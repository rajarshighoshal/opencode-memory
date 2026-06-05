//! Association-graph builder.
//!
//! Runs as a one-shot CLI step (`opencode-memory consolidate <db-path>`), invoked
//! weekly by the `memory-mcp` wrapper. Requires no embedding server: it reads the
//! embeddings already stored in the DB, computes pairwise cosine similarity, and
//! writes `memory_graph` edges for pairs inside a per-DB similarity window.
//!
//! ## Model
//! Edges are scored on a single, well-defined signal — semantic cosine of the
//! stored embeddings — mapped to `(cos + 1) / 2` so scores land on the same [0, 1]
//! scale the rest of the graph uses, plus a cheap `shared_tags` flag.
//! `connection_types` reflects exactly what we computed. The graph reader only
//! walks edges, so writing only `related` edges is safe; edges are additive
//! (`INSERT OR REPLACE`) and never auto-supersede a memory. Dangling edges (whose
//! endpoints no longer exist) are pruned each run.

use crate::error::Result;
use crate::models::tags_from_csv;
use crate::storage::blob_to_vec;
use rusqlite::{params, Connection};

/// Outcome of one consolidation run, printed by `main` for the cron log.
pub struct Report {
    pub memories: usize,
    pub lo: f64,
    pub hi: f64,
    pub edges_before: usize,
    pub edges_after: usize,
    pub pruned: usize,
    pub coverage: usize,
}

/// Linear-interpolation quantile (numpy `method="linear"` default) over a
/// pre-sorted ascending slice. `q` in [0, 1].
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Above this memory count, log a heads-up that the weekly O(n^2) consolidation
/// pass is large (it still runs — detached, WAL, lock-guarded — just slowly).
const CONSOLIDATE_WARN_N: usize = 5000;

/// Pick the similarity window [lo, hi] from already-computed pairwise similarities:
/// keep the graph sparse via the 0.88 quantile lower bound (clamped to [0.50, 0.90])
/// once there are >= 20 pairs, else 0.50; hi = 0.98. Takes the sims so the caller's
/// single pairwise pass is reused (no recomputation of the dot products).
fn pick_window_from_sims(sims: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut sims: Vec<f64> = sims.collect();
    if sims.len() < 20 {
        return (0.50, 0.98);
    }
    sims.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (quantile_sorted(&sims, 0.88).clamp(0.50, 0.90), 0.98)
}

/// Cosine of two already-L2-normalized vectors, mapped to [0, 1] via (cos+1)/2.
fn mapped_cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    (dot.clamp(-1.0, 1.0) + 1.0) / 2.0
}

/// L2-normalize; returns None for a zero vector (cannot normalize).
fn normalize(v: &[f32]) -> Option<Vec<f32>> {
    let norm: f64 = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    Some(v.iter().map(|x| (*x as f64 / norm) as f32).collect())
}

/// connection_types JSON for an edge: always "semantic"; add "shared_tags" when
/// the two memories share a real (non-"untagged") tag.
fn connection_types(tags_a: &[String], tags_b: &[String]) -> String {
    let mut ct = vec!["semantic"];
    let shared = tags_a
        .iter()
        .any(|t| t != "untagged" && tags_b.iter().any(|u| u == t));
    if shared {
        ct.push("shared_tags");
    }
    serde_json::to_string(&ct).unwrap_or_else(|_| "[\"semantic\"]".to_string())
}

/// Build the association graph for `db_path`. Idempotent (`INSERT OR REPLACE` +
/// dangling-edge prune), additive, no migration.
pub fn run(db_path: &str) -> Result<Report> {
    let conn = Connection::open(db_path)?;
    let _ = conn.busy_timeout(std::time::Duration::from_millis(5000));
    let _ = conn.pragma_update(None, "journal_mode", "WAL");
    let _ = conn.pragma_update(None, "synchronous", "NORMAL");

    // Load non-deleted memories (id -> hash/tags) and their stored embeddings (rowid == id).
    let mut hashes: Vec<String> = Vec::new();
    let mut tag_lists: Vec<Vec<String>> = Vec::new();
    let mut norms: Vec<Vec<f32>> = Vec::new();
    {
        // id -> (hash, tags)
        let mut meta: std::collections::HashMap<i64, (String, Vec<String>)> =
            std::collections::HashMap::new();
        let mut stmt = conn
            .prepare("SELECT id, content_hash, tags FROM memories WHERE deleted_at IS NULL")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })?;
        for row in rows {
            let (id, hash, tags) = row?;
            let tv = tags.as_deref().map(tags_from_csv).unwrap_or_default();
            meta.insert(id, (hash, tv));
        }

        // Embeddings keyed by rowid (== memories.id). Join in code so a memory
        // without an embedding is simply skipped.
        let mut estmt = conn.prepare("SELECT rowid, content_embedding FROM memory_embeddings")?;
        let erows = estmt.query_map([], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
        })?;
        for row in erows {
            let (rowid, blob) = row?;
            let Some((hash, tv)) = meta.remove(&rowid) else { continue };
            let Ok(vec) = blob_to_vec(&blob) else { continue };
            let Some(nv) = normalize(&vec) else { continue };
            hashes.push(hash);
            tag_lists.push(tv);
            norms.push(nv);
        }
    }

    let n = hashes.len();
    let edges_before: usize =
        conn.query_row("SELECT COUNT(*) FROM memory_graph", [], |r| r.get::<_, i64>(0))? as usize;

    // Score every pair ONCE — the 2560-dim dot products dominate runtime — and
    // reuse the buffer for both the window quantile and edge emission (previously
    // the matrix was computed twice). Surface the work magnitude before it runs.
    let pair_count = n.saturating_mul(n.saturating_sub(1)) / 2;
    if n >= 2 {
        tracing::info!(memories = n, pairs = pair_count, "consolidation: scoring pairwise cosine similarities");
        if n > CONSOLIDATE_WARN_N {
            tracing::warn!(memories = n, pairs = pair_count, "large memory count; weekly consolidation is O(n^2) and may take a while");
        }
    }
    let mut pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(pair_count);
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j, mapped_cosine(&norms[i], &norms[j])));
        }
    }
    let (lo, hi) = pick_window_from_sims(pairs.iter().map(|&(_, _, s)| s));

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);

    if n >= 2 {
        let tx = conn.unchecked_transaction()?;
        {
            let mut ins = tx.prepare(
                "INSERT OR REPLACE INTO memory_graph \
                 (source_hash, target_hash, similarity, connection_types, metadata, created_at, relationship_type) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'related')",
            )?;
            for &(i, j, sim) in &pairs {
                if sim < lo || sim > hi {
                    continue;
                }
                let ct = connection_types(&tag_lists[i], &tag_lists[j]);
                let metadata = format!(
                    "{{\"discovery_method\": \"semantic_cosine\", \"confidence\": {:.6}}}",
                    sim
                );
                // Write both directions: the graph stores symmetric edges.
                ins.execute(params![hashes[i], hashes[j], sim, ct, metadata, now])?;
                ins.execute(params![hashes[j], hashes[i], sim, ct, metadata, now])?;
            }
        }
        tx.commit()?;
    }

    // Prune dangling edges (source/target no longer live). Nothing else
    // reconciles memory_graph outside the per-row delete path, so bulk/manual
    // deletes can orphan edges. Idempotent.
    let pruned = conn.execute(
        "DELETE FROM memory_graph WHERE source_hash NOT IN \
         (SELECT content_hash FROM memories WHERE deleted_at IS NULL) \
         OR target_hash NOT IN (SELECT content_hash FROM memories WHERE deleted_at IS NULL)",
        [],
    )?;

    let edges_after: usize =
        conn.query_row("SELECT COUNT(*) FROM memory_graph", [], |r| r.get::<_, i64>(0))? as usize;
    let coverage: usize = conn.query_row(
        "SELECT COUNT(DISTINCT source_hash) FROM memory_graph",
        [],
        |r| r.get::<_, i64>(0),
    )? as usize;

    Ok(Report {
        memories: n,
        lo,
        hi,
        edges_before,
        edges_after,
        pruned,
        coverage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{ensure_vec_extension, seeded_embed};
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn ensure_ext() {
        INIT.call_once(|| {
            ensure_vec_extension();
        });
    }

    #[test]
    fn test_quantile_sorted() {
        let v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!((quantile_sorted(&v, 0.5) - 49.5).abs() < 0.01);
        assert!((quantile_sorted(&v, 0.0) - 0.0).abs() < 0.01);
        assert!((quantile_sorted(&v, 1.0) - 99.0).abs() < 0.01);
    }

    #[test]
    fn test_quantile_singleton() {
        assert_eq!(quantile_sorted(&[7.0], 0.5), 7.0);
    }

    #[test]
    fn test_mapped_cosine() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        // dot = 0 → cos=0 → mapped=0.5
        assert!((mapped_cosine(&a, &b) - 0.5).abs() < 1e-9);
        // same vector → cos=1 → mapped=1.0
        assert!((mapped_cosine(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalize_zero_vector() {
        assert!(normalize(&[0.0f32; 10]).is_none());
    }

    #[test]
    fn test_normalize_unit() {
        let v = normalize(&[3.0f32]).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_connection_types_no_shared() {
        let a = vec!["x".to_string()];
        let b = vec!["y".to_string()];
        let ct = connection_types(&a, &b);
        assert!(ct.contains("semantic"));
        assert!(!ct.contains("shared_tags"));
    }

    #[test]
    fn test_connection_types_shared() {
        let a = vec!["python".to_string(), "reference".to_string()];
        let b = vec!["python".to_string(), "other".to_string()];
        let ct = connection_types(&a, &b);
        assert!(ct.contains("semantic"));
        assert!(ct.contains("shared_tags"));
    }

    #[test]
    fn test_pick_window_empty() {
        let (lo, hi) = pick_window_from_sims(std::iter::empty());
        assert_eq!(lo, 0.50);
        assert_eq!(hi, 0.98);
    }

    #[test]
    fn test_pick_window_few_pairs() {
        // Fewer than 20 pairs -> default window (no quantile narrowing).
        let (lo, hi) = pick_window_from_sims([0.7, 0.8, 0.9].into_iter());
        assert_eq!(lo, 0.50);
        assert_eq!(hi, 0.98);
    }

    #[test]
    fn test_pick_window_quantile_clamped() {
        // >=20 pairs: lo = 0.88-quantile clamped into [0.50, 0.90].
        let (lo, hi) = pick_window_from_sims(vec![0.99f64; 25].into_iter());
        assert_eq!(lo, 0.90);
        assert_eq!(hi, 0.98);
    }

    /// Full consolidate integration test: store memories with embeddings,
    /// run the consolidator, verify edges and pruning.
    #[test]
    fn test_run_creates_and_prunes_edges() {
        ensure_ext();

        // Build a fresh temp DB with the schema + some memories.
        let db_path = std::env::temp_dir().join(format!(
            "consolidate-test-{}.db",
            std::process::id()
        ));
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.pragma_update(None, "journal_mode", "WAL").unwrap();
        conn.pragma_update(None, "busy_timeout", 5000).unwrap();
        conn.pragma_update(None, "synchronous", "NORMAL").unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                tags TEXT, memory_type TEXT, metadata TEXT,
                created_at REAL, updated_at REAL,
                created_at_iso TEXT, updated_at_iso TEXT,
                deleted_at REAL DEFAULT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                content_embedding FLOAT[2560] distance_metric=cosine
            );
            CREATE TABLE IF NOT EXISTS memory_graph (
                source_hash TEXT NOT NULL, target_hash TEXT NOT NULL,
                similarity REAL NOT NULL,
                connection_types TEXT NOT NULL, metadata TEXT,
                created_at REAL NOT NULL,
                relationship_type TEXT DEFAULT 'related',
                PRIMARY KEY (source_hash, target_hash)
            );",
        )
        .unwrap();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        // Insert 3 memories: a (seed=1), b (seed=2), c (seed=3).
        // seed-1 & seed-2 are orthogonal → mapped cos ≈0.5 (below typical window lo=0.50).
        // But with 3 memories, pairwise similarities above 0.88 quantile may include them.
        for (content, emb) in [
            ("con-a", seeded_embed(1)),
            ("con-b", seeded_embed(2)),
            ("con-c", seeded_embed(3)),
        ] {
            let h = crate::hashing::content_hash(content);
            let blob: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
            conn.execute(
                "INSERT INTO memories (content_hash, content, tags, memory_type, metadata, created_at, updated_at, created_at_iso, updated_at_iso) VALUES (?1,?2,'','note','{}',?3,?3,'2024-01-01','2024-01-01')",
                rusqlite::params![h, content, now],
            ).unwrap();
            conn.execute(
                "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (last_insert_rowid(), ?1)",
                rusqlite::params![blob],
            ).unwrap();
        }

        drop(conn);

        // Run consolidate
        let report = run(db_path.to_str().unwrap()).unwrap();
        assert!(report.memories >= 2, "expected ≥2 memories; got {}", report.memories);
        assert!(report.lo >= 0.50 && report.lo <= 0.90, "lo={}", report.lo);
        assert_eq!(report.hi, 0.98);

        // Verify edges exist in the DB
        let conn2 = rusqlite::Connection::open(&db_path).unwrap();
        let edge_count: i64 = conn2
            .query_row("SELECT COUNT(*) FROM memory_graph", [], |r| r.get(0))
            .unwrap();
        assert!(edge_count > 0, "expected some edges after consolidation");
        assert_eq!(edge_count as usize, report.edges_after);
        assert_eq!(report.pruned, 0);

        // Test idempotency: second run should not change edge count.
        let report2 = run(db_path.to_str().unwrap()).unwrap();
        assert_eq!(report2.edges_after, report.edges_after);

        // Cleanup
        drop(conn2);
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(format!("{}-wal", db_path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", db_path.display()));
    }
}
