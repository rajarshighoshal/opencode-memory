//! Opt-in salience decay + eviction (off by default).
//!
//! With `MCP_EVICTION_ENABLED=true`, a weekly pass soft-deletes memories that are
//! both old and low-salience (like `memory_delete`: content is tombstoned, but the
//! embedding + graph edges are dropped). An age floor protects recent memories.
//! See `decayed_salience` for the decay formula.

use crate::error::Result;
use rusqlite::Connection;

/// The master switch (`MCP_EVICTION_ENABLED`), parsed once for both this module
/// and storage's access-tracking check.
pub fn eviction_enabled() -> bool {
    std::env::var("MCP_EVICTION_ENABLED")
        .map(|v| v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Eviction tuning, resolved from the environment. Inert by default.
#[derive(Debug, Clone)]
pub struct EvictionParams {
    /// `MCP_EVICTION_ENABLED` — the opt-in master switch.
    pub enabled: bool,
    /// `MCP_EVICTION_MAX_AGE_DAYS` — never evict a memory younger than this.
    pub max_age_days: f64,
    /// `MCP_EVICTION_HALFLIFE_DAYS` — idle salience halves this often.
    pub halflife_days: f64,
    /// `MCP_EVICTION_MIN_SALIENCE` — evict below this decayed salience.
    pub min_salience: f64,
}

impl Default for EvictionParams {
    fn default() -> Self {
        Self {
            enabled: false,
            max_age_days: 365.0,
            halflife_days: 90.0,
            min_salience: 0.05,
        }
    }
}

impl EvictionParams {
    /// Resolve from environment; non-positive numeric overrides fall back to the default.
    pub fn from_env() -> Self {
        fn pos(key: &str, default: f64) -> f64 {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| *v > 0.0)
                .unwrap_or(default)
        }
        let d = EvictionParams::default();
        EvictionParams {
            enabled: eviction_enabled(),
            max_age_days: pos("MCP_EVICTION_MAX_AGE_DAYS", d.max_age_days),
            halflife_days: pos("MCP_EVICTION_HALFLIFE_DAYS", d.halflife_days),
            min_salience: pos("MCP_EVICTION_MIN_SALIENCE", d.min_salience),
        }
    }
}

/// Outcome of one eviction pass, logged for the maintenance record.
pub struct EvictReport {
    pub scanned: usize,
    pub evicted: usize,
}

/// Wall-clock epoch seconds.
fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Decayed salience for one memory. `clock` is the reinforcement point
/// (`max(created_at, last_accessed)`); salience halves every `halflife_days` idle.
fn decayed_salience(confidence: f64, clock: f64, now: f64, halflife_days: f64) -> f64 {
    if halflife_days <= 0.0 {
        return confidence;
    }
    let idle_days = ((now - clock) / 86_400.0).max(0.0);
    confidence * 0.5_f64.powf(idle_days / halflife_days)
}

/// Run one eviction pass: soft-delete old, low-salience rows and clean up their
/// embedding + graph edges (mirrors `Storage::delete`). No-op unless enabled.
///
/// `first_run` (the weekly stamp was absent) seeds a `last_accessed` baseline for
/// the pre-existing corpus so it isn't bulk-evicted before any search can stamp it.
pub fn run(db_path: &str, params: &EvictionParams, first_run: bool) -> Result<EvictReport> {
    if !params.enabled {
        return Ok(EvictReport {
            scanned: 0,
            evicted: 0,
        });
    }
    let mut conn = Connection::open(db_path)?;
    let _ = conn.busy_timeout(std::time::Duration::from_millis(5000));
    let _ = conn.pragma_update(None, "journal_mode", "WAL");
    let _ = conn.pragma_update(None, "synchronous", "NORMAL");

    let now = now_secs();

    // First enable: give the pre-existing corpus a fresh idle clock so aged-but-
    // never-stamped memories aren't bulk-evicted before reinforcement has any data.
    if first_run {
        conn.execute(
            "UPDATE memories SET last_accessed = ?1 WHERE last_accessed IS NULL AND deleted_at IS NULL",
            [now as i64],
        )?;
    }

    // Collect victims first (the mutation loop can't run while a read borrows conn).
    let mut victims: Vec<(i64, String)> = Vec::new();
    let scanned;
    {
        let mut stmt = conn.prepare(
            "SELECT id, content_hash, created_at, confidence, last_accessed \
             FROM memories WHERE deleted_at IS NULL",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                r.get::<_, Option<f64>>(3)?.unwrap_or(1.0),
                r.get::<_, Option<i64>>(4)?,
            ))
        })?;
        let mut n = 0usize;
        for row in rows {
            let (id, hash, created_at, confidence, last_accessed) = row?;
            n += 1;
            // Unknown/invalid creation time — treat as un-evictable, never ancient.
            if created_at <= 0.0 {
                continue;
            }
            // Age floor: never evict anything younger than max_age_days.
            let age_days = ((now - created_at) / 86_400.0).max(0.0);
            if age_days <= params.max_age_days {
                continue;
            }
            // Reinforcement: decay from the later of creation and last access.
            let clock = created_at.max(last_accessed.map(|t| t as f64).unwrap_or(created_at));
            let salience = decayed_salience(confidence, clock, now, params.halflife_days);
            if salience < params.min_salience {
                victims.push((id, hash));
            }
        }
        scanned = n;
    }

    if victims.is_empty() {
        return Ok(EvictReport {
            scanned,
            evicted: 0,
        });
    }

    let tx = conn.transaction()?;
    for (id, hash) in &victims {
        // Mirror Storage::delete: the embedding + graph deletes are best-effort (a
        // vec0 quirk must not abort the pass); the soft-delete is authoritative.
        if let Err(e) = tx.execute("DELETE FROM memory_embeddings WHERE rowid = ?1", [*id]) {
            tracing::warn!(id = *id, error = %e, "evict: embedding delete failed (tolerated)");
        }
        if let Err(e) = tx.execute(
            "DELETE FROM memory_graph WHERE source_hash = ?1 OR target_hash = ?1",
            [hash],
        ) {
            tracing::warn!(hash = %hash, error = %e, "evict: graph delete failed (tolerated)");
        }
        tx.execute(
            "UPDATE memories SET deleted_at = ?1 WHERE id = ?2 AND deleted_at IS NULL",
            rusqlite::params![now, id],
        )?;
    }
    tx.commit()?;

    Ok(EvictReport {
        scanned,
        evicted: victims.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::ensure_vec_extension;

    fn now() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }

    /// Build a temp DB with the eviction-relevant schema + the given rows.
    /// Each row: `(content, created_at, confidence, last_accessed_opt)`.
    fn build_db(tag: &str, rows: &[(&str, f64, f64, Option<i64>)]) -> String {
        ensure_vec_extension();
        let path = std::env::temp_dir().join(format!("evict-{}-{tag}.db", std::process::id()));
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
        let conn = Connection::open(&path).unwrap();
        conn.pragma_update(None, "journal_mode", "WAL").unwrap();
        conn.execute_batch(
            "CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL, content TEXT NOT NULL,
                tags TEXT, memory_type TEXT, metadata TEXT,
                created_at REAL, updated_at REAL, created_at_iso TEXT, updated_at_iso TEXT,
                deleted_at REAL DEFAULT NULL,
                parent_id TEXT, version INTEGER DEFAULT 1, confidence REAL DEFAULT 1.0,
                last_accessed INTEGER, superseded_by TEXT
            );
            CREATE VIRTUAL TABLE memory_embeddings USING vec0(content_embedding FLOAT[4] distance_metric=cosine);
            CREATE TABLE memory_graph (
                source_hash TEXT NOT NULL, target_hash TEXT NOT NULL, similarity REAL NOT NULL,
                connection_types TEXT NOT NULL, metadata TEXT, created_at REAL NOT NULL,
                relationship_type TEXT DEFAULT 'related', PRIMARY KEY (source_hash, target_hash)
            );",
        )
        .unwrap();
        for (i, (content, created_at, confidence, last_accessed)) in rows.iter().enumerate() {
            let hash = format!("hash{i}");
            conn.execute(
                "INSERT INTO memories (content_hash, content, tags, memory_type, metadata, \
                 created_at, updated_at, created_at_iso, updated_at_iso, confidence, last_accessed) \
                 VALUES (?1, ?2, '', 'note', '{}', ?3, ?3, '', '', ?4, ?5)",
                rusqlite::params![hash, content, created_at, confidence, last_accessed],
            )
            .unwrap();
        }
        drop(conn);
        path.to_string_lossy().to_string()
    }

    fn live_count(path: &str) -> i64 {
        let conn = Connection::open(path).unwrap();
        conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap()
    }

    fn cleanup(path: &str) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{path}{suffix}"));
        }
    }

    #[test]
    fn evicts_only_old_low_salience_untouched() {
        let n = now();
        let day = 86_400.0;
        let path = build_db(
            "mix",
            &[
                ("recent", n - 10.0 * day, 1.0, None), // young → age floor protects it
                ("old-untouched", n - 500.0 * day, 1.0, None), // old + decayed ≈0.02 → EVICT
                ("old-but-used", n - 500.0 * day, 1.0, Some((n - day) as i64)), // reinforced → survives
            ],
        );
        let params = EvictionParams {
            enabled: true,
            ..EvictionParams::default()
        };
        let report = run(&path, &params, false).unwrap();
        assert_eq!(report.scanned, 3);
        assert_eq!(report.evicted, 1, "only the old, untouched, low-salience memory");
        assert_eq!(live_count(&path), 2);
        cleanup(&path);
    }

    #[test]
    fn age_floor_protects_everything_when_high() {
        let n = now();
        let day = 86_400.0;
        let path = build_db("safe", &[("old", n - 500.0 * day, 1.0, None)]);
        let params = EvictionParams {
            enabled: true,
            max_age_days: 100_000.0,
            ..EvictionParams::default()
        };
        let report = run(&path, &params, false).unwrap();
        assert_eq!(report.evicted, 0);
        assert_eq!(live_count(&path), 1);
        cleanup(&path);
    }

    #[test]
    fn disabled_params_evict_nothing() {
        let n = now();
        let path = build_db("disabled", &[("old", n - 500.0 * 86_400.0, 1.0, None)]);
        // Default params are disabled → run() must be a no-op even on an aged corpus.
        let report = run(&path, &EvictionParams::default(), false).unwrap();
        assert_eq!(report.evicted, 0);
        assert_eq!(live_count(&path), 1);
        cleanup(&path);
    }

    #[test]
    fn first_run_seeds_baseline_not_bulk_evict() {
        let n = now();
        let day = 86_400.0;
        let path = build_db("firstrun", &[("aged-unstamped", n - 500.0 * day, 1.0, None)]);
        let params = EvictionParams {
            enabled: true,
            ..EvictionParams::default()
        };
        // First pass seeds last_accessed=now → aged-but-never-stamped survives.
        let r1 = run(&path, &params, true).unwrap();
        assert_eq!(r1.evicted, 0, "first pass must seed a baseline, not bulk-evict");
        assert_eq!(live_count(&path), 1);
        // Simulate the seed having aged (never re-accessed since enable).
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute("UPDATE memories SET last_accessed = ?1", [(n - 500.0 * day) as i64])
                .unwrap();
        }
        // A later, non-first pass now evicts the still-idle memory.
        let r2 = run(&path, &params, false).unwrap();
        assert_eq!(r2.evicted, 1, "an idle memory becomes eligible after the baseline ages");
        assert_eq!(live_count(&path), 0);
        cleanup(&path);
    }

    #[test]
    fn null_or_zero_created_at_is_never_evicted() {
        let path = build_db("nocreate", &[("no-created-at", 0.0, 1.0, None)]);
        let params = EvictionParams {
            enabled: true,
            ..EvictionParams::default()
        };
        let report = run(&path, &params, false).unwrap();
        assert_eq!(report.evicted, 0, "created_at<=0 must be treated as un-evictable");
        assert_eq!(live_count(&path), 1);
        cleanup(&path);
    }

    #[test]
    fn decayed_salience_halves_and_reinforces() {
        let n = 1_000_000.0;
        let day = 86_400.0;
        // Idle 180 days at halflife 90 → 0.5^2 = 0.25.
        assert!((decayed_salience(1.0, n - 180.0 * day, n, 90.0) - 0.25).abs() < 1e-6);
        // Reinforced (clock == now) → ~1.0.
        assert!((decayed_salience(1.0, n, n, 90.0) - 1.0).abs() < 1e-9);
    }
}
