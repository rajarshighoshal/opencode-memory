//! sqlite-vec storage layer.
//!
//! Opens existing DBs in place with no migration. The on-disk vec0 format
//! (`FLOAT[2560] distance_metric=cosine`, raw LE-f32 blobs) and the SHA-256
//! content_hash are part of the stable on-disk contract — anything reading or
//! writing these DBs must agree on them exactly.
//!
//! ## Extension loading (static link, preferred)
//! `sqlite-vec` 0.1.9 statically compiles `sqlite-vec.c` and exports
//! `sqlite3_vec_init`. Register it as an auto-extension ONCE at process start,
//! BEFORE opening any connection:
//! ```ignore
//! unsafe {
//!     rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
//!         sqlite_vec::sqlite3_vec_init as *const (),
//!     )));
//! }
//! ```
//! No dylib and no `enable_load_extension` needed — the extension is compiled in.
//!
//! ## PRAGMAs (kept consistent across all processes so WAL stays compatible)
//! `journal_mode=WAL`, `busy_timeout=5000`, `synchronous=NORMAL`.
//!
//! ## Schema (already present; we only ever touch `memories`, `memory_embeddings`,
//! `memory_graph`; the FTS5 mirror is maintained by triggers, the vec0 shadow
//! tables by the extension — never write them directly):
//!   memories(id INTEGER PK AUTOINCREMENT, content_hash UNIQUE, content, tags,
//!            memory_type, metadata, created_at REAL, updated_at REAL,
//!            created_at_iso, updated_at_iso, deleted_at REAL DEFAULT NULL,
//!            parent_id, version, confidence, last_accessed, superseded_by)
//!   memory_embeddings USING vec0(content_embedding FLOAT[2560] distance_metric=cosine)
//!     -- rowid == memories.id (application convention, NOT a FK)
//!   memory_content_fts USING fts5(content, content='memories', tokenize='trigram')

use crate::config::{Config, MAX_KNN_K};
use crate::error::{MemoryError, Result};
use crate::hashing::content_hash;
use crate::models::{epoch_to_iso, tags_from_csv, Memory, SearchHit, SearchMode, TagMatch};
use rusqlite::{params_from_iter, Connection, OptionalExtension};
use std::sync::Mutex;

/// `(content_hash, distance)` — a node in a graph traversal result.
type GraphNode = (String, usize);
/// `(source_hash, target_hash, similarity)` — an edge in a subgraph result.
type GraphEdge = (String, String, f64);

/// Escape LIKE metacharacters for use with `ESCAPE '\'`. Order matters:
/// backslash first (so the escapes we add aren't re-escaped), then `%` and `_`.
fn escape_like(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Wall-clock epoch seconds, used for `created_at`/`updated_at`/`deleted_at`.
fn now_epoch() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "system clock is before UNIX_EPOCH; recording timestamp as 0.0");
            0.0
        })
}

/// Recency half-life in days for the optional search reweight; 0 (default) = off.
/// A research knowledge base should not auto-forget, so created-at decay is opt-in.
fn recency_halflife_days() -> f64 {
    std::env::var("MCP_RECENCY_HALFLIFE_DAYS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(0.0)
}

/// Multiplicatively decay each hit's relevance by created-at age (exponential
/// half-life) and re-sort by the reweighted score. No-op when `halflife_days <= 0`.
fn apply_recency(hits: &mut [SearchHit], halflife_days: f64) {
    if halflife_days <= 0.0 {
        return;
    }
    let now = now_epoch();
    for h in hits.iter_mut() {
        let age_days = ((now - h.memory.created_at) / 86400.0).max(0.0);
        let decay = (-std::f64::consts::LN_2 * age_days / halflife_days).exp();
        h.relevance_score *= decay;
    }
    hits.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Parse a `YYYY-MM-DD` (or full ISO) date bound to epoch seconds for the
/// `after`/`before` filters, treating naive datetimes as UTC.
fn parse_iso_date_to_epoch(s: &str) -> Option<f64> {
    use chrono::{NaiveDate, NaiveDateTime, TimeZone, Utc};
    // Try full datetime first, then date-only at midnight UTC.
    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(Utc.from_utc_datetime(&ndt).timestamp() as f64);
    }
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let ndt = d.and_hms_opt(0, 0, 0)?;
        return Some(Utc.from_utc_datetime(&ndt).timestamp() as f64);
    }
    None
}

/// Register the statically-linked sqlite-vec extension as a SQLite
/// auto-extension. Call EXACTLY ONCE, before opening any connection.
///
/// # Safety
/// Transmutes the `sqlite3_vec_init` fn pointer into the C entrypoint signature
/// `sqlite3_auto_extension` expects. Sound because that is the documented
/// integration pattern for the `sqlite-vec` crate.
pub fn register_vec_extension() -> Result<()> {
    // SAFETY: transmute the sqlite-vec init fn pointer into the C entrypoint
    // signature that sqlite3_auto_extension expects. This is the documented
    // integration pattern for the sqlite-vec crate's static build.
    unsafe {
        let rc = rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
        if rc != rusqlite::ffi::SQLITE_OK {
            return Err(MemoryError::Other(format!(
                "sqlite3_auto_extension failed: rc={rc}"
            )));
        }
    }
    Ok(())
}

/// Owns the connection to one scope's DB. Single-process, single-connection
/// behind a `Mutex` (rusqlite `Connection` is not `Sync`); the stdio MCP server
/// handles one request at a time anyway, so contention is negligible.
pub struct Storage {
    conn: Mutex<Connection>,
    /// Embedding width this DB uses — the existing vec0 table's width, or the
    /// configured dim for a freshly-created DB.
    embedding_dim: usize,
    /// Semantic-dedup config, read from env at open time
    /// (`MCP_SEMANTIC_DEDUP_ENABLED` default true, `MCP_SEMANTIC_DEDUP_THRESHOLD`
    /// 0.85, `MCP_SEMANTIC_DEDUP_TIME_WINDOW_HOURS` 24).
    semantic_dedup_enabled: bool,
    semantic_dedup_threshold: f64,
    semantic_dedup_time_window_hours: f64,
}

/// Semantic-dedup settings resolved from the environment.
struct SemanticDedupConfig {
    enabled: bool,
    threshold: f64,
    time_window_hours: f64,
}

impl SemanticDedupConfig {
    fn from_env() -> Self {
        let enabled = std::env::var("MCP_SEMANTIC_DEDUP_ENABLED")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(true);
        let threshold = std::env::var("MCP_SEMANTIC_DEDUP_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.85);
        let time_window_hours = std::env::var("MCP_SEMANTIC_DEDUP_TIME_WINDOW_HOURS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24.0);
        Self {
            enabled,
            threshold,
            time_window_hours,
        }
    }
}

/// Arguments for [`Storage::store`]. `tags`/`memory_type`/`metadata` come from
/// `memory_store`'s `metadata.tags` / `metadata.type` (note: tags live UNDER
/// metadata in the store schema, unlike search/list/delete which take top-level tags).
pub struct StoreArgs {
    pub content: String,
    pub tags: Vec<String>,
    pub memory_type: Option<String>,
    pub metadata: serde_json::Value,
    /// If set, skip semantic dedup (exact-hash dedup still applies).
    pub conversation_id: Option<String>,
}

/// Outcome of a store. `Duplicate` carries the existing hash (exact-hash hit);
/// `SemanticDuplicate` carries the hash of the near-duplicate that blocked the
/// store (semantic-dedup hit, enabled by default).
#[derive(Debug)]
pub enum StoreOutcome {
    Stored { content_hash: String },
    Duplicate { content_hash: String },
    SemanticDuplicate { existing_hash: String },
}

/// Filters for [`Storage::search`].
#[derive(Clone)]
pub struct SearchQuery {
    pub query: Option<String>,
    pub mode: SearchMode,
    pub tags: Vec<String>,
    pub tag_match: TagMatch,
    pub after: Option<String>,
    pub before: Option<String>,
    pub limit: usize,
    pub include_superseded: bool,
}

/// Filters for [`Storage::list`] (paginated browse).
pub struct ListQuery {
    pub page: usize,
    pub page_size: usize,
    pub tags: Vec<String>,
    pub tag_match: TagMatch,
    pub memory_type: Option<String>,
}

/// A page of list results.
pub struct ListPage {
    pub memories: Vec<Memory>,
    pub page: usize,
    pub page_size: usize,
    pub total: usize,
    pub total_pages: usize,
    pub has_more: bool,
}

/// Filters for [`Storage::delete`]. Soft-delete only. No filters at all is an
/// error (mass-delete guard); `content_hash` wins over the other filters.
pub struct DeleteQuery {
    pub content_hash: Option<String>,
    pub tags: Vec<String>,
    pub tag_match: TagMatch,
    pub before: Option<String>,
    pub after: Option<String>,
    pub dry_run: bool,
}

/// Outcome of a [`Storage::delete`], carrying enough to build the unified
/// `memory_delete` response (single-hash vs bulk vs dry-run).
#[derive(Debug)]
pub struct DeleteOutcome {
    pub deleted_count: usize,
    pub deleted_hashes: Vec<String>,
    /// The storage-layer message (e.g. `"Successfully deleted memory <h>"` for a
    /// single hash). The unified handler appends `"\n\nDeleted N memories"` etc.
    pub message: String,
    /// `Some(hash)` for the single-hash branch; `None` for filter-based deletes.
    /// Carried for callers/future use; the formatted message already embeds it.
    #[allow(dead_code)]
    pub single_hash: Option<String>,
}

/// DB statistics for `memory_health` (counts, sizes, embedding info).
pub struct DbStats {
    pub total_memories: usize,
    pub database_size_bytes: u64,
    pub embedding_model: String,
    // Resolved for completeness but intentionally NOT surfaced in the health output.
    #[allow(dead_code)]
    pub embedding_dimension: usize,
}

/// Create the canonical schema if absent so a brand-new DB is usable. Idempotent
/// — every statement is `IF NOT EXISTS`, so it is a no-op on existing DBs and
/// never alters them. The vec0 extension must already be registered (main does it
/// before open). The vec0 embedding table is created `dim` wide (a fresh DB takes
/// the configured dim); the rest is an FTS5 trigram external-content mirror plus
/// sync triggers and the association-graph table.
fn ensure_schema(conn: &Connection, dim: usize) -> Result<()> {
    // Width-independent tables + indexes.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            memory_type TEXT,
            metadata TEXT,
            created_at REAL,
            updated_at REAL,
            created_at_iso TEXT,
            updated_at_iso TEXT,
            deleted_at REAL DEFAULT NULL,
            parent_id TEXT,
            version INTEGER DEFAULT 1,
            confidence REAL DEFAULT 1.0,
            last_accessed INTEGER,
            superseded_by TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash);
        CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);
        CREATE INDEX IF NOT EXISTS idx_deleted_at ON memories(deleted_at);",
    )?;
    // Embedding table — its width is fixed at creation, so it carries `dim`.
    conn.execute(
        &format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings \
             USING vec0(content_embedding FLOAT[{dim}] distance_metric=cosine)"
        ),
        [],
    )?;
    // FTS mirror + sync triggers + association graph.
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_content_fts USING fts5(
            content,
            content='memories',
            content_rowid='id',
            tokenize='trigram'
        );
        CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memory_content_fts(rowid, content) VALUES (new.id, new.content);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN
            DELETE FROM memory_content_fts WHERE rowid = old.id;
            INSERT INTO memory_content_fts(rowid, content) VALUES (new.id, new.content);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
            DELETE FROM memory_content_fts WHERE rowid = old.id;
        END;
        CREATE TABLE IF NOT EXISTS memory_graph (
            source_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            similarity REAL NOT NULL,
            connection_types TEXT NOT NULL,
            metadata TEXT,
            created_at REAL NOT NULL,
            relationship_type TEXT DEFAULT 'related',
            PRIMARY KEY (source_hash, target_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_graph_source ON memory_graph(source_hash);
        CREATE INDEX IF NOT EXISTS idx_graph_target ON memory_graph(target_hash);
        CREATE INDEX IF NOT EXISTS idx_graph_relationship ON memory_graph(relationship_type);",
    )?;
    Ok(())
}

impl Storage {
    /// Open the scope DB, apply PRAGMAs, and verify `vec_version()` resolves
    /// (proves the extension is registered). Assumes [`register_vec_extension`]
    /// already ran.
    pub fn open(cfg: &Config) -> Result<Self> {
        // Ensure the parent directory exists before opening.
        if let Some(parent) = cfg.db_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(&cfg.db_path)?;
        // PRAGMAs kept consistent across processes so WAL stays compatible.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        // Verify the vec0 extension is live (proves register_vec_extension ran).
        let _v: String = conn.query_row("SELECT vec_version()", [], |r| r.get(0))?;
        // Embedding width: an existing DB keeps the width its vec0 table was created
        // with (parsed from its `FLOAT[N]` definition); a fresh DB uses the configured
        // dim. Warn if a non-empty DB's width disagrees with MCP_EXTERNAL_EMBEDDING_DIM.
        let existing_dim: Option<usize> = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name = 'memory_embeddings'",
                [],
                |r| r.get::<_, String>(0),
            )
            .optional()?
            .and_then(|sql| {
                sql.split("FLOAT[")
                    .nth(1)
                    .and_then(|s| s.split(']').next())
                    .and_then(|n| n.trim().parse::<usize>().ok())
            });
        let embedding_dim = existing_dim.unwrap_or(cfg.embed.embedding_dim);
        if let Some(d) = existing_dim {
            if d != cfg.embed.embedding_dim {
                tracing::warn!(
                    db_dim = d,
                    configured = cfg.embed.embedding_dim,
                    "existing DB embedding width differs from MCP_EXTERNAL_EMBEDDING_DIM; using the DB's width"
                );
            }
        }
        // Bootstrap the schema for a brand-new DB (no-op on existing DBs). Without
        // this, a freshly created DB would have no `memories` table and every store
        // would fail ("no such table: memories").
        ensure_schema(&conn, embedding_dim)?;
        let dedup = SemanticDedupConfig::from_env();
        Ok(Self {
            conn: Mutex::new(conn),
            embedding_dim,
            semantic_dedup_enabled: dedup.enabled,
            semantic_dedup_threshold: dedup.threshold,
            semantic_dedup_time_window_hours: dedup.time_window_hours,
        })
    }

    /// The embedding width this DB uses (existing vec0 table's width, or the
    /// configured dim for a fresh DB).
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// `vec_version()` — for the health check + startup self-test.
    pub fn vec_version(&self) -> Result<String> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row("SELECT vec_version()", [], |r| r.get(0))?)
    }

    /// Store a memory atomically, in five steps:
    ///   1. exact-hash dedup: SELECT WHERE content_hash=? AND deleted_at IS NULL -> skip if present.
    ///   2. tombstone purge: DELETE FROM memories WHERE content_hash=? AND deleted_at IS NOT NULL
    ///      (also delete the orphaned memory_embeddings row for that old id).
    ///   3. INSERT INTO memories(...9 cols...) -> capture last_insert_rowid().
    ///   4. INSERT INTO memory_embeddings(rowid, content_embedding) with rowid = that lastrowid.
    ///   5. wrap 3-4 (and 2) in one transaction; FTS triggers fire automatically.
    /// The `embedding` is the 2560-f32 vector from [`crate::embed::EmbedClient`].
    pub fn store(&self, args: &StoreArgs, embedding: &[f32]) -> Result<StoreOutcome> {
        let hash = content_hash(&args.content);
        let mut conn = self.conn.lock().unwrap();

        // Step 1: exact-hash dedup (live rows only) -> store nothing if a live row
        // with this hash already exists.
        let existing: Option<String> = conn
            .query_row(
                "SELECT content_hash FROM memories WHERE content_hash = ?1 AND deleted_at IS NULL",
                [&hash],
                |r| r.get(0),
            )
            .ok();
        if let Some(h) = existing {
            return Ok(StoreOutcome::Duplicate { content_hash: h });
        }

        // Semantic dedup (enabled by default). Skipped when the caller signals an
        // incremental save (conversation_id set) or for session memories
        // (memory_type == "session"); exact-hash dedup above is always enforced.
        // When the incoming embedding is >= threshold cosine-similar to a memory
        // stored within the time window, store NOTHING and report the near-duplicate
        // hash.
        let skip_semantic_dedup = args.conversation_id.is_some()
            || args.memory_type.as_deref() == Some("session");
        if self.semantic_dedup_enabled && !skip_semantic_dedup {
            // cosine distance = 1 - similarity; threshold 0.85 -> distance <= 0.15.
            let max_distance = 1.0 - self.semantic_dedup_threshold;
            let cutoff = now_epoch() - self.semantic_dedup_time_window_hours * 3600.0;
            let query_blob = vec_to_blob(embedding);
            // Brute-force scalar cosine distance over recent live rows (NOT the
            // vec0 KNN MATCH) — the candidate set is bounded by the time window.
            // This is an O(live rows within the window) 2560-dim scan per store, so
            // a bulk import into a wide window pays it on every row; pass a
            // conversation_id (or memory_type="session") to skip this path entirely.
            let dup: Option<(String, f64)> = conn
                .query_row(
                    "SELECT m.content_hash,
                            vec_distance_cosine(me.content_embedding, ?1) AS distance
                     FROM memories m
                     JOIN memory_embeddings me ON m.id = me.rowid
                     WHERE m.created_at > ?2
                       AND m.deleted_at IS NULL
                     ORDER BY distance ASC
                     LIMIT 1",
                    rusqlite::params![query_blob, cutoff],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)),
                )
                .optional()?;
            if let Some((existing_hash, distance)) = dup {
                if distance <= max_distance {
                    return Ok(StoreOutcome::SemanticDuplicate { existing_hash });
                }
            }
        }

        // tags are stored as a comma-joined string (empty -> ""). memory_type
        // default is applied at the tool layer. metadata is stored as compact JSON
        // ("{}" when empty).
        let tags_str = crate::models::tags_to_csv(&args.tags);
        let metadata_str = if args.metadata.is_null()
            || (args.metadata.is_object() && args.metadata.as_object().unwrap().is_empty())
        {
            "{}".to_string()
        } else {
            serde_json::to_string(&args.metadata)?
        };

        let now = now_epoch();
        let iso = epoch_to_iso(now);
        let blob = vec_to_blob(embedding);

        // Steps 2-4 atomically (FTS triggers fire automatically; never write vec0
        // shadow / FTS tables directly).
        let tx = conn.transaction()?;
        // Step 2: tombstone purge so the content_hash UNIQUE constraint allows the
        // re-insert, plus delete the orphaned memory_embeddings row for the old id.
        {
            let old_id: Option<i64> = tx
                .query_row(
                    "SELECT id FROM memories WHERE content_hash = ?1 AND deleted_at IS NOT NULL",
                    [&hash],
                    |r| r.get(0),
                )
                .ok();
            if let Some(id) = old_id {
                tx.execute("DELETE FROM memory_embeddings WHERE rowid = ?1", [id])?;
                tx.execute(
                    "DELETE FROM memories WHERE content_hash = ?1 AND deleted_at IS NOT NULL",
                    [&hash],
                )?;
            }
        }

        // Step 3: insert the memory row (9 cols).
        tx.execute(
            "INSERT INTO memories (
                content_hash, content, tags, memory_type,
                metadata, created_at, updated_at, created_at_iso, updated_at_iso
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                hash,
                args.content,
                tags_str,
                args.memory_type,
                metadata_str,
                now,
                now,
                iso,
                iso,
            ],
        )?;
        let rowid = tx.last_insert_rowid();

        // Step 4: insert the embedding at rowid == memories.id.
        tx.execute(
            "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (?1, ?2)",
            rusqlite::params![rowid, blob],
        )?;

        tx.commit()?;
        Ok(StoreOutcome::Stored { content_hash: hash })
    }

    /// Semantic KNN search — the primary query:
    /// ```sql
    /// SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
    ///        m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso, e.distance
    /// FROM memories m
    /// INNER JOIN (SELECT rowid, distance FROM memory_embeddings
    ///             WHERE content_embedding MATCH ?1 AND k = ?2) e
    ///   ON m.id = e.rowid
    /// WHERE m.deleted_at IS NULL
    ///   AND (m.superseded_by IS NULL OR m.superseded_by = '')
    ///   {optional tag/time filters}
    /// ORDER BY e.distance LIMIT ?3
    /// ```
    /// `k` is a vec0 constraint (cap [`crate::config::MAX_KNN_K`]), bound as `AND k = ?`.
    /// `query_embedding` is serialized to the raw 10240-byte LE-f32 blob.
    pub fn search_semantic(&self, q: &SearchQuery, query_embedding: &[f32]) -> Result<Vec<SearchHit>> {
        let conn = self.conn.lock().unwrap();

        // Count embeddings; an empty table can't be KNN-searched.
        let embedding_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM memory_embeddings", [], |r| r.get(0))?;
        if embedding_count == 0 {
            return Ok(Vec::new());
        }

        // k_value: when tags/time filters are present we must scan more candidates
        // since membership is orthogonal to similarity. Cap at MAX_KNN_K.
        let k_value: i64 = if !q.tags.is_empty() || q.after.is_some() || q.before.is_some() {
            embedding_count.min(MAX_KNN_K as i64)
        } else {
            (q.limit as i64).min(MAX_KNN_K as i64)
        };

        // Build the dynamic filter clauses + bound params, in binding order:
        //   ?1 = query blob, ?2 = k, [tag params], [time params], [n_results].
        let blob = vec_to_blob(query_embedding);
        let mut params: Vec<rusqlite::types::Value> = Vec::new();
        params.push(rusqlite::types::Value::Blob(blob.to_vec()));
        params.push(rusqlite::types::Value::Integer(k_value));

        let superseded_filter = if q.include_superseded {
            ""
        } else {
            " AND (m.superseded_by IS NULL OR m.superseded_by = '')"
        };

        // Tag filter via LIKE on the comma-wrapped tags column. `any` joins the
        // per-tag clauses with OR, `all` with AND — pushed into SQL (like list()/
        // delete()) so the LIMIT operates on the correctly-filtered set, with no
        // post-truncation narrowing.
        let mut tag_conditions = String::new();
        if !q.tags.is_empty() {
            let clauses: Vec<String> = q
                .tags
                .iter()
                .map(|tag| {
                    params.push(rusqlite::types::Value::Text(format!(
                        "%,{},%",
                        escape_like(tag.trim())
                    )));
                    "(',' || REPLACE(m.tags, ' ', '') || ',') LIKE ? ESCAPE '\\'".to_string()
                })
                .collect();
            let joiner = if q.tag_match == TagMatch::All { " AND " } else { " OR " };
            tag_conditions = format!(" AND ({})", clauses.join(joiner));
        }

        // Time filters at SQL level (created_at >= / <=).
        let mut time_conditions = String::new();
        if let Some(after) = &q.after {
            if let Some(ts) = parse_iso_date_to_epoch(after) {
                time_conditions.push_str(" AND m.created_at >= ?");
                params.push(rusqlite::types::Value::Real(ts));
            }
        }
        if let Some(before) = &q.before {
            if let Some(ts) = parse_iso_date_to_epoch(before) {
                time_conditions.push_str(" AND m.created_at <= ?");
                params.push(rusqlite::types::Value::Real(ts));
            }
        }

        params.push(rusqlite::types::Value::Integer(q.limit as i64));

        let sql = format!(
            "SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                    m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                    e.distance
             FROM memories m
             INNER JOIN (
                 SELECT rowid, distance
                 FROM memory_embeddings
                 WHERE content_embedding MATCH ?1 AND k = ?2
             ) e ON m.id = e.rowid
             WHERE m.deleted_at IS NULL{superseded_filter}{tag_conditions}{time_conditions}
             ORDER BY e.distance
             LIMIT ?"
        );

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(params.iter()), |row| {
            Ok(row_to_hit(row))
        })?;
        let mut hits = Vec::new();
        for r in rows {
            hits.push(r??);
        }
        Ok(hits)
    }

    /// No-embedding filtered browse: used when a semantic-family search carries
    /// tag/time filters but no query string (so there's nothing to embed). Applies
    /// the same superseded + tag (ANY/ALL) + time clauses as [`Self::search_semantic`]
    /// minus the KNN join, ordered by `created_at` DESC. distance is fixed at 0.0
    /// (no similarity signal -> relevance 1.0 for every hit).
    pub fn search_filtered(&self, q: &SearchQuery) -> Result<Vec<SearchHit>> {
        let conn = self.conn.lock().unwrap();
        let mut params: Vec<rusqlite::types::Value> = Vec::new();

        let superseded_filter = if q.include_superseded {
            ""
        } else {
            " AND (m.superseded_by IS NULL OR m.superseded_by = '')"
        };

        let mut tag_conditions = String::new();
        if !q.tags.is_empty() {
            let clauses: Vec<String> = q
                .tags
                .iter()
                .map(|tag| {
                    params.push(rusqlite::types::Value::Text(format!(
                        "%,{},%",
                        escape_like(tag.trim())
                    )));
                    "(',' || REPLACE(m.tags, ' ', '') || ',') LIKE ? ESCAPE '\\'".to_string()
                })
                .collect();
            let joiner = if q.tag_match == TagMatch::All { " AND " } else { " OR " };
            tag_conditions = format!(" AND ({})", clauses.join(joiner));
        }

        let mut time_conditions = String::new();
        if let Some(after) = &q.after {
            if let Some(ts) = parse_iso_date_to_epoch(after) {
                time_conditions.push_str(" AND m.created_at >= ?");
                params.push(rusqlite::types::Value::Real(ts));
            }
        }
        if let Some(before) = &q.before {
            if let Some(ts) = parse_iso_date_to_epoch(before) {
                time_conditions.push_str(" AND m.created_at <= ?");
                params.push(rusqlite::types::Value::Real(ts));
            }
        }

        params.push(rusqlite::types::Value::Integer(q.limit as i64));

        let sql = format!(
            "SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                    m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                    0.0 AS distance
             FROM memories m
             WHERE m.deleted_at IS NULL{superseded_filter}{tag_conditions}{time_conditions}
             ORDER BY m.created_at DESC
             LIMIT ?"
        );

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(params.iter()), |row| Ok(row_to_hit(row)))?;
        let mut hits = Vec::new();
        for r in rows {
            hits.push(r??);
        }
        Ok(hits)
    }

    /// Dispatch on [`SearchMode`]. `Semantic`/`Hybrid`/`Ranked` use KNN; `Exact`
    /// uses a case-insensitive substring match (no FTS) for `mode=exact`.
    pub fn search(&self, q: &SearchQuery, query_embedding: Option<&[f32]>) -> Result<Vec<SearchHit>> {
        let mut hits = match q.mode {
            SearchMode::Exact => {
                let query = q.query.as_deref().unwrap_or("");
                self.search_fts(query, q)?
            }
            // Hybrid/Ranked fuse vector KNN + FTS5 BM25 via Reciprocal Rank Fusion
            // (Ranked aliases Hybrid for now). Semantic is pure vector KNN.
            SearchMode::Hybrid | SearchMode::Ranked => match query_embedding {
                Some(emb) => self.search_hybrid(q, emb)?,
                None => Vec::new(),
            },
            SearchMode::Semantic => match query_embedding {
                Some(emb) => self.search_semantic(q, emb)?,
                None => Vec::new(),
            },
        };
        // Optional recency reweight (off unless MCP_RECENCY_HALFLIFE_DAYS > 0).
        apply_recency(&mut hits, recency_halflife_days());
        Ok(hits)
    }

    /// Exact search (`mode: exact`): a case-insensitive substring match on content
    /// (`LIKE '%query%'`), NOT FTS5/BM25 (BM25 is only used by hybrid's fusion).
    /// Filters soft-deleted rows ONLY — superseded live rows are intentionally
    /// surfaced. Relevance is fixed at 1.0, ORDER BY created_at DESC.
    pub fn search_fts(&self, query: &str, q: &SearchQuery) -> Result<Vec<SearchHit>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        // NOTE: exact search filters ONLY `deleted_at IS NULL` — unlike the
        // semantic path, it does NOT filter superseded rows, so superseded live
        // rows are surfaced too. Do not add a superseded_filter here.
        let sql = "SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                    m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                    0.0 AS distance
             FROM memories m
             WHERE m.deleted_at IS NULL
               AND LOWER(m.content) LIKE '%' || LOWER(?1) || '%' ESCAPE '\\'
             ORDER BY m.created_at DESC
             LIMIT ?2".to_string();
        let mut stmt = conn.prepare(&sql)?;
        let pattern = escape_like(query);
        let rows = stmt.query_map(
            rusqlite::params![pattern, q.limit as i64],
            |row| {
                let mut hit = row_to_hit(row)?;
                // Exact matches get a fixed relevance of 1.0.
                hit.distance = 0.0;
                hit.relevance_score = 1.0;
                Ok(hit)
            },
        )?;
        let mut hits = Vec::new();
        for r in rows {
            hits.push(r?);
        }
        Ok(hits)
    }

    /// BM25 keyword search over the FTS5 trigram index (`memory_content_fts`) —
    /// the lexical arm of hybrid search. Phrase-quotes the query so arbitrary text
    /// cannot be parsed as FTS5 MATCH operators. Trigram needs >= 3 chars; shorter
    /// queries return empty. distance/relevance are placeholders — the caller fuses
    /// by RANK, not score.
    pub fn search_bm25(
        &self,
        query: &str,
        k: usize,
        include_superseded: bool,
    ) -> Result<Vec<SearchHit>> {
        let trimmed = query.trim();
        if trimmed.chars().count() < 3 {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        let phrase = format!("\"{}\"", trimmed.replace('"', "\"\""));
        let superseded_filter = if include_superseded {
            ""
        } else {
            " AND (m.superseded_by IS NULL OR m.superseded_by = '')"
        };
        let sql = format!(
            "SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                    m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                    0.0 AS distance
             FROM memories m
             INNER JOIN (
                 SELECT rowid, rank
                 FROM memory_content_fts
                 WHERE memory_content_fts MATCH ?1
                 ORDER BY rank
                 LIMIT ?2
             ) f ON m.id = f.rowid
             WHERE m.deleted_at IS NULL{superseded_filter}
             ORDER BY f.rank"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows =
            stmt.query_map(rusqlite::params![phrase, k as i64], |row| Ok(row_to_hit(row)))?;
        let mut hits = Vec::new();
        for r in rows {
            hits.push(r??);
        }
        Ok(hits)
    }

    /// Hybrid search: Reciprocal Rank Fusion of the semantic-KNN and BM25
    /// rankings. Over-fetches each arm to `limit*3`, fuses by RRF
    /// (score = Σ 1/(60 + rank)), dedups by content_hash, returns the top `limit`
    /// with relevance_score normalized to [0,1] (best = 1.0). The vector arm's
    /// `distance` is preserved on each hit for output.
    pub fn search_hybrid(&self, q: &SearchQuery, query_embedding: &[f32]) -> Result<Vec<SearchHit>> {
        let pool = q.limit.max(1).saturating_mul(3);
        let mut sem_q = q.clone();
        sem_q.limit = pool;
        let sem = self.search_semantic(&sem_q, query_embedding)?;
        let query = q.query.as_deref().unwrap_or("");
        let bm = self.search_bm25(query, pool, q.include_superseded)?;

        const RRF_K: f64 = 60.0;
        let mut score: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let mut hit_by_hash: std::collections::HashMap<String, SearchHit> =
            std::collections::HashMap::new();
        for (rank, h) in sem.iter().enumerate() {
            *score.entry(h.memory.content_hash.clone()).or_insert(0.0) +=
                1.0 / (RRF_K + rank as f64 + 1.0);
            hit_by_hash
                .entry(h.memory.content_hash.clone())
                .or_insert_with(|| h.clone());
        }
        for (rank, h) in bm.iter().enumerate() {
            *score.entry(h.memory.content_hash.clone()).or_insert(0.0) +=
                1.0 / (RRF_K + rank as f64 + 1.0);
            hit_by_hash
                .entry(h.memory.content_hash.clone())
                .or_insert_with(|| h.clone());
        }

        let mut fused: Vec<(String, f64)> = score.into_iter().collect();
        // Sort by fused score desc, then content_hash asc as a deterministic
        // tiebreaker (RRF ties are common on small corpora; HashMap iteration
        // order is otherwise random across runs).
        fused.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        fused.truncate(q.limit);
        let best = fused.first().map(|(_, s)| *s).unwrap_or(1.0).max(f64::MIN_POSITIVE);

        let mut out = Vec::new();
        for (hash, s) in fused {
            if let Some(mut h) = hit_by_hash.remove(&hash) {
                h.relevance_score = (s / best).clamp(0.0, 1.0);
                out.push(h);
            }
        }
        Ok(out)
    }

    /// Sample stored embedding-model provenance; returns `Some(found)` if any
    /// stamped memory was embedded with a model other than `configured` (None if
    /// all match or none are stamped yet — legacy rows carry no stamp).
    pub fn embedding_model_mismatch(&self, configured: &str) -> Option<String> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT DISTINCT json_extract(metadata, '$.embedding_model') AS m \
                 FROM memories \
                 WHERE deleted_at IS NULL \
                   AND json_extract(metadata, '$.embedding_model') IS NOT NULL \
                 LIMIT 50",
            )
            .ok()?;
        let mut rows = stmt.query_map([], |r| r.get::<_, String>(0)).ok()?;
        rows.find_map(|res| res.ok().filter(|m| m.as_str() != configured))
    }

    /// Paginated browse (`memory_list`). Filters soft-deleted + superseded,
    /// optional tag/type filters, ORDER BY created_at DESC, LIMIT/OFFSET.
    pub fn list(&self, q: &ListQuery) -> Result<ListPage> {
        let conn = self.conn.lock().unwrap();
        let offset = q.page.saturating_sub(1).saturating_mul(q.page_size);

        // Build shared WHERE clause + params (deleted_at IS NULL always).
        // NOTE: list does NOT filter superseded rows — only soft-deleted.
        let mut conditions: Vec<String> = vec!["m.deleted_at IS NULL".to_string()];
        let mut params: Vec<rusqlite::types::Value> = Vec::new();

        if let Some(mt) = &q.memory_type {
            conditions.push("m.memory_type = ?".to_string());
            params.push(rusqlite::types::Value::Text(mt.clone()));
        }
        if !q.tags.is_empty() {
            let joiner = if q.tag_match == TagMatch::All { " AND " } else { " OR " };
            let clauses: Vec<String> = q
                .tags
                .iter()
                .map(|tag| {
                    params.push(rusqlite::types::Value::Text(format!(
                        "%,{},%",
                        escape_like(tag.trim())
                    )));
                    "(',' || REPLACE(m.tags, ' ', '') || ',') LIKE ? ESCAPE '\\'".to_string()
                })
                .collect();
            conditions.push(format!("({})", clauses.join(joiner)));
        }
        let where_clause = conditions.join(" AND ");

        // total count over the same filtered set.
        let count_sql = format!("SELECT COUNT(*) FROM memories m WHERE {where_clause}");
        let total: i64 = conn.query_row(
            &count_sql,
            params_from_iter(params.iter()),
            |r| r.get(0),
        )?;

        // page slice
        let mut page_params = params.clone();
        page_params.push(rusqlite::types::Value::Integer(q.page_size as i64));
        page_params.push(rusqlite::types::Value::Integer(offset as i64));
        let list_sql = format!(
            "SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                    m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso
             FROM memories m
             WHERE {where_clause}
             ORDER BY m.created_at DESC
             LIMIT ? OFFSET ?"
        );
        let mut stmt = conn.prepare(&list_sql)?;
        let rows = stmt.query_map(params_from_iter(page_params.iter()), row_to_memory)?;
        let mut memories = Vec::new();
        for r in rows {
            memories.push(r?);
        }

        let total = total as usize;
        let total_pages = if q.page_size > 0 {
            total.div_ceil(q.page_size)
        } else {
            0
        };
        let has_more = offset.saturating_add(q.page_size) < total;

        Ok(ListPage {
            memories,
            page: q.page,
            page_size: q.page_size,
            total,
            total_pages,
            has_more,
        })
    }

    /// Soft-delete (`memory_delete`). NEVER hard-DELETE on user delete.
    ///   * content_hash set -> UPDATE ... SET deleted_at=now WHERE content_hash=? AND deleted_at IS NULL.
    ///   * else filter by tags/time -> UPDATE matching rows.
    ///   * no filters at all -> InvalidArg (mass-delete guard).
    ///   * dry_run -> count + collect hashes, no UPDATE.
    /// Returns a [`DeleteOutcome`] (count, matched hashes, storage message, single-hash flag).
    pub fn delete(&self, q: &DeleteQuery) -> Result<DeleteOutcome> {
        // tag_match is restricted to any/all by its enum, so no validation needed.

        // Case 1: single hash (ignores all other filters).
        if let Some(hash) = &q.content_hash {
            let conn = self.conn.lock().unwrap();
            if q.dry_run {
                // Dry-run: existence check on a LIVE row.
                let exists: Option<String> = conn
                    .query_row(
                        "SELECT content_hash FROM memories WHERE content_hash = ?1 AND deleted_at IS NULL",
                        [hash],
                        |r| r.get(0),
                    )
                    .ok();
                return match exists {
                    Some(_) => Ok(DeleteOutcome {
                        deleted_count: 1,
                        deleted_hashes: vec![hash.clone()],
                        message: format!("Would delete 1 memory with hash: {hash}"),
                        single_hash: Some(hash.clone()),
                    }),
                    None => Err(MemoryError::NotFound(format!("Memory not found: {hash}"))),
                };
            }

            // Resolve the id (to delete its embedding + graph edges), then soft-delete.
            let id: Option<i64> = conn
                .query_row(
                    "SELECT id FROM memories WHERE content_hash = ?1 AND deleted_at IS NULL",
                    [hash],
                    |r| r.get(0),
                )
                .ok();
            let Some(id) = id else {
                return Err(MemoryError::NotFound(format!(
                    "Memory with hash {hash} not found"
                )));
            };
            // The embedding delete can fail on some sqlite-vec versions; tolerate
            // it (the soft-delete still wins) but log it rather than swallow it.
            if let Err(e) = conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?1", [id]) {
                tracing::warn!(hash = %hash, error = %e, "failed to delete embedding row during soft-delete");
            }
            if let Err(e) = conn.execute(
                "DELETE FROM memory_graph WHERE source_hash = ?1 OR target_hash = ?1",
                [hash],
            ) {
                tracing::warn!(hash = %hash, error = %e, "failed to delete graph edges during soft-delete");
            }
            let changed = conn.execute(
                "UPDATE memories SET deleted_at = ?1 WHERE content_hash = ?2 AND deleted_at IS NULL",
                rusqlite::params![now_epoch(), hash],
            )?;
            if changed == 0 {
                return Err(MemoryError::NotFound(format!(
                    "Memory with hash {hash} not found"
                )));
            }
            return Ok(DeleteOutcome {
                deleted_count: 1,
                deleted_hashes: vec![hash.clone()],
                message: format!("Successfully deleted memory {hash}"),
                single_hash: Some(hash.clone()),
            });
        }

        // Case 2: no filters at all -> mass-delete guard.
        if q.tags.is_empty() && q.before.is_none() && q.after.is_none() {
            return Err(MemoryError::InvalidArg(
                "At least one filter required (content_hash, tags, before, or after)".into(),
            ));
        }

        // Case 3: filter-based deletion (tags AND/OR time range), combined with AND.
        let conn = self.conn.lock().unwrap();
        let mut conditions: Vec<String> = vec!["deleted_at IS NULL".to_string()];
        let mut sel_params: Vec<rusqlite::types::Value> = Vec::new();

        if !q.tags.is_empty() {
            let joiner = if q.tag_match == TagMatch::All { " AND " } else { " OR " };
            let clauses: Vec<String> = q
                .tags
                .iter()
                .map(|tag| {
                    sel_params.push(rusqlite::types::Value::Text(format!(
                        "%,{},%",
                        escape_like(tag.trim())
                    )));
                    "(',' || REPLACE(tags, ' ', '') || ',') LIKE ? ESCAPE '\\'".to_string()
                })
                .collect();
            conditions.push(format!("({})", clauses.join(joiner)));
        }
        if let Some(after) = &q.after {
            if let Some(ts) = parse_iso_date_to_epoch(after) {
                conditions.push("created_at >= ?".to_string());
                sel_params.push(rusqlite::types::Value::Real(ts));
            }
        }
        if let Some(before) = &q.before {
            if let Some(ts) = parse_iso_date_to_epoch(before) {
                conditions.push("created_at <= ?".to_string());
                sel_params.push(rusqlite::types::Value::Real(ts));
            }
        }
        let where_clause = conditions.join(" AND ");

        // Collect matching (id, content_hash) up front.
        let sel_sql = format!("SELECT id, content_hash FROM memories WHERE {where_clause}");
        let mut stmt = conn.prepare(&sel_sql)?;
        let rows = stmt.query_map(params_from_iter(sel_params.iter()), |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?;
        let mut matched: Vec<(i64, String)> = Vec::new();
        for r in rows {
            matched.push(r?);
        }
        drop(stmt);

        let hashes: Vec<String> = matched.iter().map(|(_, h)| h.clone()).collect();
        let n = matched.len();

        // Dry-run filter delete: report "Would delete N memories"; the handler then
        // appends "\n\nWould delete N memories\nHashes: ...".
        if q.dry_run {
            return Ok(DeleteOutcome {
                deleted_count: n,
                deleted_hashes: hashes,
                message: format!("Would delete {n} memories"),
                single_hash: None,
            });
        }

        // Soft-delete each match (+ its embedding + graph edges).
        let now = now_epoch();
        for (id, hash) in &matched {
            if let Err(e) = conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?1", [*id]) {
                tracing::warn!(hash = %hash, error = %e, "failed to delete embedding row during bulk soft-delete");
            }
            if let Err(e) = conn.execute(
                "DELETE FROM memory_graph WHERE source_hash = ?1 OR target_hash = ?1",
                [hash],
            ) {
                tracing::warn!(hash = %hash, error = %e, "failed to delete graph edges during bulk soft-delete");
            }
            conn.execute(
                "UPDATE memories SET deleted_at = ?1 WHERE id = ?2 AND deleted_at IS NULL",
                rusqlite::params![now, id],
            )?;
        }

        // Real filter delete base message. Callers grep this text, so the exact
        // wording matters:
        //   * tag-only ANY-match: "Successfully deleted N memories matching M tag(s)"
        //     when N>0, else "No memories found matching any of the M tags".
        //   * every other filter combo (time-only, tags+time, ALL-match):
        //     "Successfully deleted N memories".
        // The handler appends "\n\nDeleted N memories" in all cases.
        let tag_only_any = !q.tags.is_empty()
            && q.before.is_none()
            && q.after.is_none()
            && q.tag_match == TagMatch::Any;
        let message = if tag_only_any {
            if n > 0 {
                format!("Successfully deleted {n} memories matching {} tag(s)", q.tags.len())
            } else {
                format!("No memories found matching any of the {} tags", q.tags.len())
            }
        } else {
            format!("Successfully deleted {n} memories")
        };

        Ok(DeleteOutcome {
            deleted_count: n,
            deleted_hashes: hashes,
            message,
            single_hash: None,
        })
    }

    /// Read a live memory's `(tags, memory_type)` for the versioned-update
    /// carry-over (so a content-only new version keeps the parent's tags/type).
    /// Returns `(vec![], None)` if the hash isn't a live row.
    pub fn get_tags_type(&self, content_hash: &str) -> Result<(Vec<String>, Option<String>)> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT tags, memory_type FROM memories WHERE content_hash = ?1 AND deleted_at IS NULL",
                [content_hash],
                |r| Ok((r.get::<_, Option<String>>(0)?, r.get::<_, Option<String>>(1)?)),
            )
            .optional()?;
        Ok(match row {
            Some((tags, mt)) => (tags.as_deref().map(tags_from_csv).unwrap_or_default(), mt),
            None => (Vec::new(), None),
        })
    }

    /// Mark a memory superseded by another (sets the `superseded_by` COLUMN that
    /// every search path filters on). Called by the tool-layer versioned update
    /// after it inserts the new row. No-op if `old_hash` isn't a live row.
    pub fn mark_superseded(&self, old_hash: &str, new_hash: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = now_epoch();
        conn.execute(
            "UPDATE memories SET superseded_by = ?1, updated_at = ?2, updated_at_iso = ?3
             WHERE content_hash = ?4 AND deleted_at IS NULL",
            rusqlite::params![new_hash, now, epoch_to_iso(now), old_hash],
        )?;
        Ok(())
    }

    /// Update metadata/tags/type for a memory (`memory_update`). Non-versioned
    /// path updates in place + bumps updated_at/_iso. The versioned path (when
    /// `updates.content` is set) is handled by the tool layer, which inserts the new
    /// row and then calls [`Self::mark_superseded`] on the old one.
    /// Returns a human-readable message (the handler returns it verbatim).
    pub fn update(&self, content_hash: &str, updates: &serde_json::Value, _versioned: bool) -> Result<String> {
        // Versioned supersede is handled at the tool layer (it needs an embedding
        // for the new content); this storage method does the in-place metadata path
        // and preserves timestamps unless a structural field changes (see below).
        let conn = self.conn.lock().unwrap();

        // Read current row (live only).
        let row = conn
            .query_row(
                "SELECT content, tags, memory_type, metadata, created_at, created_at_iso,
                        updated_at, updated_at_iso
                 FROM memories WHERE content_hash = ?1 AND deleted_at IS NULL",
                [content_hash],
                |r| {
                    Ok((
                        r.get::<_, String>(1)?,          // tags
                        r.get::<_, Option<String>>(2)?,  // memory_type
                        r.get::<_, Option<String>>(3)?,  // metadata
                        r.get::<_, Option<f64>>(4)?,     // created_at
                        r.get::<_, Option<String>>(5)?,  // created_at_iso
                        r.get::<_, Option<f64>>(6)?,     // updated_at
                        r.get::<_, Option<String>>(7)?,  // updated_at_iso
                    ))
                },
            )
            .optional()?;
        let Some((cur_tags, cur_type, cur_meta, created_at, created_at_iso, cur_updated_at, cur_updated_at_iso)) =
            row
        else {
            return Err(MemoryError::NotFound(format!(
                "Memory with hash {content_hash} not found"
            )));
        };

        let obj = updates.as_object().ok_or_else(|| {
            MemoryError::InvalidArg("updates must be a dictionary".into())
        })?;

        // Apply updates.
        let mut new_tags = cur_tags;
        let mut new_type = cur_type;
        let mut new_meta: serde_json::Map<String, serde_json::Value> = match cur_meta {
            Some(s) if !s.is_empty() => serde_json::from_str(&s).unwrap_or_else(|e| {
                // Corrupt stored metadata falls back to an empty map, but the
                // corruption is LOGGED (not silently dropped).
                tracing::warn!(
                    content_hash = %content_hash,
                    error = %e,
                    data = %s.chars().take(100).collect::<String>(),
                    "stored metadata JSON parse failed; falling back to empty map"
                );
                serde_json::Map::new()
            }),
            _ => serde_json::Map::new(),
        };

        let mut updated_fields: Vec<&str> = Vec::new();
        if let Some(t) = obj.get("tags") {
            // Accept array OR CSV-string form, then normalize_tags (lowercase +
            // trim + comma->hyphen + dedup) before persisting.
            let raw: Vec<String> = match t {
                serde_json::Value::Array(arr) => arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect(),
                serde_json::Value::String(s) => crate::models::split_tag_string(s),
                _ => {
                    return Err(MemoryError::InvalidArg(
                        "Tags must be provided as a list of strings".into(),
                    ))
                }
            };
            new_tags = crate::models::normalize_tags(&raw).join(",");
            updated_fields.push("tags");
        }
        if let Some(mt) = obj.get("memory_type") {
            new_type = mt.as_str().map(|s| s.to_string());
            updated_fields.push("memory_type");
        }
        if let Some(m) = obj.get("metadata") {
            let mo = m.as_object().ok_or_else(|| {
                MemoryError::InvalidArg("Metadata must be provided as a dictionary".into())
            })?;
            for (k, v) in mo {
                new_meta.insert(k.clone(), v.clone());
            }
            updated_fields.push("custom_metadata");
        }
        // Other custom (non-protected) fields land in metadata.
        const PROTECTED: &[&str] = &[
            "content", "content_hash", "tags", "memory_type", "metadata",
            "embedding", "created_at", "created_at_iso", "updated_at", "updated_at_iso",
        ];
        for (k, v) in obj {
            if !PROTECTED.contains(&k.as_str()) {
                new_meta.insert(k.clone(), v.clone());
            }
        }

        // Timestamps: a structural change (tags / memory_type / content) advances
        // updated_at; a pure-metadata change preserves it.
        let now = now_epoch();
        let now_iso = epoch_to_iso(now);
        let structural = obj.contains_key("tags")
            || obj.contains_key("memory_type")
            || obj.contains_key("content");
        let (updated_at, updated_at_iso) = if !structural {
            (
                cur_updated_at.unwrap_or(now),
                cur_updated_at_iso.unwrap_or_else(|| now_iso.clone()),
            )
        } else {
            (now, now_iso.clone())
        };

        conn.execute(
            "UPDATE memories SET
                tags = ?1, memory_type = ?2, metadata = ?3,
                updated_at = ?4, updated_at_iso = ?5,
                created_at = ?6, created_at_iso = ?7
             WHERE content_hash = ?8 AND deleted_at IS NULL",
            rusqlite::params![
                new_tags,
                new_type,
                serde_json::to_string(&serde_json::Value::Object(new_meta))?,
                updated_at,
                updated_at_iso,
                created_at,
                created_at_iso,
                content_hash,
            ],
        )?;

        updated_fields.push("updated_at");
        Ok(format!("Updated fields: {}", updated_fields.join(", ")))
    }

    /// DB statistics for `memory_health.statistics`.
    pub fn stats(&self, embedding_model: &str) -> Result<DbStats> {
        let conn = self.conn.lock().unwrap();

        let total: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL",
            [],
            |r| r.get(0),
        )?;

        // DB file size — best-effort.
        let db_path: String = conn.query_row("PRAGMA database_list", [], |r| r.get(2)).unwrap_or_default();
        let database_size_bytes = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

        Ok(DbStats {
            total_memories: total as usize,
            database_size_bytes,
            embedding_model: embedding_model.to_string(),
            embedding_dimension: self.embedding_dim,
        })
    }

    /// `memory_graph action=connected`: BFS over `memory_graph` edges from `hash`
    /// out to `max_hops`. Edges are populated by the weekly Rust consolidator
    /// (the `consolidate` subcommand); if the table is empty this returns empty.
    pub fn graph_connected(&self, hash: &str, max_hops: usize) -> Result<Vec<(String, usize)>> {
        if hash.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        // Recursive CTE for multi-hop BFS over memory_graph, treating edges as
        // undirected (either endpoint matches). Cycle prevention via the
        // comma-delimited `path` string + instr().
        let sql = "
WITH RECURSIVE connected_memories(hash, distance, path) AS (
    SELECT ?1, 0, ?2

    UNION ALL

    SELECT
        CASE WHEN cm.hash = mg.source_hash THEN mg.target_hash ELSE mg.source_hash END,
        cm.distance + 1,
        cm.path || CASE WHEN cm.hash = mg.source_hash THEN mg.target_hash ELSE mg.source_hash END || ','
    FROM connected_memories cm
    JOIN memory_graph mg ON (cm.hash = mg.source_hash OR cm.hash = mg.target_hash)
    WHERE
        cm.distance < ?3
        AND instr(cm.path, ',' || (CASE WHEN cm.hash = mg.source_hash THEN mg.target_hash ELSE mg.source_hash END) || ',') = 0
)
SELECT DISTINCT hash, distance
FROM connected_memories
WHERE distance > 0
ORDER BY distance, hash
";
        let path_seed = format!(",{hash},");
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(
            rusqlite::params![hash, path_seed, max_hops as i64],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as usize)),
        )?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// `memory_graph action=path`: shortest path (node sequence `[from, .., to]`)
    /// over association edges within `max_hops`. Empty if unreachable. BFS via a
    /// recursive CTE that carries the path string (shortest = lowest distance).
    pub fn graph_path(&self, from: &str, to: &str, max_hops: usize) -> Result<Vec<String>> {
        if from.is_empty() || to.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        if from == to {
            // Degenerate self-path: a one-node path is valid only if the node
            // actually exists as a live memory (otherwise be consistent with the
            // from != to branch, which returns not-found for a missing node).
            let exists = conn
                .query_row(
                    "SELECT 1 FROM memories WHERE content_hash = ?1 AND deleted_at IS NULL",
                    rusqlite::params![from],
                    |_| Ok(()),
                )
                .optional()?
                .is_some();
            return Ok(if exists { vec![from.to_string()] } else { Vec::new() });
        }
        let sql = "
WITH RECURSIVE paths(hash, distance, path) AS (
    SELECT ?1, 0, ?2
    UNION ALL
    SELECT
        CASE WHEN p.hash = mg.source_hash THEN mg.target_hash ELSE mg.source_hash END,
        p.distance + 1,
        p.path || CASE WHEN p.hash = mg.source_hash THEN mg.target_hash ELSE mg.source_hash END || ','
    FROM paths p
    JOIN memory_graph mg ON (p.hash = mg.source_hash OR p.hash = mg.target_hash)
    WHERE p.distance < ?3
      AND instr(p.path, ',' || (CASE WHEN p.hash = mg.source_hash THEN mg.target_hash ELSE mg.source_hash END) || ',') = 0
)
SELECT path FROM paths WHERE hash = ?4 ORDER BY distance, path LIMIT 1
";
        let seed = format!(",{from},");
        let path_str: Option<String> = conn
            .query_row(sql, rusqlite::params![from, seed, max_hops as i64, to], |r| {
                r.get(0)
            })
            .optional()?;
        Ok(match path_str {
            Some(s) => s
                .split(',')
                .filter(|t| !t.is_empty())
                .map(|t| t.to_string())
                .collect(),
            None => Vec::new(),
        })
    }

    /// `memory_graph action=subgraph`: nodes within `max_hops` of `hash`
    /// (including `hash` at distance 0) plus every edge among them (symmetric
    /// edges deduped to `source < target`).
    pub fn graph_subgraph(
        &self,
        hash: &str,
        max_hops: usize,
    ) -> Result<(Vec<GraphNode>, Vec<GraphEdge>)> {
        if hash.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut nodes = self.graph_connected(hash, max_hops)?;
        let mut node_set: std::collections::HashSet<String> =
            nodes.iter().map(|(h, _)| h.clone()).collect();
        node_set.insert(hash.to_string());

        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT source_hash, target_hash, similarity FROM memory_graph WHERE source_hash < target_hash",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, f64>(2)?,
            ))
        })?;
        let mut edges = Vec::new();
        for row in rows {
            let (s, t, sim) = row?;
            if node_set.contains(&s) && node_set.contains(&t) {
                edges.push((s, t, sim));
            }
        }
        nodes.insert(0, (hash.to_string(), 0));
        Ok((nodes, edges))
    }

    /// `memory_graph action=infer`: link-prediction candidates — memories NOT
    /// directly connected to `hash` but sharing common neighbors, ranked by the
    /// number of shared neighbors (descending). Pure graph, no embeddings.
    pub fn graph_infer(&self, hash: &str, limit: usize) -> Result<Vec<(String, usize)>> {
        if hash.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        let sql = "
WITH nbr(h) AS (
    SELECT target_hash FROM memory_graph WHERE source_hash = ?1
    UNION
    SELECT source_hash FROM memory_graph WHERE target_hash = ?1
),
two_hop(cand, bridge) AS (
    SELECT
        CASE WHEN mg.source_hash IN (SELECT h FROM nbr) THEN mg.target_hash ELSE mg.source_hash END,
        CASE WHEN mg.source_hash IN (SELECT h FROM nbr) THEN mg.source_hash ELSE mg.target_hash END
    FROM memory_graph mg
    WHERE mg.source_hash IN (SELECT h FROM nbr) OR mg.target_hash IN (SELECT h FROM nbr)
)
SELECT cand, COUNT(DISTINCT bridge) AS shared
FROM two_hop
WHERE cand != ?1 AND cand NOT IN (SELECT h FROM nbr)
GROUP BY cand
ORDER BY shared DESC, cand
LIMIT ?2
";
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(rusqlite::params![hash, limit as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as usize))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// `memory_graph action=suggest`: semantic nearest neighbors of `hash` by the
    /// STORED embedding (no embedding-server call) that are NOT already directly
    /// connected, ranked by cosine mapped to `[0,1]`. Proposes new associations.
    pub fn graph_suggest(&self, hash: &str, limit: usize) -> Result<Vec<(String, f64)>> {
        if hash.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        let seed_blob: Option<Vec<u8>> = conn
            .query_row(
                "SELECT e.content_embedding FROM memories m \
                 JOIN memory_embeddings e ON e.rowid = m.id \
                 WHERE m.content_hash = ?1 AND m.deleted_at IS NULL",
                rusqlite::params![hash],
                |r| r.get(0),
            )
            .optional()?;
        let Some(seed_blob) = seed_blob else {
            return Ok(Vec::new());
        };
        let Some(seed) = normalize_vec(&blob_to_vec(&seed_blob)?) else {
            return Ok(Vec::new());
        };

        // Direct neighbors to exclude.
        let mut neighbors: std::collections::HashSet<String> = std::collections::HashSet::new();
        {
            let mut nstmt = conn.prepare(
                "SELECT target_hash FROM memory_graph WHERE source_hash = ?1 \
                 UNION SELECT source_hash FROM memory_graph WHERE target_hash = ?1",
            )?;
            let nrows = nstmt.query_map(rusqlite::params![hash], |r| r.get::<_, String>(0))?;
            for n in nrows {
                neighbors.insert(n?);
            }
        }

        let mut cands: Vec<(String, f64)> = Vec::new();
        {
            let mut cstmt = conn.prepare(
                "SELECT m.content_hash, e.content_embedding FROM memories m \
                 JOIN memory_embeddings e ON e.rowid = m.id WHERE m.deleted_at IS NULL",
            )?;
            let crows =
                cstmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
            for row in crows {
                let (h, blob) = row?;
                if h == hash || neighbors.contains(&h) {
                    continue;
                }
                let Ok(v) = blob_to_vec(&blob) else { continue };
                let Some(nv) = normalize_vec(&v) else { continue };
                let cos: f64 = seed
                    .iter()
                    .zip(&nv)
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum();
                cands.push((h, (cos.clamp(-1.0, 1.0) + 1.0) / 2.0));
            }
        }
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(limit);
        Ok(cands)
    }
}

/// Map a search-result row (10 cols, last = distance) to a [`SearchHit`].
fn row_to_hit(row: &rusqlite::Row) -> rusqlite::Result<SearchHit> {
    let memory = build_memory(row)?;
    let distance: f64 = row.get(9)?;
    Ok(SearchHit {
        memory,
        distance,
        relevance_score: SearchHit::relevance_from_distance(distance),
    })
}

/// Map a list-result row (9 cols) to a [`Memory`].
fn row_to_memory(row: &rusqlite::Row) -> rusqlite::Result<Memory> {
    build_memory(row)
}

/// Shared row->Memory builder for the first 9 columns
/// (content_hash, content, tags, memory_type, metadata, created_at, updated_at,
///  created_at_iso, updated_at_iso).
fn build_memory(row: &rusqlite::Row) -> rusqlite::Result<Memory> {
    let content_hash: String = row.get(0)?;
    let content: String = row.get(1)?;
    let tags_str: Option<String> = row.get(2)?;
    let memory_type: Option<String> = row.get(3)?;
    let metadata_str: Option<String> = row.get(4)?;
    let created_at: Option<f64> = row.get(5)?;
    let updated_at: Option<f64> = row.get(6)?;
    let created_at_iso: Option<String> = row.get(7)?;
    let updated_at_iso: Option<String> = row.get(8)?;

    let tags = tags_str.as_deref().map(tags_from_csv).unwrap_or_default();
    let metadata: serde_json::Value = match metadata_str.as_deref().filter(|s| !s.is_empty()) {
        // Log corrupt metadata instead of dropping it silently, then fall back to
        // an empty object.
        Some(s) => serde_json::from_str(s).unwrap_or_else(|e| {
            tracing::warn!(
                content_hash = %content_hash,
                error = %e,
                data = %s.chars().take(100).collect::<String>(),
                "stored metadata JSON parse failed; falling back to empty object"
            );
            serde_json::Value::Object(serde_json::Map::new())
        }),
        None => serde_json::Value::Object(serde_json::Map::new()),
    };

    let created_at = created_at.unwrap_or(0.0);
    let updated_at = updated_at.unwrap_or(created_at);
    Ok(Memory {
        content_hash,
        content,
        tags,
        memory_type,
        metadata,
        created_at,
        updated_at,
        created_at_iso: created_at_iso.unwrap_or_default(),
        updated_at_iso: updated_at_iso.unwrap_or_default(),
    })
}

/// Serialize a `&[f32]` to the raw little-endian byte blob sqlite-vec stores
/// (no header; dim 2560 -> 10240 bytes). On little-endian hosts the cast is a
/// no-op reinterpret. This LE layout is part of the on-disk format contract.
pub fn vec_to_blob(vec: &[f32]) -> &[u8] {
    bytemuck::cast_slice(vec)
}

/// Deserialize a stored blob back to `Vec<f32>`. Validates `blob.len() % 4 == 0`.
/// Used by the consolidator (reads stored embeddings) and the round-trip test.
pub fn blob_to_vec(blob: &[u8]) -> Result<Vec<f32>> {
    if !blob.len().is_multiple_of(4) {
        return Err(MemoryError::Other(format!(
            "embedding blob length {} not a multiple of 4",
            blob.len()
        )));
    }
    // bytemuck requires 4-byte alignment; blob from SQLite may not be aligned,
    // so build the Vec<f32> explicitly from LE chunks (correct on any platform).
    Ok(blob
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// L2-normalize a vector; `None` for a zero/non-finite vector. Used by
/// `graph_suggest` to cosine-rank stored embeddings.
fn normalize_vec(v: &[f32]) -> Option<Vec<f32>> {
    let norm: f64 = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    Some(v.iter().map(|x| (*x as f64 / norm) as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{open_test_storage, seeded_embed, store_mem};

    /// A fresh DB is created at the configured embedding width, not a hardcoded 2560.
    #[test]
    fn fresh_db_honors_configured_dim() {
        let mut cfg = crate::test_util::test_config(crate::config::Scope::Project);
        cfg.embed.embedding_dim = 384;
        let s = Storage::open(&cfg).expect("open with dim 384");
        assert_eq!(s.embedding_dim(), 384);
    }

    // ––– helpers –––

    fn insert_graph_edge(storage: &Storage, source: &str, target: &str, similarity: f64) {
        let conn = storage.conn.lock().unwrap();
        // Insert both directions, matching the real consolidator.
        conn.execute(
            "INSERT OR REPLACE INTO memory_graph (source_hash, target_hash, similarity, \
             connection_types, metadata, created_at, relationship_type) \
             VALUES (?1, ?2, ?3, '[\"semantic\"]', '{}', ?4, 'related')",
            rusqlite::params![source, target, similarity, 0.0_f64],
        )
        .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO memory_graph (source_hash, target_hash, similarity, \
             connection_types, metadata, created_at, relationship_type) \
             VALUES (?1, ?2, ?3, '[\"semantic\"]', '{}', ?4, 'related')",
            rusqlite::params![target, source, similarity, 0.0_f64],
        )
        .unwrap();
    }

    fn sq(query: &str, mode: SearchMode, limit: usize) -> SearchQuery {
        SearchQuery {
            query: Some(query.to_string()),
            mode,
            tags: vec![],
            tag_match: TagMatch::Any,
            after: None,
            before: None,
            limit,
            include_superseded: false,
        }
    }

    fn lq(page: usize, page_size: usize) -> ListQuery {
        ListQuery {
            page,
            page_size,
            tags: vec![],
            tag_match: TagMatch::Any,
            memory_type: None,
        }
    }

    // ––– existing tests (unchanged) –––

    #[test]
    fn blob_roundtrip_le() {
        let v: Vec<f32> = vec![0.0, 1.0, -1.5, std::f32::consts::PI, 2560.0];
        let blob = vec_to_blob(&v);
        assert_eq!(blob.len(), 20);
        assert_eq!(&blob[0..4], &0.0f32.to_le_bytes());
        let back = blob_to_vec(blob).unwrap();
        assert_eq!(back, v);
    }

    #[test]
    fn blob_rejects_misaligned_len() {
        assert!(blob_to_vec(&[1, 2, 3]).is_err());
    }

    #[test]
    fn escape_like_order() {
        assert_eq!(escape_like("a_b%c\\d"), "a\\_b\\%c\\\\d");
    }

    #[test]
    fn parse_iso_date() {
        assert_eq!(parse_iso_date_to_epoch("2024-06-03"), Some(1717372800.0));
        assert!(parse_iso_date_to_epoch("not-a-date").is_none());
    }

    // ––– store –––

    #[test]
    fn store_basic_roundtrip() {
        let s = open_test_storage();
        let h = store_mem(&s, "hello world", 1, &[], None);
        let hits = s
            .search(&sq("hello", SearchMode::Semantic, 5), Some(&seeded_embed(1)))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory.content, "hello world");
        assert_eq!(hits[0].memory.content_hash, h);
    }

    #[test]
    fn store_exact_duplicate() {
        let s = open_test_storage();
        let e = seeded_embed(1);
        let args = crate::test_util::str_mem("dup", &[], None);
        let r1 = s.store(&args, &e).unwrap();
        assert!(matches!(r1, StoreOutcome::Stored { .. }));
        let r2 = s.store(&args, &e).unwrap();
        assert!(matches!(r2, StoreOutcome::Duplicate { .. }));
    }

    #[test]
    fn store_with_tags() {
        let s = open_test_storage();
        store_mem(&s, "tagged", 1, &["python", "reference"], None);
        let mut q = sq("tagged", SearchMode::Semantic, 5);
        q.tags = vec!["python".into()];
        let hits = s.search(&q, Some(&seeded_embed(1))).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].memory.tags.contains(&"python".to_string()));
    }

    #[test]
    fn store_with_type() {
        let s = open_test_storage();
        store_mem(&s, "typed-mem", 1, &[], Some("insight"));
        let page = s
            .list(&ListQuery {
                page: 1,
                page_size: 10,
                tags: vec![],
                tag_match: TagMatch::Any,
                memory_type: Some("insight".into()),
            })
            .unwrap();
        assert_eq!(page.memories.len(), 1);
        assert_eq!(page.memories[0].memory_type.as_deref(), Some("insight"));
    }

    #[test]
    fn store_multiple() {
        let s = open_test_storage();
        store_mem(&s, "a", 1, &[], None);
        store_mem(&s, "b", 2, &[], None);
        store_mem(&s, "c", 3, &[], None);
        let hits = s
            .search(&sq("a", SearchMode::Semantic, 10), Some(&seeded_embed(1)))
            .unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn store_tombstone_purge() {
        let s = open_test_storage();
        let content = "tombstone-test";
        let e = seeded_embed(1);
        let args = crate::test_util::str_mem(content, &[], None);
        let h1 = match s.store(&args, &e).unwrap() {
            StoreOutcome::Stored { content_hash } => content_hash,
            other => panic!("expected Stored, got {other:?}"),
        };
        // soft-delete
        s.delete(&DeleteQuery {
            content_hash: Some(h1.clone()),
            tags: vec![],
            tag_match: TagMatch::Any,
            before: None,
            after: None,
            dry_run: false,
        })
        .unwrap();
        // re-store same content → should succeed (tombstone purged)
        let h2 = match s.store(&args, &e).unwrap() {
            StoreOutcome::Stored { content_hash } => content_hash,
            other => panic!("expected Stored after tombstone purge, got {other:?}"),
        };
        // Same content → same hash, but re-store should SUCCEED (not duplicate).
        assert_eq!(h1, h2, "tombstone-purged re-store should produce same hash");
        assert!(matches!(
            s.store(&args, &e).unwrap(),
            StoreOutcome::Duplicate { .. }
        ), "third store of same content should now be exact duplicate");
    }

    // ––– search: semantic –––

    #[test]
    fn search_semantic_empty_table() {
        let s = open_test_storage();
        let hits = s
            .search(&sq("x", SearchMode::Semantic, 5), Some(&seeded_embed(1)))
            .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn search_semantic_ranked_by_relevance() {
        let s = open_test_storage();
        store_mem(&s, "far", 10, &[], None);
        store_mem(&s, "close", 1, &[], None);
        let hits = s
            .search(&sq("close", SearchMode::Semantic, 5), Some(&seeded_embed(1)))
            .unwrap();
        assert_eq!(hits.len(), 2);
        // seed-1 vector matches the "close" embedding, so it should be first (lowest distance)
        assert!(hits[0].memory.content.contains("close"));
    }

    #[test]
    fn search_semantic_tag_filter_any() {
        let s = open_test_storage();
        store_mem(&s, "a", 1, &["x"], None);
        store_mem(&s, "b", 2, &["y"], None);
        let mut q = sq("a", SearchMode::Semantic, 5);
        q.tags = vec!["x".into()];
        let hits = s.search(&q, Some(&seeded_embed(1))).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory.content, "a");
    }

    #[test]
    fn search_semantic_tag_filter_all() {
        // ALL-match is enforced in SQL: only rows carrying every tag come back.
        let s = open_test_storage();
        store_mem(&s, "ab", 1, &["x", "y"], None);
        store_mem(&s, "a_only", 2, &["x"], None);
        store_mem(&s, "ab2", 3, &["x", "y"], None);
        let mut q = sq("ab", SearchMode::Semantic, 5);
        q.tags = vec!["x".into(), "y".into()];
        q.tag_match = TagMatch::All;
        let hits = s.search(&q, Some(&seeded_embed(1))).unwrap();
        let mut got: Vec<_> = hits.iter().map(|h| h.memory.content.clone()).collect();
        got.sort();
        assert_eq!(got, vec!["ab".to_string(), "ab2".to_string()]);
    }

    #[test]
    fn mark_superseded_hidden_unless_included() {
        let s = open_test_storage();
        let old = store_mem(&s, "old version", 1, &["t"], None);
        s.mark_superseded(&old, "newhash123").unwrap();
        // Default search hides the superseded row...
        let mut q = sq("old version", SearchMode::Semantic, 5);
        let hits = s.search(&q, Some(&seeded_embed(1))).unwrap();
        assert!(
            hits.iter().all(|h| h.memory.content_hash != old),
            "superseded row must be hidden by default"
        );
        // ...but include_superseded surfaces it again.
        q.include_superseded = true;
        let hits = s.search(&q, Some(&seeded_embed(1))).unwrap();
        assert!(
            hits.iter().any(|h| h.memory.content_hash == old),
            "include_superseded must surface the superseded row"
        );
    }

    #[test]
    fn search_filtered_browse_by_tags() {
        // No query string + a tag filter -> filtered browse (no embedding needed).
        let s = open_test_storage();
        store_mem(&s, "m1", 1, &["proj"], None);
        store_mem(&s, "m2", 2, &["other"], None);
        store_mem(&s, "m3", 3, &["proj"], None);
        let mut q = sq("", SearchMode::Semantic, 10);
        q.tags = vec!["proj".into()];
        let hits = s.search_filtered(&q).unwrap();
        let mut got: Vec<_> = hits.iter().map(|h| h.memory.content.clone()).collect();
        got.sort();
        assert_eq!(got, vec!["m1".to_string(), "m3".to_string()]);
    }

    #[test]
    fn get_tags_type_reads_parent() {
        let s = open_test_storage();
        let h = store_mem(&s, "parent", 1, &["a", "b"], Some("decision"));
        let (mut tags, mt) = s.get_tags_type(&h).unwrap();
        tags.sort();
        assert_eq!(tags, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(mt.as_deref(), Some("decision"));
        // Unknown hash -> empty carry-over.
        let (t2, m2) = s.get_tags_type("nope").unwrap();
        assert!(t2.is_empty() && m2.is_none());
    }

    #[test]
    fn search_semantic_time_filter() {
        let s = open_test_storage();
        // Store with seed-1, then filter by a point in the far future
        store_mem(&s, "old", 1, &[], None);
        let mut q = sq("old", SearchMode::Semantic, 5);
        q.after = Some("9999-01-01".to_string());
        let hits = s.search(&q, Some(&seeded_embed(1))).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn search_semantic_limit() {
        let s = open_test_storage();
        for i in 0..5u8 {
            store_mem(&s, &format!("item{i}"), i + 1, &[], None);
        }
        let hits = s
            .search(&sq("item", SearchMode::Semantic, 2), Some(&seeded_embed(1)))
            .unwrap();
        assert!(hits.len() <= 2);
    }

    // ––– search: exact –––

    #[test]
    fn search_exact_case_insensitive() {
        let s = open_test_storage();
        store_mem(&s, "HeLLo WorLD", 1, &[], None);
        let hits = s
            .search(
                &sq("hello world", SearchMode::Exact, 5),
                None, /* exact does not need embedding */
            )
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory.content, "HeLLo WorLD");
        assert!((hits[0].relevance_score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn search_exact_no_match() {
        let s = open_test_storage();
        store_mem(&s, "abc", 1, &[], None);
        let hits = s
            .search(&sq("xyz", SearchMode::Exact, 5), None)
            .unwrap();
        assert!(hits.is_empty());
    }

    // ––– search: hybrid –––

    #[test]
    fn search_hybrid_returns_results() {
        let s = open_test_storage();
        store_mem(&s, "hybrid test abc", 1, &[], None);
        store_mem(&s, "something else", 2, &[], None);
        let hits = s
            .search(
                &sq("hybrid test", SearchMode::Hybrid, 5),
                Some(&seeded_embed(1)),
            )
            .unwrap();
        // hybrid fuses semantic + BM25; should rank "hybrid test abc" first
        assert!(!hits.is_empty());
        // first result should be the matching one
        assert!(hits[0].memory.content.contains("hybrid"));
    }

    #[test]
    fn search_hybrid_no_embedding() {
        let s = open_test_storage();
        store_mem(&s, "x", 1, &[], None);
        // No query embedding → hybrid returns empty (can't fuse)
        let hits = s
            .search(&sq("x", SearchMode::Hybrid, 5), None)
            .unwrap();
        assert!(hits.is_empty());
    }

    // ––– list –––

    #[test]
    fn list_pagination_basics() {
        let s = open_test_storage();
        for i in 0..5u8 {
            store_mem(&s, &format!("item{i}"), i + 1, &[], None);
        }
        let page = s.list(&lq(1, 3)).unwrap();
        assert_eq!(page.page, 1);
        assert_eq!(page.page_size, 3);
        assert_eq!(page.total, 5);
        assert_eq!(page.memories.len(), 3);
        assert!(page.has_more);
        assert_eq!(page.total_pages, 2);
    }

    #[test]
    fn list_tag_filter() {
        let s = open_test_storage();
        store_mem(&s, "tagged-a", 1, &["foo"], None);
        store_mem(&s, "tagged-b", 2, &["bar"], None);
        let mut q = lq(1, 10);
        q.tags = vec!["foo".into()];
        let page = s.list(&q).unwrap();
        assert_eq!(page.memories.len(), 1);
        assert_eq!(page.memories[0].content, "tagged-a");
    }

    #[test]
    fn list_empty() {
        let s = open_test_storage();
        let page = s.list(&lq(1, 10)).unwrap();
        assert_eq!(page.memories.len(), 0);
        assert_eq!(page.total, 0);
    }

    // ––– delete –––

    #[test]
    fn delete_single() {
        let s = open_test_storage();
        let h = store_mem(&s, "delete-me", 1, &[], None);
        let outcome = s
            .delete(&DeleteQuery {
                content_hash: Some(h.clone()),
                tags: vec![],
                tag_match: TagMatch::Any,
                before: None,
                after: None,
                dry_run: false,
            })
            .unwrap();
        assert_eq!(outcome.deleted_count, 1);
        assert!(outcome.deleted_hashes.contains(&h));
        // should be gone from search
        let hits = s
            .search(&sq("delete-me", SearchMode::Exact, 5), None)
            .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn delete_bulk_by_tags() {
        let s = open_test_storage();
        store_mem(&s, "del-a", 1, &["temp"], None);
        store_mem(&s, "del-b", 2, &["temp"], None);
        store_mem(&s, "keep", 3, &["perm"], None);
        let outcome = s
            .delete(&DeleteQuery {
                content_hash: None,
                tags: vec!["temp".into()],
                tag_match: TagMatch::Any,
                before: None,
                after: None,
                dry_run: false,
            })
            .unwrap();
        assert_eq!(outcome.deleted_count, 2);
        // "keep" should still be there
        let hits = s
            .search(&sq("keep", SearchMode::Exact, 5), None)
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn delete_dry_run() {
        let s = open_test_storage();
        let h = store_mem(&s, "dry-run-test", 1, &[], None);
        let outcome = s
            .delete(&DeleteQuery {
                content_hash: Some(h.clone()),
                tags: vec![],
                tag_match: TagMatch::Any,
                before: None,
                after: None,
                dry_run: true,
            })
            .unwrap();
        assert_eq!(outcome.deleted_count, 1);
        assert!(outcome.message.contains("Would delete"));
        // memory should still exist
        let hits = s
            .search(&sq("dry-run-test", SearchMode::Exact, 5), None)
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn delete_mass_guard() {
        let s = open_test_storage();
        let err = s
            .delete(&DeleteQuery {
                content_hash: None,
                tags: vec![],
                tag_match: TagMatch::Any,
                before: None,
                after: None,
                dry_run: false,
            })
            .unwrap_err();
        assert!(matches!(err, MemoryError::InvalidArg(_)));
    }

    #[test]
    fn delete_not_found() {
        let s = open_test_storage();
        let err = s
            .delete(&DeleteQuery {
                content_hash: Some("nonexistent_hash_abc123".into()),
                tags: vec![],
                tag_match: TagMatch::Any,
                before: None,
                after: None,
                dry_run: false,
            })
            .unwrap_err();
        assert!(matches!(err, MemoryError::NotFound(_)));
    }

    // ––– update –––

    #[test]
    fn update_tags() {
        let s = open_test_storage();
        let h = store_mem(&s, "updatable", 1, &["old"], None);
        let msg = s
            .update(
                &h,
                &serde_json::json!({"tags": ["new-tag"]}),
                false,
            )
            .unwrap();
        assert!(msg.contains("tags"));
        // verify via list
        let page = s
            .list(&ListQuery {
                page: 1,
                page_size: 10,
                tags: vec!["new-tag".into()],
                tag_match: TagMatch::Any,
                memory_type: None,
            })
            .unwrap();
        assert_eq!(page.memories.len(), 1);
    }

    #[test]
    fn update_not_found() {
        let s = open_test_storage();
        let err = s
            .update(
                "nonexistent",
                &serde_json::json!({"tags": ["x"]}),
                false,
            )
            .unwrap_err();
        assert!(matches!(err, MemoryError::NotFound(_)));
    }

    // ––– graph –––

    #[test]
    fn graph_connected() {
        let s = open_test_storage();
        let a = store_mem(&s, "node-a", 1, &[], None);
        let b = store_mem(&s, "node-b", 2, &[], None);
        let c = store_mem(&s, "node-c", 3, &[], None);
        insert_graph_edge(&s, &a, &b, 0.8);
        insert_graph_edge(&s, &b, &c, 0.7);
        let conn = s.graph_connected(&a, 2).unwrap();
        assert_eq!(conn.len(), 2);
        let hashes: Vec<&str> = conn.iter().map(|(h, _)| h.as_str()).collect();
        assert!(hashes.contains(&b.as_str()));
        assert!(hashes.contains(&c.as_str()));
    }

    #[test]
    fn graph_path() {
        let s = open_test_storage();
        let a = store_mem(&s, "path-a", 1, &[], None);
        let b = store_mem(&s, "path-b", 2, &[], None);
        let c = store_mem(&s, "path-c", 3, &[], None);
        insert_graph_edge(&s, &a, &b, 0.8);
        insert_graph_edge(&s, &b, &c, 0.7);
        let path = s.graph_path(&a, &c, 5).unwrap();
        assert_eq!(path, vec![a.clone(), b.clone(), c.clone()]);
    }

    #[test]
    fn graph_path_unreachable() {
        let s = open_test_storage();
        let a = store_mem(&s, "iso-a", 1, &[], None);
        let b = store_mem(&s, "iso-b", 2, &[], None);
        // no edge → unreachable
        let path = s.graph_path(&a, &b, 5).unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn graph_subgraph() {
        let s = open_test_storage();
        let x = store_mem(&s, "sub-x", 1, &[], None);
        let y = store_mem(&s, "sub-y", 2, &[], None);
        insert_graph_edge(&s, &x, &y, 0.9);
        let (nodes, edges) = s.graph_subgraph(&x, 1).unwrap();
        // x (dist 0) + y (dist 1) = 2 nodes
        assert_eq!(nodes.len(), 2);
        // one edge
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn graph_infer() {
        let s = open_test_storage();
        let a = store_mem(&s, "inf-a", 1, &[], None);
        let b = store_mem(&s, "inf-b", 2, &[], None);
        let c = store_mem(&s, "inf-c", 3, &[], None);
        // a–b and b–c → infer should suggest a–c via shared neighbor b
        insert_graph_edge(&s, &a, &b, 0.8);
        insert_graph_edge(&s, &b, &c, 0.7);
        let cands = s.graph_infer(&a, 10).unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].0, c);
        assert_eq!(cands[0].1, 1); // one shared neighbor (b)
    }

    #[test]
    fn graph_suggest_returns_candidates() {
        let s = open_test_storage();
        let a = store_mem(&s, "sug-a", 1, &[], None);
        let b = store_mem(&s, "sug-b", 2, &[], None);
        // seed-1 and seed-2 are orthogonal → cosine ~0 → mapped sim ~0.5
        let sugg = s.graph_suggest(&a, 10).unwrap();
        assert_eq!(sugg.len(), 1);
        assert_eq!(sugg[0].0, b);
    }

    // ––– stats –––

    #[test]
    fn stats_counts() {
        let s = open_test_storage();
        store_mem(&s, "s1", 1, &[], None);
        store_mem(&s, "s2", 2, &[], None);
        let st = s.stats("test-model").unwrap();
        assert_eq!(st.total_memories, 2);
        assert_eq!(st.embedding_model, "test-model");
        assert_eq!(st.embedding_dimension, 2560);
    }
}
