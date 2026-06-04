//! MCP tool surface — the 8 stdio tools the opencode CLIs are allowed to call:
//! `memory_store, memory_search, memory_list, memory_delete, memory_update,
//!  memory_health, memory_stats, memory_graph`.
//!
//! The opencode.jsonc permission allow-list permits only these 8, so they are
//! the complete reachable surface. Other tools (memory_observe,
//! memory_consolidate, mistake_note_*, …) are not exposed to these CLIs and are
//! out of scope for v1.
//!
//! ## Output-format fidelity (highest-risk detail)
//! Each tool returns `content: [{type:"text", text:<string>}]` but the *string*
//! shape differs per tool — some prose, some JSON-stringified. Callers that grep
//! the text depend on these exact shapes:
//!   * store/search/delete/update -> PLAIN TEXT templates.
//!   * list/stats/graph           -> JSON-stringified.
//!   * health                     -> `"Database Health Check Results:\n"` + JSON.
//! The exact templates live in [`crate::format`].
//!
//! ## Arg schemas
//! Input structs derive `serde::Deserialize + schemars::JsonSchema`; doc-comments
//! on fields become the MCP schema descriptions clients validate against.
//! NOTE: in `memory_store`, tags live UNDER `metadata.tags` (array or CSV) — NOT
//! a top-level arg. search/list/delete/update DO take top-level `tags`.

use crate::config::Config;
use crate::embed::EmbedClient;
use crate::format;
use crate::models::SearchMode;
use crate::storage::{
    DeleteQuery, ListQuery, SearchQuery, Storage, StoreArgs as StorageStoreArgs, StoreOutcome,
};
use crate::error::MemoryError;
use crate::models::{normalize_tags, Tags, TagMatch};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, Implementation, ProtocolVersion, ServerCapabilities, ServerInfo};
use rmcp::{schemars, tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Per-process query telemetry surfaced in `memory_health`. Counters are
/// process-local (one server process per scope), so they reflect the current
/// session and reset when it ends.
#[derive(Default)]
pub struct QueryStats {
    count: AtomicU64,
    total_us: AtomicU64,
}

impl QueryStats {
    fn record(&self, elapsed_us: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_us.fetch_add(elapsed_us, Ordering::Relaxed);
    }
    /// Returns `(total_queries, average_query_time_ms)`.
    fn snapshot(&self) -> (u64, f64) {
        let n = self.count.load(Ordering::Relaxed);
        let us = self.total_us.load(Ordering::Relaxed);
        let avg_ms = if n > 0 { (us as f64 / n as f64) / 1000.0 } else { 0.0 };
        (n, avg_ms)
    }
}

/// RAII timer: records elapsed time into [`QueryStats`] on drop, so early
/// returns in a tool handler are still counted. Owns an `Arc` clone so it is
/// safe to hold across `.await` points.
struct QueryTimer {
    stats: Arc<QueryStats>,
    start: Instant,
}
impl Drop for QueryTimer {
    fn drop(&mut self) {
        self.stats.record(self.start.elapsed().as_micros() as u64);
    }
}

// ---------------------------------------------------------------------------
// Tool argument structs (JSON-Schema derived via schemars).
// ---------------------------------------------------------------------------

/// Metadata sub-object accepted by `memory_store` (this is where tags + type live).
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct StoreMetadata {
    /// Tags. Accepts an array of strings OR a comma-separated string.
    #[serde(default)]
    pub tags: Tags,
    /// Memory type (observation, insight, decision, ...).
    #[serde(rename = "type", default)]
    pub mem_type: Option<String>,
}

/// `memory_store` args. Only `content` is required.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct StoreArgs {
    /// The content to store (required).
    pub content: String,
    /// Optional metadata object; tags + type live here.
    #[serde(default)]
    pub metadata: Option<StoreMetadata>,
    /// Skips SEMANTIC dedup (exact-hash dedup still applies).
    #[serde(default)]
    pub conversation_id: Option<String>,
}

/// `memory_search` args — all optional.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchArgs {
    /// Natural-language query.
    #[serde(default)]
    pub query: Option<String>,
    /// semantic | exact | hybrid | ranked (default semantic).
    #[serde(default)]
    pub mode: Option<String>,
    /// Filter to these tags (array of strings OR comma-separated string).
    #[serde(default)]
    pub tags: Tags,
    /// any | all (default any).
    #[serde(default)]
    pub tag_match: Option<String>,
    /// YYYY-MM-DD lower bound.
    #[serde(default)]
    pub after: Option<String>,
    /// YYYY-MM-DD upper bound.
    #[serde(default)]
    pub before: Option<String>,
    /// Max results 1..100 (default 10).
    #[serde(default)]
    pub limit: Option<usize>,
    /// Include superseded memories (default false).
    #[serde(default)]
    pub include_superseded: Option<bool>,
}

/// `memory_list` args.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ListArgs {
    /// 1-based page (default 1).
    #[serde(default)]
    pub page: Option<usize>,
    /// 1..100 (default 20).
    #[serde(default)]
    pub page_size: Option<usize>,
    /// Filter to these tags (array of strings OR comma-separated string).
    #[serde(default)]
    pub tags: Tags,
    #[serde(default)]
    pub tag_match: Option<String>,
    #[serde(default)]
    pub memory_type: Option<String>,
}

/// `memory_delete` args. No filters at all -> error (mass-delete guard);
/// `content_hash` wins over everything else.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DeleteArgs {
    #[serde(default)]
    pub content_hash: Option<String>,
    /// Filter to these tags (array of strings OR comma-separated string).
    #[serde(default)]
    pub tags: Tags,
    #[serde(default)]
    pub tag_match: Option<String>,
    #[serde(default)]
    pub before: Option<String>,
    #[serde(default)]
    pub after: Option<String>,
    /// Count-only, no mutation (changes message wording to "Would delete").
    #[serde(default)]
    pub dry_run: Option<bool>,
}

/// `memory_update` args. `content_hash` + `updates` required.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct UpdateArgs {
    pub content_hash: String,
    /// Object with optional `tags`, `memory_type`, `metadata`, `content`.
    pub updates: serde_json::Value,
    /// Supersede + insert new row instead of in-place update (needs updates.content).
    #[serde(default)]
    pub versioned: Option<bool>,
}

/// `memory_graph` args.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GraphArgs {
    /// connected | path | subgraph | infer | suggest (entity/abduct actions are
    /// not implemented — they need an NLP subsystem the Rust server lacks).
    pub action: String,
    /// Source memory hash (connected | subgraph | infer | suggest | path-from).
    #[serde(default)]
    pub hash: Option<String>,
    /// BFS hop limit for connected/subgraph/path (default 2, path default 5).
    #[serde(default)]
    pub max_hops: Option<usize>,
    /// Target memory hash for `path`.
    #[serde(default)]
    pub target: Option<String>,
    /// Result cap for `suggest`/`infer` (default 10, max 100).
    #[serde(default)]
    pub limit: Option<usize>,
}

// ---------------------------------------------------------------------------
// Server struct + tool router.
// ---------------------------------------------------------------------------

/// The MCP server: holds the tool router (required by `#[tool_router]`), the
/// storage handle, and the embedding client. Cloned cheaply via `Arc` fields.
#[derive(Clone)]
pub struct MemoryServer {
    // Populated and dispatched through by the #[tool_router]/#[tool_handler]
    // macros; not read directly, hence the targeted allow.
    #[allow(dead_code)]
    tool_router: ToolRouter<MemoryServer>,
    storage: Arc<Storage>,
    embed: Arc<EmbedClient>,
    config: Arc<Config>,
    stats: Arc<QueryStats>,
}

/// Build a `CallToolResult` whose single text content is `text` (the universal
/// shape: every tool returns `content: [{type:"text", text:<string>}]`).
fn text_result(text: String) -> CallToolResult {
    CallToolResult::success(vec![Content::text(text)])
}

#[tool_router]
impl MemoryServer {
    /// Build the server from an opened storage handle + embedding client + config.
    pub fn new(storage: Arc<Storage>, embed: Arc<EmbedClient>, config: Arc<Config>) -> Self {
        Self {
            tool_router: Self::tool_router(),
            storage,
            embed,
            config,
            stats: Arc::new(QueryStats::default()),
        }
    }

    /// Start a query timer; records elapsed time into `stats` when the returned
    /// guard drops at the end of the tool handler.
    fn timer(&self) -> QueryTimer {
        QueryTimer { stats: Arc::clone(&self.stats), start: Instant::now() }
    }

    /// Embedding provenance stamped into each stored memory's metadata so a future
    /// model upgrade is detectable + re-embeddable. No schema change — metadata is
    /// a free-form JSON blob, and older rows simply lack these keys.
    fn provenance_metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "embedding_model": self.config.embed.model,
            "embedding_dimension": self.storage.embedding_dim(),
        })
    }

    /// Store a memory. Returns PLAIN TEXT:
    /// `"Memory stored successfully (hash: <sha256hex>)"` or
    /// `"Error storing memory: <msg>"` / `"Error: Content is required"`.
    #[tool(description = "Store a memory in the persistent store")]
    pub async fn memory_store(
        &self,
        Parameters(args): Parameters<StoreArgs>,
    ) -> Result<CallToolResult, McpError> {
        let _t = self.timer();
        if args.content.trim().is_empty() {
            return Ok(text_result("Error: Content is required".to_string()));
        }

        // tags + type live UNDER metadata in memory_store. Default type = "note".
        // Empty tags collapse to ["untagged"]. Tags are normalize_tags()'d
        // (lowercase + trim + comma->hyphen + dedup) before they are joined into
        // the comma-separated string persisted in the DB.
        let (raw_tags, mem_type) = match &args.metadata {
            Some(md) => (
                md.tags.clone().into_vec(),
                md.mem_type.clone().or_else(|| Some("note".to_string())),
            ),
            None => (Vec::new(), Some("note".to_string())),
        };
        let mut tags = normalize_tags(&raw_tags);
        if tags.is_empty() {
            tags.push("untagged".to_string());
        }

        // Embed the content (lazy-ensure self-heal lives in the embed client).
        let embedding = match self.embed.embed_one(&args.content).await {
            Ok(e) => e,
            Err(e) => {
                return Ok(text_result(format!(
                    "Error storing memory: Failed to generate embedding: {e}"
                )));
            }
        };

        let store_args = StorageStoreArgs {
            content: args.content.clone(),
            tags,
            memory_type: mem_type,
            // metadata: tags/type are not duplicated here — they have dedicated
            // columns and are stripped before this blob is persisted.
            metadata: self.provenance_metadata(),
            conversation_id: args.conversation_id.clone(),
        };

        match self.storage.store(&store_args, &embedding) {
            Ok(StoreOutcome::Stored { content_hash }) => {
                Ok(text_result(format::store_ok(&content_hash)))
            }
            Ok(StoreOutcome::Duplicate { content_hash }) => {
                Ok(text_result(format::store_duplicate(&content_hash)))
            }
            Ok(StoreOutcome::SemanticDuplicate { existing_hash }) => {
                Ok(text_result(format::store_semantic_duplicate(&existing_hash)))
            }
            Err(e) => Ok(text_result(format!("Error storing memory: {e}"))),
        }
    }

    /// Semantic/exact/hybrid search. Returns PLAIN TEXT formatted result list:
    /// header `"Found <n> memories (mode: <mode>) for query: '<q>'"` + per-result lines,
    /// or `"No memories found for query: '<q>'"`.
    #[tool(description = "Search stored memories by semantic similarity, tags, or time")]
    pub async fn memory_search(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, McpError> {
        let _t = self.timer();
        let mode_str = args.mode.clone().unwrap_or_else(|| "semantic".to_string());
        let mode = SearchMode::parse(&mode_str);
        let query = args.query.clone().unwrap_or_default();
        let limit = args.limit.unwrap_or(10).clamp(1, 100);

        // Normalize filter tags (lowercase + trim + dedup) so they compare
        // consistently against the normalized tags stored in the DB.
        let norm_tags = normalize_tags(&args.tags.clone().into_vec());
        let sq = SearchQuery {
            query: args.query.clone(),
            mode,
            tags: norm_tags,
            tag_match: TagMatch::parse(args.tag_match.as_deref().unwrap_or("any")),
            after: args.after.clone(),
            before: args.before.clone(),
            limit,
            include_superseded: args.include_superseded.unwrap_or(false),
        };

        // Semantic / hybrid / ranked need a query embedding; exact does not.
        let embedding = if matches!(mode, SearchMode::Exact) {
            None
        } else if query.is_empty() {
            // No query for a semantic-family search: nothing to embed, so the
            // search returns empty.
            None
        } else {
            match self.embed.embed_one(&query).await {
                Ok(e) => Some(e),
                Err(e) => {
                    return Ok(text_result(format!("Error searching memories: {e}")));
                }
            }
        };

        let hits = match self.storage.search(&sq, embedding.as_deref()) {
            Ok(h) => h,
            Err(e) => return Ok(text_result(format!("Error searching memories: {e}"))),
        };

        // For ALL tag_match, post-filter in Rust because the SQL tag filter is
        // ANY-match only. Compare case-insensitively: sq.tags is already
        // lowercased by normalize_tags, but older stored tags could be
        // mixed-case, so lowercase both sides here.
        let hits = if sq.tag_match == TagMatch::All && !sq.tags.is_empty() {
            hits.into_iter()
                .filter(|h| {
                    let stored: Vec<String> =
                        h.memory.tags.iter().map(|t| t.to_lowercase()).collect();
                    sq.tags.iter().all(|t| stored.contains(&t.to_lowercase()))
                })
                .collect()
        } else {
            hits
        };

        Ok(text_result(format::search_results(&query, &mode_str, &hits)))
    }

    /// Paginated browse. Returns a JSON STRING:
    /// `{memories:[...], page, page_size, total, total_pages, has_more}`.
    #[tool(description = "List stored memories with pagination and optional tag/type filters")]
    pub async fn memory_list(
        &self,
        Parameters(args): Parameters<ListArgs>,
    ) -> Result<CallToolResult, McpError> {
        let _t = self.timer();
        let lq = ListQuery {
            page: args.page.unwrap_or(1).max(1),
            page_size: args.page_size.unwrap_or(20).clamp(1, 100),
            tags: normalize_tags(&args.tags.clone().into_vec()),
            tag_match: TagMatch::parse(args.tag_match.as_deref().unwrap_or("any")),
            memory_type: args.memory_type.clone(),
        };

        match self.storage.list(&lq) {
            Ok(page) => {
                // Build each memory's JSON map explicitly: key order is part of
                // the output contract callers depend on.
                let mems: Vec<serde_json::Value> = page
                    .memories
                    .iter()
                    .map(|m| {
                        serde_json::json!({
                            "content": m.content,
                            "content_hash": m.content_hash,
                            "tags": m.tags,
                            "memory_type": m.memory_type,
                            "metadata": m.metadata,
                            "created_at": m.created_at,
                            "updated_at": m.updated_at,
                            "created_at_iso": m.created_at_iso,
                            "updated_at_iso": m.updated_at_iso,
                        })
                    })
                    .collect();
                let out = serde_json::json!({
                    "memories": mems,
                    "page": page.page,
                    "page_size": page.page_size,
                    "total": page.total,
                    "total_pages": page.total_pages,
                    "has_more": page.has_more,
                });
                // Serialize pretty-printed (2-space indent) JSON.
                Ok(text_result(
                    serde_json::to_string_pretty(&out).unwrap_or_else(|_| "{}".to_string()),
                ))
            }
            Err(e) => Ok(text_result(format!("Error listing memories: {e}"))),
        }
    }

    /// Soft-delete by hash or filters. Returns PLAIN TEXT:
    /// `"Memory deleted successfully: <hash>"` / `"<msg>\n\nDeleted N memories"` /
    /// `"Would delete N memories\nHashes: ..."` (dry_run) / `"Error: ..."`.
    #[tool(description = "Soft-delete memories by content_hash or tag/time filters")]
    pub async fn memory_delete(
        &self,
        Parameters(args): Parameters<DeleteArgs>,
    ) -> Result<CallToolResult, McpError> {
        let _t = self.timer();
        let dq = DeleteQuery {
            content_hash: args.content_hash.clone(),
            tags: normalize_tags(&args.tags.clone().into_vec()),
            tag_match: TagMatch::parse(args.tag_match.as_deref().unwrap_or("any")),
            before: args.before.clone(),
            after: args.after.clone(),
            dry_run: args.dry_run.unwrap_or(false),
        };
        let dry_run = dq.dry_run;

        match self.storage.delete(&dq) {
            Ok(outcome) => Ok(text_result(format::delete_msg(
                outcome.deleted_count,
                &outcome.deleted_hashes,
                dry_run,
                &outcome.message,
            ))),
            // Invalid-arg / not-found surface as plain `Error: <msg>`.
            Err(MemoryError::InvalidArg(m)) | Err(MemoryError::NotFound(m)) => {
                Ok(text_result(format!("Error: {m}")))
            }
            Err(e) => Ok(text_result(format!("Error deleting memories: {e}"))),
        }
    }

    /// Update tags/type/metadata (or versioned supersede). Returns PLAIN TEXT message.
    #[tool(description = "Update a memory's tags, type, or metadata (optionally versioned)")]
    pub async fn memory_update(
        &self,
        Parameters(args): Parameters<UpdateArgs>,
    ) -> Result<CallToolResult, McpError> {
        let _t = self.timer();
        if args.content_hash.is_empty() {
            return Ok(text_result("Error: content_hash is required".to_string()));
        }
        let updates_obj = match args.updates.as_object() {
            Some(o) if !o.is_empty() => o,
            Some(_) => {
                return Ok(text_result("Error: updates dictionary is required".to_string()))
            }
            None => return Ok(text_result("Error: updates must be a dictionary".to_string())),
        };

        let versioned = args.versioned.unwrap_or(false);

        if versioned {
            // Versioned supersede: requires updates.content.
            let new_content = updates_obj
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if new_content.is_empty() {
                return Ok(text_result(
                    "Error: versioned update requires 'content' field in updates.".to_string(),
                ));
            }

            // Resolve carry-over tags/type from the existing row unless overridden.
            // Accept array OR CSV-string form, then normalize_tags.
            let new_tags = updates_obj.get("tags").map(|v| {
                let raw: Vec<String> = match v {
                    serde_json::Value::Array(a) => a
                        .iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect(),
                    serde_json::Value::String(s) => crate::models::split_tag_string(s),
                    _ => Vec::new(),
                };
                normalize_tags(&raw)
            });
            let new_type = updates_obj
                .get("memory_type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let reason = updates_obj
                .get("reason")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            return Ok(self
                .versioned_update(&args.content_hash, new_content, new_tags, new_type, reason)
                .await);
        }

        // Non-versioned in-place metadata/tags/type update.
        match self.storage.update(&args.content_hash, &args.updates, false) {
            Ok(message) => Ok(text_result(format!(
                "Successfully updated memory metadata. {message}"
            ))),
            Err(e) => Ok(text_result(format!(
                "Failed to update memory metadata: {e}"
            ))),
        }
    }

    /// Versioned supersede: embed + store the new content (dedup-skipped), then
    /// stamp the old row's `metadata.superseded_by = <new_hash>`. The supersede
    /// marker lives in metadata, NOT in a dedicated column, so superseded rows
    /// still surface in search unless explicitly filtered out.
    async fn versioned_update(
        &self,
        content_hash: &str,
        new_content: &str,
        new_tags: Option<Vec<String>>,
        new_type: Option<String>,
        reason: Option<String>,
    ) -> CallToolResult {
        let embedding = match self.embed.embed_one(new_content).await {
            Ok(e) => e,
            Err(e) => {
                return text_result(format!("Failed versioned update: embedding failed: {e}"));
            }
        };

        let store_args = StorageStoreArgs {
            content: new_content.to_string(),
            tags: new_tags.unwrap_or_default(),
            memory_type: new_type,
            metadata: self.provenance_metadata(),
            conversation_id: Some("versioned".to_string()), // skip semantic dedup
        };
        let new_hash = match self.storage.store(&store_args, &embedding) {
            Ok(StoreOutcome::Stored { content_hash }) => content_hash,
            Ok(StoreOutcome::Duplicate { content_hash }) => content_hash,
            // conversation_id is set above, so semantic dedup is skipped and this
            // arm is unreachable in practice; treat the near-dup hash as the result.
            Ok(StoreOutcome::SemanticDuplicate { existing_hash }) => existing_hash,
            Err(e) => return text_result(format!("Failed versioned update: {e}")),
        };

        // Stamp metadata.superseded_by (+ optional evolution_reason) on the old row.
        let mut meta = serde_json::Map::new();
        meta.insert(
            "superseded_by".to_string(),
            serde_json::Value::String(new_hash.clone()),
        );
        if let Some(r) = reason {
            meta.insert("evolution_reason".to_string(), serde_json::Value::String(r));
        }
        let updates = serde_json::json!({ "metadata": serde_json::Value::Object(meta) });
        match self.storage.update(content_hash, &updates, false) {
            Ok(_) => text_result(format!(
                "Versioned update successful. New hash: {new_hash}, parent hash: {content_hash}. Memory versioned successfully"
            )),
            Err(e) => text_result(format!("Failed versioned update: {e}")),
        }
    }

    /// DB health. Returns `"Database Health Check Results:\n"` + JSON
    /// (version, validation, statistics{total_memories, embedding_dimension:2560, ...}).
    #[tool(description = "Report database health, statistics, and embedding configuration")]
    pub async fn memory_health(&self) -> Result<CallToolResult, McpError> {
        let _t = self.timer();
        let (total_queries, avg_ms) = self.stats.snapshot();
        let model = &self.config.embed.model;
        let stats = match self.storage.stats(model) {
            Ok(s) => s,
            Err(e) => {
                return Ok(text_result(format!("Error checking database health: {e}")))
            }
        };
        let size_mb = (stats.database_size_bytes as f64 / (1024.0 * 1024.0) * 100.0).round() / 100.0;

        // Health statistics block. embedding_dimension is intentionally omitted
        // here to keep this dict in sync with the documented health-stats shape.
        let statistics = serde_json::json!({
            "status": "healthy",
            "backend": "sqlite-vec",
            "total_memories": stats.total_memories,
            "has_embedding_tables": true,
            "has_embedding_model": true,
            "embedding_model": stats.embedding_model,
            "database_size_bytes": stats.database_size_bytes,
            "database_size_mb": size_mb,
        });

        let result = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "validation": {
                "status": "healthy",
                "message": "SQLite-vec database validation successful",
            },
            "statistics": statistics,
            "integrity": {},
            "performance": {
                "storage": {},
                "server": {
                    "average_query_time_ms": avg_ms,
                    "total_queries": total_queries,
                    "storage_type": "SqliteVecMemoryStorage",
                },
            },
        });

        Ok(text_result(format::health(&result)))
    }

    /// Process-local cache telemetry (NOT memory counts). Returns a minimal
    /// JSON STRING in the shape callers expect so the allow-listed call doesn't
    /// error.
    #[tool(description = "Report process-local cache and performance telemetry")]
    pub async fn memory_stats(&self) -> Result<CallToolResult, McpError> {
        let _t = self.timer();
        // Process-local cache telemetry. This binary has no global init-cache
        // (one process per scope), so the counters are zeroed — but the key SHAPE
        // (cache stats + a backend_info block) is preserved so any parser sees
        // the expected keys.
        let result = serde_json::json!({
            "total_calls": 0,
            "hit_rate": 0.0,
            "storage_cache": {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "size": 0,
                "keys": [],
            },
            "service_cache": {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "size": 0,
            },
            "performance": {
                "avg_init_time_ms": 0.0,
                "min_init_time_ms": 0.0,
                "max_init_time_ms": 0.0,
                "total_inits": 0,
            },
            "message": "MCP server caching is INACTIVE with 0.0% hit rate",
            "backend_info": {
                "storage_backend": "sqlite_vec",
                "sqlite_path": self.config.db_path.to_string_lossy(),
                "embedding_model": self.config.embed.model,
            },
        });
        Ok(text_result(
            serde_json::to_string_pretty(&result).unwrap_or_else(|_| "{}".to_string()),
        ))
    }

    /// Association-graph queries. Implemented: connected, path, subgraph, infer,
    /// suggest (pure-graph + stored-embedding). entity/abduct actions return a
    /// specific not-implemented error (need an NLP subsystem). Returns JSON STRING.
    #[tool(description = "Query the memory association graph. Implemented actions: connected (BFS neighbors within N hops), path (shortest path between hash and target), subgraph (nodes+edges within N hops of hash), infer (link-prediction candidates by shared neighbors), suggest (semantic neighbors not yet linked, from stored embeddings). entity/abduct actions are not implemented (require NLP).")]
    pub async fn memory_graph(
        &self,
        Parameters(args): Parameters<GraphArgs>,
    ) -> Result<CallToolResult, McpError> {
        let _t = self.timer();
        if args.action.is_empty() {
            return Ok(text_result(
                "Error: action parameter is required".to_string(),
            ));
        }
        const VALID: &[&str] = &[
            "connected", "path", "subgraph", "extract_entities", "infer", "suggest", "abduct",
            "list_entities", "entity_profile",
        ];
        if !VALID.contains(&args.action.as_str()) {
            return Ok(text_result(format!(
                "Error: Invalid action '{}'. Must be one of: {}",
                args.action,
                VALID.join(", ")
            )));
        }

        let dump = |v: &serde_json::Value| {
            serde_json::to_string_pretty(v).unwrap_or_else(|_| "{}".to_string())
        };

        match args.action.as_str() {
            "connected" => {
                let Some(hash) = args.hash.clone() else {
                    return Ok(text_result(
                        "Error: hash is required for 'connected' action".to_string(),
                    ));
                };
                let max_hops = args.max_hops.unwrap_or(2);
                match self.storage.graph_connected(&hash, max_hops) {
                    Ok(connected) => Ok(text_result(dump(&serde_json::json!({
                        "success": true,
                        "connected": connected.iter().map(|(h, d)| serde_json::json!({"hash": h, "distance": d})).collect::<Vec<_>>(),
                        "count": connected.len(),
                    })))),
                    Err(e) => Ok(text_result(dump(&serde_json::json!({
                        "success": false, "error": e.to_string(), "connected": [], "count": 0,
                    })))),
                }
            }
            "path" => {
                let (Some(from), Some(to)) = (args.hash.clone(), args.target.clone()) else {
                    return Ok(text_result(
                        "Error: 'path' requires 'hash' (from) and 'target' (to)".to_string(),
                    ));
                };
                let max_hops = args.max_hops.unwrap_or(5);
                match self.storage.graph_path(&from, &to, max_hops) {
                    Ok(path) => {
                        let hops = if path.is_empty() {
                            serde_json::Value::Null
                        } else {
                            serde_json::json!(path.len().saturating_sub(1))
                        };
                        Ok(text_result(dump(&serde_json::json!({
                            "success": true, "found": !path.is_empty(), "path": path, "hops": hops,
                        }))))
                    }
                    Err(e) => Ok(text_result(dump(&serde_json::json!({"success": false, "error": e.to_string()})))),
                }
            }
            "subgraph" => {
                let Some(hash) = args.hash.clone() else {
                    return Ok(text_result(
                        "Error: hash is required for 'subgraph' action".to_string(),
                    ));
                };
                let max_hops = args.max_hops.unwrap_or(2);
                match self.storage.graph_subgraph(&hash, max_hops) {
                    Ok((nodes, edges)) => Ok(text_result(dump(&serde_json::json!({
                        "success": true,
                        "nodes": nodes.iter().map(|(h, d)| serde_json::json!({"hash": h, "distance": d})).collect::<Vec<_>>(),
                        "edges": edges.iter().map(|(s, t, sim)| serde_json::json!({"source": s, "target": t, "similarity": sim})).collect::<Vec<_>>(),
                        "node_count": nodes.len(),
                        "edge_count": edges.len(),
                    })))),
                    Err(e) => Ok(text_result(dump(&serde_json::json!({"success": false, "error": e.to_string()})))),
                }
            }
            "infer" => {
                let Some(hash) = args.hash.clone() else {
                    return Ok(text_result(
                        "Error: hash is required for 'infer' action".to_string(),
                    ));
                };
                let limit = args.limit.unwrap_or(10).clamp(1, 100);
                match self.storage.graph_infer(&hash, limit) {
                    Ok(cands) => Ok(text_result(dump(&serde_json::json!({
                        "success": true,
                        "candidates": cands.iter().map(|(h, n)| serde_json::json!({"hash": h, "shared_neighbors": n})).collect::<Vec<_>>(),
                        "count": cands.len(),
                    })))),
                    Err(e) => Ok(text_result(dump(&serde_json::json!({"success": false, "error": e.to_string()})))),
                }
            }
            "suggest" => {
                let Some(hash) = args.hash.clone() else {
                    return Ok(text_result(
                        "Error: hash is required for 'suggest' action".to_string(),
                    ));
                };
                let limit = args.limit.unwrap_or(10).clamp(1, 100);
                match self.storage.graph_suggest(&hash, limit) {
                    Ok(sugg) => Ok(text_result(dump(&serde_json::json!({
                        "success": true,
                        "suggestions": sugg.iter().map(|(h, sim)| serde_json::json!({"hash": h, "similarity": sim})).collect::<Vec<_>>(),
                        "count": sugg.len(),
                    })))),
                    Err(e) => Ok(text_result(dump(&serde_json::json!({"success": false, "error": e.to_string()})))),
                }
            }
            // abduct / extract_entities / list_entities / entity_profile need an
            // NLP entity-extraction + reasoning subsystem the Rust server lacks.
            other => Ok(text_result(dump(&serde_json::json!({
                "success": false,
                "error": format!("action '{other}' is not implemented in the Rust server (needs an NLP entity-extraction/reasoning subsystem). Implemented graph actions: connected, path, subgraph, infer, suggest."),
            })))),
        }
    }
}

#[tool_handler]
impl ServerHandler for MemoryServer {
    /// Advertise the server name (`"memory"`) the opencode client expects, so
    /// capability negotiation succeeds.
    fn get_info(&self) -> ServerInfo {
        // ServerInfo (= InitializeResult) and Implementation are #[non_exhaustive],
        // so we cannot use struct-literal syntax from outside the crate. Build via
        // the public constructors / builder methods instead.
        let mut info = ServerInfo::default();
        info.protocol_version = ProtocolVersion::V_2024_11_05;
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation::new(
            crate::config::SERVER_NAME,
            env!("CARGO_PKG_VERSION"),
        );
        info.instructions =
            Some("Self-owned Rust memory MCP server over sqlite-vec".to_string());
        info
    }
}
