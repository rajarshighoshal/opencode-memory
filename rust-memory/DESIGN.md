# opencode-memory (Rust) — Architecture & Design

A **single Rust binary** providing persistent semantic memory for AI agents over
the **Model Context Protocol**. It is a **stdio MCP server** backed by **SQLite +
sqlite-vec**, with embeddings from a local **llama.cpp** endpoint. It is
schema-compatible with the common sqlite-vec memory layout, so it opens an
existing memory DB with **no data migration**.

- Data: `~/.config/opencode/memory/global.db` (global) + per-project
  `<project-root>/.opencode-memory/project.db`. sqlite-vec `vec0`, embedding dim **2560**.
- Embeddings: llama.cpp OpenAI-compatible server at
  `http://127.0.0.1:11434/v1/embeddings` (Qwen3-Embedding-4B Q8).
- Toolchain: cargo/rustc 1.96, Apple clang 21 (for the static sqlite-vec.c compile).

**Status: `cargo check` is GREEN (exit 0, zero warnings).** All module
interfaces/types compile; handler bodies are `todo!()` stubs. This is a minimal
compiling skeleton, not an implementation.

---

## 1. Module layout

```
rust-memory/
  Cargo.toml          deps + pinned versions; [[bin]] opencode-memory
  DESIGN.md           this file
  src/
    main.rs           entrypoint: tracing->stderr, scope arg, register vec ext,
                       open Storage, build EmbedClient, MemoryServer.serve(stdio())
    config.rs         Scope {Global,Project}, Config::from_env, EmbedConfig,
                       project-root walk; SERVER_NAME="memory", EMBEDDING_DIM=2560,
                       MAX_KNN_K=4096
    error.rs          EmbedError (ServerUnavailable/Http/BadResponse/DimMismatch/
                       NonFinite) + MemoryError + Result alias
    hashing.rs        content_hash = hex(sha256(content.trim().to_lowercase()))
                       (exact Python parity); unit-tested
    embed.rs          EmbedClient: embed_one/embed_batch, post_with_retry,
                       ensure_server (lazy-ensure self-heal), reorder_and_validate
    models.rs         Memory, SearchHit (relevance = 1 - d/2), SearchMode,
                       TagMatch, epoch_to_iso, tags_to_csv/from_csv
    storage.rs        register_vec_extension (auto-extension), Storage:
                       open/store/search(_semantic/_fts)/list/delete/update/stats/
                       graph_connected; vec_to_blob/blob_to_vec
    format.rs         per-tool output-string templates (prose vs JSON fidelity)
    tools.rs          8 MCP tools (#[tool_router]/#[tool]/#[tool_handler]),
                       arg structs (schemars), MemoryServer + get_info()
```

Dependency direction: `main -> {config, storage, embed, tools}`;
`tools -> {storage, embed, format, models, config}`;
`storage -> {models, config, error}`; `embed -> {config, error}`. No cycles.

---

## 2. Crates + pinned versions (resolved locally, from research)

| Crate | Requested | Resolved | Why |
|---|---|---|---|
| `rmcp` | `1.7` | **1.7.0** | Official MCP Rust SDK. Features `server, macros, transport-io, schemars`. `transport-io` gates `transport::stdio()`; `schemars` derives tool input schemas. |
| `rusqlite` | `0.40` | **0.40.0** | SQLite binding. Feature `bundled` => SQLite 3.51.x compiled in (vec0 0.1.9 compatible). |
| `sqlite-vec` | `=0.1.9` | **0.1.9** | Statically compiles `sqlite-vec.c` (`-DSQLITE_CORE`), exports `sqlite3_vec_init`. **Pinned exact** to match the on-disk vec0 format (0.1.10 is alpha only). |
| `reqwest` | `0.13` | **0.13.1** | Embedding HTTP client. `default-features=false`, features `json, rustls, rustls-native-certs, http2` (0.13 renamed `rustls-tls` -> `rustls`; avoids OpenSSL on macOS). |
| `tokio` | `1` | **1.52.3** | Async runtime; features `rt-multi-thread, macros, io-std, process, sync, time` (`process` for `llama-embed.sh ensure`; `sync` for the ensure mutex). |
| `serde` / `serde_json` | `1` | **1.0.228 / 1.0.150** | (de)serialization. |
| `schemars` | `1` | **1.2.1** | JSON-Schema derive for tool args (re-exported via `rmcp::schemars`). |
| `sha2` / `hex` | `0.10` / `0.4` | **0.10.9 / 0.4.3** | content_hash parity. |
| `bytemuck` | `1` | **1.25.0** | `cast_slice::<f32,u8>` => 10240-byte LE blob == Python `serialize_float32`. |
| `chrono` | `0.4` | **0.4.44** | `created_at_iso` (`datetime.utcfromtimestamp(ts).isoformat()+"Z"`). |
| `thiserror` | `2` | **2.0.18** | error enums. |
| `anyhow` | `1` | **1.0.102** | `main` error glue. |
| `tracing` / `tracing-subscriber` | `0.1` / `0.3` | **0.1.x / 0.3.23** | logging to **stderr only**. |

> Note: `rmcp`'s GitHub README still shows a stale `0.16.0` pin; crates.io
> max_stable is 1.7.0 — we use 1.7.

---

## 3. Reusing the existing sqlite-vec schema — DROP-IN, NO MIGRATION

The on-disk format is byte-identical to what the Rust crates produce, verified
live: `vec_version()` == `v0.1.9`, embeddings are raw LE-f32 (10240 B for dim
2560), `content_hash` is sha256 of `content.strip().lower()`. So the Rust binary
opens `global.db` and every `project.db` **as-is**.

### Schema we touch (real DDL, present in the live DBs)
```sql
CREATE TABLE memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,   -- == memory_embeddings.rowid (JOIN key, not a FK)
  content_hash TEXT UNIQUE NOT NULL, content TEXT NOT NULL,
  tags TEXT,                              -- comma-joined; "untagged" when empty
  memory_type TEXT, metadata TEXT,        -- metadata is a JSON string ("{}" empty)
  created_at REAL, updated_at REAL,       -- epoch float
  created_at_iso TEXT, updated_at_iso TEXT,
  deleted_at REAL DEFAULT NULL,           -- SOFT DELETE tombstone
  parent_id TEXT, version INTEGER DEFAULT 1, confidence REAL DEFAULT 1.0,
  last_accessed INTEGER, superseded_by TEXT
);
CREATE VIRTUAL TABLE memory_embeddings USING vec0(
  content_embedding FLOAT[2560] distance_metric=cosine
);  -- shadow tables auto-managed by the extension; NEVER write them directly
CREATE VIRTUAL TABLE memory_content_fts USING fts5(
  content, content='memories', content_rowid='id', tokenize='trigram'
);  -- kept in sync by memories_fts_ai/au/ad triggers
CREATE TABLE memory_graph (
  source_hash TEXT, target_hash TEXT, similarity REAL,
  connection_types TEXT, metadata TEXT, created_at REAL,
  relationship_type TEXT DEFAULT 'related',
  PRIMARY KEY(source_hash, target_hash)
);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  -- rows: ('distance_metric','cosine'), ('fts5_enabled','true')
```

### Extension loading — static link (preferred)
`sqlite-vec` 0.1.9 statically compiles the C and exports `sqlite3_vec_init`.
Register it **once at process start, before opening any connection**:
```rust
unsafe {
    rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
        sqlite_vec::sqlite3_vec_init as *const (),
    )));
}
```
No dylib and no `enable_load_extension` needed — the extension is compiled in.
`storage::register_vec_extension()` owns this.

### PRAGMAs (match the running Python server / cron so WAL stays compatible)
`journal_mode=WAL`, `busy_timeout=5000`, `synchronous=NORMAL`.

### Atomic write contract (`Storage::store`) — mirrors Python 5 steps
1. exact-hash dedup: `SELECT ... WHERE content_hash=? AND deleted_at IS NULL` -> skip if present.
2. tombstone purge: `DELETE FROM memories WHERE content_hash=? AND deleted_at IS NOT NULL`
   (+ delete the orphaned `memory_embeddings` row for that old id; FTS `_ad` trigger handles FTS cleanup).
3. `INSERT INTO memories(content_hash, content, tags, memory_type, metadata, created_at, updated_at, created_at_iso, updated_at_iso) VALUES (...)` -> capture `last_insert_rowid()`.
4. `INSERT INTO memory_embeddings(rowid, content_embedding) VALUES (lastrowid, blob)`.
5. wrap 2-4 in one transaction; FTS triggers fire automatically. **Never write FTS or vec0 shadow tables directly.**

### Primary read (`Storage::search_semantic`) — exact KNN SQL
```sql
SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
       m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso, e.distance
FROM memories m
INNER JOIN (SELECT rowid, distance FROM memory_embeddings
            WHERE content_embedding MATCH ?1 AND k = ?2) e
  ON m.id = e.rowid
WHERE m.deleted_at IS NULL
  AND (m.superseded_by IS NULL OR m.superseded_by = '')   -- unless include_superseded
  {optional tag filter} {optional time filter}
ORDER BY e.distance LIMIT ?3
```
- `k` is a vec0 constraint (`AND k = ?`, NOT a column; cap **4096**).
- `MATCH ?1` binds the 10240-byte LE-f32 query blob.
- distance = cosine (0..2); `relevance = max(0, 1 - distance/2)`.
- tag filter: `(',' || REPLACE(m.tags,' ','') || ',') LIKE '%,<tag>,%' ESCAPE '\'`.

### Soft delete
`UPDATE memories SET deleted_at = <now> WHERE content_hash=? AND deleted_at IS NULL`.
All reads filter `deleted_at IS NULL`. **Never hard-DELETE on user delete** (breaks id/rowid alignment + audit history).

### Re-embed?
Not needed for compatibility — same model/dim, byte-identical format. A clean
re-embed is required ONLY if the embedding model changes (schema/format unchanged):
walk `memories`, re-embed `content`, overwrite each `memory_embeddings` row by
`rowid`. Not in scope now.

---

## 4. Tool set: v1 must-have vs later

The opencode `permission` allow-list permits exactly **8 tools** per server
(`memory-global_*` / `memory-project_*`); the Python server registers ~20 but
the other 12 are unreachable by these CLIs. We implement the 8.

| Tool | Tier | Output shape (drop-in critical) |
|---|---|---|
| `memory_store` | **v1 must** | PLAIN TEXT `"Memory stored successfully (hash: <hex>)"` |
| `memory_search` | **v1 must** | PLAIN TEXT `"Found <n> memories (mode: <m>) for query: '<q>'"` + lines |
| `memory_list` | **v1 must** | JSON string `{memories, page, page_size, total, total_pages, has_more}` |
| `memory_delete` | **v1 must** | PLAIN TEXT (`"Memory deleted successfully: <h>"` / bulk / dry-run) |
| `memory_health` | **v1 must** | `"Database Health Check Results:\n"` + JSON (statistics, embedding_dimension:2560) |
| `memory_update` | v1 should | PLAIN TEXT message |
| `memory_graph` | v1 should | JSON string; **`connected`** implemented, others stub `{success:false}` |
| `memory_stats` | v1 should | JSON string (process-local cache telemetry; minimal valid object) |

**Search modes**: `semantic` fully (KNN); `exact` (FTS5/BM25), `hybrid` (RRF
fuse), `ranked` are wired but may initially fall back to semantic.

**Deferred / skipped (not allow-listed, unreachable):** `memory_observe,
memory_store_session, memory_cleanup, memory_consolidate, memory_ingest,
memory_quality, memory_harvest, memory_conflicts, memory_resolve, mistake_note_*`.

**Output-format fidelity is the #1 drop-in risk** — callers grep the prose, so
`format.rs` reproduces the per-tool templates character-for-character. Tags are
top-level in search/list/delete/update but live UNDER `metadata.tags` in
`memory_store`.

---

## 5. Embed client + lazy-ensure (`embed.rs`)

Replaces Python `ExternalEmbeddingModel`. Async, single pooled `reqwest::Client`.

- **Endpoint** (default `http://127.0.0.1:11434/v1/embeddings`, overridable via
  `MCP_EXTERNAL_EMBEDDING_URL`). NOTE: the bash wrapper's stale `:8585` default
  is wrong — we default to **11434** (the live port all other infra uses).
- **Request**: `{"input": [..strings..], "model": "qwen3-embedding-4b"}` (model
  value cosmetic; always send `input` as an array). Optional `Authorization: Bearer`.
- **Response**: `{data:[{embedding:[..2560 f32..], index}]}`. Reorder by `index`.
- **No normalization** — server already returns unit vectors (‖v‖≈1.0).
- **Validation**: `len == 2560` and all-finite (else `DimMismatch`/`NonFinite`).
- **Batching**: chunk size 32 (Python parity).

### Lazy-ensure / self-heal (key behavior)
The watchdog (`llama-embed-watchdog.sh`, 15s poll) **stops** the embed server
(~5 GB) when no agent CLI is running, so it may be down at **any** call, not just
startup. Each HTTP call runs in a retry loop:
1. send.
2. on `err.is_connect()` (ConnectionRefused) -> run `llama-embed.sh ensure` ONCE
   (blocks ~40s polling `/health`, returns 0 even on timeout), then retry.
3. on `err.is_timeout()` / 5xx -> capped exponential backoff (3 attempts: 250/500/1000 ms).
4. on exhaustion -> `EmbedError::ServerUnavailable`.
A `tokio::sync::Mutex` guards `ensure_server()` so concurrent embeds spawn at
most one `ensure`. First-call timeout >= ~40s to absorb cold model load.

---

## 6. Two-tier global/project scoping (`config.rs`)

Selected by `argv[1]` (`global` | `project`), exactly like the `memory-mcp`
wrapper. Two completely separate DB connections (one process per scope).

- **Global**: `MCP_MEMORY_SQLITE_PATH || ~/.config/opencode/memory/global.db`.
- **Project**: an explicit **absolute** `MCP_MEMORY_SQLITE_PATH` wins (no
  anchoring); otherwise anchor to the project root:
  1. `git rev-parse --show-toplevel`, else
  2. nearest ancestor (up to `$HOME`) holding a marker (`.git`, `Cargo.toml`,
     `package.json`, `pyproject.toml`, `CLAUDE.md`, ...), else
  3. CWD — then `<root>/.opencode-memory/project.db`.

In practice the wrapper already exports the resolved `MCP_MEMORY_SQLITE_PATH`, so
the in-binary walk is a fallback for direct invocation. The wrapper's
`MCP_MEMORY_BASE_DIR` env is also honored.

---

## 7. Cutover plan (point the CLIs/wrapper at the Rust binary)

The `memory-mcp` wrapper already resolves scope, anchors the DB path, runs
`llama-embed.sh ensure`, and does weekly backup/maintain/consolidate. We keep ALL
of that and change **only the exec target**:

1. `cargo build --release` -> `rust-memory/target/release/opencode-memory`.
2. In `memory-mcp`, replace the binary discovery + `exec "$memory_bin" server`
   with `exec "$RUST_MEMORY_BIN" "$scope"` (the Rust binary takes the scope as
   `argv[1]`; no `server` subcommand). Gate behind an env flag
   (`MEMORY_IMPL=rust|python`, default keep python) for instant rollback.
3. **Validation (no client change needed)**: open a DB read-only and confirm
   `vec_version()`, a known KNN self-query (distance 0.0 to an identical stored
   vector), `memory_search`/`memory_list` against the live `global.db` return the
   same hits as Python, and the `initialize` response advertises name `"memory"`.
4. Flip the flag for `memory-project` first (3 rows, low blast radius), watch a
   session, then `memory-global` (56 rows). Backups already run weekly; take a
   manual `.backup` before the first write via the Rust path.
5. The opencode.jsonc MCP command + permission allow-list are **unchanged** — the
   client still launches `memory-mcp global|project` and sees the same 8 tools.

Embedding format/model unchanged => existing vectors carry over, no re-embed.

---

## 8. Scale path (brute-force vec0 now, HNSW later)

Current volumes are tiny (global.db = 56 live memories; largest project.db = 3),
and vec0 `MATCH ... AND k=N` is a chunked linear scan — instant and correct to
~10k-100k rows for this single-user workload. **Do NOT add an ANN index now.**

If a store ever exceeds ~100k rows: add an HNSW side index (`usearch` or
`hnsw_rs` crate) keyed by `memories.id`, query it for candidate ids, then fetch
rows from `memories`. Keep vec0 as the source of truth + brute-force fallback;
the side index is purely a latency optimization and stays optional/rebuildable.

---

## 9. Open questions / things to verify against running code

- **`ProtocolVersion`**: skeleton advertises `V_2024_11_05` (matches the Python
  examples). The MCP Python SDK's `create_initialization_options()` actually
  defaults to the SDK's latest negotiated version; confirm what the live Python
  server reports in its `initialize` response and match it if it differs. (Server
  *name* is confirmed `"memory"` from `config.py:290`.)
- **`memory_stats` shape**: process-local cache telemetry — emit a minimal valid
  object; verify the live JSON keys against `handlers/utility.py` if any caller
  parses it (the policy/CLIs appear not to).
- **Versioned `memory_update`**: confirm the supersede mechanics (does it set
  `superseded_by` on the old row and insert a new `parent_id`-linked row?) against
  `services/memory_service.py` before implementing the versioned branch.
- **`memory_graph` edges**: `connected` depends on the `memory_graph` table being
  populated by the weekly `consolidate.py` cron (kept as-is for now). Decide later
  whether to port edge-building into Rust or keep relying on the Python cron.
- **rmcp 1.x churn**: macro/module paths verified against 1.7.0
  (`handler::server::router::tool::ToolRouter`,
  `handler::server::wrapper::Parameters`). `ServerInfo`/`Implementation` are
  `#[non_exhaustive]` — built via `Default::default()` + field assignment +
  `Implementation::new(...)`, not struct literals. Re-verify on any version bump.
