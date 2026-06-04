# opencode-memory

Persistent, semantic **memory for AI coding agents** — shared across **opencode**, **Claude Code**, and **Codex CLI** via the Model Context Protocol (MCP). Set it up once and every agent on your machine reads from and writes to the same memory.

It's a single self-contained **Rust** binary over **SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec)**, with embeddings from a local **[llama.cpp](https://github.com/ggml-org/llama.cpp)** server. No Python, no cloud, no external database — your memories never leave your machine.

## Why

Agents forget everything between sessions. This gives them a durable, searchable memory — decisions, preferences, gotchas, project conventions — recalled by *meaning*, not just keywords, and shared across every CLI you use.

## Features

- **Two-tier memory** — a **global** store (cross-project facts, preferences) and a **per-project** store that *follows the repo*: move or rename the project folder and its memory comes with it (it lives in `<repo>/.opencode-memory/`).
- **Semantic + hybrid search** — vector KNN over embeddings, plus a `hybrid` mode that fuses vector search with FTS5 BM25 via Reciprocal Rank Fusion, so exact technical tokens (function names, identifiers) rank too.
- **Association graph** — weekly consolidation links related memories by similarity; query `connected` neighbors, shortest `path`, `subgraph`, link-prediction (`infer`), and `suggest`.
- **8 MCP tools** — `memory_store`, `memory_search`, `memory_list`, `memory_update`, `memory_delete`, `memory_health`, `memory_stats`, `memory_graph`.
- **Local & private** — SQLite on disk, embeddings via local llama.cpp; MCP over stdio, never touches the network.
- **Pure Rust** — one statically-linked binary (sqlite-vec compiled in). No runtime dependencies beyond the embedding server.

## Requirements

- **Rust** (`cargo`) — <https://rustup.rs>
- **llama.cpp** (`llama-server` on `PATH`) — `brew install llama.cpp`, or build from source
- **sqlite3** CLI *(optional)* — used by `doctor.sh` and the backup helpers
- macOS or Linux

The default embedding model is **Qwen3-Embedding-4B** (GGUF Q8_0, dim **2560**), auto-downloaded by llama.cpp on first run. A different model works as long as it's also 2560-dim; other dimensions require changing `EMBEDDING_DIM` (and the `vec0` schema) in the source.

## Quick start

```bash
git clone https://github.com/rajarshighoshal/opencode-memory.git
cd opencode-memory
./install.sh
```

`install.sh` builds the Rust server, caches the embedding model, installs the watchdog (macOS launchd), activates the pre-push gate, and prints the config to paste into each CLI. It's idempotent — safe to re-run.

## Per-CLI setup

`install.sh` prints these with paths filled in. All three point at the same `memory-mcp` launcher with a `global` or `project` argument.

- **opencode** — merge `configs/opencode-snippet.jsonc` into the `mcp` block of `~/.config/opencode/opencode.jsonc`.
- **Claude Code** — run the printed `claude mcp add … -- /path/to/memory-mcp global` (and `project`) commands.
- **Codex** — merge `configs/codex-config-snippet.toml` into `~/.codex/config.toml`.

Each server needs only four env vars: `MCP_MEMORY_BASE_DIR`, `MCP_MEMORY_SQLITE_PATH`, `MCP_EXTERNAL_EMBEDDING_URL`, `MCP_EXTERNAL_EMBEDDING_MODEL`. Project scope auto-anchors to the repo root, so a relative `MCP_MEMORY_SQLITE_PATH` is fine.

## Search modes (`memory_search` `mode`)

| mode | what it does |
|---|---|
| `semantic` *(default)* | vector KNN over the sqlite-vec embeddings |
| `hybrid` / `ranked` | Reciprocal Rank Fusion of vector KNN + FTS5 BM25 keyword search |
| `exact` | case-insensitive substring match |

Optional recency reweight: set `MCP_RECENCY_HALFLIFE_DAYS=N` to decay relevance by age (off by default).

## Graph actions (`memory_graph` `action`)

| action | args (defaults) | returns |
|---|---|---|
| `connected` | `hash`, `max_hops` (2) | memories reachable from `hash` (BFS) |
| `path` | `hash` (from), `target` (to), `max_hops` (5) | shortest path between two memories |
| `subgraph` | `hash`, `max_hops` (2) | nodes + edges within N hops |
| `infer` | `hash`, `limit` (10) | link-prediction candidates by shared neighbors |
| `suggest` | `hash`, `limit` (10) | semantic neighbors not yet linked |

## How it works

```
        ┌──────────────────────────────────────────────┐
        │  memory-mcp  (per-session launcher)            │
        │  ├─ ensures the embedding server is up         │
        │  ├─ anchors the project DB to the repo root    │
        │  ├─ weekly backup / maintenance / graph build  │
        │  └─ exec ↓                                     │
        │     opencode-memory  (Rust MCP server, stdio)  │
        │     ├─ SQLite + sqlite-vec  (vec0, dim 2560)   │
        │     └─ embeddings ↓                            │
        │        llama.cpp  (local, :11434)              │
        └──────────────────────────────────────────────┘
                       ▲  MCP over stdio
            ┌──────────┼──────────┐
        opencode   Claude Code   Codex
```

A launchd watchdog (macOS) starts the llama.cpp embedding server only while an agent CLI is running and stops it when idle, so the model isn't resident when you're not working.

## Project layout

| Path | Purpose |
|---|---|
| `rust-memory/` | the Rust MCP server crate (+ `DESIGN.md`) |
| `memory-mcp` | per-session launcher every CLI execs |
| `llama-embed.sh`, `llama-embed-watchdog.sh` | manage the local embedding server |
| `doctor.sh` | health check (binaries, endpoint, DB integrity, graph) |
| `backup-memory.sh`, `maintain-memory.sh` | manual SQLite backup / maintenance |
| `install.sh`, `configs/` | setup + per-CLI config templates |
| `.githooks/pre-push`, `.github/workflows/ci.yml` | build/test/clippy gates |

## Development

```bash
cd rust-memory
cargo build --release
cargo test
cargo clippy -- -D warnings
```

The pre-push hook and CI both run build + test + clippy on changes to `rust-memory/`.

## License

[MIT](LICENSE) © Rajarshi Ghoshal
