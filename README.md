# opencode-memory

[![crates.io](https://img.shields.io/crates/v/opencode-memory.svg)](https://crates.io/crates/opencode-memory)
[![release](https://img.shields.io/github/v/release/rajarshighoshal/opencode-memory.svg)](https://github.com/rajarshighoshal/opencode-memory/releases/latest)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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

- **Rust** (`cargo`) — <https://rustup.rs> (to build from source)
- **llama.cpp** (`llama-server` on `PATH`) — `brew install llama.cpp`, or build from source. The binary starts it on demand; you don't run it yourself.
- **sqlite3** CLI *(optional)* — used by `doctor.sh` and the manual backup helpers
- macOS or Linux

The default embedding model is **Qwen3-Embedding-4B** (GGUF Q8_0, dim **2560**), auto-downloaded by llama.cpp on first run. **To use any other model, point `MCP_EXTERNAL_EMBEDDING_URL`/`MODEL` at it and set `MCP_EXTERNAL_EMBEDDING_DIM` to its width** — e.g. `all-MiniLM` (384), `text-embedding-3-small` (1536), BGE (1024). A freshly-created DB is built at that width; an existing DB keeps the width it was created with (auto-detected on open, with a warning if your configured dim disagrees).

## Install

It's one self-contained binary. Pick the install for your platform, then make sure `llama-server` is on `PATH` — the binary lazy-starts it the first time it needs to embed (nothing is resident until you use it). Prebuilt binaries cover `aarch64`/`x86_64` macOS and Linux. After installing, wire up your CLIs (see [Per-CLI setup](#per-cli-setup)).

### macOS

```bash
# prebuilt binary via the release installer (recommended)
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/rajarshighoshal/opencode-memory/releases/latest/download/opencode-memory-installer.sh | sh

# or, no compile, fetch the prebuilt binary
cargo binstall opencode-memory

# or compile from crates.io
cargo install opencode-memory
```

Prerequisite:

```bash
brew install llama.cpp
```

The binary starts `llama-server` on demand (on `:11434`); you never run it yourself. On first embed it downloads the default model (Qwen3-Embedding-4B, GGUF Q8_0, dim 2560).

### Linux

```bash
# prebuilt binary via the release installer (recommended)
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/rajarshighoshal/opencode-memory/releases/latest/download/opencode-memory-installer.sh | sh

# or, no compile, fetch the prebuilt binary
cargo binstall opencode-memory

# or compile from crates.io
cargo install opencode-memory
```

Prerequisite: get `llama-server` onto your `PATH` yourself — build it from <https://github.com/ggml-org/llama.cpp> or install via your distro/package manager (there's no auto-install). The binary then starts it on demand (on `:11434`) and downloads the default model (Qwen3-Embedding-4B, GGUF Q8_0, dim 2560) on first embed.

### Windows

Not supported natively. The server reads `HOME` (not `USERPROFILE`) and hard-errors for global scope when it's unset, writes logs to `/tmp`, looks for a bare `llama-server` (not `llama-server.exe`) and falls back to a Homebrew path, and its helper scripts are bash — none of which is guarded for Windows, and no Windows binary is built.

Use **WSL** (Windows Subsystem for Linux) and follow the [Linux](#linux) steps inside it.

### From source

For the batteries-included path on macOS/Linux — builds the release binary, pre-warms the model, installs the optional idle watchdog, activates the pre-push gate, and prints per-CLI config with paths filled in:

```bash
git clone https://github.com/rajarshighoshal/opencode-memory.git
cd opencode-memory
./install.sh
```

`install.sh` is bash and idempotent — safe to re-run. It also installs llama.cpp for you when missing: Homebrew on macOS (or Linux), else your distro's package manager (`pacman`/`dnf`/`nix`) — best-effort, falling back to a build-from-source pointer if no package is available (e.g. Debian/Ubuntu).

## Per-CLI setup

`install.sh` prints these with paths filled in. All three point at the same `opencode-memory` binary — `opencode-memory` if it's on `PATH`, otherwise its absolute path — with a `global` or `project` argument.

- **opencode** — merge `configs/opencode-snippet.jsonc` into the `mcp` block of `~/.config/opencode/opencode.jsonc`.
- **Claude Code** — run the printed `claude mcp add … -- /path/to/opencode-memory global` (and `project`) commands.
- **Codex** — merge `configs/codex-config-snippet.toml` into `~/.codex/config.toml`.

Because the binary is self-contained, **the command is the only required setting** — every env var is an optional override of a working default (global store under `~/.config/opencode/memory`, embeddings from llama.cpp on `:11434`, dim 2560). Project scope auto-anchors to the repo root, so it needs no path at all. To use a different embedding model, set `MCP_EXTERNAL_EMBEDDING_URL`/`MODEL` and `MCP_EXTERNAL_EMBEDDING_DIM` (its width).

> Older setups can still point at the `memory-mcp` wrapper — it's now a thin shim that just locates and execs the binary.

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

The whole server is **one self-contained binary** — it anchors project memory, starts its own embedding server, and maintains itself. No wrapper or cron required.

```
        ┌────────────────────────────────────────────────────┐
        │  opencode-memory  (one self-contained Rust binary)   │
        │  ├─ anchors the project DB to the repo root          │
        │  ├─ lazily starts the embedding server on demand     │
        │  ├─ weekly backup / maintenance / graph build        │
        │  ├─ SQLite + sqlite-vec  (vec0, cosine, dim 2560)    │
        │  └─ MCP over stdio   ·   embeddings ↓                │
        │     llama.cpp  (local, :11434)                       │
        └────────────────────────────────────────────────────┘
                       ▲  MCP over stdio
            ┌──────────┼──────────┐
        opencode   Claude Code   Codex
```

The binary lazy-starts `llama-server` the first time it needs to embed, so nothing is resident until you actually use it. *Optionally*, a launchd watchdog (macOS) adds the reverse — stopping the embedding server when no agent CLI is running — so the model isn't resident between sessions either.

## Project layout

| Path | Purpose |
|---|---|
| `rust-memory/` | the self-contained Rust MCP server crate (+ `DESIGN.md`) |
| `memory-mcp` | optional thin compat shim (the binary needs no wrapper; older configs that point here still work) |
| `llama-embed.sh`, `llama-embed-watchdog.sh` | optional helpers to start / stop-when-idle the embedding server (the binary lazy-starts it on its own) |
| `doctor.sh` | health check (binaries, endpoint, DB integrity, graph) |
| `backup-memory.sh`, `maintain-memory.sh` | manual SQLite backup / maintenance (the binary also does this weekly) |
| `install.sh`, `configs/` | setup + per-CLI config templates |
| `.githooks/pre-push`, `.github/workflows/ci.yml` | build/test/clippy gates |
| `dist-workspace.toml`, `.github/workflows/release.yml` | [cargo-dist](https://opensource.axo.dev/cargo-dist/) release: prebuilt binaries + installer on tag push |

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
