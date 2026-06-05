# opencode-memory

Persistent, semantic **memory for AI coding agents** — one self-contained Rust binary that speaks the **Model Context Protocol (MCP)** over stdio, so [opencode](https://opencode.ai), [Claude Code](https://www.anthropic.com/claude-code), [Codex CLI](https://github.com/openai/codex), and any other MCP client share the same memory.

Storage is **SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec)** (compiled in); embeddings come from a local **[llama.cpp](https://github.com/ggml-org/llama.cpp)** server the binary starts on demand. No Python, no cloud, no external database — memories never leave your machine.

## Install

```bash
cargo install opencode-memory
```

Requires [`llama-server`](https://github.com/ggml-org/llama.cpp) on `PATH` (`brew install llama.cpp`); the binary starts it on demand. Prebuilt binaries and the full multi-CLI setup live in the [project repo](https://github.com/rajarshighoshal/opencode-memory).

## Use

Point any MCP client at the binary with a scope argument — that's the only required setting:

```jsonc
{ "command": ["opencode-memory", "global"] }   // cross-project store
{ "command": ["opencode-memory", "project"] }  // per-project store, anchored to the repo root
```

The binary is self-contained: it anchors project memory to the repo root, lazily starts the embedding server, and runs its own weekly backup / integrity-check / association-graph consolidation. Every default (DB location, embedding endpoint, model, dimension) is overridable via `MCP_*` / `MEMORY_*` env vars.

## Features

- **Two-tier memory** — a **global** store and a **per-project** store that follows the repo (lives in `<repo>/.opencode-memory/`).
- **Semantic + hybrid search** — vector KNN, plus a `hybrid` mode fusing vector search with FTS5 BM25 via Reciprocal Rank Fusion.
- **Association graph** — `connected`, shortest `path`, `subgraph`, link-prediction (`infer`), and `suggest`.
- **8 MCP tools** — `memory_store`, `memory_search`, `memory_list`, `memory_update`, `memory_delete`, `memory_health`, `memory_stats`, `memory_graph`.
- **Swappable embedding model** — defaults to Qwen3-Embedding-4B (dim 2560); set `MCP_EXTERNAL_EMBEDDING_URL`/`MODEL`/`DIM` for any other.

## License

[MIT](LICENSE) © Rajarshi Ghoshal
