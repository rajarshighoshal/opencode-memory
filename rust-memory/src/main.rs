//! opencode-memory — a single Rust binary providing persistent semantic memory
//! for AI agents over the Model Context Protocol (MCP).
//!
//! A self-contained stdio MCP server, invoked as `opencode-memory <global|project>`.
//! It anchors project memory to the repo root, lazily starts a local llama.cpp
//! embedding server, and runs weekly maintenance itself — no wrapper or cron
//! required. Stores memories in SQLite + sqlite-vec and embeds via the llama.cpp
//! endpoint. Opens an existing sqlite-vec memory DB with no migration when present.
//!
//! CRITICAL: stdout is the JSON-RPC transport. ALL logging goes to stderr.

// A few cosmetic clippy lints are deliberately allowed (they do NOT hide real
// issues): the doc-list lints (`doc_lazy_continuation`, `doc_overindented_list_items`)
// fire on the dense `*`-bulleted technical doc comments — reflowing them to satisfy
// rustdoc's list rules would hurt readability; `missing_transmute_annotations` fires
// on the single documented sqlite-vec FFI transmute, where hand-writing the C
// entrypoint type is verbose and error-prone. Everything else is clippy-clean.
#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::missing_transmute_annotations
)]

mod config;
mod consolidate;
mod embed;
mod error;
mod format;
mod hashing;
mod maintenance;
mod models;
mod storage;
mod tools;

#[cfg(test)]
mod test_util;

use std::sync::Arc;

use config::{Config, Scope};
use embed::EmbedClient;
use rmcp::transport::stdio;
use rmcp::ServiceExt;
use storage::{register_vec_extension, Storage};
use tools::MemoryServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logs -> stderr ONLY (stdout is the protocol channel).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // CLI subcommand: `opencode-memory consolidate <db-path>` — one-shot
    // association-graph build. The server also runs this weekly on its own (see
    // `maintenance`); this entrypoint stays for manual/cron use. No MCP server, no
    // embedding calls (reads stored embeddings). stdout is a human/cron log here,
    // not the JSON-RPC transport.
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("consolidate") {
        let db = argv
            .get(2)
            .ok_or_else(|| anyhow::anyhow!("usage: opencode-memory consolidate <db-path>"))?;
        register_vec_extension()?;
        let r = consolidate::run(db)?;
        println!(
            "consolidate {db}: mem={} window=[{:.2},{:.2}] edges {}->{} pruned={} coverage={}/{}",
            r.memories, r.lo, r.hi, r.edges_before, r.edges_after, r.pruned, r.coverage, r.memories
        );
        return Ok(());
    }

    // argv[1] = scope, exactly like the memory-mcp wrapper (global|project).
    let scope_arg = std::env::args().nth(1).unwrap_or_default();
    let scope = Scope::parse(&scope_arg).ok_or_else(|| {
        anyhow::anyhow!("usage: opencode-memory <global|project> | consolidate <db-path>")
    })?;

    let cfg = Config::from_env(scope)?;
    tracing::info!(?scope, db = %cfg.db_path.display(), "starting opencode-memory");

    // Register the statically-linked sqlite-vec extension BEFORE opening any
    // connection, then open this scope's DB.
    register_vec_extension()?;
    let storage = Arc::new(Storage::open(&cfg)?);
    tracing::info!(vec = %storage.vec_version()?, "sqlite-vec ready");

    // Provenance drift check: warn (once, at startup) if any stored memory was
    // embedded with a different model than the one now configured — otherwise the
    // mismatch is silent and undetectable. Legacy (unstamped) rows are ignored.
    if let Some(found) = storage.embedding_model_mismatch(&cfg.embed.model) {
        tracing::warn!(
            configured = %cfg.embed.model,
            found = %found,
            "stored memories were embedded with a DIFFERENT model; semantic search across the mismatch will degrade — a re-embed is needed"
        );
    }

    // Weekly self-maintenance (backup / integrity-check + optimize / association-
    // graph consolidation) on a detached background thread. Each task is stamp-gated
    // and lock-guarded, so this is a cheap no-op when nothing is due.
    maintenance::spawn(cfg.db_path.clone());

    let embed = Arc::new(EmbedClient::new(cfg.embed.clone())?);
    let config = Arc::new(cfg);

    // Boot the stdio MCP server and block until the peer disconnects.
    let service = MemoryServer::new(storage, embed, config).serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
