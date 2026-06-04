//! Shared test utilities for opencode-memory unit tests.
//! Compiled only in test mode (`#[cfg(test)]`), never shipped in release builds.
//!
//! Provides:
//!   * `TestDB` — opens a fresh `:memory:` SQLite DB with vec0 + full schema.
//!   * `seeded_embed` — deterministic embedding vectors for predictable search results.
//!   * `store_mem` — convenience helper for storing a memory + returning its hash.
//!   * `str_mem` — convenience helper for building StoreArgs.
//!
//! The vec0 extension is registered exactly ONCE per test binary via `std::sync::Once`.

use crate::config::{Config, EmbedConfig, Scope};
use crate::models::normalize_tags;
use crate::storage::{register_vec_extension, Storage, StoreArgs, StoreOutcome};
use std::path::PathBuf;
use std::sync::Once;

static INIT_VEC: Once = Once::new();

/// Register the statically-linked vec0 extension. Safe to call many times
/// (backs off to a no-op after the first call).
pub fn ensure_vec_extension() {
    INIT_VEC.call_once(|| {
        register_vec_extension().expect("register_vec_extension failed in tests");
    });
}

/// Returns a `Config` pointing at a `:memory:` SQLite DB.
/// Each `Connection::open(":memory:")` creates a fully independent in-memory
/// database, so tests running in parallel never share state.
pub fn test_config(scope: Scope) -> Config {
    ensure_vec_extension();
    Config {
        scope,
        db_path: PathBuf::from(":memory:"),
        embed: EmbedConfig {
            url: "http://localhost:9999/v1/embeddings".to_string(),
            model: "test-model".to_string(),
            api_key: None,
            ensure_script: None,
            timeout_secs: 30,
            batch_size: 32,
            embedding_dim: crate::config::DEFAULT_EMBEDDING_DIM,
        },
    }
}

/// Open a fresh `Storage` backed by a `:memory:` DB with the full schema.
pub fn open_test_storage() -> Storage {
    let cfg = test_config(Scope::Project);
    Storage::open(&cfg).expect("open test storage")
}

/// Build a deterministic embedding vector from a seed byte.
/// Seed 0 → all zeros (invalid, for edge-case tests).
/// Seed 1..=255 → unit-norm vectors; each seed sets only the seed-th element,
/// producing mutually orthogonal vectors (cosine = 0, distance = 1.0 for any
/// pair of distinct non-zero seeds; zero seed is the null vector).
pub fn seeded_embed(seed: u8) -> Vec<f32> {
    if seed == 0 {
        return vec![0.0f32; 2560];
    }
    let mut v = vec![0.0f32; 2560];
    let idx = seed as usize;
    if idx < v.len() {
        v[idx] = 1.0;
    }
    v
}

/// Store a memory with the given content + embedding seed + tags, returning the
/// content_hash. Panics on failure (tests want a clean assert story).
pub fn store_mem(
    storage: &Storage,
    content: &str,
    embed_seed: u8,
    tags: &[&str],
    mem_type: Option<&str>,
) -> String {
    let embedding = seeded_embed(embed_seed);
    let tags: Vec<String> =
        normalize_tags(&tags.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    let store_args = StoreArgs {
        content: content.to_string(),
        tags,
        memory_type: mem_type.map(|s| s.to_string()),
        metadata: serde_json::json!({}),
        conversation_id: None,
    };
    match storage.store(&store_args, &embedding) {
        Ok(StoreOutcome::Stored { content_hash }) => content_hash,
        Ok(StoreOutcome::Duplicate { content_hash }) => {
            panic!("unexpected exact duplicate for content '{content}' (hash {content_hash})")
        }
        Ok(StoreOutcome::SemanticDuplicate { existing_hash }) => {
            panic!(
                "unexpected semantic duplicate for content '{content}' (existing {existing_hash})"
            )
        }
        Err(e) => panic!("store failed for content '{content}': {e}"),
    }
}

/// Build StoreArgs for direct use with `storage.store()`, with the given embed seed.
pub fn str_mem(content: &str, tags: &[&str], mem_type: Option<&str>) -> StoreArgs {
    let tags: Vec<String> =
        normalize_tags(&tags.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    StoreArgs {
        content: content.to_string(),
        tags,
        memory_type: mem_type.map(|s| s.to_string()),
        metadata: serde_json::json!({}),
        conversation_id: None,
    }
}
