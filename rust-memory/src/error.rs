//! Crate-wide error types.
//!
//! Two layers:
//!   * [`EmbedError`] — failures of the llama.cpp embedding client. Variants are
//!     split by failure mode (connect-refused vs bad-response vs dim-mismatch) so
//!     the retry/lazy-ensure logic can branch on them.
//!   * [`MemoryError`] — top-level error for storage + tool handlers. Tool
//!     handlers convert this into a human-readable `Error: ...` text payload
//!     rather than a JSON-RPC error for most tool failures, so `MemoryError` is
//!     mostly for the storage/main plumbing.

use thiserror::Error;

/// Embedding-client errors. The lazy-ensure loop treats [`EmbedError::ServerUnavailable`]
/// as "run `llama-embed.sh ensure` then retry".
#[derive(Debug, Error)]
pub enum EmbedError {
    /// Connection refused / timed out after retries + ensure exhausted.
    /// The llama.cpp server is down (watchdog may have stopped it mid-session).
    #[error("embedding server unavailable: {0}")]
    ServerUnavailable(String),

    /// Transport / HTTP-status error from reqwest that is not a clean connect refusal.
    #[error("embedding http error: {0}")]
    Http(#[from] reqwest::Error),

    /// Response JSON did not match the expected `{data:[{embedding,index}]}` shape.
    #[error("bad embedding response: {0}")]
    BadResponse(String),

    /// Returned vector width != the fixed 2560 the vec0 table expects.
    #[error("embedding dim mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },

    /// Embedding contained NaN/Inf; vectors must be all-finite before storing.
    #[error("embedding contained non-finite value")]
    NonFinite,

    /// Model file failed sha256 integrity check — refuse to serve NaN.
    #[error("model file corrupt: {0}")]
    CorruptModel(String),
}

/// Top-level error for storage operations and server wiring.
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),

    #[error(transparent)]
    Embed(#[from] EmbedError),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error("invalid argument: {0}")]
    InvalidArg(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("{0}")]
    Other(String),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, MemoryError>;
