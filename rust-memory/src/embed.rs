//! Embedding client for the llama.cpp OpenAI-compatible server.
//!
//! Endpoint: `POST http://127.0.0.1:11434/v1/embeddings`
//!   request : `{"input": <string|[string]>, "model": "qwen3-embedding-4b"}`
//!   response: `{"model","object","usage":{...},"data":[{"embedding":[..2560 f32..],"index":0,"object":"embedding"}]}`
//!
//! Key behaviors to preserve:
//!   * Vectors come back ALREADY L2-normalized (‖v‖≈1.0). Do NOT normalize.
//!   * Batch responses may arrive out of order — reorder by `data[i].index`.
//!   * dim is fixed at 2560; validated on every response.
//!   * Lazy ensure: on connection-refused, run `llama-embed.sh ensure` once
//!     (guarded so concurrent calls don't fan out N starts), then retry with
//!     capped exponential backoff. A watchdog can stop the server mid-session,
//!     so ensuring it once at launch is not enough.

use crate::config::{EmbedConfig, EMBEDDING_DIM};
use crate::error::EmbedError;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

/// Request body. We always send `input` as an array (single = 1-element slice)
/// so the response shape is uniform.
#[derive(Debug, Serialize)]
struct EmbedReq<'a> {
    input: &'a [String],
    model: &'a str,
}

/// One `data[i]` item.
#[derive(Debug, Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
    #[serde(default)]
    index: usize,
}

/// Top-level response.
#[derive(Debug, Deserialize)]
struct EmbedResp {
    data: Vec<EmbedItem>,
}

/// Async embedding client. Holds a single pooled `reqwest::Client`; cheap to
/// share behind an `Arc`.
pub struct EmbedClient {
    cfg: EmbedConfig,
    http: reqwest::Client,
    /// Serializes the lazy-ensure subprocess so concurrent embed calls trigger
    /// at most one `llama-embed.sh ensure`.
    ensure_lock: Mutex<()>,
}

impl EmbedClient {
    /// Build the client with a pooled HTTP connection and the configured timeout.
    pub fn new(cfg: EmbedConfig) -> Result<Self, EmbedError> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(cfg.timeout_secs))
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .build()?;
        Ok(Self {
            cfg,
            http,
            ensure_lock: Mutex::new(()),
        })
    }

    /// Embed a single string. Convenience wrapper over [`Self::embed_batch`].
    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let out = self.embed_batch(std::slice::from_ref(&text.to_string())).await?;
        out.into_iter()
            .next()
            .ok_or_else(|| EmbedError::BadResponse("empty data array".into()))
    }

    /// Embed a batch of strings, preserving input order.
    ///
    /// Chunks into `cfg.batch_size` (default 32), POSTs each chunk, reorders each
    /// chunk's results by `index`, validates `len == 2560` and all-finite, then
    /// concatenates in order. Does NOT normalize (vectors arrive L2-normalized).
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.cfg.batch_size.max(1)) {
            let items = self.post_with_retry(chunk).await?;
            let ordered = reorder_and_validate(items, chunk.len())?;
            out.extend(ordered);
        }
        Ok(out)
    }

    /// POST one chunk with the lazy-ensure + backoff retry loop.
    ///
    /// 1. send the request.
    /// 2. on `err.is_connect()` (ConnectionRefused) -> [`Self::ensure_server`] once, then retry.
    /// 3. on `err.is_timeout()` or 5xx -> capped exponential backoff (3 attempts: 250ms/500ms/1s).
    /// 4. on exhaustion -> [`EmbedError::ServerUnavailable`].
    async fn post_with_retry(&self, chunk: &[String]) -> Result<Vec<EmbedItem>, EmbedError> {
        let req = EmbedReq {
            input: chunk,
            model: &self.cfg.model,
        };
        // Backoff schedule for timeout/5xx: 250ms, 500ms, 1s (3 retries).
        let backoff_ms = [250u64, 500, 1000];
        let mut backoff_idx = 0usize;
        let mut ensured = false;
        let mut last_err: Option<EmbedError> = None;

        // Bound total attempts so a persistently-broken server eventually fails.
        for _ in 0..8 {
            let mut builder = self.http.post(&self.cfg.url).json(&req);
            if let Some(ref key) = self.cfg.api_key {
                builder = builder.bearer_auth(key);
            }

            match builder.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let parsed: EmbedResp = resp.json().await.map_err(|e| {
                            EmbedError::BadResponse(format!("invalid JSON: {e}"))
                        })?;
                        return Ok(parsed.data);
                    } else if status.is_server_error() {
                        // 5xx -> backoff + retry.
                        let body = resp.text().await.unwrap_or_default();
                        last_err = Some(EmbedError::BadResponse(format!(
                            "server status {status}: {}",
                            body.chars().take(200).collect::<String>()
                        )));
                        if backoff_idx < backoff_ms.len() {
                            tokio::time::sleep(std::time::Duration::from_millis(
                                backoff_ms[backoff_idx],
                            ))
                            .await;
                            backoff_idx += 1;
                            continue;
                        }
                        break;
                    } else {
                        // 4xx and the like are not retryable.
                        let body = resp.text().await.unwrap_or_default();
                        return Err(EmbedError::BadResponse(format!(
                            "status {status}: {}",
                            body.chars().take(200).collect::<String>()
                        )));
                    }
                }
                Err(e) if e.is_connect() => {
                    // Connection refused — the watchdog likely stopped the server.
                    // Run `llama-embed.sh ensure` ONCE, then retry immediately.
                    if !ensured {
                        ensured = true;
                        self.ensure_server().await?;
                        continue;
                    }
                    last_err = Some(EmbedError::ServerUnavailable(format!(
                        "connection refused after ensure: {e}"
                    )));
                    // After ensure, give the just-started server a beat then retry.
                    if backoff_idx < backoff_ms.len() {
                        tokio::time::sleep(std::time::Duration::from_millis(
                            backoff_ms[backoff_idx],
                        ))
                        .await;
                        backoff_idx += 1;
                        continue;
                    }
                    break;
                }
                Err(e) if e.is_timeout() => {
                    last_err = Some(EmbedError::ServerUnavailable(format!("timeout: {e}")));
                    if backoff_idx < backoff_ms.len() {
                        tokio::time::sleep(std::time::Duration::from_millis(
                            backoff_ms[backoff_idx],
                        ))
                        .await;
                        backoff_idx += 1;
                        continue;
                    }
                    break;
                }
                Err(e) => return Err(EmbedError::Http(e)),
            }
        }

        Err(last_err.unwrap_or_else(|| {
            EmbedError::ServerUnavailable("retries exhausted".into())
        }))
    }

    /// Run `llama-embed.sh ensure` (blocks up to ~40s polling /health, returns 0
    /// even on timeout). Guarded by `ensure_lock` so only one runs at a time.
    /// Best-effort: a non-fatal failure still lets the caller retry the request.
    async fn ensure_server(&self) -> Result<(), EmbedError> {
        let _guard = self.ensure_lock.lock().await;
        if let Some(ref script) = self.cfg.ensure_script {
            // Ignore the exit status: `ensure` returns 0 even on its own
            // readiness timeout, and a spawn failure just means we retry the
            // request and report ServerUnavailable on exhaustion.
            if let Err(e) = tokio::process::Command::new(script)
                .arg("ensure")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await
            {
                tracing::debug!(script = %script.display(), error = %e, "ensure script spawn failed; will retry request");
            }
        }
        Ok(())
    }
}

/// Reorder a chunk's items by their `index` field into a dense `Vec`, validating
/// no duplicate/missing/out-of-bounds indices and that every vector is exactly
/// [`EMBEDDING_DIM`] wide and finite.
fn reorder_and_validate(items: Vec<EmbedItem>, expected_len: usize) -> Result<Vec<Vec<f32>>, EmbedError> {
    if items.len() != expected_len {
        return Err(EmbedError::BadResponse(format!(
            "expected {expected_len} embeddings, got {}",
            items.len()
        )));
    }
    // Place each item at out[index], rejecting duplicate / out-of-bounds indices.
    let mut out: Vec<Option<Vec<f32>>> = (0..expected_len).map(|_| None).collect();
    for item in items {
        let idx = item.index;
        if idx >= expected_len {
            return Err(EmbedError::BadResponse(format!(
                "out-of-bounds index {idx} for batch size {expected_len}"
            )));
        }
        if out[idx].is_some() {
            return Err(EmbedError::BadResponse(format!("duplicate index {idx}")));
        }
        // Validate dimension + finiteness before accepting.
        if item.embedding.len() != EMBEDDING_DIM {
            return Err(EmbedError::DimMismatch {
                expected: EMBEDDING_DIM,
                got: item.embedding.len(),
            });
        }
        if !item.embedding.iter().all(|x| x.is_finite()) {
            return Err(EmbedError::NonFinite);
        }
        out[idx] = Some(item.embedding);
    }

    out.into_iter()
        .enumerate()
        .map(|(i, v)| {
            v.ok_or_else(|| EmbedError::BadResponse(format!("missing embedding for index {i}")))
        })
        .collect()
}
