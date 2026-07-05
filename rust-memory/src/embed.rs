//! Embedding client for the llama.cpp OpenAI-compatible server.
//!
//! Endpoint: `POST http://127.0.0.1:11434/v1/embeddings`
//!   request : `{"input": <string|[string]>, "model": "qwen3-embedding-4b"}`
//!   response: `{"model","object","usage":{...},"data":[{"embedding":[..2560 f32..],"index":0,"object":"embedding"}]}`
//!
//! Key behaviors to preserve:
//!   * Vectors come back ALREADY L2-normalized (‖v‖≈1.0). Do NOT normalize.
//!   * Batch responses may arrive out of order — reorder by `data[i].index`.
//!   * Vector width is validated on every response against the configured dim.
//!   * Lazy ensure: on connection-refused, bring the server up ONCE (guarded so
//!     concurrent calls don't fan out N starts) and block until `/health` answers,
//!     then retry. The binary starts `llama-server` itself by default — no shell
//!     script required — falling back to an `ensure_script` override when set. A
//!     watchdog may stop the server mid-session, so ensuring once at launch is not
//!     enough.

use crate::config::EmbedConfig;
use crate::error::EmbedError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::path::Path;
use std::time::Duration;
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
    /// `Option` so an absent `index` differs from a present `0` (a server omitting
    /// it would otherwise collapse every item onto slot 0).
    #[serde(default)]
    index: Option<usize>,
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
            let ordered = reorder_and_validate(items, chunk.len(), self.cfg.embedding_dim)?;
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
                    // Bring it up ONCE (blocks until /health is ready), then retry.
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
                Err(e) => {
                    // Other transport error (e.g. a watchdog-severed keep-alive
                    // mid-request). Self-heal once like the connect arm, then back
                    // off; bounded by the loop counter + `ensured`.
                    last_err = Some(EmbedError::Http(e));
                    if !ensured {
                        ensured = true;
                        self.ensure_server().await?;
                        continue;
                    }
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
            }
        }

        Err(last_err.unwrap_or_else(|| {
            EmbedError::ServerUnavailable("retries exhausted".into())
        }))
    }

    /// Bring the embedding server up and block until it answers `/health` (up to
    /// `ready_timeout_secs`). Guarded by `ensure_lock` so concurrent embed calls
    /// trigger at most one start. Best-effort: any failure still lets the caller
    /// retry the request and report `ServerUnavailable` on exhaustion.
    ///
    /// Two start paths, in precedence order:
    ///   1. `ensure_script` (power-user override) — run `<script> ensure`, which
    ///      owns its own readiness wait, so we don't poll afterward.
    ///   2. built-in — spawn `llama-server` directly, then poll `/health` here.
    ///      This is what lets a bare `cargo install` work with no shell scripts.
    async fn ensure_server(&self) -> Result<(), EmbedError> {
        let _guard = self.ensure_lock.lock().await;
        // Re-check under the lock: a concurrent caller may have already started it.
        if self.health_ok().await {
            return Ok(());
        }

        if let Some(ref script) = self.cfg.ensure_script {
            // Ignore the exit status: `ensure` returns 0 even on its own readiness
            // timeout, and a spawn failure just means we retry and report
            // ServerUnavailable on exhaustion.
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
            return Ok(());
        }

        // Only a LOCAL self-start can satisfy a loopback endpoint; for a remote URL,
        // spawning a local llama-server would be pointless, so skip it and wait.
        let local = self.url_is_loopback();
        match self.cfg.llama_server {
            Some(ref bin) if local => {
                // spawn_llama_server returns Err(CorruptModel) if sha256 fails —
                // propagate so the caller sees a clear error instead of NaN.
                self.spawn_llama_server(bin).await?;
                self.wait_until_ready().await;
            }
            _ if !local => {
                tracing::warn!(
                    url = %self.cfg.url,
                    "embedding endpoint is not loopback; not starting a local llama-server — \
                     start it yourself or set MEMORY_EMBED_ENSURE"
                );
            }
            _ => {
                tracing::warn!(
                    "embedding server unreachable and no llama-server found (set LLAMA_SERVER, \
                     put it on PATH, or set MEMORY_EMBED_ENSURE); waiting for an external server"
                );
            }
        }
        Ok(())
    }

    /// `GET <cfg.url scheme+host+port>/health` with a short timeout; `true` on a 2xx.
    /// Probes the SAME endpoint the embeddings POST targets (not a hardcoded
    /// localhost), so the self-heal logic works for remote/https configs too.
    async fn health_ok(&self) -> bool {
        self.http
            .get(self.health_url())
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Derive the `/health` URL from `cfg.url` (same scheme + authority).
    fn health_url(&self) -> String {
        match self.cfg.url.split_once("://") {
            Some((scheme, rest)) => {
                let authority = rest.split('/').next().unwrap_or(rest);
                format!("{scheme}://{authority}/health")
            }
            None => format!("http://127.0.0.1:{}/health", self.cfg.embed_port),
        }
    }

    /// True if `cfg.url`'s host is loopback — the only case where starting a LOCAL
    /// llama-server can serve the configured endpoint.
    fn url_is_loopback(&self) -> bool {
        let host = self
            .cfg
            .url
            .split_once("://")
            .map(|(_, rest)| rest.split('/').next().unwrap_or(rest))
            .map(|authority| match authority.rsplit_once(':') {
                // strip a trailing :port (all-digit); leave bracketed IPv6 intact
                Some((h, p)) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => h,
                _ => authority,
            })
            .unwrap_or("");
        matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]")
    }

    /// Spawn `llama-server` with the throughput-tuned embedding flags (mirrors the
    /// bundled `llama-embed.sh`). Uses `-m <path>` when a local GGUF is resolved
    /// (zero network I/O, no fragile `-hf` downloader); falls back to `-hf` only
    /// when no local file is found. The model path is resolved **lazily** (not at
    /// config-build time) so a `prefetch` done after the binary started is picked
    /// up on the next spawn. sha256 integrity is verified before serving — a
    /// corrupt model fails loudly instead of silently producing NaN embeddings.
    ///
    /// Returns `Err(CorruptModel)` if the sha256 check fails, so the caller can
    /// surface a clear error instead of the old "invalid JSON" confusion.
    ///
    /// A detached reaper thread `wait()`s on the child so it never becomes a
    /// zombie if it later exits (crash / idle watchdog) during this process's
    /// lifetime; if this process exits first the child is reparented to init,
    /// which reaps it. stdio goes to `/tmp/llama-embed.log` for triage.
    async fn spawn_llama_server(&self, bin: &Path) -> Result<(), EmbedError> {
        use std::process::{Command, Stdio};

        let port = self.cfg.embed_port.to_string();
        let (out, err) = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/llama-embed.log")
        {
            Ok(f) => match f.try_clone() {
                Ok(f2) => (Stdio::from(f), Stdio::from(f2)),
                Err(_) => (Stdio::null(), Stdio::null()),
            },
            Err(_) => (Stdio::null(), Stdio::null()),
        };

        // Lazily resolve the model path (not at config-build time) so a prefetch
        // done after the binary started is picked up on the next spawn.
        let model_path = crate::config::resolve_model_path(&self.cfg.embed_repo);

        let mut cmd = Command::new(bin);
        cmd.args(["--embedding", "--pooling", "last", "--host", "127.0.0.1", "--port", &port]);

        // Verify sha256 (if HF cache file) and set -m or -hf.
        match &model_path {
            Some(path) => {
                // Run the blocking sha256 hash on a spawn_blocking thread
                // so we don't stall the tokio worker for ~7.5s.
                let path_clone = path.clone();
                verify_model_integrity(path_clone).await?;
                cmd.arg("-m").arg(path);
            }
            None => {
                cmd.arg("-hf").arg(&self.cfg.embed_repo);
            }
        }

        cmd.args(["-c", "16384", "-b", "8192", "-ub", "2048", "--parallel", "8"])
            .stdin(Stdio::null())
            .stdout(out)
            .stderr(err);

        match cmd.spawn() {
            Ok(mut child) => {
                // Log AFTER successful spawn — not before — so we don't claim
                // success if the spawn failed.
                match &model_path {
                    Some(path) => tracing::info!(
                        bin = %bin.display(), model = %path.display(), port = %port,
                        "started embedding server (llama-server, -m local file)"
                    ),
                    None => tracing::warn!(
                        bin = %bin.display(), repo = %self.cfg.embed_repo, port = %port,
                        "started embedding server (llama-server, -hf fallback — no local model \
                         found, set MEMORY_EMBED_MODEL_PATH or run `llama-embed.sh prefetch`)"
                    ),
                }
                // Reap on exit so a crashed/idle-stopped server doesn't linger as a zombie.
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
            }
            Err(e) => {
                tracing::warn!(bin = %bin.display(), error = %e, "failed to spawn llama-server")
            }
        }
        Ok(())
    }

    /// Poll `/health` once a second until ready or `ready_timeout_secs` elapses.
    /// On timeout we log and return anyway — the caller's retry loop will surface
    /// `ServerUnavailable` if the server truly never came up.
    async fn wait_until_ready(&self) {
        let deadline =
            tokio::time::Instant::now() + Duration::from_secs(self.cfg.ready_timeout_secs);
        loop {
            if self.health_ok().await {
                return;
            }
            if tokio::time::Instant::now() >= deadline {
                tracing::warn!(
                    timeout_secs = self.cfg.ready_timeout_secs,
                    "embedding server not ready before timeout; retrying request anyway"
                );
                return;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
}

/// Verify model file integrity via sha256. Uses a stamp file so the expensive
/// (~4-8s for 4.28 GB) hash computation runs only once per file version.
///
/// The sha256 hash runs on a `spawn_blocking` thread so the tokio worker
/// is not stalled during the ~7.5s computation.
///
/// HF LFS convention: the blob filename IS the content sha256. We resolve the
/// symlink, compare the blob filename to the actual sha256, and cache the
/// result in `<model_path>.verified` containing "<size>:<mtime_secs>".
/// On subsequent calls, if the stamp matches, we skip verification.
///
/// For files outside the HF cache (filename isn't a 64-char hex string),
/// verification is skipped — we can't know the expected hash.
async fn verify_model_integrity(path: PathBuf) -> Result<(), EmbedError> {
    tokio::task::spawn_blocking(move || verify_model_integrity_blocking(&path))
        .await
        .map_err(|e| EmbedError::CorruptModel(format!("verify task panicked: {e}")))?
}

fn verify_model_integrity_blocking(path: &Path) -> Result<(), EmbedError> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    use std::time::UNIX_EPOCH;

    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let stamp_path = path.with_file_name(format!("{file_name}.verified"));

    let metadata = std::fs::metadata(path)
        .map_err(|e| EmbedError::CorruptModel(format!("cannot stat model file: {e}")))?;
    let mtime = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let stamp_key = format!("{}:{}", metadata.len(), mtime);

    // Fast path: stamp exists and matches — file unchanged since last verify.
    if let Ok(stamp) = std::fs::read_to_string(&stamp_path) {
        if stamp.trim() == stamp_key {
            return Ok(());
        }
    }

    // Resolve symlink → blob path. HF LFS: blob filename = expected sha256.
    let real_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let expected_sha = real_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // If the filename isn't a 64-char hex string, it's not an HF cache blob —
    // can't verify, so trust the user's explicit path and write a stamp.
    if expected_sha.len() != 64 || !expected_sha.chars().all(|c| c.is_ascii_hexdigit()) {
        let _ = std::fs::write(&stamp_path, &stamp_key);
        return Ok(());
    }

    // Slow path: compute sha256 and compare to the blob filename.
    tracing::info!(path = %path.display(), "verifying model integrity (sha256)...");
    let mut file = std::fs::File::open(path)
        .map_err(|e| EmbedError::CorruptModel(format!("cannot open model file: {e}")))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024]; // 1 MB buffer — memory-efficient for 4+ GB files
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| EmbedError::CorruptModel(format!("read error: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let actual_sha = hex::encode(hasher.finalize());

    if actual_sha != expected_sha {
        return Err(EmbedError::CorruptModel(format!(
            "sha256 mismatch: expected {expected_sha}, got {actual_sha} — \
             run `llama-embed.sh prefetch` to re-download"
        )));
    }

    let _ = std::fs::write(&stamp_path, &stamp_key);
    Ok(())
}

/// Reorder a chunk's items by their `index` field into a dense `Vec`, validating
/// no duplicate/missing/out-of-bounds indices and that every vector is exactly
/// `dim` wide and finite.
fn reorder_and_validate(items: Vec<EmbedItem>, expected_len: usize, dim: usize) -> Result<Vec<Vec<f32>>, EmbedError> {
    if items.len() != expected_len {
        return Err(EmbedError::BadResponse(format!(
            "expected {expected_len} embeddings, got {}",
            items.len()
        )));
    }
    // Place each item at out[index]. If every item omits `index`, fall back to
    // positional order instead of collapsing onto slot 0.
    let all_missing = items.iter().all(|it| it.index.is_none());
    let mut out: Vec<Option<Vec<f32>>> = (0..expected_len).map(|_| None).collect();
    for (pos, item) in items.into_iter().enumerate() {
        let idx = if all_missing {
            pos
        } else {
            match item.index {
                Some(i) => i,
                None => {
                    return Err(EmbedError::BadResponse(
                        "response mixes items with and without an `index` field".into(),
                    ))
                }
            }
        };
        if idx >= expected_len {
            return Err(EmbedError::BadResponse(format!(
                "out-of-bounds index {idx} for batch size {expected_len}"
            )));
        }
        if out[idx].is_some() {
            return Err(EmbedError::BadResponse(format!("duplicate index {idx}")));
        }
        // Validate dimension + finiteness before accepting.
        if item.embedding.len() != dim {
            return Err(EmbedError::DimMismatch {
                expected: dim,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Scope;

    fn client(url: &str) -> EmbedClient {
        let mut embed = crate::test_util::test_config(Scope::Project).embed;
        embed.url = url.to_string();
        EmbedClient::new(embed).expect("build client")
    }

    #[test]
    fn loopback_detection() {
        assert!(client("http://127.0.0.1:11434/v1/embeddings").url_is_loopback());
        assert!(client("http://localhost:11434/v1/embeddings").url_is_loopback());
        assert!(!client("http://gpu-box:11434/v1/embeddings").url_is_loopback());
        assert!(!client("https://api.example.com/v1/embeddings").url_is_loopback());
    }

    #[test]
    fn health_url_mirrors_endpoint() {
        assert_eq!(
            client("http://127.0.0.1:11434/v1/embeddings").health_url(),
            "http://127.0.0.1:11434/health"
        );
        assert_eq!(
            client("https://gpu:8443/v1/embeddings").health_url(),
            "https://gpu:8443/health"
        );
    }

    // ––– reorder_and_validate: index handling (bug-hunt) –––

    #[test]
    fn reorder_positional_when_all_index_absent() {
        // Server omitted `index` on every item → positional order, not all-slot-0.
        let items = vec![
            EmbedItem { embedding: vec![1.0; 3], index: None },
            EmbedItem { embedding: vec![2.0; 3], index: None },
        ];
        let out = reorder_and_validate(items, 2, 3).unwrap();
        assert_eq!(out[0], vec![1.0; 3]);
        assert_eq!(out[1], vec![2.0; 3]);
    }

    #[test]
    fn reorder_honors_explicit_out_of_order_index() {
        let items = vec![
            EmbedItem { embedding: vec![9.0; 3], index: Some(1) },
            EmbedItem { embedding: vec![8.0; 3], index: Some(0) },
        ];
        let out = reorder_and_validate(items, 2, 3).unwrap();
        assert_eq!(out[0], vec![8.0; 3]);
        assert_eq!(out[1], vec![9.0; 3]);
    }

    #[test]
    fn reorder_rejects_mixed_index_presence() {
        let items = vec![
            EmbedItem { embedding: vec![1.0; 3], index: Some(0) },
            EmbedItem { embedding: vec![2.0; 3], index: None },
        ];
        assert!(reorder_and_validate(items, 2, 3).is_err());
    }
}
