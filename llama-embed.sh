#!/usr/bin/env bash
# llama-embed.sh — manage the llama.cpp embedding server for the memory infra.
#
# Serves Qwen3-Embedding-4B (Q8_0) via llama.cpp's OpenAI-compatible
# /v1/embeddings on a local port. Tuned for batch throughput
# (-ub 2048 --parallel 8 → ~100 emb/sec on M-series; 23 ms single).
#
# Uses -m <path> to load a pre-downloaded GGUF from the HF cache (or an
# explicit MEMORY_EMBED_MODEL_PATH). This avoids llama.cpp's -hf downloader,
# which does a network validation call on every start and can corrupt the
# cached file if that call glitches. Pre-fetch once with `prefetch` (or
# `hf download`), then `start` loads the local file with zero network I/O.
#
# Subcommands: start | ensure (start+wait-ready) | stop | status | prefetch
set -uo pipefail

PORT="${MEMORY_EMBED_PORT:-11434}"
REPO="${MEMORY_EMBED_REPO:-Qwen/Qwen3-Embedding-4B-GGUF:Q8_0}"
LLAMA="${LLAMA_SERVER:-$(command -v llama-server || echo /opt/homebrew/bin/llama-server)}"
LOG="${MEMORY_EMBED_LOG:-/tmp/llama-embed.log}"
READY_TIMEOUT="${MEMORY_EMBED_READY_TIMEOUT:-40}"
# Explicit model file override; if unset, resolved from the HF cache.
MODEL_PATH="${MEMORY_EMBED_MODEL_PATH:-}"

is_up() { curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

# --- portable helpers --------------------------------------------------------

# Detect whether `stat` is BSD (macOS) or GNU (Linux). BSD stat uses
# -f <format>, GNU stat uses -c <format>. GNU stat -f means --filesystem.
# Tested once; the result is reused via a global.
if stat -f %z /dev/null >/dev/null 2>&1; then
  _STAT_BSD=1
else
  _STAT_BSD=0
fi

# Print file size in bytes, following symlinks. Portable across BSD/GNU stat.
file_size() {
  if [ "$_STAT_BSD" -eq 1 ]; then
    stat -L -f %z "$1" 2>/dev/null
  else
    stat -L -c %s "$1" 2>/dev/null
  fi
}

# Print file mtime in epoch seconds, following symlinks. Portable.
file_mtime() {
  if [ "$_STAT_BSD" -eq 1 ]; then
    stat -L -f %m "$1" 2>/dev/null
  else
    stat -L -c %Y "$1" 2>/dev/null
  fi
}

# Resolve a symlink to its real path (portable). Prefers readlink -f
# (standard on Linux, macOS 12.3+). Falls back to python3 for older macOS.
resolve_realpath() {
  local p="$1"
  readlink -f "$p" 2>/dev/null && return 0
  python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$p" 2>/dev/null && return 0
  # Last resort: return the path as-is (verification will skip if not 64-hex).
  echo "$p"
}

# Compute sha256, portable. Prefers sha256sum (Linux coreutils), falls back
# to shasum -a 256 (macOS / Perl).
compute_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" 2>/dev/null | awk '{print $1}'
  else
    shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'
  fi
}

# --- model path resolution ---------------------------------------------------

# Resolve the GGUF path from the HF cache (or MEMORY_EMBED_MODEL_PATH).
# Sets $MODEL_PATH on success; returns 1 on failure.
#
# HF cache layout:
#   ~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B-GGUF/
#     refs/main                    ← snapshot hash
#     snapshots/<hash>/<file>.gguf ← symlink to ../../blobs/<sha256>
resolve_model_path() {
  # 1. Explicit override — use verbatim if the file exists.
  if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
    return 0
  fi
  # 2. Resolve from the HF cache.
  local repo="${REPO%%:*}"          # Qwen/Qwen3-Embedding-4B-GGUF
  local selector="${REPO#*:}"       # Q8_0
  [ "$selector" = "$REPO" ] && selector=""
  local cache_root="${HF_HOME:-$HOME/.cache/huggingface}/hub"
  local cache_dir="$cache_root/models--${repo//\//--}"
  local refs_file="$cache_dir/refs/main"
  local snapshot
  snapshot=$(cat "$refs_file" 2>/dev/null) || {
    echo "ERROR: model not cached ($refs_file missing)." >&2
    echo "       Run: $0 prefetch" >&2
    echo "       Or:  hf download $repo <file>.gguf" >&2
    echo "       Or:  set MEMORY_EMBED_MODEL_PATH to an existing .gguf" >&2
    return 1
  }
  local snap_dir="$cache_dir/snapshots/${snapshot}"
  # Find the .gguf, preferring one whose name contains the selector (e.g. Q8_0).
  # `ls` sorts alphabetically for deterministic tie-breaking.
  local gguf
  if [ -n "$selector" ]; then
    gguf=$(ls "$snap_dir"/*"$selector"*.gguf 2>/dev/null | head -1)
  fi
  [ -n "$gguf" ] || gguf=$(ls "$snap_dir"/*.gguf 2>/dev/null | head -1)
  if [ -z "$gguf" ]; then
    echo "ERROR: no .gguf found in $snap_dir" >&2
    echo "       Run: $0 prefetch" >&2
    return 1
  fi
  MODEL_PATH="$gguf"
}

# --- integrity verification --------------------------------------------------

# Verify model integrity via sha256. Uses a stamp file so the expensive
# (~4-8s for 4.28 GB) hash computation runs only once per file version.
#
# HF LFS convention: the blob filename IS the content sha256. We resolve
# the symlink, compare the blob filename to the actual sha256, and cache
# the result in <model_path>.verified containing "<size>:<mtime>".
# On subsequent starts, if the stamp matches, we skip verification.
#
# The stamp keys on the BLOB's size+mtime (not the symlink's), matching
# Rust's std::fs::metadata which follows symlinks. This ensures shell and
# Rust agree on the stamp so there's no cross-tool re-hash ping-pong.
#
# For user-provided files (MEMORY_EMBED_MODEL_PATH pointing outside the HF
# cache, where the filename isn't a sha256), verification is skipped — we
# can't know the expected hash.
verify_model() {
  local path="$MODEL_PATH"
  local stamp="${path}.verified"
  local size mtime stamp_key
  size=$(file_size "$path") || return 1
  mtime=$(file_mtime "$path") || return 1
  stamp_key="${size}:${mtime}"

  # Fast path: stamp exists and matches — file unchanged since last verify.
  if [ -f "$stamp" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$stamp_key" ]; then
    return 0
  fi

  # Resolve symlink → blob path. HF LFS: blob filename = expected sha256.
  local real_path expected_sha
  real_path=$(resolve_realpath "$path")
  expected_sha=$(basename "$real_path")

  # If the filename isn't a 64-char hex string, it's not an HF cache blob —
  # can't verify, so trust the user's explicit path and write a stamp.
  if ! echo "$expected_sha" | grep -qE '^[0-9a-f]{64}$'; then
    echo "$stamp_key" > "$stamp" 2>/dev/null || true
    return 0
  fi

  # Slow path: compute sha256 and compare to the blob filename.
  echo "verifying model integrity (sha256)..." >&2
  local actual_sha
  actual_sha=$(compute_sha256 "$path")
  if [ -z "$actual_sha" ] || [ "$actual_sha" != "$expected_sha" ]; then
    echo "ERROR: model file corrupt! sha256 mismatch." >&2
    echo "  expected: $expected_sha" >&2
    echo "  actual:   ${actual_sha:-<empty>}" >&2
    echo "  Run: $0 prefetch  (to re-download)" >&2
    return 1
  fi
  echo "$stamp_key" > "$stamp" 2>/dev/null || true
  return 0
}

# --- server lifecycle --------------------------------------------------------

start() {
  is_up && return 0
  [ -x "$LLAMA" ] || { echo "llama-server not found ($LLAMA)" >&2; return 1; }
  resolve_model_path || return 1
  verify_model || return 1
  echo "[$(date '+%H:%M:%S')] starting llama-server -m $MODEL_PATH (port $PORT)" >> "$LOG"
  nohup "$LLAMA" -m "$MODEL_PATH" --embedding --pooling last \
    --host 127.0.0.1 --port "$PORT" -c 16384 -b 8192 -ub 2048 --parallel 8 \
    >>"$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

ensure() {
  start
  local deadline=$((SECONDS + READY_TIMEOUT))
  until is_up; do
    (( SECONDS >= deadline )) && { echo "embed server not ready in ${READY_TIMEOUT}s" >&2; return 0; }
    sleep 1
  done
}

# Graceful stop: SIGTERM all matching pids, wait up to 5s, SIGKILL survivors.
# Kills ALL llama-server processes on this port — not just the first — so a
# raced multi-start can't leave orphan processes holding the port.
stop() {
  local pids
  pids=$(pgrep -f "llama-server.*--port ${PORT}( |\$)" 2>/dev/null)
  [ -n "$pids" ] || return 0
  # SIGTERM every matching pid.
  echo "$pids" | xargs kill -TERM 2>/dev/null || true
  # Wait up to 5 seconds for graceful shutdown.
  local i=0
  while [ $i -lt 50 ]; do
    local survivors=0
    for pid in $pids; do
      kill -0 "$pid" 2>/dev/null && survivors=$((survivors + 1))
    done
    [ "$survivors" -eq 0 ] && break
    sleep 0.1
    i=$((i + 1))
  done
  # SIGKILL any survivors.
  for pid in $pids; do
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  done
}

# --- model pre-fetch ---------------------------------------------------------

# Pre-fetch the model via `hf download` (sha256-verified, atomic, no -hf).
# If the model is already cached AND passes sha256 verification, skip the
# download. If the cached file is corrupt, delete it and re-fetch.
prefetch() {
  local repo="${REPO%%:*}"
  local selector="${REPO#*:}"
  [ "$selector" = "$REPO" ] && selector=""
  if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: hf CLI not found. Install with: pip install huggingface_hub" >&2
    return 1
  fi

  # Check if we already have a cached .gguf that resolves AND verifies.
  if resolve_model_path 2>/dev/null && [ -f "$MODEL_PATH" ]; then
    if verify_model 2>/dev/null; then
      echo "model already cached and verified: $MODEL_PATH"
      return 0
    fi
    # Cached file is corrupt — delete it and the stamp, then re-fetch.
    echo "cached model failed integrity check — re-downloading..." >&2
    local real_path
    real_path=$(resolve_realpath "$MODEL_PATH")
    rm -f "${MODEL_PATH}.verified" "$real_path" 2>/dev/null || true
    # Clear MODEL_PATH so resolve_model_path re-runs after download.
    MODEL_PATH="${MEMORY_EMBED_MODEL_PATH:-}"
  fi

  echo "prefetching $repo (selector: ${selector:-any})..."
  # Download only the GGUF matching the selector (e.g. Q8_0), not all quants.
  local include_glob="*.gguf"
  [ -n "$selector" ] && include_glob="*${selector}*.gguf"
  local files
  files=$(hf download "$repo" --include "$include_glob" 2>&1) || {
    echo "ERROR: hf download failed:" >&2
    echo "$files" >&2
    return 1
  }
  # Verify it resolves and passes integrity check now.
  if resolve_model_path 2>/dev/null && [ -f "$MODEL_PATH" ]; then
    if verify_model 2>/dev/null; then
      echo "model cached and verified: $MODEL_PATH"
    else
      echo "ERROR: downloaded model failed integrity check" >&2
      return 1
    fi
  else
    echo "ERROR: download completed but model path not resolved" >&2
    return 1
  fi
}

case "${1:-ensure}" in
  start)    start ;;
  ensure)   ensure ;;
  stop)     stop ;;
  status)   is_up && echo up || echo down ;;
  prefetch) prefetch ;;
  *) echo "usage: $0 start|ensure|stop|status|prefetch" >&2; exit 64 ;;
esac
