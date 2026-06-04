#!/usr/bin/env bash
# llama-embed.sh — manage the llama.cpp embedding server for the memory infra.
#
# Replaces Ollama. Serves Qwen3-Embedding-4B (Q8_0) via llama.cpp's
# OpenAI-compatible /v1/embeddings on a local port. Tuned for batch throughput
# (-ub 2048 --parallel 8 → ~100 emb/sec on M-series; 23 ms single).
#
# Subcommands: start | ensure (start+wait-ready) | stop | status
set -uo pipefail

PORT="${MEMORY_EMBED_PORT:-11434}"
REPO="${MEMORY_EMBED_REPO:-Qwen/Qwen3-Embedding-4B-GGUF:Q8_0}"
LLAMA="${LLAMA_SERVER:-$(command -v llama-server || echo /opt/homebrew/bin/llama-server)}"
LOG="${MEMORY_EMBED_LOG:-/tmp/llama-embed.log}"
READY_TIMEOUT="${MEMORY_EMBED_READY_TIMEOUT:-40}"

is_up() { curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

start() {
  is_up && return 0
  [ -x "$LLAMA" ] || { echo "llama-server not found ($LLAMA)" >&2; return 1; }
  nohup "$LLAMA" -hf "$REPO" --embedding --pooling last \
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

stop() { pkill -f "llama-server.*--port ${PORT}( |\$)" 2>/dev/null || true; }

case "${1:-ensure}" in
  start)  start ;;
  ensure) ensure ;;
  stop)   stop ;;
  status) is_up && echo up || echo down ;;
  *) echo "usage: $0 start|ensure|stop|status" >&2; exit 64 ;;
esac
