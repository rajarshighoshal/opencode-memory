#!/usr/bin/env bash
# llama-embed-watchdog.sh
#
# Keeps the llama.cpp embedding server (memory infra) alive only while at least
# one agent CLI (opencode / claude / codex) is running; stops it when none are,
# freeing the ~5 GB model. Replaces ollama-watchdog.sh. Polls every POLL_SECS.
set -uo pipefail

POLL_SECS="${POLL_SECS:-15}"
_src="${BASH_SOURCE[0]}"
while [ -L "$_src" ]; do
  _d="$(cd -P "$(dirname "$_src")" && pwd)"; _src="$(readlink "$_src")"
  [[ "$_src" != /* ]] && _src="$_d/$_src"
done
SCRIPT_DIR="$(cd -P "$(dirname "$_src")" && pwd)"
EMBED="$SCRIPT_DIR/llama-embed.sh"
LOG="${LOG:-/tmp/llama-embed-watchdog.log}"

AGENT_CLIS=(opencode claude claude-code codex codex-cli)
AGENT_DESKTOP_PATTERNS=('/Applications/Claude\.app/' '/Applications/Codex\.app/' '/Applications/ChatGPT\.app/')

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

agent_running() {
  for a in "${AGENT_CLIS[@]}"; do pgrep -x "$a" >/dev/null 2>&1 && return 0; done
  for p in "${AGENT_DESKTOP_PATTERNS[@]}"; do pgrep -f "$p" >/dev/null 2>&1 && return 0; done
  return 1
}

trap 'log "watchdog exiting"; exit 0' INT TERM
log "watchdog started (poll=${POLL_SECS}s, embed=${EMBED})"

while :; do
  if agent_running; then
    [ "$("$EMBED" status)" = up ] || { log "agent present → starting embed"; "$EMBED" start; }
  else
    [ "$("$EMBED" status)" = up ] && { log "no agents → stopping embed"; "$EMBED" stop; }
  fi
  sleep "$POLL_SECS"
done
