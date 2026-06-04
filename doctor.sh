#!/usr/bin/env bash
# doctor.sh — health check for opencode-memory (Rust server + llama.cpp embeddings).
set -uo pipefail

failures=0
warnings=0
pass() { echo "[OK] $1"; }
warn() { echo "[WARN] $1"; warnings=$((warnings + 1)); }
fail() { echo "[FAIL] $1" >&2; failures=$((failures + 1)); }

HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:11434/health}"
EMBEDDING_URL="${EMBEDDING_URL:-http://127.0.0.1:11434/v1/embeddings}"
GLOBAL_DB="${GLOBAL_DB:-$HOME/.config/opencode/memory/global.db}"
PROJECT_DB="${PROJECT_DB:-./.opencode-memory/project.db}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_BIN="$SCRIPT_DIR/rust-memory/target/release/opencode-memory"

# --- helper scripts present + executable ---
for helper in memory-mcp llama-embed.sh llama-embed-watchdog.sh backup-memory.sh maintain-memory.sh; do
  if [[ -x "$SCRIPT_DIR/$helper" ]]; then pass "$helper executable"; else fail "$helper missing or not executable"; fi
done

# --- Rust MCP server built ---
if [[ -x "$RUST_BIN" ]]; then
  pass "rust server built: $RUST_BIN"
else
  fail "rust server not built: $RUST_BIN  (run: cd rust-memory && cargo build --release)"
fi

# --- llama.cpp present ---
if command -v llama-server >/dev/null 2>&1; then
  pass "llama-server installed: $(llama-server --version 2>&1 | head -1)"
else
  fail "llama-server not installed (https://github.com/ggml-org/llama.cpp)"
fi

# --- embedding endpoint (llama.cpp on :11434) + dimension ---
if command -v curl >/dev/null 2>&1; then
  if curl -fsS -m 3 "$HEALTH_URL" >/dev/null 2>&1; then
    resp="$(curl -fsS -m 20 -H 'Content-Type: application/json' -d '{"input":"ping","model":"q"}' "$EMBEDDING_URL" 2>/dev/null || true)"
    dim=""
    if command -v jq >/dev/null 2>&1; then
      dim="$(printf '%s' "$resp" | jq -r '.data[0].embedding | length' 2>/dev/null || echo "")"
    elif command -v python3 >/dev/null 2>&1; then
      dim="$(printf '%s' "$resp" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["data"][0]["embedding"]))' 2>/dev/null || echo "")"
    fi
    if [[ "$dim" == "2560" ]]; then pass "embedding endpoint works ($EMBEDDING_URL, dim $dim)"
    elif [[ -n "$dim" && "$dim" != "null" ]]; then warn "embedding endpoint up but dim=$dim (expected 2560)"
    elif [[ -n "$resp" ]]; then pass "embedding endpoint up ($EMBEDDING_URL; dim check skipped — install jq/python3 to verify)"
    else warn "embedding endpoint up but embed request failed"; fi
  else
    warn "embedding endpoint down ($HEALTH_URL) — the watchdog starts it when an agent runs"
  fi
else
  warn "curl not found; skipped embedding probe"
fi

# --- DB health: integrity + memory count + graph edges + dangling edges ---
check_db() {
  local label="$1" db="$2"
  if [[ ! -f "$db" ]]; then warn "$label DB not found: $db"; return 0; fi
  pass "$label DB exists: $db"
  command -v sqlite3 >/dev/null 2>&1 || { warn "sqlite3 not found; skipped $label checks"; return 0; }
  local qc mem edges dangling
  qc="$(sqlite3 "$db" 'PRAGMA quick_check;' 2>/dev/null || true)"
  [[ "$qc" == "ok" ]] && pass "$label quick_check ok" || fail "$label quick_check failed: ${qc:-none}"
  mem="$(sqlite3 "$db" 'SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL' 2>/dev/null || echo '?')"
  edges="$(sqlite3 "$db" 'SELECT COUNT(*) FROM memory_graph' 2>/dev/null || echo 0)"
  dangling="$(sqlite3 "$db" 'SELECT COUNT(*) FROM memory_graph WHERE source_hash NOT IN (SELECT content_hash FROM memories WHERE deleted_at IS NULL) OR target_hash NOT IN (SELECT content_hash FROM memories WHERE deleted_at IS NULL)' 2>/dev/null || echo 0)"
  pass "$label memories=$mem  graph_edges=$edges"
  if [[ "$dangling" =~ ^[0-9]+$ && "$dangling" -gt 0 ]]; then
    warn "$label has $dangling dangling graph edge(s) — run: $RUST_BIN consolidate $db"
  else
    pass "$label no dangling graph edges"
  fi
}
check_db global "$GLOBAL_DB"
check_db project "$PROJECT_DB"

# --- watchdog (macOS launchd only) ---
if [[ "$(uname)" == "Darwin" ]]; then
  if launchctl list 2>/dev/null | grep -q llama-embed-watchdog; then
    pass "llama-embed-watchdog launch agent loaded"
  else
    warn "llama-embed-watchdog not loaded (launchctl load ~/Library/LaunchAgents/ai.opencode.llama-embed-watchdog.plist)"
  fi
fi

echo
if [[ "$failures" -ne 0 ]]; then
  echo "Doctor finished with $failures failure(s), $warnings warning(s)." >&2
  exit 1
fi
echo "Doctor passed with $warnings warning(s)."
