#!/usr/bin/env bash
#
# install.sh — set up opencode-memory for opencode / Claude Code / Codex.
#
# What it does (pure Rust, no Python):
#   1. Verifies prerequisites (cargo/Rust, llama.cpp).
#   2. Builds the Rust MCP server.
#   3. Activates the pre-push gate (build/test/clippy + shell syntax).
#   4. Caches the embedding model (llama.cpp auto-downloads the GGUF).
#   5. Creates the global memory directory.
#   6. Installs the llama-embed watchdog (macOS launchd; manual elsewhere).
#   7. Prints the per-CLI config snippets to paste into each tool.
#
# Safe to re-run — each step is idempotent.
set -euo pipefail

# --- paths (override via env for a different layout) ---
GLOBAL_DIR="${GLOBAL_DIR:-$HOME/.config/opencode/memory}"
GLOBAL_DB="${GLOBAL_DB:-$GLOBAL_DIR/global.db}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding-4b}"
EMBEDDING_URL="${EMBEDDING_URL:-http://localhost:11434/v1/embeddings}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMORY_MCP="$SCRIPT_DIR/memory-mcp"
RUST_BIN="$SCRIPT_DIR/rust-memory/target/release/opencode-memory"

echo "==> opencode-memory install"
echo "    MEMORY_MCP = $MEMORY_MCP"
echo "    GLOBAL_DB  = $GLOBAL_DB"
echo "    EMBEDDING  = $EMBEDDING_MODEL @ $EMBEDDING_URL"
echo

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
echo "==> Checking prerequisites"
if ! command -v cargo >/dev/null 2>&1; then
  echo "ERROR: cargo/Rust not found. Install from https://rustup.rs" >&2; exit 1
fi
echo "    ✓ cargo: $(cargo --version)"
if ! command -v llama-server >/dev/null 2>&1; then
  echo "    llama.cpp not found."
  if command -v brew >/dev/null 2>&1; then
    echo "    installing via brew..."; brew install llama.cpp
  else
    echo "ERROR: install llama.cpp (https://github.com/ggml-org/llama.cpp) so 'llama-server' is on PATH." >&2; exit 1
  fi
fi
echo "    ✓ llama-server: $(llama-server --version 2>&1 | head -1)"

# ---------------------------------------------------------------------------
# 2. Build the Rust MCP server
# ---------------------------------------------------------------------------
echo "==> Building the Rust MCP server"
( cd "$SCRIPT_DIR/rust-memory" && cargo build --release --quiet )
[ -x "$RUST_BIN" ] || { echo "ERROR: build did not produce $RUST_BIN" >&2; exit 1; }
echo "    ✓ $RUST_BIN"

# ---------------------------------------------------------------------------
# 3. Activate the pre-push gate (only meaningful inside a git checkout)
# ---------------------------------------------------------------------------
if [ -d "$SCRIPT_DIR/.git" ] && [ -f "$SCRIPT_DIR/.githooks/pre-push" ]; then
  chmod +x "$SCRIPT_DIR/.githooks/pre-push"
  git -C "$SCRIPT_DIR" config core.hooksPath .githooks
  echo "==> pre-push hook activated (build/test/clippy on rust-memory changes)"
fi

# ---------------------------------------------------------------------------
# 4. Pre-warm the embedding model (llama.cpp auto-downloads the GGUF)
# ---------------------------------------------------------------------------
echo "==> Pre-warming embedding model ($EMBEDDING_MODEL via llama.cpp)"
chmod +x "$SCRIPT_DIR/llama-embed.sh" "$MEMORY_MCP" \
         "$SCRIPT_DIR/llama-embed-watchdog.sh" "$SCRIPT_DIR/doctor.sh" \
         "$SCRIPT_DIR/backup-memory.sh" "$SCRIPT_DIR/maintain-memory.sh"
"$SCRIPT_DIR/llama-embed.sh" ensure && "$SCRIPT_DIR/llama-embed.sh" stop
echo "    ✓ model cached + endpoint verified"

# ---------------------------------------------------------------------------
# 5. Create the global memory directory
# ---------------------------------------------------------------------------
mkdir -p "$GLOBAL_DIR"
echo "==> Created $GLOBAL_DIR"

# ---------------------------------------------------------------------------
# 6. Watchdog — starts the embedding server only while an agent CLI is running,
#    stops it when idle. macOS uses launchd; elsewhere run llama-embed.sh yourself
#    (or wire llama-embed-watchdog.sh into your own supervisor / systemd unit).
# ---------------------------------------------------------------------------
WATCHDOG_SCRIPT="$SCRIPT_DIR/llama-embed-watchdog.sh"
if [ "$(uname)" = "Darwin" ]; then
  echo "==> Installing llama-embed watchdog (launchd)"
  LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
  PLIST_PATH="$LAUNCH_AGENTS_DIR/ai.opencode.llama-embed-watchdog.plist"
  mkdir -p "$LAUNCH_AGENTS_DIR"
  [ -f "$PLIST_PATH" ] && launchctl unload "$PLIST_PATH" 2>/dev/null || true
  cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>ai.opencode.llama-embed-watchdog</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$WATCHDOG_SCRIPT</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/llama-embed-watchdog.out</string>
    <key>StandardErrorPath</key><string>/tmp/llama-embed-watchdog.err</string>
</dict>
</plist>
EOF
  launchctl load "$PLIST_PATH"
  echo "    ✓ watchdog loaded ($PLIST_PATH)"
else
  echo "==> Non-macOS: start the embedding server with '$SCRIPT_DIR/llama-embed.sh ensure'"
  echo "    or run '$WATCHDOG_SCRIPT' under your own supervisor (it lazy-starts/stops llama.cpp)."
fi

# ---------------------------------------------------------------------------
# 7. Print config snippets
# ---------------------------------------------------------------------------
echo
echo "==> Setup complete. Wire up your CLI tools:"
echo
echo "──────────────────────────────────────────────────────────────────"
echo "OPENCODE — merge into the 'mcp' block of ~/.config/opencode/opencode.jsonc"
echo "──────────────────────────────────────────────────────────────────"
sed -e "s|{{MEMORY_MCP}}|$MEMORY_MCP|g" \
    -e "s|{{GLOBAL_DIR}}|$GLOBAL_DIR|g" \
    -e "s|{{GLOBAL_DB}}|$GLOBAL_DB|g" \
    "$SCRIPT_DIR/configs/opencode-snippet.jsonc"

echo
echo "──────────────────────────────────────────────────────────────────"
echo "CLAUDE CODE — run:"
echo "──────────────────────────────────────────────────────────────────"
cat <<EOF
claude mcp add \\
  -e MCP_MEMORY_BASE_DIR=$GLOBAL_DIR \\
  -e MCP_MEMORY_SQLITE_PATH=$GLOBAL_DB \\
  -e MCP_EXTERNAL_EMBEDDING_URL=$EMBEDDING_URL \\
  -e MCP_EXTERNAL_EMBEDDING_MODEL=$EMBEDDING_MODEL \\
  -s user memory-global \\
  -- $MEMORY_MCP global

claude mcp add \\
  -e MCP_MEMORY_BASE_DIR=./.opencode-memory \\
  -e MCP_MEMORY_SQLITE_PATH=./.opencode-memory/project.db \\
  -e MCP_EXTERNAL_EMBEDDING_URL=$EMBEDDING_URL \\
  -e MCP_EXTERNAL_EMBEDDING_MODEL=$EMBEDDING_MODEL \\
  -s user memory-project \\
  -- $MEMORY_MCP project
EOF

echo
echo "──────────────────────────────────────────────────────────────────"
echo "CODEX — merge into ~/.codex/config.toml"
echo "──────────────────────────────────────────────────────────────────"
sed -e "s|{{MEMORY_MCP}}|$MEMORY_MCP|g" \
    -e "s|{{GLOBAL_DIR}}|$GLOBAL_DIR|g" \
    -e "s|{{GLOBAL_DB}}|$GLOBAL_DB|g" \
    "$SCRIPT_DIR/configs/codex-config-snippet.toml"

echo
echo "VERIFY:  ./doctor.sh   (then launch any CLI and ask it to list MCP tools)"
echo "Done."
