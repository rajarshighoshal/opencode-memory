#!/usr/bin/env bash
#
# install.sh — set up opencode-memory for opencode / Claude Code / Codex.
#
# What it does (pure Rust, no Python):
#   1. Verifies cargo/Rust; installs llama.cpp if missing (Homebrew, else the
#      distro package manager — best-effort, with a build-from-source fallback).
#   2. Builds the Rust MCP server.
#   3. Activates the pre-push gate (build/test/clippy + shell syntax).
#   4. Caches the embedding model (llama.cpp auto-downloads the GGUF).
#   5. Creates the global memory directory.
#   6. Installs the OPTIONAL llama-embed watchdog (stop-when-idle; the binary
#      already lazy-starts the server itself).
#   7. Prints the per-CLI config snippets (pointing at the self-contained binary).
#
# Safe to re-run — each step is idempotent.
set -euo pipefail

# --- paths (override via env for a different layout) ---
GLOBAL_DIR="${GLOBAL_DIR:-$HOME/.config/opencode/memory}"
GLOBAL_DB="${GLOBAL_DB:-$GLOBAL_DIR/global.db}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding-4b}"
EMBEDDING_URL="${EMBEDDING_URL:-http://localhost:11434/v1/embeddings}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_BIN="$SCRIPT_DIR/rust-memory/target/release/opencode-memory"
# MCP clients point straight at the self-contained binary (the memory-mcp wrapper
# is now only a thin compat shim for older configs).
MEMORY_BIN="$RUST_BIN"

echo "==> opencode-memory install"
echo "    MEMORY_BIN = $MEMORY_BIN"
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
# Best-effort llama.cpp install. Package availability varies by platform, so we try
# each manager that actually ships it IN SEQUENCE — re-probing after each so a
# failed one doesn't block the rest — and fall back to a build-from-source pointer.
# System package managers need sudo. (Arch's llama.cpp is AUR-only, hence an AUR helper.)
ensure_llamacpp() {
  if command -v llama-server >/dev/null 2>&1; then return 0; fi
  echo "    llama.cpp not found — attempting to install..."
  if command -v brew >/dev/null 2>&1; then
    echo "    -> Homebrew: brew install llama.cpp"; brew install llama.cpp || true
    if command -v llama-server >/dev/null 2>&1; then return 0; fi
  fi
  if command -v yay >/dev/null 2>&1; then
    echo "    -> yay (AUR): llama.cpp-bin"; yay -S --needed --noconfirm llama.cpp-bin || true
    if command -v llama-server >/dev/null 2>&1; then return 0; fi
  elif command -v paru >/dev/null 2>&1; then
    echo "    -> paru (AUR): llama.cpp-bin"; paru -S --needed --noconfirm llama.cpp-bin || true
    if command -v llama-server >/dev/null 2>&1; then return 0; fi
  fi
  if command -v dnf >/dev/null 2>&1; then
    echo "    -> dnf: sudo dnf install llama-cpp"; sudo dnf install -y llama-cpp || true
    if command -v llama-server >/dev/null 2>&1; then return 0; fi
  fi
  if command -v nix >/dev/null 2>&1; then
    echo "    -> nix: nix profile install nixpkgs#llama-cpp"; nix profile install nixpkgs#llama-cpp || true
    if command -v llama-server >/dev/null 2>&1; then return 0; fi
  fi
  command -v llama-server >/dev/null 2>&1
}
if ! ensure_llamacpp; then
  echo "ERROR: couldn't get 'llama-server' on PATH automatically. Install llama.cpp, then re-run:" >&2
  echo "         macOS / Linux: brew install llama.cpp" >&2
  echo "         Arch (AUR):    yay -S llama.cpp-bin   (or paru, or build from source)" >&2
  echo "         Fedora:        sudo dnf install llama-cpp" >&2
  echo "         Nix:           nix profile install nixpkgs#llama-cpp" >&2
  echo "         Debian/Ubuntu: no distro package — use Homebrew on Linux or build from source" >&2
  echo "         from source:   https://github.com/ggml-org/llama.cpp" >&2
  exit 1
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
chmod +x "$SCRIPT_DIR/llama-embed.sh" "$SCRIPT_DIR/memory-mcp" \
         "$SCRIPT_DIR/llama-embed-watchdog.sh" "$SCRIPT_DIR/doctor.sh" \
         "$SCRIPT_DIR/backup-memory.sh" "$SCRIPT_DIR/maintain-memory.sh"
# `ensure` returns 0 even if it timed out waiting for /health (a cold multi-GB
# model download can exceed the readiness window), so re-probe status to report
# honestly rather than printing a false success.
"$SCRIPT_DIR/llama-embed.sh" ensure || true
if [ "$("$SCRIPT_DIR/llama-embed.sh" status 2>/dev/null)" = "up" ]; then
  echo "    ✓ model cached + endpoint verified"
else
  echo "    ⚠ model still downloading / not ready in ${MEMORY_EMBED_READY_TIMEOUT:-40}s — the first embedding call may block while it finishes"
fi
"$SCRIPT_DIR/llama-embed.sh" stop || true

# ---------------------------------------------------------------------------
# 5. Create the global memory directory
# ---------------------------------------------------------------------------
mkdir -p "$GLOBAL_DIR"
echo "==> Created $GLOBAL_DIR"

# ---------------------------------------------------------------------------
# 6. Watchdog (OPTIONAL) — the binary already lazy-starts the embedding server on
#    demand; the watchdog only adds stop-when-idle so the model isn't resident
#    between sessions. macOS uses launchd; elsewhere wire llama-embed-watchdog.sh
#    into your own supervisor / systemd unit (or just let the binary start it).
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
# Escape sed replacement metacharacters (& | \) so a path containing them still
# templates correctly into the snippets.
sed_rhs() { printf '%s' "$1" | sed 's/[&|\\]/\\&/g'; }
MB=$(sed_rhs "$MEMORY_BIN"); GD=$(sed_rhs "$GLOBAL_DIR"); GDB=$(sed_rhs "$GLOBAL_DB")

echo
echo "==> Setup complete. Wire up your CLI tools:"
echo
echo "──────────────────────────────────────────────────────────────────"
echo "OPENCODE — merge into the 'mcp' block of ~/.config/opencode/opencode.jsonc"
echo "──────────────────────────────────────────────────────────────────"
sed -e "s|{{MEMORY_BIN}}|$MB|g" \
    -e "s|{{GLOBAL_DIR}}|$GD|g" \
    -e "s|{{GLOBAL_DB}}|$GDB|g" \
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
  -- $MEMORY_BIN global

# project scope auto-anchors to the repo root, so no path env is needed:
claude mcp add \\
  -e MCP_EXTERNAL_EMBEDDING_URL=$EMBEDDING_URL \\
  -e MCP_EXTERNAL_EMBEDDING_MODEL=$EMBEDDING_MODEL \\
  -s user memory-project \\
  -- $MEMORY_BIN project
EOF

echo
echo "──────────────────────────────────────────────────────────────────"
echo "CODEX — merge into ~/.codex/config.toml"
echo "──────────────────────────────────────────────────────────────────"
sed -e "s|{{MEMORY_BIN}}|$MB|g" \
    -e "s|{{GLOBAL_DIR}}|$GD|g" \
    -e "s|{{GLOBAL_DB}}|$GDB|g" \
    "$SCRIPT_DIR/configs/codex-config-snippet.toml"

echo
echo "VERIFY:  ./doctor.sh   (then launch any CLI and ask it to list MCP tools)"
echo "Done."
