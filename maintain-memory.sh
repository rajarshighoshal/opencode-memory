#!/usr/bin/env bash
set -euo pipefail

GLOBAL_DB="${GLOBAL_DB:-$HOME/.config/opencode/memory/global.db}"
PROJECT_DB="${PROJECT_DB:-./.opencode-memory/project.db}"

maintain_one() {
  local label="$1"
  local db_path="$2"

  if [[ ! -f "$db_path" ]]; then
    echo "[SKIP] $label missing: $db_path"
    return 0
  fi

  if ! command -v sqlite3 >/dev/null 2>&1; then
    echo "[FAIL] sqlite3 not found" >&2
    return 1
  fi

  local db_dir backup_dir backup_path qc
  db_dir="$(cd "$(dirname "$db_path")" && pwd)"
  backup_dir="$db_dir/backups"
  mkdir -p "$backup_dir"
  backup_path="$backup_dir/${label}-pre-maintenance-$(date +%Y%m%d-%H%M%S).db"

  sqlite3 "$db_path" ".backup '$backup_path'"
  qc="$(sqlite3 "$db_path" 'PRAGMA quick_check;' 2>/dev/null || true)"
  if [[ "$qc" != "ok" ]]; then
    echo "[FAIL] $label quick_check failed: ${qc:-no output}" >&2
    echo "[INFO] backup kept at: $backup_path" >&2
    return 1
  fi

  sqlite3 "$db_path" 'PRAGMA optimize;'
  echo "[OK] $label maintained: $db_path"
  echo "[OK] $label backup: $backup_path"
}

maintain_one global "$GLOBAL_DB"
maintain_one project "$PROJECT_DB"
