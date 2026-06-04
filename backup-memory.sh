#!/usr/bin/env bash
set -euo pipefail

backup_one() {
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

  local db_dir backup_dir backup_path
  db_dir="$(cd "$(dirname "$db_path")" && pwd)"
  backup_dir="$db_dir/backups"
  mkdir -p "$backup_dir"
  backup_path="$backup_dir/${label}-$(date +%Y%m%d-%H%M%S).db"
  sqlite3 "$db_path" ".backup '$backup_path'"
  echo "[OK] $label backup: $backup_path"
}

GLOBAL_DB="${GLOBAL_DB:-$HOME/.config/opencode/memory/global.db}"
PROJECT_DB="${PROJECT_DB:-./.opencode-memory/project.db}"

backup_one global "$GLOBAL_DB"
backup_one project "$PROJECT_DB"
