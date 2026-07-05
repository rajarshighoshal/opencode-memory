//! In-process weekly maintenance: backup, integrity check + optimize, and
//! association-graph consolidation.
//!
//! Folded in from the former `memory-mcp` shell wrapper so the binary is
//! self-contained — no external cron or scripts needed. [`spawn`] kicks the work
//! onto a detached background thread at serve startup so it never blocks the stdio
//! MCP loop. Each task is:
//!   * stamp-gated — a `.weekly-*.stamp` file next to the DB holds the last-run
//!     epoch; a task runs only if its stamp is missing or older than a week.
//!   * lock-guarded — an atomically-created `.lock` dir prevents two concurrent
//!     server processes (the three agent CLIs can launch at once) from double-running.
//! SQLite WAL makes the separate maintenance connection safe alongside the live
//! server connection. Failures are logged and swallowed: maintenance must never
//! take down serving.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::Connection;

/// One week in seconds — the cadence for every maintenance task.
const WEEK_SECS: u64 = 604_800;
/// Backups older than this (90 days) are pruned after each successful backup.
const BACKUP_RETENTION_SECS: u64 = 90 * 86_400;
/// `busy_timeout` for the short-lived maintenance connections, matching the server.
const BUSY_TIMEOUT_MS: i64 = 5_000;

/// Spawn maintenance on a detached background thread. Returns immediately; the
/// thread runs backup → integrity/optimize → consolidate, each only if due.
pub fn spawn(db_path: PathBuf) {
    std::thread::spawn(move || run_due(&db_path));
}

/// Run each due maintenance task in order. On a fresh install the DB may not exist
/// yet (nothing stored), so a missing file is a clean no-op.
fn run_due(db_path: &Path) {
    if !db_path.exists() {
        return;
    }
    let db_dir = match db_path.parent() {
        Some(d) => d.to_path_buf(),
        None => return,
    };
    backup_if_due(db_path, &db_dir);
    maintain_if_due(db_path, &db_dir);
    consolidate_if_due(db_path, &db_dir);
    evict_if_due(db_path, &db_dir);
}

/// Current wall-clock epoch seconds.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// True if `stamp` is genuinely missing or older than `interval` seconds. A stamp
/// that exists but is corrupt (truncated/garbled write) is treated as NOT due and
/// re-stamped, so a bad write doesn't re-run the (expensive) task on every startup.
fn is_due(stamp: &Path, interval: u64) -> bool {
    match std::fs::read_to_string(stamp) {
        Ok(s) => match s.trim().parse::<u64>() {
            Ok(last) => now_secs().saturating_sub(last) >= interval,
            Err(_) => {
                tracing::warn!(stamp = %stamp.display(), contents = %s.trim(), "maintenance stamp is unparseable; re-stamping and skipping this run");
                stamp_now(stamp);
                false
            }
        },
        Err(_) => true, // absent -> due
    }
}

/// Write the current epoch to `stamp` (best-effort).
fn stamp_now(stamp: &Path) {
    let _ = std::fs::write(stamp, now_secs().to_string());
}

/// A lock dir older than this is presumed stale — left by a process that died
/// before Drop ran. Comfortably longer than any real maintenance run.
const STALE_LOCK_SECS: u64 = 2 * WEEK_SECS;

/// RAII lock backed by an atomic `create_dir`: present = held. Released on drop.
/// `try_acquire` returns `None` when another live process already holds it, but
/// self-heals a stale lock (older than [`STALE_LOCK_SECS`]) so a crash can't block
/// maintenance forever.
struct DirLock(PathBuf);

impl DirLock {
    fn try_acquire(path: PathBuf) -> Option<Self> {
        match std::fs::create_dir(&path) {
            Ok(()) => Some(DirLock(path)),
            Err(_) => {
                // Held. If the existing lock is stale, a previous run likely died
                // before Drop — recover it once. Otherwise it's a live concurrent run.
                match lock_age_secs(&path) {
                    Ok(age) if age >= STALE_LOCK_SECS => {
                        let _ = std::fs::remove_dir(&path);
                        match std::fs::create_dir(&path) {
                            Ok(()) => {
                                tracing::warn!(lock = %path.display(), age_secs = age, "recovered a stale maintenance lock");
                                Some(DirLock(path))
                            }
                            Err(_) => None,
                        }
                    }
                    _ => {
                        tracing::debug!(lock = %path.display(), "maintenance lock held by another run; skipping");
                        None
                    }
                }
            }
        }
    }
}

/// Seconds since the lock dir was last modified (its creation time, in practice).
fn lock_age_secs(path: &Path) -> std::io::Result<u64> {
    let mtime = std::fs::metadata(path)?.modified()?;
    Ok(SystemTime::now()
        .duration_since(mtime)
        .map(|d| d.as_secs())
        .unwrap_or(0))
}

impl Drop for DirLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.0);
    }
}

/// Open a short-lived maintenance connection with the standard busy timeout.
fn open_conn(db_path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open(db_path)?;
    conn.pragma_update(None, "busy_timeout", BUSY_TIMEOUT_MS)?;
    Ok(conn)
}

/// Weekly: write a compacted copy into `<db_dir>/backups/<label>-<ts>.db` via
/// `VACUUM INTO` (one statement, no WAL juggling), then prune copies > 90 days.
fn backup_if_due(db_path: &Path, db_dir: &Path) {
    let stamp = db_dir.join(".weekly-backup.stamp");
    if !is_due(&stamp, WEEK_SECS) {
        return;
    }
    let Some(_lock) = DirLock::try_acquire(db_dir.join(".backup.lock")) else {
        return;
    };

    let backup_dir = db_dir.join("backups");
    if std::fs::create_dir_all(&backup_dir).is_err() {
        return;
    }
    let label = db_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("memory")
        .to_string();
    let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let backup_path = backup_dir.join(format!("{label}-{ts}.db"));

    match backup_via_vacuum(db_path, &backup_path) {
        Ok(()) => {
            stamp_now(&stamp);
            prune_old_backups(&backup_dir, &label);
            tracing::info!(backup = %backup_path.display(), "weekly memory backup written");
        }
        Err(e) => tracing::warn!(error = %e, "weekly memory backup failed"),
    }
}

/// `VACUUM INTO '<dest>'`. The path is single-quote-escaped and inlined because
/// `VACUUM` can't run with a bound parameter; `dest` is one we construct, so this
/// is not user input.
fn backup_via_vacuum(db_path: &Path, dest: &Path) -> rusqlite::Result<()> {
    let conn = open_conn(db_path)?;
    let dest_sql = dest.to_string_lossy().replace('\'', "''");
    conn.execute_batch(&format!("VACUUM INTO '{dest_sql}';"))
}

/// Delete `<label>-*.db` backups whose mtime is older than the retention window.
fn prune_old_backups(backup_dir: &Path, label: &str) {
    let Some(cutoff) = SystemTime::now().checked_sub(Duration::from_secs(BACKUP_RETENTION_SECS))
    else {
        return;
    };
    let prefix = format!("{label}-");
    let Ok(entries) = std::fs::read_dir(backup_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name.starts_with(&prefix) && name.ends_with(".db")) {
            continue;
        }
        if let Ok(modified) = entry.metadata().and_then(|m| m.modified()) {
            if modified < cutoff {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

/// Weekly: gate on `PRAGMA quick_check`, then `PRAGMA optimize`. Skips optimize
/// (and does not stamp) if the integrity check doesn't come back clean.
fn maintain_if_due(db_path: &Path, db_dir: &Path) {
    let stamp = db_dir.join(".weekly-maintenance.stamp");
    if !is_due(&stamp, WEEK_SECS) {
        return;
    }
    let Some(_lock) = DirLock::try_acquire(db_dir.join(".maintenance.lock")) else {
        return;
    };

    let conn = match open_conn(db_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "weekly maintenance: open failed");
            return;
        }
    };
    let qc: String = conn
        .query_row("PRAGMA quick_check", [], |r| r.get(0))
        .unwrap_or_else(|_| "error".to_string());
    if qc != "ok" {
        tracing::warn!(db = %db_path.display(), quick_check = %qc, "memory DB integrity check failed; skipping optimize");
        return;
    }
    let _ = conn.execute_batch("PRAGMA optimize;");
    stamp_now(&stamp);
}

/// Weekly: rebuild the association graph (`memory_graph` edges) so the graph
/// `connected`/`path`/`subgraph` actions stay current. Reads stored embeddings
/// directly — no embedding-server call. The stamp is written BEFORE the (slow)
/// run so a crash mid-build doesn't make every subsequent startup retry it.
fn consolidate_if_due(db_path: &Path, db_dir: &Path) {
    let stamp = db_dir.join(".weekly-consolidate.stamp");
    if !is_due(&stamp, WEEK_SECS) {
        return;
    }
    let Some(_lock) = DirLock::try_acquire(db_dir.join(".consolidate.lock")) else {
        return;
    };
    stamp_now(&stamp);
    match crate::consolidate::run(&db_path.to_string_lossy()) {
        Ok(r) => tracing::info!(
            memories = r.memories, edges_after = r.edges_after, pruned = r.pruned,
            "weekly association-graph consolidation done"
        ),
        Err(e) => tracing::warn!(error = %e, "weekly consolidation failed"),
    }
}

/// Weekly opt-in eviction (off unless `MCP_EVICTION_ENABLED=true`). Gate checked
/// first so a disabled install never locks or stamps.
fn evict_if_due(db_path: &Path, db_dir: &Path) {
    let params = crate::evict::EvictionParams::from_env();
    if !params.enabled {
        return; // opt-in; default off
    }
    let stamp = db_dir.join(".weekly-evict.stamp");
    // A missing stamp means this is the first-ever pass → seed a reinforcement
    // baseline instead of grading a corpus that has no access history yet.
    let first_run = !stamp.exists();
    if !is_due(&stamp, WEEK_SECS) {
        return;
    }
    let Some(_lock) = DirLock::try_acquire(db_dir.join(".evict.lock")) else {
        return;
    };
    stamp_now(&stamp);
    match crate::evict::run(&db_path.to_string_lossy(), &params, first_run) {
        Ok(r) => tracing::info!(
            scanned = r.scanned,
            evicted = r.evicted,
            "weekly eviction pass done"
        ),
        Err(e) => tracing::warn!(error = %e, "weekly eviction failed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn due_when_stamp_missing() {
        let dir = std::env::temp_dir().join(format!("om-maint-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let stamp = dir.join(".weekly-missing.stamp");
        let _ = std::fs::remove_file(&stamp);
        assert!(is_due(&stamp, WEEK_SECS), "missing stamp must be due");

        stamp_now(&stamp);
        assert!(!is_due(&stamp, WEEK_SECS), "fresh stamp must not be due");
        // A zero-interval is always due, even right after stamping.
        assert!(is_due(&stamp, 0), "zero interval is always due");
        let _ = std::fs::remove_file(&stamp);
    }

    #[test]
    fn dir_lock_is_exclusive() {
        let dir = std::env::temp_dir().join(format!("om-lock-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let lock_path = dir.join(".some.lock");
        let _ = std::fs::remove_dir(&lock_path);

        let held = DirLock::try_acquire(lock_path.clone()).expect("first acquire");
        assert!(DirLock::try_acquire(lock_path.clone()).is_none(), "second must fail while held");
        drop(held);
        assert!(DirLock::try_acquire(lock_path.clone()).is_some(), "acquire after release");
        let _ = std::fs::remove_dir(&lock_path);
    }
}
