//! Runtime configuration, resolved from environment + argv scope.
//!
//! Mirrors the `memory-mcp` bash wrapper's env contract so the Rust binary is a
//! drop-in: same `MCP_MEMORY_SQLITE_PATH`, same embedding env vars, same
//! `global | project` scope argument.

use std::path::PathBuf;

/// Server name advertised in the MCP `initialize` response. The Python low-level
/// server uses `Server(SERVER_NAME)` with `SERVER_NAME = "memory"` (config.py:290),
/// so we advertise the same name for byte-identical client negotiation.
pub const SERVER_NAME: &str = "memory";

/// Fixed embedding dimension of the existing vec0 tables (`FLOAT[2560]`).
pub const EMBEDDING_DIM: usize = 2560;

/// vec0 KNN hard cap (`_SQLITE_VEC_MAX_KNN_K` in the Python impl).
pub const MAX_KNN_K: usize = 4096;

/// Two-tier scope, selected by argv\[1\] (`global` or `project`), exactly like the
/// `memory-mcp` wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Cross-project user-level store: `~/.config/opencode/memory/global.db`.
    Global,
    /// Per-project store anchored to the project root: `<root>/.opencode-memory/project.db`.
    Project,
}

impl Scope {
    /// Parse the argv scope token. Returns `None` for anything but `global`/`project`
    /// (the wrapper exits 64 on bad usage).
    pub fn parse(arg: &str) -> Option<Self> {
        match arg {
            "global" => Some(Scope::Global),
            "project" => Some(Scope::Project),
            _ => None,
        }
    }
}

/// Embedding-client configuration. Defaults match the live infra (port 11434,
/// NOT the wrapper's stale :8585 default).
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    /// `MCP_EXTERNAL_EMBEDDING_URL`, default `http://127.0.0.1:11434/v1/embeddings`.
    pub url: String,
    /// `MCP_EXTERNAL_EMBEDDING_MODEL`, default `qwen3-embedding-4b` (value cosmetic — llama.cpp serves the loaded GGUF).
    pub model: String,
    /// `MCP_EXTERNAL_EMBEDDING_API_KEY`, optional Bearer token.
    pub api_key: Option<String>,
    /// Absolute path to `llama-embed.sh` for lazy-ensure self-heal. Resolved from
    /// `MEMORY_EMBED_ENSURE` or next to the binary.
    pub ensure_script: Option<PathBuf>,
    /// Per-request timeout; >= ~40s to absorb cold model load on the first call.
    pub timeout_secs: u64,
    /// Batch size for `encode` (Python uses 32; keep identical for parity).
    pub batch_size: usize,
}

/// Fully resolved configuration for one server process.
#[derive(Debug, Clone)]
pub struct Config {
    /// Retained for context/telemetry; the DB path is derived from it at construction.
    #[allow(dead_code)]
    pub scope: Scope,
    /// Absolute path to the sqlite-vec DB for this scope (`MCP_MEMORY_SQLITE_PATH`).
    pub db_path: PathBuf,
    pub embed: EmbedConfig,
}

impl Config {
    /// Resolve full config from the scope arg + environment.
    ///
    /// Honors an explicit absolute `MCP_MEMORY_SQLITE_PATH` (wins, no anchoring);
    /// otherwise derives the path from scope:
    ///   * Global  -> `$MCP_MEMORY_SQLITE_PATH || ~/.config/opencode/memory/global.db`
    ///   * Project -> project-root walk (git toplevel, then marker walk, then CWD)
    ///                + `/.opencode-memory/project.db`, matching the wrapper.
    ///
    /// In practice the `memory-mcp` wrapper already exports the resolved
    /// `MCP_MEMORY_SQLITE_PATH`, so the project-root walk here is a fallback for
    /// direct invocation.
    pub fn from_env(scope: Scope) -> crate::error::Result<Self> {
        let home = std::env::var_os("HOME").map(PathBuf::from);

        // DB path resolution mirrors the memory-mcp wrapper:
        //   * An explicit ABSOLUTE MCP_MEMORY_SQLITE_PATH always wins (no anchoring).
        //   * Otherwise: Global -> ~/.config/opencode/memory/global.db;
        //               Project -> <project-root>/.opencode-memory/project.db.
        let env_path = std::env::var_os("MCP_MEMORY_SQLITE_PATH").map(PathBuf::from);
        let db_path = match env_path {
            Some(p) if p.is_absolute() => p,
            _ => match scope {
                Scope::Global => {
                    let base = home
                        .clone()
                        .ok_or_else(|| crate::error::MemoryError::Other("HOME not set".into()))?;
                    base.join(".config/opencode/memory/global.db")
                }
                Scope::Project => {
                    let cwd = std::env::current_dir()
                        .map_err(|e| crate::error::MemoryError::Other(format!("cwd: {e}")))?;
                    Config::find_project_root(&cwd)
                        .join(".opencode-memory")
                        .join("project.db")
                }
            },
        };

        // Embedding config — port-11434 defaults (NOT the wrapper's stale :8585).
        let url = std::env::var("MCP_EXTERNAL_EMBEDDING_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:11434/v1/embeddings".to_string());
        let model = std::env::var("MCP_EXTERNAL_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "qwen3-embedding-4b".to_string());
        let api_key = std::env::var("MCP_EXTERNAL_EMBEDDING_API_KEY").ok().filter(|s| !s.is_empty());

        // Resolve llama-embed.sh for lazy-ensure self-heal: explicit override
        // (MEMORY_EMBED_ENSURE), else next to the running binary, else at the repo
        // root (binary lives at <repo>/rust-memory/target/release/opencode-memory).
        let ensure_script = std::env::var_os("MEMORY_EMBED_ENSURE")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::current_exe().ok().and_then(|exe| {
                    exe.parent().map(|d| d.join("llama-embed.sh"))
                })
            })
            .or_else(|| {
                std::env::current_exe()
                    .ok()
                    .and_then(|exe| exe.ancestors().nth(4).map(|root| root.join("llama-embed.sh")))
            })
            .filter(|p| p.exists());

        let timeout_secs = std::env::var("MCP_EXTERNAL_EMBEDDING_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);

        Ok(Config {
            scope,
            db_path,
            embed: EmbedConfig {
                url,
                model,
                api_key,
                ensure_script,
                timeout_secs,
                batch_size: 32,
            },
        })
    }

    /// Walk from `cwd` to the project root (git toplevel, else nearest ancestor
    /// holding a project marker, else `cwd`). Mirrors the wrapper's detection.
    pub fn find_project_root(cwd: &std::path::Path) -> PathBuf {
        // 1. git toplevel
        if let Ok(out) = std::process::Command::new("git")
            .arg("-C")
            .arg(cwd)
            .args(["rev-parse", "--show-toplevel"])
            .output()
        {
            if out.status.success() {
                let top = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !top.is_empty() {
                    return PathBuf::from(top);
                }
            }
        }

        // 2. nearest ancestor (up to $HOME) holding a project marker.
        let home = std::env::var_os("HOME").map(PathBuf::from);
        const DIR_MARKERS: &[&str] = &[".opencode-memory", ".git", ".hg", ".svn"];
        const FILE_MARKERS: &[&str] = &[
            "pyproject.toml", "setup.py", "package.json", "Cargo.toml", "go.mod",
            "pom.xml", "CMakeLists.txt", "CLAUDE.md", "AGENTS.md",
        ];
        let mut d: &std::path::Path = cwd;
        loop {
            if Some(d) == home.as_deref() || d.parent().is_none() {
                break;
            }
            let has_marker = DIR_MARKERS.iter().any(|m| d.join(m).is_dir())
                || FILE_MARKERS.iter().any(|m| d.join(m).is_file());
            if has_marker {
                return d.to_path_buf();
            }
            match d.parent() {
                Some(p) => d = p,
                None => break,
            }
        }

        // 3. CWD as last resort.
        cwd.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_parse_global() {
        assert_eq!(Scope::parse("global"), Some(Scope::Global));
    }

    #[test]
    fn scope_parse_project() {
        assert_eq!(Scope::parse("project"), Some(Scope::Project));
    }

    #[test]
    fn scope_parse_invalid() {
        assert_eq!(Scope::parse("anything_else"), None);
        assert_eq!(Scope::parse(""), None);
    }

    #[test]
    fn project_root_finds_via_git() {
        // This test runs inside this crate's git repo, so find_project_root
        // should resolve the root via the git toplevel.
        let root = Config::find_project_root(
            &std::env::current_dir().expect("cwd"),
        );
        assert!(root.join(".git").is_dir(), "expected root with .git; got {root:?}");
    }

    #[test]
    fn project_root_finds_via_parent_cargo_toml() {
        // CWD is the crate root (rust-memory/) which has Cargo.toml.
        // But git toplevel wins, returning the workspace root.
        let cwd = std::env::current_dir().expect("cwd");
        assert!(cwd.join("Cargo.toml").is_file(), "test must run in crate root");
        let root = Config::find_project_root(&cwd);
        // Workspace root has .git; Cargo.toml lives in rust-memory/ not at root.
        assert!(root.join(".git").is_dir(), "expected git-toplevel as root; got {root:?}");
    }
}
