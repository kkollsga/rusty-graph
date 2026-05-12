//! pyo3 wrappers exposing `mcp-methods` framework tools to Python.
//!
//! 0.9.21+: instead of reimplementing read_source / grep / list_source /
//! github_issues / git_api / workspace tools in pure Python, kglite
//! depends on the pure-Rust `mcp-methods` crate (0.3.26+, three-crate
//! split with zero pyo3 in the library half) and exposes its functions
//! to the Python entry point via this module.
//!
//! The kglite cdylib already uses pyo3's `extension-module` feature
//! (no libpython link, abi3 portable across Python 3.10+), so wrapping
//! more Rust code here costs nothing in the wheel matrix. From the
//! agent's perspective the tool surface and output format are exactly
//! what the 0.9.18 Rust binary produced — same crate, same code paths.
//!
//! Registered as a submodule `kglite._mcp_internal` so it stays clearly
//! internal. The Python `kglite.mcp_server.server` entry point is the
//! intended consumer.

use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Mutex;

use mcp_methods::cache::ElementCache as RustElementCache;
use mcp_methods::github::{git_api_internal, github_issues_rust, has_git_token};
use mcp_methods::server::source::{self, GrepOpts, ListOpts, ReadOpts};

// ---------------------------------------------------------------------------
// source tools
// ---------------------------------------------------------------------------

/// File slice read against the configured source roots. Mirrors
/// `mcp_methods::server::source::read_source`. Path traversal attempts
/// return an `Error: ...` string rather than raising — same convention
/// as the 0.9.18 binary.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    file_path, source_roots, *,
    start_line=None, end_line=None,
    grep=None, grep_context=None,
    max_matches=None, max_chars=None,
))]
fn read_source(
    file_path: &str,
    source_roots: Vec<String>,
    start_line: Option<usize>,
    end_line: Option<usize>,
    grep: Option<String>,
    grep_context: Option<usize>,
    max_matches: Option<usize>,
    max_chars: Option<usize>,
) -> String {
    let opts = ReadOpts {
        start_line,
        end_line,
        grep,
        grep_context,
        max_matches,
        max_chars,
    };
    source::read_source(file_path, &source_roots, &opts)
}

/// Recursive regex search across source roots (ripgrep walker, .gitignore
/// aware). Output format: `path:lineno:content` for matches,
/// `path-lineno-content` for context.
#[pyfunction]
#[pyo3(signature = (
    pattern, source_roots, *,
    glob=None, context=0,
    max_results=None, case_insensitive=false,
))]
fn grep(
    pattern: &str,
    source_roots: Vec<String>,
    glob: Option<String>,
    context: usize,
    max_results: Option<usize>,
    case_insensitive: bool,
) -> String {
    let opts = GrepOpts {
        glob,
        context,
        max_results,
        case_insensitive,
    };
    source::grep(&source_roots, pattern, &opts)
}

/// Tree-style directory listing relative to the primary source root.
#[pyfunction]
#[pyo3(signature = (
    source_roots, *,
    path=String::from("."), depth=1,
    glob=None, dirs_only=false,
))]
fn list_source(
    source_roots: Vec<String>,
    path: String,
    depth: usize,
    glob: Option<String>,
    dirs_only: bool,
) -> String {
    // resolve_dir_under_roots picks the first root as primary.
    let target = match source::resolve_dir_under_roots(&path, &source_roots) {
        Some(p) => p,
        None => return format!("Error: path '{}' does not exist or access denied.", path),
    };
    let primary_root = match source_roots
        .first()
        .and_then(|d| PathBuf::from(d).canonicalize().ok())
    {
        Some(p) => p,
        None => return "Error: no source roots configured.".to_string(),
    };
    let opts = ListOpts {
        depth,
        glob,
        dirs_only,
    };
    source::list_source(&target, &primary_root, &opts)
}

/// Liveness probe — kept on the pyo3 side since it's trivial and there's
/// no equivalent in `mcp-methods` (the Rust framework exposes ping via
/// its `Server` boilerplate, which we don't use).
#[pyfunction]
#[pyo3(signature = (message=None))]
fn ping(message: Option<String>) -> String {
    message.unwrap_or_else(|| "pong".to_string())
}

// ---------------------------------------------------------------------------
// github tools
// ---------------------------------------------------------------------------

/// Generic GitHub REST GET. Relative paths (`pulls?state=open`) auto-
/// prefix with `/repos/<repo>/`; absolute paths (`search/issues?q=...`)
/// pass through. Returns pretty-printed JSON, truncated at `truncate_at`
/// chars.
#[pyfunction]
#[pyo3(signature = (repo, path, *, truncate_at=80_000))]
fn git_api(repo: &str, path: &str, truncate_at: usize) -> String {
    git_api_internal(repo, path, truncate_at)
}

/// Module-level check — used by the Python entry point to decide
/// whether to register the github_* tools at boot.
#[pyfunction]
fn has_github_token() -> bool {
    has_git_token()
}

/// `github_issues` tool. Stateful via [`RustElementCache`] for
/// drill-down (cb_N, comment_N, patch_N). Holds the cache across calls
/// so a follow-up `fetch_issue(number=N, element_id="cb_3")` doesn't
/// re-fetch from GitHub.
#[pyclass]
struct GithubIssues {
    cache: Mutex<RustElementCache>,
}

#[pymethods]
impl GithubIssues {
    #[new]
    fn new() -> Self {
        Self {
            cache: Mutex::new(RustElementCache::new()),
        }
    }

    /// FETCH mode: number given. Returns full issue/PR/discussion body
    /// (with element collapsing when over budget) — drill-down via
    /// element_id + optional lines/grep/context. `refresh=true` bypasses
    /// the cache.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        repo, number, *,
        element_id=None, lines=None,
        grep=None, context=3, refresh=false,
    ))]
    fn fetch(
        &self,
        py: Python<'_>,
        repo: &str,
        number: u64,
        element_id: Option<String>,
        lines: Option<String>,
        grep: Option<String>,
        context: usize,
        refresh: bool,
    ) -> String {
        py.detach(|| {
            let mut guard = self.cache.lock().unwrap();
            guard.fetch_issue(
                repo,
                number,
                element_id.as_deref(),
                lines.as_deref(),
                grep.as_deref(),
                context,
                refresh,
            )
        })
    }

    /// SEARCH or LIST mode — neither number nor element_id; query or
    /// no-args. Mirrors `github_issues_rust` dispatch.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        repo=None, query=None, kind=String::from("all"),
        state=String::from("open"), sort=None, limit=20, labels=None,
    ))]
    fn search_or_list(
        &self,
        py: Python<'_>,
        repo: Option<String>,
        query: Option<String>,
        kind: String,
        state: String,
        sort: Option<String>,
        limit: usize,
        labels: Option<String>,
    ) -> String {
        py.detach(|| {
            github_issues_rust(
                repo.as_deref(),
                None,
                query.as_deref(),
                &kind,
                &state,
                sort.as_deref(),
                limit,
                labels.as_deref(),
            )
        })
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "_mcp_internal")?;
    m.add_function(wrap_pyfunction!(read_source, &m)?)?;
    m.add_function(wrap_pyfunction!(grep, &m)?)?;
    m.add_function(wrap_pyfunction!(list_source, &m)?)?;
    m.add_function(wrap_pyfunction!(ping, &m)?)?;
    m.add_function(wrap_pyfunction!(git_api, &m)?)?;
    m.add_function(wrap_pyfunction!(has_github_token, &m)?)?;
    m.add_class::<GithubIssues>()?;
    parent.add_submodule(&m)?;
    // Register in sys.modules so `import kglite._mcp_internal` works.
    // pyo3's `add_submodule` only sets the attribute; the import system
    // looks up qualified names in sys.modules separately.
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("kglite._mcp_internal", &m)?;
    Ok(())
}
