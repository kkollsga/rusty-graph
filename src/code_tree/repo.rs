//! GitHub shallow-clone helper (ported from repo.py).

use crate::graph::KnowledgeGraph;
use pyo3::{exceptions::PyRuntimeError, PyResult};
use std::path::{Path, PathBuf};
use std::process::Command;

#[allow(clippy::too_many_arguments)]
pub fn clone_and_build(
    repo: &str,
    save_to: Option<&Path>,
    clone_to: Option<&Path>,
    branch: Option<&str>,
    token: Option<&str>,
    verbose: bool,
    include_tests: bool,
    max_loc_per_file: Option<usize>,
) -> PyResult<KnowledgeGraph> {
    if !repo.contains('/') || repo.matches('/').count() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "repo must be in 'org/repo' format, got: {:?}",
            repo
        )));
    }

    let env_token: Option<String> = std::env::var("GITHUB_TOKEN").ok();
    let token = token.map(str::to_string).or(env_token);

    if let Some(parent) = clone_to {
        let repo_path = clone_repo(repo, parent, branch, token.as_deref(), verbose)?;
        return crate::code_tree::builder::run_with_options(
            &repo_path,
            verbose,
            include_tests,
            save_to,
            max_loc_per_file,
        );
    }

    let tmp = tempfile::tempdir().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let repo_path = clone_repo(repo, tmp.path(), branch, token.as_deref(), verbose)?;
    crate::code_tree::builder::run_with_options(
        &repo_path,
        verbose,
        include_tests,
        save_to,
        max_loc_per_file,
    )
}

fn clone_repo(
    repo: &str,
    parent: &Path,
    branch: Option<&str>,
    token: Option<&str>,
    verbose: bool,
) -> PyResult<PathBuf> {
    let (org, name) = repo.split_once('/').ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("bad repo format: {:?}", repo))
    })?;
    let repo_path = parent.join(org).join(name);

    if repo_path.exists() {
        if verbose {
            eprintln!("Using existing clone at {}", repo_path.display());
        }
        return Ok(repo_path);
    }

    if let Some(parent_dir) = repo_path.parent() {
        std::fs::create_dir_all(parent_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("mkdir failed: {}", e)))?;
    }

    let url = if let Some(t) = token {
        format!("https://x-access-token:{}@github.com/{}.git", t, repo)
    } else {
        format!("https://github.com/{}.git", repo)
    };

    let mut cmd = Command::new("git");
    cmd.args(["clone", "--depth", "1"]);
    if let Some(b) = branch {
        cmd.args(["--branch", b]);
    }
    cmd.arg(&url).arg(&repo_path);

    if verbose {
        eprintln!("Cloning https://github.com/{}.git ...", repo);
    }

    let output = cmd
        .output()
        .map_err(|e| PyRuntimeError::new_err(format!("git command failed: {}", e)))?;

    if !output.status.success() {
        let mut stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if let Some(t) = token {
            stderr = stderr.replace(t, "***");
        }
        return Err(PyRuntimeError::new_err(format!(
            "git clone failed: {}",
            stderr
        )));
    }

    if verbose {
        eprintln!("Cloned to {}", repo_path.display());
    }
    Ok(repo_path)
}
