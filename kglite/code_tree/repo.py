"""Clone a GitHub repository and build a code_tree knowledge graph."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import kglite

from .builder import build


def repo_tree(
    repo: str,
    *,
    save_to: str | Path | None = None,
    clone_to: str | Path | None = None,
    branch: str | None = None,
    token: str | None = None,
    verbose: bool = False,
    include_tests: bool = True,
) -> kglite.KnowledgeGraph:
    """Clone a GitHub repository and build a knowledge graph from its source code.

    Downloads the repo with a shallow clone, parses all supported languages
    using tree-sitter, and returns a populated :class:`KnowledgeGraph`.

    By default the cloned files are deleted after parsing. Pass ``clone_to``
    to keep them at a specific location.

    Args:
        repo: GitHub ``org/repo`` identifier (e.g. ``"pydata/xarray"``).
        save_to: Optional path to persist the graph as a ``.kgl`` file.
        clone_to: Directory to clone into. When provided the files are kept.
            When *None* (default) a temporary directory is used and removed
            after the graph is built.
        branch: Branch, tag, or commit to clone. Defaults to the repo's
            default branch.
        token: GitHub personal access token for private repositories.
            Can also be set via the ``GITHUB_TOKEN`` environment variable.
        verbose: Print progress messages.
        include_tests: Include test files in the graph (default *True*).

    Returns:
        A :class:`KnowledgeGraph` populated with the repository's code
        structure (functions, classes, modules, call edges, etc.).

    Raises:
        ValueError: If *repo* is not in ``org/repo`` format.
        RuntimeError: If ``git clone`` fails (e.g. repo not found, no network).
        ImportError: If tree-sitter is not installed.

    Example::

        from kglite.code_tree import repo_tree

        # Build graph, discard cloned files
        g = repo_tree("pydata/xarray")

        # Keep the clone and save the graph
        g = repo_tree("pydata/xarray", clone_to="./repos", save_to="xarray.kgl")
    """
    if "/" not in repo or repo.count("/") != 1:
        raise ValueError(
            f"repo must be in 'org/repo' format, got: {repo!r}"
        )

    token = token or os.environ.get("GITHUB_TOKEN")

    if clone_to is not None:
        repo_path = _clone(repo, Path(clone_to), branch=branch, token=token, verbose=verbose)
        return build(
            str(repo_path),
            save_to=save_to,
            verbose=verbose,
            include_tests=include_tests,
        )

    # Clone to a tempdir and clean up afterwards
    tmp = tempfile.mkdtemp(prefix="kglite_repo_")
    try:
        repo_path = _clone(repo, Path(tmp), branch=branch, token=token, verbose=verbose)
        return build(
            str(repo_path),
            save_to=save_to,
            verbose=verbose,
            include_tests=include_tests,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _clone(
    repo: str,
    parent: Path,
    *,
    branch: str | None = None,
    token: str | None = None,
    verbose: bool = False,
) -> Path:
    """Shallow-clone a GitHub repo into *parent/org/name*."""
    org, name = repo.split("/", 1)
    repo_path = parent / org / name

    if repo_path.exists():
        if verbose:
            print(f"Using existing clone at {repo_path}")
        return repo_path

    repo_path.parent.mkdir(parents=True, exist_ok=True)

    if token:
        url = f"https://x-access-token:{token}@github.com/{repo}.git"
    else:
        url = f"https://github.com/{repo}.git"

    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(repo_path)]

    if verbose:
        # Don't print the token
        print(f"Cloning https://github.com/{repo}.git ...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        # Strip token from error messages
        stderr = result.stderr.strip()
        if token:
            stderr = stderr.replace(token, "***")
        raise RuntimeError(f"git clone failed: {stderr}")

    if verbose:
        print(f"Cloned to {repo_path}")

    return repo_path
