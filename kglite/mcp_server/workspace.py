"""Workspace mode: clone GitHub repos and build their code-tree graphs.

Mirrors `crates/kglite-mcp-server/src/main.rs`'s workspace path which
in turn called into the mcp-methods Rust crate's `workspace` module.
The framework's surface there exposed `Workspace::open(canon, stale_after_days, hook)`
plus a `set_root_dir` tool that swaps the active repo at runtime.

This Python implementation keeps the same operator surface:
- `--workspace DIR` clones `org/repo` into `DIR/org/repo/` via shallow
  `git clone --depth 1`.
- A `set_root_dir` MCP tool re-points the active graph to a different
  clone in the workspace, building the code-tree on activate.
- `manifest.workspace.kind: local` + `root: ./repos` skips the clone
  and uses a fixed local directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import subprocess
import time
from typing import Any

log = logging.getLogger("kglite.mcp_server.workspace")


@dataclass
class Workspace:
    root: Path
    stale_after_days: int = 7
    kind: str = "remote"  # "remote" or "local"

    def list_repos(self) -> list[Path]:
        """Return paths to all cloned repos under `root`."""
        if not self.root.is_dir():
            return []
        out: list[Path] = []
        for org in self.root.iterdir():
            if not org.is_dir():
                continue
            for repo in org.iterdir():
                if repo.is_dir() and (repo / ".git").exists():
                    out.append(repo)
        return out

    def activate(self, repo: str) -> Path:
        """Clone `org/repo` if not present (or stale), then return the
        local path."""
        if self.kind == "local":
            # Local workspace: repo is a relative path under root.
            target = (self.root / repo).resolve()
            try:
                target.relative_to(self.root.resolve())
            except ValueError:
                raise ValueError(f"local workspace: {repo!r} escapes the root")
            if not target.is_dir():
                raise FileNotFoundError(f"local workspace: {target} not found")
            return target

        # Remote workspace: clone via git.
        if "/" not in repo or repo.count("/") != 1:
            raise ValueError(f"repo must be in 'org/repo' format, got: {repo!r}")
        org, name = repo.split("/", 1)
        target = self.root / org / name

        if target.is_dir() and (target / ".git").is_dir():
            # Stale check: if older than threshold, do `git fetch`.
            mtime = max((target / ".git").stat().st_mtime, target.stat().st_mtime)
            if time.time() - mtime > self.stale_after_days * 86400:
                log.info("refreshing stale clone: %s", target)
                self._run(["git", "fetch", "--depth", "1"], cwd=target)
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{repo}.git"
        log.info("cloning %s ...", url)
        self._run(["git", "clone", "--depth", "1", url, str(target)])
        return target

    def delete(self, repo: str) -> None:
        if self.kind == "local":
            raise ValueError("delete not supported in local workspace mode")
        if "/" not in repo:
            raise ValueError(f"repo must be in 'org/repo' format, got: {repo!r}")
        org, name = repo.split("/", 1)
        target = self.root / org / name
        if target.is_dir():
            import shutil

            shutil.rmtree(target)
            log.info("deleted clone %s", target)

    @staticmethod
    def _run(cmd: list[str], cwd: Path | None = None) -> None:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"command {cmd[:2]!r} failed: {result.stderr.strip()}")


def repo_management_tool_schema() -> dict[str, Any]:
    """JSON-Schema for the workspace `repo_management` tool."""
    return {
        "type": "object",
        "properties": {
            "repo": {
                "type": ["string", "null"],
                "description": "Activate this 'org/repo'. Omit to list cloned repos.",
            },
            "delete": {
                "type": ["boolean", "null"],
                "description": "If true, remove the clone instead of activating.",
            },
            "update": {
                "type": ["boolean", "null"],
                "description": "If true, `git fetch --depth 1` on the active clone.",
            },
        },
    }
