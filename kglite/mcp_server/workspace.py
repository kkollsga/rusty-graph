"""Workspace mode — thin wrapper around mcp-methods Rust.

0.9.24: replaced the pure-Python git-subprocess + path-validation
implementation with `kglite._mcp_internal.Workspace` (mcp-methods
0.3.28). Rust handles:
- Repo cloning + stale sweeping + inventory persistence.
- Atomic active-root swaps via RwLock (no sandbox-narrowing bug —
  the configured root is immutable for the lifetime of the Workspace,
  and `clone_or_update`'s local branch reads the current bound root
  rather than the immutable configured root, so post-activate hooks
  fire against the target of the most recent `set_root_dir`).
- `last_built_sha` gating for post-activate hook reruns.

Sandbox check: the Python wrapper still enforces "target must be under
the configured workspace root" on `set_root_dir_tool`, but uses
`workspace_dir()` (set at construction, never narrowed) as the
boundary — the pre-0.9.24 bug came from using the mutable active
root as the boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from kglite import _mcp_internal

log = logging.getLogger("kglite.mcp_server.workspace")


class Workspace:
    """Workspace handle — github clone-tracker OR local-directory bind.

    Preserves the pre-0.9.24 surface: `root` property, `kind`
    attribute, `active_repo_name()`, `repo_management_tool(args)`,
    `set_root_dir_tool(path)`. server.py and the test suite see the
    same shape; the implementation is now ~150 LOC of Python deleted
    in favour of the validated Rust path.
    """

    def __init__(
        self,
        root: Path,
        kind: str = "remote",
        stale_after_days: int = 7,
    ) -> None:
        # Preserve the original label vocabulary ("remote" / "local")
        # for the rest of the Python codebase even though mcp-methods
        # spells github mode as "github".
        self._kind_label = "local" if kind == "local" else "remote"
        if kind == "local":
            self._inner = _mcp_internal.Workspace.open_local(str(root))
        else:
            self._inner = _mcp_internal.Workspace.open(str(root), stale_after_days)
        # Cache the configured workspace dir at construction so the
        # sandbox check on `set_root_dir_tool` always validates against
        # the immutable boundary, not the (mutable) active root.
        self._configured_root = Path(self._inner.workspace_dir()).resolve()

    def set_post_activate(self, callback: Callable[[str, str], None]) -> None:
        """Register a `(repo_path, repo_name) -> None` callback fired
        after each successful activate / set_root_dir. One-shot per
        Workspace instance — calling twice raises."""
        self._inner.set_post_activate(callback)

    @property
    def kind(self) -> str:
        """'local' or 'remote'. Matches the pre-0.9.24 label set."""
        return self._kind_label

    @property
    def root(self) -> Path:
        """Active source root. Reflects the most recent successful
        `set_root_dir` / `activate` via mcp-methods' atomic RwLock —
        reads through to `active_repo_path()` directly."""
        active = self._inner.active_repo_path()
        if active:
            return Path(active)
        return self._configured_root

    def active_repo_name(self) -> str | None:
        """Active repo as 'org/repo' (remote mode) or
        'local/<basename>' (local mode), or None when nothing is
        active. Consumed by github_issues / github_api for default-
        repo resolution."""
        return self._inner.active_repo_name()

    def repo_management_tool(self, args: dict[str, Any]) -> str:
        """Dispatch the `repo_management` MCP tool. The Rust side
        formats the user-facing body string."""
        return self._inner.repo_management(
            name=args.get("name"),
            delete=bool(args.get("delete")),
            update=bool(args.get("update")),
            force_rebuild=bool(args.get("force_rebuild")),
        )

    def set_root_dir_tool(self, path: str) -> str:
        """Dispatch the `set_root_dir` MCP tool (local-workspace only).

        Sandbox check: the resolved target must be a descendant of
        the configured workspace dir (set at __init__, immovable).
        Without this check mcp-methods would accept any absolute path
        on the filesystem — convenient but a privilege boundary the
        operator's manifest declares.
        """
        if self.kind != "local":
            return "Error: set_root_dir requires local-workspace mode."
        if not path:
            return "Error: set_root_dir requires a `path` argument."

        target = Path(path)
        if not target.is_absolute():
            # Resolve relative paths against the CURRENT active root —
            # matches the pre-0.9.24 path-join semantics so agents can
            # say `set_root_dir({"path": "subdir"})` after activating
            # the parent.
            target = (self.root / target).resolve()
        else:
            target = target.resolve()

        try:
            target.relative_to(self._configured_root)
        except ValueError:
            return f"Error: path {path!r} escapes the workspace root."

        result = self._inner.set_root_dir(str(target))

        # mcp-methods returns "Cloned/Updated/Activated 'name' at PATH."
        # on success. Normalise to the kglite operator-facing wording
        # ("Active root set to PATH.") so server.py's rebind logic and
        # the F2 / F4 regression tests stay shape-stable across the
        # 0.9.24 refactor.
        body_lower = result.lower()
        if "failed" in body_lower or body_lower.startswith("error"):
            return result
        return f"Active root set to {target}."


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
