"""Multi-graph workspace mode for ``kglite-mcp-server``.

Pass ``--workspace DIR`` to the CLI and the server boots without a
graph; the agent activates one with ``repo_management('org/repo')``,
which clones the repo, builds a code-graph, and pins it as the
active graph for ``cypher_query`` / ``graph_overview`` /
``read_source`` / ``grep`` / ``list_source``.

Layout under the workspace dir::

    workspace/
      repos/<org>/<repo>/         ← cloned source
      graphs/<org>/<repo>.kgl     ← built knowledge graph
      inventory.json              ← per-repo access tracking
      temp/                       ← CSV exports + localhost server root
      workspace_mcp.yaml          ← optional config (read by the CLI)

Inventory tracks ``last_accessed`` / ``access_count`` / ``cloned_at``
so repos idle for more than ``stale_after_days`` (default 7) are
auto-swept on each ``repo_management`` call. The active repo is
always exempt from sweeping.

This is the dynamic-graph counterpart to the static single-graph CLI.
The two modes are mutually exclusive — a workspace replaces ``--graph``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

_REPO_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")


@dataclass
class WorkspaceConfig:
    """Tunable knobs for the workspace. Pulled from CLI flags + manifest."""

    stale_after_days: int = 7
    clone_depth: int = 1
    fetch_timeout_seconds: int = 1800


@dataclass
class Workspace:
    """Multi-graph state, on-disk layout, and the ``repo_management`` tool.

    The CLI constructs one of these per server, registers
    :meth:`repo_management` on the FastMCP instance, and wires
    ``cypher_query`` / ``graph_overview`` / source tools through
    :meth:`current_graph` / :meth:`current_source_dirs`.
    """

    workspace_dir: Path
    config: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    active_graph: Any = None
    active_repo_name: str | None = None
    active_repo_path: Path | None = None
    active_graph_path: Path | None = None

    # ── derived paths ──
    @property
    def repos_dir(self) -> Path:
        return self.workspace_dir / "repos"

    @property
    def graphs_dir(self) -> Path:
        return self.workspace_dir / "graphs"

    @property
    def inventory_path(self) -> Path:
        return self.workspace_dir / "inventory.json"

    @property
    def temp_dir(self) -> Path:
        return self.workspace_dir / "temp"

    # ── providers wired into core_tools / source_access ──
    def current_graph(self):
        return self.active_graph

    def current_source_dirs(self) -> list[str]:
        return [str(self.active_repo_path)] if self.active_repo_path else []

    # ── one-shot setup at server boot ──
    def initialise(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir.mkdir(exist_ok=True)
        self.graphs_dir.mkdir(exist_ok=True)
        self._reconcile_inventory()

    # ── inventory ──
    def _now_iso(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _load_inventory(self) -> dict:
        if not self.inventory_path.exists():
            return {}
        try:
            return json.loads(self.inventory_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_inventory(self, inv: dict) -> None:
        self.inventory_path.write_text(json.dumps(inv, indent=2, sort_keys=True))

    def _reconcile_inventory(self) -> None:
        """Sync inventory with on-disk state.

        Adds entries for repos found on disk but missing from inventory
        (using directory mtime as ``last_accessed`` so old idle repos
        sweep on first call). Marks any non-stale entry whose repo dir
        is gone as stale — preserves access history even when the
        directory was manually deleted.
        """
        inv = self._load_inventory()
        on_disk: set[str] = set()
        if self.repos_dir.exists():
            for org_dir in self.repos_dir.iterdir():
                if not org_dir.is_dir() or org_dir.name.startswith("."):
                    continue
                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir() or repo_dir.name.startswith("."):
                        continue
                    rname = f"{org_dir.name}/{repo_dir.name}"
                    on_disk.add(rname)
                    if rname not in inv:
                        mtime_iso = datetime.fromtimestamp(repo_dir.stat().st_mtime).isoformat(timespec="seconds")
                        inv[rname] = {
                            "cloned_at": mtime_iso,
                            "last_accessed": mtime_iso,
                            "access_count": 0,
                            "stale": False,
                        }
        for rname, entry in inv.items():
            if rname not in on_disk and not entry.get("stale"):
                entry["stale"] = True
        self._save_inventory(inv)

    def _bump_access(self, repo_name: str, *, action: str) -> None:
        inv = self._load_inventory()
        now = self._now_iso()
        entry = inv.get(repo_name, {"access_count": 0})
        entry["last_accessed"] = now
        entry["access_count"] = entry.get("access_count", 0) + 1
        entry["stale"] = False
        if action == "cloned" or "cloned_at" not in entry:
            entry["cloned_at"] = now
        inv[repo_name] = entry
        self._save_inventory(inv)

    def _mark_stale(self, repo_name: str) -> None:
        inv = self._load_inventory()
        if repo_name in inv:
            inv[repo_name]["stale"] = True
            self._save_inventory(inv)

    def _prune_empty_org_dirs(self) -> None:
        for root in (self.repos_dir, self.graphs_dir):
            if not root.exists():
                continue
            for org_dir in root.iterdir():
                if not org_dir.is_dir():
                    continue
                children = list(org_dir.iterdir())
                real = [c for c in children if not c.name.startswith(".")]
                if real:
                    continue
                for c in children:
                    try:
                        c.unlink() if c.is_file() else shutil.rmtree(c, ignore_errors=True)
                    except OSError:
                        pass
                try:
                    org_dir.rmdir()
                except OSError:
                    pass

    def _sweep_stale(self) -> list[str]:
        """Delete repo+graph for non-stale entries idle past the threshold.

        The active repo is exempt. Stale (already-deleted) entries are
        skipped. Returns the list of newly-swept repo names so the caller
        can surface them to the agent.
        """
        inv = self._load_inventory()
        cutoff = datetime.now() - timedelta(days=self.config.stale_after_days)
        swept: list[str] = []
        for rname, entry in inv.items():
            if entry.get("stale"):
                continue
            if rname == self.active_repo_name:
                continue
            try:
                last = datetime.fromisoformat(entry["last_accessed"])
            except (KeyError, ValueError):
                continue
            if last >= cutoff:
                continue
            org, repo = rname.split("/", 1)
            repo_path = self.repos_dir / org / repo
            graph_path = self.graphs_dir / org / f"{repo}.kgl"
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            if graph_path.exists():
                graph_path.unlink()
            entry["stale"] = True
            swept.append(rname)
        if swept:
            self._save_inventory(inv)
            self._prune_empty_org_dirs()
        return swept

    # ── git + graph build ──
    def _git(self, *args, cwd: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=self.config.fetch_timeout_seconds,
        )

    def _clone_or_update(self, repo_name: str) -> tuple[Path, str]:
        """Clone the repo if missing; otherwise fetch and fast-forward.

        Returns ``(repo_path, action)`` where ``action`` is one of
        ``"cloned"``, ``"updated"``, ``"current"``.
        """
        org, repo = repo_name.split("/", 1)
        repo_path = self.repos_dir / org / repo
        if not repo_path.exists():
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://github.com/{repo_name}.git"
            result = self._git("clone", "--depth", str(self.config.clone_depth), url, str(repo_path))
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
            return repo_path, "cloned"

        self._git("fetch", "--depth", str(self.config.clone_depth), "origin", cwd=repo_path)
        local = self._git("rev-parse", "HEAD", cwd=repo_path)
        remote = self._git("rev-parse", "FETCH_HEAD", cwd=repo_path)
        if local.stdout.strip() != remote.stdout.strip():
            self._git("reset", "--hard", "FETCH_HEAD", cwd=repo_path)
            return repo_path, "updated"
        return repo_path, "current"

    def _build_graph(self, repo_name: str, repo_path: Path) -> tuple[Path, Any]:
        """Build a code_tree graph for ``repo_path`` and save it.

        Imported lazily so the workspace module doesn't pull tree-sitter
        in for users who only ever load pre-built ``.kgl`` files.
        """
        from kglite.code_tree import build as build_code_tree

        org, repo = repo_name.split("/", 1)
        graph_dir = self.graphs_dir / org
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graph_dir / f"{repo}.kgl"
        graph = build_code_tree(str(repo_path), save_to=str(graph_path), verbose=False)
        return graph_path, graph

    # ── activation helpers ──
    def _activate(
        self,
        name: str,
        repo_path: Path,
        graph_path: Path | None,
        graph: Any,
    ) -> None:
        self.active_repo_name = name
        self.active_repo_path = repo_path
        self.active_graph_path = graph_path
        self.active_graph = graph

    def _deactivate(self) -> None:
        self.active_repo_name = None
        self.active_repo_path = None
        self.active_graph_path = None
        self.active_graph = None

    # ── formatting ──
    @staticmethod
    def _format_relative(iso_ts: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_ts)
        except (TypeError, ValueError):
            return "unknown"
        delta = datetime.now() - dt
        days = delta.days
        if days <= 0:
            hours = delta.seconds // 3600
            return "just now" if hours < 1 else f"{hours}h ago"
        if days == 1:
            return "1d ago"
        return f"{days}d ago"

    def _list_inventory(self) -> str:
        inv = self._load_inventory()
        if not inv:
            return "No repos cloned yet. Call repo_management('org/repo') to clone one."
        live_lines: list[str] = []
        stale_lines: list[str] = []
        for rname in sorted(inv.keys()):
            entry = inv[rname]
            org, repo = rname.split("/", 1)
            is_stale = entry.get("stale", False)
            is_active = rname == self.active_repo_name
            has_graph = (self.graphs_dir / org / f"{repo}.kgl").exists()
            count = entry.get("access_count", 0)
            ago = self._format_relative(entry.get("last_accessed", ""))
            access_str = f"{count} access{'es' if count != 1 else ''}, last {ago}"
            if is_stale:
                stale_lines.append(f"  {rname}  [STALE — re-fetch with repo_management('{rname}')]  ({access_str})")
            else:
                graph_status = "graph built" if has_graph else "no graph"
                marker = " [active]" if is_active else ""
                live_lines.append(f"  {rname}  ({graph_status}){marker}  ({access_str})")
        sections = []
        if live_lines:
            sections.append(f"{len(live_lines)} live repo(s):\n" + "\n".join(live_lines))
        if stale_lines:
            sections.append(f"{len(stale_lines)} stale repo(s):\n" + "\n".join(stale_lines))
        return "\n\n".join(sections)

    def _sweep_note(self, swept: list[str]) -> str:
        if not swept:
            return ""
        return f"[Swept {len(swept)} idle repo(s) (>{self.config.stale_after_days}d): {', '.join(swept)}]\n\n"

    # ── main entry point: repo_management ──
    def repo_management(
        self,
        name: str | None = None,
        delete: bool = False,
        update: bool = False,
        rebuild_graph: bool = False,
    ) -> str:
        """Manage repositories — clone, set active, update, delete, or list.

        FIRST STEP — call this to activate a repo before using any other tool.

        ``repo_management()`` lists all known repos. ``repo_management(name=...)``
        clones (if needed) and activates. ``update=True`` fetches upstream changes
        and rebuilds if anything changed. ``delete=True`` removes the repo and its
        graph. ``rebuild_graph=True`` rebuilds the graph even when the repo is
        already up to date — useful after upgrading kglite or code_tree.
        """
        swept = self._sweep_stale()
        prefix = self._sweep_note(swept)

        # List mode: no name, no update flag
        if name is None and not update:
            return prefix + self._list_inventory()

        # Update mode: refresh the active repo
        if update:
            if self.active_repo_name is None:
                return prefix + "No active repository. Call repo_management('org/repo') first."
            return prefix + self._do_update()

        # Both delete and set require a valid repo name
        if name is None:
            return prefix + "Provide a repo name (e.g. repo_management('org/repo'))."
        validation_err = _validate_repo_name(name)
        if validation_err:
            return prefix + validation_err

        if delete:
            return prefix + self._do_delete(name)

        return prefix + self._do_set(name, rebuild_graph=rebuild_graph)

    def _do_update(self) -> str:
        active_name = self.active_repo_name
        active_path = self.active_repo_path
        assert active_name is not None and active_path is not None  # narrowed by caller
        try:
            t0 = time.perf_counter()
            _, action = self._clone_or_update(active_name)
            fetch_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            return f"Failed to check for updates: {e}"
        self._bump_access(active_name, action=action)
        if action == "current":
            return f"'{active_name}' is already up to date."
        try:
            t0 = time.perf_counter()
            graph_path, graph = self._build_graph(active_name, active_path)
            build_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            return f"Repo updated but graph rebuild failed: {e}"
        self.active_graph = graph
        self.active_graph_path = graph_path
        schema = graph.schema()
        return (
            f"Updated '{active_name}' to latest.\n"
            f"Fetch: {fetch_ms:.0f}ms | Rebuild: {build_ms:.0f}ms\n"
            f"Graph: {schema['node_count']} nodes, {schema['edge_count']} edges.\n\n" + graph.describe()
        )

    def _do_delete(self, name: str) -> str:
        org, repo = name.split("/", 1)
        repo_path = self.repos_dir / org / repo
        graph_path = self.graphs_dir / org / f"{repo}.kgl"
        deleted: list[str] = []
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)
            deleted.append("repo")
        if graph_path.exists():
            graph_path.unlink()
            deleted.append("graph")
        self._mark_stale(name)
        self._prune_empty_org_dirs()
        if not deleted:
            return f"Nothing to delete — '{name}' not found."
        if self.active_repo_name == name:
            self._deactivate()
            return f"Deleted {', '.join(deleted)}. Active repo cleared."
        return f"Deleted {', '.join(deleted)}."

    def _do_set(self, name: str, *, rebuild_graph: bool) -> str:
        try:
            t0 = time.perf_counter()
            repo_path, action = self._clone_or_update(name)
            clone_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            return f"Failed to clone/update {name}: {e}"
        self._bump_access(name, action=action)

        org, repo = name.split("/", 1)
        graph_path = self.graphs_dir / org / f"{repo}.kgl"

        # Up-to-date repo with an existing graph: just load the .kgl
        if graph_path.exists() and action == "current" and not rebuild_graph:
            try:
                import kglite

                graph = kglite.load(str(graph_path))
            except Exception as e:
                return f"Repo is current but failed to load existing graph: {e}"
            self._activate(name, repo_path, graph_path, graph)
            schema = graph.schema()
            return (
                f"Repo '{name}' is up to date (no changes).\n"
                f"Loaded existing graph: {schema['node_count']} nodes, "
                f"{schema['edge_count']} edges.\n\n" + graph.describe()
            )

        # Pin the repo as active before building the graph so non-graph
        # tools (read_source / grep / list_source) work even if the
        # build step explodes.
        self._activate(name, repo_path, None, None)
        try:
            t0 = time.perf_counter()
            graph_path, graph = self._build_graph(name, repo_path)
            build_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            return (
                f"Failed to build knowledge graph for {name}: {e}\n\n"
                f"The repo is cloned at {repo_path}. Graph-based tools "
                f"(cypher_query, graph_overview) are unavailable, but these "
                f"still work:\n"
                f"  - grep(pattern) — regex search across source files\n"
                f"  - read_source(file_path=...) — read source code by path\n"
                f"  - list_source() — browse directory structure"
            )
        self.active_graph = graph
        self.active_graph_path = graph_path

        schema = graph.schema()
        verb = {"cloned": "Cloned and built", "updated": "Updated and rebuilt"}.get(action, "Built")
        return (
            f"{verb} graph for '{name}'.\n"
            f"Clone/update: {clone_ms:.0f}ms | Graph build: {build_ms:.0f}ms\n"
            f"Graph: {schema['node_count']} nodes, {schema['edge_count']} edges.\n\n" + graph.describe()
        )


def register_repo_management(mcp, workspace: Workspace) -> None:
    """Register the ``repo_management`` MCP tool on the server."""

    @mcp.tool()
    def repo_management(
        name: str | None = None,
        delete: bool = False,
        update: bool = False,
        rebuild_graph: bool = False,
    ) -> str:
        """Manage repositories — clone, set active, update, delete, or list.

        FIRST STEP — call this to activate a repo before using any other tool.

        repo_management()                              — list all known repos
        repo_management("pydata/xarray")              — clone (if needed) and activate
        repo_management(update=True)                   — fetch upstream + rebuild if changed
        repo_management("pydata/xarray", delete=True) — delete repo + its graph
        repo_management("pydata/xarray", rebuild_graph=True) — force graph rebuild

        Idle repos (last access > stale_after_days) are auto-swept on each
        call. The active repo is exempt. Stale entries are preserved in
        the inventory so historical access counts aren't lost.
        """
        return workspace.repo_management(name=name, delete=delete, update=update, rebuild_graph=rebuild_graph)


def _validate_repo_name(name: str) -> str | None:
    if not _REPO_RE.match(name):
        return (
            f"Invalid repo name {name!r}. Expected 'org/repo' format with only "
            "letters, digits, dots, hyphens, or underscores."
        )
    return None
