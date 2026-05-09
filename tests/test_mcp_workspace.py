"""Tests for kglite.mcp_server.workspace — multi-graph mode state, inventory,
and the ``repo_management`` tool. Network-touching paths (clone, build) are
patched out so tests are hermetic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import subprocess
from typing import Any

import pytest

from kglite.mcp_server.workspace import (
    Workspace,
    WorkspaceConfig,
    _validate_repo_name,
    register_repo_management,
)


class _CaptureMcp:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *_args, **_kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


class _StubGraph:
    def __init__(self, nodes: int = 100, edges: int = 200) -> None:
        self._nodes = nodes
        self._edges = edges

    def schema(self) -> dict:
        return {"node_count": self._nodes, "edge_count": self._edges}

    def describe(self, **_kw) -> str:
        return "STUB_DESCRIBE"


def _make_workspace(tmp_path: Path, **cfg) -> Workspace:
    ws = Workspace(workspace_dir=tmp_path / "ws", config=WorkspaceConfig(**cfg))
    ws.initialise()
    return ws


def _seed_inventory_entry(ws: Workspace, name: str, *, days_ago: int = 0, stale: bool = False, count: int = 1) -> None:
    """Create a fake repo + inventory entry, dated ``days_ago`` ago."""
    org, repo = name.split("/", 1)
    repo_path = ws.repos_dir / org / repo
    repo_path.mkdir(parents=True, exist_ok=True)
    if not stale:
        # Marker file so the dir is non-empty
        (repo_path / "README.md").write_text("hi\n")
    inv = ws._load_inventory()
    last = (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")
    inv[name] = {
        "cloned_at": last,
        "last_accessed": last,
        "access_count": count,
        "stale": stale,
    }
    if stale:
        # Also remove the on-disk dir to match "stale" semantics
        import shutil

        shutil.rmtree(repo_path, ignore_errors=True)
    ws._save_inventory(inv)


class TestValidation:
    def test_valid_name(self) -> None:
        assert _validate_repo_name("pydata/xarray") is None

    def test_dots_and_dashes_ok(self) -> None:
        assert _validate_repo_name("my-org.x/repo_v2") is None

    def test_no_slash_rejected(self) -> None:
        assert _validate_repo_name("xarray") is not None

    def test_two_slashes_rejected(self) -> None:
        assert _validate_repo_name("a/b/c") is not None

    def test_special_chars_rejected(self) -> None:
        assert _validate_repo_name("foo/bar; rm -rf") is not None


class TestPaths:
    def test_initialise_creates_layout(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        assert ws.workspace_dir.is_dir()
        assert ws.repos_dir.is_dir()
        assert ws.graphs_dir.is_dir()

    def test_temp_dir_path(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        assert ws.temp_dir == ws.workspace_dir / "temp"

    def test_inventory_path(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        assert ws.inventory_path == ws.workspace_dir / "inventory.json"


class TestInventoryReconcile:
    def test_picks_up_unrecorded_repo(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        # Pretend a repo was manually copied in
        repo_dir = ws.repos_dir / "ghost" / "repo"
        repo_dir.mkdir(parents=True)
        (repo_dir / "README").write_text("hi\n")
        ws._reconcile_inventory()
        inv = ws._load_inventory()
        assert "ghost/repo" in inv
        assert inv["ghost/repo"]["stale"] is False

    def test_marks_missing_repo_stale(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        _seed_inventory_entry(ws, "vanished/repo", days_ago=1)
        # Manually delete the dir
        import shutil

        shutil.rmtree(ws.repos_dir / "vanished" / "repo")
        ws._reconcile_inventory()
        inv = ws._load_inventory()
        assert inv["vanished/repo"]["stale"] is True


class TestSweep:
    def test_sweeps_idle_repo(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path, stale_after_days=7)
        _seed_inventory_entry(ws, "old/repo", days_ago=30)
        swept = ws._sweep_stale()
        assert swept == ["old/repo"]
        inv = ws._load_inventory()
        assert inv["old/repo"]["stale"] is True
        assert not (ws.repos_dir / "old" / "repo").exists()

    def test_active_repo_exempt(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path, stale_after_days=7)
        _seed_inventory_entry(ws, "active/repo", days_ago=30)
        ws.active_repo_name = "active/repo"
        ws.active_repo_path = ws.repos_dir / "active" / "repo"
        swept = ws._sweep_stale()
        assert swept == []
        assert (ws.repos_dir / "active" / "repo").exists()

    def test_recent_repo_kept(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path, stale_after_days=7)
        _seed_inventory_entry(ws, "fresh/repo", days_ago=2)
        swept = ws._sweep_stale()
        assert swept == []

    def test_already_stale_skipped(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path, stale_after_days=7)
        _seed_inventory_entry(ws, "ghost/repo", days_ago=30, stale=True)
        swept = ws._sweep_stale()
        assert swept == []


class TestListMode:
    def test_empty_workspace(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        out = ws.repo_management()
        assert "No repos cloned yet" in out

    def test_lists_repos_with_active_marker(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        _seed_inventory_entry(ws, "a/b", days_ago=1, count=3)
        ws.active_repo_name = "a/b"
        ws.active_repo_path = ws.repos_dir / "a" / "b"
        out = ws.repo_management()
        assert "a/b" in out and "[active]" in out
        assert "3 accesses" in out

    def test_stale_section(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        _seed_inventory_entry(ws, "live/one", days_ago=1)
        _seed_inventory_entry(ws, "ghost/two", days_ago=30, stale=True)
        out = ws.repo_management()
        assert "live/one" in out
        assert "ghost/two" in out
        assert "STALE" in out


class TestDelete:
    def test_delete_existing(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        _seed_inventory_entry(ws, "doomed/repo", days_ago=1)
        # Also create a fake graph file
        graph_dir = ws.graphs_dir / "doomed"
        graph_dir.mkdir(parents=True, exist_ok=True)
        (graph_dir / "repo.kgl").write_bytes(b"\x00")
        out = ws.repo_management(name="doomed/repo", delete=True)
        assert "Deleted" in out
        assert not (ws.repos_dir / "doomed" / "repo").exists()
        assert not (ws.graphs_dir / "doomed" / "repo.kgl").exists()
        inv = ws._load_inventory()
        assert inv["doomed/repo"]["stale"] is True

    def test_delete_active_clears_active(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        _seed_inventory_entry(ws, "active/repo", days_ago=1)
        ws.active_repo_name = "active/repo"
        ws.active_repo_path = ws.repos_dir / "active" / "repo"
        out = ws.repo_management(name="active/repo", delete=True)
        assert "Active repo cleared" in out
        assert ws.active_repo_name is None

    def test_delete_unknown(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        out = ws.repo_management(name="never/here", delete=True)
        assert "Nothing to delete" in out

    def test_delete_invalid_name(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        out = ws.repo_management(name="not-an-org-repo", delete=True)
        assert "Invalid repo name" in out


class TestSetMode:
    def test_clone_path_with_mocked_subprocess(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _make_workspace(tmp_path)

        def fake_run(cmd, **kw) -> subprocess.CompletedProcess:
            # Simulate `git clone` by creating the target directory
            if cmd[1] == "clone":
                target = Path(cmd[-1])
                target.mkdir(parents=True, exist_ok=True)
                (target / "README.md").write_text("hi\n")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        # Patch the graph build to avoid tree-sitter dep
        def fake_build(repo_path: str, save_to: str = "", verbose: bool = False) -> Any:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            Path(save_to).write_bytes(b"\x00")
            return _StubGraph(nodes=42, edges=84)

        import kglite.code_tree

        monkeypatch.setattr(kglite.code_tree, "build", fake_build, raising=False)
        # workspace.py imports lazily, so patch where the lookup happens
        import kglite.mcp_server.workspace as ws_mod

        def patched_build_graph(self, repo_name: str, repo_path: Path):
            org, repo = repo_name.split("/", 1)
            graph_dir = self.graphs_dir / org
            graph_dir.mkdir(parents=True, exist_ok=True)
            graph_path = graph_dir / f"{repo}.kgl"
            graph_path.write_bytes(b"\x00")
            return graph_path, _StubGraph(nodes=42, edges=84)

        monkeypatch.setattr(ws_mod.Workspace, "_build_graph", patched_build_graph)

        out = ws.repo_management(name="my/repo")
        assert "Cloned and built" in out or "Built" in out
        assert ws.active_repo_name == "my/repo"
        assert ws.active_graph is not None
        assert "42 nodes" in out and "84 edges" in out
        assert "STUB_DESCRIBE" in out

    def test_set_invalid_name(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        out = ws.repo_management(name="bad name with spaces")
        assert "Invalid repo name" in out

    def test_clone_failure_returns_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _make_workspace(tmp_path)

        def fake_run(cmd, **kw) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="not found")

        monkeypatch.setattr(subprocess, "run", fake_run)
        out = ws.repo_management(name="bogus/repo")
        assert "Failed to clone" in out


class TestUpdateMode:
    def test_no_active_repo(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        out = ws.repo_management(update=True)
        assert "No active repository" in out


class TestRegisterRepoManagement:
    def test_registers_tool_with_docstring(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        mcp = _CaptureMcp()
        register_repo_management(mcp, ws)
        assert "repo_management" in mcp.tools
        fn = mcp.tools["repo_management"]
        assert fn.__doc__ and "FIRST STEP" in fn.__doc__

    def test_tool_calls_workspace_method(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        mcp = _CaptureMcp()
        register_repo_management(mcp, ws)
        out = mcp.tools["repo_management"]()
        assert "No repos cloned" in out


class TestProviderHooks:
    def test_current_graph_returns_active(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        assert ws.current_graph() is None
        ws.active_graph = _StubGraph()
        assert ws.current_graph() is not None

    def test_current_source_dirs_empty_when_inactive(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        assert ws.current_source_dirs() == []

    def test_current_source_dirs_active(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        ws.active_repo_path = tmp_path / "ws" / "repos" / "x" / "y"
        assert ws.current_source_dirs() == [str(tmp_path / "ws" / "repos" / "x" / "y")]
