"""Tests for kglite.mcp_server.builtins — save_graph and temp_cleanup."""

from __future__ import annotations

from pathlib import Path

from kglite.mcp_server.builtins import make_temp_cleanup_hook, register_save_graph


class _CaptureMcp:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *_args, **_kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


class _StubGraph:
    def __init__(self, *, raise_on_save: bool = False) -> None:
        self.saved_to: str | None = None
        self.raise_on_save = raise_on_save

    def save(self, path: str) -> None:
        if self.raise_on_save:
            raise RuntimeError("disk full")
        self.saved_to = path

    def schema(self) -> dict:
        return {"node_count": 7, "edge_count": 11}


class TestSaveGraph:
    def test_registers_tool(self, tmp_path: Path) -> None:
        graph = _StubGraph()
        graph_path = tmp_path / "demo.kgl"
        graph_path.write_bytes(b"\x00")
        mcp = _CaptureMcp()
        register_save_graph(mcp, graph, graph_path)
        assert "save_graph" in mcp.tools

    def test_save_writes_path(self, tmp_path: Path) -> None:
        graph = _StubGraph()
        graph_path = tmp_path / "demo.kgl"
        graph_path.write_bytes(b"\x00")
        mcp = _CaptureMcp()
        register_save_graph(mcp, graph, graph_path)
        result = mcp.tools["save_graph"]()
        assert graph.saved_to == str(graph_path)
        assert "Saved" in result and "7 nodes" in result and "11 edges" in result

    def test_save_failure_returns_error_string(self, tmp_path: Path) -> None:
        graph = _StubGraph(raise_on_save=True)
        mcp = _CaptureMcp()
        register_save_graph(mcp, graph, tmp_path / "demo.kgl")
        result = mcp.tools["save_graph"]()
        assert "save_graph error" in result and "disk full" in result


class TestTempCleanupHook:
    def test_on_overview_clears_temp_dir(self, tmp_path: Path) -> None:
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        (temp_dir / "old.csv").write_text("a,b\n1,2\n")
        hook = make_temp_cleanup_hook(temp_dir, "on_overview")
        hook()
        assert not temp_dir.exists()

    def test_never_is_no_op(self, tmp_path: Path) -> None:
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        (temp_dir / "x.csv").write_text("x\n")
        hook = make_temp_cleanup_hook(temp_dir, "never")
        hook()
        assert (temp_dir / "x.csv").exists()

    def test_missing_temp_dir_is_safe(self, tmp_path: Path) -> None:
        hook = make_temp_cleanup_hook(tmp_path / "never_existed", "on_overview")
        hook()  # no error

    def test_unknown_mode_returns_noop(self, tmp_path: Path) -> None:
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        (temp_dir / "x.csv").write_text("x\n")
        hook = make_temp_cleanup_hook(temp_dir, "weekly_at_3am")
        hook()
        assert (temp_dir / "x.csv").exists()
