"""Tests for kglite.mcp_server.core_tools — graph_overview / cypher_query
registration with provider callable + overview_prefix injection.
"""

from __future__ import annotations

from pathlib import Path

from kglite.mcp_server.core_tools import register_core_tools


class _CaptureMcp:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *_args, **_kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


class _Result(list):
    """Simple list-like cypher result that matches the protocol the tool uses."""


class _StubGraph:
    def __init__(self) -> None:
        self.describe_calls: list[dict] = []
        self.cypher_calls: list[tuple[str, dict]] = []
        self.next_describe = "DESCRIBE_OUTPUT"
        self.next_cypher: object = _Result()

    def describe(self, **kwargs) -> str:
        self.describe_calls.append(kwargs)
        return self.next_describe

    def cypher(self, query: str, **kwargs) -> object:
        self.cypher_calls.append((query, kwargs))
        return self.next_cypher

    def schema(self) -> dict:
        return {"node_count": 0, "edge_count": 0}


class TestGraphOverviewBare:
    def test_bare_call_includes_prefix(self, tmp_path: Path) -> None:
        graph = _StubGraph()
        mcp = _CaptureMcp()
        register_core_tools(
            mcp,
            graph_provider=lambda: graph,
            temp_dir=tmp_path / "temp",
            overview_prefix="STICKY CONTEXT",
        )
        out = mcp.tools["graph_overview"]()
        assert out.startswith("STICKY CONTEXT")
        assert "DESCRIBE_OUTPUT" in out

    def test_bare_call_runs_pre_overview_hook(self, tmp_path: Path) -> None:
        graph = _StubGraph()
        mcp = _CaptureMcp()
        calls: list[int] = []
        register_core_tools(
            mcp,
            graph_provider=lambda: graph,
            temp_dir=tmp_path / "temp",
            pre_overview_hook=lambda: calls.append(1),
        )
        mcp.tools["graph_overview"]()
        assert calls == [1]

    def test_drilldown_skips_prefix_and_hook(self, tmp_path: Path) -> None:
        graph = _StubGraph()
        mcp = _CaptureMcp()
        calls: list[int] = []
        register_core_tools(
            mcp,
            graph_provider=lambda: graph,
            temp_dir=tmp_path / "temp",
            overview_prefix="SHOULD_NOT_APPEAR",
            pre_overview_hook=lambda: calls.append(1),
        )
        out = mcp.tools["graph_overview"](types=["Foo"])
        assert "SHOULD_NOT_APPEAR" not in out
        assert calls == []

    def test_provider_returning_none(self, tmp_path: Path) -> None:
        mcp = _CaptureMcp()
        register_core_tools(
            mcp,
            graph_provider=lambda: None,
            temp_dir=tmp_path / "temp",
            no_graph_message="custom no-graph message",
        )
        out = mcp.tools["graph_overview"]()
        assert "custom no-graph message" in out


class TestCypherQuery:
    def test_basic_query(self, tmp_path: Path) -> None:
        graph = _StubGraph()
        graph.next_cypher = _Result([{"a": 1}])
        mcp = _CaptureMcp()
        register_core_tools(mcp, graph_provider=lambda: graph, temp_dir=tmp_path / "temp")
        out = mcp.tools["cypher_query"]("MATCH (n) RETURN n")
        assert "1 row" in out

    def test_no_active_graph(self, tmp_path: Path) -> None:
        mcp = _CaptureMcp()
        register_core_tools(mcp, graph_provider=lambda: None, temp_dir=tmp_path / "temp")
        out = mcp.tools["cypher_query"]("MATCH (n) RETURN n")
        assert "No graph" in out or "no graph" in out

    def test_empty_results(self, tmp_path: Path) -> None:
        graph = _StubGraph()
        graph.next_cypher = _Result()
        mcp = _CaptureMcp()
        register_core_tools(mcp, graph_provider=lambda: graph, temp_dir=tmp_path / "temp")
        assert "No results." in mcp.tools["cypher_query"]("MATCH (n) RETURN n")

    def test_csv_export_writes_file(self, tmp_path: Path) -> None:
        graph = _StubGraph()
        graph.next_cypher = "a,b\n1,2\n"  # str signals FORMAT CSV path
        mcp = _CaptureMcp()
        register_core_tools(mcp, graph_provider=lambda: graph, temp_dir=tmp_path / "temp")
        out = mcp.tools["cypher_query"]("MATCH (n) RETURN n FORMAT CSV")
        assert "CSV exported" in out
        # First call shows the read_source warning
        assert "DO NOT" in out
        files = list((tmp_path / "temp").glob("data-*.csv"))
        assert len(files) == 1
        # Second call doesn't repeat the warning
        graph.next_cypher = "a,b\n3,4\n"
        out2 = mcp.tools["cypher_query"]("MATCH (n) RETURN n FORMAT CSV")
        assert "DO NOT" not in out2
