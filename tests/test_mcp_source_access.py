"""Tests for kglite.mcp_server.source_access — the mcp-methods shim layer."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mcp_methods")

from kglite.mcp_server.source_access import (  # noqa: E402
    _resolve_under_roots,
    register,
    register_dynamic,
)


class _CaptureMcp:
    """Minimal FastMCP stand-in that captures registered tool functions."""

    def __init__(self) -> None:
        self.tools: dict[str, callable] = {}

    def tool(self, *_args, **_kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    """Build a small file tree for read_source/grep/list_source to act on."""
    (tmp_path / "data.json").write_text('{"name": "Alice", "role": "engineer"}\n')
    (tmp_path / "notes.txt").write_text("line one\nline two with marker\nline three\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested file\n")
    return tmp_path


class TestRegisterShape:
    def test_register_creates_three_tools(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)])
        assert set(mcp.tools) == {"read_source", "grep", "list_source"}

    def test_register_requires_at_least_one_root(self) -> None:
        mcp = _CaptureMcp()
        with pytest.raises(ValueError, match="at least one source root"):
            register(mcp, [])


class TestReadSource:
    def test_full_file_read(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)])
        out = mcp.tools["read_source"](file_path="notes.txt")
        assert "line one" in out
        assert "line three" in out

    def test_grep_filter(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)])
        out = mcp.tools["read_source"](file_path="notes.txt", grep="marker")
        assert "marker" in out
        # Lines without "marker" outside the grep_context window should be excluded
        assert "matches" in out  # mcp-methods header includes match count

    def test_path_traversal_blocked(self, source_tree: Path, tmp_path: Path) -> None:
        # Create a sibling file outside the sandbox
        sibling = tmp_path.parent / "outside.txt"
        sibling.write_text("secret\n")
        try:
            mcp = _CaptureMcp()
            register(mcp, [str(source_tree)])
            out = mcp.tools["read_source"](file_path="../outside.txt")
            assert "secret" not in out
            assert "Error" in out or "not found" in out.lower() or "denied" in out.lower()
        finally:
            sibling.unlink(missing_ok=True)


class TestGrep:
    def test_finds_pattern(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)])
        out = mcp.tools["grep"](pattern="Alice")
        assert "data.json" in out

    def test_glob_filter(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)])
        # Pattern that exists in both .json and .txt files; glob restricts to .txt
        (source_tree / "extra.json").write_text('{"key": "marker"}\n')
        out_txt = mcp.tools["grep"](pattern="marker", glob="*.txt")
        assert "notes.txt" in out_txt
        assert "extra.json" not in out_txt


class TestListSource:
    def test_lists_root(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)])
        out = mcp.tools["list_source"]()
        assert "data.json" in out
        assert "notes.txt" in out

    def test_lists_subdirectory(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)])
        out = mcp.tools["list_source"](path="sub")
        assert "nested.txt" in out

    def test_path_traversal_rejected(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)])
        out = mcp.tools["list_source"](path="../..")
        assert "outside the configured source roots" in out


class TestResolveUnderRoots:
    def test_dot_returns_root(self, tmp_path: Path) -> None:
        root = str(tmp_path.resolve())
        assert _resolve_under_roots(".", [root]) == root

    def test_subpath_resolves(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        root = str(tmp_path.resolve())
        assert _resolve_under_roots("sub", [root]) == str(sub.resolve())

    def test_escape_returns_none(self, tmp_path: Path) -> None:
        root = str(tmp_path.resolve())
        assert _resolve_under_roots("../..", [root]) is None

    def test_multi_root_accepts_either(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        # Resolution is against root_a; root_b acceptance only matters if the
        # resolved path happens to fall under it.
        result = _resolve_under_roots(".", [str(root_a.resolve()), str(root_b.resolve())])
        assert result == str(root_a.resolve())


class _StubGraphSource:
    """Stub graph for qualified_name= tests. ``responses`` is a dict from
    name to either the dict source() returns, or an Exception to raise."""

    def __init__(self, responses: dict, cypher_rows: list | None = None) -> None:
        self.responses = responses
        self.cypher_rows = cypher_rows or []
        self.cypher_calls: list[tuple[str, dict]] = []

    def source(self, name: str, node_type=None):
        v = self.responses[name]
        if isinstance(v, Exception):
            raise v
        return v

    def cypher(self, query: str, params=None, **kwargs):
        self.cypher_calls.append((query, params or kwargs))
        return self.cypher_rows


class TestQualifiedName:
    def test_resolves_to_file_slice(self, source_tree: Path) -> None:
        graph = _StubGraphSource(
            responses={
                "MyClass.do_thing": {
                    "file_path": "notes.txt",
                    "line_number": 1,
                    "end_line": 2,
                }
            }
        )
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)], graph_provider=lambda: graph)
        out = mcp.tools["read_source"](qualified_name="MyClass.do_thing")
        assert "line one" in out

    def test_no_graph_provider_returns_error(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)])  # no graph_provider
        out = mcp.tools["read_source"](qualified_name="X")
        assert "not available" in out.lower()

    def test_provider_returns_none(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)], graph_provider=lambda: None)
        out = mcp.tools["read_source"](qualified_name="X")
        assert "active graph" in out.lower()

    def test_ambiguous_match(self, source_tree: Path) -> None:
        graph = _StubGraphSource(
            responses={
                "do_thing": {
                    "ambiguous": True,
                    "matches": [
                        {"qualified_name": "A.do_thing"},
                        {"qualified_name": "B.do_thing"},
                    ],
                }
            }
        )
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)], graph_provider=lambda: graph)
        out = mcp.tools["read_source"](qualified_name="do_thing")
        assert "Ambiguous" in out
        assert "A.do_thing" in out and "B.do_thing" in out

    def test_error_with_suffix_fallback_unique(self, source_tree: Path) -> None:
        graph = _StubGraphSource(
            responses={"helper": {"error": "name 'helper' not found"}},
            cypher_rows=[
                {
                    "qualified_name": "mod.util.helper",
                    "file_path": "notes.txt",
                    "line_number": 2,
                    "end_line": 2,
                }
            ],
        )
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)], graph_provider=lambda: graph)
        out = mcp.tools["read_source"](qualified_name="helper")
        assert "line two" in out

    def test_error_with_suffix_fallback_no_match(self, source_tree: Path) -> None:
        graph = _StubGraphSource(
            responses={"nope": {"error": "name 'nope' not found"}},
            cypher_rows=[],
        )
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)], graph_provider=lambda: graph)
        out = mcp.tools["read_source"](qualified_name="nope")
        assert "not found" in out

    def test_both_args_rejected(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)])
        out = mcp.tools["read_source"](file_path="x.txt", qualified_name="Y")
        assert "only one" in out.lower()

    def test_neither_arg_rejected(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register(mcp, [str(source_tree)])
        out = mcp.tools["read_source"]()
        assert "Provide either" in out


class TestRegisterDynamic:
    def test_provider_returns_active_dirs(self, source_tree: Path) -> None:
        current_dirs = [str(source_tree)]
        mcp = _CaptureMcp()
        register_dynamic(mcp, lambda: current_dirs)
        out = mcp.tools["read_source"](file_path="notes.txt")
        assert "line one" in out

    def test_provider_returns_empty(self, source_tree: Path) -> None:
        mcp = _CaptureMcp()
        register_dynamic(mcp, lambda: [])
        out = mcp.tools["read_source"](file_path="x.txt")
        assert "no active" in out.lower()

    def test_provider_changes_at_runtime(self, source_tree: Path, tmp_path: Path) -> None:
        other_root = tmp_path / "other"
        other_root.mkdir()
        (other_root / "hello.txt").write_text("hello world\n")
        current = [str(source_tree)]
        mcp = _CaptureMcp()
        register_dynamic(mcp, lambda: current)
        out1 = mcp.tools["read_source"](file_path="notes.txt")
        assert "line one" in out1
        # Switch active root
        current[:] = [str(other_root)]
        out2 = mcp.tools["read_source"](file_path="hello.txt")
        assert "hello world" in out2
