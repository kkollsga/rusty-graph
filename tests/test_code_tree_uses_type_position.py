"""USES_TYPE edges carry a `position` property — parameter | return | both | signature.

Phase 4.3 refactored the resolver so a type appearing in a parameter and the
return value collapses to position='both'; pure-parameter or pure-return uses
get the simple label. Enables cleanly distinguishing consumers from producers.
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _write(tmp_path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))


def _positions(graph, fn_qsuffix: str, type_short: str) -> str | None:
    rows = graph.cypher(
        """
        MATCH (f:Function)-[r:USES_TYPE]->(t)
        WHERE f.qualified_name ENDS WITH $fs AND t.name = $tn
        RETURN r.position AS p LIMIT 1
        """,
        params={"fs": fn_qsuffix, "tn": type_short},
    ).to_list()
    return rows[0]["p"] if rows else None


def test_position_parameter_only(tmp_path):
    _write(
        tmp_path,
        {
            "Cargo.toml": """
            [package]
            name = "demo"
            version = "0.1.0"
            """,
            "src/lib.rs": """
            pub struct Widget;
            pub fn consumes(w: Widget) -> bool { true }
            """,
        },
    )
    g = build(str(tmp_path))
    assert _positions(g, "::consumes", "Widget") == "parameter"


def test_position_return_only(tmp_path):
    _write(
        tmp_path,
        {
            "Cargo.toml": """
            [package]
            name = "demo"
            version = "0.1.0"
            """,
            "src/lib.rs": """
            pub struct Widget;
            pub fn produces() -> Widget { Widget }
            """,
        },
    )
    g = build(str(tmp_path))
    assert _positions(g, "::produces", "Widget") == "return"


def test_position_both(tmp_path):
    _write(
        tmp_path,
        {
            "Cargo.toml": """
            [package]
            name = "demo"
            version = "0.1.0"
            """,
            "src/lib.rs": """
            pub struct Widget;
            pub fn transform(w: Widget) -> Widget { w }
            """,
        },
    )
    g = build(str(tmp_path))
    assert _positions(g, "::transform", "Widget") == "both"


def test_go_method_return_type_resolved(tmp_path):
    """Regression: Go methods previously had their NAME mistaken for the return type
    because get_return_type naively saw the receiver `parameter_list` first and
    returned the next named child. Fix uses tree-sitter-go's `result` named field.
    """
    _write(
        tmp_path,
        {
            "go.mod": "module demo\n\ngo 1.21\n",
            "main.go": """
            package demo

            type Call struct{}

            // Method on a struct — receiver is *Call, return is *Call. Before the
            // fix, get_return_type returned "Once" (the method name) because the
            // receiver parameter_list tripped the scan.
            func (c *Call) Once() *Call { return c }

            // Same shape with a real parameter.
            func (c *Call) Run(n int) *Call { return c }
            """,
        },
    )
    g = build(str(tmp_path))
    # Once should produce *Call → return position USES_TYPE edge.
    pos = _positions(g, ".Once", "Call")
    assert pos in ("return", "both"), f"expected return/both, got {pos!r}"
    # Run takes int param, returns *Call → both positions seen.
    pos = _positions(g, ".Run", "Call")
    assert pos in ("return", "both"), f"expected return/both, got {pos!r}"


def test_go_method_receiver_position(tmp_path):
    """Go method receiver — captured via ParameterKind::Receiver, USES_TYPE
    edge labeled position='receiver' (not 'signature' fallback)."""
    _write(
        tmp_path,
        {
            "go.mod": "module demo\n\ngo 1.21\n",
            "main.go": """
            package demo

            type Call struct{}

            func (c *Call) lock() {}
            """,
        },
    )
    g = build(str(tmp_path))
    pos = _positions(g, ".lock", "Call")
    assert pos == "receiver", f"expected 'receiver', got {pos!r}"


def test_rust_method_receiver_position(tmp_path):
    """Rust &self method — receiver type injected from owning impl block."""
    _write(
        tmp_path,
        {
            "Cargo.toml": """
            [package]
            name = "demo"
            version = "0.1.0"
            """,
            "src/lib.rs": """
            pub struct Foo;
            impl Foo {
                pub fn bar(&self) {}
            }
            """,
        },
    )
    g = build(str(tmp_path))
    pos = _positions(g, "::bar", "Foo")
    assert pos == "receiver", f"expected 'receiver', got {pos!r}"


def test_method_receiver_plus_return_collapses_to_both(tmp_path):
    """func (c *Call) Once() *Call — receiver + return → 'both'."""
    _write(
        tmp_path,
        {
            "go.mod": "module demo\n\ngo 1.21\n",
            "main.go": """
            package demo

            type Call struct{}

            func (c *Call) Once() *Call { return c }
            """,
        },
    )
    g = build(str(tmp_path))
    pos = _positions(g, ".Once", "Call")
    assert pos == "both", f"expected 'both', got {pos!r}"


def test_no_duplicate_edges_for_both_positions(tmp_path):
    """Single edge per (function, type) — position aggregated, not duplicated."""
    _write(
        tmp_path,
        {
            "Cargo.toml": """
            [package]
            name = "demo"
            version = "0.1.0"
            """,
            "src/lib.rs": """
            pub struct Widget;
            pub fn transform(w: Widget) -> Widget { w }
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        """
        MATCH (f:Function)-[r:USES_TYPE]->(t)
        WHERE f.qualified_name ENDS WITH "::transform" AND t.name = "Widget"
        RETURN count(*) AS n
        """
    ).to_list()
    assert rows[0]["n"] == 1
