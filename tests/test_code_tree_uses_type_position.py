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
