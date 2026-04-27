"""Procedure nodes synthesized from `@procedure: NAME` doc-comment annotations.

Phase 4.5 added a generic, language-agnostic mechanism: any function with
`@procedure: NAME` (or `@cypher_procedure: NAME`) in its docstring/leading
comment gets a Procedure node + IMPLEMENTED_BY edge. Useful for surfacing
RPC method registries, Cypher CALL procedures, command-bus dispatchers,
and similar string-keyed registries as first-class graph entities.
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


def test_procedure_annotation_python(tmp_path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/m.py": """
            def regular():
                '''A normal function.'''
                return 1

            def special():
                '''Compute X.

                @procedure: do_x
                '''
                return 2
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        """
        MATCH (p:Procedure)-[:IMPLEMENTED_BY]->(f:Function)
        RETURN p.name AS proc, f.qualified_name AS q
        """
    ).to_list()
    # qualified_name has a tmp_path-derived prefix; match by suffix.
    matched = [(r["proc"], r["q"]) for r in rows if r["proc"] == "do_x" and r["q"].endswith(".pkg.m.special")]
    assert matched, rows


def test_procedure_annotation_rust(tmp_path):
    _write(
        tmp_path,
        {
            "Cargo.toml": """
            [package]
            name = "demo"
            version = "0.1.0"
            """,
            "src/lib.rs": """
            /// Regular function.
            pub fn ordinary() {}

            /// Compute pagerank centrality scores.
            ///
            /// @procedure: pagerank
            pub fn pagerank_impl() {}
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        """
        MATCH (p:Procedure {name: "pagerank"})-[:IMPLEMENTED_BY]->(f:Function)
        RETURN f.qualified_name AS q
        """
    ).to_list()
    assert rows, "expected pagerank Procedure with IMPLEMENTED_BY edge"
    assert rows[0]["q"].endswith("::pagerank_impl")


def test_no_annotation_no_procedure_node(tmp_path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/m.py": """
            def f():
                '''Plain docstring without procedure marker.'''
                return 1
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher("MATCH (p:Procedure) RETURN count(*) AS n").to_list()
    assert rows[0]["n"] == 0


def test_alternate_annotation_form(tmp_path):
    """`@cypher_procedure: NAME` form should also work."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/m.py": """
            def f():
                '''@cypher_procedure: louvain'''
                return 1
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher("MATCH (p:Procedure {name: 'louvain'})-[:IMPLEMENTED_BY]->(f) RETURN f.name AS n").to_list()
    assert rows and rows[0]["n"] == "f"


def test_multiple_aliases_per_function(tmp_path):
    """A function with multiple @procedure: NAME annotations registers under each alias."""
    _write(
        tmp_path,
        {
            "Cargo.toml": """
            [package]
            name = "demo"
            version = "0.1.0"
            """,
            "src/lib.rs": """
            /// Compute betweenness centrality.
            ///
            /// @procedure: betweenness
            /// @procedure: betweenness_centrality
            pub fn betweenness_impl() {}
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        "MATCH (p:Procedure)-[:IMPLEMENTED_BY]->(f:Function) "
        "WHERE f.qualified_name ENDS WITH '::betweenness_impl' "
        "RETURN p.name AS proc ORDER BY p.name"
    ).to_list()
    names = [r["proc"] for r in rows]
    assert names == ["betweenness", "betweenness_centrality"], names
