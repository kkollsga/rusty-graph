"""0.9.0 gate item §6 — size() over pattern expressions.

Today `RETURN size((:A)-[:R]->(:B))` errors at parse with
"Unexpected token in expression: Colon". `size()` only accepts
lists / strings / collections.

Target: `size((pattern))` returns the count of matches of the inner
pattern, scoped to the outer row's bindings. Same shape as the
existing `count { ... }` subquery expression that landed in 0.8.16
— largely reuses that code path.

These tests are xfail-strict — see test_v0_9_05_integer_division.py
header for the workflow.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

NOT_IMPLEMENTED = "0.9.0 §6 — size() over pattern expressions not parsed; flip when fixed."


@pytest.fixture
def linked_graph():
    """3 :A nodes, 2 :B nodes; A0→B0, A0→B1, A1→B0; A2 has no out edges."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame([{"id": i, "title": f"A{i}"} for i in range(3)]),
        "A",
        "id",
        "title",
    )
    g.add_nodes(
        pd.DataFrame([{"id": i, "title": f"B{i}"} for i in range(2)]),
        "B",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame(
            [{"src": 0, "tgt": 0}, {"src": 0, "tgt": 1}, {"src": 1, "tgt": 0}]
        ),
        "R",
        "A",
        "src",
        "B",
        "tgt",
    )
    return g


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_size_of_unbound_pattern(linked_graph):
    """Total count of matches of the pattern across the whole graph."""
    rows = list(linked_graph.cypher("RETURN size((:A)-[:R]->(:B)) AS n"))
    assert rows[0]["n"] == 3


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_size_with_per_row_binding(linked_graph):
    """Per-row out-degree via pattern expression — the canonical
    use case for size((node)-[]->(...))."""
    rows = list(
        linked_graph.cypher(
            "MATCH (a:A) RETURN a.id AS id, size((a)-[:R]->(:B)) AS deg ORDER BY a.id"
        )
    )
    assert [(r["id"], r["deg"]) for r in rows] == [(0, 2), (1, 1), (2, 0)]


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_size_in_where_clause(linked_graph):
    """size((pattern)) in WHERE filter — common 'has at least N
    neighbors' shape."""
    rows = list(
        linked_graph.cypher(
            "MATCH (a:A) WHERE size((a)-[:R]->(:B)) >= 2 RETURN a.id AS id"
        )
    )
    ids = [r["id"] for r in rows]
    assert ids == [0]


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_size_of_pattern_with_undirected(linked_graph):
    """Undirected variant — counts edges in either direction."""
    rows = list(linked_graph.cypher("RETURN size((:A)--(:B)) AS n"))
    assert rows[0]["n"] == 3
