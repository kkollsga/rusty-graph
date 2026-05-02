"""0.9.0 gate item §2 — NULLS LAST / NULLS FIRST in ORDER BY.

`ORDER BY x DESC NULLS LAST` is rejected at parse with
"Unexpected token at start of clause: Identifier(\"NULLS\")" today.
openCypher / Neo4j semantics: NULL values sort either before or
after non-NULL based on the modifier. Default is NULLS LAST for
ASC, NULLS FIRST for DESC (Neo4j 5+).

These tests are xfail-strict — see test_v0_9_05_integer_division.py
header for the workflow.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

NOT_IMPLEMENTED = "0.9.0 §2 — NULLS LAST/FIRST not parsed; flip when fixed."


@pytest.fixture
def nullable_graph():
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        [
            {"id": 1, "title": "A", "score": 10.0},
            {"id": 2, "title": "B", "score": None},
            {"id": 3, "title": "C", "score": 5.0},
            {"id": 4, "title": "D", "score": None},
        ]
    )
    g.add_nodes(df, "X", "id", "title")
    return g


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_order_by_desc_nulls_last(nullable_graph):
    rows = list(
        nullable_graph.cypher(
            "MATCH (n:X) RETURN n.title AS t ORDER BY n.score DESC NULLS LAST, n.id ASC"
        )
    )
    titles = [r["t"] for r in rows]
    # 10.0 (A), 5.0 (C), then NULLs broken by id: B(2), D(4)
    assert titles == ["A", "C", "B", "D"]


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_order_by_asc_nulls_first(nullable_graph):
    rows = list(
        nullable_graph.cypher(
            "MATCH (n:X) RETURN n.title AS t ORDER BY n.score ASC NULLS FIRST, n.id ASC"
        )
    )
    titles = [r["t"] for r in rows]
    # NULLs first broken by id: B(2), D(4); then 5.0 (C), 10.0 (A)
    assert titles == ["B", "D", "C", "A"]


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_order_by_asc_nulls_last_explicit(nullable_graph):
    """ASC NULLS LAST is the Neo4j-default for ASC, but assert it
    explicitly to lock the parser shape."""
    rows = list(
        nullable_graph.cypher(
            "MATCH (n:X) RETURN n.title AS t ORDER BY n.score ASC NULLS LAST, n.id ASC"
        )
    )
    titles = [r["t"] for r in rows]
    assert titles == ["C", "A", "B", "D"]


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_order_by_desc_nulls_first_explicit(nullable_graph):
    """DESC NULLS FIRST is the Neo4j-default for DESC, but assert
    it explicitly."""
    rows = list(
        nullable_graph.cypher(
            "MATCH (n:X) RETURN n.title AS t ORDER BY n.score DESC NULLS FIRST, n.id ASC"
        )
    )
    titles = [r["t"] for r in rows]
    assert titles == ["B", "D", "A", "C"]
