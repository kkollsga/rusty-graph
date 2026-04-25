"""Parity oracle for Cypher `count { <pattern> }` subquery expressions.

The `count { ... }` shape used to fail on every storage backend with a
parser error (the identifier-followed-by-brace dispatch routed to map
projection). 0.8.16 adds a parser-level special case so `count { ... }`
becomes a `CountSubquery` expression, evaluated per outer row against
the pattern's match count. This test pins the behaviour across the
three storage modes.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


def _seed(graph):
    people = pd.DataFrame({"nid": [1, 2, 3, 4], "name": ["A", "B", "C", "D"], "age": [20, 30, 40, 50]})
    graph.add_nodes(people, "P", "nid", "name")
    graph.add_connections(
        pd.DataFrame({"s": [1, 2, 3, 1], "t": [2, 3, 4, 4]}),
        "KNOWS",
        "P",
        "s",
        "P",
        "t",
    )
    graph.add_connections(
        pd.DataFrame({"s": [1, 1, 2, 3], "t": [2, 3, 4, 1]}),
        "WORKS_AT",
        "P",
        "s",
        "P",
        "t",
    )


def _make_graph(mode, tmp_path):
    if mode == "default":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    if mode == "disk":
        return KnowledgeGraph(storage="disk", path=str(tmp_path))
    raise ValueError(mode)


STORAGE_MODES = ["default", "mapped", "disk"]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_typed_outdegree_subquery(mode, tmp_path):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    q = "MATCH (a)-[:KNOWS]->(b) WITH a, count{(a)-[:WORKS_AT]->()} AS n RETURN a.name AS name, n ORDER BY name, n"
    df = g.cypher(q).to_df()
    rows = list(zip(df["name"], df["n"]))
    # A: KNOWS→B, KNOWS→D (fires twice), WORKS_AT out = 2
    # B: KNOWS→C, WORKS_AT out = 1
    # C: KNOWS→D, WORKS_AT out = 1
    assert sorted(rows) == [("A", 2), ("A", 2), ("B", 1), ("C", 1)]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_typed_indegree_subquery(mode, tmp_path):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    q = "MATCH (a)-[:KNOWS]->(b) WITH a, count{(a)<-[:WORKS_AT]-()} AS n RETURN a.name AS name, n ORDER BY name, n"
    df = g.cypher(q).to_df()
    rows = list(zip(df["name"], df["n"]))
    # WORKS_AT edges s→t: (1,2), (1,3), (2,4), (3,1).
    # Incoming WORKS_AT per a: A(1)←C(3), B(2)←A(1), C(3)←A(1). All =1.
    # (A appears twice because it's in two KNOWS rows.)
    assert sorted(rows) == [("A", 1), ("A", 1), ("B", 1), ("C", 1)]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_subquery_with_where(mode, tmp_path):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    # Count only WORKS_AT edges whose target is older than 30.
    # WORKS_AT targets: A→B(30), A→C(40), B→D(50), C→A(20). >30 → C, D.
    # By source: A has {B,C} → one target (C) is >30.  B has {D} → 1.  C has {A} → 0.
    q = (
        "MATCH (a)-[:KNOWS]->(b) "
        "WITH a, count{(a)-[:WORKS_AT]->(x) WHERE x.age > 30} AS n "
        "RETURN a.name AS name, n ORDER BY name, n"
    )
    df = g.cypher(q).to_df()
    rows = list(zip(df["name"], df["n"]))
    assert sorted(rows) == [("A", 1), ("A", 1), ("B", 1), ("C", 0)]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_untyped_outdegree_subquery(mode, tmp_path):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    q = "MATCH (a)-[:KNOWS]->(b) WITH a, count{(a)-[]->()} AS n RETURN a.name AS name, n ORDER BY name, n"
    df = g.cypher(q).to_df()
    # All outgoing: A has 2 KNOWS + 2 WORKS_AT = 4
    #               B has 1 KNOWS + 1 WORKS_AT = 2
    #               C has 1 KNOWS + 1 WORKS_AT = 2
    rows = list(zip(df["name"], df["n"]))
    assert sorted(rows) == [("A", 4), ("A", 4), ("B", 2), ("C", 2)]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_zero_matches(mode, tmp_path):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    # FOLLOWS doesn't exist; count should be 0, not an error.
    q = "MATCH (a)-[:KNOWS]->(b) WITH a, count{(a)-[:FOLLOWS]->()} AS n RETURN a.name AS name, n ORDER BY name, n"
    df = g.cypher(q).to_df()
    rows = list(zip(df["name"], df["n"]))
    assert all(n == 0 for _, n in rows)
