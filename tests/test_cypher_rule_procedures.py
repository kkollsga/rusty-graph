"""Tests for the structural-validator CALL procedures.

Replaces ``tests/test_rules.py`` (deleted in 0.8.19). Each procedure is
exercised in isolation against a tiny graph plus end-to-end against a
realistic edge-direction setup. Procedures verified:

- ``orphan_node({type})``
- ``self_loop({type, edge})``
- ``cycle_2step({type, edge})``
- ``missing_required_edge({type, edge})``
- ``missing_inbound_edge({type, edge})``
- ``duplicate_title({type})``

Plus the direction validator that fires before iteration when a caller
passes the wrong (type, edge) pair for ``missing_required_edge`` or
``missing_inbound_edge``.
"""

import pandas as pd
import pytest

import kglite


@pytest.fixture
def integrity_graph():
    """4 LawSections (one orphaned, two share a title) + 1 Law."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "sec1", "title": "§1"},
                {"id": "sec2", "title": "§2"},
                {"id": "sec3", "title": "§3-orphan"},  # zero-degree
                {"id": "sec4", "title": "§1"},  # duplicates sec1's title
            ]
        ),
        "LawSection",
        "id",
        "title",
    )
    g.add_nodes(
        pd.DataFrame([{"id": "lov1", "title": "Test Law"}]),
        "Law",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame(
            [
                {"s": "sec1", "t": "lov1"},
                {"s": "sec2", "t": "lov1"},
                {"s": "sec4", "t": "lov1"},
            ]
        ),
        "SECTION_OF",
        "LawSection",
        "s",
        "Law",
        "t",
    )
    return g


# ---------------------------------------------------------------------
# orphan_node


def test_orphan_node_finds_zero_degree(integrity_graph):
    rows = list(integrity_graph.cypher("CALL orphan_node({type: 'LawSection'}) YIELD node RETURN node.id AS id"))
    assert len(rows) == 1
    assert rows[0]["id"] == "sec3"


def test_orphan_node_clean_type_returns_zero(integrity_graph):
    rows = list(integrity_graph.cypher("CALL orphan_node({type: 'Law'}) YIELD node RETURN count(node) AS c"))
    assert rows[0]["c"] == 0


def test_orphan_node_unknown_type_errors(integrity_graph):
    with pytest.raises((ValueError, RuntimeError), match="no nodes"):
        integrity_graph.cypher("CALL orphan_node({type: 'NoSuchType'}) YIELD node RETURN node")


def test_orphan_node_missing_param_errors(integrity_graph):
    with pytest.raises((ValueError, RuntimeError), match="missing required parameter"):
        integrity_graph.cypher("CALL orphan_node({}) YIELD node RETURN node")


# ---------------------------------------------------------------------
# self_loop + cycle_2step


@pytest.fixture
def loop_graph():
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "a", "title": "A"},
                {"id": "b", "title": "B"},
                {"id": "c", "title": "C-self"},
            ]
        ),
        "T",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame(
            [
                {"s": "a", "t": "b"},  # a→b
                {"s": "b", "t": "a"},  # b→a (forms a 2-cycle)
                {"s": "c", "t": "c"},  # self-loop on c
            ]
        ),
        "E",
        "T",
        "s",
        "T",
        "t",
    )
    return g


def test_self_loop_finds_node(loop_graph):
    rows = list(loop_graph.cypher("CALL self_loop({type: 'T', edge: 'E'}) YIELD node RETURN node.id AS id"))
    assert [r["id"] for r in rows] == ["c"]


def test_cycle_2step_finds_pair(loop_graph):
    rows = list(
        loop_graph.cypher(
            "CALL cycle_2step({type: 'T', edge: 'E'}) YIELD node_a, node_b RETURN node_a.id AS a, node_b.id AS b"
        )
    )
    assert len(rows) == 1
    pair = (rows[0]["a"], rows[0]["b"])
    assert pair == ("a", "b") or pair == ("b", "a")


def test_cycle_2step_yields_named_node_a_node_b(loop_graph):
    """Variables must be node_a / node_b — `end` is reserved by Cypher (CASE)."""
    rows = list(loop_graph.cypher("CALL cycle_2step({type: 'T', edge: 'E'}) YIELD node_a, node_b RETURN count(*) AS c"))
    assert rows[0]["c"] == 1


# ---------------------------------------------------------------------
# missing_required_edge / missing_inbound_edge + direction validation


@pytest.fixture
def directional_graph():
    """Wellbore -[IN_LICENCE]-> Licence (outbound from Wellbore)."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "w1", "title": "W1"},
                {"id": "w2", "title": "W2"},
                {"id": "w3", "title": "W3-no-licence"},
            ]
        ),
        "Wellbore",
        "id",
        "title",
    )
    g.add_nodes(
        pd.DataFrame([{"id": "l1", "title": "PL001"}]),
        "Licence",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame(
            [
                {"s": "w1", "t": "l1"},
                {"s": "w2", "t": "l1"},
            ]
        ),
        "IN_LICENCE",
        "Wellbore",
        "s",
        "Licence",
        "t",
    )
    return g


def test_missing_required_edge_finds_no_outbound(directional_graph):
    rows = list(
        directional_graph.cypher(
            "CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node RETURN node.id AS id"
        )
    )
    assert [r["id"] for r in rows] == ["w3"]


def test_missing_inbound_edge_finds_no_inbound(directional_graph):
    """All licences have inbound IN_LICENCE; an unreferenced licence wouldn't."""
    directional_graph.add_nodes(
        pd.DataFrame([{"id": "l_orphan", "title": "PL999"}]),
        "Licence",
        "id",
        "title",
    )
    rows = list(
        directional_graph.cypher(
            "CALL missing_inbound_edge({type: 'Licence', edge: 'IN_LICENCE'}) YIELD node RETURN node.id AS id"
        )
    )
    assert [r["id"] for r in rows] == ["l_orphan"]


def test_direction_validator_blocks_outbound_misuse(directional_graph):
    """Calling missing_required_edge with type=Licence flips on direction validator."""
    with pytest.raises((ValueError, RuntimeError), match="DirectionMismatch"):
        directional_graph.cypher(
            "CALL missing_required_edge({type: 'Licence', edge: 'IN_LICENCE'}) YIELD node RETURN node"
        )


def test_direction_validator_blocks_inbound_misuse(directional_graph):
    """Calling missing_inbound_edge with type=Wellbore flips on direction validator."""
    with pytest.raises((ValueError, RuntimeError), match="DirectionMismatch"):
        directional_graph.cypher(
            "CALL missing_inbound_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node RETURN node"
        )


# ---------------------------------------------------------------------
# duplicate_title


def test_duplicate_title_emits_one_row_per_duplicate(integrity_graph):
    rows = list(integrity_graph.cypher("CALL duplicate_title({type: 'LawSection'}) YIELD node RETURN node.id AS id"))
    # sec1 + sec4 share '§1' — both flagged
    assert sorted(r["id"] for r in rows) == ["sec1", "sec4"]


def test_duplicate_title_aggregates_downstream(integrity_graph):
    rows = list(
        integrity_graph.cypher(
            "CALL duplicate_title({type: 'LawSection'}) YIELD node "
            "WITH node.title AS title, collect(node) AS dups "
            "WITH title, size(dups) AS dup_count "
            "RETURN title, dup_count"
        )
    )
    assert len(rows) == 1
    assert rows[0]["title"] == "§1"
    assert rows[0]["dup_count"] == 2


# ---------------------------------------------------------------------
# Composability — rule procedures flow into surrounding Cypher


def test_call_composes_with_where_and_order_by(integrity_graph):
    """Rule output flows naturally into WHERE / ORDER BY / LIMIT."""
    rows = list(
        integrity_graph.cypher(
            "CALL duplicate_title({type: 'LawSection'}) YIELD node WHERE node.id <> 'sec1' RETURN node.id AS id"
        )
    )
    assert [r["id"] for r in rows] == ["sec4"]


def test_call_composes_with_match_and_aggregation(directional_graph):
    """Cross-reference query results against rule output in a single Cypher pass."""
    # Add wellbore w4 with no IN_LICENCE so missing_required_edge has 2 hits
    directional_graph.add_nodes(
        pd.DataFrame([{"id": "w4", "title": "W4"}]),
        "Wellbore",
        "id",
        "title",
    )
    rows = list(
        directional_graph.cypher(
            "CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node "
            "WHERE node.title CONTAINS 'W' "
            "RETURN count(node) AS c"
        )
    )
    assert rows[0]["c"] == 2


# ---------------------------------------------------------------------
# Procedure catalogue + YIELD validation


def test_list_procedures_includes_rule_procedures(integrity_graph):
    rows = list(integrity_graph.cypher("CALL list_procedures() YIELD name RETURN name"))
    names = {r["name"] for r in rows}
    for p in (
        "orphan_node",
        "self_loop",
        "cycle_2step",
        "missing_required_edge",
        "missing_inbound_edge",
        "duplicate_title",
    ):
        assert p in names, f"missing procedure: {p}"


def test_invalid_yield_column_rejected(integrity_graph):
    with pytest.raises((ValueError, RuntimeError), match="does not yield"):
        integrity_graph.cypher("CALL orphan_node({type: 'LawSection'}) YIELD bogus RETURN bogus")
