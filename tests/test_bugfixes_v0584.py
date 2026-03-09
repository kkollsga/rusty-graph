"""Tests for bug fixes in v0.5.84."""

import random

import pandas as pd

from kglite import KnowledgeGraph
import pytest

# ===========================================================================
# Bug 2: Edge traversal without ORDER BY returns broken results
# ===========================================================================


@pytest.fixture
def edge_graph():
    """Graph with 100 persons, 50 companies, 300 WORKS_AT edges."""
    random.seed(42)
    g = KnowledgeGraph()

    g.add_nodes(pd.DataFrame([{"id": i, "name": f"Person_{i}"} for i in range(100)]), "Person", "id", "name")

    g.add_nodes(pd.DataFrame([{"id": i, "name": f"Company_{i}"} for i in range(50)]), "Company", "id", "name")

    edges = []
    for p in range(100):
        for c in random.sample(range(50), 3):
            edges.append({"person_id": p, "company_id": c, "role": random.choice(["eng", "mgr", "dir"])})

    g.add_connections(
        data=pd.DataFrame(edges),
        connection_type="WORKS_AT",
        source_type="Person",
        source_id_field="person_id",
        target_type="Company",
        target_id_field="company_id",
        columns=["role"],
    )
    return g


class TestBug2EdgeTraversalWithoutOrderBy:
    """Edge traversal without ORDER BY should return correct rows."""

    def test_limit_respected_without_order_by(self, edge_graph):
        res = edge_graph.cypher("""
            MATCH (p:Person)-[w:WORKS_AT]->(c:Company)
            RETURN p.title, c.title, w.role
            LIMIT 5
        """)
        assert len(res) == 5

    def test_no_nulls_without_order_by(self, edge_graph):
        res = edge_graph.cypher("""
            MATCH (p:Person)-[w:WORKS_AT]->(c:Company)
            RETURN p.title, c.title, w.role
            LIMIT 10
        """)
        for row in res:
            d = dict(row)
            assert d["p.title"] is not None, f"p.title is None in {d}"
            assert d["c.title"] is not None, f"c.title is None in {d}"
            assert d["w.role"] is not None, f"w.role is None in {d}"

    def test_same_result_with_and_without_order_by(self, edge_graph):
        # Both should return the correct number of total rows
        res_no_order = edge_graph.cypher("""
            MATCH (p:Person)-[w:WORKS_AT]->(c:Company)
            RETURN p.title, c.title, w.role
        """)
        res_with_order = edge_graph.cypher("""
            MATCH (p:Person)-[w:WORKS_AT]->(c:Company)
            RETURN p.title, c.title, w.role
            ORDER BY p.title
        """)
        assert len(res_no_order) == len(res_with_order) == 300

    def test_aggregation_unaffected(self, edge_graph):
        """Aggregation queries should still work correctly."""
        res = edge_graph.cypher("""
            MATCH ()-[w:WORKS_AT]->() RETURN count(w) AS cnt
        """)
        assert dict(res[0])["cnt"] == 300


# ===========================================================================
# Bug 1: create_connections() silently creates 0 edges
# ===========================================================================


@pytest.fixture
def simple_graph():
    """Simple graph with Person → Company edges."""
    g = KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        ),
        "Person",
        "id",
        "name",
    )
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": 10, "name": "Acme"},
                {"id": 20, "name": "Globex"},
            ]
        ),
        "Company",
        "id",
        "name",
    )
    g.add_connections(
        data=pd.DataFrame(
            [
                {"pid": 1, "cid": 10},
                {"pid": 1, "cid": 20},
                {"pid": 2, "cid": 10},
            ]
        ),
        connection_type="WORKS_AT",
        source_type="Person",
        source_id_field="pid",
        target_type="Company",
        target_id_field="cid",
    )
    return g


class TestBug1CreateConnections:
    """create_connections() should create edges from the traversal chain."""

    def test_create_connections_basic(self, simple_graph):
        result = (
            simple_graph.select("Person").traverse("WORKS_AT", target_type="Company").create_connections("PERSON_AT")
        )

        res = result.cypher("MATCH ()-[:PERSON_AT]->() RETURN count(*) AS cnt")
        assert dict(res[0])["cnt"] == 3

    def test_create_connections_preserves_existing(self, simple_graph):
        result = (
            simple_graph.select("Person").traverse("WORKS_AT", target_type="Company").create_connections("PERSON_AT")
        )

        # Original WORKS_AT edges should still exist
        res = result.cypher("MATCH ()-[:WORKS_AT]->() RETURN count(*) AS cnt")
        assert dict(res[0])["cnt"] == 3

    def test_create_connections_correct_pairs(self, simple_graph):
        result = (
            simple_graph.select("Person").traverse("WORKS_AT", target_type="Company").create_connections("PERSON_AT")
        )

        res = result.cypher("""
            MATCH (p:Person)-[:PERSON_AT]->(c:Company)
            RETURN p.title, c.title
            ORDER BY p.title, c.title
        """)
        pairs = [(dict(r)["p.title"], dict(r)["c.title"]) for r in res]
        assert ("Alice", "Acme") in pairs
        assert ("Alice", "Globex") in pairs
        assert ("Bob", "Acme") in pairs

    def test_create_connections_different_type_no_conflict(self, simple_graph):
        """Creating edges of type B should not affect existing edges of type A."""
        result = (
            simple_graph.select("Person").traverse("WORKS_AT", target_type="Company").create_connections("SHORTCUT")
        )

        # SHORTCUT edges created
        res = result.cypher("MATCH ()-[:SHORTCUT]->() RETURN count(*) AS cnt")
        assert dict(res[0])["cnt"] == 3

        # WORKS_AT edges still intact
        res = result.cypher("MATCH ()-[:WORKS_AT]->() RETURN count(*) AS cnt")
        assert dict(res[0])["cnt"] == 3


# ===========================================================================
# Bug 3: describe() documents wrong parameter name
# ===========================================================================


class TestBug3DescribeParameterName:
    def test_describe_fluent_loading_uses_columns(self):
        g = KnowledgeGraph()
        desc = g.describe(fluent=["loading"])
        # Should say "columns=" not bare "properties=" (extra_properties= is fine)
        assert "columns=" in desc
        assert ", properties=" not in desc


# ===========================================================================
# Bug 4: traverse() first arg meaning with method='contains'
# ===========================================================================


class TestBug4TraverseContainsTargetType:
    def test_target_type_kwarg_with_contains_method(self):
        """When method='contains' is used, target_type= should be respected."""
        g = KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame(
                [
                    {"id": 1, "name": "RegionA", "wkt_geometry": "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"},
                ]
            ),
            "Region",
            "id",
            "name",
        )
        g.set_spatial("Region", geometry="wkt_geometry")

        g.add_nodes(
            pd.DataFrame(
                [
                    {"id": 2, "name": "SiteX", "lat": 5.0, "lon": 5.0},
                ]
            ),
            "Site",
            "id",
            "name",
        )
        g.set_spatial("Site", location=("lat", "lon"))

        # This should work: first arg = target node type (current behavior)
        result = g.select("Region").traverse("Site", method="contains")
        assert len(result.collect()) == 1

        # Also works with target_type= explicitly
        result2 = g.select("Region").traverse(target_type="Site", method="contains")
        assert len(result2.collect()) == 1
