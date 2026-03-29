"""Tests for connection property aggregation (sum, avg, count, store_as)."""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def discovery_field_graph():
    """Graph with Discovery nodes connected to Field nodes via EXTENDS_INTO edges.

    Discovery A -> Field 1 (share_pct=30), Field 2 (share_pct=70)  => sum=100
    Discovery B -> Field 2 (share_pct=50), Field 3 (share_pct=50)  => sum=100
    Discovery C -> Field 4 (share_pct=100)                         => sum=100
    """
    graph = KnowledgeGraph()

    discoveries = pd.DataFrame(
        {
            "discovery_id": [1, 2, 3],
            "name": ["Discovery A", "Discovery B", "Discovery C"],
        }
    )
    fields = pd.DataFrame(
        {
            "field_id": [101, 102, 103, 104],
            "name": ["Field 1", "Field 2", "Field 3", "Field 4"],
        }
    )

    graph.add_nodes(discoveries, "Discovery", "discovery_id", "name")
    graph.add_nodes(fields, "Field", "field_id", "name")

    connections = pd.DataFrame(
        {
            "discovery_id": [1, 1, 2, 2, 3],
            "field_id": [101, 102, 102, 103, 104],
            "share_pct": [30.0, 70.0, 50.0, 50.0, 100.0],
        }
    )
    graph.add_connections(
        connections,
        "EXTENDS_INTO",
        "Discovery",
        "discovery_id",
        "Field",
        "field_id",
        columns=["share_pct"],
    )
    return graph


@pytest.fixture
def company_license_graph():
    """Graph with Company nodes connected to License nodes via OWNS edges.

    Company A owns: License 1 (20%), License 2 (40%), License 3 (60%) => avg=40
    Company B owns: License 4 (50%), License 5 (100%)                 => avg=75
    """
    graph = KnowledgeGraph()

    companies = pd.DataFrame(
        {
            "company_id": [1, 2],
            "name": ["Company A", "Company B"],
        }
    )
    licenses = pd.DataFrame(
        {
            "license_id": [101, 102, 103, 104, 105],
            "name": ["License 1", "License 2", "License 3", "License 4", "License 5"],
        }
    )

    graph.add_nodes(companies, "Company", "company_id", "name")
    graph.add_nodes(licenses, "License", "license_id", "name")

    connections = pd.DataFrame(
        {
            "company_id": [1, 1, 1, 2, 2],
            "license_id": [101, 102, 103, 104, 105],
            "ownership_pct": [20.0, 40.0, 60.0, 50.0, 100.0],
        }
    )
    graph.add_connections(
        connections,
        "OWNS",
        "Company",
        "company_id",
        "License",
        "license_id",
        columns=["ownership_pct"],
    )
    return graph


@pytest.fixture
def prospect_estimate_graph():
    """Graph with Prospect nodes connected to Estimate nodes via HAS_ESTIMATE edges.

    Prospect A -> Est 1 (weight=0.5), Est 2 (weight=0.5) => sum=1.0
    Prospect B -> Est 3 (weight=1.0)                      => sum=1.0
    """
    graph = KnowledgeGraph()

    prospects = pd.DataFrame(
        {
            "prospect_id": [1, 2],
            "name": ["Prospect A", "Prospect B"],
        }
    )
    estimates = pd.DataFrame(
        {
            "estimate_id": [101, 102, 103],
            "name": ["Est 1", "Est 2", "Est 3"],
        }
    )

    graph.add_nodes(prospects, "Prospect", "prospect_id", "name")
    graph.add_nodes(estimates, "Estimate", "estimate_id", "name")

    connections = pd.DataFrame(
        {
            "prospect_id": [1, 1, 2],
            "estimate_id": [101, 102, 103],
            "weight": [0.5, 0.5, 1.0],
        }
    )
    graph.add_connections(
        connections,
        "HAS_ESTIMATE",
        "Prospect",
        "prospect_id",
        "Estimate",
        "estimate_id",
        columns=["weight"],
    )
    return graph


@pytest.fixture
def parent_child_graph():
    """Graph with Parent nodes connected to Child nodes via HAS_CHILD edges.

    Parent A -> 3 children
    Parent B -> 1 child
    Parent C -> 2 children
    """
    graph = KnowledgeGraph()

    parents = pd.DataFrame(
        {
            "parent_id": [1, 2, 3],
            "name": ["Parent A", "Parent B", "Parent C"],
        }
    )
    children = pd.DataFrame(
        {
            "child_id": [101, 102, 103, 104, 105],
            "name": ["Child 1", "Child 2", "Child 3", "Child 4", "Child 5"],
        }
    )

    graph.add_nodes(parents, "Parent", "parent_id", "name")
    graph.add_nodes(children, "Child", "child_id", "name")

    connections = pd.DataFrame(
        {
            "parent_id": [1, 1, 1, 2, 3, 3],
            "child_id": [101, 102, 103, 104, 104, 105],
            "dummy": [1, 1, 1, 1, 1, 1],
        }
    )
    graph.add_connections(
        connections,
        "HAS_CHILD",
        "Parent",
        "parent_id",
        "Child",
        "child_id",
        columns=["dummy"],
    )
    return graph


class TestConnectionAggregation:
    """Tests for aggregate_connections=True in calculate()."""

    def test_sum_connection_properties(self, discovery_field_graph):
        """Sum share_pct on EXTENDS_INTO edges; each discovery should total 100."""
        result = (
            discovery_field_graph.select("Discovery")
            .traverse("EXTENDS_INTO")
            .calculate("sum(share_pct)", aggregate_connections=True)
        )

        assert len(result) == 3
        for value in result.values():
            assert abs(value - 100.0) < 0.01

    def test_avg_connection_properties(self, company_license_graph):
        """Avg ownership_pct on OWNS edges: Company A=40, Company B=75."""
        result = (
            company_license_graph.select("Company")
            .traverse("OWNS")
            .calculate("avg(ownership_pct)", aggregate_connections=True)
        )

        assert len(result) == 2
        values = list(result.values())
        assert any(abs(v - 40.0) < 0.01 for v in values)
        assert any(abs(v - 75.0) < 0.01 for v in values)

    def test_store_connection_aggregation(self, prospect_estimate_graph):
        """Store sum(weight) as total_weight on parent Prospect nodes."""
        updated_graph = (
            prospect_estimate_graph.select("Prospect")
            .traverse("HAS_ESTIMATE")
            .calculate("sum(weight)", store_as="total_weight", aggregate_connections=True)
        )

        prospects = updated_graph.select("Prospect")
        props = prospects.get_properties(["total_weight"])

        values = [t[0] for t in props if t[0] is not None]
        assert len(values) == 2
        assert all(abs(v - 1.0) < 0.01 for v in values)

    def test_count_connections(self, parent_child_graph):
        """Count connections per parent: A=3, B=1, C=2."""
        result = (
            parent_child_graph.select("Parent")
            .traverse("HAS_CHILD")
            .calculate("count(dummy)", aggregate_connections=True)
        )

        assert len(result) == 3
        values = list(result.values())
        assert 3 in values
        assert 1 in values
        assert 2 in values

    def test_error_without_traversal(self):
        """aggregate_connections without a prior traverse should raise an error."""
        graph = KnowledgeGraph()
        nodes = pd.DataFrame(
            {
                "node_id": [1, 2, 3],
                "name": ["Node A", "Node B", "Node C"],
                "value": [10, 20, 30],
            }
        )
        graph.add_nodes(nodes, "Node", "node_id", "name")

        with pytest.raises(Exception, match=r"(?i)traversal|2 selection levels"):
            graph.select("Node").calculate("sum(value)", aggregate_connections=True)
