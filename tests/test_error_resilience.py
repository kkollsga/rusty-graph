"""Tests that failed operations do not corrupt or wipe existing graph data.

Regression tests for a bug where extract_or_clone_graph would replace
self.inner with an empty DirGraph, then if the operation failed with ?,
the original graph data was lost.
"""

import pandas as pd
import pytest
import kglite


@pytest.fixture
def graph_with_data():
    """Create a graph with some initial data to verify preservation."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [28, 35, 42],
    })
    g.add_nodes(data=df, node_type="Person", unique_id_field="user_id", node_title_field="name")

    edges_df = pd.DataFrame({"source_id": [1, 2], "target_id": [2, 3]})
    g.add_connections(
        data=edges_df,
        connection_type="KNOWS",
        source_type="Person",
        source_id_field="source_id",
        target_type="Person",
        target_id_field="target_id",
    )
    return g


class TestAddNodesErrorResilience:
    def test_failed_add_nodes_preserves_existing_data(self, graph_with_data):
        """Failed add_nodes should not wipe existing nodes."""
        g = graph_with_data
        assert g.graph_info()["node_count"] == 3

        # This should fail — unique_id_field doesn't exist in the DataFrame
        with pytest.raises(Exception):
            bad_df = pd.DataFrame({"wrong_col": [4], "name": ["Dave"]})
            g.add_nodes(data=bad_df, node_type="BadType", unique_id_field="nonexistent_id", node_title_field="name")

        # Original data must still be intact
        assert g.graph_info()["node_count"] == 3
        assert g.graph_info()["type_count"] == 1

        result = g.cypher("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob", "Charlie"]

    def test_failed_add_nodes_preserves_connections(self, graph_with_data):
        """Failed add_nodes should not wipe existing connections."""
        g = graph_with_data
        assert g.graph_info()["edge_count"] == 2

        with pytest.raises(Exception):
            bad_df = pd.DataFrame({"x": [1]})
            g.add_nodes(data=bad_df, node_type="X", unique_id_field="missing", node_title_field="x")

        assert g.graph_info()["edge_count"] == 2


class TestAddConnectionsErrorResilience:
    def test_failed_add_connections_preserves_data(self, graph_with_data):
        """Failed add_connections should not wipe existing data."""
        g = graph_with_data
        assert g.graph_info()["node_count"] == 3
        assert g.graph_info()["edge_count"] == 2

        with pytest.raises(Exception):
            bad_df = pd.DataFrame({"x": [1]})
            g.add_connections(
                data=bad_df,
                connection_type="BAD",
                source_type="NonexistentType",
                source_id_field="missing_src",
                target_type="Person",
                target_id_field="missing_tgt",
            )

        assert g.graph_info()["node_count"] == 3
        assert g.graph_info()["edge_count"] == 2


class TestCypherErrorResilience:
    def test_failed_cypher_parse_preserves_data(self, graph_with_data):
        """Cypher parse error should not wipe graph."""
        g = graph_with_data
        assert g.graph_info()["node_count"] == 3

        with pytest.raises(Exception):
            g.cypher("THIS IS NOT VALID CYPHER")

        assert g.graph_info()["node_count"] == 3

    def test_failed_cypher_mutation_preserves_data(self, graph_with_data):
        """Failed mutation (DELETE node with edges) should not wipe graph."""
        g = graph_with_data

        # Alice has connections — plain DELETE (not DETACH) should fail
        with pytest.raises(Exception):
            g.cypher("MATCH (p:Person {name: 'Alice'}) DELETE p")

        assert g.graph_info()["node_count"] == 3
        assert g.graph_info()["edge_count"] == 2

    def test_successful_operations_still_work(self, graph_with_data):
        """Verify normal mutations still work after the refactor."""
        g = graph_with_data

        g.cypher("CREATE (:Person {name: 'Dave', age: 22})")
        assert g.graph_info()["node_count"] == 4

        g.cypher("MATCH (p:Person {name: 'Dave'}) SET p.city = 'Oslo'")
        result = g.cypher("MATCH (p:Person {name: 'Dave'}) RETURN p.city")
        assert result[0]["p.city"] == "Oslo"

        g.cypher("MATCH (p:Person {name: 'Dave'}) DETACH DELETE p")
        assert g.graph_info()["node_count"] == 3


class TestBulkLoadErrorResilience:
    def test_failed_add_nodes_bulk_preserves_data(self, graph_with_data):
        """Failed add_nodes_bulk should not wipe existing data."""
        g = graph_with_data
        assert g.graph_info()["node_count"] == 3

        # add_nodes_bulk with bad data
        with pytest.raises(Exception):
            g.add_nodes_bulk([
                {"node_type": "Bad", "unique_id_field": "missing_id", "node_title_field": "name", "data": pd.DataFrame({"x": [1]})}
            ])

        assert g.graph_info()["node_count"] == 3
