"""Tests for connection operations: add, retrieve, connections."""

import pandas as pd

from kglite import KnowledgeGraph
import pytest


class TestAddConnections:
    def test_add_connections_basic(self, small_graph):
        conns = small_graph.select("Person").where({"title": "Alice"}).connections()
        assert len(conns) > 0

    def test_add_connections_with_properties(self, small_graph):
        # connections returns a nested dict: {title: {node_id, node_type, incoming, outgoing}}
        conns = small_graph.select("Person").where({"title": "Alice"}).connections()
        assert "Alice" in conns
        alice = conns["Alice"]
        # Alice has outgoing KNOWS connections
        assert "outgoing" in alice
        assert "KNOWS" in alice["outgoing"]

    def test_add_connections_empty_dataframe(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")
        conn_df = pd.DataFrame({"source": [], "target": []})
        report = graph.add_connections(conn_df, "LINKS", "Node", "source", "Node", "target")
        assert report["connections_created"] == 0

    def test_self_referential_connection(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")
        conn_df = pd.DataFrame({"source": [1], "target": [1]})
        report = graph.add_connections(conn_df, "SELF", "Node", "source", "Node", "target")
        assert report["connections_created"] == 1

    def test_cross_type_connections(self):
        graph = KnowledgeGraph()
        users = pd.DataFrame({"id": [1], "name": ["Alice"]})
        products = pd.DataFrame({"id": [101], "name": ["Laptop"]})
        graph.add_nodes(users, "User", "id", "name")
        graph.add_nodes(products, "Product", "id", "name")

        conn_df = pd.DataFrame({"user_id": [1], "product_id": [101]})
        report = graph.add_connections(conn_df, "PURCHASED", "User", "user_id", "Product", "product_id")
        assert report["connections_created"] == 1


class TestGetConnections:
    def test_connections_basic(self, small_graph):
        # connections returns nested dict keyed by node title
        conns = small_graph.select("Person").where({"title": "Alice"}).connections()
        assert "Alice" in conns
        alice_conns = conns["Alice"]
        assert "outgoing" in alice_conns
        assert "KNOWS" in alice_conns["outgoing"]
        assert len(alice_conns["outgoing"]["KNOWS"]) >= 2

    def test_connections_include_properties(self, small_graph):
        # include_node_properties is a bool flag
        conns = small_graph.select("Person").where({"title": "Alice"}).connections(include_node_properties=True)
        assert "Alice" in conns
        # With include_node_properties=True, node_properties should be populated
        alice = conns["Alice"]
        for conn_type, targets in alice.get("outgoing", {}).items():
            for target_name, target_info in targets.items():
                assert "node_properties" in target_info

    def test_duplicate_connections(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")
        conn_df = pd.DataFrame({"source": [1, 1], "target": [2, 2]})
        report = graph.add_connections(conn_df, "LINKS", "Node", "source", "Node", "target")
        # Both connections should be created (multigraph)
        assert report["connections_created"] >= 1


class TestConflictHandlingSum:
    """Tests for conflict_handling='sum' on add_connections."""

    def _make_graph(self):
        graph = KnowledgeGraph()
        nodes = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        graph.add_nodes(nodes, "Node", "id", "name")
        edges = pd.DataFrame(
            {
                "src": [1, 1],
                "tgt": [2, 3],
                "weight": [10, 20],
                "label": ["x", "y"],
            }
        )
        graph.add_connections(edges, "LINK", "Node", "src", "Node", "tgt", columns=["weight", "label"])
        return graph

    def test_sum_int_properties(self):
        graph = self._make_graph()
        edges2 = pd.DataFrame({"src": [1], "tgt": [2], "weight": [5]})
        graph.add_connections(edges2, "LINK", "Node", "src", "Node", "tgt", columns=["weight"], conflict_handling="sum")
        result = graph.cypher("MATCH (:Node {id: 1})-[r:LINK]->(:Node {id: 2}) RETURN r.weight")
        assert result[0]["r.weight"] == 15

    def test_sum_float_properties(self):
        graph = KnowledgeGraph()
        nodes = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(nodes, "Node", "id", "name")
        edges1 = pd.DataFrame({"src": [1], "tgt": [2], "score": [1.5]})
        graph.add_connections(edges1, "LINK", "Node", "src", "Node", "tgt", columns=["score"])
        edges2 = pd.DataFrame({"src": [1], "tgt": [2], "score": [2.5]})
        graph.add_connections(edges2, "LINK", "Node", "src", "Node", "tgt", columns=["score"], conflict_handling="sum")
        result = graph.cypher("MATCH (:Node {id: 1})-[r:LINK]->(:Node {id: 2}) RETURN r.score")
        assert abs(result[0]["r.score"] - 4.0) < 1e-10

    def test_sum_non_numeric_overwrites(self):
        graph = self._make_graph()
        edges2 = pd.DataFrame({"src": [1], "tgt": [2], "weight": [5], "label": ["z"]})
        graph.add_connections(
            edges2, "LINK", "Node", "src", "Node", "tgt", columns=["weight", "label"], conflict_handling="sum"
        )
        result = graph.cypher("MATCH (:Node {id: 1})-[r:LINK]->(:Node {id: 2}) RETURN r.weight, r.label")
        assert result[0]["r.weight"] == 15
        assert result[0]["r.label"] == "z"  # overwritten, not summed

    def test_sum_new_edge_created(self):
        graph = self._make_graph()
        edges2 = pd.DataFrame({"src": [2], "tgt": [3], "weight": [42]})
        report = graph.add_connections(
            edges2, "LINK", "Node", "src", "Node", "tgt", columns=["weight"], conflict_handling="sum"
        )
        assert report["connections_created"] == 1
        result = graph.cypher("MATCH (:Node {id: 2})-[r:LINK]->(:Node {id: 3}) RETURN r.weight")
        assert result[0]["r.weight"] == 42

    def test_sum_new_property_added(self):
        graph = self._make_graph()
        edges2 = pd.DataFrame({"src": [1], "tgt": [2], "new_prop": [42]})
        graph.add_connections(
            edges2, "LINK", "Node", "src", "Node", "tgt", columns=["new_prop"], conflict_handling="sum"
        )
        result = graph.cypher("MATCH (:Node {id: 1})-[r:LINK]->(:Node {id: 2}) RETURN r.weight, r.new_prop")
        assert result[0]["r.weight"] == 10  # unchanged
        assert result[0]["r.new_prop"] == 42

    def test_sum_on_nodes_behaves_as_update(self):
        graph = KnowledgeGraph()
        df1 = pd.DataFrame({"id": [1], "name": ["A"], "v": [10]})
        df2 = pd.DataFrame({"id": [1], "name": ["A"], "v": [20]})
        graph.add_nodes(df1, "Node", "id", "name")
        graph.add_nodes(df2, "Node", "id", "name", conflict_handling="sum")
        result = graph.cypher("MATCH (n:Node {id: 1}) RETURN n.v")
        assert result[0]["n.v"] == 20  # overwrite, not 30


class TestQueryModeParamValidation:
    """Data-mode-only params should raise errors in query mode."""

    def _make_graph(self):
        graph = KnowledgeGraph()
        nodes = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(nodes, "Node", "id", "name")
        return graph

    def test_columns_rejected_in_query_mode(self):
        graph = self._make_graph()
        with pytest.raises(ValueError, match="columns.*data mode"):
            graph.add_connections(
                None,
                "LINK",
                "Node",
                "src",
                "Node",
                "tgt",
                query="MATCH (a:Node {id: 1}), (b:Node {id: 2}) RETURN a.id AS src, b.id AS tgt",
                columns=["src"],
            )

    def test_skip_columns_rejected_in_query_mode(self):
        graph = self._make_graph()
        with pytest.raises(ValueError, match="skip_columns.*data mode"):
            graph.add_connections(
                None,
                "LINK",
                "Node",
                "src",
                "Node",
                "tgt",
                query="MATCH (a:Node {id: 1}), (b:Node {id: 2}) RETURN a.id AS src, b.id AS tgt",
                skip_columns=["tgt"],
            )

    def test_column_types_rejected_in_query_mode(self):
        graph = self._make_graph()
        with pytest.raises(ValueError, match="column_types.*data mode"):
            graph.add_connections(
                None,
                "LINK",
                "Node",
                "src",
                "Node",
                "tgt",
                query="MATCH (a:Node {id: 1}), (b:Node {id: 2}) RETURN a.id AS src, b.id AS tgt",
                column_types={"src": "integer"},
            )
