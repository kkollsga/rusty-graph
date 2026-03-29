"""Tests for expand, to_subgraph, subgraph_stats."""

import pandas as pd

from kglite import KnowledgeGraph


class TestExpand:
    def test_expand_single_hop(self, small_graph):
        alice = small_graph.select("Person").where({"title": "Alice"})
        expanded = alice.expand(hops=1)
        assert expanded.len() >= 2  # Alice + at least Bob/Charlie

    def test_expand_multiple_hops(self, petroleum_graph):
        play = petroleum_graph.select("Play").where({"title": "North Sea Play"})
        expanded = play.expand(hops=2)
        assert expanded.len() > 1

    def test_expand_does_not_include_isolated(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        graph.add_nodes(df, "Node", "id", "name")
        conn_df = pd.DataFrame({"source": [1], "target": [2]})
        graph.add_connections(conn_df, "LINKS", "Node", "source", "Node", "target")

        start = graph.select("Node").where({"title": "A"})
        expanded = start.expand(hops=1)
        # Should include A and B but not C (isolated from A)
        assert expanded.len() == 2

    def test_expand_from_empty(self, small_graph):
        empty = small_graph.select("NonExistent")
        expanded = empty.expand(hops=2)
        assert expanded.len() == 0


class TestSubgraph:
    def test_to_subgraph_basic(self, small_graph):
        alice = small_graph.select("Person").where({"title": "Alice"})
        expanded = alice.expand(hops=1)
        subgraph = expanded.to_subgraph()
        assert subgraph.select("Person").len() >= 2

    def test_subgraph_is_independent(self, small_graph):
        selection = small_graph.select("Person")
        subgraph = selection.to_subgraph()
        # Subgraph should be a separate graph
        assert subgraph.select("Person").len() == 3

    def test_subgraph_stats(self, small_graph):
        selection = small_graph.select("Person")
        stats = selection.subgraph_stats()
        assert stats is not None

    def test_explain_shows_expand(self, small_graph):
        result = small_graph.select("Person").where({"title": "Alice"}).expand(hops=1)
        explanation = result.explain()
        assert "EXPAND" in explanation.upper() or "expand" in explanation.lower()


class TestExpandExtended:
    """Additional expand tests migrated from pytest/test_subgraph_extraction.py."""

    @staticmethod
    def _hub_graph():
        """Build a Central/Connected/Peripheral/Isolated hub graph."""
        graph = KnowledgeGraph()

        central_df = pd.DataFrame(
            {"id": [1, 2, 3], "title": ["Center1", "Center2", "Center3"], "group": ["A", "A", "B"]}
        )
        graph.add_nodes(central_df, "Central", "id", "title")

        connected_df = pd.DataFrame(
            {
                "id": [10, 20, 30, 40, 50],
                "title": ["Connected1", "Connected2", "Connected3", "Connected4", "Connected5"],
                "value": [100, 200, 300, 400, 500],
            }
        )
        graph.add_nodes(connected_df, "Connected", "id", "title")

        peripheral_df = pd.DataFrame({"id": [100, 200, 300], "title": ["Peripheral1", "Peripheral2", "Peripheral3"]})
        graph.add_nodes(peripheral_df, "Peripheral", "id", "title")

        isolated_df = pd.DataFrame({"id": [1000, 2000], "title": ["Isolated1", "Isolated2"]})
        graph.add_nodes(isolated_df, "Isolated", "id", "title")

        central_to_connected = pd.DataFrame({"central_id": [1, 1, 2, 2, 3], "connected_id": [10, 20, 30, 40, 50]})
        graph.add_connections(
            central_to_connected, "HAS_CONNECTED", "Central", "central_id", "Connected", "connected_id"
        )

        connected_to_peripheral = pd.DataFrame({"connected_id": [10, 20, 30], "peripheral_id": [100, 200, 300]})
        graph.add_connections(
            connected_to_peripheral, "HAS_PERIPHERAL", "Connected", "connected_id", "Peripheral", "peripheral_id"
        )

        return graph

    def test_expand_default_hops(self):
        """expand() without arguments defaults to 1 hop."""
        graph = self._hub_graph()
        expanded = graph.select("Central").where({"title": "Center1"}).expand()
        nodes = expanded.collect().to_df()
        titles = nodes["title"].tolist()

        assert "Center1" in titles
        assert "Connected1" in titles
        # 2-hop nodes should not be included
        assert "Peripheral1" not in titles

    def test_to_subgraph_preserves_connections(self):
        """Edges between selected nodes survive subgraph extraction."""
        graph = self._hub_graph()
        expanded = graph.select("Central").expand(hops=1)
        subgraph = expanded.to_subgraph()

        stats = subgraph.select("Central").expand(hops=1).subgraph_stats()
        assert stats["edge_count"] > 0, "Subgraph should have preserved edges"
        assert "HAS_CONNECTED" in stats["connection_types"]
        assert "Central" in stats["node_types"]
        assert "Connected" in stats["node_types"]

    def test_expand_and_subgraph_combined(self):
        """Workflow: select -> expand -> subgraph extracts a neighbourhood."""
        graph = self._hub_graph()
        subgraph = graph.select("Central").where({"title": "Center1"}).expand(hops=2).to_subgraph()

        all_nodes = (
            subgraph.select("Central").count()
            + subgraph.select("Connected").count()
            + subgraph.select("Peripheral").count()
        )
        assert all_nodes > 1  # More than just Center1

    def test_expand_from_multiple_nodes(self):
        """Expanding from all Central nodes reaches every Connected node."""
        graph = self._hub_graph()
        expanded = graph.select("Central").expand(hops=1)
        nodes = expanded.collect().to_df()

        # 3 Central + 5 Connected
        assert len(nodes) >= 8
