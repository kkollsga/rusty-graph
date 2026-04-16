"""Tests for disk-mode iterative updates: add/remove nodes and edges on loaded graphs."""

import shutil
import tempfile

import pandas as pd
import pytest

from kglite import KnowledgeGraph, load


@pytest.fixture
def disk_dir():
    """Temporary directory for disk graph, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="kglite_disk_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def build_small_disk_graph(path: str) -> KnowledgeGraph:
    """Build a small disk graph, save, and return a freshly loaded copy."""
    g = KnowledgeGraph(storage="disk", path=path)
    nodes = pd.DataFrame(
        {
            "nid": ["A", "B", "C", "D", "E"],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "type": ["Person"] * 5,
        }
    )
    g.add_nodes(nodes, "Person", "nid", "name")
    edges = pd.DataFrame(
        {
            "from_id": ["A", "A", "B", "C", "D"],
            "to_id": ["B", "C", "C", "D", "E"],
        }
    )
    g.add_connections(edges, "KNOWS", "Person", "from_id", "Person", "to_id")
    g.save(path)
    # Load fresh copy to test post-load behavior
    return load(path)


class TestPhase1OverflowInEdgeIterators:
    """Verify edges_directed returns overflow edges (Phase 1 fix)."""

    def test_overflow_edges_visible_in_query(self, disk_dir):
        g = build_small_disk_graph(disk_dir)
        # Add edge post-load (goes to overflow)
        g.add_connections(
            pd.DataFrame({"from_id": ["E"], "to_id": ["A"]}),
            "KNOWS",
            "Person",
            "from_id",
            "Person",
            "to_id",
        )
        # Query should see all 6 edges (5 original + 1 new)
        result = g.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt").to_df()
        assert result["cnt"][0] == 6

    def test_overflow_edges_visible_in_traversal(self, disk_dir):
        g = build_small_disk_graph(disk_dir)
        g.add_connections(
            pd.DataFrame({"from_id": ["E"], "to_id": ["A"]}),
            "KNOWS",
            "Person",
            "from_id",
            "Person",
            "to_id",
        )
        # Eve should now have an outgoing KNOWS edge to Alice
        result = g.cypher("MATCH (e:Person {id: 'E'})-[:KNOWS]->(target) RETURN target.id AS tid").to_df()
        assert "A" in result["tid"].values


class TestPhase2MmapLifecycle:
    """Save/load cycle with same-dir CSR build (Phase 2 fix)."""

    def test_save_load_small_graph(self, disk_dir):
        """Build → save → load → query. The original panic case."""
        g = KnowledgeGraph(storage="disk", path=disk_dir)
        nodes = pd.DataFrame(
            {
                "nid": list(range(100)),
                "name": [f"Node_{i}" for i in range(100)],
            }
        )
        g.add_nodes(nodes, "Item", "nid", "name")
        edges = pd.DataFrame(
            {
                "from_id": [i % 100 for i in range(200)],
                "to_id": [(i * 7 + 3) % 100 for i in range(200)],
            }
        )
        g.add_connections(edges, "LINKS", "Item", "from_id", "Item", "to_id")
        g.save(disk_dir)

        # Load and verify
        g2 = load(disk_dir)
        result = g2.cypher("MATCH ()-[r:LINKS]->() RETURN count(r) AS cnt").to_df()
        assert result["cnt"][0] == 200

    def test_save_load_preserves_node_data(self, disk_dir):
        g = build_small_disk_graph(disk_dir)
        result = g.cypher("MATCH (n:Person {id: 'A'}) RETURN n.title AS name").to_df()
        assert result["name"][0] == "Alice"


class TestPhase3PostLoadEdgeOverflow:
    """Post-load add_edge routes to overflow, visible immediately."""

    def test_add_single_edge_after_load(self, disk_dir):
        g = build_small_disk_graph(disk_dir)
        original = g.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt").to_df()["cnt"][0]
        g.add_connections(
            pd.DataFrame({"from_id": ["D"], "to_id": ["A"]}),
            "KNOWS",
            "Person",
            "from_id",
            "Person",
            "to_id",
        )
        updated = g.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt").to_df()["cnt"][0]
        assert updated == original + 1

    def test_add_multiple_edges_after_load(self, disk_dir):
        g = build_small_disk_graph(disk_dir)
        new_edges = pd.DataFrame(
            {
                "from_id": ["D", "E", "B"],
                "to_id": ["A", "A", "D"],
            }
        )
        g.add_connections(new_edges, "KNOWS", "Person", "from_id", "Person", "to_id")
        result = g.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt").to_df()
        assert result["cnt"][0] == 8  # 5 original + 3 new

    def test_add_new_connection_type_after_load(self, disk_dir):
        g = build_small_disk_graph(disk_dir)
        g.add_connections(
            pd.DataFrame({"from_id": ["A"], "to_id": ["D"]}),
            "WORKS_WITH",
            "Person",
            "from_id",
            "Person",
            "to_id",
        )
        # Both edge types visible
        knows = g.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt").to_df()["cnt"][0]
        works = g.cypher("MATCH ()-[r:WORKS_WITH]->() RETURN count(r) AS cnt").to_df()["cnt"][0]
        assert knows == 5
        assert works == 1

    def test_save_reload_overflow_edges(self, disk_dir):
        """Add edges after load, save, reload — edges should persist."""
        g = build_small_disk_graph(disk_dir)
        g.add_connections(
            pd.DataFrame({"from_id": ["E", "D"], "to_id": ["A", "A"]}),
            "KNOWS",
            "Person",
            "from_id",
            "Person",
            "to_id",
        )
        g.save(disk_dir)
        g2 = load(disk_dir)
        result = g2.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt").to_df()
        assert result["cnt"][0] == 7  # 5 + 2


class TestPhase5Compaction:
    """Compact merges overflow back into CSR."""

    def test_compact_merges_overflow(self, disk_dir):
        g = build_small_disk_graph(disk_dir)
        # Add overflow edges
        new_edges = pd.DataFrame(
            {
                "from_id": ["D", "E", "B"],
                "to_id": ["A", "A", "D"],
            }
        )
        g.add_connections(new_edges, "KNOWS", "Person", "from_id", "Person", "to_id")
        merged = g.compact()
        assert merged == 3
        # All edges still visible
        result = g.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt").to_df()
        assert result["cnt"][0] == 8

    def test_compact_noop_when_no_overflow(self, disk_dir):
        g = build_small_disk_graph(disk_dir)
        merged = g.compact()
        assert merged == 0

    def test_compact_then_save_reload(self, disk_dir):
        g = build_small_disk_graph(disk_dir)
        g.add_connections(
            pd.DataFrame({"from_id": ["E"], "to_id": ["A"]}),
            "KNOWS",
            "Person",
            "from_id",
            "Person",
            "to_id",
        )
        g.compact()
        g.save(disk_dir)
        g2 = load(disk_dir)
        result = g2.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt").to_df()
        assert result["cnt"][0] == 6

    def test_compact_preserves_traversal(self, disk_dir):
        g = build_small_disk_graph(disk_dir)
        g.add_connections(
            pd.DataFrame({"from_id": ["E"], "to_id": ["A"]}),
            "KNOWS",
            "Person",
            "from_id",
            "Person",
            "to_id",
        )
        g.compact()
        # A→B, A→C still work after compaction
        result = g.cypher("MATCH (a:Person {id: 'A'})-[:KNOWS]->(b) RETURN b.id AS bid ORDER BY bid").to_df()
        assert list(result["bid"]) == ["B", "C"]

    def test_compact_not_disk_returns_zero(self):
        """Non-disk graph returns 0, no error."""
        g = KnowledgeGraph()
        assert g.compact() == 0


class TestPhase6ConnTypeIndexOverflow:
    """Connection-type inverted index augmented with overflow."""

    def test_new_conn_type_in_overflow(self, disk_dir):
        g = build_small_disk_graph(disk_dir)
        g.add_connections(
            pd.DataFrame({"from_id": ["A"], "to_id": ["D"]}),
            "EMPLOYS",
            "Person",
            "from_id",
            "Person",
            "to_id",
        )
        # Should find the EMPLOYS edge via Cypher (uses conn_type index for optimization)
        result = g.cypher("MATCH ()-[r:EMPLOYS]->() RETURN count(r) AS cnt").to_df()
        assert result["cnt"][0] == 1
