"""Tests for graph maintenance operations: reindex(), vacuum(), graph_info()."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


@pytest.fixture
def graph_with_data():
    """Graph with 10 Person nodes and KNOWS edges for maintenance testing."""
    graph = KnowledgeGraph()
    people = pd.DataFrame({
        'person_id': list(range(1, 11)),
        'name': [f'Person_{i}' for i in range(1, 11)],
        'age': [20 + i for i in range(1, 11)],
        'city': ['Oslo'] * 5 + ['Bergen'] * 5,
    })
    graph.add_nodes(people, 'Person', 'person_id', 'name')

    edges = pd.DataFrame({
        'from_id': list(range(1, 10)),
        'to_id': list(range(2, 11)),
    })
    graph.add_connections(edges, 'KNOWS', 'Person', 'from_id', 'Person', 'to_id')
    return graph


class TestGraphInfo:
    """Tests for graph_info() diagnostic method."""

    def test_graph_info_clean_graph(self, graph_with_data):
        """graph_info on a graph with no deletions should show zero tombstones."""
        info = graph_with_data.graph_info()
        assert info['node_tombstones'] == 0
        assert info['fragmentation_ratio'] == 0.0
        assert info['node_count'] == info['node_capacity']
        assert info['edge_count'] == 9
        assert info['type_count'] >= 1

    def test_graph_info_after_delete(self, graph_with_data):
        """After deleting nodes via Cypher, tombstones should appear."""
        info_before = graph_with_data.graph_info()
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")
        info_after = graph_with_data.graph_info()
        assert info_after['node_count'] == info_before['node_count'] - 1
        assert info_after['node_tombstones'] == 1
        assert info_after['node_capacity'] == info_before['node_capacity']
        assert info_after['fragmentation_ratio'] > 0.0

    def test_graph_info_empty_graph(self, empty_graph):
        """Empty graph should have all zeros."""
        info = empty_graph.graph_info()
        assert info['node_count'] == 0
        assert info['node_capacity'] == 0
        assert info['node_tombstones'] == 0
        assert info['edge_count'] == 0
        assert info['fragmentation_ratio'] == 0.0

    def test_graph_info_has_all_fields(self, graph_with_data):
        """graph_info should return all documented fields."""
        info = graph_with_data.graph_info()
        expected_fields = [
            'node_count', 'node_capacity', 'node_tombstones',
            'edge_count', 'fragmentation_ratio',
            'type_count', 'property_index_count', 'composite_index_count',
        ]
        for field in expected_fields:
            assert field in info, f"Missing field: {field}"

    def test_graph_info_index_counts(self, graph_with_data):
        """Index counts should reflect created indexes."""
        info = graph_with_data.graph_info()
        assert info['property_index_count'] == 0
        assert info['composite_index_count'] == 0

        # Create some indexes
        graph_with_data.create_index('Person', 'age')
        graph_with_data.create_index('Person', 'city')
        info = graph_with_data.graph_info()
        assert info['property_index_count'] == 2

    def test_graph_info_multiple_deletes(self, graph_with_data):
        """Multiple deletions should accumulate tombstones."""
        info_before = graph_with_data.graph_info()
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_1'}) DETACH DELETE p")
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_3'}) DETACH DELETE p")
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")
        info_after = graph_with_data.graph_info()
        assert info_after['node_count'] == info_before['node_count'] - 3
        assert info_after['node_tombstones'] == 3


class TestReindex:
    """Tests for reindex() method."""

    def test_reindex_basic(self, graph_with_data):
        """reindex() should succeed without error and preserve graph data."""
        info_before = graph_with_data.graph_info()
        graph_with_data.reindex()
        info_after = graph_with_data.graph_info()
        assert info_after['node_count'] == info_before['node_count']

    def test_reindex_after_delete_fixes_type_filter(self, graph_with_data):
        """After DELETE + reindex, type_filter should work correctly."""
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")
        graph_with_data.reindex()
        filtered = graph_with_data.type_filter('Person')
        assert filtered.node_count() == 9

    def test_reindex_preserves_property_indexes(self, graph_with_data):
        """reindex() should rebuild existing property indexes."""
        graph_with_data.create_index('Person', 'city')
        graph_with_data.reindex()
        info = graph_with_data.graph_info()
        assert info['property_index_count'] == 1

    def test_reindex_preserves_data(self, graph_with_data):
        """reindex() should not change any node/edge data."""
        info_before = graph_with_data.graph_info()
        graph_with_data.reindex()
        info_after = graph_with_data.graph_info()
        assert info_before['node_count'] == info_after['node_count']
        assert info_before['edge_count'] == info_after['edge_count']

    def test_reindex_idempotent(self, graph_with_data):
        """Calling reindex() multiple times should be safe."""
        graph_with_data.reindex()
        info1 = graph_with_data.graph_info()
        graph_with_data.reindex()
        graph_with_data.reindex()
        info3 = graph_with_data.graph_info()
        assert info1['node_count'] == info3['node_count']


class TestVacuum:
    """Tests for vacuum() graph compaction method."""

    def test_vacuum_clean_graph_noop(self, graph_with_data):
        """vacuum() on a clean graph should be a no-op."""
        result = graph_with_data.vacuum()
        assert result['nodes_remapped'] == 0
        assert result['tombstones_removed'] == 0

    def test_vacuum_after_delete(self, graph_with_data):
        """vacuum() should remove tombstones after deletion."""
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")
        info_before = graph_with_data.graph_info()
        assert info_before['node_tombstones'] == 1

        result = graph_with_data.vacuum()
        nodes_before = info_before['node_count']

        assert result['nodes_remapped'] == nodes_before  # All surviving nodes remapped
        assert result['tombstones_removed'] == 1
        info_after = graph_with_data.graph_info()
        assert info_after['node_tombstones'] == 0
        assert info_after['node_count'] == nodes_before
        assert info_after['node_capacity'] == nodes_before

    def test_vacuum_preserves_node_data(self, graph_with_data):
        """After vacuum, all surviving node data should be intact."""
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")
        graph_with_data.vacuum()

        # Check surviving nodes via Cypher
        result = graph_with_data.cypher(
            "MATCH (p:Person) RETURN p.name ORDER BY p.name"
        )
        names = sorted([row['p.name'] for row in result])
        expected = sorted([f'Person_{i}' for i in range(1, 11) if i != 5])
        assert names == expected

    def test_vacuum_preserves_edges(self, graph_with_data):
        """After vacuum, surviving edges should be intact."""
        edges_before = graph_with_data.graph_info()['edge_count']
        # Delete Person_1 (has edge to Person_2)
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_1'}) DETACH DELETE p")
        edges_after_delete = graph_with_data.graph_info()['edge_count']
        graph_with_data.vacuum()

        info = graph_with_data.graph_info()
        # Edge count should be same as after delete (vacuum preserves, doesn't change)
        assert info['edge_count'] == edges_after_delete
        assert info['edge_count'] < edges_before  # Some edges were removed with the node

    def test_vacuum_enables_correct_operations(self, graph_with_data):
        """After vacuum, all graph operations should work correctly."""
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")
        graph_with_data.vacuum()

        # Type filter should work
        filtered = graph_with_data.type_filter('Person')
        assert filtered.node_count() == 9

        # Cypher MATCH should work
        result = graph_with_data.cypher("MATCH (p:Person) RETURN count(p) as cnt")
        assert result[0]['cnt'] == 9

    def test_vacuum_resets_selection(self, graph_with_data):
        """vacuum() should reset the selection since indices change."""
        # Build a selection
        filtered = graph_with_data.type_filter('Person')
        assert filtered.node_count() == 10

        # Delete and vacuum on the original graph
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")
        graph_with_data.vacuum()

        # Selection is cleared; node_count() returns total graph count (9 remaining)
        assert graph_with_data.node_count() == 9
        # After re-filtering, we get the correct count
        assert graph_with_data.type_filter('Person').node_count() == 9

    def test_vacuum_then_create(self, graph_with_data):
        """After vacuum, CREATE should still work correctly."""
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")
        graph_with_data.vacuum()

        # Create a new node
        graph_with_data.cypher(
            "CREATE (p:Person {name: 'NewPerson', age: 99})"
        )
        assert graph_with_data.last_mutation_stats['nodes_created'] == 1

        # Verify the new node exists
        result = graph_with_data.cypher(
            "MATCH (p:Person {name: 'NewPerson'}) RETURN p.age"
        )
        assert len(result) == 1
        assert result[0]['p.age'] == 99

    def test_vacuum_heavy_fragmentation(self):
        """vacuum() should handle heavy fragmentation (50% deleted)."""
        graph = KnowledgeGraph()
        people = pd.DataFrame({
            'person_id': list(range(1, 101)),
            'name': [f'Person_{i}' for i in range(1, 101)],
            'age': [20 + i for i in range(1, 101)],
        })
        graph.add_nodes(people, 'Person', 'person_id', 'name')

        info_before = graph.graph_info()

        # Delete every other person
        for i in range(1, 101, 2):
            graph.cypher(f"MATCH (p:Person {{name: 'Person_{i}'}}) DETACH DELETE p")

        info_after_delete = graph.graph_info()
        assert info_after_delete['node_count'] == info_before['node_count'] - 50
        assert info_after_delete['node_tombstones'] == 50

        result = graph.vacuum()
        assert result['tombstones_removed'] == 50

        info_after_vacuum = graph.graph_info()
        assert info_after_vacuum['node_tombstones'] == 0
        assert info_after_vacuum['fragmentation_ratio'] == 0.0

    def test_vacuum_idempotent(self, graph_with_data):
        """Calling vacuum() twice should be safe; second call is a no-op."""
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")
        result1 = graph_with_data.vacuum()
        assert result1['nodes_remapped'] > 0

        result2 = graph_with_data.vacuum()
        assert result2['nodes_remapped'] == 0
        assert result2['tombstones_removed'] == 0

    def test_vacuum_with_indexes(self, graph_with_data):
        """vacuum() should rebuild property indexes correctly."""
        graph_with_data.create_index('Person', 'city')
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")

        graph_with_data.vacuum()

        # Index should still exist
        info = graph_with_data.graph_info()
        assert info['property_index_count'] == 1

    def test_vacuum_full_lifecycle(self):
        """Full lifecycle: create → delete → vacuum → create → query."""
        graph = KnowledgeGraph()
        people = pd.DataFrame({
            'person_id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [28, 35, 42],
        })
        graph.add_nodes(people, 'Person', 'person_id', 'name')

        edges = pd.DataFrame({
            'from_id': [1, 2],
            'to_id': [2, 3],
        })
        graph.add_connections(edges, 'KNOWS', 'Person', 'from_id', 'Person', 'to_id')

        # Delete Bob (middle node, has edges)
        graph.cypher("MATCH (p:Person {name: 'Bob'}) DETACH DELETE p")

        # Vacuum
        result = graph.vacuum()
        assert result['tombstones_removed'] == 1

        # Verify graph state
        info = graph.graph_info()
        assert info['node_tombstones'] == 0
        assert info['edge_count'] == 0  # Both edges touched Bob

        # Create new node and edge
        graph.cypher("CREATE (p:Person {name: 'Dave', age: 30})")
        result = graph.cypher("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        names = sorted([row['p.name'] for row in result])
        assert names == ['Alice', 'Charlie', 'Dave']


class TestMaintenanceWorkflow:
    """Integration tests for the reindex/vacuum/graph_info workflow."""

    def test_recommended_workflow(self, graph_with_data):
        """Test the recommended maintenance workflow: check → vacuum if needed."""
        # Delete some nodes
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_1'}) DETACH DELETE p")
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_3'}) DETACH DELETE p")
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_7'}) DETACH DELETE p")

        # Check health
        info = graph_with_data.graph_info()
        assert info['node_tombstones'] == 3
        assert info['fragmentation_ratio'] > 0.0

        # Vacuum if fragmented
        if info['fragmentation_ratio'] > 0.1:
            result = graph_with_data.vacuum()
            assert result['tombstones_removed'] == 3

        # Verify clean state
        info = graph_with_data.graph_info()
        assert info['fragmentation_ratio'] == 0.0

    def test_reindex_then_vacuum(self, graph_with_data):
        """reindex() then vacuum() should work correctly."""
        info_before = graph_with_data.graph_info()
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")
        graph_with_data.reindex()
        result = graph_with_data.vacuum()
        assert result['nodes_remapped'] == info_before['node_count'] - 1
        info = graph_with_data.graph_info()
        assert info['node_tombstones'] == 0

    def test_vacuum_then_reindex(self, graph_with_data):
        """vacuum() then reindex() should be safe (reindex is already called internally)."""
        info_before = graph_with_data.graph_info()
        graph_with_data.cypher("MATCH (p:Person {name: 'Person_5'}) DETACH DELETE p")
        graph_with_data.vacuum()
        graph_with_data.reindex()
        info = graph_with_data.graph_info()
        assert info['node_tombstones'] == 0
        assert info['node_count'] == info_before['node_count'] - 1
