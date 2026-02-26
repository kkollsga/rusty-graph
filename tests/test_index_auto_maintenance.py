"""Tests for index auto-maintenance during Cypher mutations (CREATE, SET, REMOVE, MERGE)."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


@pytest.fixture
def indexed_graph():
    """Graph with Person nodes, KNOWS edges, and property indexes on 'city' and 'age'."""
    graph = KnowledgeGraph()
    people = pd.DataFrame({
        'person_id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [28, 35, 42],
        'city': ['Oslo', 'Bergen', 'Oslo'],
    })
    graph.add_nodes(people, 'Person', 'person_id', 'name')

    edges = pd.DataFrame({
        'from_id': [1, 2],
        'to_id': [2, 3],
    })
    graph.add_connections(edges, 'KNOWS', 'Person', 'from_id', 'Person', 'to_id')

    graph.create_index('Person', 'city')
    graph.create_index('Person', 'age')
    return graph


def _names_from_filter(graph, conditions):
    """Helper: type_filter('Person') + filter(conditions), return set of names via Cypher."""
    graph.select('Person')
    result = graph.where(conditions)
    count = result.len()
    # Also verify via get_nodes which gives property dicts
    nodes = result.collect()
    return count, {n.get('title', n.get('name', '')) for n in nodes}


class TestCreateUpdatesIndex:
    """Cypher CREATE should auto-update property indexes."""

    def test_create_node_appears_in_index(self, indexed_graph):
        """After CREATE, the new node should be findable via index-backed filter."""
        indexed_graph.cypher("CREATE (p:Person {name: 'Dave', city: 'Stavanger', age: 25})")

        count, names = _names_from_filter(indexed_graph, {'city': 'Stavanger'})
        assert count == 1
        assert 'Dave' in names

    def test_create_multiple_nodes_indexed(self, indexed_graph):
        """Creating multiple nodes should all appear in indexes."""
        indexed_graph.cypher("CREATE (p:Person {name: 'Eve', city: 'Oslo', age: 22})")
        indexed_graph.cypher("CREATE (p:Person {name: 'Frank', city: 'Oslo', age: 33})")

        count, names = _names_from_filter(indexed_graph, {'city': 'Oslo'})
        # Original: Alice, Charlie in Oslo + Eve, Frank = 4
        assert count == 4
        assert {'Alice', 'Charlie', 'Eve', 'Frank'} <= names


class TestSetUpdatesIndex:
    """Cypher SET should update property indexes (remove old, add new)."""

    def test_set_updates_city_index(self, indexed_graph):
        """After SET city, old value gone from index, new value present."""
        indexed_graph.cypher("MATCH (p:Person {name: 'Alice'}) SET p.city = 'Tromsø'")

        count_oslo, names_oslo = _names_from_filter(indexed_graph, {'city': 'Oslo'})
        assert 'Alice' not in names_oslo
        assert 'Charlie' in names_oslo

        count_tromso, names_tromso = _names_from_filter(indexed_graph, {'city': 'Tromsø'})
        assert count_tromso == 1
        assert 'Alice' in names_tromso

    def test_set_updates_age_index(self, indexed_graph):
        """After SET age, index reflects new value."""
        indexed_graph.cypher("MATCH (p:Person {name: 'Bob'}) SET p.age = 99")

        count_old, _ = _names_from_filter(indexed_graph, {'age': 35})
        assert count_old == 0

        count_new, names_new = _names_from_filter(indexed_graph, {'age': 99})
        assert count_new == 1
        assert 'Bob' in names_new

    def test_set_name_updates_index(self, indexed_graph):
        """SET name should update the name property in the index."""
        indexed_graph.create_index('Person', 'name')
        indexed_graph.cypher("MATCH (p:Person {name: 'Alice'}) SET p.name = 'Alicia'")

        count_old, _ = _names_from_filter(indexed_graph, {'name': 'Alice'})
        assert count_old == 0

        count_new, _ = _names_from_filter(indexed_graph, {'name': 'Alicia'})
        assert count_new == 1


class TestRemoveUpdatesIndex:
    """Cypher REMOVE should remove values from property indexes."""

    def test_remove_property_clears_index(self, indexed_graph):
        """After REMOVE p.city, the node should no longer appear in city index."""
        indexed_graph.cypher("MATCH (p:Person {name: 'Alice'}) REMOVE p.city")

        count, names = _names_from_filter(indexed_graph, {'city': 'Oslo'})
        assert 'Alice' not in names
        assert 'Charlie' in names

    def test_remove_age_clears_index(self, indexed_graph):
        """After REMOVE p.age, the node should no longer appear in age index."""
        indexed_graph.cypher("MATCH (p:Person {name: 'Bob'}) REMOVE p.age")

        count, _ = _names_from_filter(indexed_graph, {'age': 35})
        assert count == 0


class TestMergeUpdatesIndex:
    """Cypher MERGE should maintain indexes for both match and create paths."""

    def test_merge_create_path_indexed(self, indexed_graph):
        """MERGE create path should add new node to index."""
        indexed_graph.cypher(
            "MERGE (p:Person {name: 'Grace'}) ON CREATE SET p.city = 'Bodø', p.age = 29"
        )

        count, names = _names_from_filter(indexed_graph, {'city': 'Bodø'})
        assert count == 1
        assert 'Grace' in names

    def test_merge_match_path_set_updates_index(self, indexed_graph):
        """MERGE match path with ON MATCH SET should update index."""
        indexed_graph.cypher(
            "MERGE (p:Person {name: 'Alice'}) ON MATCH SET p.city = 'Drammen'"
        )

        _, names_oslo = _names_from_filter(indexed_graph, {'city': 'Oslo'})
        assert 'Alice' not in names_oslo

        count_drammen, names_drammen = _names_from_filter(indexed_graph, {'city': 'Drammen'})
        assert count_drammen == 1
        assert 'Alice' in names_drammen


class TestTypeMetadata:
    """Cypher CREATE should populate type metadata (replaces old SchemaNode tests)."""

    def test_cypher_create_populates_metadata(self):
        """CREATE for a new type should populate type metadata, not create extra graph nodes."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (a:Animal {name: 'Rex', species: 'Dog'})")

        info = graph.graph_info()
        # Only 1 actual node — no SchemaNode pollution
        assert info['node_count'] == 1
        assert info['type_count'] == 1  # Just Animal

        # Schema string should mention Animal type
        schema = graph.schema_text()
        assert 'Animal' in schema

    def test_mixed_python_cypher_metadata_consistent(self):
        """Metadata should be consistent whether nodes added via Python or Cypher."""
        graph = KnowledgeGraph()
        people = pd.DataFrame({
            'person_id': [1],
            'name': ['Alice'],
            'age': [28],
        })
        graph.add_nodes(people, 'Person', 'person_id', 'name')

        # Add another Person via Cypher — metadata should merge cleanly
        graph.cypher("CREATE (p:Person {name: 'Bob', age: 35})")

        # Schema should mention Person type
        schema = graph.schema_text()
        assert 'Person' in schema

        # No SchemaNode graph nodes should exist — MATCH returns nothing
        result = graph.cypher("MATCH (s:SchemaNode) RETURN s.name")
        assert len(result) == 0


class TestCompositeIndexMaintenance:
    """Composite indexes should also be maintained by mutations."""

    def test_set_updates_composite_index(self):
        """SET on a property in a composite index should update the index."""
        graph = KnowledgeGraph()
        people = pd.DataFrame({
            'person_id': [1, 2],
            'name': ['Alice', 'Bob'],
            'city': ['Oslo', 'Bergen'],
            'age': [28, 35],
        })
        graph.add_nodes(people, 'Person', 'person_id', 'name')
        graph.create_composite_index('Person', ['city', 'age'])

        # SET Alice's city — composite index should update
        graph.cypher("MATCH (p:Person {name: 'Alice'}) SET p.city = 'Bergen'")

        # Verify via Cypher — Alice should now be findable with Bergen+28
        result = graph.cypher(
            "MATCH (p:Person) WHERE p.city = 'Bergen' AND p.age = 28 RETURN p.name"
        )
        names = [row['p.name'] for row in result]
        assert 'Alice' in names

        # Verify old combo is gone — no one has Oslo+28 anymore
        result_old = graph.cypher(
            "MATCH (p:Person) WHERE p.city = 'Oslo' AND p.age = 28 RETURN p.name"
        )
        assert len(result_old) == 0
