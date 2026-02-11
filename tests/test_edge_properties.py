"""Tests for relationship/edge property access in Cypher queries."""

import pytest
from kglite import KnowledgeGraph


@pytest.fixture
def graph_with_edge_props():
    """Graph with relationships that have properties."""
    graph = KnowledgeGraph()

    # Create nodes
    graph.cypher("CREATE (:Movie {title: 'Inception', year: 2010})")
    graph.cypher("CREATE (:Movie {title: 'The Matrix', year: 1999})")
    graph.cypher("CREATE (:Person {name: 'Alice', age: 30})")
    graph.cypher("CREATE (:Person {name: 'Bob', age: 25})")
    graph.cypher("CREATE (:Person {name: 'Charlie', age: 35})")

    # Create relationships with properties
    graph.cypher("""
        MATCH (m:Movie {title: 'Inception'}), (p:Person {name: 'Alice'})
        CREATE (p)-[:RATED {score: 5, comment: 'Excellent', date: '2023-01-15'}]->(m)
    """)
    graph.cypher("""
        MATCH (m:Movie {title: 'Inception'}), (p:Person {name: 'Bob'})
        CREATE (p)-[:RATED {score: 4, comment: 'Good', date: '2023-02-20'}]->(m)
    """)
    graph.cypher("""
        MATCH (m:Movie {title: 'The Matrix'}), (p:Person {name: 'Alice'})
        CREATE (p)-[:RATED {score: 5, comment: 'Amazing', date: '2023-01-10'}]->(m)
    """)
    graph.cypher("""
        MATCH (m:Movie {title: 'The Matrix'}), (p:Person {name: 'Charlie'})
        CREATE (p)-[:RATED {score: 3, comment: 'Okay', date: '2023-03-01'}]->(m)
    """)

    return graph


class TestEdgePropertyAccess:
    """Test edge property access in RETURN, WHERE, ORDER BY, and aggregations."""

    def test_return_edge_properties(self, graph_with_edge_props):
        """Return edge properties in RETURN clause."""
        result = graph_with_edge_props.cypher("""
            MATCH (p:Person)-[r:RATED]->(m:Movie)
            RETURN r.score, r.comment, type(r)
        """)

        assert len(result) == 4
        assert 'r.score' in result[0]
        assert 'r.comment' in result[0]
        assert 'type(r)' in result[0]

        # Check that type(r) returns the relationship type
        assert all(row['type(r)'] == 'RATED' for row in result)

        # Check scores
        scores = [row['r.score'] for row in result]
        assert set(scores) == {3, 4, 5}

    def test_where_edge_property_filter(self, graph_with_edge_props):
        """Filter by edge properties in WHERE clause."""
        result = graph_with_edge_props.cypher("""
            MATCH (p:Person)-[r:RATED]->(m:Movie)
            WHERE r.score > 3
            RETURN p.name, m.title, r.score
        """)

        assert len(result) == 3  # Alice (2x) and Bob rated > 3
        scores = [row['r.score'] for row in result]
        assert all(s > 3 for s in scores)

    def test_where_edge_property_equals(self, graph_with_edge_props):
        """Filter by edge property equality."""
        result = graph_with_edge_props.cypher("""
            MATCH (p:Person)-[r:RATED]->(m:Movie)
            WHERE r.score = 5
            RETURN p.name, m.title
        """)

        assert len(result) == 2  # Alice rated both movies with 5
        names = [row['p.name'] for row in result]
        assert all(name == 'Alice' for name in names)

    def test_order_by_edge_property(self, graph_with_edge_props):
        """Sort by edge properties."""
        result = graph_with_edge_props.cypher("""
            MATCH (p:Person)-[r:RATED]->(m:Movie)
            RETURN p.name, r.score
            ORDER BY r.score DESC
        """)

        scores = [row['r.score'] for row in result]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 5  # Highest score first

    def test_aggregate_edge_properties(self, graph_with_edge_props):
        """Aggregate functions on edge properties."""
        result = graph_with_edge_props.cypher("""
            MATCH (p:Person)-[r:RATED]->(m:Movie)
            RETURN avg(r.score) AS avg_score, min(r.score) AS min_score, max(r.score) AS max_score
        """)

        assert len(result) == 1
        row = result[0]
        assert row['avg_score'] == 4.25  # (5+4+5+3)/4
        assert row['min_score'] == 3
        assert row['max_score'] == 5

    def test_group_by_with_edge_property_aggregation(self, graph_with_edge_props):
        """Group by node and aggregate edge properties."""
        result = graph_with_edge_props.cypher("""
            MATCH (p:Person)-[r:RATED]->(m:Movie)
            RETURN p.name, avg(r.score) AS avg_score
            ORDER BY p.name
        """)

        assert len(result) == 3  # Alice, Bob, Charlie

        # Alice rated 2 movies with 5 → avg = 5.0
        alice = [row for row in result if row['p.name'] == 'Alice'][0]
        assert alice['avg_score'] == 5.0

        # Bob rated 1 movie with 4 → avg = 4.0
        bob = [row for row in result if row['p.name'] == 'Bob'][0]
        assert bob['avg_score'] == 4.0

        # Charlie rated 1 movie with 3 → avg = 3.0
        charlie = [row for row in result if row['p.name'] == 'Charlie'][0]
        assert charlie['avg_score'] == 3.0

    def test_edge_property_with_multiple_conditions(self, graph_with_edge_props):
        """Multiple WHERE conditions including edge properties."""
        result = graph_with_edge_props.cypher("""
            MATCH (p:Person)-[r:RATED]->(m:Movie)
            WHERE r.score >= 4 AND m.year >= 2000
            RETURN p.name, m.title, r.score
            ORDER BY r.score DESC
        """)

        # Alice (Inception, 5) and Bob (Inception, 4)
        assert len(result) == 2
        scores = [row['r.score'] for row in result]
        assert scores == [5, 4]

    def test_edge_string_property_filter(self, graph_with_edge_props):
        """Filter by string edge property."""
        result = graph_with_edge_props.cypher("""
            MATCH (p:Person)-[r:RATED]->(m:Movie)
            WHERE r.comment = 'Excellent'
            RETURN p.name, m.title
        """)

        assert len(result) == 1
        assert result[0]['p.name'] == 'Alice'
        assert result[0]['m.title'] == 'Inception'

    def test_return_both_node_and_edge_properties(self, graph_with_edge_props):
        """Mix node and edge properties in RETURN."""
        result = graph_with_edge_props.cypher("""
            MATCH (p:Person)-[r:RATED]->(m:Movie)
            RETURN p.name, p.age, r.score, r.comment, m.title, m.year
            WHERE r.score = 5
        """)

        assert len(result) == 2
        for row in result:
            assert row['p.name'] == 'Alice'
            assert row['p.age'] == 30
            assert row['r.score'] == 5
            assert row['r.comment'] in ['Excellent', 'Amazing']
