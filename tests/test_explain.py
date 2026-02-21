"""Tests for EXPLAIN query plan output."""

import pytest
from kglite import KnowledgeGraph


@pytest.fixture
def graph():
    g = KnowledgeGraph()
    g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
    g.cypher("CREATE (:Person {name: 'Bob', age: 25})")
    g.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    return g


class TestExplainBasic:
    """Basic EXPLAIN functionality."""

    def test_explain_returns_string(self, graph):
        """EXPLAIN returns a string, not a list."""
        result = graph.cypher("EXPLAIN MATCH (n:Person) RETURN n.name")
        assert isinstance(result, str)

    def test_explain_contains_query_plan(self, graph):
        """Output starts with 'Query Plan:'."""
        result = graph.cypher("EXPLAIN MATCH (n:Person) RETURN n.name")
        assert result.startswith("Query Plan:")

    def test_explain_shows_node_scan(self, graph):
        """MATCH shows as NodeScan with type."""
        result = graph.cypher("EXPLAIN MATCH (n:Person) RETURN n.name")
        assert "NodeScan (MATCH) :Person" in result

    def test_explain_shows_filter(self, graph):
        """WHERE shows as Filter."""
        result = graph.cypher("EXPLAIN MATCH (n:Person) WHERE n.age > 25 RETURN n.name")
        assert "Filter (WHERE)" in result

    def test_explain_does_not_execute(self, graph):
        """EXPLAIN on a mutation does not actually mutate."""
        graph.cypher("EXPLAIN CREATE (:Person {name: 'Charlie'})")
        # If it had executed, there would be 3 Persons
        result = graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        assert result[0]['cnt'] == 2


class TestExplainOptimizations:
    """EXPLAIN shows optimizations."""

    def test_explain_shows_fusion(self, graph):
        """Optional match fusion is visible in EXPLAIN output."""
        result = graph.cypher("""
            EXPLAIN
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
            WITH p, count(f) AS friends
            RETURN p.name, friends
        """)
        assert "FusedOptionalMatchAggregate" in result
        assert "optional_match_fusion=1" in result

    def test_explain_shows_projection(self, graph):
        """RETURN shows as Projection with column names."""
        result = graph.cypher("EXPLAIN MATCH (n:Person) RETURN n.name AS name, n.age AS age")
        assert "Projection (RETURN)" in result
        assert "name" in result
        assert "age" in result

    def test_explain_shows_topk_fusion(self, graph):
        """ORDER BY + LIMIT fuses into FusedOrderByTopK."""
        result = graph.cypher("""
            EXPLAIN MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT 5
        """)
        assert "FusedOrderByTopK" in result
        assert "k=5" in result
        assert "order_by_topk_fusion" in result

    def test_explain_shows_sort_and_limit_unfused(self, graph):
        """ORDER BY without LIMIT shows as separate Sort step."""
        result = graph.cypher("""
            EXPLAIN MATCH (n:Person) RETURN n.name ORDER BY n.name
        """)
        assert "Sort (ORDER BY)" in result
