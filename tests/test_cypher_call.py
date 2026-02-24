"""Tests for CALL procedure() YIELD columns — graph algorithm support in Cypher."""

import pytest
from kglite import KnowledgeGraph


@pytest.fixture
def graph():
    """Small directed graph for algorithm testing."""
    g = KnowledgeGraph()
    # Star topology: center node connected to 5 leaves
    g.cypher("CREATE (:Person {name: 'Center'})")
    for i in range(1, 6):
        g.cypher(f"CREATE (:Person {{name: 'Leaf{i}'}})")
    for i in range(1, 6):
        g.cypher(f"""
            MATCH (c:Person {{name: 'Center'}}), (l:Person {{name: 'Leaf{i}'}})
            CREATE (c)-[:KNOWS]->(l)
        """)
    # Add cross-links between some leaves for community structure
    g.cypher("""
        MATCH (a:Person {name: 'Leaf1'}), (b:Person {name: 'Leaf2'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    g.cypher("""
        MATCH (a:Person {name: 'Leaf2'}), (b:Person {name: 'Leaf1'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    g.cypher("""
        MATCH (a:Person {name: 'Leaf3'}), (b:Person {name: 'Leaf4'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    return g


class TestCallPagerank:
    """Test CALL pagerank() YIELD node, score."""

    def test_basic(self, graph):
        result = graph.cypher("""
            CALL pagerank() YIELD node, score
            RETURN node.name, score
            ORDER BY score DESC
        """)
        assert len(result) == 6
        # Center should have lowest pagerank in a star (outgoing only, no incoming)
        names = [r['node.name'] for r in result]
        assert isinstance(result[0]['score'], float)
        # All leaves should appear
        for i in range(1, 6):
            assert f'Leaf{i}' in names

    def test_with_limit(self, graph):
        result = graph.cypher("""
            CALL pagerank() YIELD node, score
            RETURN node.name, score
            ORDER BY score DESC LIMIT 3
        """)
        assert len(result) == 3

    def test_with_params(self, graph):
        result = graph.cypher("""
            CALL pagerank({damping_factor: 0.5}) YIELD node, score
            RETURN node.name, score
            ORDER BY score DESC
        """)
        assert len(result) == 6
        assert all(isinstance(r['score'], float) for r in result)

    def test_with_where_filter(self, graph):
        result = graph.cypher("""
            CALL pagerank() YIELD node, score
            WHERE node.name <> 'Center'
            RETURN node.name, score
            ORDER BY score DESC
        """)
        names = [r['node.name'] for r in result]
        assert 'Center' not in names
        assert len(result) == 5

    def test_node_type_access(self, graph):
        result = graph.cypher("""
            CALL pagerank() YIELD node, score
            RETURN node.type AS type, node.name AS name, score
            LIMIT 1
        """)
        assert result[0]['type'] == 'Person'


class TestCallBetweenness:
    """Test CALL betweenness() YIELD node, score."""

    def test_basic(self, graph):
        result = graph.cypher("""
            CALL betweenness() YIELD node, score
            RETURN node.name, score
            ORDER BY score DESC LIMIT 3
        """)
        assert len(result) == 3
        # Center node should have high betweenness in star topology
        assert result[0]['node.name'] == 'Center'

    def test_alias(self, graph):
        result = graph.cypher("""
            CALL betweenness_centrality() YIELD node, score
            RETURN node.name, score
            LIMIT 1
        """)
        assert 'node.name' in result[0]

    def test_normalized_param(self, graph):
        result = graph.cypher("""
            CALL betweenness({normalized: false}) YIELD node, score
            RETURN node.name, score
            ORDER BY score DESC LIMIT 1
        """)
        assert isinstance(result[0]['score'], float)


class TestCallDegree:
    """Test CALL degree() YIELD node, score."""

    def test_basic(self, graph):
        result = graph.cypher("""
            CALL degree() YIELD node, score
            RETURN node.name, score
            ORDER BY score DESC
        """)
        assert len(result) == 6
        # Center has most connections
        assert result[0]['node.name'] == 'Center'

    def test_alias_form(self, graph):
        result = graph.cypher("""
            CALL degree_centrality() YIELD node, score
            RETURN node.name, score
            ORDER BY score DESC LIMIT 1
        """)
        assert result[0]['node.name'] == 'Center'


class TestCallCloseness:
    """Test CALL closeness() YIELD node, score."""

    def test_basic(self, graph):
        result = graph.cypher("""
            CALL closeness() YIELD node, score
            RETURN node.name, score
            ORDER BY score DESC LIMIT 1
        """)
        assert isinstance(result[0]['score'], float)


class TestCallLouvain:
    """Test CALL louvain() YIELD node, community."""

    def test_basic(self, graph):
        result = graph.cypher("""
            CALL louvain() YIELD node, community
            RETURN node.name, community
        """)
        assert len(result) == 6
        assert all(isinstance(r['community'], int) for r in result)

    def test_aggregation(self, graph):
        result = graph.cypher("""
            CALL louvain() YIELD node, community
            RETURN community, count(*) AS size
            ORDER BY size DESC
        """)
        assert len(result) > 0
        total = sum(r['size'] for r in result)
        assert total == 6

    def test_with_resolution(self, graph):
        result = graph.cypher("""
            CALL louvain({resolution: 0.5}) YIELD node, community
            RETURN community, count(*) AS size
        """)
        assert len(result) > 0


class TestCallLabelPropagation:
    """Test CALL label_propagation() YIELD node, community."""

    def test_basic(self, graph):
        result = graph.cypher("""
            CALL label_propagation() YIELD node, community
            RETURN node.name, community
        """)
        assert len(result) == 6
        assert all(isinstance(r['community'], int) for r in result)


class TestCallConnectedComponents:
    """Test CALL connected_components() YIELD node, component."""

    def test_single_component(self, graph):
        result = graph.cypher("""
            CALL connected_components() YIELD node, component
            RETURN component, count(*) AS size
        """)
        # Star graph is fully connected
        assert len(result) == 1
        assert result[0]['size'] == 6

    def test_multiple_components(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'A'})")
        g.cypher("CREATE (:Person {name: 'B'})")
        g.cypher("CREATE (:Person {name: 'C'})")
        g.cypher("""
            MATCH (a:Person {name: 'A'}), (b:Person {name: 'B'})
            CREATE (a)-[:KNOWS]->(b)
        """)
        # A-B connected, C isolated
        result = g.cypher("""
            CALL connected_components() YIELD node, component
            RETURN component, count(*) AS size
            ORDER BY size DESC
        """)
        assert len(result) == 2
        assert result[0]['size'] == 2
        assert result[1]['size'] == 1

    def test_node_access(self, graph):
        result = graph.cypher("""
            CALL connected_components() YIELD node, component
            RETURN node.name, node.type, component
            ORDER BY node.name
        """)
        assert len(result) == 6


class TestCallYieldAlias:
    """Test YIELD with AS aliases."""

    def test_yield_alias(self, graph):
        result = graph.cypher("""
            CALL pagerank() YIELD node AS n, score AS s
            RETURN n.name, s
            ORDER BY s DESC LIMIT 3
        """)
        assert len(result) == 3
        assert 'n.name' in result[0]
        assert 's' in result[0]

    def test_yield_partial(self, graph):
        """YIELD only score, not node."""
        result = graph.cypher("""
            CALL pagerank() YIELD score
            RETURN score
            ORDER BY score DESC LIMIT 3
        """)
        assert len(result) == 3
        assert 'score' in result[0]


class TestCallErrors:
    """Test error handling."""

    def test_unknown_procedure(self, graph):
        with pytest.raises(RuntimeError, match="Unknown procedure"):
            graph.cypher("CALL unknown_algo() YIELD node, score")

    def test_invalid_yield_column(self, graph):
        with pytest.raises(RuntimeError, match="does not yield"):
            graph.cypher("CALL pagerank() YIELD node, community")

    def test_missing_yield(self, graph):
        with pytest.raises(ValueError, match="YIELD"):
            graph.cypher("CALL pagerank()")


class TestCallExplain:
    """Test EXPLAIN with CALL."""

    def test_explain(self, graph):
        result = graph.cypher("""
            EXPLAIN CALL pagerank() YIELD node, score
            RETURN node.name, score
            ORDER BY score DESC LIMIT 10
        """)
        ops = [r['operation'] for r in result.to_list()]
        assert any('Call' in op for op in ops)
