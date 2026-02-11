"""Tests for OPTIONAL MATCH + WITH count() fusion optimization."""

import pytest
import time
from kglite import KnowledgeGraph


@pytest.fixture
def star_graph():
    """Star graph: Center connected to 10 spokes via LINK."""
    g = KnowledgeGraph()
    g.cypher("CREATE (:Hub {name: 'Center'})")
    for i in range(10):
        g.cypher(f"CREATE (:Spoke {{name: 'S{i}'}})")
    for i in range(10):
        g.cypher(f"""
            MATCH (h:Hub {{name: 'Center'}}), (s:Spoke {{name: 'S{i}'}})
            CREATE (h)-[:LINK]->(s)
        """)
    return g


@pytest.fixture
def multi_type_graph():
    """Graph with multiple node types for chained OPTIONAL MATCH testing."""
    g = KnowledgeGraph()
    # Create 5 Person nodes
    for i in range(5):
        g.cypher(f"CREATE (:Person {{name: 'P{i}'}})")
    # Create 3 Project nodes
    for i in range(3):
        g.cypher(f"CREATE (:Project {{name: 'Proj{i}'}})")
    # Create 4 Skill nodes
    for i in range(4):
        g.cypher(f"CREATE (:Skill {{name: 'Skill{i}'}})")
    # P0 works on Proj0, Proj1; P1 works on Proj0; P2 works on Proj2
    g.cypher("MATCH (p:Person {name: 'P0'}), (pr:Project {name: 'Proj0'}) CREATE (p)-[:WORKS_ON]->(pr)")
    g.cypher("MATCH (p:Person {name: 'P0'}), (pr:Project {name: 'Proj1'}) CREATE (p)-[:WORKS_ON]->(pr)")
    g.cypher("MATCH (p:Person {name: 'P1'}), (pr:Project {name: 'Proj0'}) CREATE (p)-[:WORKS_ON]->(pr)")
    g.cypher("MATCH (p:Person {name: 'P2'}), (pr:Project {name: 'Proj2'}) CREATE (p)-[:WORKS_ON]->(pr)")
    # P0 has Skill0, Skill1; P1 has Skill2
    g.cypher("MATCH (p:Person {name: 'P0'}), (s:Skill {name: 'Skill0'}) CREATE (p)-[:HAS_SKILL]->(s)")
    g.cypher("MATCH (p:Person {name: 'P0'}), (s:Skill {name: 'Skill1'}) CREATE (p)-[:HAS_SKILL]->(s)")
    g.cypher("MATCH (p:Person {name: 'P1'}), (s:Skill {name: 'Skill2'}) CREATE (p)-[:HAS_SKILL]->(s)")
    return g


class TestFusionCorrectness:
    """Correctness tests: fused results must match non-fused results."""

    def test_basic_optional_match_count(self, star_graph):
        """OPTIONAL MATCH + WITH count produces correct count."""
        result = star_graph.cypher("""
            MATCH (h:Hub)
            OPTIONAL MATCH (h)-[:LINK]->(s:Spoke)
            WITH h, count(s) AS cnt
            RETURN h.name, cnt
        """)
        assert len(result) == 1
        assert result[0]['h.name'] == 'Center'
        assert result[0]['cnt'] == 10

    def test_no_matches_returns_zero(self, star_graph):
        """When OPTIONAL MATCH finds nothing, count should be 0."""
        result = star_graph.cypher("""
            MATCH (s:Spoke)
            OPTIONAL MATCH (s)-[:LINK]->(other:Hub)
            WITH s, count(other) AS cnt
            RETURN s.name, cnt
            ORDER BY s.name
        """)
        assert len(result) == 10
        for row in result:
            assert row['cnt'] == 0

    def test_mixed_counts(self, multi_type_graph):
        """Different nodes have different counts."""
        result = multi_type_graph.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:WORKS_ON]->(pr:Project)
            WITH p, count(pr) AS project_count
            RETURN p.name, project_count
            ORDER BY p.name
        """)
        counts = {r['p.name']: r['project_count'] for r in result}
        assert counts['P0'] == 2
        assert counts['P1'] == 1
        assert counts['P2'] == 1
        assert counts['P3'] == 0
        assert counts['P4'] == 0

    def test_with_where_on_count(self, multi_type_graph):
        """WHERE on the aggregated count filters correctly."""
        result = multi_type_graph.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:WORKS_ON]->(pr:Project)
            WITH p, count(pr) AS cnt
            WHERE cnt > 0
            RETURN p.name, cnt
            ORDER BY p.name
        """)
        names = [r['p.name'] for r in result]
        assert names == ['P0', 'P1', 'P2']

    def test_chained_optional_match(self, multi_type_graph):
        """Two chained OPTIONAL MATCH + WITH count clauses."""
        result = multi_type_graph.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:WORKS_ON]->(pr:Project)
            WITH p, count(pr) AS projects
            OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
            WITH p, projects, count(s) AS skills
            RETURN p.name, projects, skills
            ORDER BY p.name
        """)
        data = {r['p.name']: (r['projects'], r['skills']) for r in result}
        assert data['P0'] == (2, 2)
        assert data['P1'] == (1, 1)
        assert data['P2'] == (1, 0)
        assert data['P3'] == (0, 0)
        assert data['P4'] == (0, 0)

    def test_node_properties_survive_fusion(self, star_graph):
        """Node properties accessible after fused OPTIONAL MATCH + WITH."""
        star_graph.cypher("MATCH (h:Hub {name: 'Center'}) SET h.city = 'Oslo'")
        result = star_graph.cypher("""
            MATCH (h:Hub)
            OPTIONAL MATCH (h)-[:LINK]->(s:Spoke)
            WITH h, count(s) AS cnt
            RETURN h.name, h.city, cnt
        """)
        assert result[0]['h.city'] == 'Oslo'
        assert result[0]['cnt'] == 10


class TestFusionPerformance:
    """Performance test: fusion should be significantly faster."""

    def test_fusion_faster_than_expansion(self):
        """Build a graph where fusion matters and verify it runs fast."""
        g = KnowledgeGraph()
        # Create 200 Person nodes, each connected to 20 Items
        for i in range(200):
            g.cypher(f"CREATE (:Person {{name: 'P{i}'}})")
        for i in range(200):
            for j in range(20):
                g.cypher(f"CREATE (:Item {{name: 'I{i}_{j}'}})")
                g.cypher(f"""
                    MATCH (p:Person {{name: 'P{i}'}}), (it:Item {{name: 'I{i}_{j}'}})
                    CREATE (p)-[:OWNS]->(it)
                """)

        start = time.time()
        result = g.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:OWNS]->(i:Item)
            WITH p, count(i) AS item_count
            RETURN p.name, item_count
            ORDER BY p.name
        """)
        elapsed = time.time() - start

        assert len(result) == 200
        # Each person should have 20 items
        for row in result:
            assert row['item_count'] == 20

        # With fusion: should be well under 1 second
        # Without fusion (200 * 20 = 4000 expanded rows): would be slower
        assert elapsed < 2.0, f"Fusion query took {elapsed:.2f}s, expected < 2s"
