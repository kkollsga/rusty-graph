"""Tests for EXPLAIN and PROFILE query plan output."""

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
    """Basic EXPLAIN functionality — now returns a structured ResultView."""

    def test_explain_returns_result_view(self, graph):
        """EXPLAIN returns a ResultView with columns [step, operation, estimated_rows]."""
        result = graph.cypher("EXPLAIN MATCH (n:Person) RETURN n.name")
        assert hasattr(result, "columns")
        assert "step" in result.columns
        assert "operation" in result.columns
        assert "estimated_rows" in result.columns

    def test_explain_has_plan_steps(self, graph):
        """Each clause becomes a row in the result."""
        result = graph.cypher("EXPLAIN MATCH (n:Person) RETURN n.name")
        rows = result.to_list()
        assert len(rows) >= 2  # At least Match and Return
        assert rows[0]["operation"] == "Match :Person"

    def test_explain_shows_node_scan(self, graph):
        """MATCH shows with type name."""
        result = graph.cypher("EXPLAIN MATCH (n:Person) RETURN n.name")
        ops = [r["operation"] for r in result.to_list()]
        assert any("Match :Person" in op for op in ops)

    def test_explain_shows_filter(self, graph):
        """WHERE shows as a step."""
        result = graph.cypher("EXPLAIN MATCH (n:Person) WHERE n.age > 25 RETURN n.name")
        ops = [r["operation"] for r in result.to_list()]
        assert "Where" in ops

    def test_explain_does_not_execute(self, graph):
        """EXPLAIN on a mutation does not actually mutate."""
        graph.cypher("EXPLAIN CREATE (:Person {name: 'Charlie'})")
        # If it had executed, there would be 3 Persons
        result = graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        assert result[0]['cnt'] == 2

    def test_explain_cardinality_estimates(self, graph):
        """MATCH steps include estimated row counts based on type_indices."""
        result = graph.cypher("EXPLAIN MATCH (n:Person) RETURN n.name")
        rows = result.to_list()
        match_row = rows[0]
        assert match_row["estimated_rows"] == 2  # 2 Person nodes


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
        ops = [r["operation"] for r in result.to_list()]
        assert any("FusedOptionalMatchAggregate" in op for op in ops)

    def test_explain_shows_projection(self, graph):
        """Return step exists in plan."""
        result = graph.cypher("EXPLAIN MATCH (n:Person) RETURN n.name AS name, n.age AS age")
        ops = [r["operation"] for r in result.to_list()]
        assert any("Return" in op for op in ops)

    def test_explain_shows_topk_fusion(self, graph):
        """ORDER BY + LIMIT fuses into FusedOrderByTopK."""
        result = graph.cypher("""
            EXPLAIN MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT 5
        """)
        ops = [r["operation"] for r in result.to_list()]
        assert any("FusedOrderByTopK" in op for op in ops)

    def test_explain_shows_sort_unfused(self, graph):
        """ORDER BY without LIMIT shows as separate step."""
        result = graph.cypher("""
            EXPLAIN MATCH (n:Person) RETURN n.name ORDER BY n.name
        """)
        ops = [r["operation"] for r in result.to_list()]
        assert any("OrderBy" in op for op in ops)


class TestProfile:
    """PROFILE executes the query and returns per-clause statistics."""

    def test_profile_returns_results_and_stats(self, graph):
        """PROFILE returns results plus a profile list."""
        result = graph.cypher("PROFILE MATCH (n:Person) RETURN n.name")
        assert len(result) == 2  # 2 Person nodes
        assert result.profile is not None
        assert len(result.profile) >= 2

    def test_profile_clause_stats_structure(self, graph):
        """Each profile entry has clause, rows_in, rows_out, elapsed_us."""
        result = graph.cypher("PROFILE MATCH (n:Person) RETURN n.name")
        step = result.profile[0]
        assert "clause" in step
        assert "rows_in" in step
        assert "rows_out" in step
        assert "elapsed_us" in step

    def test_profile_row_counts(self, graph):
        """MATCH produces rows, WHERE filters them."""
        result = graph.cypher("PROFILE MATCH (n:Person) WHERE n.age > 28 RETURN n.name")
        profile = result.profile
        # Match should produce 2 rows
        match_step = profile[0]
        assert match_step["clause"].startswith("Match")
        assert match_step["rows_out"] == 2
        # Where should filter to 1 (Alice, age 30)
        where_step = profile[1]
        assert where_step["clause"] == "Where"
        assert where_step["rows_in"] == 2
        assert where_step["rows_out"] == 1

    def test_profile_fused_count(self, graph):
        """Fused count optimization is shown in PROFILE."""
        result = graph.cypher("PROFILE MATCH (n:Person) RETURN count(n)")
        profile = result.profile
        assert len(profile) == 1
        assert "FusedCountTypedNode" in profile[0]["clause"]

    def test_profile_mutation(self, graph):
        """PROFILE works with mutation queries."""
        result = graph.cypher("PROFILE MATCH (n:Person) WHERE n.age < 26 SET n.young = true")
        assert result.stats is not None
        assert result.stats["properties_set"] == 1
        assert result.profile is not None
        assert len(result.profile) >= 2

    def test_profile_is_none_for_normal_queries(self, graph):
        """Normal queries (no PROFILE prefix) have profile=None."""
        result = graph.cypher("MATCH (n:Person) RETURN n.name")
        assert result.profile is None
