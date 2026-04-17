"""Regression tests for Tier A Cypher correctness fixes:

1. HAVING with aggregate expressions (previously: silent empty result when
   the HAVING predicate references count(m) instead of its alias).
2. WHERE n:Label predicate (previously: syntax error).
3. rand() / random() distinct values within a tight loop (previously: could
   collide when SystemTime resolution was insufficient).
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def people_graph():
    g = KnowledgeGraph()
    people = pd.DataFrame([{"id": i, "title": n} for i, n in enumerate(["a", "b", "c", "d"], 1)])
    g.add_nodes(people, "Person", "id", "title")
    # Node 1 has 3 outgoing R edges, node 2 has 1
    edges = pd.DataFrame([(1, 2), (1, 3), (1, 4), (2, 3)], columns=["s", "t"])
    g.add_connections(edges, "R", "Person", "s", "Person", "t")
    return g


@pytest.fixture
def labeled_graph():
    g = KnowledgeGraph()
    persons = pd.DataFrame([{"id": 1, "title": "Alice"}, {"id": 2, "title": "Bob"}])
    orgs = pd.DataFrame([{"id": 3, "title": "Acme"}, {"id": 4, "title": "Ghi"}])
    g.add_nodes(persons, "Person", "id", "title")
    g.add_nodes(orgs, "Org", "id", "title")
    return g


# ── HAVING ────────────────────────────────────────────────────────────────


class TestHavingAggregate:
    def test_having_with_aggregate_expression(self, people_graph):
        """HAVING count(m) > 1 should match aliased count(m) AS c."""
        rows = people_graph.cypher("MATCH (n:Person)-[:R]->(m) RETURN n.id AS nid, count(m) AS c HAVING count(m) > 1")
        assert len(rows) == 1
        assert rows[0]["nid"] == 1
        assert rows[0]["c"] == 3

    def test_having_with_alias(self, people_graph):
        """HAVING c > 1 — alias form already worked, keep as baseline."""
        rows = people_graph.cypher("MATCH (n:Person)-[:R]->(m) RETURN n.id AS nid, count(m) AS c HAVING c > 1")
        assert len(rows) == 1
        assert rows[0]["nid"] == 1

    def test_having_no_alias(self, people_graph):
        """HAVING count(m) > 1 with no alias on the RETURN item."""
        rows = people_graph.cypher("MATCH (n:Person)-[:R]->(m) RETURN n.id AS nid, count(m) HAVING count(m) > 1")
        assert len(rows) == 1
        assert rows[0]["nid"] == 1

    def test_having_distinct(self, people_graph):
        """HAVING over count(DISTINCT ...)."""
        rows = people_graph.cypher(
            "MATCH (n:Person)-[:R]->(m) RETURN n.id AS nid, count(DISTINCT m) AS c HAVING count(DISTINCT m) > 1"
        )
        assert len(rows) == 1

    def test_having_filters_all_out(self, people_graph):
        """A HAVING that rejects every group should return empty, not error."""
        rows = people_graph.cypher("MATCH (n:Person)-[:R]->(m) RETURN n.id AS nid, count(m) AS c HAVING count(m) > 100")
        assert len(rows) == 0

    def test_having_over_all_rows(self, people_graph):
        """HAVING on a single-group aggregate (no GROUP BY)."""
        rows = people_graph.cypher("MATCH (n:Person)-[:R]->(m) RETURN count(m) AS total HAVING count(m) > 0")
        assert len(rows) == 1
        assert rows[0]["total"] == 4


# ── WHERE n:Label ─────────────────────────────────────────────────────────


class TestWhereLabel:
    def test_simple_label_match(self, labeled_graph):
        rows = labeled_graph.cypher("MATCH (n) WHERE n:Person RETURN count(n) AS c")
        assert rows[0]["c"] == 2

    def test_negated_label(self, labeled_graph):
        rows = labeled_graph.cypher("MATCH (n) WHERE NOT n:Person RETURN count(n) AS c")
        assert rows[0]["c"] == 2  # Orgs only

    def test_or_labels(self, labeled_graph):
        rows = labeled_graph.cypher("MATCH (n) WHERE n:Person OR n:Org RETURN count(n) AS c")
        assert rows[0]["c"] == 4

    def test_and_property(self, labeled_graph):
        rows = labeled_graph.cypher("MATCH (n) WHERE n:Person AND n.id = 1 RETURN n.id AS nid")
        assert len(rows) == 1
        assert rows[0]["nid"] == 1

    def test_nonexistent_label_returns_zero(self, labeled_graph):
        rows = labeled_graph.cypher("MATCH (n) WHERE n:Missing RETURN count(n) AS c")
        assert rows[0]["c"] == 0


# ── rand() / random() ─────────────────────────────────────────────────────


class TestRand:
    def test_rand_distinct_per_row(self):
        """rand() across many rows should produce distinct values.

        The previous SystemTime-per-call seeding could return identical
        values when two calls hit the same nanosecond.
        """
        g = KnowledgeGraph()
        df = pd.DataFrame([{"id": i} for i in range(500)])
        g.add_nodes(df, "N", "id")
        rows = g.cypher("MATCH (n:N) RETURN rand() AS r")
        assert len(rows) == 500
        values = [r["r"] for r in rows]
        assert all(0.0 <= v <= 1.0 for v in values)
        # Allow for a handful of birthday-paradox collisions but require
        # overwhelming uniqueness — prior behavior could produce many duplicates.
        assert len(set(values)) >= 495

    def test_rand_is_row_dependent(self):
        """rand() inside a WHERE should NOT be constant-folded away."""
        g = KnowledgeGraph()
        df = pd.DataFrame([{"id": i} for i in range(100)])
        g.add_nodes(df, "N", "id")
        rows = g.cypher("MATCH (n:N) WHERE rand() < 0.5 RETURN n.id")
        # Probabilistic: expect roughly half, very loose bounds.
        assert 10 <= len(rows) <= 90
