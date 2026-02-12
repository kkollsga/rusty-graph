"""Tests for multi-statement transactions (Phase 3).

Transactions provide snapshot isolation: mutations are applied to a working
copy and only committed to the original graph on tx.commit(). On rollback
(or exception), no changes are applied.
"""
import pytest
import kglite
import pandas as pd


@pytest.fixture
def graph():
    """Graph with a few Person nodes."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["Alice", "Bob", "Charlie"],
        "age": [30, 25, 35],
    })
    g.add_nodes(df, "Person", "id", "title")
    return g


class TestTransactionCommit:
    """Committed transactions apply changes to the original graph."""

    def test_commit_creates_nodes(self, graph):
        """Nodes created in a transaction should appear after commit."""
        tx = graph.begin()
        tx.cypher("CREATE (n:Person {id: 4, title: 'Dave', age: 40})")
        tx.commit()

        result = graph.cypher("MATCH (n:Person) WHERE n.title = 'Dave' RETURN n.age")
        assert len(result) == 1
        assert result[0]["n.age"] == 40

    def test_commit_deletes_nodes(self, graph):
        """Nodes deleted in a transaction should be gone after commit."""
        tx = graph.begin()
        tx.cypher("MATCH (n:Person) WHERE n.title = 'Bob' DETACH DELETE n")
        tx.commit()

        result = graph.cypher("MATCH (n:Person) RETURN n.title ORDER BY n.title")
        titles = [r["n.title"] for r in result]
        assert titles == ["Alice", "Charlie"]

    def test_commit_sets_properties(self, graph):
        """Property changes in a transaction should persist after commit."""
        tx = graph.begin()
        tx.cypher("MATCH (n:Person) WHERE n.title = 'Alice' SET n.age = 31")
        tx.commit()

        result = graph.cypher("MATCH (n:Person) WHERE n.title = 'Alice' RETURN n.age")
        assert result[0]["n.age"] == 31

    def test_multi_statement_commit(self, graph):
        """Multiple statements in one transaction all apply on commit."""
        tx = graph.begin()
        tx.cypher("CREATE (n:Person {id: 4, title: 'Dave', age: 40})")
        tx.cypher("CREATE (n:Person {id: 5, title: 'Eve', age: 28})")
        tx.cypher("MATCH (n:Person) WHERE n.title = 'Bob' SET n.age = 26")
        tx.commit()

        result = graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        assert result[0]["cnt"] == 5

        result = graph.cypher("MATCH (n:Person) WHERE n.title = 'Bob' RETURN n.age")
        assert result[0]["n.age"] == 26


class TestTransactionRollback:
    """Rolled-back transactions discard all changes."""

    def test_rollback_discards_creates(self, graph):
        """Nodes created in a rolled-back transaction should not appear."""
        tx = graph.begin()
        tx.cypher("CREATE (n:Person {id: 4, title: 'Dave', age: 40})")
        tx.rollback()

        result = graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        assert result[0]["cnt"] == 3

    def test_rollback_discards_deletes(self, graph):
        """Nodes deleted in a rolled-back transaction should still exist."""
        tx = graph.begin()
        tx.cypher("MATCH (n:Person) WHERE n.title = 'Bob' DETACH DELETE n")
        tx.rollback()

        result = graph.cypher("MATCH (n:Person) WHERE n.title = 'Bob' RETURN n.age")
        assert len(result) == 1
        assert result[0]["n.age"] == 25

    def test_rollback_discards_sets(self, graph):
        """Property changes in a rolled-back transaction should not persist."""
        tx = graph.begin()
        tx.cypher("MATCH (n:Person) WHERE n.title = 'Alice' SET n.age = 99")
        tx.rollback()

        result = graph.cypher("MATCH (n:Person) WHERE n.title = 'Alice' RETURN n.age")
        assert result[0]["n.age"] == 30

    def test_drop_without_commit_discards(self, graph):
        """Dropping a transaction without commit should discard changes."""
        tx = graph.begin()
        tx.cypher("CREATE (n:Person {id: 4, title: 'Dave', age: 40})")
        del tx  # Dropped without commit

        result = graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        assert result[0]["cnt"] == 3


class TestTransactionIsolation:
    """Changes in a transaction are not visible in the original graph until commit."""

    def test_original_graph_unchanged_before_commit(self, graph):
        """Original graph should not see uncommitted changes."""
        tx = graph.begin()
        tx.cypher("CREATE (n:Person {id: 4, title: 'Dave', age: 40})")

        # Original graph should still have 3 nodes
        result = graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        assert result[0]["cnt"] == 3

        # Working copy should have 4
        result = tx.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        assert result[0]["cnt"] == 4

        tx.rollback()

    def test_read_within_transaction_sees_changes(self, graph):
        """Reads within a transaction should see uncommitted mutations."""
        tx = graph.begin()
        tx.cypher("CREATE (n:Person {id: 4, title: 'Dave', age: 40})")
        tx.cypher("MATCH (n:Person) WHERE n.title = 'Alice' SET n.age = 99")

        result = tx.cypher("MATCH (n:Person) WHERE n.title = 'Dave' RETURN n.age")
        assert len(result) == 1
        assert result[0]["n.age"] == 40

        result = tx.cypher("MATCH (n:Person) WHERE n.title = 'Alice' RETURN n.age")
        assert result[0]["n.age"] == 99

        tx.rollback()


class TestContextManager:
    """Context manager auto-commits on success and auto-rollbacks on exception."""

    def test_context_manager_auto_commits(self, graph):
        """Exiting a with-block without exception should auto-commit."""
        with graph.begin() as tx:
            tx.cypher("CREATE (n:Person {id: 4, title: 'Dave', age: 40})")

        result = graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        assert result[0]["cnt"] == 4

    def test_context_manager_auto_rollbacks_on_exception(self, graph):
        """Exiting a with-block with an exception should auto-rollback."""
        with pytest.raises(ValueError):
            with graph.begin() as tx:
                tx.cypher("CREATE (n:Person {id: 4, title: 'Dave', age: 40})")
                raise ValueError("Simulated error")

        result = graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        assert result[0]["cnt"] == 3

    def test_context_manager_rollback_on_cypher_error(self, graph):
        """Cypher error in context manager should roll back all changes."""
        with pytest.raises(Exception):
            with graph.begin() as tx:
                tx.cypher("CREATE (n:Person {id: 4, title: 'Dave', age: 40})")
                tx.cypher("INVALID CYPHER SYNTAX !!!")

        result = graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        assert result[0]["cnt"] == 3


class TestTransactionErrors:
    """Error handling for invalid transaction operations."""

    def test_cypher_after_commit_raises(self, graph):
        """Using a committed transaction should raise an error."""
        tx = graph.begin()
        tx.commit()
        with pytest.raises(Exception, match="already committed"):
            tx.cypher("MATCH (n) RETURN n")

    def test_cypher_after_rollback_raises(self, graph):
        """Using a rolled-back transaction should raise an error."""
        tx = graph.begin()
        tx.rollback()
        with pytest.raises(Exception, match="already committed"):
            tx.cypher("MATCH (n) RETURN n")

    def test_double_commit_raises(self, graph):
        """Committing twice should raise an error."""
        tx = graph.begin()
        tx.commit()
        with pytest.raises(Exception, match="already committed"):
            tx.commit()

    def test_double_rollback_raises(self, graph):
        """Rolling back twice should raise an error."""
        tx = graph.begin()
        tx.rollback()
        with pytest.raises(Exception, match="already committed"):
            tx.rollback()


class TestTransactionWithParams:
    """Transactions should support parameterized queries."""

    def test_params_in_transaction(self, graph):
        """Parameters should work within transactions."""
        tx = graph.begin()
        tx.cypher(
            "CREATE (n:Person {id: $id, title: $name, age: $age})",
            params={"id": 4, "name": "Dave", "age": 40},
        )
        tx.commit()

        result = graph.cypher("MATCH (n:Person) WHERE n.title = 'Dave' RETURN n.age")
        assert len(result) == 1
        assert result[0]["n.age"] == 40
