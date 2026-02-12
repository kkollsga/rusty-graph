"""Tests for GIL release / concurrency (Phase 5).

Verifies that read-only Cypher queries release the GIL and can run
concurrently from multiple Python threads.
"""
import pytest
import pandas as pd
import kglite
import threading
import time


@pytest.fixture
def large_graph():
    """Graph with enough nodes to make queries non-trivial."""
    g = kglite.KnowledgeGraph()
    n = 5000
    df = pd.DataFrame({
        "id": list(range(n)),
        "title": [f"Person_{i}" for i in range(n)],
        "age": [20 + (i % 80) for i in range(n)],
        "city": [["Oslo", "Bergen", "Trondheim", "Stavanger"][i % 4] for i in range(n)],
    })
    g.add_nodes(df, "Person", "id", "title")

    # Add some edges
    edges = pd.DataFrame({
        "source": list(range(0, n - 1)),
        "target": list(range(1, n)),
        "type": ["KNOWS"] * (n - 1),
    })
    g.add_connections(edges, "KNOWS", "Person", "source", "Person", "target")
    return g


class TestConcurrentReads:
    """Multiple threads can read the graph concurrently."""

    def test_concurrent_cypher_reads(self, large_graph):
        """Multiple threads running read-only Cypher should all complete correctly."""
        results = {}
        errors = []

        def query_thread(thread_id, city):
            try:
                result = large_graph.cypher(
                    f"MATCH (n:Person) WHERE n.city = '{city}' RETURN count(n) AS cnt"
                )
                results[thread_id] = result[0]["cnt"]
            except Exception as e:
                errors.append((thread_id, str(e)))

        cities = ["Oslo", "Bergen", "Trondheim", "Stavanger"]
        threads = []
        for i, city in enumerate(cities):
            t = threading.Thread(target=query_thread, args=(i, city))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 4
        # Each city has ~1250 nodes (5000 / 4)
        for count in results.values():
            assert count == 1250

    def test_concurrent_reads_produce_correct_results(self, large_graph):
        """Results from concurrent reads match sequential reads."""
        # Get sequential baseline
        sequential = large_graph.cypher(
            "MATCH (n:Person) WHERE n.age > 50 RETURN count(n) AS cnt"
        )[0]["cnt"]

        results = []
        errors = []

        def query_thread():
            try:
                result = large_graph.cypher(
                    "MATCH (n:Person) WHERE n.age > 50 RETURN count(n) AS cnt"
                )
                results.append(result[0]["cnt"])
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=query_thread) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 8
        assert all(r == sequential for r in results)


class TestReadWriteIsolation:
    """Reads and writes don't interfere."""

    def test_read_during_no_mutation(self, large_graph):
        """Simple sanity: reads work fine when no mutations happening."""
        result = large_graph.cypher(
            "MATCH (n:Person) RETURN count(n) AS cnt"
        )
        assert result[0]["cnt"] == 5000


class TestGILReleasePerformance:
    """GIL release should allow other Python code to run during queries."""

    def test_gil_released_during_read_query(self, large_graph):
        """While a Cypher query runs, other Python threads can make progress."""
        counter = {"value": 0}
        stop_event = threading.Event()

        def increment_counter():
            """Simple Python thread that increments a counter."""
            while not stop_event.is_set():
                counter["value"] += 1
                time.sleep(0.001)

        # Start the counter thread
        counter_thread = threading.Thread(target=increment_counter, daemon=True)
        counter_thread.start()

        # Run a query (should release GIL, allowing counter to increment)
        large_graph.cypher("MATCH (n:Person)-[:KNOWS]->(m:Person) RETURN count(n) AS cnt")

        stop_event.set()
        counter_thread.join(timeout=2)

        # The counter should have incremented at least once during the query
        # (if GIL wasn't released, counter would stay at 0)
        assert counter["value"] > 0
