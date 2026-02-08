"""Performance benchmark tests. Run with: pytest -m benchmark"""

import pytest
import time
import pandas as pd
from rusty_graph import KnowledgeGraph


pytestmark = pytest.mark.benchmark


class TestNodeLoadPerformance:
    @pytest.mark.parametrize("count", [100, 1000, 10000])
    def test_bulk_node_loading(self, count):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': list(range(count)),
            'name': [f'Node_{i}' for i in range(count)],
            'value': list(range(count)),
        })
        start = time.time()
        graph.add_nodes(df, 'Item', 'id', 'name')
        elapsed = time.time() - start
        assert graph.type_filter('Item').node_count() == count
        # Just record timing, no hard threshold
        print(f"  {count} nodes: {elapsed*1000:.1f}ms")


class TestFilterPerformance:
    def test_filter_performance(self, large_graph):
        start = time.time()
        result = large_graph.type_filter('Item').filter({'category': 'Cat_0'})
        elapsed = time.time() - start
        assert result.node_count() == 200
        print(f"  Filter 10k nodes: {elapsed*1000:.1f}ms")

    def test_indexed_filter_performance(self, large_graph):
        large_graph.create_index('Item', 'category')
        start = time.time()
        result = large_graph.type_filter('Item').filter({'category': 'Cat_0'})
        elapsed = time.time() - start
        assert result.node_count() == 200
        print(f"  Indexed filter 10k nodes: {elapsed*1000:.1f}ms")


class TestLightweightMethods:
    def test_node_count_vs_get_nodes(self, large_graph):
        start = time.time()
        count = large_graph.type_filter('Item').node_count()
        count_time = time.time() - start

        start = time.time()
        nodes = large_graph.type_filter('Item').get_nodes()
        get_time = time.time() - start

        assert count == len(nodes) == 10000
        # node_count should be significantly faster
        print(f"  node_count: {count_time*1000:.1f}ms, get_nodes: {get_time*1000:.1f}ms")


class TestPatternMatchPerformance:
    def test_pattern_match_scale(self):
        graph = KnowledgeGraph()
        n = 100
        people = pd.DataFrame({
            'id': list(range(n)),
            'name': [f'Person_{i}' for i in range(n)],
        })
        graph.add_nodes(people, 'Person', 'id', 'name')

        edges = []
        for i in range(n):
            for j in range(i + 1, min(i + 3, n)):
                edges.append({'from_id': i, 'to_id': j})
        graph.add_connections(pd.DataFrame(edges), 'KNOWS', 'Person', 'from_id',
                              'Person', 'to_id')

        start = time.time()
        results = graph.match_pattern('(a:Person)-[:KNOWS]->(b:Person)', max_matches=200)
        elapsed = time.time() - start
        assert len(results) > 0
        print(f"  Pattern match {n} people: {elapsed*1000:.1f}ms, {len(results)} matches")


def _build_cypher_bench_graph(n_people=100, n_companies=20):
    """Build a graph with people, companies, KNOWS and WORKS_AT edges."""
    graph = KnowledgeGraph()
    people = pd.DataFrame({
        'id': list(range(n_people)),
        'name': [f'Person_{i}' for i in range(n_people)],
        'age': [20 + (i % 50) for i in range(n_people)],
        'city': [f'City_{i % 10}' for i in range(n_people)],
        'salary': [30000 + i * 100 for i in range(n_people)],
    })
    graph.add_nodes(people, 'Person', 'id', 'name')

    companies = pd.DataFrame({
        'id': list(range(n_companies)),
        'name': [f'Company_{i}' for i in range(n_companies)],
    })
    graph.add_nodes(companies, 'Company', 'id', 'name')

    # KNOWS: each person knows 2-3 others
    knows = []
    for i in range(n_people):
        for j in range(i + 1, min(i + 3, n_people)):
            knows.append({'from_id': i, 'to_id': j})
    graph.add_connections(pd.DataFrame(knows), 'KNOWS', 'Person', 'from_id',
                          'Person', 'to_id')

    # WORKS_AT: each person works at one company
    works = pd.DataFrame({
        'person_id': list(range(n_people)),
        'company_id': [i % n_companies for i in range(n_people)],
    })
    graph.add_connections(works, 'WORKS_AT', 'Person', 'person_id',
                          'Company', 'company_id')

    return graph


class TestCypherPerformance:
    """Cypher query benchmarks at different complexity levels."""

    def test_simple_match(self):
        """MATCH (n:Person) RETURN n.title — simple node scan."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher("MATCH (n:Person) RETURN n.title")
        elapsed = time.time() - start
        assert len(result['rows']) == 500
        print(f"  Cypher simple match 500 nodes: {elapsed*1000:.1f}ms")

    def test_where_filter(self):
        """MATCH + WHERE — filter with predicate."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher(
            "MATCH (n:Person) WHERE n.age > 40 RETURN n.title, n.age"
        )
        elapsed = time.time() - start
        assert len(result['rows']) > 0
        print(f"  Cypher WHERE filter 500 nodes: {elapsed*1000:.1f}ms, {len(result['rows'])} rows")

    def test_edge_traversal(self):
        """MATCH (a)-[:KNOWS]->(b) — single-hop edge pattern."""
        graph = _build_cypher_bench_graph(200)
        start = time.time()
        result = graph.cypher(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title, b.title"
        )
        elapsed = time.time() - start
        assert len(result['rows']) > 0
        print(f"  Cypher edge traversal 200 nodes: {elapsed*1000:.1f}ms, {len(result['rows'])} rows")

    def test_cross_type_join(self):
        """MATCH (p:Person)-[:WORKS_AT]->(c:Company) — cross-type join."""
        graph = _build_cypher_bench_graph(500, 50)
        start = time.time()
        result = graph.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.title, c.title"
        )
        elapsed = time.time() - start
        assert len(result['rows']) == 500
        print(f"  Cypher cross-type join 500 people -> 50 companies: {elapsed*1000:.1f}ms")

    def test_aggregation(self):
        """GROUP BY + COUNT — aggregation query."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher(
            "MATCH (n:Person) RETURN n.city AS city, count(*) AS cnt, avg(n.salary) AS avg_salary"
        )
        elapsed = time.time() - start
        assert len(result['rows']) > 0
        print(f"  Cypher aggregation 500 nodes: {elapsed*1000:.1f}ms, {len(result['rows'])} groups")

    def test_order_limit(self):
        """ORDER BY + LIMIT — sorting and pagination."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher(
            "MATCH (n:Person) RETURN n.title, n.salary ORDER BY n.salary DESC LIMIT 10"
        )
        elapsed = time.time() - start
        assert len(result['rows']) == 10
        salaries = [r['n.salary'] for r in result['rows']]
        assert salaries == sorted(salaries, reverse=True)
        print(f"  Cypher ORDER BY + LIMIT 500 nodes: {elapsed*1000:.1f}ms")

    def test_complex_query(self):
        """Multi-hop + WHERE + aggregation — realistic complex query."""
        graph = _build_cypher_bench_graph(200, 20)
        start = time.time()
        result = graph.cypher("""
            MATCH (p:Person)-[:WORKS_AT]->(c:Company)
            WHERE p.age > 30
            RETURN c.title AS company, count(*) AS headcount, avg(p.salary) AS avg_salary
            ORDER BY headcount DESC
        """)
        elapsed = time.time() - start
        assert len(result['rows']) > 0
        print(f"  Cypher complex query 200 nodes: {elapsed*1000:.1f}ms, {len(result['rows'])} rows")

    def test_cypher_vs_fluent_api(self):
        """Compare Cypher vs fluent API for the same operation."""
        graph = _build_cypher_bench_graph(500)

        # Cypher approach
        start = time.time()
        cypher_result = graph.cypher(
            "MATCH (n:Person) WHERE n.age > 40 RETURN n.title"
        )
        cypher_time = time.time() - start

        # Fluent API approach
        start = time.time()
        fluent_result = graph.type_filter('Person').filter({'age': {'>': 40}}).get_nodes()
        fluent_time = time.time() - start

        assert len(cypher_result['rows']) == len(fluent_result)
        print(f"  Cypher: {cypher_time*1000:.1f}ms vs Fluent API: {fluent_time*1000:.1f}ms "
              f"({len(cypher_result['rows'])} results)")

    def test_cypher_indexed_vs_unindexed(self):
        """Compare Cypher WHERE with and without property index."""
        graph = _build_cypher_bench_graph(500)
        query = "MATCH (n:Person) WHERE n.city = 'City_0' RETURN n.title"

        # Without index
        start = time.time()
        result_no_idx = graph.cypher(query)
        no_idx_time = time.time() - start

        # Create index and re-run
        graph.create_index('Person', 'city')
        start = time.time()
        result_idx = graph.cypher(query)
        idx_time = time.time() - start

        assert len(result_no_idx['rows']) == len(result_idx['rows'])
        assert len(result_idx['rows']) == 50  # 500 people / 10 cities
        print(f"  Cypher WHERE no index: {no_idx_time*1000:.1f}ms, "
              f"with index: {idx_time*1000:.1f}ms "
              f"({len(result_idx['rows'])} results)")

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_cypher_scaling(self, n):
        """Measure how Cypher scales with graph size."""
        graph = _build_cypher_bench_graph(n)
        start = time.time()
        result = graph.cypher(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN a.title, b.title"
        )
        elapsed = time.time() - start
        assert len(result['rows']) > 0
        print(f"  Cypher scaling {n} nodes: {elapsed*1000:.1f}ms, {len(result['rows'])} rows")
