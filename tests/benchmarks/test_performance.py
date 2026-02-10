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
        assert len(result) == 500
        print(f"  Cypher simple match 500 nodes: {elapsed*1000:.1f}ms")

    def test_where_filter(self):
        """MATCH + WHERE — filter with predicate."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher(
            "MATCH (n:Person) WHERE n.age > 40 RETURN n.title, n.age"
        )
        elapsed = time.time() - start
        assert len(result) > 0
        print(f"  Cypher WHERE filter 500 nodes: {elapsed*1000:.1f}ms, {len(result)} rows")

    def test_edge_traversal(self):
        """MATCH (a)-[:KNOWS]->(b) — single-hop edge pattern."""
        graph = _build_cypher_bench_graph(200)
        start = time.time()
        result = graph.cypher(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title, b.title"
        )
        elapsed = time.time() - start
        assert len(result) > 0
        print(f"  Cypher edge traversal 200 nodes: {elapsed*1000:.1f}ms, {len(result)} rows")

    def test_cross_type_join(self):
        """MATCH (p:Person)-[:WORKS_AT]->(c:Company) — cross-type join."""
        graph = _build_cypher_bench_graph(500, 50)
        start = time.time()
        result = graph.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.title, c.title"
        )
        elapsed = time.time() - start
        assert len(result) == 500
        print(f"  Cypher cross-type join 500 people -> 50 companies: {elapsed*1000:.1f}ms")

    def test_aggregation(self):
        """GROUP BY + COUNT — aggregation query."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher(
            "MATCH (n:Person) RETURN n.city AS city, count(*) AS cnt, avg(n.salary) AS avg_salary"
        )
        elapsed = time.time() - start
        assert len(result) > 0
        print(f"  Cypher aggregation 500 nodes: {elapsed*1000:.1f}ms, {len(result)} groups")

    def test_order_limit(self):
        """ORDER BY + LIMIT — sorting and pagination."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher(
            "MATCH (n:Person) RETURN n.title, n.salary ORDER BY n.salary DESC LIMIT 10"
        )
        elapsed = time.time() - start
        assert len(result) == 10
        salaries = [r['n.salary'] for r in result]
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
        assert len(result) > 0
        print(f"  Cypher complex query 200 nodes: {elapsed*1000:.1f}ms, {len(result)} rows")

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

        assert len(cypher_result) == len(fluent_result)
        print(f"  Cypher: {cypher_time*1000:.1f}ms vs Fluent API: {fluent_time*1000:.1f}ms "
              f"({len(cypher_result)} results)")

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

        assert len(result_no_idx) == len(result_idx)
        assert len(result_idx) == 50  # 500 people / 10 cities
        print(f"  Cypher WHERE no index: {no_idx_time*1000:.1f}ms, "
              f"with index: {idx_time*1000:.1f}ms "
              f"({len(result_idx)} results)")

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_cypher_scaling(self, n):
        """Measure how Cypher scales with graph size."""
        graph = _build_cypher_bench_graph(n)
        start = time.time()
        result = graph.cypher(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN a.title, b.title"
        )
        elapsed = time.time() - start
        assert len(result) > 0
        print(f"  Cypher scaling {n} nodes: {elapsed*1000:.1f}ms, {len(result)} rows")


class TestCypherMutationPerformance:
    """Benchmarks for DELETE, REMOVE, and MERGE operations."""

    # ------------------------------------------------------------------ DELETE
    def test_delete_single_node(self):
        """DETACH DELETE a single node — baseline cost."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Person_0'
            DETACH DELETE n
        """)
        elapsed = time.time() - start
        assert result['stats']['nodes_deleted'] == 1
        print(f"  Delete single node: {elapsed*1000:.2f}ms")

    def test_delete_relationship_only(self):
        """DELETE r — remove edges without touching nodes."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher("""
            MATCH (a:Person)-[r:KNOWS]->(b:Person)
            WHERE a.name = 'Person_0'
            DELETE r
        """)
        elapsed = time.time() - start
        assert result['stats']['relationships_deleted'] > 0
        print(f"  Delete relationships for 1 node: {elapsed*1000:.2f}ms, "
              f"{result['stats']['relationships_deleted']} edges removed")

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_detach_delete_all_scaling(self, n):
        """DETACH DELETE all nodes — measures index cleanup scaling."""
        graph = _build_cypher_bench_graph(n)
        start = time.time()
        result = graph.cypher("MATCH (n:Person) DETACH DELETE n")
        elapsed = time.time() - start
        assert result['stats']['nodes_deleted'] == n
        print(f"  Detach delete all {n} people: {elapsed*1000:.1f}ms, "
              f"{result['stats']['relationships_deleted']} edges")

    def test_delete_subset(self):
        """DETACH DELETE a filtered subset — common real-world pattern."""
        graph = _build_cypher_bench_graph(1000)
        start = time.time()
        result = graph.cypher("""
            MATCH (n:Person) WHERE n.age > 60
            DETACH DELETE n
        """)
        elapsed = time.time() - start
        deleted = result['stats']['nodes_deleted']
        print(f"  Delete subset (age>60): {elapsed*1000:.1f}ms, "
              f"{deleted} nodes, {result['stats']['relationships_deleted']} edges")

    # ------------------------------------------------------------------ REMOVE
    def test_remove_single_property(self):
        """REMOVE n.prop — single property from one node."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Person_0'
            REMOVE n.salary
        """)
        elapsed = time.time() - start
        assert result['stats']['properties_removed'] == 1
        print(f"  Remove 1 property from 1 node: {elapsed*1000:.2f}ms")

    def test_remove_property_all_nodes(self):
        """REMOVE n.prop across all nodes — bulk property removal."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher("MATCH (n:Person) REMOVE n.salary")
        elapsed = time.time() - start
        assert result['stats']['properties_removed'] == 500
        print(f"  Remove salary from 500 nodes: {elapsed*1000:.1f}ms")

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_remove_scaling(self, n):
        """REMOVE property scaling across graph sizes."""
        graph = _build_cypher_bench_graph(n)
        start = time.time()
        result = graph.cypher("MATCH (n:Person) REMOVE n.city, n.salary")
        elapsed = time.time() - start
        assert result['stats']['properties_removed'] == n * 2
        print(f"  Remove 2 props from {n} nodes: {elapsed*1000:.1f}ms, "
              f"{result['stats']['properties_removed']} removed")

    # ------------------------------------------------------------------ MERGE
    def test_merge_creates_single_node(self):
        """MERGE creating one new node — CREATE path."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher(
            "MERGE (n:Person {name: 'NewPerson'}) ON CREATE SET n.age = 99"
        )
        elapsed = time.time() - start
        assert result['stats']['nodes_created'] == 1
        print(f"  Merge create 1 node: {elapsed*1000:.2f}ms")

    def test_merge_matches_existing_node(self):
        """MERGE matching an existing node — MATCH path."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher(
            "MERGE (n:Person {name: 'Person_0'}) ON MATCH SET n.visits = 1"
        )
        elapsed = time.time() - start
        assert result['stats']['nodes_created'] == 0
        print(f"  Merge match existing (500 nodes scanned): {elapsed*1000:.2f}ms")

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_merge_match_scaling(self, n):
        """MERGE matching existing — scales with type_indices size."""
        graph = _build_cypher_bench_graph(n)
        # Match the last person — worst case linear scan
        target = f'Person_{n-1}'
        start = time.time()
        result = graph.cypher(
            f"MERGE (n:Person {{name: '{target}'}}) ON MATCH SET n.visits = 1"
        )
        elapsed = time.time() - start
        assert result['stats']['nodes_created'] == 0
        print(f"  Merge match last of {n} nodes: {elapsed*1000:.2f}ms")

    def test_merge_relationship_exists(self):
        """MERGE edge that already exists — edge scan path."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher("""
            MATCH (a:Person {name: 'Person_0'}), (b:Person {name: 'Person_1'})
            MERGE (a)-[r:KNOWS]->(b)
        """)
        elapsed = time.time() - start
        assert result['stats']['relationships_created'] == 0
        print(f"  Merge existing edge: {elapsed*1000:.2f}ms")

    def test_merge_relationship_new(self):
        """MERGE edge that doesn't exist — CREATE path."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        result = graph.cypher("""
            MATCH (a:Person {name: 'Person_0'}), (b:Person {name: 'Person_99'})
            MERGE (a)-[r:FRIENDS]->(b)
        """)
        elapsed = time.time() - start
        assert result['stats']['relationships_created'] == 1
        print(f"  Merge create new edge: {elapsed*1000:.2f}ms")

    def test_merge_batch_create_nodes(self):
        """UNWIND + MERGE — batch creating nodes that don't exist."""
        graph = _build_cypher_bench_graph(100)
        start = time.time()
        for i in range(100):
            graph.cypher(
                f"MERGE (n:City {{name: 'City_{i}'}}) ON CREATE SET n.pop = {i * 1000}"
            )
        elapsed = time.time() - start
        # Verify all created
        check = graph.cypher("MATCH (n:City) RETURN count(*) AS cnt")
        assert check['rows'][0]['cnt'] == 100
        print(f"  Merge-create 100 cities (sequential): {elapsed*1000:.1f}ms "
              f"({elapsed*10:.2f}ms/op)")

    def test_merge_batch_match_nodes(self):
        """Sequential MERGE matching existing nodes — amortized lookup cost."""
        graph = _build_cypher_bench_graph(500)
        start = time.time()
        for i in range(100):
            graph.cypher(
                f"MERGE (n:Person {{name: 'Person_{i}'}}) ON MATCH SET n.visited = 1"
            )
        elapsed = time.time() - start
        print(f"  Merge-match 100 of 500 people (sequential): {elapsed*1000:.1f}ms "
              f"({elapsed*10:.2f}ms/op)")


class TestCypherMutationCombined:
    """Benchmarks for realistic mutation workflows."""

    def test_create_set_delete_lifecycle(self):
        """Full lifecycle: CREATE -> SET -> DELETE."""
        graph = _build_cypher_bench_graph(200)
        times = {}

        # CREATE
        start = time.time()
        graph.cypher("CREATE (n:TempNode {name: 'temp', val: 42})")
        times['create'] = time.time() - start

        # SET
        start = time.time()
        graph.cypher("""
            MATCH (n:TempNode) WHERE n.name = 'temp'
            SET n.val = 100, n.extra = 'data'
        """)
        times['set'] = time.time() - start

        # DELETE
        start = time.time()
        graph.cypher("MATCH (n:TempNode) DETACH DELETE n")
        times['delete'] = time.time() - start

        for op, t in times.items():
            print(f"  {op}: {t*1000:.2f}ms")

    def test_merge_then_remove(self):
        """MERGE + REMOVE workflow — idempotent upsert then cleanup."""
        graph = _build_cypher_bench_graph(500)

        start = time.time()
        graph.cypher(
            "MERGE (n:Person {name: 'Person_0'}) ON MATCH SET n.temp_flag = 'processing'"
        )
        merge_time = time.time() - start

        start = time.time()
        graph.cypher("""
            MATCH (n:Person) WHERE n.name = 'Person_0'
            REMOVE n.temp_flag
        """)
        remove_time = time.time() - start

        print(f"  Merge+SET: {merge_time*1000:.2f}ms, REMOVE: {remove_time*1000:.2f}ms")

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_full_mutation_pipeline(self, n):
        """MATCH -> CREATE edges -> DELETE old edges — graph restructuring."""
        graph = _build_cypher_bench_graph(n, n // 5)

        # Delete all WORKS_AT edges
        start = time.time()
        result = graph.cypher("MATCH ()-[r:WORKS_AT]->() DELETE r")
        delete_time = time.time() - start
        deleted = result['stats']['relationships_deleted']

        # Re-create edges with different assignments
        start = time.time()
        for i in range(min(n, 100)):
            company = (i + 1) % (n // 5)
            graph.cypher(f"""
                MATCH (p:Person {{name: 'Person_{i}'}}), (c:Company {{name: 'Company_{company}'}})
                CREATE (p)-[:WORKS_AT]->(c)
            """)
        create_time = time.time() - start
        created = min(n, 100)

        print(f"  Pipeline {n} nodes: delete {deleted} edges={delete_time*1000:.1f}ms, "
              f"create {created} edges={create_time*1000:.1f}ms")
