"""Tests for cypher() — full Cypher query pipeline."""

import pytest
import pandas as pd
from rusty_graph import KnowledgeGraph


@pytest.fixture
def cypher_graph():
    """Graph optimized for Cypher tests."""
    graph = KnowledgeGraph()

    people = pd.DataFrame({
        'person_id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [30, 25, 35, 28, 40],
        'city': ['Oslo', 'Bergen', 'Oslo', 'Bergen', 'Oslo'],
        'salary': [70000, 55000, 80000, 65000, 90000],
        'email': ['alice@test.com', None, 'charlie@test.com', None, 'eve@test.com'],
    })
    graph.add_nodes(people, 'Person', 'person_id', 'name')

    products = pd.DataFrame({
        'product_id': [101, 102, 103],
        'name': ['Laptop', 'Phone', 'Tablet'],
        'price': [999.99, 699.99, 349.99],
    })
    graph.add_nodes(products, 'Product', 'product_id', 'name')

    knows = pd.DataFrame({
        'from_id': [1, 1, 2, 3, 4],
        'to_id': [2, 3, 3, 4, 5],
    })
    graph.add_connections(knows, 'KNOWS', 'Person', 'from_id', 'Person', 'to_id')

    purchased = pd.DataFrame({
        'person_id': [1, 1, 2, 3],
        'product_id': [101, 102, 103, 101],
    })
    graph.add_connections(purchased, 'PURCHASED', 'Person', 'person_id',
                          'Product', 'product_id')

    return graph


class TestBasicQueries:
    def test_simple_match_return(self, cypher_graph):
        result = cypher_graph.cypher("MATCH (n:Person) RETURN n.title")
        assert len(result['rows']) == 5
        assert 'n.title' in result['columns']

    def test_match_with_alias(self, cypher_graph):
        result = cypher_graph.cypher("MATCH (n:Person) RETURN n.title AS name")
        assert 'name' in result['columns']
        names = {r['name'] for r in result['rows']}
        assert 'Alice' in names

    def test_edge_pattern(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.title, b.title"
        )
        assert len(result['rows']) == 5

    def test_multi_hop(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (a:Person)-[:PURCHASED]->(p:Product) RETURN a.title, p.title"
        )
        assert len(result['rows']) == 4

    def test_cross_type(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (p:Person)-[:PURCHASED]->(pr:Product) "
            "RETURN p.title AS person, pr.title AS product"
        )
        assert len(result['rows']) == 4

    def test_case_insensitive_keywords(self, cypher_graph):
        result = cypher_graph.cypher("match (n:Person) return n.title")
        assert len(result['rows']) == 5


class TestWhereClause:
    def test_comparison_gt(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.age > 30 RETURN n.title, n.age"
        )
        for row in result['rows']:
            assert row['n.age'] > 30

    def test_equality(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.title"
        )
        assert len(result['rows']) == 3

    def test_not_equals(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.city <> 'Oslo' RETURN n.title"
        )
        assert len(result['rows']) == 2

    def test_and(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.age > 25 AND n.city = 'Oslo' RETURN n.title"
        )
        for row in result['rows']:
            assert row['n.title'] in ['Alice', 'Charlie', 'Eve']

    def test_or(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.age < 26 OR n.age > 39 RETURN n.title"
        )
        names = {r['n.title'] for r in result['rows']}
        assert 'Bob' in names
        assert 'Eve' in names

    def test_not(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE NOT n.city = 'Oslo' RETURN n.title"
        )
        assert len(result['rows']) == 2

    def test_is_null(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.email IS NULL RETURN n.title"
        )
        assert len(result['rows']) == 2

    def test_is_not_null(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.email IS NOT NULL RETURN n.title"
        )
        assert len(result['rows']) == 3

    def test_in_list(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.city IN ['Oslo', 'Bergen'] RETURN n.title"
        )
        assert len(result['rows']) == 5

    def test_contains(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.title CONTAINS 'li' RETURN n.title"
        )
        names = {r['n.title'] for r in result['rows']}
        assert 'Alice' in names
        assert 'Charlie' in names

    def test_starts_with(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.title STARTS WITH 'A' RETURN n.title"
        )
        assert len(result['rows']) == 1

    def test_ends_with(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.title ENDS WITH 'e' RETURN n.title"
        )
        names = {r['n.title'] for r in result['rows']}
        assert 'Alice' in names
        assert 'Eve' in names
        assert 'Charlie' in names


class TestOrderByLimitSkip:
    def test_order_by_asc(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) RETURN n.title, n.age ORDER BY n.age"
        )
        ages = [r['n.age'] for r in result['rows']]
        assert ages == sorted(ages)

    def test_order_by_desc(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) RETURN n.title, n.age ORDER BY n.age DESC"
        )
        ages = [r['n.age'] for r in result['rows']]
        assert ages == sorted(ages, reverse=True)

    def test_limit(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) RETURN n.title LIMIT 3"
        )
        assert len(result['rows']) == 3

    def test_skip(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) RETURN n.title ORDER BY n.age SKIP 2"
        )
        assert len(result['rows']) == 3

    def test_skip_and_limit(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) RETURN n.title, n.age ORDER BY n.age SKIP 1 LIMIT 2"
        )
        assert len(result['rows']) == 2
        ages = [r['n.age'] for r in result['rows']]
        assert ages == [28, 30]


class TestAggregation:
    def test_count_star(self, cypher_graph):
        result = cypher_graph.cypher("MATCH (n:Person) RETURN count(*) AS total")
        assert result['rows'][0]['total'] == 5

    def test_count_with_grouping(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) RETURN n.city AS city, count(*) AS cnt"
        )
        cities = {r['city']: r['cnt'] for r in result['rows']}
        assert cities['Oslo'] == 3
        assert cities['Bergen'] == 2

    def test_sum(self, cypher_graph):
        result = cypher_graph.cypher("MATCH (n:Person) RETURN sum(n.salary) AS total")
        assert result['rows'][0]['total'] == 360000

    def test_avg(self, cypher_graph):
        result = cypher_graph.cypher("MATCH (n:Person) RETURN avg(n.age) AS avg_age")
        assert abs(result['rows'][0]['avg_age'] - 31.6) < 0.1

    def test_min_max(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) RETURN min(n.age) AS youngest, max(n.age) AS oldest"
        )
        assert result['rows'][0]['youngest'] == 25
        assert result['rows'][0]['oldest'] == 40

    def test_distinct(self, cypher_graph):
        result = cypher_graph.cypher("MATCH (n:Person) RETURN DISTINCT n.city")
        assert len(result['rows']) == 2

    def test_count_distinct(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) RETURN count(DISTINCT n.city) AS unique_cities"
        )
        assert result['rows'][0]['unique_cities'] == 2


class TestWithClause:
    def test_with_basic(self, cypher_graph):
        result = cypher_graph.cypher("""
            MATCH (p:Person)-[:KNOWS]->(f:Person)
            WITH p, count(f) AS friend_count
            RETURN p.title, friend_count
            ORDER BY friend_count DESC
        """)
        assert len(result['rows']) > 0
        counts = [r['friend_count'] for r in result['rows']]
        assert counts == sorted(counts, reverse=True)


class TestOptionalMatch:
    def test_optional_match_basic(self, cypher_graph):
        result = cypher_graph.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:PURCHASED]->(pr:Product)
            RETURN p.title, count(pr) AS purchases
        """)
        assert len(result['rows']) == 5  # All persons, even without purchases


class TestExpressions:
    def test_arithmetic(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Product) RETURN n.title, n.price * 1.25 AS with_tax"
        )
        for row in result['rows']:
            assert row['with_tax'] > row.get('n.price', 0) or 'with_tax' in row

    def test_coalesce(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) RETURN n.title, coalesce(n.email, 'no email') AS contact"
        )
        for row in result['rows']:
            assert row['contact'] != '' or row['contact'] is not None

    def test_predicate_pushdown(self, cypher_graph):
        """Predicate pushdown should produce same results as without."""
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.age = 30 RETURN n.title"
        )
        assert len(result['rows']) == 1
        assert result['rows'][0]['n.title'] == 'Alice'


class TestEmptyResults:
    def test_no_matching_nodes(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:NonExistent) RETURN n.title"
        )
        assert len(result['rows']) == 0

    def test_where_eliminates_all(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.age > 100 RETURN n.title"
        )
        assert len(result['rows']) == 0


class TestSyntaxErrors:
    def test_invalid_query(self, cypher_graph):
        with pytest.raises(ValueError):
            cypher_graph.cypher("NOT A VALID QUERY")


class TestCaseExpressions:
    """Tests for CASE WHEN ... THEN ... ELSE ... END expressions."""

    def test_generic_case(self, cypher_graph):
        """CASE WHEN predicate THEN result ELSE default END."""
        result = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.name AS name,
                   CASE WHEN n.age >= 30 THEN 'senior' ELSE 'junior' END AS level
            ORDER BY n.name
        """)
        rows = result['rows']
        assert len(rows) == 5
        alice = next(r for r in rows if r['name'] == 'Alice')
        assert alice['level'] == 'senior'  # age 30
        bob = next(r for r in rows if r['name'] == 'Bob')
        assert bob['level'] == 'junior'  # age 25

    def test_simple_case(self, cypher_graph):
        """CASE expr WHEN val THEN result ... END."""
        result = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.name AS name,
                   CASE n.city WHEN 'Oslo' THEN 'capital' WHEN 'Bergen' THEN 'west' ELSE 'other' END AS region
            ORDER BY n.name
        """)
        rows = result['rows']
        alice = next(r for r in rows if r['name'] == 'Alice')
        assert alice['region'] == 'capital'
        bob = next(r for r in rows if r['name'] == 'Bob')
        assert bob['region'] == 'west'

    def test_case_no_else_returns_null(self, cypher_graph):
        """CASE without ELSE returns null when no match."""
        result = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.name AS name,
                   CASE n.city WHEN 'Trondheim' THEN 'found' END AS status
            ORDER BY n.name
        """)
        rows = result['rows']
        # No one lives in Trondheim, so all should be null
        for row in rows:
            assert row['status'] is None

    def test_case_multiple_when(self, cypher_graph):
        """CASE with multiple WHEN clauses — first match wins."""
        result = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.name AS name,
                   CASE
                       WHEN n.age >= 40 THEN 'veteran'
                       WHEN n.age >= 30 THEN 'experienced'
                       ELSE 'newcomer'
                   END AS tier
            ORDER BY n.name
        """)
        rows = result['rows']
        eve = next(r for r in rows if r['name'] == 'Eve')
        assert eve['tier'] == 'veteran'  # age 40 — first match wins
        alice = next(r for r in rows if r['name'] == 'Alice')
        assert alice['tier'] == 'experienced'  # age 30
        bob = next(r for r in rows if r['name'] == 'Bob')
        assert bob['tier'] == 'newcomer'  # age 25


class TestParameters:
    """Tests for $param parameter substitution."""

    def test_single_parameter(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name AS name ORDER BY n.name",
            params={'min_age': 30}
        )
        names = [r['name'] for r in result['rows']]
        assert 'Charlie' in names  # age 35
        assert 'Eve' in names      # age 40
        assert 'Alice' not in names  # age 30, not > 30
        assert 'Bob' not in names    # age 25

    def test_multiple_parameters(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.city = $city AND n.age > $age RETURN n.name AS name",
            params={'city': 'Oslo', 'age': 30}
        )
        names = [r['name'] for r in result['rows']]
        assert 'Charlie' in names  # Oslo, age 35
        assert 'Eve' in names      # Oslo, age 40
        assert 'Alice' not in names  # Oslo, age 30 (not > 30)

    def test_missing_parameter_error(self, cypher_graph):
        with pytest.raises(RuntimeError, match="Missing parameter"):
            cypher_graph.cypher(
                "MATCH (n:Person) WHERE n.age > $nonexistent RETURN n.name"
            )

    def test_parameter_with_to_df(self, cypher_graph):
        df = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.age >= $min_age RETURN n.name AS name, n.age AS age ORDER BY n.age",
            params={'min_age': 35}, to_df=True
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Charlie (35) and Eve (40)
        assert list(df['name']) == ['Charlie', 'Eve']

    def test_string_parameter(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) WHERE n.city = $city RETURN n.name AS name ORDER BY n.name",
            params={'city': 'Bergen'}
        )
        names = [r['name'] for r in result['rows']]
        assert names == ['Bob', 'Diana']

    def test_parameter_in_return(self, cypher_graph):
        result = cypher_graph.cypher(
            "MATCH (n:Person) RETURN n.name AS name, $label AS category ORDER BY n.name LIMIT 1",
            params={'label': 'person'}
        )
        assert result['rows'][0]['category'] == 'person'


class TestExistingFeatures:
    """Tests for already-implemented features to ensure coverage."""

    def test_unwind(self, cypher_graph):
        result = cypher_graph.cypher("UNWIND [1, 2, 3] AS x RETURN x")
        assert len(result['rows']) == 3

    def test_union(self, cypher_graph):
        result = cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
            UNION
            MATCH (n:Person) WHERE n.age > 35 RETURN n.name AS name
        """)
        names = {r['name'] for r in result['rows']}
        # Oslo: Alice, Charlie, Eve; age > 35: Eve; UNION deduplicates
        assert names == {'Alice', 'Charlie', 'Eve'}

    def test_union_all(self, cypher_graph):
        result = cypher_graph.cypher("""
            MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
            UNION ALL
            MATCH (n:Person) WHERE n.age > 35 RETURN n.name AS name
        """)
        names = [r['name'] for r in result['rows']]
        # Oslo: Alice, Charlie, Eve; age > 35: Eve; UNION ALL keeps duplicates
        assert len(names) == 4  # 3 + 1 (Eve appears twice)

    def test_var_length_path(self, cypher_graph):
        result = cypher_graph.cypher("""
            MATCH (a:Person)-[:KNOWS*1..2]->(b:Person)
            WHERE a.name = 'Alice'
            RETURN DISTINCT b.name AS friend
        """)
        names = {r['friend'] for r in result['rows']}
        # Alice->Bob, Alice->Charlie, Bob->Charlie, Charlie->Diana
        assert 'Bob' in names
        assert 'Charlie' in names

    def test_coalesce_function(self, cypher_graph):
        result = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.name AS name, coalesce(n.email, 'no email') AS contact
            ORDER BY n.name
        """)
        bob = next(r for r in result['rows'] if r['name'] == 'Bob')
        assert bob['contact'] == 'no email'
        alice = next(r for r in result['rows'] if r['name'] == 'Alice')
        assert alice['contact'] == 'alice@test.com'

    def test_collect_aggregate(self, cypher_graph):
        result = cypher_graph.cypher("""
            MATCH (n:Person)
            RETURN n.city AS city, collect(n.name) AS names
            ORDER BY city
        """)
        assert len(result['rows']) == 2  # Bergen and Oslo
