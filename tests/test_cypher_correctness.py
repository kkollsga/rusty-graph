"""Comprehensive Cypher correctness regression tests.

Tests all known edge cases and validates semantics against Neo4j/openCypher
specifications where applicable.
"""
import pytest
import kglite as rg


@pytest.fixture
def chain_graph():
    """A→B→C→D chain graph for path testing."""
    g = rg.KnowledgeGraph()
    g.cypher("CREATE (a:Person {name: 'Alice'})")
    g.cypher("CREATE (b:Person {name: 'Bob'})")
    g.cypher("CREATE (c:Person {name: 'Charlie'})")
    g.cypher("CREATE (d:Person {name: 'Diana'})")
    g.cypher("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)")
    g.cypher("MATCH (a:Person {name: 'Bob'}), (b:Person {name: 'Charlie'}) CREATE (a)-[:KNOWS]->(b)")
    g.cypher("MATCH (a:Person {name: 'Charlie'}), (b:Person {name: 'Diana'}) CREATE (a)-[:KNOWS]->(b)")
    return g


@pytest.fixture
def social_graph():
    """Graph with multiple relationship types for complex queries."""
    g = rg.KnowledgeGraph()
    g.cypher("CREATE (a:Person {name: 'Alice', age: 30})")
    g.cypher("CREATE (b:Person {name: 'Bob', age: 25})")
    g.cypher("CREATE (c:Person {name: 'Charlie', age: 35})")
    g.cypher("CREATE (p:Product {name: 'Widget', price: 10})")
    g.cypher("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)")
    g.cypher("MATCH (a:Person {name: 'Alice'}), (c:Person {name: 'Charlie'}) CREATE (a)-[:KNOWS]->(c)")
    g.cypher("MATCH (a:Person {name: 'Alice'}), (p:Product {name: 'Widget'}) CREATE (a)-[:BOUGHT]->(p)")
    return g


# ========================================================================
# OPTIONAL MATCH semantics
# ========================================================================

class TestOptionalMatchSemantics:
    def test_optional_match_first_clause_no_match_returns_null_row(self):
        """OPTIONAL MATCH as first clause with no matches returns one null row."""
        g = rg.KnowledgeGraph()
        g.cypher("CREATE (a:Person {name: 'Alice'})")
        result = g.cypher("OPTIONAL MATCH (n:NonExistent) RETURN n")
        assert len(result) == 1
        assert result[0]['n'] is None

    def test_optional_match_first_clause_with_match_returns_results(self):
        """OPTIONAL MATCH as first clause with matches returns normal results."""
        g = rg.KnowledgeGraph()
        g.cypher("CREATE (a:Person {name: 'Alice'})")
        g.cypher("CREATE (b:Person {name: 'Bob'})")
        result = g.cypher("OPTIONAL MATCH (n:Person) RETURN n.name ORDER BY n.name")
        assert len(result) == 2
        assert result[0]['n.name'] == 'Alice'
        assert result[1]['n.name'] == 'Bob'

    def test_optional_match_after_match_no_match_preserves_row(self, social_graph):
        """OPTIONAL MATCH after MATCH preserves row with NULLs when unmatched."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company)
            RETURN p.name, c
            ORDER BY p.name
        """)
        assert len(result) == 3
        for row in result:
            assert row['c'] is None  # No WORKS_AT relationships exist

    def test_optional_match_first_clause_with_edge(self):
        """OPTIONAL MATCH with edge pattern as first clause."""
        g = rg.KnowledgeGraph()
        g.cypher("CREATE (a:Person {name: 'Alice'})")
        result = g.cypher("OPTIONAL MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b")
        assert len(result) == 1
        assert result[0]['r'] is None
        assert result[0]['b'] is None


# ========================================================================
# Path binding consistency
# ========================================================================

class TestPathBindingConsistency:
    def test_nodes_shortest_path_includes_all_nodes(self, chain_graph):
        """nodes(p) for shortestPath includes source, intermediates, and target."""
        result = chain_graph.cypher("""
            MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..5]->(d:Person {name: 'Diana'}))
            RETURN nodes(p) AS path_nodes
        """)
        assert len(result) == 1
        nodes = result[0]['path_nodes']
        titles = [n['title'] for n in nodes]
        assert titles == ['Alice', 'Bob', 'Charlie', 'Diana']

    def test_nodes_variable_length_includes_all_nodes(self, chain_graph):
        """nodes(p) for variable-length path includes source through target."""
        result = chain_graph.cypher("""
            MATCH p = (a:Person {name: 'Alice'})-[:KNOWS*1..5]->(d:Person {name: 'Diana'})
            RETURN nodes(p) AS path_nodes
        """)
        assert len(result) == 1
        nodes = result[0]['path_nodes']
        titles = [n['title'] for n in nodes]
        assert titles == ['Alice', 'Bob', 'Charlie', 'Diana']

    def test_nodes_consistency_shortest_vs_varlen(self, chain_graph):
        """nodes(p) returns same result for shortestPath and variable-length."""
        sp_result = chain_graph.cypher("""
            MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..5]->(d:Person {name: 'Diana'}))
            RETURN nodes(p) AS n
        """)
        vl_result = chain_graph.cypher("""
            MATCH p = (a:Person {name: 'Alice'})-[:KNOWS*3]->(d:Person {name: 'Diana'})
            RETURN nodes(p) AS n
        """)
        sp_titles = [n['title'] for n in sp_result[0]['n']]
        vl_titles = [n['title'] for n in vl_result[0]['n']]
        assert sp_titles == vl_titles == ['Alice', 'Bob', 'Charlie', 'Diana']

    def test_relationships_shortest_path(self, chain_graph):
        """relationships(p) for shortestPath returns all edge types."""
        result = chain_graph.cypher("""
            MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..5]->(d:Person {name: 'Diana'}))
            RETURN relationships(p) AS rels
        """)
        assert result[0]['rels'] == ['KNOWS', 'KNOWS', 'KNOWS']

    def test_relationships_variable_length(self, chain_graph):
        """relationships(p) for variable-length path returns all edge types."""
        result = chain_graph.cypher("""
            MATCH p = (a:Person {name: 'Alice'})-[:KNOWS*1..5]->(d:Person {name: 'Diana'})
            RETURN relationships(p) AS rels
        """)
        assert result[0]['rels'] == ['KNOWS', 'KNOWS', 'KNOWS']

    def test_length_matches_node_count_minus_one(self, chain_graph):
        """length(p) equals len(nodes(p)) - 1 for all path types."""
        result = chain_graph.cypher("""
            MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..5]->(d:Person {name: 'Diana'}))
            RETURN length(p) AS len, nodes(p) AS n
        """)
        assert result[0]['len'] == len(result[0]['n']) - 1

    def test_length_single_hop(self, chain_graph):
        """length(p) = 1 for single-hop path assignment."""
        result = chain_graph.cypher("""
            MATCH p = (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person)
            RETURN a.name, b.name, length(p) AS len
        """)
        assert len(result) == 1
        assert result[0]['len'] == 1

    def test_nodes_cyclic_graph(self):
        """nodes(p) handles cyclic graphs correctly (no source duplication)."""
        g = rg.KnowledgeGraph()
        g.cypher("CREATE (a:Node {name: 'A'})")
        g.cypher("CREATE (b:Node {name: 'B'})")
        g.cypher("CREATE (c:Node {name: 'C'})")
        g.cypher("MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:LINK]->(b)")
        g.cypher("MATCH (b:Node {name: 'B'}), (c:Node {name: 'C'}) CREATE (b)-[:LINK]->(c)")
        g.cypher("MATCH (c:Node {name: 'C'}), (a:Node {name: 'A'}) CREATE (c)-[:LINK]->(a)")

        result = g.cypher("""
            MATCH p = (a:Node {name: 'A'})-[:LINK*1..3]->(b:Node {name: 'A'})
            RETURN length(p) AS len, nodes(p) AS n
        """)
        assert len(result) >= 1
        for row in result:
            titles = [n['title'] for n in row['n']]
            assert titles[0] == 'A'  # Starts with A
            assert titles[-1] == 'A'  # Ends with A (cycle)
            assert row['len'] == len(titles) - 1


# ========================================================================
# UNWIND nested lists
# ========================================================================

class TestUnwindNested:
    def test_unwind_flat_list(self):
        """UNWIND with flat list still works correctly."""
        g = rg.KnowledgeGraph()
        result = g.cypher("WITH [1, 2, 3] AS items UNWIND items AS x RETURN x")
        values = [r['x'] for r in result]
        assert values == [1, 2, 3]

    def test_unwind_nested_list(self):
        """UNWIND with nested lists preserves inner lists as values."""
        g = rg.KnowledgeGraph()
        result = g.cypher("WITH [[1, 2], [3, 4]] AS items UNWIND items AS x RETURN x")
        assert len(result) == 2
        # Inner lists are parsed and returned as Python lists
        assert result[0]['x'] == [1, 2]
        assert result[1]['x'] == [3, 4]

    def test_unwind_empty_list(self):
        """UNWIND with empty list produces no rows."""
        g = rg.KnowledgeGraph()
        result = g.cypher("WITH [] AS items UNWIND items AS x RETURN x")
        assert len(result) == 0

    def test_unwind_string_list(self):
        """UNWIND with list of quoted strings."""
        g = rg.KnowledgeGraph()
        result = g.cypher("""WITH ["hello", "world"] AS items UNWIND items AS x RETURN x""")
        values = [r['x'] for r in result]
        assert values == ['hello', 'world']


# ========================================================================
# Regex error handling
# ========================================================================

class TestRegexErrors:
    def test_regex_valid_match(self, social_graph):
        """Valid regex matches correctly."""
        result = social_graph.cypher("""
            MATCH (p:Person) WHERE p.name =~ '^A.*' RETURN p.name
        """)
        assert len(result) == 1
        assert result[0]['p.name'] == 'Alice'

    def test_regex_valid_no_match(self, social_graph):
        """Valid regex that matches nothing returns empty."""
        result = social_graph.cypher("""
            MATCH (p:Person) WHERE p.name =~ '^Z.*' RETURN p.name
        """)
        assert len(result) == 0

    def test_regex_invalid_pattern_raises_error(self, social_graph):
        """Invalid regex pattern raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Invalid regular expression"):
            social_graph.cypher("""
                MATCH (p:Person) WHERE p.name =~ '[unclosed' RETURN p.name
            """)


# ========================================================================
# Aggregation edge cases
# ========================================================================

class TestAggregationEdgeCases:
    def test_sum_empty_returns_zero(self):
        """sum() on empty set returns 0 (matches Neo4j/SQL convention)."""
        g = rg.KnowledgeGraph()
        g.cypher("CREATE (a:Person {name: 'Alice', age: 30})")
        result = g.cypher("MATCH (n:NonExistent) RETURN sum(n.age) AS total")
        # Aggregation in RETURN always produces one row (Neo4j convention)
        assert len(result) == 1
        assert result[0]['total'] == 0

    def test_avg_empty_returns_null(self, social_graph):
        """avg() with all null values returns null."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            RETURN avg(p.nonexistent) AS avg_val
        """)
        assert len(result) == 1
        assert result[0]['avg_val'] is None

    def test_count_star_empty_returns_zero(self):
        """count(*) on empty match returns 0 (Neo4j convention)."""
        g = rg.KnowledgeGraph()
        g.cypher("CREATE (a:Person {name: 'Alice'})")
        result = g.cypher("MATCH (n:NonExistent) RETURN count(*) AS cnt")
        # Aggregation in RETURN always produces one row
        assert len(result) == 1
        assert result[0]['cnt'] == 0

    def test_collect_empty_returns_empty_list(self, social_graph):
        """collect() with no matching values returns empty list."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            RETURN collect(p.nonexistent) AS items
        """)
        assert len(result) == 1
        assert result[0]['items'] == '[]' or result[0]['items'] == []

    def test_sum_with_values(self, social_graph):
        """sum() with actual values works correctly."""
        result = social_graph.cypher("""
            MATCH (p:Person) RETURN sum(p.age) AS total
        """)
        assert result[0]['total'] == 90  # 30 + 25 + 35

    def test_avg_with_values(self, social_graph):
        """avg() with actual values works correctly."""
        result = social_graph.cypher("""
            MATCH (p:Person) RETURN avg(p.age) AS average
        """)
        assert result[0]['average'] == 30.0  # (30 + 25 + 35) / 3


# ========================================================================
# Anonymous nodes in patterns
# ========================================================================

class TestAnonymousNodes:
    def test_anonymous_intermediate_node(self, chain_graph):
        """Pattern with anonymous intermediate node works."""
        result = chain_graph.cypher("""
            MATCH (a:Person {name: 'Alice'})-[:KNOWS]->()-[:KNOWS]->(c:Person)
            RETURN c.name
        """)
        assert len(result) == 1
        assert result[0]['c.name'] == 'Charlie'

    def test_anonymous_double_intermediate(self, chain_graph):
        """Pattern with two anonymous intermediate nodes."""
        result = chain_graph.cypher("""
            MATCH (a:Person {name: 'Alice'})-[:KNOWS]->()-[:KNOWS]->()-[:KNOWS]->(d:Person)
            RETURN d.name
        """)
        assert len(result) == 1
        assert result[0]['d.name'] == 'Diana'

    def test_anonymous_with_named_endpoints(self, chain_graph):
        """Anonymous intermediate doesn't affect named endpoint binding."""
        result = chain_graph.cypher("""
            MATCH (a:Person)-[:KNOWS]->()-[:KNOWS]->(c:Person)
            RETURN a.name, c.name
            ORDER BY a.name
        """)
        assert len(result) >= 1
        names = [(r['a.name'], r['c.name']) for r in result]
        assert ('Alice', 'Charlie') in names


# ========================================================================
# Direction enforcement in variable-length paths
# ========================================================================

class TestDirectionEnforcement:
    def test_outgoing_varlen_respects_direction(self):
        """Variable-length outgoing path does not traverse incoming edges."""
        g = rg.KnowledgeGraph()
        g.cypher("CREATE (a:Node {name: 'A'})")
        g.cypher("CREATE (b:Node {name: 'B'})")
        g.cypher("CREATE (c:Node {name: 'C'})")
        # A→B and C→B (B is a sink from A, but C points TO B, not from B)
        g.cypher("MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:LINK]->(b)")
        g.cypher("MATCH (c:Node {name: 'C'}), (b:Node {name: 'B'}) CREATE (c)-[:LINK]->(b)")

        # A-[:LINK*1..3]->? should only find B (not C, since C→B is incoming to B)
        result = g.cypher("""
            MATCH (a:Node {name: 'A'})-[:LINK*1..3]->(target:Node)
            RETURN target.name
        """)
        names = [r['target.name'] for r in result]
        assert 'B' in names
        assert 'C' not in names  # C→B is not reachable from A via outgoing

    def test_incoming_varlen_respects_direction(self):
        """Variable-length incoming path traverses edges in reverse."""
        g = rg.KnowledgeGraph()
        g.cypher("CREATE (a:Node {name: 'A'})")
        g.cypher("CREATE (b:Node {name: 'B'})")
        g.cypher("CREATE (c:Node {name: 'C'})")
        g.cypher("MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:LINK]->(b)")
        g.cypher("MATCH (b:Node {name: 'B'}), (c:Node {name: 'C'}) CREATE (b)-[:LINK]->(c)")

        # <-[:LINK*1..3]- from C should find B and A
        result = g.cypher("""
            MATCH (c:Node {name: 'C'})<-[:LINK*1..3]-(source:Node)
            RETURN source.name ORDER BY source.name
        """)
        names = [r['source.name'] for r in result]
        assert 'A' in names
        assert 'B' in names

    def test_bidirectional_varlen_traverses_both(self):
        """Bidirectional variable-length path traverses both directions."""
        g = rg.KnowledgeGraph()
        g.cypher("CREATE (a:Node {name: 'A'})")
        g.cypher("CREATE (b:Node {name: 'B'})")
        g.cypher("CREATE (c:Node {name: 'C'})")
        g.cypher("MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:LINK]->(b)")
        g.cypher("MATCH (c:Node {name: 'C'}), (b:Node {name: 'B'}) CREATE (c)-[:LINK]->(b)")

        # B-[:LINK*1..3]-? (undirected) should find both A and C
        result = g.cypher("""
            MATCH (b:Node {name: 'B'})-[:LINK*1..3]-(other:Node)
            RETURN other.name ORDER BY other.name
        """)
        names = [r['other.name'] for r in result]
        assert 'A' in names
        assert 'C' in names


# ========================================================================
# WITH seeding edge cases (phantom row regression)
# ========================================================================

class TestWithSeeding:
    def test_empty_match_with_aggregation_returns_empty(self):
        """MATCH with no results + WITH aggregation returns empty (no phantom row)."""
        g = rg.KnowledgeGraph()
        g.cypher("CREATE (a:Person {name: 'Alice'})")
        result = g.cypher("""
            MATCH (n:NonExistent)
            WITH n, count(n) AS c
            RETURN n, c
        """)
        assert len(result) == 0

    def test_with_as_first_clause_literal(self):
        """WITH as first clause evaluates literal expressions."""
        g = rg.KnowledgeGraph()
        result = g.cypher("WITH 42 AS x RETURN x")
        assert len(result) == 1
        assert result[0]['x'] == 42

    def test_unwind_literal_list(self):
        """WITH + UNWIND literal list works correctly."""
        g = rg.KnowledgeGraph()
        result = g.cypher("WITH [1, 2, 3] AS items UNWIND items AS x RETURN x")
        values = [r['x'] for r in result]
        assert values == [1, 2, 3]


# ========================================================================
# Math functions
# ========================================================================

class TestMathFunctions:
    """Tests for abs, ceil, floor, round, sqrt, sign."""

    def test_abs_integer(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN abs(-5) AS val")
        assert result[0]['val'] == 5

    def test_abs_float(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN abs(-3.14) AS val")
        assert abs(result[0]['val'] - 3.14) < 1e-10

    def test_abs_positive_unchanged(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN abs(42) AS val")
        assert result[0]['val'] == 42

    def test_ceil(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN ceil(2.3) AS val")
        assert result[0]['val'] == 3.0

    def test_ceil_negative(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN ceil(-2.7) AS val")
        assert result[0]['val'] == -2.0

    def test_ceiling_alias(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN ceiling(2.3) AS val")
        assert result[0]['val'] == 3.0

    def test_floor(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN floor(2.7) AS val")
        assert result[0]['val'] == 2.0

    def test_floor_negative(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN floor(-2.3) AS val")
        assert result[0]['val'] == -3.0

    def test_round(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN round(2.5) AS val")
        assert result[0]['val'] == 3.0

    def test_round_down(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN round(2.4) AS val")
        assert result[0]['val'] == 2.0

    def test_sqrt(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN sqrt(16) AS val")
        assert result[0]['val'] == 4.0

    def test_sqrt_negative_returns_null(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN sqrt(-1) AS val")
        assert result[0]['val'] is None

    def test_sign_negative(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN sign(-5) AS val")
        assert result[0]['val'] == -1

    def test_sign_zero(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN sign(0) AS val")
        assert result[0]['val'] == 0

    def test_sign_positive(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN sign(3.7) AS val")
        assert result[0]['val'] == 1

    def test_null_propagation(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN abs(null) AS a, ceil(null) AS b, round(null) AS c")
        assert result[0]['a'] is None
        assert result[0]['b'] is None
        assert result[0]['c'] is None

    def test_math_in_unwind_pipeline(self):
        """Math functions work in UNWIND pipeline."""
        g = rg.KnowledgeGraph()
        result = g.cypher("""
            UNWIND [1.1, 2.5, 3.9, -4.2] AS x
            RETURN x, round(x) AS rounded, abs(x) AS absolute
            ORDER BY x
        """)
        assert len(result) == 4
        assert result[0]['rounded'] == -4.0
        assert result[0]['absolute'] == 4.2
        assert result[2]['rounded'] == 3.0


# ========================================================================
# String coercion on +
# ========================================================================

class TestStringCoercion:
    """Tests for automatic string coercion when + has a string operand."""

    def test_int_plus_string(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN 2024 + '-06' AS val")
        assert result[0]['val'] == '2024-06'

    def test_string_plus_int(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN 'year: ' + 2024 AS val")
        assert result[0]['val'] == 'year: 2024'

    def test_float_plus_string(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN 3.14 + ' pi' AS val")
        assert result[0]['val'] == '3.14 pi'

    def test_bool_plus_string(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN 'is: ' + true AS val")
        assert result[0]['val'] == 'is: true'

    def test_string_plus_null_returns_null(self):
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN 'hello' + null AS val")
        assert result[0]['val'] is None

    def test_unwind_int_string_concat(self):
        """The original failing case: UNWIND integer + string literal."""
        g = rg.KnowledgeGraph()
        result = g.cypher("""
            UNWIND [2020, 2021, 2022] AS yr
            RETURN yr + '-6' AS period
            ORDER BY period
        """)
        assert len(result) == 3
        assert result[0]['period'] == '2020-6'
        assert result[1]['period'] == '2021-6'
        assert result[2]['period'] == '2022-6'

    def test_string_plus_string_still_works(self):
        """Ensure the existing String + String case is not broken."""
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN 'hello' + ' world' AS val")
        assert result[0]['val'] == 'hello world'

    def test_int_plus_int_still_works(self):
        """Ensure numeric + numeric is not affected."""
        g = rg.KnowledgeGraph()
        result = g.cypher("RETURN 3 + 4 AS val")
        assert result[0]['val'] == 7


# ========================================================================
# collect(node)[index] property access (regression: #collect-noderef)
# ========================================================================


class TestCollectNodePropertyAccess:
    """Regression tests for collect(node)[0] returning the node's title
    string instead of preserving the node object for property access.

    Bug: WITH f, collect(fr)[0] AS lr  →  lr.prop returned lr's title for
    every property instead of the actual property value.
    """

    @pytest.fixture
    def reserves_graph(self):
        g = rg.KnowledgeGraph()
        g.cypher("CREATE (f:Field {name: 'ULA'})")
        g.cypher(
            "CREATE (fr:FieldReserves {title: '2025-12-31', oil: 158.185, gas: 87.706})"
        )
        g.cypher(
            "CREATE (fr2:FieldReserves {title: '2024-01-01', oil: 150.0, gas: 80.0})"
        )
        g.cypher(
            "MATCH (fr:FieldReserves {title: '2025-12-31'}), (f:Field {name: 'ULA'}) "
            "CREATE (fr)-[:OF_FIELD]->(f)"
        )
        g.cypher(
            "MATCH (fr:FieldReserves {title: '2024-01-01'}), (f:Field {name: 'ULA'}) "
            "CREATE (fr)-[:OF_FIELD]->(f)"
        )
        return g

    def test_collect_index_property_access(self, reserves_graph):
        """collect(node)[0].prop should return the actual property, not title."""
        result = reserves_graph.cypher(
            "MATCH (fr:FieldReserves)-[:OF_FIELD]->(f:Field {name: 'ULA'}) "
            "WITH f, fr ORDER BY fr.title DESC "
            "WITH f, collect(fr)[0] AS lr "
            "RETURN lr.oil AS oil, lr.gas AS gas"
        )
        assert len(result) == 1
        assert result[0]["oil"] == 158.185
        assert result[0]["gas"] == 87.706

    def test_collect_index_negative(self, reserves_graph):
        """collect(node)[-1] should access the last element."""
        result = reserves_graph.cypher(
            "MATCH (fr:FieldReserves)-[:OF_FIELD]->(f:Field {name: 'ULA'}) "
            "WITH f, fr ORDER BY fr.title DESC "
            "WITH f, collect(fr)[-1] AS lr "
            "RETURN lr.oil AS oil"
        )
        assert len(result) == 1
        assert result[0]["oil"] == 150.0

    def test_collect_index_title_still_works(self, reserves_graph):
        """collect(node)[0].title should still return the title."""
        result = reserves_graph.cypher(
            "MATCH (fr:FieldReserves)-[:OF_FIELD]->(f:Field {name: 'ULA'}) "
            "WITH f, fr ORDER BY fr.title DESC "
            "WITH f, collect(fr)[0] AS lr "
            "RETURN lr.title AS t"
        )
        assert result[0]["t"] == "2025-12-31"

    def test_collect_index_bare_variable(self, reserves_graph):
        """RETURN lr (without property) should still show the title."""
        result = reserves_graph.cypher(
            "MATCH (fr:FieldReserves)-[:OF_FIELD]->(f:Field {name: 'ULA'}) "
            "WITH f, fr ORDER BY fr.title DESC "
            "WITH f, collect(fr)[0] AS lr "
            "RETURN lr"
        )
        assert result[0]["lr"] == "2025-12-31"

    def test_collect_scalar_property_unchanged(self, reserves_graph):
        """collect(node.prop)[0] should still work for scalar properties."""
        result = reserves_graph.cypher(
            "MATCH (fr:FieldReserves)-[:OF_FIELD]->(f:Field {name: 'ULA'}) "
            "WITH f, collect(fr.oil) AS oils "
            "RETURN oils"
        )
        assert len(result) == 1
        oils_str = result[0]["oils"]
        # oils should be a JSON-style list string
        assert "158.185" in str(oils_str)
        assert "150.0" in str(oils_str)
