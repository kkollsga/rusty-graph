"""Tests for FFI boundary: Python-Rust value conversion (py_in.rs / py_out.rs).

Verifies that values survive round-trips through add_nodes -> get_nodes/get_properties,
including edge cases like NaN, None, large integers, unicode, and various pandas dtypes.
"""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graph():
    return KnowledgeGraph()


# ---------------------------------------------------------------------------
# Value Round-Trip Tests (py_in -> py_out)
# ---------------------------------------------------------------------------

class TestValueRoundTrips:
    """Verify that Python values survive the Python->Rust->Python round-trip."""

    def test_integer_round_trip(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': [42]})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node['val'] == 42
        assert isinstance(node['val'], int)

    def test_float_round_trip(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': [3.14]})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert abs(node['val'] - 3.14) < 1e-10

    def test_string_round_trip(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': ['hello']})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node['val'] == 'hello'

    def test_boolean_round_trip(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'flag': [True]})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node['flag'] is True

    def test_none_round_trip(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': [None]})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node.get('val') is None

    def test_date_string_round_trip(self, graph):
        """Date strings are stored and retrievable."""
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'dt': ['2025-01-15']})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert '2025-01-15' in str(node['dt'])

    def test_negative_integer(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': [-999]})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node['val'] == -999

    def test_large_integer(self, graph):
        """Large integers within i64 range should survive."""
        big = 2**53  # largest exact int in float64
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': [big]})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node['val'] == big

    def test_zero_float(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': [0.0]})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node['val'] == 0.0

    def test_empty_string(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': ['']})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node['val'] == ''

    def test_unicode_string(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': ['cafe\u0301 \u2603 \U0001F600']})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert '\u2603' in node['val']


# ---------------------------------------------------------------------------
# NaN / NA / NaT Handling
# ---------------------------------------------------------------------------

class TestNullHandling:
    """Verify that various null-like values are handled correctly."""

    def test_nan_becomes_null(self, graph):
        """NaN float values should become null in the graph."""
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': [float('nan')]})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node.get('val') is None

    def test_pd_na_becomes_null(self, graph):
        """pd.NA should become null."""
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': pd.array([pd.NA], dtype='Int64')})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node.get('val') is None

    def test_mixed_nulls_and_values(self, graph):
        """Mix of real values and nulls in same column."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'val': [10, None, 30],
        })
        graph.add_nodes(df, 'T', 'id', 'name')
        nodes = graph.select('T').collect()
        vals = {n['title']: n.get('val') for n in nodes}
        assert vals['A'] == 10
        assert vals['B'] is None
        assert vals['C'] == 30


# ---------------------------------------------------------------------------
# Pandas Dtype Handling
# ---------------------------------------------------------------------------

class TestPandasDtypes:
    """Verify correct handling of various pandas dtypes during ingestion."""

    def test_int64_dtype(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': pd.array([42], dtype='int64')})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node['val'] == 42

    def test_nullable_int64_dtype(self, graph):
        """Nullable Int64 (capital I) dtype."""
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': pd.array([42], dtype='Int64')})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node['val'] == 42

    def test_float32_dtype(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': pd.array([3.14], dtype='float32')})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert abs(node['val'] - 3.14) < 0.01  # float32 precision

    def test_string_dtype(self, graph):
        """pandas StringDtype (pd.StringDtype())."""
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': pd.array(['hello'], dtype='string')})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node['val'] == 'hello'

    def test_boolean_dtype(self, graph):
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'flag': pd.array([True], dtype='boolean')})
        graph.add_nodes(df, 'T', 'id', 'name')
        node = graph.select('T').collect()[0]
        assert node['flag'] is True

    def test_category_dtype_rejected(self, graph):
        """Category dtype is not supported — must convert to string first."""
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['A', 'B'],
            'color': pd.Categorical(['red', 'blue']),
        })
        with pytest.raises(ValueError, match="Unsupported column type"):
            graph.add_nodes(df, 'T', 'id', 'name')

    def test_category_as_string_workaround(self, graph):
        """Category converted to string works fine."""
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['A', 'B'],
            'color': pd.Categorical(['red', 'blue']),
        })
        df['color'] = df['color'].astype(str)
        graph.add_nodes(df, 'T', 'id', 'name')
        nodes = graph.select('T').collect()
        colors = {n['title']: n['color'] for n in nodes}
        assert colors['A'] == 'red'
        assert colors['B'] == 'blue'


# ---------------------------------------------------------------------------
# Filter Condition Parsing (pydict_to_filter_conditions)
# ---------------------------------------------------------------------------

class TestFilterConditions:
    """Verify all filter operators via the .where() API."""

    @pytest.fixture
    def populated_graph(self, graph):
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'city': ['Oslo', 'Bergen', 'Oslo', 'Stavanger', 'Bergen'],
        })
        graph.add_nodes(df, 'Person', 'id', 'name')
        return graph

    def test_equals(self, populated_graph):
        result = populated_graph.select('Person').where({'age': 30})
        assert result.len() == 1
        assert result.collect()[0]['title'] == 'Bob'

    def test_not_equals(self, populated_graph):
        result = populated_graph.select('Person').where({'age': {'!=': 30}})
        assert result.len() == 4

    def test_greater_than(self, populated_graph):
        result = populated_graph.select('Person').where({'age': {'>': 35}})
        assert result.len() == 2

    def test_greater_than_equals(self, populated_graph):
        result = populated_graph.select('Person').where({'age': {'>=': 35}})
        assert result.len() == 3

    def test_less_than(self, populated_graph):
        result = populated_graph.select('Person').where({'age': {'<': 35}})
        assert result.len() == 2

    def test_less_than_equals(self, populated_graph):
        result = populated_graph.select('Person').where({'age': {'<=': 35}})
        assert result.len() == 3

    def test_in_list(self, populated_graph):
        result = populated_graph.select('Person').where({'age': {'==': [25, 45]}})
        assert result.len() == 2

    def test_in_operator(self, populated_graph):
        result = populated_graph.select('Person').where({'city': {'in': ['Oslo', 'Bergen']}})
        assert result.len() == 4

    def test_between(self, populated_graph):
        result = populated_graph.select('Person').where({'age': {'between': [30, 40]}})
        assert result.len() == 3

    def test_contains(self, populated_graph):
        result = populated_graph.select('Person').where({'city': {'contains': 'erg'}})
        assert result.len() == 2  # Bergen x2

    def test_starts_with(self, populated_graph):
        result = populated_graph.select('Person').where({'city': {'starts_with': 'Os'}})
        assert result.len() == 2

    def test_ends_with(self, populated_graph):
        result = populated_graph.select('Person').where({'city': {'ends_with': 'lo'}})
        assert result.len() == 2  # Oslo x2

    def test_is_null(self, populated_graph):
        """is_null finds nodes where property is missing."""
        result = populated_graph.select('Person').where({'email': {'is_null': True}})
        assert result.len() == 5  # no one has email

    def test_is_not_null(self, populated_graph):
        result = populated_graph.select('Person').where({'age': {'is_not_null': True}})
        assert result.len() == 5  # all have age

    def test_combined_filters(self, populated_graph):
        """Multiple filter conditions applied together (AND)."""
        result = populated_graph.select('Person').where({
            'age': {'>': 25},
            'city': 'Oslo',
        })
        assert result.len() == 1
        assert result.collect()[0]['title'] == 'Charlie'


# ---------------------------------------------------------------------------
# Cypher Value Output (value_to_py)
# ---------------------------------------------------------------------------

class TestCypherValueOutput:
    """Verify that Cypher RETURN correctly converts Rust values to Python."""

    def test_return_integer(self, graph):
        result = graph.cypher("UNWIND [1] AS x RETURN x")
        assert result[0]['x'] == 1
        assert isinstance(result[0]['x'], int)

    def test_return_float(self, graph):
        result = graph.cypher("UNWIND [1] AS x RETURN 3.14 AS val")
        assert abs(result[0]['val'] - 3.14) < 1e-10

    def test_return_string(self, graph):
        result = graph.cypher("UNWIND [1] AS x RETURN 'hello' AS val")
        assert result[0]['val'] == 'hello'

    def test_return_boolean_true(self, graph):
        result = graph.cypher("UNWIND [1] AS x RETURN true AS val")
        assert result[0]['val'] is True

    def test_return_boolean_false(self, graph):
        result = graph.cypher("UNWIND [1] AS x RETURN false AS val")
        assert result[0]['val'] is False

    def test_return_null(self, graph):
        result = graph.cypher("UNWIND [1] AS x RETURN null AS val")
        assert result[0]['val'] is None

    def test_return_list(self, graph):
        result = graph.cypher("UNWIND [1] AS x RETURN [1, 2, 3] AS val")
        assert result[0]['val'] == [1, 2, 3]

    def test_return_arithmetic(self, graph):
        result = graph.cypher("UNWIND [1] AS x RETURN 2 + 3 AS val")
        assert result[0]['val'] == 5


# ---------------------------------------------------------------------------
# Multiple Values / Batch Ingestion
# ---------------------------------------------------------------------------

class TestBatchIngestion:
    """Verify batch DataFrame ingestion handles mixed types correctly."""

    def test_multiple_rows(self, graph):
        df = pd.DataFrame({
            'id': range(100),
            'name': [f'Node_{i}' for i in range(100)],
            'val': list(range(100)),
        })
        graph.add_nodes(df, 'T', 'id', 'name')
        assert graph.select('T').len() == 100

    def test_mixed_int_float_column(self, graph):
        """Column with both int and float values."""
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['A', 'B'],
            'val': [1, 2.5],  # pandas infers float64
        })
        graph.add_nodes(df, 'T', 'id', 'name')
        nodes = graph.select('T').collect()
        vals = sorted([n['val'] for n in nodes])
        assert vals[0] == 1.0
        assert vals[1] == 2.5

    def test_empty_dataframe(self, graph):
        """Empty DataFrame should not crash."""
        df = pd.DataFrame({'id': [], 'name': []})
        graph.add_nodes(df, 'T', 'id', 'name')
        assert graph.select('T').len() == 0
