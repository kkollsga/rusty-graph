"""Property-based tests using Hypothesis.

These tests verify invariants that should hold for ALL valid inputs,
not just hand-picked examples. Hypothesis generates random inputs to
find edge cases humans wouldn't write.
"""

from hypothesis import given, settings
from hypothesis import strategies as st
import pandas as pd
import pytest

from kglite import KnowledgeGraph

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Node type names: ASCII letters only (KGLite may have constraints on names)
node_type_st = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
    min_size=1,
    max_size=20,
)

# Property values that KGLite supports
prop_value_st = st.one_of(
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e15, max_value=1e15),
    st.text(min_size=0, max_size=100),
)


# ---------------------------------------------------------------------------
# Property: Node count invariant
# ---------------------------------------------------------------------------
@given(
    n=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=30)
def test_node_count_invariant(n):
    """Adding N nodes increases count by exactly N."""
    graph = KnowledgeGraph()
    df = pd.DataFrame({"nid": list(range(n)), "name": [f"item_{i}" for i in range(n)]})
    graph.add_nodes(df, "Thing", "nid", "name")
    result = graph.select("Thing").collect()
    assert len(result) == n


# ---------------------------------------------------------------------------
# Property: Filter correctness
# ---------------------------------------------------------------------------
@given(
    n=st.integers(min_value=5, max_value=30),
    target_city=st.sampled_from(["Oslo", "Bergen", "Stavanger"]),
)
@settings(max_examples=30)
def test_filter_returns_only_matching(n, target_city):
    """WHERE filter only returns nodes matching the condition."""
    cities = ["Oslo", "Bergen", "Stavanger"]
    graph = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "nid": list(range(n)),
            "name": [f"p_{i}" for i in range(n)],
            "city": [cities[i % len(cities)] for i in range(n)],
        }
    )
    graph.add_nodes(df, "Person", "nid", "name")

    result = graph.select("Person").where({"city": target_city}).collect()
    for node in result:
        assert node["city"] == target_city


# ---------------------------------------------------------------------------
# Property: Index transparency
# ---------------------------------------------------------------------------
@given(
    n=st.integers(min_value=5, max_value=30),
    target_city=st.sampled_from(["Oslo", "Bergen", "Stavanger"]),
)
@settings(max_examples=20)
def test_index_does_not_change_results(n, target_city):
    """Creating an index should not change query results."""
    cities = ["Oslo", "Bergen", "Stavanger"]
    graph = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "nid": list(range(n)),
            "name": [f"p_{i}" for i in range(n)],
            "city": [cities[i % len(cities)] for i in range(n)],
        }
    )
    graph.add_nodes(df, "Person", "nid", "name")

    # Query without index
    before = graph.select("Person").where({"city": target_city}).collect()

    # Create index and query again
    graph.create_index("Person", "city")
    after = graph.select("Person").where({"city": target_city}).collect()

    assert len(before) == len(after)
    before_ids = {n["id"] for n in before}
    after_ids = {n["id"] for n in after}
    assert before_ids == after_ids


# ---------------------------------------------------------------------------
# Property: Cypher-fluent parity
# ---------------------------------------------------------------------------
@given(n=st.integers(min_value=1, max_value=30))
@settings(max_examples=20)
def test_cypher_fluent_parity(n):
    """MATCH (n:T) RETURN n should match select(T).collect() results."""
    graph = KnowledgeGraph()
    df = pd.DataFrame({"nid": list(range(n)), "name": [f"item_{i}" for i in range(n)]})
    graph.add_nodes(df, "Widget", "nid", "name")

    fluent = graph.select("Widget").collect()
    cypher_result = graph.cypher("MATCH (n:Widget) RETURN n.id AS id")
    cypher_ids = {row["id"] for row in cypher_result}

    assert len(fluent) == len(cypher_ids)
    fluent_ids = {node["id"] for node in fluent}
    assert fluent_ids == cypher_ids


# ---------------------------------------------------------------------------
# Property: Delete consistency
# ---------------------------------------------------------------------------
@given(
    n=st.integers(min_value=3, max_value=20),
    delete_fraction=st.floats(min_value=0.1, max_value=0.9),
)
@settings(max_examples=20)
def test_delete_consistency(n, delete_fraction):
    """Deleted nodes should not appear in any subsequent query."""
    graph = KnowledgeGraph()
    df = pd.DataFrame({"nid": list(range(n)), "name": [f"item_{i}" for i in range(n)]})
    graph.add_nodes(df, "Thing", "nid", "name")

    # Delete some nodes
    n_delete = max(1, int(n * delete_fraction))
    for i in range(n_delete):
        graph.cypher(f"MATCH (n:Thing {{id: {i}}}) DELETE n")

    remaining = graph.select("Thing").collect()
    remaining_ids = {node["id"] for node in remaining}

    # Deleted IDs should not be present
    for i in range(n_delete):
        assert i not in remaining_ids

    # Remaining count should be correct
    assert len(remaining) == n - n_delete


# ---------------------------------------------------------------------------
# Property: Sort correctness
# ---------------------------------------------------------------------------
@given(n=st.integers(min_value=2, max_value=30))
@settings(max_examples=20)
def test_sort_produces_ordered_results(n):
    """Sorting by a numeric property produces correctly ordered results."""
    graph = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "nid": list(range(n)),
            "name": [f"item_{i}" for i in range(n)],
            "score": [float(n - i) for i in range(n)],  # reverse order
        }
    )
    graph.add_nodes(df, "Item", "nid", "name")

    result = graph.select("Item").sort("score").collect()
    scores = [node["score"] for node in result]
    assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# Property: Type roundtrip
# ---------------------------------------------------------------------------
@given(
    int_val=st.integers(min_value=-(2**31), max_value=2**31 - 1),
    float_val=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    str_val=st.text(min_size=0, max_size=50),
)
@settings(max_examples=30)
def test_type_roundtrip(int_val, float_val, str_val):
    """Values put in should come out with the expected types."""
    graph = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "nid": [1],
            "name": ["test"],
            "int_prop": [int_val],
            "float_prop": [float_val],
            "str_prop": [str_val],
        }
    )
    graph.add_nodes(df, "TypeTest", "nid", "name")
    result = graph.select("TypeTest").collect()
    assert len(result) == 1
    node = result[0]
    assert node["int_prop"] == int_val
    assert node["str_prop"] == str_val
    # Floats may have precision differences
    assert abs(node["float_prop"] - float_val) < 1e-6 or node["float_prop"] == pytest.approx(float_val)
