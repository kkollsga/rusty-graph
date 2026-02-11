"""Benchmark-specific fixtures with larger graphs."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


@pytest.fixture
def large_graph():
    """Large graph with 10,000 nodes for performance testing."""
    graph = KnowledgeGraph()
    df = pd.DataFrame({
        'id': list(range(10000)),
        'name': [f'Node_{i}' for i in range(10000)],
        'category': [f'Cat_{i % 50}' for i in range(10000)],
        'value': [i * 1.5 for i in range(10000)],
        'region': [f'Region_{i % 10}' for i in range(10000)],
    })
    graph.add_nodes(df, 'Item', 'id', 'name')
    return graph
