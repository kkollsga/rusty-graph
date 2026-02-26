"""Tests for index management: create, drop, list, composite, performance."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


@pytest.fixture
def indexed_graph():
    """Graph with data suitable for indexing tests."""
    graph = KnowledgeGraph()
    df = pd.DataFrame({
        'id': list(range(1, 101)),
        'name': [f'Node_{i}' for i in range(1, 101)],
        'category': [f'Cat_{i % 5}' for i in range(100)],
        'value': [i * 10 for i in range(100)],
    })
    graph.add_nodes(df, 'Item', 'id', 'name')
    return graph


class TestCreateDrop:
    def test_create_index(self, indexed_graph):
        indexed_graph.create_index('Item', 'category')
        assert indexed_graph.has_index('Item', 'category')

    def test_drop_index(self, indexed_graph):
        indexed_graph.create_index('Item', 'category')
        indexed_graph.drop_index('Item', 'category')
        assert not indexed_graph.has_index('Item', 'category')

    def test_has_index_false(self, indexed_graph):
        assert not indexed_graph.has_index('Item', 'nonexistent')

    def test_list_indexes(self, indexed_graph):
        indexed_graph.create_index('Item', 'category')
        indexed_graph.create_index('Item', 'value')
        indexes = indexed_graph.list_indexes()
        assert len(indexes) >= 2


class TestIndexUsage:
    def test_filter_uses_index(self, indexed_graph):
        indexed_graph.create_index('Item', 'category')
        result = indexed_graph.select('Item').where({'category': 'Cat_0'})
        assert result.len() == 20

    def test_filter_no_index_same_result(self, indexed_graph):
        # Without index
        result_no_idx = indexed_graph.select('Item').where({'category': 'Cat_1'})
        count_no_idx = result_no_idx.len()

        # With index
        indexed_graph.create_index('Item', 'category')
        result_idx = indexed_graph.select('Item').where({'category': 'Cat_1'})
        count_idx = result_idx.len()

        assert count_no_idx == count_idx

    def test_index_stats(self, indexed_graph):
        indexed_graph.create_index('Item', 'category')
        stats = indexed_graph.index_stats('Item', 'category')
        assert stats is not None

    def test_rebuild_indexes(self, indexed_graph):
        indexed_graph.create_index('Item', 'category')
        indexed_graph.rebuild_indexes()
        assert indexed_graph.has_index('Item', 'category')


class TestCompositeIndex:
    def test_create_composite(self, indexed_graph):
        indexed_graph.create_composite_index('Item', ['category', 'value'])
        assert indexed_graph.has_composite_index('Item', ['category', 'value'])

    def test_drop_composite(self, indexed_graph):
        indexed_graph.create_composite_index('Item', ['category', 'value'])
        indexed_graph.drop_composite_index('Item', ['category', 'value'])
        assert not indexed_graph.has_composite_index('Item', ['category', 'value'])

    def test_list_composite(self, indexed_graph):
        indexed_graph.create_composite_index('Item', ['category', 'value'])
        composites = indexed_graph.list_composite_indexes()
        assert len(composites) >= 1


class TestEmptyGraph:
    def test_index_on_empty(self):
        graph = KnowledgeGraph()
        # Should not error
        indexes = graph.list_indexes()
        assert len(indexes) == 0
