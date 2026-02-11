"""Tests for node operations: add, retrieve, property mapping, counting."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph
import kglite


class TestModuleAPI:
    def test_module_imports(self):
        assert hasattr(kglite, 'KnowledgeGraph')
        assert hasattr(kglite, 'load')

    def test_graph_creation(self):
        graph = KnowledgeGraph()
        assert graph is not None
        assert isinstance(graph.get_schema(), str)


class TestAddNodes:
    def test_add_nodes_basic(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C'], 'value': [10, 20, 30]})
        report = graph.add_nodes(df, 'TestNode', 'id', 'name')
        assert report['nodes_created'] == 3

    def test_add_nodes_empty_dataframe(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [], 'name': []})
        report = graph.add_nodes(df, 'EmptyType', 'id', 'name')
        assert report['nodes_created'] == 0

    def test_add_nodes_with_columns(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1], 'name': ['A'], 'keep': ['yes'], 'drop': ['no']
        })
        graph.add_nodes(df, 'Node', 'id', 'name', columns=['id', 'name', 'keep'])
        node = graph.type_filter('Node').get_nodes()[0]
        assert 'keep' in node
        assert 'drop' not in node

    def test_add_nodes_conflict_update(self):
        graph = KnowledgeGraph()
        df1 = pd.DataFrame({'id': [1], 'name': ['A'], 'v': [10]})
        df2 = pd.DataFrame({'id': [1], 'name': ['A'], 'v': [20]})
        graph.add_nodes(df1, 'Node', 'id', 'name')
        graph.add_nodes(df2, 'Node', 'id', 'name', conflict_handling='update')
        node = graph.type_filter('Node').get_nodes()[0]
        assert node['v'] == 20

    def test_add_nodes_conflict_skip(self):
        graph = KnowledgeGraph()
        df1 = pd.DataFrame({'id': [1], 'name': ['A'], 'v': [10]})
        df2 = pd.DataFrame({'id': [1], 'name': ['A'], 'v': [20]})
        graph.add_nodes(df1, 'Node', 'id', 'name')
        graph.add_nodes(df2, 'Node', 'id', 'name', conflict_handling='skip')
        node = graph.type_filter('Node').get_nodes()[0]
        assert node['v'] == 10

    def test_add_nodes_null_values(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'optional': ['value', None, 'other'],
        })
        graph.add_nodes(df, 'Node', 'id', 'name')
        nodes = graph.type_filter('Node').get_nodes()
        assert len(nodes) == 3


class TestPropertyMapping:
    def test_id_field_renamed(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'user_id': [1], 'name': ['Alice']})
        graph.add_nodes(df, 'User', 'user_id', 'name')
        node = graph.type_filter('User').get_nodes()[0]
        assert node['id'] == 1
        assert 'user_id' not in node

    def test_title_field_renamed(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'full_name': ['Alice']})
        graph.add_nodes(df, 'User', 'id', 'full_name')
        node = graph.type_filter('User').get_nodes()[0]
        assert node['title'] == 'Alice'
        assert 'full_name' not in node

    def test_other_fields_preserved(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'age': [30], 'city': ['Oslo']})
        graph.add_nodes(df, 'User', 'id', 'name')
        node = graph.type_filter('User').get_nodes()[0]
        assert node['age'] == 30
        assert node['city'] == 'Oslo'


class TestRetrieveNodes:
    def test_get_nodes(self, small_graph):
        nodes = small_graph.type_filter('Person').get_nodes()
        assert len(nodes) == 3
        titles = {n['title'] for n in nodes}
        assert titles == {'Alice', 'Bob', 'Charlie'}

    def test_node_count(self, small_graph):
        count = small_graph.type_filter('Person').node_count()
        assert count == 3

    def test_get_titles(self, small_graph):
        # get_titles returns flat list when no traversal (single parent)
        titles = small_graph.type_filter('Person').get_titles()
        assert isinstance(titles, list)
        assert set(titles) == {'Alice', 'Bob', 'Charlie'}

    def test_get_ids(self, small_graph):
        ids = small_graph.type_filter('Person').get_ids()
        assert len(ids) == 3
        for item in ids:
            assert 'id' in item
            assert 'title' in item
            assert 'type' in item

    def test_id_values(self, small_graph):
        ids = small_graph.type_filter('Person').id_values()
        assert set(ids) == {1, 2, 3}

    def test_indices(self, small_graph):
        indices = small_graph.type_filter('Person').indices()
        assert len(indices) == 3

    def test_get_properties(self, small_graph):
        # get_properties returns flat list when no traversal (single parent)
        props = small_graph.type_filter('Person').get_properties(['age', 'city'])
        assert isinstance(props, list)
        assert len(props) == 3
        for row in props:
            assert len(row) == 2  # (age, city)

    def test_get_node_by_id(self, small_graph):
        node = small_graph.get_node_by_id('Person', 1)
        assert node is not None
        assert node['title'] == 'Alice'
        assert node['age'] == 28

    def test_get_node_by_id_not_found(self, small_graph):
        node = small_graph.get_node_by_id('Person', 999)
        assert node is None

    def test_type_filter_nonexistent(self, small_graph):
        result = small_graph.type_filter('NonExistent')
        assert result.node_count() == 0
