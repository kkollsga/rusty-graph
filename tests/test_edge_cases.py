"""Tests for edge cases: empty graphs, nulls, special characters, boundary values."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


class TestEmptyGraph:
    def test_empty_graph_type_filter(self):
        graph = KnowledgeGraph()
        result = graph.select('NonExistent')
        assert result.len() == 0

    def test_empty_graph_get_nodes(self):
        graph = KnowledgeGraph()
        result = graph.select('Any')
        assert len(result.collect()) == 0

    def test_empty_graph_schema(self):
        graph = KnowledgeGraph()
        schema = graph.schema_text()
        assert isinstance(schema, str)

    def test_empty_graph_connected_components(self):
        graph = KnowledgeGraph()
        components = graph.connected_components()
        assert len(components) == 0


class TestSingleNode:
    def test_single_node_operations(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['Solo']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        assert graph.select('Node').len() == 1
        nodes = graph.select('Node').collect()
        assert nodes[0]['title'] == 'Solo'


class TestSpecialCharacters:
    def test_special_chars_in_properties(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1],
            'name': ["O'Brien & Co."],
            'desc': ['Has "quotes" and <brackets>'],
        })
        graph.add_nodes(df, 'Node', 'id', 'name')
        node = graph.select('Node').collect()[0]
        assert node['title'] == "O'Brien & Co."
        assert '"quotes"' in node['desc']

    def test_unicode_in_properties(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1],
            'name': ['Test'],
            'content': ['Hello World'],
        })
        graph.add_nodes(df, 'Node', 'id', 'name')
        node = graph.select('Node').collect()[0]
        assert node['content'] is not None

    def test_very_long_strings(self):
        graph = KnowledgeGraph()
        long_str = 'x' * 10000
        df = pd.DataFrame({'id': [1], 'name': ['LongNode'], 'data': [long_str]})
        graph.add_nodes(df, 'Node', 'id', 'name')
        node = graph.select('Node').collect()[0]
        assert len(node['data']) == 10000


class TestNumericEdgeCases:
    def test_zero_values(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'value': [0]})
        graph.add_nodes(df, 'Node', 'id', 'name')
        result = graph.select('Node').where({'value': 0})
        assert result.len() == 1

    def test_negative_values(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'value': [-100]})
        graph.add_nodes(df, 'Node', 'id', 'name')
        result = graph.select('Node').where({'value': {'<': 0}})
        assert result.len() == 1

    def test_float_precision(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'value': [3.14159265358979]})
        graph.add_nodes(df, 'Node', 'id', 'name')
        node = graph.select('Node').collect()[0]
        assert abs(node['value'] - 3.14159265358979) < 1e-10


class TestManyNodeTypes:
    def test_many_types(self):
        graph = KnowledgeGraph()
        for i in range(20):
            df = pd.DataFrame({'id': [1], 'name': [f'Node_{i}']})
            graph.add_nodes(df, f'Type_{i}', 'id', 'name')

        for i in range(20):
            assert graph.select(f'Type_{i}').len() == 1


class TestDeepTraversal:
    def test_deep_chain(self):
        graph = KnowledgeGraph()
        # Create a chain: A -> B -> C -> D -> E
        for i in range(5):
            df = pd.DataFrame({'id': [i], 'name': [f'Node_{i}']})
            graph.add_nodes(df, 'Chain', 'id', 'name')
        for i in range(4):
            conn = pd.DataFrame({'from_id': [i], 'to_id': [i + 1]})
            graph.add_connections(conn, 'NEXT', 'Chain', 'from_id', 'Chain', 'to_id')

        # Traverse 4 hops from start — each hop accumulates all reachable nodes
        start = graph.select('Chain').where({'id': 0})
        hop1 = start.traverse('NEXT')
        assert hop1.len() >= 1
        hop2 = hop1.traverse('NEXT')
        assert hop2.len() >= 1
        hop3 = hop2.traverse('NEXT')
        assert hop3.len() >= 1
        hop4 = hop3.traverse('NEXT')
        assert hop4.len() >= 1
