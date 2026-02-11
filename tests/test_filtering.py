"""Tests for filtering, sorting, limiting, null checks, orphan filtering."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


class TestBasicFiltering:
    def test_filter_exact_match(self, small_graph):
        result = small_graph.type_filter('Person').filter({'age': 28})
        assert result.node_count() == 1

    def test_filter_greater_than(self, social_graph):
        result = social_graph.type_filter('Person').filter({'age': {'>': 30}})
        assert result.node_count() > 0
        nodes = result.get_nodes()
        for n in nodes:
            assert n['age'] > 30

    def test_filter_less_than(self, social_graph):
        result = social_graph.type_filter('Person').filter({'age': {'<': 25}})
        nodes = result.get_nodes()
        for n in nodes:
            assert n['age'] < 25

    def test_filter_greater_equal(self, social_graph):
        result = social_graph.type_filter('Person').filter({'age': {'>=': 40}})
        nodes = result.get_nodes()
        for n in nodes:
            assert n['age'] >= 40

    def test_filter_less_equal(self, social_graph):
        result = social_graph.type_filter('Person').filter({'age': {'<=': 22}})
        nodes = result.get_nodes()
        for n in nodes:
            assert n['age'] <= 22

    def test_filter_multiple_conditions(self, social_graph):
        result = social_graph.type_filter('Person').filter({
            'age': {'>': 25},
            'city': 'Oslo',
        })
        nodes = result.get_nodes()
        for n in nodes:
            assert n['age'] > 25
            assert n['city'] == 'Oslo'

    def test_filter_in_operator(self, social_graph):
        result = social_graph.type_filter('Person').filter({
            'city': {'in': ['Oslo', 'Bergen']},
        })
        nodes = result.get_nodes()
        for n in nodes:
            assert n['city'] in ['Oslo', 'Bergen']

    def test_filter_no_matches(self, small_graph):
        result = small_graph.type_filter('Person').filter({'title': 'NonExistent'})
        assert result.node_count() == 0

    def test_filter_chained(self, social_graph):
        result = (social_graph.type_filter('Person')
                  .filter({'city': 'Oslo'})
                  .filter({'age': {'>': 23}}))
        nodes = result.get_nodes()
        for n in nodes:
            assert n['city'] == 'Oslo'
            assert n['age'] > 23


class TestNullFiltering:
    def test_is_null(self, social_graph):
        result = social_graph.type_filter('Person').filter({'email': {'is_null': True}})
        assert result.node_count() > 0

    def test_is_not_null(self, social_graph):
        result = social_graph.type_filter('Person').filter({'email': {'is_not_null': True}})
        assert result.node_count() > 0

    def test_null_and_not_null_partition(self, social_graph):
        null_count = social_graph.type_filter('Person').filter(
            {'email': {'is_null': True}}).node_count()
        not_null_count = social_graph.type_filter('Person').filter(
            {'email': {'is_not_null': True}}).node_count()
        total = social_graph.type_filter('Person').node_count()
        assert null_count + not_null_count == total

    def test_filter_on_missing_property(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        result = graph.type_filter('Node').filter({'nonexistent': {'is_null': True}})
        assert result.node_count() == 2


class TestStringPredicates:
    def test_filter_contains(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Alice Smith', 'Bob Jones', 'Carol Smith', 'Dave Lee'],
        })
        graph.add_nodes(df, 'Person', 'id', 'name')
        result = graph.type_filter('Person').filter({'title': {'contains': 'Smith'}})
        assert result.node_count() == 2
        for n in result.get_nodes():
            assert 'Smith' in n['title']

    def test_filter_starts_with(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alpha', 'Beta', 'Alphabet'],
        })
        graph.add_nodes(df, 'Item', 'id', 'name')
        result = graph.type_filter('Item').filter({'title': {'starts_with': 'Alp'}})
        assert result.node_count() == 2
        for n in result.get_nodes():
            assert n['title'].startswith('Alp')

    def test_filter_ends_with(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['report.csv', 'data.json', 'output.csv'],
        })
        graph.add_nodes(df, 'File', 'id', 'name')
        result = graph.type_filter('File').filter({'title': {'ends_with': '.csv'}})
        assert result.node_count() == 2
        for n in result.get_nodes():
            assert n['title'].endswith('.csv')

    def test_filter_contains_no_match(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        graph.add_nodes(df, 'Person', 'id', 'name')
        result = graph.type_filter('Person').filter({'title': {'contains': 'xyz'}})
        assert result.node_count() == 0

    def test_filter_string_predicates_combined(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Alice Smith', 'Bob Jones', 'Alice Jones', 'Carol Smith'],
            'email': ['alice@test.com', 'bob@test.com', 'alice2@other.com', 'carol@test.com'],
        })
        graph.add_nodes(df, 'Person', 'id', 'name')
        # Combine starts_with with another filter
        result = graph.type_filter('Person').filter({
            'title': {'starts_with': 'Alice'},
        })
        assert result.node_count() == 2


class TestSorting:
    def test_sort_ascending(self, social_graph):
        nodes = social_graph.type_filter('Person').sort('age').get_nodes()
        ages = [n['age'] for n in nodes]
        assert ages == sorted(ages)

    def test_sort_descending(self, social_graph):
        nodes = social_graph.type_filter('Person').sort('age', ascending=False).get_nodes()
        ages = [n['age'] for n in nodes]
        assert ages == sorted(ages, reverse=True)

    def test_sort_multi_field(self, social_graph):
        nodes = social_graph.type_filter('Person').sort([
            ('city', True), ('age', True)
        ]).get_nodes()
        for i in range(len(nodes) - 1):
            if nodes[i]['city'] == nodes[i + 1]['city']:
                assert nodes[i]['age'] <= nodes[i + 1]['age']


class TestLimiting:
    def test_max_nodes(self, social_graph):
        result = social_graph.type_filter('Person').max_nodes(5)
        assert result.node_count() == 5

    def test_max_nodes_more_than_total(self, small_graph):
        result = small_graph.type_filter('Person').max_nodes(100)
        assert result.node_count() == 3


class TestOrphanFiltering:
    def test_filter_orphans_include(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        conn_df = pd.DataFrame({'source': [1], 'target': [2]})
        graph.add_connections(conn_df, 'LINKS', 'Node', 'source', 'Node', 'target')

        orphans = graph.type_filter('Node').filter_orphans(include_orphans=True)
        assert orphans.node_count() == 1  # Node 3 is orphan

    def test_filter_orphans_exclude(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        conn_df = pd.DataFrame({'source': [1], 'target': [2]})
        graph.add_connections(conn_df, 'LINKS', 'Node', 'source', 'Node', 'target')

        connected = graph.type_filter('Node').filter_orphans(include_orphans=False)
        assert connected.node_count() == 2  # Nodes 1 and 2
