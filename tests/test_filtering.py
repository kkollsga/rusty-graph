"""Tests for filtering, sorting, limiting, null checks, orphan filtering."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


class TestBasicFiltering:
    def test_filter_exact_match(self, small_graph):
        result = small_graph.select('Person').where({'age': 28})
        assert result.len() == 1

    def test_filter_greater_than(self, social_graph):
        result = social_graph.select('Person').where({'age': {'>': 30}})
        assert result.len() > 0
        nodes = result.collect()
        for n in nodes:
            assert n['age'] > 30

    def test_filter_less_than(self, social_graph):
        result = social_graph.select('Person').where({'age': {'<': 25}})
        nodes = result.collect()
        for n in nodes:
            assert n['age'] < 25

    def test_filter_greater_equal(self, social_graph):
        result = social_graph.select('Person').where({'age': {'>=': 40}})
        nodes = result.collect()
        for n in nodes:
            assert n['age'] >= 40

    def test_filter_less_equal(self, social_graph):
        result = social_graph.select('Person').where({'age': {'<=': 22}})
        nodes = result.collect()
        for n in nodes:
            assert n['age'] <= 22

    def test_filter_multiple_conditions(self, social_graph):
        result = social_graph.select('Person').where({
            'age': {'>': 25},
            'city': 'Oslo',
        })
        nodes = result.collect()
        for n in nodes:
            assert n['age'] > 25
            assert n['city'] == 'Oslo'

    def test_filter_in_operator(self, social_graph):
        result = social_graph.select('Person').where({
            'city': {'in': ['Oslo', 'Bergen']},
        })
        nodes = result.collect()
        for n in nodes:
            assert n['city'] in ['Oslo', 'Bergen']

    def test_filter_no_matches(self, small_graph):
        result = small_graph.select('Person').where({'title': 'NonExistent'})
        assert result.len() == 0

    def test_filter_chained(self, social_graph):
        result = (social_graph.select('Person')
                  .where({'city': 'Oslo'})
                  .where({'age': {'>': 23}}))
        nodes = result.collect()
        for n in nodes:
            assert n['city'] == 'Oslo'
            assert n['age'] > 23


class TestNullFiltering:
    def test_is_null(self, social_graph):
        result = social_graph.select('Person').where({'email': {'is_null': True}})
        assert result.len() > 0

    def test_is_not_null(self, social_graph):
        result = social_graph.select('Person').where({'email': {'is_not_null': True}})
        assert result.len() > 0

    def test_null_and_not_null_partition(self, social_graph):
        null_count = social_graph.select('Person').where(
            {'email': {'is_null': True}}).len()
        not_null_count = social_graph.select('Person').where(
            {'email': {'is_not_null': True}}).len()
        total = social_graph.select('Person').len()
        assert null_count + not_null_count == total

    def test_filter_on_missing_property(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        result = graph.select('Node').where({'nonexistent': {'is_null': True}})
        assert result.len() == 2


class TestStringPredicates:
    def test_filter_contains(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Alice Smith', 'Bob Jones', 'Carol Smith', 'Dave Lee'],
        })
        graph.add_nodes(df, 'Person', 'id', 'name')
        result = graph.select('Person').where({'title': {'contains': 'Smith'}})
        assert result.len() == 2
        for n in result.collect():
            assert 'Smith' in n['title']

    def test_filter_starts_with(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alpha', 'Beta', 'Alphabet'],
        })
        graph.add_nodes(df, 'Item', 'id', 'name')
        result = graph.select('Item').where({'title': {'starts_with': 'Alp'}})
        assert result.len() == 2
        for n in result.collect():
            assert n['title'].startswith('Alp')

    def test_filter_ends_with(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['report.csv', 'data.json', 'output.csv'],
        })
        graph.add_nodes(df, 'File', 'id', 'name')
        result = graph.select('File').where({'title': {'ends_with': '.csv'}})
        assert result.len() == 2
        for n in result.collect():
            assert n['title'].endswith('.csv')

    def test_filter_contains_no_match(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        graph.add_nodes(df, 'Person', 'id', 'name')
        result = graph.select('Person').where({'title': {'contains': 'xyz'}})
        assert result.len() == 0

    def test_filter_string_predicates_combined(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Alice Smith', 'Bob Jones', 'Alice Jones', 'Carol Smith'],
            'email': ['alice@test.com', 'bob@test.com', 'alice2@other.com', 'carol@test.com'],
        })
        graph.add_nodes(df, 'Person', 'id', 'name')
        # Combine starts_with with another filter
        result = graph.select('Person').where({
            'title': {'starts_with': 'Alice'},
        })
        assert result.len() == 2


class TestSorting:
    def test_sort_ascending(self, social_graph):
        nodes = social_graph.select('Person').sort('age').collect()
        ages = [n['age'] for n in nodes]
        assert ages == sorted(ages)

    def test_sort_descending(self, social_graph):
        nodes = social_graph.select('Person').sort('age', ascending=False).collect()
        ages = [n['age'] for n in nodes]
        assert ages == sorted(ages, reverse=True)

    def test_sort_multi_field(self, social_graph):
        nodes = social_graph.select('Person').sort([
            ('city', True), ('age', True)
        ]).collect()
        for i in range(len(nodes) - 1):
            if nodes[i]['city'] == nodes[i + 1]['city']:
                assert nodes[i]['age'] <= nodes[i + 1]['age']


class TestLimiting:
    def test_max_nodes(self, social_graph):
        result = social_graph.select('Person').limit(5)
        assert result.len() == 5

    def test_max_nodes_more_than_total(self, small_graph):
        result = small_graph.select('Person').limit(100)
        assert result.len() == 3


class TestOrphanFiltering:
    def test_filter_orphans_include(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        conn_df = pd.DataFrame({'source': [1], 'target': [2]})
        graph.add_connections(conn_df, 'LINKS', 'Node', 'source', 'Node', 'target')

        orphans = graph.select('Node').where_orphans(include_orphans=True)
        assert orphans.len() == 1  # Node 3 is orphan

    def test_filter_orphans_exclude(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        conn_df = pd.DataFrame({'source': [1], 'target': [2]})
        graph.add_connections(conn_df, 'LINKS', 'Node', 'source', 'Node', 'target')

        connected = graph.select('Node').where_orphans(include_orphans=False)
        assert connected.len() == 2  # Nodes 1 and 2
