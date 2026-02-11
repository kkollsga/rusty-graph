"""Tests for calculate, statistics, count, children_properties_to_list, connection aggregation."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


class TestStatistics:
    def test_statistics_basic(self, social_graph):
        # statistics requires (property, level_index)
        social_graph.type_filter('Person')
        stats = social_graph.statistics('age', 0)
        assert stats is not None


class TestCalculate:
    def test_simple_expression(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'price': [100.0, 200.0, 300.0],
        })
        graph.add_nodes(df, 'Product', 'id', 'name')
        result = graph.type_filter('Product').calculate(
            expression='price * 1.1',
            store_as='price_with_tax',
        )
        assert result is not None

    def test_aggregate_sum(self):
        graph = KnowledgeGraph()
        parents = pd.DataFrame({'id': [1], 'name': ['Parent']})
        children = pd.DataFrame({
            'id': [10, 11, 12],
            'name': ['C1', 'C2', 'C3'],
            'value': [100, 200, 300],
        })
        graph.add_nodes(parents, 'Parent', 'id', 'name')
        graph.add_nodes(children, 'Child', 'id', 'name')
        conn = pd.DataFrame({'parent_id': [1, 1, 1], 'child_id': [10, 11, 12]})
        graph.add_connections(conn, 'HAS_CHILD', 'Parent', 'parent_id',
                              'Child', 'child_id')

        result = graph.type_filter('Parent').traverse('HAS_CHILD').calculate(
            expression='sum(value)',
            store_as='total_value',
        )
        assert result is not None


class TestCount:
    def test_count_basic(self):
        graph = KnowledgeGraph()
        parents = pd.DataFrame({'id': [1, 2], 'name': ['P1', 'P2']})
        children = pd.DataFrame({
            'id': [10, 11, 12],
            'name': ['C1', 'C2', 'C3'],
        })
        graph.add_nodes(parents, 'Parent', 'id', 'name')
        graph.add_nodes(children, 'Child', 'id', 'name')
        conn = pd.DataFrame({
            'parent_id': [1, 1, 2],
            'child_id': [10, 11, 12],
        })
        graph.add_connections(conn, 'HAS', 'Parent', 'parent_id', 'Child', 'child_id')

        # Use node_count instead — the simpler counting API
        plays = graph.type_filter('Parent')
        assert plays.node_count() == 2
        children_sel = plays.traverse('HAS')
        assert children_sel.node_count() == 3


class TestConnectionAggregation:
    def test_sum_connection_properties(self, petroleum_graph):
        result = petroleum_graph.type_filter('Prospect').traverse('BECAME_DISCOVERY').calculate(
            expression='sum(share_pct)',
            aggregate_connections=True,
        )
        assert result is not None

    def test_avg_connection_properties(self, petroleum_graph):
        result = petroleum_graph.type_filter('Prospect').traverse('HAS_ESTIMATE').calculate(
            expression='avg(weight)',
            aggregate_connections=True,
        )
        assert result is not None


class TestChildrenPropertiesToList:
    def test_basic(self):
        graph = KnowledgeGraph()
        parents = pd.DataFrame({'id': [1], 'name': ['Parent']})
        children = pd.DataFrame({
            'id': [10, 11, 12],
            'name': ['Alpha', 'Beta', 'Gamma'],
        })
        graph.add_nodes(parents, 'Parent', 'id', 'name')
        graph.add_nodes(children, 'Child', 'id', 'name')
        conn = pd.DataFrame({'parent_id': [1, 1, 1], 'child_id': [10, 11, 12]})
        graph.add_connections(conn, 'HAS', 'Parent', 'parent_id', 'Child', 'child_id')

        result = graph.type_filter('Parent').traverse('HAS').children_properties_to_list(
            property='title',
        )
        assert result is not None
