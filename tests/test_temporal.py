"""Tests for temporal queries: datetime columns, valid_at, valid_during."""

import pytest
import pandas as pd
from rusty_graph import KnowledgeGraph


class TestDateTimeHandling:
    def test_datetime_column_type(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'created': ['2020-01-01', '2021-06-15', '2022-12-31'],
        })
        graph.add_nodes(df, 'Item', 'id', 'name',
                        column_types={'created': 'datetime'})
        nodes = graph.type_filter('Item').get_nodes()
        assert len(nodes) == 3

    def test_datetime_comparison_filter(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'date': ['2020-01-01', '2021-06-15', '2022-12-31'],
        })
        graph.add_nodes(df, 'Item', 'id', 'name',
                        column_types={'date': 'datetime'})
        result = graph.type_filter('Item').filter({'date': {'>=': '2021-01-01'}})
        assert result.node_count() == 2


class TestValidAt:
    def test_valid_at_basic(self, petroleum_graph):
        result = petroleum_graph.type_filter('Estimate').valid_at('2020-06-15')
        assert result.node_count() > 0

    def test_valid_at_custom_fields(self, petroleum_graph):
        result = petroleum_graph.type_filter('Prospect').valid_at(
            '2020-06-15',
            date_from_field='date_from',
            date_to_field='date_to',
        )
        assert result.node_count() > 0

    def test_valid_at_no_matches(self, petroleum_graph):
        result = petroleum_graph.type_filter('Estimate').valid_at('2000-01-01')
        assert result.node_count() == 0


class TestValidDuring:
    def test_valid_during_basic(self, petroleum_graph):
        result = petroleum_graph.type_filter('Estimate').valid_during(
            '2020-01-01', '2020-06-30'
        )
        assert result.node_count() > 0

    def test_valid_during_partial_overlap(self, petroleum_graph):
        result = petroleum_graph.type_filter('Estimate').valid_during(
            '2020-10-01', '2021-03-31'
        )
        assert result.node_count() > 0

    def test_valid_during_no_overlap(self, petroleum_graph):
        result = petroleum_graph.type_filter('Estimate').valid_during(
            '2000-01-01', '2000-12-31'
        )
        assert result.node_count() == 0
