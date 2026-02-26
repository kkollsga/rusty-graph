"""Tests for expand, to_subgraph, subgraph_stats."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


class TestExpand:
    def test_expand_single_hop(self, small_graph):
        alice = small_graph.select('Person').where({'title': 'Alice'})
        expanded = alice.expand(hops=1)
        assert expanded.len() >= 2  # Alice + at least Bob/Charlie

    def test_expand_multiple_hops(self, petroleum_graph):
        play = petroleum_graph.select('Play').where({'title': 'North Sea Play'})
        expanded = play.expand(hops=2)
        assert expanded.len() > 1

    def test_expand_does_not_include_isolated(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        conn_df = pd.DataFrame({'source': [1], 'target': [2]})
        graph.add_connections(conn_df, 'LINKS', 'Node', 'source', 'Node', 'target')

        start = graph.select('Node').where({'title': 'A'})
        expanded = start.expand(hops=1)
        # Should include A and B but not C (isolated from A)
        assert expanded.len() == 2

    def test_expand_from_empty(self, small_graph):
        empty = small_graph.select('NonExistent')
        expanded = empty.expand(hops=2)
        assert expanded.len() == 0


class TestSubgraph:
    def test_to_subgraph_basic(self, small_graph):
        alice = small_graph.select('Person').where({'title': 'Alice'})
        expanded = alice.expand(hops=1)
        subgraph = expanded.to_subgraph()
        assert subgraph.select('Person').len() >= 2

    def test_subgraph_is_independent(self, small_graph):
        selection = small_graph.select('Person')
        subgraph = selection.to_subgraph()
        # Subgraph should be a separate graph
        assert subgraph.select('Person').len() == 3

    def test_subgraph_stats(self, small_graph):
        selection = small_graph.select('Person')
        stats = selection.subgraph_stats()
        assert stats is not None

    def test_explain_shows_expand(self, small_graph):
        result = small_graph.select('Person').where({'title': 'Alice'}).expand(hops=1)
        explanation = result.explain()
        assert 'EXPAND' in explanation.upper() or 'expand' in explanation.lower()
