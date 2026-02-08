"""Tests for graph algorithms: shortest path, all paths, connected components, centrality."""

import pytest
import pandas as pd
from rusty_graph import KnowledgeGraph


class TestShortestPath:
    def test_shortest_path_basic(self, social_graph):
        path = social_graph.shortest_path(
            source_type='Person', source_id=1,
            target_type='Person', target_id=5,
        )
        assert path is not None
        assert len(path) >= 2

    def test_shortest_path_length(self, social_graph):
        length = social_graph.shortest_path_length(
            source_type='Person', source_id=1,
            target_type='Person', target_id=5,
        )
        assert length >= 1

    def test_shortest_path_ids(self, social_graph):
        ids = social_graph.shortest_path_ids(
            source_type='Person', source_id=1,
            target_type='Person', target_id=5,
        )
        assert len(ids) >= 2

    def test_shortest_path_indices(self, social_graph):
        indices = social_graph.shortest_path_indices(
            source_type='Person', source_id=1,
            target_type='Person', target_id=5,
        )
        assert len(indices) >= 2

    def test_shortest_path_not_found(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        # No connections, no path
        path = graph.shortest_path(
            source_type='Node', source_id=1,
            target_type='Node', target_id=2,
        )
        assert path is None or len(path) == 0

    def test_shortest_path_same_node(self, social_graph):
        path = social_graph.shortest_path(
            source_type='Person', source_id=1,
            target_type='Person', target_id=1,
        )
        # Path to self is either length 0 or 1 (just the node)
        assert path is not None


class TestAllPaths:
    def test_all_paths_basic(self, social_graph):
        paths = social_graph.all_paths(
            source_type='Person', source_id=1,
            target_type='Person', target_id=5,
            max_hops=4,
        )
        assert len(paths) >= 1

    def test_all_paths_limited_hops(self, small_graph):
        paths = small_graph.all_paths(
            source_type='Person', source_id=1,
            target_type='Person', target_id=3,
            max_hops=1,
        )
        # Direct path Alice->Charlie exists
        assert len(paths) >= 1


class TestConnectedComponents:
    def test_connected_components(self, social_graph):
        components = social_graph.connected_components()
        assert len(components) >= 1

    def test_are_connected(self, social_graph):
        result = social_graph.are_connected(
            source_type='Person', source_id=1,
            target_type='Person', target_id=5,
        )
        assert result is True

    def test_are_not_connected(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        result = graph.are_connected(
            source_type='Node', source_id=1,
            target_type='Node', target_id=2,
        )
        assert result is False


class TestCentrality:
    def test_betweenness_centrality(self, social_graph):
        result = social_graph.betweenness_centrality()
        assert result is not None
        assert len(result) > 0

    def test_degree_centrality(self, social_graph):
        result = social_graph.degree_centrality()
        assert result is not None

    def test_get_degrees(self, social_graph):
        # get_degrees needs a selection — use type_filter first
        degrees = social_graph.type_filter('Person').get_degrees()
        assert degrees is not None
        assert isinstance(degrees, dict)
        assert len(degrees) > 0

    def test_pagerank(self, social_graph):
        result = social_graph.pagerank()
        assert result is not None
        assert len(result) > 0

    def test_closeness_centrality(self, social_graph):
        result = social_graph.closeness_centrality()
        assert result is not None
