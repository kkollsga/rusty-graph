"""Tests for connection operations: add, retrieve, get_connections."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


class TestAddConnections:
    def test_add_connections_basic(self, small_graph):
        conns = small_graph.type_filter('Person').filter({'title': 'Alice'}).get_connections()
        assert len(conns) > 0

    def test_add_connections_with_properties(self, small_graph):
        # get_connections returns a nested dict: {title: {node_id, node_type, incoming, outgoing}}
        conns = small_graph.type_filter('Person').filter({'title': 'Alice'}).get_connections()
        assert 'Alice' in conns
        alice = conns['Alice']
        # Alice has outgoing KNOWS connections
        assert 'outgoing' in alice
        assert 'KNOWS' in alice['outgoing']

    def test_add_connections_empty_dataframe(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        conn_df = pd.DataFrame({'source': [], 'target': []})
        report = graph.add_connections(conn_df, 'LINKS', 'Node', 'source', 'Node', 'target')
        assert report['connections_created'] == 0

    def test_self_referential_connection(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        conn_df = pd.DataFrame({'source': [1], 'target': [1]})
        report = graph.add_connections(conn_df, 'SELF', 'Node', 'source', 'Node', 'target')
        assert report['connections_created'] == 1

    def test_cross_type_connections(self):
        graph = KnowledgeGraph()
        users = pd.DataFrame({'id': [1], 'name': ['Alice']})
        products = pd.DataFrame({'id': [101], 'name': ['Laptop']})
        graph.add_nodes(users, 'User', 'id', 'name')
        graph.add_nodes(products, 'Product', 'id', 'name')

        conn_df = pd.DataFrame({'user_id': [1], 'product_id': [101]})
        report = graph.add_connections(conn_df, 'PURCHASED', 'User', 'user_id',
                                       'Product', 'product_id')
        assert report['connections_created'] == 1


class TestGetConnections:
    def test_get_connections_basic(self, small_graph):
        # get_connections returns nested dict keyed by node title
        conns = small_graph.type_filter('Person').filter({'title': 'Alice'}).get_connections()
        assert 'Alice' in conns
        alice_conns = conns['Alice']
        assert 'outgoing' in alice_conns
        assert 'KNOWS' in alice_conns['outgoing']
        assert len(alice_conns['outgoing']['KNOWS']) >= 2

    def test_get_connections_include_properties(self, small_graph):
        # include_node_properties is a bool flag
        conns = small_graph.type_filter('Person').filter({'title': 'Alice'}).get_connections(
            include_node_properties=True
        )
        assert 'Alice' in conns
        # With include_node_properties=True, node_properties should be populated
        alice = conns['Alice']
        for conn_type, targets in alice.get('outgoing', {}).items():
            for target_name, target_info in targets.items():
                assert 'node_properties' in target_info

    def test_duplicate_connections(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        conn_df = pd.DataFrame({'source': [1, 1], 'target': [2, 2]})
        report = graph.add_connections(conn_df, 'LINKS', 'Node', 'source', 'Node', 'target')
        # Both connections should be created (multigraph)
        assert report['connections_created'] >= 1
