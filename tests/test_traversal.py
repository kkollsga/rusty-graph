"""Tests for graph traversal and connection property filtering."""

import pytest
import pandas as pd
from rusty_graph import KnowledgeGraph


class TestBasicTraversal:
    def test_traverse_outgoing(self, small_graph):
        alice = small_graph.type_filter('Person').filter({'title': 'Alice'})
        friends = alice.traverse(connection_type='KNOWS', direction='outgoing')
        assert friends.node_count() == 2  # Bob and Charlie

    def test_traverse_incoming(self, small_graph):
        charlie = small_graph.type_filter('Person').filter({'title': 'Charlie'})
        known_by = charlie.traverse(connection_type='KNOWS', direction='incoming')
        assert known_by.node_count() >= 1  # At least Alice and Bob know Charlie

    def test_traverse_cross_type(self, social_graph):
        person = social_graph.type_filter('Person').filter({'title': 'Person_1'})
        companies = person.traverse(connection_type='WORKS_AT', direction='outgoing')
        assert companies.node_count() >= 1

    def test_traverse_with_max_nodes(self, social_graph):
        # max_nodes limits per-parent, not globally — test with a single person
        person = social_graph.type_filter('Person').filter({'title': 'Person_1'})
        friends = person.traverse(connection_type='KNOWS', max_nodes=5)
        assert friends.node_count() <= 5


class TestTraversalFiltering:
    def test_filter_target(self, social_graph):
        people = social_graph.type_filter('Person')
        old_friends = people.traverse(
            connection_type='KNOWS',
            filter_target={'age': {'>': 35}},
        )
        nodes = old_friends.get_nodes()
        for n in nodes:
            assert n['age'] > 35

    def test_filter_connection_properties(self, petroleum_graph):
        prospects = petroleum_graph.type_filter('Prospect')
        high_share = prospects.traverse(
            connection_type='BECAME_DISCOVERY',
            filter_connection={'share_pct': {'>=': 70.0}},
        )
        assert high_share.node_count() > 0

    def test_filter_connection_null(self, petroleum_graph):
        prospects = petroleum_graph.type_filter('Prospect')
        with_weight = prospects.traverse(
            connection_type='HAS_ESTIMATE',
            filter_connection={'weight': {'is_not_null': True}},
        )
        assert with_weight.node_count() > 0

    def test_sort_target(self, social_graph):
        people = social_graph.type_filter('Person').filter({'title': 'Person_1'})
        friends = people.traverse(
            connection_type='KNOWS',
            sort_target='age',
        )
        nodes = friends.get_nodes()
        if len(nodes) > 1:
            ages = [n['age'] for n in nodes]
            assert ages == sorted(ages)


class TestTraversalChaining:
    def test_multi_hop_traversal(self, petroleum_graph):
        plays = petroleum_graph.type_filter('Play')
        prospects = plays.traverse(connection_type='HAS_PROSPECT')
        discoveries = prospects.traverse(connection_type='BECAME_DISCOVERY')
        assert discoveries.node_count() > 0

    def test_traverse_then_filter(self, social_graph):
        person = social_graph.type_filter('Person').filter({'title': 'Person_1'})
        friends = person.traverse(connection_type='KNOWS')
        old_friends = friends.filter({'age': {'>': 30}})
        nodes = old_friends.get_nodes()
        for n in nodes:
            assert n['age'] > 30
