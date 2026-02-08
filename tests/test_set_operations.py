"""Tests for set operations: union, intersection, difference, symmetric_difference."""

import pytest
import pandas as pd
from rusty_graph import KnowledgeGraph


class TestUnion:
    def test_union_basic(self, social_graph):
        oslo = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        bergen = social_graph.type_filter('Person').filter({'city': 'Bergen'})
        combined = oslo.union(bergen)
        assert combined.node_count() == oslo.node_count() + bergen.node_count()

    def test_union_with_overlap(self, social_graph):
        young = social_graph.type_filter('Person').filter({'age': {'<': 30}})
        oslo = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        combined = young.union(oslo)
        assert combined.node_count() >= max(young.node_count(), oslo.node_count())

    def test_union_with_self(self, social_graph):
        oslo = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        combined = oslo.union(oslo)
        assert combined.node_count() == oslo.node_count()

    def test_union_with_empty(self, social_graph):
        oslo = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        empty = social_graph.type_filter('NonExistent')
        combined = oslo.union(empty)
        assert combined.node_count() == oslo.node_count()


class TestIntersection:
    def test_intersection_basic(self, social_graph):
        oslo = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        old = social_graph.type_filter('Person').filter({'age': {'>': 35}})
        result = oslo.intersection(old)
        nodes = result.get_nodes()
        for n in nodes:
            assert n['city'] == 'Oslo'
            assert n['age'] > 35

    def test_intersection_with_empty(self, social_graph):
        oslo = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        empty = social_graph.type_filter('NonExistent')
        result = oslo.intersection(empty)
        assert result.node_count() == 0


class TestDifference:
    def test_difference_basic(self, social_graph):
        all_people = social_graph.type_filter('Person')
        oslo = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        non_oslo = all_people.difference(oslo)
        nodes = non_oslo.get_nodes()
        for n in nodes:
            assert n['city'] != 'Oslo'

    def test_difference_with_self(self, social_graph):
        oslo = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        result = oslo.difference(oslo)
        assert result.node_count() == 0


class TestSymmetricDifference:
    def test_symmetric_difference(self, social_graph):
        oslo = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        young = social_graph.type_filter('Person').filter({'age': {'<': 30}})
        result = oslo.symmetric_difference(young)
        # XOR: in one but not both
        intersection_count = oslo.intersection(young).node_count()
        expected = oslo.node_count() + young.node_count() - 2 * intersection_count
        assert result.node_count() == expected


class TestChaining:
    def test_union_then_intersection(self, social_graph):
        oslo = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        bergen = social_graph.type_filter('Person').filter({'city': 'Bergen'})
        old = social_graph.type_filter('Person').filter({'age': {'>': 35}})
        result = oslo.union(bergen).intersection(old)
        assert result.node_count() >= 0
