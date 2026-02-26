"""Tests for set operations: union, intersection, difference, symmetric_difference."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


class TestUnion:
    def test_union_basic(self, social_graph):
        oslo = social_graph.select('Person').where({'city': 'Oslo'})
        bergen = social_graph.select('Person').where({'city': 'Bergen'})
        combined = oslo.union(bergen)
        assert combined.len() == oslo.len() + bergen.len()

    def test_union_with_overlap(self, social_graph):
        young = social_graph.select('Person').where({'age': {'<': 30}})
        oslo = social_graph.select('Person').where({'city': 'Oslo'})
        combined = young.union(oslo)
        assert combined.len() >= max(young.len(), oslo.len())

    def test_union_with_self(self, social_graph):
        oslo = social_graph.select('Person').where({'city': 'Oslo'})
        combined = oslo.union(oslo)
        assert combined.len() == oslo.len()

    def test_union_with_empty(self, social_graph):
        oslo = social_graph.select('Person').where({'city': 'Oslo'})
        empty = social_graph.select('NonExistent')
        combined = oslo.union(empty)
        assert combined.len() == oslo.len()


class TestIntersection:
    def test_intersection_basic(self, social_graph):
        oslo = social_graph.select('Person').where({'city': 'Oslo'})
        old = social_graph.select('Person').where({'age': {'>': 35}})
        result = oslo.intersection(old)
        nodes = result.collect()
        for n in nodes:
            assert n['city'] == 'Oslo'
            assert n['age'] > 35

    def test_intersection_with_empty(self, social_graph):
        oslo = social_graph.select('Person').where({'city': 'Oslo'})
        empty = social_graph.select('NonExistent')
        result = oslo.intersection(empty)
        assert result.len() == 0


class TestDifference:
    def test_difference_basic(self, social_graph):
        all_people = social_graph.select('Person')
        oslo = social_graph.select('Person').where({'city': 'Oslo'})
        non_oslo = all_people.difference(oslo)
        nodes = non_oslo.collect()
        for n in nodes:
            assert n['city'] != 'Oslo'

    def test_difference_with_self(self, social_graph):
        oslo = social_graph.select('Person').where({'city': 'Oslo'})
        result = oslo.difference(oslo)
        assert result.len() == 0


class TestSymmetricDifference:
    def test_symmetric_difference(self, social_graph):
        oslo = social_graph.select('Person').where({'city': 'Oslo'})
        young = social_graph.select('Person').where({'age': {'<': 30}})
        result = oslo.symmetric_difference(young)
        # XOR: in one but not both
        intersection_count = oslo.intersection(young).len()
        expected = oslo.len() + young.len() - 2 * intersection_count
        assert result.len() == expected


class TestChaining:
    def test_union_then_intersection(self, social_graph):
        oslo = social_graph.select('Person').where({'city': 'Oslo'})
        bergen = social_graph.select('Person').where({'city': 'Bergen'})
        old = social_graph.select('Person').where({'age': {'>': 35}})
        result = oslo.union(bergen).intersection(old)
        assert result.len() >= 0
