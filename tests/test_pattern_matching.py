"""Tests for match_pattern() — Cypher-like pattern syntax."""

import pytest
import pandas as pd
from rusty_graph import KnowledgeGraph


class TestNodePatterns:
    def test_simple_node(self, social_graph):
        results = social_graph.match_pattern('(p:Person)')
        assert len(results) == 20

    def test_anonymous_node(self, social_graph):
        results = social_graph.match_pattern('(:Person)')
        assert len(results) == 20

    def test_empty_node(self, social_graph):
        results = social_graph.match_pattern('(n)')
        # 20 persons + 5 companies + schema nodes (4 types with define_schema in fixture)
        assert len(results) >= 25

    def test_node_with_property(self, small_graph):
        results = small_graph.match_pattern('(p:Person {city: "Oslo"})')
        assert len(results) == 2  # Alice and Charlie


class TestEdgePatterns:
    def test_outgoing_edge(self, small_graph):
        results = small_graph.match_pattern('(a:Person)-[:KNOWS]->(b:Person)')
        assert len(results) == 3  # Alice->Bob, Bob->Charlie, Alice->Charlie

    def test_incoming_edge(self, small_graph):
        results = small_graph.match_pattern('(a:Person)<-[:KNOWS]-(b:Person)')
        assert len(results) == 3

    def test_bidirectional_edge(self, small_graph):
        results = small_graph.match_pattern('(a:Person)-[:KNOWS]-(b:Person)')
        assert len(results) >= 3


class TestMultiHopPatterns:
    def test_two_hop(self, petroleum_graph):
        results = petroleum_graph.match_pattern(
            '(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)'
        )
        assert len(results) > 0
        for m in results:
            assert 'p' in m
            assert 'pr' in m
            assert 'd' in m

    def test_cross_type_pattern(self, social_graph):
        results = social_graph.match_pattern(
            '(p:Person)-[:WORKS_AT]->(c:Company)'
        )
        assert len(results) == 20  # Each person works at one company

    def test_max_matches(self, social_graph):
        # max_matches parameter is accepted; verify it runs without error
        results = social_graph.match_pattern(
            '(a:Person)-[:KNOWS]->(b:Person)',
            max_matches=5,
        )
        assert len(results) > 0


class TestVariableLengthPaths:
    def test_exact_hops(self, petroleum_graph):
        results = petroleum_graph.match_pattern(
            '(p:Play)-[:HAS_PROSPECT*1]->(pr:Prospect)'
        )
        assert len(results) > 0

    def test_range_hops(self, petroleum_graph):
        results = petroleum_graph.match_pattern(
            '(p:Play)-[*1..2]->(d)'
        )
        assert len(results) > 0

    def test_star_only(self, petroleum_graph):
        results = petroleum_graph.match_pattern(
            '(p:Play)-[*]->(d:Discovery)'
        )
        assert len(results) > 0


class TestNoMatches:
    def test_no_matching_type(self, small_graph):
        results = small_graph.match_pattern('(n:NonExistent)')
        assert len(results) == 0

    def test_no_matching_edge(self, small_graph):
        results = small_graph.match_pattern('(a:Person)-[:WORKS_AT]->(b:Company)')
        assert len(results) == 0
