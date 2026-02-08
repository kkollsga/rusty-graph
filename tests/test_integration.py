"""Cross-feature integration tests."""

import pytest
import pandas as pd
import tempfile
import os
from rusty_graph import KnowledgeGraph
import rusty_graph


class TestCrossFeatureWorkflows:
    def test_temporal_with_spatial(self, petroleum_graph):
        """Combine temporal and spatial queries."""
        valid = petroleum_graph.type_filter('Prospect').valid_at(
            '2020-06-15',
            date_from_field='date_from',
            date_to_field='date_to',
        )
        nearby = valid.within_bounds(
            lat_field='latitude', lon_field='longitude',
            min_lat=59.0, max_lat=65.0, min_lon=3.0, max_lon=8.0,
        )
        assert nearby.node_count() > 0

    def test_pattern_match_with_filter(self, petroleum_graph):
        """Pattern match then apply additional filtering."""
        matches = petroleum_graph.match_pattern(
            '(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)'
        )
        assert len(matches) > 0
        for m in matches:
            assert 'p' in m
            assert 'pr' in m

    def test_index_with_set_operations(self, social_graph):
        """Use index for fast filtering combined with set operations."""
        social_graph.create_index('Person', 'city')
        oslo = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        bergen = social_graph.type_filter('Person').filter({'city': 'Bergen'})
        combined = oslo.union(bergen)
        assert combined.node_count() == oslo.node_count() + bergen.node_count()

    def test_subgraph_with_export(self, small_graph):
        """Extract subgraph and export it."""
        selection = small_graph.type_filter('Person').filter({'title': 'Alice'})
        expanded = selection.expand(hops=1)
        subgraph = expanded.to_subgraph()
        result = subgraph.export_string(format='d3')
        assert '"nodes"' in result

    def test_schema_with_updates(self):
        """Define schema, add data, validate, then update."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B'], 'status': ['active', 'active']})
        graph.add_nodes(df, 'Item', 'id', 'name')

        graph.define_schema({
            'nodes': {'Item': {'required': ['id', 'title', 'status']}}
        })
        errors = graph.validate_schema()
        assert len(errors) == 0

        result = graph.type_filter('Item').update({'status': 'archived'})
        assert result['nodes_updated'] == 2

    def test_cypher_with_pattern_match_parity(self, social_graph):
        """Verify cypher and match_pattern produce consistent results."""
        pattern_results = social_graph.match_pattern(
            '(a:Person)-[:WORKS_AT]->(c:Company)'
        )
        cypher_result = social_graph.cypher(
            "MATCH (a:Person)-[:WORKS_AT]->(c:Company) RETURN a.title, c.title"
        )
        assert len(pattern_results) == len(cypher_result['rows'])

    def test_full_pipeline(self, petroleum_graph):
        """Full workflow: filter -> traverse -> aggregate -> export."""
        plays = petroleum_graph.type_filter('Play')
        prospects = plays.traverse('HAS_PROSPECT')
        assert prospects.node_count() > 0

        explanation = prospects.explain()
        assert isinstance(explanation, str)

        subgraph = prospects.to_subgraph()
        export = subgraph.export_string(format='d3')
        assert len(export) > 0

    def test_save_load_roundtrip_with_features(self):
        """Full roundtrip: create graph with features, save, load, verify."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': list(range(10)),
            'name': [f'N_{i}' for i in range(10)],
            'cat': [f'C_{i % 3}' for i in range(10)],
        })
        graph.add_nodes(df, 'Node', 'id', 'name')
        graph.create_index('Node', 'cat')
        graph.define_schema({'nodes': {'Node': {'required': ['id', 'title']}}})

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = rusty_graph.load(path)
            assert loaded.type_filter('Node').node_count() == 10
            assert loaded.has_index('Node', 'cat')
            assert loaded.has_schema()
        finally:
            os.unlink(path)
