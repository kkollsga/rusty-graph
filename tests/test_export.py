"""Tests for export: GraphML, GEXF, D3 JSON, CSV, export_string."""

import pytest
import tempfile
import os
import json
import pandas as pd
from kglite import KnowledgeGraph


class TestExportToFile:
    def test_export_graphml(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as f:
            path = f.name
        try:
            small_graph.export(path, format='graphml')
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_export_gexf(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix='.gexf', delete=False) as f:
            path = f.name
        try:
            small_graph.export(path, format='gexf')
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_export_d3_json(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            small_graph.export(path, format='d3')
            assert os.path.exists(path)
            with open(path) as fh:
                data = json.load(fh)
            assert 'nodes' in data
            assert 'links' in data
        finally:
            os.unlink(path)

    def test_export_json(self, small_graph):
        """Export as JSON (alias for d3)."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            small_graph.export(path, format='json')
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


class TestExportString:
    def test_graphml_string(self, small_graph):
        result = small_graph.export_string(format='graphml')
        assert isinstance(result, str)
        assert len(result) > 0
        assert 'graphml' in result.lower()

    def test_d3_string(self, small_graph):
        result = small_graph.export_string(format='d3')
        data = json.loads(result)
        assert 'nodes' in data
        assert 'links' in data

    def test_export_empty_graph(self):
        graph = KnowledgeGraph()
        result = graph.export_string(format='graphml')
        assert isinstance(result, str)


class TestExportStringFormats:
    """Ensure all string export formats produce valid output."""

    def test_gexf_string(self, small_graph):
        result = small_graph.export_string(format='gexf')
        assert isinstance(result, str)
        assert 'gexf' in result.lower()

    def test_graphml_contains_nodes(self, small_graph):
        result = small_graph.export_string(format='graphml')
        assert '<node' in result
        assert '<edge' in result

    def test_d3_node_count(self, small_graph):
        result = small_graph.export_string(format='d3')
        data = json.loads(result)
        assert len(data['nodes']) >= 2

    def test_special_chars_in_export(self):
        """Properties with special XML characters don't break export."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1],
            'name': ["O'Brien & <Co>"],
            'desc': ['Has "quotes"'],
        })
        graph.add_nodes(df, 'T', 'id', 'name')
        # Should not crash
        graphml = graph.export_string(format='graphml')
        assert isinstance(graphml, str)
        d3 = graph.export_string(format='d3')
        data = json.loads(d3)
        assert len(data['nodes']) == 1


class TestExportWithSelection:
    def test_export_selection_only(self, social_graph):
        selection = social_graph.type_filter('Person').filter({'city': 'Oslo'})
        result = selection.export_string(format='d3', selection_only=True)
        data = json.loads(result)
        assert len(data['nodes']) > 0

    def test_export_expanded_selection(self, small_graph):
        expanded = small_graph.type_filter('Person').filter({'title': 'Alice'}).expand(hops=1)
        subgraph = expanded.to_subgraph()
        result = subgraph.export_string(format='d3')
        data = json.loads(result)
        assert len(data['nodes']) >= 2
