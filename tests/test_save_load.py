"""Tests for save/load persistence."""

import pytest
import tempfile
import os
import pandas as pd
from rusty_graph import KnowledgeGraph
import rusty_graph


class TestBasicSaveLoad:
    def test_save_and_load(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name
        try:
            small_graph.save(path)
            loaded = rusty_graph.load(path)
            assert loaded.type_filter('Person').node_count() == 3
        finally:
            os.unlink(path)

    def test_save_load_preserves_properties(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name
        try:
            small_graph.save(path)
            loaded = rusty_graph.load(path)
            alice = loaded.get_node_by_id('Person', 1)
            assert alice['title'] == 'Alice'
            assert alice['age'] == 28
        finally:
            os.unlink(path)

    def test_save_load_preserves_connections(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name
        try:
            small_graph.save(path)
            loaded = rusty_graph.load(path)
            alice = loaded.type_filter('Person').filter({'title': 'Alice'})
            friends = alice.traverse(connection_type='KNOWS', direction='outgoing')
            assert friends.node_count() == 2
        finally:
            os.unlink(path)


class TestSaveLoadWithFeatures:
    def test_save_load_with_indexes(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': list(range(10)),
            'name': [f'N_{i}' for i in range(10)],
            'cat': [f'C_{i % 3}' for i in range(10)],
        })
        graph.add_nodes(df, 'Node', 'id', 'name')
        graph.create_index('Node', 'cat')

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = rusty_graph.load(path)
            assert loaded.has_index('Node', 'cat')
        finally:
            os.unlink(path)

    def test_save_load_with_schema(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        graph.define_schema({'nodes': {'Node': {'required': ['id', 'title']}}})

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = rusty_graph.load(path)
            assert loaded.has_schema()
        finally:
            os.unlink(path)

    def test_save_load_large(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': list(range(1000)),
            'name': [f'Node_{i}' for i in range(1000)],
            'value': list(range(1000)),
        })
        graph.add_nodes(df, 'LargeType', 'id', 'name')

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = rusty_graph.load(path)
            assert loaded.type_filter('LargeType').node_count() == 1000
        finally:
            os.unlink(path)
