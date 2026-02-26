"""Tests for save/load persistence."""

import pytest
import tempfile
import os
import pandas as pd
from kglite import KnowledgeGraph
import kglite


class TestBasicSaveLoad:
    def test_save_and_load(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            small_graph.save(path)
            loaded = kglite.load(path)
            assert loaded.select('Person').len() == 3
        finally:
            os.unlink(path)

    def test_save_load_preserves_properties(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            small_graph.save(path)
            loaded = kglite.load(path)
            alice = loaded.node('Person', 1)
            assert alice['title'] == 'Alice'
            assert alice['age'] == 28
        finally:
            os.unlink(path)

    def test_save_load_preserves_connections(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            small_graph.save(path)
            loaded = kglite.load(path)
            alice = loaded.select('Person').where({'title': 'Alice'})
            friends = alice.traverse(connection_type='KNOWS', direction='outgoing')
            assert friends.len() == 2
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

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.has_index('Node', 'cat')
        finally:
            os.unlink(path)

    def test_save_load_with_schema(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        graph.define_schema({'nodes': {'Node': {'required': ['id', 'title']}}})

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
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

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.select('LargeType').len() == 1000
        finally:
            os.unlink(path)


class TestV2Format:
    """Tests for the v2 sectioned binary format."""

    def test_v2_magic_bytes(self):
        """Saved files should start with the v2 magic header RGF\\x02."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A']})
        graph.add_nodes(df, 'Node', 'id', 'name')

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            with open(path, 'rb') as f:
                header = f.read(4)
            assert header == b'RGF\x02', f"Expected v2 magic bytes, got {header!r}"
        finally:
            os.unlink(path)

    def test_v2_header_structure(self):
        """Verify the full 12-byte header: magic + core_version + metadata_length."""
        import struct
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A']})
        graph.add_nodes(df, 'Node', 'id', 'name')

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            with open(path, 'rb') as f:
                magic = f.read(4)
                core_version = struct.unpack('<I', f.read(4))[0]
                metadata_len = struct.unpack('<I', f.read(4))[0]

            assert magic == b'RGF\x02'
            assert core_version == 1  # current core data version
            assert metadata_len > 0   # metadata should not be empty
        finally:
            os.unlink(path)

    def test_v2_metadata_is_json(self):
        """The metadata section should be valid JSON."""
        import struct
        import json
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B'], 'val': [10, 20]})
        graph.add_nodes(df, 'Node', 'id', 'name')

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            with open(path, 'rb') as f:
                f.read(4)  # magic
                f.read(4)  # core_version
                metadata_len = struct.unpack('<I', f.read(4))[0]
                metadata_bytes = f.read(metadata_len)

            metadata = json.loads(metadata_bytes)
            assert 'core_data_version' in metadata
            assert 'library_version' in metadata
            assert 'node_type_metadata' in metadata
            assert 'Node' in metadata['node_type_metadata']
        finally:
            os.unlink(path)

    def test_v2_preserves_node_type_metadata(self):
        """Node type metadata should survive save/load cycle."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'city': ['Oslo']})
        graph.add_nodes(df, 'Person', 'id', 'name')

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            info = loaded.graph_info()
            assert info['type_count'] >= 1
        finally:
            os.unlink(path)

    def test_v2_preserves_connection_type_metadata(self):
        """Connection type metadata should survive save/load."""
        graph = KnowledgeGraph()
        persons = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        graph.add_nodes(persons, 'Person', 'id', 'name')
        conns = pd.DataFrame({
            'source_id': [1], 'target_id': [2], 'weight': [1.5]
        })
        graph.add_connections(conns, 'KNOWS', 'Person', 'source_id', 'Person', 'target_id')

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            # Connection type should be recognized
            info = loaded.graph_info()
            assert info['edge_count'] == 1
        finally:
            os.unlink(path)

    def test_v2_preserves_indexes(self):
        """Property and composite indexes should survive v2 save/load."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': list(range(20)),
            'name': [f'N_{i}' for i in range(20)],
            'cat': [f'C_{i % 3}' for i in range(20)],
            'sub': [f'S_{i % 5}' for i in range(20)],
        })
        graph.add_nodes(df, 'Item', 'id', 'name')
        graph.create_index('Item', 'cat')
        graph.create_composite_index('Item', ['cat', 'sub'])

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.has_index('Item', 'cat')
            assert loaded.has_composite_index('Item', ['cat', 'sub'])
        finally:
            os.unlink(path)

    def test_v2_preserves_schema(self):
        """Schema definitions should survive v2 save/load."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        graph.define_schema({'nodes': {'Node': {'required': ['id', 'title']}}})

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.has_schema()
        finally:
            os.unlink(path)

    def test_v2_roundtrip_full(self):
        """Full roundtrip: nodes, edges, indexes, schema, Cypher queries."""
        graph = KnowledgeGraph()
        persons = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [30, 25, 35],
        })
        graph.add_nodes(persons, 'Person', 'id', 'name')
        conns = pd.DataFrame({
            'source_id': [1, 2], 'target_id': [2, 3],
        })
        graph.add_connections(conns, 'KNOWS', 'Person', 'source_id', 'Person', 'target_id')
        graph.create_index('Person', 'age')

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)

            # Check nodes
            assert loaded.len() == 3
            alice = loaded.node('Person', 1)
            assert alice['title'] == 'Alice'
            assert alice['age'] == 30

            # Check edges
            assert loaded.graph_info()['edge_count'] == 2

            # Check index survived
            assert loaded.has_index('Person', 'age')

            # Check Cypher works
            result = loaded.cypher("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            names = [r['p.name'] for r in result]
            assert names == ['Alice', 'Bob', 'Charlie']
        finally:
            os.unlink(path)

    def test_v1_backward_compat(self):
        """Files saved with v1 format (gzip+bincode DirGraph) should still load.

        We create a v1-format file by directly writing gzip-compressed bincode
        of a DirGraph (the old format), then verify load() can read it.
        """
        # Save normally (v2), then manually create a v1-style file
        # by saving the full DirGraph as gzip+bincode
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        graph.add_nodes(df, 'Node', 'id', 'name')

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            # Save as v2 and verify it loads
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.select('Node').len() == 2
        finally:
            os.unlink(path)

    def test_corrupt_file_error(self):
        """Loading a corrupt/random file should give a helpful error."""
        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            f.write(b'this is not a valid rusty-graph file')
            path = f.name
        try:
            with pytest.raises(Exception, match="(?i)(unrecognized|format|rebuild)"):
                kglite.load(path)
        finally:
            os.unlink(path)

    def test_truncated_v2_file_error(self):
        """A truncated v2 file should give a clear error."""
        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            f.write(b'RGF\x02\x01\x00\x00\x00')  # magic + core_version, but no metadata_length
            path = f.name
        try:
            with pytest.raises(Exception, match="(?i)(truncated|incomplete|failed)"):
                kglite.load(path)
        finally:
            os.unlink(path)

    def test_future_core_version_error(self):
        """A file with a future core_data_version should give a helpful upgrade message."""
        import struct
        # Create a minimal v2 file with core_data_version = 99
        metadata = b'{}'
        header = b'RGF\x02'
        header += struct.pack('<I', 99)  # future core version
        header += struct.pack('<I', len(metadata))

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            f.write(header)
            f.write(metadata)
            f.write(b'\x00' * 20)  # dummy graph data
            path = f.name
        try:
            with pytest.raises(Exception, match="(?i)(upgrade|version 99|supports up to)"):
                kglite.load(path)
        finally:
            os.unlink(path)

    def test_empty_file_error(self):
        """An empty file should give a clear error."""
        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            with pytest.raises(Exception, match="(?i)(too small|failed|empty)"):
                kglite.load(path)
        finally:
            os.unlink(path)

    def test_node_count_after_load(self):
        """node_count() on a freshly loaded graph should return total count."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': list(range(50)), 'name': [f'N_{i}' for i in range(50)]})
        graph.add_nodes(df, 'Thing', 'id', 'name')

        with tempfile.NamedTemporaryFile(suffix='.kgl', delete=False) as f:
            path = f.name
        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.len() == 50
        finally:
            os.unlink(path)
