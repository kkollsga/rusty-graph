"""Tests for schema definition and validation."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


class TestSchemaDefinition:
    def test_define_schema_basic(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B'], 'value': [10, 20]})
        graph.add_nodes(df, 'Node', 'id', 'name')

        graph.define_schema({
            'nodes': {
                'Node': {
                    'required': ['id', 'title'],
                    'types': {'id': 'integer', 'title': 'string'},
                }
            }
        })
        assert graph.has_schema()

    def test_clear_schema(self):
        graph = KnowledgeGraph()
        graph.define_schema({'nodes': {'Node': {'required': ['id']}}})
        assert graph.has_schema()
        graph.clear_schema()
        assert not graph.has_schema()

    def test_schema_definition(self):
        graph = KnowledgeGraph()
        schema_def = {
            'nodes': {
                'Node': {
                    'required': ['id', 'title'],
                    'types': {'id': 'integer'},
                }
            }
        }
        graph.define_schema(schema_def)
        retrieved = graph.schema_definition()
        assert retrieved is not None


class TestSchemaValidation:
    def test_valid_graph(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B'], 'value': [10, 20]})
        graph.add_nodes(df, 'Node', 'id', 'name')

        graph.define_schema({
            'nodes': {
                'Node': {
                    'required': ['id', 'title'],
                    'types': {'id': 'integer', 'title': 'string'},
                }
            }
        })
        errors = graph.validate_schema()
        assert len(errors) == 0

    def test_missing_required_field(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A']})
        graph.add_nodes(df, 'Node', 'id', 'name')

        graph.define_schema({
            'nodes': {
                'Node': {
                    'required': ['id', 'title', 'missing_field'],
                }
            }
        })
        errors = graph.validate_schema()
        assert len(errors) > 0

    def test_type_mismatch(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'count': ['not_a_number']})
        graph.add_nodes(df, 'Node', 'id', 'name')

        graph.define_schema({
            'nodes': {
                'Node': {
                    'types': {'count': 'integer'},
                }
            }
        })
        errors = graph.validate_schema()
        assert len(errors) > 0

    def test_connection_schema(self):
        graph = KnowledgeGraph()
        users = pd.DataFrame({'id': [1], 'name': ['A']})
        products = pd.DataFrame({'id': [101], 'name': ['P']})
        graph.add_nodes(users, 'User', 'id', 'name')
        graph.add_nodes(products, 'Product', 'id', 'name')
        conn_df = pd.DataFrame({'user_id': [1], 'product_id': [101]})
        graph.add_connections(conn_df, 'PURCHASED', 'User', 'user_id',
                              'Product', 'product_id')

        graph.define_schema({
            'connections': {
                'PURCHASED': {'source': 'User', 'target': 'Product'},
            }
        })
        errors = graph.validate_schema()
        assert len(errors) == 0
