"""Tests for batch operations: add_nodes_bulk, add_connections_bulk, update, connector API."""

import pytest
import pandas as pd
from rusty_graph import KnowledgeGraph


class TestBatchUpdate:
    def test_basic_update(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'status': ['active', 'active', 'active'],
        })
        graph.add_nodes(df, 'Node', 'id', 'name')
        result = graph.type_filter('Node').update({'status': 'inactive'})
        assert result['nodes_updated'] == 3

    def test_update_with_filter(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10, 20, 30],
        })
        graph.add_nodes(df, 'Node', 'id', 'name')
        result = graph.type_filter('Node').filter({'value': {'>': 15}}).update(
            {'processed': True}
        )
        assert result['nodes_updated'] == 2

    def test_update_preserves_selection(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        result = graph.type_filter('Node').update({'flag': True}, keep_selection=True)
        updated_graph = result['graph']
        assert updated_graph is not None

    def test_update_multiple_properties(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A']})
        graph.add_nodes(df, 'Node', 'id', 'name')
        result = graph.type_filter('Node').update({
            'count': 42,
            'ratio': 3.14,
            'active': True,
            'label': 'updated',
        })
        # nodes_updated counts individual property updates (1 node * 4 properties = 4)
        assert result['nodes_updated'] == 4

    def test_update_numeric_types(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({'id': [1], 'name': ['A'], 'val': [0]})
        graph.add_nodes(df, 'Node', 'id', 'name')
        result = graph.type_filter('Node').update({'val': 999})
        updated = result['graph']
        node = updated.type_filter('Node').get_nodes()[0]
        assert node['val'] == 999


class TestBulkOperations:
    def test_add_nodes_bulk(self):
        graph = KnowledgeGraph()
        nodes = [
            {'node_type': 'Person', 'id_field': 'person_id',
             'data': pd.DataFrame({
                 'person_id': [1, 2], 'name': ['A', 'B'], 'age': [25, 30]
             })},
            {'node_type': 'Company', 'id_field': 'company_id',
             'data': pd.DataFrame({
                 'company_id': [100], 'name': ['Corp']
             })},
        ]
        for item in nodes:
            graph.add_nodes(item['data'], item['node_type'],
                            item['id_field'], 'name')

        assert graph.type_filter('Person').node_count() == 2
        assert graph.type_filter('Company').node_count() == 1

    def test_add_connections_bulk(self):
        graph = KnowledgeGraph()
        df1 = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        df2 = pd.DataFrame({'id': [10, 11], 'name': ['X', 'Y']})
        graph.add_nodes(df1, 'Source', 'id', 'name')
        graph.add_nodes(df2, 'Target', 'id', 'name')

        conn = pd.DataFrame({'source': [1, 2], 'target': [10, 11]})
        report = graph.add_connections(conn, 'LINKS', 'Source', 'source',
                                       'Target', 'target')
        assert report['connections_created'] == 2


class TestConnectorAPI:
    def test_add_nodes_bulk_api(self):
        graph = KnowledgeGraph()
        people = pd.DataFrame({
            'person_id': list(range(1, 6)),
            'name': [f'Person_{i}' for i in range(1, 6)],
        })
        graph.add_nodes_bulk([{
            'data': people,
            'node_type': 'Person',
            'unique_id_field': 'person_id',
            'node_title_field': 'name',
        }])
        assert graph.type_filter('Person').node_count() == 5

    def test_add_connections_bulk_api(self):
        graph = KnowledgeGraph()
        people = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        companies = pd.DataFrame({'id': [100], 'name': ['Corp']})
        graph.add_nodes(people, 'Person', 'id', 'name')
        graph.add_nodes(companies, 'Company', 'id', 'name')

        # add_connections_bulk expects: connection_name, source_type, target_type,
        # and data with columns named source_id and target_id
        conns = pd.DataFrame({'source_id': [1, 2], 'target_id': [100, 100]})
        graph.add_connections_bulk([{
            'data': conns,
            'connection_name': 'WORKS_AT',
            'source_type': 'Person',
            'target_type': 'Company',
        }])

        result = graph.type_filter('Person').traverse('WORKS_AT')
        assert result.node_count() >= 1
