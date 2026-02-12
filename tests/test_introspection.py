"""Tests for schema introspection methods."""

import pytest
from kglite import KnowledgeGraph


# ── schema() ────────────────────────────────────────────────────────────────

class TestSchema:
    def test_schema_has_all_keys(self, small_graph):
        s = small_graph.schema()
        assert 'node_types' in s
        assert 'connection_types' in s
        assert 'indexes' in s
        assert 'node_count' in s
        assert 'edge_count' in s

    def test_schema_node_counts(self, small_graph):
        s = small_graph.schema()
        assert s['node_types']['Person']['count'] == 3
        assert s['node_count'] == 3

    def test_schema_node_properties(self, small_graph):
        props = small_graph.schema()['node_types']['Person']['properties']
        assert 'age' in props
        assert 'city' in props

    def test_schema_connection_types(self, small_graph):
        s = small_graph.schema()
        knows = s['connection_types']['KNOWS']
        assert knows['count'] == 3
        assert 'Person' in knows['source_types']
        assert 'Person' in knows['target_types']

    def test_schema_edge_count(self, small_graph):
        s = small_graph.schema()
        assert s['edge_count'] == 3

    def test_schema_indexes_empty_by_default(self, small_graph):
        s = small_graph.schema()
        assert s['indexes'] == []

    def test_schema_includes_indexes(self, small_graph):
        small_graph.create_index('Person', 'city')
        s = small_graph.schema()
        assert 'Person.city' in s['indexes']

    def test_schema_empty_graph(self):
        g = KnowledgeGraph()
        s = g.schema()
        assert s['node_types'] == {}
        assert s['connection_types'] == {}
        assert s['indexes'] == []
        assert s['node_count'] == 0
        assert s['edge_count'] == 0

    def test_schema_multiple_types(self, social_graph):
        s = social_graph.schema()
        assert 'Person' in s['node_types']
        assert 'Company' in s['node_types']
        assert s['node_types']['Person']['count'] == 20
        assert s['node_types']['Company']['count'] == 5

    def test_schema_multiple_connection_types(self, social_graph):
        s = social_graph.schema()
        assert 'KNOWS' in s['connection_types']
        assert 'WORKS_AT' in s['connection_types']
        works_at = s['connection_types']['WORKS_AT']
        assert 'Person' in works_at['source_types']
        assert 'Company' in works_at['target_types']


# ── connection_types() ──────────────────────────────────────────────────────

class TestConnectionTypes:
    def test_returns_list(self, small_graph):
        ct = small_graph.connection_types()
        assert isinstance(ct, list)
        assert len(ct) == 1
        assert ct[0]['type'] == 'KNOWS'

    def test_has_counts(self, small_graph):
        ct = small_graph.connection_types()
        assert ct[0]['count'] == 3

    def test_has_source_target_types(self, social_graph):
        ct = social_graph.connection_types()
        works_at = [c for c in ct if c['type'] == 'WORKS_AT'][0]
        assert works_at['source_types'] == ['Person']
        assert works_at['target_types'] == ['Company']

    def test_empty_graph(self):
        g = KnowledgeGraph()
        assert g.connection_types() == []

    def test_multiple_types(self, social_graph):
        ct = social_graph.connection_types()
        types = {c['type'] for c in ct}
        assert types == {'KNOWS', 'WORKS_AT'}


# ── properties() ────────────────────────────────────────────────────────────

class TestProperties:
    def test_basic_properties(self, small_graph):
        props = small_graph.properties('Person')
        assert 'type' in props
        assert 'title' in props
        assert 'id' in props
        assert 'age' in props
        assert 'city' in props

    def test_builtin_type_field(self, small_graph):
        props = small_graph.properties('Person')
        assert props['type']['non_null'] == 3
        assert props['type']['unique'] == 1
        assert props['type']['values'] == ['Person']

    def test_non_null_counts(self, small_graph):
        props = small_graph.properties('Person')
        assert props['id']['non_null'] == 3
        assert props['age']['non_null'] == 3

    def test_unique_counts(self, small_graph):
        props = small_graph.properties('Person')
        assert props['id']['unique'] == 3
        # city: Oslo, Bergen → 2 unique values
        assert props['city']['unique'] == 2

    def test_low_cardinality_values(self, small_graph):
        props = small_graph.properties('Person')
        # city has 2 unique values → should include values
        assert 'values' in props['city']
        assert set(props['city']['values']) == {'Oslo', 'Bergen'}

    def test_nullable_field(self, social_graph):
        props = social_graph.properties('Person')
        # email is None for odd-numbered persons (10 of 20)
        assert props['email']['non_null'] == 10

    def test_unknown_type_raises(self, small_graph):
        with pytest.raises(KeyError):
            small_graph.properties('NonExistent')

    def test_has_type_info(self, small_graph):
        props = small_graph.properties('Person')
        assert 'type' in props['age']
        assert props['city']['type'] in ('str', 'String')

    def test_ghost_property_suppressed(self):
        """Properties in metadata but absent on all nodes should not appear."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Item {name: 'A', color: 'red'})")
        # SET a new property, then REMOVE it from the only node
        g.cypher("MATCH (n:Item) SET n.temp = 99")
        g.cypher("MATCH (n:Item) REMOVE n.temp")
        props = g.properties('Item')
        # temp was removed from the node — non_null would be 0, so it should be excluded
        assert 'temp' not in props
        assert 'color' in props

    def test_max_values_default(self, small_graph):
        """Default max_values=20 includes values for low-cardinality props."""
        props = small_graph.properties('Person')
        assert 'values' in props['city']

    def test_max_values_zero_suppresses(self, small_graph):
        """max_values=0 should never include the values list."""
        props = small_graph.properties('Person', max_values=0)
        assert 'values' not in props['city']
        assert 'values' not in props['id']

    def test_max_values_custom(self, small_graph):
        """max_values=1 excludes city (2 unique) but includes type (1 unique)."""
        props = small_graph.properties('Person', max_values=1)
        assert 'values' not in props['city']  # 2 unique > 1
        assert 'values' in props['type']       # 1 unique <= 1


# ── neighbors_schema() ──────────────────────────────────────────────────────

class TestNeighborsSchema:
    def test_outgoing(self, small_graph):
        ns = small_graph.neighbors_schema('Person')
        assert len(ns['outgoing']) > 0
        knows_out = [e for e in ns['outgoing'] if e['connection_type'] == 'KNOWS']
        assert len(knows_out) == 1
        assert knows_out[0]['target_type'] == 'Person'
        assert knows_out[0]['count'] == 3

    def test_incoming(self, small_graph):
        ns = small_graph.neighbors_schema('Person')
        knows_in = [e for e in ns['incoming'] if e['connection_type'] == 'KNOWS']
        assert len(knows_in) == 1
        assert knows_in[0]['source_type'] == 'Person'
        assert knows_in[0]['count'] == 3

    def test_cross_type_connections(self, social_graph):
        ns = social_graph.neighbors_schema('Person')
        works_at = [e for e in ns['outgoing'] if e['connection_type'] == 'WORKS_AT']
        assert len(works_at) == 1
        assert works_at[0]['target_type'] == 'Company'

    def test_company_incoming(self, social_graph):
        ns = social_graph.neighbors_schema('Company')
        works_at = [e for e in ns['incoming'] if e['connection_type'] == 'WORKS_AT']
        assert len(works_at) == 1
        assert works_at[0]['source_type'] == 'Person'
        assert works_at[0]['count'] == 20

    def test_unknown_type_raises(self, small_graph):
        with pytest.raises(KeyError):
            small_graph.neighbors_schema('NonExistent')


# ── sample() ────────────────────────────────────────────────────────────────

class TestSample:
    def test_default_count(self, social_graph):
        nodes = social_graph.sample('Person')
        assert len(nodes) == 5

    def test_custom_count(self, social_graph):
        nodes = social_graph.sample('Person', n=3)
        assert len(nodes) == 3

    def test_more_than_available(self, small_graph):
        nodes = small_graph.sample('Person', n=100)
        assert len(nodes) == 3

    def test_returns_full_dicts(self, small_graph):
        nodes = small_graph.sample('Person', n=1)
        node = nodes[0]
        assert 'type' in node
        assert 'title' in node
        assert 'id' in node
        assert node['type'] == 'Person'

    def test_unknown_type_raises(self, small_graph):
        with pytest.raises(KeyError):
            small_graph.sample('NonExistent')


# ── indexes() ───────────────────────────────────────────────────────────────

class TestIndexes:
    def test_empty_by_default(self, small_graph):
        assert small_graph.indexes() == []

    def test_equality_index(self, small_graph):
        small_graph.create_index('Person', 'city')
        idxs = small_graph.indexes()
        eq_idxs = [i for i in idxs if i['type'] == 'equality']
        assert len(eq_idxs) == 1
        assert eq_idxs[0]['node_type'] == 'Person'
        assert eq_idxs[0]['property'] == 'city'

    def test_composite_index(self, small_graph):
        small_graph.create_composite_index('Person', ['city', 'age'])
        idxs = small_graph.indexes()
        comp_idxs = [i for i in idxs if i['type'] == 'composite']
        assert len(comp_idxs) == 1
        assert comp_idxs[0]['node_type'] == 'Person'
        assert comp_idxs[0]['properties'] == ['city', 'age']

    def test_mixed_indexes(self, small_graph):
        small_graph.create_index('Person', 'city')
        small_graph.create_composite_index('Person', ['city', 'age'])
        idxs = small_graph.indexes()
        assert len(idxs) == 2
        types = {i['type'] for i in idxs}
        assert types == {'equality', 'composite'}


# ── agent_describe() ──────────────────────────────────────────────────────

import xml.etree.ElementTree as ET


class TestAgentDescribe:
    def test_returns_string(self, small_graph):
        result = small_graph.agent_describe()
        assert isinstance(result, str)

    def test_is_valid_xml(self, small_graph):
        root = ET.fromstring(small_graph.agent_describe())
        assert root.tag == 'kglite'

    def test_root_attributes(self, small_graph):
        root = ET.fromstring(small_graph.agent_describe())
        assert root.attrib['nodes'] == '3'
        assert root.attrib['edges'] == '3'

    def test_node_types_present(self, small_graph):
        root = ET.fromstring(small_graph.agent_describe())
        types = root.findall('.//node_types/type')
        assert len(types) == 1
        assert types[0].attrib['name'] == 'Person'
        assert types[0].attrib['count'] == '3'

    def test_node_properties_listed(self, small_graph):
        root = ET.fromstring(small_graph.agent_describe())
        person = root.find(".//node_types/type[@name='Person']")
        prop_names = {p.attrib['name'] for p in person.findall('prop')}
        assert 'age' in prop_names
        assert 'city' in prop_names

    def test_builtin_fields_not_in_props(self, small_graph):
        """Built-in fields (type, title, id) should be in notes, not per-type props."""
        root = ET.fromstring(small_graph.agent_describe())
        person = root.find(".//node_types/type[@name='Person']")
        prop_names = {p.attrib['name'] for p in person.findall('prop')}
        assert 'type' not in prop_names
        assert 'title' not in prop_names
        assert 'id' not in prop_names

    def test_connections_present(self, small_graph):
        root = ET.fromstring(small_graph.agent_describe())
        conns = root.findall('.//connections/conn')
        assert len(conns) == 1
        assert conns[0].attrib['type'] == 'KNOWS'
        assert conns[0].attrib['from'] == 'Person'
        assert conns[0].attrib['to'] == 'Person'

    def test_multiple_types(self, social_graph):
        root = ET.fromstring(social_graph.agent_describe())
        type_names = {t.attrib['name'] for t in root.findall('.//node_types/type')}
        assert type_names == {'Person', 'Company'}

    def test_multiple_connection_types(self, social_graph):
        root = ET.fromstring(social_graph.agent_describe())
        conn_types = {c.attrib['type'] for c in root.findall('.//connections/conn')}
        assert conn_types == {'KNOWS', 'WORKS_AT'}

    def test_indexes_section(self, small_graph):
        small_graph.create_index('Person', 'city')
        root = ET.fromstring(small_graph.agent_describe())
        idxs = root.findall('.//indexes/idx')
        assert len(idxs) == 1
        assert idxs[0].attrib['on'] == 'Person.city'

    def test_empty_graph(self):
        g = KnowledgeGraph()
        root = ET.fromstring(g.agent_describe())
        assert root.attrib['nodes'] == '0'
        assert root.attrib['edges'] == '0'
        assert root.findall('.//node_types/type') == []

    def test_cypher_ref_present(self, small_graph):
        root = ET.fromstring(small_graph.agent_describe())
        cypher_ref = root.find('cypher_ref')
        assert cypher_ref is not None
        assert cypher_ref.find('clauses') is not None
        assert cypher_ref.find('functions') is not None
        assert cypher_ref.find('not_supported') is not None

    def test_api_section_present(self, small_graph):
        root = ET.fromstring(small_graph.agent_describe())
        api = root.find('api')
        assert api is not None
        sigs = [m.attrib['sig'] for m in api.findall('method')]
        assert any('cypher' in s for s in sigs)

    def test_static_sections_identical(self, small_graph, social_graph):
        """The cypher_ref and api sections should be identical across graphs."""
        root1 = ET.fromstring(small_graph.agent_describe())
        root2 = ET.fromstring(social_graph.agent_describe())
        ref1 = ET.tostring(root1.find('cypher_ref'), encoding='unicode')
        ref2 = ET.tostring(root2.find('cypher_ref'), encoding='unicode')
        assert ref1 == ref2

    def test_notes_mention_builtins(self, small_graph):
        root = ET.fromstring(small_graph.agent_describe())
        notes_xml = ET.tostring(root.find('.//notes'), encoding='unicode')
        assert 'type' in notes_xml
        assert 'title' in notes_xml
        assert 'id' in notes_xml
