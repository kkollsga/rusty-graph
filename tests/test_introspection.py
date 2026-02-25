"""Tests for schema introspection methods."""

import pandas as pd
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


# ── describe() ─────────────────────────────────────────────────────────────

import xml.etree.ElementTree as ET


class TestDescribe:
    """Tests for the new describe() method (progressive disclosure)."""

    # -- Inventory mode (no args) -- small graph auto-inlines detail --

    def test_returns_string(self, small_graph):
        result = small_graph.describe()
        assert isinstance(result, str)

    def test_is_valid_xml(self, small_graph):
        root = ET.fromstring(small_graph.describe())
        assert root.tag == 'graph'

    def test_root_attributes(self, small_graph):
        root = ET.fromstring(small_graph.describe())
        assert root.attrib['nodes'] == '3'
        assert root.attrib['edges'] == '3'

    def test_conventions_present(self, small_graph):
        root = ET.fromstring(small_graph.describe())
        conv = root.find('conventions')
        assert conv is not None
        assert '.id' in conv.text
        assert '.title' in conv.text

    def test_no_cypher_ref(self, small_graph):
        """describe() should NOT include a full Cypher reference."""
        root = ET.fromstring(small_graph.describe())
        assert root.find('cypher_ref') is None

    def test_connections_present(self, social_graph):
        root = ET.fromstring(social_graph.describe())
        conns = root.findall('.//connections/conn')
        conn_types = {c.attrib['type'] for c in conns}
        assert 'KNOWS' in conn_types
        assert 'WORKS_AT' in conn_types

    def test_extensions_present(self, small_graph):
        root = ET.fromstring(small_graph.describe())
        ext = root.find('extensions')
        assert ext is not None
        # Algorithms should always be present
        alg = ext.find('algorithms')
        assert alg is not None

    def test_empty_graph(self):
        g = KnowledgeGraph()
        root = ET.fromstring(g.describe())
        assert root.attrib['nodes'] == '0'
        assert root.attrib['edges'] == '0'

    # -- Small graph: auto-inlined detail --

    def test_small_graph_inlines_types(self, small_graph):
        """≤15 types → full detail for all types inline."""
        root = ET.fromstring(small_graph.describe())
        types = root.findall('.//types/type')
        assert len(types) == 1
        assert types[0].attrib['name'] == 'Person'

    def test_small_graph_has_properties(self, small_graph):
        root = ET.fromstring(small_graph.describe())
        person = root.find(".//types/type[@name='Person']")
        props = person.findall('.//properties/prop')
        prop_names = {p.attrib['name'] for p in props}
        assert 'age' in prop_names
        assert 'city' in prop_names

    def test_small_graph_excludes_builtins(self, small_graph):
        """Built-in fields (type, title, id) should NOT be in properties."""
        root = ET.fromstring(small_graph.describe())
        person = root.find(".//types/type[@name='Person']")
        props = person.findall('.//properties/prop')
        prop_names = {p.attrib['name'] for p in props}
        assert 'type' not in prop_names
        assert 'title' not in prop_names
        assert 'id' not in prop_names

    def test_small_graph_has_samples(self, small_graph):
        root = ET.fromstring(small_graph.describe())
        person = root.find(".//types/type[@name='Person']")
        samples = person.findall('.//samples/node')
        assert len(samples) > 0
        assert 'title' in samples[0].attrib

    def test_small_graph_has_connections_topology(self, social_graph):
        root = ET.fromstring(social_graph.describe())
        person = root.find(".//types/type[@name='Person']")
        out_conns = person.findall('.//connections/out')
        assert any(c.attrib['type'] == 'WORKS_AT' for c in out_conns)

    # -- Large graph: flat descriptor format --

    def test_large_graph_flat_types(self, large_schema_graph):
        """>15 types → flat comma-separated list, no inline detail."""
        root = ET.fromstring(large_schema_graph.describe())
        types_el = root.find('types')
        # Should have text content (comma-separated descriptors), not <band> children
        assert types_el.findall('band') == []
        assert types_el.text is not None and len(types_el.text.strip()) > 0

    def test_large_graph_descriptor_format(self, large_schema_graph):
        """Descriptors should be Name[size,complexity] format."""
        root = ET.fromstring(large_schema_graph.describe())
        types_text = root.find('types').text.strip()
        # Should contain descriptors like Type0[l,vl], Type5[m,h], etc.
        import re
        descriptors = re.findall(r'\w+\[\w+,\w+(?:,\w+)*\]', types_text)
        assert len(descriptors) == 20

    def test_large_graph_has_hint(self, large_schema_graph):
        root = ET.fromstring(large_schema_graph.describe())
        hint = root.find('hint')
        assert hint is not None
        assert 'describe' in hint.text

    def test_large_graph_has_types_count(self, large_schema_graph):
        root = ET.fromstring(large_schema_graph.describe())
        types_el = root.find('types')
        assert 'count' in types_el.attrib
        assert int(types_el.attrib['count']) > 15

    # -- Focused detail mode (types= arg) --

    def test_focused_returns_detail(self, social_graph):
        root = ET.fromstring(social_graph.describe(types=['Person']))
        person = root.find(".//type[@name='Person']")
        assert person is not None
        assert person.find('properties') is not None
        assert person.find('samples') is not None

    def test_focused_connections_topology(self, social_graph):
        root = ET.fromstring(social_graph.describe(types=['Person']))
        person = root.find(".//type[@name='Person']")
        out_conns = person.findall('.//connections/out')
        assert any(c.attrib['type'] == 'WORKS_AT' for c in out_conns)
        in_conns = person.findall('.//connections/in')
        assert any(c.attrib['type'] == 'KNOWS' for c in in_conns)

    def test_focused_multiple_types(self, social_graph):
        root = ET.fromstring(social_graph.describe(types=['Person', 'Company']))
        type_names = {t.attrib['name'] for t in root.findall('.//type')}
        assert type_names == {'Person', 'Company'}

    def test_focused_no_extensions(self, social_graph):
        """Focused mode should NOT include extensions section."""
        root = ET.fromstring(social_graph.describe(types=['Person']))
        assert root.find('extensions') is None

    def test_focused_invalid_type_raises(self, small_graph):
        with pytest.raises(ValueError):
            small_graph.describe(types=['NonExistent'])

    def test_focused_error_lists_available(self, social_graph):
        """Error message should list available types."""
        try:
            social_graph.describe(types=['Bogus'])
        except ValueError as e:
            msg = str(e)
            assert 'Person' in msg
            assert 'Company' in msg

    # -- Extensions conditional --

    def test_no_timeseries_extension_without_ts(self, social_graph):
        root = ET.fromstring(social_graph.describe())
        ext = root.find('extensions')
        assert ext.find('timeseries') is None

    def test_no_spatial_extension_without_spatial(self, social_graph):
        root = ET.fromstring(social_graph.describe())
        ext = root.find('extensions')
        assert ext.find('spatial') is None


# ── Tiered describe() ────────────────────────────────────────────────────


class TestDescribeTiers:
    """Tests for core/supporting node tiers in describe()."""

    # -- set_parent_type API --

    def test_set_parent_type_basic(self, social_graph):
        """set_parent_type succeeds for valid types."""
        social_graph.set_parent_type('Company', 'Person')

    def test_set_parent_type_invalid_child(self, social_graph):
        with pytest.raises(ValueError, match='not found'):
            social_graph.set_parent_type('Bogus', 'Person')

    def test_set_parent_type_invalid_parent(self, social_graph):
        with pytest.raises(ValueError, match='not found'):
            social_graph.set_parent_type('Company', 'Bogus')

    # -- Inventory only shows core types --

    def test_inventory_core_only(self, tiered_graph):
        """Inventory should only list core types when tiers are set."""
        result = tiered_graph.describe()
        root = ET.fromstring(result)
        types_el = root.find('types')
        types_text = types_el.text.strip() if types_el.text else ''
        # Core types should be present
        assert 'Project' in types_text
        assert 'Facility' in types_text
        assert 'Region' in types_text
        # Supporting types should NOT be listed as standalone entries
        assert 'ProjectBudget[' not in types_text
        assert 'ProjectPhase[' not in types_text
        assert 'FacilitySpec[' not in types_text

    def test_inventory_core_supporting_counts(self, tiered_graph):
        """Types element should have core= and supporting= attributes."""
        root = ET.fromstring(tiered_graph.describe())
        types_el = root.find('types')
        assert types_el.attrib['core'] == '17'  # 3 domain + 14 filler
        assert types_el.attrib['supporting'] == '3'

    def test_inventory_plus_n_suffix(self, tiered_graph):
        """Parent types should have +N suffix showing child count."""
        root = ET.fromstring(tiered_graph.describe())
        types_text = root.find('types').text.strip()
        # Project has 2 children (ProjectBudget, ProjectPhase)
        assert '+2' in types_text
        # Facility has 1 child (FacilitySpec)
        assert '+1' in types_text

    # -- Descriptor format --

    def test_descriptor_format_with_flags(self, tiered_graph):
        """Descriptors should use Name[size,complexity,flags] format."""
        root = ET.fromstring(tiered_graph.describe())
        types_text = root.find('types').text.strip()
        import re
        # Facility has location → should have 'loc' flag
        facility_match = re.search(r'Facility\[(\w+,\w+(?:,\w+)*)\]', types_text)
        assert facility_match is not None
        parts = facility_match.group(1).split(',')
        assert len(parts) >= 2  # at least size,complexity
        assert parts[0] in ('vs', 's', 'm', 'l', 'vl')  # size tier
        assert parts[1] in ('vl', 'l', 'm', 'h', 'vh')  # complexity

    # -- Capability bubbling --

    def test_capability_bubbling_ts(self, tiered_graph):
        """ProjectBudget has timeseries → Project descriptor should show ts."""
        root = ET.fromstring(tiered_graph.describe())
        types_text = root.find('types').text.strip()
        import re
        project_match = re.search(r'Project\[([^\]]+)\]', types_text)
        assert project_match is not None
        flags = project_match.group(1)
        assert 'ts' in flags

    # -- Connection map excludes supporting→parent edges --

    def test_connection_map_excludes_of_edges(self, tiered_graph):
        """OF_PROJECT, OF_FACILITY edges should be excluded from connections."""
        root = ET.fromstring(tiered_graph.describe())
        conns = root.findall('.//connections/conn')
        conn_types = {c.attrib['type'] for c in conns}
        assert 'OF_PROJECT' not in conn_types
        assert 'OF_FACILITY' not in conn_types

    def test_connection_map_includes_core_edges(self, tiered_graph):
        """Core→core edges should still be in connections."""
        root = ET.fromstring(tiered_graph.describe())
        conns = root.findall('.//connections/conn')
        conn_types = {c.attrib['type'] for c in conns}
        assert 'HAS_PROJECT' in conn_types
        assert 'HAS_FACILITY' in conn_types

    # -- Focused detail: supporting section --

    def test_focused_detail_supporting_section(self, tiered_graph):
        """Focused detail for a parent type should include <supporting> section."""
        root = ET.fromstring(tiered_graph.describe(types=['Project']))
        project = root.find(".//type[@name='Project']")
        supporting = project.find('supporting')
        assert supporting is not None
        text = supporting.text
        assert 'ProjectBudget' in text
        assert 'ProjectPhase' in text

    def test_focused_detail_supporting_descriptors(self, tiered_graph):
        """Supporting section should use compact descriptor format."""
        root = ET.fromstring(tiered_graph.describe(types=['Project']))
        project = root.find(".//type[@name='Project']")
        supporting_text = project.find('supporting').text
        import re
        # Should have descriptor format Name[size,complexity]
        descriptors = re.findall(r'\w+\[\w+,\w+(?:,\w+)*\]', supporting_text)
        assert len(descriptors) == 2  # ProjectBudget and ProjectPhase

    def test_focused_detail_no_supporting_for_leaf(self, tiered_graph):
        """Types without children should NOT have a <supporting> section."""
        root = ET.fromstring(tiered_graph.describe(types=['Region']))
        region = root.find(".//type[@name='Region']")
        assert region.find('supporting') is None

    # -- No tiers = backwards compat --

    def test_no_tiers_all_types_shown(self, social_graph):
        """When no parent_types set, all types appear in inventory."""
        root = ET.fromstring(social_graph.describe())
        # social_graph has 2 types (Person, Company) → auto-inlined detail
        types = root.findall('.//types/type')
        type_names = {t.attrib['name'] for t in types}
        assert type_names == {'Person', 'Company'}

    def test_no_tiers_count_attribute(self, large_schema_graph):
        """Without tiers, types element uses count= not core=/supporting=."""
        root = ET.fromstring(large_schema_graph.describe())
        types_el = root.find('types')
        assert 'count' in types_el.attrib
        assert 'core' not in types_el.attrib
        assert 'supporting' not in types_el.attrib


# ── Read-only describe() ─────────────────────────────────────────────────


class TestDescribeReadOnly:
    """Tests for read-only mode in describe()."""

    def test_read_only_notice_in_inventory(self, social_graph):
        """Read-only graph should include <read-only> element."""
        social_graph.read_only(True)
        root = ET.fromstring(social_graph.describe())
        ro = root.find('read-only')
        assert ro is not None
        assert 'CREATE' in ro.text
        assert 'SET' in ro.text
        assert 'DELETE' in ro.text
        assert 'REMOVE' in ro.text
        assert 'MERGE' in ro.text

    def test_no_read_only_notice_when_writable(self, social_graph):
        """Writable graph should NOT include <read-only> element."""
        root = ET.fromstring(social_graph.describe())
        assert root.find('read-only') is None

    def test_read_only_notice_in_focused_detail(self, social_graph):
        """Focused detail should also show read-only notice."""
        social_graph.read_only(True)
        root = ET.fromstring(social_graph.describe(types=['Person']))
        ro = root.find('read-only')
        assert ro is not None

    def test_read_only_notice_in_large_inventory(self, large_schema_graph):
        """Large graph inventory should show read-only notice."""
        large_schema_graph.read_only(True)
        root = ET.fromstring(large_schema_graph.describe())
        ro = root.find('read-only')
        assert ro is not None


# ── Exploration Hints ────────────────────────────────────────────────────────

class TestExplorationHints:
    def test_disconnected_types_shown(self):
        """Disconnected types appear in <exploration_hints>."""
        g = KnowledgeGraph()
        # Two connected types + one orphan
        df_a = pd.DataFrame({'id': [1, 2], 'name': ['A1', 'A2']})
        g.add_nodes(df_a, 'TypeA', 'id', 'name')
        df_b = pd.DataFrame({'id': [10, 20], 'name': ['B1', 'B2']})
        g.add_nodes(df_b, 'TypeB', 'id', 'name')
        df_orphan = pd.DataFrame({'id': [100, 200, 300], 'name': ['O1', 'O2', 'O3']})
        g.add_nodes(df_orphan, 'Orphan', 'id', 'name')
        # Connect A -> B
        edges = pd.DataFrame({'from': [1, 2], 'to': [10, 20]})
        g.add_connections(edges, 'LINKS', 'TypeA', 'from', 'TypeB', 'to')

        root = ET.fromstring(g.describe())
        hints = root.find('exploration_hints')
        assert hints is not None
        disc = hints.find('disconnected')
        assert disc is not None
        types = [t.get('name') for t in disc.findall('type')]
        assert 'Orphan' in types
        assert 'TypeA' not in types
        assert 'TypeB' not in types

    def test_no_hints_empty_graph(self):
        """Empty graph has no exploration_hints section."""
        g = KnowledgeGraph()
        root = ET.fromstring(g.describe())
        assert root.find('exploration_hints') is None

    def test_no_hints_zero_edges(self):
        """Nodes but no edges → no exploration_hints (all disconnected = not useful)."""
        g = KnowledgeGraph()
        df_a = pd.DataFrame({'id': [1], 'name': ['A1']})
        g.add_nodes(df_a, 'TypeA', 'id', 'name')
        df_b = pd.DataFrame({'id': [2], 'name': ['B1']})
        g.add_nodes(df_b, 'TypeB', 'id', 'name')
        root = ET.fromstring(g.describe())
        assert root.find('exploration_hints') is None

    def test_no_disconnected_when_all_connected(self):
        """All types have edges → no <disconnected> section."""
        g = KnowledgeGraph()
        df_a = pd.DataFrame({'id': [1], 'name': ['A1']})
        g.add_nodes(df_a, 'TypeA', 'id', 'name')
        df_b = pd.DataFrame({'id': [10], 'name': ['B1']})
        g.add_nodes(df_b, 'TypeB', 'id', 'name')
        edges = pd.DataFrame({'from': [1], 'to': [10]})
        g.add_connections(edges, 'LINKS', 'TypeA', 'from', 'TypeB', 'to')

        root = ET.fromstring(g.describe())
        hints = root.find('exploration_hints')
        # No disconnected types, and join candidates only for disconnected pairs,
        # so hints section should be absent entirely
        assert hints is None

    def test_join_candidates_property_overlap(self):
        """Two disconnected types with overlapping property values → join candidate."""
        g = KnowledgeGraph()
        # TypeA and TypeB share 'region' property with overlapping values
        df_a = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A1', 'A2', 'A3'],
            'region': ['North', 'South', 'East'],
        })
        g.add_nodes(df_a, 'TypeA', 'id', 'name')
        df_b = pd.DataFrame({
            'id': [10, 20, 30],
            'name': ['B1', 'B2', 'B3'],
            'region': ['North', 'West', 'East'],
        })
        g.add_nodes(df_b, 'TypeB', 'id', 'name')
        # TypeC connects to TypeA to make the graph have edges
        df_c = pd.DataFrame({'id': [100], 'name': ['C1']})
        g.add_nodes(df_c, 'TypeC', 'id', 'name')
        edges = pd.DataFrame({'from': [100], 'to': [1]})
        g.add_connections(edges, 'LINKS', 'TypeC', 'from', 'TypeA', 'to')

        root = ET.fromstring(g.describe())
        hints = root.find('exploration_hints')
        assert hints is not None
        jc = hints.find('join_candidates')
        assert jc is not None
        candidates = jc.findall('candidate')
        # TypeA and TypeB share region with 2 overlapping values (North, East)
        assert len(candidates) >= 1
        attrs = candidates[0].attrib
        # The candidate should reference TypeA/TypeB.region
        left_right = {attrs['left'], attrs['right']}
        assert 'TypeA.region' in left_right or 'TypeB.region' in left_right
        assert int(attrs['overlap']) == 2

    def test_join_candidates_not_for_connected_pairs(self):
        """Already-connected type pairs don't appear as join candidates."""
        g = KnowledgeGraph()
        df_a = pd.DataFrame({
            'id': [1, 2],
            'name': ['A1', 'A2'],
            'region': ['North', 'South'],
        })
        g.add_nodes(df_a, 'TypeA', 'id', 'name')
        df_b = pd.DataFrame({
            'id': [10, 20],
            'name': ['B1', 'B2'],
            'region': ['North', 'South'],
        })
        g.add_nodes(df_b, 'TypeB', 'id', 'name')
        # Connect them — they share 'region' but are already connected
        edges = pd.DataFrame({'from': [1], 'to': [10]})
        g.add_connections(edges, 'LINKS', 'TypeA', 'from', 'TypeB', 'to')

        root = ET.fromstring(g.describe())
        hints = root.find('exploration_hints')
        # Either no hints or no join_candidates section
        if hints is not None:
            jc = hints.find('join_candidates')
            if jc is not None:
                for c in jc.findall('candidate'):
                    pair = {c.get('left').split('.')[0], c.get('right').split('.')[0]}
                    assert pair != {'TypeA', 'TypeB'}
