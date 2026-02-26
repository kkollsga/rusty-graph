"""Tests for field alias resolution (Phase 1: Rename Hell fix).

When users call add_nodes(df, 'Type', 'npdid', 'prospect_name'),
the original column names should still work as property accessors in Cypher queries,
where() calls, and the fluent API.
"""
import pytest
import pandas as pd
import kglite
import tempfile
import os


@pytest.fixture
def graph_with_aliases():
    """Graph where original column names differ from canonical id/title."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({
        "npdid": [1, 2, 3],
        "prospect_name": ["Alpha", "Beta", "Gamma"],
        "status": ["active", "inactive", "active"],
    })
    g.add_nodes(df, "Prospect", "npdid", "prospect_name")
    return g


@pytest.fixture
def graph_default_fields():
    """Graph where id/title fields use default names (no aliasing)."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({
        "id": [10, 20, 30],
        "title": ["X", "Y", "Z"],
        "category": ["A", "B", "A"],
    })
    g.add_nodes(df, "Item", "id", "title")
    return g


class TestCypherAliasResolution:
    """Cypher queries should resolve original column names to id/title."""

    def test_cypher_alias_id_field(self, graph_with_aliases):
        """n.npdid should resolve to the id field."""
        result = graph_with_aliases.cypher("MATCH (n:Prospect) RETURN n.npdid ORDER BY n.npdid")
        values = [r["n.npdid"] for r in result]
        assert values == [1, 2, 3]

    def test_cypher_alias_title_field(self, graph_with_aliases):
        """n.prospect_name should resolve to the title field."""
        result = graph_with_aliases.cypher(
            "MATCH (n:Prospect) RETURN n.prospect_name ORDER BY n.prospect_name"
        )
        values = [r["n.prospect_name"] for r in result]
        assert values == ["Alpha", "Beta", "Gamma"]

    def test_cypher_canonical_still_works(self, graph_with_aliases):
        """n.id and n.title should still work alongside aliases."""
        result = graph_with_aliases.cypher(
            "MATCH (n:Prospect) RETURN n.id, n.title ORDER BY n.id"
        )
        assert result[0]["n.id"] == 1
        assert result[0]["n.title"] == "Alpha"

    def test_cypher_where_with_alias(self, graph_with_aliases):
        """WHERE clause should resolve aliases."""
        result = graph_with_aliases.cypher(
            "MATCH (n:Prospect) WHERE n.npdid = 2 RETURN n.prospect_name"
        )
        assert len(result) == 1
        assert result[0]["n.prospect_name"] == "Beta"

    def test_cypher_no_alias_no_interference(self, graph_default_fields):
        """When fields use default names, no aliasing occurs."""
        result = graph_default_fields.cypher("MATCH (n:Item) RETURN n.id ORDER BY n.id")
        values = [r["n.id"] for r in result]
        assert values == [10, 20, 30]

    def test_cypher_regular_property_unaffected(self, graph_with_aliases):
        """Regular properties (not aliased) should work normally."""
        result = graph_with_aliases.cypher(
            "MATCH (n:Prospect) WHERE n.status = 'active' RETURN n.npdid ORDER BY n.npdid"
        )
        values = [r["n.npdid"] for r in result]
        assert values == [1, 3]


class TestFilterAliasResolution:
    """Fluent API where() should resolve original column names."""

    def test_filter_by_alias_id(self, graph_with_aliases):
        """where({'npdid': 2}) should find the node."""
        g = graph_with_aliases
        result = g.where({"type": "Prospect"}).where({"npdid": 2}).collect()
        assert len(result) == 1
        assert result[0]["title"] == "Beta"

    def test_filter_by_alias_title(self, graph_with_aliases):
        """where({'prospect_name': 'Alpha'}) should find the node."""
        g = graph_with_aliases
        result = g.where({"type": "Prospect"}).where({"prospect_name": "Alpha"}).collect()
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_filter_by_canonical_still_works(self, graph_with_aliases):
        """where({'id': 1}) should still work."""
        g = graph_with_aliases
        result = g.where({"type": "Prospect"}).where({"id": 1}).collect()
        assert len(result) == 1
        assert result[0]["title"] == "Alpha"


class TestSaveLoadAliases:
    """Aliases should survive save/load round-trips."""

    def test_aliases_persist(self, graph_with_aliases):
        """Save and reload should preserve alias resolution."""
        with tempfile.NamedTemporaryFile(suffix=".kglite", delete=False) as f:
            path = f.name

        try:
            graph_with_aliases.save(path)
            g2 = kglite.load(path)

            # Alias should work after reload
            result = g2.cypher("MATCH (n:Prospect) WHERE n.npdid = 1 RETURN n.prospect_name")
            assert len(result) == 1
            assert result[0]["n.prospect_name"] == "Alpha"
        finally:
            os.unlink(path)


class TestMultipleNodeTypes:
    """Aliases are per-node-type — different types can have different aliases."""

    def test_different_aliases_per_type(self):
        g = kglite.KnowledgeGraph()
        df1 = pd.DataFrame({
            "npdid": [1, 2], "prospect_name": ["A", "B"], "area": ["North", "South"],
        })
        g.add_nodes(df1, "Prospect", "npdid", "prospect_name")

        df2 = pd.DataFrame({
            "well_id": [10, 20], "well_name": ["W1", "W2"], "depth": [100, 200],
        })
        g.add_nodes(df2, "Well", "well_id", "well_name")

        # Each type should resolve its own aliases
        result = g.cypher("MATCH (n:Prospect) WHERE n.npdid = 1 RETURN n.prospect_name")
        assert result[0]["n.prospect_name"] == "A"

        result = g.cypher("MATCH (n:Well) WHERE n.well_id = 10 RETURN n.well_name")
        assert result[0]["n.well_name"] == "W1"

    def test_alias_does_not_cross_types(self):
        """npdid alias on Prospect should not affect Well type."""
        g = kglite.KnowledgeGraph()
        df1 = pd.DataFrame({"npdid": [1], "name": ["A"]})
        g.add_nodes(df1, "Prospect", "npdid")

        df2 = pd.DataFrame({"id": [10], "title": ["W1"], "npdid_ref": [1]})
        g.add_nodes(df2, "Well", "id", "title")

        # n.npdid_ref on Well should NOT resolve to id (it's a regular property there)
        result = g.cypher("MATCH (n:Well) RETURN n.npdid_ref")
        assert result[0]["n.npdid_ref"] == 1


class TestDescribeAliases:
    """describe() should include alias info in XML."""

    def test_aliases_in_xml(self, graph_with_aliases):
        xml = graph_with_aliases.describe()
        assert 'id_alias="npdid"' in xml
        assert 'title_alias="prospect_name"' in xml

    def test_no_aliases_when_default(self, graph_default_fields):
        xml = graph_default_fields.describe()
        assert 'id_alias="' not in xml
        assert 'title_alias="' not in xml
