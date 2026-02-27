"""Tests for the show() display method."""
import pytest
import pandas as pd
import kglite


@pytest.fixture
def discovery_graph():
    """Graph with Discovery -> Prospect -> Wellbore chains."""
    g = kglite.KnowledgeGraph()
    # Discoveries
    df_disc = pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["Johan Sverdrup", "Troll", "Ekofisk"],
        "status": ["producing", "producing", "producing"],
    })
    g.add_nodes(df_disc, "Discovery", "id", "title")

    # Prospects
    df_pros = pd.DataFrame({
        "id": [10, 20, 30],
        "title": ["Alpha", "Beta", "Gamma"],
        "area": ["North Sea", "North Sea", "Barents"],
    })
    g.add_nodes(df_pros, "Prospect", "id", "title")

    # Wellbores
    df_well = pd.DataFrame({
        "id": [100, 200, 300],
        "title": ["W1", "W2", "W3"],
        "depth": [1500, 2500, 3500],
    })
    g.add_nodes(df_well, "Wellbore", "id", "title")

    # Discovery -> Prospect connections
    g.cypher("MATCH (d:Discovery {id: 1}), (p:Prospect {id: 10}) CREATE (d)-[:HAS_PROSPECT]->(p)")
    g.cypher("MATCH (d:Discovery {id: 1}), (p:Prospect {id: 20}) CREATE (d)-[:HAS_PROSPECT]->(p)")
    g.cypher("MATCH (d:Discovery {id: 2}), (p:Prospect {id: 30}) CREATE (d)-[:HAS_PROSPECT]->(p)")
    # Prospect -> Wellbore connections
    g.cypher("MATCH (p:Prospect {id: 10}), (w:Wellbore {id: 100}) CREATE (p)-[:TESTED_BY]->(w)")
    g.cypher("MATCH (p:Prospect {id: 20}), (w:Wellbore {id: 200}) CREATE (p)-[:TESTED_BY]->(w)")
    g.cypher("MATCH (p:Prospect {id: 30}), (w:Wellbore {id: 300}) CREATE (p)-[:TESTED_BY]->(w)")

    return g


class TestShowSingleLevel:
    """show() on a selection without traversals."""

    def test_basic_id_title(self, discovery_graph):
        output = discovery_graph.select("Discovery").show(["id", "title"])
        assert "Discovery(1, Johan Sverdrup)" in output
        assert "Discovery(2, Troll)" in output
        assert "Discovery(3, Ekofisk)" in output

    def test_single_column(self, discovery_graph):
        output = discovery_graph.select("Discovery").show(["id"])
        assert "Discovery(1)" in output
        assert "Discovery(2)" in output

    def test_default_columns(self, discovery_graph):
        """Default columns are id and title."""
        output = discovery_graph.select("Discovery").show()
        assert "Discovery(1, Johan Sverdrup)" in output

    def test_extra_property(self, discovery_graph):
        output = discovery_graph.select("Discovery").show(["title", "status"])
        assert "Discovery(Johan Sverdrup, producing)" in output

    def test_missing_property_skipped(self, discovery_graph):
        """Properties not on a type are silently skipped."""
        output = discovery_graph.select("Discovery").show(["id", "nonexistent"])
        assert "Discovery(1)" in output

    def test_empty_selection(self, discovery_graph):
        output = discovery_graph.select("NonExistent").show()
        assert "empty" in output.lower()

    def test_limit(self, discovery_graph):
        output = discovery_graph.select("Discovery").show(["id"], limit=2)
        lines = [l for l in output.strip().split("\n") if l.startswith("Discovery")]
        assert len(lines) == 2
        assert "... and 1 more" in output


class TestShowMultiLevel:
    """show() after traverse() — displays traversal chains."""

    def test_two_level_chain(self, discovery_graph):
        output = (
            discovery_graph
            .select("Discovery")
            .traverse("HAS_PROSPECT")
            .show(["id", "title"])
        )
        # Discovery 1 connects to Prospect 10 and 20
        assert "Discovery(1, Johan Sverdrup) -> Prospect(" in output
        # Discovery 2 connects to Prospect 30
        assert "Discovery(2, Troll) -> Prospect(30, Gamma)" in output

    def test_three_level_chain(self, discovery_graph):
        output = (
            discovery_graph
            .select("Discovery")
            .traverse("HAS_PROSPECT")
            .traverse("TESTED_BY")
            .show(["id", "title"])
        )
        # Full chain: Discovery -> Prospect -> Wellbore
        assert "->" in output
        # Should contain wellbore info
        assert "Wellbore(" in output

    def test_single_column_chain(self, discovery_graph):
        output = (
            discovery_graph
            .select("Discovery")
            .traverse("HAS_PROSPECT")
            .show(["id"])
        )
        assert "Discovery(1) -> Prospect(" in output

    def test_dead_end_omitted(self, discovery_graph):
        """Roots with no traversal results are omitted."""
        output = (
            discovery_graph
            .select("Discovery")
            .traverse("HAS_PROSPECT")
            .show(["id"])
        )
        # Discovery 3 has no HAS_PROSPECT connections → not in output
        assert "Discovery(3)" not in output

    def test_chain_limit(self, discovery_graph):
        output = (
            discovery_graph
            .select("Discovery")
            .traverse("HAS_PROSPECT")
            .show(["id"], limit=1)
        )
        chain_lines = [l for l in output.strip().split("\n") if "->" in l]
        assert len(chain_lines) == 1

    def test_no_results(self):
        """Traversal that produces no matching targets."""
        g = kglite.KnowledgeGraph()
        df1 = pd.DataFrame({"id": [1], "title": ["A"]})
        g.add_nodes(df1, "Source", "id", "title")
        df2 = pd.DataFrame({"id": [2], "title": ["B"]})
        g.add_nodes(df2, "Target", "id", "title")
        # Create connection type but not from our source node
        g.cypher("MATCH (t:Target {id: 2}), (s:Source {id: 1}) CREATE (t)-[:LINK]->(s)")
        # Traverse outgoing LINK from Source — no targets
        output = g.select("Source").traverse("LINK", direction="outgoing").show()
        assert "no traversal" in output.lower() or "empty" in output.lower()


class TestShowWithAliases:
    """show() should resolve field aliases."""

    def test_alias_id(self):
        g = kglite.KnowledgeGraph()
        df = pd.DataFrame({"npdid": [1, 2], "prospect_name": ["A", "B"]})
        g.add_nodes(df, "Prospect", "npdid", "prospect_name")
        output = g.select("Prospect").show(["npdid"])
        assert "Prospect(1)" in output
        assert "Prospect(2)" in output

    def test_alias_title(self):
        g = kglite.KnowledgeGraph()
        df = pd.DataFrame({"npdid": [1], "prospect_name": ["Alpha"]})
        g.add_nodes(df, "Prospect", "npdid", "prospect_name")
        output = g.select("Prospect").show(["prospect_name"])
        assert "Prospect(Alpha)" in output
