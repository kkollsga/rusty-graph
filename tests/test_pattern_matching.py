"""Tests for match_pattern() — Cypher-like pattern syntax."""


class TestNodePatterns:
    def test_simple_node(self, social_graph):
        results = social_graph.match_pattern("(p:Person)")
        assert len(results) == 20

    def test_anonymous_node(self, social_graph):
        results = social_graph.match_pattern("(:Person)")
        assert len(results) == 20

    def test_empty_node(self, social_graph):
        results = social_graph.match_pattern("(n)")
        # 20 persons + 5 companies + schema nodes (4 types with define_schema in fixture)
        assert len(results) >= 25

    def test_node_with_property(self, small_graph):
        results = small_graph.match_pattern('(p:Person {city: "Oslo"})')
        assert len(results) == 2  # Alice and Charlie


class TestEdgePatterns:
    def test_outgoing_edge(self, small_graph):
        results = small_graph.match_pattern("(a:Person)-[:KNOWS]->(b:Person)")
        assert len(results) == 3  # Alice->Bob, Bob->Charlie, Alice->Charlie

    def test_incoming_edge(self, small_graph):
        results = small_graph.match_pattern("(a:Person)<-[:KNOWS]-(b:Person)")
        assert len(results) == 3

    def test_bidirectional_edge(self, small_graph):
        results = small_graph.match_pattern("(a:Person)-[:KNOWS]-(b:Person)")
        assert len(results) >= 3


class TestMultiHopPatterns:
    def test_two_hop(self, petroleum_graph):
        results = petroleum_graph.match_pattern(
            "(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)"
        )
        assert len(results) > 0
        for m in results:
            assert "p" in m
            assert "pr" in m
            assert "d" in m

    def test_cross_type_pattern(self, social_graph):
        results = social_graph.match_pattern("(p:Person)-[:WORKS_AT]->(c:Company)")
        assert len(results) == 20  # Each person works at one company

    def test_max_matches(self, social_graph):
        # max_matches parameter is accepted; verify it runs without error
        results = social_graph.match_pattern(
            "(a:Person)-[:KNOWS]->(b:Person)",
            max_matches=5,
        )
        assert len(results) > 0


class TestVariableLengthPaths:
    def test_exact_hops(self, petroleum_graph):
        results = petroleum_graph.match_pattern("(p:Play)-[:HAS_PROSPECT*1]->(pr:Prospect)")
        assert len(results) > 0

    def test_range_hops(self, petroleum_graph):
        results = petroleum_graph.match_pattern("(p:Play)-[*1..2]->(d)")
        assert len(results) > 0

    def test_star_only(self, petroleum_graph):
        results = petroleum_graph.match_pattern("(p:Play)-[*]->(d:Discovery)")
        assert len(results) > 0


class TestNoMatches:
    def test_no_matching_type(self, small_graph):
        results = small_graph.match_pattern("(n:NonExistent)")
        assert len(results) == 0

    def test_no_matching_edge(self, small_graph):
        results = small_graph.match_pattern("(a:Person)-[:WORKS_AT]->(b:Company)")
        assert len(results) == 0


class TestPropertyFilters:
    """Property filter tests migrated from pytest/test_pattern_matching.py."""

    def test_node_property_filter_string(self, small_graph):
        """Filter nodes by string property (city)."""
        results = small_graph.match_pattern('(p:Person {city: "Oslo"})')
        assert len(results) == 2  # Alice and Charlie

    def test_node_property_filter_int(self, small_graph):
        """Filter nodes by integer property (age) and check value."""
        results = small_graph.match_pattern("(p:Person {age: 28})")
        assert len(results) == 1
        assert results[0]["p"]["properties"]["age"] == 28

    def test_edge_property_filter(self, small_graph):
        """Filter edges by property (since)."""
        results = small_graph.match_pattern("(a:Person)-[k:KNOWS {since: 2020}]->(b:Person)")
        assert len(results) == 1
        assert "k" in results[0]
        assert results[0]["k"]["properties"]["since"] == 2020


class TestResultFormat:
    """Result structure tests migrated from pytest/test_pattern_matching.py."""

    @staticmethod
    def _result_graph():
        """Small graph with email property and weighted edge."""
        import pandas as pd

        from kglite import KnowledgeGraph

        graph = KnowledgeGraph()
        people = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["Alice", "Bob"],
                "email": ["alice@test.com", "bob@test.com"],
            }
        )
        graph.add_nodes(people, "Person", "id", "name")

        knows = pd.DataFrame({"from_id": [1], "to_id": [2], "strength": [0.8]})
        graph.add_connections(knows, "KNOWS", "Person", "from_id", "Person", "to_id", columns=["strength"])
        return graph

    def test_node_binding_structure(self):
        """Node bindings contain type, title, id, properties."""
        graph = self._result_graph()
        matches = graph.match_pattern("(p:Person)")
        assert len(matches) > 0

        m = matches[0]
        assert "p" in m
        assert "type" in m["p"]
        assert "title" in m["p"]
        assert "id" in m["p"]
        assert "properties" in m["p"]

    def test_edge_binding_structure(self):
        """Edge bindings contain connection_type and properties."""
        graph = self._result_graph()
        matches = graph.match_pattern("(a:Person)-[k:KNOWS]->(b:Person)")
        assert len(matches) == 1

        m = matches[0]
        assert "k" in m
        assert "connection_type" in m["k"]
        assert "properties" in m["k"]
        assert m["k"]["connection_type"] == "KNOWS"

    def test_properties_accessible(self):
        """Node properties dict includes extra columns (email)."""
        graph = self._result_graph()
        matches = graph.match_pattern("(p:Person)")
        for m in matches:
            assert "email" in m["p"]["properties"]

    def test_edge_properties_accessible(self):
        """Edge properties dict includes extra columns (strength)."""
        graph = self._result_graph()
        matches = graph.match_pattern("(a:Person)-[k:KNOWS]->(b:Person)")
        assert len(matches) == 1
        assert matches[0]["k"]["properties"]["strength"] == 0.8


class TestVariableHandling:
    """Variable binding tests migrated from pytest/test_pattern_matching.py."""

    @staticmethod
    def _var_graph():
        """Small graph with Node type and CONN edges."""
        import pandas as pd

        from kglite import KnowledgeGraph

        graph = KnowledgeGraph()
        nodes = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        graph.add_nodes(nodes, "Node", "id", "name")

        edges = pd.DataFrame({"src": [1, 2], "dst": [2, 3]})
        graph.add_connections(edges, "CONN", "Node", "src", "Node", "dst")
        return graph

    def test_anonymous_edge(self):
        """Edge without a variable still returns node bindings."""
        graph = self._var_graph()
        matches = graph.match_pattern("(a:Node)-[:CONN]->(b:Node)")
        assert len(matches) == 2
        for m in matches:
            assert "a" in m
            assert "b" in m

    def test_node_only_variable(self):
        """Bare variable without type matches all nodes."""
        graph = self._var_graph()
        matches = graph.match_pattern("(x)")
        # At least the 3 regular nodes (schema nodes may add more)
        assert len(matches) >= 3

    def test_multiple_variables(self):
        """Named node and edge variables all appear in results."""
        graph = self._var_graph()
        matches = graph.match_pattern("(a:Node)-[e:CONN]->(b:Node)")
        for m in matches:
            assert "a" in m
            assert "e" in m
            assert "b" in m
