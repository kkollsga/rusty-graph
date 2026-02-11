"""Tests for Cypher map projections: n {.prop1, .prop2, alias: expr}"""
import pytest
import kglite
import pandas as pd


@pytest.fixture
def social_graph():
    g = kglite.KnowledgeGraph()
    persons = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [30, 25, 35],
        "city": ["Oslo", "Bergen", "Oslo"],
    })
    companies = pd.DataFrame({
        "id": [10, 11],
        "name": ["Acme", "Globex"],
        "industry": ["Tech", "Manufacturing"],
    })
    g.add_nodes(persons, "Person", unique_id_field="id", node_title_field="name")
    g.add_nodes(companies, "Company", unique_id_field="id", node_title_field="name")
    conns = pd.DataFrame({
        "source_id": [1, 2, 3],
        "target_id": [10, 11, 10],
        "since": [2020, 2021, 2019],
    })
    g.add_connections(conns, "WORKS_AT", "Person", "source_id", "Company", "target_id")
    return g


class TestMapProjectionBasic:
    def test_property_shorthand(self, social_graph):
        """Test .prop shorthand syntax."""
        result = social_graph.cypher(
            "MATCH (p:Person) RETURN p {.name, .age} AS info ORDER BY p.name"
        )
        assert len(result) == 3
        assert result[0]['info'] == {'name': 'Alice', 'age': 30}
        assert result[1]['info'] == {'name': 'Bob', 'age': 25}
        assert result[2]['info'] == {'name': 'Charlie', 'age': 35}

    def test_single_property(self, social_graph):
        """Test projection with a single property."""
        result = social_graph.cypher(
            "MATCH (p:Person) RETURN p {.name} AS info ORDER BY p.name LIMIT 1"
        )
        assert result[0]['info'] == {'name': 'Alice'}

    def test_system_properties(self, social_graph):
        """Test projection with system properties (id, type, title)."""
        result = social_graph.cypher(
            "MATCH (p:Person) RETURN p {.id, .name, .type} AS info ORDER BY p.name LIMIT 1"
        )
        assert result[0]['info']['id'] == 1
        assert result[0]['info']['name'] == 'Alice'
        assert result[0]['info']['type'] == 'Person'

    def test_excludes_unselected_properties(self, social_graph):
        """Map projection should only include the specified properties."""
        result = social_graph.cypher(
            "MATCH (p:Person) RETURN p {.name} AS info ORDER BY p.name LIMIT 1"
        )
        assert 'age' not in result[0]['info']
        assert 'city' not in result[0]['info']

    def test_null_property(self, social_graph):
        """Projecting a non-existent property should return null."""
        result = social_graph.cypher(
            "MATCH (p:Person) RETURN p {.name, .nonexistent} AS info ORDER BY p.name LIMIT 1"
        )
        assert result[0]['info']['name'] == 'Alice'
        assert result[0]['info']['nonexistent'] is None


class TestMapProjectionComputed:
    def test_alias_with_expression(self, social_graph):
        """Test alias: expr syntax in projection."""
        result = social_graph.cypher(
            "MATCH (p:Person) RETURN p {.name, label: p.type} AS info ORDER BY p.name LIMIT 1"
        )
        assert result[0]['info']['name'] == 'Alice'
        assert result[0]['info']['label'] == 'Person'

    def test_alias_with_property_access(self, social_graph):
        """Test computed field accessing another property."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            RETURN p {.name, location: p.city} AS info
            ORDER BY p.name LIMIT 1
        """)
        assert result[0]['info']['name'] == 'Alice'
        assert result[0]['info']['location'] == 'Oslo'

    def test_mixed_shorthand_and_alias(self, social_graph):
        """Test mixing .prop shorthand with alias: expr."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            RETURN p {.name, .age, home: p.city} AS info
            ORDER BY p.name
        """)
        assert result[0]['info'] == {'name': 'Alice', 'age': 30, 'home': 'Oslo'}


class TestMapProjectionIntegration:
    def test_with_where_clause(self, social_graph):
        """Map projections with WHERE filtering."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE p.city = 'Oslo'
            RETURN p {.name, .city} AS info
            ORDER BY p.name
        """)
        assert len(result) == 2
        assert result[0]['info']['name'] == 'Alice'
        assert result[1]['info']['name'] == 'Charlie'

    def test_with_relationship_pattern(self, social_graph):
        """Map projections in multi-hop queries."""
        result = social_graph.cypher("""
            MATCH (p:Person)-[:WORKS_AT]->(c:Company)
            RETURN p {.name}, c {.name, .industry} AS company
            ORDER BY p.name
        """)
        assert len(result) == 3
        assert result[0]['company']['name'] == 'Acme'
        assert result[0]['company']['industry'] == 'Tech'

    def test_without_alias(self, social_graph):
        """Map projection without AS alias uses auto-generated column name."""
        result = social_graph.cypher(
            "MATCH (p:Person) RETURN p {.name} ORDER BY p.name LIMIT 1"
        )
        # Column name should be something like "p {.name}"
        row = result[0]
        values = list(row.values())
        assert values[0] == {'name': 'Alice'}

    def test_with_limit(self, social_graph):
        """Map projections work with LIMIT."""
        result = social_graph.cypher(
            "MATCH (p:Person) RETURN p {.name, .age} AS info ORDER BY p.age LIMIT 2"
        )
        assert len(result) == 2
        assert result[0]['info']['age'] == 25
        assert result[1]['info']['age'] == 30
