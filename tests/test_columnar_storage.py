"""Integration tests for columnar property storage (Phase B).

Tests verify that enable_columnar() / disable_columnar() preserve
property semantics across Cypher queries, mutations, save/load, and
bulk operations.
"""

import os
import tempfile

import pandas as pd
import pytest

import kglite

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def person_graph():
    """Graph with 5 Person nodes having mixed property types."""
    kg = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "full_name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [30, 25, 35, 28, 42],
            "score": [1.5, 2.7, 3.9, 4.1, 0.8],
            "active": [True, False, True, True, False],
        }
    )
    kg.add_nodes(df, "Person", "id", "id")
    return kg


@pytest.fixture
def multi_type_graph():
    """Graph with Person + Company node types."""
    kg = kglite.KnowledgeGraph()
    persons = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "full_name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
        }
    )
    companies = pd.DataFrame(
        {
            "id": [10, 20],
            "company_name": ["Acme", "Globex"],
            "employees": [100, 200],
        }
    )
    kg.add_nodes(persons, "Person", "id", "id")
    kg.add_nodes(companies, "Company", "id", "id")
    edges = pd.DataFrame({"from": [1, 2], "to": [10, 20]})
    kg.add_connections(edges, "WORKS_AT", "Person", "from", "Company", "to")
    return kg


# ── Basic enable/disable ─────────────────────────────────────────────────────


class TestColumnarBasic:
    def test_enable_disable_flag(self, person_graph):
        assert not person_graph.is_columnar
        person_graph.enable_columnar()
        assert person_graph.is_columnar
        person_graph.disable_columnar()
        assert not person_graph.is_columnar

    def test_enable_idempotent(self, person_graph):
        person_graph.enable_columnar()
        person_graph.enable_columnar()  # second call should be safe
        assert person_graph.is_columnar

    def test_disable_on_non_columnar(self, person_graph):
        person_graph.disable_columnar()  # no-op, should not crash
        assert not person_graph.is_columnar


# ── Property preservation ────────────────────────────────────────────────────


class TestColumnarPropertyPreservation:
    def test_all_properties_survive_enable(self, person_graph):
        before = person_graph.cypher(
            "MATCH (n:Person) RETURN n.full_name, n.age, n.score, n.active ORDER BY n.age"
        ).to_list()

        person_graph.enable_columnar()

        after = person_graph.cypher(
            "MATCH (n:Person) RETURN n.full_name, n.age, n.score, n.active ORDER BY n.age"
        ).to_list()
        assert before == after

    def test_roundtrip_enable_disable(self, person_graph):
        before = person_graph.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()

        person_graph.enable_columnar()
        person_graph.disable_columnar()

        after = person_graph.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()
        assert before == after

    def test_multi_type_properties(self, multi_type_graph):
        kg = multi_type_graph
        persons_before = kg.cypher("MATCH (n:Person) RETURN n.full_name, n.age ORDER BY n.age").to_list()
        companies_before = kg.cypher(
            "MATCH (n:Company) RETURN n.company_name, n.employees ORDER BY n.employees"
        ).to_list()

        kg.enable_columnar()

        persons_after = kg.cypher("MATCH (n:Person) RETURN n.full_name, n.age ORDER BY n.age").to_list()
        companies_after = kg.cypher(
            "MATCH (n:Company) RETURN n.company_name, n.employees ORDER BY n.employees"
        ).to_list()
        assert persons_before == persons_after
        assert companies_before == companies_after


# ── Cypher queries on columnar storage ───────────────────────────────────────


class TestColumnarCypher:
    def test_where_int_filter(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher(
            "MATCH (n:Person) WHERE n.age > 30 RETURN n.full_name ORDER BY n.full_name"
        ).to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Charlie", "Eve"]

    def test_where_float_filter(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher(
            "MATCH (n:Person) WHERE n.score >= 3.0 RETURN n.full_name ORDER BY n.full_name"
        ).to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Charlie", "Diana"]

    def test_where_bool_filter(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher(
            "MATCH (n:Person) WHERE n.active = true RETURN n.full_name ORDER BY n.full_name"
        ).to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Alice", "Charlie", "Diana"]

    def test_where_string_equals(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Bob' RETURN n.age").to_list()
        assert result == [{"n.age": 25}]

    def test_order_by_columnar(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher("MATCH (n:Person) RETURN n.full_name ORDER BY n.score DESC").to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Diana", "Charlie", "Bob", "Alice", "Eve"]

    def test_aggregation_on_columnar(self, person_graph):
        person_graph.enable_columnar()
        result = person_graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt, avg(n.age) AS avg_age").to_list()
        assert result[0]["cnt"] == 5
        assert abs(result[0]["avg_age"] - 32.0) < 0.01

    def test_relationship_traversal_with_columnar(self, multi_type_graph):
        kg = multi_type_graph
        kg.enable_columnar()
        result = kg.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.full_name, c.company_name ORDER BY p.full_name"
        ).to_list()
        assert len(result) == 2
        assert result[0]["p.full_name"] == "Alice"
        assert result[0]["c.company_name"] == "Acme"


# ── Save/Load ────────────────────────────────────────────────────────────────


class TestColumnarSaveLoad:
    def test_save_load_roundtrip(self, person_graph):
        person_graph.enable_columnar()
        before = person_graph.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()

        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "test.kgl")
            person_graph.save(fp)
            kg2 = kglite.load(fp)

        # Loaded graph should NOT be columnar (backward compat)
        assert not kg2.is_columnar

        after = kg2.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()
        assert before == after

    def test_save_load_multi_type(self, multi_type_graph):
        kg = multi_type_graph
        kg.enable_columnar()

        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "test.kgl")
            kg.save(fp)
            kg2 = kglite.load(fp)

        # Check both types survived
        persons = kg2.cypher("MATCH (n:Person) RETURN n.full_name ORDER BY n.full_name").to_list()
        companies = kg2.cypher("MATCH (n:Company) RETURN n.company_name ORDER BY n.company_name").to_list()
        assert [r["n.full_name"] for r in persons] == ["Alice", "Bob", "Charlie"]
        assert [r["n.company_name"] for r in companies] == ["Acme", "Globex"]

    def test_save_load_preserves_edges(self, multi_type_graph):
        kg = multi_type_graph
        kg.enable_columnar()

        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "test.kgl")
            kg.save(fp)
            kg2 = kglite.load(fp)

        result = kg2.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.full_name, c.company_name ORDER BY p.full_name"
        ).to_list()
        assert len(result) == 2


# ── Mutations on columnar storage ────────────────────────────────────────────


class TestColumnarMutations:
    def test_set_property_cypher(self, person_graph):
        person_graph.enable_columnar()
        person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Alice' SET n.age = 99")
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Alice' RETURN n.age").to_list()
        assert result == [{"n.age": 99}]

    def test_set_new_property_cypher(self, person_graph):
        person_graph.enable_columnar()
        person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Bob' SET n.email = 'bob@test.com'")
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Bob' RETURN n.email").to_list()
        assert result == [{"n.email": "bob@test.com"}]

    def test_remove_property_cypher(self, person_graph):
        person_graph.enable_columnar()
        person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Charlie' REMOVE n.score")
        result = person_graph.cypher("MATCH (n:Person) WHERE n.full_name = 'Charlie' RETURN n.score").to_list()
        assert result == [{"n.score": None}]


# ── Node count and graph stats ───────────────────────────────────────────────


class TestColumnarStats:
    def test_node_count_unchanged(self, person_graph):
        count_before = person_graph.cypher("MATCH (n:Person) RETURN count(n) AS c").to_list()[0]["c"]
        person_graph.enable_columnar()
        count_after = person_graph.cypher("MATCH (n:Person) RETURN count(n) AS c").to_list()[0]["c"]
        assert count_before == count_after == 5

    def test_graph_info_with_columnar(self, person_graph):
        person_graph.enable_columnar()
        info = person_graph.graph_info()
        assert info["node_count"] == 5


# ── Mmap directory format ────────────────────────────────────────────────────


class TestMmapStorage:
    def test_save_load_mmap_basic(self, person_graph, tmp_path):
        person_graph.enable_columnar()
        before = person_graph.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()

        mmap_dir = str(tmp_path / "graph_mmap")
        person_graph.save_mmap(mmap_dir)

        kg2 = kglite.load_mmap(mmap_dir)
        assert kg2.is_columnar
        after = kg2.cypher("MATCH (n:Person) RETURN n.full_name, n.age, n.score ORDER BY n.age").to_list()
        assert before == after

    def test_save_load_mmap_multi_type(self, multi_type_graph, tmp_path):
        kg = multi_type_graph
        kg.enable_columnar()

        mmap_dir = str(tmp_path / "multi_mmap")
        kg.save_mmap(mmap_dir)

        kg2 = kglite.load_mmap(mmap_dir)
        persons = kg2.cypher("MATCH (n:Person) RETURN n.full_name ORDER BY n.full_name").to_list()
        companies = kg2.cypher("MATCH (n:Company) RETURN n.company_name ORDER BY n.company_name").to_list()
        assert [r["n.full_name"] for r in persons] == ["Alice", "Bob", "Charlie"]
        assert [r["n.company_name"] for r in companies] == ["Acme", "Globex"]

    def test_save_load_mmap_preserves_edges(self, multi_type_graph, tmp_path):
        kg = multi_type_graph
        kg.enable_columnar()

        mmap_dir = str(tmp_path / "edges_mmap")
        kg.save_mmap(mmap_dir)

        kg2 = kglite.load_mmap(mmap_dir)
        result = kg2.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.full_name, c.company_name ORDER BY p.full_name"
        ).to_list()
        assert len(result) == 2
        assert result[0]["p.full_name"] == "Alice"

    def test_mmap_query_after_load(self, person_graph, tmp_path):
        person_graph.enable_columnar()
        mmap_dir = str(tmp_path / "query_mmap")
        person_graph.save_mmap(mmap_dir)

        kg2 = kglite.load_mmap(mmap_dir)
        result = kg2.cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n.full_name ORDER BY n.full_name").to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Charlie", "Eve"]

    def test_mmap_directory_structure(self, person_graph, tmp_path):
        person_graph.enable_columnar()
        mmap_dir = str(tmp_path / "structure_mmap")
        person_graph.save_mmap(mmap_dir)

        mmap_path = tmp_path / "structure_mmap"
        assert (mmap_path / "manifest.json").exists()
        assert (mmap_path / "topology.zst").exists()
        assert (mmap_path / "Person").is_dir()
        # Should have column files for each property
        person_files = list((mmap_path / "Person").iterdir())
        assert len(person_files) > 0

    def test_mmap_save_non_columnar_graph(self, person_graph, tmp_path):
        """save_mmap on a non-columnar graph saves topology only (no column dirs)."""
        mmap_dir = str(tmp_path / "non_col_mmap")
        person_graph.save_mmap(mmap_dir)

        kg2 = kglite.load_mmap(mmap_dir)
        # No column stores were saved, so it loads as non-columnar
        assert not kg2.is_columnar
        result = kg2.cypher("MATCH (n:Person) RETURN n.full_name ORDER BY n.full_name").to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Alice", "Bob", "Charlie", "Diana", "Eve"]

    def test_load_enable_columnar_on_loaded_graph(self, person_graph, tmp_path):
        """Standard .kgl load followed by enable_columnar should work."""
        fp = str(tmp_path / "test.kgl")
        person_graph.save(fp)
        kg2 = kglite.load(fp)
        kg2.enable_columnar()
        result = kg2.cypher("MATCH (n:Person) RETURN n.full_name ORDER BY n.full_name").to_list()
        names = [r["n.full_name"] for r in result]
        assert names == ["Alice", "Bob", "Charlie", "Diana", "Eve"]
