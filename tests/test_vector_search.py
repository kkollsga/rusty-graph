"""Tests for embedding storage and vector search functionality."""

import math
import os
import tempfile

import pandas as pd
import pytest

import kglite


@pytest.fixture
def graph_with_embeddings():
    """Create a graph with articles and embeddings."""
    graph = kglite.KnowledgeGraph()

    # Create articles
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
            "category": ["politics", "sports", "politics", "tech", "sports"],
        }
    )
    graph.add_nodes(df, "Article", "id", "title")

    # Set embeddings (3-dimensional for simplicity)
    embeddings = {
        1: [1.0, 0.0, 0.0],  # Alpha: points along x-axis
        2: [0.0, 1.0, 0.0],  # Beta: points along y-axis
        3: [0.9, 0.1, 0.0],  # Gamma: mostly x-axis (similar to Alpha)
        4: [0.0, 0.0, 1.0],  # Delta: points along z-axis
        5: [0.1, 0.9, 0.0],  # Epsilon: mostly y-axis (similar to Beta)
    }
    result = graph.set_embeddings("Article", "summary_emb", embeddings)

    assert result["embeddings_stored"] == 5
    assert result["dimension"] == 3
    assert result["skipped"] == 0

    return graph


# ── set_embeddings / get_embeddings ────────────────────────────────────────


class TestSetGetEmbeddings:
    def test_set_and_get_embeddings(self, graph_with_embeddings):
        graph = graph_with_embeddings
        embs = graph.type_filter("Article").get_embeddings("summary_emb")

        assert len(embs) == 5
        assert embs[1] == [1.0, 0.0, 0.0]
        assert embs[2] == [0.0, 1.0, 0.0]

    def test_get_embeddings_filtered_selection(self, graph_with_embeddings):
        graph = graph_with_embeddings
        embs = (
            graph.type_filter("Article")
            .filter({"category": "politics"})
            .get_embeddings("summary_emb")
        )

        assert len(embs) == 2
        assert 1 in embs  # Alpha
        assert 3 in embs  # Gamma

    def test_set_embeddings_with_missing_ids(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "title")

        result = graph.set_embeddings(
            "Node", "emb", {1: [1.0, 2.0], 2: [3.0, 4.0], 999: [5.0, 6.0]}
        )

        assert result["embeddings_stored"] == 2
        assert result["skipped"] == 1

    def test_set_embeddings_inconsistent_dimensions(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "title")

        with pytest.raises(ValueError, match="Inconsistent embedding dimensions"):
            graph.set_embeddings(
                "Node", "emb", {1: [1.0, 2.0], 2: [3.0, 4.0, 5.0]}
            )

    def test_set_embeddings_empty_dict(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"]})
        graph.add_nodes(df, "Node", "id", "title")

        result = graph.set_embeddings("Node", "emb", {})
        assert result["embeddings_stored"] == 0


# ── list_embeddings / remove_embeddings ────────────────────────────────────


class TestListRemoveEmbeddings:
    def test_list_embeddings(self, graph_with_embeddings):
        graph = graph_with_embeddings
        listing = graph.list_embeddings()

        assert len(listing) == 1
        info = listing[0]
        assert info["node_type"] == "Article"
        assert info["property_name"] == "summary_emb"
        assert info["dimension"] == 3
        assert info["count"] == 5

    def test_remove_embeddings(self, graph_with_embeddings):
        graph = graph_with_embeddings
        graph.remove_embeddings("Article", "summary_emb")

        assert len(graph.list_embeddings()) == 0

    def test_list_embeddings_empty(self):
        graph = kglite.KnowledgeGraph()
        assert graph.list_embeddings() == []


# ── vector_search ──────────────────────────────────────────────────────────


class TestVectorSearch:
    def test_basic_cosine_search(self, graph_with_embeddings):
        graph = graph_with_embeddings

        # Search for vectors similar to [1, 0, 0] (Alpha direction)
        results = graph.type_filter("Article").vector_search(
            "summary_emb", [1.0, 0.0, 0.0], top_k=3
        )

        assert len(results) == 3
        # Alpha (exact match) should be first
        assert results[0]["id"] == 1
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
        # Gamma (0.9, 0.1, 0) should be second
        assert results[1]["id"] == 3
        # Scores should be descending
        assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]

    def test_search_with_filter(self, graph_with_embeddings):
        graph = graph_with_embeddings

        # Search only within "sports" category
        results = (
            graph.type_filter("Article")
            .filter({"category": "sports"})
            .vector_search("summary_emb", [0.0, 1.0, 0.0], top_k=10)
        )

        # Only sports articles: Beta(2) and Epsilon(5)
        assert len(results) == 2
        ids = [r["id"] for r in results]
        assert 2 in ids
        assert 5 in ids

    def test_search_result_contains_properties(self, graph_with_embeddings):
        graph = graph_with_embeddings
        results = graph.type_filter("Article").vector_search(
            "summary_emb", [1.0, 0.0, 0.0], top_k=1
        )

        assert len(results) == 1
        r = results[0]
        assert "id" in r
        assert "title" in r
        assert "type" in r
        assert "score" in r
        assert "category" in r
        assert r["title"] == "Alpha"
        assert r["type"] == "Article"

    def test_search_top_k_limits_results(self, graph_with_embeddings):
        graph = graph_with_embeddings
        results = graph.type_filter("Article").vector_search(
            "summary_emb", [1.0, 0.0, 0.0], top_k=2
        )

        assert len(results) == 2

    def test_search_dot_product_metric(self, graph_with_embeddings):
        graph = graph_with_embeddings
        results = graph.type_filter("Article").vector_search(
            "summary_emb",
            [1.0, 0.0, 0.0],
            top_k=3,
            metric="dot_product",
        )

        assert len(results) == 3
        # Alpha has highest dot product with [1,0,0]
        assert results[0]["id"] == 1
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_search_euclidean_metric(self, graph_with_embeddings):
        graph = graph_with_embeddings
        results = graph.type_filter("Article").vector_search(
            "summary_emb",
            [1.0, 0.0, 0.0],
            top_k=3,
            metric="euclidean",
        )

        assert len(results) == 3
        # Alpha has distance 0 (neg_euclidean = 0.0)
        assert results[0]["id"] == 1
        assert results[0]["score"] == pytest.approx(0.0, abs=1e-5)

    def test_search_dimension_mismatch(self, graph_with_embeddings):
        graph = graph_with_embeddings

        with pytest.raises(ValueError, match="dimension"):
            graph.type_filter("Article").vector_search(
                "summary_emb", [1.0, 0.0], top_k=3  # 2D instead of 3D
            )

    def test_search_invalid_metric(self, graph_with_embeddings):
        graph = graph_with_embeddings

        with pytest.raises(ValueError, match="Unknown metric"):
            graph.type_filter("Article").vector_search(
                "summary_emb", [1.0, 0.0, 0.0], metric="manhattan"
            )

    def test_search_to_df(self, graph_with_embeddings):
        graph = graph_with_embeddings
        df = graph.type_filter("Article").vector_search(
            "summary_emb", [1.0, 0.0, 0.0], top_k=3, to_df=True
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "score" in df.columns
        assert "id" in df.columns

    def test_search_empty_selection(self):
        graph = kglite.KnowledgeGraph()
        results = graph.vector_search("emb", [1.0, 0.0, 0.0], top_k=3)
        assert results == []


# ── Embeddings invisible to node API ──────────────────────────────────────


class TestEmbeddingsInvisible:
    def test_embeddings_not_in_get_nodes(self, graph_with_embeddings):
        graph = graph_with_embeddings
        nodes = graph.type_filter("Article").get_nodes()

        for node in nodes:
            assert "summary_emb" not in node

    def test_embeddings_not_in_to_df(self, graph_with_embeddings):
        graph = graph_with_embeddings
        df = graph.type_filter("Article").to_df()

        assert "summary_emb" not in df.columns


# ── Persistence ───────────────────────────────────────────────────────────


class TestEmbeddingPersistence:
    def test_save_and_load_with_embeddings(self, graph_with_embeddings):
        graph = graph_with_embeddings

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name

        try:
            graph.save(path)

            loaded = kglite.load(path)

            # Check embeddings survived save/load
            listing = loaded.list_embeddings()
            assert len(listing) == 1
            assert listing[0]["node_type"] == "Article"
            assert listing[0]["property_name"] == "summary_emb"
            assert listing[0]["dimension"] == 3
            assert listing[0]["count"] == 5

            # Verify vector search still works
            results = loaded.type_filter("Article").vector_search(
                "summary_emb", [1.0, 0.0, 0.0], top_k=3
            )
            assert len(results) == 3
            assert results[0]["id"] == 1
            assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

            # Verify actual embedding values
            embs = loaded.type_filter("Article").get_embeddings("summary_emb")
            assert embs[1] == pytest.approx([1.0, 0.0, 0.0])
            assert embs[2] == pytest.approx([0.0, 1.0, 0.0])
        finally:
            os.unlink(path)

    def test_save_and_load_without_embeddings(self):
        """Ensure save/load still works for graphs without embeddings."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "title")

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name

        try:
            graph.save(path)
            loaded = kglite.load(path)

            assert loaded.list_embeddings() == []
            assert loaded.type_filter("Node").node_count() == 2
        finally:
            os.unlink(path)


# ── add_nodes with embedding columns ─────────────────────────────────────


class TestAddNodesEmbeddings:
    def test_add_nodes_with_embedding_column_type(self):
        graph = kglite.KnowledgeGraph()

        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "text_emb": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            }
        )
        graph.add_nodes(
            df,
            "Doc",
            "id",
            "title",
            column_types={"text_emb": "embedding"},
        )

        # Embeddings should be stored
        listing = graph.list_embeddings()
        assert len(listing) == 1
        assert listing[0]["property_name"] == "text_emb"
        assert listing[0]["dimension"] == 2
        assert listing[0]["count"] == 3

        # Embeddings should NOT be in node properties
        nodes = graph.type_filter("Doc").get_nodes()
        for node in nodes:
            assert "text_emb" not in node

        # Vector search should work
        results = graph.type_filter("Doc").vector_search(
            "text_emb", [1.0, 0.0], top_k=2
        )
        assert len(results) == 2
        assert results[0]["id"] == 1  # Most similar to [1, 0]


# ── Cypher vector_score() function ────────────────────────────────────────


class TestCypherVectorScore:
    def test_vector_score_in_return(self, graph_with_embeddings):
        graph = graph_with_embeddings
        result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title AS title, vector_score(n, 'summary_emb', [1.0, 0.0, 0.0]) AS score "
            "ORDER BY score DESC LIMIT 3",
            to_df=True,
        )
        assert len(result) == 3
        # Alpha (exact match) should be first
        assert result.iloc[0]["title"] == "Alpha"
        assert result.iloc[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_vector_score_in_where(self, graph_with_embeddings):
        graph = graph_with_embeddings
        result = graph.cypher(
            "MATCH (n:Article) "
            "WHERE vector_score(n, 'summary_emb', [1.0, 0.0, 0.0]) > 0.9 "
            "RETURN n.title AS title",
            to_df=True,
        )
        titles = result["title"].tolist()
        # Alpha (1.0) and Gamma (0.99+) should pass threshold
        assert "Alpha" in titles
        assert "Gamma" in titles
        assert "Beta" not in titles

    def test_vector_score_with_property_filter(self, graph_with_embeddings):
        graph = graph_with_embeddings
        result = graph.cypher(
            "MATCH (n:Article) "
            "WHERE n.category = 'sports' "
            "RETURN n.title AS title, vector_score(n, 'summary_emb', [0.0, 1.0, 0.0]) AS score "
            "ORDER BY score DESC",
            to_df=True,
        )
        assert len(result) == 2
        # Beta and Epsilon are sports, Beta is closer to [0,1,0]
        assert result.iloc[0]["title"] == "Beta"

    def test_vector_score_with_params(self, graph_with_embeddings):
        graph = graph_with_embeddings
        result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title AS title, vector_score(n, 'summary_emb', $qvec) AS score "
            "ORDER BY score DESC LIMIT 2",
            params={"qvec": [1.0, 0.0, 0.0]},
            to_df=True,
        )
        assert len(result) == 2
        assert result.iloc[0]["title"] == "Alpha"

    def test_vector_score_dot_product_metric(self, graph_with_embeddings):
        graph = graph_with_embeddings
        result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title AS title, "
            "vector_score(n, 'summary_emb', [1.0, 0.0, 0.0], 'dot_product') AS score "
            "ORDER BY score DESC LIMIT 1",
            to_df=True,
        )
        assert result.iloc[0]["title"] == "Alpha"
        assert result.iloc[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_vector_score_missing_embedding(self, graph_with_embeddings):
        graph = graph_with_embeddings
        with pytest.raises(RuntimeError, match="no embedding"):
            graph.cypher(
                "MATCH (n:Article) "
                "RETURN vector_score(n, 'nonexistent_emb', [1.0, 0.0, 0.0]) AS score"
            )

    def test_vector_score_dimension_mismatch(self, graph_with_embeddings):
        graph = graph_with_embeddings
        with pytest.raises(RuntimeError, match="dimension"):
            graph.cypher(
                "MATCH (n:Article) "
                "RETURN vector_score(n, 'summary_emb', [1.0, 0.0]) AS score"
            )
