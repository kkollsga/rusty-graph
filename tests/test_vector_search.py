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
            "summary": ["alpha text", "beta text", "gamma text", "delta text", "epsilon text"],
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
    result = graph.set_embeddings("Article", "summary", embeddings)

    assert result["embeddings_stored"] == 5
    assert result["dimension"] == 3
    assert result["skipped"] == 0

    return graph


# ── set_embeddings / embeddings ────────────────────────────────────────


class TestSetGetEmbeddings:
    def test_set_and_embeddings(self, graph_with_embeddings):
        graph = graph_with_embeddings
        embs = graph.select("Article").embeddings("summary")

        assert len(embs) == 5
        assert embs[1] == [1.0, 0.0, 0.0]
        assert embs[2] == [0.0, 1.0, 0.0]

    def test_embeddings_filtered_selection(self, graph_with_embeddings):
        graph = graph_with_embeddings
        embs = (
            graph.select("Article")
            .where({"category": "politics"})
            .embeddings("summary")
        )

        assert len(embs) == 2
        assert 1 in embs  # Alpha
        assert 3 in embs  # Gamma

    def test_set_embeddings_with_missing_ids(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "title")

        result = graph.set_embeddings(
            "Node", "title", {1: [1.0, 2.0], 2: [3.0, 4.0], 999: [5.0, 6.0]}
        )

        assert result["embeddings_stored"] == 2
        assert result["skipped"] == 1

    def test_set_embeddings_inconsistent_dimensions(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "title": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "title")

        with pytest.raises(ValueError, match="Inconsistent embedding dimensions"):
            graph.set_embeddings(
                "Node", "title", {1: [1.0, 2.0], 2: [3.0, 4.0, 5.0]}
            )

    def test_set_embeddings_empty_dict(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"]})
        graph.add_nodes(df, "Node", "id", "title")

        result = graph.set_embeddings("Node", "title", {})
        assert result["embeddings_stored"] == 0

    def test_embeddings_two_arg_form(self, graph_with_embeddings):
        """embeddings(node_type, text_column) returns all embeddings."""
        graph = graph_with_embeddings
        embs = graph.embeddings("Article", "summary")

        assert len(embs) == 5
        assert embs[1] == [1.0, 0.0, 0.0]
        assert embs[2] == [0.0, 1.0, 0.0]

    def test_embeddings_two_arg_nonexistent(self, graph_with_embeddings):
        """Two-arg form returns empty dict for nonexistent store."""
        graph = graph_with_embeddings
        embs = graph.embeddings("Article", "nonexistent")
        assert embs == {}

    def test_embedding_single_node(self, graph_with_embeddings):
        """embedding(node_type, text_column, node_id) returns one vector."""
        graph = graph_with_embeddings
        vec = graph.embedding("Article", "summary", 1)
        assert vec == [1.0, 0.0, 0.0]

        vec2 = graph.embedding("Article", "summary", 4)
        assert vec2 == [0.0, 0.0, 1.0]

    def test_embedding_nonexistent_node(self, graph_with_embeddings):
        """embedding returns None for a node ID that doesn't exist."""
        graph = graph_with_embeddings
        assert graph.embedding("Article", "summary", 999) is None

    def test_embedding_nonexistent_store(self, graph_with_embeddings):
        """embedding returns None for a property name that doesn't exist."""
        graph = graph_with_embeddings
        assert graph.embedding("Article", "no_such", 1) is None


class TestSetEmbeddingsValidation:
    """Tests for set_embeddings source column validation."""

    def test_set_embeddings_missing_node_type(self):
        """set_embeddings raises when node type doesn't exist."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"]})
        graph.add_nodes(df, "Node", "id", "title")

        with pytest.raises(ValueError, match="does not exist"):
            graph.set_embeddings("NonExistent", "title", {1: [1.0, 2.0]})

    def test_set_embeddings_missing_source_column(self):
        """set_embeddings raises when source column doesn't exist on nodes."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"]})
        graph.add_nodes(df, "Node", "id", "title")

        with pytest.raises(ValueError, match="Source column"):
            graph.set_embeddings("Node", "nonexistent", {1: [1.0, 2.0]})

    def test_set_embeddings_builtin_columns_accepted(self):
        """Builtin columns (id, title, type) are always accepted."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"]})
        graph.add_nodes(df, "Node", "id", "title")

        for col in ["id", "title", "type"]:
            result = graph.set_embeddings("Node", col, {1: [1.0, 2.0]})
            assert result["embeddings_stored"] == 1


# ── list_embeddings / remove_embeddings ────────────────────────────────────


class TestListRemoveEmbeddings:
    def test_list_embeddings(self, graph_with_embeddings):
        graph = graph_with_embeddings
        listing = graph.list_embeddings()

        assert len(listing) == 1
        info = listing[0]
        assert info["node_type"] == "Article"
        assert info["text_column"] == "summary"
        assert info["dimension"] == 3
        assert info["count"] == 5

    def test_remove_embeddings(self, graph_with_embeddings):
        graph = graph_with_embeddings
        graph.remove_embeddings("Article", "summary")

        assert len(graph.list_embeddings()) == 0

    def test_list_embeddings_empty(self):
        graph = kglite.KnowledgeGraph()
        assert graph.list_embeddings() == []


# ── vector_search ──────────────────────────────────────────────────────────


class TestVectorSearch:
    def test_basic_cosine_search(self, graph_with_embeddings):
        graph = graph_with_embeddings

        # Search for vectors similar to [1, 0, 0] (Alpha direction)
        results = graph.select("Article").vector_search(
            "summary", [1.0, 0.0, 0.0], top_k=3
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
            graph.select("Article")
            .where({"category": "sports"})
            .vector_search("summary", [0.0, 1.0, 0.0], top_k=10)
        )

        # Only sports articles: Beta(2) and Epsilon(5)
        assert len(results) == 2
        ids = [r["id"] for r in results]
        assert 2 in ids
        assert 5 in ids

    def test_search_result_contains_properties(self, graph_with_embeddings):
        graph = graph_with_embeddings
        results = graph.select("Article").vector_search(
            "summary", [1.0, 0.0, 0.0], top_k=1
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
        results = graph.select("Article").vector_search(
            "summary", [1.0, 0.0, 0.0], top_k=2
        )

        assert len(results) == 2

    def test_search_dot_product_metric(self, graph_with_embeddings):
        graph = graph_with_embeddings
        results = graph.select("Article").vector_search(
            "summary",
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
        results = graph.select("Article").vector_search(
            "summary",
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
            graph.select("Article").vector_search(
                "summary", [1.0, 0.0], top_k=3  # 2D instead of 3D
            )

    def test_search_invalid_metric(self, graph_with_embeddings):
        graph = graph_with_embeddings

        with pytest.raises(ValueError, match="Unknown metric"):
            graph.select("Article").vector_search(
                "summary", [1.0, 0.0, 0.0], metric="manhattan"
            )

    def test_search_to_df(self, graph_with_embeddings):
        graph = graph_with_embeddings
        df = graph.select("Article").vector_search(
            "summary", [1.0, 0.0, 0.0], top_k=3, to_df=True
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "score" in df.columns
        assert "id" in df.columns

    def test_search_empty_selection(self):
        graph = kglite.KnowledgeGraph()
        results = graph.vector_search("text", [1.0, 0.0, 0.0], top_k=3)
        assert results == []


# ── Embeddings invisible to node API ──────────────────────────────────────


class TestEmbeddingsInvisible:
    def test_embeddings_not_in_get_nodes(self, graph_with_embeddings):
        graph = graph_with_embeddings
        nodes = graph.select("Article").collect()

        for node in nodes:
            assert "summary_emb" not in node

    def test_embeddings_not_in_to_df(self, graph_with_embeddings):
        graph = graph_with_embeddings
        df = graph.select("Article").to_df()

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
            assert listing[0]["text_column"] == "summary"
            assert listing[0]["dimension"] == 3
            assert listing[0]["count"] == 5

            # Verify vector search still works
            results = loaded.select("Article").vector_search(
                "summary", [1.0, 0.0, 0.0], top_k=3
            )
            assert len(results) == 3
            assert results[0]["id"] == 1
            assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

            # Verify actual embedding values
            embs = loaded.select("Article").embeddings("summary")
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
            assert loaded.select("Node").len() == 2
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
        assert listing[0]["text_column"] == "text"
        assert listing[0]["dimension"] == 2
        assert listing[0]["count"] == 3

        # Embeddings should NOT be in node properties
        nodes = graph.select("Doc").collect()
        for node in nodes:
            assert "text_emb" not in node

        # Vector search should work
        results = graph.select("Doc").vector_search(
            "text", [1.0, 0.0], top_k=2
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


# ── embed_texts / search_text (text-level API) ────────────────────────────


class MockEmbedder:
    """Mock embedding model for testing. Maps text to simple vectors."""

    def __init__(self, dimension=3):
        self.dimension = dimension

    def embed(self, texts):
        """Deterministic embeddings: hash text to a unit vector."""
        import hashlib

        vectors = []
        for text in texts:
            h = hashlib.md5(text.encode()).digest()
            raw = [float(b) for b in h[: self.dimension]]
            norm = math.sqrt(sum(x * x for x in raw))
            if norm > 0:
                vectors.append([x / norm for x in raw])
            else:
                vectors.append([0.0] * self.dimension)
        return vectors


class TestSetEmbedder:
    def test_set_embedder_basic(self):
        """set_embedder registers a valid model."""
        graph = kglite.KnowledgeGraph()
        model = MockEmbedder(dimension=3)
        graph.set_embedder(model)  # Should not raise

    def test_set_embedder_missing_dimension(self):
        """Model without dimension attribute raises."""
        graph = kglite.KnowledgeGraph()

        class BadModel:
            def embed(self, texts):
                return []

        with pytest.raises(AttributeError, match="dimension"):
            graph.set_embedder(BadModel())

    def test_set_embedder_missing_embed(self):
        """Model without embed method raises."""
        graph = kglite.KnowledgeGraph()

        class BadModel:
            dimension = 3

        with pytest.raises(AttributeError, match="embed"):
            graph.set_embedder(BadModel())


class TestEmbedTexts:
    def test_embed_texts_basic(self):
        """Embed a text column and verify embeddings are stored."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["Alpha", "Beta", "Gamma"],
                "summary": ["AI research paper", "Sports article", "Tech news"],
            }
        )
        graph.add_nodes(df, "Article", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=3))
        result = graph.embed_texts("Article", "summary", show_progress=False)

        assert result["embedded"] == 3
        assert result["skipped"] == 0
        assert result["dimension"] == 3

        # Verify embeddings are stored as {text_column}_emb
        listing = graph.list_embeddings()
        assert len(listing) == 1
        assert listing[0]["text_column"] == "summary"
        assert listing[0]["count"] == 3

    def test_embed_texts_missing_text_skipped(self):
        """Nodes with null/missing text column are skipped."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "summary": ["Has text", None, "Also has text"],
            }
        )
        graph.add_nodes(df, "Doc", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=3))
        result = graph.embed_texts("Doc", "summary", show_progress=False)

        assert result["embedded"] == 2
        assert result["skipped"] == 1

    def test_embed_texts_batch_sizes(self):
        """Test with various batch sizes."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": list(range(10)),
                "title": [f"Node{i}" for i in range(10)],
                "text": [f"Text content {i}" for i in range(10)],
            }
        )
        graph.add_nodes(df, "Item", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=4))

        # batch_size=1 (many batches) — uses "text" column → "text_emb"
        result1 = graph.embed_texts("Item", "text", batch_size=1, show_progress=False)
        assert result1["embedded"] == 10

        # Verify embeddings stored
        embs = graph.embeddings("Item", "text")
        assert len(embs) == 10

    def test_embed_texts_dimension_mismatch(self):
        """Model returns wrong dimension → error."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"], "text": ["some text"]})
        graph.add_nodes(df, "Node", "id", "title")

        class BadModel:
            dimension = 5  # Claims 5

            def embed(self, texts):
                return [[1.0, 2.0, 3.0]] * len(texts)  # Returns 3

        graph.set_embedder(BadModel())
        with pytest.raises(ValueError, match="dimension"):
            graph.embed_texts("Node", "text", show_progress=False)

    def test_embed_texts_no_matching_nodes(self):
        """No nodes of that type → embedded=0."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"], "text": ["hello"]})
        graph.add_nodes(df, "Node", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=3))
        result = graph.embed_texts("Other", "text", show_progress=False)

        assert result["embedded"] == 0
        assert result["skipped"] == 0

    def test_embed_texts_no_model_set(self):
        """Calling embed_texts without set_embedder raises with skeleton."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"], "text": ["hello"]})
        graph.add_nodes(df, "Node", "id", "title")

        with pytest.raises(RuntimeError, match="set_embedder"):
            graph.embed_texts("Node", "text")

    def test_embed_texts_show_progress(self):
        """show_progress=True works (uses tqdm if available)."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": list(range(5)),
                "title": [f"N{i}" for i in range(5)],
                "text": [f"Text {i}" for i in range(5)],
            }
        )
        graph.add_nodes(df, "Item", "id", "title")
        graph.set_embedder(MockEmbedder(dimension=3))

        # show_progress=True (default) — should work even if tqdm is missing
        result = graph.embed_texts("Item", "text", batch_size=2, show_progress=True)
        assert result["embedded"] == 5

    def test_embed_texts_show_progress_false(self):
        """show_progress=False skips tqdm."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {"id": [1, 2], "title": ["A", "B"], "text": ["hello", "world"]}
        )
        graph.add_nodes(df, "Node", "id", "title")
        graph.set_embedder(MockEmbedder(dimension=3))

        result = graph.embed_texts("Node", "text", show_progress=False)
        assert result["embedded"] == 2

    def test_embed_texts_skips_existing(self):
        """Re-running embed_texts skips already-embedded nodes."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "text": ["hello", "world", "test"],
            }
        )
        graph.add_nodes(df, "Item", "id", "title")
        graph.set_embedder(MockEmbedder(dimension=3))

        # First run: embeds all 3
        r1 = graph.embed_texts("Item", "text", show_progress=False)
        assert r1["embedded"] == 3
        assert r1["skipped_existing"] == 0

        # Second run: all 3 already exist → skip all
        r2 = graph.embed_texts("Item", "text", show_progress=False)
        assert r2["embedded"] == 0
        assert r2["skipped_existing"] == 3

    def test_embed_texts_skips_existing_partial(self):
        """Adding new nodes then re-running embeds only new ones."""
        graph = kglite.KnowledgeGraph()
        df1 = pd.DataFrame(
            {"id": [1, 2], "title": ["A", "B"], "text": ["hello", "world"]}
        )
        graph.add_nodes(df1, "Item", "id", "title")
        graph.set_embedder(MockEmbedder(dimension=3))

        r1 = graph.embed_texts("Item", "text", show_progress=False)
        assert r1["embedded"] == 2

        # Add more nodes
        df2 = pd.DataFrame(
            {"id": [3, 4], "title": ["C", "D"], "text": ["foo", "bar"]}
        )
        graph.add_nodes(df2, "Item", "id", "title")

        # Re-run: only new nodes get embedded
        r2 = graph.embed_texts("Item", "text", show_progress=False)
        assert r2["embedded"] == 2
        assert r2["skipped_existing"] == 2

        # All 4 now have embeddings
        embs = graph.embeddings("Item", "text")
        assert len(embs) == 4

    def test_embed_texts_replace(self):
        """replace=True re-embeds everything."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {"id": [1, 2], "title": ["A", "B"], "text": ["hello", "world"]}
        )
        graph.add_nodes(df, "Item", "id", "title")
        graph.set_embedder(MockEmbedder(dimension=3))

        r1 = graph.embed_texts("Item", "text", show_progress=False)
        assert r1["embedded"] == 2

        # replace=True forces re-embed
        r2 = graph.embed_texts("Item", "text", show_progress=False, replace=True)
        assert r2["embedded"] == 2
        assert r2["skipped_existing"] == 0


class TestSearchText:
    def test_search_text_basic(self):
        """Search with text query and verify results."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
                "summary": [
                    "artificial intelligence",
                    "football game",
                    "machine learning",
                    "quantum physics",
                    "basketball match",
                ],
            }
        )
        graph.add_nodes(df, "Article", "id", "title")

        model = MockEmbedder(dimension=8)
        graph.set_embedder(model)
        graph.embed_texts("Article", "summary", show_progress=False)

        # Search with text — uses "summary" which resolves to "summary_emb"
        results = graph.select("Article").search_text(
            "summary", "artificial intelligence", top_k=3
        )

        assert len(results) == 3
        # First result should be the exact text match
        assert results[0]["id"] == 1
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
        assert "title" in results[0]

    def test_search_text_with_filter(self):
        """search_text within a filtered selection."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "category": ["tech", "sports", "tech"],
                "text": ["AI paper", "Football", "ML study"],
            }
        )
        graph.add_nodes(df, "Doc", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=4))
        graph.embed_texts("Doc", "text", show_progress=False)

        results = (
            graph.select("Doc")
            .where({"category": "tech"})
            .search_text("text", "AI paper", top_k=5)
        )

        # Only tech docs
        ids = [r["id"] for r in results]
        assert 2 not in ids  # sports doc excluded
        assert 1 in ids
        assert 3 in ids

    def test_search_text_to_df(self):
        """search_text with to_df=True returns DataFrame."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {"id": [1, 2], "title": ["A", "B"], "text": ["hello", "world"]}
        )
        graph.add_nodes(df, "Node", "id", "title")

        graph.set_embedder(MockEmbedder(dimension=3))
        graph.embed_texts("Node", "text", show_progress=False)

        result = graph.select("Node").search_text(
            "text", "hello", top_k=2, to_df=True
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "score" in result.columns

    def test_search_text_matches_vector_search(self):
        """search_text and vector_search produce identical results."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "text": ["alpha beta", "gamma delta", "epsilon zeta"],
            }
        )
        graph.add_nodes(df, "Node", "id", "title")

        model = MockEmbedder(dimension=5)
        graph.set_embedder(model)
        graph.embed_texts("Node", "text", show_progress=False)

        # Get query vector manually
        query_vec = model.embed(["alpha beta"])[0]

        # Both paths — search_text uses "text" → "text_emb"
        text_results = graph.select("Node").search_text(
            "text", "alpha beta", top_k=3
        )
        vec_results = graph.select("Node").vector_search(
            "text", query_vec, top_k=3
        )

        assert len(text_results) == len(vec_results)
        for tr, vr in zip(text_results, vec_results):
            assert tr["id"] == vr["id"]
            assert tr["score"] == pytest.approx(vr["score"], abs=1e-5)

    def test_search_text_no_model_set(self):
        """Calling search_text without set_embedder raises with skeleton."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"], "text": ["hello"]})
        graph.add_nodes(df, "Node", "id", "title")

        with pytest.raises(RuntimeError, match="set_embedder"):
            graph.select("Node").search_text("text", "hello")


# ── load/unload lifecycle ─────────────────────────────────────────────────


class MockEmbedderWithLifecycle:
    """Mock embedder that tracks load/unload calls."""

    def __init__(self, dimension=3):
        self.dimension = dimension
        self.load_count = 0
        self.unload_count = 0

    def load(self):
        self.load_count += 1

    def unload(self):
        self.unload_count += 1

    def embed(self, texts):
        import hashlib

        vectors = []
        for text in texts:
            h = hashlib.md5(text.encode()).digest()
            raw = [float(b) for b in h[: self.dimension]]
            norm = math.sqrt(sum(x * x for x in raw))
            if norm > 0:
                vectors.append([x / norm for x in raw])
            else:
                vectors.append([0.0] * self.dimension)
        return vectors


class TestEmbedderLifecycle:
    def _make_graph(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["A", "B", "C"],
                "text": ["alpha", "beta", "gamma"],
            }
        )
        graph.add_nodes(df, "Node", "id", "title", ["text"])
        return graph

    def test_embed_texts_calls_load_and_unload(self):
        """embed_texts calls load() before and unload() after embedding."""
        graph = self._make_graph()
        model = MockEmbedderWithLifecycle(dimension=3)
        graph.set_embedder(model)

        result = graph.embed_texts("Node", "text", show_progress=False)

        assert result["embedded"] == 3
        assert model.load_count == 1
        assert model.unload_count == 1

    def test_embed_texts_calls_unload_on_empty(self):
        """unload() is called even when there are no texts to embed."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"]})
        graph.add_nodes(df, "Node", "id", "title")

        model = MockEmbedderWithLifecycle(dimension=3)
        graph.set_embedder(model)

        result = graph.embed_texts("Node", "text", show_progress=False)

        assert result["embedded"] == 0
        assert model.load_count == 1
        assert model.unload_count == 1

    def test_search_text_calls_load_and_unload(self):
        """search_text calls load() before and unload() after embedding the query."""
        graph = self._make_graph()
        model = MockEmbedderWithLifecycle(dimension=3)
        graph.set_embedder(model)
        graph.embed_texts("Node", "text", show_progress=False)
        model.load_count = 0
        model.unload_count = 0

        results = graph.select("Node").search_text("text", "alpha", top_k=2)

        assert len(results) == 2
        assert model.load_count == 1
        assert model.unload_count == 1

    def test_multiple_embed_calls_load_unload_each_time(self):
        """Each embed_texts call triggers its own load/unload cycle."""
        graph = self._make_graph()
        model = MockEmbedderWithLifecycle(dimension=3)
        graph.set_embedder(model)

        graph.embed_texts("Node", "text", show_progress=False)
        graph.embed_texts("Node", "text", show_progress=False, replace=True)

        assert model.load_count == 2
        assert model.unload_count == 2

    def test_model_without_load_unload_works(self):
        """Models without load/unload still work fine (methods are optional)."""
        graph = self._make_graph()
        model = MockEmbedder(dimension=3)  # no load/unload
        graph.set_embedder(model)

        result = graph.embed_texts("Node", "text", show_progress=False)
        assert result["embedded"] == 3

        results = graph.select("Node").search_text("text", "alpha", top_k=2)
        assert len(results) == 2

    def test_unload_called_on_embed_error(self):
        """unload() is called even if embed() raises an exception."""
        graph = self._make_graph()

        class FailingEmbedder:
            dimension = 3
            load_count = 0
            unload_count = 0

            def load(self):
                self.load_count += 1

            def unload(self):
                self.unload_count += 1

            def embed(self, texts):
                raise RuntimeError("boom")

        model = FailingEmbedder()
        graph.set_embedder(model)

        with pytest.raises(RuntimeError, match="boom"):
            graph.embed_texts("Node", "text", show_progress=False)

        assert model.load_count == 1
        assert model.unload_count == 1


# ── text_score() in Cypher (AST rewrite to vector_score) ──────────────────


class TestCypherTextScore:
    """Tests for text_score() Cypher function."""

    def _make_graph(self):
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
                "summary": [
                    "artificial intelligence",
                    "football game",
                    "machine learning",
                    "quantum physics",
                    "basketball match",
                ],
            }
        )
        graph.add_nodes(df, "Article", "id", "title", ["summary"])
        model = MockEmbedder(dimension=8)
        graph.set_embedder(model)
        graph.embed_texts("Article", "summary", show_progress=False)
        return graph, model

    def test_text_score_in_return(self):
        """text_score in RETURN with ORDER BY + LIMIT."""
        graph, _ = self._make_graph()
        result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title AS title, "
            "text_score(n, 'summary', 'artificial intelligence') AS score "
            "ORDER BY score DESC LIMIT 3",
            to_df=True,
        )
        assert len(result) == 3
        # The same text should score highest (cosine = 1.0)
        assert result.iloc[0]["title"] == "Alpha"
        assert result.iloc[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_text_score_in_where(self):
        """text_score in WHERE clause threshold filter."""
        graph, _ = self._make_graph()
        result = graph.cypher(
            "MATCH (n:Article) "
            "WHERE text_score(n, 'summary', 'artificial intelligence') > 0.99 "
            "RETURN n.title AS title",
            to_df=True,
        )
        titles = result["title"].tolist()
        assert "Alpha" in titles

    def test_text_score_with_param(self):
        """text_score with $param for query text."""
        graph, _ = self._make_graph()
        result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title AS title, "
            "text_score(n, 'summary', $query) AS score "
            "ORDER BY score DESC LIMIT 1",
            params={"query": "artificial intelligence"},
            to_df=True,
        )
        assert result.iloc[0]["title"] == "Alpha"

    def test_text_score_matches_vector_score(self):
        """text_score produces same results as manual embed + vector_score."""
        graph, model = self._make_graph()
        query = "artificial intelligence"
        query_vec = model.embed([query])[0]

        ts_result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title, text_score(n, 'summary', $q) AS score "
            "ORDER BY score DESC",
            params={"q": query},
            to_df=True,
        )
        vs_result = graph.cypher(
            "MATCH (n:Article) "
            "RETURN n.title, vector_score(n, 'summary_emb', $v) AS score "
            "ORDER BY score DESC",
            params={"v": query_vec},
            to_df=True,
        )
        assert len(ts_result) == len(vs_result)
        for i in range(len(ts_result)):
            assert ts_result.iloc[i]["n.title"] == vs_result.iloc[i]["n.title"]
            assert ts_result.iloc[i]["score"] == pytest.approx(
                vs_result.iloc[i]["score"], abs=1e-5
            )

    def test_text_score_no_embedder_error(self):
        """text_score without set_embedder raises helpful error."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["A"], "summary": ["hello"]})
        graph.add_nodes(df, "Article", "id", "title", ["summary"])
        graph.set_embeddings("Article", "summary", {1: [1.0, 0.0, 0.0]})

        with pytest.raises(RuntimeError, match="set_embedder"):
            graph.cypher(
                "MATCH (n:Article) "
                "RETURN text_score(n, 'summary', 'hello') AS score"
            )

    def test_text_score_wrong_arg_count(self):
        """text_score with wrong number of args raises clear error."""
        graph, _ = self._make_graph()
        with pytest.raises(ValueError, match="3 arguments"):
            graph.cypher(
                "MATCH (n:Article) "
                "RETURN text_score(n, 'summary') AS score"
            )

    def test_text_score_non_string_column(self):
        """text_score with non-string column name raises."""
        graph, _ = self._make_graph()
        with pytest.raises(ValueError, match="string literal"):
            graph.cypher(
                "MATCH (n:Article) "
                "RETURN text_score(n, n.summary, 'hello') AS score"
            )

    def test_text_score_fused_topk(self):
        """text_score benefits from fused top-k optimization."""
        graph, _ = self._make_graph()
        plan = graph.cypher(
            "EXPLAIN MATCH (n:Article) "
            "RETURN n.title AS title, "
            "text_score(n, 'summary', 'artificial intelligence') AS score "
            "ORDER BY score DESC LIMIT 2"
        )
        ops = [r['operation'] for r in plan.to_list()]
        assert any("FusedVectorScoreTopK" in op for op in ops)

    def test_text_score_load_unload_lifecycle(self):
        """text_score triggers embedder load/unload lifecycle."""
        graph = kglite.KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "title": ["A", "B"],
                "text": ["hello", "world"],
            }
        )
        graph.add_nodes(df, "Node", "id", "title", ["text"])
        model = MockEmbedderWithLifecycle(dimension=3)
        graph.set_embedder(model)
        graph.embed_texts("Node", "text", show_progress=False)
        model.load_count = 0
        model.unload_count = 0

        graph.cypher(
            "MATCH (n:Node) "
            "RETURN text_score(n, 'text', 'hello') AS score"
        )
        assert model.load_count == 1
        assert model.unload_count == 1

    def test_text_score_with_property_filter(self):
        """text_score combined with WHERE property filter."""
        graph, _ = self._make_graph()
        graph.cypher("MATCH (n:Article) WHERE n.id = 1 SET n.category = 'tech'")
        graph.cypher("MATCH (n:Article) WHERE n.id = 3 SET n.category = 'tech'")

        result = graph.cypher(
            "MATCH (n:Article) "
            "WHERE n.category = 'tech' "
            "RETURN n.title, "
            "text_score(n, 'summary', 'machine learning') AS score "
            "ORDER BY score DESC",
            to_df=True,
        )
        assert len(result) == 2
        titles = result["n.title"].tolist()
        assert "Alpha" in titles or "Gamma" in titles

    def test_text_score_regular_queries_unaffected(self):
        """Regular queries without text_score work normally."""
        graph, _ = self._make_graph()
        result = graph.cypher(
            "MATCH (n:Article) RETURN n.title LIMIT 3",
            to_df=True,
        )
        assert len(result) == 3
