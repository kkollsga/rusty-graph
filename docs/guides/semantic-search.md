# Semantic Search

Store embedding vectors alongside nodes and query them with fast similarity search. Embeddings are stored separately from node properties — they don't appear in `collect()`, `to_df()`, or regular Cypher property access.

## Text-Level API (Recommended)

Register an embedding model once, then embed and search using text column names. The model runs on the Python side — KGLite only stores the resulting vectors.

```python
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._timer = None
        self.dimension = 384  # set in load() if unknown

    def load(self):
        """Called automatically before embedding. Loads model on demand."""
        import threading
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()

    def unload(self, cooldown=60):
        """Called automatically after embedding. Releases after cooldown."""
        import threading
        def _release():
            self._model = None
            self._timer = None
        self._timer = threading.Timer(cooldown, _release)
        self._timer.start()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()

# Register once on the graph
graph.set_embedder(Embedder())

# Embed a text column — stores vectors as "summary_emb" automatically
graph.embed_texts("Article", "summary")
# Embedding Article.summary: 100%|████████| 1000/1000 [00:05<00:00]
# → {'embedded': 1000, 'skipped': 3, 'skipped_existing': 0, 'dimension': 384}

# Search with text — resolves "summary" → "summary_emb" internally
results = graph.select("Article").search_text("summary", "machine learning", top_k=10)
# [{'id': 42, 'title': '...', 'type': 'Article', 'score': 0.95, ...}, ...]
```

**Key details:**

- **Auto-naming:** text column `"summary"` → embedding store key `"summary_emb"` (auto-derived)
- **Incremental:** re-running `embed_texts` skips nodes that already have embeddings — only new nodes get embedded. Pass `replace=True` to force re-embed.
- **Progress bar:** shows a tqdm progress bar by default. Disable with `show_progress=False`.
- **Load/unload lifecycle:** if the model has optional `load()` / `unload()` methods, they are called automatically before and after each embedding operation.
- **Not serialized:** the model is not saved with `save()` — call `set_embedder()` again after deserializing.

```python
# Add new articles, then re-embed — only new ones are processed
graph.embed_texts("Article", "summary")
# → {'embedded': 50, 'skipped': 0, 'skipped_existing': 1000, ...}

# Force full re-embed
graph.embed_texts("Article", "summary", replace=True)

# Combine with filters
results = (graph
    .select("Article")
    .where({"category": "politics"})
    .search_text("summary", "foreign policy", top_k=10))
```

## Low-Level Vector API

If you manage vectors yourself, use the low-level API:

### Storing Embeddings

```python
# Explicit: pass a dict of {node_id: vector}
graph.set_embeddings('Article', 'summary', {
    1: [0.1, 0.2, 0.3, ...],
    2: [0.4, 0.5, 0.6, ...],
})

# Or auto-detect during add_nodes with column_types
df = pd.DataFrame({
    'id': [1, 2, 3],
    'title': ['A', 'B', 'C'],
    'text_emb': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
})
graph.add_nodes(df, 'Doc', 'id', 'title', column_types={'text_emb': 'embedding'})
```

### Vector Search

```python
# Basic search — returns list of dicts sorted by similarity
results = graph.select('Article').vector_search('summary', query_vec, top_k=10)
# [{'id': 5, 'title': '...', 'type': 'Article', 'score': 0.95, ...}, ...]

# Filtered search — only search within a subset
results = (graph
    .select('Article')
    .where({'category': 'politics'})
    .vector_search('summary', query_vec, top_k=10))

# DataFrame output
df = graph.select('Article').vector_search('summary', query_vec, top_k=10, to_df=True)

# Distance metrics: 'cosine' (default), 'dot_product', 'euclidean'
results = graph.select('Article').vector_search(
    'summary', query_vec, top_k=10, metric='dot_product')
```

### Semantic Search in Cypher

`text_score()` enables semantic search directly in Cypher queries. It automatically embeds the query text using the registered model (via `set_embedder()`) and computes similarity:

```python
# Requires: set_embedder() + embed_texts()
graph.cypher("""
    MATCH (n:Article)
    RETURN n.title, text_score(n, 'summary', 'machine learning') AS score
    ORDER BY score DESC LIMIT 10
""")

# With parameters
graph.cypher("""
    MATCH (n:Article)
    WHERE text_score(n, 'summary', $query) > 0.8
    RETURN n.title
""", params={'query': 'artificial intelligence'})

# Combine with graph filters
graph.cypher("""
    MATCH (n:Article)-[:CITED_BY]->(m:Article)
    WHERE n.category = 'politics'
    RETURN m.title, text_score(m, 'summary', 'foreign policy') AS score
    ORDER BY score DESC LIMIT 5
""")
```

## Embedding Utilities

```python
graph.list_embeddings()
# [{'node_type': 'Article', 'text_column': 'summary', 'dimension': 384, 'count': 1000}]

graph.remove_embeddings('Article', 'summary')

# Retrieve all embeddings for a type (no selection needed)
embs = graph.embeddings('Article', 'summary')
# {1: [0.1, 0.2, ...], 2: [0.4, 0.5, ...], ...}

# Retrieve embeddings for current selection only
embs = graph.select('Article').where({'category': 'politics'}).embeddings('summary')

# Get a single node's embedding (O(1) lookup, returns None if not found)
vec = graph.embedding('Article', 'summary', node_id)
```

Embeddings persist across `save()`/`load()` cycles automatically.

## Embedding Export / Import

Export embeddings to a standalone `.kgle` file so they survive graph rebuilds:

```python
# Export all embeddings
stats = graph.export_embeddings("embeddings.kgle")
# {'stores': 2, 'embeddings': 5000}

# Export only specific node types
graph.export_embeddings("embeddings.kgle", ["Article", "Author"])

# Import into a fresh graph — matches by (node_type, node_id)
result = graph.import_embeddings("embeddings.kgle")
# {'stores': 2, 'imported': 4800, 'skipped': 200}
```
