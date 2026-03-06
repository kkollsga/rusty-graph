# KGLite — Lightweight Knowledge Graph for Python

[![PyPI version](https://img.shields.io/pypi/v/kglite)](https://pypi.org/project/kglite/)
[![Python versions](https://img.shields.io/pypi/pyversions/kglite)](https://pypi.org/project/kglite/)
[![License: MIT](https://img.shields.io/pypi/l/kglite)](https://github.com/kkollsga/kglite/blob/main/LICENSE)
[![Docs](https://img.shields.io/readthedocs/kglite)](https://kglite.readthedocs.io)

An embedded, in-memory knowledge graph database for Python — built in Rust for speed, with a Cypher query engine, semantic search, and first-class support for RAG pipelines and AI agents. No server, no setup, no infrastructure. Just `pip install kglite` and go.

## Why KGLite?

- **Zero infrastructure** — runs inside your Python process. No database server to install, configure, or maintain.
- **Fast** — Rust core (via PyO3 + petgraph) with zero-copy where possible. Load millions of nodes without leaving Python.
- **Query with Cypher** — familiar graph query language for pattern matching, mutations, aggregations, and traversals.
- **Built for AI** — semantic search with `text_score()`, schema introspection via `describe()`, and a ready-made MCP server for LLM tool use.
- **DataFrames in, DataFrames out** — bulk-load from pandas, query results as DataFrames. Fits naturally into data science workflows.

## Quick Start

```bash
pip install kglite
```

```python
import kglite

graph = kglite.KnowledgeGraph()

# Create nodes and relationships
graph.cypher("CREATE (:Person {name: 'Alice', age: 28, city: 'Oslo'})")
graph.cypher("CREATE (:Person {name: 'Bob', age: 35, city: 'Bergen'})")
graph.cypher("""
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    CREATE (a)-[:KNOWS]->(b)
""")

# Query — returns a ResultView (lazy; data stays in Rust until accessed)
result = graph.cypher("""
    MATCH (p:Person) WHERE p.age > 30
    RETURN p.name AS name, p.city AS city
    ORDER BY p.age DESC
""")
for row in result:
    print(row['name'], row['city'])

# Or get a pandas DataFrame
df = graph.cypher("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age", to_df=True)

# Persist to disk and reload
graph.save("my_graph.kgl")
loaded = kglite.load("my_graph.kgl")
```

## Use Cases

### RAG & Retrieval Pipelines

Store documents, chunks, and entities as a knowledge graph. Use `text_score()` for semantic similarity search and Cypher for structured retrieval — combine both for hybrid RAG.

```python
graph.cypher("""
    MATCH (c:Chunk)
    RETURN c.text, text_score(c.embedding, $query_vec) AS score
    ORDER BY score DESC LIMIT 5
""", params={"query_vec": query_embedding})
```

### AI Agent Memory & Tool Use

Give LLM agents a structured, queryable memory. `describe()` generates a progressive-disclosure schema that agents can reason over, and the included MCP server exposes the graph as a tool.

```python
xml = graph.describe()  # schema for agent context
prompt = f"You have a knowledge graph:\n{xml}\nAnswer using graph.cypher()."
```

### Data Exploration & Analysis

Load CSVs or DataFrames, explore relationships, run graph algorithms (shortest path, centrality, community detection), and export results — all without leaving your notebook.

```python
graph.add_nodes(data=users_df, node_type='User', unique_id_field='user_id', node_title_field='name')
graph.cypher("MATCH path = shortestPath((a:User {name:'Alice'})-[*]-(b:User {name:'Eve'})) RETURN path")
```

### Codebase Analysis

Parse Python and Rust codebases into a knowledge graph with functions, classes, calls, and imports. Search, trace dependencies, and review code structure.

```python
from kglite.code_tree import build
graph = build(".")
graph.cypher("MATCH (f:Function) RETURN f.name, f.file ORDER BY f.name")
```

## Key Features

| Feature | Description |
|---|---|
| **Cypher queries** | MATCH, CREATE, SET, DELETE, MERGE, aggregations, ORDER BY, LIMIT, SKIP |
| **Semantic search** | Vector embeddings + `text_score()` for similarity ranking |
| **Graph algorithms** | Shortest path, centrality, community detection, clustering |
| **Spatial** | Coordinates, WKT geometry, distance and containment queries |
| **Timeseries** | Time-indexed data with `ts_*()` Cypher functions |
| **Bulk loading** | Fluent API (`add_nodes` / `add_connections`) for DataFrames |
| **Blueprints** | Declarative CSV-to-graph loading via JSON config |
| **Import/Export** | Save/load snapshots, GraphML, CSV export |
| **AI integration** | `describe()` introspection, MCP server, agent prompts |
| **Code analysis** | Parse codebases via tree-sitter (`kglite.code_tree`) |

## Documentation

Full docs at **[kglite.readthedocs.io](https://kglite.readthedocs.io)**:

- [Getting Started](https://kglite.readthedocs.io/en/latest/getting-started.html) — installation, first graph, core concepts
- [Cypher Guide](https://kglite.readthedocs.io/en/latest/guides/cypher.html) — queries, mutations, parameters
- [Semantic Search](https://kglite.readthedocs.io/en/latest/guides/semantic-search.html) — embeddings, vector search
- [AI Agents](https://kglite.readthedocs.io/en/latest/guides/ai-agents.html) — MCP server, `describe()`, agent prompts
- [API Reference](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html) — full auto-generated reference

## Requirements

Python 3.10+ (CPython) | macOS (ARM/Intel), Linux (x86_64/aarch64), Windows (x86_64) | `pandas >= 1.5`

## License

MIT — see [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE) for details.
