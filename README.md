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
import pandas as pd
import kglite

# Three storage modes — pick by graph size:
#   default (in-memory)   — small/medium graphs, fastest queries
#   storage="mapped"      — mmap columns, RAM-friendly as you grow
#   storage="disk", path=…  — 100M+ nodes, Wikidata-scale, loaded lazily
graph = kglite.KnowledgeGraph()

# Bulk-load nodes from a DataFrame (also: add_nodes_bulk, from_blueprint,
# load_ntriples, or Cypher CREATE for ad-hoc inserts).
people = pd.DataFrame({
    "id":   ["alice", "bob", "eve"],
    "name": ["Alice", "Bob", "Eve"],
    "age":  [28, 35, 41],
    "city": ["Oslo", "Bergen", "Trondheim"],
})
graph.add_nodes(people, node_type="Person", unique_id_field="id", node_title_field="name")

# Bulk-load relationships the same way (also: add_connections_bulk,
# add_connections_from_source for auto-filter by loaded types).
knows = pd.DataFrame({"src": ["alice", "bob"], "tgt": ["bob", "eve"]})
graph.add_connections(knows, connection_type="KNOWS",
                      source_type="Person", source_id_field="src",
                      target_type="Person", target_id_field="tgt")

# Query — returns a ResultView (lazy; data stays in Rust until accessed).
result = graph.cypher("""
    MATCH (p:Person) WHERE p.age > 30
    RETURN p.name AS name, p.city AS city
    ORDER BY p.age DESC
""")
for row in result:
    print(row['name'], row['city'])

# Or get a pandas DataFrame directly.
df = graph.cypher("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age", to_df=True)

# Persist to disk and reload.
graph.save("my_graph.kgl")
loaded = kglite.load("my_graph.kgl")
```

## Use Cases

### Codebase analysis

Parse Python, Rust, TypeScript, Go, Java, C#, and C++ into a graph of
functions, classes, calls, and imports. Trace who-calls-what, find
dead code, and review structure without leaving your editor. Pairs
naturally with the MCP server so an agent can reason over your repo.

```python
from kglite.code_tree import build

graph = build(".")                                # parse current directory
graph.cypher("""
    MATCH (f:Function)-[:CALLS]->(g:Function)
    RETURN g.name, count(f) AS callers
    ORDER BY callers DESC LIMIT 10
""")
```

### Agentic AI — memory and tool use

Give an LLM a structured memory it can query. `describe()` emits a
compact XML schema that fits in a system prompt, and the bundled MCP
server exposes the whole graph as a Cypher tool — drop-in for Claude,
Cursor, or any MCP-capable agent.

```python
xml = graph.describe()                            # schema for the agent's context
prompt = f"You have a knowledge graph:\n{xml}\nAnswer via graph.cypher()."
# Or: python examples/mcp_server.py path/to/graph.kgl
```

### RAG retrieval

Store documents, chunks, and entities together as one graph. Combine
`text_score()` semantic similarity with Cypher structure — hybrid
retrieval in one query, no second vector DB.

```python
graph.cypher("""
    MATCH (c:Chunk)-[:IN_DOC]->(d:Document)
    RETURN c.text, d.title,
           text_score(c.embedding, $query_vec) AS score
    ORDER BY score DESC LIMIT 5
""", params={"query_vec": query_embedding})
```

### Data exploration and analysis

Load CSVs or DataFrames, walk relationships, run graph algorithms
(shortest path, centrality, community detection), and export — all
from a notebook.

```python
graph.add_nodes(users_df, node_type="User", unique_id_field="user_id", node_title_field="name")
graph.cypher("""
    MATCH path = shortestPath((a:User {name:'Alice'})-[*]-(b:User {name:'Eve'}))
    RETURN path
""")
```

## Examples

The [`examples/`](https://github.com/kkollsga/kglite/tree/main/examples)
directory has runnable, self-contained scripts covering each of the
use cases above:

- **[`code_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/code_graph.py)**
  — build a code knowledge graph from a source directory via
  `code_tree.build`. Produces `Function`, `Class`, `Module`, `File`
  nodes with `CALLS`, `DEFINES`, `IMPORTS` edges.
- **[`legal_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/legal_graph.py)**
  — end-to-end `add_nodes` / `add_connections` from pandas DataFrames,
  covering laws, regulations, and court decisions with citation
  relationships. Good template for adapting to your own domain.
- **[`mcp_server.py`](https://github.com/kkollsga/kglite/blob/main/examples/mcp_server.py)**
  — drop-in MCP server that exposes any `.kgl` file to an LLM (Claude,
  Cursor, …) as a Cypher query tool, with schema disclosure and
  code-graph–aware helpers.
- **[`spatial_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/spatial_graph.py)**
  — declarative CSV→graph loading via a JSON blueprint; regions,
  facilities, and sensors with lat/lon coordinates and pipeline-path
  traversal queries.
- **[`wikidata_disk.py`](https://github.com/kkollsga/kglite/blob/main/examples/wikidata_disk.py)**
  — Wikidata-scale build + disk-mode storage; loads hundreds of
  millions of triples via `load_ntriples` into a mmap-backed graph.

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
