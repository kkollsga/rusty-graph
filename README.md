# KGLite

[![PyPI version](https://img.shields.io/pypi/v/kglite)](https://pypi.org/project/kglite/)
[![Python versions](https://img.shields.io/pypi/pyversions/kglite)](https://pypi.org/project/kglite/)
[![License: MIT](https://img.shields.io/pypi/l/kglite)](https://github.com/kkollsga/kglite/blob/main/LICENSE)

A knowledge graph that runs inside your Python process. Load data, query with Cypher, do semantic search — no server, no setup, no infrastructure.

> **Two APIs:** Use **Cypher** for querying, mutations, and semantic search. Use the **fluent API** (`add_nodes` / `add_connections`) for bulk-loading DataFrames. Most agent and application code only needs `cypher()`.

| | |
|---|---|
| Embedded, in-process | No server, no network; `import` and go |
| In-memory | Persistence via `save()`/`load()` snapshots |
| Cypher subset | Querying + mutations + `text_score()` for semantic search |
| Single-label nodes | Each node has exactly one type |
| Fluent bulk loading | Import DataFrames with `add_nodes()` / `add_connections()` |

**Requirements:** Python 3.10+ (CPython) | macOS (ARM/Intel), Linux (x86_64/aarch64), Windows (x86_64) | `pandas >= 1.5`

```bash
pip install kglite
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Using with AI Agents](#using-with-ai-agents)
- [Core Concepts](#core-concepts)
- [How It Works](#how-it-works)
- [Return Types](#return-types)
- [Schema Introspection](#schema-introspection)
- [Cypher Queries](#cypher-queries) | [Full Cypher Reference](CYPHER.md)
- [Fluent API: Data Loading](#fluent-api-data-loading)
- [Fluent API: Querying](#fluent-api-querying)
- [Semantic Search](#semantic-search)
- [Graph Algorithms](#graph-algorithms)
- [Spatial Operations](#spatial-operations)
- [Analytics](#analytics)
- [Schema and Indexes](#schema-and-indexes)
- [Import and Export](#import-and-export)
- [Performance](#performance)
- [Common Gotchas](#common-gotchas)
- [Graph Maintenance](#graph-maintenance)
- [Common Recipes](#common-recipes)
- [API Quick Reference](#api-quick-reference)

---

## Quick Start

```python
import kglite

graph = kglite.KnowledgeGraph()

# Create nodes and relationships
graph.cypher("CREATE (:Person {name: 'Alice', age: 28, city: 'Oslo'})")
graph.cypher("CREATE (:Person {name: 'Bob', age: 35, city: 'Bergen'})")
graph.cypher("CREATE (:Person {name: 'Charlie', age: 42, city: 'Oslo'})")
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

# Quick peek at first rows
result.head()      # first 5 rows (returns a new ResultView)
result.head(3)     # first 3 rows

# Or get a pandas DataFrame
df = graph.cypher("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age", to_df=True)

# Persist to disk and reload
graph.save("my_graph.kgl")
loaded = kglite.load("my_graph.kgl")
```

### Loading Data from DataFrames

For bulk loading (thousands of rows), use the fluent API:

```python
import pandas as pd

users_df = pd.DataFrame({
    'user_id': [1001, 1002, 1003],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [28, 35, 42]
})

graph.add_nodes(data=users_df, node_type='User', unique_id_field='user_id', node_title_field='name')

edges_df = pd.DataFrame({'source_id': [1001, 1002], 'target_id': [1002, 1003]})
graph.add_connections(data=edges_df, connection_type='KNOWS', source_type='User',
                      source_id_field='source_id', target_type='User', target_id_field='target_id')

graph.cypher("MATCH (u:User) WHERE u.age > 30 RETURN u.name, u.age")
```

---

## Using with AI Agents

KGLite is designed to work as a self-contained knowledge layer for AI agents. No external database, no server process, no network — just a Python object with a Cypher interface that an agent can query directly.

### The idea

1. **Load or build a graph** from your data (DataFrames, CSVs, APIs)
2. **Give the agent `agent_describe()`** — a single XML string containing the full schema, Cypher reference, property values, and embedding info
3. **The agent writes Cypher queries** using `graph.cypher()` — no other API to learn
4. **Semantic search works natively** — `text_score()` in Cypher, backed by any embedding model you wrap

No vector database, no graph database, no infrastructure. The graph lives in memory and persists to a single `.kgl` file.

### Quick setup

```python
xml = graph.agent_describe()  # schema + Cypher reference + property values as XML
prompt = f"You have a knowledge graph:\n{xml}\nAnswer the user's question using graph.cypher()."
```

### MCP server

Expose the graph to any MCP-compatible agent (Claude, etc.) with a thin server:

```python
from mcp.server.fastmcp import FastMCP
import kglite

graph = kglite.load("my_graph.kgl")
mcp = FastMCP("knowledge-graph")

@mcp.tool()
def describe() -> str:
    """Get the graph schema and Cypher reference."""
    return graph.agent_describe()

@mcp.tool()
def query(cypher: str) -> str:
    """Run a Cypher query and return results."""
    result = graph.cypher(cypher, to_df=True)
    return result.to_markdown()

mcp.run(transport="stdio")
```

The agent calls `describe()` once to learn the schema, then uses `query()` for everything — traversals, aggregations, filtering, and semantic search via `text_score()`.

### Adding semantic search (5-minute setup)

Semantic search lets agents find nodes by meaning, not just exact property matches. Here's the minimal path:

```python
# 1. Wrap any embedding model (local or remote)
class Embedder:
    dimension = 384
    def embed(self, texts: list[str]) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts).tolist()

# 2. Register it on the graph
graph.set_embedder(Embedder())

# 3. Embed a text column (one-time, incremental on re-run)
graph.embed_texts("Article", "summary")

# 4. Now agents can search by meaning in Cypher — no extra API
graph.cypher("""
    MATCH (a:Article)
    WHERE text_score(a, 'summary', 'climate policy') > 0.5
    RETURN a.title, text_score(a, 'summary', 'climate policy') AS score
    ORDER BY score DESC LIMIT 10
""")
```

The model wrapper works with any provider — OpenAI, Cohere, local sentence-transformers, Ollama. See [Semantic Search](#semantic-search) for the full API including load/unload lifecycle, incremental embedding, and low-level vector access.

### Semantic search in agent workflows

```python
# Wrap any local or remote model — only needs .dimension and .embed()
class OpenAIEmbedder:
    dimension = 1536
    def embed(self, texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(input=texts, model="text-embedding-3-small")
        return [e.embedding for e in response.data]

graph.set_embedder(OpenAIEmbedder())
graph.embed_texts("Article", "summary")  # one-time: vectorize all articles

# Now agents can use text_score() in Cypher — no extra API needed
graph.cypher("""
    MATCH (a:Article)
    WHERE text_score(a, 'summary', 'climate policy') > 0.5
    RETURN a.title, text_score(a, 'summary', 'climate policy') AS score
    ORDER BY score DESC LIMIT 10
""")
```

The model wrapper pattern works with any provider (OpenAI, Cohere, local sentence-transformers, Ollama) — see the [Semantic Search](#semantic-search) section for a full load/unload lifecycle example.

### Tips for agent prompts

1. **Start with `agent_describe()`** — gives the agent schema, types, property names with sample values, counts, and full Cypher syntax in one XML string
2. **Use `properties(type)`** for deeper column discovery — shows types, nullability, unique counts, and sample values
3. **Use `sample(type, n=3)`** before writing queries — lets the agent see real data shapes
4. **Prefer Cypher** over the fluent API in agent contexts — closer to natural language, easier for LLMs to generate
5. **Use parameters** (`params={'x': val}`) to prevent injection when passing user input to queries
6. **ResultView is lazy** — agents can call `len(result)` to check row count without converting all rows

### What `agent_describe()` returns

- **Dynamic** (per-graph): node types with counts, property names/types/sample values, connection types with endpoints, indexes, field aliases, embedding stores
- **Static** (always the same): supported Cypher clauses, WHERE operators, functions (including spatial and semantic), mutation syntax, notes

---

## Core Concepts

**Nodes** have three built-in fields — `type` (label), `title` (display name), `id` (unique within type) — plus arbitrary properties. Each node has exactly one type.

**Relationships** connect two nodes with a type (e.g., `:KNOWS`) and optional properties. The Cypher API calls them "relationships"; the fluent API calls them "connections" — same thing.

**Selections** (fluent API) are lightweight views — a set of node indices that flow through chained operations like `type_filter().filter().traverse()`. They don't copy data.

**Atomicity.** Each `cypher()` call is atomic — if any clause fails, the graph remains unchanged. For multi-statement atomicity, use `graph.begin()` transactions. Durability only via explicit `save()`.

---

## How It Works

KGLite stores nodes and relationships in a Rust graph structure ([petgraph](https://github.com/petgraph/petgraph)). Python only sees lightweight handles — data converts to Python objects on access, not on query.

- **Cypher queries** parse, optimize, and execute entirely in Rust, then return a `ResultView` (lazy — rows convert to Python dicts only when accessed)
- **Fluent API** chains build a *selection* (a set of node indices) — no data is copied until you call `get_nodes()`, `to_df()`, etc.
- **Persistence** is via `save()`/`load()` binary snapshots — there is no WAL or auto-save

---

## Return Types

All node-related methods use a consistent key order: **`type`, `title`, `id`**, then other properties.

### Cypher

| Query type | Returns |
|-----------|---------|
| Read (`MATCH...RETURN`) | `ResultView` — lazy container, rows converted on access |
| Read with `to_df=True` | `pandas.DataFrame` |
| Mutation (`CREATE`, `SET`, `DELETE`, `MERGE`) | `ResultView` with `.stats` dict |
| `EXPLAIN` prefix | `str` (query plan, not executed) |

**Spatial return types:** `point()` values are returned as `{'latitude': float, 'longitude': float}` dicts.

### ResultView

`ResultView` is a lazy result container returned by `cypher()`, centrality methods, `get_nodes()`, and `sample()`. Data stays in Rust and is only converted to Python objects when you access it — making `cypher()` calls fast even for large result sets.

```python
result = graph.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age")

len(result)        # row count (O(1), no conversion)
result[0]          # single row as dict (converts that row only)
result[-1]         # negative indexing works

for row in result: # iterate rows as dicts (one at a time)
    print(row)

result.head()      # first 5 rows → new ResultView
result.head(3)     # first 3 rows → new ResultView
result.tail(2)     # last 2 rows → new ResultView

result.to_list()   # all rows as list[dict] (full conversion)
result.to_df()     # pandas DataFrame (full conversion)

result.columns     # column names: ['n.name', 'n.age']
result.stats       # mutation stats (None for read queries)
```

Because `ResultView` supports iteration and indexing, it works anywhere you'd use a list of dicts — existing code that iterates over `cypher()` results continues to work unchanged.

### Node dicts

Every method that returns node data uses the same dict shape:

```python
{'type': 'Person', 'title': 'Alice', 'id': 1, 'age': 28, 'city': 'Oslo'}
#  ^^^^             ^^^^^             ^^^       ^^^ other properties
```

### Retrieval methods (cheapest to most expensive)

| Method | Returns | Notes |
|--------|---------|-------|
| `node_count()` | `int` | No materialization |
| `indices()` | `list[int]` | Raw graph indices |
| `id_values()` | `list[Any]` | Flat list of IDs |
| `get_ids()` | `list[{type, title, id}]` | Identification only |
| `get_titles()` | `list[str]` | Flat list (see below) |
| `get_properties(['a','b'])` | `list[tuple]` | Flat list (see below) |
| `get_nodes()` | `ResultView` or grouped dict | Full node dicts |
| `to_df()` | `DataFrame` | Columns: `type, title, id, ...props` |
| `get_node_by_id(type, id)` | `dict \| None` | O(1) hash lookup |

### Flat vs. grouped results

`get_titles()`, `get_properties()`, and `get_nodes()` automatically flatten when there is only one parent group (the common case). After a traversal with multiple parent groups, they return grouped dicts instead:

```python
# No traversal (single group) → flat list
graph.type_filter('Person').get_titles()
# ['Alice', 'Bob', 'Charlie']

# After traversal (multiple groups) → grouped dict
graph.type_filter('Person').traverse('KNOWS').get_titles()
# {'Alice': ['Bob'], 'Bob': ['Charlie']}

# Override with flatten_single_parent=False to always get grouped
graph.type_filter('Person').get_titles(flatten_single_parent=False)
# {'Root': ['Alice', 'Bob', 'Charlie']}
```

### Centrality methods

All centrality methods (`pagerank`, `betweenness_centrality`, `closeness_centrality`, `degree_centrality`) return:

| Mode | Returns |
|------|---------|
| Default | `ResultView` of `{type, title, id, score}` sorted by score desc |
| `as_dict=True` | `{id: score}` — keyed by node ID (unique per type) |
| `to_df=True` | `DataFrame` with columns `type, title, id, score` |

---

## Schema Introspection

Methods for exploring graph structure — what types exist, what properties they have, and how they connect. Useful for discovering an unfamiliar graph or building dynamic UIs.

### `schema()` — Full graph overview

```python
s = graph.schema()
# {
#   'node_types': {
#     'Person': {'count': 500, 'properties': {'age': 'Int64', 'city': 'String'}},
#     'Company': {'count': 50, 'properties': {'founded': 'Int64'}},
#   },
#   'connection_types': {
#     'KNOWS': {'count': 1200, 'source_types': ['Person'], 'target_types': ['Person']},
#     'WORKS_AT': {'count': 500, 'source_types': ['Person'], 'target_types': ['Company']},
#   },
#   'indexes': ['Person.city', 'Person.(city, age)'],
#   'node_count': 550,
#   'edge_count': 1700,
# }
```

### `connection_types()` — Edge type inventory

```python
graph.connection_types()
# [
#   {'type': 'KNOWS', 'count': 1200, 'source_types': ['Person'], 'target_types': ['Person']},
#   {'type': 'WORKS_AT', 'count': 500, 'source_types': ['Person'], 'target_types': ['Company']},
# ]
```

### `properties(node_type, max_values=20)` — Property details

Per-property statistics for a single node type. Only properties that exist on at least one node are included. The `values` list is included when the unique count is at or below `max_values` (default 20). Set `max_values=0` to never include values, or raise it to see more (e.g., `max_values=100`).

```python
graph.properties('Person')
# {
#   'type':  {'type': 'str', 'non_null': 500, 'unique': 1, 'values': ['Person']},
#   'title': {'type': 'str', 'non_null': 500, 'unique': 500},
#   'id':    {'type': 'int', 'non_null': 500, 'unique': 500},
#   'city':  {'type': 'str', 'non_null': 500, 'unique': 3, 'values': ['Bergen', 'Oslo', 'Stavanger']},
#   'age':   {'type': 'int', 'non_null': 500, 'unique': 45},
#   'email': {'type': 'str', 'non_null': 250, 'unique': 250},
# }

# See all values even for higher-cardinality properties
graph.properties('Person', max_values=100)
```

Raises `KeyError` if the node type doesn't exist.

### `neighbors_schema(node_type)` — Connection topology

Outgoing and incoming connections grouped by (connection type, endpoint type):

```python
graph.neighbors_schema('Person')
# {
#   'outgoing': [
#     {'connection_type': 'KNOWS', 'target_type': 'Person', 'count': 1200},
#     {'connection_type': 'WORKS_AT', 'target_type': 'Company', 'count': 500},
#   ],
#   'incoming': [
#     {'connection_type': 'KNOWS', 'source_type': 'Person', 'count': 1200},
#   ],
# }
```

Raises `KeyError` if the node type doesn't exist.

### `sample(node_type, n=5)` — Quick data peek

Returns the first N nodes of a type as a `ResultView`:

```python
result = graph.sample('Person', n=3)
result[0]          # {'type': 'Person', 'title': 'Alice', 'id': 1, 'age': 28, 'city': 'Oslo'}
result.to_list()   # all rows as list[dict]
result.to_df()     # as DataFrame
```

Returns fewer than N if the type has fewer nodes. Raises `KeyError` if the node type doesn't exist.

### `indexes()` — Unified index list

```python
graph.indexes()
# [
#   {'node_type': 'Person', 'property': 'city', 'type': 'equality'},
#   {'node_type': 'Person', 'properties': ['city', 'age'], 'type': 'composite'},
# ]
```

### `agent_describe()` — AI agent context

Returns a self-contained XML string summarizing the graph structure and supported Cypher syntax. Designed to be included directly in an LLM prompt:

```python
xml = graph.agent_describe()
prompt = f"You have a knowledge graph:\n{xml}\nAnswer the user's question using cypher()."
```

The output includes:

- **Dynamic** (per-graph): node types with counts and property schemas, connection types, indexes
- **Static** (always the same): supported Cypher subset, key API methods, single-label model notes

---

## Cypher Queries

A substantial Cypher subset. See [CYPHER.md](CYPHER.md) for the full reference with examples of every clause.

> **Single-label note:** Each node has exactly one type. `labels(n)` returns a string, not a list. `SET n:OtherLabel` is not supported.

```python
result = graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WHERE p.age > 30 AND f.city = 'Oslo'
    RETURN p.name AS person, f.name AS friend, p.age AS age
    ORDER BY p.age DESC
    LIMIT 10
""")

# Read queries → ResultView (iterate, index, or convert)
for row in result:
    print(f"{row['person']} knows {row['friend']}")

# Pass to_df=True for a DataFrame
df = graph.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age", to_df=True)
```

### Mutations

```python
# CREATE
result = graph.cypher("CREATE (n:Person {name: 'Alice', age: 30, city: 'Oslo'})")
print(result.stats['nodes_created'])  # 1

# SET
graph.cypher("MATCH (n:Person {name: 'Bob'}) SET n.age = 26")

# DELETE / DETACH DELETE
graph.cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")

# MERGE
graph.cypher("""
    MERGE (n:Person {name: 'Alice'})
    ON CREATE SET n.created = 'today'
    ON MATCH SET n.updated = 'today'
""")
```

### Transactions

```python
with graph.begin() as tx:
    tx.cypher("CREATE (:Person {name: 'Alice', age: 30})")
    tx.cypher("CREATE (:Person {name: 'Bob', age: 25})")
    tx.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    # Commits on exit; rolls back on exception
```

### Parameters

```python
graph.cypher(
    "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age",
    params={'min_age': 25}
)
```

### Semantic search in Cypher

`text_score()` enables semantic search directly in Cypher. Requires `set_embedder()` + `embed_texts()`:

```python
graph.cypher("""
    MATCH (n:Article)
    WHERE text_score(n, 'summary', 'machine learning') > 0.8
    RETURN n.title, text_score(n, 'summary', 'machine learning') AS score
    ORDER BY score DESC LIMIT 10
""")
```

### Supported Cypher Subset

| Category | Supported |
|----------|-----------|
| **Clauses** | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `ORDER BY`, `SKIP`, `LIMIT`, `UNWIND`, `UNION`/`UNION ALL`, `CREATE`, `SET`, `DELETE`, `DETACH DELETE`, `REMOVE`, `MERGE`, `EXPLAIN` |
| **Patterns** | Node `(n:Type)`, relationship `-[:REL]->`, variable-length `*1..3`, undirected `-[:REL]-`, properties `{key: val}`, `p = shortestPath(...)` |
| **WHERE** | `=`, `<>`, `<`, `>`, `<=`, `>=`, `=~` (regex), `AND`, `OR`, `NOT`, `IS NULL`, `IS NOT NULL`, `IN [...]`, `CONTAINS`, `STARTS WITH`, `ENDS WITH`, `EXISTS { pattern }`, `EXISTS(( pattern ))` |
| **Functions** | `toUpper`, `toLower`, `toString`, `toInteger`, `toFloat`, `size`, `type`, `id`, `labels`, `coalesce`, `count`, `sum`, `avg`, `min`, `max`, `collect`, `std`, `text_score` |
| **Spatial** | `point`, `distance`, `wkt_contains`, `wkt_intersects`, `wkt_centroid`, `latitude`, `longitude` |
| **Not supported** | `CALL`/stored procedures, `FOREACH`, subqueries, `SET n:Label` (label mutation), multi-label |

See [CYPHER.md](CYPHER.md) for full examples of every feature.

---

## Fluent API: Data Loading

> For most use cases, use [Cypher queries](#cypher-queries). The fluent API is for bulk operations from DataFrames or complex data pipelines.

### Adding Nodes

```python
products_df = pd.DataFrame({
    'product_id': [101, 102, 103],
    'title': ['Laptop', 'Phone', 'Tablet'],
    'price': [999.99, 699.99, 349.99],
    'stock': [45, 120, 30]
})

report = graph.add_nodes(
    data=products_df,
    node_type='Product',
    unique_id_field='product_id',
    node_title_field='title',
    columns=['product_id', 'title', 'price', 'stock'],       # whitelist columns (None = all)
    column_types={'launch_date': 'datetime'},                  # explicit type hints
    conflict_handling='update'  # 'update' | 'replace' | 'skip' | 'preserve'
)
print(f"Created {report['nodes_created']} nodes in {report['processing_time_ms']}ms")
```

### Property Mapping

When adding nodes, `unique_id_field` and `node_title_field` are **mapped** to `id` and `title`. The original column names become **aliases** — they work in Cypher queries and `filter()`, but results always use the canonical names.

| Your DataFrame Column | Stored As | Alias? |
|-----------------------|-----------|--------|
| `unique_id_field` (e.g., `user_id`) | `id` | `n.user_id` resolves to `n.id` |
| `node_title_field` (e.g., `name`) | `title` | `n.name` resolves to `n.title` |
| All other columns | Same name | — |

```python
# After adding with unique_id_field='user_id', node_title_field='name':
graph.cypher("MATCH (u:User) WHERE u.user_id = 1001 RETURN u")  # OK — alias resolves to id
graph.type_filter('User').filter({'user_id': 1001})              # OK — alias works here too
graph.type_filter('User').filter({'id': 1001})                   # Also OK — canonical name

# Results always use canonical names:
# {'id': 1001, 'title': 'Alice', 'type': 'User', ...}  — NOT 'user_id' or 'name'
```

### Creating Connections

```python
purchases_df = pd.DataFrame({
    'user_id': [1001, 1001, 1002],
    'product_id': [101, 103, 102],
    'date': ['2023-01-15', '2023-02-10', '2023-01-20'],
    'quantity': [1, 2, 1]
})

graph.add_connections(
    data=purchases_df,
    connection_type='PURCHASED',
    source_type='User',
    source_id_field='user_id',
    target_type='Product',
    target_id_field='product_id',
    columns=['date', 'quantity']
)
```

> `source_type` and `target_type` each refer to a single node type. To connect nodes of the same type, set both to the same value (e.g., `source_type='Person', target_type='Person'`).

### Working with Dates

```python
graph.add_nodes(
    data=estimates_df,
    node_type='Estimate',
    unique_id_field='estimate_id',
    node_title_field='name',
    column_types={'valid_from': 'datetime', 'valid_to': 'datetime'}
)

graph.type_filter('Estimate').filter({'valid_from': {'>=': '2020-06-01'}})
graph.type_filter('Estimate').valid_at('2020-06-15')
graph.type_filter('Estimate').valid_during('2020-01-01', '2020-06-30')
```

### Batch Property Updates

```python
result = graph.type_filter('Prospect').filter({'status': 'Inactive'}).update({
    'is_active': False,
    'deactivation_reason': 'status_inactive'
})

updated_graph = result['graph']
print(f"Updated {result['nodes_updated']} nodes")
```

### Operation Reports

Operations that modify the graph return detailed reports:

```python
report = graph.add_nodes(data=df, node_type='Product', unique_id_field='product_id')
# report keys: operation, timestamp, nodes_created, nodes_updated, nodes_skipped,
#              processing_time_ms, has_errors, errors

graph.get_last_report()       # most recent operation report
graph.get_operation_index()   # sequential index of last operation
graph.get_report_history()    # all reports
```

---

## Fluent API: Querying

> For most queries, prefer [Cypher](#cypher-queries). The fluent API is for building reusable query chains or when you need `explain()` and selection-based workflows.

### Filtering

```python
graph.type_filter('Product').filter({'price': 999.99})
graph.type_filter('Product').filter({'price': {'<': 500.0}, 'stock': {'>': 50}})
graph.type_filter('Product').filter({'id': {'in': [101, 103]}})
graph.type_filter('Product').filter({'category': {'is_null': True}})

# Orphan nodes (no connections)
graph.filter_orphans(include_orphans=True)
```

### Sorting

```python
graph.type_filter('Product').sort('price')
graph.type_filter('Product').sort('price', ascending=False)
graph.type_filter('Product').sort([('stock', False), ('price', True)])
```

### Traversing the Graph

```python
alice = graph.type_filter('User').filter({'title': 'Alice'})
alice_products = alice.traverse(connection_type='PURCHASED', direction='outgoing')

# Filter and sort traversal targets
expensive = alice.traverse(
    connection_type='PURCHASED',
    filter_target={'price': {'>=': 500.0}},
    sort_target='price',
    max_nodes=10
)

# Get connection information
alice.get_connections(include_node_properties=True)
```

### Set Operations

```python
n3 = graph.type_filter('Prospect').filter({'geoprovince': 'N3'})
m3 = graph.type_filter('Prospect').filter({'geoprovince': 'M3'})

n3.union(m3)                    # all nodes from both (OR)
n3.intersection(m3)             # nodes in both (AND)
n3.difference(m3)               # nodes in n3 but not m3
n3.symmetric_difference(m3)     # nodes in exactly one (XOR)
```

### Retrieving Results

```python
people = graph.type_filter('Person')

# Lightweight (no property materialization)
people.node_count()                     # → 3
people.indices()                        # → [0, 1, 2]
people.id_values()                      # → [1, 2, 3]

# Medium (partial materialization)
people.get_ids()                        # → [{'type': 'Person', 'title': 'Alice', 'id': 1}, ...]
people.get_titles()                     # → ['Alice', 'Bob', 'Charlie']
people.get_properties(['age', 'city'])  # → [(28, 'Oslo'), (35, 'Bergen'), (42, 'Oslo')]

# Full materialization
people.get_nodes()                      # → [{'type': 'Person', 'title': 'Alice', 'id': 1, 'age': 28, ...}, ...]
people.to_df()                          # → DataFrame with columns type, title, id, age, city, ...

# Single node lookup (O(1))
graph.get_node_by_id('Person', 1)       # → {'type': 'Person', 'title': 'Alice', ...} or None
```

### Debugging Selections

```python
result = graph.type_filter('User').filter({'id': 1001})
print(result.explain())
# TYPE_FILTER User (1000 nodes) -> FILTER (1 nodes)
```

### Pattern Matching

For simpler pattern-based queries without full Cypher clause support:

```python
results = graph.match_pattern(
    '(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)'
)

for match in results:
    print(f"Play: {match['p']['title']}, Discovery: {match['d']['title']}")

# With property conditions
graph.match_pattern('(u:User)-[:PURCHASED]->(p:Product {category: "Electronics"})')

# Limit results for large graphs
graph.match_pattern('(a:Person)-[:KNOWS]->(b:Person)', max_matches=100)
```

---

## Semantic Search

Store embedding vectors alongside nodes and query them with fast similarity search. Embeddings are stored separately from node properties — they don't appear in `get_nodes()`, `to_df()`, or regular Cypher property access.

### Text-Level API (Recommended)

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
results = graph.type_filter("Article").search_text("summary", "machine learning", top_k=10)
# [{'id': 42, 'title': '...', 'type': 'Article', 'score': 0.95, ...}, ...]
```

**Key details:**

- **Auto-naming:** text column `"summary"` → embedding store key `"summary_emb"` (auto-derived)
- **Incremental:** re-running `embed_texts` skips nodes that already have embeddings — only new nodes get embedded. Pass `replace=True` to force re-embed.
- **Progress bar:** shows a tqdm progress bar by default. Disable with `show_progress=False`.
- **Load/unload lifecycle:** if the model has optional `load()` / `unload()` methods, they are called automatically before and after each embedding operation. Use this to load on demand and release after a cooldown.
- **Not serialized:** the model is not saved with `save()` — call `set_embedder()` again after deserializing.

```python
# Add new articles, then re-embed — only new ones are processed
graph.embed_texts("Article", "summary")
# → {'embedded': 50, 'skipped': 0, 'skipped_existing': 1000, ...}

# Force full re-embed
graph.embed_texts("Article", "summary", replace=True)

# Combine with filters
results = (graph
    .type_filter("Article")
    .filter({"category": "politics"})
    .search_text("summary", "foreign policy", top_k=10))
```

Calling `embed_texts()` or `search_text()` without `set_embedder()` raises an error with a full skeleton showing the required model interface.

### Storing Embeddings (Low-Level)

If you manage vectors yourself, use the low-level API:

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

### Vector Search (Low-Level)

Search operates on the current selection — combine with `type_filter()` and `filter()` for scoped queries:

```python
# Basic search — returns list of dicts sorted by similarity
results = graph.type_filter('Article').vector_search('summary', query_vec, top_k=10)
# [{'id': 5, 'title': '...', 'type': 'Article', 'score': 0.95, ...}, ...]
# 'score' is always included: cosine similarity [-1,1], dot_product, or negative euclidean distance

# Filtered search — only search within a subset
results = (graph
    .type_filter('Article')
    .filter({'category': 'politics'})
    .vector_search('summary', query_vec, top_k=10))

# DataFrame output
df = graph.type_filter('Article').vector_search('summary', query_vec, top_k=10, to_df=True)

# Distance metrics: 'cosine' (default), 'dot_product', 'euclidean'
results = graph.type_filter('Article').vector_search(
    'summary', query_vec, top_k=10, metric='dot_product')
```

### Semantic Search in Cypher

`text_score()` enables semantic search directly in Cypher queries.
It automatically embeds the query text using the registered model (via `set_embedder()`) and computes similarity:

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

### Embedding Utilities

```python
graph.list_embeddings()
# [{'node_type': 'Article', 'text_column': 'summary', 'dimension': 384, 'count': 1000}]

graph.remove_embeddings('Article', 'summary')

# Retrieve all embeddings for a type (no selection needed)
embs = graph.get_embeddings('Article', 'summary')
# {1: [0.1, 0.2, ...], 2: [0.4, 0.5, ...], ...}

# Retrieve embeddings for current selection only
embs = graph.type_filter('Article').filter({'category': 'politics'}).get_embeddings('summary')

# Get a single node's embedding (O(1) lookup, returns None if not found)
vec = graph.get_embedding('Article', 'summary', node_id)
```

Embeddings persist across `save()`/`load()` cycles automatically.

### Embedding Export / Import

Export embeddings to a standalone `.kgle` file so they survive graph rebuilds. Embeddings are keyed by node ID — import resolves IDs against the current graph, skipping any that no longer exist.

```python
# Export all embeddings
stats = graph.export_embeddings("embeddings.kgle")
# {'stores': 2, 'embeddings': 5000}

# Export only specific node types
graph.export_embeddings("embeddings.kgle", ["Article", "Author"])

# Export specific (node_type, property) pairs — empty list = all properties for that type
graph.export_embeddings("embeddings.kgle", {
    "Article": ["summary", "title"],  # only these two
    "Author": [],                     # all embedding properties for Author
})

# Import into a fresh graph — matches by (node_type, node_id)
graph2 = kglite.KnowledgeGraph()
graph2.add_nodes(articles_df, 'Article', 'id', 'title')
result = graph2.import_embeddings("embeddings.kgle")
# {'stores': 2, 'imported': 4800, 'skipped': 200}
```

This is useful when rebuilding a graph from scratch (e.g., re-running a build script) without re-generating expensive embeddings.

---

## Graph Algorithms

### Shortest Path

```python
result = graph.shortest_path(source_type='Person', source_id=1, target_type='Person', target_id=100)
if result:
    for node in result["path"]:
        print(f"{node['type']}: {node['title']}")
    print(f"Connections: {result['connections']}")
    print(f"Path length: {result['length']}")
```

Lightweight variants when you don't need full path data:

```python
graph.shortest_path_length(...)    # → int | None (hop count only)
graph.shortest_path_ids(...)       # → list[id] | None (node IDs along path)
graph.shortest_path_indices(...)   # → list[int] | None (raw graph indices, fastest)
```

All path methods support `connection_types`, `via_types`, and `timeout_ms` for filtering and safety.

Batch variant for computing many distances at once:

```python
distances = graph.shortest_path_lengths_batch('Person', [(1, 5), (2, 8), (3, 10)])
# → [2, None, 5]  (None where no path exists, same order as input)
```

Much faster than calling `shortest_path_length` in a loop — builds the adjacency list once.

### All Paths

```python
paths = graph.all_paths(
    source_type='Play', source_id=1,
    target_type='Wellbore', target_id=100,
    max_hops=4,
    max_results=100  # Prevent OOM on dense graphs
)
```

### Connected Components

```python
components = graph.connected_components()
# Returns list of lists: [[node_dicts...], [node_dicts...], ...]
print(f"Found {len(components)} connected components")
print(f"Largest component: {len(components[0])} nodes")

graph.are_connected(source_type='Person', source_id=1, target_type='Person', target_id=100)
```

### Centrality Algorithms

All centrality methods return a `ResultView` of `{type, title, id, score}` rows, sorted by score descending.

```python
graph.betweenness_centrality(top_k=10)
graph.betweenness_centrality(normalized=True, sample_size=500)
graph.pagerank(top_k=10, damping_factor=0.85)
graph.degree_centrality(top_k=10)
graph.closeness_centrality(top_k=10)

# Alternative output formats
graph.pagerank(as_dict=True)      # → {1: 0.45, 2: 0.32, ...} (keyed by id)
graph.pagerank(to_df=True)        # → DataFrame with type, title, id, score columns
```

### Community Detection

```python
# Louvain modularity optimization (recommended)
result = graph.louvain_communities()
# {'communities': {0: [{type, title, id}, ...], 1: [...]},
#  'modularity': 0.45, 'num_communities': 2}

for comm_id, members in result['communities'].items():
    names = [m['title'] for m in members]
    print(f"Community {comm_id}: {names}")

# With edge weights and resolution tuning
result = graph.louvain_communities(weight_property='strength', resolution=1.5)

# Label propagation (faster, less precise)
result = graph.label_propagation(max_iterations=100)
```

### Node Degrees

```python
degrees = graph.type_filter('Person').get_degrees()
# Returns: {'Alice': 5, 'Bob': 3, ...}
```

---

## Spatial Operations

> Spatial queries are also available in Cypher via `point()`, `distance()`, `wkt_contains()`, `wkt_intersects()`, and `wkt_centroid()`. See [CYPHER.md](CYPHER.md#spatial-functions).

### Bounding Box

```python
graph.type_filter('Discovery').within_bounds(
    lat_field='latitude', lon_field='longitude',
    min_lat=58.0, max_lat=62.0, min_lon=1.0, max_lon=5.0
)
```

### Distance Queries (Haversine)

```python
graph.type_filter('Wellbore').near_point_km(
    center_lat=60.5, center_lon=3.2, max_distance_km=50.0,
    lat_field='latitude', lon_field='longitude'
)
```

### WKT Geometry Intersection

```python
graph.type_filter('Field').intersects_geometry(
    'POLYGON((1 58, 5 58, 5 62, 1 62, 1 58))',
    geometry_field='wkt_geometry'
)
```

### Point-in-Polygon

```python
graph.type_filter('Block').contains_point(lat=60.5, lon=3.2, geometry_field='wkt_geometry')
```

---

## Analytics

### Statistics

```python
price_stats = graph.type_filter('Product').statistics('price')
unique_cats = graph.type_filter('Product').unique_values(property='category', max_length=10)
```

### Calculations

```python
graph.type_filter('Product').calculate(expression='price * 1.1', store_as='price_with_tax')

graph.type_filter('User').traverse('PURCHASED').calculate(
    expression='sum(price * quantity)', store_as='total_spent'
)

graph.type_filter('User').traverse('PURCHASED').count(store_as='product_count', group_by_parent=True)
```

### Connection Aggregation

```python
graph.type_filter('Discovery').traverse('EXTENDS_INTO').calculate(
    expression='sum(share_pct)',
    aggregate_connections=True
)
```

Supported: `sum`, `avg`/`mean`, `min`, `max`, `count`, `std`.

---

## Schema and Indexes

### Schema Definition

```python
graph.define_schema({
    'nodes': {
        'Prospect': {
            'required': ['npdid_prospect', 'prospect_name'],
            'optional': ['prospect_status'],
            'types': {'npdid_prospect': 'integer', 'prospect_name': 'string'}
        }
    },
    'connections': {
        'HAS_ESTIMATE': {'source': 'Prospect', 'target': 'ProspectEstimate'}
    }
})

errors = graph.validate_schema()
schema = graph.get_schema()
```

### Indexes

Two index types:

| Method | Accelerates | Use for |
|--------|-------------|---------|
| `create_index()` | Equality (`= value`) | Exact lookups |
| `create_range_index()` | Range (`>`, `<`, `>=`, `<=`) | Numeric/date filtering |

Both also accelerate Cypher `WHERE` clauses. Composite indexes support multi-property equality.

```python
graph.create_index('Prospect', 'prospect_geoprovince')        # equality index
graph.create_range_index('Person', 'age')                      # B-Tree range index
graph.create_composite_index('Person', ['city', 'age'])        # composite equality

graph.list_indexes()
graph.drop_index('Prospect', 'prospect_geoprovince')
```

Indexes are maintained automatically by all mutation operations.

---

## Import and Export

### Saving and Loading

```python
graph.save("my_graph.kgl")
loaded_graph = kglite.load("my_graph.kgl")
```

Save files (`.kgl`) use a pinned binary format (bincode with explicit little-endian, fixed-int encoding). Files are forward-compatible within the same major version. For sharing across machines or long-term archival, prefer a portable format (GraphML, CSV).

### Embedding Snapshots

Export embeddings separately so they survive graph rebuilds. See [Embedding Export / Import](#embedding-export--import) under Semantic Search for full details.

```python
graph.export_embeddings("embeddings.kgle")                          # all embeddings
graph.export_embeddings("embeddings.kgle", ["Article"])             # by node type
graph.export_embeddings("embeddings.kgle", {"Article": ["summary"]})  # by type + property

result = graph.import_embeddings("embeddings.kgle")
# {'stores': 2, 'imported': 4800, 'skipped': 200}
```

### Export Formats

```python
graph.export('my_graph.graphml', format='graphml')  # Gephi, yEd
graph.export('my_graph.gexf', format='gexf')        # Gephi native
graph.export('my_graph.json', format='d3')           # D3.js
graph.export('my_graph.csv', format='csv')           # creates _nodes.csv + _edges.csv

graphml_string = graph.export_string(format='graphml')
```

### Subgraph Extraction

```python
subgraph = (
    graph.type_filter('Company')
    .filter({'title': 'Acme Corp'})
    .expand(hops=2)
    .to_subgraph()
)
subgraph.export('acme_network.graphml', format='graphml')
```

---

## Performance

### Tips

1. **Batch operations** — add nodes/connections in batches, not individually
2. **Specify columns** — only include columns you need to reduce memory
3. **Filter by type first** — `type_filter()` before `filter()` for narrower scans
4. **Create indexes** — on frequently filtered equality conditions (~3x on 100k+ nodes)
5. **Use lightweight methods** — `node_count()`, `indices()`, `get_node_by_id()` skip property materialization
6. **Cypher LIMIT** — use `LIMIT` to avoid scanning entire result sets

### Threading

The Python GIL is released during heavy Rust operations, allowing other Python threads to run concurrently:

| Operation | GIL Released? | Notes |
|-----------|:---:|-------|
| `save()` | Yes | Serialization + compression + file write |
| `load()` | Yes | File read + decompression + deserialization |
| `export_embeddings()` | Yes | Serialization + compression + file write |
| `cypher()` (reads) | Yes | Query parsing, optimization, and execution |
| `vector_search()` | Yes | Similarity computation (uses rayon internally) |
| `search_text()` | Partial | Model embedding needs GIL; vector search releases it |
| `add_nodes()` | No | DataFrame conversion requires GIL throughout |
| `import_embeddings()` | No | Mutates graph in-place |
| `cypher()` (mutations) | No | Must hold exclusive lock on graph |

For concurrent access from multiple threads, mutations (`add_nodes`, `CREATE`/`SET`/`DELETE` Cypher) require external synchronization. Read-only operations (`cypher` reads, `vector_search`, `save`) can run while other Python threads execute.

---

## Common Gotchas

- **Single-label only.** Each node has exactly one type. `labels(n)` returns a string, not a list. `SET n:OtherLabel` is not supported.
- **`id` and `title` are canonical.** `add_nodes(unique_id_field='user_id')` stores the column as `id`. The original name works as an alias in Cypher (`n.user_id` resolves to `n.id`), but results always return canonical names (`id`, `title`).
- **Save files use a pinned binary format.** `.kgl` and `.kgle` files use bincode with explicitly pinned encoding options (little-endian, fixed-int). Files are compatible across OS and CPU architecture within the same major version. For long-term archival or sharing with non-kglite tools, use `export()` (GraphML, CSV).
- **Indexes:** `create_index()` accelerates equality only (`=`). For range queries (`>`, `<`, `>=`, `<=`), use `create_range_index()`.
- **Flat vs. grouped results.** After traversal with multiple parents, `get_titles()`, `get_nodes()`, and `get_properties()` return grouped dicts instead of flat lists. Use `flatten_single_parent=False` to always get grouped output.
- **No auto-persistence.** The graph lives in memory. `save()` is manual — crashes lose unsaved work.

---

## Graph Maintenance

After heavy mutation workloads (DELETE, REMOVE), internal storage accumulates tombstones. Monitor with `graph_info()`.

```python
info = graph.graph_info()
# {'node_count': 950, 'node_capacity': 1000, 'node_tombstones': 50,
#  'edge_count': 2800, 'fragmentation_ratio': 0.05,
#  'type_count': 3, 'property_index_count': 2, 'composite_index_count': 0}

if info['fragmentation_ratio'] > 0.3:
    result = graph.vacuum()
    print(f"Reclaimed {result['tombstones_removed']} slots, remapped {result['nodes_remapped']} nodes")
```

`vacuum()` rebuilds the graph with contiguous indices and rebuilds all indexes. **Resets the current selection** — call between query chains.

`reindex()` rebuilds indexes only. Recovery tool, not routine maintenance — indexes are maintained automatically by all mutations.

---

## Common Recipes

### Upsert with MERGE

```python
graph.cypher("""
    MERGE (p:Person {email: 'alice@example.com'})
    ON CREATE SET p.created = '2024-01-01', p.name = 'Alice'
    ON MATCH SET p.last_seen = '2024-01-15'
""")
```

### Top-K Nodes by Centrality

```python
top_nodes = graph.pagerank(top_k=10)
for node in top_nodes:
    print(f"{node['title']}: {node['score']:.3f}")
```

### 2-Hop Neighborhood

```python
graph.cypher("""
    MATCH (me:Person {name: 'Alice'})-[:KNOWS*2]-(fof:Person)
    WHERE fof <> me
    RETURN DISTINCT fof.name
""")
```

### Export Subgraph

```python
subgraph = (
    graph.type_filter('Person')
    .filter({'name': 'Alice'})
    .expand(hops=2)
    .to_subgraph()
)
subgraph.export('alice_network.graphml', format='graphml')
```

### Parameterized Queries

```python
graph.cypher(
    "MATCH (p:Person) WHERE p.city = $city AND p.age > $min_age RETURN p.name",
    params={'city': 'Oslo', 'min_age': 25}
)
```

### Delete Subgraph

```python
graph.cypher("""
    MATCH (u:User) WHERE u.status = 'inactive'
    DETACH DELETE u
""")
```

### Aggregation with Relationship Properties

```python
graph.cypher("""
    MATCH (p:Person)-[r:RATED]->(m:Movie)
    RETURN p.name, avg(r.score) AS avg_rating, count(m) AS movies_rated
    ORDER BY avg_rating DESC
""")
```

---

## API Quick Reference

### Graph lifecycle

```python
graph = kglite.KnowledgeGraph()     # create
graph.save("file.kgl")              # persist
graph = kglite.load("file.kgl")     # reload
graph.graph_info()                   # → dict with node_count, edge_count, fragmentation_ratio, ...
graph.get_schema()                   # → str summary of types and connections
graph.node_types                     # → ['Person', 'Product', ...]
```

### Cypher (recommended for most tasks)

```python
graph.cypher("MATCH (n:Person) RETURN n.name")                          # → ResultView
graph.cypher("MATCH (n:Person) RETURN n.name", to_df=True)              # → DataFrame
graph.cypher("MATCH (n:Person) RETURN n.name", params={'x': 1})         # parameterized
graph.cypher("CREATE (:Person {name: 'Alice'})")                        # → ResultView (.stats has counts)
```

### Data loading (fluent API)

```python
graph.add_nodes(data=df, node_type='T', unique_id_field='id')           # → report dict
graph.add_connections(data=df, connection_type='REL',
    source_type='A', source_id_field='src',
    target_type='B', target_id_field='tgt')                              # → report dict
```

### Selection chain (fluent API)

```python
graph.type_filter('Person')                        # select by type → KnowledgeGraph
    .filter({'age': {'>': 25}})                    # filter → KnowledgeGraph
    .sort('age', ascending=False)                  # sort → KnowledgeGraph
    .traverse('KNOWS', direction='outgoing')       # traverse → KnowledgeGraph
    .get_nodes()                                   # materialize → ResultView or grouped dict
```

### Semantic search

```python
# Text-level API (recommended) — register model once, embed & search by column name
graph.set_embedder(model)                                                    # register model (.dimension, .embed())
graph.embed_texts('Article', 'summary')                                      # embed text column → stored as summary_emb
graph.type_filter('Article').search_text('summary', 'find AI papers', top_k=10)  # text query search

# Low-level vector API — bring your own vectors
graph.set_embeddings('Article', 'summary', {id: vec, ...})             # store embeddings
graph.type_filter('Article').vector_search('summary', qvec, top_k=10)  # similarity search
graph.list_embeddings()                                                 # list all embedding stores
graph.remove_embeddings('Article', 'summary')                           # remove an embedding store
graph.get_embeddings('Article', 'summary')                              # retrieve all vectors for type
graph.type_filter('Article').get_embeddings('summary')                  # retrieve vectors for selection
graph.get_embedding('Article', 'summary', node_id)                      # single node vector (or None)
graph.export_embeddings('emb.kgle')                                     # export all embeddings to file
graph.export_embeddings('emb.kgle', ['Article'])                        # export by node type
graph.export_embeddings('emb.kgle', {'Article': ['summary']})           # export by type + property
graph.import_embeddings('emb.kgle')                                     # import embeddings from file
# Cypher: text_score(n, 'summary', 'query text') — semantic search in Cypher, needs set_embedder()
```

### Introspection

```python
graph.schema()                                # → full graph overview (types, counts, connections, indexes)
graph.connection_types()                      # → list of edge types with counts and endpoint types
graph.properties('Person')                    # → per-property stats (type, non_null, unique, values)
graph.properties('Person', max_values=50)     # → include values list for up to 50 unique values
graph.neighbors_schema('Person')              # → outgoing/incoming connection topology
graph.sample('Person', n=5)                   # → first N nodes as ResultView
graph.indexes()                               # → all indexes with type info
graph.agent_describe()                        # → XML string for LLM prompt context
```

### Algorithms

```python
graph.shortest_path(source_type, source_id, target_type, target_id)  # → {path, connections, length} | None
graph.all_paths(source_type, source_id, target_type, target_id)      # → list[{path, connections, length}]
graph.pagerank(top_k=10)                                             # → ResultView of {type, title, id, score}
graph.betweenness_centrality(top_k=10)                               # → ResultView of {type, title, id, score}
graph.louvain_communities()                                          # → {communities, modularity, num_communities}
graph.connected_components()                                         # → list[list[node_dict]]
```
