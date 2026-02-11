# KGLite

[![PyPI version](https://img.shields.io/pypi/v/kglite)](https://pypi.org/project/kglite/)
[![Python versions](https://img.shields.io/pypi/pyversions/kglite)](https://pypi.org/project/kglite/)
[![License: MIT](https://img.shields.io/pypi/l/kglite)](https://github.com/kkollsga/kglite/blob/main/LICENSE)

An embedded knowledge graph engine for Python. Import and go — no server, no setup.

> **For AI agents:** see [`kglite.pyi`](kglite.pyi) for full type stubs with signatures and docstrings.

| | |
|---|---|
| Embedded, in-process | No server, no network; `import` and go |
| In-memory | Persistence via `save()`/`load()` snapshots |
| Cypher subset | Querying + mutations; returns `dict` or DataFrame |
| Single-label nodes | Each node has exactly one type |
| Single-threaded | Designed for single-threaded use (see [Threading](#threading)) |

**Requirements:** Python 3.10+ (CPython) | macOS (ARM/Intel), Linux (x86_64/aarch64), Windows (x86_64) | `pandas >= 1.5`

```bash
pip install kglite
```

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

# Query — returns list[dict]
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

## Core Concepts

**Nodes** have three built-in fields — `type` (label), `title` (display name), `id` (unique within type) — plus arbitrary properties. Each node has exactly one type.

**Relationships** connect two nodes with a type (e.g., `:KNOWS`) and optional properties. The Cypher API calls them "relationships"; the fluent API calls them "connections" — same thing.

**Selections** (fluent API) are lightweight views — a set of node indices that flow through chained operations like `type_filter().filter().traverse()`. They don't copy data.

**Atomicity.** Each `cypher()` call is atomic — if any clause fails, the graph remains unchanged. There are no multi-statement transactions. Durability only via explicit `save()`.

---

## Return Types

All node-related methods use a consistent key order: **`type`, `title`, `id`**, then other properties.

### Cypher

| Query type | Returns |
|-----------|---------|
| Read (`MATCH...RETURN`) | `list[dict]` — one dict per row, keyed by column alias |
| Read with `to_df=True` | `pandas.DataFrame` |
| Mutation (`CREATE`, `SET`, `DELETE`, `MERGE`) | `{'stats': {'nodes_created': int, ...}}` |
| Mutation with `RETURN` | `{'rows': [...], 'stats': {...}}` |
| `EXPLAIN` prefix | `str` (query plan, not executed) |

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
| `get_nodes()` | `list[dict]` | Full node dicts |
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
| Default | `list[{type, title, id, score}]` sorted by score desc |
| `as_dict=True` | `{id: score}` — keyed by node ID (unique per type) |
| `to_df=True` | `DataFrame` with columns `type, title, id, score` |

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
graph.cypher("MATCH (n:Person) RETURN n.name")                          # → list[dict]
graph.cypher("MATCH (n:Person) RETURN n.name", to_df=True)              # → DataFrame
graph.cypher("MATCH (n:Person) RETURN n.name", params={'x': 1})         # parameterized
graph.cypher("CREATE (:Person {name: 'Alice'})")                        # → {'stats': {...}}
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
    .get_nodes()                                   # materialize → list[dict]
```

### Introspection

```python
graph.schema()                                # → full graph overview (types, counts, connections, indexes)
graph.connection_types()                      # → list of edge types with counts and endpoint types
graph.properties('Person')                    # → per-property stats (type, non_null, unique, values)
graph.neighbors_schema('Person')              # → outgoing/incoming connection topology
graph.sample('Person', n=5)                   # → first N nodes as dicts
graph.indexes()                               # → all indexes with type info
```

### Algorithms

```python
graph.shortest_path(source_type, source_id, target_type, target_id)  # → {path, connections, length} | None
graph.all_paths(source_type, source_id, target_type, target_id)      # → list[{path, connections, length}]
graph.pagerank(top_k=10)                                             # → list[{type, title, id, score}]
graph.betweenness_centrality(top_k=10)                               # → list[{type, title, id, score}]
graph.louvain_communities()                                          # → {communities, modularity, num_communities}
graph.connected_components()                                         # → list[list[node_dict]]
```

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

### `properties(node_type)` — Property details

Per-property statistics for a single node type. Includes `values` list when unique count is 20 or fewer (low cardinality).

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

Returns the first N nodes of a type as full dicts:

```python
graph.sample('Person', n=3)
# [
#   {'type': 'Person', 'title': 'Alice', 'id': 1, 'age': 28, 'city': 'Oslo'},
#   {'type': 'Person', 'title': 'Bob', 'id': 2, 'age': 35, 'city': 'Bergen'},
#   {'type': 'Person', 'title': 'Charlie', 'id': 3, 'age': 42, 'city': 'Oslo'},
# ]
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

---

## Cypher Queries

A substantial Cypher subset. See the [Supported Cypher Subset](#supported-cypher-subset) table for exact coverage.

> **Single-label note:** Each node has exactly one type. `labels(n)` returns a string, not a list. `SET n:OtherLabel` is not supported.

```python
result = graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WHERE p.age > 30 AND f.city = 'Oslo'
    RETURN p.name AS person, f.name AS friend, p.age AS age
    ORDER BY p.age DESC
    LIMIT 10
""")

# Read queries → list[dict]
for row in result:
    print(f"{row['person']} knows {row['friend']}")

# Pass to_df=True for a DataFrame
df = graph.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age", to_df=True)
```

### WHERE Clause

```python
# Comparisons: =, <>, <, >, <=, >=
graph.cypher("MATCH (n:Product) WHERE n.price >= 500 RETURN n.title, n.price")

# Boolean operators: AND, OR, NOT
graph.cypher("MATCH (n:Person) WHERE n.age > 25 AND NOT n.city = 'Oslo' RETURN n.name")

# Null checks
graph.cypher("MATCH (n:Person) WHERE n.email IS NOT NULL RETURN n.name")

# String predicates: CONTAINS, STARTS WITH, ENDS WITH
graph.cypher("MATCH (n:Person) WHERE n.name CONTAINS 'ali' RETURN n.name")

# IN lists
graph.cypher("MATCH (n:Person) WHERE n.city IN ['Oslo', 'Bergen'] RETURN n.name")

# Regex matching with =~
graph.cypher("MATCH (n:Person) WHERE n.name =~ '(?i)^ali.*' RETURN n.name")
graph.cypher("MATCH (n:Person) WHERE n.email =~ '.*@example\\.com$' RETURN n.name")
```

### Relationship Properties

Relationships can have properties. Access them with `r.property` syntax:

```python
# Create relationships with properties
graph.cypher("""
    MATCH (p:Person {name: 'Alice'}), (m:Movie {title: 'Inception'})
    CREATE (p)-[:RATED {score: 5, comment: 'Excellent'}]->(m)
""")

# Access, filter, aggregate, sort by relationship properties
graph.cypher("MATCH (p)-[r:RATED]->(m) RETURN p.name, r.score, r.comment, type(r)")
graph.cypher("MATCH (p)-[r:RATED]->(m) WHERE r.score >= 4 RETURN p.name, m.title")
graph.cypher("MATCH (p)-[r:RATED]->(m) RETURN avg(r.score) AS avg_rating")
graph.cypher("MATCH ()-[r:RATED]->(m) RETURN m.title, r.score ORDER BY r.score DESC")
```

### Aggregation

```python
graph.cypher("MATCH (n:Person) RETURN n.city, count(*) AS population ORDER BY population DESC")
graph.cypher("MATCH (n:Person) RETURN avg(n.age) AS avg_age, min(n.age), max(n.age)")

# DISTINCT
graph.cypher("MATCH (n:Person) RETURN DISTINCT n.city")
graph.cypher("MATCH (n:Person) RETURN count(DISTINCT n.city) AS unique_cities")
```

### WITH Clause

```python
graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WITH p, count(f) AS friend_count
    WHERE friend_count > 3
    RETURN p.name, friend_count
    ORDER BY friend_count DESC
""")
```

### OPTIONAL MATCH

Left outer join — keeps rows even when no match:

```python
graph.cypher("""
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
    RETURN p.name, count(f) AS friends
""")
```

### Built-in Functions

| Function | Description |
|----------|-------------|
| `toUpper(expr)` | Convert to uppercase |
| `toLower(expr)` | Convert to lowercase |
| `toString(expr)` | Convert to string |
| `toInteger(expr)` | Convert to integer |
| `toFloat(expr)` | Convert to float |
| `size(expr)` | Length of string or list |
| `type(r)` | Relationship type |
| `id(n)` | Node ID |
| `labels(n)` | Node type (string, not list — single-label) |
| `coalesce(a, b, ...)` | First non-null argument |
| `length(p)` | Path hop count |
| `nodes(p)` | Nodes in a path |
| `relationships(p)` | Relationships in a path |

### Arithmetic

```python
graph.cypher("MATCH (n:Product) RETURN n.title, n.price * 1.25 AS price_with_tax")
```

### CASE Expressions

```python
# Generic form
graph.cypher("""
    MATCH (n:Person)
    RETURN n.name,
           CASE WHEN n.age >= 18 THEN 'adult' ELSE 'minor' END AS category
""")

# Simple form
graph.cypher("""
    MATCH (n:Person)
    RETURN n.name,
           CASE n.city WHEN 'Oslo' THEN 'capital' WHEN 'Bergen' THEN 'west coast' ELSE 'other' END AS region
""")
```

### List Comprehensions

`[x IN list WHERE predicate | expression]` syntax:

```python
# Map: double each number
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] | x * 2] AS doubled")
# [2, 4, 6, 8, 10]

# Filter only
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 3] AS filtered")
# [4, 5]

# Filter + map
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 3 | x * 2] AS result")
# [8, 10]

# With collect() — transform aggregated values
graph.cypher("""
    MATCH (p:Person)
    WITH collect(p.name) AS names
    RETURN [x IN names | toUpper(x)] AS upper_names
""")
```

> **Note:** List comprehensions require at least one row in the pipeline. Use `UNWIND [1] AS _` or a preceding `MATCH`/`WITH` to provide the row context.

### Map Projections

`n {.prop1, .prop2, alias: expr}` syntax — select specific properties from a node:

```python
# Select only name and age (returns a dict per row)
graph.cypher("MATCH (p:Person) RETURN p {.name, .age} AS info")
# [{'info': {'name': 'Alice', 'age': 30}}, {'info': {'name': 'Bob', 'age': 25}}]

# Mix shorthand properties with computed values
graph.cypher("""
    MATCH (p:Person)-[:WORKS_AT]->(c:Company)
    RETURN p {.name, .age, company: c.name} AS info
""")

# System properties (id, type) work too
graph.cypher("MATCH (p:Person) RETURN p {.name, .type, .id} AS info LIMIT 1")
# [{'info': {'name': 'Alice', 'type': 'Person', 'id': 1}}]
```

### Parameters

```python
graph.cypher(
    "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age",
    params={'min_age': 25}
)

# Parameters in inline pattern properties
graph.cypher(
    "MATCH (n:Person {name: $name}) RETURN n.age",
    params={'name': 'Alice'}
)

# Parameters with DataFrame output
df = graph.cypher(
    "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age ORDER BY n.age",
    params={'min_age': 20}, to_df=True
)
```

### UNWIND

Expand a list into rows:

```python
graph.cypher("UNWIND [1, 2, 3] AS x RETURN x, x * 2 AS doubled")
```

### UNION

```python
graph.cypher("""
    MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
    UNION
    MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name
""")
```

### Variable-Length Paths

```python
# 1 to 3 hops
graph.cypher("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) WHERE a.name = 'Alice' RETURN b.name")

# Exact 2 hops
graph.cypher("MATCH (a:Person)-[:KNOWS*2]->(b:Person) RETURN a.name, b.name")
```

### WHERE EXISTS

Check for subpattern existence. Both brace `{ }` and parenthesis `(( ))` syntax are supported:

```python
# Brace syntax
graph.cypher("MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->(:Person) } RETURN p.name")

# Parenthesis syntax (equivalent)
graph.cypher("MATCH (p:Person) WHERE EXISTS((p)-[:KNOWS]->(:Person)) RETURN p.name")

# Negation
graph.cypher("""
    MATCH (p:Person)
    WHERE NOT EXISTS { (p)-[:PURCHASED]->(:Product) }
    RETURN p.name
""")
```

### shortestPath()

BFS shortest path between two nodes. Supports directed (`->`) and undirected (`-`) syntax:

```python
# Directed — only follows edges in their defined direction
result = graph.cypher("""
    MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Dave'}))
    RETURN length(p), nodes(p), relationships(p), a.name, b.name
""")

# Undirected — traverses edges in both directions (same as fluent API)
result = graph.cypher("""
    MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]-(b:Person {name: 'Dave'}))
    RETURN length(p), nodes(p), relationships(p)
""")

# No path → empty list (not an error)
```

**Path functions:** `length(p)` returns hop count, `nodes(p)` returns node list, `relationships(p)` returns edge type list.

### CREATE / SET / DELETE / REMOVE / MERGE

```python
# CREATE — returns stats
result = graph.cypher("CREATE (n:Person {name: 'Alice', age: 30, city: 'Oslo'})")
print(result['stats']['nodes_created'])  # 1

# CREATE relationship between existing nodes
graph.cypher("""
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    CREATE (a)-[:KNOWS]->(b)
""")

# SET — update properties
result = graph.cypher("MATCH (n:Person {name: 'Bob'}) SET n.age = 26, n.city = 'Stavanger'")
print(result['stats']['properties_set'])  # 2

# DELETE — plain DELETE errors if node has relationships; DETACH removes all
graph.cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")

# REMOVE — remove properties (id/type are immutable)
graph.cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n.city")

# MERGE — match or create
graph.cypher("""
    MERGE (n:Person {name: 'Alice'})
    ON CREATE SET n.created = 'today'
    ON MATCH SET n.updated = 'today'
""")
```

### DataFrame Output

```python
df = graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WITH p, count(f) AS friends
    RETURN p.name, p.city, friends
    ORDER BY friends DESC
""", to_df=True)
```

### EXPLAIN

Prefix any Cypher query with `EXPLAIN` to see the query plan without executing it:

```python
plan = graph.cypher("""
    EXPLAIN
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
    WITH p, count(f) AS friends
    RETURN p.name, friends
""")
print(plan)
# Query Plan:
#   1. NodeScan (MATCH) :Person
#   2. FusedOptionalMatchAggregate (optimized OPTIONAL MATCH + count)
#   3. Projection (RETURN) [p.name, friends]
# Optimizations: optional_match_fusion=1
```

### Supported Cypher Subset

| Category | Supported |
|----------|-----------|
| **Clauses** | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `ORDER BY`, `SKIP`, `LIMIT`, `UNWIND`, `UNION`/`UNION ALL`, `CREATE`, `SET`, `DELETE`, `DETACH DELETE`, `REMOVE`, `MERGE`, `EXPLAIN` |
| **Patterns** | Node `(n:Type)`, relationship `-[:REL]->`, variable-length `*1..3`, undirected `-[:REL]-`, properties `{key: val}`, `p = shortestPath(...)` |
| **WHERE** | `=`, `<>`, `<`, `>`, `<=`, `>=`, `=~` (regex), `AND`, `OR`, `NOT`, `IS NULL`, `IS NOT NULL`, `IN [...]`, `CONTAINS`, `STARTS WITH`, `ENDS WITH`, `EXISTS { pattern }`, `EXISTS(( pattern ))` |
| **RETURN** | `n.prop`, `r.prop`, `AS` aliases, `DISTINCT`, arithmetic `+`/`-`/`*`/`/`, map projections `n {.prop1, .prop2}` |
| **Aggregation** | `count(*)`, `count(expr)`, `sum`, `avg`/`mean`, `min`, `max`, `collect`, `std` |
| **Expressions** | `CASE WHEN...THEN...ELSE...END`, `$param`, `[x IN list WHERE ... \| expr]` |
| **Functions** | `toUpper`, `toLower`, `toString`, `toInteger`, `toFloat`, `size`, `length`, `type`, `id`, `labels`, `coalesce`, `nodes(p)`, `relationships(p)` |
| **Mutations** | `CREATE (n:Label {props})`, `CREATE (a)-[:TYPE]->(b)`, `SET n.prop = expr`, `DELETE`, `DETACH DELETE`, `REMOVE n.prop`, `MERGE ... ON CREATE SET ... ON MATCH SET` |
| **Not supported** | `CALL`/stored procedures, `FOREACH`, subqueries, `SET n:Label` (label mutation), `REMOVE n:Label`, multi-label |

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

When adding nodes, `unique_id_field` and `node_title_field` are **renamed** to `id` and `title`. The original column names no longer exist as properties.

| Your DataFrame Column | Stored As | Why |
|-----------------------|-----------|-----|
| `unique_id_field` (e.g., `user_id`) | `id` | Canonical identifier |
| `node_title_field` (e.g., `name`) | `title` | Display/label field |
| All other columns | Same name | Preserved as-is |

```python
# After adding with unique_id_field='user_id', node_title_field='name':
graph.type_filter('User').filter({'user_id': 1001})  # WRONG — field was renamed
graph.type_filter('User').filter({'id': 1001})        # CORRECT
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

All centrality methods return `list[{type, title, id, score}]`, sorted by score descending.

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

Indexes accelerate **equality lookups only** (`WHERE n.prop = value`). Range conditions (`<`, `>`, `<=`, `>=`) always scan.

```python
graph.create_index('Prospect', 'prospect_geoprovince')
graph.create_composite_index('Person', ['city', 'age'])

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

> **Portability:** Save files use bincode serialization and are **not guaranteed portable** across OS, CPU architecture, or library versions. Always re-export via a portable format (GraphML, CSV) when sharing across machines.

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

Designed for single-threaded use. The Rust code does not release the Python GIL during operations. If you share a graph instance across threads, guard access with your own lock.

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
