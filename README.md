# KGLite

[![PyPI version](https://img.shields.io/pypi/v/kglite)](https://pypi.org/project/kglite/)
[![Python versions](https://img.shields.io/pypi/pyversions/kglite)](https://pypi.org/project/kglite/)
[![License: MIT](https://img.shields.io/pypi/l/kglite)](https://github.com/kkollsga/kglite/blob/main/LICENSE)

An embedded knowledge graph engine for Python.

**Use it for:** local analytics, ETL pipelines, notebooks, embedding in apps, fast prototyping.
**Not for:** multi-user server deployments, cross-call transactions, HA/replication.

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

## Feature Matrix

| Feature | Status |
|---------|--------|
| Embedded / in-process | Yes |
| Cypher queries (MATCH, CREATE, SET, DELETE, MERGE, ...) | Yes |
| DataFrame output (`to_df=True`) | Yes |
| Graph algorithms (shortest path, centrality, communities) | Yes |
| Persistence (binary snapshots) | Yes |
| Multi-label nodes | No |
| Transactions across calls | No |
| Concurrency / thread safety | No |
| Server mode | No |

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

# Mutations return stats
result = graph.cypher("CREATE (:Person {name: 'Dave', age: 22})")
print(result['stats'])  # {'nodes_created': 1, 'relationships_created': 0, ...}

# Check graph size
print(graph.graph_info())  # {'node_count': 4, 'edge_count': 1, ...}

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

**Nodes** have four built-in fields: `id` (unique within type), `title` (display name), `type` (label), plus arbitrary properties. Each node has exactly one type — `labels(n)` returns a string, not a list.

**Relationships** connect two nodes with a type (e.g., `:KNOWS`) and optional properties. The Cypher API calls them "relationships"; the fluent API calls them "connections" — they're the same thing.

**Return shape.** Read queries return `list[dict]` — each dict is one row keyed by column alias. Mutation queries (CREATE, SET, DELETE, REMOVE, MERGE) return `{'stats': {...}}` with counts like `nodes_created`, `properties_set`, etc. Mutations with a RETURN clause return `{'rows': [...], 'stats': {...}}`. Pass `to_df=True` to get a pandas DataFrame instead (read queries only).

**Atomicity.** Each `cypher()` call is atomic at the statement level — if any clause fails, the graph remains unchanged (copy-on-write internally). There are no multi-statement transactions: two separate `cypher()` calls are independent. Durability only via explicit `save()`.

**Selections** (fluent API only) are lightweight views — a set of node indices that flow through chained operations like `type_filter().filter().traverse()`. They don't copy data. Use `explain()` to see the pipeline.

**Tombstones.** `DELETE` leaves empty slots in the internal graph storage. After heavy deletion, check `graph.graph_info()['fragmentation_ratio']` and call `graph.vacuum()` if it exceeds 0.3 (see [Graph Maintenance](#graph-maintenance)).

---

## Table of Contents

- [Cypher Queries](#cypher-queries) — MATCH, CREATE, SET, DELETE, MERGE, aggregation, shortestPath
- [Common Recipes](#common-recipes) — Quick copy/paste snippets
- [Advanced API: Data Management](#advanced-api-data-management) | [Querying](#advanced-api-querying) | [Pattern Matching](#pattern-matching)
- [Graph Algorithms](#graph-algorithms) | [Spatial Operations](#spatial-operations) | [Analytics](#analytics)
- [Schema and Indexes](#schema-and-indexes) | [Import and Export](#import-and-export) | [Performance](#performance) | [Graph Maintenance](#graph-maintenance)

---

## When to Use What

| Interface | Best For | Key Benefits |
|-----------|----------|--------------|
| **Cypher** (recommended) | Ad-hoc queries, exploration, analytics, mutations | Standard syntax, declarative, familiar to Neo4j users |
| **Fluent API** (advanced) | Bulk loading from DataFrames, multi-step pipelines | Chainable operations, `explain()`, computed properties |
| **Pattern matching** (specialized) | Quick structural checks without full Cypher overhead | Lightweight, minimal parsing |

**Start with Cypher** for most tasks. Use the fluent API when bulk-loading from pandas DataFrames or building pipelines that store intermediate computed properties. Use pattern matching for simple structural queries where you don't need WHERE/RETURN clauses.

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

### Create Index for Speed

```python
graph.create_index('Product', 'category')

# ~3x faster on 100k+ node graphs (equality only; depends on selectivity)
result = graph.cypher("MATCH (p:Product) WHERE p.category = 'Electronics' RETURN p.name")
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
# [{'info': {'name': 'Alice', 'age': 30, 'company': 'Acme'}}, ...]

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

Check for subpattern existence. The outer variable (`p`) is bound from MATCH.
Both brace `{ }` and parenthesis `(( ))` syntax are supported:

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

# With property filter in inner pattern
graph.cypher("""
    MATCH (p:Person)
    WHERE EXISTS { (p)-[:KNOWS]->(:Person {city: 'Oslo'}) }
    RETURN p.name
""")
```

### shortestPath()

Directed BFS shortest path between two nodes:

```python
result = graph.cypher("""
    MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Dave'}))
    RETURN length(p), nodes(p), relationships(p), a.name, b.name
""")
# [{'length(p)': 3, 'nodes(p)': [...], ...}]

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

**Error example:** `DELETE` on a node with relationships returns:
`"Cannot delete node with existing relationships. Use DETACH DELETE to remove the node and all its relationships."`

### Mutation Semantics

**Atomicity:** Each `cypher()` call is atomic at the statement level — if any clause fails, the graph remains unchanged (copy-on-write internally). There are no multi-statement transactions; two separate `cypher()` calls are independent. Durability only via explicit `save()`.

**Index maintenance:** Property and composite indexes are updated automatically by all mutation operations (CREATE, SET, DELETE, REMOVE, MERGE).

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

Returns a string (not data). Mutation queries with `EXPLAIN` are not executed.

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

## Advanced API: Data Management

<details>
<summary>Expand section</summary>

> For most use cases, use [Cypher queries](#cypher-queries). The fluent API below is for bulk operations from DataFrames or complex data pipelines.

### Adding Nodes

```python
products_df = pd.DataFrame({
    'product_id': [101, 102, 103],
    'title': ['Laptop', 'Phone', 'Tablet'],
    'price': [999.99, 699.99, 349.99],
    'stock': [45, 120, 30]
})

graph.add_nodes(
    data=products_df,
    node_type='Product',
    unique_id_field='product_id',
    node_title_field='title',
    columns=['product_id', 'title', 'price', 'stock'],
    column_types={'launch_date': 'datetime'},  # explicit type hints (see Working with Dates)
    conflict_handling='update'  # 'update' | 'replace' | 'skip' | 'preserve'
)
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

Use `explain()` to verify node counts at each step:

```python
result = graph.type_filter('User').filter({'id': 1001})
print(result.explain())
# TYPE_FILTER User (1000 nodes) -> FILTER (1 nodes)
```

### Retrieving Nodes

```python
products = graph.type_filter('Product')
products.get_nodes()                       # all properties
products.get_properties(['price', 'stock'])  # specific properties
products.get_titles()                       # just titles
```

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

> **Note:** `source_type` and `target_type` each refer to a single node type. To connect nodes of the same type, set both to the same value (e.g., `source_type='Person', target_type='Person'`).

### Batch Property Updates

```python
result = graph.type_filter('Prospect').filter({'status': 'Inactive'}).update({
    'is_active': False,
    'deactivation_reason': 'status_inactive'
})

updated_graph = result['graph']
print(f"Updated {result['nodes_updated']} nodes")
```

</details>

---

## Advanced API: Querying

<details>
<summary>Expand section</summary>

> For most queries, prefer [Cypher](#cypher-queries). The fluent API below is for building reusable query chains or when you need `explain()` and selection-based workflows.

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

</details>

---

## Pattern Matching

<details>
<summary>Expand section</summary>

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

</details>

---

## Graph Algorithms

<details>
<summary>Expand section</summary>

### Shortest Path

```python
path = graph.shortest_path(source_type='Person', source_id=1, target_type='Person', target_id=100)
for node in path:
    print(f"{node['type']}: {node['title']}")
```

### All Paths

```python
paths = graph.all_paths(
    source_type='Play', source_id=1,
    target_type='Wellbore', target_id=100,
    max_hops=4
)
```

### Connected Components

```python
components = graph.connected_components()
# Returns list of lists: [[node_indices...], [node_indices...], ...]
print(f"Found {len(components)} connected components")
print(f"Largest component: {len(components[0])} nodes")

graph.are_connected(source_type='Person', source_id=1, target_type='Person', target_id=100)
```

### Centrality Algorithms

All centrality methods return a list of dicts with `type`, `title`, `id`, and `score` keys, sorted by score descending.

```python
graph.betweenness_centrality(top_k=10)
graph.betweenness_centrality(normalized=True, sample_size=500)
graph.pagerank(top_k=10, damping_factor=0.85)
graph.degree_centrality(top_k=10)
graph.closeness_centrality(top_k=10)
```

### Community Detection

Identify clusters of densely connected nodes.

```python
# Louvain modularity optimization (recommended)
result = graph.louvain_communities()
# {'communities': {0: [{title, type, id}, ...], 1: [...]},
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

</details>

---

## Spatial Operations

<details>
<summary>Expand section</summary>

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

</details>

---

## Analytics

<details>
<summary>Expand section</summary>

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

</details>

---

## Schema and Indexes

<details>
<summary>Expand section</summary>

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

</details>

---

## Import and Export

<details>
<summary>Expand section</summary>

### Saving and Loading

```python
graph.save("my_graph.kgl")
loaded_graph = kglite.load("my_graph.kgl")
```

> **Portability:** Save files use bincode serialization and are **not guaranteed portable** across OS, CPU architecture, or library versions. Always re-export via a portable format (GraphML, CSV) when sharing across machines. Each file includes a format version and the library version that wrote it — check with `graph_info()['format_version']` and `graph_info()['library_version']` after loading. If the internal data structures change between releases, loading will fail with a clear version mismatch error rather than silent corruption.

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

</details>

---

## Performance

<details>
<summary>Expand section</summary>

### Tips

1. **Batch operations** — add nodes/connections in batches, not individually
2. **Specify columns** — only include columns you need to reduce memory
3. **Filter by type first** — `type_filter()` before `filter()` for narrower scans
4. **Create indexes** — on frequently filtered equality conditions (~3x on 100k+ nodes; depends on selectivity)
5. **Use lightweight methods** — `node_count()`, `indices()`, `get_node_by_id()` skip property materialization
6. **Cypher LIMIT** — use `LIMIT` to avoid scanning entire result sets

### Lightweight Methods

| Method | Returns | Speed |
|--------|---------|-------|
| `node_count()` | Integer count (total graph if no filter, selection count after filter) | Fastest |
| `indices()` | List of node indices | Fast |
| `id_values()` | List of ID values | Fast |
| `get_ids()` | List of `{id, title, type}` dicts | Medium |
| `get_nodes()` | List of full node dicts | Slowest |

Lightweight path methods: `shortest_path_length()`, `shortest_path_indices()`, `shortest_path_ids()`.

### Performance Model

kglite is optimized for **knowledge graph workloads** — complex multi-step queries on heterogeneous, property-rich graphs. Operations have overhead compared to raw graph algorithms because they build selections, materialize Python dicts, and support the full query API.

**Speed claims caveat:** The "~3x" index speedup was measured on equality-filtered queries over 100k+ node graphs. Actual improvement depends on graph size, selectivity, and property cardinality. On small graphs (<1k nodes) the overhead of index lookup may not be noticeable. Always benchmark on your own data.

### Threading

Designed for single-threaded use. The Rust code does not release the Python GIL during operations. If you share a graph instance across threads, guard access with your own lock.

</details>

---

## Graph Maintenance

<details>
<summary>Expand section</summary>

After heavy mutation workloads (DELETE, REMOVE), the internal graph storage accumulates tombstones. Use `graph_info()` to monitor storage health.

### Diagnostics

```python
info = graph.graph_info()
# {'node_count': 950, 'node_capacity': 1000, 'node_tombstones': 50,
#  'edge_count': 2800, 'fragmentation_ratio': 0.05,
#  'type_count': 3, 'property_index_count': 2, 'composite_index_count': 0}
```

### Vacuum — Compact Storage

```python
if info['fragmentation_ratio'] > 0.3:
    result = graph.vacuum()
    print(f"Reclaimed {result['tombstones_removed']} slots, remapped {result['nodes_remapped']} nodes")
```

`vacuum()` rebuilds the graph with contiguous indices and rebuilds all indexes. **Resets the current selection** — call between query chains.

### Reindex — Rebuild Indexes

```python
graph.reindex()
```

Recovery tool, not routine maintenance. Indexes are maintained automatically by all mutations. Use `reindex()` only if you suspect corruption (e.g., after a crash during `save()`).

### Recommended Workflow

```python
info = graph.graph_info()
if info['fragmentation_ratio'] > 0.3:
    graph.vacuum()
```

</details>

---

## Operation Reports

<details>
<summary>Expand section</summary>

Operations that modify the graph return detailed reports:

```python
report = graph.add_nodes(data=df, node_type='Product', unique_id_field='product_id')
print(f"Created {report['nodes_created']} nodes in {report['processing_time_ms']}ms")

if report['has_errors']:
    print(f"Errors: {report['errors']}")
```

**Node report fields:** `operation`, `timestamp`, `nodes_created`, `nodes_updated`, `nodes_skipped`, `processing_time_ms`, `has_errors`, `errors`.

**Connection report fields:** `connections_created`, `connections_skipped`, `property_fields_tracked`.

```python
graph.get_last_report()
graph.get_operation_index()
graph.get_report_history()
```

</details>
