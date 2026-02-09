# Rusty Graph

[![PyPI version](https://img.shields.io/pypi/v/rusty-graph)](https://pypi.org/project/rusty-graph/)
[![Python versions](https://img.shields.io/pypi/pyversions/rusty-graph)](https://pypi.org/project/rusty-graph/)
[![License: MIT](https://img.shields.io/pypi/l/rusty-graph)](https://github.com/kkollsga/rusty-graph/blob/main/LICENSE)

A high-performance graph database library for Python, written in Rust.

```bash
pip install rusty-graph
pip install rusty-graph --upgrade  # upgrade
```

## ⚡ Start Here

**Most users should:**

- **Use Cypher queries** for creating, reading, updating, and deleting data — it's declarative, familiar, and returns DataFrames
- **Use `add_nodes()` / `add_connections()`** when bulk-loading from pandas DataFrames (thousands of rows at once)
- **Use `create_index()`** on frequently-filtered properties for ~3x speedup on large graphs

**Compatibility note:** rusty_graph implements a Cypher subset (see [Supported Cypher Subset](#supported-cypher-subset)). Not supported: `CALL`, subqueries, `SET n:Label` (label mutation). If you need Neo4j-specific features, check the compatibility table first.

## Quick Start

```python
import rusty_graph

graph = rusty_graph.KnowledgeGraph()

# Create nodes with Cypher
graph.cypher("CREATE (:Person {name: 'Alice', age: 28, city: 'Oslo'})")
graph.cypher("CREATE (:Person {name: 'Bob', age: 35, city: 'Bergen'})")
graph.cypher("CREATE (:Person {name: 'Charlie', age: 42, city: 'Oslo'})")

# Create relationships
graph.cypher("""
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    CREATE (a)-[:KNOWS]->(b)
""")

# Query with filters, aggregation, and sorting
result = graph.cypher("""
    MATCH (p:Person)
    WHERE p.age > 30
    RETURN p.name AS name, p.city AS city, p.age AS age
    ORDER BY age DESC
""")

for row in result['rows']:
    print(f"{row['name']}, {row['age']}, {row['city']}")

# Get results as a pandas DataFrame
df = graph.cypher("MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age", to_df=True)
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

# Now query with Cypher
graph.cypher("MATCH (u:User) WHERE u.age > 30 RETURN u.name, u.age")
```

### Key Semantics

- **Atomic mutations** — each `cypher()` call is all-or-nothing; failures leave the graph unchanged
- **Indexes maintained automatically** — CREATE/SET/DELETE/REMOVE keep property indexes in sync
- **Deletes create tombstones** — after heavy deletion, use `graph.vacuum()` when `fragmentation_ratio > 0.3`

## Table of Contents

### Getting Started

- [When to Use What](#when-to-use-what) — Choosing between Cypher, fluent API, and pattern matching
- [Common Recipes](#common-recipes) — Quick copy/paste examples for common tasks

### Query Interfaces

- [Cypher Queries](#cypher-queries) — MATCH, CREATE, SET, DELETE, REMOVE, MERGE, aggregation (recommended)
- [Advanced API: Data Management](#advanced-api-data-management) — Bulk loading from DataFrames
- [Advanced API: Querying](#advanced-api-querying) — Fluent filtering, traversal, set operations
- [Pattern Matching](#pattern-matching) — Lightweight structural pattern queries

### Graph Analysis

- [Graph Algorithms](#graph-algorithms) — Shortest path, all paths, connected components, centrality
- [Spatial Operations](#spatial-operations) — Bounding box, distance, WKT geometry, point-in-polygon
- [Analytics](#analytics) — Statistics, calculations, connection aggregation

### Advanced Topics

- [Schema and Indexes](#schema-and-indexes) — Schema definition, validation, index management
- [Import and Export](#import-and-export) — Save/load, GraphML, GEXF, D3 JSON, CSV, subgraphs
- [Performance](#performance) — Tips, lightweight methods, performance model
- [Graph Maintenance](#graph-maintenance) — Reindex, vacuum, graph health diagnostics
- [Operation Reports](#operation-reports) — Tracking graph mutations

---

## When to Use What

rusty_graph offers three query interfaces:

| Interface | Best For | Key Benefits |
|-----------|----------|--------------|
| **Cypher** (recommended) | Ad-hoc queries, exploration, analytics, mutations | Standard syntax, declarative, returns DataFrames, familiar to Neo4j users |
| **Fluent API** (advanced) | Bulk loading from DataFrames, multi-step pipelines with stored intermediate results | Chainable operations, `explain()`, computed properties stored on nodes |
| **Pattern matching** (specialized) | Quick structural checks without full Cypher overhead | Lightweight, minimal parsing |

**Start with Cypher** for most tasks — it's the most expressive and widely understood interface. Use the fluent API when you need to load data from pandas DataFrames or build complex pipelines that store intermediate computed properties. Use pattern matching for simple structural queries where you don't need WHERE/RETURN clauses.

All three interfaces share the same underlying graph engine and have similar performance characteristics.

---

## Common Recipes

Quick copy/paste snippets for common tasks:

### Upsert with MERGE

```python
# Create node if it doesn't exist, update timestamp if it does
graph.cypher("""
    MERGE (p:Person {email: 'alice@example.com'})
    ON CREATE SET p.created = '2024-01-01', p.name = 'Alice'
    ON MATCH SET p.last_seen = '2024-01-15'
""")
```

### Top-K Nodes by Centrality

```python
# Find the 10 most connected people
top_nodes = graph.pagerank(top_k=10)
for node in top_nodes:
    print(f"{node['title']}: {node['score']:.3f}")
```

### 2-Hop Neighborhood

```python
# Find friends-of-friends
graph.cypher("""
    MATCH (me:Person {name: 'Alice'})-[:KNOWS*2]-(friend_of_friend:Person)
    WHERE friend_of_friend <> me
    RETURN DISTINCT friend_of_friend.name
""")
```

### Export Subgraph

```python
# Export Alice's 2-hop network as GraphML
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
# Create index before filtering
graph.create_index('Product', 'category')

# Now ~3x faster on large graphs
result = graph.cypher("MATCH (p:Product) WHERE p.category = 'Electronics' RETURN p.name")
```

### Delete Subgraph

```python
# Delete all inactive users and their relationships
graph.cypher("""
    MATCH (u:User)
    WHERE u.status = 'inactive'
    DETACH DELETE u
""")
```

### Parameterized Queries

```python
# Safe value substitution — prevents injection, enables reuse
graph.cypher(
    "MATCH (p:Person) WHERE p.city = $city AND p.age > $min_age RETURN p.name",
    params={'city': 'Oslo', 'min_age': 25}
)
```

### Aggregation with Relationship Properties

```python
# Average movie rating per user
graph.cypher("""
    MATCH (p:Person)-[r:RATED]->(m:Movie)
    RETURN p.name, avg(r.score) AS avg_rating, count(m) AS movies_rated
    ORDER BY avg_rating DESC
""")
```

---

## Cypher Queries

A substantial Cypher subset covering most day-to-day querying and mutation. See the [Supported Cypher Subset](#supported-cypher-subset) table for exact coverage.

```python
result = graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WHERE p.age > 30 AND f.city = 'Oslo'
    RETURN p.name AS person, f.name AS friend, p.age AS age
    ORDER BY p.age DESC
    LIMIT 10
""")

# Returns: {'columns': ['person', 'friend', 'age'], 'rows': [{'person': 'Alice', ...}, ...]}
for row in result['rows']:
    print(f"{row['person']} knows {row['friend']}")

# Or get results as a pandas DataFrame
df = graph.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age", to_df=True)
```

### WHERE Clause

Full predicate support with boolean logic:

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
```

### Relationship Properties

Relationships (edges) can have properties just like nodes. Access them with `r.property` syntax:

```python
# Create relationships with properties
graph.cypher("""
    MATCH (p:Person {name: 'Alice'}), (m:Movie {title: 'Inception'})
    CREATE (p)-[:RATED {score: 5, comment: 'Excellent', date: '2023-01-15'}]->(m)
""")

# Access relationship properties in RETURN
graph.cypher("MATCH (p)-[r:RATED]->(m) RETURN p.name, r.score, r.comment, type(r)")

# Filter by relationship properties
graph.cypher("MATCH (p)-[r:RATED]->(m) WHERE r.score >= 4 RETURN p.name, m.title")

# Aggregate relationship properties
graph.cypher("MATCH (p:Person)-[r:RATED]->(m:Movie) RETURN avg(r.score) AS avg_rating")

# Sort by relationship properties
graph.cypher("MATCH ()-[r:RATED]->(m) RETURN m.title, r.score ORDER BY r.score DESC")
```

### Aggregation

Standard aggregate functions with automatic grouping:

```python
# count, sum, avg, min, max, collect
graph.cypher("MATCH (n:Person) RETURN n.city AS city, count(*) AS population ORDER BY population DESC")
graph.cypher("MATCH (n:Person) RETURN avg(n.age) AS avg_age, min(n.age) AS youngest, max(n.age) AS oldest")

# DISTINCT
graph.cypher("MATCH (n:Person) RETURN DISTINCT n.city")
graph.cypher("MATCH (n:Person) RETURN count(DISTINCT n.city) AS unique_cities")
```

### WITH Clause

Intermediate projections for multi-stage queries:

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

Left outer join — keeps rows even when no match is found:

```python
graph.cypher("""
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
    RETURN p.name, count(f) AS friends
""")
```

### Built-in Functions

Scalar functions available in RETURN and WHERE:

| Function | Description |
|----------|-------------|
| `toUpper(expr)` | Convert to uppercase |
| `toLower(expr)` | Convert to lowercase |
| `toString(expr)` | Convert to string |
| `toInteger(expr)` | Convert to integer |
| `toFloat(expr)` | Convert to float |
| `size(expr)` | Length of string or list |
| `type(r)` | Connection type of a relationship |
| `id(n)` | Node ID |
| `labels(n)` | Node type/labels |
| `coalesce(a, b, ...)` | First non-null argument |

### Arithmetic

Arithmetic expressions in RETURN:

```python
graph.cypher("MATCH (n:Product) RETURN n.title, n.price * 1.25 AS price_with_tax")
```

### CASE Expressions

Conditional logic in RETURN and WHERE:

```python
# Generic form: CASE WHEN predicate THEN result
graph.cypher("""
    MATCH (n:Person)
    RETURN n.name,
           CASE WHEN n.age >= 18 THEN 'adult' ELSE 'minor' END AS category
""")

# Simple form: CASE expression WHEN value THEN result
graph.cypher("""
    MATCH (n:Person)
    RETURN n.name,
           CASE n.city WHEN 'Oslo' THEN 'capital' WHEN 'Bergen' THEN 'west coast' ELSE 'other' END AS region
""")
```

### List Comprehensions

Transform and filter lists inline with `[x IN list WHERE predicate | expression]` syntax:

```python
# Basic map: double each number
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] | x * 2] AS doubled")
# Result: [2, 4, 6, 8, 10]

# Filter only: numbers greater than 3
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 3] AS filtered")
# Result: [4, 5]

# Combined filter and map
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 3 | x * 2] AS result")
# Result: [8, 10]

# With collect() — transform aggregated values
graph.cypher("""
    MATCH (p:Person)
    WITH collect(p.name) AS names
    RETURN [x IN names | toUpper(x)] AS upper_names
""")

# Multiple comprehensions
graph.cypher("""
    UNWIND [1] AS _
    RETURN
        [x IN [1, 2, 3] | x * 2] AS doubled,
        [x IN [1, 2, 3] WHERE x > 1] AS filtered
""")
```

### Parameters

Use `$param` syntax for safe value substitution:

```python
# Parameterized queries
graph.cypher(
    "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age",
    params={'min_age': 25}
)

# Multiple parameters
graph.cypher(
    "MATCH (n:Person) WHERE n.city = $city AND n.age > $age RETURN n.name",
    params={'city': 'Oslo', 'age': 30}
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

Combine results from multiple queries:

```python
# UNION removes duplicates; UNION ALL keeps them
graph.cypher("""
    MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
    UNION
    MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name
""")
```

### Variable-Length Paths

Traverse relationships with hop ranges:

```python
# Friends-of-friends within 1 to 3 hops
graph.cypher("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) WHERE a.name = 'Alice' RETURN b.name")

# Exact 2 hops
graph.cypher("MATCH (a:Person)-[:KNOWS*2]->(b:Person) RETURN a.name, b.name")
```

### CREATE — Create Nodes and Edges

```python
# Create a node
result = graph.cypher("CREATE (n:Person {name: 'Alice', age: 30, city: 'Oslo'})")
print(result['stats']['nodes_created'])  # 1

# Create a full path (nodes + edge)
graph.cypher("CREATE (a:Team {name: 'Alpha'})-[:MEMBER]->(b:Team {name: 'Beta'})")

# Create an edge between existing nodes
graph.cypher("""
    MATCH (a:Person) WHERE a.name = 'Alice'
    MATCH (b:Person) WHERE b.name = 'Bob'
    CREATE (a)-[:KNOWS]->(b)
""")

# Use parameters
graph.cypher("CREATE (n:Person {name: $name, age: $age})", params={'name': 'Eve', 'age': 28})

# Return created data
result = graph.cypher("CREATE (n:City {name: 'Bergen'}) RETURN n.name")
```

### SET — Update Properties

```python
# Update a property
graph.cypher("MATCH (n:Person) WHERE n.name = 'Alice' SET n.city = 'Bergen'")

# Update multiple properties
result = graph.cypher("""
    MATCH (n:Person) WHERE n.name = 'Bob'
    SET n.age = 26, n.city = 'Stavanger'
""")
print(result['stats']['properties_set'])  # 2

# Arithmetic expressions
graph.cypher("MATCH (n:Person) WHERE n.name = 'Alice' SET n.age = 30 + 1")
```

### DELETE / DETACH DELETE — Remove Nodes and Edges

```python
# Delete a node with no connections
graph.cypher("MATCH (n:Person) WHERE n.name = 'Alice' DELETE n")

# DETACH DELETE removes the node and all its connections
graph.cypher("MATCH (n:Person) WHERE n.name = 'Alice' DETACH DELETE n")
result = graph.cypher("MATCH (n:Person) DETACH DELETE n")
print(result['stats']['nodes_deleted'])       # 5
print(result['stats']['relationships_deleted'])  # 3

# Delete a relationship only
graph.cypher("MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE a.name = 'Alice' DELETE r")
```

### REMOVE — Remove Properties

```python
# Remove a property from nodes
graph.cypher("MATCH (n:Person) WHERE n.name = 'Alice' REMOVE n.city")

# Remove multiple properties
result = graph.cypher("MATCH (n:Person) REMOVE n.city, n.age")
print(result['stats']['properties_removed'])  # 2
```

### MERGE — Match or Create

```python
# Create a node only if it doesn't exist
graph.cypher("MERGE (n:Person {name: 'Alice'})")

# ON CREATE SET — set properties only when creating
graph.cypher("MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.age = 30, n.city = 'Oslo'")

# ON MATCH SET — set properties only when matching existing
graph.cypher("MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.visits = 1")

# Both ON CREATE and ON MATCH
graph.cypher("""
    MERGE (n:Person {name: 'Alice'})
    ON CREATE SET n.created = 'today'
    ON MATCH SET n.updated = 'today'
""")

# Merge a relationship between existing nodes
graph.cypher("""
    MATCH (a:Person) WHERE a.name = 'Alice'
    MATCH (b:Person) WHERE b.name = 'Bob'
    MERGE (a)-[:KNOWS]->(b)
""")
```

### Mutation Semantics

**Atomicity**: Each `cypher()` call is atomic at the statement level.
If any clause fails, the graph remains unchanged. Internally, mutations
operate on a copy-on-write snapshot that is only committed on success.

**Index Maintenance**: Property and composite indexes are maintained
automatically by all mutation operations:

| Operation | Index behaviour |
|-----------|----------------|
| CREATE    | New node added to all matching indexes |
| SET       | Old value removed, new value inserted |
| REMOVE    | Old value removed from affected indexes |
| DELETE    | Node removed from all indexes |
| MERGE     | Delegates to CREATE/SET |

### DataFrame Output

Pass `to_df=True` to get results as a pandas DataFrame instead of a dict:

```python
df = graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WITH p, count(f) AS friends
    RETURN p.name, p.city, friends
    ORDER BY friends DESC
""", to_df=True)

print(df)
#     p.name    p.city  friends
# 0    Alice      Oslo        5
# 1      Bob    Bergen        3
```

### Optimization

The Cypher executor uses several optimizations:

- **Predicate pushdown**: WHERE equality conditions are moved into MATCH pattern properties for filtering during pattern matching
- **Lazy binding**: Rows carry lightweight `NodeIndex` references during execution; properties are resolved on demand
- **Short-circuit evaluation**: AND/OR predicates short-circuit naturally

### Supported Cypher Subset

| Category | Supported |
|----------|-----------|
| **Clauses** | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `ORDER BY`, `SKIP`, `LIMIT`, `UNWIND`, `UNION`, `UNION ALL`, `CREATE`, `SET`, `DELETE`, `DETACH DELETE`, `REMOVE`, `MERGE` |
| **Patterns** | Node `(n:Type)`, relationship `-[:REL]->`, variable-length `*1..3`, undirected `-[:REL]-`, inline properties `{key: val}` |
| **WHERE** | `=`, `<>`, `<`, `>`, `<=`, `>=`, `AND`, `OR`, `NOT`, `IS NULL`, `IS NOT NULL`, `IN [...]`, `CONTAINS`, `STARTS WITH`, `ENDS WITH` |
| **RETURN** | Property access `n.prop`, relationship properties `r.prop`, aliases `AS`, `DISTINCT`, arithmetic `+`, `-`, `*`, `/` |
| **Aggregation** | `count(*)`, `count(expr)`, `sum`, `avg`/`mean`, `min`, `max`, `collect`, `std` |
| **Expressions** | `CASE WHEN ... THEN ... ELSE ... END`, parameters `$param`, list comprehensions `[x IN list WHERE ... \| expr]` |
| **Scalar functions** | `toUpper`/`upper`, `toLower`/`lower`, `toString`, `toInteger`/`toInt`, `toFloat`, `size`/`length`, `type`, `id`, `labels`, `coalesce` |
| **Mutations** | `CREATE (n:Label {props})`, `CREATE (a)-[:TYPE]->(b)`, `SET n.prop = expr`, `DELETE n`, `DETACH DELETE n`, `REMOVE n.prop`, `MERGE (n:Label {props})`, `ON CREATE SET`, `ON MATCH SET` |
| **Not supported** | `CALL`, subqueries, `SET n:Label` (label mutation), `REMOVE n:Label` |

---

## Advanced API: Data Management

> **Note:** For most use cases, use [Cypher queries](#cypher-queries) for data manipulation. The fluent API below is useful for bulk operations from DataFrames or when building complex data pipelines.

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
    columns=['product_id', 'title', 'price', 'stock'],          # optional: which columns to include
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
# Adding with unique_id_field='user_id', node_title_field='name'
# Stored as: {'id': 1001, 'title': 'Alice', 'type': 'User', 'age': 28}

graph.type_filter('User').filter({'user_id': 1001})  # WRONG - returns 0 nodes
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

Specify date columns with `column_types` for date-based filtering:

```python
graph.add_nodes(
    data=estimates_df,
    node_type='Estimate',
    unique_id_field='estimate_id',
    node_title_field='name',
    column_types={'valid_from': 'datetime', 'valid_to': 'datetime'}
)

# Date comparisons with ISO format strings
graph.type_filter('Estimate').filter({'valid_from': {'>=': '2020-06-01'}})

# Temporal queries: find entities valid at a point in time
graph.type_filter('Estimate').valid_at('2020-06-15')
graph.type_filter('Contract').valid_at('2021-03-01', date_from_field='start_date', date_to_field='end_date')

# Find entities with overlapping validity periods
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

### Batch Property Updates

```python
result = graph.type_filter('Prospect').filter({'status': 'Inactive'}).update({
    'is_active': False,
    'deactivation_reason': 'status_inactive'
})

updated_graph = result['graph']
print(f"Updated {result['nodes_updated']} nodes")

# Preserve selection for chaining
result = selection.update({'processed': True}, keep_selection=True)
```

---

## Advanced API: Querying

> **Note:** For most queries, prefer [Cypher](#cypher-queries) for clarity and SQL-like syntax. The fluent API below is useful for building reusable query chains or when you need the `explain()` and selection-based workflows.

### Filtering

```python
# Exact match
graph.type_filter('Product').filter({'price': 999.99})

# Comparison operators
graph.type_filter('Product').filter({'price': {'<': 500.0}, 'stock': {'>': 50}})

# IN operator
graph.type_filter('Product').filter({'id': {'in': [101, 103]}})

# Null checks
graph.type_filter('Product').filter({'category': {'is_null': True}})
graph.type_filter('Product').filter({'category': {'is_not_null': True}})

# Orphan nodes (no connections)
graph.filter_orphans(include_orphans=True)
graph.filter_orphans(include_orphans=False)
```

### Sorting

```python
# Single field
graph.type_filter('Product').sort('price')
graph.type_filter('Product').sort('price', ascending=False)

# Multi-field: list of (field, ascending) tuples
graph.type_filter('Product').sort([('stock', False), ('price', True)])

# Sorting works in filter, type_filter, traverse, filter_orphans
graph.type_filter('Product').filter({'stock': {'>': 10}}, sort='price')
```

### Limiting Results

```python
graph.type_filter('Product').max_nodes(5)
```

### Traversing the Graph

```python
alice = graph.type_filter('User').filter({'title': 'Alice'})
alice_products = alice.traverse(connection_type='PURCHASED', direction='outgoing')

# Filter and sort traversal targets
expensive_purchases = alice.traverse(
    connection_type='PURCHASED',
    filter_target={'price': {'>=': 500.0}},
    sort_target='price',
    max_nodes=10
)

# Filter on connection properties
graph.type_filter('Discovery').traverse(
    connection_type='EXTENDS_INTO',
    filter_connection={'share_pct': {'>=': 50.0}}
)

# Get connection information
alice.get_connections(include_node_properties=True)
```

### Set Operations

Combine, intersect, or subtract selections:

```python
n3 = graph.type_filter('Prospect').filter({'geoprovince': 'N3'})
m3 = graph.type_filter('Prospect').filter({'geoprovince': 'M3'})

n3.union(m3)                    # all nodes from both (OR)
n3.intersection(m3)             # nodes in both (AND)
n3.difference(m3)               # nodes in n3 but not m3
n3.symmetric_difference(m3)     # nodes in exactly one (XOR)

# Chaining
selection_a.union(selection_b).intersection(selection_c)
```

---

## Pattern Matching

For simpler pattern-based queries without full Cypher clause support, use `match_pattern()` directly:

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

**Supported syntax:**

- Node patterns: `(variable:NodeType)` or `(variable:NodeType {property: "value"})`
- Relationship patterns: `-[:CONNECTION_TYPE]->`
- Multiple hops: `(a)-[:REL1]->(b)-[:REL2]->(c)`

---

## Graph Algorithms

### Shortest Path

```python
path = graph.shortest_path(source_type='Person', source_id=1, target_type='Person', target_id=100)
for node in path:
    print(f"{node['node_type']}: {node['title']}")
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
print(f"Found {len(components)} connected components")

# Check if two specific nodes are connected
graph.are_connected(source_type='Person', source_id=1, target_type='Person', target_id=100)
```

### Centrality Algorithms

All centrality methods return a list of dicts with `node_type`, `title`, `id`, and `score` keys, sorted by score descending. Use `top_k` to limit results.

```python
# Betweenness — identifies bridges/brokers
graph.betweenness_centrality(top_k=10)
graph.betweenness_centrality(normalized=True, sample_size=500)  # approximate for large graphs

# PageRank — importance based on incoming link structure
graph.pagerank(top_k=10, damping_factor=0.85)

# Degree — number of connections
graph.degree_centrality(top_k=10)

# Closeness — how quickly a node can reach all others
graph.closeness_centrality(top_k=10)
```

### Node Degrees

```python
# Get degree (connection count) for each node in the current selection
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

### Proximity from WKT Centroids

```python
graph.type_filter('Field').near_point_km_from_wkt(
    center_lat=60.5, center_lon=3.2, max_distance_km=100.0,
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
# Simple expression
graph.type_filter('Product').calculate(expression='price * 1.1', store_as='price_with_tax')

# Aggregation across traversal
graph.type_filter('User').traverse('PURCHASED').calculate(
    expression='sum(price * quantity)', store_as='total_spent'
)

# Count with grouping
graph.type_filter('User').traverse('PURCHASED').count(store_as='product_count', group_by_parent=True)
```

### Children Properties to List

```python
graph.type_filter('User').traverse('PURCHASED').children_properties_to_list(
    property='title',
    filter={'price': {'<': 500.0}},
    sort='price',
    max_nodes=5,
    store_as='purchased_products'
)
```

### Connection Aggregation

Aggregate properties stored on connections (edges):

```python
graph.type_filter('Discovery').traverse('EXTENDS_INTO').calculate(
    expression='sum(share_pct)',
    aggregate_connections=True
)
```

Supported aggregate functions: `sum`, `avg`/`mean`, `min`, `max`, `count`, `std`.

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
schema = graph.get_schema()  # returns formatted string
```

### Indexes

Create indexes for faster equality filtering (~3x speedup):

```python
graph.create_index('Prospect', 'prospect_geoprovince')
graph.list_indexes()
graph.drop_index('Prospect', 'prospect_geoprovince')
```

---

## Import and Export

### Saving and Loading

```python
graph.save("my_graph.bin")
loaded_graph = rusty_graph.load("my_graph.bin")
```

### Export Formats

```python
graph.export('my_graph.graphml', format='graphml')  # Gephi, yEd
graph.export('my_graph.gexf', format='gexf')        # Gephi native
graph.export('my_graph.json', format='d3')           # D3.js
graph.export('my_graph.csv', format='csv')

# Export as string
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
4. **Create indexes** — `create_index()` on frequently filtered properties
5. **Use lightweight methods** — `node_count()`, `indices()`, `get_node_by_id()` skip property materialization
6. **Limit results** — `max_nodes()` and `max_matches` to bound work on large graphs
7. **Specify direction** — in traversals, specify `direction` when possible
8. **Cypher LIMIT** — use `LIMIT` in Cypher queries to avoid scanning entire result sets

### Lightweight Methods

For performance-critical code, use methods that skip property materialization:

| Method | Returns | Speed |
|--------|---------|-------|
| `node_count()` | Integer count | Fastest |
| `indices()` | List of node indices | Fast |
| `id_values()` | List of ID values | Fast |
| `get_ids()` | List of `{id, title, type}` dicts | Medium |
| `get_nodes()` | List of full node dicts | Slowest |
| `to_df()` | pandas DataFrame | Same as `get_nodes()` |

```python
count = graph.type_filter('User').node_count()        # 50-1000x faster than len(get_nodes())
user = graph.get_node_by_id('User', 12345)             # O(1) lookup vs type_filter + filter
```

Lightweight path methods: `shortest_path_length()`, `shortest_path_indices()`, `shortest_path_ids()`.

### Performance Model

Rusty Graph is optimized for **knowledge graph workloads** — complex multi-step queries on heterogeneous, property-rich graphs. Operations have overhead compared to raw graph algorithms because they build selections, materialize Python dicts, and support the full query API (explain, undo, reports).

For benchmarking, use engine-only methods (`node_count()`, `indices()`, `explain()`) to measure pure graph traversal speed. Use end-to-end methods (`get_nodes()`, `get_properties()`) when measuring the full Python-facing workload.

---

## Graph Maintenance

After heavy mutation workloads (DELETE, REMOVE), the internal graph storage accumulates tombstones — empty slots left by deleted nodes. Use `graph_info()` to monitor storage health and `vacuum()` / `reindex()` to maintain performance.

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

`vacuum()` rebuilds the graph with contiguous indices and rebuilds all indexes. **Resets the current selection** — call between query chains, not mid-chain.

### Reindex — Rebuild Indexes

```python
graph.reindex()  # Rebuilds type, property, and composite indexes from graph state
```

Lighter than `vacuum()` — only rebuilds index structures without compacting storage. Use after bulk mutations when you suspect index drift.

### Recommended Workflow

```python
# After a batch of DELETE/REMOVE operations:
info = graph.graph_info()
if info['fragmentation_ratio'] > 0.3:
    graph.vacuum()    # Compact + reindex (heavy)
elif info['node_tombstones'] > 0:
    graph.reindex()   # Just fix indexes (light)
```

---

## Operation Reports

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
graph.get_last_report()       # most recent report
graph.get_operation_index()   # sequential counter
graph.get_report_history()    # full history
```
