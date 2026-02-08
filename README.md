# Rusty Graph

A high-performance graph database library for Python, written in Rust.

```bash
pip install rusty-graph
pip install rusty-graph --upgrade  # upgrade
```

## Quick Start

```python
import rusty_graph
import pandas as pd

graph = rusty_graph.KnowledgeGraph()

users_df = pd.DataFrame({
    'user_id': [1001, 1002, 1003],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [28, 35, 42]
})

graph.add_nodes(
    data=users_df,
    node_type='User',
    unique_id_field='user_id',
    node_title_field='name'
)

print(graph.get_schema())
```

## Table of Contents

- [Data Management](#data-management) — Adding nodes, connections, dates, batch updates
- [Querying](#querying) — Filtering, traversal, set operations
- [Cypher Queries](#cypher-queries) — Full Cypher query language with WHERE, RETURN, ORDER BY, aggregation
- [Pattern Matching](#pattern-matching) — Cypher-like pattern syntax for multi-hop traversals
- [Graph Algorithms](#graph-algorithms) — Shortest path, all paths, connected components
- [Spatial Operations](#spatial-operations) — Bounding box, distance, WKT geometry, point-in-polygon
- [Analytics](#analytics) — Statistics, calculations, connection aggregation
- [Schema and Indexes](#schema-and-indexes) — Schema definition, validation, index management
- [Import and Export](#import-and-export) — Save/load, GraphML, GEXF, D3 JSON, CSV, subgraphs
- [Performance](#performance) — Tips, lightweight methods, performance model
- [Operation Reports](#operation-reports) — Tracking graph mutations

---

## Data Management

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

## Querying

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
graph.type_filter('Product').filter({'stock': {'>': 10}}, sort_spec='price')
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

## Cypher Queries

Query the graph using Cypher — the standard graph query language. Supports `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `ORDER BY`, `SKIP`, `LIMIT`, `WITH`, aggregation functions, and arithmetic expressions.

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

### Optimization

The Cypher executor uses several optimizations:

- **Predicate pushdown**: WHERE equality conditions are moved into MATCH pattern properties for filtering during pattern matching
- **Lazy binding**: Rows carry lightweight `NodeIndex` references during execution; properties are resolved on demand
- **Short-circuit evaluation**: AND/OR predicates short-circuit naturally

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
graph.type_filter('Field').intersects(
    geometry_field='wkt_geometry',
    wkt='POLYGON((1 58, 5 58, 5 62, 1 62, 1 58))'
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
    sort_spec='price',
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

Create indexes for faster equality filtering (~3.3x speedup):

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

```python
count = graph.type_filter('User').node_count()        # 50-1000x faster than len(get_nodes())
user = graph.get_node_by_id('User', 12345)             # O(1) lookup vs type_filter + filter
```

Lightweight path methods: `shortest_path_length()`, `shortest_path_indices()`, `shortest_path_ids()`.

### Performance Model

Rusty Graph is optimized for **knowledge graph workloads** — complex multi-step queries on heterogeneous, property-rich graphs. Operations have overhead compared to raw graph algorithms because they build selections, materialize Python dicts, and support the full query API (explain, undo, reports).

For benchmarking, use engine-only methods (`node_count()`, `indices()`, `explain()`) to measure pure graph traversal speed. Use end-to-end methods (`get_nodes()`, `get_properties()`) when measuring the full Python-facing workload.

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
