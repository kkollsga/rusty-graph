# Rusty Graph Python Library

A high-performance graph database library with Python bindings written in Rust.

## Table of Contents

- [Installation](#installation)
- [Introduction](#introduction)
- [Basic Usage](#basic-usage)
- [Important: Node Property Mapping](#️-important-node-property-mapping)
- [Working with Nodes](#working-with-nodes)
- [Working with Dates](#working-with-dates)
- [Creating Connections](#creating-connections)
- [Filtering and Querying](#filtering-and-querying)
  - [Basic Filtering](#basic-filtering)
  - [Null Value Checks](#null-value-checks)
  - [Filtering Orphan Nodes](#filtering-orphan-nodes)
  - [Sorting Results](#sorting-results)
  - [Limiting Results](#limiting-results)
- [Traversing the Graph](#traversing-the-graph)
- [Set Operations on Selections](#set-operations-on-selections)
- [Path Finding and Graph Algorithms](#path-finding-and-graph-algorithms)
- [Pattern Matching](#pattern-matching)
- [Subgraph Extraction](#subgraph-extraction)
- [Spatial and Geometry Operations](#spatial-and-geometry-operations)
- [Schema Definition and Validation](#schema-definition-and-validation)
- [Index Management](#index-management)
- [Export Formats](#export-formats)
- [Statistics and Calculations](#statistics-and-calculations)
- [Saving and Loading](#saving-and-loading)
- [Operation Reports](#operation-reports)
- [Performance Tips](#performance-tips)
- [Performance Model](#performance-model)

## Installation

```bash
pip install rusty-graph
# upgrade
pip install rusty-graph --upgrade
```

## Introduction

Rusty Graph is a Rust-based project that aims to empower the generation of high-performance knowledge graphs within Python environments. Specifically designed for aggregating and merging data from SQL databases, Rusty Graph facilitates the seamless transition of relational database information into structured knowledge graphs. By leveraging Rust's efficiency and Python's flexibility, Rusty Graph offers an optimal solution for data scientists and developers looking to harness the power of knowledge graphs in their data-driven applications.

## Key Features

- **Efficient Data Integration:** Easily import and merge data from SQL databases to construct knowledge graphs, optimizing for performance and scalability.
- **High-Performance Operations:** Utilize Rust's performance capabilities to handle graph operations, making Rusty Graph ideal for working with large-scale data.
- **Python Compatibility:** Directly integrate Rusty Graph into Python projects, allowing for a smooth workflow within Python-based data analysis and machine learning pipelines.
- **Flexible Graph Manipulation:** Create, modify, and query knowledge graphs with a rich set of features, catering to complex data structures and relationships.
- **Graph Algorithms:** Built-in shortest path, all paths, and connected components algorithms powered by petgraph.
- **Pattern Matching:** Cypher-like query syntax for expressive multi-hop graph traversals.
- **Spatial Operations:** Geographic queries including bounding box, distance (Haversine), and WKT geometry intersection.
- **Export Formats:** Export to GraphML, GEXF, D3 JSON, and CSV for visualization and interoperability.

## Basic Usage

```python
import rusty_graph
import pandas as pd

# Create a new knowledge graph
graph = rusty_graph.KnowledgeGraph()

# Create some data using pandas
users_df = pd.DataFrame({
    'user_id': [1001, 1002, 1003],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [28, 35, 42]
})

# Add nodes to the graph
graph.add_nodes(
    data=users_df,
    node_type='User',
    unique_id_field='user_id', 
    node_title_field='name'
)

# View graph schema
print(graph.get_schema())
```

## ⚠️ Important: Node Property Mapping

When you add nodes to the graph, the column names are **mapped to internal property names**. Understanding this mapping is critical for correct filtering and querying.

### Property Mapping Rules

| Your DataFrame Column | Stored As | Why |
|-----------------------|-----------|-----|
| `unique_id_field` (e.g., `user_id`) | `id` | Canonical node identifier for lookups and filtering |
| `node_title_field` (e.g., `name`) | `title` | Display/label field (renamed, not preserved) |
| All other columns | Same name | Properties are preserved as-is |

**Important:** The `unique_id_field` and `node_title_field` columns are **renamed**, not copied. The original column names (`user_id`, `name`) do not exist as properties on the stored node.

### Example: What Actually Gets Stored

```python
# Your DataFrame
users_df = pd.DataFrame({
    'user_id': [1001, 1002],
    'name': ['Alice', 'Bob'],
    'age': [28, 35]
})

# Adding nodes
graph.add_nodes(
    data=users_df,
    node_type='User',
    unique_id_field='user_id',   # → stored as 'id'
    node_title_field='name'       # → stored as 'title'
)

# What the node actually looks like:
# {'id': 1001, 'title': 'Alice', 'type': 'User', 'age': 28}
#
# NOTE: There is NO 'user_id' or 'name' property!
```

### Correct Filtering

```python
# ❌ WRONG - These properties don't exist:
graph.type_filter('User').filter({'user_id': 1001})  # Returns 0 nodes!
graph.type_filter('User').filter({'name': 'Alice'})  # Returns 0 nodes!

# ✅ CORRECT - Use the mapped property names:
graph.type_filter('User').filter({'id': 1001})       # Works!
graph.type_filter('User').filter({'title': 'Alice'}) # Works!
graph.type_filter('User').filter({'age': 28})        # Works (not mapped)
```

### Best Practice: Use explain()

Always use `explain()` when developing queries to verify node counts at each step:

```python
result = graph.type_filter('User').filter({'id': 1001})
print(result.explain())
# Output: TYPE_FILTER User (1000 nodes) -> FILTER (1 nodes)
#                                               ^^^^^^^^
#                                          Verify this is > 0!
```

If you see `(0 nodes)` after a filter, you're likely using the wrong property name.

## Working with Nodes

### Adding Nodes

```python
# Add products to graph
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
    # Optional: specify which columns to include
    columns=['product_id', 'title', 'price', 'stock', 'category'],
    # Optional: how to handle conflicts with existing nodes
    conflict_handling='update'  # Options: 'update', 'replace', 'skip', 'preserve'
)
```

### Retrieving Nodes

```python
# Get all products
products = graph.type_filter('Product')

# Get node information
product_nodes = products.get_nodes()
print(product_nodes)

# Get specific properties
prices = products.get_properties(['price', 'stock'])
print(prices)

# Get only titles
titles = products.get_titles()
print(titles)
```

## Working with Dates

Rusty Graph supports native DateTime values for date-based filtering and operations.

### Specifying Date Columns

When adding nodes, use the `column_types` parameter to specify which columns should be parsed as dates:

```python
import pandas as pd

# Create data with date columns
estimates_df = pd.DataFrame({
    'estimate_id': [1, 2, 3],
    'name': ['Estimate A', 'Estimate B', 'Estimate C'],
    'valid_from': ['2020-01-01', '2020-06-15', '2021-01-01'],
    'valid_to': ['2020-12-31', '2021-06-14', '2021-12-31'],
    'value': [100.5, 250.3, 180.0]
})

# Add nodes with date columns specified
graph.add_nodes(
    data=estimates_df,
    node_type='Estimate',
    unique_id_field='estimate_id',
    node_title_field='name',
    column_types={'valid_from': 'datetime', 'valid_to': 'datetime'}
)
```

### Filtering on Date Fields

Date fields can be filtered using comparison operators. ISO format strings (YYYY-MM-DD) work correctly for date comparisons:

```python
# Find estimates valid after a specific date
recent_estimates = graph.type_filter('Estimate').filter({
    'valid_from': {'>=': '2020-06-01'}
})

# Find estimates within a date range
active_in_2020 = graph.type_filter('Estimate').filter({
    'valid_from': {'<=': '2020-12-31'},
    'valid_to': {'>=': '2020-01-01'}
})
```

### Temporal Queries

For entities with validity periods (like estimates, contracts, or versions), Rusty Graph provides convenient methods to query based on time:

```python
# Find entities valid at a specific point in time
# Default field names: 'date_from' and 'date_to'
valid_estimates = graph.type_filter('Estimate').valid_at('2020-06-15')

# Use custom field names if your data uses different column names
active_contracts = graph.type_filter('Contract').valid_at(
    '2021-03-01',
    date_from_field='start_date',
    date_to_field='end_date'
)

# Find entities valid during a date range (overlapping periods)
overlapping = graph.type_filter('Estimate').valid_during('2020-01-01', '2020-06-30')

# Chain with other operations
high_value_valid = (
    graph.type_filter('Estimate')
    .valid_at('2020-06-15')
    .filter({'value': {'>=': 100.0}})
)
```

**Note:** `valid_at(date)` finds nodes where `date_from <= date <= date_to`. `valid_during(start, end)` finds nodes whose validity period overlaps with the given range.

### Batch Property Updates

Update properties on multiple nodes at once based on a selection:

```python
# Select nodes and update them in batch
result = graph.type_filter('Prospect').filter({'status': 'Inactive'}).update({
    'is_active': False,
    'deactivation_reason': 'status_inactive'
})

# Access the updated graph and count
updated_graph = result['graph']
nodes_updated = result['nodes_updated']
print(f"Updated {nodes_updated} nodes")

# Use keep_selection=True to preserve the selection for chaining
result = selection.update({'processed': True}, keep_selection=True)

# Update with different value types
graph.type_filter('Node').update({
    'count': 42,           # Integer
    'ratio': 3.14159,      # Float
    'active': True,        # Boolean
    'category': 'updated'  # String
})
```

**Note:** The `update()` method returns a dictionary with `graph` (the updated KnowledgeGraph), `nodes_updated` (count of updated nodes), and `report_index` (index of the operation report). By default, the selection is cleared after update; use `keep_selection=True` to preserve it.

### Query Explain

Get insight into how your queries are executed with the `explain()` method:

```python
# Build a query chain
result = (
    graph.type_filter('Prospect')
    .filter({'region': 'North'})
    .traverse('HAS_ESTIMATE')
)

# See the execution plan
print(result.explain())
# Output: TYPE_FILTER Prospect (6775 nodes) -> FILTER (3200 nodes) -> TRAVERSE HAS_ESTIMATE (10954 nodes)

# Works with temporal queries too
valid_estimates = graph.type_filter('Estimate').valid_at('2020-06-15')
print(valid_estimates.explain())
# Output: TYPE_FILTER Estimate (1000 nodes) -> VALID_AT (450 nodes)
```

**Note:** The `explain()` method shows each operation in the query chain with the actual number of nodes at each step. This helps you understand query performance and optimize your queries.

## Creating Connections

```python
# Purchase data
purchases_df = pd.DataFrame({
    'user_id': [1001, 1001, 1002],
    'product_id': [101, 103, 102],
    'date': ['2023-01-15', '2023-02-10', '2023-01-20'],
    'quantity': [1, 2, 1]
})

# Create connections
graph.add_connections(
    data=purchases_df,
    connection_type='PURCHASED',
    source_type='User',
    source_id_field='user_id',
    target_type='Product',
    target_id_field='product_id',
    # Optional additional fields to include
    columns=['date', 'quantity']
)

# Create connections from currently selected nodes
users = graph.type_filter('User')
products = graph.type_filter('Product')
# This would connect all users to all products with a 'VIEWED' connection
users.selection_to_new_connections(connection_type='VIEWED')
```

## Filtering and Querying

### Basic Filtering

```python
# Filter by exact match
expensive_products = graph.type_filter('Product').filter({'price': 999.99})

# Filter using operators
affordable_products = graph.type_filter('Product').filter({
    'price': {'<': 500.0}
})

# Multiple conditions
popular_affordable = graph.type_filter('Product').filter({
    'price': {'<': 500.0},
    'stock': {'>': 50}
})

# In operator (note: product_id was the unique_id_field, so use 'id')
selected_products = graph.type_filter('Product').filter({
    'id': {'in': [101, 103]}
})
```

### Null Value Checks

You can filter nodes based on whether a field is null (missing) or not null:

```python
# Find nodes where a field is null or missing
nodes_without_category = graph.type_filter('Product').filter({
    'category': {'is_null': True}
})

# Find nodes where a field exists and is not null
nodes_with_category = graph.type_filter('Product').filter({
    'category': {'is_not_null': True}
})

# Combine with other conditions
incomplete_products = graph.type_filter('Product').filter({
    'description': {'is_null': True},
    'price': {'>': 0}
})
```

### Filtering Orphan Nodes

Orphan nodes are nodes that have no connections (no incoming or outgoing edges). You can filter to include or exclude orphan nodes:

```python
# Get only orphan nodes
orphans = graph.filter_orphans(include_orphans=True)

# Get only nodes that have at least one connection
connected = graph.filter_orphans(include_orphans=False)

# Filter orphans with sorting and limits
recent_orphans = graph.filter_orphans(
    include_orphans=True, 
    sort_spec='created_date', 
    max_nodes=100
)

# Chain with other operations
product_orphans = graph.type_filter('Product').filter_orphans(include_orphans=True)
```

### Sorting Results

Rusty Graph offers flexible options for sorting nodes based on their properties. The `sort_spec` parameter can be used in various methods including `type_filter()`, `filter()`, `filter_orphans()`, `traverse()`, and the standalone `sort()` method.

#### Sort Specification Format Options

1. **Single field string**: Sorts by the specified field in ascending order.
   ```python
   # Sort products by price (lowest to highest)
   sorted_products = graph.type_filter('Product').sort('price')
   
   # Can also be used in other methods
   cheap_products = graph.type_filter('Product').filter(
       {'stock': {'>': 10}}, 
       sort_spec='price'
   )
   ```

2. **Field with direction**: Explicitly specify ascending or descending order.
   ```python
   # Sort products by price (highest to lowest)
   expensive_first = graph.type_filter('Product').sort('price', ascending=False)
   ```

3. **List of tuples**: For multi-field sorting with different directions.
   ```python
   # First sort by stock (descending), then by price (ascending)
   # This prioritizes high-stock items, and for items with equal stock,
   # shows the cheapest ones first
   complex_sort = graph.type_filter('Product').sort([
       ('stock', False),  # False = descending order
       ('price', True)    # True = ascending order
   ])
   ```

4. **Dictionary with field and direction**: Alternative format for single field sorting.
   ```python
   # Sort by rating in descending order
   top_rated = graph.type_filter('Product').filter(
       {}, 
       sort_spec={'field': 'rating', 'ascending': False}
   )
   ```

#### Using Sort Specifications in Different Methods

Sort specifications work consistently across methods:

```python
# In type_filter
latest_users = graph.type_filter('User', sort_spec='creation_date', max_nodes=10)

# In filter
new_expensive = graph.type_filter('Product').filter(
    {'price': {'>': 500.0}},
    sort_spec=[('creation_date', False), ('price', True)]
)

# In traversal
alice_recent_purchases = graph.type_filter('User').filter({'title': 'Alice'}).traverse(
    connection_type='PURCHASED',
    sort_target='date',
    max_nodes=5
)

# In filter_orphans
recent_orphans = graph.filter_orphans(
    include_orphans=True,
    sort_spec='last_modified',
    max_nodes=20
)

# In children_properties_to_list
expensive_products = graph.type_filter('User').traverse('PURCHASED').children_properties_to_list(
    property='title',
    sort_spec='price',  # Sort children by price before creating the list
    max_nodes=3,
    store_as='top_expensive_purchases'
)
```

### Limiting Results

```python
# Get at most 5 nodes per group
limited_products = graph.type_filter('Product').max_nodes(5)
```

## Traversing the Graph

```python
# Find products purchased by a specific user
alice = graph.type_filter('User').filter({'title': 'Alice'})
alice_products = alice.traverse(
    connection_type='PURCHASED',
    direction='outgoing'
)

# Access the resulting products
alice_product_data = alice_products.get_nodes()

# Filter the traversal target nodes
expensive_purchases = alice.traverse(
    connection_type='PURCHASED',
    filter_target={'price': {'>=': 500.0}},
    sort_target='price',
    max_nodes=10
)

# Get connection information
connection_data = alice.get_connections(include_node_properties=True)
```

### Filtering on Connection Properties

You can filter traversals based on properties stored on the connections themselves:

```python
# Traverse only through connections with specific property values
high_share_blocks = graph.type_filter('Discovery').traverse(
    connection_type='EXTENDS_INTO',
    filter_connection={'share_pct': {'>=': 50.0}}
)

# Combine connection and target filters
result = graph.type_filter('Discovery').traverse(
    connection_type='EXTENDS_INTO',
    filter_connection={'year': 2021},
    filter_target={'status': 'active'}
)

# Filter connections with null/not-null checks
discounted = user.traverse(
    connection_type='PURCHASED',
    filter_connection={'discount': {'is_not_null': True}}
)
```

## Set Operations on Selections

Rusty Graph supports set operations to combine, intersect, or subtract selections. These operations create new selections without modifying the originals.

### Union

Combines all nodes from both selections (logical OR):

```python
# Select prospects from different geoprovinces
n3_prospects = graph.type_filter('Prospect').filter({'geoprovince': 'N3'})
m3_prospects = graph.type_filter('Prospect').filter({'geoprovince': 'M3'})

# Combine both selections
combined = n3_prospects.union(m3_prospects)
print(f"Total prospects: {len(combined.get_nodes())}")
```

### Intersection

Keeps only nodes present in both selections (logical AND):

```python
# Select large discoveries and discoveries in a specific block
large_discoveries = graph.type_filter('Discovery').filter({'oil_reserves': {'>=': 100.0}})
block_34_discoveries = graph.type_filter('Block').filter({'block_id': 34}).traverse('CONTAINS', direction='incoming')

# Get large discoveries in block 34
result = large_discoveries.intersection(block_34_discoveries)
```

### Difference

Keeps nodes in the first selection but not in the second (subtraction):

```python
# Get all prospects
all_prospects = graph.type_filter('Prospect')

# Get prospects that have estimates
with_estimates = graph.type_filter('ProspectEstimate').traverse('BELONGS_TO', direction='incoming')

# Get prospects WITHOUT estimates
without_estimates = all_prospects.difference(with_estimates)
```

### Symmetric Difference

Keeps nodes that are in exactly one selection but not both (exclusive OR):

```python
# Nodes in category A or B but not both
exclusive_nodes = category_a.symmetric_difference(category_b)
```

### Chaining Operations

Set operations can be chained for complex queries:

```python
# (A union B) intersection C
result = selection_a.union(selection_b).intersection(selection_c)

# A difference (B intersection C)
b_inter_c = selection_b.intersection(selection_c)
result = selection_a.difference(b_inter_c)
```

## Path Finding and Graph Algorithms

Rusty Graph provides efficient implementations of common graph algorithms powered by petgraph.

### Shortest Path

Find the shortest path between two nodes:

```python
# Find shortest path between two nodes
path = graph.shortest_path(
    source_type='Person',
    source_id=1,
    target_type='Person',
    target_id=100
)

# Path is a list of node dictionaries
for node in path:
    print(f"{node['node_type']}: {node['title']}")
```

### All Paths

Find all paths between nodes up to a maximum number of hops:

```python
# Find all paths up to 4 hops
paths = graph.all_paths(
    source_type='Play',
    source_id=1,
    target_type='Wellbore',
    target_id=100,
    max_hops=4
)

# Returns a list of paths, each path is a list of nodes
print(f"Found {len(paths)} paths")
for i, path in enumerate(paths):
    print(f"Path {i+1}: {' -> '.join(n['title'] for n in path)}")
```

### Connected Components

Identify connected components in the graph:

```python
# Get all connected components
components = graph.connected_components()

# Returns a list of components, each component is a list of node IDs
print(f"Found {len(components)} connected components")
for i, component in enumerate(components):
    print(f"Component {i+1}: {len(component)} nodes")
```

## Pattern Matching

Query the graph using Cypher-like pattern syntax for expressive multi-hop queries:

```python
# Simple pattern: Find plays with prospects that became discoveries
results = graph.match_pattern(
    '(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)'
)

# Access matched variables
for match in results:
    print(f"Play: {match['p']['title']}")
    print(f"Prospect: {match['pr']['title']}")
    print(f"Discovery: {match['d']['title']}")

# Pattern with property conditions
results = graph.match_pattern(
    '(u:User)-[:PURCHASED]->(p:Product {category: "Electronics"})'
)

# Limit results for performance on large graphs
results = graph.match_pattern(
    '(a:Person)-[:KNOWS]->(b:Person)',
    max_matches=100
)
```

**Supported pattern syntax:**

- Node patterns: `(variable:NodeType)` or `(variable:NodeType {property: "value"})`
- Relationship patterns: `-[:CONNECTION_TYPE]->`
- Multiple hops: Chain patterns like `(a)-[:REL1]->(b)-[:REL2]->(c)`

## Subgraph Extraction

Extract a portion of the graph for isolated analysis or export:

```python
# Start with a selection and expand to include neighbors
subgraph = (
    graph.type_filter('Company')
    .filter({'title': 'Acme Corp'})
    .expand(hops=2)  # Include all nodes within 2 hops
    .to_subgraph()   # Create independent subgraph
)

# The subgraph is a fully functional KnowledgeGraph
print(f"Subgraph has {subgraph.node_count()} nodes")

# Save the subgraph
subgraph.save('acme_network.bin')

# Export to visualization format
subgraph.export('acme_network.graphml', format='graphml')
```

### Expand Method

The `expand()` method uses breadth-first search to include neighboring nodes:

```python
# Expand selection by 1 hop (immediate neighbors)
expanded = graph.type_filter('Person').filter({'title': 'Alice'}).expand(hops=1)

# Expand by 3 hops for broader context
broad_context = selection.expand(hops=3)
```

## Spatial and Geometry Operations

Query nodes based on geographic location and geometry. Useful for GIS applications and location-based analysis.

### Bounding Box Queries

Find nodes within a rectangular geographic area:

```python
# Find discoveries within a bounding box
north_sea_discoveries = graph.type_filter('Discovery').within_bounds(
    lat_field='latitude',
    lon_field='longitude',
    min_lat=58.0,
    max_lat=62.0,
    min_lon=1.0,
    max_lon=5.0
)
```

### Distance Queries (Haversine)

Find nodes within a radius of a point using great-circle distance:

```python
# Find wellbores within 50km of a location
nearby_wellbores = graph.type_filter('Wellbore').near_point_km(
    center_lat=60.5,
    center_lon=3.2,
    max_distance_km=50.0,
    lat_field='latitude',
    lon_field='longitude'
)
```

### WKT Geometry Intersection

Find nodes whose geometry intersects with a WKT polygon:

```python
# Define a polygon in WKT format
search_area = 'POLYGON((1 58, 5 58, 5 62, 1 62, 1 58))'

# Find fields that intersect the polygon
fields_in_area = graph.type_filter('Field').intersects(
    geometry_field='wkt_geometry',
    wkt=search_area
)
```

### Proximity from WKT Centroids

Find nodes within a radius of a reference point, using the centroid of WKT geometries for distance calculation. This eliminates the need for external dependencies like shapely:

```python
# Find fields within 100km of a location, using polygon centroids
nearby_fields = graph.type_filter('Field').near_point_km_from_wkt(
    center_lat=60.5,
    center_lon=3.2,
    max_distance_km=100.0,
    geometry_field='wkt_geometry'  # Field containing WKT polygons
)

# The centroid is automatically extracted from WKT geometries
# Supports: POLYGON, MULTIPOLYGON, LINESTRING, MULTILINESTRING, POINT, MULTIPOINT
```

### Point-in-Polygon Containment

Find nodes whose WKT geometry contains a specific point:

```python
# Find which blocks contain a specific coordinate
containing_blocks = graph.type_filter('Block').contains_point(
    lat=60.5,
    lon=3.2,
    geometry_field='wkt_geometry'
)

# This is useful for determining which regions/areas a point falls within
```

## Schema Definition and Validation

Define expected structure and validate your graph data:

### Defining a Schema

```python
# Define schema for node types and connections
graph.define_schema({
    'nodes': {
        'Prospect': {
            'required': ['npdid_prospect', 'prospect_name'],
            'optional': ['prospect_status', 'prospect_geoprovince'],
            'types': {
                'npdid_prospect': 'integer',
                'prospect_name': 'string',
                'prospect_ns_dec': 'float'
            }
        },
        'ProspectEstimate': {
            'required': ['estimate_id'],
            'types': {
                'estimate_id': 'integer',
                'value': 'float'
            }
        }
    },
    'connections': {
        'HAS_ESTIMATE': {
            'source': 'Prospect',
            'target': 'ProspectEstimate'
        }
    }
})
```

### Validating Against Schema

```python
# Validate the graph against the defined schema
errors = graph.validate_schema()

if errors:
    print("Validation errors found:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Graph validates successfully!")
```

### Getting Current Schema

```python
# View the current schema (auto-generated from data)
schema = graph.get_schema()
print(schema)
```

**Note:** `get_schema()` returns a formatted string representation of the schema for display purposes, not a Python dictionary.

## Index Management

Create indexes for faster filtering on frequently queried properties:

### Creating Indexes

```python
# Create an index on a property
graph.create_index('Prospect', 'prospect_geoprovince')

# Indexed properties get O(1) lookup for equality filters
# This query will be much faster with an index:
north_prospects = graph.type_filter('Prospect').filter({
    'prospect_geoprovince': 'North Sea'
})
```

### Listing and Dropping Indexes

```python
# List all indexes
indexes = graph.list_indexes()
for idx in indexes:
    print(f"Index on {idx['node_type']}.{idx['property']}")

# Drop an index
graph.drop_index('Prospect', 'prospect_geoprovince')
```

**Performance Note:** Benchmarks show ~3.3x speedup for equality filters on indexed properties. Create indexes on properties you frequently filter by exact value.

## Export Formats

Export your graph to various formats for visualization and interoperability:

### Export to File

```python
# GraphML format (compatible with Gephi, yEd, etc.)
graph.export('my_graph.graphml', format='graphml')

# GEXF format (Gephi native format)
graph.export('my_graph.gexf', format='gexf')

# D3.js JSON format (for web visualization)
graph.export('my_graph.json', format='d3')

# CSV format (nodes and edges as CSV)
graph.export('my_graph.csv', format='csv')
```

### Export to String

Get export data as a string for programmatic use:

```python
# Get GraphML as string
graphml_string = graph.export_string(format='graphml')

# Get D3 JSON as string
d3_json = graph.export_string(format='d3')

# Export only current selection
selected_json = graph.type_filter('Person').export_string(
    format='d3',
    selection_only=True
)
```

### Export Subgraphs

Combine with subgraph extraction for partial exports:

```python
# Export just a portion of the graph
subgraph = (
    graph.type_filter('Company')
    .filter({'region': 'Europe'})
    .expand(hops=2)
    .to_subgraph()
)
subgraph.export('europe_companies.graphml', format='graphml')
```

## Statistics and Calculations

### Basic Statistics

```python
# Get statistics for a property
price_stats = graph.type_filter('Product').statistics('price')
print(price_stats)

# Calculate unique values
unique_categories = graph.type_filter('Product').unique_values(
    property='category',
    # Store result in node property
    store_as='category_list',
    max_length=10
)

# Convert children properties to a comma-separated list in parent nodes
# Option 1: Store results in parent nodes
users_with_products = graph.type_filter('User').traverse('PURCHASED').children_properties_to_list(
    property='title',  # Default is 'title' if not specified
    filter={'price': {'<': 500.0}},  # Optional filtering of children
    sort_spec='price',  # Optional sorting of children
    max_nodes=5,  # Optional limit of children per parent
    store_as='purchased_products',  # Property name to store the list in parent
    max_length=100,  # Optional maximum string length (adds "..." if truncated)
    keep_selection=False  # Whether to keep the current selection
)

# Option 2: Get results as a dictionary without storing them
product_names = graph.type_filter('User').traverse('PURCHASED').children_properties_to_list(
    property='title',
    sort_spec='price',
    max_nodes=5
)
print(product_names)  # Returns {'User1': 'Product1, Product2', 'User2': 'Product3, Product4, Product5'}
```

### Custom Calculations

```python
# Simple calculation: tax inclusive price
with_tax = graph.type_filter('Product').calculate(
    expression='price * 1.1',
    store_as='price_with_tax'
)

# Aggregate calculations per group
user_spending = graph.type_filter('User').traverse('PURCHASED').calculate(
    expression='sum(price * quantity)',
    store_as='total_spent'
)

# Count operations
products_per_user = graph.type_filter('User').traverse('PURCHASED').count(
    store_as='product_count',
    group_by_parent=True
)
```

### Aggregating Connection Properties

Aggregate properties stored on connections (edges) rather than nodes. This is useful when you have data like ownership percentages, weights, or quantities stored on the connections themselves.

```python
# Sum connection properties
# For each Discovery, sum the share_pct on its EXTENDS_INTO connections
total_shares = graph.type_filter('Discovery').traverse('EXTENDS_INTO').calculate(
    expression='sum(share_pct)',
    aggregate_connections=True  # Key parameter for connection aggregation
)
print(total_shares)  # Returns {'Discovery A': 100.0, 'Discovery B': 100.0}

# Average connection properties
avg_ownership = graph.type_filter('Company').traverse('OWNS').calculate(
    expression='avg(ownership_pct)',
    aggregate_connections=True
)

# Count connections
connection_count = graph.type_filter('Parent').traverse('HAS_CHILD').calculate(
    expression='count(any_property)',  # Use any property that exists on connections
    aggregate_connections=True
)

# Store aggregated results on parent nodes
updated_graph = graph.type_filter('Prospect').traverse('HAS_ESTIMATE').calculate(
    expression='sum(weight)',
    aggregate_connections=True,
    store_as='total_weight'  # Stores result on parent Prospect nodes
)
```

**Supported aggregate functions for connections:**

- `sum(property)` - Sum of property values
- `avg(property)` / `mean(property)` - Average of property values
- `min(property)` - Minimum value
- `max(property)` - Maximum value
- `count(property)` - Count of connections (with non-null property values)
- `std(property)` - Standard deviation

**Note:** Connection aggregation requires a traversal before `calculate()`. The results are grouped by the parent (source) node of the traversal.

## Saving and Loading

```python
# Save graph to file
graph.save("my_graph.bin")

# Load graph from file
loaded_graph = rusty_graph.load("my_graph.bin")
```

## Operation Reports

Rusty Graph provides detailed reports for operations that modify the graph, helping you track what changed and diagnose issues.

### Getting Operation Reports

```python
# Add nodes and get the report
report = graph.add_nodes(
    data=df,
    node_type='Product',
    unique_id_field='product_id'
)
print(f"Created {report['nodes_created']} nodes in {report['processing_time_ms']}ms")

# Check for errors
if report['has_errors']:
    print(f"Errors: {report['errors']}")
```

### Report Fields

Node operation reports include:

- `operation`: Type of operation performed
- `timestamp`: When the operation occurred
- `nodes_created`: Number of new nodes created
- `nodes_updated`: Number of existing nodes updated
- `nodes_skipped`: Number of nodes skipped (e.g., due to conflicts)
- `processing_time_ms`: Time taken in milliseconds
- `has_errors`: Boolean indicating if errors occurred
- `errors`: List of error messages (if any)

Connection operation reports include:

- `connections_created`: Number of new connections created
- `connections_skipped`: Number of connections skipped
- `property_fields_tracked`: Number of property fields on connections

### Operation History

```python
# Get the most recent operation report
last_report = graph.get_last_report()

# Get the operation index (sequential counter)
op_index = graph.get_operation_index()

# Get full operation history
history = graph.get_report_history()
for report in history:
    print(f"{report['operation']}: {report['timestamp']}")
```

## Performance Tips

1. **Batch Operations**: Add nodes and connections in batches rather than individually.

2. **Specify Columns**: When adding nodes or connections, explicitly specify which columns to include to reduce memory usage.

3. **Use Indexing**: Filter on node type first before applying other filters.

4. **Avoid Overloading**: Keep node property count reasonable; too many properties per node will increase memory usage.

5. **Conflict Handling**: Choose the appropriate conflict handling strategy:
   - Use `'update'` to merge new properties with existing ones
   - Use `'replace'` for a complete overwrite
   - Use `'skip'` to avoid any changes to existing nodes
   - Use `'preserve'` to only add missing properties

6. **Connection Direction**: Specify direction in traversals when possible to improve performance.

7. **Limit Results**: Use `max_nodes()` to limit result size when working with large datasets.

8. **Create Indexes**: Use `create_index()` on frequently filtered properties for ~3.3x speedup on equality filters.

9. **Use Pattern Matching Limits**: When using `match_pattern()`, set `max_matches` to avoid scanning the entire graph.

10. **Use Lightweight Methods**: For counting or index-only operations, use `node_count()` or `indices()` instead of `get_nodes()` - they're 50-1000x faster because they skip property materialization.

11. **Use get_node_by_id()**: For single-node lookups by ID, use `get_node_by_id('NodeType', id)` instead of `type_filter().filter()` - it's O(1) after the first call.

## Performance Model

Rusty Graph is optimized for **knowledge graph workloads** - complex multi-step queries on heterogeneous, property-rich graphs. It is NOT optimized for micro-benchmarks of raw graph algorithms.

### What Rusty Graph is Good At

- Complex multi-hop traversals with filtering at each step
- Property-rich nodes with many attributes
- Schema-aware queries with validation
- Aggregations and calculations across traversals
- Combining data from multiple sources

### Understanding the Overhead

Rusty Graph operations have overhead compared to raw graph algorithms because they provide additional features:

- Building and tracking selections at each step (enables `explain()`, chaining)
- Materializing Python dictionaries for node properties
- Schema validation and type checking
- Supporting the full query API (undo, reports, etc.)

### Benchmarking: Engine-Only vs End-to-End

When benchmarking, distinguish between two modes:

| Mode | Methods | What's Measured |
|------|---------|-----------------|
| **Engine-only** | `node_count()`, `indices()`, `get_node_by_id()`, `explain()` | Pure graph traversal speed |
| **End-to-end** | `get_nodes()`, `get_properties()` | Includes Python dict materialization |

**For fair performance comparisons**, use engine-only methods. Use end-to-end methods when measuring the full Python-facing workload including property access.

### Optimizing Your Queries

```python
# SLOW: Materializes all properties for every node
nodes = graph.type_filter('User').get_nodes()
count = len(nodes)

# FAST: Just counts without materialization (1000x faster)
count = graph.type_filter('User').node_count()

# SLOW: Scans all nodes of type
user = graph.type_filter('User').filter({'id': 12345}).get_nodes()[0]

# FAST: O(1) hash lookup after first call
user = graph.get_node_by_id('User', 12345)
```
