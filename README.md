# Rusty Graph Python Library

A high-performance graph database library with Python bindings written in Rust.

## Table of Contents

- [Installation](#installation)
- [Introduction](#introduction)
- [Basic Usage](#basic-usage)
- [Working with Nodes](#working-with-nodes)
- [Creating Connections](#creating-connections)
- [Filtering and Querying](#filtering-and-querying)
- [Traversing the Graph](#traversing-the-graph)
- [Statistics and Calculations](#statistics-and-calculations)
- [Saving and Loading](#saving-and-loading)
- [Performance Tips](#performance-tips)

## Installation

```bash
pip install rusty-graph
```

## Introduction

Rusty Graph is a Rust-based project that aims to empower the generation of high-performance knowledge graphs within Python environments. Specifically designed for aggregating and merging data from SQL databases, Rusty Graph facilitates the seamless transition of relational database information into structured knowledge graphs. By leveraging Rust's efficiency and Python's flexibility, Rusty Graph offers an optimal solution for data scientists and developers looking to harness the power of knowledge graphs in their data-driven applications.

## Key Features
- **Efficient Data Integration:** Easily import and merge data from SQL databases to construct knowledge graphs, optimizing for performance and scalability.
- **High-Performance Operations:** Utilize Rust's performance capabilities to handle graph operations, making Rusty Graph ideal for working with large-scale data.
- **Python Compatibility:** Directly integrate Rusty Graph into Python projects, allowing for a smooth workflow within Python-based data analysis and machine learning pipelines.
- **Flexible Graph Manipulation:** Create, modify, and query knowledge graphs with a rich set of features, catering to complex data structures and relationships.

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

# In operator
selected_products = graph.type_filter('Product').filter({
    'product_id': {'in': [101, 103]}
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

```python
# Sort by a single field (ascending by default)
sorted_products = graph.type_filter('Product').sort('price')

# Sort explicitly by direction
expensive_first = graph.type_filter('Product').sort('price', ascending=False)

# Sort by multiple fields
sorted_complex = graph.type_filter('Product').sort([
    ('stock', False),  # Highest stock first
    ('price', True)    # Then by price, lowest first
])
```

### Limiting Results

```python
# Get at most 5 nodes per group
limited_products = graph.type_filter('Product').max_nodes(5)
```

## Traversing the Graph

```python
# Find products purchased by a specific user
alice = graph.type_filter('User').filter({'name': 'Alice'})
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
users_with_products = graph.type_filter('User').traverse('PURCHASED').children_properties_to_list(
    property='title',  # Default is 'title' if not specified
    filter={'price': {'<': 500.0}},  # Optional filtering of children
    sort_spec='price',  # Optional sorting of children
    max_nodes=5,  # Optional limit of children per parent
    store_as='purchased_products',  # Property name to store the list in parent
    max_length=100,  # Optional maximum string length (adds "..." if truncated)
    keep_selection=False  # Whether to keep the current selection
)
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

## Saving and Loading

```python
# Save graph to file
graph.save("my_graph.bin")

# Load graph from file
loaded_graph = rusty_graph.load("my_graph.bin")
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