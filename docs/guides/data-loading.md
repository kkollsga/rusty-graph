# Data Loading

> For most use cases, use [Cypher queries](cypher.md). The fluent API is for bulk operations from DataFrames or complex data pipelines.

## End-to-end walkthrough — DataFrames in, queries out

This is the full path from raw `pandas` tables to a queryable graph
on disk. Most domain graphs land somewhere on this template.

### 1. Shape your tables

Two flat tables — one row per node, one row per edge — is enough.
The columns you'll point `add_nodes` / `add_connections` at:

```python
import pandas as pd
import kglite

users = pd.DataFrame({
    "user_id": [1001, 1002, 1003],
    "name":    ["Alice", "Bob",   "Carol"],
    "country": ["US",    "UK",    "US"],
})

products = pd.DataFrame({
    "sku":   ["P-101", "P-102", "P-103"],
    "title": ["Laptop", "Phone", "Tablet"],
    "price": [999.99,  699.99,  349.99],
})

orders = pd.DataFrame({
    "user_id":  [1001, 1001, 1002, 1003],
    "sku":      ["P-101", "P-103", "P-102", "P-101"],
    "date":     ["2024-01-15", "2024-02-10", "2024-01-20", "2024-03-04"],
    "quantity": [1, 2, 1, 1],
})
```

### 2. Load nodes

```python
graph = kglite.KnowledgeGraph()

graph.add_nodes(users,    "User",    "user_id", "name")
graph.add_nodes(products, "Product", "sku",     "title")
```

`unique_id_field` (3rd arg) is what makes a row identifiable;
`node_title_field` (4th, optional) is the human-readable label.
Both get [aliased](#property-mapping) so queries can use either
the original column name or the canonical `id` / `title`.

### 3. Load edges

```python
graph.add_connections(
    orders,
    connection_type="ORDERED",
    source_type="User",      source_id_field="user_id",
    target_type="Product",   target_id_field="sku",
    columns=["date", "quantity"],   # extra props go on the edge
)
```

The two `(type, id_field)` pairs tell `add_connections` how to
look up endpoints in the existing nodes. Any other columns ride
along as edge properties.

### 4. Query

```python
# How many distinct products has each user ordered?
graph.cypher("""
    MATCH (u:User)-[:ORDERED]->(p:Product)
    RETURN u.name AS user, count(DISTINCT p) AS unique_products
    ORDER BY unique_products DESC
""")

# Total revenue by country, layering arithmetic on edge + node props.
graph.cypher("""
    MATCH (u:User)-[r:ORDERED]->(p:Product)
    RETURN u.country, sum(r.quantity * p.price) AS revenue
    ORDER BY revenue DESC
""")
```

### 5. Save & reload

```python
graph.save("orders.kgl")
g2 = kglite.load("orders.kgl")
```

That's the whole loop — three `add_*` calls, one Cypher query.
Everything below this section is reference detail for when the
template needs to bend (timeseries, dates, batch updates, RDF,
declarative blueprints).

## Adding Nodes

```python
import pandas as pd

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

## Property Mapping

When adding nodes, `unique_id_field` and `node_title_field` are **mapped** to `id` and `title`. The original column names become **aliases** — they work in Cypher queries and `filter()`, but results always use the canonical names.

| Your DataFrame Column | Stored As | Alias? |
|-----------------------|-----------|--------|
| `unique_id_field` (e.g., `user_id`) | `id` | `n.user_id` resolves to `n.id` |
| `node_title_field` (e.g., `name`) | `title` | `n.name` resolves to `n.title` |
| All other columns | Same name | — |

```python
# After adding with unique_id_field='user_id', node_title_field='name':
graph.cypher("MATCH (u:User) WHERE u.user_id = 1001 RETURN u")  # OK — alias resolves to id
graph.select('User').where({'user_id': 1001})              # OK — alias works here too
graph.select('User').where({'id': 1001})                   # Also OK — canonical name

# Results always use canonical names:
# {'id': 1001, 'title': 'Alice', 'type': 'User', ...}  — NOT 'user_id' or 'name'
```

## Creating Connections

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

## Working with Dates

```python
graph.add_nodes(
    data=estimates_df,
    node_type='Estimate',
    unique_id_field='estimate_id',
    node_title_field='name',
    column_types={'valid_from': 'datetime', 'valid_to': 'datetime'}
)

graph.select('Estimate').where({'valid_from': {'>=': '2020-06-01'}})
graph.select('Estimate').valid_at('2020-06-15')
graph.select('Estimate').valid_during('2020-01-01', '2020-06-30')
```

## Batch Property Updates

```python
result = graph.select('Prospect').where({'status': 'Inactive'}).update({
    'is_active': False,
    'deactivation_reason': 'status_inactive'
})

updated_graph = result['graph']
print(f"Updated {result['nodes_updated']} nodes")
```

## Operation Reports

Operations that modify the graph return detailed reports:

```python
report = graph.add_nodes(data=df, node_type='Product', unique_id_field='product_id')
# report keys: operation, timestamp, nodes_created, nodes_updated, nodes_skipped,
#              processing_time_ms, has_errors, errors

graph.last_report()       # most recent operation report
graph.operation_index()   # sequential index of last operation
graph.report_history()    # all reports
```

## N-Triples and RDF

`load_ntriples()` streams an RDF/N-Triples file directly into the
graph — designed for Wikidata `latest-truthy` dumps but works with
any N-Triples input:

```python
graph = kglite.KnowledgeGraph(storage="disk", path="/data/wd")
graph.load_ntriples("latest-truthy.nt.bz2", languages=["en"], verbose=True)
```

Compressed inputs (`.bz2`, `.gz`, `.zst`, `.zstd`) are decoded
on-the-fly. Multistream `.bz2` files (the format Wikidata ships)
use a parallel decoder under the hood — ~3× faster than the
single-threaded `MultiBzDecoder`.

`verbose=True` emits one `[Phase N]` line at every gate
(streaming → columnar build → edges → CSR → finalising) plus
periodic in-phase progress for the long Phase 1. Sub-step timings
(CSR step 1/4, peer-count histograms, mmap layout, …) live behind
an env var:

```bash
KGLITE_BUILD_DEBUG=1 python build_graph.py
```

For Wikidata or Sodir specifically, prefer the bundled lifecycle
wrappers — they handle download, cooldown, and resume on top of
`load_ntriples`:

```python
from kglite.datasets import wikidata
g = wikidata.open("/data/wd")
```

See the {doc}`datasets` guide for full coverage.

## Blueprints

Build a complete graph from CSV files using a declarative JSON blueprint — see the [Blueprints guide](blueprints.md) for a full walkthrough.

```python
graph = kglite.from_blueprint("blueprint.json", verbose=True)
```
