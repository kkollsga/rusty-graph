# Data Loading

> For most use cases, use [Cypher queries](cypher.md). The fluent API is for bulk operations from DataFrames or complex data pipelines.

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

## Blueprints

Build a complete graph from CSV files using a declarative JSON blueprint. Instead of writing `add_nodes` / `add_connections` calls, describe your node types, properties, connections, and timeseries in JSON — and `from_blueprint()` handles the rest.

```python
graph = kglite.from_blueprint("blueprint.json", verbose=True)
```

### Blueprint Structure

```json
{
  "settings": {
    "root": "/path/to/csv/files",
    "output": "output/graph.kgl"
  },
  "nodes": {
    "Person": {
      "csv": "persons.csv",
      "pk": "person_id",
      "title": "name",
      "properties": {
        "age": "int",
        "city": "string",
        "salary": "float",
        "hired": "date"
      },
      "skipped": ["internal_code"],
      "filter": {"status": "Active", "age": {">": 18}},
      "connections": {
        "fk_edges": {
          "WORKS_AT": {"target": "Company", "fk": "company_id"}
        },
        "junction_edges": {
          "KNOWS": {
            "csv": "friendships.csv",
            "source_fk": "person_a",
            "target": "Person",
            "target_fk": "person_b",
            "properties": ["since"],
            "property_types": {"since": "date"}
          }
        }
      },
      "sub_nodes": {
        "Review": {
          "csv": "reviews.csv",
          "pk": "auto",
          "parent_fk": "person_id",
          "title": "summary",
          "properties": {"rating": "int"},
          "skipped": ["person_id"]
        }
      },
      "timeseries": {
        "time_key": {"year": "yr", "month": "mo"},
        "resolution": "month",
        "channels": {"sales": "monthly_sales"},
        "units": {"sales": "USD"}
      }
    },
    "Company": {
      "pk": "company_id",
      "title": "company_id",
      "properties": {},
      "skipped": []
    }
  }
}
```

### Key Concepts

| Concept | Description |
|---|---|
| **`pk`** | Primary key column. Use `"auto"` for auto-generated sequential IDs. |
| **`title`** | Column used as the node's display name. |
| **`properties`** | Map column names to types: `"string"`, `"int"`, `"float"`, `"date"`, `"geometry"`, `"location.lat"`, `"location.lon"`. Unspecified columns are auto-detected. |
| **`skipped`** | Columns to exclude from properties (e.g., FK columns you don't want stored). |
| **`filter`** | Row-level filtering. Equality: `{"status": "Active"}`. Operators: `{"age": {">": 18}}` (supports `=`, `!=`, `>`, `<`, `>=`, `<=`). |
| **FK edges** | One-to-many: a column in the source CSV references the PK of a target node type. |
| **Junction edges** | Many-to-many via a separate CSV lookup table. Can attach properties to the edges. |
| **Sub-nodes** | Hierarchical children. Must have `parent_fk` pointing to the parent's PK column. |
| **Manual nodes** | Node types without a `csv` field — created automatically from distinct FK values pointing to them. |
| **Timeseries** | Time-indexed channels attached to nodes. `time_key` can be a single column (`"date_col"`) or a composite (`{"year": "yr", "month": "mo"}`). |

### Loading Options

```python
# Basic load
graph = kglite.from_blueprint("blueprint.json")

# Verbose output + auto-save to settings.output path
graph = kglite.from_blueprint("blueprint.json", verbose=True, save=True)

# Skip auto-save (just build in memory)
graph = kglite.from_blueprint("blueprint.json", save=False)
```

### Loading Phases

`from_blueprint()` processes nodes in dependency order:

1. **Manual nodes** — types without CSV (created from distinct FK values)
2. **Core nodes** — types with CSV files
3. **Sub-nodes** — hierarchical children of core nodes
4. **FK edges** — direct foreign key relationships
5. **Junction edges** — many-to-many via lookup tables

Missing CSV files and invalid rows are handled gracefully — the graph is still created with whatever data is available.
