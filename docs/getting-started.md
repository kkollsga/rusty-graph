# Getting Started

## Installation

```bash
pip install kglite
```

For code-tree parsing (optional):

```bash
pip install kglite[code-tree]
```

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

## Loading Data from DataFrames

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

## Loading a Public Dataset

KGLite ships one-call wrappers for well-known public sources. Each
handles download, caching, cooldown, and graph build:

```python
from kglite.datasets import wikidata, sodir

# Wikidata: parallel-decoded multistream bz2 → disk-cached graph
g = wikidata.open("/data/wd")                              # full graph
g = wikidata.open("/data/wd", entity_limit_millions=100)   # 100M slice

# Sodir: petroleum-domain graph, in-memory by default
g = sodir.open("/data/sodir")
```

Re-running just loads the cached graph — sub-second on Wikidata
slices, ~2 s for the full Sodir graph. See {doc}`guides/datasets`
for the full API including cooldown semantics, complement
blueprints, and parallel-fetch tuning.

## Next Steps

- {doc}`guides/cypher` — full Cypher coverage, parameters, count
  subqueries, semantic search.
- {doc}`guides/data-loading` — bulk-load DataFrames, N-Triples,
  blueprint-driven CSV ingest.
- {doc}`guides/datasets` — Wikidata + Sodir lifecycle wrappers.
- {doc}`core-concepts` — storage modes, return types, the
  fluent / Cypher split.
