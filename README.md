# KGLite — Lightweight Knowledge Graph for Python

[![PyPI version](https://img.shields.io/pypi/v/kglite)](https://pypi.org/project/kglite/)
[![Python versions](https://img.shields.io/pypi/pyversions/kglite)](https://pypi.org/project/kglite/)
[![License: MIT](https://img.shields.io/pypi/l/kglite)](https://github.com/kkollsga/kglite/blob/main/LICENSE)
[![Docs](https://img.shields.io/readthedocs/kglite)](https://kglite.readthedocs.io)

KGLite is an embedded knowledge graph for Python: `pip install`, no
server, no setup. It speaks Cypher, loads pandas DataFrames, and
ships with the connective tissue for AI agents — an MCP server so
Claude / Cursor / any MCP-capable LLM can query your graph as a
tool, a `describe()` method that emits a compact XML schema for
system prompts, and a `code_tree` parser that turns any source
directory into a graph of functions, classes, calls, and imports
across 9 languages.

Three storage modes scale from in-memory (millisecond queries on
small graphs) to mmap-backed on disk (1 B+ edges, Wikidata-scale).
Bundled dataset wrappers turn `pip install kglite` into a queryable
Wikidata or petroleum-domain graph in one line.

## Why KGLite?

- **Built for LLM agents** — `describe()` XML schema, bundled MCP
  server, and an agent-oriented query surface (`cypher()`,
  `graph.select(...).traverse(...)`).
- **One-line public datasets** — `wikidata.open(path)` and
  `sodir.open(path)` handle fetch, parallel build, and caching;
  re-runs reload the cached graph instantly.
- **Codebase → graph in one line** — `kglite.code_tree.build(".")`
  parses Python, Rust, TypeScript, Go, Java, C#, C++, and more
  into `Function` / `Class` / `Module` nodes with `CALLS` /
  `DEFINES` / `IMPORTS` edges.
- **Scales without leaving Python** — in-memory for prototyping,
  mmap-backed for notebook-scale, disk-mode CSR for graphs too
  large for RAM. Same API across modes.
- **Query with Cypher** — `MATCH`, `MERGE`, `OPTIONAL MATCH`,
  aggregations, parameters, semantic search via `text_score()`.
- **DataFrames in, DataFrames out** — bulk-load from pandas, query
  results as DataFrames.

## Quick Start

```bash
pip install kglite
```

```python
import pandas as pd
import kglite

# Three storage modes — pick by graph size:
#   default (in-memory)   — small/medium graphs, fastest queries
#   storage="mapped"      — mmap columns, RAM-friendly as you grow
#   storage="disk", path=…  — 100M+ nodes, Wikidata-scale, loaded lazily
graph = kglite.KnowledgeGraph()

# Bulk-load nodes from a DataFrame (also: add_nodes_bulk, from_blueprint,
# load_ntriples, or Cypher CREATE for ad-hoc inserts).
people = pd.DataFrame({
    "id":   ["alice", "bob", "eve"],
    "name": ["Alice", "Bob", "Eve"],
    "age":  [28, 35, 41],
    "city": ["Oslo", "Bergen", "Trondheim"],
})
graph.add_nodes(people, node_type="Person", unique_id_field="id", node_title_field="name")

# Bulk-load relationships the same way (also: add_connections_bulk,
# add_connections_from_source for auto-filter by loaded types).
knows = pd.DataFrame({"src": ["alice", "bob"], "tgt": ["bob", "eve"]})
graph.add_connections(knows, connection_type="KNOWS",
                      source_type="Person", source_id_field="src",
                      target_type="Person", target_id_field="tgt")

# Query — returns a ResultView (lazy; data stays in Rust until accessed).
result = graph.cypher("""
    MATCH (p:Person) WHERE p.age > 30
    RETURN p.name AS name, p.city AS city
    ORDER BY p.age DESC
""")
for row in result:
    print(row['name'], row['city'])

# Or get a pandas DataFrame directly.
df = graph.cypher("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age", to_df=True)

# Persist to disk and reload.
graph.save("my_graph.kgl")
loaded = kglite.load("my_graph.kgl")
```

## Try it instantly: ready-to-query datasets

Two bundled wrappers turn well-known public sources into queryable
graphs without writing a loader. Each call handles the *fetch +
build + cache* cycle, returns a `KnowledgeGraph` you can `cypher()`
against, and respects a per-dataset cooldown so re-running just
loads the cached graph in seconds. KGLite is independent of the
upstream organisations — see each module docstring for
non-affiliation notes.

### Wikidata

Single-stream `latest-truthy.nt.bz2` from
[dumps.wikimedia.org](https://dumps.wikimedia.org/wikidatawiki/entities/) —
parallel-decoded with a bit-level block scanner, parsed, built into a
queryable graph in one call:

```python
from kglite.datasets import wikidata

g = wikidata.open("/data/wd")                                    # full graph
g = wikidata.open("/data/wd", entity_limit_millions=100)         # 100M slice
g = wikidata.open("/data/wd", storage="memory",                  # in-memory, fast tests
                  entity_limit_millions=10)
```

### Sodir (Norwegian Offshore Directorate)

Petroleum-domain graph from the public ArcGIS REST FeatureServer at
[factmaps.sodir.no](https://factmaps.sodir.no/api/rest/services/DataService) —
33 baseline node types (Field, Wellbore, Discovery, Licence,
Stratigraphy, …), ~480 k nodes, parallel-fetched and built in
seconds:

```python
from kglite.datasets import sodir

g = sodir.open("/data/sodir")  # in-memory by default; ~30s first run
g = sodir.open("/data/sodir", complement_blueprint="my_extras.json")  # extend
```

Two-tier cooldown — cheap row-count probes every 14 days; full
per-dataset re-fetch every 30 days. Add a *complement blueprint* to
extend the baseline (new node types, custom edges) without touching
the canonical schema; the file is persisted into the workdir on
first use and auto-loaded after.

## Use Cases

### Agentic AI — memory and tool use

Give an LLM a structured memory it can query. `describe()` emits a
compact XML schema that fits in a system prompt, and the bundled MCP
server exposes the whole graph as a Cypher tool — drop-in for Claude,
Cursor, or any MCP-capable agent.

```python
xml = graph.describe()                            # schema for the agent's context
prompt = f"You have a knowledge graph:\n{xml}\nAnswer via graph.cypher()."
# Or: python examples/mcp_server.py path/to/graph.kgl
```

### Codebase analysis

Parse Python, Rust, TypeScript, Go, Java, C#, and C++ into a graph of
functions, classes, calls, and imports. Trace who-calls-what, find
dead code, and review structure without leaving your editor. Pairs
naturally with the MCP server so an agent can reason over your repo.

```python
from kglite.code_tree import build

graph = build(".")                                # parse current directory
graph.cypher("""
    MATCH (f:Function)-[:CALLS]->(g:Function)
    RETURN g.name, count(f) AS callers
    ORDER BY callers DESC LIMIT 10
""")
```

### RAG retrieval

Store documents, chunks, and entities together as one graph. Combine
`text_score()` semantic similarity with Cypher structure — hybrid
retrieval in one query, no second vector DB.

```python
graph.cypher("""
    MATCH (c:Chunk)-[:IN_DOC]->(d:Document)
    RETURN c.text, d.title,
           text_score(c.embedding, $query_vec) AS score
    ORDER BY score DESC LIMIT 5
""", params={"query_vec": query_embedding})
```

### Data exploration and analysis

Load CSVs or DataFrames, walk relationships, run graph algorithms
(shortest path, centrality, community detection), and export — all
from a notebook.

```python
graph.add_nodes(users_df, node_type="User", unique_id_field="user_id", node_title_field="name")
graph.cypher("""
    MATCH path = shortestPath((a:User {name:'Alice'})-[*]-(b:User {name:'Eve'}))
    RETURN path
""")
```

## Examples

The [`examples/`](https://github.com/kkollsga/kglite/tree/main/examples)
directory has runnable, self-contained scripts covering each of the
use cases above:

- **[`code_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/code_graph.py)**
  — build a code knowledge graph from a source directory via
  `code_tree.build`. Produces `Function`, `Class`, `Module`, `File`
  nodes with `CALLS`, `DEFINES`, `IMPORTS` edges.
- **[`legal_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/legal_graph.py)**
  — end-to-end `add_nodes` / `add_connections` from pandas DataFrames,
  covering laws, regulations, and court decisions with citation
  relationships. Good template for adapting to your own domain.
- **[`mcp_server.py`](https://github.com/kkollsga/kglite/blob/main/examples/mcp_server.py)**
  — drop-in MCP server that exposes any `.kgl` file to an LLM (Claude,
  Cursor, …) as a Cypher query tool, with schema disclosure and
  code-graph–aware helpers.
- **[`spatial_graph.py`](https://github.com/kkollsga/kglite/blob/main/examples/spatial_graph.py)**
  — declarative CSV→graph loading via a JSON blueprint; regions,
  facilities, and sensors with lat/lon coordinates and pipeline-path
  traversal queries.

For Wikidata- and Sodir-scale builds, see the [Public datasets](#public-datasets)
section above — `kglite.datasets.wikidata.open(...)` and
`kglite.datasets.sodir.open(...)` cover those workflows in one call.

## Benchmarks

KGLite builds and queries Wikidata-scale graphs on a laptop.
Measured with
[`bench/wiki_benchmark.py`](https://github.com/kkollsga/kglite/blob/main/bench/wiki_benchmark.py)
on an M-series MacBook.

**Ingest** — full pipeline from compressed N-Triples to a queryable graph:

| dataset   | triples | nodes  | edges  | ingest  | throughput       | peak RAM |
|-----------|--------:|-------:|-------:|--------:|------------------|---------:|
| wiki100m  |  100 M  |  938 K |  748 K |   29 s  | 3.4 M triples/s  |  1.3 GB  |
| wiki500m  |  500 M  |  5.6 M |  6.7 M |  157 s  | 3.2 M triples/s  |  5.2 GB  |
| wiki1000m |    1 B  | 14.7 M | 15.4 M |  395 s  | 2.5 M triples/s  |  7.0 GB  |

Reloading a saved 1 B-triple graph from disk (7 GB on-disk): **3.5 s**.

**Query latency on the 1 B-triple graph** (mapped storage). Type names
match the labels Wikidata ships per language — with `languages=["en"]`
(the default), `Q5` is renamed to `human`:

| Cypher                                                              |     wall |
|---------------------------------------------------------------------|---------:|
| `MATCH (n)-[:P31]->(:human) RETURN count(n)` — typed aggregation    |   0.5 ms |
| `MATCH (a)-[:P31]->(b)-[:P279]->(c) LIMIT 10` — 2-hop typed         |   0.9 ms |
| `MATCH (a)-[:P31]->(b {nid:'Q64'}) RETURN a LIMIT 20` — pivot       |     1 ms |
| `MATCH (a)-[:P31]->(:human)` `MATCH (a)-[:P27]->(c) LIMIT 10` — join |   44 ms |

Disk and mapped storage track within 1 % on build; mapped wins on
query shapes backed by its in-memory inverted index, disk wins on
unbounded typed traversals by staying on sorted-CSR mmap I/O.

No server, no tuning, same Python process as your code.

## Key Features

| Feature | Description |
|---|---|
| **Cypher queries** | MATCH, CREATE, SET, DELETE, MERGE, aggregations, ORDER BY, LIMIT, SKIP |
| **Semantic search** | Vector embeddings + `text_score()` for similarity ranking |
| **Graph algorithms** | Shortest path, centrality, community detection, clustering |
| **Spatial** | Coordinates, WKT geometry, distance and containment queries |
| **Timeseries** | Time-indexed data with `ts_*()` Cypher functions |
| **Bulk loading** | Fluent API (`add_nodes` / `add_connections`) for DataFrames |
| **Blueprints** | Declarative CSV-to-graph loading via JSON config |
| **Import/Export** | Save/load snapshots, GraphML, CSV export |
| **AI integration** | `describe()` introspection, MCP server, agent prompts |
| **Code analysis** | Parse codebases via tree-sitter (`kglite.code_tree`) |

## Documentation

Full docs at **[kglite.readthedocs.io](https://kglite.readthedocs.io)**:

- [Getting Started](https://kglite.readthedocs.io/en/latest/getting-started.html) — installation, first graph, core concepts
- [Cypher Guide](https://kglite.readthedocs.io/en/latest/guides/cypher.html) — queries, mutations, parameters
- [Semantic Search](https://kglite.readthedocs.io/en/latest/guides/semantic-search.html) — embeddings, vector search
- [AI Agents](https://kglite.readthedocs.io/en/latest/guides/ai-agents.html) — MCP server, `describe()`, agent prompts
- [API Reference](https://kglite.readthedocs.io/en/latest/autoapi/kglite/index.html) — full auto-generated reference

## Requirements

Python 3.10+ (CPython) | macOS (ARM/Intel), Linux (x86_64/aarch64), Windows (x86_64) | `pandas >= 1.5`

## License

MIT — see [LICENSE](https://github.com/kkollsga/kglite/blob/main/LICENSE) for details.
