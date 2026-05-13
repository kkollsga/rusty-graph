# KGLite

A knowledge graph that runs inside your Python process. Load data, query with Cypher, do semantic search — no server, no setup, no infrastructure.

```{rubric} Two APIs
```

Use **Cypher** for querying, mutations, and semantic search. Use the **fluent API** (`add_nodes` / `add_connections`) for bulk-loading DataFrames. Most agent and application code only needs `cypher()`.

| | |
|---|---|
| Embedded, in-process | No server, no network; `import` and go |
| In-memory | Persistence via `save()`/`load()` snapshots |
| Cypher subset | Querying + mutations + `text_score()` for semantic search |
| Single-label nodes | Each node has exactly one type |
| Fluent bulk loading | Import DataFrames with `add_nodes()` / `add_connections()` |

**Requirements:** Python 3.10+ (CPython) | macOS (ARM), Linux (x86_64/aarch64), Windows (x86_64) | `pandas >= 1.5`

```bash
pip install kglite
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

getting-started
```

```{toctree}
:maxdepth: 2
:caption: How-to Guides

guides/index
guides/data-loading
guides/cypher
guides/mcp-servers
guides/datasets
guides/blueprints
guides/querying
guides/traversal-hierarchy
guides/semantic-search
guides/spatial
guides/timeseries
guides/graph-algorithms
guides/import-export
guides/ai-agents
guides/code-tree
guides/recipes
```

```{toctree}
:maxdepth: 2
:caption: Explanation

core-concepts
explanation/architecture
explanation/design-decisions
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/cypher-reference
reference/fluent-api
autoapi/index
```

```{toctree}
:maxdepth: 1
:caption: Project

contributing
adding-a-storage-backend
adding-a-query-language
changelog
migrations/mcp-0.6-to-0.9
migrations/mcp-pre-0.9.20
```
