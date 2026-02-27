# Using with AI Agents

KGLite is designed to work as a self-contained knowledge layer for AI agents. No external database, no server process, no network — just a Python object with a Cypher interface that an agent can query directly.

## The Idea

1. **Load or build a graph** from your data (DataFrames, CSVs, APIs)
2. **Give the agent `describe()`** — a progressive-disclosure XML schema that scales from tiny to massive graphs
3. **The agent writes Cypher queries** using `graph.cypher()` — no other API to learn
4. **Semantic search works natively** — `text_score()` in Cypher, backed by any embedding model you wrap

No vector database, no graph database, no infrastructure. The graph lives in memory and persists to a single `.kgl` file.

## Quick Setup

```python
xml = graph.describe()  # inventory overview — types, connections, Cypher extensions
prompt = f"You have a knowledge graph:\n{xml}\nAnswer the user's question using graph.cypher()."
```

## MCP Server

Expose the graph to any MCP-compatible agent (Claude, etc.) with a thin server:

```python
from mcp.server.fastmcp import FastMCP
import kglite

graph = kglite.load("my_graph.kgl")
mcp = FastMCP("knowledge-graph")

@mcp.tool()
def describe(types: list[str] | None = None) -> str:
    """Get the graph schema and Cypher reference."""
    return graph.describe(types=types)

@mcp.tool()
def query(cypher: str) -> str:
    """Run a Cypher query and return results."""
    result = graph.cypher(cypher, to_df=True)
    return result.to_markdown()

mcp.run(transport="stdio")
```

The agent calls `describe()` once to learn the schema, then uses `query()` for everything — traversals, aggregations, filtering, and semantic search via `text_score()`. For large graphs, call `describe(types=['Field', 'Well'])` to drill into specific types.

For code graphs, additional tools make exploration easier — see `examples/mcp_server.py` for a full example with `find_entity`, `read_source`, and `entity_context` tools.

## Adding Semantic Search (5-Minute Setup)

Semantic search lets agents find nodes by meaning, not just exact property matches:

```python
# 1. Wrap any embedding model (local or remote)
class Embedder:
    dimension = 384
    def embed(self, texts: list[str]) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts).tolist()

# 2. Register it on the graph
graph.set_embedder(Embedder())

# 3. Embed a text column (one-time, incremental on re-run)
graph.embed_texts("Article", "summary")

# 4. Now agents can search by meaning in Cypher — no extra API
graph.cypher("""
    MATCH (a:Article)
    WHERE text_score(a, 'summary', 'climate policy') > 0.5
    RETURN a.title, text_score(a, 'summary', 'climate policy') AS score
    ORDER BY score DESC LIMIT 10
""")
```

The model wrapper works with any provider — OpenAI, Cohere, local sentence-transformers, Ollama. See [Semantic Search](semantic-search.md) for the full API.

## Tips for Agent Prompts

1. **Start with `describe()`** — gives the agent an inventory of types with capability flags, connection map, and non-standard Cypher extensions
2. **Drill into types with `describe(types=['Field'])`** — shows properties, connections, timeseries/spatial config, supporting children, and sample nodes
3. **Use `properties(type)`** for deeper column discovery — shows types, nullability, unique counts, and sample values
4. **Use `sample(type, n=3)`** before writing queries — lets the agent see real data shapes
5. **Prefer Cypher** over the fluent API in agent contexts — closer to natural language, easier for LLMs to generate
6. **Use parameters** (`params={'x': val}`) to prevent injection when passing user input to queries
7. **ResultView is lazy** — agents can call `len(result)` to check row count without converting all rows

## What `describe()` Returns

- **Inventory mode** (`describe()`): node types as compact descriptors `TypeName[size,complexity,flags]` sorted by count, connection map, Cypher extensions. Core/supporting type tiers hide child types behind `+N` suffixes. For small graphs (≤15 types), full detail is inlined automatically.
- **Focused mode** (`describe(types=['Field'])`): detailed properties with types, connection topology, timeseries/spatial config, supporting children, and sample nodes.
- **Cypher reference** (`describe(cypher=True)`): full language reference including all supported clauses, operators, built-in functions, predicates, and examples.
