# MCP Servers

Expose a KGLite graph to AI agents via the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP). The agent gets Cypher access to your graph through tool calls — no API to learn, no infrastructure to manage.

KGLite ships the server as a console script. For most graphs the
out-of-the-box `kglite-mcp-server` is everything you need; the rest
of this guide is for when you want to fork it and add domain-specific
tools.

## Quick Start

### 1. Install with the MCP extra

```bash
pip install "kglite[mcp]"
```

This pulls in the official `mcp` Python package alongside KGLite and
registers the `kglite-mcp-server` console script.

### 2. Point it at a graph file

```bash
kglite-mcp-server --graph /path/to/my_graph.kgl
```

The server speaks MCP over stdio and exposes two tools out of the
box:

- `graph_overview(...)` — wraps `describe()` for progressive schema
  disclosure (types, connections, Cypher reference).
- `cypher_query(query, timeout_ms=...)` — runs any Cypher query;
  inline result up to 15 rows, append `FORMAT CSV` for a localhost-
  served file export.

Optional: `--embedder all-MiniLM-L6-v2` to register a
`sentence-transformers` model so `text_score()` works inside Cypher.

### 3. Register with Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "my-graph": {
      "command": "kglite-mcp-server",
      "args": ["--graph", "/abs/path/to/my_graph.kgl"]
    }
  }
}
```

For Claude Code, add to `.claude/settings.json` with the same shape.
The agent can now call `graph_overview()` to learn the schema and
`cypher_query()` to query.

### Forking the bundled server

When the two built-in tools aren't enough — you want domain-specific
helpers, custom security guardrails, or extra logging — copy
[`examples/mcp_server.py`](https://github.com/kkollsga/kglite/blob/main/examples/mcp_server.py)
and edit it. That file is a thin wrapper around `kglite.mcp_server.main`;
replacing the body with your own `FastMCP` setup gets you the same
runtime as the bundled CLI but with full control over the tool surface.
The patterns documented in the rest of this guide all apply.

## Core Tools

Every KGLite MCP server should expose three tools:

### graph_overview — Schema discovery

The agent's entry point. Uses `describe()` with progressive disclosure so agents can explore without overwhelming context.

```python
@mcp.tool()
def graph_overview(
    types: list[str] | None = None,
    connections: bool | list[str] | None = None,
    cypher: bool | list[str] | None = None,
) -> str:
    """Get graph schema, connection details, or Cypher language reference.

    Three independent axes — call with no args first for the overview:
      graph_overview()                            — inventory of node types
      graph_overview(types=["Function"])           — property schemas, samples
      graph_overview(connections=True)             — all connection types overview
      graph_overview(connections=["CALLS"])         — deep-dive with properties
      graph_overview(cypher=True)                  — Cypher clauses, functions, procedures
      graph_overview(cypher=["text_score","MATCH"]) — detailed docs with examples"""
    return graph.describe(types=types, connections=connections, cypher=cypher)
```

The three-axis design means agents start broad and drill in. A first call to `graph_overview()` returns a compact inventory (type names, sizes, capability flags). The agent then calls `graph_overview(types=["Field"])` to see properties and connections for a specific type, or `graph_overview(cypher=True)` for the full Cypher language reference.

### cypher_query — Query execution

The workhorse. Handles both regular queries (dict-per-row) and `FORMAT CSV` (file export).

```python
@mcp.tool()
def cypher_query(query: str) -> str:
    """Run a Cypher query against the knowledge graph.
    Append FORMAT CSV to export full results to a CSV file (no row limit)."""
    try:
        result = graph.cypher(query)
        if isinstance(result, str):  # FORMAT CSV
            return _handle_csv_export(result)
        if len(result) == 0:
            return "No results."
        rows = [str(dict(row)) for row in result[:15]]
        header = f"{len(result)} row(s)"
        if len(result) > 15:
            header += " (showing first 15)"
        return header + ":\n" + "\n".join(rows)
    except Exception as e:
        return f"Cypher error: {e}"
```

**Row limit**: Show 15 rows inline. This keeps responses readable for the AI agent without truncating too aggressively. Agents that need full results use `FORMAT CSV`.

### bug_report — Cypher bug tracking

Let agents file bug reports when queries return unexpected results. Reports are written to `reported_bugs.md` with timestamps.

```python
@mcp.tool()
def bug_report(query: str, result: str, expected: str, description: str) -> str:
    """File a Cypher bug report to reported_bugs.md."""
    return graph.bug_report(query, result, expected, description)
```

## FORMAT CSV Export

When agents need full result sets (not just 15 rows), they append `FORMAT CSV` to the query:

```cypher
MATCH (n:Field) RETURN n.name, n.status, n.reserves FORMAT CSV
```

KGLite returns a CSV string directly from Rust — no pandas, no Python overhead. The server saves it to a temp file and serves it over a local HTTP server with CORS enabled:

```python
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

TEMP_DIR = Path("./temp")
_file_server_port = None

def _ensure_file_server():
    """Start a background HTTP server serving temp/ files on a random port."""
    global _file_server_port
    if _file_server_port is not None:
        return _file_server_port
    TEMP_DIR.mkdir(exist_ok=True)

    class CORSHandler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(TEMP_DIR), **kw)
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    server = HTTPServer(("127.0.0.1", 0), CORSHandler)
    _file_server_port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return _file_server_port

def _handle_csv_export(csv_string: str) -> str:
    """Save CSV to temp file and return URL + inspection hint."""
    TEMP_DIR.mkdir(exist_ok=True)
    filename = f"data-{datetime.now():%Y%m%d-%H%M%S}.csv"
    (TEMP_DIR / filename).write_text(csv_string)
    port = _ensure_file_server()
    url = f"http://localhost:{port}/{filename}"
    return (
        f"CSV exported: {url}\n"
        f"read_source(file_path='./temp/{filename}') — add rows=[0,4] to inspect."
    )
```

**Why a local HTTP server?** It lets agents generate HTML artifacts (charts, tables) that fetch the CSV at runtime via `fetch()`, rather than hardcoding thousands of rows into the artifact source. The CORS header makes this work from sandbox iframes.

## Optional: read_source

For servers that export CSV data, add a `read_source` tool so agents can inspect exported files without loading everything into context:

```python
@mcp.tool()
def read_source(file_path: str, rows: list[int] | None = None) -> str:
    """Read a CSV file exported by FORMAT CSV.

    Args:
        file_path: Path to the exported CSV file (e.g. './temp/data-20260304.csv').
        rows: Optional [start, end] row interval (0-indexed, inclusive).
              E.g. [0, 4] returns the header + first 5 data rows."""
    path = (TEMP_DIR.parent / file_path).resolve()
    # Guard: only allow files inside temp/
    if not path.is_relative_to(TEMP_DIR):
        return f"Access denied: {file_path}"
    if not path.exists():
        return f"File not found: {file_path}"

    lines = path.read_text().splitlines()
    header = lines[0] if lines else ""

    if rows and len(rows) == 2:
        start, end = rows[0] + 1, rows[1] + 2  # skip header, inclusive
        selected = lines[start:end]
        return header + "\n" + "\n".join(selected) + f"\n\n[rows {rows[0]}-{rows[1]} of {len(lines) - 1}]"
    return "\n".join(lines)
```

For code graphs (built with `code_tree`), `read_source` can also resolve qualified names to source file locations — see `examples/mcp_server.py` for this pattern.

## Server Instructions

The `instructions` parameter on `FastMCP` tells the agent what the graph contains and how to use it. Keep it concise — the agent reads this on every conversation start.

```python
schema = graph.schema()
mcp = FastMCP(
    "my-domain-graph",
    instructions=(
        f"Knowledge graph with {schema['node_count']} nodes and "
        f"{schema['edge_count']} edges covering [your domain]. "
        "Call graph_overview() first to learn the schema and Cypher reference, "
        "then use cypher_query() to explore."
    ),
)
```

For domain-specific servers, add a **preamble** to `graph_overview()` with workflow tips:

```python
OVERVIEW_PREAMBLE = """Usage tips:
- Start broad: MATCH (n:NodeType) RETURN n LIMIT 5 to explore properties
- Use graph_overview(types=['Person']) to see all properties for a type
- Use graph_overview(connections=True) for the full connection map
- Use graph_overview(cypher=True) for Cypher reference
- Append FORMAT CSV for full data exports (no row limit)"""

@mcp.tool()
def graph_overview(types=None, connections=None, cypher=None):
    """..."""
    if types or connections or cypher:
        return graph.describe(types=types, connections=connections, cypher=cypher)
    return OVERVIEW_PREAMBLE + "\n\n" + graph.describe()
```

## Timing Decorator

Add execution time to every tool response. This helps agents (and you) identify slow queries:

```python
import functools
import time

def _timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        return result + f"\n\n⏱ {ms:.0f}ms"
    return wrapper

@mcp.tool()
@_timed
def cypher_query(query: str) -> str:
    ...
```

## Adding Semantic Search

Register an embedding model so agents can use `text_score()` in Cypher for meaning-based search:

```python
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

graph.set_embedder(Embedder())
```

The agent can then write queries like:

```cypher
MATCH (a:Article)
WHERE text_score(a, 'summary', 'renewable energy policy') > 0.4
RETURN a.title, text_score(a, 'summary', 'renewable energy policy') AS score
ORDER BY score DESC LIMIT 10
```

For production servers, implement `load()` / `unload()` methods with a cooldown timer to free GPU memory between queries. See [Semantic Search](semantic-search.md) for the full protocol.

## Mutable Graphs

If agents need to modify the graph (CREATE, SET, DELETE, MERGE), add a `save_graph` tool:

```python
@mcp.tool()
def save_graph() -> str:
    """Save the current graph state to disk.

    Call after mutations via cypher_query to persist changes.
    Without saving, changes only exist in memory."""
    try:
        graph.save(str(graph_path))
        s = graph.schema()
        return f"Saved ({s['node_count']} nodes, {s['edge_count']} edges)."
    except Exception as e:
        return f"Save failed: {e}"
```

Without this tool, mutations from `cypher_query` live only in memory and are lost when the server restarts.

## Code Graph Tools

For graphs built with `code_tree` (codebase analysis), the example server auto-detects code types and enables extra tools:

| Tool | Purpose |
|------|---------|
| `find_entity` | Search code entities by name (exact, contains, starts_with) |
| `read_source` | Resolve entities to file paths + line ranges |
| `entity_context` | Get relationships: calls, callers, methods, imports |

These are conditionally registered — see `examples/mcp_server.py` for the pattern.

## Security

**Path traversal**: Always validate file paths against an allowed root. Never let agents read arbitrary files on disk:

```python
path = (allowed_root / user_path).resolve()
if not path.is_relative_to(allowed_root):
    return "Access denied."
```

**Read-only mode**: If agents should only query (not mutate), enable read-only mode:

```python
graph.read_only(True)
```

This rejects CREATE, SET, DELETE, REMOVE, and MERGE at the Cypher level.

**Query parameters**: When passing user input to Cypher, use `params` to prevent injection:

```python
graph.cypher("MATCH (n) WHERE n.name = $name RETURN n", params={"name": user_input})
```

## Minimal Server Template

A complete, copy-paste starting point:

```python
#!/usr/bin/env python3
"""MCP server exposing a KGLite knowledge graph."""

import kglite
from mcp.server.fastmcp import FastMCP

graph = kglite.load("my_graph.kgl")
schema = graph.schema()

mcp = FastMCP(
    "my-graph",
    instructions=(
        f"Knowledge graph ({schema['node_count']} nodes, {schema['edge_count']} edges). "
        "Call graph_overview() first, then cypher_query() to explore."
    ),
)

@mcp.tool()
def graph_overview(
    types: list[str] | None = None,
    connections: bool | list[str] | None = None,
    cypher: bool | list[str] | None = None,
) -> str:
    """Get graph schema, connection details, or Cypher language reference."""
    return graph.describe(types=types, connections=connections, cypher=cypher)

@mcp.tool()
def cypher_query(query: str) -> str:
    """Run a Cypher query. Append FORMAT CSV for full export."""
    try:
        result = graph.cypher(query)
        if isinstance(result, str):
            return result  # FORMAT CSV
        if len(result) == 0:
            return "No results."
        rows = [str(dict(row)) for row in result[:15]]
        header = f"{len(result)} row(s)"
        if len(result) > 15:
            header += " (showing first 15)"
        return header + ":\n" + "\n".join(rows)
    except Exception as e:
        return f"Cypher error: {e}"

@mcp.tool()
def bug_report(query: str, result: str, expected: str, description: str) -> str:
    """File a Cypher bug report to reported_bugs.md."""
    return graph.bug_report(query, result, expected, description)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

See [`examples/mcp_server.py`](https://github.com/kkollsga/kglite/blob/main/examples/mcp_server.py)
for a full-featured server with CSV export and optional embedder
support — that file is the canonical place to fork when you want to
register custom tools alongside the bundled `graph_overview` /
`cypher_query`.
