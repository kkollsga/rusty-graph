# MCP Servers

Expose a KGLite graph to AI agents via the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP). The agent gets Cypher access to your graph through tool calls — no API to learn, no infrastructure to manage.

KGLite ships the server as a console script. For most graphs the
out-of-the-box `kglite-mcp-server` is everything you need. For
project-specific tools — semantic search, source-file access,
parameterised Cypher lookups, custom Python hooks — drop a YAML
manifest next to your graph and the bundled server picks it up
automatically. **No fork required for most customisation.**

## Quick Start

### 1. Install with the MCP extra

```bash
pip install "kglite[mcp]"
```

This pulls in `mcp`, `mcp-methods`, and `PyYAML` alongside KGLite and
registers the `kglite-mcp-server` console script.

### 2. Point it at a graph file

```bash
kglite-mcp-server --graph /path/to/my_graph.kgl
```

The server speaks MCP over stdio and exposes two tools out of the box:

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

## Customising with a manifest

A **manifest** is a YAML file that sits next to your graph and tells
`kglite-mcp-server` to register additional tools. Drop a file named
`<graph_basename>_mcp.yaml` alongside your graph and it loads
automatically:

```
demo.kgl
demo_mcp.yaml      ← auto-detected sibling
```

Or point at any path with `--mcp-config`:

```bash
kglite-mcp-server --graph demo.kgl --mcp-config /path/to/manifest.yaml
```

A manifest can declare three kinds of additions, all optional:

| Section | What it does | Trust |
|---|---|---|
| `source_root:` | Auto-registers `read_source` / `grep` / `list_source` over a directory | None — read-only |
| `tools: cypher: \|` | Parameterised Cypher templates as named MCP tools | None — read-only |
| `tools: python: ...` | Load custom Python functions as MCP tools | Two-signal opt-in |

### `source_root:` — first-class source-file access

Most knowledge graphs index *something* — a codebase, a JSON corpus,
scraped documents. The agent flow is almost always: Cypher narrows
the search, source read for the top hits, occasional grep for
context that didn't make it into the graph. Wire it in with one line:

```yaml
# demo_mcp.yaml
source_root: ./data
```

`./data` is resolved relative to the yaml file's directory, so a
manifest in `/proj/demo_mcp.yaml` exposes `/proj/data`. Use `../`
to point at a sibling directory:

```yaml
source_root: ../scrape
```

For multi-root setups, use `source_roots:`:

```yaml
source_roots:
  - ./data
  - ../shared/lookups
```

This auto-registers three tools, all sandboxed to the configured
roots:

- **`read_source(file_path, start_line=, end_line=, grep=, grep_context=2, max_chars=, max_matches=)`** — read a file relative to the source root. Use `grep="pattern"` to filter to matching lines instead of dumping everything (essential for large files — agents can search a 50 MB JSON without exhausting context).
- **`grep(pattern, glob="*", context=0, max_results=50, case_insensitive=False)`** — regex search across all files in the source roots. Backed by ripgrep crates, gitignore-aware by default.
- **`list_source(path=".", depth=1, glob=None, dirs_only=False)`** — directory tree under the first source root. `path="."` lists the root itself; `depth=2+` produces a recursive tree.

All path resolution is sandboxed — `..` traversal that escapes the
configured roots is rejected.

### `tools:` — inline Cypher tools

Declare Cypher templates as named MCP tools. Each entry becomes a
top-level tool the agent can call by name with typed parameters:

```yaml
tools:
  - name: similar_sessions
    description: Top-k semantically similar sessions for a session id.
    parameters:
      type: object
      properties:
        session_id:
          type: string
        top_k:
          type: integer
          default: 5
      required: [session_id]
    cypher: |
      MATCH (s:Session {id: $session_id})-[r:SIMILAR_TO]->(t:Session)
      RETURN t.id AS id, t.title AS title, r.score AS score
      ORDER BY score DESC LIMIT $top_k
```

The agent sees `similar_sessions(session_id, top_k=5)` as a regular
MCP tool. Param names in the Cypher (`$session_id`, `$top_k`) bind
to the JSON Schema properties at call time.

**Validation runs at server startup**, not at agent call time:

- Every `$param` in the Cypher must appear in `parameters.properties`
- The schema itself must be valid JSON Schema (Draft 2020-12)

Typos surface at boot with a clear error pointing at the yaml file —
not 30 seconds into a conversation.

Manifest Cypher tools cap output at 15 rows / 2k chars. For full
result exports, agents use the bundled `cypher_query` with
`FORMAT CSV`.

### `python:` — custom hook tools (trust-gated)

When inline Cypher isn't enough — you want tool logic that does HTTP
fetches, transforms results, or wraps non-Cypher Python — declare a
hook pointing at a `.py` file:

```yaml
trust:
  allow_python_tools: true

tools:
  - name: session_detail
    description: Full source JSON for a session by id.
    python: ./gcn_tools.py
    function: session_detail
```

```python
# gcn_tools.py
import json
from pathlib import Path

DATA = json.loads(Path(__file__).parent.joinpath("data/sessions.json").read_text())

def session_detail(session_id: str) -> str:
    """Return the canonical session record."""
    record = next((s for s in DATA["sessions"] if s["id"] == session_id), None)
    return json.dumps(record, indent=2) if record else f"Session {session_id} not found."
```

The function's signature, type hints, and docstring are picked up by
FastMCP's introspection — no synthesis layer.

**Loading requires both signals**:

1. Manifest declares `trust.allow_python_tools: true`
2. CLI started with `--trust-tools`

Either alone refuses to load with a startup error. The two-signal
design means the manifest *can* request Python execution, and the
operator independently authorises it. CI / daemons that should never
load arbitrary code just leave `--trust-tools` off.

```bash
kglite-mcp-server --graph demo.kgl --trust-tools
```

File paths resolve relative to the manifest, same contract as
`source_root`.

### Top-level fields

```yaml
name: GCN-2026 Graph                  # FastMCP server display name (optional)
instructions: |                       # Replaces default instructions (optional)
  Custom prompt shown to the agent at server-info time.
source_root: ./data                   # OR source_roots: [./data, ../alt]
trust:
  allow_python_tools: false           # Default: false
tools:
  - name: ...                         # See sections above
```

Anything else fails fast at load time with the offending key
listed.

## End-to-end example: GCN-2026 conference graph

A real graph indexing 1,143 sessions across 4 days. Manifest
co-locates with the graph + the source data:

```
gcn2026/
├── googlenext2026.kgl
├── googlenext2026_mcp.yaml      ← auto-detected
├── gcn_tools.py                 ← custom python hook
└── data/
    ├── sessions/
    │   └── classified.json
    └── speakers/
```

```yaml
# googlenext2026_mcp.yaml
name: GCN-2026 Graph
instructions: |
  Google Cloud Next 2026 conference data — 1,143 sessions across
  Apr 21–24, 1,812 speakers, 526 companies. Use cypher_query for
  structured questions, read_source/grep for raw JSON in ./data,
  similar_sessions for embedding-based recommendations,
  session_detail for fields not lifted to the graph.

source_root: ./data

trust:
  allow_python_tools: true

tools:
  - name: similar_sessions
    description: Top-k semantically similar sessions for a session id.
    parameters:
      type: object
      properties:
        session_id: {type: string}
        top_k:      {type: integer, default: 5}
      required: [session_id]
    cypher: |
      MATCH (s:Session {id: $session_id})-[r:SIMILAR_TO]->(t:Session)
      RETURN t.id AS id, t.title AS title, r.score AS score
      ORDER BY score DESC LIMIT $top_k

  - name: session_detail
    description: Raw source JSON for a session by id.
    python: ./gcn_tools.py
    function: session_detail
```

Run with:

```bash
kglite-mcp-server --graph googlenext2026.kgl --trust-tools
```

Tools registered (visible in any MCP-aware agent):

- `graph_overview`, `cypher_query` — bundled
- `read_source`, `grep`, `list_source` — from `source_root`
- `similar_sessions` — inline Cypher
- `session_detail` — Python hook

That's six tools from ~30 lines of YAML and ~10 lines of Python.

## Forking the bundled server

When the manifest tiers aren't enough — you need custom CSV-export
logic, FastMCP middleware, conditional tool registration, or anything
that touches the server lifecycle — fork
[`examples/mcp_server.py`](https://github.com/kkollsga/kglite/blob/main/examples/mcp_server.py).
That file is a thin wrapper around `kglite.mcp_server.main`; replace
the body with your own `FastMCP` setup to get full control.

When deciding between manifest vs fork:

| Need | Manifest | Fork |
|---|---|---|
| Read-only tools (Cypher templates, source access) | ✅ | overkill |
| Custom Python tool logic | ✅ (`python:` tier) | works too |
| Tool registration conditional on graph schema | ⚠️ no | ✅ |
| Custom MCP middleware / hooks | ❌ | ✅ |
| Replacing `cypher_query` / `graph_overview` | ❌ | ✅ |
| FastMCP transport other than stdio | ❌ | ✅ |

Most projects never need to fork.

### Minimal fork template

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
            return result
        if len(result) == 0:
            return "No results."
        rows = [str(row) for row in result[:15]]
        header = f"{len(result)} row(s)"
        if len(result) > 15:
            header += " (showing first 15)"
        return header + ":\n" + "\n".join(rows)
    except Exception as e:
        return f"Cypher error: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## Patterns for forks

### FORMAT CSV export

When agents need full result sets (not just 15 rows), they append
`FORMAT CSV` to the query. KGLite returns a CSV string directly from
Rust. The bundled server saves it to a temp file and serves it over
a localhost HTTP server with CORS:

```python
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

TEMP_DIR = Path("./temp")
_file_server_port = None

def _ensure_file_server():
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
```

The CORS header lets agents generate HTML artifacts that fetch the
CSV at runtime via `fetch()`, rather than hardcoding thousands of
rows into the artifact source.

### Adding semantic search

```python
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def load(self) -> None: pass
    def unload(self) -> None: pass

graph.set_embedder(Embedder())
```

The agent can then write queries like:

```cypher
MATCH (a:Article)
WHERE text_score(a, 'summary', 'renewable energy') > 0.4
RETURN a.title, text_score(a, 'summary', 'renewable energy') AS score
ORDER BY score DESC LIMIT 10
```

For production servers, implement `load()` / `unload()` with a
cooldown timer. See [Semantic Search](semantic-search.md).

### Mutable graphs

If agents need to modify the graph (CREATE, SET, DELETE, MERGE), add
a `save_graph` tool — without it, mutations live only in memory:

```python
@mcp.tool()
def save_graph() -> str:
    """Save the current graph state to disk."""
    try:
        graph.save(str(graph_path))
        s = graph.schema()
        return f"Saved ({s['node_count']} nodes, {s['edge_count']} edges)."
    except Exception as e:
        return f"Save failed: {e}"
```

### Security

**Read-only mode** rejects mutations at the Cypher level:

```python
graph.read_only(True)
```

**Path traversal** — when accepting file paths from agents, always
validate against an allowed root. (The bundled `read_source` /
`grep` / `list_source` tools already do this for the configured
`source_root`.)

```python
path = (allowed_root / user_path).resolve()
if not path.is_relative_to(allowed_root):
    return "Access denied."
```

**Query parameters** — when passing user input to Cypher, use
`params` to prevent injection:

```python
graph.cypher("MATCH (n) WHERE n.name = $name RETURN n", params={"name": user_input})
```
