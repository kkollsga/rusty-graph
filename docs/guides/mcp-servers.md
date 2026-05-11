# MCP Servers

> [Model Context Protocol](https://modelcontextprotocol.io/) is the
> protocol Claude / Cursor / agentic CLIs use to call tools. Your
> KGLite graph becomes a server that speaks it over stdin/stdout, and
> the agent gets Cypher access to your data through ordinary tool
> calls — no API to learn, no infrastructure to manage.

KGLite ships `kglite-mcp-server` as a single Rust binary built on the
[mcp-methods] framework (rmcp + manifest-driven tool registration).
For most graphs the out-of-the-box server is everything you need.
For project-specific tools — semantic search, source-file access,
parameterised Cypher lookups, custom Python hooks — drop a YAML
manifest next to your graph and the bundled server picks it up
automatically. **No fork required for most customisation.**

[mcp-methods]: https://github.com/kkollsga/mcp-methods

## Quick Start

### 1. Install

```bash
pip install kglite
```

`kglite-mcp-server` ships with the wheel — it lands on your `PATH`
automatically through the `[project.scripts]` entry point. No `cargo
install`, no `PYO3_PYTHON=` dance, no `install_name_tool` patching.
Run `kglite-mcp-server --help` to confirm.

The binary itself is built in Rust against `kglite::api` and goes
through no Python boundary at tool-call time. The `extensions.embedder`
backend uses [fastembed-rs](https://github.com/Anush008/fastembed-rs)
to run ONNX embedding models directly — no `torch` or
`sentence-transformers` required in your venv.

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

### 2½. Five tools from one yaml line

Drop a sibling YAML file next to your graph and you get three more
tools without writing any Python:

```yaml
# my_graph_mcp.yaml
source_root: ./data
```

That auto-registers `read_source`, `grep`, and `list_source` over the
`./data` directory (sandboxed, ripgrep-backed, gitignore-aware) — five
tools total. Cypher narrows the search at the graph level; the agent
follows up with `read_source` for the top hits or `grep` for context
the graph didn't lift. Full reference is in
[Customising with a manifest](#customising-with-a-manifest) below.

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

### `python:` — custom hook tools (removed in 0.9.18)

Earlier releases let manifests load Python tool functions
(`tools[].python: ./tools.py`) and Python embedder factories
(`embedder: { module, class }`) through an in-process CPython runtime.
0.9.18 dropped both: the binary no longer embeds Python, and the
mcp-methods Python interface isn't on its dep graph. If you need
custom tool logic, write the tool against `kglite::api` in a
downstream Rust binary (see **Building a downstream binary** below)
or fold the logic into a parameterised Cypher template.

For embedders, see `extensions.embedder:` in the next section — the
operator-facing shape is the same single-block declaration, but the
runtime uses [fastembed-rs](https://github.com/Anush008/fastembed-rs)
to run BAAI/bge-m3 (or any of fastembed's catalog) natively without
Python.

### Top-level fields

```yaml
name: My Graph                        # Server display name (optional)
instructions: |                       # Replaces default instructions (optional)
  Custom prompt shown to the agent at server-info time.
source_root: ./data                   # OR source_roots: [./data, ../alt]
builtins:
  save_graph: false                   # Default false — gate write-back tool.
  temp_cleanup: on_overview           # Wipe temp/ on every bare graph_overview().
extensions:                           # kglite-specific addons (see matrix below).
  embedder:
    backend: fastembed
    model: BAAI/bge-m3
  csv_http_server:
    port: 8765
    dir: temp/
tools:
  - name: ...                         # See sections above
```

Anything else fails fast at load time with the offending key
listed.

### Common boot errors

The manifest is validated before `mcp.run()` is called, so most
configuration mistakes surface as a one-line `ERROR:` to stderr at
startup with a non-zero exit code. The recurring ones:

| Error message | What it means | Fix |
|---|---|---|
| `ERROR: <path>: unknown top-level keys: ['foo']` | Typo or unsupported key in manifest. | Compare against the [top-level field list](#top-level-fields). |
| `ERROR: <path>: source root './data' resolves to '/abs/.../data' which is not an existing directory` | The path is relative-to-yaml; it didn't land on a real directory. | Check the path; create the directory; or use `source_roots:` if you have multiple. |
| `ERROR: <path>: cypher tool 'foo': cypher references $params ['bar'] not declared in parameters.properties` | A `$param` in the Cypher template isn't in the JSON Schema. | Add it under `parameters.properties` (and to `required:` if it's mandatory). |
| `ERROR: <path>: cypher tool 'foo': invalid parameters schema: ...` | The `parameters:` block isn't valid JSON Schema (Draft 2020-12). | Check `type`, nested types in `properties`, and `required:` list. |
| `ERROR: <path>: manifest declares N python tool(s) but '--trust-tools' was not passed on the CLI` | Python hooks declared, but the operator hasn't authorised them. | Add `--trust-tools` to the CLI invocation after auditing the manifest's `python:` entries. |
| `ERROR: <path>: manifest declares N python tool(s) but trust.allow_python_tools is not set in the manifest` | The reverse: CLI passed `--trust-tools` but yaml doesn't opt in. | Add `trust:\n  allow_python_tools: true` to the yaml. |
| `ERROR: --mcp-config path does not exist: <path>` | Explicit `--mcp-config` value points at a missing file. | Check the path. Sibling auto-detect is `<basename>_mcp.yaml`. |
| `ERROR: python tool 'foo': function 'bar' not found in <path>` | The yaml `function:` name doesn't match anything in the .py file. | Check the function name. Class methods aren't supported — use module-level functions. |

Exit code 3 is reserved for manifest / validation errors; exit 1 for
graph-file-not-found; exit 2 for missing `[mcp]` extras. Wrapping
scripts can branch on those.

## End-to-end example: a conference catalog graph

A graph indexing conference sessions, speakers, and companies, with
embedding-derived similarity edges between sessions. Manifest
co-locates with the graph file and the source data:

```
conference/
├── conference.kgl
├── conference_mcp.yaml          ← auto-detected
├── tools.py                     ← custom python hook
└── data/
    ├── sessions/
    │   └── classified.json
    └── speakers/
```

```yaml
# conference_mcp.yaml
name: Conference Graph
instructions: |
  Conference catalog — sessions, speakers, companies, plus
  similarity edges between sessions. Use cypher_query for
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
    python: ./tools.py
    function: session_detail
```

Run with:

```bash
kglite-mcp-server --graph conference.kgl --trust-tools
```

Tools registered (visible in any MCP-aware agent):

- `graph_overview`, `cypher_query` — bundled
- `read_source`, `grep`, `list_source` — from `source_root`
- `similar_sessions` — inline Cypher
- `session_detail` — Python hook

That's six tools from ~30 lines of YAML and ~10 lines of Python.

## Building a downstream binary

When the manifest tiers aren't enough — you need conditional tool
registration based on graph schema, custom transports, or tools that
need to share state with the kglite-specific dispatch — build a
*downstream binary* on top of the [`mcp-server`][mcp-server-crate]
framework. `kglite-mcp-server` itself is exactly that: a 514-LoC Rust
crate that registers `cypher_query` / `graph_overview` / `save_graph`
on top of the framework's source / GitHub / python-tool surface.

[mcp-server-crate]: https://github.com/kkollsga/mcp-methods/tree/main/crates/mcp-server

The shape:

```rust
use mcp_server::{McpServer, ServerOptions, apply_python_extensions};
use rmcp::handler::server::router::tool::ToolRoute;

let options = ServerOptions::from_manifest(manifest.as_ref(), "My Server")
    .with_static_source_roots(vec!["/path/to/data".into()]);
let mut server = McpServer::new(options);

// Register your domain tools dynamically:
server.tool_router_mut().add_route(ToolRoute::new_dyn(
    Tool::new("my_tool", "Description", schema),
    move |ctx| Box::pin(async move { /* … */ }),
));

server.serve(rmcp::transport::stdio()).await?;
```

Adding a new tool to `kglite-mcp-server` itself is a single
`make_route()` call in `crates/kglite-mcp-server/src/tools.rs` (~5
lines). When deciding between manifest vs downstream binary:

| Need | Manifest | Downstream binary |
|---|---|---|
| Read-only tools (Cypher templates, source access) | ✅ | overkill |
| Custom Python tool logic | ✅ (`python:` tier) | works too |
| Tool registration conditional on graph schema | ⚠️ no | ✅ |
| Custom rmcp transports / middleware | ❌ | ✅ |
| Replacing `cypher_query` / `graph_overview` | ❌ | ✅ |

Most projects never need a downstream binary.

## Built-in patterns

### `FORMAT CSV` export

When agents need full result sets (not just 15 rows), they append
`FORMAT CSV` to the query. The Rust binary saves it to a temp file
and serves it over a localhost HTTP server with CORS — agents can
generate HTML artifacts that `fetch()` the CSV at runtime instead
of hardcoding thousands of rows into the artifact source.

### Mutable graphs

`save_graph` is built in: when the manifest sets `builtins.save_graph: true`
(single-graph mode), the tool registers automatically and persists
post-mutation graph state to the source `.kgl` path.

### Custom embedders

The manifest's `embedder: { module, class, kwargs }` block — gated
by `trust.allow_embedder: true` + `--trust-tools` — instantiates a
user-supplied class and binds it to the active graph via
`graph.set_embedder()`. The agent can then write queries like:

```cypher
MATCH (a:Article)
WHERE text_score(a, 'summary', 'renewable energy') > 0.4
RETURN a.title, text_score(a, 'summary', 'renewable energy') AS score
ORDER BY score DESC LIMIT 10
```

See [Semantic Search](semantic-search.md) for the embedder protocol.

### Security

- **Read-only mode** rejects mutations at the Cypher level — set via
  `graph.read_only(True)` before binding to the server, or use
  single-graph mode with `builtins.save_graph: false` (the default).
- **Path traversal** is blocked by the framework's source tools: the
  bundled `read_source` / `grep` / `list_source` canonicalise every
  path against the configured `source_root` before any I/O.

**Query parameters** — when passing user input to Cypher, use
`params` to prevent injection:

```python
graph.cypher("MATCH (n) WHERE n.name = $name RETURN n", params={"name": user_input})
```

## Field-by-mode reference

Quick lookup of which manifest keys take effect in which CLI mode.
"—" means the key is parsed and validated but has no behavioural
effect in that mode (it's preserved so the same YAML can be moved
between modes without edits).

| Manifest key | `--graph` | `--workspace` | `--watch` | `--source-root` | bare |
|---|---|---|---|---|---|
| `name`, `instructions` | yes | yes | yes | yes | yes |
| `source_root` / `source_roots` | yes (overrides parent-of-`.kgl`) | — | — | yes (canonical mode) | yes |
| `workspace.kind: local` | — | — | — | — | promoted into workspace mode |
| `env_file` | yes | yes | yes | yes | yes |
| `tools[].cypher` | yes | yes | yes | — (no graph) | — |
| `tools[].python`, `embedder:` | removed in 0.9.18 — see migration below |
| `trust.allow_*` | parsed, no-op | parsed, no-op | parsed, no-op | parsed, no-op | parsed, no-op |
| `builtins.save_graph: true` | yes | — | — | — | — |
| `builtins.temp_cleanup: on_overview` | yes | yes | yes | yes | yes |
| `extensions.embedder` (fastembed) | yes | yes | yes | — (no graph) | — |
| `extensions.csv_http_server` | yes | yes | yes | yes | yes |
| `extensions.<other>` (passthrough) | parsed, opaque | parsed, opaque | parsed, opaque | parsed, opaque | parsed, opaque |

## Migration: 0.9.17 → 0.9.18

### Embedders: `embedder:` → `extensions.embedder:`

The framework-level `embedder:` block (Python class factory) is gone.
Replace with `extensions.embedder:` (Rust-native fastembed-rs):

```yaml
# Before (0.9.17 and earlier — no longer parsed)
embedder:
  module: ./embedder.py
  class: BgeM3Embedder
trust:
  allow_embedder: true

# After (0.9.18+)
extensions:
  embedder:
    backend: fastembed
    model: BAAI/bge-m3            # or any fastembed catalog name
```

Operators with custom `embedder.py` files don't need them any more —
fastembed-rs supports BAAI/bge-m3, bge-small/base/large-en-v1.5,
all-MiniLM-L6-v2, and the multilingual-e5 family natively, downloading
ONNX weights on first use to `~/.cache/fastembed/`.

### `tools[].python:` → Rust shim or Cypher template

Python tool hooks are removed in 0.9.18. Two replacements depending
on shape:

- If the function is mostly Cypher with light parameter munging,
  promote it to a `tools[].cypher` entry with a `$param` template.
- If it has real logic (HTTP fetch, file parse), write a small
  downstream Rust binary that embeds the kglite crate directly —
  see **Building a downstream binary** above. The binary calls
  `kglite::api::CypherExecutor` / `compute_description` / etc.
  without any Python boundary.

### Wheel install

`pip install kglite` now lands `kglite-mcp-server` on `PATH` directly.
The 0.9.17-era discovery flow (`otool -L`, `PYO3_PYTHON=`,
`install_name_tool -add_rpath`) is unnecessary. If your shell still
points at an old `cargo install` binary, drop it and let `pip` win.

### CSV-over-HTTP

The new `extensions.csv_http_server` block opts into a localhost HTTP
listener that serves `FORMAT CSV` exports as URLs instead of inline
strings:

```yaml
extensions:
  csv_http_server:
    port: 8765
    dir: temp/                    # relative to the manifest
    cors_origin: "*"              # optional, defaults to "*"
```

With this set, a `cypher_query` that ends in `FORMAT CSV` writes the
result to `temp/kglite-<hash>.csv` and returns a `http://127.0.0.1:8765/...`
URL the agent can fetch when ready. Useful for million-row exports
that would otherwise blow the MCP response budget.
