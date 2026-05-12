# MCP Servers

> [Model Context Protocol](https://modelcontextprotocol.io/) is the
> protocol Claude / Cursor / agentic CLIs use to call tools. Your
> KGLite graph becomes a server that speaks it over stdin/stdout, and
> the agent gets Cypher access to your data through ordinary tool
> calls — no API to learn, no infrastructure to manage.

KGLite ships `kglite-mcp-server` as a Python entry point on top of
the pure-Rust [mcp-methods] framework (rmcp + manifest-driven tool
registration in Rust; Python orchestration thin enough that hot-path
dispatch is sub-microsecond per call). Install with
`pip install 'kglite[mcp]'`; the server lands on PATH and the agent
gets `graph_overview` + `cypher_query` over MCP stdio. For
project-specific tools — semantic search, source-file access,
parameterised Cypher lookups, query preprocessing — drop a YAML
manifest next to your graph and the bundled server picks it up
automatically. **No fork required for most customisation.**

[mcp-methods]: https://github.com/kkollsga/mcp-methods

## Quick Start

### 1. Install

```bash
pip install 'kglite[mcp]'
```

`kglite-mcp-server` ships with the wheel as a Python console-script
entry point. Run `kglite-mcp-server --help` to confirm.

The `[mcp]` extras pull `mcp` (the official Python SDK), `fastembed`,
`aiohttp`, `pyyaml`, and `watchdog` — only installed when you
actually want to run the server. Plain `pip install kglite` gives
you the graph engine without the server deps.

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

A manifest can declare several kinds of additions, all optional:

| Section | What it does | Trust |
|---|---|---|
| `source_root:` / `source_roots:` | Auto-registers `read_source` / `grep` / `list_source` over the directory tree | None — read-only |
| `tools[].cypher` | Parameterised Cypher templates as named MCP tools | None — read-only |
| `extensions.embedder` | Registers an embedder so `text_score()` works inside Cypher | `trust.allow_embedder: true` |
| `extensions.csv_http_server` | Localhost listener that serves `FORMAT CSV` exports as URLs | None |
| `extensions.cypher_preprocessor` | Manifest-declared Python hook that rewrites queries before execution | `trust.allow_query_preprocessor: true` |
| `workspace:` | Bind a local directory (or clone-and-track GitHub repos) as the active source root | None |
| `builtins.save_graph: true` | Registers `save_graph` so the agent can persist mutations | None |

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

### `extensions.embedder` — semantic search inside Cypher

Wire bge-m3 (or any fastembed-catalog model) so `text_score()` works
inside `cypher_query`. Worked example at
{doc}`../examples/manifest_with_embedder`. Reference under
[`extensions:` schema reference](#extensions-schema-reference) below.

### `extensions.cypher_preprocessor` — rewrite agent input

Manifest-declared Python hook that fires before every `cypher_query`
and `tools[].cypher` invocation. Useful for domain-specific
identifier coercion (Wikidata Q-numbers → integers), date format
normalisation, multi-tenant scoping, etc. Worked example at
{doc}`../examples/manifest_cypher_preprocessor`. Reference below.

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
| `ERROR: --mcp-config path does not exist: <path>` | Explicit `--mcp-config` value points at a missing file. | Check the path. Sibling auto-detect is `<basename>_mcp.yaml`. |
| `ERROR: extensions.cypher_preprocessor requires trust.allow_query_preprocessor: true` | Preprocessor declared without the trust gate. | Add `trust:\n  allow_query_preprocessor: true` to the manifest. |
| `ERROR: extensions.cypher_preprocessor.module file does not exist: <path>` | Preprocessor module path is wrong (paths are manifest-relative). | Check that the `.py` file exists at the configured path. |

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
  session_detail for the full session record by id.

source_root: ./data

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
    description: Full record for a session by id.
    parameters:
      type: object
      properties:
        session_id: {type: string}
      required: [session_id]
    cypher: |
      MATCH (s:Session {id: $session_id})
      OPTIONAL MATCH (s)-[:PRESENTED_BY]->(speaker:Speaker)
      OPTIONAL MATCH (speaker)-[:WORKS_AT]->(company:Company)
      RETURN s, collect(DISTINCT speaker) AS speakers,
             collect(DISTINCT company) AS companies
```

Run with:

```bash
kglite-mcp-server --graph conference.kgl
```

Tools registered (visible in any MCP-aware agent):

- `graph_overview`, `cypher_query`, `ping` — bundled
- `read_source`, `grep`, `list_source` — from `source_root`
- `similar_sessions` — inline Cypher
- `session_detail` — inline Cypher

That's seven tools from ~35 lines of YAML, zero Python. For shapes
that need real Python logic (HTTP fetch, file parse, custom
identifier rewriting), see {doc}`../examples/manifest_cypher_preprocessor`
for the query-preprocessor hook, or **Building a downstream binary**
below for full Rust integration.

## Building a downstream binary

When the manifest tiers aren't enough — you need conditional tool
registration based on graph schema, custom transports, or tools that
need to share state with the kglite-specific dispatch — build a
*downstream binary* on top of the pure-Rust [`mcp-methods`](https://crates.io/crates/mcp-methods)
crate. `kglite-mcp-server` itself is exactly that: a small Rust
crate (see [`crates/kglite-mcp-server`](https://github.com/kkollsga/kglite/tree/main/crates/kglite-mcp-server))
that registers `cypher_query` / `graph_overview` / `save_graph` on
top of the framework's source / GitHub / workspace surface.

The shape (using mcp-methods 0.3.30+ from crates.io):

```rust
use mcp_methods::server::{McpServer, ServerOptions, load_manifest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let manifest = load_manifest(std::path::Path::new("manifest.yaml"))?;
    let options = ServerOptions::from_manifest(&manifest, "My Server")
        .with_static_source_roots(vec!["/path/to/data".into()]);
    let mut server = McpServer::new(options);

    // Register your domain tools (typed args via #[derive(Deserialize)]):
    server.register_typed_tool::<MyArgs, _>(
        "my_tool",
        "What the tool does.",
        |args: MyArgs| async move {
            // ... your logic ...
            Ok(format!("response body"))
        },
    );

    server.serve(rmcp::transport::stdio()).await?;
    Ok(())
}
```

For the canonical worked example with full setup, see the
[mcp-methods downstream-binary guide](https://mcp-methods.readthedocs.io/en/latest/guides/downstream-binary.html)
or the runnable
[`examples/downstream_binary/`](https://github.com/kkollsga/mcp-methods/tree/main/examples/downstream_binary)
in the mcp-methods repo.

Adding a new tool to `kglite-mcp-server` itself is a single
registration call in `crates/kglite-mcp-server/src/tools.rs` (~5
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

## Reference

The full programmable surface of `kglite-mcp-server`, with the
"what's documented enough that an agent or operator can rely on
it?" stance: anything in this section is treated as a contract.

### Mode × YAML-field acceptance matrix

Which manifest key takes effect in which CLI mode. "—" means the
key parses cleanly but has no behavioural effect in that mode (the
same YAML can move between modes without edits). The graph file is
the discriminator for `--graph` / `--workspace` / `--watch` /
`--source-root` / bare:

| Manifest key | `--graph` | `--workspace` | `--watch` | `--source-root` | bare (no graph) |
|---|---|---|---|---|---|
| `name`, `instructions`, `overview_prefix` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `source_root` / `source_roots` | ✓ (overrides parent-of-`.kgl`) | — | — | ✓ (canonical) | ✓ |
| `env_file` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `workspace.kind: local` + `workspace.root: <dir>` | — | — | — | — | promotes into local-workspace mode |
| `workspace.watch: true` | — | — | ✓ (auto-rebuild) | — | ✓ when `workspace.kind: local` |
| `tools[].cypher` | ✓ | ✓ (per active repo) | ✓ | — (no graph) | — |
| `trust.allow_python_tools` / `allow_embedder` / `allow_query_preprocessor` | parsed, used by matching extension | parsed, used by matching extension | parsed, used by matching extension | parsed, used by matching extension | parsed, used by matching extension |
| `builtins.save_graph: true` | ✓ (registers `save_graph`) | — (multiple graphs) | — | — | — |
| `builtins.temp_cleanup: on_overview` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `extensions.embedder` | ✓ | ✓ (per active repo) | ✓ | — (no graph) | — |
| `extensions.csv_http_server` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `extensions.cypher_preprocessor` | ✓ | ✓ | ✓ | — (no graph) | — |
| `extensions.<other>` (passthrough) | parsed, opaque to framework | parsed, opaque | parsed, opaque | parsed, opaque | parsed, opaque |
| legacy top-level `embedder:` (pre-0.9.18) | parsed and ignored | parsed and ignored | parsed and ignored | parsed and ignored | parsed and ignored |
| `tools[].python:` (pre-0.9.18) | not loaded by Python entry point; mcp-methods Rust framework still parses it | (same) | (same) | (same) | (same) |

Unknown keys at the top level (or under `builtins:` / `workspace:` /
`trust:` / `tools[]` / `embedder:`) fail validation at boot with a
non-zero exit and an `ERROR: <path>: unknown ... keys: [...]`
message. Keys under `extensions:` are deliberately unvalidated —
they're the downstream-binary passthrough zone.

### Tool gating

Which tool registers, under what conditions. `tools/list` only ever
shows what's registered, so this also answers "what set of tools
will my agent see?"

| Tool | Registered when | Notes |
|---|---|---|
| `cypher_query` | always | Returns inline rows or CSV URL — see "Tool response formats". |
| `graph_overview` | always | Always available even with no graph: returns the no-graph message. |
| `ping` | always | Liveness probe. |
| `read_code_source` | always | Requires an active graph at call time (returns the no-graph message otherwise). |
| `save_graph` | `--graph` mode AND `builtins.save_graph: true` | Other modes have no single graph to save back to. |
| `read_source` / `grep` / `list_source` | a source root is configured (`--source-root`, `--graph` parent auto-bind, manifest `source_root:`, or active workspace repo) | All three register together; never registered independently. |
| `repo_management` | `--workspace` mode OR `workspace.kind: local` in manifest | Local-mode rejects `name=` and `delete=true`; both are github-only. |
| `set_root_dir` | `workspace.kind: local` only | Sandboxed against the manifest-declared `workspace.root` for the lifetime of the server. |
| `github_issues` / `github_api` | `GITHUB_TOKEN` (or `GH_TOKEN`) reachable at boot | Token loaded from process env, walk-up `.env`, or explicit `env_file:`. Tools are registered together; never one without the other. |
| Manifest `tools[].cypher` entries | the manifest declares them AND the mode supports cypher (anything but `--source-root` and bare) | Tool names cannot collide with the built-ins above. |

### Tool response formats

Bundled-tool response shapes are treated as version-stable contracts
across patch releases — they're tagged below per stability. Manifest
`tools[].cypher` responses inherit `cypher_query`'s format.

| Tool | Response shape | Stability |
|---|---|---|
| `ping` | `<message>` (default `pong`) | Stable. |
| `cypher_query` (inline) | `<N> row(s)[ (showing first 15)]:\n<TAB-joined column names>\n<TAB-joined repr'd values per row>\n` | Stable post-0.9.22 (the 0.9.21 row-formatter regression is the canonical "this is now a contract" event). |
| `cypher_query FORMAT CSV` with `csv_http_server` | `FORMAT CSV: <N> row(s) written to <url>\nFetch with: curl <url>` | Stable. |
| `cypher_query FORMAT CSV` without `csv_http_server` | Inline CSV body. | Stable. |
| `cypher_query` errors | `Cypher error: <engine message>` | Stable. |
| `graph_overview` | XML schema (see `describe()` output) — types / connections / cypher panes depending on args. | Stable; the XML shape is the canonical agent-facing format. |
| `read_source` | First line: `<path>  (lines X-Y of Z)`, body lines: `   <lineno>: <text>`. Truncation footer when `max_chars` trips: `... (truncated)`. | Stable. |
| `read_source` (path errors) | `Error: path '<path>' does not exist or access denied.` | Stable. |
| `grep` | `<path>:<line>:<text>` for matches, `<path>-<line>-<text>` for context lines. | Stable. |
| `list_source` | Tree-formatted directory listing relative to the primary source root. | Stable. |
| `read_code_source` | First line: `// <qualified_name> (<path>:<start>-<end>)`, body lines: `   <lineno>: <text>`. | Stable. |
| `save_graph` | `Saved <path> (<N> nodes, <M> edges).` (or `Saved <path>.` when schema unavailable). | Stable. |
| `save_graph` (no graph) | `save_graph requires --graph mode (no source path bound).` | Stable. |
| `repo_management` (list) | `<N> live repo(s):\n  <repo>[ [active]]  (<count> access[es], last <when>)` | Stable. |
| `repo_management` (activate) | `Cloned 'org/repo' at <path>.` / `Updated 'org/repo' at <path>.` / `Activated (already up to date) 'org/repo' at <path>.` | Stable. |
| `set_root_dir` (success) | `Active root set to <absolute_path>.` | Stable. |
| `set_root_dir` (escape) | `Error: path '<path>' escapes the workspace root.` | Stable. |
| `github_issues` (FETCH) | Issue/PR/discussion body with `cb_N` / `patch_N` / `comment_N` / `review_N` placeholders for collapsed elements. Drill down with `element_id=<placeholder>`. | Stable. |
| `github_issues` (LIST/SEARCH) | `<N> discussions in org/repo (<state>):` then per-line summary. | Stable. |
| `github_api` | Pretty-printed JSON body, truncated to `truncate_at` chars (default 80 000). | Stable. |
| (any tool, no active graph) | `No active graph. Pass --graph X.kgl, or activate one via repo_management('org/repo').` | Stable. |

If a future release needs to change a stable shape, that's a breaking
change tracked in the `CHANGELOG.md` "Changed" section (not "Fixed")
and the version bumps minor, not patch.

### `extensions:` schema reference

The `extensions:` block is the kglite-specific addon namespace. The
keys validated below are first-class — they have parser-level
validation, default values, and contracts. Anything else under
`extensions.*` is opaque passthrough.

Machine-readable JSON Schema (Draft 2020-12) for each first-class
block lives under [`docs/schemas/extensions/`][schemas-dir] in the
repo:

- [`csv_http_server.json`][schema-csv]
- [`embedder.json`][schema-embedder]
- [`cypher_preprocessor.json`][schema-preprocessor]

The schemas are anchored to the Python parsers by the regression
test `tests/test_extensions_schemas.py` — any drift between
"what the parser accepts" and "what the schema accepts" surfaces
as a test failure on the next CI run.

[schemas-dir]: https://github.com/kkollsga/kglite/tree/main/docs/schemas/extensions
[schema-csv]: https://github.com/kkollsga/kglite/blob/main/docs/schemas/extensions/csv_http_server.json
[schema-embedder]: https://github.com/kkollsga/kglite/blob/main/docs/schemas/extensions/embedder.json
[schema-preprocessor]: https://github.com/kkollsga/kglite/blob/main/docs/schemas/extensions/cypher_preprocessor.json

#### `extensions.embedder`

Registers an embedder so `text_score()` works inside Cypher.

```yaml
extensions:
  embedder:
    backend: fastembed              # required; only "fastembed" is currently supported
    model: BAAI/bge-m3              # required; see "Embedder backend × model catalog" below
    cooldown: 900                   # optional; seconds (default 900). 0 = never release.
```

| Field | Type | Default | Constraint |
|---|---|---|---|
| `backend` | string | (required) | Must be `"fastembed"`. |
| `model` | string | (required) | Must be in the catalog table below. |
| `cooldown` | int | 900 | seconds; `0` disables auto-release. bge-m3 only — other models ignore this field. |

The legacy `embedder:` block (top-level, 0.9.17 and earlier) is
parsed by the framework but the kglite Python entry point does not
load Python embedder factories — use `extensions.embedder:` and an
in-catalog model.

#### `extensions.csv_http_server`

Spawns a localhost HTTP listener (loopback only) that serves CSV
exports produced by `cypher_query ... FORMAT CSV`.

```yaml
extensions:
  csv_http_server:
    port: 8765                      # optional; default 8765
    dir: temp/                      # optional; default temp/ (relative to manifest)
    cors_origin: "*"                # optional; default "*"
```

Also accepts shorthand:

```yaml
extensions:
  csv_http_server: true             # defaults — port 8765, dir temp/
  # or
  csv_http_server: false            # explicitly disabled (same as absent)
```

| Field | Type | Default | Constraint |
|---|---|---|---|
| `port` | int | 8765 | `0 ≤ port ≤ 65535`. |
| `dir` | string | `temp` | Path; resolved against the manifest's parent directory. |
| `cors_origin` | string | `"*"` | Sent in `Access-Control-Allow-Origin`. Use a specific origin for tighter security. |

Only GETs of flat filenames inside `dir` are served. No directory
listings, no write surface from the HTTP layer (writes only come
from the Cypher executor via `FORMAT CSV`).

#### `extensions.cypher_preprocessor` (0.9.25+)

Manifest-declarable hook fired before every `cypher_query` and
`tools[].cypher` invocation. Optionally rewrites query + params
before they reach `graph.cypher(...)`.

```yaml
trust:
  allow_query_preprocessor: true    # required gate; mirrors allow_embedder

extensions:
  cypher_preprocessor:
    module: ./wikidata_preprocessor.py   # path; relative to manifest, or absolute
    class: WikidataPreprocessor          # callable class OR
    # function: rewrite                  # alternative: module-level free function
    kwargs:                              # optional, passed to constructor
      log_rewrites: false
```

Protocol the loaded callable must implement:

```python
class CypherPreprocessor(Protocol):
    def rewrite(self, query: str, params: dict | None) -> tuple[str, dict | None]:
        """Rewrite the query and/or params before execution.
        Either may be returned unchanged.
        Raising ValueError / TypeError surfaces to the agent as
        'preprocessor: <msg>' inside the tool response."""
```

The hook fires for `cypher_query` and `tools[].cypher` only. It is
**not** called for `graph_overview`, `read_source`, `grep`,
`list_source`, `read_code_source`, `repo_management`, `set_root_dir`,
`github_issues`, `github_api`, `ping`, or `save_graph`.

| Field | Type | Default | Constraint |
|---|---|---|---|
| `module` | string | (required) | Path to a `.py` file. Manifest-relative or absolute. |
| `class` | string | one of `class`/`function` required | Importable class name; instantiated with `kwargs`. |
| `function` | string | one of `class`/`function` required | Module-level function with signature `(query, params) -> (query, params)`. |
| `kwargs` | mapping | `{}` | Passed to the class constructor (ignored for `function:`). |

#### `extensions.<other>` (passthrough)

Any other key under `extensions:` parses cleanly and is preserved on
the loaded `Manifest.extensions` dict. The framework does not
validate inner shape. Downstream consumers (kglite-mcp-server, your
own server binaries) read whatever they need from this map.

### `tools[].cypher` template reference

Manifest entries shaped like

```yaml
tools:
  - name: <identifier>
    description: <agent-visible explanation>
    parameters: <JSON Schema object>
    cypher: |
      <Cypher template with $param placeholders>
```

become first-class MCP tools. Behaviour:

**Name** — must match `^[a-zA-Z_][a-zA-Z0-9_]*$`. Cannot collide
with built-in tool names (`cypher_query` / `graph_overview` etc.).

**`$param` substitution** — Cypher templates pass through to
`graph.cypher(query, params=args)` unchanged. The kglite Cypher
engine does typed parameter binding (no string interpolation) — the
JSON value of `args[$name]` becomes a typed value at the
`MATCH (n {field: $name})` site, so injection is impossible by
construction. The agent supplies values per the JSON Schema; kglite
binds them in-engine.

**JSON Schema flavour** — `parameters:` accepts the subset of
JSON Schema (draft 2020-12) that the MCP SDK supports for tool
input. Practically: `type`, `properties`, `required`, `default`,
`description`, `enum`, `items` (for arrays), `minimum`/`maximum`,
`minLength`/`maxLength`, `pattern`. Nested objects work; bring the
schema's complexity in proportion to the tool's parameter
complexity.

**Parameter validation** — the MCP client enforces schema validation
before the tool is dispatched. A type mismatch (string supplied for
an `integer` field) raises an MCP-level error before
`graph.cypher()` runs; the agent receives a structured tool error
rather than a Cypher error.

**Tool errors** — if `graph.cypher()` raises, the response body is
`Cypher error: <engine message>` (the same envelope as
`cypher_query`). Empty result sets render as `No results.`.

**`FORMAT CSV` inheritance** — manifest cypher tools share the
formatting path with `cypher_query`. Append `FORMAT CSV` inside the
template (or `$_csv_format` if you want to gate it on a parameter),
and the tool's output follows the same inline-vs-URL behaviour
documented under "Tool response formats."

**Boot-time validation** — every `$param` named in the template
must appear in `parameters.properties`. Mismatch fails at boot,
not at agent call time.

Worked examples — see `docs/examples/manifest_cypher_tool.md`
through `docs/examples/manifest_wikidata.md`.

### Embedder backend × model catalog

`extensions.embedder.backend: fastembed` is currently the only
supported backend. The `model:` string routes internally to one of
two implementations depending on what's in fastembed-python's
catalog:

| Model name | Dimension | Internal backend |
|---|:---:|---|
| `BAAI/bge-m3` | 1024 | Direct ONNX wrapper (`kglite.mcp_server.bge_m3.BgeM3Embedder`) — fastembed-python's catalog doesn't carry bge-m3, so we go through a hand-written ONNX inference path with HuggingFace Hub downloads. |
| `BAAI/bge-small-en-v1.5`, `bge-small-en-v1.5` | 384 | fastembed-python (`fastembed.TextEmbedding`). |
| `BAAI/bge-base-en-v1.5`, `bge-base-en-v1.5` | 768 | fastembed-python. |
| `BAAI/bge-large-en-v1.5`, `bge-large-en-v1.5` | 1024 | fastembed-python. |
| `sentence-transformers/all-MiniLM-L6-v2`, `all-MiniLM-L6-v2` | 384 | fastembed-python. |
| `intfloat/multilingual-e5-large`, `multilingual-e5-large` | 1024 | fastembed-python. |
| `intfloat/multilingual-e5-base`, `multilingual-e5-base` | 768 | fastembed-python. |

Both internal backends cache ONNX weights at `~/.cache/fastembed/`
(or `FASTEMBED_CACHE_PATH` if set). First call downloads; subsequent
calls re-use the cache.

Adding a new model: open an issue with the HuggingFace identifier
and target dimension. fastembed-python adds models periodically, in
which case the kglite-side change is a one-line addition to
`kglite/mcp_server/embedder.py::KNOWN_MODELS`.

### Path resolution and manifest discovery

**Relative paths in manifests resolve against the manifest's own
directory.** This applies to `source_root`, `env_file`, every
entry in `source_roots`, `workspace.root`,
`extensions.csv_http_server.dir`, and
`extensions.cypher_preprocessor.module`. The rule is unconditional:
no path is interpreted relative to `cwd` unless explicitly
absolute.

Manifest discovery order:

1. `--mcp-config <path>` — explicit path; absolute or
   resolved against cwd.
2. `--graph X.kgl` — auto-detects `<dirname>/<basename>_mcp.yaml`
   next to the graph file (the "sibling" pattern).
3. `--workspace DIR` / `--watch DIR` — auto-detects
   `DIR/workspace_mcp.yaml`.
4. `--source-root` / bare — no auto-detection. Pass `--mcp-config`
   explicitly if you want a manifest.

`.env` discovery order:

1. Manifest `env_file: <path>` — explicit; absolute or relative to
   manifest dir.
2. Otherwise walks upward from the mode path (or cwd in bare mode)
   looking for a `.env` file. Loads the first one found.

Existing process env vars are never overwritten by `.env` —
`GITHUB_TOKEN=...` in your shell wins over the file.

### Operator notes

#### PyPI simple-index lag after publish

After a `kglite` release publishes to PyPI, the `simple/` index
that `pip install` consults can lag the JSON metadata by ~few
minutes. The first `pip install kglite==X.Y.Z` after publish may
return `No matching distribution found`. Workaround:

```bash
pip install --index-url https://pypi.org/simple/ --no-cache-dir 'kglite[mcp]==X.Y.Z'
```

The `--index-url` forces a direct fetch (some mirrors cache
longer); `--no-cache-dir` bypasses pip's local cache. Wait a few
minutes if you'd rather not pass flags — the lag is consistent.

This is a PyPI / mirror-cache behaviour, not a kglite packaging
problem.

#### Conda + multiple Pythons

`pip install kglite` against a conda env's Python (`conda
activate myenv && pip install kglite`) Just Works post-0.9.20 —
no `PYO3_PYTHON=`, no `install_name_tool` patching. If your shell
PATH lifts an older `kglite-mcp-server` from a different env, run
`which kglite-mcp-server` to confirm which install the script points
at.

#### Watch mode rebuild costs

`workspace.watch: true` + `--watch DIR` rebuilds the code-tree graph
on every debounced file change (500 ms default debounce). For source
trees over 100k LoC this costs a few seconds per rebuild. The
rebuild runs on a background thread; queries against the previous
graph keep working until the new graph atomically swaps in.

## Migration: 0.9.19 → 0.9.20

`kglite-mcp-server` is now a Python entry point instead of a bundled
Rust binary. Operator action:

```bash
pip install --upgrade 'kglite[mcp]'
```

YAMLs unchanged. Tool surface unchanged. fastembed cache directory
unchanged (`~/.cache/fastembed/`). Performance unchanged (kglite's
Python `cypher()` releases the GIL for execution, so the wrapping
layer is sub-microsecond).

What disappeared:
- `kglite/_bin/kglite-mcp-server` binary inside the wheel (no
  longer built).
- `install_name_tool` / `patchelf` / mold / per-Python-version
  wheel matrix in CI (no longer needed).
- The 0.9.18 conda install_name regression (impossible by
  construction — there's no binary to mis-link).

Wheel matrix is back to 3 abi3 wheels per release, same as pre-0.9.18.

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

## Worked examples

End-to-end manifest snippets, each focused on one feature:

```{toctree}
:maxdepth: 1

../examples/manifest_cypher_tool
../examples/manifest_with_embedder
../examples/manifest_workspace
../examples/manifest_cypher_preprocessor
```
