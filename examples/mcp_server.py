#!/usr/bin/env python3
"""MCP server exposing a KGLite knowledge graph to Claude.

Loads any .kgl graph file and provides Cypher query access via MCP (stdio).
Works with graphs built by legal_graph.py, code_graph.py, spatial_graph.py,
or any other KGLite graph.

Tools:
    graph_overview  — progressive schema disclosure (types, connections, Cypher ref)
    cypher_query    — run any Cypher query (up to 15 rows inline, FORMAT CSV for full export)
    search          — name-based node lookup (requires global label index)
    bug_report      — file a timestamped Cypher bug report

Code graph tools (auto-enabled when the graph has Function/Class nodes):
    find_entity     — search code entities by name
    read_source     — resolve entities to source file locations
    entity_context  — get neighborhood of a code entity

Optional rule-pack tooling (off by default — set ENABLE_RULE_PACKS=True
in this file to enable):
    rules_run       — execute a structural-validator rule pack
    plus a <rule_packs> block in graph_overview() advertising bundled packs.

Usage:
    python mcp_server.py --graph legal_graph.kgl
    python mcp_server.py --graph my_codebase.kgl --embedder all-MiniLM-L6-v2

Claude Desktop config:
    {
      "mcpServers": {
        "my-graph": {
          "command": "python",
          "args": ["/path/to/mcp_server.py", "--graph", "/path/to/graph.kgl"]
        }
      }
    }
"""

import argparse
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import sys
import threading

from mcp.server.fastmcp import FastMCP

import kglite

# -- Args & loading --------------------------------------------------------

parser = argparse.ArgumentParser(description="KGLite MCP Server")
parser.add_argument("--graph", default="graph.kgl", help="Path to .kgl file")
parser.add_argument("--embedder", default=None, help="sentence-transformers model (optional)")
parser.add_argument("--name", default="KGLite Graph", help="Server display name")
args = parser.parse_args()

graph_path = Path(args.graph)
if not graph_path.exists():
    print(f"ERROR: {graph_path} not found.", file=sys.stderr)
    sys.exit(1)

TEMP_DIR = graph_path.parent / "temp"

graph = kglite.load(str(graph_path))

if args.embedder:
    from sentence_transformers import SentenceTransformer

    class Embedder:
        def __init__(self, model_name):
            self._model = SentenceTransformer(model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()

        def embed(self, texts: list[str]) -> list[list[float]]:
            return self._model.encode(texts, show_progress_bar=False).tolist()

    graph.set_embedder(Embedder(args.embedder))

# -- CSV file server -------------------------------------------------------

_file_server_port = None
_csv_hint_shown = False


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


# -- Optional features -----------------------------------------------------
#
# Flip to ``True`` to expose KGLite's rule-pack tooling: a ``rules_run``
# MCP tool plus the ``<rule_packs>`` block in ``graph_overview()`` so
# agents discover available structural validators inline with the
# schema. Recommended for graphs where data integrity matters (legal,
# petroleum, ontology-quality work); leave off for code graphs and
# small social graphs where the rules add noise without value.
ENABLE_RULE_PACKS = False

if ENABLE_RULE_PACKS:
    import kglite.rules

    kglite.rules.advertise()


# -- MCP server ------------------------------------------------------------

schema = graph.schema()
mcp = FastMCP(
    args.name,
    instructions=(
        f"Knowledge graph with {schema['node_count']} nodes and {schema['edge_count']} edges. "
        "Call graph_overview() first to learn the schema, then cypher_query() to query."
    ),
)


@mcp.tool()
def graph_overview(
    types: list[str] | None = None,
    connections: bool | list[str] | None = None,
    cypher: bool | list[str] | None = None,
    max_pairs: int | None = None,
) -> str:
    """Get graph schema, connection details, or Cypher language reference.

    Three independent axes — call with no args first for the overview:
      graph_overview()                            — inventory + connections with property names
      graph_overview(types=["Type"])              — property schemas, samples
      graph_overview(connections=True)            — all connection types with properties
      graph_overview(connections=["CALLS"])       — deep-dive: property stats, sample edges
      graph_overview(cypher=True)                 — Cypher reference
      graph_overview(cypher=["temporal","MATCH"]) — detailed docs with examples

    `max_pairs` caps the (src_type, tgt_type) breakdown in
    connections=[...] deep-dives (default 50). Raise it when a connection
    type has many distinct endpoint pairs and you want the full list.
    """
    try:
        return graph.describe(types=types, connections=connections, cypher=cypher, max_pairs=max_pairs)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def cypher_query(query: str, timeout_ms: int | None = None) -> str:
    """Run a Cypher query against the knowledge graph. Returns up to 15 rows
    inline. Append FORMAT CSV to export full results to a CSV file (no row
    limit). Call graph_overview() first if you need the schema.

    timeout_ms: Deadline in milliseconds. None (default) uses the
    backend-aware default (Disk 10_000, Mapped 60_000, Memory none).
    Pass 0 to disable the deadline entirely for this call — use
    sparingly, typically only after describe()/EXPLAIN confirmed the
    plan is anchored on an index."""
    try:
        result = graph.cypher(query, timeout_ms=timeout_ms)
        if isinstance(result, str):  # FORMAT CSV returns a string
            TEMP_DIR.mkdir(exist_ok=True)
            filename = f"data-{datetime.now():%Y%m%d-%H%M%S}.csv"
            (TEMP_DIR / filename).write_text(result)
            port = _ensure_file_server()
            url = f"http://localhost:{port}/{filename}"
            rel_path = f"./temp/{filename}"
            global _csv_hint_shown
            if not _csv_hint_shown:
                _csv_hint_shown = True
                return (
                    f"CSV exported: {url}\n\n"
                    f"⚠️ DO NOT call read_source to load this data into context.\n"
                    f"The CSV URL is served locally with CORS enabled — use fetch() in a generated\n"
                    f"HTML file to load it at runtime. Do not read the data and hardcode it.\n"
                    f"read_source is only for inspecting column names or small row slices.\n\n"
                    f"read_source(file_path='{rel_path}') — add rows=[0,4] to inspect structure only."
                )
            return (
                f"CSV exported: {url}\nread_source(file_path='{rel_path}') — add rows=[0,4] to inspect structure only."
            )
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
def search(text: str, property: str = "label", limit: int = 10) -> str:
    """Find nodes whose `property` (default 'label') matches `text`.

    Tries exact match first, then prefix match. Returns up to `limit`
    hits with type, title, and id. Requires a global index — the graph
    author runs `graph.create_global_index('label')` once to build it.

    Use this when you have a human-readable name (e.g. 'Equinor',
    'Viking Age') but don't know which node type it belongs to. Skips
    the "guess the type" ceremony that MATCH (n:Type {...}) requires."""
    try:
        hits = graph.search(text, property=property, limit=limit)
        if not hits:
            return (
                f"No nodes matching '{text}' on property '{property}'. "
                "If this is unexpected, confirm a global index exists: "
                "graph.create_global_index('{property}')."
            )
        lines = [f"{len(hits)} hit(s):"]
        for h in hits:
            lines.append(f"  [{h['type']}] {h['title']}  id={h['id_value']}  (node={h['id']})")
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


@mcp.tool()
def bug_report(query: str, result: str, expected: str, description: str) -> str:
    """File a Cypher bug report to reported_bugs.md."""
    try:
        return graph.bug_report(query, result, expected, description)
    except Exception as e:
        return f"Error: {e}"


# -- Rule packs (gated by ENABLE_RULE_PACKS at top of file) ----------------

if ENABLE_RULE_PACKS:

    @mcp.tool()
    def rules_run(
        name: str = "structural_integrity",
        type: str | None = None,
        edge: str | None = None,
        only: list[str] | None = None,
        limit: int | None = None,
        timeout_ms: int | None = None,
    ) -> str:
        """Run a rule pack and return a per-rule violation summary.

        Rule packs are named structural validators (orphan nodes,
        missing-parent edges, cycles, duplicate titles, etc.) that
        surface data-integrity gaps not visible from normal queries.
        Available packs and per-rule parameter schemas appear in the
        ``<rule_packs>`` block of ``graph_overview()``.

        Most bundled rules take ``type`` (node label) and ``edge``
        (relationship type). Pass ``only=['rule_a']`` to run a subset,
        ``limit`` to override the default 1000-row cap, or
        ``timeout_ms`` to override the per-rule timeout (useful on
        large graphs where heavy rules need more than the graph's
        default Cypher budget).
        """
        try:
            kwargs: dict[str, object] = {}
            if type is not None:
                kwargs["type"] = type
            if edge is not None:
                kwargs["edge"] = edge
            report = graph.rules.run(name, only=only, limit=limit, timeout_ms=timeout_ms, **kwargs)
            s = report.summary
            lines = [
                f"Pack: {s['pack']} v{s['version']}",
                f"Total violations: {s['total_violations']:,} across {s['rule_count']} rule(s)",
            ]
            if s.get("any_truncated"):
                lines.append("⚠ One or more rules hit the LIMIT — re-run higher.")
            if s["rules_with_errors"]:
                lines.append(f"Rules with errors: {s['rules_with_errors']}")
            sev = s["by_severity"]
            sev_line = ", ".join(f"{k}: {v:,}" for k, v in sev.items() if v > 0)
            if sev_line:
                lines.append(f"By severity: {sev_line}")
            lines.append("")
            for rname, info in s["by_rule"].items():
                err = f" — error: {info['error']}" if info["error"] else ""
                trunc = " (truncated)" if info["truncated"] else ""
                lines.append(
                    f"  {rname} [{info['severity']}]: {info['violations']:,}{trunc} ({info['elapsed_ms']:.1f}ms){err}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"


# -- Code graph tools (auto-enabled for code_tree graphs) ------------------

CODE_TYPES = {"Function", "Class", "Struct", "Enum", "Module", "File"}
has_code = bool(CODE_TYPES & set(schema.get("node_types", {}).keys()))

if has_code:

    @mcp.tool()
    def find_entity(
        name: str,
        node_type: str | None = None,
        match_type: str | None = None,
    ) -> str:
        """Search code entities by name. Returns qualified_name, file_path, line.

        match_type: 'exact' (default), 'contains', or 'starts_with'."""
        try:
            results = graph.find(name, node_type=node_type, match_type=match_type)
            if not results:
                return f"No entities matching '{name}'."
            lines = [f"{len(results)} match(es):"]
            for r in results:
                qn = r.get("qualified_name", r.get("id", "?"))
                lines.append(f"  {r.get('type', '?')}: {qn}  ({r.get('file_path', '?')}:{r.get('line_number', '?')})")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def read_source(names: list[str], node_type: str | None = None) -> str:
        """Resolve code entity names to source file locations.
        Returns file_path, line range, and signature."""
        try:
            results = graph.source(names, node_type=node_type)
            lines = []
            for r in results:
                if r.get("error"):
                    lines.append(f"{r.get('name', '?')}: {r['error']}")
                elif r.get("ambiguous"):
                    lines.append(f"{r.get('name', '?')}: ambiguous — use find_entity to disambiguate")
                else:
                    sig = f"  {r['signature']}" if r.get("signature") else ""
                    lines.append(f"{r.get('type', '?')}: {r.get('qualified_name', '?')}")
                    lines.append(
                        f"  {r.get('file_path', '?')}:{r.get('line_number', '?')}-{r.get('end_line', '?')}{sig}"
                    )
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    def entity_context(name: str, node_type: str | None = None, hops: int = 1) -> str:
        """Get all relationships of a code entity (calls, callers, methods, types).
        Set hops > 1 for multi-hop expansion."""
        try:
            import json

            ctx = graph.context(name, node_type=node_type, hops=hops)
            if ctx.get("error"):
                return ctx["error"]
            if ctx.get("ambiguous"):
                matches = ctx.get("matches", [])
                return f"Ambiguous: {len(matches)} matches. Use a qualified_name."
            return json.dumps(ctx, indent=2, default=str)
        except Exception as e:
            return f"Error: {e}"


# -- Main ------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
