#!/usr/bin/env python3
"""MCP server exposing a KGLite knowledge graph to Claude.

Loads any .kgl graph file and provides Cypher query access via MCP (stdio).
Works with graphs built by legal_graph.py, code_graph.py, spatial_graph.py,
or any other KGLite graph.

Tools:
    graph_overview  — progressive schema disclosure (types, connections, Cypher ref)
    cypher_query    — run any Cypher query (returns up to 200 rows)
    bug_report      — file a timestamped Cypher bug report

Code graph tools (auto-enabled when the graph has Function/Class nodes):
    find_entity     — search code entities by name
    read_source     — resolve entities to source file locations
    entity_context  — get neighborhood of a code entity

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
import sys
from pathlib import Path

import kglite
from mcp.server.fastmcp import FastMCP

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
) -> str:
    """Get graph schema, connection details, or Cypher language reference.

    Three independent axes — call with no args first for the overview:
      graph_overview()                            — inventory of node types
      graph_overview(types=["Type"])              — property schemas, samples
      graph_overview(connections=True)            — all connection types
      graph_overview(connections=["CITES"])        — deep-dive with properties
      graph_overview(cypher=True)                 — Cypher reference
      graph_overview(cypher=["temporal","MATCH"])  — detailed docs with examples
    """
    try:
        return graph.describe(types=types, connections=connections, cypher=cypher)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def cypher_query(query: str) -> str:
    """Run a Cypher query against the knowledge graph. Returns up to 200 rows.

    Supports MATCH, WHERE, RETURN, ORDER BY, LIMIT, WITH, OPTIONAL MATCH,
    UNWIND, UNION, CREATE, SET, DELETE, MERGE, aggregations, path patterns,
    CALL procedures (pagerank, louvain, etc.), and spatial/temporal functions.
    Call graph_overview() first if you need the schema."""
    try:
        result = graph.cypher(query)
        if len(result) == 0:
            return "No results."
        rows = [str(dict(row)) for row in result[:200]]
        header = f"{len(result)} row(s)"
        if len(result) > 200:
            header += " (showing first 200)"
        return header + ":\n" + "\n".join(rows)
    except Exception as e:
        return f"Cypher error: {e}"


@mcp.tool()
def bug_report(query: str, result: str, expected: str, description: str) -> str:
    """File a Cypher bug report to reported_bugs.md."""
    try:
        return graph.bug_report(query, result, expected, description)
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
                    lines.append(f"  {r.get('file_path', '?')}:{r.get('line_number', '?')}-{r.get('end_line', '?')}{sig}")
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
