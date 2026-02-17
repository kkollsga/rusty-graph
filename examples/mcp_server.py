#!/usr/bin/env python3
"""
MCP server exposing a KGLite knowledge graph over stdio.

Loads a .kgl graph file and exposes Cypher query access for
Claude Code / Claude Desktop via the Model Context Protocol.

Tools:
    graph_overview  — schema, Cypher reference, and example queries
    cypher_query    — run any Cypher query (including text_score() for semantic search)
    find_entity     — search code entities by name across all types
    read_source     — resolve entities to source locations (file, line range, line count)
    entity_context  — get full neighborhood of a code entity

Usage:
    python mcp_server.py                          # uses default graph.kgl
    python mcp_server.py --graph my_data.kgl      # custom graph file

Claude Desktop config (~/.claude/claude_desktop_config.json):
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

# ---------------------------------------------------------------------------
# Generic embedder wrapper
# ---------------------------------------------------------------------------

class EmbedderModel:
    """
    Generic wrapper that turns any sentence-transformers model into a
    KGLite-compatible embedder (requires .dimension and .embed()).

    Usage:
        embedder = EmbedderModel("all-MiniLM-L6-v2")
        graph.set_embedder(embedder)
        graph.embed_texts("Article", "summary")

    You can also subclass this and override embed() to use any embedding
    backend (OpenAI, Cohere, local ONNX, etc.).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, **kwargs)
        self.dimension: int = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return a list of float vectors."""
        return self._model.encode(texts, show_progress_bar=False).tolist()


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="KGLite MCP Server")
    parser.add_argument(
        "--graph",
        type=str,
        default="graph.kgl",
        help="Path to the .kgl graph file (default: graph.kgl)",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=None,
        help="Sentence-transformers model name to enable semantic search (optional)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="KGLite Graph",
        help="Display name for the MCP server (default: 'KGLite Graph')",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

args = parse_args()
graph_path = Path(args.graph)

if not graph_path.exists():
    print(f"ERROR: {graph_path} not found.", file=sys.stderr)
    sys.exit(1)

graph = kglite.load(str(graph_path))

# Optional: register an embedder for text_score() support in Cypher
if args.embedder:
    graph.set_embedder(EmbedderModel(args.embedder))

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

schema_info = graph.schema()
node_count = schema_info.get("node_count", "?")
edge_count = schema_info.get("edge_count", "?")

mcp = FastMCP(
    args.name,
    instructions=(
        f"Knowledge graph with {node_count} nodes and {edge_count} edges. "
        "Call graph_overview first to learn the schema and Cypher reference, "
        "then use cypher_query to run queries."
    ),
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def graph_overview() -> str:
    """Get the full schema of the knowledge graph — node types with
    properties, edge types, indexes, Cypher reference, and example queries.
    Call this first to understand what you can query."""
    return graph.agent_describe()


@mcp.tool()
def cypher_query(query: str) -> str:
    """Run a Cypher query against the knowledge graph. Supports MATCH, WHERE,
    RETURN, ORDER BY, LIMIT, aggregations, path traversals, CREATE, SET,
    DELETE, and more. Call graph_overview first if you need the schema.
    Returns up to 200 rows as formatted text."""
    try:
        result = graph.cypher(query)
        if len(result) == 0:
            return "Query returned no results."
        rows = []
        for row in result:
            rows.append(str(dict(row)))
            if len(rows) >= 200:
                break
        header = f"Returned {len(result)} row(s)"
        if len(result) > 200:
            header += " (showing first 200)"
        return header + ":\n" + "\n".join(rows)
    except Exception as e:
        return f"Cypher query error: {e}"


@mcp.tool()
def find_entity(name: str, node_type: str | None = None) -> str:
    """Search code entities by name across all types (Function, Struct, Class,
    Enum, Trait, etc.). Returns matching entities with qualified_name, file_path,
    and line_number for disambiguation. Use qualified_name with read_source or
    entity_context for exact lookups."""
    try:
        results = graph.find(name, node_type=node_type)
        if not results:
            return f"No code entities found matching '{name}'."
        lines = [f"Found {len(results)} match(es) for '{name}':"]
        for r in results:
            qn = r.get("qualified_name", r.get("id", "?"))
            fp = r.get("file_path", "?")
            ln = r.get("line_number", "?")
            lines.append(f"  {r.get('type', '?')}: {qn}  ({fp}:{ln})")
        return "\n".join(lines)
    except Exception as e:
        return f"find_entity error: {e}"


@mcp.tool()
def read_source(names: list[str], node_type: str | None = None) -> str:
    """Resolve one or more code entity names to their source locations.
    Returns file_path, line_number, end_line, and line_count for each.
    Accepts names or qualified_names. Use find_entity first if a name
    is ambiguous."""
    try:
        results = graph.source(names, node_type=node_type)
        lines = []
        for r in results:
            name = r.get("name", "?")
            if r.get("error"):
                lines.append(f"{name}: {r['error']}")
            elif r.get("ambiguous"):
                matches = r.get("matches", [])
                lines.append(f"{name}: ambiguous ({len(matches)} matches) — use find_entity to disambiguate")
            else:
                fp = r.get("file_path", "?")
                ln = r.get("line_number", "?")
                el = r.get("end_line", "?")
                lc = r.get("line_count", "?")
                qn = r.get("qualified_name", "")
                sig = r.get("signature", "")
                lines.append(f"{r.get('type', '?')}: {qn}")
                lines.append(f"  file: {fp}:{ln}-{el} ({lc} lines)")
                if sig:
                    lines.append(f"  signature: {sig}")
        return "\n".join(lines)
    except Exception as e:
        return f"read_source error: {e}"


@mcp.tool()
def entity_context(name: str, node_type: str | None = None, hops: int = 1) -> str:
    """Get the full neighborhood of a code entity — its properties and all
    relationships grouped by type (HAS_METHOD, CALLS, called_by, USES_TYPE,
    etc.). Accepts a name or qualified_name. Set hops > 1 for multi-hop
    expansion."""
    try:
        import json
        ctx = graph.context(name, node_type=node_type, hops=hops)
        if ctx.get("error"):
            return ctx["error"]
        if ctx.get("ambiguous"):
            matches = ctx.get("matches", [])
            lines = [f"Ambiguous name '{name}' — {len(matches)} matches:"]
            for m in matches:
                qn = m.get("qualified_name", m.get("id", "?"))
                lines.append(f"  {m.get('type', '?')}: {qn}")
            lines.append("Use a qualified_name for an exact match.")
            return "\n".join(lines)
        return json.dumps(ctx, indent=2, default=str)
    except Exception as e:
        return f"entity_context error: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
