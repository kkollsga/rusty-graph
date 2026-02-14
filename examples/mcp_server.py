#!/usr/bin/env python3
"""
MCP server exposing a KGLite knowledge graph over stdio.

Loads a .kgl graph file and exposes Cypher query access for
Claude Code / Claude Desktop via the Model Context Protocol.

Tools:
    graph_overview  — schema, Cypher reference, and example queries
    cypher_query    — run any Cypher query (including text_score() for semantic search)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
