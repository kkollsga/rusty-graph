"""CLI entry point for ``kglite-mcp-server``.

Tools registered with the MCP host:

- ``graph_overview``: progressive schema disclosure (types, connections,
  Cypher reference). Wraps ``KnowledgeGraph.describe()``.
- ``cypher_query``: runs any Cypher query against the loaded graph.
  Returns up to 15 rows inline; append ``FORMAT CSV`` for a full
  on-disk export (served over a localhost HTTP endpoint with CORS).

Everything beyond these two — name lookup, code-entity navigation,
structural validators — is callable from Cypher itself, so the agent
discovers the surface via ``graph_overview(cypher=True)``.

For a customisable starting point that you can edit (e.g. registering
project-specific tools), see ``examples/mcp_server.py`` in the
repository — that file is a thin wrapper around :func:`main` and the
canonical place to fork.

Claude Desktop config::

    {
      "mcpServers": {
        "my-graph": {
          "command": "kglite-mcp-server",
          "args": ["--graph", "/abs/path/to/graph.kgl"]
        }
      }
    }
"""

from __future__ import annotations

import argparse
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import itertools
from pathlib import Path
import sys
import threading

import kglite


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kglite-mcp-server",
        description="Serve a KGLite .kgl graph file over MCP (stdio).",
    )
    parser.add_argument("--graph", default="graph.kgl", help="Path to .kgl file (default: graph.kgl)")
    parser.add_argument(
        "--embedder",
        default=None,
        help="sentence-transformers model name to enable text_score(); optional",
    )
    parser.add_argument("--name", default="KGLite Graph", help="Server display name")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_arg_parser().parse_args(argv)

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print(
            "ERROR: the `mcp` package is not installed. Reinstall with the\n"
            "       MCP extra:    pip install 'kglite[mcp]'",
            file=sys.stderr,
        )
        return 2

    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found.", file=sys.stderr)
        return 1

    temp_dir = graph_path.parent / "temp"
    graph = kglite.load(str(graph_path))

    if args.embedder:
        from sentence_transformers import SentenceTransformer

        class _Embedder:
            def __init__(self, model_name: str) -> None:
                self._model = SentenceTransformer(model_name)
                self.dimension = self._model.get_sentence_embedding_dimension()

            def embed(self, texts: list[str]) -> list[list[float]]:
                return self._model.encode(texts, show_progress_bar=False).tolist()

            def load(self) -> None:
                pass

            def unload(self) -> None:
                pass

        graph.set_embedder(_Embedder(args.embedder))

    # ── localhost CSV server (lazy) ───────────────────────────────────────
    file_server_state: dict[str, int | bool] = {"port": 0, "csv_hint_shown": False}

    def _ensure_file_server() -> int:
        if file_server_state["port"]:
            return file_server_state["port"]  # type: ignore[return-value]
        temp_dir.mkdir(exist_ok=True)

        class _CORSHandler(SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(temp_dir), **kw)

            def end_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                super().end_headers()

        server = HTTPServer(("127.0.0.1", 0), _CORSHandler)
        file_server_state["port"] = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return file_server_state["port"]  # type: ignore[return-value]

    # ── MCP server ────────────────────────────────────────────────────────
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
        built-in default of 180_000 ms (3 min). Pass 0 to disable the
        deadline entirely for this call — use sparingly, typically only
        after describe()/EXPLAIN confirmed the plan is anchored on an index."""
        try:
            result = graph.cypher(query, timeout_ms=timeout_ms)
            if isinstance(result, str):  # FORMAT CSV returns a string
                temp_dir.mkdir(exist_ok=True)
                filename = f"data-{datetime.now():%Y%m%d-%H%M%S}.csv"
                (temp_dir / filename).write_text(result)
                port = _ensure_file_server()
                url = f"http://localhost:{port}/{filename}"
                rel_path = f"./temp/{filename}"
                if not file_server_state["csv_hint_shown"]:
                    file_server_state["csv_hint_shown"] = True
                    return (
                        f"CSV exported: {url}\n\n"
                        f"⚠️ DO NOT call read_source to load this data into context.\n"
                        f"The CSV URL is served locally with CORS enabled — use fetch() in a generated\n"
                        f"HTML file to load it at runtime. Do not read the data and hardcode it.\n"
                        f"read_source is only for inspecting column names or small row slices.\n\n"
                        f"read_source(file_path='{rel_path}') — add rows=[0,4] to inspect structure only."
                    )
                return (
                    f"CSV exported: {url}\n"
                    f"read_source(file_path='{rel_path}') — add rows=[0,4] to inspect structure only."
                )
            if len(result) == 0:
                return "No results."
            rows = [str(row) for row in itertools.islice(result, 15)]
            header = f"{len(result)} row(s)"
            if len(result) > 15:
                header += " (showing first 15)"
            return header + ":\n" + "\n".join(rows)
        except Exception as e:
            return f"Cypher error: {e}"

    mcp.run(transport="stdio")
    return 0
