"""Built-in ``graph_overview`` and ``cypher_query`` MCP tools.

Both single-graph and workspace modes register these. The graph is
accessed through a provider callable so the workspace mode can swap
the active graph at runtime without re-registering tools.

CSV exports go to a per-server ``temp/`` directory served over a
lazy-launched localhost HTTP endpoint with CORS. Agents fetch the
URL rather than reading the CSV into context.
"""

from __future__ import annotations

from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import itertools
from pathlib import Path
import threading
from typing import Any, Callable

GraphProvider = Callable[[], Any]
PreOverviewHook = Callable[[], None]


def register_core_tools(
    mcp,
    *,
    graph_provider: GraphProvider,
    temp_dir: Path,
    overview_prefix: str | None = None,
    pre_overview_hook: PreOverviewHook | None = None,
    no_graph_message: str = (
        "No graph is currently loaded. In workspace mode, call repo_management('org/repo') to activate one."
    ),
) -> None:
    """Register graph_overview and cypher_query tools on ``mcp``.

    ``graph_provider`` returns the active :class:`KnowledgeGraph` or
    ``None``. The tools surface a friendly message in the latter case
    rather than crashing.

    ``overview_prefix`` is prepended to ``graph.describe()`` output on
    bare ``graph_overview()`` calls (no args). ``pre_overview_hook`` is
    invoked first — typically to clear the CSV temp dir.
    """
    file_server_state: dict[str, int | bool] = {"port": 0, "csv_hint_shown": False}

    def _ensure_file_server() -> int:
        if file_server_state["port"]:
            return file_server_state["port"]  # type: ignore[return-value]
        temp_dir.mkdir(parents=True, exist_ok=True)
        directory = str(temp_dir)

        class _CORSHandler(SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=directory, **kw)

            def end_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                super().end_headers()

        server = HTTPServer(("127.0.0.1", 0), _CORSHandler)
        file_server_state["port"] = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return file_server_state["port"]  # type: ignore[return-value]

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

        ``max_pairs`` caps the (src_type, tgt_type) breakdown in
        connections=[...] deep-dives (default 50). Raise it when a connection
        type has many distinct endpoint pairs and you want the full list.
        """
        graph = graph_provider()
        if graph is None:
            return no_graph_message
        is_bare = not (types or connections or cypher)
        if is_bare and pre_overview_hook is not None:
            try:
                pre_overview_hook()
            except Exception:
                pass
        try:
            described = graph.describe(types=types, connections=connections, cypher=cypher, max_pairs=max_pairs)
        except Exception as e:
            return f"Error: {e}"
        if is_bare and overview_prefix:
            return overview_prefix.rstrip() + "\n\n" + described
        return described

    @mcp.tool()
    def cypher_query(query: str, timeout_ms: int | None = None) -> str:
        """Run a Cypher query against the knowledge graph. Returns up to 15 rows
        inline. Append FORMAT CSV to export full results to a CSV file (no row
        limit). Call graph_overview() first if you need the schema.

        timeout_ms: Deadline in milliseconds. None (default) uses the
        built-in default of 180_000 ms (3 min). Pass 0 to disable the
        deadline entirely for this call — use sparingly, typically only
        after describe()/EXPLAIN confirmed the plan is anchored on an index."""
        graph = graph_provider()
        if graph is None:
            return no_graph_message
        try:
            result = graph.cypher(query, timeout_ms=timeout_ms)
        except Exception as e:
            return f"Cypher error: {e}"

        if isinstance(result, str):  # FORMAT CSV returns a string
            temp_dir.mkdir(parents=True, exist_ok=True)
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
                f"CSV exported: {url}\nread_source(file_path='{rel_path}') — add rows=[0,4] to inspect structure only."
            )
        if len(result) == 0:
            return "No results."
        rows = [str(row) for row in itertools.islice(result, 15)]
        header = f"{len(result)} row(s)"
        if len(result) > 15:
            header += " (showing first 15)"
        return header + ":\n" + "\n".join(rows)
