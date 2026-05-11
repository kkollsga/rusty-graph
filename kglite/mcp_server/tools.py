"""Built-in MCP tools: cypher_query, graph_overview, save_graph.

Mirrors `crates/kglite-mcp-server/src/tools.rs` with the same handler
shapes, status messages, and 15-row inline cap. Differences:

- We call `kglite.KnowledgeGraph.cypher(...)`/`.describe(...)`/`.save(...)`
  directly (kglite Python API). The GIL is released inside `cypher()`
  for execution, so the wrapping overhead is just Python attr-lookup
  and argument marshalling — sub-microsecond per call.
- Mutating queries route to `graph.cypher(...)` too (the Python API
  handles read/write internally, unlike the Rust shim which had to
  branch on `is_mutation_query`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

import kglite
from kglite.mcp_server.csv_http import CsvHttpConfig, write_csv

NO_GRAPH = "No active graph. Pass --graph X.kgl, or activate one via repo_management('org/repo')."


@dataclass
class ActiveGraph:
    """The kglite graph + its source path (for save_graph)."""

    graph: kglite.KnowledgeGraph
    source_path: Path | None


class GraphState:
    """Shared active-graph slot, swappable at runtime (workspace mode
    re-activates the slot when the operator switches repos). The lock
    serialises swap + read access; tool calls hold the read view for
    the duration of the call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: ActiveGraph | None = None

    def load_kgl(self, path: Path) -> None:
        graph = kglite.load(str(path))
        with self._lock:
            self._active = ActiveGraph(graph=graph, source_path=path)

    def build_code_tree(self, dir_path: Path) -> None:
        graph = kglite.code_tree.build(str(dir_path))
        with self._lock:
            self._active = ActiveGraph(graph=graph, source_path=None)

    def bind_embedder(self, embedder: Any) -> None:
        """Bind an embedder instance to the active graph. No-op if no
        active graph yet (workspace mode loads the graph later)."""
        with self._lock:
            if self._active is None:
                return
            self._active.graph.set_embedder(embedder)

    def active(self) -> ActiveGraph | None:
        with self._lock:
            return self._active

    def schema(self) -> tuple[int, int] | None:
        a = self.active()
        if a is None:
            return None
        s = a.graph.schema()
        return (s["node_count"], s["edge_count"])


def run_cypher(
    state: GraphState,
    query: str,
    params: dict[str, Any] | None = None,
    csv_http: CsvHttpConfig | None = None,
) -> str:
    """Run a Cypher query against the active graph. Returns the
    rendered tool body — inline 15-row preview by default, CSV (or URL
    when csv_http is configured) when `FORMAT CSV` appears in the
    query."""
    active = state.active()
    if active is None:
        return NO_GRAPH

    try:
        result = active.graph.cypher(query, params=params or {})
    except Exception as e:  # noqa: BLE001 — surface the engine's error verbatim
        return f"Cypher error: {e}"

    # `FORMAT CSV` is recognised inside kglite's cypher() — when set,
    # the call returns a CSV string instead of a ResultView. Use that
    # signal to route URL-vs-inline.
    if isinstance(result, str):
        if csv_http is not None:
            try:
                name = write_csv(csv_http, result)
            except Exception as e:  # noqa: BLE001
                # Best-effort: fall back to inline CSV on write failure.
                return result + f"\n# (csv_http write failed: {e}; CSV inlined)"
            url = csv_http.url_for(name)
            row_count = _count_csv_rows(result)
            return f"FORMAT CSV: {row_count} row(s) written to {url}\nFetch with: curl {url}"
        return result

    return _format_inline(result)


def _count_csv_rows(csv: str) -> int:
    # Strip trailing newline so we don't double-count an empty final line.
    lines = csv.rstrip("\n").splitlines()
    return max(0, len(lines) - 1)  # minus header


def _format_inline(result: Any) -> str:
    """Render a kglite ResultView as a 15-row inline preview. Matches
    the Rust shim's format."""
    rows = list(result)
    total = len(rows)
    if total == 0:
        return "No results."
    columns = result.columns
    header = f"{total} row(s) (showing first 15):\n" if total > 15 else f"{total} row(s):\n"
    out = [header, "\t".join(columns), "\n"]
    for row in rows[:15]:
        # ResultView rows are tuples of values; render each as repr-ish
        # text, tab-separated. Matches the Rust shim's push_value_repr.
        out.append("\t".join(_repr(v) for v in row))
        out.append("\n")
    return "".join(out)


def _repr(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    return str(v)


def run_overview(
    state: GraphState,
    types: list[str] | None,
    connections: Any,
    cypher: Any,
    temp_cleanup_dir: Path | None,
) -> str:
    active = state.active()
    if active is None:
        return NO_GRAPH

    # Wipe the temp dir on bare overview when temp_cleanup is on. "Bare"
    # = no drill-down args. Matches the Rust shim's P4 behaviour.
    if temp_cleanup_dir is not None and types is None and connections is None and cypher is None:
        _wipe_dir(temp_cleanup_dir)

    kwargs: dict[str, Any] = {}
    if types is not None:
        kwargs["types"] = types
    if connections is not None:
        kwargs["connections"] = connections
    if cypher is not None:
        kwargs["cypher"] = cypher
    try:
        return active.graph.describe(**kwargs)
    except Exception as e:  # noqa: BLE001
        return f"graph_overview error: {e}"


def run_save(state: GraphState) -> str:
    active = state.active()
    if active is None:
        return NO_GRAPH
    if active.source_path is None:
        return "save_graph requires --graph mode (no source path bound)."
    try:
        active.graph.save(str(active.source_path))
    except Exception as e:  # noqa: BLE001
        return f"save_graph error: {e}"
    s = state.schema()
    if s is None:
        return f"Saved {active.source_path}."
    nodes, edges = s
    return f"Saved {active.source_path} ({nodes} nodes, {edges} edges)."


def _wipe_dir(dir_path: Path) -> None:
    """Best-effort directory wipe — log on errors, never raise. Matches
    the Rust shim's `wipe_temp_dir` semantics."""
    import logging
    import shutil

    log = logging.getLogger("kglite.mcp_server.tools")
    if not dir_path.is_dir():
        log.debug("temp_cleanup: directory does not exist: %s", dir_path)
        return
    wiped = 0
    for entry in dir_path.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            wiped += 1
        except OSError as e:
            log.debug("temp_cleanup: remove %s failed: %s", entry, e)
    if wiped > 0:
        log.info("temp_cleanup: wiped %d entries from %s", wiped, dir_path)
