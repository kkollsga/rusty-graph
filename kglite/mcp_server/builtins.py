"""Pre-blessed builtin tools that don't need the ``--trust-tools`` gate.

These wrap small kglite-internal operations the CLI knows how to
perform safely without loading user code. Currently:

- ``save_graph`` — persist mutations made through Cypher (CREATE/SET/
  DELETE) back to the graph file. Common pattern for write workflows.
- ``temp_cleanup`` — wipe the CSV-export ``temp/`` directory on bare
  ``graph_overview()`` calls so it doesn't grow unbounded across long
  sessions.

Each builtin is opt-in via ``builtins:`` in the manifest. The CLI
constructs callables that close over the relevant CLI-level state
(graph, graph_path, temp_dir) and registers them only when the flag
is set.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable


def register_save_graph(mcp, graph, graph_path: Path) -> None:
    """Register a ``save_graph()`` tool that persists the graph to disk."""

    @mcp.tool()
    def save_graph() -> str:
        """Persist the in-memory graph to its source ``.kgl`` file.

        Call this after CREATE / SET / DELETE Cypher mutations to
        durably save the changes. No-op safe — the file is always
        rewritten in full.
        """
        try:
            graph.save(str(graph_path))
            schema = graph.schema()
            return f"Saved {graph_path} ({schema.get('node_count', '?')} nodes, {schema.get('edge_count', '?')} edges)."
        except Exception as e:
            return f"save_graph error: {e}"


def make_temp_cleanup_hook(temp_dir: Path, mode: str) -> Callable[[], None]:
    """Return a hook the CLI calls before running a bare ``graph_overview()``.

    ``mode`` is the manifest's ``builtins.temp_cleanup`` value. When
    ``"on_overview"`` the hook clears ``temp_dir`` (best-effort, swallows
    OSError so a busy-file doesn't break the overview call). Other
    values produce a no-op hook.
    """
    if mode == "on_overview":

        def _clear() -> None:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        return _clear

    def _noop() -> None:
        return None

    return _noop
