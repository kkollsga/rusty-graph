"""MCP server exposing a KGLite knowledge graph to an LLM.

Run with the bundled CLI::

    pip install "kglite[mcp]"
    kglite-mcp-server --graph my_graph.kgl

See :mod:`kglite.mcp_server.cli` for the entry point and
:mod:`kglite.mcp_server.source_access` for the source-file tool shims
backed by the ``mcp-methods`` package.
"""

from __future__ import annotations

from kglite.mcp_server.cli import main

__all__ = ["main"]
