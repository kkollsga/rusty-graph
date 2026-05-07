#!/usr/bin/env python3
"""Example MCP server entry point — fork this only when the manifest
isn't enough.

For most customisation, drop a YAML manifest next to your graph
instead of forking::

    demo.kgl
    demo_mcp.yaml      ← source_root:, inline cypher tools, python hooks

See ``docs/guides/mcp-servers.md`` for the manifest reference.

The bundled CLI also ships as a console script::

    pip install "kglite[mcp]"
    kglite-mcp-server --graph my_graph.kgl

This file just re-exports :func:`kglite.mcp_server.main` so that
existing ``python examples/mcp_server.py --graph X`` invocations and
Claude Desktop configs that point at this path keep working. Fork it
when you need to replace the bundled tools, swap the transport, or
register custom MCP middleware — the manifest covers everything else.
"""

import sys

from kglite.mcp_server import main

if __name__ == "__main__":
    sys.exit(main())
