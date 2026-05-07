#!/usr/bin/env python3
"""Example MCP server entry point — fork this if you want to register
your own tools alongside the bundled ``graph_overview`` and
``cypher_query``.

The same server is shipped as a console script with the package::

    pip install "kglite[mcp]"
    kglite-mcp-server --graph my_graph.kgl

This file just re-exports :func:`kglite.mcp_server.main` so that
existing ``python examples/mcp_server.py --graph X`` invocations and
Claude Desktop configs that point at this path keep working. To
customise tool surface, copy this file and add ``@mcp.tool()``
definitions before ``mcp.run()``.
"""

import sys

from kglite.mcp_server import main

if __name__ == "__main__":
    sys.exit(main())
