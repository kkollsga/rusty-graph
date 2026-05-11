"""kglite MCP server — Python implementation.

0.9.20: the bundled Rust binary `kglite-mcp-server` was retired in
favour of this Python entry point, talking the MCP stdio protocol via
the official `mcp` SDK and calling the existing kglite Python API
(`graph.cypher` / `graph.describe` / `graph.source`) for all heavy
lifting. Cypher execution still happens in pure Rust under the GIL
release inside `kglite.KnowledgeGraph.cypher()`, so the hot-path
performance is identical to the prior bundled-binary release.

Install: `pip install 'kglite[mcp]'`. Run: `kglite-mcp-server --help`.
"""
