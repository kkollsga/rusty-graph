"""YAML-declared `tools[].cypher` registration.

Mirrors `crates/kglite-mcp-server/src/cypher_tools.rs`. Each manifest
entry like

    tools:
      - name: session_detail
        description: Full source JSON for a session by id.
        cypher: |
          MATCH (s:Session {id: $session_id}) RETURN s
        parameters:
          type: object
          properties:
            session_id: { type: string }
          required: [session_id]

becomes an MCP tool that substitutes `$session_id` in the template
and dispatches to kglite's `graph.cypher(template, params=args)`. The
schema published with the tool is taken from `parameters:` verbatim
when present (must be a JSON-Schema object), otherwise a permissive
empty-object schema.
"""

from __future__ import annotations

from typing import Any

from mcp import types as mcp_types

from kglite.mcp_server.csv_http import CsvHttpConfig
from kglite.mcp_server.manifest import CypherTool
from kglite.mcp_server.tools import GraphState, run_cypher


def build_tool_attrs(tools: list[CypherTool]) -> list[mcp_types.Tool]:
    """Return mcp.Tool attrs for each manifest cypher tool."""
    attrs: list[mcp_types.Tool] = []
    for spec in tools:
        schema = spec.parameters if spec.parameters else {"type": "object", "properties": {}}
        attrs.append(
            mcp_types.Tool(
                name=spec.name,
                description=spec.description or f"Manifest-declared Cypher tool {spec.name!r}.",
                inputSchema=schema,
            )
        )
    return attrs


async def call_cypher_tool(
    spec: CypherTool,
    state: GraphState,
    args: dict[str, Any],
    csv_http: CsvHttpConfig | None,
) -> str:
    """Execute a manifest cypher template with arg substitution. Errors
    bubble up as text bodies (consistent with the Rust shim)."""
    return run_cypher(state, spec.cypher, params=args, csv_http=csv_http)
