"""Register manifest-declared cypher tools as MCP @tool() callables.

Each :class:`~kglite.mcp_server.manifest.CypherTool` becomes a closure
captured against the graph + the cypher template, with its
``inspect.Signature`` synthesised from the user-declared
``parameters:`` JSON Schema so FastMCP's introspection produces the
right input schema on the wire.

Validation runs at load time — any ``$param`` referenced in the
cypher must appear in ``parameters.properties``, and the schema
itself must be a valid JSON Schema. Errors fail server startup so
typos surface during boot, not on the first agent call.

Result formatting is intentionally minimal (15-row inline truncation,
no FORMAT CSV export). The bundled ``cypher_query`` tool is the path
to large CSV exports; manifest cypher tools are for small parameterised
lookups.
"""

from __future__ import annotations

import inspect
import itertools
import re

from kglite.mcp_server.manifest import CypherTool, ManifestError

_PARAM_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _collect_cypher_params(query: str) -> set[str]:
    """Return the set of ``$param`` names referenced in a cypher string."""
    return set(_PARAM_RE.findall(query))


def _validate_spec(spec: CypherTool) -> None:
    """Validate a CypherTool at load time. Raises :class:`ManifestError`."""
    if spec.parameters is not None:
        try:
            from jsonschema import Draft202012Validator

            Draft202012Validator.check_schema(spec.parameters)
        except Exception as e:
            raise ManifestError(f"cypher tool {spec.name!r}: invalid parameters schema: {e}") from e

    cypher_params = _collect_cypher_params(spec.cypher)
    declared = set((spec.parameters or {}).get("properties", {}).keys())
    missing = cypher_params - declared
    if missing:
        raise ManifestError(
            f"cypher tool {spec.name!r}: cypher references $params {sorted(missing)} "
            "that are not declared in parameters.properties"
        )


def _first_nonempty_line(s: str) -> str:
    for line in s.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _format_inline(result) -> str:
    """Format a Cypher result for inline return (15-row cap, no CSV export)."""
    if isinstance(result, str):
        # FORMAT CSV path — manifest cypher tools don't support file export;
        # truncate so we never dump huge CSVs into the agent context.
        if len(result) > 2000:
            return result[:2000] + "\n[... truncated; use cypher_query for FORMAT CSV exports]"
        return result
    n = len(result)
    if n == 0:
        return "No results."
    rows = [str(r) for r in itertools.islice(result, 15)]
    header = f"{n} row(s)" + (" (showing first 15)" if n > 15 else "")
    return header + ":\n" + "\n".join(rows)


def _build_callable(graph, spec: CypherTool):
    """Build a closure with a synthesised signature matching ``spec.parameters``."""
    schema = spec.parameters or {"type": "object", "properties": {}}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    cypher = spec.cypher

    def fn(**kwargs):
        try:
            result = graph.cypher(cypher, params=kwargs)
        except Exception as e:
            return f"Cypher error: {e}"
        return _format_inline(result)

    parameters: list[inspect.Parameter] = []
    annotations: dict = {}
    # Required-then-optional ordering so the synthesised signature is canonical.
    for name in [n for n in props if n in required] + [n for n in props if n not in required]:
        prop_schema = props[name]
        py_type = _TYPE_MAP.get(prop_schema.get("type"), str)
        annotations[name] = py_type
        if name in required:
            parameters.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=py_type))
        else:
            default = prop_schema.get("default")
            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=py_type,
                )
            )
    annotations["return"] = str
    fn.__name__ = spec.name
    fn.__qualname__ = spec.name
    fn.__doc__ = spec.description or _first_nonempty_line(cypher)
    fn.__signature__ = inspect.Signature(parameters=parameters, return_annotation=str)  # type: ignore[attr-defined]
    fn.__annotations__ = annotations
    return fn


def register_cypher_tools(mcp, graph, specs: list[CypherTool]) -> int:
    """Validate and register each spec on the MCP server. Returns count."""
    for spec in specs:
        _validate_spec(spec)
    for spec in specs:
        fn = _build_callable(graph, spec)
        mcp.tool()(fn)
    return len(specs)
