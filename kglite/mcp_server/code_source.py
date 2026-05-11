"""`read_code_source` tool — resolve qualified name → file slice.

Mirrors `crates/kglite-mcp-server/src/code_source.rs`. Calls
`graph.source(qualified_name, node_type=...)` (kglite Python API)
which returns a dict with `file_path` / `line_number` / `end_line`
fields. Then reads the corresponding file slice from the configured
source roots, with optional `start_line` / `end_line` / `grep` /
`grep_context` / `max_chars` filters.

The framework's Rust `mcp_methods::server::source::read_source`
helper isn't available here (we're not pulling the mcp-methods Python
package — operator explicitly asked us not to depend on it). The
slicing is straightforward Python: re-implemented inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from kglite.mcp_server.tools import NO_GRAPH, GraphState


@dataclass
class SourceLookup:
    file_path: str
    line_number: int
    end_line: int


def lookup(state: GraphState, qualified_name: str, node_type: str | None) -> SourceLookup | str:
    """Resolve a qualified name to a source location, or return a
    user-facing error string."""
    active = state.active()
    if active is None:
        return NO_GRAPH
    try:
        result = active.graph.source(qualified_name, node_type=node_type)
    except Exception as e:  # noqa: BLE001
        return f"graph.source({qualified_name!r}) failed: {e}"
    if isinstance(result, dict) and "error" in result:
        return str(result["error"])
    if isinstance(result, dict) and result.get("ambiguous"):
        matches = result.get("matches", [])
        return f"ambiguous qualified_name {qualified_name!r}; matches: {matches}. Pass `node_type` to narrow."
    if not isinstance(result, dict):
        return f"graph.source({qualified_name!r}) returned unexpected shape: {type(result).__name__}"
    file_path = result.get("file_path")
    if not file_path:
        return f"graph.source({qualified_name!r}) returned no file_path"
    line_number = int(result.get("line_number") or 1)
    end_line = int(result.get("end_line") or line_number)
    return SourceLookup(file_path=str(file_path), line_number=line_number, end_line=end_line)


def read(
    state: GraphState,
    args: dict[str, Any],
    source_roots: list[Path],
) -> str:
    qname = args.get("qualified_name")
    if not isinstance(qname, str) or not qname:
        return "read_code_source: missing required argument `qualified_name`."
    node_type = args.get("node_type")
    if node_type is not None and not isinstance(node_type, str):
        return "read_code_source: `node_type` must be a string if provided."

    lookup_result = lookup(state, qname, node_type)
    if isinstance(lookup_result, str):
        return f"read_code_source: {lookup_result}"

    if not source_roots:
        return (
            "read_code_source: no source roots configured. Pass "
            "`--source-root DIR`, `--workspace DIR`, or declare "
            "`source_root:` in the manifest."
        )

    start_line = args.get("start_line") or lookup_result.line_number
    end_line = args.get("end_line") or lookup_result.end_line
    grep_pattern = args.get("grep")
    grep_context = args.get("grep_context") or 2
    max_chars = args.get("max_chars")

    # Resolve the file against the source roots — first one that
    # contains the relative path wins. Mirrors the framework's
    # `read_source` semantics.
    resolved: Path | None = None
    rel = Path(lookup_result.file_path)
    for root in source_roots:
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            resolved = candidate
            break
    if resolved is None:
        # Try the file_path as-is (might already be absolute).
        absolute = Path(lookup_result.file_path)
        if absolute.is_absolute() and absolute.is_file():
            resolved = absolute
    if resolved is None:
        return f"read_code_source: source file not found in any configured root: {lookup_result.file_path}"

    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"read_code_source: read failed: {e}"

    lines = text.splitlines()
    start = max(1, int(start_line))
    end = min(len(lines), int(end_line))
    slice_lines = lines[start - 1 : end]

    if grep_pattern:
        try:
            rx = re.compile(grep_pattern)
        except re.error as e:
            return f"read_code_source: invalid grep pattern: {e}"
        kept: list[tuple[int, str]] = []
        for i, line in enumerate(slice_lines):
            if rx.search(line):
                lo = max(0, i - int(grep_context))
                hi = min(len(slice_lines), i + int(grep_context) + 1)
                for j in range(lo, hi):
                    kept.append((j, slice_lines[j]))
        # Dedup while preserving order.
        seen = set()
        body_lines = []
        for j, line in kept:
            if j in seen:
                continue
            seen.add(j)
            body_lines.append(f"{start + j:5}: {line}")
        body = "\n".join(body_lines)
    else:
        body = "\n".join(f"{start + i:5}: {line}" for i, line in enumerate(slice_lines))

    if max_chars and len(body) > max_chars:
        body = body[: int(max_chars)] + "\n... (truncated)"

    header = f"// {qname} ({lookup_result.file_path}:{lookup_result.line_number}-{lookup_result.end_line})"
    return f"{header}\n{body}"


# JSON-Schema for the tool — mirrors the Rust shim's schema exactly.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "qualified_name": {
            "type": "string",
            "description": (
                "Fully-qualified entity name to resolve (e.g. "
                "'kglite.code_tree.builder.build', "
                "'KnowledgeGraph::cypher')."
            ),
        },
        "node_type": {
            "type": ["string", "null"],
            "description": (
                "Optional node-type hint when the qualified name is ambiguous (e.g. 'Function', 'Struct')."
            ),
        },
        "start_line": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": ("Override the entity's start line (1-indexed). Defaults to the entity's defined start."),
        },
        "end_line": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": (
                "Override the entity's end line (1-indexed, inclusive). Defaults to the entity's defined end."
            ),
        },
        "grep": {
            "type": ["string", "null"],
            "description": "Regex pattern to filter lines. Returns matching lines plus context.",
        },
        "grep_context": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": "Lines of context around each grep match (default 2).",
        },
        "max_chars": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": "Cap output size in characters.",
        },
    },
    "required": ["qualified_name"],
}

DESCRIPTION = (
    "Read source code by fully-qualified entity name. Resolves the "
    "name through the active graph's `graph.source()` (which uses "
    "the code-tree node attributes), then reads the corresponding "
    "file slice from the configured source root(s). Equivalent to "
    "cypher → graph.source → read_source in a single MCP call. "
    "Same line-range / grep / max_chars filters as `read_source`."
)
