"""Source-file MCP tools backed by the ``mcp-methods`` package.

When a yaml manifest declares ``source_root:`` (or ``source_roots:``), the
bundled MCP server auto-registers ``read_source`` / ``grep`` / ``list_source``
tools that are sandboxed to those roots. This module is the thin shim layer
that shapes those tools' signatures and docstrings to match the kglite
agent conventions.

Roots arrive at :func:`register` already resolved to absolute paths
(relative-to-yaml resolution happens upstream in the manifest loader).
All path traversal protection is delegated to ``mcp-methods`` which
canonicalises against the supplied ``allowed_dirs`` / ``source_dirs``.

When a graph is also passed, ``read_source`` accepts ``qualified_name=``
to resolve a code-entity name (function, class, method) to its file
slice via :meth:`KnowledgeGraph.source` — a single round-trip instead
of cypher-then-read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

SourceProvider = Callable[[], list[str]]
GraphProvider = Callable[[], Any]


def register(
    mcp,
    source_dirs: list[str],
    *,
    graph_provider: GraphProvider | None = None,
) -> None:
    """Register read_source / grep / list_source tools on the MCP server.

    ``source_dirs`` must be absolute, canonicalised paths. All file reads
    and pattern searches are confined to those directories.

    ``graph_provider`` is an optional zero-arg callable returning the
    active :class:`KnowledgeGraph`. When provided, ``read_source``
    accepts ``qualified_name=...`` and resolves via ``graph.source()``.
    Workspace mode passes a callable that returns the active repo's
    graph; single-graph mode passes ``lambda: graph``.
    """
    if not source_dirs:
        raise ValueError("register() requires at least one source root")

    register_dynamic(mcp, lambda: source_dirs, graph_provider=graph_provider)


def register_dynamic(
    mcp,
    source_dirs_provider: SourceProvider,
    *,
    graph_provider: GraphProvider | None = None,
) -> None:
    """Variant of :func:`register` for workspace mode where the source
    roots change as the active repo switches.

    ``source_dirs_provider`` is a zero-arg callable returning the
    current absolute source roots. Returning an empty list signals
    "no active repo" and the tools return a friendly error.
    """
    import mcp_methods

    def _resolve_roots_or_error(action: str) -> tuple[list[str], str]:
        """Return ``(roots, "")`` when active, or ``([], err_msg)`` when not.

        Tools call this and short-circuit on a non-empty error string —
        keeps the type tractable by always returning a list.
        """
        roots = source_dirs_provider()
        if not roots:
            return [], f"Cannot {action}: no active source root. Activate a repo first."
        return list(roots), ""

    @mcp.tool()
    def read_source(
        file_path: str | None = None,
        qualified_name: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        grep: str | None = None,
        grep_context: int = 2,
        max_matches: int | None = None,
        max_chars: int | None = None,
    ) -> str:
        """Read a source file from the configured source root(s).

        Two ways to identify what to read:
          - ``file_path``: relative path under the source root (e.g. ``src/lib.rs``).
          - ``qualified_name``: code-entity name resolved through the active graph
            (e.g. ``MyClass.my_method``). Requires that the graph carries
            ``qualified_name`` + ``file_path`` properties on code nodes.

        Pass ``grep="pattern"`` to filter to lines matching a regex (with
        ``grep_context`` lines of surrounding context). Pass ``start_line`` /
        ``end_line`` to slice. Pass ``max_chars`` to cap output size.

        Path traversal attempts are rejected.
        """
        if file_path is None and qualified_name is None:
            return "Provide either file_path or qualified_name."
        if file_path is not None and qualified_name is not None:
            return "Provide only one of file_path / qualified_name, not both."

        if qualified_name is not None:
            resolved = _resolve_qualified_name(qualified_name, graph_provider)
            if isinstance(resolved, str):
                return resolved  # error message
            file_path, qn_start, qn_end = resolved
            if start_line is None:
                start_line = qn_start
            if end_line is None:
                end_line = qn_end

        roots, err = _resolve_roots_or_error("read source")
        if err:
            return err
        return mcp_methods.read_file(
            file_path,
            roots,
            start_line=start_line,
            end_line=end_line,
            grep=grep,
            grep_context=grep_context,
            max_matches=max_matches,
            max_chars=max_chars,
        )

    @mcp.tool()
    def grep(
        pattern: str,
        glob: str = "*",
        context: int = 0,
        max_results: int | None = 50,
        case_insensitive: bool = False,
    ) -> str:
        """Search source files under the configured source root(s) using ripgrep.

        ``pattern`` is a regex. ``glob`` filters file paths (e.g. ``"*.json"``).
        ``context`` adds N surrounding lines per match. Set ``case_insensitive=True``
        for case-insensitive matching. ``max_results`` caps the total matches
        returned (None for unlimited; default 50).
        """
        roots, err = _resolve_roots_or_error("grep")
        if err:
            return err
        return mcp_methods.ripgrep_files(
            roots,
            pattern,
            glob=glob,
            context=context,
            case_insensitive=case_insensitive,
            max_results=max_results,
        )

    @mcp.tool()
    def list_source(
        path: str = ".",
        depth: int = 1,
        glob: str | None = None,
        dirs_only: bool = False,
    ) -> str:
        """List directory contents under the configured source root.

        ``path`` is resolved against the first configured source root
        (``"."`` lists the root itself). ``depth`` controls recursion
        (1 = flat ls, 2+ = tree). ``glob`` filters entries (e.g. ``"*.py"``).
        ``dirs_only=True`` shows only directories.
        """
        roots, err = _resolve_roots_or_error("list source")
        if err:
            return err
        target = _resolve_under_roots(path, roots)
        if target is None:
            return f"Error: path '{path}' resolves outside the configured source roots."
        return mcp_methods.list_dir(
            target,
            depth=depth,
            glob=glob,
            dirs_only=dirs_only,
            relative_to=roots[0],
        )


def _resolve_qualified_name(
    qualified_name: str,
    graph_provider: GraphProvider | None,
) -> tuple[str, int | None, int | None] | str:
    """Resolve ``qualified_name`` to ``(file_path, start_line, end_line)``.

    Returns a string error message on any failure (no active graph,
    name not found, ambiguous match). Bare names without dots are
    extended with a fallback Cypher suffix-match — useful when the
    agent passes a short symbol name like ``_helper_fn`` rather than
    the fully qualified path.
    """
    if graph_provider is None:
        return "qualified_name lookup is not available — no graph is bound to this server."
    graph = graph_provider()
    if graph is None:
        return "qualified_name lookup requires an active graph; none is currently loaded."

    try:
        loc = graph.source(qualified_name)
    except Exception as e:
        return f"qualified_name lookup failed: {e}"

    if isinstance(loc, dict) and "error" in loc and not loc.get("ambiguous"):
        # Suffix fallback: the agent may have passed a short name (e.g.
        # `bellman_ford` instead of `crate::path::bellman_ford`). The
        # graph's qualified_name property typically encodes the dotted
        # path; ENDS WITH catches the short form.
        rows = _suffix_lookup(graph, qualified_name)
        if rows is None:
            return loc["error"]
        if len(rows) == 1:
            row = dict(rows[0])
            file_path = row.get("file_path") or row.get("n.file_path")
            start_line = row.get("line_number") or row.get("n.line_number")
            end_line = row.get("end_line") or row.get("n.end_line")
            if not file_path:
                return loc["error"]
            return file_path, start_line, end_line
        if len(rows) > 1:
            names = []
            for r in rows[:10]:
                d = dict(r)
                names.append(d.get("qualified_name") or d.get("n.qualified_name") or "?")
            return (
                f"Ambiguous name '{qualified_name}', matches:\n"
                + "\n".join(f"  - {n}" for n in names)
                + "\n\nUse the full qualified name to disambiguate."
            )
        return loc["error"]

    if isinstance(loc, dict) and loc.get("ambiguous"):
        names = []
        for m in loc.get("matches", []):
            names.append(m.get("qualified_name") or m.get("name") or "?")
        return f"Ambiguous name '{qualified_name}', matches:\n" + "\n".join(f"  - {n}" for n in names)

    if isinstance(loc, dict):
        file_path = loc.get("file_path")
        if not file_path:
            return f"qualified_name '{qualified_name}' has no file_path on the graph node."
        return file_path, loc.get("line_number"), loc.get("end_line")

    return f"unexpected source() return type: {type(loc).__name__}"


def _suffix_lookup(graph, name: str):
    """Best-effort fallback: ENDS WITH suffix lookup against qualified_name.

    Returns None if Cypher is unavailable or the schema doesn't carry
    a ``qualified_name`` property. Tries both the bare name and a
    ``.<name>`` form (matches method names dotted onto a class path).
    """
    queries = [
        (
            "MATCH (n) WHERE n.qualified_name ENDS WITH $suffix "
            "RETURN n.qualified_name AS qualified_name, n.file_path AS file_path, "
            "n.line_number AS line_number, n.end_line AS end_line LIMIT 10",
            {"suffix": name},
        ),
        (
            "MATCH (n) WHERE n.qualified_name ENDS WITH $suffix "
            "RETURN n.qualified_name AS qualified_name, n.file_path AS file_path, "
            "n.line_number AS line_number, n.end_line AS end_line LIMIT 10",
            {"suffix": f".{name}"},
        ),
    ]
    for query, params in queries:
        try:
            rows = graph.cypher(query, params=params)
        except Exception:
            return None
        if rows:
            return rows
    return []


def _resolve_under_roots(path: str, source_dirs: list[str]) -> str | None:
    """Resolve ``path`` against the first source root and verify the result
    stays within one of the configured roots. Returns the absolute path on
    success, ``None`` when the resolution escapes the sandbox.
    """
    primary = Path(source_dirs[0])
    candidate = (primary / path).resolve() if path != "." else primary
    candidate_str = str(candidate)
    for root in source_dirs:
        if candidate_str == root or candidate_str.startswith(root + "/"):
            return candidate_str
    return None
