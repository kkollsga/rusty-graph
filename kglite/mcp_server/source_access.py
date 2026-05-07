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
"""

from __future__ import annotations

from pathlib import Path


def register(mcp, source_dirs: list[str]) -> None:
    """Register read_source / grep / list_source tools on the MCP server.

    ``source_dirs`` must be absolute, canonicalised paths. All file reads
    and pattern searches are confined to those directories.
    """
    if not source_dirs:
        raise ValueError("register() requires at least one source root")

    import mcp_methods

    primary_root = source_dirs[0]

    @mcp.tool()
    def read_source(
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        grep: str | None = None,
        grep_context: int = 2,
        max_matches: int | None = None,
        max_chars: int | None = None,
    ) -> str:
        """Read a source file from the configured source root(s).

        Pass ``grep="pattern"`` to filter to lines matching a regex (with
        ``grep_context`` lines of surrounding context). Pass ``start_line`` /
        ``end_line`` to slice. Pass ``max_chars`` to cap output size.

        ``file_path`` is resolved against the configured source root(s);
        attempts to escape the sandbox are rejected.
        """
        return mcp_methods.read_file(
            file_path,
            source_dirs,
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
        return mcp_methods.ripgrep_files(
            source_dirs,
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
        target = _resolve_under_roots(path, source_dirs)
        if target is None:
            return f"Error: path '{path}' resolves outside the configured source roots."
        return mcp_methods.list_dir(
            target,
            depth=depth,
            glob=glob,
            dirs_only=dirs_only,
            relative_to=primary_root,
        )


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
