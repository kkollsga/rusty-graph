"""Register manifest-declared python: hooks as MCP @tool() callables.

Two-signal opt-in: a manifest may declare ``python:`` tools only if it
sets ``trust.allow_python_tools: true`` AND the operator runs the CLI
with ``--trust-tools``. Either signal alone refuses to load — the
yaml-side flag tells you the manifest *can* request python execution,
the CLI flag tells you the operator authored or audited it.

File paths are resolved relative to the manifest's directory so the
yaml stays portable. The named function is loaded via importlib (no
``sys.path`` mutation) and registered with its existing signature +
docstring intact, so FastMCP's introspection produces the right input
schema for whatever the user wrote.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from kglite.mcp_server.manifest import ManifestError, PythonTool


def register_python_tools(
    mcp,
    specs: list[PythonTool],
    *,
    manifest_dir: Path,
    trust_flag: bool,
    allow_python_tools: bool,
) -> int:
    """Validate trust gates, load each python tool, register on the server.

    Returns the count registered. Raises :class:`ManifestError` when trust
    signals are missing or any tool fails to load.
    """
    if not specs:
        return 0
    if not allow_python_tools:
        raise ManifestError(
            f"manifest declares {len(specs)} python tool(s) but "
            "`trust.allow_python_tools: true` is not set in the manifest"
        )
    if not trust_flag:
        raise ManifestError(
            f"manifest declares {len(specs)} python tool(s) but the CLI was "
            "started without --trust-tools (refusing to load arbitrary code)"
        )

    for spec in specs:
        fn = _load_function(spec, manifest_dir)
        # Preserve the manifest-declared name; everything else (signature,
        # docstring, annotations) is whatever the user wrote.
        fn.__name__ = spec.name
        if spec.description:
            fn.__doc__ = spec.description
        mcp.tool()(fn)
    return len(specs)


def _load_function(spec: PythonTool, manifest_dir: Path):
    file_path = (manifest_dir / spec.python).resolve()
    if not file_path.is_file():
        raise ManifestError(
            f"python tool {spec.name!r}: file {spec.python!r} resolves to {str(file_path)!r} which does not exist"
        )

    module_name = f"_kglite_mcp_user_tool_{spec.name}"
    module_spec = importlib.util.spec_from_file_location(module_name, file_path)
    if module_spec is None or module_spec.loader is None:
        raise ManifestError(f"python tool {spec.name!r}: could not load module from {file_path}")

    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as e:
        raise ManifestError(f"python tool {spec.name!r}: error executing {file_path}: {e}") from e

    fn = getattr(module, spec.function, None)
    if fn is None:
        raise ManifestError(f"python tool {spec.name!r}: function {spec.function!r} not found in {file_path}")
    if not callable(fn):
        raise ManifestError(f"python tool {spec.name!r}: {spec.function!r} in {file_path} is not callable")
    return fn
