"""CLI entry point for ``kglite-mcp-server``.

Two operating modes:

1. **Single-graph (default).** Pass ``--graph X.kgl`` and the server
   loads a single, pre-built graph and registers the standard tools:
   ``graph_overview`` and ``cypher_query``, plus any extras declared
   in a sibling ``<basename>_mcp.yaml`` manifest.

2. **Workspace (multi-graph).** Pass ``--workspace DIR`` instead. The
   server starts with no graph; the agent activates one with
   ``repo_management('org/repo')`` (clones the repo, builds a code-
   graph, pins it as active). All other tools then operate on whichever
   repo is active. Idle repos auto-sweep after ``stale_after_days``.

When **not** to use either mode:

- If you need fundamentally different graph-side semantics (e.g. a
  query rewriter that intercepts every Cypher call), fork
  ``examples/mcp_server.py``. The manifest covers most cases — source
  roots, custom embedders, cypher/python tools, builtins like
  ``save_graph`` and ``temp_cleanup`` — but it is not infinitely
  extensible.
- The CSV temp dir is deliberately *outside* the source-root sandbox.
  Agents must ``fetch()`` the localhost URL rather than reading the
  CSV with ``read_source``. ``read_source`` will refuse paths under
  ``./temp/`` if you set ``source_root: .`` because the resolver
  canonicalises against the configured root only.

Claude Desktop config::

    {
      "mcpServers": {
        "my-graph": {
          "command": "kglite-mcp-server",
          "args": ["--graph", "/abs/path/to/graph.kgl"]
        },
        "open-source": {
          "command": "kglite-mcp-server",
          "args": ["--workspace", "/abs/path/to/workspace/"]
        }
      }
    }
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import kglite
from kglite.mcp_server import source_access
from kglite.mcp_server.builtins import make_temp_cleanup_hook, register_save_graph
from kglite.mcp_server.core_tools import register_core_tools
from kglite.mcp_server.cypher_tools import register_cypher_tools
from kglite.mcp_server.embedder import build_sentence_transformer_embedder, load_embedder
from kglite.mcp_server.manifest import (
    CypherTool,
    Manifest,
    ManifestError,
    PythonTool,
    find_sibling_manifest,
    find_workspace_manifest,
    load_manifest,
)
from kglite.mcp_server.python_tools import register_python_tools
from kglite.mcp_server.workspace import Workspace, WorkspaceConfig, register_repo_management


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kglite-mcp-server",
        description="Serve a KGLite .kgl graph file (or workspace of repos) over MCP (stdio).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--graph",
        default=None,
        help="Path to .kgl file (default: graph.kgl). Single-graph mode.",
    )
    mode.add_argument(
        "--workspace",
        default=None,
        help=(
            "Path to a workspace directory for multi-graph mode. The agent activates "
            "repos via repo_management('org/repo') which clones, builds, and pins "
            "the active graph. Mutually exclusive with --graph."
        ),
    )
    parser.add_argument(
        "--embedder",
        default=None,
        help="sentence-transformers model name to enable text_score(); optional",
    )
    parser.add_argument("--name", default=None, help="Server display name")
    parser.add_argument(
        "--mcp-config",
        default=None,
        help=(
            "Path to a manifest YAML file. Defaults to <graph_basename>_mcp.yaml "
            "next to the graph file in single-graph mode, or "
            "<workspace>/workspace_mcp.yaml in workspace mode, when present."
        ),
    )
    parser.add_argument(
        "--trust-tools",
        action="store_true",
        help=(
            "Allow loading python: tool hooks AND custom embedders declared in the "
            "manifest. Required alongside `trust.allow_python_tools: true` / "
            "`trust.allow_embedder: true` in the yaml — both signals must be present, "
            "otherwise loading is refused."
        ),
    )
    parser.add_argument(
        "--stale-after-days",
        type=int,
        default=7,
        help="Workspace mode: auto-sweep repos idle for more than N days (default 7).",
    )
    return parser


def _resolve_source_roots(raw_roots: list[str], yaml_path: Path) -> list[str]:
    """Resolve manifest source roots relative to the yaml file's directory.

    Returns canonicalised absolute path strings. Raises :class:`ManifestError`
    if any root does not resolve to an existing directory.
    """
    base = yaml_path.parent
    resolved: list[str] = []
    for raw in raw_roots:
        candidate = (base / raw).resolve()
        if not candidate.is_dir():
            raise ManifestError(
                f"source root {raw!r} resolves to {str(candidate)!r} which is not an existing directory",
                yaml_path,
            )
        resolved.append(str(candidate))
    return resolved


def _apply_manifest(
    mcp,
    graph,
    manifest: Manifest,
    *,
    trust_tools: bool = False,
    graph_provider=None,
    register_source_roots: bool = True,
) -> dict:
    """Wire manifest-declared tools onto the MCP server.

    ``graph`` is the live graph (or ``None`` in workspace mode pre-activation).
    ``graph_provider`` is a zero-arg callable returning the current graph;
    used by source_access for ``qualified_name=`` lookups. When omitted,
    falls back to ``lambda: graph``.

    ``register_source_roots`` is set to ``False`` by workspace mode where
    source roots are managed dynamically against the active repo and
    re-registering would clash with the per-active-repo provider.
    """
    summary: dict = {"source_roots": [], "cypher_tools": 0, "python_tools": 0}
    if graph_provider is None:
        graph_provider = lambda: graph  # noqa: E731

    if manifest.source_roots and register_source_roots:
        resolved = _resolve_source_roots(manifest.source_roots, manifest.yaml_path)
        source_access.register(mcp, resolved, graph_provider=graph_provider)
        summary["source_roots"] = resolved

    cypher_specs = [t for t in manifest.tools if isinstance(t, CypherTool)]
    if cypher_specs:
        if graph is None:
            raise ManifestError(
                "manifest declares cypher tools but no graph is loaded "
                "(workspace mode binds these per-repo and isn't supported here yet)"
            )
        summary["cypher_tools"] = register_cypher_tools(mcp, graph, cypher_specs)

    python_specs = [t for t in manifest.tools if isinstance(t, PythonTool)]
    if python_specs:
        summary["python_tools"] = register_python_tools(
            mcp,
            python_specs,
            manifest_dir=manifest.yaml_path.parent,
            trust_flag=trust_tools,
            allow_python_tools=manifest.trust.allow_python_tools,
        )

    return summary


def _load_manifest_from_args(args, default_path: Path | None) -> Manifest | None:
    """Locate and load the manifest. Returns ``None`` when no manifest is in
    play. Raises :class:`ManifestError` on validation failure or when an
    explicit ``--mcp-config`` path does not exist.

    ``default_path`` is the path used to derive a sibling manifest when
    ``--mcp-config`` is not given (a graph file in single-graph mode,
    or a workspace dir in workspace mode).
    """
    manifest_path: Path | None
    if args.mcp_config:
        manifest_path = Path(args.mcp_config)
        if not manifest_path.is_file():
            raise ManifestError(f"--mcp-config path does not exist: {manifest_path}")
    elif default_path is None:
        return None
    elif default_path.is_dir():
        manifest_path = find_workspace_manifest(default_path)
    else:
        manifest_path = find_sibling_manifest(default_path)
    if manifest_path is None:
        return None
    return load_manifest(manifest_path)


def _resolve_embedder(args, manifest: Manifest | None):
    """Return an embedder instance or ``None``.

    Manifest-declared factories take precedence over the ``--embedder``
    shortcut. Both paths are independent of each other so a yaml that
    forgot to declare the embedder isn't silently overridden.
    """
    if manifest is not None and manifest.embedder is not None:
        return load_embedder(
            manifest.embedder,
            manifest_dir=manifest.yaml_path.parent,
            trust_flag=args.trust_tools,
            allow_embedder=manifest.trust.allow_embedder,
        )
    if args.embedder:
        return build_sentence_transformer_embedder(args.embedder)
    return None


def _print_load_summary(*, manifest: Manifest | None, summary: dict | None, mode: str) -> None:
    """One-line stderr report at boot — what the server registered."""
    parts: list[str] = [f"mode: {mode}"]
    if manifest is not None:
        parts.append(f"manifest: {manifest.yaml_path}")
    if summary:
        if summary.get("source_roots"):
            parts.append(f"source roots: {summary['source_roots']}")
        if summary.get("cypher_tools"):
            parts.append(f"{summary['cypher_tools']} cypher tool(s)")
        if summary.get("python_tools"):
            parts.append(f"{summary['python_tools']} python tool(s)")
        if summary.get("embedder"):
            parts.append("embedder loaded")
        if summary.get("save_graph"):
            parts.append("save_graph enabled")
        if summary.get("temp_cleanup") and summary["temp_cleanup"] != "never":
            parts.append(f"temp_cleanup: {summary['temp_cleanup']}")
    print("kglite-mcp-server: " + "; ".join(parts), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_arg_parser().parse_args(argv)

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print(
            "ERROR: the `mcp` package is not installed. Reinstall with the\n"
            "       MCP extra:    pip install 'kglite[mcp]'",
            file=sys.stderr,
        )
        return 2

    if args.workspace is not None:
        return _run_workspace_mode(args, FastMCP)
    return _run_single_graph_mode(args, FastMCP)


def _run_single_graph_mode(args, FastMCP) -> int:
    graph_arg = args.graph or "graph.kgl"
    graph_path = Path(graph_arg)
    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found.", file=sys.stderr)
        return 1

    try:
        manifest = _load_manifest_from_args(args, graph_path)
    except ManifestError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    graph = kglite.load(str(graph_path))

    try:
        embedder = _resolve_embedder(args, manifest)
    except ManifestError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    if embedder is not None:
        graph.set_embedder(embedder)

    schema = graph.schema()
    server_name = args.name or (manifest.name if manifest and manifest.name else None) or "KGLite Graph"
    default_instructions = (
        f"Knowledge graph with {schema['node_count']} nodes and {schema['edge_count']} edges. "
        "Call graph_overview() first to learn the schema, then cypher_query() to query."
    )
    instructions = manifest.instructions if manifest and manifest.instructions else default_instructions
    mcp = FastMCP(server_name, instructions=instructions)

    temp_dir = graph_path.parent / "temp"
    overview_prefix = manifest.overview_prefix if manifest else None
    builtins_cfg = manifest.builtins if manifest else None
    cleanup_mode = builtins_cfg.temp_cleanup if builtins_cfg else "never"
    pre_overview_hook = make_temp_cleanup_hook(temp_dir, cleanup_mode)

    register_core_tools(
        mcp,
        graph_provider=lambda: graph,
        temp_dir=temp_dir,
        overview_prefix=overview_prefix,
        pre_overview_hook=pre_overview_hook,
    )

    summary: dict = {"embedder": embedder is not None}
    if manifest is not None:
        try:
            applied = _apply_manifest(mcp, graph, manifest, trust_tools=args.trust_tools)
        except ManifestError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3
        summary.update(applied)

    if builtins_cfg and builtins_cfg.save_graph:
        register_save_graph(mcp, graph, graph_path)
        summary["save_graph"] = True
    if builtins_cfg and builtins_cfg.temp_cleanup != "never":
        summary["temp_cleanup"] = builtins_cfg.temp_cleanup

    _print_load_summary(manifest=manifest, summary=summary, mode="single-graph")
    mcp.run(transport="stdio")
    return 0


def _run_workspace_mode(args, FastMCP) -> int:
    workspace_dir = Path(args.workspace).resolve()
    if workspace_dir.exists() and not workspace_dir.is_dir():
        print(f"ERROR: --workspace path is not a directory: {workspace_dir}", file=sys.stderr)
        return 1

    try:
        manifest = _load_manifest_from_args(args, workspace_dir)
    except ManifestError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    workspace = Workspace(
        workspace_dir=workspace_dir,
        config=WorkspaceConfig(stale_after_days=args.stale_after_days),
    )
    workspace.initialise()

    server_name = args.name or (manifest.name if manifest and manifest.name else None) or "KGLite Workspace"
    default_instructions = (
        "Multi-repo workspace. FIRST STEP: call repo_management('org/repo') to clone "
        "a GitHub repo and activate its knowledge graph. After activation, "
        "graph_overview() / cypher_query() / read_source() / grep() / list_source() "
        "operate on the active repo. Idle repos auto-sweep after "
        f"{args.stale_after_days} days."
    )
    instructions = manifest.instructions if manifest and manifest.instructions else default_instructions
    mcp = FastMCP(server_name, instructions=instructions)

    overview_prefix = manifest.overview_prefix if manifest else None
    builtins_cfg = manifest.builtins if manifest else None
    cleanup_mode = builtins_cfg.temp_cleanup if builtins_cfg else "on_overview"
    pre_overview_hook = make_temp_cleanup_hook(workspace.temp_dir, cleanup_mode)

    register_repo_management(mcp, workspace)
    register_core_tools(
        mcp,
        graph_provider=workspace.current_graph,
        temp_dir=workspace.temp_dir,
        overview_prefix=overview_prefix,
        pre_overview_hook=pre_overview_hook,
        no_graph_message=(
            "No active repository. Call repo_management('org/repo') to clone "
            "and activate one, or repo_management() to list available repos."
        ),
    )
    source_access.register_dynamic(
        mcp,
        workspace.current_source_dirs,
        graph_provider=workspace.current_graph,
    )

    summary: dict = {"workspace_dir": str(workspace_dir)}
    if manifest is not None:
        try:
            applied = _apply_manifest(
                mcp,
                workspace.active_graph,
                manifest,
                trust_tools=args.trust_tools,
                graph_provider=workspace.current_graph,
                register_source_roots=False,
            )
        except ManifestError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3
        summary.update(applied)

    if builtins_cfg and builtins_cfg.save_graph:

        def _ws_save() -> str:
            graph = workspace.current_graph()
            graph_path = workspace.active_graph_path
            if graph is None or graph_path is None:
                return "No active graph to save. Call repo_management('org/repo') first."
            try:
                graph.save(str(graph_path))
                schema = graph.schema()
                return (
                    f"Saved {graph_path} ({schema.get('node_count', '?')} nodes, "
                    f"{schema.get('edge_count', '?')} edges)."
                )
            except Exception as e:
                return f"save_graph error: {e}"

        _ws_save.__doc__ = (
            "Persist the active repo's graph to its .kgl file. Use after "
            "CREATE / SET / DELETE Cypher mutations to durably save changes."
        )
        _ws_save.__name__ = "save_graph"
        mcp.tool()(_ws_save)
        summary["save_graph"] = True
    if cleanup_mode != "never":
        summary["temp_cleanup"] = cleanup_mode

    _print_load_summary(manifest=manifest, summary=summary, mode="workspace")
    mcp.run(transport="stdio")
    return 0
