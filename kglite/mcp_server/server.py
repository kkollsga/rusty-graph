"""`kglite-mcp-server` — Python entry point (0.9.20+).

Replaces the pre-0.9.20 bundled Rust binary. The architecture is
identical: same modes (`--graph` / `--workspace` / `--watch` /
`--source-root` / bare), same YAML schema, same tool surface
(`cypher_query`, `graph_overview`, `save_graph`, `read_code_source`,
manifest-declared cypher tools, optional `csv_http_server`).

Why Python instead of Rust: a Rust binary that depends on the kglite
crate transitively links libpython at a specific Python version,
forcing per-Python wheel matrices (3 OS × 4 Python = 12 wheels in
0.9.18/0.9.19). The Python entry-point bypasses that entirely — the
wheel is back to 3 abi3 wheels, and kglite's Python `cypher()`
method already releases the GIL so the hot-path performance is
unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys
from typing import Any

log = logging.getLogger("kglite.mcp_server")


def main() -> None:
    """Console-script entry point. Synchronous wrapper that pipes into
    the asyncio loop."""
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging()

    try:
        # Lazy imports so a missing `mcp` extras dep produces an
        # actionable error rather than a cryptic ModuleNotFoundError
        # from somewhere deep in the call chain.
        _check_extras()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"kglite-mcp-server: {e}\n")
        sys.exit(1)

    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        sys.exit(0)


def _check_extras() -> None:
    """Verify the `mcp` extras are installed. Without them the server
    can't function; without a clear error, users hit confusing
    `ModuleNotFoundError` deep in the stack."""
    missing = []
    for mod, pkg in [
        ("mcp", "mcp"),
        ("yaml", "pyyaml"),
        ("aiohttp", "aiohttp"),
        ("fastembed", "fastembed"),
        ("watchdog", "watchdog"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise SystemExit(
            "kglite-mcp-server requires the MCP extras. Install with:\n"
            f"  pip install 'kglite[mcp]'\n"
            f"Missing packages: {', '.join(missing)}"
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kglite-mcp-server",
        description="MCP server for KGLite knowledge graphs (Python entry point).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--graph", type=Path, help="Path to a .kgl knowledge graph file. Loaded at boot.")
    mode.add_argument(
        "--source-root",
        type=Path,
        dest="source_root",
        help="Source-root mode (no graph).",
    )
    mode.add_argument(
        "--workspace",
        type=Path,
        help="Workspace mode: clone GitHub repos and build code-tree graphs.",
    )
    mode.add_argument(
        "--watch",
        type=Path,
        help="Watch mode: rebuild the code-tree graph on file changes.",
    )
    p.add_argument("--mcp-config", type=Path, dest="mcp_config")
    p.add_argument("--name")
    p.add_argument(
        "--trust-tools",
        action="store_true",
        help="Legacy flag (0.9.17 era); no-op in 0.9.20+ since python tools were removed.",
    )
    p.add_argument(
        "--stale-after-days",
        type=int,
        default=7,
        dest="stale_after_days",
        help="Workspace mode: refresh clones older than N days (default 7).",
    )
    return p


def _setup_logging() -> None:
    level = logging.INFO
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


async def _async_main(args: argparse.Namespace) -> None:
    from kglite.mcp_server.csv_http import (
        CsvHttpConfig,
    )
    from kglite.mcp_server.csv_http import (
        from_manifest_value as csv_http_from_manifest,
    )
    from kglite.mcp_server.csv_http import (
        spawn as csv_http_spawn,
    )
    from kglite.mcp_server.embedder import from_manifest_value as embedder_from_manifest
    from kglite.mcp_server.preprocessor import PreprocessorError
    from kglite.mcp_server.preprocessor import from_manifest_value as preprocessor_from_manifest
    from kglite.mcp_server.tools import GraphState
    from kglite.mcp_server.watch import start as watch_start
    from kglite.mcp_server.workspace import Workspace

    mode = _pick_mode(args)
    _validate_mode_paths(mode)

    manifest = _load_manifest(args, mode)
    _load_env(manifest, mode)

    csv_http_cfg: CsvHttpConfig | None = None
    embedder_adapter = None
    preprocessor = None
    if manifest is not None:
        ext = manifest.extensions
        csv_http_cfg = csv_http_from_manifest(ext.get("csv_http_server"), manifest.base_dir)
        embedder_adapter = embedder_from_manifest(ext.get("embedder"))
        try:
            preprocessor = preprocessor_from_manifest(
                ext.get("cypher_preprocessor"),
                manifest.base_dir,
                manifest.trust.allow_query_preprocessor,
            )
        except PreprocessorError as e:
            sys.stderr.write(f"ERROR: {e}\n")
            sys.exit(3)

    if csv_http_cfg is not None:
        await csv_http_spawn(csv_http_cfg)

    graph_state = GraphState()
    workspace: Workspace | None = None

    # Mode-specific binding.
    if mode["kind"] == "graph":
        graph_state.load_kgl(mode["path"])
    elif mode["kind"] == "watch":
        graph_state.build_code_tree(mode["path"])
    elif mode["kind"] == "workspace":
        workspace = Workspace(root=mode["path"], stale_after_days=args.stale_after_days)
    elif mode["kind"] == "local_workspace":
        workspace = Workspace(
            root=mode["path"],
            kind="local",
            stale_after_days=args.stale_after_days,
        )
        # 0.9.28: local-workspace mode binds the workspace root as the
        # active root at construction (see mcp-methods workspace.rs
        # open_local), but the post-activate hook is registered AFTER
        # construction so the initial active-binding doesn't fire it.
        # Build the code-tree explicitly at boot so the first
        # cypher_query (before any set_root_dir) sees a populated graph
        # — matches watch mode's boot-time build.
        graph_state.build_code_tree(mode["path"])

    if embedder_adapter is not None:
        graph_state.bind_embedder(embedder_adapter)

    # Boot summary to stderr (MCP stdio uses stdout for protocol).
    _print_boot_summary(mode, manifest, graph_state)

    # Wire the watcher if applicable.
    watch_handle: Any = None
    if mode["kind"] == "watch":
        watch_handle = watch_start(
            mode["path"],
            lambda: graph_state.build_code_tree(mode["path"]),
        )
    elif mode["kind"] == "local_workspace" and manifest and manifest.workspace and manifest.workspace.watch:
        watch_handle = watch_start(
            mode["path"],
            lambda: graph_state.build_code_tree(mode["path"]),
        )
    _ = watch_handle  # keep alive until process exit

    # Build the MCP server.
    server = _build_server(
        graph_state=graph_state,
        manifest=manifest,
        mode=mode,
        csv_http_cfg=csv_http_cfg,
        workspace=workspace,
        preprocessor=preprocessor,
    )

    # Serve.
    from mcp.server.stdio import stdio_server

    fallback_name = _fallback_name(mode["kind"])
    server_name = args.name or (manifest.name if manifest else None) or fallback_name
    instructions = manifest.instructions if manifest else None

    async with stdio_server() as (read_stream, write_stream):
        init_opts = server.create_initialization_options(
            notification_options=None,
            experimental_capabilities={},
        )
        # Override the name + instructions populated by
        # `create_initialization_options` so the boot YAML can shape
        # what the agent sees on `initialize`.
        init_opts.server_name = server_name
        if instructions is not None:
            init_opts.instructions = instructions
        await server.run(read_stream, write_stream, init_opts)


def _pick_mode(args: argparse.Namespace) -> dict[str, Any]:
    if args.graph:
        return {"kind": "graph", "path": args.graph}
    if args.source_root:
        return {"kind": "source_root", "path": args.source_root}
    if args.workspace:
        return {"kind": "workspace", "path": args.workspace}
    if args.watch:
        return {"kind": "watch", "path": args.watch}
    return {"kind": "bare", "path": None}


def _validate_mode_paths(mode: dict[str, Any]) -> None:
    if mode["kind"] == "graph":
        p = mode["path"]
        # 0.9.26: `--graph` also accepts disk-backed graph directories
        # (built via `kglite.KnowledgeGraph(storage="disk", path=...)`),
        # not just single `.kgl` files. Disk graphs are a directory of
        # column-store files identified by a `disk_graph_meta.json`
        # sentinel — the same marker the Rust loader checks
        # (src/graph/io/file.rs::load_file). Without this branch,
        # operators deploying Wikidata-scale graphs (the documented
        # `storage="disk"` path for graphs >50M nodes) hit a
        # misleading "does not exist" error even though the path is
        # a valid graph directory.
        if not (p.is_file() or (p.is_dir() and (p / "disk_graph_meta.json").is_file())):
            sys.stderr.write(f"--graph path is not a .kgl file or disk-backed graph directory: {p}\n")
            sys.exit(1)
    if mode["kind"] in {"source_root", "watch", "workspace"} and not mode["path"].is_dir():
        sys.stderr.write(f"path does not exist or is not a directory: {mode['path']}\n")
        sys.exit(1)


def _load_manifest(args: argparse.Namespace, mode: dict[str, Any]) -> Any:
    from kglite.mcp_server.manifest import (
        ManifestError,
        find_sibling_manifest,
        find_workspace_manifest,
        load_manifest,
    )

    path: Path | None = None
    if args.mcp_config:
        if not args.mcp_config.is_file():
            sys.stderr.write(f"--mcp-config path does not exist: {args.mcp_config}\n")
            sys.exit(3)
        path = args.mcp_config
    elif mode["kind"] == "graph":
        path = find_sibling_manifest(mode["path"])
    elif mode["kind"] in {"workspace", "watch"}:
        path = find_workspace_manifest(mode["path"])

    if path is None:
        return None
    try:
        manifest = load_manifest(path)
    except ManifestError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(3)

    # Promote `workspace.kind: local` over CLI flags.
    if manifest.workspace and manifest.workspace.kind == "local" and manifest.workspace.root:
        mode["kind"] = "local_workspace"
        mode["path"] = manifest.workspace.root

    return manifest


def _load_env(manifest: Any, mode: dict[str, Any]) -> None:
    """Load `.env` from the manifest's `env_file:` or a walk-up
    discovery rooted at the mode's directory.

    0.9.24: the walk-up case delegates to mcp-methods Rust
    (`kglite._mcp_internal.load_env_walk`) which mirrors the
    framework's `load_env_for_mode` semantics — same quoting rules,
    same no-overwrite-existing-env behaviour, same comment/blank-line
    handling. Explicit `env_file:` paths are still loaded inline
    (small enough that we don't need to round-trip through Rust)."""
    from kglite import _mcp_internal

    if manifest and manifest.env_file:
        env_path = manifest.env_file
        if not env_path.is_file():
            return
        # Minimal .env loader (don't pull python-dotenv as a hard dep).
        import os

        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
        log.info("loaded env file: %s", env_path)
        return

    start = mode["path"] if mode["path"] else Path.cwd()
    if start.is_file():
        start = start.parent
    loaded = _mcp_internal.load_env_walk(str(start.resolve()))
    if loaded:
        log.info("loaded env file: %s", loaded)


def _fallback_name(kind: str) -> str:
    return {
        "graph": "KGLite (single-graph)",
        "source_root": "KGLite (source-root)",
        "workspace": "KGLite (workspace)",
        "local_workspace": "KGLite (local-workspace)",
        "watch": "KGLite (watch)",
        "bare": "KGLite",
    }[kind]


def _print_boot_summary(mode: dict[str, Any], manifest: Any, state: Any) -> None:
    parts = [f"mode: {mode['kind']}"]
    if mode["path"]:
        parts[-1] += f" [{mode['path']}]"
    if manifest is not None:
        parts.append(f"manifest: {manifest.yaml_path}")
    s = state.schema()
    if s is not None:
        nodes, edges = s
        parts.append(f"graph: {nodes} nodes, {edges} edges")
    sys.stderr.write("kglite-mcp-server: " + "; ".join(parts) + "\n")


BUNDLED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "cypher_query",
        "graph_overview",
        "ping",
        "read_code_source",
        "save_graph",
        "read_source",
        "grep",
        "list_source",
        "repo_management",
        "set_root_dir",
        "github_issues",
        "github_api",
    }
)
"""Set of bundled tool names kglite-mcp-server can register. The
manifest's `tools[].bundled:` overrides (mcp-methods 0.3.31+) are
validated against this set at boot: any name not in here surfaces
as a clear `ERROR: unknown bundled tool ...` so operators catch
typos before runtime."""


def _build_server(
    graph_state: Any,
    manifest: Any,
    mode: dict[str, Any],
    csv_http_cfg: Any,
    workspace: Any,
    preprocessor: Any = None,
) -> Any:
    """Construct the mcp.Server and register all tools."""
    from mcp import types
    from mcp.server.lowlevel import Server

    import kglite._mcp_internal as mcp_internal
    from kglite.mcp_server.code_source import DESCRIPTION as CODE_SOURCE_DESC
    from kglite.mcp_server.code_source import SCHEMA as CODE_SOURCE_SCHEMA
    from kglite.mcp_server.code_source import read as read_code_source
    from kglite.mcp_server.cypher_tools import build_tool_attrs as cypher_tool_attrs
    from kglite.mcp_server.cypher_tools import call_cypher_tool
    from kglite.mcp_server.tools import run_cypher, run_overview, run_save

    fallback = _fallback_name(mode["kind"])
    server_name = (manifest.name if manifest else None) or fallback
    server: Any = Server(server_name)

    # 0.9.27: bundled-tool overrides from manifest. Validate names
    # against the catalogue at boot — unknown names error out with
    # the full list of valid bundled tools (matches the strict-key
    # validation pattern operators expect from manifest parsing).
    bundled_overrides_list = manifest.bundled_overrides if manifest else []
    unknown_bundled = {b.name for b in bundled_overrides_list} - BUNDLED_TOOL_NAMES
    if unknown_bundled:
        sys.stderr.write(
            "ERROR: unknown bundled tool name(s) in manifest `tools[].bundled:`: "
            f"{sorted(unknown_bundled)}. Valid names: {sorted(BUNDLED_TOOL_NAMES)}\n"
        )
        sys.exit(3)
    bundled_overrides: dict[str, Any] = {b.name: b for b in bundled_overrides_list}
    hidden_bundled_names: set[str] = {b.name for b in bundled_overrides_list if b.hidden}

    def _apply_override(tool: types.Tool) -> types.Tool:
        """Apply a manifest `bundled:` override to a freshly-built
        tool, returning a new Tool with the override applied (or the
        original if no override exists for this tool name)."""
        override = bundled_overrides.get(tool.name)
        if override is None or override.description is None:
            return tool
        return tool.model_copy(update={"description": override.description})

    builtins = manifest.builtins if manifest else None
    save_graph_enabled = bool(builtins and builtins.save_graph)
    temp_cleanup_dir: Path | None = None
    if builtins and builtins.temp_cleanup_on_overview:
        if csv_http_cfg is not None:
            temp_cleanup_dir = csv_http_cfg.dir
        elif manifest is not None:
            temp_cleanup_dir = manifest.base_dir / "temp"

    source_roots: list[Path] = list(manifest.source_roots) if manifest else []
    if mode["kind"] == "graph" and not source_roots and mode["path"]:
        # Auto-bind the .kgl's parent as the source root, same default
        # as the Rust shim.
        source_roots = [mode["path"].resolve().parent]
    elif mode["kind"] == "source_root" and mode["path"]:
        source_roots = [mode["path"].resolve()]
    elif mode["kind"] == "local_workspace" and mode["path"] and not source_roots:
        # 0.9.23: local-workspace mode binds the workspace root as the
        # source root by default (set_root_dir rebinds it at runtime).
        # Without this the fallback below picked manifest.base_dir which
        # is the YAML's parent — usually NOT what the operator wanted.
        source_roots = [mode["path"].resolve()]
    elif manifest is not None and not source_roots:
        # Fallback: any manifest-driven boot without an explicit
        # source_root binds the manifest's directory. Matches the
        # 0.9.18 Rust binary's behaviour — operator's sodir-style
        # YAMLs (no source_root: declared) registered read_source /
        # grep / list_source under that parent. Without this fallback
        # we'd silently lose three tools per such manifest, which is
        # exactly the 0.9.20 regression class.
        source_roots = [manifest.base_dir]

    manifest_cypher_tools = manifest.tools if manifest else []
    cypher_tool_lookup = {t.name: t for t in manifest_cypher_tools}

    # 0.9.23: source roots must be re-read on each tool call so
    # `set_root_dir` can rebind them at runtime. Mutate `source_roots`
    # in place from the handler; expose a lambda for the source tools
    # to fetch the live list. Previously this was captured once at
    # build time and set_root_dir didn't actually change anything the
    # source tools could see (the regression Cat F2 catches).
    def _live_source_roots() -> list[str]:
        return [str(p) for p in source_roots]

    # 0.9.28: wire workspace post-activate so `repo_management(...)`
    # and local-workspace `set_root_dir(...)` actually build the
    # code-tree graph for the active path AND rebind source_roots so
    # subsequent read_source/grep target the active clone. Prior to
    # 0.9.28 the workspace was instantiated but this hook was never
    # called — the activation tools were path-tracking-only, no graph
    # build, source_roots stayed pointed at the workspace dir. That's
    # why operator's open-source / code-review deployments couldn't
    # migrate from 0.6.x.
    if workspace is not None:

        def _post_activate(repo_path: str, _repo_name: str) -> None:
            # Hook signature: (repo_path, repo_name). The Rust wrapper
            # passes path first (mcp_tools.rs::DeferredPyHook::invoke),
            # which matches mcp-methods' Rust hook contract.
            target = Path(repo_path)
            graph_state.build_code_tree(target)
            source_roots[:] = [target]

        workspace.set_post_activate(_post_activate)

    # GithubIssues holds an ElementCache for drill-down across calls.
    # Constructed once per process; lives for the server's lifetime.
    github_token_present = mcp_internal.has_github_token()
    github_issues = mcp_internal.GithubIssues() if github_token_present else None

    # Workspace-mode tools. `repo_management` registers in remote
    # workspace mode; `set_root_dir` in local workspace mode. Matches
    # the 0.9.18 binary's per-mode tool surface.
    is_workspace = mode["kind"] == "workspace"
    is_local_workspace = mode["kind"] == "local_workspace"

    @server.list_tools()
    async def _list() -> list[types.Tool]:
        # ───────────────────────────────────────────────────────────
        # kglite-specific tools
        # ───────────────────────────────────────────────────────────
        tools: list[types.Tool] = [
            types.Tool(
                name="cypher_query",
                description=(
                    "Run a Cypher query against the active knowledge graph. "
                    "Returns up to 15 rows inline; append FORMAT CSV to "
                    "export results"
                    + (
                        " — large CSVs are written to the csv_http_server directory and returned as a fetch URL."
                        if csv_http_cfg is not None
                        else " to a CSV string."
                    )
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Cypher query string. Append `FORMAT CSV` for CSV-encoded output.",
                        }
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="graph_overview",
                description=(
                    "Inspect the active graph's schema. With no args returns "
                    "the inventory; pass types=[...] / connections=true|[...] "
                    "/ cypher=true|[...] for drill-down."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "types": {"type": ["array", "null"], "items": {"type": "string"}},
                        "connections": {"type": ["array", "boolean", "null"]},
                        "cypher": {"type": ["array", "boolean", "null"]},
                    },
                },
            ),
            types.Tool(
                name="read_code_source",
                description=CODE_SOURCE_DESC,
                inputSchema=CODE_SOURCE_SCHEMA,
            ),
        ]
        if save_graph_enabled:
            tools.append(
                types.Tool(
                    name="save_graph",
                    description="Persist the active graph to its source .kgl file (single-graph mode only).",
                    inputSchema={"type": "object", "properties": {}},
                )
            )

        # ───────────────────────────────────────────────────────────
        # Framework tools — wrapped from mcp-methods Rust crate via
        # kglite._mcp_internal. Same output format as the 0.9.18 binary.
        # ───────────────────────────────────────────────────────────
        tools.append(
            types.Tool(
                name="ping",
                description="Liveness probe. Returns the supplied message (default 'pong').",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Optional message to echo back.",
                        }
                    },
                },
            )
        )
        if _live_source_roots():
            tools.extend(_framework_source_tools())
        if github_issues is not None:
            tools.extend(_framework_github_tools())
        if is_workspace:
            tools.append(_framework_repo_management_tool())
        if is_local_workspace:
            tools.append(_framework_set_root_dir_tool())

        # 0.9.27: apply manifest `tools[].bundled:` overrides.
        # `hidden: true` drops the tool from tools/list (and is also
        # rejected at call time in `_call` below). `description: "..."`
        # replaces the agent-facing description without touching the
        # input schema.
        tools = [_apply_override(t) for t in tools if t.name not in hidden_bundled_names]

        # YAML-declared cypher tools (overrides do NOT apply here —
        # cypher tools carry their own description in the manifest
        # entry; the override block is for the bundled catalogue
        # only).
        tools.extend(cypher_tool_attrs(manifest_cypher_tools))
        return tools

    @server.call_tool()
    async def _call(name: str, args: dict[str, Any]) -> list[types.TextContent]:
        # 0.9.27: hidden bundled tools (manifest declared
        # `tools[].bundled: NAME / hidden: true`) are absent from
        # `tools/list` but the dispatcher still needs to refuse
        # calls if an agent tries the name directly. Reject early
        # with a clear error rather than falling through to "unknown
        # tool" (which would be confusing — the tool DOES exist in
        # the catalogue, the manifest just hid it).
        if name in hidden_bundled_names:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error: tool {name!r} is hidden by manifest configuration.",
                )
            ]
        if name == "cypher_query":
            body = run_cypher(
                graph_state,
                args.get("query", ""),
                csv_http=csv_http_cfg,
                preprocessor=preprocessor,
            )
        elif name == "graph_overview":
            body = run_overview(
                graph_state,
                args.get("types"),
                args.get("connections"),
                args.get("cypher"),
                temp_cleanup_dir,
            )
        elif name == "save_graph":
            body = run_save(graph_state)
        elif name == "read_code_source":
            body = read_code_source(graph_state, args, source_roots)
        # ── framework tools, forward to Rust wrappers ────────────────
        elif name == "ping":
            body = mcp_internal.ping(args.get("message"))
        elif name == "read_source":
            body = mcp_internal.read_source(
                args["file_path"],
                _live_source_roots(),
                start_line=args.get("start_line"),
                end_line=args.get("end_line"),
                grep=args.get("grep"),
                grep_context=args.get("grep_context"),
                max_matches=args.get("max_matches"),
                max_chars=args.get("max_chars"),
            )
        elif name == "grep":
            body = mcp_internal.grep(
                args["pattern"],
                _live_source_roots(),
                glob=args.get("glob"),
                context=args.get("context", 0),
                max_results=args.get("max_results"),
                case_insensitive=args.get("case_insensitive", False),
            )
        elif name == "list_source":
            body = mcp_internal.list_source(
                _live_source_roots(),
                path=args.get("path", "."),
                depth=args.get("depth", 1),
                glob=args.get("glob"),
                dirs_only=args.get("dirs_only", False),
            )
        elif name == "github_issues":
            if github_issues is None:
                body = "Error: github_issues requires GITHUB_TOKEN / GH_TOKEN to be set."
            else:
                body = _call_github_issues(github_issues, args, workspace)
        elif name == "github_api":
            body = mcp_internal.git_api(
                args.get("repo_name") or _active_repo_for_github(workspace),
                args["path"],
                truncate_at=args.get("truncate_at", 80_000),
            )
        elif name == "repo_management":
            if workspace is None:
                body = "Error: repo_management requires workspace mode."
            else:
                body = workspace.repo_management_tool(args)
        elif name == "set_root_dir":
            if workspace is None:
                body = "Error: set_root_dir requires local-workspace mode."
            else:
                body = workspace.set_root_dir_tool(args.get("path", ""))
                # source_roots rebind + code-tree rebuild happen via
                # the workspace post-activate hook wired above (0.9.28).
                # mcp-methods workspace.rs fires the hook on both
                # repo_management(activate) AND set_root_dir success,
                # so this branch no longer needs an inline rebind.
        # ── manifest-declared cypher tools ───────────────────────────
        elif name in cypher_tool_lookup:
            body = await call_cypher_tool(
                cypher_tool_lookup[name],
                graph_state,
                args,
                csv_http_cfg,
                preprocessor=preprocessor,
            )
        else:
            body = f"unknown tool: {name!r}"
        return [types.TextContent(type="text", text=body)]

    return server


def _framework_source_tools() -> list:
    """JSON-Schema definitions for the file-system tools. Source roots
    must be configured for these to be useful — caller gates by
    checking source_roots is non-empty."""
    from mcp import types

    return [
        types.Tool(
            name="read_source",
            description=(
                "Read a file slice from one of the configured source roots. "
                "Line range / regex filter / truncation; path traversal rejected."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "grep": {"type": "string"},
                    "grep_context": {"type": "integer"},
                    "max_matches": {"type": "integer"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="grep",
            description=(
                "Recursive regex search across source roots. .gitignore-aware. "
                "Output: `path:lineno:content` for matches, `path-lineno-content` "
                "for context lines."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string"},
                    "context": {"type": "integer"},
                    "max_results": {"type": "integer"},
                    "case_insensitive": {"type": "boolean"},
                },
                "required": ["pattern"],
            },
        ),
        types.Tool(
            name="list_source",
            description="Tree-style directory listing relative to the primary source root.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "depth": {"type": "integer"},
                    "glob": {"type": "string"},
                    "dirs_only": {"type": "boolean"},
                },
            },
        ),
    ]


def _framework_github_tools() -> list:
    from mcp import types

    return [
        types.Tool(
            name="github_issues",
            description=(
                "Fetch / search / list GitHub issues, PRs, and discussions. "
                "Drill-down with `element_id` after a fetch (cb_N for code blocks, "
                "comment_N for review comments, patch_N for diffs)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "repo_name": {"type": "string"},
                    "query": {"type": "string"},
                    "kind": {"type": "string", "enum": ["issue", "pr", "discussion", "all"]},
                    "state": {"type": "string", "enum": ["open", "closed", "all"]},
                    "sort": {"type": "string"},
                    "limit": {"type": "integer"},
                    "labels": {"type": "string"},
                    "element_id": {"type": "string"},
                    "lines": {"type": "string"},
                    "grep": {"type": "string"},
                    "context": {"type": "integer"},
                    "refresh": {"type": "boolean"},
                },
            },
        ),
        types.Tool(
            name="github_api",
            description=(
                "Generic GitHub REST GET. Relative paths auto-prefix with "
                "`/repos/<repo>/`; absolute paths pass through. Returns pretty-"
                "printed JSON, truncated at `truncate_at` chars (default 80k)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "repo_name": {"type": "string"},
                    "truncate_at": {"type": "integer"},
                },
                "required": ["path"],
            },
        ),
    ]


def _call_github_issues(github_issues: Any, args: dict[str, Any], workspace: Any) -> str:
    """Route a github_issues call between FETCH (number given) and
    SEARCH/LIST (no number) via the wrapped Rust ElementCache.

    0.9.23: when `repo_name` isn't passed and a workspace is active,
    default to the workspace's active repo. Without this the agent had
    to repeat `repo_name='org/repo'` on every call even after
    `repo_management('org/repo')` activated one. Operator caught this
    on 0.9.22 redeploy."""
    repo = args.get("repo_name") or _active_repo_for_github(workspace) or None
    number = args.get("number")
    if number is not None:
        return github_issues.fetch(
            repo or "",
            int(number),
            element_id=args.get("element_id"),
            lines=args.get("lines"),
            grep=args.get("grep"),
            context=args.get("context", 3),
            refresh=args.get("refresh", False),
        )
    return github_issues.search_or_list(
        repo=repo,
        query=args.get("query"),
        kind=args.get("kind", "all"),
        state=args.get("state", "open"),
        sort=args.get("sort"),
        limit=args.get("limit", 20),
        labels=args.get("labels"),
    )


def _framework_repo_management_tool() -> Any:
    from mcp import types

    return types.Tool(
        name="repo_management",
        description=(
            "Activate (or update / delete) a cloned GitHub repo in the "
            "workspace. With `name='org/repo'`: clone if missing, build "
            "code-tree, set as active. With `update=true`: refresh active "
            "clone. With `delete=true`: remove clone + inventory entry. "
            "With no args: list known repos."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "delete": {"type": "boolean"},
                "update": {"type": "boolean"},
                "force_rebuild": {"type": "boolean"},
            },
        },
    )


def _framework_set_root_dir_tool() -> Any:
    from mcp import types

    return types.Tool(
        name="set_root_dir",
        description=(
            "Swap the active source root (local-workspace mode only). "
            "Canonicalises the path, rebinds the source tools, and "
            "triggers a code-tree rebuild for the new root."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    )


def _active_repo_for_github(workspace: Any) -> str:
    """Return the active repo (`org/repo`) for github_api when not
    explicitly passed. Falls back to empty string — the Rust function
    handles the empty case with a clear error."""
    if workspace is None:
        return ""
    name = getattr(workspace, "active_repo_name", lambda: None)()
    return name or ""


if __name__ == "__main__":
    main()
