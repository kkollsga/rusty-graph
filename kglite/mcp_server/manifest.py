"""YAML manifest parser for `kglite-mcp-server`.

Schema matches the Rust shim's 0.9.19 manifest exactly so operators
don't have to edit their YAMLs:

    name: My Graph
    instructions: |
      ...
    overview_prefix: |
      ...
    source_root: ./data          # or source_roots: [./a, ./b]
    env_file: .env
    builtins:
      save_graph: false
      temp_cleanup: on_overview
    extensions:
      embedder:
        backend: fastembed
        model: BAAI/bge-m3
      csv_http_server:
        port: 8765
        dir: temp/
        cors_origin: "*"
    workspace:
      kind: local
      root: ./repos
      watch: true
    tools:
      - name: ...
        cypher: ...
        parameters: {...}
        description: ...

Strict-key validation: unknown top-level / nested keys raise
`ManifestError`. This catches typos before boot rather than silently
ignoring them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_TOP_LEVEL = {
    "name",
    "instructions",
    "overview_prefix",
    "source_root",
    "source_roots",
    "env_file",
    "builtins",
    "extensions",
    "workspace",
    "tools",
    "trust",  # legacy; parsed and ignored (binary no longer loads python tools)
    "embedder",  # legacy 0.9.17 key; warned and ignored
}

ALLOWED_BUILTINS = {"save_graph", "temp_cleanup"}
ALLOWED_TEMP_CLEANUP = {"never", "on_overview"}
ALLOWED_WORKSPACE = {"kind", "root", "watch"}
ALLOWED_WORKSPACE_KIND = {"local"}
ALLOWED_TOOL_KEYS = {"name", "description", "cypher", "parameters"}


class ManifestError(ValueError):
    """Raised when manifest parsing/validation fails. Message is
    pre-formatted with the YAML path."""


@dataclass
class Builtins:
    save_graph: bool = False
    temp_cleanup_on_overview: bool = False


@dataclass
class WorkspaceCfg:
    kind: str = "local"
    root: Path | None = None
    watch: bool = False


@dataclass
class CypherTool:
    name: str
    cypher: str
    description: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Manifest:
    yaml_path: Path
    base_dir: Path
    name: str | None = None
    instructions: str | None = None
    overview_prefix: str | None = None
    source_roots: list[Path] = field(default_factory=list)
    env_file: Path | None = None
    builtins: Builtins = field(default_factory=Builtins)
    extensions: dict[str, Any] = field(default_factory=dict)
    workspace: WorkspaceCfg | None = None
    tools: list[CypherTool] = field(default_factory=list)


def load_manifest(path: Path) -> Manifest:
    """Parse + validate a YAML manifest from `path`. Returns the
    resolved `Manifest` with paths canonicalised against the YAML's
    parent directory."""
    if not path.is_file():
        raise ManifestError(f"manifest file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ManifestError(f"{path}: YAML parse error: {e}") from e
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: top-level YAML must be a mapping")

    base_dir = path.resolve().parent
    unknown = set(raw.keys()) - ALLOWED_TOP_LEVEL
    if unknown:
        raise ManifestError(f"{path}: unknown top-level keys: {sorted(unknown)}")

    if "embedder" in raw:
        # Legacy 0.9.17 Python embedder factory. The 0.9.18 framework
        # dropped support; warn and ignore so operators on stale YAMLs
        # see a clear deprecation rather than a silent boot.
        import warnings

        warnings.warn(
            f"{path}: 'embedder:' is deprecated; move to "
            "'extensions.embedder:' (see docs/guides/mcp-servers.md). "
            "The legacy block is ignored.",
            DeprecationWarning,
            stacklevel=2,
        )

    m = Manifest(yaml_path=path.resolve(), base_dir=base_dir)
    m.name = _str(raw.get("name"), path, "name")
    m.instructions = _str(raw.get("instructions"), path, "instructions")
    m.overview_prefix = _str(raw.get("overview_prefix"), path, "overview_prefix")

    if "source_root" in raw and "source_roots" in raw:
        raise ManifestError(f"{path}: cannot set both 'source_root' and 'source_roots'")
    if "source_root" in raw:
        m.source_roots = [_resolve_path(raw["source_root"], base_dir, path, "source_root")]
    elif "source_roots" in raw:
        sr = raw["source_roots"]
        if not isinstance(sr, list):
            raise ManifestError(f"{path}: source_roots must be a list")
        m.source_roots = [_resolve_path(s, base_dir, path, "source_roots[]") for s in sr]

    if "env_file" in raw:
        m.env_file = _resolve_path(raw["env_file"], base_dir, path, "env_file")

    if "builtins" in raw:
        b = raw["builtins"]
        if not isinstance(b, dict):
            raise ManifestError(f"{path}: builtins must be a mapping")
        unknown_b = set(b.keys()) - ALLOWED_BUILTINS
        if unknown_b:
            raise ManifestError(f"{path}: unknown builtins keys: {sorted(unknown_b)}")
        m.builtins.save_graph = bool(b.get("save_graph", False))
        tc = b.get("temp_cleanup", "never")
        if tc not in ALLOWED_TEMP_CLEANUP:
            raise ManifestError(f"{path}: builtins.temp_cleanup must be one of {sorted(ALLOWED_TEMP_CLEANUP)}")
        m.builtins.temp_cleanup_on_overview = tc == "on_overview"

    if "extensions" in raw:
        ext = raw["extensions"]
        if not isinstance(ext, dict):
            raise ManifestError(f"{path}: extensions must be a mapping")
        m.extensions = ext

    if "workspace" in raw:
        w = raw["workspace"]
        if not isinstance(w, dict):
            raise ManifestError(f"{path}: workspace must be a mapping")
        unknown_w = set(w.keys()) - ALLOWED_WORKSPACE
        if unknown_w:
            raise ManifestError(f"{path}: unknown workspace keys: {sorted(unknown_w)}")
        kind = w.get("kind", "local")
        if kind not in ALLOWED_WORKSPACE_KIND:
            raise ManifestError(f"{path}: workspace.kind must be one of {sorted(ALLOWED_WORKSPACE_KIND)}")
        m.workspace = WorkspaceCfg(
            kind=kind,
            root=_resolve_path(w["root"], base_dir, path, "workspace.root") if "root" in w else None,
            watch=bool(w.get("watch", False)),
        )

    if "tools" in raw:
        tools = raw["tools"]
        if not isinstance(tools, list):
            raise ManifestError(f"{path}: tools must be a list")
        for i, t in enumerate(tools):
            if not isinstance(t, dict):
                raise ManifestError(f"{path}: tools[{i}] must be a mapping")
            unknown_t = set(t.keys()) - ALLOWED_TOOL_KEYS
            if unknown_t:
                raise ManifestError(f"{path}: tools[{i}] unknown keys: {sorted(unknown_t)}")
            if "name" not in t or "cypher" not in t:
                raise ManifestError(f"{path}: tools[{i}] requires 'name' and 'cypher'")
            params = t.get("parameters", {})
            if params and not isinstance(params, dict):
                raise ManifestError(f"{path}: tools[{i}].parameters must be a mapping")
            m.tools.append(
                CypherTool(
                    name=str(t["name"]),
                    cypher=str(t["cypher"]),
                    description=t.get("description"),
                    parameters=params,
                )
            )
    return m


def _str(value: Any, path: Path, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestError(f"{path}: {key} must be a string")
    return value


def _resolve_path(value: Any, base: Path, manifest: Path, key: str) -> Path:
    if not isinstance(value, str):
        raise ManifestError(f"{manifest}: {key} must be a string path")
    p = (base / value).resolve()
    return p


def find_sibling_manifest(graph_path: Path) -> Path | None:
    """Return `<graph_path_stem>_mcp.yaml` next to the graph file if it
    exists. Auto-detect pattern for `--graph X.kgl`."""
    candidate = graph_path.parent / f"{graph_path.stem}_mcp.yaml"
    return candidate if candidate.is_file() else None


def find_workspace_manifest(dir_path: Path) -> Path | None:
    """Return `<dir>/workspace_mcp.yaml` if it exists. Auto-detect for
    `--workspace DIR` or `--watch DIR`."""
    candidate = dir_path / "workspace_mcp.yaml"
    return candidate if candidate.is_file() else None
