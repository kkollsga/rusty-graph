"""YAML manifest parser — thin wrapper around mcp-methods Rust.

0.9.24: YAML parsing, strict-unknown-key validation, and per-field
schema enforcement (tools, embedder, workspace, builtins, env_file)
all happen in `mcp_methods::server::manifest`. This module loads the
validated dict via `_mcp_internal.Manifest.load(path).as_dict()` and
populates the dataclasses that `server.py` consumes.

Schema reference: see `crates/mcp-methods/src/server/manifest.rs`
(`Manifest::to_json` defines the canonical JSON shape, stable across
patch releases per the mcp-methods 0.3.27 contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kglite import _mcp_internal


class ManifestError(ValueError):
    """Raised when manifest parsing/validation fails. Wraps the
    Rust-side error message verbatim (`mcp_methods::ManifestError`
    surfaces the YAML path + the specific violation)."""


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
    """Parse + validate a YAML manifest. mcp-methods Rust owns the
    parser; this function adapts the validated JSON dict into kglite's
    dataclass surface, resolving relative paths against the YAML's
    parent directory."""
    try:
        rust = _mcp_internal.Manifest.load(str(path))
    except ValueError as e:
        raise ManifestError(str(e)) from None

    data = rust.as_dict()
    yaml_path = Path(data["yaml_path"]).resolve()
    base_dir = yaml_path.parent

    def _resolve(s: str) -> Path:
        return (base_dir / s).resolve()

    m = Manifest(
        yaml_path=yaml_path,
        base_dir=base_dir,
        name=data.get("name"),
        instructions=data.get("instructions"),
        overview_prefix=data.get("overview_prefix"),
        source_roots=[_resolve(s) for s in (data.get("source_roots") or [])],
        env_file=_resolve(data["env_file"]) if data.get("env_file") else None,
        extensions=data.get("extensions") or {},
    )

    b = data.get("builtins") or {}
    m.builtins = Builtins(
        save_graph=bool(b.get("save_graph", False)),
        temp_cleanup_on_overview=(b.get("temp_cleanup") == "on_overview"),
    )

    w = data.get("workspace")
    if w is not None:
        m.workspace = WorkspaceCfg(
            kind=w["kind"],
            root=_resolve(w["root"]) if w.get("root") else None,
            watch=bool(w.get("watch", False)),
        )

    for t in data.get("tools") or []:
        # kglite only registers cypher tools at this layer; mcp-methods
        # accepts `python:` tools too, but those are a framework
        # feature kglite's Python entry point does not consume.
        if t.get("kind") != "cypher":
            continue
        m.tools.append(
            CypherTool(
                name=t["name"],
                cypher=t["cypher"],
                description=t.get("description"),
                parameters=t.get("parameters") or {},
            )
        )

    return m


def find_sibling_manifest(graph_path: Path) -> Path | None:
    """Auto-detect `<graph_stem>_mcp.yaml` next to a graph file."""
    result = _mcp_internal.Manifest.find_sibling(str(graph_path))
    return Path(result) if result else None


def find_workspace_manifest(dir_path: Path) -> Path | None:
    """Auto-detect `workspace_mcp.yaml` in a workspace directory."""
    result = _mcp_internal.Manifest.find_workspace(str(dir_path))
    return Path(result) if result else None
