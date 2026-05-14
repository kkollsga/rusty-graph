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
class Trust:
    """Manifest `trust:` flags. Mirror the keys in
    `mcp_methods::server::TrustConfig`; the Python wrapper consumes the
    JSON dict produced by Manifest::to_json(). Defaults are False —
    forces operators to opt in explicitly per extension."""

    allow_python_tools: bool = False
    allow_embedder: bool = False
    allow_query_preprocessor: bool = False


@dataclass
class Builtins:
    save_graph: bool = False
    temp_cleanup_on_overview: bool = False


@dataclass
class WorkspaceCfg:
    kind: str = "local"
    root: Path | None = None
    watch: bool = False
    # 0.9.29 / mcp-methods 0.3.33+: opt-in declaration for parent-walk
    # manifest discovery. Single pattern (`"./repos"`, `"./prod-*"`),
    # list of patterns (`["./repos", "./clones"]`), or None (no
    # parent-walk allowed). Glob syntax follows the `globset` crate
    # (`*` / `?` / `[abc]`). Matched against the workspace dir's
    # basename — single-segment match only; multi-segment paths and
    # `..` are rejected at parse time by the framework.
    applies_to: str | list[str] | None = None


@dataclass
class CypherTool:
    name: str
    cypher: str
    description: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class BundledOverride:
    """Operator-declared customisation of a bundled tool's agent-
    facing surface. Comes from a `tools[].bundled: <name>` entry in
    the manifest (mcp-methods 0.3.31+).

    - `description=None` means "keep the default description."
    - `description=""` means "explicitly clear the description"
      (rare but preserved as distinct from None).
    - `hidden=True` means "omit from tools/list AND reject calls."
    - `rename=None` means "expose under the canonical bundled name."
      `rename="foo_bar"` exposes the bundled tool to the agent as
      `foo_bar`. mcp-methods 0.3.34+; per-deployment disambiguation
      for ToolSearch when multiple kglite servers expose identical
      surfaces. Server-side collision check (rename must not shadow
      another registered tool) happens at boot in server.py.

    Validation against the actual bundled-tool catalogue happens
    in `server.py` at boot — the framework doesn't know what
    bundled tools kglite provides."""

    name: str
    description: str | None = None
    hidden: bool = False
    rename: str | None = None


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
    bundled_overrides: list[BundledOverride] = field(default_factory=list)
    trust: Trust = field(default_factory=Trust)
    # 0.9.31 / mcp-methods 0.3.36+: opt-in skills declaration.
    # Polymorphic JSON shape:
    #   None or False  → skills disabled
    #   True           → bundled framework + downstream-binary defaults only
    #   "./pack"       → single path source
    #   [True, "./a"]  → list; True = bundled, str = path
    # The framework's `SkillRegistry.from_manifest(...)` does the
    # three-layer composition (project / domain-pack / bundled).
    # kglite's wrapper passes this through to the framework.
    skills: bool | str | list[bool | str] | None = None


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

    # 0.9.31: pass `skills` through verbatim. The framework's
    # to_json shape is polymorphic — `False` for disabled, an array
    # mixing `true` (bundled marker) and path strings otherwise.
    # mcp_methods.SkillRegistry.from_manifest reads it back from
    # the YAML directly; we keep the parsed shape on the dataclass
    # for callers that want to inspect it without reopening the YAML.
    skills_raw = data.get("skills", False)
    if skills_raw is False:
        skills_value: bool | str | list[bool | str] | None = None
    elif isinstance(skills_raw, list):
        skills_value = list(skills_raw)
    else:
        # Bool true or a path string. Framework normalises to a list
        # at runtime; we pass the raw shape through.
        skills_value = skills_raw

    m = Manifest(
        yaml_path=yaml_path,
        base_dir=base_dir,
        name=data.get("name"),
        instructions=data.get("instructions"),
        overview_prefix=data.get("overview_prefix"),
        source_roots=[_resolve(s) for s in (data.get("source_roots") or [])],
        env_file=_resolve(data["env_file"]) if data.get("env_file") else None,
        extensions=data.get("extensions") or {},
        skills=skills_value,
    )

    b = data.get("builtins") or {}
    m.builtins = Builtins(
        save_graph=bool(b.get("save_graph", False)),
        temp_cleanup_on_overview=(b.get("temp_cleanup") == "on_overview"),
    )

    t = data.get("trust") or {}
    m.trust = Trust(
        allow_python_tools=bool(t.get("allow_python_tools", False)),
        allow_embedder=bool(t.get("allow_embedder", False)),
        # mcp-methods 0.3.29+: emits `allow_query_preprocessor` under
        # trust. Older versions don't emit the key — bool(None) is
        # False, preserving the same default semantics across pin
        # versions.
        allow_query_preprocessor=bool(t.get("allow_query_preprocessor", False)),
    )

    w = data.get("workspace")
    if w is not None:
        # `applies_to` arrives polymorphic from mcp-methods: None |
        # str (single pattern) | list[str] (multi-pattern). Pass through
        # in the same shape rather than normalising — the framework's
        # find_workspace_manifest interprets it.
        applies_to = w.get("applies_to")
        if applies_to is not None and not isinstance(applies_to, (str, list)):
            raise ManifestError(f"workspace.applies_to must be a string or list of strings (got {applies_to!r})")
        m.workspace = WorkspaceCfg(
            kind=w["kind"],
            root=_resolve(w["root"]) if w.get("root") else None,
            watch=bool(w.get("watch", False)),
            applies_to=applies_to,
        )

    for t in data.get("tools") or []:
        kind = t.get("kind")
        if kind == "cypher":
            m.tools.append(
                CypherTool(
                    name=t["name"],
                    cypher=t["cypher"],
                    description=t.get("description"),
                    parameters=t.get("parameters") or {},
                )
            )
        elif kind == "bundled":
            # mcp-methods 0.3.31+: `tools[].bundled:` overrides. The
            # framework records the operator's declared customisation;
            # kglite validates the name against its actual bundled-
            # tool catalogue at boot in server.py. mcp-methods 0.3.34+:
            # `rename:` field for per-deployment disambiguation
            # (multiple kglite servers sharing the same canonical
            # bundled surface).
            m.bundled_overrides.append(
                BundledOverride(
                    name=t["name"],
                    description=t.get("description"),
                    hidden=bool(t.get("hidden", False)),
                    rename=t.get("rename"),
                )
            )
        else:
            # `python:` tools (kind == "python") are a framework
            # feature kglite's Python entry point does not consume.
            # Other unknown kinds (future additions, etc.) are
            # silently ignored — forward-compatible.
            continue

    return m


def find_sibling_manifest(graph_path: Path) -> Path | None:
    """Auto-detect `<graph_stem>_mcp.yaml` next to a graph file."""
    result = _mcp_internal.Manifest.find_sibling(str(graph_path))
    return Path(result) if result else None


def find_workspace_manifest(dir_path: Path) -> Path | None:
    """Auto-detect `workspace_mcp.yaml` for a workspace directory.

    Checks two locations in priority order (since mcp-methods 0.3.33,
    picked up by kglite 0.9.29):

    1. Inside the workspace dir itself
       (`<workspace_dir>/workspace_mcp.yaml`). The documented primary
       location — no opt-in required.
    2. As a sibling of the workspace dir
       (`<workspace_dir>/../workspace_mcp.yaml`) **iff** that parent
       manifest declares `workspace.applies_to` and the workspace
       dir's basename matches one of the declared patterns. The
       natural layout for github-clone-tracker workspaces is
       `parent/workspace_mcp.yaml` + `parent/repos/` — opt-in via
       `applies_to: ./repos` (literal) or `applies_to: ./*` (any
       child).

    Without the `applies_to` opt-in the framework refuses the
    parent fallback by design: an unconditional walk would silently
    inherit the wrong manifest if the operator pointed `--workspace`
    at any sibling of a workspace-manifest directory. The opt-in
    declaration gives the manifest author explicit control over
    discovery scope.

    The opt-in check + glob matching happens inside the mcp-methods
    Rust path (`server/manifest.rs::find_workspace_manifest`).
    """
    result = _mcp_internal.Manifest.find_workspace(str(dir_path))
    return Path(result) if result else None
