"""YAML manifest schema + loader for ``kglite-mcp-server``.

A manifest is a YAML file sitting next to the graph file
(``<basename>_mcp.yaml``) that declares additional tools and source
access. The loader parses, validates, and returns a :class:`Manifest`
dataclass; consumers (CLI wiring, tool registration) operate on the
validated structure.

Path strings (``source_root``, ``python:`` tool paths) are kept as the
raw user input — relative-to-yaml resolution happens at the use site so
the data class stays pure and testable.

Validation is fail-fast and user-facing: the caller surfaces
:class:`ManifestError` messages directly to the operator running
``kglite-mcp-server``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

# `yaml` (PyYAML) is part of the [mcp] extra; lazy-imported inside
# load_manifest() so the dataclasses below can be imported without
# the optional dep installed (CI runs without the extras).

_VALID_TOOL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_TOOL_KINDS = ("cypher", "python")
_ALLOWED_TOP_KEYS = frozenset(
    {
        "name",
        "instructions",
        "overview_prefix",
        "source_root",
        "source_roots",
        "trust",
        "tools",
        "embedder",
        "builtins",
    }
)
_ALLOWED_TRUST_KEYS = frozenset({"allow_python_tools", "allow_embedder"})
_ALLOWED_TOOL_KEYS = frozenset({"name", "description", "parameters", "cypher", "python", "function"})
_ALLOWED_EMBEDDER_KEYS = frozenset({"module", "class", "kwargs"})
_ALLOWED_BUILTIN_KEYS = frozenset({"save_graph", "temp_cleanup"})
_VALID_TEMP_CLEANUP = frozenset({"never", "on_overview"})


class ManifestError(ValueError):
    """Raised when a manifest fails to load or validate."""

    def __init__(self, msg: str, path: Path | None = None):
        if path is not None:
            msg = f"{path}: {msg}"
        super().__init__(msg)


@dataclass
class TrustConfig:
    allow_python_tools: bool = False
    allow_embedder: bool = False


@dataclass
class CypherTool:
    name: str
    cypher: str
    description: str | None = None
    parameters: dict | None = None


@dataclass
class PythonTool:
    name: str
    python: str
    function: str
    description: str | None = None
    parameters: dict | None = None


ToolSpec = CypherTool | PythonTool


@dataclass
class EmbedderConfig:
    """Custom-embedder factory declared in the manifest.

    ``module`` is a Python file path relative to the yaml directory (e.g.
    ``./embedder.py``). ``klass`` is the symbol to instantiate. ``kwargs``
    are passed verbatim to the constructor. Loading is gated by
    ``trust.allow_embedder: true`` plus the operator's ``--trust-tools``
    flag, identical to the python-tools gate.
    """

    module: str
    klass: str
    kwargs: dict = field(default_factory=dict)


@dataclass
class BuiltinsConfig:
    """Pre-blessed tools that don't need ``--trust-tools``.

    These wrap small kglite-internal operations that the CLI knows how
    to perform safely. Currently:

    - ``save_graph``: registers a ``save_graph()`` MCP tool that calls
      ``graph.save(graph_path)``. Use after CREATE/SET/DELETE Cypher
      mutations to persist to disk.
    - ``temp_cleanup``: ``"on_overview"`` clears the CSV export temp
      directory on bare ``graph_overview()`` calls. ``"never"`` (default)
      keeps the existing behaviour.
    """

    save_graph: bool = False
    temp_cleanup: str = "never"


@dataclass
class Manifest:
    yaml_path: Path
    name: str | None = None
    instructions: str | None = None
    overview_prefix: str | None = None
    source_roots: list[str] = field(default_factory=list)
    trust: TrustConfig = field(default_factory=TrustConfig)
    tools: list[ToolSpec] = field(default_factory=list)
    embedder: EmbedderConfig | None = None
    builtins: BuiltinsConfig = field(default_factory=BuiltinsConfig)


def find_sibling_manifest(graph_path: Path) -> Path | None:
    """Return ``<graph_basename>_mcp.yaml`` next to ``graph_path``, or ``None``."""
    candidate = graph_path.parent / f"{graph_path.stem}_mcp.yaml"
    return candidate if candidate.is_file() else None


def find_workspace_manifest(workspace_dir: Path) -> Path | None:
    """Return ``<workspace>/workspace_mcp.yaml`` or ``None``."""
    candidate = workspace_dir / "workspace_mcp.yaml"
    return candidate if candidate.is_file() else None


def load_manifest(yaml_path: Path) -> Manifest:
    """Parse and validate a manifest YAML file. Raises :class:`ManifestError`.

    Requires PyYAML (installed via ``pip install "kglite[mcp]"``).
    """
    import yaml  # type: ignore[import-untyped]

    text = yaml_path.read_text()
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ManifestError(f"YAML parse error: {e}", yaml_path) from e
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ManifestError(f"top-level must be a mapping, got {type(raw).__name__}", yaml_path)
    return _build(raw, yaml_path)


def _build(raw: dict, yaml_path: Path) -> Manifest:
    unknown = set(raw) - _ALLOWED_TOP_KEYS
    if unknown:
        raise ManifestError(
            f"unknown top-level keys: {sorted(unknown)}. Allowed: {sorted(_ALLOWED_TOP_KEYS)}",
            yaml_path,
        )

    if "source_root" in raw and "source_roots" in raw:
        raise ManifestError("specify either source_root (str) or source_roots (list), not both", yaml_path)

    source_roots: list[str] = []
    if "source_root" in raw:
        v = raw["source_root"]
        if not isinstance(v, str) or not v:
            raise ManifestError(
                f"source_root must be a non-empty string, got {type(v).__name__}",
                yaml_path,
            )
        source_roots = [v]
    elif "source_roots" in raw:
        v = raw["source_roots"]
        if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
            raise ManifestError("source_roots must be a list of non-empty strings", yaml_path)
        if not v:
            raise ManifestError("source_roots must be non-empty when set", yaml_path)
        source_roots = list(v)

    trust = TrustConfig()
    if "trust" in raw:
        tv = raw["trust"]
        if not isinstance(tv, dict):
            raise ManifestError("trust must be a mapping", yaml_path)
        trust_unknown = set(tv) - _ALLOWED_TRUST_KEYS
        if trust_unknown:
            raise ManifestError(
                f"unknown trust keys: {sorted(trust_unknown)}. Allowed: {sorted(_ALLOWED_TRUST_KEYS)}",
                yaml_path,
            )
        if "allow_python_tools" in tv:
            apt = tv["allow_python_tools"]
            if not isinstance(apt, bool):
                raise ManifestError("trust.allow_python_tools must be a bool", yaml_path)
            trust.allow_python_tools = apt
        if "allow_embedder" in tv:
            ae = tv["allow_embedder"]
            if not isinstance(ae, bool):
                raise ManifestError("trust.allow_embedder must be a bool", yaml_path)
            trust.allow_embedder = ae

    tools: list[ToolSpec] = []
    seen_names: set[str] = set()
    if "tools" in raw:
        tv = raw["tools"]
        if not isinstance(tv, list):
            raise ManifestError("tools must be a list", yaml_path)
        for i, entry in enumerate(tv):
            tool = _build_tool(entry, i, yaml_path)
            if tool.name in seen_names:
                raise ManifestError(f"duplicate tool name: {tool.name!r}", yaml_path)
            seen_names.add(tool.name)
            tools.append(tool)

    embedder = _build_embedder(raw.get("embedder"), yaml_path)
    builtins = _build_builtins(raw.get("builtins"), yaml_path)

    return Manifest(
        yaml_path=yaml_path,
        name=_optional_str(raw, "name", yaml_path),
        instructions=_optional_str(raw, "instructions", yaml_path),
        overview_prefix=_optional_str(raw, "overview_prefix", yaml_path),
        source_roots=source_roots,
        trust=trust,
        tools=tools,
        embedder=embedder,
        builtins=builtins,
    )


def _optional_str(raw: dict, key: str, yaml_path: Path) -> str | None:
    if key not in raw:
        return None
    v = raw[key]
    if v is None:
        return None
    if not isinstance(v, str):
        raise ManifestError(f"{key} must be a string", yaml_path)
    return v


def _build_tool(entry: object, idx: int, yaml_path: Path) -> ToolSpec:
    if not isinstance(entry, dict):
        raise ManifestError(f"tools[{idx}] must be a mapping", yaml_path)

    unknown = set(entry) - _ALLOWED_TOOL_KEYS
    if unknown:
        raise ManifestError(
            f"tools[{idx}] has unknown keys: {sorted(unknown)}. Allowed: {sorted(_ALLOWED_TOOL_KEYS)}",
            yaml_path,
        )

    name = entry.get("name")
    if not isinstance(name, str) or not _VALID_TOOL_NAME.match(name):
        raise ManifestError(
            f"tools[{idx}] needs a string `name:` matching {_VALID_TOOL_NAME.pattern}",
            yaml_path,
        )

    kinds_present = [k for k in _TOOL_KINDS if k in entry]
    if not kinds_present:
        raise ManifestError(
            f"tools[{idx}] ({name!r}) needs exactly one of: {list(_TOOL_KINDS)}",
            yaml_path,
        )
    if len(kinds_present) > 1:
        raise ManifestError(
            f"tools[{idx}] ({name!r}) has multiple kinds set ({kinds_present}); pick one",
            yaml_path,
        )
    kind = kinds_present[0]

    description = _optional_str(entry, "description", yaml_path)
    parameters = entry.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise ManifestError(f"tools[{idx}] ({name!r}).parameters must be a mapping", yaml_path)

    if kind == "cypher":
        cypher = entry["cypher"]
        if not isinstance(cypher, str) or not cypher.strip():
            raise ManifestError(f"tools[{idx}] ({name!r}).cypher must be a non-empty string", yaml_path)
        return CypherTool(name=name, cypher=cypher, description=description, parameters=parameters)

    # kind == "python"
    python = entry["python"]
    function = entry.get("function")
    if not isinstance(python, str) or not python:
        raise ManifestError(f"tools[{idx}] ({name!r}).python must be a non-empty path string", yaml_path)
    if not isinstance(function, str) or not _VALID_TOOL_NAME.match(function):
        raise ManifestError(
            f"tools[{idx}] ({name!r}) python tools need `function:` set to a valid Python identifier",
            yaml_path,
        )
    return PythonTool(
        name=name,
        python=python,
        function=function,
        description=description,
        parameters=parameters,
    )


def _build_embedder(raw: object, yaml_path: Path) -> EmbedderConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ManifestError("embedder must be a mapping", yaml_path)
    unknown = set(raw) - _ALLOWED_EMBEDDER_KEYS
    if unknown:
        raise ManifestError(
            f"unknown embedder keys: {sorted(unknown)}. Allowed: {sorted(_ALLOWED_EMBEDDER_KEYS)}",
            yaml_path,
        )
    module = raw.get("module")
    if not isinstance(module, str) or not module:
        raise ManifestError("embedder.module must be a non-empty string (path or dotted name)", yaml_path)
    klass = raw.get("class")
    if not isinstance(klass, str) or not _VALID_TOOL_NAME.match(klass):
        raise ManifestError(
            f"embedder.class must be a valid identifier matching {_VALID_TOOL_NAME.pattern}",
            yaml_path,
        )
    kwargs = raw.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ManifestError("embedder.kwargs must be a mapping", yaml_path)
    return EmbedderConfig(module=module, klass=klass, kwargs=kwargs)


def _build_builtins(raw: object, yaml_path: Path) -> BuiltinsConfig:
    if raw is None:
        return BuiltinsConfig()
    if not isinstance(raw, dict):
        raise ManifestError("builtins must be a mapping", yaml_path)
    unknown = set(raw) - _ALLOWED_BUILTIN_KEYS
    if unknown:
        raise ManifestError(
            f"unknown builtins keys: {sorted(unknown)}. Allowed: {sorted(_ALLOWED_BUILTIN_KEYS)}",
            yaml_path,
        )
    cfg = BuiltinsConfig()
    if "save_graph" in raw:
        sv = raw["save_graph"]
        if not isinstance(sv, bool):
            raise ManifestError("builtins.save_graph must be a bool", yaml_path)
        cfg.save_graph = sv
    if "temp_cleanup" in raw:
        tc = raw["temp_cleanup"]
        if not isinstance(tc, str) or tc not in _VALID_TEMP_CLEANUP:
            raise ManifestError(
                f"builtins.temp_cleanup must be one of {sorted(_VALID_TEMP_CLEANUP)}",
                yaml_path,
            )
        cfg.temp_cleanup = tc
    return cfg
