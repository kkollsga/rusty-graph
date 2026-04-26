"""Rule + RulePack data model and YAML loader.

A YAML rule pack file looks like::

    name: structural_integrity
    version: "1.0"
    description: |
      Universal structural validators.
    rules:
      - name: orphan_node
        description_for_agent: |
          A node with zero edges. Likely an ingest artifact.
        severity: medium
        parameters:
          type: string
        match: |
          MATCH (n:{type})
          WHERE NOT EXISTS { (n)--() }
        columns:
          - n.id AS node_id
          - n.title AS title

The loader validates structure and rejects rules whose first MATCH is
unanchored (no node label) unless ``unsafe_unanchored: true`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import re
from typing import Any

import yaml  # type: ignore[import-untyped]

DEFAULT_LIMIT = 1000

_VALID_SEVERITIES = frozenset({"low", "medium", "high", "blocker"})
_VALID_DIRECTIONS = frozenset({"", "outbound", "inbound"})


@dataclass(frozen=True)
class Rule:
    """A single structural validator within a rule pack."""

    name: str
    description: str
    description_for_agent: str
    severity: str
    parameters: dict[str, str]  # name -> declared type (informational)
    match: str  # match clause body — substituted with {placeholder} params
    columns: tuple[str, ...]  # RETURN-clause expressions
    default_limit: int = DEFAULT_LIMIT
    unsafe_unanchored: bool = False
    # When set, the runner verifies that the (``type``, ``edge``) parameter
    # pair represents an actual edge in the graph in the requested
    # direction before executing. Prevents trivial rule firing when the
    # author picked the wrong direction (e.g. asking for incoming
    # IN_LICENCE on a Wellbore — the edge always flows outward, so the
    # rule would match every Wellbore meaninglessly).
    validates_direction: str = ""
    # Per-rule Cypher timeout in milliseconds. ``None`` (default) means
    # use the graph's default timeout. Set this on rules known to be
    # expensive on large graphs — e.g. orphan_node on a 13M-node type
    # needs more than the standard 60s budget to finish. Caller-supplied
    # ``timeout_ms`` to ``g.rules.run(...)`` always wins when both are set.
    default_timeout_ms: int | None = None


@dataclass(frozen=True)
class RulePack:
    """A named, versioned collection of rules."""

    name: str
    version: str
    description: str
    rules: tuple[Rule, ...]
    usage_hint: str = ""

    @property
    def rule_names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.rules)

    def get_rule(self, name: str) -> Rule:
        for rule in self.rules:
            if rule.name == name:
                return rule
        raise KeyError(f"Rule '{name}' not found in pack '{self.name}'")


class RulePackLoadError(ValueError):
    """Raised when a YAML file cannot be loaded as a RulePack."""


def load_pack(path: str | Path) -> RulePack:
    """Load a YAML rule pack from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rule pack file not found: {path}")
    return loads_pack(path.read_text(encoding="utf-8"), source=str(path))


def loads_pack(text: str, source: str = "<string>") -> RulePack:
    """Parse a YAML string into a RulePack."""
    raw = yaml.safe_load(text)
    return _build_pack(raw, source=source)


def load_bundled(name: str) -> RulePack:
    """Load a rule pack bundled inside ``kglite.rules.packs``."""
    pkg = files("kglite.rules.packs")
    candidate = pkg / f"{name}.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(f"Bundled rule pack '{name}' not found")
    return loads_pack(
        candidate.read_text(encoding="utf-8"),
        source=f"kglite.rules.packs/{name}.yaml",
    )


def peek_bundled(name: str) -> dict[str, Any] | None:
    """Read header-level metadata from a bundled YAML pack without full validation.

    Returns ``{name, version, description, rule_count, usage_hint}`` or
    ``None`` when the YAML can't be parsed. Cheaper than ``load_bundled``
    and tolerant of malformed rule bodies — used by ``g.rules.list()`` to
    expose pack inventory before any pack has been loaded.
    """
    pkg = files("kglite.rules.packs")
    candidate = pkg / f"{name}.yaml"
    if not candidate.is_file():
        return None
    try:
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    rules = raw.get("rules") or []
    return {
        "name": str(raw.get("name", name)),
        "version": str(raw.get("version", "?")),
        "description": str(raw.get("description", "")),
        "rule_count": len(rules) if isinstance(rules, list) else 0,
        "usage_hint": str(raw.get("usage_hint", "")),
    }


def _require(d: dict, key: str, context: str) -> Any:
    if key not in d:
        raise RulePackLoadError(f"Missing required key '{key}' in {context}")
    return d[key]


def _build_pack(raw: Any, source: str) -> RulePack:
    if not isinstance(raw, dict):
        raise RulePackLoadError(f"Pack file must be a YAML mapping at top level (in {source})")
    name = str(_require(raw, "name", source))
    version = str(_require(raw, "version", source))
    description = str(raw.get("description", ""))
    usage_hint = str(raw.get("usage_hint", ""))
    rules_raw = _require(raw, "rules", source)
    if not isinstance(rules_raw, list) or not rules_raw:
        raise RulePackLoadError(f"'rules' must be a non-empty list (in {source})")
    rules = tuple(_build_rule(r, source) for r in rules_raw)
    return RulePack(
        name=name,
        version=version,
        description=description,
        rules=rules,
        usage_hint=usage_hint,
    )


def _build_rule(raw: Any, source: str) -> Rule:
    if not isinstance(raw, dict):
        raise RulePackLoadError(f"Each rule must be a mapping (in {source})")
    name = str(_require(raw, "name", f"rule in {source}"))
    description = str(raw.get("description", ""))
    description_for_agent = str(raw.get("description_for_agent", description))
    severity = str(raw.get("severity", "medium"))
    if severity not in _VALID_SEVERITIES:
        raise RulePackLoadError(f"Rule '{name}' has invalid severity '{severity}'. Valid: {sorted(_VALID_SEVERITIES)}")
    parameters_raw = raw.get("parameters") or {}
    if not isinstance(parameters_raw, dict):
        raise RulePackLoadError(f"Rule '{name}' parameters must be a mapping")
    parameters = {str(k): str(v) for k, v in parameters_raw.items()}

    match = _require(raw, "match", f"rule '{name}' in {source}")
    if not isinstance(match, str) or not match.strip():
        raise RulePackLoadError(f"Rule '{name}' match must be a non-empty string")

    columns_raw = raw.get("columns", [])
    columns: tuple[str, ...]
    if isinstance(columns_raw, str):
        columns = (columns_raw,)
    elif isinstance(columns_raw, list):
        columns = tuple(str(c) for c in columns_raw)
    else:
        raise RulePackLoadError(f"Rule '{name}' columns must be a string or list of strings")

    default_limit_raw = raw.get("default_limit", DEFAULT_LIMIT)
    try:
        default_limit = int(default_limit_raw)
    except (TypeError, ValueError) as e:
        raise RulePackLoadError(f"Rule '{name}' default_limit must be an integer") from e
    if default_limit < 1:
        raise RulePackLoadError(f"Rule '{name}' default_limit must be >= 1")

    unsafe_unanchored = bool(raw.get("unsafe_unanchored", False))

    if not unsafe_unanchored:
        _validate_anchoring(name, match)

    default_timeout_raw = raw.get("default_timeout_ms")
    default_timeout_ms: int | None
    if default_timeout_raw is None:
        default_timeout_ms = None
    else:
        try:
            default_timeout_ms = int(default_timeout_raw)
        except (TypeError, ValueError) as e:
            raise RulePackLoadError(f"Rule '{name}' default_timeout_ms must be an integer or null") from e
        if default_timeout_ms < 0:
            raise RulePackLoadError(f"Rule '{name}' default_timeout_ms must be >= 0 (0 = unbounded)")

    validates_direction = str(raw.get("validates_direction", "")).strip()
    if validates_direction not in _VALID_DIRECTIONS:
        raise RulePackLoadError(
            f"Rule '{name}' has invalid validates_direction "
            f"'{validates_direction}'. Valid: 'outbound', 'inbound', or omit."
        )
    if validates_direction:
        # Direction validation requires both 'type' and 'edge' parameters.
        for required in ("type", "edge"):
            if required not in parameters:
                raise RulePackLoadError(
                    f"Rule '{name}' declares validates_direction but is "
                    f"missing required parameter '{required}'. The runner "
                    f"validates that the (type, edge) pair points in the "
                    f"requested direction in the graph's actual schema."
                )

    return Rule(
        name=name,
        description=description,
        description_for_agent=description_for_agent,
        severity=severity,
        parameters=parameters,
        match=match,
        columns=columns,
        default_limit=default_limit,
        unsafe_unanchored=unsafe_unanchored,
        validates_direction=validates_direction,
        default_timeout_ms=default_timeout_ms,
    )


# Match the first node pattern after a MATCH keyword. Captures:
#   group 1: optional identifier (e.g., 'n')
#   group 2: optional ":Label" or ":{template}"
_FIRST_NODE_PATTERN = re.compile(r"\(\s*([A-Za-z_]\w*)?\s*(:[^\s)]+)?")
_MATCH_KEYWORD = re.compile(r"\bmatch\b", re.IGNORECASE)


def _validate_anchoring(rule_name: str, match: str) -> None:
    """Reject MATCH clauses whose first node has no label or template.

    Only checks the *first* MATCH; subsequent MATCHes typically join
    onto the first via shared variables, so the planner will use the
    anchored first MATCH as its starting point.

    Rules that use CALL (e.g. ``connected_components``) without any
    MATCH are accepted — those operations have their own bounded
    semantics defined by the procedure.
    """
    keyword = _MATCH_KEYWORD.search(match)
    if keyword is None:
        # CALL-only or other MATCH-less rule. Trust it.
        return
    after_match = match[keyword.end() :]
    node = _FIRST_NODE_PATTERN.search(after_match)
    if node is None:
        raise RulePackLoadError(f"Rule '{rule_name}' has an unparseable MATCH (no node pattern found)")
    if not node.group(2):
        raise RulePackLoadError(
            f"Rule '{rule_name}' MATCH starts with an unanchored node "
            f"pattern '{node.group(0).strip()}'. Add a node label "
            f"(e.g. '(n:Type)' or '(n:{{type}})'), or set "
            f"'unsafe_unanchored: true' to override (only safe on small graphs)."
        )
