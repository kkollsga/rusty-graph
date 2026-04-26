"""``_RulesAccessor`` — the object returned by ``g.rules``.

Loads, lists, runs, and describes rule packs against a single graph.
Wraps a small LRU cache keyed on ``(pack_name, frozen_params,
id(graph))`` so re-running the same pack within a session returns the
cached ``RuleReport`` without re-issuing Cypher queries.
"""

from __future__ import annotations

import builtins
from collections import OrderedDict
import time
from typing import TYPE_CHECKING, Any

from .pack import RulePack, load_bundled, load_pack, peek_bundled
from .report import RuleReport, _RuleResult
from .runner import compile_rule

if TYPE_CHECKING:
    from kglite.kglite import KnowledgeGraph

# Bundled packs auto-discovered when the user calls ``run("name")`` on
# a pack that hasn't been explicitly ``.load()``-ed.
_BUNDLED_NAMES: tuple[str, ...] = ("structural_integrity",)

_CACHE_CAPACITY = 16


class _RulesAccessor:
    """The object returned by ``g.rules``.

    Methods
    -------
    list()
        List loaded and bundled rule packs.
    load(path)
        Load a YAML rule pack from disk and register it.
    run(name, only=None, limit=None, **params)
        Execute a pack and return a :class:`RuleReport`.
    describe(name)
        Return a dict describing a pack and its rules (LLM-facing).
    """

    def __init__(self, graph: "KnowledgeGraph"):
        self._graph = graph
        self._packs: dict[str, RulePack] = {}
        self._cache: OrderedDict[tuple, RuleReport] = OrderedDict()
        # No initial XML push — rule pack discovery is opt-in. The first
        # `load()` / `run()` call activates per-graph advertising.

    # ------------------------------------------------------------------
    # Public API

    def list(self) -> builtins.list[dict[str, Any]]:
        """Return inventory of loaded + bundled-but-not-yet-loaded packs.

        Bundled packs that haven't been loaded yet are peeked from disk
        (header-only, no rule validation) so the inventory reports real
        version, description, and rule_count rather than placeholders.
        """
        loaded: dict[str, dict[str, Any]] = {
            name: {
                "name": name,
                "version": pack.version,
                "description": pack.description,
                "usage_hint": getattr(pack, "usage_hint", ""),
                "rule_count": len(pack.rules),
                "loaded": True,
            }
            for name, pack in self._packs.items()
        }
        for bundled in _BUNDLED_NAMES:
            if bundled in loaded:
                continue
            peek = peek_bundled(bundled)
            if peek is not None:
                loaded[bundled] = {
                    "name": peek["name"],
                    "version": peek["version"],
                    "description": peek["description"],
                    "usage_hint": peek["usage_hint"],
                    "rule_count": peek["rule_count"],
                    "loaded": False,
                }
            else:
                loaded[bundled] = {
                    "name": bundled,
                    "version": "(unavailable)",
                    "description": "(bundled YAML could not be peeked)",
                    "usage_hint": "",
                    "rule_count": -1,
                    "loaded": False,
                }
        return list(loaded.values())

    def load(self, path: str) -> RulePack:
        """Load a YAML rule pack file and register it under its name."""
        pack = load_pack(path)
        self._packs[pack.name] = pack
        self._cache.clear()  # invalidate; new rules might shadow cached
        self._push_pack_xml()
        return pack

    def describe(self, name: str) -> dict[str, Any]:
        """Return a dict describing the pack and each rule (LLM-facing)."""
        pack = self._resolve_pack(name)
        return {
            "name": pack.name,
            "version": pack.version,
            "description": pack.description,
            "usage_hint": pack.usage_hint,
            "rules": [
                {
                    "name": r.name,
                    "severity": r.severity,
                    "description_for_agent": r.description_for_agent,
                    "parameters": dict(r.parameters),
                }
                for r in pack.rules
            ],
        }

    def run(
        self,
        name: str,
        *,
        only: builtins.list[str] | None = None,
        limit: int | None = None,
        timeout_ms: int | None = None,
        **params: Any,
    ) -> RuleReport:
        """Execute a rule pack and return a structured report.

        Parameters
        ----------
        name
            Rule pack name (loaded packs first, then bundled).
        only
            If given, only run the rules with these names.
        limit
            Override every rule's ``default_limit`` for this run.
        timeout_ms
            Per-rule Cypher timeout in milliseconds. Wins over each
            rule's ``default_timeout_ms`` when both are set. ``None``
            (default) defers to the rule's ``default_timeout_ms`` and
            then to the graph's default timeout.
        **params
            Rule parameters (string keys substituted into ``{...}``
            placeholders in match/columns).
        """
        pack = self._resolve_pack(name)
        rules_to_run = [pack.get_rule(rn) for rn in only] if only is not None else list(pack.rules)

        cache_key = self._cache_key(pack, params, only, limit, timeout_ms)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        # Cache schema info per-run; cheap if already cached on the graph
        # but a single call avoids redundant work across rules.
        edge_schema: dict[str, dict[str, Any]] | None = None

        results: builtins.list[_RuleResult] = []
        for rule in rules_to_run:
            cypher, used = compile_rule(rule, dict(params), limit=limit)
            t0 = time.perf_counter()
            error: str | None = None
            rv: Any = None

            # Direction-aware pre-check: if the rule declares which way an
            # edge should flow, verify the (type, edge) pair is consistent
            # with the graph's actual schema before issuing Cypher.
            if rule.validates_direction:
                if edge_schema is None:
                    edge_schema = self._build_edge_schema()
                error = _validate_direction(rule, used, edge_schema)

            effective_timeout_ms = timeout_ms if timeout_ms is not None else rule.default_timeout_ms

            if error is None:
                try:
                    if effective_timeout_ms is not None:
                        rv = self._graph.cypher(cypher, timeout_ms=effective_timeout_ms)
                    else:
                        rv = self._graph.cypher(cypher)
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            count = 0
            if rv is not None:
                try:
                    count = len(rv)
                except Exception:
                    count = 0
            effective_limit = limit if limit is not None else rule.default_limit
            truncated = count >= effective_limit

            results.append(
                _RuleResult(
                    rule_name=rule.name,
                    severity=rule.severity,
                    description_for_agent=rule.description_for_agent,
                    cypher=cypher,
                    used_params=used,
                    result_view=rv,
                    elapsed_ms=elapsed_ms,
                    truncated=truncated,
                    error=error,
                )
            )

        report = RuleReport(
            pack_name=pack.name,
            pack_version=pack.version,
            rule_results=results,
        )
        self._store_in_cache(cache_key, report)
        self._push_pack_xml()
        return report

    # ------------------------------------------------------------------
    # Internals

    def _push_pack_xml(self) -> None:
        """Re-render this accessor's pack XML and push it to the graph.

        The Rust core stores the string and splices it into ``describe()``
        XML; cached read on every call. Failures here are swallowed —
        agent discovery via ``describe()`` will fall back to the
        module-level cold default (or no block) without raising.
        """
        try:
            from .render import render_rule_packs_xml

            xml = render_rule_packs_xml(self)
            setter = getattr(self._graph, "_set_rule_pack_xml", None)
            if setter is not None:
                setter(xml or None)
        except Exception:
            pass

    def _resolve_pack(self, name: str) -> RulePack:
        if name in self._packs:
            return self._packs[name]
        if name in _BUNDLED_NAMES:
            pack = load_bundled(name)
            self._packs[name] = pack
            return pack
        raise KeyError(
            f"Rule pack '{name}' not loaded. Available: {sorted(set(self._packs.keys()) | set(_BUNDLED_NAMES))}"
        )

    def _cache_key(
        self,
        pack: RulePack,
        params: dict[str, Any],
        only: builtins.list[str] | None,
        limit: int | None,
        timeout_ms: int | None,
    ) -> tuple:
        frozen_params = tuple(sorted(params.items()))
        frozen_only = tuple(only) if only is not None else None
        return (
            pack.name,
            pack.version,
            frozen_params,
            frozen_only,
            limit,
            timeout_ms,
            id(self._graph),
        )

    def _store_in_cache(self, key: tuple, report: RuleReport) -> None:
        self._cache[key] = report
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_CAPACITY:
            self._cache.popitem(last=False)

    def _build_edge_schema(self) -> dict[str, dict[str, Any]]:
        """Index of edge type → ``{source_types, target_types, count}`` from the graph."""
        try:
            entries = self._graph.connection_types() or []
        except Exception:
            return {}
        schema: dict[str, dict[str, Any]] = {}
        for entry in entries:
            edge_type = entry.get("type")
            if not edge_type:
                continue
            schema[edge_type] = {
                "source_types": set(entry.get("source_types") or []),
                "target_types": set(entry.get("target_types") or []),
                "count": entry.get("count", 0),
            }
        return schema


def _validate_direction(
    rule,
    params: dict[str, str],
    edge_schema: dict[str, dict[str, Any]],
) -> str | None:
    """Return an error string when the rule's (type, edge) direction is wrong.

    ``None`` means validation passed (or was skipped because the edge type
    isn't yet present in the graph — empty/partial graphs shouldn't trip
    the validator).
    """
    node_type = params.get("type")
    edge_type = params.get("edge")
    if not node_type or not edge_type:
        return None  # Compiler will already have raised if these were missing
    edge_info = edge_schema.get(edge_type)
    if edge_info is None:
        # Edge type doesn't exist in the graph at all — graph might be
        # empty/building. Don't second-guess the author.
        return None
    sources = sorted(edge_info["source_types"])
    targets = sorted(edge_info["target_types"])
    flows = f"{_type_list(sources)} → {_type_list(targets)}"

    if rule.validates_direction == "outbound":
        if node_type in edge_info["source_types"]:
            return None
        # The user wants outgoing E from T but E never starts at T.
        # If T appears as a target → suggest the inbound rule.
        if node_type in edge_info["target_types"]:
            return (
                f"DirectionMismatch: rule '{rule.name}' expects "
                f"'{edge_type}' to flow OUTWARD from '{node_type}', but "
                f"'{edge_type}' flows {flows} — '{node_type}' is on the "
                f"target side. Use 'missing_inbound_edge' instead, with "
                f"the same parameters."
            )
        return (
            f"DirectionMismatch: rule '{rule.name}' expects "
            f"'{edge_type}' to flow OUTWARD from '{node_type}', but "
            f"'{edge_type}' flows {flows} — '{node_type}' isn't a "
            f"source for this edge. Set type to one of {sources}, or "
            f"pick a different edge."
        )
    if rule.validates_direction == "inbound":
        if node_type in edge_info["target_types"]:
            return None
        if node_type in edge_info["source_types"]:
            return (
                f"DirectionMismatch: rule '{rule.name}' expects "
                f"'{edge_type}' to flow INTO '{node_type}', but "
                f"'{edge_type}' flows {flows} — '{node_type}' is on the "
                f"source side. Use 'missing_required_edge' instead, with "
                f"the same parameters."
            )
        return (
            f"DirectionMismatch: rule '{rule.name}' expects "
            f"'{edge_type}' to flow INTO '{node_type}', but "
            f"'{edge_type}' flows {flows} — '{node_type}' isn't a "
            f"target for this edge. Set type to one of {targets}, or "
            f"pick a different edge."
        )
    return None


def _type_list(types: builtins.list[str]) -> str:
    """Format a sorted list of type names for an error message."""
    if not types:
        return "(none)"
    if len(types) == 1:
        return types[0]
    return "{" + ", ".join(types) + "}"
