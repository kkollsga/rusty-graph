"""Rule packs — agent-discoverable structural validators.

A rule pack is a named collection of structural validators (orphan-node
checks, cycle detection, missing-parent, etc.) that compile to Cypher
queries and produce a structured ``RuleReport``. Packs are loaded from
YAML files; the bundled ``structural_integrity`` pack ships universal
rules that work on any property graph.

Usage::

    g = kglite.load("legal")
    g.rules.list()                       # discover available packs
    report = g.rules.run("structural_integrity", type="LawSection")
    report.summary                       # counts + severities
    report.violations_for("orphan_node") # rows for a single rule

See ``docs/guides/rules.md`` for full documentation.
"""

from .pack import (
    DEFAULT_LIMIT,
    Rule,
    RulePack,
    RulePackLoadError,
    load_bundled,
    load_pack,
    loads_pack,
)
from .render import render_rule_packs_xml
from .report import RuleReport

__all__ = [
    "DEFAULT_LIMIT",
    "Rule",
    "RulePack",
    "RulePackLoadError",
    "RuleReport",
    "advertise",
    "load_bundled",
    "load_pack",
    "loads_pack",
    "render_rule_packs_xml",
]


def advertise() -> None:
    """Enable `<rule_packs>` advertising in `g.describe()` for cold graphs.

    By default rule packs are silent: a fresh `kglite.load(...)` produces
    a `describe()` with no `<rule_packs>` block until the user calls
    `g.rules.run(...)` (which advertises that one graph) or this function
    (which sets a module-level default visible to every subsequent
    `describe()` call across all graphs).

    Idempotent — safe to call multiple times. No-op if the bundled
    YAML can't be peeked.
    """
    try:
        from kglite.kglite import _set_default_rule_pack_xml
    except ImportError:
        return  # Native module not yet importable (e.g. stubgen).
    try:
        xml = render_rule_packs_xml(None)
    except Exception:
        return  # Bundled YAML missing / unreadable.
    _set_default_rule_pack_xml(xml or None)


def _disable_advertising() -> None:
    """Internal: clear the module-level default. Used by tests."""
    try:
        from kglite.kglite import _set_default_rule_pack_xml
    except ImportError:
        return
    _set_default_rule_pack_xml(None)
