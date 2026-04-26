"""Render the ``<rule_packs>`` XML block consumed by ``g.describe()``.

The output is a pre-formatted XML string that the Rust ``compute_description``
function splices in before the closing ``</graph>`` tag. Two modes:

- **Cold (no accessor)** — peek bundled YAML headers and emit a pack-only
  inventory. Used as the module-level default that ``kglite.rules`` pushes
  on import via ``_set_default_rule_pack_xml``.
- **Warm (accessor present)** — emit pack inventory plus per-rule schemas
  and last-run summaries from the accessor's cache. Pushed per-graph via
  ``KnowledgeGraph._set_rule_pack_xml`` after each ``load()`` / ``run()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .accessor import _RulesAccessor


def render_rule_packs_xml(accessor: "_RulesAccessor | None") -> str:
    """Build the ``<rule_packs>...</rule_packs>`` block as a string.

    Returns the empty string when there's nothing to advertise (no bundled
    packs found, no loaded packs). The caller is responsible for splicing
    the result into the surrounding ``<graph>`` document.
    """
    from .accessor import _BUNDLED_NAMES

    packs: list[dict] = []
    last_runs: dict = {}

    if accessor is not None:
        packs = accessor.list()
        cache = getattr(accessor, "_cache", {})
        for key, report in cache.items():
            pack_name = key[0] if isinstance(key, tuple) and key else None
            if pack_name and pack_name not in last_runs:
                last_runs[pack_name] = report.summary
    else:
        # No accessor created yet — peek bundled YAML headers so an
        # agent's first describe() call surfaces real version + rule_count.
        from .pack import peek_bundled

        for name in _BUNDLED_NAMES:
            peek = peek_bundled(name)
            if peek is None:
                packs.append(
                    {
                        "name": name,
                        "version": "(unavailable)",
                        "rule_count": -1,
                        "loaded": False,
                        "usage_hint": "",
                    }
                )
            else:
                packs.append(
                    {
                        "name": peek["name"],
                        "version": peek["version"],
                        "rule_count": peek["rule_count"],
                        "loaded": False,
                        "usage_hint": peek["usage_hint"],
                    }
                )

    if not packs:
        return ""

    # When the accessor is available, surface per-rule details for any
    # loaded pack so agents can introspect parameter schemas without a
    # second tool call. Bundled packs that haven't been loaded skip
    # this — the lazy peek doesn't include rule details.
    rule_details: dict[str, list[dict]] = {}
    if accessor is not None:
        for pack_name, pack in accessor._packs.items():
            rule_details[pack_name] = [
                {
                    "name": r.name,
                    "severity": r.severity,
                    "params": ",".join(r.parameters.keys()),
                }
                for r in pack.rules
            ]

    lines = ["  <rule_packs>\n"]
    for pack_info in packs:
        name = pack_info["name"]
        ver = pack_info["version"]
        loaded = "true" if pack_info["loaded"] else "false"
        rule_count = pack_info["rule_count"]
        usage_hint = pack_info.get("usage_hint", "")
        attrs = [
            f'name="{_xml_escape(name)}"',
            f'version="{_xml_escape(ver)}"',
            f'loaded="{loaded}"',
            f'rule_count="{rule_count}"',
        ]
        last = last_runs.get(name)
        if last is not None:
            total = last.get("total_violations", 0)
            sev = last.get("by_severity", {})
            sev_str = ",".join(f"{k}={v}" for k, v in sev.items() if v)
            attrs.append(f'last_run_violations="{total}"')
            attrs.append(f'last_run_severity="{_xml_escape(sev_str)}"')
            if last.get("any_truncated"):
                attrs.append('last_run_truncated="true"')
        if usage_hint.strip():
            attrs.append(f'usage_hint="{_xml_escape(usage_hint.strip())}"')

        rules = rule_details.get(name)
        if rules:
            lines.append(f"    <pack {' '.join(attrs)}>\n")
            for r in rules:
                params_str = r["params"] or "(none)"
                lines.append(
                    f'      <rule name="{_xml_escape(r["name"])}" '
                    f'severity="{r["severity"]}" '
                    f'params="{_xml_escape(params_str)}"/>\n'
                )
            lines.append("    </pack>\n")
        else:
            lines.append(f"    <pack {' '.join(attrs)}/>\n")
    lines.append("    <hint>Run packs with g.rules.run(name); per-rule schemas appear once a pack is loaded.</hint>\n")
    lines.append("  </rule_packs>\n")
    return "".join(lines)


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("\n", " ")
