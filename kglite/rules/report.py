"""``RuleReport`` — the structured output of running a rule pack.

Aggregates per-rule ``ResultView`` objects lazily. ``.summary`` returns
counts and severity dispositions without materialising rows;
``.violations_for(name)`` returns a single rule's rows; ``.rows``
materialises everything as a pandas DataFrame on first access.

The lazy shape protects LLM callers — accessing ``.summary`` on a
report with 100k violations does not materialise pandas DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class _RuleResult:
    """One rule's contribution to a RuleReport."""

    rule_name: str
    severity: str
    description_for_agent: str
    cypher: str
    used_params: dict[str, str]
    result_view: Any  # ResultView from kglite.kglite
    elapsed_ms: float = 0.0
    truncated: bool = False
    error: str | None = None

    @property
    def violation_count(self) -> int:
        if self.error is not None:
            return 0
        try:
            return len(self.result_view)
        except Exception:
            return 0


@dataclass
class RuleReport:
    """Aggregated structural-violation findings from a rule pack run.

    Most accessors are lazy — ``.summary`` is cheap; ``.rows``
    materialises a DataFrame on first call.
    """

    pack_name: str
    pack_version: str
    rule_results: list[_RuleResult] = field(default_factory=list)
    _rows_cache: Any | None = None  # pd.DataFrame, materialised lazily
    _suspect_index: dict[str, list[tuple[str, str]]] | None = None

    # ------------------------------------------------------------------
    # Cheap accessors

    @property
    def summary(self) -> dict[str, Any]:
        """Counts, severities, per-rule status. No DataFrame materialisation."""
        by_rule: dict[str, dict[str, Any]] = {}
        by_severity: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "blocker": 0}
        total = 0
        errors = 0
        any_truncated = False
        for r in self.rule_results:
            count = r.violation_count
            total += count
            if r.error is None:
                by_severity[r.severity] = by_severity.get(r.severity, 0) + count
            else:
                errors += 1
            if r.truncated:
                any_truncated = True
            by_rule[r.rule_name] = {
                "violations": count,
                "severity": r.severity,
                "truncated": r.truncated,
                "elapsed_ms": r.elapsed_ms,
                "error": r.error,
            }
        return {
            "pack": self.pack_name,
            "version": self.pack_version,
            "total_violations": total,
            "rule_count": len(self.rule_results),
            "rules_with_errors": errors,
            "any_truncated": any_truncated,
            "by_severity": by_severity,
            "by_rule": by_rule,
        }

    @property
    def violation_count(self) -> int:
        return sum(r.violation_count for r in self.rule_results)

    @property
    def has_blockers(self) -> bool:
        """True if any rule with severity=blocker found violations."""
        return any(r.severity == "blocker" and r.violation_count > 0 for r in self.rule_results)

    def violations_for(self, rule_name: str) -> "pd.DataFrame":
        """Return one rule's rows as a DataFrame (materialised on demand)."""
        for r in self.rule_results:
            if r.rule_name == rule_name:
                return _result_view_to_df(r)
        raise KeyError(f"Rule '{rule_name}' not found in report")

    def is_suspect(self, node_id: Any) -> list[tuple[str, str]]:
        """Return ``[(rule_name, severity), ...]`` for rules that flagged ``node_id``.

        Empty list when the node is clean. The index is built lazily on the
        first call and cached. The node id is matched against any column whose
        name ends in ``_id`` or equals ``id``/``node_id`` — works with the
        bundled rule columns and any custom rule that follows the same
        naming convention.
        """
        index = self._suspect_index
        if index is None:
            index = self._build_suspect_index()
            self._suspect_index = index
        return index.get(str(node_id), [])

    def _build_suspect_index(self) -> dict[str, list[tuple[str, str]]]:
        index: dict[str, list[tuple[str, str]]] = {}
        for r in self.rule_results:
            if r.error is not None or r.violation_count == 0:
                continue
            df = _result_view_to_df(r)
            if df.empty:
                continue
            id_columns = [c for c in df.columns if c == "id" or c == "node_id" or c.endswith("_id")]
            if not id_columns:
                continue
            seen_for_rule: set[str] = set()
            for col in id_columns:
                for value in df[col].astype(str):
                    if value in seen_for_rule:
                        continue
                    seen_for_rule.add(value)
                    index.setdefault(value, []).append((r.rule_name, r.severity))
        return index

    # ------------------------------------------------------------------
    # Lazy DataFrame view

    @property
    def rows(self) -> "pd.DataFrame":
        """All violations across all rules, with a leading ``rule_name`` column.

        Materialised on first access and cached. Use ``.violations_for(name)``
        if you only need one rule's rows.
        """
        if self._rows_cache is None:
            self._rows_cache = self._build_rows()
        return self._rows_cache

    def _build_rows(self) -> "pd.DataFrame":
        import pandas as pd

        frames = []
        for r in self.rule_results:
            if r.error is not None or r.violation_count == 0:
                continue
            df = _result_view_to_df(r)
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "severity", r.severity)
            df.insert(0, "rule_name", r.rule_name)
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["rule_name", "severity"])
        return pd.concat(frames, ignore_index=True, sort=False)

    # ------------------------------------------------------------------
    # Markdown export

    def to_markdown(self, sample_rows: int = 5) -> str:
        """Render an agent-pasteable markdown summary.

        ``sample_rows`` controls how many violation rows are previewed
        per rule (after the per-rule headers).
        """
        s = self.summary
        lines: list[str] = []
        lines.append(f"# Rule pack: `{self.pack_name}` v{self.pack_version}")
        lines.append("")
        lines.append(f"**Total violations:** {s['total_violations']:,} across {s['rule_count']} rules")
        sev = s["by_severity"]
        sev_line = ", ".join(f"{k}: {v:,}" for k, v in sev.items() if v > 0)
        if sev_line:
            lines.append(f"**By severity:** {sev_line}")
        if s["rules_with_errors"]:
            lines.append(f"**Rules with errors:** {s['rules_with_errors']}")
        lines.append("")

        for r in self.rule_results:
            count = r.violation_count
            truncated = " (truncated)" if r.truncated else ""
            lines.append(f"## `{r.rule_name}` — {count:,} violations{truncated}")
            lines.append(f"*severity:* `{r.severity}` · *time:* {r.elapsed_ms:.1f}ms")
            if r.error:
                lines.append(f"*error:* `{r.error}`")
                lines.append("")
                continue
            if r.description_for_agent:
                lines.append("")
                lines.append(r.description_for_agent.strip())
            if count > 0 and sample_rows > 0:
                df = _result_view_to_df(r).head(sample_rows)
                if not df.empty:
                    lines.append("")
                    lines.append(_df_to_markdown_table(df))
            lines.append("")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Helpers


def _result_view_to_df(result: _RuleResult) -> "pd.DataFrame":
    """Materialise a per-rule ResultView into a DataFrame."""
    import pandas as pd

    if result.error is not None:
        return pd.DataFrame()
    rv = result.result_view
    if rv is None:
        return pd.DataFrame()
    try:
        return rv.to_df()
    except Exception:
        # Fall back to manual list construction for ResultViews that
        # don't expose to_df directly.
        try:
            return pd.DataFrame(rv.to_list())
        except Exception:
            return pd.DataFrame()


def _df_to_markdown_table(df: "pd.DataFrame") -> str:
    """Minimal markdown table without depending on tabulate."""
    if df.empty:
        return ""
    headers = list(df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("|" + "|".join(" --- " for _ in headers) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_md_cell(row[h]) for h in headers) + " |")
    return "\n".join(lines)


_MD_LIST_PREVIEW = 3  # show this many list elements before truncating with "…"


def _md_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return _format_list_cell(value)
    s = str(value)
    # Some Cypher list values come through as Python-list-shaped strings
    # (e.g. "[1, 2, 3]"). Format them as collections rather than literals.
    if len(s) > 2 and s.startswith("[") and s.endswith("]") and ("," in s or s.count(",") == 0):
        return _format_list_cell_str(s)
    return s.replace("|", "\\|").replace("\n", " ")


def _format_list_cell(items: list) -> str:
    """Render a Python list as 'a, b, c (+N more)' for markdown tables."""
    n = len(items)
    if n <= _MD_LIST_PREVIEW:
        body = ", ".join(_md_cell_inner(i) for i in items)
    else:
        head = ", ".join(_md_cell_inner(i) for i in items[:_MD_LIST_PREVIEW])
        body = f"{head} (+{n - _MD_LIST_PREVIEW} more)"
    return body.replace("|", "\\|").replace("\n", " ")


def _format_list_cell_str(s: str) -> str:
    """Best-effort split of a list-literal string into a friendlier cell."""
    inner = s[1:-1].strip()
    if not inner:
        return s
    parts = [p.strip() for p in inner.split(",")]
    return _format_list_cell(parts)


def _md_cell_inner(value: Any) -> str:
    s = str(value)
    return s.replace("|", "\\|").replace("\n", " ")
