"""Rule compiler — turns a Rule + params dict into an executable Cypher query.

The compiler substitutes ``{placeholder}`` tokens in the rule's match
clause and column expressions, appends a RETURN with the configured
columns, and bounds the result with ``LIMIT N``. The default LIMIT
(1000) protects agents from accidentally materialising large violation
sets; callers can override per ``run()`` invocation.
"""

from __future__ import annotations

import re

from .pack import Rule

# Match {placeholder} where placeholder is a bare identifier.
# Cypher map syntax (``{prop: 'x'}``) and parameter syntax (``$name``)
# both fail to match this pattern, so they pass through untouched.
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class RuleParameterError(ValueError):
    """Raised when required parameters are missing at compile time."""


def compile_rule(
    rule: Rule,
    params: dict[str, str] | None = None,
    *,
    limit: int | None = None,
) -> tuple[str, dict[str, str]]:
    """Compile a Rule into a complete Cypher query.

    Returns ``(cypher, used_params)`` — the used_params is the subset
    of supplied params that were actually substituted, suitable for
    cache keying.
    """
    supplied = dict(params or {})

    placeholders: set[str] = set(_PLACEHOLDER.findall(rule.match))
    for col in rule.columns:
        placeholders.update(_PLACEHOLDER.findall(col))

    missing = placeholders - set(supplied.keys())
    if missing:
        raise RuleParameterError(f"Rule '{rule.name}' missing required parameters: {sorted(missing)}")

    used = {key: str(supplied[key]) for key in placeholders}

    match_body = _substitute(rule.match, used).rstrip()
    if rule.columns:
        columns_body = ", ".join(_substitute(c, used) for c in rule.columns)
    else:
        columns_body = "*"

    effective_limit = limit if limit is not None else rule.default_limit
    if effective_limit < 1:
        raise RuleParameterError(f"limit must be >= 1 (got {effective_limit})")

    cypher = f"{match_body}\nRETURN {columns_body}\nLIMIT {effective_limit}"
    return cypher, used


def _substitute(template: str, params: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda m: params[m.group(1)], template)
