"""0.9.0 gate item §1 — better Cypher error messages.

Today the parser emits token-level errors with no source positions
and no distinction between "you typo'd" and "feature not yet
supported". The agent's recent UX feedback singled this out as the
gating item that compounds with the dialect gaps in §2/§3/§6 — when
the user can't tell "wrong" from "not implemented", every gap feels
twice as rough.

Target behavior (post-§1):
1. Errors carry source offsets (line + column).
2. Known unimplemented features ("NULLS LAST", "n.d.year",
   "size((:A)-[:R]->(:B))") emit intent-level messages naming
   the feature, not raw tokens.
3. Errors include a caret-pointing excerpt or column number.

These tests are xfail-strict — see test_v0_9_05_integer_division.py
header for the workflow.

Note: tests in this file may also begin to pass as §2/§3/§6 land
(because the test query no longer errors). When that happens, this
file's xfail markers will flip in lockstep with §1's actual fix —
which is fine; §1 lands last per the readiness doc precisely
because it benefits from the prior fixes.
"""

from __future__ import annotations

import pytest

import kglite

NOT_IMPLEMENTED = (
    "0.9.0 §1 — Cypher error messages are token-level today; flip when "
    "source-offset + intent-message + caret-pointing land."
)


def _try_cypher_capture_error(g: kglite.KnowledgeGraph, query: str) -> str:
    """Run a query expected to fail; return the error message string."""
    try:
        # Force materialization in case the API is lazy
        list(g.cypher(query))
    except Exception as e:  # noqa: BLE001 — we want any error
        return str(e)
    pytest.fail(f"expected an error from query: {query!r}")


# ---------------------------------------------------------------------------
# Source offsets — every parser error should know where it is.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_error_carries_line_number():
    g = kglite.KnowledgeGraph()
    msg = _try_cypher_capture_error(g, "MATCH (n)\nRETURN n.))")
    # Accept either explicit "line 2" or "2:" position prefix
    assert "line 2" in msg.lower() or "2:" in msg, f"missing line info: {msg!r}"


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_error_carries_column_number():
    g = kglite.KnowledgeGraph()
    msg = _try_cypher_capture_error(g, "RETURN ))")
    # Either column number or caret marker
    assert "col" in msg.lower() or "column" in msg.lower() or "^" in msg, f"missing column info: {msg!r}"


# ---------------------------------------------------------------------------
# Intent-level messages for known unimplemented features.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_intent_message_for_nulls_last():
    """The query is a known-unimplemented feature (§2). The error
    should name it, not point at the token."""
    g = kglite.KnowledgeGraph()
    msg = _try_cypher_capture_error(g, "MATCH (n) RETURN n ORDER BY n.x DESC NULLS LAST").lower()
    # Either: feature names the gap, or the feature lands in §2 first
    # (in which case this test no longer errors and xfail flips).
    assert "nulls" in msg, f"error doesn't mention NULLS: {msg!r}"
    assert "not yet" in msg or "not implemented" in msg or "unsupported" in msg, (
        f"error doesn't read as intent-level: {msg!r}"
    )


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_intent_message_for_datetime_accessor():
    """`n.d.year` is §3. Same shape: error names the feature."""
    g = kglite.KnowledgeGraph()
    msg = _try_cypher_capture_error(g, "MATCH (n:X) RETURN n.joined.year AS y").lower()
    # Expect message mentions datetime / temporal / accessor
    assert any(kw in msg for kw in ("datetime", "temporal", "accessor", "not yet", "not implemented")), (
        f"error doesn't read as intent-level: {msg!r}"
    )


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_intent_message_for_pattern_in_size():
    """`size((:A)-[:R]->(:B))` is §6."""
    g = kglite.KnowledgeGraph()
    msg = _try_cypher_capture_error(g, "RETURN size((:A)-[:R]->(:B)) AS n").lower()
    # Expect message to read as intent-level rather than "Unexpected token: Colon"
    assert "pattern" in msg or "not yet" in msg or "not implemented" in msg, (
        f"error doesn't read as intent-level: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Caret pointing for typos.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED)
def test_caret_or_column_for_typo_extra_paren():
    """The user's reported case: typo'd extra `)` in CASE produces a
    parser error that should locate the bad `)`, not say "Expected
    Then, found RParen" with no positional info."""
    g = kglite.KnowledgeGraph()
    msg = _try_cypher_capture_error(g, "MATCH (n:X) RETURN CASE WHEN count(n)) THEN 1 END")
    # Either caret marker or numeric column
    assert "^" in msg or "col" in msg.lower() or "column" in msg.lower(), f"no positional info on typo error: {msg!r}"
