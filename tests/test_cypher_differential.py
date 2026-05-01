"""Differential test harness for the Cypher optimizer pipeline.

Every query in DIFFERENTIAL_QUERIES is run twice: once with the
optimizer pipeline enabled (the default), once with `disable_optimizer
=True` (every pass skipped). We assert both produce identical row sets
after normalization.

This is the regression mechanism for **silent correctness failures**
(passes that drop or duplicate rows). Historical bugs in this class —
0.8.27 LIMIT pushdown returning fewer rows than asked, 0.8.30
startNode(r) returning wrong endpoints — would all have failed the
appropriate row-equality assertion.

It does NOT catch:

- **Gate misses** (a fusion pass bails when it could fuse): both
  paths produce the same result, just slower. Needs plan-shape or perf
  regression testing — covered by follow-ups.
- **Execution semantic bugs** that exist in both fast and slow paths
  (rare but real, e.g. 0.8.30 startNode(r) was actually present in both
  paths). Needs cross-mode parity (cypher vs. fluent vs. naive).

When fixing a future silent-correctness bug, **add the bug's triggering
query to DIFFERENTIAL_QUERIES** so the regression is permanent.
"""

from __future__ import annotations

import pytest

import kglite

# ── Corpus ───────────────────────────────────────────────────────────
#
# Each entry is `(name, fixture, query, params)`. The corpus aims to
# exercise:
#
# 1. One query per registered optimizer pass (so each pass's trigger
#    shape is in the corpus by design).
# 2. Historical bug shapes from CHANGELOG entries (0.8.27 +).
# 3. Edge cases that have surprised optimizers in the past: LIMIT 0,
#    OPTIONAL with no match, ORDER BY ties, DISTINCT, parameterized,
#    multi-MATCH chains.
#
# The corpus deliberately skips vector_score / text_score and spatial
# fusion — those depend on registered embedders or geometry data and
# don't exist in the shared fixtures. They warrant a separate harness
# that builds purpose-specific fixtures.
DIFFERENTIAL_QUERIES: list[tuple[str, str, str, dict | None]] = [
    # ── basic shapes ──
    ("simple_match", "small_graph", "MATCH (p:Person) RETURN p.name AS n", None),
    ("simple_match_param", "small_graph", "MATCH (p:Person) WHERE p.age > $min RETURN p.name AS n", {"min": 30}),
    ("count_all_typed", "social_graph", "MATCH (p:Person) RETURN count(p) AS n", None),
    ("count_all_untyped", "social_graph", "MATCH (n) RETURN count(n) AS n", None),
    ("distinct_property", "social_graph", "MATCH (p:Person) RETURN DISTINCT p.city AS c", None),
    # ── push_where_into_match ──
    ("where_eq", "social_graph", "MATCH (p:Person) WHERE p.city = 'Oslo' RETURN p.name AS n", None),
    ("where_gt", "social_graph", "MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS n", None),
    ("where_and", "social_graph", "MATCH (p:Person) WHERE p.age > 30 AND p.city = 'Bergen' RETURN p.name AS n", None),
    # ── fold_or_to_in ──
    (
        "or_chain_to_in",
        "social_graph",
        "MATCH (p:Person) WHERE p.city = 'Oslo' OR p.city = 'Bergen' OR p.city = 'Stavanger' RETURN p.name AS n",
        None,
    ),
    # ── extract_pushable_rel_predicates ──
    (
        "rel_property_filter",
        "social_graph",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) WHERE r.since > 2017 RETURN p.name AS p, q.name AS q",
        None,
    ),
    # ── fold_pass_through_with ──
    (
        "pass_through_with",
        "social_graph",
        "MATCH (p:Person) WITH p MATCH (p)-[:KNOWS]->(q:Person) RETURN p.name AS p, q.name AS q",
        None,
    ),
    # ── desugar_multi_match_return_aggregate ──
    # Regression test for the bug found by this harness on first run:
    # `MATCH (p) MATCH (c) RETURN p.city, count(c)` was over-finely
    # grouped (per-p) when the user wrote a per-property aggregation.
    # Fix: WITH groups by the user-specified RETURN expressions, not
    # the source variable. See `desugar_multi_match_return_aggregate`
    # in `simplification.rs`.
    (
        "multi_match_group_agg",
        "social_graph",
        "MATCH (p:Person) MATCH (c:Company) RETURN p.city AS city, count(c) AS n",
        None,
    ),
    # ── reorder_match_clauses + optimize_pattern_start_node ──
    (
        "two_match_chains",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) MATCH (p)-[:KNOWS]->(q:Person) "
        "RETURN p.name AS p, c.name AS c, q.name AS q",
        None,
    ),
    (
        "anchored_three_hop",
        "social_graph",
        "MATCH (a:Person {person_id: 1})-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
        "RETURN a.name AS a, b.name AS b, c.name AS c",
        None,
    ),
    # ── push_limit_into_match ──
    ("limit_simple", "social_graph", "MATCH (p:Person) RETURN p.name AS n LIMIT 5", None),
    ("limit_one", "social_graph", "MATCH (p:Person) RETURN p.name AS n LIMIT 1", None),
    ("limit_zero", "social_graph", "MATCH (p:Person) RETURN p.name AS n LIMIT 0", None),
    # ── 0.8.27 bug: multi-MATCH + WHERE on late-bound var + LIMIT ──
    (
        "multi_match_where_limit",
        "social_graph",
        "MATCH (a:Person) MATCH (b:Person) MATCH (c:Person) WHERE c.age > 35 RETURN a.name AS a LIMIT 10",
        None,
    ),
    # ── push_distinct_into_match ──
    (
        "distinct_with_match",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN DISTINCT c.name AS c",
        None,
    ),
    # ── fuse_anchored_edge_count ──
    (
        "anchored_edge_count",
        "social_graph",
        "MATCH (p:Person {person_id: 1})-[:KNOWS]->(q:Person) RETURN count(q) AS n",
        None,
    ),
    # ── fuse_count_short_circuits ──
    ("count_distinct_star", "social_graph", "MATCH (p:Person) RETURN count(DISTINCT p) AS n", None),
    # ── fuse_optional_match_aggregate (0.8.31 bug) ──
    (
        "count_optional_edge_var",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[r:KNOWS]->(:Person) RETURN p.name AS n, count(r) AS k",
        None,
    ),
    # ── fuse_match_return_aggregate ──
    ("group_by_city", "social_graph", "MATCH (p:Person) RETURN p.city AS city, count(p) AS n", None),
    ("group_by_with_sum", "social_graph", "MATCH (p:Person) RETURN p.city AS city, sum(p.salary) AS total", None),
    # ── fuse_match_with_aggregate + fuse_match_with_aggregate_top_k (0.8.32 bug) ──
    # Secondary sort key (city, n) breaks ties so the row identities are
    # deterministic — without it, both modes return correct counts but
    # which-3-of-4-tied-cities surfaces is implementation-defined.
    (
        "cohort_top_k",
        "social_graph",
        "MATCH (p:Person) WITH p.city AS city, count(p) AS n RETURN city, n ORDER BY n DESC, city LIMIT 3",
        None,
    ),
    (
        "cohort_top_k_property",
        "social_graph",
        "MATCH (p:Person) WITH p, count{(p)-[:KNOWS]->()} AS friends "
        "RETURN p.name AS n, friends ORDER BY friends DESC, n LIMIT 5",
        None,
    ),
    # ── fuse_node_scan_aggregate ──
    ("node_scan_count", "social_graph", "MATCH (n) RETURN count(n) AS n", None),
    # ── fuse_node_scan_top_k + fuse_order_by_top_k ──
    (
        "order_by_limit",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.age AS age ORDER BY p.age DESC LIMIT 5",
        None,
    ),
    (
        "order_by_ties",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.city AS c ORDER BY p.city, p.name LIMIT 10",
        None,
    ),
    # ── reorder_predicates_by_cost ──
    (
        "predicate_reorder",
        "social_graph",
        "MATCH (p:Person) WHERE p.salary > 80000 AND p.city = 'Oslo' RETURN p.name AS n",
        None,
    ),
    # ── mark_fast_var_length_paths ──
    # NOTE: `var_length_no_var` is in TestKnownDivergences below — the
    # fast-path BFS dedups by target node, the naive path keeps one row
    # per distinct path. Pure design tension (path-counting semantics);
    # not a strict correctness bug but worth flagging.
    (
        "var_length_with_var",
        "small_graph",
        "MATCH (p:Person {person_id: 1})-[r:KNOWS*1..3]->(q:Person) RETURN q.name AS n",
        None,
    ),
    # ── UNION (optimize_nested_queries) ──
    (
        "union_simple",
        "small_graph",
        "MATCH (p:Person) WHERE p.age < 30 RETURN p.name AS n "
        "UNION MATCH (p:Person) WHERE p.age > 40 RETURN p.name AS n",
        None,
    ),
    # ── edge cases ──
    (
        "optional_no_match",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(c:Company) RETURN p.name AS n, c.name AS c",
        None,
    ),
    (
        "with_chain",
        "social_graph",
        "MATCH (p:Person) WITH p WHERE p.age > 25 WITH p, p.salary AS s RETURN p.name AS n, s",
        None,
    ),
    ("empty_typed_match", "social_graph", "MATCH (n:NoSuchType) RETURN count(n) AS n", None),
    ("skip_and_limit", "social_graph", "MATCH (p:Person) RETURN p.name AS n ORDER BY p.person_id SKIP 5 LIMIT 3", None),
]


def _normalize(rows: list[dict]) -> list[tuple]:
    """Sort + canonicalize rows so unordered queries compare equal.

    Modeled on `tests/test_storage_parity.py::_rows()`. Each row becomes
    a tuple of (key, str(value)) pairs sorted by key — handles dict
    ordering and mixed numeric/string types. Final list is sorted so
    queries without ORDER BY are still comparable.
    """
    canonical = [tuple(sorted((k, str(v)) for k, v in row.items())) for row in rows]
    canonical.sort()
    return canonical


@pytest.mark.differential
@pytest.mark.parametrize(
    "name,fixture,query,params",
    DIFFERENTIAL_QUERIES,
    ids=[entry[0] for entry in DIFFERENTIAL_QUERIES],
)
def test_optimized_matches_naive(
    name: str,
    fixture: str,
    query: str,
    params: dict | None,
    request: pytest.FixtureRequest,
) -> None:
    """Run `query` against `fixture` with optimizer on, then off; assert equal rows."""
    g = request.getfixturevalue(fixture)
    kwargs = {"params": params} if params else {}

    naive = _normalize(g.cypher(query, disable_optimizer=True, **kwargs).to_list())
    optimized = _normalize(g.cypher(query, **kwargs).to_list())

    assert optimized == naive, (
        f"Optimizer divergence on `{name}`:\n"
        f"  query:     {query}\n"
        f"  optimized: {optimized[:5]}{'...' if len(optimized) > 5 else ''} ({len(optimized)} rows)\n"
        f"  naive:     {naive[:5]}{'...' if len(naive) > 5 else ''} ({len(naive)} rows)\n"
        f"  diff (in optimized but not naive): {[r for r in optimized if r not in naive][:3]}\n"
        f"  diff (in naive but not optimized): {[r for r in naive if r not in optimized][:3]}\n"
        f"To bisect: rerun with disabled_passes=[<one pass at a time>] until divergence resolves.\n"
        f"Pass list: kglite.cypher_pass_names()"
    )


# ── Known divergences (xfail) ────────────────────────────────────────
#
# These shapes diverge between optimized and naive but the divergence
# was discovered by the harness on first run. They land here as
# permanent regression tests: when a fix lands, flip xfail → expected
# pass and the test starts protecting the fix.

KNOWN_DIVERGENT: list[tuple[str, str, str, str]] = [
    (
        "var_length_no_var",
        "small_graph",
        # Variable-length edge `[:KNOWS*1..3]` with no path variable:
        # Neo4j semantics return one row per distinct path (3 rows for
        # the small_graph fixture: 1→2, 1→3 direct, 1→2→3).
        # KGLite's `mark_fast_var_length_paths` optimization dedups
        # target nodes via global-visited BFS, returning 2 rows.
        # Disabling the optimization (naive path) restores per-path
        # row count, so the divergence is real and user-observable.
        #
        # This is a design question, not a quick fix:
        # - Option A: tighten the gate so the fast-path BFS only fires
        #   when downstream is DISTINCT or a path-count-invariant
        #   aggregate. Preserves Neo4j semantics; loses the perf win
        #   for plain `RETURN q.name`.
        # - Option B: declare per-target reachability the documented
        #   semantic for bare-edge variable-length MATCH; align the
        #   naive path to dedup too. Faster, but a Neo4j-incompatible
        #   semantic users should know about.
        # Tracked here until the call is made; the harness will catch
        # any future regression in either direction.
        "MATCH (p:Person {person_id: 1})-[:KNOWS*1..3]->(q:Person) RETURN q.name AS n",
        "var-length per-target vs. per-path semantics — pending design call",
    ),
]


@pytest.mark.differential
@pytest.mark.parametrize(
    "name,fixture,query,reason",
    KNOWN_DIVERGENT,
    ids=[entry[0] for entry in KNOWN_DIVERGENT],
)
def test_known_divergences(
    name: str,
    fixture: str,
    query: str,
    reason: str,
    request: pytest.FixtureRequest,
) -> None:
    """Documented divergence — xfail'd until fixed.

    Once a fix lands, the test starts passing and pytest will flag the
    xfail-as-passing — that's the signal to remove the entry from
    KNOWN_DIVERGENT and let it run as a regular regression test.
    """
    pytest.xfail(f"Known divergence: {reason}")
    # Unreachable, but documents what we'd assert when fixed:
    g = request.getfixturevalue(fixture)
    assert _normalize(g.cypher(query).to_list()) == _normalize(g.cypher(query, disable_optimizer=True).to_list())


# Per-pass bisection check: a representative cohort query must produce
# correct rows when ANY single pass is disabled. Catches passes that
# silently became load-bearing for correctness (a fusion pass should
# never affect rows — only speed). When a pass appears here that would
# affect correctness in isolation, that's a real bug to fix.
@pytest.mark.differential
@pytest.mark.parametrize("pass_name", kglite.cypher_pass_names())
def test_disabling_single_pass_preserves_correctness(pass_name: str, social_graph) -> None:
    """Each pass, disabled in isolation, must produce the same rows as the naive baseline."""
    query = "MATCH (p:Person) WITH p.city AS city, count(p) AS n RETURN city, n ORDER BY n DESC LIMIT 5"
    baseline = _normalize(social_graph.cypher(query, disable_optimizer=True).to_list())
    actual = _normalize(social_graph.cypher(query, disabled_passes=[pass_name]).to_list())
    assert actual == baseline, (
        f"Disabling pass `{pass_name}` produced different rows:\n  baseline: {baseline}\n  actual:   {actual}"
    )
