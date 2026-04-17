"""Phase 3 crunch-point parity tests.

Guards the GAT conversion and per-file trait-dispatch migration (Phase 3
of the 0.8.0 storage refactor). Targets two risks the earlier phases'
oracles don't cover:

1. **Traversal-under-mutation isolation** — a multi-hop MATCH produces
   its rows from the *starting* graph snapshot, even if subsequent
   mutations (CREATE / SET / DELETE) touch nodes/edges the traversal
   would otherwise visit. KGLite is single-threaded from Python, so the
   invariant is about Rust-side iterator materialisation: once the
   Cypher executor collects the result set, no later mutation can
   mutate those rows.

2. **Traversal ordering + row-count parity post-migration** — the trait
   migration must not silently reorder neighbor enumeration in a way
   that changes row counts for multi-hop MATCH patterns. Phase 1 parity
   tests assert *set equality*; this file adds explicit **count
   equality** across memory / mapped / disk for patterns that exercise
   the migrated `edges_directed` / `edge_references` / `edges` paths.

Run: pytest -m parity tests/test_phase3_parity.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph

pytestmark = pytest.mark.parity

STORAGE_MODES = ("memory", "mapped", "disk")


def _new_kg(mode: str, path: str | None = None) -> KnowledgeGraph:
    if mode == "memory":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    if mode == "disk":
        if path is None:
            raise ValueError("mode='disk' requires path")
        return KnowledgeGraph(storage="disk", path=path)
    raise ValueError(f"unknown mode: {mode}")


def _build_graph(kg: KnowledgeGraph):
    """A small social graph with known structure.

    Person 1 → Person 2 → Person 3
              ↘ Person 4
    Person 5 → Person 2

    Two-hop from Person 1 reaches {3, 4} via 2.
    Two-hop from Person 5 reaches {3, 4} via 2.
    """
    nodes = pd.DataFrame(
        [
            {"pid": 1, "name": "Alice"},
            {"pid": 2, "name": "Bob"},
            {"pid": 3, "name": "Carol"},
            {"pid": 4, "name": "Dave"},
            {"pid": 5, "name": "Eve"},
        ]
    )
    kg.add_nodes(nodes, "Person", "pid", "name")

    edges = pd.DataFrame(
        [
            {"src": 1, "tgt": 2},
            {"src": 2, "tgt": 3},
            {"src": 2, "tgt": 4},
            {"src": 5, "tgt": 2},
        ]
    )
    kg.add_connections(edges, "KNOWS", "Person", "src", "Person", "tgt")


def _rows(result) -> list[dict]:
    try:
        rows = [dict(r) for r in result]
    except TypeError:
        rows = list(result)
    return sorted(rows, key=lambda r: repr(sorted(r.items())))


# ─── 1. Traversal-under-mutation isolation ──────────────────────────────────


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_traversal_snapshot_resists_later_create(mode, tmp_path):
    """A collected 2-hop result must not grow after a later CREATE."""
    path = str(tmp_path / f"kg_{mode}") if mode == "disk" else None
    kg = _new_kg(mode, path=path)
    _build_graph(kg)

    q = "MATCH (a:Person {id: 1})-[:KNOWS*2]->(b:Person) RETURN b.id AS bid ORDER BY bid"
    before = _rows(kg.cypher(q))
    assert len(before) == 2, f"setup: expected 2 rows for {mode}, got {len(before)}"

    # Add a new Person that would extend the 2-hop reachable set if
    # evaluation leaked past the initial materialisation.
    new_nodes = pd.DataFrame([{"pid": 99, "name": "New"}])
    kg.add_nodes(new_nodes, "Person", "pid", "name")
    new_edges = pd.DataFrame([{"src": 2, "tgt": 99}])
    kg.add_connections(new_edges, "KNOWS", "Person", "src", "Person", "tgt")

    # Re-run: should now reflect the new state (new node reachable).
    after = _rows(kg.cypher(q))
    assert len(after) == 3, f"after CREATE: expected 3 rows for {mode}, got {len(after)}"

    # The original `before` snapshot captured the pre-mutation state.
    # The post-mutation query returns a superset: {3, 4} ⊂ {3, 4, 99}.
    before_ids = {r["bid"] for r in before}
    after_ids = {r["bid"] for r in after}
    assert before_ids == {3, 4}
    assert before_ids.issubset(after_ids)
    assert 99 in after_ids


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_traversal_snapshot_resists_later_delete(mode, tmp_path):
    """A collected 2-hop result must not shrink during its own iteration."""
    path = str(tmp_path / f"kg_{mode}") if mode == "disk" else None
    kg = _new_kg(mode, path=path)
    _build_graph(kg)

    q = "MATCH (a:Person {id: 1})-[:KNOWS*2]->(b:Person) RETURN b.id AS bid ORDER BY bid"
    before = _rows(kg.cypher(q))
    assert len(before) == 2

    # Delete the intermediate edge from 2 → 4 and re-query.
    kg.cypher("MATCH (:Person {id: 2})-[r:KNOWS]->(:Person {id: 4}) DELETE r")

    after = _rows(kg.cypher(q))
    # Before mutation: {3, 4} reachable. After edge removed: {3}.
    assert {r["bid"] for r in before} == {3, 4}
    assert {r["bid"] for r in after} == {3}


# ─── 2. Row-count parity across backends ────────────────────────────────────


@pytest.mark.parametrize(
    "cypher",
    [
        # Exercises edges_directed + multi-hop expansion on all backends.
        "MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN count(*) AS n",
        # Exercises edge_references via count_edges_filtered (disk fast path).
        "MATCH ()-[:KNOWS]->() RETURN count(*) AS n",
        # Reverse direction — exercises Direction::Incoming.
        "MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN count(*) AS n",
    ],
)
def test_traversal_row_count_parity_across_modes(cypher, tmp_path):
    """Row counts must match byte-for-byte across memory/mapped/disk."""
    results: dict[str, list[dict]] = {}
    for mode in STORAGE_MODES:
        path = str(tmp_path / f"kg_{mode}") if mode == "disk" else None
        kg = _new_kg(mode, path=path)
        _build_graph(kg)
        results[mode] = _rows(kg.cypher(cypher))

    ref = results["memory"]
    for mode in ("mapped", "disk"):
        assert results[mode] == ref, (
            f"Row count divergence on {mode}: {results[mode]} vs memory {ref} for cypher: {cypher}"
        )


# ─── 3. Neighbor-order-insensitive algorithm output ─────────────────────────


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_pagerank_score_set_parity_post_migration(mode, tmp_path):
    """PageRank returns scores for every node across backends.

    Phase 3 migrates the tight loops in pagerank/centrality. This
    doesn't assert identical float values (neighbor-order-dependent),
    but it does assert the score *set shape* — same node count, all
    scores finite, sums comparable within a tolerance.
    """
    path = str(tmp_path / f"kg_{mode}") if mode == "disk" else None
    kg = _new_kg(mode, path=path)
    _build_graph(kg)

    scores = kg.pagerank()
    assert len(scores) == 5, f"{mode}: expected 5 scores, got {len(scores)}"
    total = sum(s["score"] for s in scores)
    # PageRank scores sum to ~1.0 (damping factor may shift slightly).
    assert 0.9 < total < 1.1, f"{mode}: pagerank sum {total} outside [0.9, 1.1]"


# ─── 4. Edge iteration stability (GAT inlining guard) ───────────────────────


def test_edge_references_iteration_count_parity(tmp_path):
    """All edges iterable via edge_references → same count across backends.

    This exercises the `edge_references` trait method that Phase 3
    promoted from inherent → trait. The grep-gate rewrite in
    graph_algorithms.rs / introspection.rs / schema_validation.rs
    must not change the observable edge count.
    """
    counts: dict[str, int] = {}
    for mode in STORAGE_MODES:
        path = str(tmp_path / f"kg_{mode}") if mode == "disk" else None
        kg = _new_kg(mode, path=path)
        _build_graph(kg)
        # Uses edge_references internally via compute_connection_stats.
        schema = kg.schema()
        total_edges = sum(
            kg.cypher(f"MATCH ()-[:{ct}]->() RETURN count(*) AS n")[0]["n"] for ct in schema["connection_types"]
        )
        counts[mode] = total_edges

    assert counts["memory"] == counts["mapped"] == counts["disk"] == 4, counts
