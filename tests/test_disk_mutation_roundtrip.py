"""Mutation + save + reload roundtrip parity across memory / mapped / disk.

Locks in the 0.8.11 fixes that make `save → reload` preserve:
  - column_stores for types added post-`load_ntriples` via `add_nodes`
    (bug E; previously the columns.bin guard in `DirGraph::save_disk`
    dropped sidecars, so reloaded graphs returned `None` for every
    property on nodes of the new type),
  - cross-segment edges produced by seal,
  - CSR arrays when heap-backed by `reconcile_seg0_csr` (bug D),
  - clean segment layout (no stale `seg_NNN > 0` after compact-rewrite —
    bug C).

**Not covered**: Cypher `SET n.prop = X` on disk-backed graphs. That
path is a no-op for persistence across 0.8.10 / 0.8.11 (tracked as a
pre-existing limitation in CHANGELOG); an `xfail` guard in
`test_cypher_set_persists_disk_limitation_xfail` prevents silent
regression once it's fixed.

Run: pytest tests/test_disk_mutation_roundtrip.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

STORAGE_MODES = ("memory", "mapped", "disk")


def _new_kg(mode: str, path: str | None = None) -> KnowledgeGraph:
    if mode == "memory":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    if mode == "disk":
        assert path is not None
        return KnowledgeGraph(storage="disk", path=path)
    raise ValueError(mode)


def _save_and_reload(kg: KnowledgeGraph, save_path: str, mode: str) -> KnowledgeGraph:
    kg.save(save_path)
    if mode == "memory":
        return kglite.load(save_path)
    # mapped + disk reload via the load entrypoint — the directory
    # carries the mode info.
    return kglite.load(save_path)


def _rows(rv) -> list[dict]:
    return rv.to_list() if hasattr(rv, "to_list") else list(rv)


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_add_nodes_new_type_survives_save_reload(mode, tmp_path):
    """add_nodes with a new type → save → reload → properties must
    round-trip. Bug E regression."""
    graph_path = str(tmp_path / f"g_{mode}")
    save_path = str(tmp_path / f"saved_{mode}") if mode != "disk" else graph_path
    kg = _new_kg(mode, path=graph_path if mode == "disk" else None)

    df = pd.DataFrame([{"id": f"N_{i}", "title": f"Title {i}", "age": 20 + i} for i in range(5)])
    kg.add_nodes(df, "NewEntity", "id", "title")

    pre_rows = _rows(
        kg.cypher(
            "MATCH (n:NewEntity) RETURN n.id, n.title, n.age ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    assert len(pre_rows) == 5
    assert pre_rows[0]["n.id"] == "N_0"
    assert pre_rows[0]["n.title"] == "Title 0"
    assert pre_rows[0]["n.age"] == 20

    reloaded = _save_and_reload(kg, save_path, mode)
    post_rows = _rows(
        reloaded.cypher(
            "MATCH (n:NewEntity) RETURN n.id, n.title, n.age ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    assert post_rows == pre_rows, f"{mode}: property roundtrip mismatch"


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_add_edges_survive_save_reload(mode, tmp_path):
    """add_connections for a new edge type → save → reload → edges
    must be enumerable post-reload."""
    graph_path = str(tmp_path / f"g_{mode}")
    kg = _new_kg(mode, path=graph_path if mode == "disk" else None)

    nodes_df = pd.DataFrame([{"id": f"N_{i}", "title": f"N{i}"} for i in range(4)])
    kg.add_nodes(nodes_df, "Person", "id", "title")

    edges_df = pd.DataFrame(
        [
            {"src": "N_0", "tgt": "N_1"},
            {"src": "N_1", "tgt": "N_2"},
            {"src": "N_2", "tgt": "N_3"},
        ]
    )
    kg.add_connections(edges_df, "KNOWS", "Person", "src", "Person", "tgt")

    pre_count = _rows(
        kg.cypher(
            "MATCH (:Person)-[:KNOWS]->(:Person) RETURN count(*) AS n",
            timeout_ms=10_000,
        )
    )[0]["n"]
    assert pre_count == 3

    reloaded = _save_and_reload(kg, graph_path, mode)
    post_count = _rows(
        reloaded.cypher(
            "MATCH (:Person)-[:KNOWS]->(:Person) RETURN count(*) AS n",
            timeout_ms=10_000,
        )
    )[0]["n"]
    assert post_count == 3, f"{mode}: edge count regressed after reload"


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_detach_delete_count_survives_save_reload(mode, tmp_path):
    """DETACH DELETE → save → reload → surviving count and ids must
    round-trip. Full property persistence is covered by the in-memory
    modes here; see `test_detach_delete_property_persistence_*` for
    the mode-specific property coverage."""
    graph_path = str(tmp_path / f"g_{mode}")
    kg = _new_kg(mode, path=graph_path if mode == "disk" else None)

    nodes_df = pd.DataFrame([{"id": f"P_{i}", "title": f"P{i}", "age": 20 + i} for i in range(10)])
    kg.add_nodes(nodes_df, "Thing", "id", "title")

    deleted = _rows(
        kg.cypher(
            "MATCH (n:Thing) WHERE n.id >= 'P_5' DETACH DELETE n RETURN count(n) AS n",
            timeout_ms=10_000,
        )
    )[0]["n"]
    assert deleted == 5

    reloaded = _save_and_reload(kg, graph_path, mode)
    post = _rows(
        reloaded.cypher(
            "MATCH (n:Thing) RETURN n.id ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    assert [r["n.id"] for r in post] == [f"P_{i}" for i in range(5)], (
        f"{mode}: surviving id set diverged after delete + reload"
    )


@pytest.mark.parametrize("mode", ["memory", "mapped"])
def test_detach_delete_property_persistence_in_memory_modes(mode, tmp_path):
    """DETACH DELETE preserves surviving nodes' full property values on
    memory + mapped modes. Disk mode has a pre-existing property
    corruption on delete — tracked by the xfail sibling below."""
    graph_path = str(tmp_path / f"g_{mode}")
    kg = _new_kg(mode, path=graph_path if mode == "disk" else None)
    kg.add_nodes(
        pd.DataFrame([{"id": f"P_{i}", "title": f"P{i}", "age": 20 + i} for i in range(10)]),
        "Thing",
        "id",
        "title",
    )
    kg.cypher(
        "MATCH (n:Thing) WHERE n.id >= 'P_5' DETACH DELETE n",
        timeout_ms=10_000,
    )
    save_path = graph_path if mode == "disk" else str(tmp_path / f"saved_{mode}")
    reloaded = _save_and_reload(kg, save_path, mode)
    post = _rows(
        reloaded.cypher(
            "MATCH (n:Thing) RETURN n.id, n.title, n.age ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    for r in post:
        idx = int(r["n.id"][2:])
        assert r["n.title"] == f"P{idx}", f"{mode}: title mismatch for {r['n.id']}"
        assert r["n.age"] == 20 + idx, f"{mode}: age mismatch for {r['n.id']}"


@pytest.mark.xfail(
    reason="DETACH DELETE on disk-backed graphs corrupts surviving nodes' "
    "title and some age values. Pre-existing since 0.8.10. Sibling of "
    "the Cypher-SET disk limitation — same root cause (node mutation "
    "routing). Scheduled for a follow-up release.",
    strict=True,
)
def test_detach_delete_property_persistence_disk_xfail(tmp_path):
    """Sentinel for disk-mode DELETE property-persistence fix. Flips
    from xfail → pass when the column-store mutation routing is
    unified."""
    graph_path = str(tmp_path / "g_disk")
    kg = KnowledgeGraph(storage="disk", path=graph_path)
    kg.add_nodes(
        pd.DataFrame([{"id": f"P_{i}", "title": f"P{i}", "age": 20 + i} for i in range(10)]),
        "Thing",
        "id",
        "title",
    )
    kg.cypher(
        "MATCH (n:Thing) WHERE n.id >= 'P_5' DETACH DELETE n",
        timeout_ms=10_000,
    )
    kg.save(graph_path)
    del kg
    reloaded = kglite.load(graph_path)
    post = _rows(
        reloaded.cypher(
            "MATCH (n:Thing) RETURN n.id, n.title, n.age ORDER BY n.id",
            timeout_ms=10_000,
        )
    )
    for r in post:
        idx = int(r["n.id"][2:])
        assert r["n.title"] == f"P{idx}"
        assert r["n.age"] == 20 + idx


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_seal_then_compact_rewrite_cleans_stale_segs(mode, tmp_path):
    """Disk-specific regression: subsequent save after a seal must
    either seal or compact-rewrite without leaving stale seg_NNN dirs.
    Memory/mapped run as controls (trivially pass).
    Bug C regression."""
    graph_path = str(tmp_path / f"g_{mode}")
    kg = _new_kg(mode, path=graph_path if mode == "disk" else None)

    # Initial save.
    kg.add_nodes(
        pd.DataFrame([{"id": f"A_{i}", "title": f"A{i}"} for i in range(3)]),
        "A",
        "id",
        "title",
    )
    if mode == "disk":
        kg.save(graph_path)

    # Tail: add nodes + intra-tail edges — triggers seal on disk.
    kg.add_nodes(
        pd.DataFrame([{"id": f"B_{i}", "title": f"B{i}"} for i in range(3)]),
        "B",
        "id",
        "title",
    )
    kg.add_connections(
        pd.DataFrame([{"src": "B_0", "tgt": "B_1"}, {"src": "B_1", "tgt": "B_2"}]),
        "T",
        "B",
        "src",
        "B",
        "tgt",
    )
    if mode == "disk":
        kg.save(graph_path)

    # Now mutate existing data via add_connections (no new nodes).
    # On disk this falls to compact-rewrite; must clean the stale
    # seg_001 and persist heap-backed arrays.
    kg.add_connections(
        pd.DataFrame([{"src": "A_0", "tgt": "A_1"}]),
        "T2",
        "A",
        "src",
        "A",
        "tgt",
    )

    save_path = graph_path if mode == "disk" else str(tmp_path / f"saved_{mode}")
    reloaded = _save_and_reload(kg, save_path, mode)

    # All three edges must survive.
    t_count = _rows(reloaded.cypher("MATCH ()-[:T]->() RETURN count(*) AS n", timeout_ms=10_000))[0]["n"]
    t2_count = _rows(reloaded.cypher("MATCH ()-[:T2]->() RETURN count(*) AS n", timeout_ms=10_000))[0]["n"]
    assert t_count == 2, f"{mode}: :T edge count regressed"
    assert t2_count == 1, f"{mode}: :T2 edge count regressed"

    # On disk, confirm no stale seg_NNN > 0 remains.
    if mode == "disk":
        seg_dirs = sorted(p.name for p in Path(graph_path).iterdir() if p.is_dir() and p.name.startswith("seg_"))
        assert seg_dirs == ["seg_000"], f"compact-rewrite must remove stale segments; found {seg_dirs}"


@pytest.mark.xfail(
    reason="Cypher SET on disk-backed graphs is a no-op for persistence. "
    "The mutation lands in an arena copy of NodeData; clear_arenas "
    "drops it without writing back to ColumnStore. Pre-existing since "
    "0.8.10. Fixing requires unifying the DirGraph/DiskGraph column "
    "store Arc ownership — scheduled for a follow-up release.",
    strict=True,
)
def test_cypher_set_persists_disk_limitation_xfail(tmp_path):
    """Sentinel that flips from xfail → pass once Cypher SET persistence
    lands for disk graphs. Keeps us from silently regressing: pytest
    errors ("XPASS") if this test ever starts passing without the
    xfail being removed."""
    graph_path = str(tmp_path / "g_disk")
    kg = KnowledgeGraph(storage="disk", path=graph_path)
    kg.add_nodes(
        pd.DataFrame([{"id": "P_0", "title": "P0", "age": 20}]),
        "Person",
        "id",
        "title",
    )
    kg.cypher(
        "MATCH (n:Person) SET n.age = 100 RETURN count(n) AS n",
        timeout_ms=10_000,
    )
    kg.save(graph_path)
    del kg
    reloaded = kglite.load(graph_path)
    age = _rows(reloaded.cypher("MATCH (n:Person) RETURN n.age", timeout_ms=10_000))[0]["n.age"]
    assert age == 100, "SET should persist through save + reload"
