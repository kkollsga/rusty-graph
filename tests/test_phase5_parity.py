"""Phase 5 crunch-point parity tests.

Guards the columnar-cleanup + per-backend-impls phase of the 0.8.0
storage refactor. Tests here:

- **enum-match audit** — confirms that `GraphBackend::<Variant>` match
  sites are confined to the documented whitelist (the dispatcher in
  `schema.rs`, the PyO3 boundary in `mod.rs`, the trait declarations in
  `storage/mod.rs`, and the three disk-internal boundary files
  `ntriples.rs`, `io_operations.rs`, `batch_operations.rs` which reach
  into `DiskGraph` internals for bulk-path performance).
- **graph.copy() CoW correctness** — mutating a copy leaves the
  original unchanged on every backend. This is the Phase 0 crunch-point
  re-asserted after Phase 5's per-backend impls + ColumnStore split.
- **binary-size regression gate** — the release `.dylib` stays under
  the +20% budget relative to the Phase 4 baseline.

Run: pytest -m parity tests/test_phase5_parity.py
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pandas as pd
import pytest

from kglite import KnowledgeGraph

pytestmark = pytest.mark.parity

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files allowed to carry `GraphBackend::<Variant>` enum-match patterns.
# Everything else should dispatch through the `GraphRead` / `GraphWrite`
# traits. Each whitelist entry has a justification — if the list grows,
# revisit the design instead of adding another file.
ENUM_MATCH_WHITELIST = {
    # Trait / enum declarations + the 3-arm dispatcher — required.
    "storage/mod.rs": "GraphRead / GraphWrite trait surface",
    "schema.rs": "GraphBackend enum + dispatcher impls",
    "mod.rs": "PyO3 boundary (KnowledgeGraph class registration)",
    # Disk-internal boundary: reach into DiskGraph fields (node_slots,
    # pending_edges, column_stores, data_dir, qnum_to_idx) for bulk-path
    # performance. Documented and intentional; Phase 7's backend-subdir
    # reorg homes these with the disk backend.
    "ntriples.rs": "disk-internal bulk-build (ntriples loader)",
    "io_operations.rs": "disk-internal .kgl load_disk_dir path",
    "batch_operations.rs": "disk-internal update-path row_id lookup",
}

ENUM_MATCH_PATTERN = re.compile(r"GraphBackend::[A-Z]")


def _list_rs_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.rs"))


def test_enum_match_audit():
    """`GraphBackend::<Variant>` matches only appear in whitelisted files."""

    src_graph = REPO_ROOT / "src" / "graph"
    offenders: dict[Path, int] = {}
    for rs in _list_rs_files(src_graph):
        rel = rs.relative_to(src_graph).as_posix()
        if rel in ENUM_MATCH_WHITELIST:
            continue
        # storage/ subdir: only mod.rs is the trait surface; impls.rs is
        # the per-backend impls (no GraphBackend match).
        if rel == "storage/impls.rs":
            # The per-backend impls file MUST NOT carry enum matches —
            # that would defeat the whole point of splitting.
            text = rs.read_text()
            hits = ENUM_MATCH_PATTERN.findall(text)
            if hits:
                offenders[rs] = len(hits)
            continue
        text = rs.read_text()
        hits = ENUM_MATCH_PATTERN.findall(text)
        if hits:
            offenders[rs] = len(hits)

    assert not offenders, (
        "GraphBackend:: enum matches leaked outside the whitelist:\n"
        + "\n".join(f"  {p.relative_to(REPO_ROOT)}: {n} hit(s)" for p, n in offenders.items())
        + "\n\nAdd the file to ENUM_MATCH_WHITELIST (with a written justification) "
        + "or route the call through GraphRead / GraphWrite."
    )


def test_graph_copy_cow_correctness_memory():
    """Mutating the copy does not affect the original (in-memory backend)."""

    kg = KnowledgeGraph()
    df = pd.DataFrame([{"pid": 1, "name": "Alice", "age": 30}, {"pid": 2, "name": "Bob", "age": 25}])
    kg.add_nodes(df, "Person", "pid", "name")

    kg2 = kg.copy()
    kg2.add_nodes(
        pd.DataFrame([{"pid": 1, "name": "Alice Updated", "age": 99}]),
        "Person",
        "pid",
        "name",
        conflict_handling="update",
    )

    orig = kg.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")
    mod = kg2.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")

    orig_rows = [dict(r) for r in orig]
    mod_rows = [dict(r) for r in mod]

    assert orig_rows == [{"age": 30}], f"original mutated unexpectedly: {orig_rows}"
    assert mod_rows == [{"age": 99}], f"copy update did not apply: {mod_rows}"


def test_graph_copy_cow_correctness_mapped():
    """Mutating the copy does not affect the original (mapped backend)."""

    kg = KnowledgeGraph(storage="mapped")
    df = pd.DataFrame([{"pid": 1, "name": "Alice", "age": 30}, {"pid": 2, "name": "Bob", "age": 25}])
    kg.add_nodes(df, "Person", "pid", "name")

    kg2 = kg.copy()
    kg2.add_nodes(
        pd.DataFrame([{"pid": 1, "name": "Alice Updated", "age": 99}]),
        "Person",
        "pid",
        "name",
        conflict_handling="update",
    )

    orig = [dict(r) for r in kg.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")]
    mod = [dict(r) for r in kg2.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")]

    assert orig == [{"age": 30}], f"mapped original mutated: {orig}"
    assert mod == [{"age": 99}], f"mapped copy update lost: {mod}"


def test_binary_size_regression():
    """Release `.dylib` size stays under the +20% Phase 5 budget.

    Phase 4 exit baseline: 6,996,688 bytes (≈6.67 MB). Phase 5 gate:
    8,396,025 bytes (+20%). Lifts when Phase 7 does the directory
    reorg and hard cap tightens.
    """

    candidates = [
        REPO_ROOT / "target" / "release" / "libkglite.dylib",
        REPO_ROOT / "target" / "release" / "libkglite.so",
    ]
    bin_path = next((p for p in candidates if p.exists()), None)
    if bin_path is None:
        pytest.skip("release build not present — run `cargo build --release` first")

    size = bin_path.stat().st_size
    baseline = 6_996_688
    gate = int(baseline * 1.20)
    assert size <= gate, (
        f"{bin_path.name} = {size:,} bytes > gate {gate:,} "
        f"(+20% over Phase 4 baseline {baseline:,}). "
        "Investigate what grew before raising the gate."
    )


def test_dead_code_check():
    """`cargo clippy -- -D dead_code` flags nothing in the graph module."""

    result = subprocess.run(
        ["cargo", "clippy", "--release", "--", "-D", "dead_code"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail("cargo clippy found dead-code warnings:\n" + (result.stdout or "") + "\n" + (result.stderr or ""))
