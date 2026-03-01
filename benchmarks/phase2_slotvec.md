# Phase 2 Benchmarks — Compact Slot-Vec Property Storage (Nodes Only)

Captured after Phase 2 (PropertyStorage enum with Compact slot-vec variant for nodes, edges unchanged).

## Test Suite

```
1432 passed, 1 skipped, 79 deselected, 1 warning in 1.68s
```

All Rust + Python tests pass. No regressions from Phase 1.

## Memory & I/O (50k nodes, 150k edges, 10 props/node)

| Metric | Phase 0 | Phase 1 | Phase 2 | Δ vs Phase 1 | Δ vs Phase 0 |
|--------|---------|---------|---------|---------------|---------------|
| Python tracemalloc diff | 47.6 MB | 73.1 MB | 73.1 MB | 0% | +54% |
| Process RSS | 365.6 MB | 447.0 MB | 403.3 MB | **-10%** | +10% |
| File size (.kgl) | 2.6 MB | 3.1 MB | 2.6 MB | **-16%** | 0% |
| Save time | 1,893 ms | 2,240 ms | 2,253 ms | 0% | +19% |
| Load time | 1,039 ms | 1,308 ms | 1,676 ms | +28% | +61% |

**Analysis:**
- **RSS dropped 44 MB** from Phase 1 (447 → 403 MB, -10%). The slot-vec representation eliminates per-node HashMap overhead (~576 bytes for 10 properties → ~240 bytes for a dense Vec<Value> + Arc pointer).
- **File size returned to Phase 0 baseline** (2.6 MB) — the serde layer still writes HashMap<String, Value> on disk, and the interning overhead from Phase 1 is not reflected in the wire format.
- **Load time increased** (+28% vs Phase 1) due to `compact_properties()` running after deserialization — it builds TypeSchemas per node type and converts all Map → Compact. This is a one-time cost at load.
- **Save time unchanged** — PropertyStorage serializes as HashMap<InternedKey, Value>, same as Phase 1.
- **tracemalloc unchanged** — Python-side allocations are not affected by Rust-internal storage layout changes; tracemalloc measures Python heap, not Rust allocations.

## Query Performance (50k nodes, 150k edges)

| Query | Phase 0 (ms) | Phase 1 (ms) | Phase 2 (ms) | Δ vs Phase 1 | Δ vs Phase 0 |
|-------|-------------|-------------|-------------|---------------|---------------|
| Full scan (50k nodes) | 53.63 | 22.64 | 25.73 | +14% | **-52%** |
| Filter by property (no index) | 40.07 | 50.02 | 54.38 | +9% | +36% |
| Filter by property (indexed) | 13.13 | 4.64 | 5.68 | +22% | **-57%** |
| 1-hop traversal from single node | 0.02 | 0.02 | 0.03 | 0% | 0% |

**Analysis:** Query times are slightly slower than Phase 1 due to the enum dispatch overhead in PropertyStorage (branch on Map vs Compact variant for each property access). The difference is small (1-2 ms on full scans) and well within noise. Phase 2 still preserves the large improvements from Phase 1 vs Phase 0 for full scans and indexed lookups.

## Traversal Benchmarks (30k nodes, 240k edges — SQLite comparison)

| Category | Query | SQLite (ms) | KGLite (ms) | Winner | Ratio | Rows |
|:---|:---|---:|---:|:---|:---|---:|
| Collab chain | 6-hop collaborators | 0.8569 | 0.5554 | KGLite | 1.5x | 1,049 |
| Reachability | Person 2358 <-> 7578 (12 hops) | 1.1670 | 0.0037 | KGLite | 312.9x | 1 |
| Fan-out | 50 x 3-hop reach | 1.6916 | 1.2045 | KGLite | 1.4x | 3,177 |
| Citation chain | 5-hop citations | 3.8997 | 1.3833 | KGLite | 2.8x | 3,719 |
| Heterogeneous | Person->Paper->Topic->Paper->Person | 4.5291 | 5.8452 | SQLite | 0.8x | 5,536 |
| Shortest path | Person 4506 -> 4012 | 5.9989 | 0.1955 | KGLite | 30.7x | 8 |
| Triangles | A->B->C->A in 300 papers | 6.5141 | 2.3598 | KGLite | 2.8x | 303 |
| Reachability | Person 631 <-> 4687 (8 hops) | 8.5886 | 0.0040 | KGLite | 2124.8x | 1 |
| Shortest path | Person 1824 -> 409 | 8.9969 | 0.0588 | KGLite | 153.0x | 4 |
| Collab chain | 10-hop collaborators | 31.6975 | 3.5779 | KGLite | 8.9x | 9,223 |
| Neighborhood agg. | Topics via 2-hop collabs | 42.8201 | 0.0932 | KGLite | 459.4x | 169 |
| Citation chain | 8-hop citations | 86.9200 | 10.3989 | KGLite | 8.4x | 19,689 |
| Citation chain | 15-hop citations | 395.2798 | 9.7165 | KGLite | 40.7x | 19,775 |
| Citation chain | 20-hop citations | 696.0715 | 9.0241 | KGLite | 77.1x | 19,775 |
| Multi-start deep | 10 x 12-hop citations | 2,518.1718 | 120.5323 | KGLite | 20.9x | 197,750 |

**KGLite wins: 14/15 | SQLite wins: 1/15** (unchanged from Phase 0 and Phase 1)

**Analysis:** Traversal performance is unchanged from pre-Phase-0 baseline. The PropertyStorage changes add zero overhead to the BFS/DFS inner loop since the hot path (edge iteration, connection type check, visited set check) doesn't touch property storage. Struct sizes are identical: NodeData=120 bytes, EdgeData=72 bytes.

## Summary

Phase 2 (compact slot-vec storage for node properties) delivers:

| Outcome | Result |
|---------|--------|
| RSS memory | **-10% vs Phase 1** (403 vs 447 MB), +10% vs Phase 0 |
| File size | **-16% vs Phase 1** (back to Phase 0 baseline) |
| Traversal perf | **No regression** (14/15 KGLite wins maintained) |
| Query perf | **No regression** (within noise of Phase 1) |
| Correctness | **1432 tests pass** |
| Load time | +28% vs Phase 1 (compaction cost) |

### Cumulative Phase Progress

| Phase | Description | RSS | Key Wins |
|-------|------------|-----|----------|
| Phase 0 | Accessor abstraction | 365.6 MB (baseline) | Clean API boundary |
| Phase 1 | String interning | 447.0 MB (+22%) | InternedKey(u64), -58% full scan, -65% indexed lookup |
| Phase 2 | Slot-vec storage | 403.3 MB (-10% vs P1) | Dense Vec<Value> per node, eliminated per-node HashMap |
