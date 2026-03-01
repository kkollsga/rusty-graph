# Phase 0 Baseline Benchmarks

Captured after Phase 0 (accessor abstraction layer) — before Phase 1 (string interning).

## Test Suite

```
1432 passed, 1 skipped, 79 deselected, 1 warning in 3.86s
```

All Rust + Python tests pass. No regressions.

## Memory & I/O (50k nodes, 150k edges, 10 props/node)

| Metric | Value |
|--------|-------|
| Python tracemalloc diff | 47.6 MB |
| Process RSS | 365.6 MB |
| File size (.kgl) | 2.6 MB |
| Save time | 1,893 ms |
| Load time | 1,039 ms |

## Query Performance (50k nodes, 150k edges)

| Query | Time (median of 10) | Rows |
|-------|---------------------|------|
| Full scan (50k nodes) | 53.63 ms | 50,000 |
| Filter by property (no index) | 40.07 ms | 12,595 |
| Filter by property (indexed) | 13.13 ms | 12,595 |
| 1-hop traversal from single node | 0.02 ms | 2 |

## Traversal Benchmarks (30k nodes, 240k edges — SQLite comparison)

| Category | Query | SQLite (ms) | KGLite (ms) | Winner | Ratio | Rows |
|:---|:---|---:|---:|:---|:---|---:|
| Collab chain | 6-hop collaborators | 0.8305 | 1.9002 | SQLite | 0.4x | 1,049 |
| Reachability | Person 2358 <-> 7578 (12 hops) | 1.1147 | 0.0223 | KGLite | 50.0x | 1 |
| Fan-out | 50 x 3-hop reach | 1.5872 | 5.9318 | SQLite | 0.3x | 3,177 |
| Citation chain | 5-hop citations | 4.2856 | 6.9766 | SQLite | 0.6x | 3,719 |
| Heterogeneous | Person->Paper->Topic->Paper->Person | 4.6082 | 22.4347 | SQLite | 0.2x | 5,536 |
| Shortest path | Person 4506 -> 4012 | 6.0107 | 1.2793 | KGLite | 4.7x | 8 |
| Triangles | A->B->C->A in 300 papers | 6.2226 | 8.6214 | SQLite | 0.7x | 303 |
| Reachability | Person 631 <-> 4687 (8 hops) | 8.1079 | 0.0232 | KGLite | 349.4x | 1 |
| Shortest path | Person 1824 -> 409 | 8.9961 | 0.4830 | KGLite | 18.6x | 4 |
| Collab chain | 10-hop collaborators | 31.7926 | 17.4788 | KGLite | 1.8x | 9,223 |
| Neighborhood agg. | Topics via 2-hop collabs | 39.3843 | 0.4173 | KGLite | 94.4x | 169 |
| Citation chain | 8-hop citations | 86.5259 | 37.3398 | KGLite | 2.3x | 19,689 |
| Citation chain | 15-hop citations | 389.9986 | 37.8648 | KGLite | 10.3x | 19,775 |
| Citation chain | 20-hop citations | 680.6732 | 37.3725 | KGLite | 18.2x | 19,775 |
| Multi-start deep | 10 x 12-hop citations | 2,466.3443 | 449.6322 | KGLite | 5.5x | 197,750 |

**KGLite wins: 10/15 | SQLite wins: 5/15**

## Notes

- Phase 0 is a pure refactoring (accessor methods). No performance change expected or observed.
- These numbers serve as the baseline for measuring Phase 1 (string interning) and Phase 2 (slot-vec) improvements.
- Memory numbers reflect Python-side tracemalloc; actual Rust heap usage is larger (RSS includes Rust allocations not tracked by Python).
