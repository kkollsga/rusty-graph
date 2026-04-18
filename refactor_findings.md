# KGLite 0.8.0 — Refactor findings to revisit

Living log of issues surfaced during the Phase 0–11 refactor that are
**not release blockers** but are worth returning to. Each entry has:
(a) where it was observed, (b) likely root cause, (c) suggested next
action. When an item is addressed, move it to the bottom `## Resolved`
section rather than deleting — preserves the paper trail.

Last updated: end of Phase 12 (2026-04-18).

---

## A. Performance — flagged but under release gates

### A1. `find_20x_10000_memory` regressed +4.7 % vs v0.7.17

- **Where**: `tests/benchmarks/phase11_delta.md` — memory p50 4.436 ms → 4.642 ms. Only cell that sits just under the +5 % memory block gate.
- **Hypothesis**: Phase 3 routed `find()` through the `GraphRead` trait + GAT iterator. The monomorphisation cost may be paying off *most* of the time but not on this specific loop shape (20 random-name lookups).
- **Action**: profile with `cargo flamegraph` on `python -c "import kglite; ..."` running 20k iterations of the same pattern. Confirm whether the GAT bounce is real or whether it's thermal noise. If real, add a specialised fast path in `pyapi/kg_core.rs::find` that skips the trait call on memory mode.
- **Risk if ignored**: low. Absolute delta is 0.2 ms per 20 lookups.

### A2. Minor flags: `simple_filter_memory` +2.3 %, `simple_filter_mapped` +3.1 %, `multi_predicate_mapped` +4.3 %

- **Where**: same delta table.
- **Observation**: All three live in the sub-millisecond regime (0.55–0.66 ms). Absolute deltas are 12–27 µs per call — within single-trial noise.
- **Action**: re-measure at N=50 on a cold machine. If still above +2 %, profile. Otherwise close.
- **Risk if ignored**: none — none hit any gate.

### A3. `limit_100_any_edge` 11.26 s on Wikidata

- **Where**: `tests/benchmarks/phase11_wikidata_partial.csv`.
- **Observation**: One Wikidata query takes 11.26 s vs ~40 ms for peers. The script itself labels this a "known pain point" (see `bench/benchmark_wikidata_cypher.py` header). Pattern is `MATCH ()-[r]->() RETURN r LIMIT 100` — no anchor, so the planner walks the full edge space until 100 rows materialise.
- **Hypothesis**: Cypher planner does not push `LIMIT` into the edge scan on unanchored patterns across storage-mode boundaries.
- **Action**: a planner pass that propagates `LIMIT N` into `edge_iter` when no WHERE clause exists. Same pattern already works on anchored matches.
- **Risk if ignored**: moderate — MCP users who paginate by LIMIT on large graphs will hit this cliff.

### A4. Disk-mode penalties on `legal` dataset

- **Where**: `tests/benchmarks/phase11_api_benchmark.txt` COMPARISON section (lines 487–503).
- **Observations**:
  - `legal EXISTS subquery`: memory 3 ms → mapped 3 ms → **disk 46 ms** (15× slower)
  - `legal OPTIONAL MATCH`: memory 78 ms → mapped 83 ms → **disk 193 ms** (2.5×)
  - `legal describe()`: memory 92 ms → mapped 93 ms → **disk 263 ms** (2.8×)
- **Hypothesis**: EXISTS/OPTIONAL MATCH materialise intermediate node refs through the disk backend's columnar accessors more than necessary. `describe()` walks sample-per-type which on disk hits cold pages.
- **Action**: add `legal EXISTS subquery` to `tests/benchmarks/phase11_harness.py` as a dedicated cell so the regression is visible on every run. Profile disk path specifically.
- **Risk if ignored**: low — disk is a new product with no v0.7.17 baseline, and the MCP-primary story lives on memory/mapped. But 46 ms vs 3 ms is a 15× user-visible gap on a common pattern.

### A6. Wikidata disk-graph build is ~16 % slower than v0.7.17

- **Where**: post-Phase-12 rebuild of `latest-truthy.nt.zst` (7.65B triples → 124M entities / 863M edges) against the current code (`/Volumes/EksternalHome/Data/Wikidata/wikidata_disk_graph_p12rebuild`). Same source file as the April 2026 v2 build.
- **Observation**: `load_ntriples` 4747 s (vs v2 4238 s, +12 %); `rebuild_caches` 236 s (vs 130 s, +81 %). Total projected ~85 min vs v2's 73 min. **Semantic output is identical** (same entity/edge/skip counts, same triples scanned).
- **Phase breakdown (exposed by current verbose output, not reported by v2):**
  - Phase 1a triple ingest: 2339 s
  - Phase 1b columnar write: 1080 s
  - Phase 2 edge creation: 113 s
  - **Phase 3 CSR build: 1181 s** ← clear cost center
- **Hypothesis (refined, 2026-04-18)**: not the CSR algorithm — byte-identical vs v0.7.17 per `git diff`. The cost is **trait-dispatch overhead** on hot iteration paths. Phase 2–3 introduced `GraphRead` / `GraphWrite` traits; `compute_type_connectivity` now does 863M `Box<dyn Iterator>::next()` calls + 1.7B virtual `node_type_of` calls per rebuild. At ~50 ns extra per call that totals ~90 s — matches the observed +106 s (130 → 236 s) delta on `rebuild_caches`. Same mechanism inflates `load_ntriples` (124M node-adds + 863M edge-adds now routed through `GraphWrite` trait). Build mode is release (confirmed `target/release/libkglite.dylib`), not a debug-build regression. Memory is cleared correctly (every buffer `drop`-ped between phases, no leaks). I/O pattern in Step 3a scatter has not changed between versions — pre-existing random-ish mmap write, buffered via 512 MB chunks.
- **Output size**: 72 GB (vs v2's 78 GB, slightly smaller).
- **Action**: (a) add `#[inline(always)]` to the `GraphBackend` → `GraphRead`/`GraphWrite` trampoline wrappers in `src/graph/storage/backend.rs` and re-profile; (b) add a disk-backend fast path in `introspection/connectivity.rs::compute_type_connectivity` that `match`es on the backend enum and calls `DiskGraph::edge_endpoint_keys` inherently; (c) Phase 11's in-memory harness missed this because ≤50k-node builds have sub-second CSR phases — at 100M+ edge scale the virtual-call amplification dominates. Target for 0.8.1.
- **Crash note**: the rebuild itself was killed externally during `save()` (RSS = 3.08 GB at that point, no OOM). Likely a Claude Code background-task lifecycle limit; running standalone in a terminal would complete normally.
- **Risk if ignored**: moderate for Wikidata users (1/6 slowdown on a 70+ min build is noticeable). Low for memory/mapped workflows where the cost center doesn't exist.
- **Artifact**: partial rebuild left at `/Volumes/EksternalHome/Data/Wikidata/wikidata_disk_graph_p12rebuild` (72 GB). `save()` phase was externally interrupted; graph is likely loadable from the completed `load_ntriples` + `rebuild_caches` state, but should be re-verified or re-run before being used as a baseline.

### A5. `wiki5m build` on mapped is 70 % slower than memory

- **Where**: `tests/benchmarks/phase11_api_benchmark.txt` line 529 — memory 1073 ms, mapped 1827 ms, disk 1394 ms.
- **Observation**: memory is *faster than disk* at wiki5m scale (1.07 s vs 1.39 s), which is unexpected — typically disk's bulk-load path is fastest. Mapped is the clear slowest.
- **Hypothesis**: mapped's `add_connections` path allocates more intermediate string buffers than either memory (direct petgraph insert) or disk (bulk CSR build). Worth profiling during `add_connections` on mapped.
- **Action**: add a construction benchmark cell for `wiki5m`-shape data (string-heavy properties + many edge types) at mapped. If confirmed, optimise the mapped write path.
- **Risk if ignored**: low for now — wiki5m is a benchmark dataset, not a product use case.

---

## B. Memory behaviour oddities

### B1. `mapped_scale_5gb` RSS grows during queries

- **Where**: `tests/benchmarks/phase11_mapped_scale_5gb.txt` lines 25–30.
- **Observation**: post-build steady RSS is 2.44 GB; during queries RSS climbs to 4.75 GB, then `two_hop` drops it back to 3.13 GB. The "steady" claim in the summary is inconsistent with the query-time RSS.
- **Hypothesis**: mapped-mode query execution materialises per-query column slices that stick around in mmap page tables until the next major allocation pressure cycle. Not a leak (RSS drops on `two_hop`), but the RSS story is not monotonic.
- **Action**: instrument `test_mapped_scale.py` to record RSS before *and* after each query, and drop `gc.collect()` + `malloc_zone_pressure_relief()` (macOS) between queries to see if the elevated RSS is retainable or releasable. Document the actual memory envelope in AUDIT.
- **Risk if ignored**: moderate — users sizing RAM for mapped mode may plan for 2.4 GB and hit 4.75 GB mid-query.

### B2. Wikidata bench OOM-killed at query 29/45

- **Where**: `tests/benchmarks/phase11_wikidata.txt` — SIGKILL 137 after ~30 minutes.
- **Root cause**: residual RSS from the preceding 5 GB mapped run (~3 GB in page cache) + 78 GB disk graph's page-cache warmup on a machine with ~3 GB free memory at the start.
- **Action**: (a) re-run Wikidata bench in isolation after a cold boot; (b) sequence Phase 11 so Wikidata runs *before* any mapped-scale bench; (c) add an explicit "minimum free RAM" precondition check to `bench/benchmark_wikidata_cypher.py`.
- **Risk if ignored**: the 29 completed queries cover 5 of 8 categories. Ship decision: 0.8.0 can ship without the full 45; re-run pre-release manually.

---

## C. API surface — documentation / UX

### C2. `Cargo.toml` version still `0.7.17`

- **Where**: `Cargo.toml` line 3.
- **Observation**: Version wasn't bumped when the 0.8.0 refactor landed on main. Phase 13 handles this, but it means `kglite.__version__` currently reports `"0.7.17"` on the 0.8.0 codebase — including in the Phase 11 JSON baselines (`"kglite_version": "0.7.17"`).
- **Action (Phase 13)**: bump Cargo.toml + pyproject.toml + CHANGELOG.md together, tag `v0.8.0`.
- **Risk if ignored**: confused users / bug reports citing the wrong version.

---

## D. Rust hygiene debt (from earlier phase report-outs)

### D1. `RecordingGraph` carries `#[allow(dead_code)]` in release builds

- **Where**: `src/graph/storage/recording.rs:78`, `:85`, `:92` (three instances).
- **Observation**: `RecordingGraph` is only wired into tests via the parity framework. Release builds never instantiate it but the struct must compile. Current approach is blanket `#[allow(dead_code)]`.
- **Action**: gate behind `#[cfg(any(test, feature = "validation"))]` as originally planned for Phase 8 (see todo.md line 87). Would reduce release binary size by an unknown amount (~small).
- **Risk if ignored**: none functional. Lint/audit cleanliness only.

### D2. Inherent `GraphBackend` read/write methods shadow trait methods

- **Where**: `src/graph/storage/backend.rs` + the `GraphRead`/`GraphWrite` traits.
- **Observation**: Rust prefers inherent methods at name resolution. Trait dispatch is enforced via UFCS or `use Trait` scope. Deletion of the shadowing inherents was deferred from Phase 5 → Phase 8 → never happened.
- **Action**: delete the inherent methods in a dedicated small PR; every caller must route through the trait. Will surface any accidentally-inherent call site.
- **Risk if ignored**: low — works today, trait dispatch is always explicit at call sites. But the shadowing is a subtle footgun for new contributors.

### D3. `Graph` type alias is effectively redundant

- **Where**: `src/graph/schema.rs` — `pub type Graph = GraphBackend;` re-exported from `storage::backend`.
- **Observation**: Callers use `Graph` and `GraphBackend` interchangeably. Phase 9 report-out (todo.md line 891) flagged this as a Rust-idiom wart to drop in a hygiene pass.
- **Action**: rename all `Graph` uses to `GraphBackend`, delete the alias. Low-risk bulk rename.
- **Risk if ignored**: cosmetic.

---

## E. Test-suite cleanup

### E3. Variable-length path stress coverage is shallow

- **Where**: `tests/test_edge_cases_parity.py::test_long_chain_traversal_1000_hops` + `tests/test_stress.py::test_deep_traversal_10k_hops`.
- **Observation**: Both tests use `RETURN count(b)` which the Cypher planner short-circuits. Effective test coverage does not exercise actual path enumeration.
- **Action**: add a variant `RETURN p` (where `p = path`) that forces path materialisation — will either expose stack-depth limits or measure real deep-traversal cost.
- **Risk if ignored**: moderate — the "1k/10k-hop works" claim is only true for counting, not materialisation.

### E4. Datetime columns exist but are unsnapshotted in golden fixtures

- **Where**: `tests/golden/build_golden_graph.py` populates `joined_at` but `tests/golden/queries.py` never queries it.
- **Observation**: Datetime round-trip through cypher is not pinned in the golden suite.
- **Action**: add a `cypher_dates_roundtrip` query to `queries.py` after verifying datetime serialisation stability across platforms. Low-risk Phase 12 addition.
- **Risk if ignored**: low. Datetime parity across modes is tested in `test_edge_cases_parity.py`, just not via golden diff.

---

## F. Coverage gaps carried forward

| Item | Where flagged | Deferred to |
|---|---|---|
| Cross-hardware (Linux CI) validation | AUDIT §3 Deferred; Phase 11 report-out | Post-0.8.0 |
| 30 GB mapped-niche full run | AUDIT §3 Deferred | Manual pre-release gate |
| Full 45-query Wikidata bench | Phase 11 report-out B2 | Pre-release manual (after cold boot) |
| `benchmark_kglite.py` MCP-focused bench | Phase 11 report-out | Post-0.8.0 (api_benchmark.py covers near-term) |
| Phase 6 historical table in AUDIT §3 | Phase 11 report-out | Phase 12 doc pass — decide retain vs drop |
| Fuzzing malformed `.kgl` | Phase 10 plan deferred | 0.9.0 |
| `Value::String` → `Cow<'static, str>` / `Arc<str>` | todo.md line 696 | Post-0.8.0 |
| Remote / RPC backend | todo.md line 698 | Post-0.8.0 stretch |

---

## G. Architectural decisions still open (from todo.md §Deferred decisions)

- **PyO3 conversion helpers location**: `datatypes/` or `pyapi/convert/`? Decide in Phase 12 based on cleaner imports.
- **`RecordingGraph` release gating**: `#[cfg(test)]` vs `#[cfg(feature = "validation")]`. See D1 above.
- **`#[pyclass]` convention scope**: see D5. Phase 12 CLAUDE.md clarification.

---

## Resolved

Move entries here as they're addressed. Include the phase/commit that
closed them plus a one-line note so the paper trail survives.

- **C1 `find()` is code-entity-specific** — Phase 12. Docstring in `kglite/__init__.pyi::find` now carries a "⚠ Code-entity search only" warning and points users at `select().where()` / `cypher()` for general node lookup. Signature unchanged; `git diff v0.7.17 HEAD -- kglite/__init__.pyi` has no `def/class` lines.
- **C3 Stale `# currently xfail` comment** — Phase 12. Comment in `tests/test_cypher.py:1227` replaced with a note that the `TestBug*` classes regression-guard historic issues; all still pass.
- **D4 `languages/fluent/` empty stubs** — Phase 12. `ARCHITECTURE.md` now explicitly notes that fluent-chain state lives in `pyapi/kg_fluent.rs` and `languages/fluent/` is reserved scaffolding for future Rust-side extraction. The new `docs/adding-a-query-language.md` repeats this under "The `languages/` umbrella".
- **D5 `#[pyclass]` convention scope** — Phase 12. `CLAUDE.md` now says only `#[pymethods] impl` blocks must live under `pyapi/`; struct `#[pyclass]` attributes may stay with the struct definition (`KnowledgeGraph`, `ResultView`).
- **E1 `test_exception_list_still_applies` no-op** — Phase 12. Deleted from `tests/test_phase7_parity.py`. Phase 7 parity test count drops by 1 (restore if `GOD_FILE_EXCEPTIONS` ever gains an entry).
- **E2 `test_file_count_budget` self-skipping** — Phase 12. Deleted from `tests/test_phase6_parity.py`; also dropped the now-unused `subprocess` import. Phase 6 parity test count drops by 1 skipped entry.
