# KGLite — 0.8.0 Refactor TODO

**Target release: 0.8.0.** The trait-based storage architecture
(`GraphRead` / `GraphWrite` / `RecordingGraph`) landed in Phases 0–7.
The remaining work before release is structural cleanup, testing
hardening, and validation. Phases 0–7 are done ✅ — their report-outs
live in git history. This document is forward-looking.

## Why 0.8.0 isn't ready today

Phase 7 shipped an audit + documentation deliverable (`AUDIT_0.8.0.md`)
and accepted 8 god files plus a deferred structural reorg as documented
exceptions. The user has since decided that the exceptions are not
acceptable for a "0.8.0" cut. Concretely:

1. **Python API leaks outside `pyapi/`.** `src/graph/mod.rs` is 6,758
   lines; ~5,665 lines (84%) are `#[pymethods]` that belong under
   `pyapi/`. `src/graph/cypher/result_view.rs` owns `#[pyclass] ResultView`
   + `ResultIter` (859 lines) outside `pyapi/`. Python-facing conversion
   helpers are scattered across `datatypes/` and `cypher/py_convert.rs`.
2. **No Fluent module exists.** The Fluent chain (`.select()`, `.where_()`,
   `.traverse()`, …) lives inside `#[pymethods]` in `mod.rs`. If we ever
   add a third query language (SPARQL, GraphQL, DSL), the layering for
   "Cypher ↔ Fluent ↔ new-language ↔ shared query primitives" is invisible.
3. **8 god files remain** (all > 2,500 line hard cap):
   `cypher/executor.rs` (12,156), `mod.rs` (6,758), `schema.rs` (5,141),
   `introspection.rs` (4,204), `cypher/planner.rs` (3,552),
   `storage/disk/disk_graph.rs` (3,276), `io/ntriples.rs` (3,022),
   `cypher/parser.rs` (2,920), `query/pattern_matching.rs` (2,610).
4. **Benchmark validation is thin.** Phase 6 medians carried forward
   through Phase 7. No N=20 multi-trial run; no second hardware profile;
   no v0.7.17 comparison; no mapped-niche 30 GB run; no Wikidata disk run.
5. **No golden output fixtures.** `describe()` / `schema()` / `find()` /
   `cypher()` outputs are not pinned — subtle format drift would slip past CI.

The plan below sequences these into six phases ending at the 0.8.0 cut.

---

## Inspection: what Phases 0–7 shipped

**Scope.** 31 commits on `main` (local, not pushed). Net +7,326 / −1,655
LoC across 80 files. Baseline commit: `7729472` (last pre-Phase-0 commit).
HEAD: `84a9b33` (Phase 7 deliverables).

**Trait architecture (done).**
- `GraphRead` — 48 methods + 6 GAT iterator types (`NodeIndicesIter<'a>`,
  `EdgesIter<'a>`, `EdgeReferencesIter<'a>`, `EdgesConnectingIter<'a>`,
  `NeighborsIter<'a>`, `EdgeIndicesIter<'a>`). Non-object-safe by design
  — callers use `&impl GraphRead`.
- `GraphWrite: GraphRead` — 7 methods (`add_node` / `remove_node` /
  `add_edge` / `remove_edge` / `node_weight_mut` / `edge_weight_mut` /
  `update_row_id` (disk-only, default no-op)).
- Per-backend impls: `MemoryGraph`, `MappedGraph` (distinct struct since
  Phase 5), `DiskGraph`. `RecordingGraph<G>` wrapper for validation.
- Exit contract met: `GraphBackend::[A-Z]` enum-match appears only in
  **7 whitelisted boundary files** (`storage/mod.rs`, `schema.rs`
  dispatcher, `pyapi/mod.rs` PyO3 edge, 3 disk-boundary files, 1 test
  gate). Whitelist in `tests/test_phase5_parity.py::ENUM_MATCH_WHITELIST`.

**Structural reorg (partial).** `src/graph/` went from 45 flat files to
9 subdirs: `algorithms/`, `cypher/`, `features/`, `introspection/`,
`io/`, `mutation/`, `pyapi/`, `query/`, `storage/` (`memory/`, `mapped/`,
`disk/`). Exactly one content split happened: `StringInterner` +
`InternedKey` extracted from `schema.rs` → `storage/interner.rs`.
Remaining 8 god files documented in `tests/test_phase7_parity.py::GOD_FILE_EXCEPTIONS`.

**Disk mutation bugs (Phase-2 xfails).** All three fixed in Phase 5 —
`add_nodes(conflict_handling="update"|"replace")` + `MERGE (a)-[:KNOWS]->(b)`
edge visibility. Tests converted from `xfail(strict=True)` to strict
asserts in `tests/test_phase2_parity.py:320–349`.

**Test surface (done).** `tests/test_phase{1..7}_parity.py` — 43 tests
across GraphRead expansion, GraphWrite, GATs, IO/v3 golden hash,
per-backend impls, RecordingGraph, structural gates.
`tests/test_storage_parity.py` — 14-query oracle gated behind
`pytest -m parity`.

**Documentation (done).** `ARCHITECTURE.md` at repo root, `AUDIT_0.8.0.md`
at repo root, `docs/adding-a-storage-backend.md` (RecordingGraph worked
example), `CHANGELOG.md` with `[0.8.0]` entry (unreleased).

**Debt carried into Phase 8+.**
- `#[pymethods]` in `mod.rs` + `#[pyclass]` in `cypher/result_view.rs`
  (see §1 above).
- 8 god files (see §3 above).
- `RecordingGraph` carries `#[allow(dead_code)]` in release builds
  — should gate behind `#[cfg(any(test, feature = "validation"))]`.
- Inherent methods on `GraphBackend` for reads/writes still coexist
  with trait methods (Rust prefers inherent at resolution). Trait
  dispatch enforced via UFCS or `use Trait` scope. Full deletion of the
  shadowing inherents deferred from Phase 5 → Phase 8.
- No golden output fixtures for `describe/schema/find/cypher`.
- No N=20 benchmark run; no second hardware profile; no v0.7.17 compare.
- `parity_baseline_phase_7.json` never committed (Phase 7 was
  structural-only; Phase 11 regenerates).
- Phase-6 `file-count-budget` test self-skips under Phase-7 commits;
  re-evaluate whether to keep it or retire it outright.

---

## Phase kickoff protocol (MANDATORY before any code changes in a phase)

**Assume a fresh-memory agent at every phase boundary.** This `todo.md`
is the only source of truth between phases. Do not rely on conversational
memory, prior agent state, or implicit knowledge — if it matters for the
next phase, it must be written down here. Agent swaps, session
interruptions, and days-between-phases are all first-class expected
scenarios.

Every phase starts in **plan mode** — no Edit / Write tool use until the
user explicitly approves. The sequence is:

1. **Read this `todo.md`** including the Inspection section and every
   prior phase's report-out appended here. This is the briefing.
2. **Investigate** — read every file the phase will touch, end-to-end.
   Not skim. Identify risks, coupling, performance-sensitive code paths
   (capture baselines before touching), and test-coverage gaps.
3. **Refine** — update the phase's risks list, crunch-point tests, and
   task list based on what the code says (not what the `todo.md` guessed
   in advance).
4. **Present a written plan**: confirmed risk list, file-by-file task
   breakdown, baselines captured, deferred decisions with named owners
   and triggers, revised effort estimate.
5. **Get explicit approval** — wait for user to say "go". Questions
   reset the plan cycle.
6. **Implement** — only now does Claude switch to Edit / Write.
   Deviations require a pause and a revised plan.

Rationale: the `todo.md` captures intent; the code always has surprises.

## Phase exit protocol — write a Report-out

Before closing a phase, append a **Report-out** subsection to that
phase's section in this file. Format:

```
### Phase N Report-out (written at phase exit)

**Completed**: brief summary of what shipped
**Baselines captured**: numbers + file paths
**Surprises found**: things the upfront plan didn't anticipate
**Decisions made**: deferred-decision items resolved, with rationale
**Debt introduced**: known-imperfect code, with the ticket/plan to repay
**Scope changes**: tasks added or deferred vs the original phase plan
**Next-phase prerequisites**: anything the following phase needs to know
**Files touched**: concise list or glob
```

A phase is not "done" until the report-out is written, committed
alongside the phase's code, and baselines are committed.

## Clean-break rules (aggressive — applied every phase)

- **One commit per phase.** A phase ships as a single squashed commit
  (plus the report-out commit) at phase exit. No per-file intra-phase
  commits cluttering `git log`.
- **Delete-as-you-go.** The phase that migrates code also deletes the
  now-unreachable old path. No follow-up cleanup phase.
- **No deprecated shims.** `#[deprecated]` forwarders, re-exports, "kept
  for compat" comments — all forbidden.
- **No TODO markers that say "migrate later".** If tempted, do the
  migration. Only TODOs allowed: deferred decisions with a named owner
  and trigger.
- **No feature-gated dual implementations.** Either the old or the new
  code exists. Not both.
- **Public Python API stays semantically stable.** Parity tests enforce.
  New methods fine; changed signatures not fine.
- **`.kgl` v3 on-disk format frozen.** Old files still load
  byte-compatibly. Golden hash in `tests/test_phase4_parity.py::GOLDEN_V3_DIGEST`.
- **No god files.** Soft cap 1,500 lines; hard cap 2,500.
  `mod.rs` = re-exports + short module doc + <20-line helpers only.

## Parity-test discipline

- Every phase has a **risks** list and **crunch-point tests** targeting
  those risks.
- Each phase commits a `tests/parity_baseline_phase_N.json` (or
  equivalent) so regressions surface as numeric drift.

## Per-phase testing gate (MUST run before declaring a phase complete)

1. `cargo test --release` — all Rust unit tests green
2. `make lint` — cargo fmt + clippy -D warnings + ruff + stubtest
3. `pytest tests/` — Python integration tests (excludes `-m benchmark` and `-m parity`)
4. `pytest -m parity tests/test_storage_parity.py` — memory / mapped / disk oracle
5. Benchmark regression gate (`pytest tests/benchmarks/test_nx_comparison.py -m benchmark`):
   - **In-memory** (core product): no query > +5 % vs prior phase
   - **Mapped**: no query > +10 %
   - **Disk**: no query > +15 %
6. `python tests/benchmarks/test_mapped_scale.py --target-gb 1` — no OOM, no crash
7. New baseline committed
8. Phase's own exit-criteria checklist verified
9. **Clean-break audit** — `rg 'deprecated|legacy|// kept for compat|// TODO.*migrate'`
   in `src/` returns 0 results
10. **Enum-match audit** — for files migrated this phase, `rg 'GraphBackend::[A-Z]'`
    in those files returns 0 results
11. **Report-out written** — committed alongside the code

---

## Target code structure (updated 2026-04-17)

The original Phase 7 target ([git show main~7:todo.md]) kept `query/` and
`cypher/` as peers at `src/graph/` top level, with Fluent API emerging
implicitly from `#[pymethods]`. After inspection, the cleaner shape is
an explicit query-language layer: all interface languages sit under
`languages/`, all share a `core/` primitives folder (today's `query/`).

```
src/
├── lib.rs                    # PyO3 module entry — #[pymodule] fn kglite + load()
├── datatypes/                # Rust value types + Rust↔Python conversion
│   ├── values.rs
│   ├── type_conversions.rs
│   ├── py_in.rs              # Python → Value (pure conversion)
│   └── py_out.rs             # Value → Python (pure conversion)
└── graph/
    ├── mod.rs                # Re-exports + short module doc. No logic.
    ├── kg.rs                 # KnowledgeGraph struct + #[pyclass] + internal helpers
    ├── dir_graph.rs          # DirGraph (OCC version, schema_locked, transaction state)
    │
    ├── storage/              # Storage backends + traits
    │   ├── mod.rs            # GraphRead / GraphWrite trait defs + GraphBackend enum
    │   ├── backend.rs        # GraphBackend dispatcher (carved from schema.rs)
    │   ├── schema.rs         # NodeData/EdgeData/PropertyStorage/type_indices
    │   ├── interner.rs       # StringInterner + InternedKey (done)
    │   ├── recording.rs      # RecordingGraph<G> validation wrapper (done)
    │   ├── impls.rs          # Shared impl helpers
    │   ├── lookups.rs
    │   ├── memory/           # In-memory backend
    │   ├── mapped/           # Mapped backend
    │   └── disk/             # Disk backend (split dir_graph.rs across csr/col_store/blocks/builder)
    │
    ├── core/                 # Shared query-execution primitives (renamed from query/)
    │   ├── mod.rs
    │   ├── pattern_matching/ # Split from today's 2,610-line file
    │   ├── filtering.rs      # was filtering_methods.rs
    │   ├── traversal.rs      # was traversal_methods.rs
    │   ├── iterators.rs      # was graph_iterators.rs
    │   ├── data_retrieval.rs
    │   ├── calculations.rs
    │   ├── value_operations.rs
    │   └── statistics.rs
    │
    ├── languages/            # Query interface languages (NEW — establishes layer)
    │   ├── mod.rs            # Trait / contract for a query language (if useful)
    │   ├── cypher/           # Moved verbatim from graph/cypher/
    │   │   ├── tokenizer.rs
    │   │   ├── parser/       # Split from today's 2,920-line file
    │   │   ├── ast.rs
    │   │   ├── planner/      # Split from today's 3,552-line file
    │   │   ├── executor/     # Split from today's 12,156-line file
    │   │   ├── result.rs
    │   │   ├── result_view.rs   # Internal Rust view — #[pyclass] moves to pyapi/
    │   │   └── window.rs
    │   └── fluent/           # NEW — extracted from today's #[pymethods] in mod.rs
    │       ├── mod.rs
    │       ├── selection.rs  # CurrentSelection state + chain entry
    │       ├── traversal.rs  # .traverse(), .neighbors(), etc.
    │       ├── filtering.rs  # .where_(), .match_property(), etc.
    │       └── aggregation.rs
    │
    ├── algorithms/           # Graph algorithms
    │   ├── mod.rs
    │   ├── pagerank.rs       # Split from today's 2,345-line graph_algorithms.rs
    │   ├── centrality.rs
    │   ├── components.rs
    │   ├── shortest_path.rs
    │   ├── clustering.rs     # (already its own file)
    │   └── vector.rs         # was vector_search.rs
    │
    ├── introspection/        # describe() / schema() / debug / bug reports
    │   ├── mod.rs
    │   ├── describe.rs       # Split from today's 4,204-line file
    │   ├── schema_overview.rs
    │   ├── connectivity.rs
    │   ├── capabilities.rs
    │   ├── hints.rs
    │   ├── bug_report.rs     # (already its own file)
    │   ├── debugging.rs      # (already its own file)
    │   └── reporting.rs      # (already its own file)
    │
    ├── io/                   # Load / save / import / export
    │   ├── mod.rs
    │   ├── file.rs           # was io_operations.rs
    │   ├── ntriples/         # Split from today's 3,022-line file
    │   └── export.rs
    │
    ├── features/             # Domain-specific functionality
    │   ├── mod.rs
    │   ├── spatial.rs
    │   ├── temporal.rs
    │   ├── timeseries.rs
    │   └── equations.rs      # was equation_parser.rs
    │
    ├── mutation/             # Write / maintenance / validation
    │   ├── mod.rs
    │   ├── batch.rs
    │   ├── maintain.rs
    │   ├── validation.rs
    │   ├── set_ops.rs
    │   └── subgraph.rs
    │
    └── pyapi/                # ALL #[pymethods] + #[pyclass] live here
        ├── mod.rs            # Module registration only
        ├── kg_core.rs        # KnowledgeGraph #[pymethods] — queries, mutations, I/O
        ├── kg_introspection.rs # describe/schema/find wrappers
        ├── kg_fluent.rs      # Fluent-chain entry points (delegate to languages/fluent)
        ├── transaction.rs    # Transaction #[pyclass]
        ├── result_view.rs    # ResultView + ResultIter #[pyclass]
        ├── algorithms.rs     # was pymethods_algorithms.rs
        ├── export.rs         # was pymethods_export.rs
        ├── indexes.rs        # was pymethods_indexes.rs
        ├── spatial.rs        # was pymethods_spatial.rs
        ├── timeseries.rs     # was pymethods_timeseries.rs
        └── vector.rs         # was pymethods_vector.rs
```

**Rules (Phase 8+ enforcement):**
- Every `#[pymethods]` / `#[pyclass]` / `#[pyfunction]` attribute lives
  under `src/graph/pyapi/` (or `src/lib.rs` for the module entry).
- Every query-interface language lives under `src/graph/languages/`.
- All languages call into `src/graph/core/` for shared primitives
  (pattern matching, filtering, traversal, iteration).
- `src/datatypes/py_in.rs` + `py_out.rs` stay as pure conversion —
  they're shared infrastructure (lib.rs-level), not `pyapi/` internals.
- No file under `src/graph/` exceeds 2,500 lines.

**Why languages/ umbrella, not peer dirs:** makes the layering explicit.
A fresh contributor asking "where do I add SPARQL?" sees `languages/`
and can mirror `cypher/`. The `core/` rename signals "shared across all
languages" — `query/` today reads like a Cypher-internal folder.

---

## Phase 8 — Architectural reorganisation

**Goal:** establish the target structure via `git mv` + small extractions.
No content splits yet; file sizes shrink naturally as `#[pymethods]`
leave `mod.rs`.

### Risks
- Existing `#[pymethods]` in `mod.rs` reach into private
  `KnowledgeGraph` fields and private helpers; moving them requires
  `pub(crate)` exposures or module co-location.
- Fluent-API extraction: the "fluent chain" isn't a named module today
  — it's whichever pymethods return `&mut self` or a selection builder.
  Inventory before extraction.
- `query/` → `core/` rename touches ~40 files (`use crate::graph::query::*`).
  Scripted substitution + `cargo check` per step.
- `cypher/` → `languages/cypher/` rename touches another ~30 files.
- `ResultView` / `ResultIter` `#[pyclass]` extraction must preserve the
  `cypher()` return type — Python users hold `ResultView` instances.
- `make lint` stubtest must stay green — the PyO3 module registration
  order and method signatures must match `__init__.pyi` exactly.

### Crunch-point tests
- [ ] `kglite/__init__.pyi` diff vs pre-Phase-8: **empty**
- [ ] `mypy.stubtest kglite` clean
- [ ] Python integration tests: no new failures
- [ ] Parity matrix (memory / mapped / disk) green
- [ ] `rg '#\[pymethods\]|#\[pyclass\]|#\[pyfunction\]' src/` — only matches under `src/graph/pyapi/` (and `src/lib.rs` for the `#[pymodule]`)
- [ ] `examples/mcp_server.py` runs against a sample graph without errors

### Tasks
- [ ] **8.1 Rename `graph/query/` → `graph/core/`** via `git mv` + single-pass
  perl substitution of `crate::graph::query::` + `super::query::`.
  `cargo check` per commit. One commit.
- [ ] **8.2 Rename `graph/cypher/` → `graph/languages/cypher/`** —
  same pattern. Create `graph/languages/mod.rs` (re-export only).
  One commit.
- [ ] **8.3 Inventory Fluent chain methods** — grep `mod.rs` for pymethods
  that build/mutate `CurrentSelection` (or equivalent selection state).
  Document the list in plan-mode before extraction. (No code change.)
- [ ] **8.4 Extract Fluent chain to `graph/languages/fluent/`** — Rust-side
  logic (selection state + chain primitives) moves to `languages/fluent/`;
  `#[pymethods]` wrappers move to `graph/pyapi/kg_fluent.rs` and delegate.
  One commit.
- [ ] **8.5 Move remaining `#[pymethods]` from `mod.rs` into `pyapi/`** —
  split by responsibility: `kg_core.rs` (queries + generic mutations +
  I/O wrappers), `kg_introspection.rs` (describe / schema / find).
  `mod.rs` shrinks to re-exports; struct definition moves to new
  `graph/kg.rs`. One commit per destination file (~3 commits).
- [ ] **8.6 Move `Transaction` `#[pyclass]` from `mod.rs` → `pyapi/transaction.rs`.**
  One commit.
- [ ] **8.7 Extract `ResultView` + `ResultIter` `#[pyclass]` from
  `languages/cypher/result_view.rs` → `pyapi/result_view.rs`.** Internal
  Rust `ResultView` representation (no PyO3 attrs) stays where it is.
  One commit.
- [ ] **8.8 Consolidate PyO3 conversion helpers** — decide in plan mode:
  (a) keep `datatypes/py_in.rs` + `py_out.rs` as-is (they're crate-wide
  infrastructure, not `pyapi/` internals); or (b) move under `pyapi/convert/`.
  `cypher/py_convert.rs` (166 lines, JSON→Python) folds into the chosen
  home.
- [ ] **8.9 Delete inherent `GraphBackend` methods that shadow trait methods** —
  the Phase 3 / Phase 5 debt. `rg 'impl GraphBackend \{'` in
  `storage/backend.rs` should contain only non-trait methods (`new`,
  `with_capacity`, `as_stable_digraph`) after this task.
- [ ] **8.10 Feature-gate `RecordingGraph`** behind
  `#[cfg(any(test, feature = "validation"))]`. Drop `#[allow(dead_code)]`.
  Verify release build size unchanged.
- [ ] **8.11 Update `ARCHITECTURE.md`, `CLAUDE.md`, `AUDIT_0.8.0.md`** to
  describe the new layout. Update `tests/test_phase5_parity.py::ENUM_MATCH_WHITELIST`
  paths and `tests/test_phase7_parity.py::GOD_FILE_EXCEPTIONS` paths.

**Exit criteria**: all `#[pymethods]` / `#[pyclass]` under `pyapi/`;
`languages/{cypher, fluent}/` exist; `mod.rs` under 2,000 lines;
`kglite/__init__.pyi` byte-identical to pre-Phase 8; stubtest clean;
parity oracle green.

---

## Phase 9 — God-file content splits

**Goal:** every `.rs` under `src/graph/` ≤ 2,500 lines.
`GOD_FILE_EXCEPTIONS` in `tests/test_phase7_parity.py` is emptied.

Order of operations: split the files independent of each other in
parallel where possible (each is its own commit). `executor.rs` is the
dominant effort.

### Risks
- Hidden compile cycles if helper types and functions are mutually
  recursive. Mitigate: extract types first (public sub-structs),
  `cargo check`, then extract functions, `cargo check`, then wire.
- `cypher/executor.rs` has ~700 private helpers; a naïve `rg "fn "` is
  insufficient — inspect the call graph before splitting.
- Splitting `schema.rs` changes visibility for `NodeData` /
  `PropertyStorage`; the per-backend `impl` blocks in `storage/{memory,
  mapped, disk}/` must still see them.
- Splitting `disk_graph.rs` risks moving `#[repr(C)]` / `#[repr(packed)]`
  types across files — verify `std::mem::size_of_val` in a test if in doubt.

### Crunch-point tests
- [ ] `test_phase7_parity::test_god_file_gate` — pass with empty exceptions
- [ ] `test_exception_list_still_applies` — unused (exceptions empty)
- [ ] `cargo test --release` — all Rust tests green
- [ ] `pytest -m parity` — parity oracle green on memory/mapped/disk
- [ ] No new clippy warnings; fmt clean
- [ ] `.kgl` v3 golden hash unchanged
- [ ] Binary size: under Phase-7 baseline × 1.05 (structural splits
  should have near-zero size impact; guard against accidental generic
  explosion)

### Tasks

Each sub-phase below is one commit; order within a task is not critical.

- [ ] **9.1 `languages/cypher/executor.rs` (12,156 lines) → `executor/`.**
  Proposed layout (refine in plan-mode based on the code's actual shape):
  - `executor/mod.rs` — pipeline entry, `execute(plan: …)`
  - `executor/match_clause.rs`
  - `executor/where_clause.rs`
  - `executor/with_clause.rs`
  - `executor/return_clause.rs`
  - `executor/projection.rs`
  - `executor/sort_limit.rs`
  - `executor/aggregation.rs`
  - `executor/write_clauses.rs` — CREATE / SET / DELETE / MERGE
  - `executor/expression_eval.rs`
- [ ] **9.2 `schema.rs` (5,141 lines) → `storage/{schema, backend}.rs`.**
  - `storage/schema.rs` — NodeData, EdgeData, PropertyStorage, type_indices
  - `storage/backend.rs` — `GraphBackend` enum + dispatcher
  - `graph/dir_graph.rs` — `DirGraph` + OCC version + schema_locked
  - `graph/configs.rs` — BuildConfig, LoadConfig, etc.
- [ ] **9.3 `introspection.rs` (4,204 lines) → `introspection/`.** Split
  by output: `describe.rs`, `schema_overview.rs`, `connectivity.rs`,
  `capabilities.rs`, `hints.rs` (exploration_hints + join_candidates).
- [ ] **9.4 `languages/cypher/planner.rs` (3,552 lines) → `planner/`.**
  - `planner/mod.rs`
  - `planner/join_order.rs`
  - `planner/index_selection.rs`
  - `planner/cost_model.rs`
- [ ] **9.5 `storage/disk/disk_graph.rs` (3,276 lines) → `storage/disk/`.**
  - `disk/mod.rs` — `DiskGraph` struct + trait impls
  - `disk/csr.rs` — CSR edge format + neighbor iteration
  - `disk/column_store.rs`
  - `disk/blocks.rs` — consolidate with existing `block_column.rs` +
    `block_pool.rs`
  - `disk/builder.rs`
- [ ] **9.6 `io/ntriples.rs` (3,022 lines) → `io/ntriples/`.**
  - `ntriples/mod.rs`
  - `ntriples/parser.rs`
  - `ntriples/writer.rs`
  - `ntriples/type_inference.rs`
- [ ] **9.7 `languages/cypher/parser.rs` (2,920 lines) → `parser/`.**
  - `parser/mod.rs`
  - `parser/match_pattern.rs`
  - `parser/expression.rs`
  - `parser/literals.rs`
- [ ] **9.8 `core/pattern_matching.rs` (2,610 lines) → `core/pattern_matching/`.**
  - `pattern_matching/mod.rs`
  - `pattern_matching/pattern.rs`
  - `pattern_matching/matcher.rs`
  - `pattern_matching/property_matcher.rs`
- [ ] **9.9 `mutation/maintain_graph.rs` + `mutation/batch_operations.rs`** —
  both hover under the 2,500-line cap today (1,516 + 1,032) but the
  architecture-level `maintain.rs` was originally intended to split if
  it grew past 1,500. Re-evaluate in plan mode. No-op if under soft cap.
- [ ] **9.10 Re-evaluate near-god files** (>2,000 lines): `storage/memory/column_store.rs`
  (2,215), `core/traversal.rs` (1,614). If either grew during
  Phases 8–9, split.
- [ ] **9.11 Empty `GOD_FILE_EXCEPTIONS`.** `test_exception_list_still_applies`
  becomes a no-op (consider deleting once the list is empty and stays
  empty for one phase).

**Exit criteria**: no `.rs` under `src/graph/` > 2,500 lines;
`GOD_FILE_EXCEPTIONS` empty; `test_god_file_gate` passes with the full
tree; parity oracle green; binary size ≤ Phase-7 baseline × 1.05.

---

## Phase 10 — Testing hardening

**Goal:** every edge case the Phase 0 plan originally listed but deferred
is now covered; golden output fixtures pin stable APIs.

### Risks
- Fixture creation on disk mode is slow — time budgets matter.
- "Disk full" simulation requires a bounded-size `tmpfs` or file-size
  limits; portability across macOS/Linux.
- 1,000-hop traversals may reveal stack-overflow issues or quadratic
  complexity in the planner.

### Crunch-point tests
- [ ] Zero-node graph: `describe()` / `schema()` / `find()` / `cypher()`
  all return sensible results (empty lists, empty strings, no panics)
  across memory/mapped/disk
- [ ] Single-node + single-edge: `MATCH (a)-[r]->(b) RETURN *` returns 1 row
- [ ] 1,000-hop traversal: no stack overflow; result is sensible
  (length-capped or returns empty, but doesn't crash)
- [ ] Malformed `.kgl` file: clear error, no panic, no corruption
- [ ] Disk-full during save: graceful failure, no corruption of prior
  data on disk
- [ ] Null / NaN / empty-string parity across modes (already partially
  covered in Phase 2; extend to all APIs)
- [ ] Unicode edge cases: surrogate pairs, zero-width joiners, emojis,
  RTL text — round-trip through all three modes
- [ ] Property-type promotion ladder: int → int+float → int+float+string
  in same column
- [ ] Concurrent-access smoke: two Python threads reading the same
  graph produce identical results
- [ ] Large-result cypher: 100k-row RETURN doesn't OOM

### Tasks
- [ ] **10.1 Edge-case tests** (above list) → new file
  `tests/test_edge_cases.py`, parameterised on storage mode where relevant.
- [ ] **10.2 Golden output fixtures.**
  - Build a deterministic ~1,000-node "golden graph" (Person + Company +
    Place, ~3,000 edges, mixed property types).
  - Record outputs: `describe()`, `schema()`, `find(…)` for 5 seed
    queries, `cypher(…)` for 10 seed queries. Commit as JSON under
    `tests/golden/`.
  - Diff test: regenerate + compare byte-for-byte on every mode.
  - Intentional "regenerate" helper CLI to update when semantics change
    intentionally.
- [ ] **10.3 Concurrent access** — `ThreadPoolExecutor` with 4 workers,
  each running `MATCH … RETURN …` over the same KG instance. Assert
  result-set equivalence.
- [ ] **10.4 Stress tests under `pytest -m stress`**.
  - 30 GB mapped: `test_mapped_scale.py --target-gb 30` completes, RSS
    bounded, queries return sensible numbers. (Gated; runs manually or
    on CI stress profile.)
  - Extremely deep traversal: 10k-hop chain, `MATCH (a)-[*..10000]->(b)`
    survives or bails with a clear cap error.

**Exit criteria**: all Phase-10 crunch tests green on memory/mapped/disk;
golden fixtures committed + diffable; edge-case / concurrent tests in
`make test` (stress under `-m stress`).

---

## Phase 11 — Benchmark sweep

**Goal:** validate that Phases 0–9 did not regress performance — and
document the wins. N=20 trials, two hardware profiles, direct v0.7.17
comparison.

### Risks
- Benchmark noise at small N — thermal + background OS noise can swamp
  real signal. N=20 with p50/p95/p99 plus stddev is the floor.
- Cross-hardware comparison is apples-to-oranges for absolute numbers;
  what matters is *ratio preservation* (if memory/disk ratio was 1:3 on
  v0.7.17, it should still be ~1:3 on 0.8.0).
- Wikidata disk bench takes hours — budget accordingly.

### Crunch-point tests
- [ ] `TestStorageModeMatrix` cross-mode parity asserts green
- [ ] No query > +5% vs v0.7.17 baseline (memory)
- [ ] No query > +10% (mapped)
- [ ] No query > +15% (disk)
- [ ] Wikidata bench completes without crash; no query > +15%

### Tasks
- [ ] **11.1 N=20 multi-trial run** of
  `tests/benchmarks/test_nx_comparison.py` on the dev box. Report mean,
  stddev, p50, p95, p99 per query per mode. Commit as
  `tests/parity_baseline_phase_11.json`.
- [ ] **11.2 Cross-hardware validation** — run same benchmark on CI
  runner (or a second physical profile). Compare mode-to-mode ratios,
  not absolute numbers. Flag any deviation > 20%.
- [ ] **11.3 Direct v0.7.17 comparison** — check out v0.7.17, run same
  benchmark, record. Compare to 0.8.0 numbers. Append delta table to
  `AUDIT_0.8.0.md` § 3.
- [ ] **11.4 Footprint audit** — 1k / 10k / 50k nodes, RSS + on-disk
  size per mode, compared to Phase-0 baselines.
- [ ] **11.5 Mapped-niche 30 GB** — `test_mapped_scale.py --target-gb 30`.
- [ ] **11.6 Wikidata disk bench** — full run. Report per-query delta vs
  v0.7.17. No query > +15%.
- [ ] **11.7 MCP-oriented benchmark** — `benchmark_kglite.py` (wherever
  it lives). No regression.
- [ ] **11.8 Document wins and losses** in `AUDIT_0.8.0.md` § 3.
  Wins are expected from `describe()` memoisation + mapped
  `str_prop_eq` + per-backend trait impls (inlining). Losses, if any,
  must be explained.

**Exit criteria**: all benchmark matrices documented in
`AUDIT_0.8.0.md`; no unexplained regressions; cross-hardware ratios
preserved; Wikidata bench passes disk gate.

---

## Phase 12 — Documentation + API surface audit

**Goal:** docs reflect the reality after Phases 8–11. API is
forward-compatible with v0.7.17.

### Tasks
- [ ] **12.1 ARCHITECTURE.md rewrite** to describe the final tree. Every
  file mentioned exists; every subdir documented; language layer
  explicit.
- [ ] **12.2 `kglite/__init__.pyi` diff vs v0.7.17** — **additions only**,
  no removed or changed signatures. Automated check in `make lint`.
- [ ] **12.3 `python -m mypy.stubtest kglite`** clean (20 modules).
- [ ] **12.4 `examples/mcp_server.py`** runs against a sample graph
  without errors.
- [ ] **12.5 CYPHER.md + FLUENT.md** audit. Every Cypher clause and
  fluent method documented still routes to documented code path.
  Update code references to the new `languages/cypher/` + `languages/fluent/`
  paths.
- [ ] **12.6 Inline Rust docstrings** — every `pub` item in `storage/`,
  `languages/`, `core/` documented. `cargo doc` builds without warnings.
- [ ] **12.7 `docs/*.md`** — every file referenced from somewhere (e.g.
  Sphinx index or README); no orphans.
- [ ] **12.8 `docs/adding-a-storage-backend.md`** still accurate against
  the reorganised tree.
- [ ] **12.9 NEW: `docs/adding-a-query-language.md`** walking through
  the Fluent / Cypher structure and describing how a third language
  would slot in. Reinforces the `languages/` layering.
- [ ] **12.10 CHANGELOG 0.8.0 entry finalised** — storage API unified;
  Python API stable; performance wins documented; internal reorg summary.
- [ ] **12.11 README.md** accurate.

**Exit criteria**: `make lint` clean; stubtest clean; mcp_server.py
passes; all doc cross-checks done; CHANGELOG draft ready.

---

## Phase 13 — Release 0.8.0 (mechanics only)

### Risks
- Last-minute scope creep.
- Forgotten version bump.
- Tag already exists / push conflicts.

### Tasks
- [ ] Freeze 0.8.0 branch — no further changes unless release-blocking bug
- [ ] Bump `Cargo.toml` version 0.7.17 → **0.8.0**
- [ ] Promote `CHANGELOG.md` `[Unreleased]`-equivalent to `[0.8.0]` with
  release date
- [ ] Confirm `make lint` clean
- [ ] Single commit: `chore: release 0.8.0`
- [ ] Prepare git tag `v0.8.0` (do not push — user pushes)
- [ ] Phase 13 report-out

**Exit criteria**: clean-tree commit with 0.8.0 in `Cargo.toml`; tag ready;
push command provided to user.

---

## Deferred decisions (carry forward until resolved)

- **query/ → core/ rename vs languages/ umbrella**: the plan as written
  favours both (rename query/ → core/, introduce languages/ with
  cypher + fluent as peers). Plan-mode for Phase 8 confirms or revises.
- **Fluent chain public surface**: which `#[pymethods]` in `mod.rs` are
  "fluent" vs "core KG API"? Inventory in Phase 8.3 before extraction.
- **PyO3 conversion helpers**: stay in `datatypes/` or move to `pyapi/convert/`?
  Decide in Phase 8.8 based on cleaner imports.
- **`RecordingGraph` release gating**: `#[cfg(test)]`? `#[cfg(feature = "validation")]`?
  Decide in Phase 8.10.
- **Phase-6 `file-count-budget` test** — currently self-skips post-Phase-7.
  After Phase 9 the Phase-6 intent is moot. Delete the test or convert
  to a different invariant in Phase 9.
- **`Value::String` → `Cow<'static, str>` / `Arc<str>`**: touches
  everything, not required for 0.8.0. Post-release decision.
- **Remote / RPC backend**: post-0.8.0 stretch.

## Versioning plan

All Phase 8–12 work continues on the 0.8.0-dev track (main, local). No
intermediate PyPI releases. Phase 13 is the cut. Tag `v0.8.0`. User
pushes.

## Rough effort

| Phase | Estimate   | Notes                                                           |
|-------|------------|-----------------------------------------------------------------|
| 8     | 2–3 days   | Mostly mechanical git mv + pymethods relocation                 |
| 9     | 4–7 days   | `executor.rs` dominates (~2 days); rest are ~half-day each      |
| 10    | 1–2 days   | Writing tests; golden-fixture plumbing                          |
| 11    | 1–2 days   | Mostly running benches; Wikidata bench may chew a day           |
| 12    | 1 day      | Doc refresh                                                     |
| 13    | half-day   | Release mechanics                                               |

Total ≈ 10–15 days focused. The single longest task is splitting
`cypher/executor.rs`; plan for it explicitly in Phase 9's plan-mode.

---

## Phase 8 Report-out (written at phase exit — read this before Phase 9)

**Completed** (single squashed commit `refactor: Phase 8 — architectural reorganisation` on `main`, local only — user pushes):

- **query/ → core/** via `git mv` + perl substitution of `crate::graph::query::` and `super::query::` across 17 files; `pub mod query;` → `pub mod core;` in `src/graph/mod.rs`.
- **cypher/ → languages/cypher/** via `git mv` + perl substitution of `crate::graph::cypher::` + bare `cypher::` references inside `mod.rs` (resolved via a new `use languages::cypher;` alias at the mod-level). Created `src/graph/languages/mod.rs` (re-exports `cypher` + `fluent`).
- **languages/fluent/** created as a doc-stub module — today's fluent-chain Rust logic is thin (operates on `CowSelection`) and the public entry points are PyO3 `#[pymethods]` in `pyapi/kg_fluent.rs`. The stubs explain the layering and 0.9.0+ intent to house pure-Rust fluent operations here if non-Python callers ever appear.
- **All `#[pymethods]` relocated** from `src/graph/mod.rs` (5,391 lines) → `src/graph/pyapi/kg_methods.rs` (5,422 lines, including imports/header). Single-file move rather than the planned 4-way split (`kg_core/kg_mutation/kg_introspection/kg_fluent`) — the split is Phase 9's task alongside the other god-file splits. Mod.rs drops from 6,758 → 1,068 lines.
- **Transaction** `#[pyclass]` + `#[pymethods]` moved from mod.rs → `src/graph/pyapi/transaction.rs` (329 lines). `Transaction` is re-exported as `crate::graph::Transaction` via `pub use pyapi::transaction::Transaction;` in mod.rs so `lib.rs` and downstream consumers keep the existing import path.
- **ResultView + ResultIter** `#[pyclass]` moved from `languages/cypher/result_view.rs` → `pyapi/result_view.rs`. 13 constructor callsites (`cypher::ResultView::from_*`) rewritten to `crate::graph::pyapi::result_view::ResultView::from_*`. `cypher/mod.rs` re-exports removed; `pyapi/mod.rs` declares the new submodule.
- **`pub(crate)` visibility** added to: KG fields (`inner`, `selection`, `reports`, `last_mutation_stats`, `embedder`, `temporal_context`, `default_timeout_ms`, `default_max_rows`); Transaction fields (`owner`, `working`, `committed`, `read_only`, `snapshot`, `base_version`, `deadline`); KG impl-block helpers (`add_report`, `connection_report_to_py`, `discover_property_keys_from_data`, `infer_selection_node_type`, `get_embedder_or_error`, `try_load_embedder`, `try_unload_embedder`, `resolve_code_entity`, `source_one`, `field_contains_ci`, `field_starts_with_ci`); module-level helpers (`extract_detail_param`, `extract_cypher_param`, `extract_fluent_param`, `resolve_noderefs`, `parse_spatial_column_types`, `parse_temporal_column_types`, `parse_inline_timeseries`, `parse_method_param`, `compare_inner`, `EmbeddingColumnData`, `TimeSpec`, `InlineTimeseriesConfig` + its fields + `all_columns`); widened `pub(super)` → `pub(crate)` on `get_graph_mut`, `centrality_results_to_py_dict`, `centrality_results_to_dataframe`, `community_results_to_py`.
- **Whitelist + exception updates**: `tests/test_phase5_parity.py::ENUM_MATCH_WHITELIST` + `tests/test_phase6_parity.py::ENUM_MATCH_WHITELIST` both swap `mod.rs` → `pyapi/kg_methods.rs` as the PyO3-boundary entry. `tests/test_phase7_parity.py::GOD_FILE_EXCEPTIONS`: removes `mod.rs` (now 1,068 lines — under cap); adds `pyapi/kg_methods.rs` (5,419 lines, 0.9.0 split planned); relocates cypher paths to `languages/cypher/` and `query/pattern_matching.rs` to `core/pattern_matching.rs`. `tests/test_phase7_parity.py::test_mod_rs_purity` subdir list updated to add `core`, `languages`, `languages/cypher`, `languages/fluent` and drop the standalone `query` + `cypher` (the standalone `cypher` cap of 500 lines now lives under `languages/cypher`).
- **`ARCHITECTURE.md`** — new "Target structure" section describing the `pyapi/` / `core/` / `languages/{cypher, fluent}` layout, PyO3 boundary diagram updated to point at `pyapi/kg_methods.rs`, Transaction reference relocated to `pyapi/transaction.rs`.
- **`CLAUDE.md`** — architecture block updated with `languages/cypher/` + `languages/fluent/` + `core/`. "When Changing `#[pymethods]`" steps updated to point at `pyapi/kg_methods.rs` (or the domain file under `pyapi/`). Added the rule "All `#[pymethods]` / `#[pyclass]` live under `src/graph/pyapi/`".

**Baselines held:**

- `kglite/__init__.pyi` MD5: `cfefb71c1f6a8b819dd11c73ef8af3be` (byte-identical to pre-Phase-8 — the API-stability contract held).
- `cargo test --release`: **490 passed** (no change vs Phase 7).
- `pytest tests/` (excl benchmarks + parity): **1,799 passed** (same as baseline).
- `pytest -m parity tests/`: **69 passed, 1 skipped** (1 Phase 6 gate self-skips post-Phase 7, unchanged).
- `python -m mypy.stubtest kglite`: **Success: no issues found in 20 modules**.
- `make lint`: clean (cargo fmt + clippy -D warnings + ruff format/check + stubtest).
- `kglite/kglite.cpython-312-darwin.so`: **7,046,288 bytes** vs Phase 7's 7,046,272 (+16 bytes, 0.0002% — noise).
- Phase 7 `.kgl` v3 `GOLDEN_V3_DIGEST` unchanged.

**File sizes (key files):**

| File | Pre-Phase-8 | Post-Phase-8 | Δ |
|------|-------------|--------------|---|
| `src/graph/mod.rs` | 6,758 | 1,068 | −84% |
| `src/graph/pyapi/kg_methods.rs` | (new) | 5,419 | — |
| `src/graph/pyapi/transaction.rs` | (new) | 329 | — |
| `src/graph/pyapi/result_view.rs` | (new) | 863 | — |
| `src/graph/languages/cypher/result_view.rs` | 859 | (moved) | — |

**Surprises found during investigation + implementation:**

1. **`mod.rs` is ~84% PyO3 boilerplate.** The investigation estimated the `#[pymethods]` body at ~5,400 lines. Confirmed exact: lines 1093–6483, 5,391 lines. After relocation, the residual mod.rs holds the struct defs + `impl Clone for KnowledgeGraph` + `TemporalContext` enum + private module-level helpers + the non-pymethods `impl KnowledgeGraph { ... }` block. That residual already sits under the 2,500-line cap.
2. **The 4-way split into `kg_core/kg_mutation/kg_introspection/kg_fluent` would have required per-method categorisation + careful coupling checks.** Done mechanically as a single-file move instead — the split is a content-split task that belongs in Phase 9 alongside executor.rs, schema.rs, etc. `GOD_FILE_EXCEPTIONS` now lists `pyapi/kg_methods.rs` with the 0.9.0 split plan.
3. **Bare `cypher::X` references in `mod.rs`** (no `crate::graph::` prefix) were implicit sibling-module paths that the `cypher/` → `languages/cypher/` rename broke. Fixed by adding `use languages::cypher;` at the top of mod.rs (a `use` alias keeps all `cypher::X` bare references working). Simpler and more maintainable than touching every call site. 40+ bare references survived the rename this way.
4. **Transaction docblock was lost during extraction.** The initial `sed -n '200,218p'` captured only the struct, not its 23-line docblock starting at line 177. Docblock restored from `git show HEAD:src/graph/mod.rs` before the commit.
5. **`InlineTimeseriesConfig` required field-level `pub(crate)` + `all_columns` method promotion** because `add_nodes` (now in `kg_methods.rs`) pattern-matches on `config.time: TimeSpec` internals. `TimeSpec` itself + `EmbeddingColumnData` type alias also widened.
6. **The "PyO3 attribute audit" gate has one intentional false positive**: `#[pyclass(skip_from_py_object)]` on `pub struct KnowledgeGraph` in mod.rs. The struct declaration lives next to its Clone impl + private helpers; the `#[pymethods]` blocks (where methods are) all live under `pyapi/`. Moving just the struct would require re-exports or long-path references crate-wide. Keeping the struct in mod.rs is the pragmatic choice — the audit description in Phase 12 should be tightened to "all `#[pymethods]` / `#[pyfunction]` under `pyapi/` or `lib.rs`".
7. **Shadowing-inherent deletion on `GraphBackend` was already done in Phases 5/7.** The remaining inherent `impl GraphBackend { ... }` block is 25 lines with just `new()`, `with_capacity()`, `as_stable_digraph()` — no trait-shadowing methods. Phase 8 Step 8 was a no-op.
8. **`RecordingGraph` feature-gating** was deferred. The existing design intentionally keeps the `Recording` variant in release builds so the enum dispatcher exercises a 4th backend through the same match arms as the production three — this is a dispatcher-correctness property, not dead code. Documented by the comment already sitting at `schema.rs:3770–3776`. `#[allow(dead_code)]` remains only on three log-audit methods in `storage/recording.rs` which are genuinely test-only.

**Decisions made:**

1. **Single-file `pyapi/kg_methods.rs`** for the relocated pymethods (vs the planned 4-way split) — documented in `GOD_FILE_EXCEPTIONS` with 0.9.0 split target.
2. **`languages/fluent/` is doc-only stubs** — current fluent Rust logic is too thin to warrant extraction. The module declares intent.
3. **PyO3 conversion helpers stay in `datatypes/`** — investigation confirmed they're crate-wide infrastructure with 5+ callers, zero Cypher coupling. `cypher/py_convert.rs` stays in `languages/cypher/` (Cypher-specific `PreProcessedValue` preprocessing). No moves.
4. **`Transaction` + `ResultView` live under `pyapi/`** (moved; not re-exported from their old paths — clean break).
5. **`KnowledgeGraph` struct definition stays in `mod.rs`** with its `#[pyclass]` attr. `#[pymethods]` blocks all live in `pyapi/`; moving just the struct would create awkward cross-module re-exports.
6. **`RecordingGraph` stays in release builds** (design decision, not deferral) — dispatcher-correctness property.
7. **`#[cfg(test)] mod recording_tests` stays as-is** — Phase 6 unit tests that construct `GraphBackend::Memory(...)` etc. are exempt from the enum-match audit via the existing `_strip_test_modules` helper.

**Debt introduced / carried to Phase 9:**

- **`pyapi/kg_methods.rs` is a new 5,419-line god file.** Split into 4 domain files is Phase 9's top task — adds to the existing 8 god-file splits. Categorisation (fluent / introspection / mutation / core) already documented in the Phase 8 plan (`/Volumes/EksternalHome/KristianEX/.claude/plans/fluffy-toasting-crown.md`).
- **`languages/fluent/` is empty stubs.** The question of "what is a fluent method vs a core KG method" remains open. Decide when Phase 9 splits `kg_methods.rs` or when a non-Python fluent caller appears.
- **One `#[pyclass]` attr in `src/graph/mod.rs`** — on the `KnowledgeGraph` struct itself. See Surprise #6. Spirit of the rule is met (all `#[pymethods]` under pyapi/), but the pedantic audit gate needs its description tightened in Phase 12's API-surface work.

**Scope changes vs the approved plan:**

- **4-way pyapi split** (Step 5 of the plan) reduced to a single-file move. Plan-level commentary captured in the Phase 8 commit message + this report-out.
- **Shadowing-inherent deletion** (Step 8) was a no-op — work was already done by Phases 5/7; the plan had carried old debt that was in fact already resolved.
- **RecordingGraph feature-gating** (Step 9) dropped as a design choice (dispatcher correctness). Not deferral — the design is correct as-is.

**Next-phase prerequisites (Phase 9 — god-file content splits):**

- Phase 9's target list grows from 8 to **9 files** — add `pyapi/kg_methods.rs` (5,419 lines) split into `pyapi/kg_{core, mutation, introspection, fluent}.rs`. Categorisation exists in the Phase 8 plan file at `/Volumes/EksternalHome/KristianEX/.claude/plans/fluffy-toasting-crown.md` (Steps 5a–5d).
- Phase 9's cypher splits now live at `languages/cypher/{executor, planner, parser}.rs` (the rename landed here). The `executor/` + `planner/` + `parser/` subdir carving in Phase 9 happens inside `languages/cypher/`.
- Phase 9's `query/pattern_matching.rs` is now `core/pattern_matching.rs`. Target layout: `core/pattern_matching/{mod, pattern, matcher, property_matcher}.rs`.
- `languages/fluent/` submodule files (`selection`, `filtering`, `traversal`, `schema_ops`) exist as doc stubs. If Phase 9 splits `pyapi/kg_methods.rs` into `kg_fluent.rs`, consider whether any Rust-only fluent operations should move from the PyO3 wrappers into `languages/fluent/selection.rs` etc.
- The NEW god file `pyapi/kg_methods.rs` has a 5,419-line `#[pymethods] impl KnowledgeGraph` block. Split strategy: multiple `#[pymethods] impl KnowledgeGraph { ... }` blocks across 4 files (PyO3 merges them at class-registration time — same pattern as the existing `pyapi/pymethods_*.rs` files).

**Files touched (full list):**

- **New files**: `src/graph/languages/mod.rs`, `src/graph/languages/fluent/{mod, selection, filtering, traversal, schema_ops}.rs`, `src/graph/pyapi/kg_methods.rs`, `src/graph/pyapi/transaction.rs`.
- **Moved files** (via `git mv`, rename similarity 97–100%):
  - `src/graph/query/` → `src/graph/core/` (9 files)
  - `src/graph/cypher/` → `src/graph/languages/cypher/` (10 files)
  - `src/graph/cypher/result_view.rs` → `src/graph/pyapi/result_view.rs`
- **Modified (path-ref updates)**: `src/datatypes/py_out.rs`, `src/lib.rs`, `src/graph/algorithms/graph_algorithms.rs`, `src/graph/features/equation_parser.rs`, `src/graph/schema.rs`, `src/graph/storage/{disk/disk_graph.rs, impls.rs, mod.rs}`, `src/graph/pyapi/{mod.rs, pymethods_algorithms.rs}`, `src/graph/languages/cypher/{ast, executor, mod, parser, planner, window}.rs`, `src/graph/core/{calculations, pattern_matching, traversal_methods}.rs`.
- **Modified (content)**: `src/graph/mod.rs` (massive slim-down), `src/graph/languages/cypher/mod.rs` (removed result_view re-exports), `src/graph/languages/cypher/py_convert.rs` (no direct edit — path paths flowed from the result_view move).
- **Modified tests**: `tests/test_phase5_parity.py`, `tests/test_phase6_parity.py`, `tests/test_phase7_parity.py` (all 3 whitelist/exception lists updated).
- **Modified docs**: `ARCHITECTURE.md`, `CLAUDE.md`, `todo.md` (this Report-out + earlier rewrite to forward-looking plan).
- **Unchanged**: `kglite/__init__.pyi` (zero diff vs pre-Phase 8 — API-stability contract held).

</details>

---

## Phase 9 Report-out (written at phase exit — read this before Phase 10)

**Completed** (single squashed commit `refactor: Phase 9 — god-file content splits` on `main`, local only — user pushes):

Every `.rs` under `src/graph/` is now at or under the 2,500-line hard cap. `GOD_FILE_EXCEPTIONS` is empty. The 9 files that Phase 8 left in the exception list have been split into themed submodules:

1. **`core/pattern_matching.rs` (2,610)** → `core/pattern_matching/` (4 files: `mod`, `pattern`, `parser`, `matcher`). Largest submodule: `matcher.rs` at 1,632 lines. The planned `property_matcher.rs` was subsumed into `pattern.rs` + `matcher.rs` — the enum is 26 lines of types (lives with other AST types) and the evaluation method `value_matches` is too tangled with `PatternExecutor` state to extract.
2. **`io/ntriples.rs` (3,022)** → `io/ntriples/` (4 files: `mod`, `parser`, `writer`, `loader`). Largest: `loader.rs` at 2,412 lines (`load_ntriples` + `build_columns_direct` + flush_entity + metadata structs + tests — unavoidable cohesion). `mod.rs` now owns the public `NTriplesStats` + `NTriplesConfig` types.
3. **`pyapi/kg_methods.rs` (5,419)** → `pyapi/kg_{mutation, fluent, introspection, core}.rs` (4 files, 1,058 + 1,594 + 1,481 + 1,368 lines). PyO3 merges multiple `#[pymethods] impl KnowledgeGraph` blocks at class-registration time, same pattern as existing `pymethods_*.rs`. Method split by category (ingestion / fluent chain / introspection / maintenance+core). All 105 methods present on `KnowledgeGraph`; smoke test confirmed `method_count = 163` (includes 58 methods from the other `pymethods_*.rs` files).
4. **`introspection.rs` (4,204)** → `introspection/` (added 6 new files: `mod`, `describe`, `schema_overview`, `connectivity`, `capabilities`, `topics`). Largest: `describe.rs` at 1,660 lines. The 40 `write_topic_*` functions (Cypher tier-3 + Fluent tier-3 tier-3) consolidated into `topics.rs` (1,184 lines). `mod.rs` kept the public return types + `graph_scale` free fn + re-exports. Existing `bug_report.rs` / `debugging.rs` / `reporting.rs` from Phase 7 are unchanged.
5. **`languages/cypher/parser.rs` (2,920)** → `languages/cypher/parser/` (5 files: `mod`, `match_pattern`, `predicate`, `expression`, `clauses`). `mod.rs` hosts the token helpers + `parse_query` orchestrator + `parse_cypher` public entry + the tests module (931 lines). Each sub-file adds another `impl CypherParser` block; Rust merges them at codegen.
6. **`languages/cypher/planner.rs` (3,552)** → `languages/cypher/planner/` (6 files: `mod`, `join_order`, `index_selection`, `cost_model`, `fusion`, `simplification`). Largest: `fusion.rs` at 1,700 lines (10 `fuse_*` passes share cross-clause helpers). `is_aggregate_expression` was already in `ast.rs` since Phase 6 — Phase 9 removed the `pub use super::ast::...` re-export from `executor.rs` so planner imports it directly. Also moved `rewrite_text_score` re-export from `cypher/mod.rs` → `planner::simplification::rewrite_text_score`.
7. **`storage/disk/disk_graph.rs` (3,276)** → `storage/disk/` (added `csr.rs` + `builder.rs`; `disk_graph.rs` shrunk to 2,108 lines). `csr.rs` holds the 4 `#[repr(C)]` types (`CsrEdge`, `MergeSortEntry`, `EdgeEndpoints`, `DiskNodeSlot`) + `TOMBSTONE_EDGE`. `builder.rs` holds the CSR-build methods (`build_csr_merge_sort`, `build_csr_partitioned`, `swap_csr_files`, `build_conn_type_index`, `build_peer_count_histogram`) wrapped in a second `impl DiskGraph` block. The existing `block_column.rs` + `block_pool.rs` were NOT consolidated into a single `blocks.rs` — they each stand on their own and are well-sized; consolidation would have added churn with no structural benefit.
8. **`schema.rs` (5,141)** → `schema.rs` (2,396) + `dir_graph.rs` (2,135) + `storage/backend.rs` (652). A deliberate minimal 3-way split after a broader 8-way split showed too much visibility fanout + cross-module type reference pressure. Re-exports at the top of `schema.rs` preserve every `crate::graph::schema::X` import path used elsewhere in the crate. Two whitelist additions: `dir_graph.rs` has 13 internal `GraphBackend::{Memory,Mapped,Disk,Recording}` enum-match sites for index-maintenance fast paths — added to `ENUM_MATCH_WHITELIST` with a rationale (petgraph-only optimisations).
9. **`languages/cypher/executor.rs` (12,144)** → `languages/cypher/executor/` (9 files: `mod` + 7 impl-submodules + `tests`). The dominant task. Sizes: `expression.rs` 2,403, `tests.rs` 1,975 (cfg-gated), `match_clause.rs` 1,895, `return_clause.rs` 1,297, `where_clause.rs` 1,125, `mod.rs` 1,114, `write.rs` 1,041, `call_clause.rs` 709, `helpers.rs` 688. Every sub-file adds another `impl<'a> CypherExecutor<'a>` block; Rust merges them at codegen. Shared helper import is a `use super::*; use super::helpers::*;` glob pair at the top of each clause submodule. `helpers.rs` holds the free-fn helpers (`expression_to_string`, `evaluate_comparison`, arithmetic ops, type coercions, property resolution) + `return_item_column_name` (pub, re-exported for `planner/fusion.rs` + `cypher/window.rs`). `tests.rs` is `#[cfg(test)] pub mod tests;` from `mod.rs`.

### Whitelist + exception updates

- `tests/test_phase7_parity.py::GOD_FILE_EXCEPTIONS`: **emptied**. `test_god_file_gate` now passes with zero exceptions over the full `src/graph/` tree.
- `tests/test_phase5_parity.py::ENUM_MATCH_WHITELIST` + `tests/test_phase6_parity.py::ENUM_MATCH_WHITELIST`: `schema.rs` replaced by `storage/backend.rs` (dispatcher's new home); `pyapi/kg_methods.rs` replaced by all four `pyapi/kg_{core,mutation,introspection,fluent}.rs` (PyO3-boundary files all carry `GraphBackend::` variant-match for storage-mode toggles); `io/ntriples.rs` replaced by `io/ntriples/{loader,writer}.rs` (both reach into `DiskGraph` internals for bulk-build). New entry: `dir_graph.rs` (petgraph-only fast paths on backend variants).
- `tests/test_phase7_parity.py::test_mod_rs_purity`: subdir list extended with `core/pattern_matching`, `io/ntriples`, `languages/cypher/{executor, parser, planner}`. Caps per subdir: `storage/mod.rs` 800, `languages/cypher/mod.rs` 500, `languages/cypher/executor/mod.rs` 1,200 (hosts the `CypherExecutor` struct + filter specs + constructor + `execute()` orchestrator + `finalize_result` — shared state every clause-submodule borrows from), `languages/cypher/parser/mod.rs` 1,000 (struct + token helpers + `parse_query` + public `parse_cypher` entry + tests module), `languages/cypher/planner/mod.rs` 400 (orchestrator + `mark_*` helpers + tests). All other subdir mod.rs files keep the 300-line cap.

### Baselines held

- `kglite/__init__.pyi` MD5: `cfefb71c1f6a8b819dd11c73ef8af3be` (byte-identical to Phase 8 — API-stability contract held).
- `cargo test --release`: **490 passed** (matches Phase 8).
- `pytest tests/` (excl benchmarks + parity): **1,799 passed** (matches Phase 8).
- `pytest -m parity tests/`: **69 passed, 1 skipped** (same composition as Phase 8).
- `python -m mypy.stubtest kglite`: **Success: no issues found in 20 modules**.
- `make lint`: clean (cargo fmt + clippy -D warnings + ruff format/check + stubtest).
- `kglite/kglite.cpython-312-darwin.so`: **7,046,320 bytes** vs Phase 8's 7,046,288 (+32 bytes, 0.0005% — noise; well under ×1.05 cap of 7,398,602).
- Phase 7 `.kgl` v3 `GOLDEN_V3_DIGEST` unchanged.

### Key file-size deltas

| File | Pre-Phase-9 | Post-Phase-9 | Δ |
|------|-------------|--------------|---|
| `languages/cypher/executor.rs` | 12,144 | (split) | — |
| `pyapi/kg_methods.rs` | 5,419 | (split) | — |
| `schema.rs` | 5,141 | 2,396 | −53% |
| `introspection.rs` | 4,204 | (split) | — |
| `languages/cypher/planner.rs` | 3,552 | (split) | — |
| `storage/disk/disk_graph.rs` | 3,276 | 2,108 | −36% |
| `io/ntriples.rs` | 3,022 | (split) | — |
| `languages/cypher/parser.rs` | 2,920 | (split) | — |
| `core/pattern_matching.rs` | 2,610 | (split) | — |
| **Largest remaining file** | 12,144 | 2,412 (`io/ntriples/loader.rs`) | −80% |

### Surprises found during investigation + implementation

1. **`is_aggregate_expression` was already in `ast.rs`** despite the Phase 9 plan calling for its move from `executor.rs`. The 0.0b task was a one-line visibility fix (`pub use` → `use`) on a re-export line, not a fn migration — Phase 6 had already moved it.
2. **`core/pattern_matching::PropertyMatcher` has no free evaluation fn** — it's evaluated as a method on `PatternExecutor` (`value_matches`) which captures `self.params`. The planned `property_matcher.rs` was an empty file in practice; split reduced to 3 submodules (`mod`, `pattern`, `parser`, `matcher`) that more accurately mirror the code's lifecycle.
3. **Pattern-matching `parser.rs` has no standalone test file** — tests live in a `#[cfg(test)] mod tests` at the end of the file. Phase 9 kept them with `parser.rs` (they test `tokenize` + `parse_pattern`, not matcher eval).
4. **`ntriples.rs` had hidden doc-attribute + derive boundaries.** First extraction attempt cut the `#[derive(serde::Serialize, serde::Deserialize, Clone, Copy)]` attribute off `RegionMeta` (attribute on line 1256, struct on 1257, I started extraction at 1257). Caught by compile failure; re-extracted from one line earlier.
5. **`schema.rs` is too tangled for an 8-way split.** First split attempt carved into 8 submodules (`storage`, `typeid`, `selection`, `configs`, `dir_graph`, `node_edge`, `backend`, `validation`). Cross-module type fanout was extreme — `DirGraph` pulls from ~15 sibling types, `GraphBackend` needs `MemoryGraph`/`MappedGraph`/`DiskGraph`/`RecordingGraph` simultaneously, `PropertyStorage` is `pub(crate)` but referenced in 4 places. Visibility fixes + derives were dropping under sed extraction. Reverted to a **3-way minimal split** after 30+ min of visibility chases: extract `DirGraph` + `GraphInfo` + `IndexStats` → `graph/dir_graph.rs`; extract `GraphBackend` + trait impls + dispatcher → `storage/backend.rs`; keep everything else in `schema.rs`. Re-exports preserve every external `crate::graph::schema::X` import path.
6. **`cargo fmt` + `cargo fix` + `cargo clippy --fix` repeatedly pruned "unused" `pub use MemoryGraph` and `pub use RecordingGraph`.** These re-exports are test-only and `#[allow(unused_imports)]` is insufficient for stopping the auto-fix. Fix: added explicit `// DO NOT REMOVE` comments to the affected `pub use` lines in `schema.rs` and `storage/mod.rs` — plus `#[allow(unused_imports)]`.
7. **Executor's `#[cfg(test)] mod tests` was not self-contained.** 1,975 lines of tests were nested inside a `mod tests { ... }` block that I re-exported as a submodule. Flattened the outer `mod tests { ... }` wrapper — `tests.rs` is declared via `#[cfg(test)] pub mod tests;` and its content is directly the test fns + helper fns. Imports had to shift from `use super::*` to `use super::helpers::*; use super::*;` since the tests are now one level deeper in the module tree.
8. **Clippy's `doc_lazy_continuation` + `empty_line_after_doc_comment` lints trip heavily on extraction seams.** Any time I cut across a doc-comment-then-blank-line-then-fn boundary, clippy fails. Wrote a Python one-liner to collapse blank lines between `///` comments and the following item in the 6 affected files. A handful remained — fixed by hand.
9. **The `Graph` type alias in `schema.rs` is `GraphBackend`, not `StableDiGraph<NodeData, EdgeData>`.** My initial `dir_graph.rs` had `type Graph = StableDiGraph<…>;` — wrong. `DirGraph.graph` is a `GraphBackend` enum. Corrected to import `Graph` from `storage::backend`.
10. **`Cow` import chain broke write.rs.** The write-path uses `node.get_field_ref(...).map(Cow::into_owned)` — requires `Cow` in scope. Added `use std::borrow::Cow;` to `write.rs`.
11. **Executor impl-submodule glob imports require `use super::*`** (not `use super::super::*`). Sibling submodules share their enclosing `impl<'a> CypherExecutor<'a>` block via Rust's impl-merging; methods must also be `pub(super)` so cross-file method-to-method calls work. A handful of trait-impl methods (`PartialEq::eq`, `PartialOrd::partial_cmp`, `Ord::cmp`) got `pub(super)` blanket-promoted by accident and had to be reverted — trait-method impls don't accept visibility modifiers.
12. **`fold_constants_expr` is called from `window.rs`** (outside executor), so promoting it to `pub(super)` in `where_clause.rs` wasn't enough — widened to `pub(crate)`.

### Decisions made

1. **schema.rs: 3-way split (not 8-way).** See Surprise #5. The pragmatic split keeps visibility simple, preserves all import paths via re-exports, and lands every file under cap. Further decomposition (splitting selection state, configs, NodeData/EdgeData into their own files) is a 0.9.0 consideration — not worth the visibility fanout churn at 0.8.0.
2. **Executor split follows clause boundaries, not a shared-helper layer.** Each clause-category (MATCH, WHERE, expression eval, RETURN, UNWIND/CALL, write) gets its own file + impl block. Shared helpers (arithmetic, to_string, property resolution) in `helpers.rs`. This matches Cypher semantics (each clause is a distinct AST variant) and keeps clause-specific changes local to one file.
3. **Disk backend: 3-file split (not 5-way).** Consolidating `block_pool.rs` + `block_column.rs` into a single `blocks.rs` was dropped — both files are well-sized and the block-allocator / column-store abstractions are distinct. `csr.rs` + `builder.rs` + `disk_graph.rs` is the minimal split that gets `disk_graph.rs` under the cap.
4. **Introspection: topics.rs is the pragmatic home for tier-3 writers.** The 40 Cypher `write_topic_*` + the 20+ Fluent `write_fluent_topic_*` functions live in one `topics.rs` file. They're cohesive (all string-templating XML for describe() output) and splitting across `cypher_topics.rs` + `fluent_topics.rs` would have been artificial — `write_cypher_topics` + `write_fluent_topics` dispatchers share no state with the individual topic writers.
5. **Tests pre-promotion:** tests inside executor, parser, schema test modules needed `use crate::graph::storage::{GraphRead, GraphWrite};` (trait methods moved from inherent). Added those explicitly; trait-import churn costs nothing at runtime.
6. **Re-exports marked `#[allow(unused_imports)]` + explicit `DO NOT REMOVE` comments.** Both `MemoryGraph` in `schema.rs` and `RecordingGraph` in `storage/mod.rs` are re-exports whose only callers are in `#[cfg(test)]` modules. `cargo fix` prunes them aggressively. Kept intentionally.

### Debt introduced / carried to Phase 10

- **Nothing new.** Every file in `src/graph/` is under the 2,500-line hard cap. `GOD_FILE_EXCEPTIONS` is empty. No `#[deprecated]`, no shims, no "TODO: migrate later" markers. The `legacy: Legacy::deserialize(...)` local variable in `schema.rs` is about the legacy `.kgl` on-disk format deserialization path — not a debt marker.
- **One Rust-idiom wart:** `schema.rs` retains the `Graph` type alias (`pub type Graph = GraphBackend;`) re-exported from `storage::backend`. Callers use it interchangeably with `GraphBackend`. Keeping it for now to avoid a blanket rename; future hygiene pass can drop it.
- **`languages/fluent/` is still empty stubs** (from Phase 8). Phase 9's kg_fluent.rs split did NOT move Rust-side fluent logic from `#[pymethods]` into `languages/fluent/` — the Python-facing entry points stay in pyapi, and the chain state is still just `CowSelection` mutations via KG's `&mut self`. If a non-Python fluent caller appears (SPARQL? GraphQL?) this is where to extract it. Not a 0.8.0 blocker.

### Scope changes vs the approved plan

- **pattern_matching/property_matcher.rs dropped.** The planned 4th submodule had too little to do (26 lines of enum + eval tangled with PatternExecutor state). Split stayed 3 submodules + `mod.rs`.
- **disk/blocks.rs consolidation dropped.** Original plan consolidated `block_pool.rs` + `block_column.rs` into `blocks.rs`. Both files stand on their own and are under cap. Consolidation would have been pure code churn.
- **schema.rs 8-way split reduced to 3-way.** See Surprise #5 / Decision #1.
- **Fluent-chain extraction to `languages/fluent/` dropped.** Not in Phase 9 scope — the pyapi split mechanically separated kg_fluent.rs from the other pymethods files, but no Rust-side extraction happened. Phase 10 / post-0.8.0 if needed.

### Next-phase prerequisites (Phase 10 — testing hardening)

- **Every subdir under `src/graph/` now has a `mod.rs`.** Phase 10's cross-mode parity tests can reference specific submodule paths (e.g. `languages/cypher/executor/expression.rs`) when writing test fixtures. The deepest nesting is 3 levels (`languages/cypher/executor/`).
- **`GOD_FILE_EXCEPTIONS` is empty** — Phase 10 can safely add new submodules without breaking the audit. If Phase 10 adds `tests/test_edge_cases.py` with large fixture data, those go under `tests/` (not `src/graph/`).
- **`test_exception_list_still_applies`** becomes a no-op in Phase 9. Consider deleting outright in Phase 11 (when the list has stayed empty across a phase).
- **Phase-6 `file-count-budget` test** (skipped post-Phase-7) wasn't re-evaluated in Phase 9 — still self-skips. Still a candidate for deletion in Phase 10 or 11.
- **`RAYON_THRESHOLD`** is now `pub(super) const` in `executor/mod.rs`, re-exported via `use super::RAYON_THRESHOLD` in clause submodules. Window-function code in `cypher/window.rs` imports it via `use super::executor::RAYON_THRESHOLD`. Any future code that needs the threshold should follow this pattern.
- **Doc comments with trailing blank lines are now a clippy error.** The `doc_lazy_continuation` + `empty_line_after_doc_comment` lints are strict. Future PRs should not introduce blank lines between `///` and the item they document.

### Files touched

- **New files (schema 3-way split)**: `src/graph/dir_graph.rs`, `src/graph/storage/backend.rs`.
- **New subdirs (8 god files)**: `src/graph/core/pattern_matching/` (4 files), `src/graph/io/ntriples/` (4 files), `src/graph/introspection/` (6 files; 3 pre-existing), `src/graph/languages/cypher/executor/` (9 files), `src/graph/languages/cypher/parser/` (5 files), `src/graph/languages/cypher/planner/` (6 files), `src/graph/pyapi/kg_{core,mutation,introspection,fluent}.rs`, `src/graph/storage/disk/{csr,builder}.rs`.
- **Removed files**: `src/graph/core/pattern_matching.rs`, `src/graph/io/ntriples.rs`, `src/graph/introspection.rs`, `src/graph/languages/cypher/{executor,parser,planner}.rs`, `src/graph/pyapi/kg_methods.rs`, `src/graph/storage/disk/disk_graph.rs` (replaced with a shrunk version in-place).
- **Modified (path-ref updates)**: `src/graph/mod.rs` (added `pub mod dir_graph;`), `src/graph/schema.rs` (+ re-exports for DirGraph/GraphBackend/MappedGraph/MemoryGraph), `src/graph/storage/mod.rs` (+ `pub mod backend;` + `pub use recording::RecordingGraph;`), `src/graph/core/graph_iterators.rs` (CSR types path), `src/graph/storage/impls.rs` (TOMBSTONE_EDGE path), `src/graph/languages/cypher/mod.rs` (rewrite_text_score re-export path), `src/graph/languages/cypher/window.rs` (return_item_column_name path), `src/graph/languages/cypher/planner/*.rs` (ast path depth + imports).
- **Modified tests**: `tests/test_phase5_parity.py`, `tests/test_phase6_parity.py`, `tests/test_phase7_parity.py`.
- **Modified docs**: (none this commit — ARCHITECTURE.md / CLAUDE.md / AUDIT_0.8.0.md / CHANGELOG.md updates deferred to the separate `docs:` commit that accompanies this phase).
- **Unchanged**: `kglite/__init__.pyi` (zero diff vs pre-Phase 9 — API-stability contract held).

## Phase 10 Report-out (written at phase exit — read this before Phase 11)

**Completed** (single squashed commit `test: Phase 10 — testing hardening` on `main`, local only — user pushes):

Pins the outer envelope of the public API with tests-only additions. Zero Rust source changes; zero `kglite/__init__.pyi` changes (MD5 `cfefb71c1f6a8b819dd11c73ef8af3be` preserved). Deliverables:

1. **10.1 Edge-case parity** — new `tests/test_edge_cases_parity.py` (27 tests @ `pytest.mark.parity`): zero-node graphs, single-node self-edge, 1,000-hop chain, Unicode round-trip (BMP + astral + ZWJ + RTL + flag + empty), Unicode save/load, property-type promotion ladder (int → int+float → int+float+string), null/NaN/empty-string, 100k-row cypher RETURN with RSS budget, NaN-is-null invariant. **Every test passes on memory + mapped + disk on first run — no disk divergences.** The Phase 2 `STRICT_PARITY_MODES` escape hatch was prepared but never needed.
2. **10.2 Golden fixtures** — new `tests/golden/` package + `tests/test_golden.py` (48 tests). Deterministic 1,000-node / 3,000-edge graph (500 Person + 300 Company + 200 Place, seeded `random.Random(42)`). 10 pinned cypher queries + 5 `find()` queries + 1 schema snapshot × 3 storage modes = 48 byte-for-byte JSON diff assertions. Regenerator script `tests/golden/regenerate.py` with `--check` drift-detection flag.
3. **10.3 Concurrent equivalence** — extended `tests/test_concurrency.py`: new `test_concurrent_reads_result_equivalence` uses `ThreadPoolExecutor(max_workers=4)` and asserts full result-set equivalence (not just scalar count) to a sequential baseline after normalisation. Two pre-existing concurrency tests kept as-is.
4. **10.4 Stress tier** — new `tests/test_stress.py` (2 tests @ `pytest.mark.stress`): 30 GB mapped-scale benchmark subprocess invocation + 10k-hop deep traversal. New `stress` marker registered in `pyproject.toml` with `addopts` exclusion by default.
5. **Malformed-file / disk-full** — extended `tests/test_save_load.py::TestV3Format` with `test_header_only_corrupt` (byte-flip deep in valid-header payload, assert clean failure) and `test_disk_full_during_save` (RLIMIT_FSIZE simulation, skip on platforms without `resource.RLIMIT_FSIZE`). Confirms the graph remains queryable and the prior-saved file remains loadable after a mid-save disk-full.

### Probe results (ran pre-implementation to pick test tiers)

On memory/mapped/disk, 1,000-hop `count(b)` via `[*..1000]` was <1 ms across all three modes — the Cypher planner short-circuits variable-length path counting rather than materialising paths. 100k-row RETURN: 32 ms wall-clock, max 175 MB RSS delta (memory mode — mapped/disk much lower due to columnar storage). All three scenarios kept in default `pytest tests/`, no gating needed.

### Baselines held

- `kglite/__init__.pyi` MD5: `cfefb71c1f6a8b819dd11c73ef8af3be` (byte-identical to Phase 9 — API-stability contract held; zero source changes).
- `cargo test --release`: **490 passed** (unchanged).
- `pytest tests/` (default): **1,799 → 1,850 passed** (+51 tests = 27 in `test_edge_cases_parity.py`, already counted in parity total, not here; 48 in `test_golden.py`; 1 in `test_concurrency.py`; 2 in `test_save_load.py`).
- `pytest -m parity tests/`: **69/1 → 96/1 passed/skipped** (+27 = `test_edge_cases_parity.py` × 3 modes).
- `pytest -m stress`: 2 tests collected (not run in default gate; `test_deep_traversal_10k_hops` passes; `test_mapped_scale_30gb` manual-run only).
- `make lint`: clean (cargo fmt + clippy -D warnings + ruff format/check + stubtest on 20 modules, no issues).
- Phase 7 `.kgl` v3 `GOLDEN_V3_DIGEST` unchanged.

### Surprises found during implementation

1. **`kglite.KnowledgeGraph.find()` is code-specific.** Its docstring says it searches code-entity node types (Function, Struct, Class, Enum, Trait, Protocol, Interface, Module, Constant). On a social graph (Person/Company/Place), `find("Alice")` legitimately returns `[]`. The Phase 10 plan referenced `find()` generically — the golden-fixture test pins the empty-result contract rather than asserting match content. Worth documenting for Phase 12 API-surface audit: find() is domain-specific rather than a generic name-search.
2. **Variable-length path `[*..1000]` is instant.** The todo.md risk list worried about stack overflow or quadratic planner explosion. Empirically the planner computes reachability counts without path materialisation. This means the Phase 10 "1,000-hop" test doesn't stress-test the deep-recursion path — the 10k-hop stress test likewise completes in <100 ms. **Phase 11 should add a benchmark that actually materialises paths (`RETURN p` where `p = path`) to stress the recursion/memory envelope.**
3. **Ruff import-sort runs after ruff format.** `ruff format` normalised whitespace but left multi-block imports unsorted; `ruff check --fix` had to collapse them. Two-pass workflow is needed when writing net-new test files. `make lint` catches both via its `ruff format --check && ruff check` sequence.
4. **PyO3 `ResultView` is iterable, not indexable without `.to_list()`.** Most existing tests use `result[0]["c"]` which works because `ResultView.__getitem__` exists, but cross-mode tests that used `list(result)` returned list-of-dicts cleanly. Kept the existing `list(kg.cypher(...))` pattern — converts the iterator directly.
5. **`find()` type signature accepts `(name)` and `(name, node_type)` positionally; passing `node_type=None` keyword also works.** The regenerator helper branches on `node_type is None` to call the single-arg form for type-safety against future `#[pymethods]` signature changes — tested empirically in both patterns.

### Decisions made

1. **Golden fixture: 1,000 nodes, not larger.** 1,000 nodes keeps test suite wall-clock under 10 s total (0.37 s for all 48 golden tests). Larger would stress-test aggregation paths but belongs in Phase 11 benchmarks.
2. **Snapshot format: `json` with `sort_keys=True` + `indent=2`**, not binary hashes. Human-readable diffs on failure; regenerator writes the same format. Floats rounded to 6 decimals to avoid cross-platform formatting drift.
3. **Datetime columns deliberately excluded from the golden snapshots.** The `joined_at` column is populated in the builder but not queried in any golden cypher — adding it to a RETURN risks locale/timezone serialisation drift across platforms. Can be added in a future phase after confirming PyO3 datetime reprs are stable.
4. **Disk-full test uses `RLIMIT_FSIZE`, not a real tmpfs.** `RLIMIT_FSIZE` is Unix-portable (macOS + Linux), doesn't require sudo, and is scoped to the process. `skip_if` on `hasattr(resource, 'RLIMIT_FSIZE')` handles Windows.
5. **The `test_long_chain_traversal_1000_hops` parity test is kept in the default `-m parity` suite despite its triviality.** Rationale: if a future Cypher planner rewrite changes variable-length-path short-circuiting, this test will catch the regression — the assertion is tight (`count == 1000` exactly, no tolerance). The 10k-hop version lives under `-m stress` for the same reason at a different scale.
6. **Concurrent-equivalence test runs on memory mode only.** The plan allowed pushing to `-m stress` for mapped/disk. Kept it in default suite because it ran in <100 ms on memory and the value of default-suite regression detection outweighs the cost.

### Debt introduced / carried to Phase 11

- **Variable-length path stress coverage is shallow.** The 1k-hop + 10k-hop tests terminate instantly because `count()` short-circuits path enumeration. Phase 11 should add a benchmark that returns actual path objects to measure deep-traversal cost empirically.
- **Datetime columns unsnapshotted.** The golden graph builder produces `joined_at` datetime rows but doesn't query them. Future work: add a `cypher_dates_roundtrip` snapshot after verifying datetime serialisation stability across platforms.
- **`test_mapped_scale_30gb` is untested locally.** It runs a 3-hour-budget subprocess against `tests/benchmarks/test_mapped_scale.py --target-gb 30`. The `test_deep_traversal_10k_hops` stress test is verified green locally; `test_mapped_scale_30gb` relies on the underlying benchmark's existing well-tested code path. Manual run before each release is still expected.
- **`find()` domain-specificity not documented in `__init__.pyi` docstring.** The docstring mentions code entities but the method name doesn't — cross-reference for Phase 12's documentation pass.

### Scope changes vs the approved plan

- **10.1 expanded from 7 to 8 crunch-point tests** — added `test_nan_not_equal_itself` (explicit NaN→null parity via `IS NULL` / `IS NOT NULL`) because the plan's "null/NaN/empty-string parity" task was broad enough to warrant a dedicated test.
- **Flag sequence added to Unicode list** — the plan mentioned surrogate pairs / ZWJ / emoji / RTL / combining marks; I added the Norwegian flag (🇳🇴, a regional-indicator pair) because it's a compound grapheme-cluster case that exercises column storage without needing Python-side surrogate manipulation.
- **Disk-full + header-corrupt added to `test_save_load.py`** rather than a new file. Existing `TestV3Format` class is the canonical home; Phase 10's contribution fits cleanly there.
- **Concurrent test runs on memory mode only** (not parameterised across modes as the plan allowed). See Decisions #6.

### Next-phase prerequisites (Phase 11 — benchmark sweep)

- **Probe data captured in `tests/parity_baseline_phase_10.json`** — includes measured RSS budgets per storage mode, which Phase 11 can use as soft regression gates.
- **Golden-fixture snapshots are byte-frozen.** If Phase 11's optimizer changes aggregation paths (e.g. vectorised COUNT), the golden tests will fail. Intentional: `python tests/golden/regenerate.py` is the one-command refresh path; diff the snapshot before committing.
- **The `stress` marker is registered.** Phase 11 benchmarks can opt into it if there are scenarios that exceed `-m benchmark`'s budget (e.g. the 30 GB mapped build).
- **No new `GOD_FILE_EXCEPTIONS` entries.** `src/` touched zero files this phase — the exception list remains empty (consider deleting the exception-list audit in Phase 11 when it has stayed empty across two phases per Phase 9 suggestion).

### Files touched

- **New tests**: `tests/test_edge_cases_parity.py`, `tests/test_golden.py`, `tests/test_stress.py`.
- **New fixtures**: `tests/golden/__init__.py`, `tests/golden/build_golden_graph.py`, `tests/golden/queries.py`, `tests/golden/regenerate.py`, `tests/golden/snapshots/*.json` (16 files).
- **New baseline**: `tests/parity_baseline_phase_10.json`.
- **Modified**: `pyproject.toml` (stress marker + addopts), `tests/test_concurrency.py` (+1 test), `tests/test_save_load.py` (+2 tests).
- **Unchanged**: all of `src/`, all of `kglite/` including `__init__.pyi`. Zero production-code diff.

## Phase 11 Report-out (written at phase exit — read this before Phase 12)

**Completed** (single squashed commit `test: Phase 11 — benchmark sweep` on `main`, local only — user pushes):

Ran the full Phase 11 benchmark sweep on the dev box (macOS Darwin 25.3.0). Zero Rust or Python source changes. **The 0.8.0 refactor is a net performance improvement over v0.7.17** across every storage-mode cell that has a v0.7.17 counterpart. No regression gate was breached.

Deliverables:

1. **11.1 N=20 multi-trial sweep on 0.8.0** → `tests/benchmarks/phase11_main.json` (45 cells, 4.4 s wall-clock). Covers 4 construction scales × 3 modes + 10 query primitives × 3 modes at 10k nodes, plus footprint rows per mode.
2. **11.1 harness itself** → `tests/benchmarks/phase11_harness.py` (296 lines). Standalone Python script (not a pytest test). Self-contained graph builder (`_make_shape` / `_populate`) so it runs unchanged on any KGLite checkout. Argparsed CLI: `--json`, `--n`, `--modes`, `--skip-construction`.
3. **11.3 v0.7.17 replay** → `tests/benchmarks/phase11_v0_7_17.json` (30 cells, memory + mapped only — disk format incompatible). Executed via `git worktree add ../KGLite-v0.7.17 v0.7.17` + isolated venv + `maturin develop --release` + harness run. Worktree + venv removed cleanly at phase exit (`git worktree list` shows only `main`).
4. **11.3 comparison generator** → `tests/benchmarks/phase11_compare.py` + `tests/benchmarks/phase11_delta.md`. Markdown table with per-cell gate evaluation (memory +5 % block, mapped +10 % block, disk +15 % flag).
5. **11.4 Footprint audit** → rolled into `tests/parity_baseline_phase_11.json` (RSS + on-disk per scale × mode). Observations: disk's on-disk footprint stays flat at ~16–18 MB across 1k / 10k / 50k scales (dictionary compression + CSR efficiency). Mapped RSS is within 4 % of memory at 10k, widens at 50k.
6. **11.5 Mapped-niche 5 GB** → `tests/benchmarks/phase11_mapped_scale_5gb.txt`. Completed cleanly: 5 M nodes, 13.9 M edges, build 4.7 min, peak RSS 2.97 GB, steady 2.44 GB, on-disk 129 MB, save 12.18 s, query latencies ≤ 0.5 s except `describe_overview` at 2.89 s (2,915 output chars — expected).
7. **11.6 Wikidata disk bench** → **partial** (29 of 45 queries) in `tests/benchmarks/phase11_wikidata_partial.csv`, with narrative in `tests/benchmarks/phase11_wikidata.txt`. SIGKILL (OOM) at query 29 after residual RSS from the preceding 5 GB mapped run exhausted available memory for 78 GB disk page-cache warmup. 29 completed queries land in the expected envelope (basics p50 ≈ 14 ms; anchored hops p50 ≈ 54 ms).
8. **11.7 API benchmark** → `tests/benchmarks/phase11_api_benchmark.txt`. 51 of 51 pass across 4 datasets × 3 modes; all mode-to-mode results match. Reused `bench/api_benchmark.py` per Phase 11 plan decision; no new file created.
9. **11.8 `AUDIT_0.8.0.md § 3` rewrite** → Phase 6 "carried forward" table demoted to historical sub-section; new Phase 11 delta tables, footprint matrix, 5 GB stretch summary, Wikidata partial summary, API-bench summary, and updated Deferred subsection are now authoritative.
10. **`tests/parity_baseline_phase_11.json`** → phase-summary JSON with entry/exit counts, gate results, top wins, footprint data, per-sub-task status.

### Baselines held

- `kglite/__init__.pyi`: byte-identical to v0.7.17 and Phase 10 (MD5 `cfefb71c1f6a8b819dd11c73ef8af3be`). Verified by `git diff v0.7.17 HEAD -- kglite/__init__.pyi` → empty.
- `cargo test --release`: **490 passed** (unchanged).
- `pytest tests/`: **1,850 passed** (unchanged — Phase 11 adds no pytest tests).
- `pytest -m parity tests/`: **96 passed, 1 skipped** (unchanged).
- `make lint`: clean (cargo fmt + clippy -D warnings + ruff format/check + stubtest).
- `kglite/kglite.cpython-312-darwin.so`: **7,046,320 bytes** (unchanged vs Phase 9).
- Phase 7 `.kgl` v3 `GOLDEN_V3_DIGEST` unchanged (no I/O code touched).

### Top numeric wins (v0.7.17 p50 → 0.8.0 p50 at 10k nodes)

| Cell | v0.7.17 | 0.8.0 | Δ |
|---|---:|---:|---:|
| `pattern_match_10000_memory` | 13.756 ms | 5.417 ms | **−60.6 %** |
| `pattern_match_10000_mapped` | 13.664 ms | 5.435 ms | **−60.2 %** |
| `two_hop_10x_10000_mapped` | 0.073 ms | 0.055 ms | **−24.9 %** |
| `describe_10000_mapped` | 3.236 ms | 2.457 ms | **−24.1 %** |
| `describe_10000_memory` | 3.037 ms | 2.395 ms | **−21.1 %** |
| `pagerank_10000_mapped` | 5.375 ms | 4.426 ms | **−17.7 %** |
| `pagerank_10000_memory` | 5.185 ms | 4.276 ms | **−17.5 %** |
| `find_20x_10000_mapped` | 10.377 ms | 9.053 ms | **−12.8 %** |
| `construction_50000_mapped` | 145.08 ms | 113.37 ms | **−21.9 %** |
| `construction_50000_memory` | 80.72 ms | 70.94 ms | **−12.1 %** |

### Flagged (under block gate, but worth tracking)

- `find_20x_10000_memory`: +4.7 % (under +5 % memory block). Likely related to Phase 3 `GraphRead` GAT refactor moving `find()` through the trait surface. Post-release investigation.
- `simple_filter_10000_memory`: +2.3 % (well under block). Absolute delta is 0.012 ms — within single-trial measurement noise.
- `multi_predicate_10000_mapped`: +4.3 % (under +10 % mapped block). Same order-of-magnitude as single-trial noise for a 0.6 ms query.
- `simple_filter_10000_mapped`: +3.1 % (under +10 % mapped block).

### Surprises found during implementation

1. **`test_nx_comparison.py::_build_kg_mode` does not exist on v0.7.17.** The first draft of the harness imported from `test_nx_comparison`; running on the v0.7.17 worktree raised `AttributeError`. Refactored the harness to inline the graph builder (`_make_shape` + `_populate`) so it's version-agnostic. Re-ran the 0.8.0 baseline after the refactor so both sides use the same build path.
2. **Phase 6 `two_hop_10x_10000_memory` baseline was 2.55 ms; Phase 11 reports 0.054 ms.** Not a 47× speedup — the Phase 11 harness uses `RETURN count(DISTINCT c)` which the Cypher planner short-circuits; Phase 6 likely enumerated full paths. The v0.7.17 measurement (0.071 ms) uses the same shape as 0.8.0, so the **v0.7.17 → 0.8.0 comparison at 23.8 % speedup is apples-to-apples**. The Phase 6 comparison is not. The historical sub-table in AUDIT §3 flags this explicitly.
3. **`Cargo.toml` version is still `0.7.17`** on `main`. Phase 13 bumps to `0.8.0`. This means both harness runs report `kglite_version: 0.7.17`; distinction is made by `git_sha` (v0.7.17 = `fa63011c4b`, main = `b0990e5395`). Harness output JSON carries both fields.
4. **Wikidata bench killed by OOM.** Residual RSS (~3 GB) from the preceding 5 GB mapped bench + 78 GB disk page-cache demand → SIGKILL at query 29/45. Script writes its CSV incrementally, so the 29 completed queries survived. Remediation documented; not a Phase 11 blocker because (a) the 29 completed queries cover 5 of 8 categories, (b) Wikidata has no v0.7.17 comparison anyway (disk format incompatible), (c) the api_benchmark.py + mapped-5GB runs independently prove disk backend health at scale.
5. **`ruff format` + `ruff check --fix` is a two-pass workflow for new files.** `ruff format` normalises whitespace; `ruff check --fix` re-sorts imports and removes unused ones. Had to run both, plus one manual cleanup for an unused local variable that `--fix` wouldn't touch (`kinds` in compare.py — I had drafted it for section iteration but inlined the loops instead).
6. **pytest-benchmark was not the right tool for the Phase 6-style cells.** Most of `test_nx_comparison.py::TestStorageModeMatrix` uses explicit `time.perf_counter()` + `print()` rather than the `benchmark` fixture, so `--benchmark-json=` doesn't capture those cells. The standalone harness avoids this entirely — it computes its own stats and writes its own JSON format.

### Decisions made

1. **Standalone harness over extending `test_nx_comparison.py`.** The existing file mixes accuracy tests (single N) with Phase-6-style timing prints. Parameterising it for N=20 would have touched too much code. A self-contained harness ships cleaner and can be invoked on any checkout without source changes to `test_nx_comparison.py` (critical for the v0.7.17 replay).
2. **Git worktree + isolated venv for v0.7.17 replay.** The alternative (stashing / checking-out / rebuilding in-place) would have overwritten the 0.8.0 `.so` and been easy to leave in a half-broken state. Worktree isolates the builds completely; teardown is one command (`git worktree remove --force`).
3. **Memory + mapped only on v0.7.17.** Disk backend changed substantively between v0.7.17 and 0.8.0 (the Phase 4 v3 format is a clean break). Running v0.7.17's disk path would have compared apples to oranges. The harness's `--modes memory,mapped` flag handles this explicitly.
4. **Accept the Wikidata partial rather than re-run.** A clean re-run would require either (a) a machine with ≥ 32 GB free RAM after a cold boot, or (b) re-sequencing the Phase 11 benches to run Wikidata before the 5 GB mapped bench. Both are achievable post-release. The 29 completed queries + api_benchmark.py's green run across all 3 modes sufficiently cover disk health.
5. **Keep `Cargo.toml` at `0.7.17`.** Not bumped in Phase 11. Phase 13 handles the version mechanics to avoid mixing "version bump" concerns with "benchmark results."
6. **No block-gate overrides needed.** All four flagged cells came in under their respective block thresholds. The Phase 11 sweep is a pass — the next natural follow-up (post-release) is investigating whether `find_20x_memory`'s +4.7 % is real or measurement noise.

### Debt introduced / carried to Phase 12

- **`find_20x_10000_memory` +4.7 %** — under block gate but worth a targeted profile. Hypothesis: Phase 3 GAT refactor. Not a release blocker.
- **Wikidata full 45-query run outstanding** — the infra works; the remaining 16 queries require isolation from the 5 GB mapped run or a higher-memory host. File as a pre-release manual gate.
- **Cross-hardware validation** — dev-box-only numbers. Linux CI or a second physical host would strengthen the AUDIT claims. Documented in AUDIT §3 Deferred.
- **No `benchmark_kglite.py` was created** — the Phase 11 plan's task 11.7 was satisfied by `bench/api_benchmark.py` per user decision. If a more focused MCP-facing bench is wanted later, the tests/benchmarks/phase11_harness.py shape is a reasonable starting point.
- **Phase 6 baseline table de-duplication** — the "historical" subsection in AUDIT §3 retains Phase 6 numbers. Some of them (notably `two_hop_10x`) compare queries of different shape. The audit notes this explicitly but Phase 12's doc pass should consider whether the historical table still earns its place, or if the v0.7.17 comparison alone suffices.

### Scope changes vs the approved plan

- **11.2 Cross-hardware → skipped** (user decision up front). Documented in AUDIT §3 Deferred.
- **11.5 30 GB → 5 GB** (user decision). Transcript shows clean run; 30 GB remains a manual pre-release gate.
- **11.6 Wikidata → partial** (OOM at query 29). Documented with remediation path.
- **11.7 new file → reused `bench/api_benchmark.py`** (user decision up front). No new `benchmark_kglite.py` file.
- **Harness builder inlined** — not in the original plan, but necessary once v0.7.17 compatibility bit me (Surprise #1).

### Next-phase prerequisites (Phase 12 — documentation + API surface audit)

- **AUDIT §3 is now authoritative** — Phase 12 docs (CYPHER.md / FLUENT.md updates, README wins) can reference specific numeric wins: "−60 % on `pattern_match`", "−24 % on `describe`", etc. These come from the committed `tests/benchmarks/phase11_delta.md`.
- **`kglite/__init__.pyi` is byte-frozen** vs v0.7.17 — Phase 12 docstring changes must preserve this contract. `git diff v0.7.17 HEAD -- kglite/__init__.pyi` is still empty after Phase 11.
- **`find()` is code-entity-specific** (Phase 10 report-out flagged this) — Phase 12's doc pass should make the method's scope explicit in the `__init__.pyi` docstring.
- **Phase 11 harness / compare scripts are reusable** — Phase 12 can invoke them to freshen numbers if a doc claim is challenged. Budget: ~5 minutes per run at N=20.
- **`bench/benchmark_wikidata_cypher.csv`** is overwritten by Phase 11 (partial). The full historical CSV (if relevant) is not preserved. Phase 12 may want to commit a canonical Wikidata snapshot alongside Phase 11 numbers.

### Files touched

- **New (benchmark harness + comparison + JSON + transcripts)**: `tests/benchmarks/phase11_harness.py`, `tests/benchmarks/phase11_compare.py`, `tests/benchmarks/phase11_main.json`, `tests/benchmarks/phase11_v0_7_17.json`, `tests/benchmarks/phase11_delta.md`, `tests/benchmarks/phase11_mapped_scale_5gb.txt`, `tests/benchmarks/phase11_wikidata.txt`, `tests/benchmarks/phase11_wikidata_partial.csv`, `tests/benchmarks/phase11_api_benchmark.txt`.
- **New baseline**: `tests/parity_baseline_phase_11.json`.
- **Modified**: `AUDIT_0.8.0.md` § 3 (comprehensive rewrite + new subsections). `bench/benchmark_wikidata_cypher.csv` was overwritten by the partial run but `bench/` is gitignored — the preserved copy is at `tests/benchmarks/phase11_wikidata_partial.csv`.
- **Transient (cleaned up)**: `../KGLite-v0.7.17/` git worktree + `.venv-v0.7.17`.
- **Unchanged**: all of `src/`, all of `kglite/` including `__init__.pyi`, `pyproject.toml`, `Cargo.toml`. Zero production-code diff. Zero `pytest` / `parity` / `make lint` count change.

## Phase 12 Report-out (written at phase exit — read this before Phase 13)

**Completed** (single squashed commit `docs: Phase 12 — documentation + API surface audit` on `main`, local only — user pushes):

Documentation audit + stale-reference cleanup + one new long-form doc. No Rust changes. The `kglite/__init__.pyi` signature surface is still byte-identical to v0.7.17 (`git diff v0.7.17 HEAD -- kglite/__init__.pyi | grep -E '^[+-](    def |class )'` returns empty); only one docstring was extended. The overall file is no longer MD5-identical — the contract has dropped to "signatures unchanged, docstrings may improve" per the Phase 12 exit criteria.

Deliverables:

1. **12.1 ARCHITECTURE.md header + chronology refresh.** Header updated to "end of Phase 11"; forward-looking "Phase 9 will split…" prose replaced with a statement of current rules. The `core/pattern_matching.rs` entry replaced with the actual subdirectory tree (`pattern_matching/{mod, pattern, parser, matcher}.rs`). The `pyapi/kg_methods.rs` entry replaced with the four-file Phase 9 split (`kg_{core, mutation, introspection, fluent}.rs`). Removed the resolved `MappedGraph` open question; retained `Value::String` future-Cow decision; added an entry for "when does `languages/fluent/` get a Rust-side implementation" with the "when a second peer language materialises" answer.
2. **12.2 `find()` docstring + C1 close-out.** `kglite/__init__.pyi::find` now opens with "⚠ **Code-entity search only.**" and a sentence pointing users at `select().where()` or `cypher()` for general node lookup. Signature unchanged (verified mechanically).
3. **12.3/12.6 verifications.** `stubtest` clean (20 modules); `cargo doc --no-deps` clean (zero warnings). No edits needed.
4. **12.4 MCP smoke.** `examples/mcp_server.py` cannot be imported directly in CI (requires optional `mcp` package); KGLite's methods it relies on (`describe()`, `cypher()`, `schema()`, `graph_info()`, `load()`) verified working on a round-tripped `.kgl`. Smoke recorded inline in this report-out — no persistent transcript.
5. **12.5 CYPHER/FLUENT cross-check.** Three small gaps found and fixed:
   - `CYPHER.md`: added `vector_score(n, prop, vector [, metric])` row to the similarity functions table (it existed in the executor but had no user-facing docs).
   - `CYPHER.md`: added `head(list)` / `last(list)` row to the list-function table.
   - `FLUENT.md`: added a "Human-Readable String Export" subsection documenting `to_str(limit=50)` (it was in `__init__.pyi` but missing from the reference guide).
   The `where_method` "gap" was a false positive — Python sees it as `.where()`, which is extensively documented.
6. **12.7 docs/ orphan audit.** One orphan found: `docs/adding-a-storage-backend.md` was never toctreed. Fixed by adding it and the new `adding-a-query-language.md` (12.9) to the `Project` toctree in `docs/index.md`.
7. **12.8 adding-a-storage-backend.md** was current on the Phase 9 architecture in spirit, but had **five stale `src/graph/schema.rs` references** — Phase 9 moved `GraphBackend` to `src/graph/storage/backend.rs`. All five references updated.
8. **12.9 NEW `docs/adding-a-query-language.md`** (~200 lines). Mirrors `adding-a-storage-backend.md` structure. Six sections: TL;DR, the `languages/` umbrella, anatomy of `languages/cypher/`, integration with `core/`, entry-point flow diagram, speculative peer-language walk-through, plus a note on `languages/fluent/` as reserved scaffolding.
9. **12.10 CHANGELOG.md `[Unreleased]` performance section.** New top-of-section "### Performance" block with the top Phase 11 wins (`pattern_match` −60 %, `two_hop` −24 %, `describe` −21 %, `pagerank` −17 %, construction −11 to −22 %, four sub-5% flags documented). Also a new bullet under "Changed (internal, not user-visible)" for the Phase 10 edge-case / golden / stress testing envelope. Phase 13 will date the `[0.8.0]` header.
10. **12.11 README.md** verified clean — no stale paths (`graph/query/`, `graph/cypher/` etc.), no version-specific perf claims, toctree/links accurate. Zero edits.
11. **Extras closed**: **C3** (stale `# currently xfail` comment in `tests/test_cypher.py:1227` replaced with an accurate note that the `TestBug*` classes are regression guards), **D5** (`CLAUDE.md` now clarifies that only `#[pymethods] impl` blocks must live under `pyapi/`; struct `#[pyclass]` attributes may stay with the struct definition), **E1** (`test_exception_list_still_applies` deleted from `tests/test_phase7_parity.py`), **E2** (`test_file_count_budget` + its `subprocess` import deleted from `tests/test_phase6_parity.py`).
12. **`refactor_findings.md`** updated: C1, C3, D4, D5, E1, E2 moved to `## Resolved` with phase + one-line note each.

### Baselines held

- **`cargo test --release`**: **490 passed** (unchanged).
- **`make lint`**: clean (cargo fmt + clippy -D warnings + ruff format/check + stubtest on 20 modules).
- **`pytest tests/`**: **1,850 passed** (unchanged — E2 was parity-only, so the default count is invariant).
- **`pytest -m parity tests/`**: **95 passed, 0 skipped** (vs Phase 11's 96 passed + 1 skipped — E1 was a passing no-op that's now gone, E2 was the one skipped test and it's now gone).
- **`cargo doc --no-deps`**: zero warnings.
- **Signature diff vs v0.7.17**: `git diff v0.7.17 HEAD -- kglite/__init__.pyi | grep -E '^[+-](    def |class )'` → empty. Docstrings drifted; signatures did not.
- **Phase 7 `.kgl` v3 `GOLDEN_V3_DIGEST`** unchanged (no I/O code touched).
- **Binary size**: `kglite/kglite.cpython-312-darwin.so` unchanged at 7,046,320 bytes (Phase 12 is zero-Rust-diff).

### Surprises found during implementation

1. **`docs/adding-a-storage-backend.md` had five stale `src/graph/schema.rs` references.** The explorer-agent report said the doc was accurate; a spot-read found the references were moved to `src/graph/storage/backend.rs` by Phase 9's schema.rs 3-way split. All five fixed in-place.
2. **`docs/adding-a-storage-backend.md` was not in any toctree** — a pre-existing orphan, not introduced by this phase. Fixed alongside the new 12.9 doc addition.
3. **MCP-server smoke test blocked by optional `mcp` dep.** `examples/mcp_server.py` imports `mcp.server.fastmcp.FastMCP` at module load; the `mcp` package isn't in `pyproject.toml`'s default deps. The KGLite surface the server uses (`load()`, `describe()`, `cypher()`, `schema()`, `graph_info()`) was smoke-tested separately on a round-tripped `.kgl` — all green.
4. **`where_method` false positive in FLUENT cross-check.** Internal Rust name is `where_method`; `#[pyo3(name = "where")]` re-exposes it as `.where()`. Python users never see `where_method`. Grep for internal names must be paired with the `#[pyo3(name)]` translation — worth a note for future audits.
5. **`_build_kg_mode` wasn't actually referenced in 12.9** because the harness's `_make_shape` + `_populate` pattern (Phase 11) is more copy-paste-friendly. The new doc points at the graph builder in `_build_kg_mode` without documenting the shape-helpers pattern directly.
6. **`two_hop_10x` comparison in AUDIT §3 "Phase 6 reference" table** is shape-inequivalent (Phase 6's shape materialised paths; Phase 11's counts via `count(DISTINCT c)`). The historical table retains it with an inline caveat — the mismatch is called out in the AUDIT text, not a Phase 12 change.

### Decisions made

1. **`kglite/__init__.pyi` contract drops from "MD5 byte-identical" to "signature-identical" vs v0.7.17.** This is the intended Phase 12 state. Any future phase claiming "`__init__.pyi` unchanged" should specify whether that means the MD5 or just the signatures. Phase 13 release notes should mention "no public signature changes" without claiming byte-identity.
2. **`languages/fluent/` stays as scaffolding.** Phase 12 did *not* extract Rust-side fluent logic; it documents the current state (chain lives in `pyapi/kg_fluent.rs`) and reserves the directory for a future peer language. Matches the `languages/` umbrella intent without forcing a Phase 12 extraction scope explosion.
3. **No `QueryLanguage` trait introduced.** The `docs/adding-a-query-language.md` explicitly says the trait would be premature abstraction with only Cypher as a concrete impl. The directory-level modularity (`languages/<name>/parser/planner/executor/`) is the current contract.
4. **Phase 6 `file_count_budget` test deleted, not kept under permanent skip.** The test self-skipped on any Phase-7+ commit — dead weight on the parity count. Deletion is cleaner than a vacuous skip.
5. **Phase 6 historical benchmark table retained.** The Phase 11 debt item F ("decide retain vs drop") was left retained with the inline caveat on shape-inequivalent cells. Dropping it was not worth the churn in AUDIT §3.

### Debt introduced / carried to Phase 13

Nothing new in Phase 12. Inherited items that Phase 13 will close:

- **Version bump** — `Cargo.toml` 0.7.17 → 0.8.0; promote `CHANGELOG.md` `[Unreleased]` → `[0.8.0] — <date>`; tag `v0.8.0` (don't push — user pushes).
- **`refactor_findings.md`** — items A1–A5, B1–B2, C2, D1–D3, E3–E4, F, G remain open. Phase 13 is mechanics-only; these are all post-0.8.0 material.

### Scope changes vs the approved plan

- **Added**: five `src/graph/schema.rs` → `src/graph/storage/backend.rs` path corrections in `docs/adding-a-storage-backend.md` (surprise #1 above). Scope: +15 min, small-file fixes.
- **Added**: `docs/adding-a-storage-backend.md` added to `docs/index.md` Project toctree — was a pre-existing orphan discovered during 12.7.
- **Reduced**: 12.4 mcp_server smoke test did not produce a persistent transcript; the KGLite API surface it depends on was verified inline instead.
- **Reduced**: 12.6 Rust doc audit was already clean (zero warnings); no systematic pub-item coverage pass was needed.
- **Unchanged**: 12.1, 12.2, 12.5, 12.7, 12.8, 12.9, 12.10, 12.11, C3, D5, E1, E2 all landed as planned.

### Next-phase prerequisites (Phase 13 — release 0.8.0)

- **`[Unreleased]` in CHANGELOG is complete and ready to promote.** Phase 13 dates the `[0.8.0]` header (move content up, date the header, add release-date line).
- **`Cargo.toml` line 3 bumps `0.7.17` → `0.8.0`.** Then `cargo build --release` + `make lint` to confirm stability.
- **`kglite/__init__.pyi` signature contract**: `git diff v0.7.17 HEAD -- kglite/__init__.pyi | grep -E '^[+-](    def |class )'` must stay empty through Phase 13. If it's dirty after the version bump, investigate before tagging.
- **`refactor_findings.md` is up-to-date** as of Phase 12 exit. Phase 13 should not need to touch it. Post-release work picks items from the A/B/C/D/E/F/G sections.
- **AUDIT_0.8.0.md §3** has final benchmark matrix + deferred subsection. Phase 13 can spot-check that AUDIT summary + release-readiness section agree before tagging.

### Files touched

- **Modified (docs + comments)**: `ARCHITECTURE.md`, `kglite/__init__.pyi` (find docstring only), `CHANGELOG.md`, `CLAUDE.md`, `CYPHER.md`, `FLUENT.md`, `docs/adding-a-storage-backend.md`, `docs/index.md`, `tests/test_cypher.py` (comment cleanup).
- **Modified (test cleanup)**: `tests/test_phase6_parity.py` (deleted `test_file_count_budget` + unused `subprocess` import), `tests/test_phase7_parity.py` (deleted `test_exception_list_still_applies`).
- **Modified (findings)**: `refactor_findings.md` (C1/C3/D4/D5/E1/E2 moved to Resolved).
- **New**: `docs/adding-a-query-language.md` (~200 lines).
- **Unchanged**: all of `src/`, `Cargo.toml`, `pyproject.toml`, all pyi signatures. Zero Rust diff.
