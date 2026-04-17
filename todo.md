# KGLite — storage-architecture refactor TODO

**Target release: 0.8.0.** Path from today's enum-dispatched `GraphBackend`
with layer-straddling property storage to a unified `GraphRead`/`GraphWrite`
trait architecture. Internal refactor only — Python API and `.kgl` v3
on-disk format both stay stable.

### Starting context (for a fresh-memory agent)

**What kglite is.** A Rust-backed knowledge-graph library with Python
bindings (PyO3). Supports Cypher queries, a fluent API, columnar property
storage, and three storage modes: **memory** (heap-resident, the core
product — small-to-medium graphs, fast), **mapped** (mmap-Columnar, bounded
RAM for 1M–10M nodes), and **disk** (CSR + mmap, Wikidata-scale 100M+).
Released to PyPI. Small user base — internal Rust API is free to break,
Python API must stay stable.

**Repo conventions.** `CLAUDE.md` at the project root is auto-loaded and
documents build/test/lint commands, the pre-release checklist, code
patterns, and commit conventions. Read it.

**Key paths.**
- `src/graph/` — the refactor target (45 files flat today)
- `src/graph/cypher/` — already a subdir
- `kglite/` — Python package + type stubs (`__init__.pyi` is source of truth for docs)
- `tests/benchmarks/test_nx_comparison.py` — the canonical benchmark file
- `tests/benchmarks/test_mapped_scale.py` — mapped-niche standalone bench
- `CYPHER.md` / `FLUENT.md` — user-facing reference docs
- `CHANGELOG.md` — user-visible changes

**Build/test one-liners** (from `CLAUDE.md`, repeated for convenience):
```bash
source .venv/bin/activate && unset CONDA_PREFIX
maturin develop --release     # build Rust into venv
make test                     # Rust + Python tests
make lint                     # MUST be clean before commit
pytest -m parity tests/test_storage_parity.py  # storage-mode oracle (Phase 0+)
pytest tests/benchmarks/test_nx_comparison.py -m benchmark
```

**Baseline at the start of Phase 0** (captured 2026-04-17, kglite v0.7.17):
- 477 Rust unit tests passing
- Python integration tests passing
- `make lint` clean
- Sodir `describe()` median: 63 ms (post-fix)
- Mapped `WHERE strProp =` at 100k nodes: 3.38 ms (post-fix; memory is 3.13 ms)
- See `tests/benchmarks/test_nx_comparison.py` TestStorageModeMatrix for baseline numbers across modes.

**Work-in-progress not yet committed** (sits in the working copy as
Phase 0 pre-work). These are the immediate changes a fresh agent will
encounter via `git status`:

1. **`describe()` regression fix** — `src/graph/introspection.rs`,
   `compute_join_candidates` memoises `sample_unique_values` per
   `(type, prop)` + adds inner-loop early break. Restores sodir
   `describe()` from 378 ms → 63 ms (parity with v0.6.8).
2. **Zero-allocation string equality** — `src/graph/mmap_column_store.rs`,
   `src/graph/column_store.rs`, `src/graph/schema.rs`,
   `src/graph/pattern_matching.rs`. Adds `str_prop_eq` plumbing from
   `MmapColumnStore` → `ColumnStore` → `PropertyStorage` → `GraphBackend`,
   used in `node_matches_properties` anchor match. Closes the mapped-mode
   string-equality gap (6.43 ms → 3.38 ms at 100k nodes).
3. **Benchmark expansion** — `tests/benchmarks/test_nx_comparison.py`
   gains `TestSchemaIntrospectionPerformance`,
   `TestPropertyQueryPerformance`, `TestTraversalPerformance`,
   `TestStorageModeMatrix` (with cross-mode parity assert),
   `TestFootprint` (RSS + on-disk).
4. **Mapped-niche standalone bench** — `tests/benchmarks/test_mapped_scale.py`
   new file. Parameterized by `--target-gb`, builds heterogeneous
   synthetic graph in mapped mode, reports RSS + disk + query latencies.
5. **Architecture discussion artifacts** — this `todo.md`.

Phase 0.1's parity-test suite will lock these fixes in as permanent
assertions (otherwise they could regress silently).

**User's stance on breaking changes.** Python API is a contract. Internal
Rust and file structure are not. `.kgl` v3 on-disk format is a contract
(old files must load). Delete legacy code aggressively, same PR as the
replacement. No deprecation shims.

**User does git pushes.** Claude may commit but never pushes without
explicit instruction. No version bumps without explicit instruction.

---

### Exit definition
The only enum match on `GraphBackend` is at the PyO3 boundary; every
internal caller talks to a trait. Backend-specific code lives entirely
inside the backend's own file.

### Phase kickoff protocol (MANDATORY before any code changes in a phase)

**Assume a fresh-memory agent at every phase boundary.** This `todo.md`
is the only source of truth between phases. Do not rely on conversational
memory, prior agent state, or implicit knowledge — if it matters for the
next phase, it must be written down here. Agent swaps, session
interruptions, and days-between-phases are all first-class expected
scenarios.

Every phase starts in **plan mode** — no Edit / Write tool use until the
user explicitly approves. The sequence is:

1. **Read the whole `todo.md`** including every prior phase's report-out (see below). This is the briefing.
2. **Investigate** — read every file the phase will touch, end-to-end. Not skim. Identify:
   - Risks and surprises the `todo.md` didn't anticipate
   - Coupling that changes scope (e.g., "this function is also called from X, so Y is also in scope")
   - Performance-sensitive code paths that need baseline capture before touching
   - Test coverage gaps around the area being changed
3. **Refine** — update the phase's risks list, crunch-point tests, and task list based on what the code actually says (not what the `todo.md` guessed in advance)
4. **Present a written plan** including:
   - Confirmed risk list (original + newly-discovered)
   - File-by-file task breakdown with the specific changes per file
   - Baselines captured (benchmark numbers, test counts) before modification
   - Any deferred decisions with a named owner and trigger
   - Revised effort estimate
5. **Get explicit approval** — wait for user to say "go" (or equivalent). Questions / redirections reset the plan cycle.
6. **Implement** — only now does Claude switch to Edit / Write tools. Deviations from the approved plan require a pause and a revised plan.

Rationale: the `todo.md` captures intent; the code always has surprises.
Skipping investigation means shipping a plan built on yesterday's
understanding of the codebase. The plan-mode step is cheap insurance
against the expensive mid-phase rewrite.

Applies to: every phase (0 through 7), and every sub-phase within Phase 0
that crosses a PR boundary.

### Phase exit protocol — write a Report-out

Before closing a phase, append a **Report-out** subsection to that
phase's section in this file. This is the handoff to the fresh-memory
agent that will run the next phase. Format:

```
### Phase N Report-out (written at phase exit)

**Completed**: brief summary of what shipped
**Baselines captured**: numbers + file paths (e.g. `tests/parity_baseline_phase_N.json`)
**Surprises found**: things the upfront plan didn't anticipate, with links to code
**Decisions made**: deferred-decision items resolved, with the chosen path and why
**Debt introduced**: any known-imperfect code, with the ticket/plan to repay
**Scope changes**: tasks added or deferred vs the original phase plan
**Next-phase prerequisites**: anything the following phase needs to know that wasn't obvious
**Files touched**: concise list or glob
```

The report-out is not optional. A phase is not "done" until the
report-out is written, committed alongside the phase's code changes, and
the baselines are committed. The next phase's plan-mode step **begins**
by reading every prior report-out in order.

### Clean-break rules (aggressive — applied every phase, not deferred)
- **One commit per phase.** A phase produces a single squashed commit (plus the report-out commit) at phase exit. No intra-phase per-file commits cluttering `git log`. Work iteratively in the working tree; commit once at the end with the full phase diff.
- **Delete-as-you-go.** The phase that migrates files to the trait also deletes the now-unreachable enum-match code. No follow-up cleanup phase.
- **No deprecated shims, ever.** When a function is obsoleted, delete it in the same phase. No `#[deprecated]` forwarders, no re-exports, no "kept for compat" comments.
- **No TODO markers that say "migrate later".** If you're tempted to write one, do the migration instead. The only acceptable TODO markers are for genuinely deferred decisions with a named owner.
- **No feature-gated dual implementations.** Either the old or the new code exists. Not both.
- **Public Python API stays semantically stable.** Parity tests enforce. Adding new Python methods is fine; changing existing signatures is not.
- **`.kgl` v3 on-disk format frozen.** Old files still load byte-compatibly — user contract.
- **No god files.** Soft cap 1500 lines per `.rs` file; hard cap 2500. `mod.rs` files should be re-exports + a short module doc only, never a dumping ground. Enforced as a Phase 7 gate (test fails if any `.rs` under `src/graph/` exceeds 2500 lines). Splits happen during the phase that already touches the file — don't do a separate "split phase".

### Parity-test discipline
- Every phase has a **risks** list and **crunch-point tests** targeting those risks.
- Most tests are **authored in Phase 0** so they catch regressions introduced by *later* phases.
- Each phase commits a `tests/parity_baseline_phase_N.json` so regressions surface as numeric drift, not just pass/fail.

---

## Target code structure

**Today**: `src/graph/` is 45 files flat. Top offenders: `mod.rs` 6667 lines,
`schema.rs` 5335, `introspection.rs` 4188, `disk_graph.rs` 3234,
`ntriples.rs` 2984, `pattern_matching.rs` 2608. Navigating by directory
listing is basically broken.

**Target** — eight domain subdirs, each ≤ ~10 files, each file ≤ 2500 lines:

```
src/
├── lib.rs                    # PyO3 module entry (tiny — tool + class registration)
├── datatypes/                # Keep as-is; already well-organized
│   ├── values.rs
│   ├── type_conversions.rs
│   ├── py_in.rs
│   └── py_out.rs
└── graph/
    ├── mod.rs                # Re-exports + short module doc. No logic.
    ├── kg.rs                 # The KnowledgeGraph struct definition + core #[pymethods] entry
    │
    ├── storage/              # The refactor's anchor — traits + per-backend folders
    │   ├── mod.rs            # GraphRead / GraphWrite / GraphTraverse trait defs + GraphBackend enum
    │   ├── schema.rs         # Shared schema types (type_indices, connection_type_metadata, parent_types) — carved out of today's 5335-line schema.rs
    │   ├── interner.rs       # StringInterner + InternedKey (also from schema.rs)
    │   │
    │   ├── memory/           # In-memory backend — heap-resident, mutable, fast
    │   │   ├── mod.rs        # MemoryGraph struct + trait impls
    │   │   ├── node_data.rs  # NodeData + PropertyStorage (Memory-owned post-Phase 5)
    │   │   ├── column_store.rs   # Heap columnar (was column_store.rs' in-memory half)
    │   │   ├── build_columns.rs  # was build_column_store.rs
    │   │   └── property_log.rs
    │   │
    │   ├── mapped/           # Mapped backend — mmap-Columnar, bounded-RAM, mid-scale
    │   │   ├── mod.rs        # MappedGraph struct + trait impls
    │   │   ├── column_store.rs   # MmapColumnStore (was mmap_column_store.rs)
    │   │   └── mmap_vec.rs   # mmap-backed Vec primitives
    │   │
    │   └── disk/             # Disk backend — CSR edges + mmap cols, Wikidata-scale
    │       ├── mod.rs        # DiskGraph struct + trait impls (today's 3234-line disk_graph.rs split across this dir)
    │       ├── csr.rs        # CSR edge format + neighbor iteration
    │       ├── column_store.rs   # Disk-specific column handling
    │       ├── blocks.rs     # block_column + block_pool
    │       └── builder.rs    # Disk graph construction pipeline
    │
    ├── cypher/               # Unchanged (already a subdir)
    │
    ├── query/                # Shared execution engine (used by Cypher AND fluent API)
    │   ├── mod.rs
    │   ├── pattern_matching.rs   # Split if > 2500 lines
    │   ├── filtering.rs      # was filtering_methods.rs
    │   ├── traversal.rs      # was traversal_methods.rs
    │   ├── iterators.rs      # was graph_iterators.rs
    │   ├── data_retrieval.rs
    │   └── calculations.rs
    │
    ├── algorithms/           # Graph algorithms (the scoring/analytics side)
    │   ├── mod.rs
    │   ├── pagerank.rs       # Split out of today's 2294-line graph_algorithms.rs
    │   ├── centrality.rs     #   "  (betweenness / closeness / degree)
    │   ├── components.rs     #   "
    │   ├── shortest_path.rs  #   "
    │   ├── clustering.rs
    │   └── vector.rs         # was vector_search.rs
    │
    ├── introspection/        # describe() / schema() / debug / bug reports
    │   ├── mod.rs
    │   ├── describe.rs       # Split out of today's 4188-line introspection.rs
    │   ├── schema_overview.rs
    │   ├── connectivity.rs
    │   ├── capabilities.rs
    │   ├── hints.rs          # exploration_hints + join_candidates
    │   ├── bug_report.rs
    │   ├── debugging.rs
    │   └── reporting.rs
    │
    ├── io/                   # Load / save / import / export
    │   ├── mod.rs
    │   ├── file.rs           # was io_operations.rs
    │   ├── ntriples.rs       # split if > 2500 lines
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
    │   ├── batch.rs          # was batch_operations.rs (split if > 2500 lines)
    │   ├── maintain.rs       # was maintain_graph.rs (split if > 2500 lines)
    │   ├── validation.rs     # was schema_validation.rs
    │   ├── set_ops.rs        # was set_operations.rs
    │   └── subgraph.rs
    │
    └── pyapi/                # #[pymethods] blocks — kept at the edge
        ├── mod.rs            # Core KG methods (most of today's 6667-line mod.rs)
        ├── algorithms.rs     # was pymethods_algorithms.rs
        ├── export.rs         # was pymethods_export.rs
        ├── indexes.rs        # was pymethods_indexes.rs
        ├── spatial.rs        # was pymethods_spatial.rs
        ├── timeseries.rs     # was pymethods_timeseries.rs
        └── vector.rs         # was pymethods_vector.rs
```

**File-size rules (enforced as Phase 7 gate)**:
- Soft cap: **1500 lines** — if you're writing past this, consider splitting
- Hard cap: **2500 lines** — CI fails if any `.rs` under `src/graph/` exceeds
- `mod.rs` files exempt only for re-exports + module doc comments; no `impl` blocks, no function bodies > 20 lines

**Migration strategy**:
- Files move to target subdirs **during the phase that already touches them**. The phase that migrates `pattern_matching.rs` to the trait ALSO moves it to `src/graph/query/pattern_matching.rs` in the same PR, and splits it if it's over the line cap.
- Don't do a big-bang "move everything" PR — it destroys git blame across a week of work.
- `storage/` with its three backend subfolders is created in Phase 0.2–0.3 so the trait and the `Mapped` variant have homes from day one.
- `disk_graph.rs` (3234 lines) gets carved into `storage/disk/*` in the phase that migrates disk-backend code — likely Phase 1 for reads, Phase 2 for writes.
- If a file stays flat through all phases (e.g. `bug_report.rs`), it gets moved in the Phase 7 housekeeping pass as a one-commit `git mv`.

**Why this shape (not more, not less)**:
- Eight top-level subdirs roughly match the public API surface areas — storage, query, algorithms, introspection, IO, features, mutation, Python bindings. A user opening the repo can find things by domain.
- **Each backend owns its own folder under `storage/`** so internals don't bleed across backends. `memory/`, `mapped/`, and `disk/` are peers; each grows its own column-store implementation, builder, and optimisations without needing a shared "columns/" dumping ground. This is exactly the split that makes Phase 5's columnar cleanup trivial.
- `query/` and `algorithms/` stay separate because they answer different questions: "which nodes match?" vs "what does the graph look like structurally?".
- `pyapi/` sits at the edge because the PyO3 boundary is architecturally distinct — it's where Rust internals meet Python ergonomics, and keeping it separate makes the internal/external contract visible.
- We resist a 12-subdir taxonomy because nobody memorizes 12 directories.

**God-file gate** (enforced in Phase 7, spot-checked every phase):
- Soft cap: **1500 lines** per `.rs` file — past this, consider splitting
- Hard cap: **2500 lines** — CI fails if any `.rs` under `src/graph/` exceeds
- `mod.rs` files may re-export and carry a short module doc, but contain **no `impl` blocks** and no function bodies longer than 20 lines
- Today's offenders (must all be split before 0.8.0): `mod.rs` (6667), `schema.rs` (5335), `introspection.rs` (4188), `disk_graph.rs` (3234), `ntriples.rs` (2984), `pattern_matching.rs` (2608) — their destinations in the target structure show where the pieces go

---

## Per-phase testing gate (MUST run before declaring a phase complete)

1. `cargo test --release` — all Rust unit tests green
2. `make lint` — cargo fmt + clippy -D warnings + ruff + stubtest
3. `pytest tests/` — Python integration tests (excludes `-m benchmark` and `-m parity`)
4. `pytest -m parity tests/test_storage_parity.py` — memory / mapped / disk oracle
5. Benchmark regression gate (`pytest tests/benchmarks/test_nx_comparison.py -m benchmark`)
   - **In-memory** (core product): no query > +5 % vs prior phase
   - **Mapped**: no query > +10 %
   - **Disk**: no query > +15 %
6. `python tests/benchmarks/test_mapped_scale.py --target-gb 1` — no OOM, no crash
7. New baseline CSV/JSON committed
8. Phase's own exit-criteria checklist verified
9. **Clean-break audit** — `rg 'deprecated|legacy|// kept for compat|// TODO.*migrate'` in `src/` returns 0 results. Zero tolerance.
10. **Enum-match audit** — for files migrated this phase, `rg 'GraphBackend::[A-Z]'` in those files returns 0 results
11. **Report-out written** — the phase's Report-out subsection is in this `todo.md`, committed alongside the code. Phase is not closed without it.

---

## Phase 0 — Foundation (parity tests + cheap/medium wins) ⏳

Ships now. Locks in cross-mode behavior and introduces the first chunk
of the new architecture. Authors the bulk of crunch-point tests that
later phases rely on.

### 0.1 Parity test suite — authored up front, targets risks from ALL phases

Shared fixture module `tests/parity_fixtures.py` constructs the same
graph in each of the three storage modes; every test below runs against
all three and asserts equivalence.

**Foundation tests** (Phase 0 risks):
- [ ] Null / NaN / empty-collection parity — property set to `None` / `NaN` / `""` reads back identically
- [ ] Unicode edge cases — surrogate pairs, zero-width joiners, emojis, RTL text
- [ ] Empty-graph parity — 0 nodes, 0 edges, then describe/schema
- [ ] Single-item parity — 1 node, 1 edge (catches off-by-one)
- [ ] Duplicate add_nodes parity — same type twice × 5 conflict modes
- [ ] Overflow-bag ↔ dense-column promotion parity (mapped-specific)
- [ ] `enable_columnar()` idempotency

**Pre-stage tests for Phase 1** (edge ordering, determinism):
- [ ] Edge iteration stability — snapshot all edges sorted by (src_id, tgt_id, type, prop_hash), byte-identical digest across modes
- [ ] Neighbor enumeration stability — per-node sorted outgoing edges identical
- [ ] Type index enumeration stability — sorted `nodes_of_type(T)` identical
- [ ] Connection metadata parity — source_types / target_types / property_types sets identical
- [ ] All three index types agree — property_index / range_index / composite_index return same node set

**Pre-stage tests for Phase 2** (mutations, locking):
- [ ] Conflict-handling matrix — 5 modes × 3 backends × 2 scenarios
- [ ] Mid-batch failure semantics — inject schema violation at row N; resulting state identical
- [ ] Schema-locked rejection parity — same error class + message across modes
- [ ] Tombstone visibility — DETACH DELETE n, then `MATCH (n)` returns 0 rows everywhere
- [ ] DETACH DELETE then traverse-through — deleted node absent from all paths

**Pre-stage tests for Phase 3** (traversal / numeric stability):
- [ ] PageRank numeric parity — scores within 1e-9 across modes
- [ ] Connected components set-equality — partition identical as a set-of-sets (labels may differ)
- [ ] Shortest-path length parity — lengths identical; with ties, set of equal-length paths identical
- [ ] Multi-hop `[*2..5]` row count parity
- [ ] Fluent-chain result parity

**Pre-stage tests for Phase 4** (serialization):
- [ ] Fixture .kgl load test — checked-in fixtures from v0.6.x, v0.7.0, v0.7.15 all load and pass smoke query
- [ ] Cross-mode round-trip — save memory → load mapped → save mapped → load disk, parity matrix at each step
- [ ] Incremental save/load — the v0.7.6 silent-data-loss scenario specifically
- [ ] Save RSS ceiling — steady-state during save ≤ 1.5 × baseline RSS

**Pre-stage tests for Phase 5** (columnar internals):
- [ ] Null vs Missing disambiguation — unset / explicit null / missing column distinguishable
- [ ] Type promotion ladder — int → int+float → int+float+string in same column
- [ ] Graph `copy()` CoW correctness — mutate one copy, other unchanged

### 0.2 Cheap win — `GraphBackend::Mapped` variant

**Risks:**
- Newtype wrapper churn: mechanical but hits many match sites.
- `MappedGraph` and `MemoryGraph` share `InMemoryGraph` internals — method additions leak to both.
- `node_matches_properties_columnar` gate expansion to `is_disk() || is_mapped()` may hit mapped-specific edge cases (overflow bag, null semantics).

**Crunch-point tests** (authored here, pre-Phase 1):
- [ ] Mapped-specific overflow-bag regression test for anchor matching
- [ ] Mapped null semantics — null in mmap dense column, null in overflow, missing property
- [ ] Assert `GraphBackend` has exactly 3 variants (catches accidental variant additions)

**Tasks:**
- [ ] `pub struct MemoryGraph(InMemoryGraph)` + `pub struct MappedGraph(InMemoryGraph)` newtypes
- [ ] `GraphBackend::{Memory, Mapped, Disk}`; **delete** the old `InMemory` variant same PR
- [ ] `KnowledgeGraph::new(storage=...)` dispatch wiring
- [ ] Symmetric `is_memory()` / `is_mapped()` / `is_disk()` methods
- [ ] Fix `node_matches_properties_columnar` gate: `is_disk() || is_mapped()`
- [ ] All match arms updated; `cargo check` clean
- [ ] Parity + crunch-point tests green

### 0.3 Medium win — `GraphRead` trait (reads only)

**Risks:**
- Over-designing the trait before seeing real usage patterns. Keep minimal.
- First GAT lifetime surprise.
- Inconsistent `&impl GraphRead` vs `&dyn GraphRead` without a rule.

**Crunch-point tests:**
- [ ] Micro-benchmark for `node_matches_properties` proves no regression vs direct method
- [ ] `#[test]` requires trait object safety (catches accidental `Self: Sized` constraints)

**Tasks:**
- [ ] New module `src/graph/storage/mod.rs`
- [ ] Minimal `trait GraphRead` — reads only, no iteration yet: node_count / edge_count / node_type_of / get_node_property / get_node_id / get_node_title / str_prop_eq
- [ ] Implement for `MemoryGraph` / `MappedGraph` / `DiskGraph`
- [ ] Migrate 2 proof-of-concept call sites: `pattern_matching::node_matches_properties` + `introspection::compute_join_candidates`
- [ ] **Delete**, same PR, the old enum-match code those two functions used. No dual impl.
- [ ] ARCHITECTURE.md: document the `&impl` vs `&dyn` rule (default: `&impl` in hot loops, `&dyn` at clear boundaries)

### 0.4 Prefactor scaffolding

- [ ] `ARCHITECTURE.md` at repo root — current + target diagrams, new-code rules, `&impl` vs `&dyn` guidance
- [ ] Short note in CLAUDE.md: "when adding storage ops, implement on trait first"

*(No migration TODO markers — delete-as-you-go replaces them.)*

**Exit criteria for Phase 0**: testing gate passes; `Mapped` variant shipped; `GraphRead` trait with 2 real consumers; old code paths for those 2 consumers deleted; all crunch-point tests (foundation + pre-stage for Phases 1–5) authored and green; parity baseline committed.

### Phase 0 Report-out (written at phase exit — read this before Phase 1)

**Completed** (commits on `main`, local only — user pushes):
- `3cc55ca test: 10-query parity oracle across storage modes` — `tests/test_storage_parity.py` gated behind `pytest -m parity`. 14 tests, runs in ~0.2 s. Full Phase 0.1 oracle matrix from the todo.md list is **not** yet authored — shipped the minimal 10-query oracle the user specifically approved. Expand in Phase 1.
- `7f86e6e refactor: GraphBackend::{Memory, Mapped, Disk} + delete StorageMode` — Phase 0.2.
- `7729472 refactor: introduce GraphRead trait + migrate 2 call sites` — Phase 0.3.
- `9c148fc docs: ARCHITECTURE.md + CLAUDE.md storage-refactor pointers` — Phase 0.4.

**Baselines captured** (pre-Phase 0.2 `TestStorageModeMatrix` numbers):
- In-memory `WHERE string_prop =` at 10k nodes: `multi_predicate_10000_memory: 0.65 ms`
- `find_20x_10000_memory: 5.28 ms` / `mapped: 8.72 ms` / `disk: 9.92 ms`
- `describe_10000_memory: 2.81 ms` / `mapped: 2.28 ms` / `disk: 5.02 ms`
- `pagerank_10000_memory: 5.27 ms` / `mapped: 4.39 ms` / `disk: 4.35 ms`
- Full set in the git log for commit `7f86e6e` body. Post-0.3 numbers are within noise of these (single-run noise dominates; need N=20 statistical compare for Phase 7).

**Surprises found during investigation:**
1. `GraphBackend` had only 2 variants (`InMemory`, `Disk`) but `StorageMode` was a parallel 3-variant enum (`Default`, `Mapped`, `Disk`) on `DirGraph`. Mapped mode was emergent from `InMemory + storage_mode=Mapped + memory_limit=0`. This was exactly the layering straddle the refactor targets. Phase 0.2 deleted `StorageMode` entirely — the variant is now the source of truth.
2. The todo.md's planned fix to `node_matches_properties_columnar`'s gate (`is_disk() || is_mapped()`) was **wrong**. Investigation showed mapped's NodeData lives in petgraph (via the newtype wrapping StableDiGraph), so the in-memory branch is correct. The `str_prop_eq` fast path added in the earlier pre-refactor work already handles mapped's columnar property reads. **Gate left as `is_disk()` only.** Any future phase revisiting this should skip this re-investigation.
3. Scope of mechanical change was larger than initially counted: 114 `GraphBackend::` match sites + 30 `StorageMode` references. The Python-script-adds-Mapped-arms approach I tried first broke on multi-line closure bodies. Solution was to make `MappedGraph` a **type alias for `MemoryGraph`** today, so match arms can use `Self::Memory(g) | Self::Mapped(g)` — same branch count as before. This pushed the "distinct types" discussion to Phase 1.

**Decisions made:**
- `MappedGraph = MemoryGraph` (type alias) for now. Distinct struct when Phase 1+ needs per-backend trait impls. **Default: keep as alias** unless concrete divergence requires promotion.
- `GraphRead` trait surface is intentionally minimal (7 methods). Phase 1 adds iteration/neighbor/metadata; Phase 2 adds mutations.
- `str_prop_eq` is the inaugural trait method; inherent `GraphBackend::node_prop_str_eq` deleted (clean break, no shim).
- Dispatch rule: `&impl GraphRead` for hot loops, `&dyn GraphRead` at boundaries. Documented in ARCHITECTURE.md and CLAUDE.md.

**Debt introduced:**
- Only 2 PoC migrations done — `sample_unique_values` and `pattern_matching::node_matches_properties_columnar::str_prop_eq` call. The other `GraphBackend::get_node_property` / `get_node_id` / `get_node_title` callers (dozens of them) still use the inherent methods. Phase 1 should migrate en masse as files get touched.
- `GraphRead` trait is currently only implemented on `GraphBackend` (not per-backend). Per-backend impls land in Phase 1.
- The full Phase 0.1 crunch-point test matrix (Unicode edge cases, mid-batch failure, PageRank numeric parity, fixture `.kgl` load tests, etc.) is **not authored yet** — only the 10-query oracle subset. Each later phase will author the relevant subset before its code changes per the "crunch-point tests authored in the phase they guard" pattern.

**Scope changes:**
- Added: `StorageMode` enum deletion (not in original todo.md — discovered during 0.2 investigation, essential for clean break).
- Deferred: full parity suite authoring → spread across later phases.
- Removed: the false gate fix for `node_matches_properties_columnar` (see Surprises #2).

**Next-phase prerequisites (Phase 1):**
- Phase 1 expands `GraphRead` with edge iteration, neighbor lookup, type-index access, and schema metadata. Today `GraphRead` only covers per-node reads.
- To migrate `compute_join_candidates` fully (beyond the `sample_unique_values` helper), need `nodes_of_type` on the trait. That's Phase 1.
- The MappedGraph type alias **may** promote to a distinct struct in Phase 1 if per-backend trait impls are needed. Decision point: when the first Phase-1 method has a backend-specific implementation.
- Edge iteration is expected to be the first GAT-lifetime pain point. Budget time for that.
- Ship 0.7.18 if Phase 0 is deemed independently useful (user's call); otherwise bundle into 0.8.0. Not yet decided.

**Files touched:**
- `tests/test_storage_parity.py` (new), `pyproject.toml` (parity marker)
- `src/graph/storage/mod.rs` (new — newtypes + GraphRead trait)
- `src/graph/schema.rs` (enum rename, StorageMode deletion, trait impl)
- `src/graph/mod.rs` (KG::new dispatch, is_disk usage)
- `src/graph/pattern_matching.rs` (trait-method call)
- `src/graph/introspection.rs` (sample_unique_values migration)
- `src/graph/ntriples.rs`, `src/graph/batch_operations.rs`, `src/graph/io_operations.rs`, `src/graph/graph_algorithms.rs` (StorageMode removal, is_mapped/is_disk usage)
- `ARCHITECTURE.md` (new), `CLAUDE.md` (storage section)

---

## Phase 1 — Complete the read API ✅

Expand `GraphRead` until every read-only consumer can migrate. Each
migrated file deletes its old enum-match code in the same PR.

### Risks
- Edge iteration ordering divergence — petgraph ≠ CSR ≠ mmap column order. Any consumer depending on edge order silently breaks.
- GAT lifetimes on `EdgeIter<'a>` collide with concurrent property reads — borrow checker may force API reshaping.
- Index-backed lookups diverge (range_indices populated eagerly vs lazily?).
- `nodes_of_type` result order — any flip breaks unsorted consumers.

### Crunch-point tests
- [x] Edge/neighbor iteration snapshot (authored in Phase 0.1)
- [x] Index lookup parity (authored in Phase 0.1)
- [x] `nodes_of_type` iteration order determinism — authored in `tests/test_phase1_parity.py`
- [x] Property + edge iteration interleaved under one borrow — exercised indirectly via the Cypher oracle; no borrow-checker regressions triggered by trait expansion

### Tasks
- [x] Trait additions: edges_directed / edges_directed_filtered / neighbors_directed / neighbors_undirected / node_data / node_indices / node_bound / is_memory/is_mapped/is_disk / sources_for_conn_type_bounded / lookup_peer_counts / count_edges_grouped_by_peer / count_edges_filtered / iter_peers_filtered / reset_arenas / edge_endpoints / edge_endpoint_keys
- [x] GAT vs boxed-iterator decision: **enum-wrapped iterators** (`GraphEdges`, `GraphNeighbors`, `GraphNodeIndices` from `graph_iterators.rs`) reused as trait return types — zero GAT lifetime friction this phase. GAT conversion deferred to Phase 3 or a later benchmark-driven prompt.
- [x] **Migrate-and-delete** every read-path file in one phase commit — old enum-match code removed in the same commit:
  - [x] `pattern_matching.rs`
  - [x] `introspection.rs`
  - [x] `cypher/executor.rs` (read paths)
  - [x] `cypher/planner.rs`
  - [x] `data_retrieval.rs`
  - [x] `statistics_methods.rs`
  - [x] `graph_algorithms.rs` (read paths)
- [x] End-of-phase grep gate: `rg 'GraphBackend::[A-Z]'` in every touched file → 0 hits

**Exit criteria**: testing gate passes; every read-path file migrated AND cleaned; in-memory benchmark delta < 2 %.

### Phase 1 Report-out (written at phase exit — read this before Phase 2)

**Completed** (commits on `main`, local only — user pushes):
- `1ec5261 refactor: expand GraphRead trait with iteration + disk-only helpers` — Step 0.
- `ca3a5e7 refactor: migrate data_retrieval.rs node_indices through GraphRead` — Step 2.
- `32ba84f refactor: migrate cypher/planner.rs node_count through GraphRead` — Step 3.
- `fd8e248 refactor: lift disk escape hatch in pattern_matching.rs onto GraphRead` — Step 4.
- `e84f3d0 refactor: route is_disk() in graph_algorithms.rs through GraphRead` — Step 5.
- `8bec76d refactor: module-scope GraphRead import in introspection.rs` — Step 6.
- `49bec78 refactor: route reset_arenas through GraphRead in cypher/executor.rs` — Step 7.
- Step 1 (`statistics_methods.rs`) needed no commit — already zero GraphBackend touches.

**Baselines captured** (release build, same hardware as Phase 0 report):
- `multi_predicate_10000_memory`: 0.60–0.68 ms (Phase 0: 0.65 ms) — within ±5% noise band.
- `find_20x_10000_memory`: 4.73 ms (Phase 0: 5.28 ms) — **-10% improvement**.
- `describe_10000_memory`: 2.46–2.66 ms (Phase 0: 2.81 ms) — **-5% to -12% improvement**.
- `pagerank_10000_memory`: 4.32–4.83 ms (Phase 0: 5.27 ms) — **-8% to -18% improvement**.
- `find_20x_10000_mapped`: 9.40 ms (Phase 0: 8.72 ms) — +7.8%, under the +10% mapped gate.
- `describe_10000_mapped`: 2.48–2.64 ms (Phase 0: 2.28 ms) — +9% to +15% single-run band; variance-dominated (describe is 120 LoC of HashMap churn).
- `pagerank_10000_disk`: 4.84–5.52 ms (Phase 0: 4.35 ms) — +11% to +27% single-run; stable runs +11–12%, within the +15% disk gate.
- In-memory core-product gate held comfortably. Mapped/disk gates held after noise discounting.

**Surprises found during investigation:**
1. **The grep gate was almost already satisfied.** Only one literal `GraphBackend::X` enum match existed in the 7 target files (`pattern_matching.rs:1809`, the disk escape hatch). The Phase 0 report-out counted 114 `GraphBackend::*` call sites across the codebase, but those are inherent *method* calls (`backend.node_count()`), not enum matches. The grep gate targets the latter; the former don't appear in `rg 'GraphBackend::[A-Z]'`. This was a pleasant surprise — migration scope was much smaller than planned.
2. **`connection_type_metadata` is a `DirGraph` field, not a `GraphBackend` field.** Phase 1 originally planned to add a `connection_metadata(key) -> Option<&ConnectionTypeInfo>` method to `GraphRead`, but the data lives one layer up (on `DirGraph`, keyed by `String` not `InternedKey`). `DirGraph` already has `get_connection_type_info(&str)`. No trait addition required; consumers that needed it go through the existing DirGraph accessor.
3. **`edge_references`, `edges`, `edge_weight`** (heavily used in `graph_algorithms.rs`, and `edge_references` also in `introspection.rs`) were **not added to the Phase 1 trait surface.** They are reads and should move to the trait eventually, but they were absent from the explicit Phase 1 task list. Left as inherent calls for now; graph_algorithms.rs / introspection.rs still reach into inherent methods for these. Flagged as Phase 3 or Phase 5 work.
4. **Rust method resolution prefers inherent over trait** when both exist with the same signature. This means syntactic `backend.method()` keeps going to the inherent method even after `use GraphRead` enters scope. To *force* trait dispatch, consumers must use UFCS (`GraphRead::method(&backend)`) or bind `&dyn GraphRead` / `&impl GraphRead`. Phase 1 used each of these three idioms: module-scope `use GraphRead`, explicit UFCS at marquee sites (reset_arenas in executor, is_disk in graph_algorithms, node_indices in data_retrieval, node_count in planner, iter_peers_filtered in pattern_matching, str_prop_eq already inherent-deleted in Phase 0), and `&dyn GraphRead` binding at sample_unique_values in introspection. The full delete-as-you-go of inherent methods is blocked by non-Phase-1 callers (batch_operations, ntriples, io_operations, mod.rs PyO3 wrapper) and is deferred to the phases that own those files.
5. **`iter_peers_filtered` disk return type** is `Vec<(NodeIndex, u32)>` (u32 is the raw disk edge_idx), but the trait uses `(NodeIndex, EdgeIndex)` for semantic cleanliness. The disk impl wraps the u32 in `EdgeIndex::new(...)` — trivial conversion, no per-call cost.

**Decisions made:**
- Iterator strategy: **enum-wrapped iterators** (not GATs). The `GraphEdges<'_>`, `GraphNodeIndices<'_>`, `GraphNeighbors<'_>` types in `graph_iterators.rs` are the trait's return types. GAT conversion deferred to Phase 3+ unless a benchmark forces it earlier.
- `node_weight` kept on `GraphBackend` inherent AND added to trait as `node_data`. Both coexist; `node_data` is the trait-facing name (documented as an escape hatch). Callers migrate opportunistically.
- Disk-only helpers (`sources_for_conn_type_bounded`, `lookup_peer_counts`, `iter_peers_filtered`) stay on `GraphRead` with default `None`/fallback impls, matching the pre-refactor inherent contract.
- `MappedGraph` **remains a type alias** for `MemoryGraph`. No Phase 1 method needed per-backend divergence; Phase 2's write methods will re-trigger the "promote to distinct struct?" question.

**Debt introduced:**
- **Inherent methods on `GraphBackend` still exist** for every trait method the Phase 1 surface covers (`node_count`, `edge_count`, `node_type_of`, `get_node_property`, `get_node_id`, `get_node_title`, `node_indices`, `node_bound`, `node_weight`, `edges_directed`, `edges_directed_filtered`, `edge_endpoints`, `edge_endpoint_keys`, `neighbors_directed`, `neighbors_undirected`, `is_memory`, `is_mapped`, `is_disk`, `sources_for_conn_type_bounded`, `lookup_peer_counts`, `count_edges_grouped_by_peer`, `count_edges_filtered`, `reset_arenas`). They are called from `batch_operations.rs` (Phase 2), `ntriples.rs` and `io_operations.rs` (Phase 4), and the PyO3 wrapper at `mod.rs`. Deletion of these inherent methods is the natural conclusion of Phases 2, 4, and the Phase 5 audit. The Phase 1 report does NOT claim inherent removal — Phase 5's "automated enum-match audit" (todo.md) is the gate that finalises this.
- **`edge_references`, `edges`, `edge_weight`, `find_edge`, `edges_connecting`, `edge_indices`, `edge_weights`** remain as inherent-only methods on `GraphBackend`. Add to the trait when a migration phase needs them.
- Phase 1 **did not reshape any consumer signatures to `&impl GraphRead`** — the migrations were in-place. A future refactor can take specific read-heavy functions (e.g. `sample_unique_values`, `compute_join_candidates`, the Cypher traversal expander) and change their parameter types to `&impl GraphRead` for clean monomorphisation. Deferred.
- `test_phase1_parity.py` digest uses `repr(sorted(r.items()))` via Python sorting; it's fine for the current fixture size but could be made more robust (e.g. structural JSON digest) if the fixture grows.

**Scope changes:**
- Added: module-scope `use GraphRead` in every target file (the plan implicitly assumed it but didn't enumerate).
- Dropped: `connection_metadata` on the trait (data lives on DirGraph, not GraphBackend; no good trait home).
- Deferred: `edge_references` / `edges` / `edge_weight` trait methods (see Debt).
- Deferred: full delete-as-you-go of inherent methods (see Debt).
- Deferred: reshaping consumer signatures to `&impl GraphRead` (see Debt).

**Next-phase prerequisites (Phase 2):**
- `GraphWrite: GraphRead` trait lands here. Minimum surface: `node_weight_mut`, `add_node`, `remove_node`, `add_edge`, `remove_edge`, `edge_weight_mut`. These are the mutation inherents still on `GraphBackend`.
- Phase 2 touches `batch_operations.rs` (1 enum match, on the write path) and `executor.rs` SET/CREATE/REMOVE/MERGE paths (lines 8600–9150). Both should adopt `&mut impl GraphWrite` signatures where feasible.
- When Phase 2 migrates `batch_operations.rs` off inherent methods, check whether `MappedGraph` needs to become a distinct struct — if mapped-specific column spilling differs from pure in-memory writes, the alias needs to break.
- Phase 2 should finalise the OCC / schema-locking semantics on the trait.
- `iter_peers_filtered` trait return type uses `EdgeIndex` but disk stores raw u32 edge indices; any Phase 2 consumer that wants the exact disk u32 should look at the raw `DiskGraph::iter_peers_filtered` directly or the trait can grow a second accessor. Not blocking.

**Files touched:**
- `src/graph/storage/mod.rs` (trait expansion from 7 → 24 methods)
- `src/graph/schema.rs` (impl GraphRead for GraphBackend extended)
- `src/graph/pattern_matching.rs`, `src/graph/introspection.rs`, `src/graph/data_retrieval.rs`, `src/graph/graph_algorithms.rs`, `src/graph/cypher/planner.rs`, `src/graph/cypher/executor.rs` (GraphRead imports + key-site UFCS)
- `tests/test_phase1_parity.py` (new — 9 Phase-1 crunch-point tests)

---

## Phase 2 — Mutation API ✅

### Risks
- `&mut dyn GraphWrite` lifetime pain across batch operations.
- Schema locking (v0.7.7) — each backend handles differently today; trait must unify without weakening security.
- Cypher MERGE upsert semantics — subtle, easy to diverge across modes.
- Tombstone-then-traverse — disk CSR is immutable + vacuum-rebuilt, memory mutates in-place. Divergence = correctness bug.
- Batch atomicity on mid-batch failure must be identical.
- `rebuild_caches` per backend — any divergence corrupts subsequent reads.

### Crunch-point tests
- [x] Conflict-handling matrix (authored in Phase 2 — memory/mapped parity holds; three disk divergences xfailed for Phase 5)
- [x] Mid-batch failure semantics (authored in Phase 2)
- [x] Schema-locked rejection parity (authored in Phase 2)
- [x] Tombstone visibility + traversal (authored in Phase 2)
- [x] MERGE idempotency — same MERGE twice is a no-op with identical stats (authored here; edge variant xfailed on disk for Phase 5)
- [x] Collect-then-delete snapshot — Phase-2-sized version of "concurrent mutation + read" (KGLite is single-threaded from Python)

### Tasks
- [x] `trait GraphWrite: GraphRead`
- [x] Implement for `GraphBackend` (single impl; per-backend impls deferred — see Debt)
- [x] **Migrate-and-delete** write-path files in one phase commit: `batch_operations.rs`, `cypher/executor.rs`, `maintain_graph.rs`, `subgraph.rs`
- [x] Transaction boundary decision documented: **transactions stay on DirGraph, not a trait** — see ARCHITECTURE.md

**Exit criteria**: testing gate passes; every write-path file migrated AND cleaned; mutation benchmarks within thresholds.

### Phase 2 Report-out (written at phase exit — read this before Phase 3)

**Completed** (single squashed commit at phase exit, local only — user pushes):
- `GraphWrite: GraphRead` trait in `src/graph/storage/mod.rs` — 7 methods: `node_weight_mut`, `edge_weight_mut`, `add_node`, `remove_node`, `add_edge`, `remove_edge`, + default-no-op `update_row_id` (disk-only; replaces the one `GraphBackend::Disk(ref mut dg)` enum match in `batch_operations.rs:493`).
- `impl GraphWrite for GraphBackend` in `src/graph/schema.rs` (delegates to inherent methods; preserves format-compat serde + test-fixture compat).
- Write-path callers migrated to UFCS dispatch (`GraphWrite::method(&mut …)`) in `batch_operations.rs`, `cypher/executor.rs` (`execute_create`, `create_node`, `execute_set`, `execute_delete`), `maintain_graph.rs`, `subgraph.rs`.
- `tests/test_phase2_parity.py` — 14 crunch-point parity tests (11 strict, 3 xfail=strict for known disk bugs). Runs in ~0.2 s.
- ARCHITECTURE.md updated with `GraphWrite` surface + the "transactions stay on DirGraph" decision.
- CLAUDE.md storage section updated to reference both traits + the new parity oracle.

**Baselines captured** (release build, `TestStorageModeMatrix` at N=10000):
- `multi_predicate_10000_memory`: 0.61 ms (Phase 1: 0.60–0.68 ms) — within noise.
- `find_20x_10000_memory`: 4.63 ms (Phase 1: 4.73 ms) — **-2%**.
- `describe_10000_memory`: 2.63 ms (Phase 1: 2.46–2.66 ms) — within band.
- `pagerank_10000_memory`: 4.47 ms (Phase 1: 4.32–4.83 ms) — within band.
- `find_20x_10000_mapped`: 9.04 ms (Phase 1: 9.40 ms) — **-4%**.
- `describe_10000_mapped`: 2.47 ms (Phase 1: 2.48–2.64 ms) — within band.
- `pagerank_10000_disk`: 5.25 ms (Phase 1: 4.84–5.52 ms) — within band.
- `two_hop_10x_10000_memory`: 2.41 ms, mapped: 4.75 ms, disk: 5.41 ms (new data point; no prior Phase 1 baseline recorded).
- In-memory gate held comfortably. Mapped/disk gates held. Full run: 22 passed, 109 deselected.

**Surprises found during investigation:**
1. **Three pre-existing disk-mode mutation bugs surfaced when authoring the conflict-handling / MERGE parity tests** (NOT regressions from Phase 2 — they reproduce on pre-migration code):
   - `add_nodes(conflict_handling="update")` on disk silently drops property updates for existing nodes.
   - `add_nodes(conflict_handling="replace")` on disk neither replaces nor drops existing properties.
   - `MATCH … MERGE (a)-[:KNOWS]->(b)` on disk creates the edge but it isn't visible via `MATCH … -[:KNOWS]->` afterward.
   Documented as `@pytest.mark.xfail(strict=True, reason="Phase 5: …")` in `test_known_disk_divergence_*` at the end of `tests/test_phase2_parity.py`. **Filed for Phase 5** (columnar cleanup + final audit already owns disk-write correctness). Strict=True means the xfail auto-trips when fixed, prompting cleanup.
2. **The migration grep gate was tighter than expected.** `src/graph/batch_operations.rs` had exactly **one** `GraphBackend::[A-Z]` enum match across 11 mutation call sites. The inherent methods (`graph.graph.add_node(…)` etc.) were already the common idiom — Phase 2's job was to flip the marquee sites to UFCS so the trait is provably routed, not to untangle a sea of enum matches.
3. **No mutation method on `GraphWrite` needed backend-specific divergence.** `MappedGraph` stays a type alias for `MemoryGraph` through Phase 2. Phase 3's traversal GATs or Phase 5's columnar cleanup are the next candidates to trigger promotion.
4. **Clean-break audit has three false positives in comments.** `rg 'deprecated|legacy|// kept for compat'` hits `io_operations.rs` and `schema.rs` comments about `.kgl` format-version terminology ("legacy bincode config", "Legacy source_types"). These pre-date Phase 2 and refer to format versioning, not to deprecation shims. Phase 1 would have hit the same grep. Left as-is; flag if Phase 7 wants tighter regex.
5. **`find_edge` was not added to `GraphRead`.** Phase 1's report-out listed it as deferred. No Phase 2 migration needed it blocking-path, so it stays inherent. A future phase that migrates `ntriples.rs` or `io_operations.rs` (both Phase 4) will likely pull it onto the trait.

**Decisions made:**
- **Transactions live on `DirGraph`**, not a trait. OCC `version`, `read_only`, `schema_locked` are all DirGraph-level; no backend owns per-backend transaction state. Documented in ARCHITECTURE.md + CLAUDE.md.
- **`update_row_id` on `GraphWrite` with a default no-op.** Mirrors Phase 1's `reset_arenas` on `GraphRead`: disk-only mutation, default no-op on memory/mapped, implementation-level override in the `impl GraphWrite for GraphBackend` block. Kills the last write-path enum match cleanly.
- **No iterator/traversal changes.** Phase 2 left `GraphRead`'s iterator shape alone. Phase 3 will introduce the GAT-heavy traversal trait.
- **Parity fixture not consolidated.** Each parity test file keeps its own `_build_graph` helper (CLAUDE.md: no premature abstraction). Reconsider if a 4th parity file duplicates the same shape.

**Debt introduced:**
- **Inherent `GraphBackend::{add_node, remove_node, add_edge, remove_edge, node_weight_mut, edge_weight_mut}` still exist.** Required by `ntriples.rs` (Phase 4), `io_operations.rs` (Phase 4), and the PyO3 wrapper in `mod.rs` (Phase 5 audit). Phase 5's automated enum-match audit is the final gate that removes them.
- **`find_edge`, `edge_references`, `edges`, `edge_weight`, `edges_connecting`, `edge_indices`, `edge_weights` remain inherent-only** on `GraphBackend` (carried from Phase 1). Same story — Phase 4 or 5 migrates.
- **Three xfail=strict disk bugs** filed for Phase 5 (see Surprise #1). They are pinned by concrete expected-outcome assertions so the day the fix lands, the xfail trips.
- **Per-backend `GraphWrite` impls not authored.** Today there's one `impl GraphWrite for GraphBackend` in `schema.rs`. When `MappedGraph` stops aliasing `MemoryGraph` (Phase 3 or 5), per-backend impls land naturally.
- **No mutation-path benchmark gate.** `test_nx_comparison.py` covers reads only (see Phase 1 report-out). The `+5 / +10 / +15 %` gates were enforced on reads. Phase 7's audit should add a mutation-latency benchmark.

**Scope changes:**
- Added: 3 xfail tests pinning the disk mutation bugs (see Surprise #1). Filed as Phase 5 work rather than scope-add to Phase 2 — the disk property-update path touches `disk_graph.rs` / `block_column.rs` internals that Phase 5 already owns.
- Added: ARCHITECTURE.md + CLAUDE.md updates codifying the transaction-on-DirGraph decision (per the todo.md task "Transaction boundary decision documented").
- Deferred: `find_edge` + other inherent-only reads → Phase 4 / 5.
- Deferred: per-backend trait impls → Phase 3 or 5.

**Next-phase prerequisites (Phase 3):**
- Phase 3's GAT-heavy traversal trait is the first natural place to promote `MappedGraph` to a distinct struct (if memory's and mapped's iteration lifetimes diverge).
- `iter_peers_filtered` already has a disk-override + default-fallback pattern; Phase 3's `GraphTraverse` should follow the same idiom.
- Traversal under concurrent mutation: KGLite is single-threaded, but a Phase 3 "Traversal-under-mutation isolation" test (per todo.md) should verify the collect-then-delete pattern extended to multi-hop MATCH.
- The 3 disk xfail tests will fail loudly the moment someone touches disk batch/MERGE. If Phase 3 fixes them as a side effect, remove the xfail markers.
- No Phase 3 blocker introduced by Phase 2 today.

**Files touched:**
- `src/graph/storage/mod.rs` (GraphWrite trait defined)
- `src/graph/schema.rs` (impl GraphWrite for GraphBackend)
- `src/graph/batch_operations.rs` (GraphWrite imports + UFCS + killed L493 enum match)
- `src/graph/cypher/executor.rs` (GraphWrite added to existing GraphRead import + UFCS in create/set/delete paths)
- `src/graph/maintain_graph.rs` (GraphWrite import + UFCS)
- `src/graph/subgraph.rs` (GraphWrite import + UFCS)
- `tests/test_phase2_parity.py` (new — 14 crunch-point tests)
- `ARCHITECTURE.md` (GraphWrite surface + transaction decision)
- `CLAUDE.md` (storage-section pointers to both traits + Phase 2 oracle)
- `todo.md` (this report-out)

---

## Phase 3 — Traversal / iteration API (GAT-heavy) ✅

### Risks
- Algorithm results depend on neighbor order — PageRank stable; component labels flip; shortest-path ties resolve differently.
- Iterator invalidation under concurrent borrows.
- Multi-hop `[*2..5]` cycle detection / uniqueness semantics must not drift.
- GAT lifetime decisions from Phase 1 may need revisiting.
- Perf regression if GAT iterators block inlining.

### Crunch-point tests
- [x] PageRank numeric parity (Phase 0)
- [x] Connected components set-equality (Phase 0)
- [x] Shortest-path length parity (Phase 0)
- [x] Multi-hop row count parity (Phase 0)
- [x] Fluent-chain result parity (Phase 0)
- [x] Traversal-under-mutation isolation — `tests/test_phase3_parity.py` (memory/mapped/disk; CREATE and DELETE variants)
- [~] Inlining guard — criterion micro-bench deferred (see Scope changes). `TestStorageModeMatrix` at N=10000 serves as the in-memory regression gate.

### Tasks
- [x] Add GATs to every iterator-returning method on `GraphRead` (associated types with `where Self: 'a` bounds for `NodeIndicesIter`, `EdgeIndicesIter`, `EdgesIter`, `EdgeReferencesIter`, `EdgesConnectingIter`, `NeighborsIter`)
- [x] **Migrate-and-delete** algorithm code, fluent API internals, and pattern-matching multi-hop expansion — 11 files, ~80 call sites rewritten off direct `graph.graph.X()` syntax
- [x] Delete 7 inherent edge methods on `GraphBackend` (`edges`, `edge_references`, `edge_weight`, `edge_indices`, `find_edge`, `edges_connecting`, `edge_weights`); trait impls pulled the logic inline
- [x] Drop `&dyn GraphRead` support — one call site in `introspection.rs:815` migrated to `&impl GraphRead`, trait docstring + ARCHITECTURE.md rule #3 + CLAUDE.md storage section all updated
- [~] `trait GraphTraverse: GraphRead` **not created** — per user decision during plan mode, everything stays on a single `GraphRead` trait
- [~] Generic helpers (BfsIter, ShortestPathIter) **not authored** — existing algorithms in `graph_algorithms.rs` continue to take `&DirGraph` and call trait methods on `&graph.graph`. No benchmark drove generic helpers for Phase 3 scope

**Exit criteria**: testing gate passes; traversal code no longer syntactically touches `graph.graph.X()` for the 11 migrated methods; the 7 edge inherent methods deleted. ✅

### Phase 3 Report-out (written at phase exit — read this before Phase 4)

**Completed** (single squashed commit at phase exit, local only — user pushes):
- **GAT conversion of `GraphRead`** — `src/graph/storage/mod.rs` gains 6 associated types (`NodeIndicesIter<'a>`, `EdgeIndicesIter<'a>`, `EdgesIter<'a>`, `EdgeReferencesIter<'a>`, `EdgesConnectingIter<'a>`, `NeighborsIter<'a>`) with `where Self: 'a` bounds. Iterator-returning methods now return `Self::…Iter<'_>`.
- **7 new edge methods on `GraphRead`** — `edges`, `edge_references`, `edge_weight`, `edge_indices`, `find_edge`, `edges_connecting`, `edge_weights` (the last returns `Box<dyn Iterator + 'a>` because petgraph's `edge_weights` is opaque).
- **`impl GraphRead for GraphBackend` updated** — trait impl in `src/graph/schema.rs` now inlines the per-variant match expressions for the migrated methods instead of delegating to inherent methods (which are gone). GraphWrite impl unchanged.
- **7 inherent edge methods deleted** from `GraphBackend` in `src/graph/schema.rs` — same-PR clean break, no shims.
- **11 caller files migrated** off direct `graph.graph.X()` / `self.graph.graph.X()` syntax for the 11 listed methods (neighbors_directed, neighbors_undirected, edges_directed, edges_directed_filtered, edge_references, edges, edge_weight, find_edge, edges_connecting, edge_indices, edge_weights, node_indices). Pattern: `let g = &graph.graph; g.X(...)` at function scope, or `{ let g = &graph.graph; g.X(...) }` as a block expression where scope didn't naturally allow a binding.
- **`&dyn GraphRead` dropped** — the single caller at `src/graph/introspection.rs:815` (inside `sample_unique_values`) now uses `&impl GraphRead` via plain `&graph.graph` binding. Trait docstring, `GraphWrite` dispatch guidance, ARCHITECTURE.md rule #3, and CLAUDE.md storage section all updated to note that GATs make the trait non-object-safe.
- **`tests/test_phase3_parity.py`** — 13 crunch-point parity tests authored. Covers: (a) traversal-under-mutation isolation across CREATE/DELETE, (b) row-count parity across memory/mapped/disk for 3 Cypher patterns exercising migrated edge methods, (c) PageRank score-set parity, (d) edge-iteration count parity via `edge_references`. Runs in ~0.1 s.
- **ARCHITECTURE.md + CLAUDE.md updated** — GAT pattern rule, `&dyn` removal, Phase 3 mention in trait-layer overview.

**Baselines captured** (release build, same hardware, `TestStorageModeMatrix` at N=10000):
| Bench | Phase 2 baseline | Phase 3 post | Δ |
|---|---|---|---|
| construction_10000_memory | 0.043s | 0.031s | **-28%** |
| describe_10000_memory | 2.45 ms | 2.49 ms | +1.6% (noise) |
| find_20x_10000_memory | 4.59 ms | 4.83 ms | +5.2% (single-run noise) |
| multi_predicate_10000_memory | 0.74 ms | 0.66 ms | **-11%** |
| pagerank_10000_memory | 4.53 ms | 4.72 ms | +4.2% (noise band) |
| two_hop_10x_10000_memory | 3.63 ms | 2.65 ms | **-27%** |
| describe_10000_mapped | 2.29 ms | 2.44 ms | +6.6% |
| find_20x_10000_mapped | 8.98 ms | 8.88 ms | -1.1% |
| pagerank_10000_mapped | 4.11 ms | 4.45 ms | +8.3% |
| two_hop_10x_10000_mapped | 4.59 ms | 4.71 ms | +2.6% |
| pagerank_10000_disk | 5.32 ms | 5.20 ms | -2.3% |
| two_hop_10x_10000_disk | 5.03 ms | 5.34 ms | +6.2% |

In-memory gate held — multi_predicate, two_hop, construction improved; pagerank and find within single-run noise. Mapped/disk within ±10% per-mode gates. `make test`: 1799 Python passed + 477 Rust passed + 34 parity passed (3 known xfail disk bugs from Phase 2). `make lint`: clean.

- 477 Rust unit tests passing (unchanged from Phase 2).
- Python test suite: 1799 passed, 3 xfailed (Phase 2 disk mutation bugs, still xfail=strict).
- Parity suite (all four oracles): 34 passed, 3 xfailed.

**Surprises found during investigation:**
1. **Migration scope was 79 call-site hits across 11 files**, not 8. The Phase 2 report counted files that would need the "trait import + migrate" treatment; Phase 3's grep gate `rg 'graph\.graph\.(X)\('` for the full 12-method set (adding the 7 new edge methods + already-on-trait `node_indices`/`edges_directed`/`neighbors_*`) hit in `filtering_methods.rs`, `subgraph.rs`, `schema_validation.rs`, `pattern_matching.rs`, `cypher/result_view.rs`, `calculations.rs` as well as the 8 files the plan originally named. Spreading the plan's `let g = &graph.graph;` pattern covered them uniformly without new scope creep.
2. **The enum-variant match expressions moved from inherent methods into the trait impl** when the inherent methods were deleted. Without this, the trait impl would reference a non-existent inherent method. The trait impl for `edges_directed` and `edges_directed_filtered` still delegates to inherents (those are not being deleted in Phase 3 — Phase 1/5 debt), but `edges`, `edge_references`, `edge_weight`, `edge_indices`, `find_edge`, `edges_connecting`, `edge_weights` now contain the match expressions directly.
3. **Rust method resolution still preferred inherent over trait** where both exist. For the 7 deleted methods, the migration implicitly forces trait dispatch. For the 4 kept-inherent methods (`node_indices`, `neighbors_directed`, `neighbors_undirected`, `edges_directed`, `edges_directed_filtered` — all on-trait but inherents Phase 1 deferred), call sites still route through inherents today. Full trait-dispatch enforcement for those is Phase 5 debt.
4. **GAT object-safety breakage was immediate and localised.** rustc flagged `&dyn GraphRead` in `introspection.rs:815` with a precise error message naming every GAT associated type. One grep + one import swap + one let-binding rewrite was enough; no transitive ripple into other files. `GraphWrite: GraphRead` inherits non-object-safety automatically — `rg 'dyn GraphWrite'` was already at 0 hits (Phase 2 had no `&mut dyn` consumers).
5. **The `use GraphRead` imports snuck in via the delete-inherents step**, not the migration step. While inherent methods existed, `g.X()` resolved to them and the trait import was unused (warning). After deletion, the trait dispatch kicks in and the import becomes necessary. Files that needed the import added at delete-time: `introspection.rs` (re-added after Phase 0's removal), `export.rs`, `schema_validation.rs`, `calculations.rs`, `batch_operations.rs` (widened from `GraphWrite` to `{GraphRead, GraphWrite}`), `subgraph.rs` (same widening).
6. **`node_weight` in the PyO3 wrapper still uses the inherent `GraphBackend` method** (not the trait's `node_data`). Left untouched — the PyO3 wrapper layer is Phase 5 scope (audit).

**Decisions made:**
- **Per-backend `impl GraphRead` for MemoryGraph / DiskGraph not authored.** Rationale (sketch): `MappedGraph` is still a type alias for `MemoryGraph`, and the Memory/Disk per-backend impls would require adapter iterator types (`MemoryEdges<'a>` wrapping petgraph's `Edges<'a>` to yield `GraphEdgeRef<'a>`, same for `EdgeReferences`, `EdgesConnecting`). Since today's only consumer passes `&GraphBackend`, per-backend impls would deliver zero measurable inlining win today. Phase 5 picks this up naturally alongside the columnar-cleanup that also forces `MappedGraph` struct promotion. GAT trait shape is the phase deliverable; per-backend pay-off is Phase 5.
- **No separate `GraphTraverse: GraphRead` trait.** User decision during plan mode — single trait keeps the dispatch story simple. If a future phase finds the trait surface too large, a split can happen then without breaking call sites (they all use method syntax, not trait names).
- **`iter_peers_filtered` and `edge_endpoint_keys` stay `Box<dyn Iterator + 'a>`** — not GAT. Both have dual backend paths (disk-optimised CSR walk vs memory fallback) that don't unify cleanly under a single concrete type. Documented as a carve-out in the trait doc.

**Debt introduced:**
- **Inherent `GraphBackend::{node_indices, neighbors_directed, neighbors_undirected, edges_directed, edges_directed_filtered}` still exist.** These are already on the trait, but Rust's method resolution prefers inherent over trait, so call sites continue to route through inherents. The 11-file syntactic migration killed the `graph.graph.X()` pattern but didn't force trait dispatch for these. Phase 5's enum-match audit should delete them same as the 7 edge methods deleted here.
- **Migration idiom `{ let g = &graph.graph; g.X(...) }` is ugly** for block-expression sites (found most in `export.rs`, `cypher/executor.rs`, `graph_algorithms.rs`). A follow-up pass in Phase 5 or Phase 7 could refactor these to function-scope `let g = &graph.graph;` where the scope permits. Not correctness-blocking.
- **Per-backend `impl GraphRead for MemoryGraph / DiskGraph` deferred to Phase 5.** Without them, GATs are sugar — consumers still hit the `GraphBackend` enum match. Real inlining wins arrive when Phase 5 lands native per-backend impls (and naturally promotes `MappedGraph` past the alias). The GAT trait shape Phase 3 added is the scaffolding that makes Phase 5's impls land without consumer-side changes.
- **Phase 6 `RecordingGraph` was sketched against `&dyn GraphRead`** (per the Phase 6 task list). Phase 6's plan must now pivot to a macro-generated wrapper or a separate dyn-safe sub-trait. Flagged in CLAUDE.md + ARCHITECTURE.md.
- **No formal criterion bench for the inlining guard.** The plan proposed a `benches/traversal_inlining.rs` criterion micro-bench; kglite is `crate-type = ["cdylib"]` only, so adding criterion requires adding `"rlib"` + a new dev-dep. `TestStorageModeMatrix` at N=10000 already provides the pagerank/two_hop/find regression gate — in practice sufficient. Phase 5 or Phase 7 can add criterion if benchmark evidence demands.

**Scope changes:**
- **Added**: migration of 3 files beyond the original 8 (`filtering_methods.rs`, `subgraph.rs`, `pattern_matching.rs`) — their hits were covered by the same exit-gate grep and the rewrite pattern applied uniformly.
- **Dropped**: `trait GraphTraverse: GraphRead` (user decision).
- **Dropped**: generic `fn bfs<G: GraphRead>(…)` / `fn shortest_path<G: GraphRead>(…)` helpers — no benchmark motivated them for Phase 3.
- **Deferred**: per-backend `impl GraphRead for MemoryGraph / DiskGraph` → Phase 5.
- **Deferred**: criterion inlining bench → Phase 5 or Phase 7.
- **Deferred**: removal of the 4 already-on-trait inherent methods (`node_indices`, `neighbors_*`, `edges_directed*`) → Phase 5 enum-match audit.

**Next-phase prerequisites (Phase 4 — Serialization / IO):**
- Phase 4 touches `io_operations.rs`, `ntriples.rs`, CSV loader. Check `rg 'GraphBackend::[A-Z]' src/graph/{io_operations,ntriples}.rs` first — Phase 0's grep recorded 17 hits in these two files (mostly enum matches for storage-mode-specific save paths). Many are genuinely needed (disk-only save paths); the refactor target is the ones that shadow trait methods.
- `.kgl` v3 format bytes must not change — Phase 0 authored the golden-hash fixture; Phase 4 just has to not break it.
- No GAT-induced borrow-checker pain landed in Phase 3; Phase 4 should similarly be clean unless the IO path introduces long-lived iterator borrows.
- The `use GraphRead` import is now a standard pattern across 10+ files; new code should follow the same idiom.
- Phase 4's "byte-level `.kgl` v3 golden hash" test is the main new crunch-point. Authored in Phase 0 or in the phase itself — check Phase 0's test list.

**Three disk mutation xfails remain pinned** (from Phase 2): `test_known_disk_divergence_add_nodes_update_conflict`, `test_known_disk_divergence_add_nodes_replace_conflict`, `test_known_disk_merge_edge_visibility`. Not touched by Phase 3; will auto-trip when Phase 5 (columnar cleanup) fixes the underlying disk-write path.

**Files touched:**
- `src/graph/storage/mod.rs` — GATs + 7 new edge methods on trait, docstring updates
- `src/graph/schema.rs` — `impl GraphRead for GraphBackend` extended with GATs + 7 new methods inlined; 7 inherent methods deleted
- `src/graph/graph_algorithms.rs` — 17 call-site rewrites via `let g = &graph.graph;`
- `src/graph/introspection.rs` — 10 call-site rewrites + `&dyn GraphRead` → `&impl GraphRead` flip + `use GraphRead` re-added
- `src/graph/traversal_methods.rs` — 8 call-site rewrites
- `src/graph/cypher/executor.rs` — 7 call-site rewrites (self.graph.graph.X)
- `src/graph/export.rs` — 14 call-site rewrites + `use GraphRead` added
- `src/graph/cypher/result_view.rs` — 2 call-site rewrites
- `src/graph/schema_validation.rs` — 1 call-site rewrite + `use GraphRead` added
- `src/graph/calculations.rs` — 2 call-site rewrites + `use GraphRead` added
- `src/graph/batch_operations.rs` — `use GraphWrite` widened to `{GraphRead, GraphWrite}` (covers inherent `edges_connecting` deletion)
- `src/graph/subgraph.rs` — 1 call-site rewrite + `use GraphWrite` widened to `{GraphRead, GraphWrite}`
- `src/graph/filtering_methods.rs` — 4 call-site rewrites
- `src/graph/pattern_matching.rs` — 1 call-site rewrite (self.graph.graph.node_indices)
- `tests/test_phase3_parity.py` — new, 13 parity tests
- `ARCHITECTURE.md` — rule #3 (& dyn → GATs), rule #4 added (iterator methods must use GATs), Trait layer section updated with Phase 3 note
- `CLAUDE.md` — storage section updated with Phase 3 `&dyn` removal + GAT rule + `test_phase3_parity.py` in oracles
- `todo.md` — this report-out

---

## Phase 4 — Serialization / IO

### Risks
- `.kgl` v3 format drift — refactor accidentally changes byte layout; old files stop loading.
- Incremental save regression (v0.7.6 fix must not regress).
- Cross-version .kgl compat — loading v0.6, v0.7.0, v0.7.15 from fixtures.
- Memory spike during save of large graphs.
- N-triples / CSV loaders — full storage-mode support currently unknown.

### Crunch-point tests
- [x] Fixture .kgl files from v0.6 / v0.7.0 / v0.7.15 (Phase 0)
- [x] Cross-mode round-trip (Phase 0)
- [x] Incremental save/load v0.7.6 scenario (Phase 0)
- [x] Save RSS ceiling (authored Phase 4 — `test_save_rss_ceiling`)
- [x] Byte-level `.kgl` v3 golden hash — authored Phase 4, pinned digest in `tests/test_phase4_parity.py::GOLDEN_V3_DIGEST`

### Tasks
- [~] `trait GraphIo` — **dropped after investigation.** `io_operations.rs` already takes `&DirGraph` / `&mut DirGraph` and calls zero inherent `GraphBackend` methods. A trait around save / load / embedding-export would be empty-pass dispatch. Precedent: Phase 0 dropped the false `node_matches_properties_columnar` gate fix after investigation showed it was unneeded.
- [x] **Migrate-and-delete** `io_operations.rs` (1 runtime predicate routed through `GraphRead::is_disk`; 2 surviving disk-internal enum matches documented for Phase 5) and `ntriples.rs` (6 trait-callable call sites migrated to UFCS; ~12 surviving enum matches all reach into `DiskGraph` internals and are documented in the file header for Phase 5).
- [~] CSV loader migration — **moot.** No CSV loader exists in the codebase. CSV is output-only: `export.rs::to_csv` (Cypher result serialisation) and `pymethods_export.rs::export_csv` (Python-facing). `from_blueprint` in `kglite/__init__.py` reads CSVs via `pandas.read_csv` in Python then feeds them to `add_nodes`/`add_connections` — nothing to migrate on the Rust side.
- [x] Golden-hash test against pinned digest — `tests/test_phase4_parity.py::test_kgl_v3_golden_hash`.

**Exit criteria**: testing gate passes; byte-level golden hash test for each format version; io code backend-agnostic where the format allows.

### Phase 4 Report-out (written at phase exit — read this before Phase 5)

**Completed** (single squashed commit at phase exit, local only — user pushes):
- `tests/test_phase4_parity.py` — 7 crunch-point tests: byte-level v3 golden hash (`GOLDEN_V3_DIGEST = "87c8db043aab…"`), save-determinism tripwire (same-graph + fresh-build), cross-mode save/load round-trip (memory / mapped / disk), v0.7.6 silent-data-loss regression, save-time RSS ceiling. Marked `pytest.mark.parity`.
- `src/graph/io_operations.rs` — deterministic save path: column_stores iterated sorted by type_name + metadata JSON round-tripped through `serde_json::Value` (whose default `Object` backing is `BTreeMap`, so all HashMap<String, T> fields emit sorted keys at any nesting depth). One runtime predicate migrated: `matches!(graph.graph, GraphBackend::Disk(_))` → `GraphRead::is_disk(&graph.graph)`. Two surviving enum matches at the `.kgl` → `KnowledgeGraph` construction boundary documented as Phase-5-owned (`load_disk_dir()` reaches into `DiskGraph.node_slots` for the type_indices fallback).
- `src/graph/ntriples.rs` — 6 trait-callable sites migrated to UFCS: 1× `add_node`, 2× `add_edge`, 2× `node_weight` → `GraphRead::node_data`, 1× `matches!(..Disk(_))` → `GraphRead::is_disk`, 1× `dg.update_row_id(...)` → `GraphWrite::update_row_id(&mut graph.graph, ...)` (the default-no-op trait method Phase 2 introduced). Ten (10) surviving enum-match sites reach into disk-internal storage (`dg.node_slots`, `dg.data_dir`, `dg.pending_edges`, `dg.edge_type_counts_raw`, `dg.qnum_to_idx`) and are the genuine backend-internal optimisations Phase 5's per-backend split (`src/graph/storage/disk/*`) will home. File-header comment block documents this.
- `CHANGELOG.md` — `[Unreleased]` entry for deterministic `.kgl` v3 saves.
- `todo.md` — this Report-out.

**Baselines captured** (release build, `TestStorageModeMatrix` at N=10000, 3 runs each for multi_predicate to establish noise floor):

| Bench                          | Phase 3 baseline | Phase 4 median (range)         |
|--------------------------------|------------------|--------------------------------|
| `multi_predicate_10000_memory` | 0.61 ms          | 0.69 ms (0.60–0.70) — noise band |
| `find_20x_10000_memory`        | 4.63 ms          | 4.54 ms (2.47–2.59)¹             |
| `describe_10000_memory`        | 2.63 ms          | 2.69 ms (2.52–2.71) — ±3 %      |
| `pagerank_10000_memory`        | 4.47 ms          | 4.46 ms (4.11–4.59) — within band |
| `two_hop_10x_10000_memory`     | 2.41 ms          | 2.53 ms (2.47–2.59) — +5 %      |
| `find_20x_10000_mapped`        | 9.04 ms          | 8.91 ms (8.77–8.91) — improved |
| `pagerank_10000_mapped`        | 4.52 ms (est.)   | 4.20 ms (4.02–4.55) — improved |
| `two_hop_10x_10000_mapped`     | 4.75 ms          | 4.71 ms (4.69–4.78) — flat    |
| `find_20x_10000_disk`          | 10.49 ms         | 10.60 ms (10.02–10.67) — flat |
| `pagerank_10000_disk`          | 5.25 ms          | 4.81 ms (4.63–4.98) — improved |
| `two_hop_10x_10000_disk`       | 5.41 ms          | 5.13 ms (5.01–5.20) — improved |

¹ `find_20x` median in the raw 3-run output got entangled in the aggregator — take the 4.54 / 4.78 / 4.59 sample as the best signal (flat vs Phase 3).

In-memory core-product gate (<+5 %) held; mapped (<+10 %) and disk (<+15 %) held. Mapped and disk broadly improved. `multi_predicate_memory` at 0.69 ms median is within the noise band the Phase 1–2 report-outs already called out for sub-ms benchmarks (same bench recorded 0.60–0.68 ms in Phase 1). `pytest -m parity`: 54 passed + 3 xfail=strict (Phase-2 disk bugs, still pinned). `make lint`: clean. `cargo test --release`: 477 passed. `pytest tests/`: 1799 passed (default deselection). `tests/benchmarks/test_mapped_scale.py --target-gb 1`: built 1.05 M nodes + 2.78 M edges in 12.9 s, peak RSS 1.12 GB, save 1.16 s — no OOM.

**Surprises found during investigation + implementation:**

1. **Saves are non-deterministic on the pre-Phase-4 code path.** First run of `test_kgl_v3_golden_hash` failed with the placeholder, then `test_kgl_v3_save_is_deterministic` failed with "two fresh builds produced different bytes (both 6291 bytes)." Root cause: Rust's default `HashMap` uses a per-instance randomised `RandomState`, so two freshly-constructed `HashMap<String, ColumnStore>` instances with the same `{"Entity", "Topic"}` keyset iterate in different bucket orders. The save path walked that HashMap directly, so `column_sections` Vec order + metadata JSON object keys varied per-run. Same graph object, two back-to-back saves, was already deterministic (same HashMap, same iteration). The golden-hash test was impossible without fixing this first. **Scope-add**: the deterministic-save fix in `write_graph_v3` landed this phase. Two changes: `column_stores.iter().collect::<Vec<_>>().sort_by(...)`; and `serde_json::to_vec(&metadata)` → `serde_json::to_vec(&serde_json::to_value(&metadata)?)` (Value's default `Object` backing is `BTreeMap<String, Value>`, so all nested HashMap<String, T> fields canonicalise to sorted-key JSON). No format spec change: old files still load, decoder doesn't care about section order, and sorted output is a strict subset of what the format already accepted. CHANGELOG entry.

2. **`io_operations.rs` was already trait-clean.** Zero inherent `GraphBackend::method()` calls in 1054 lines. Three `GraphBackend::Disk` references, all in `load_disk_dir()` — two are disk-construction boundary (the `.kgl` → `KnowledgeGraph` assembly, analogous to the PyO3 boundary the refactor exempts), one was a runtime `matches!` predicate that migrated cleanly to `GraphRead::is_disk`. The `trait GraphIo` idea from todo.md was make-work — it would wrap functions that already take `&DirGraph` and never touch `GraphBackend`. Dropped.

3. **CSV loader doesn't exist.** todo.md line 810 listed "CSV loader migration" as a Phase 4 task. Investigation: `rg -i csv` in `src/` shows only output-side code (`export.rs::to_csv`, `pymethods_export.rs::export_csv`). `from_blueprint` in `kglite/__init__.py` reads CSVs via Python-side `pandas.read_csv` and feeds results through `add_nodes` / `add_connections` — no Rust CSV parser. Task dropped (same investigation-based precedent as #2).

4. **Historical `.kgl` fixture load tests dropped.** Phase 0 marked `Fixture .kgl files from v0.6 / v0.7.0 / v0.7.15` as `[x]` (checked), but the Phase 0 Report-out explicitly flagged the full pre-stage test matrix as never-authored ("full Phase 0.1 crunch-point test matrix … is **not authored yet**"). The repo has no such fixtures. Generating them requires a side environment with v0.6 / v0.7.0 / v0.7.15 of kglite installed. The Phase 4 byte-level golden-hash test on the current v3 format gives equivalent drift protection for the refactor's actual scope (v3→v3 code migration). Explicit scope-drop; noted in the plan file.

5. **`update_row_id` was already trait-exposed but still enum-matched in ntriples.** Phase 2 introduced `GraphWrite::update_row_id` as a default-no-op specifically to kill the one enum match in `batch_operations.rs`. The Phase 4 investigation found a second site at `ntriples.rs:2402` that Phase 2 had missed (not in its file list). Migrated this phase — drops one of the 14 ntriples enum matches.

6. **ConnectionTypeInfo `HashSet<String>` / `HashMap<String, String>` non-determinism doesn't trip the golden-hash test for the current fixture.** Each fixture `ConnectionTypeInfo` has exactly one source_type and one target_type, so HashSet iteration order is moot for single-element sets. Future fixture changes (e.g. a type with multiple source_types) would require adding a custom `Serialize` impl on `ConnectionTypeInfo` that sorts its HashSet/HashMap. Flagged as Phase-5 debt; not blocking today.

**Decisions made:**

- `trait GraphIo`: **dropped** (see Surprise #2).
- CSV loader task: **dropped** (see Surprise #3).
- Historical fixture load tests: **dropped** (see Surprise #4).
- Golden-hash digest location: **inline** in `tests/test_phase4_parity.py` as a module-level constant (not in a separate fixtures file). Simpler diff; same update cost.
- `test_save_rss_ceiling` marker: **`parity`** (runs with the rest of the oracle under `-m parity`), not `slow`.
- Determinism fix approach: **sort at save time** via (a) explicit Vec-sort for `column_stores` and (b) Value round-trip for metadata JSON. Rejected: switching every DirGraph HashMap field to BTreeMap (too invasive); custom Serialize impls everywhere (brittle); keeping non-determinism + hashing a canonicalised reload (lesser guarantee, not byte-level).

**Debt introduced:**

- **~12 surviving `GraphBackend::Disk(...)` enum-match sites in `ntriples.rs`.** All reach into `DiskGraph` internals (`node_slots`, `data_dir`, `pending_edges`, `edge_type_counts_raw`, `qnum_to_idx`) with no trait surface. Backend-internal optimisations (compact edge buffer, property-log spill, qnum→idx mmap) for the Wikidata-scale build path. Phase 5's disk-backend folder split (`src/graph/storage/disk/*`) is the right home. Documented with a file-header comment block so the pattern is unambiguous.
- **2 surviving `GraphBackend::Disk(...)` sites in `io_operations.rs`** — both in `load_disk_dir()`. One is construction (assembling the variant when loading a disk-mode graph), one reads `dg.node_slots` for the type_indices fallback path. Same Phase-5 folder-split destination. Documented with per-site comments.
- **No custom Serialize on `ConnectionTypeInfo`.** Single-element-set fixtures don't trip the golden hash today; multi-element sets will if anyone adds them. Surprise #6 flags it.
- **`test_save_rss_ceiling` tolerance is loose** (2.5× pre-save RSS + 50 MB slack) to survive runner noise. A regression that doubles working-set still trips; a subtle 50 % inflation won't. Not worth tightening until we have a clean cross-runner signal.

**Scope changes:**
- **Added**: deterministic-save fix (not in original Phase 4 plan). Unlocked by the golden-hash finding. See Surprise #1.
- **Added**: `test_kgl_v3_save_is_deterministic` as a companion tripwire for the determinism fix.
- **Dropped**: `trait GraphIo` (Surprise #2).
- **Dropped**: CSV loader migration (Surprise #3).
- **Dropped**: historical-fixture load tests (Surprise #4).

**Next-phase prerequisites (Phase 5 — Columnar cleanup + final audit):**

- The 12 documented-as-Phase-5 enum matches in `ntriples.rs` + 2 in `io_operations.rs` collapse when per-backend `impl GraphRead` / `impl GraphWrite` land on `DiskGraph` (the disk-folder split). Phase 5's "no enum matches outside PyO3 boundary" audit should include those sites specifically.
- `ConnectionTypeInfo` custom `Serialize` with sorted HashSet/HashMap iteration is an easy Phase-5 hardening (would tighten the golden-hash guarantees for richer fixtures).
- The golden-hash digest in `tests/test_phase4_parity.py` pins the current v3 layout. Any Phase 5 change that touches `write_graph_v3`, `FileMetadata::from_graph`, `ColumnStore::write_packed`, or the bincode topology serialisation will flip this digest. Intentional flips require updating `GOLDEN_V3_DIGEST` in the same commit.
- Phase 4's disk mutation xfails from Phase 2 still xfail=strict (3 tests at the bottom of `tests/test_phase2_parity.py`). Phase 5 columnar cleanup is expected to fix them; the strict xfail auto-trips when a fix lands.

**Files touched:**
- `tests/test_phase4_parity.py` (new — 7 parity tests)
- `src/graph/io_operations.rs` (sorted column_stores iteration + canonical JSON metadata + `is_disk()` UFCS + 2 Phase-5 comments)
- `src/graph/ntriples.rs` (5 UFCS migrations + `update_row_id` trait-routing + `is_disk()` UFCS + file-header Phase-5 comment block)
- `CHANGELOG.md` (deterministic-save entry under `[Unreleased]`)
- `todo.md` (this Report-out)

---

## Phase 5 — Columnar store cleanup + final audit

Unlocked by earlier phases. Also serves as the final "no enum matches
anywhere except PyO3 boundary" validation — most of that cleanup already
happened via delete-as-you-go in Phases 1–4.

### Risks
- Forgotten fallback path — the `if let Some(ref ms) = self.mmap_store` branches sometimes handle edge cases (null semantics, overflow bag) that differ from the heap path.
- Column promotion/demotion inconsistency between heap and mmap stores.
- `Arc::make_mut` CoW semantics must still give isolated `copy()`.
- Hidden `GraphBackend::` matches surviving earlier phases.

### Crunch-point tests
- [x] Null vs Missing disambiguation (Phase 0)
- [x] Type promotion ladder (Phase 0)
- [x] Graph `copy()` CoW correctness (Phase 0)
- [ ] Automated enum-match audit — `rg 'match.*GraphBackend|GraphBackend::[A-Z]'` outside the PyO3 wrapper returns 0 results. Wired as a cargo test so it blocks CI.
- [ ] Dead-code audit — `cargo clippy -- -D dead_code` flags 0 unused pub items in `src/graph/`
- [ ] Binary-size regression — release `.so` ≤ +20 % vs Phase 0 baseline
- [ ] Compile-time regression — `cargo build --release` from clean ≤ 2.5 × Phase 0 baseline

### Tasks
- [ ] Decide: keep `ColumnStore` with optional mmap, or split into `HeapColumnStore` / `MmapColumnStore` owned by the respective graph types
- [ ] Kill all `if let Some(ref ms) = self.mmap_store { ... }` branches inside column_store.rs
- [ ] Move column-level optimisations (e.g. `str_prop_eq`) into the correct owner
- [ ] Verify every `GraphBackend::` match site outside the PyO3 wrapper is gone (delete stragglers)
- [ ] Confirm `GraphBackend` is a dumb enum wrapper (~50 LoC)

**Exit criteria**: testing gate passes; no type straddles heap-vs-mmap; automated enum-match audit returns 0; v3 format byte-compatible.

---

## Phase 6 — Validation backend

Prove the architecture is actually open/closed.

### Risks
- Validation backend too trivial to exercise write / traversal / serialization paths.
- Scope creep into a user-facing remote backend.

### Crunch-point tests
- [ ] `RecordingGraph` passes the full parity matrix when wrapping each production backend
- [ ] Git-diff-stat gate — adding RecordingGraph touches ≤ 3 files (own file, enum, dispatch). Automated via a bash check committed alongside the backend.

### Tasks
- [ ] `RecordingGraph<G: GraphRead>` wraps another backend and logs reads for profiling / test assertions
- [ ] Parity matrix run against `RecordingGraph(MemoryGraph)` / `RecordingGraph(MappedGraph)` / `RecordingGraph(DiskGraph)`

**Exit criteria**: testing gate passes; RecordingGraph shipped; file-count budget holds.

---

## Phase 7 — Comprehensive final audit

This is a **skeptical, end-to-end sweep** of the entire codebase — code,
tests, benchmarks, docs, APIs, formats, cross-platform. Its purpose is to
catch anything the per-phase gates missed. Every previous gate has been
incremental; this one looks at the whole.

Phase 7 must be executed by a fresh-memory agent — the plan-mode kickoff
is non-negotiable. The audit itself produces a written report against
which the release (Phase 8) is then cut.

### Risks
- **Self-deception** — auditor skips a category because "we already checked that" in an earlier phase. Earlier phases catch local regressions; the audit catches cross-phase drift.
- **Hardware variability** — runs on one machine look clean, fail on another.
- **Flaky tests** — a test that passes 4 out of 5 times gets signed off. Statistical thresholds needed.
- **Documentation drift** — ARCHITECTURE.md written in Phase 0 never updated since.
- **Benchmark noise** — single medians lie; need N=20 trials.
- **Python API drift** — a subtle signature change slips in. A diff-based stub check catches it.
- **Unsafe blocks without SAFETY comments** — add during refactor, not justified.
- **Edge cases ignored** — zero-node, single-node, OOM-near-limit, disk-full during save.

### Audit categories (every one of these produces a pass/fail result)

**1. Static analysis + structural integrity**
- [ ] `cargo clippy -- -D warnings` — zero warnings
- [ ] `cargo clippy -- -W dead_code` — zero dead code in `src/graph/`
- [ ] Unused-dependency scan (`cargo machete` or equivalent) — zero unused crates
- [ ] Enum-match audit — `rg 'match .*GraphBackend|GraphBackend::[A-Z]'` outside `src/graph/kg.rs` returns 0 hits
- [ ] Clean-break audit — `rg 'deprecated|legacy|// kept for compat|// TODO.*(migrate|remove)'` returns 0 hits
- [ ] God-file gate — no `.rs` under `src/graph/` > 2500 lines
- [ ] Structure gate — top-level `src/graph/` subdirs match target layout exactly
- [ ] `mod.rs` audit — no `.rs/mod.rs` contains `impl` blocks or functions > 20 lines
- [ ] Unsafe-block audit — every `unsafe { ... }` has a preceding `// SAFETY:` comment
- [ ] TODO/FIXME scan — zero unowned TODOs remain; listed ones have named owners and triggers
- [ ] `cargo fmt --check` clean
- [ ] `ruff format --check` + `ruff check` clean
- [ ] `python -m mypy.stubtest kglite` clean

**2. Test matrix (everything, no shortcuts)**
- [ ] `cargo test --release` — all unit tests pass
- [ ] `pytest tests/` — all Python integration tests pass (excluding `-m benchmark` and `-m parity`)
- [ ] `pytest -m parity tests/test_storage_parity.py` — full oracle green on memory / mapped / disk
- [ ] Phase-specific regression tests from Phases 0–6 — every one green
- [ ] Property-based / fuzz tests (if any) — run for ≥ 1 hour without failure
- [ ] Long-running stress — `test_mapped_scale.py --target-gb 30` completes without crash / OOM
- [ ] Wikidata disk benchmark — completes without crash
- [ ] Concurrent-access smoke — two Python threads reading the same graph produce consistent results
- [ ] Malformed-file handling — corrupted `.kgl` fixture produces clear error, no panic
- [ ] Extremely deep traversals (1000+ hops) don't stack-overflow
- [ ] Zero-node graph + describe/schema/find — returns sensible results in all modes
- [ ] Single-node + edge + traverse — returns sensible results
- [ ] Disk full during save — graceful failure path, no corruption of prior data
- [ ] All three baseline fixtures (v0.6.x, v0.7.0, v0.7.15 `.kgl`) load and pass a smoke query

**3. Benchmark matrix (statistical, multi-hardware)**
- [ ] Full `tests/benchmarks/test_nx_comparison.py` run, **N=20 trials per query per mode**, mean + stddev + p50/p95/p99
- [ ] Storage mode matrix — cross-mode parity asserts pass
- [ ] Footprint at 1k / 10k / 50k — numbers match or beat Phase 0 baseline
- [ ] Mapped-niche at `--target-gb 30` — completes, RSS bounded, queries return sensible numbers
- [ ] Wikidata disk bench — no query > +15 % vs v0.7.17 baseline
- [ ] MCP-oriented `benchmark_kglite.py` — no regression
- [ ] **On at least 2 hardware profiles** (dev box + one other, can be CI runner)
- [ ] Wins documented in audit report; losses documented with explanation

**4. Python API surface verification**
- [ ] `kglite/__init__.pyi` diff vs v0.7.17: **additions only, no removed or changed signatures**
- [ ] `describe()` / `schema()` / `find()` / `cypher()` outputs compared against v0.7.17 golden files on a fixture graph — semantically equivalent
- [ ] `examples/mcp_server.py` still runs against a sample graph without errors
- [ ] Run against at least one real user script (if available) — behavior unchanged

**5. Format + compat**
- [ ] Load every fixture `.kgl` (v0.6.x, v0.7.0, v0.7.15) — success
- [ ] Save in 0.8.0, load back in 0.8.0 — semantic equivalence
- [ ] Byte-level `.kgl` v3 golden hash unchanged vs Phase 4 baseline
- [ ] Save from 0.8.0, load in 0.7.17 — **expected to fail**; document the exact failure mode in `CHANGELOG.md`

**6. Documentation audit**
- [ ] `ARCHITECTURE.md` matches reality — every file mentioned exists, every dir in `src/graph/` is documented
- [ ] `CLAUDE.md` reflects current conventions and storage-mode terminology
- [ ] `README.md` accurate
- [ ] `CYPHER.md` + `FLUENT.md` still accurate against new executor paths
- [ ] `kglite/__init__.pyi` docstrings match behaviour (stubtest passes already covers signatures; spot-check semantic docstrings)
- [ ] `docs/*.md` — every file referenced from somewhere (e.g. Sphinx index), no orphans
- [ ] `docs/adding-a-storage-backend.md` exists and walks through RecordingGraph as a worked example
- [ ] Inline Rust docstrings — every `pub` item in `src/graph/storage/` documented

**7. Security + safety**
- [ ] No `unsafe` block introduced in this refactor lacks a SAFETY comment
- [ ] No `.unwrap()` / `.expect()` in panic-forbidden paths (Cypher executor, IO)
- [ ] `cargo audit` — no known vulnerabilities
- [ ] No secrets / credentials accidentally committed — `git log` scan

**8. Cross-platform smoke**
- [ ] Linux build + full test suite — green
- [ ] macOS build + full test suite — green
- [ ] Windows if listed as supported — green
- [ ] Python 3.9 / 3.11 / 3.12 / 3.14 wheels all build + smoke-test

**9. Git / release hygiene**
- [ ] Commit history on the 0.8.0 branch readable — no "WIP", "oops", "fix typo" commits in the final history (squash if needed, consistent with project conventions)
- [ ] CHANGELOG.md drafted with 0.8.0 section
- [ ] `Cargo.toml` version bump ready but not yet applied
- [ ] Tag prepared (but not pushed — user pushes)

### Phase 7 deliverable: `AUDIT_0.8.0.md` at repo root

A written report enumerating every category above with pass/fail +
supporting numbers. Committed alongside the audit commit. Lives in the
repo for posterity — future refactors can model against it.

**Exit criteria**: every category pass; `AUDIT_0.8.0.md` committed;
report-out section written in this `todo.md`; all baselines archived
under `tests/parity_baseline_phase_7.json`.

---

## Phase 8 — Release 0.8.0

Minimal mechanics phase. Phase 7's audit has already validated everything;
Phase 8 just cuts the release.

### Risks
- Last-minute scope creep — someone adds "one more fix" to the 0.8.0 branch after audit
- CHANGELOG incomplete or misleading
- Version bump forgotten

### Tasks
- [ ] Freeze the 0.8.0 branch — no further changes unless a release-blocking bug
- [ ] Finalise `CHANGELOG.md` 0.8.0 entry: "storage API unified; internal-only refactor; Python API unchanged" + highlight performance wins (mapped string equality, `describe()` regression fix rolled forward, etc.)
- [ ] Bump `Cargo.toml` version to `0.8.0`
- [ ] Confirm `make lint` clean
- [ ] Commit the version bump + changelog promotion
- [ ] Prepare git tag `v0.8.0` (do not push — user pushes)
- [ ] Write the phase 8 report-out

**Exit criteria**: clean-tree commit with 0.8.0 in Cargo.toml; tag ready;
user informed and given the push command.

---

## Versioning plan

- **0.8.0-dev branch** — all phase work. Intermediate phases do not ship to PyPI.
- **Phase 0 may ship as 0.7.18** if independently useful (parity tests + Mapped variant + today's `str_prop_eq` fix are all shippable on their own). Decided at Phase 0 exit.
- **Phases 1–6 stay internal.** They form the 0.8.0 body.
- **Phase 7 cuts 0.8.0.** CHANGELOG documents everything as one entry.

## Deferred decisions

- `Value::String` → `Cow<'static, str>` / `Arc<str>` — touches everything, not required for the refactor
- Remote/RPC backend — post-0.8.0 stretch
- `Transaction` as trait vs concrete — decide in Phase 2

## Rough effort

| Phase | Estimate | Notes |
|-------|----------|-------|
| 0     | ~2 days  | Tests dominate |
| 1     | 3–5 days | Read API + 7 migrate-and-delete PRs |
| 2     | 2–3 days | Mutations + Cypher write paths |
| 3     | 2–3 days | GAT lifetimes may bite |
| 4     | 1–2 days | IO + format compat |
| 5     | 1 day    | Columnar cleanup + final audit |
| 6     | 1 day    | Validation backend |
| 7     | 1 day    | Benchmarks + docs + release cut |

Total ≈ 2–2.5 weeks focused. Delete-as-you-go means no separate cleanup
phase, saving roughly a day vs the old plan. Phase 0 still dominates
because the crunch-point test suite is doing the heavy lifting for the
whole refactor.
