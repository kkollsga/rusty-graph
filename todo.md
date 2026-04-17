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

### Clean-break rules (aggressive — applied every PR, not deferred)
- **Delete-as-you-go.** The PR that migrates a file to the trait is the same PR that deletes the now-unreachable enum-match code. No follow-up cleanup phase.
- **No deprecated shims, ever.** When a function is obsoleted, delete it in the same commit. No `#[deprecated]` forwarders, no re-exports, no "kept for compat" comments.
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

## Phase 1 — Complete the read API 🔜

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
- [ ] `nodes_of_type` iteration order determinism — requires the trait method, authored here
- [ ] Property + edge iteration interleaved under one borrow — compiles and runs

### Tasks
- [ ] Trait additions: `nodes_of_type` / `edges_from` / `edges_to` / `neighbors_out` / `neighbors_in` / `degree` / metadata accessors / type-connectivity / edge-count caches
- [ ] GAT vs boxed-iterator decision per method, documented inline
- [ ] **Migrate-and-delete** one file per PR — old enum-match code removed same commit:
  - [ ] `pattern_matching.rs`
  - [ ] `introspection.rs`
  - [ ] `cypher/executor.rs` (read paths)
  - [ ] `cypher/planner.rs`
  - [ ] `data_retrieval.rs`
  - [ ] `statistics_methods.rs`
  - [ ] `graph_algorithms.rs` (read paths)
- [ ] Per-PR grep gate: `rg 'GraphBackend::[A-Z]'` in the touched file → 0 hits

**Exit criteria**: testing gate passes; every read-path file migrated AND cleaned; in-memory benchmark delta < 2 %.

---

## Phase 2 — Mutation API

### Risks
- `&mut dyn GraphWrite` lifetime pain across batch operations.
- Schema locking (v0.7.7) — each backend handles differently today; trait must unify without weakening security.
- Cypher MERGE upsert semantics — subtle, easy to diverge across modes.
- Tombstone-then-traverse — disk CSR is immutable + vacuum-rebuilt, memory mutates in-place. Divergence = correctness bug.
- Batch atomicity on mid-batch failure must be identical.
- `rebuild_caches` per backend — any divergence corrupts subsequent reads.

### Crunch-point tests
- [x] Conflict-handling matrix (Phase 0)
- [x] Mid-batch failure semantics (Phase 0)
- [x] Schema-locked rejection parity (Phase 0)
- [x] Tombstone visibility + traversal (Phase 0)
- [ ] MERGE idempotency — same MERGE twice is a no-op with identical stats (authored here)
- [ ] Concurrent mutation + read — long-running MATCH during mutation; snapshot isolation matches across modes

### Tasks
- [ ] `trait GraphWrite: GraphRead`
- [ ] Implement for each backend; preserve OCC + read-only txns + schema locking
- [ ] **Migrate-and-delete** write-path files + Cypher CREATE / SET / DELETE / MERGE paths
- [ ] Transaction boundary decision (trait or concrete type) documented

**Exit criteria**: testing gate passes; every write-path file migrated AND cleaned; mutation benchmarks within thresholds.

---

## Phase 3 — Traversal / iteration API (GAT-heavy)

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
- [ ] Traversal-under-mutation isolation — long MATCH, concurrent CREATE; MATCH sees its starting snapshot
- [ ] Inlining guard — micro-benchmark of innermost traversal loop, fails if slower than Phase 2 baseline

### Tasks
- [ ] `trait GraphTraverse: GraphRead` with GAT iterators
- [ ] Generic helpers (BfsIter, ShortestPathIter) over `&impl GraphTraverse`
- [ ] **Migrate-and-delete** algorithm code, fluent API internals, pattern-matching multi-hop expansion

**Exit criteria**: testing gate passes; traversal code no longer touches `GraphBackend` directly; old traversal helpers deleted.

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
- [x] Save RSS ceiling (Phase 0)
- [ ] Byte-level `.kgl` v3 golden hash — format unchanged after Phase 4 migration

### Tasks
- [ ] `trait GraphIo` — save / load / save_incremental / load_incremental / format version enum
- [ ] **Migrate-and-delete** `io_operations.rs`, ntriples loader, CSV loader
- [ ] Golden-hash test against checked-in digests

**Exit criteria**: testing gate passes; byte-level golden hash test for each format version; io code backend-agnostic where the format allows.

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
