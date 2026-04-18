# KGLite — architecture (as of end of Phase 11 of the 0.8.0 refactor)

This document is the living spec for how storage is layered in kglite.
It's updated as each phase of the 0.8.0 refactor lands. See `todo.md`
for the full plan.

The source tree under `src/graph/` has three layering rules: (a) every
`#[pymethods] impl` block lives under `src/graph/pyapi/`, (b) query
languages live under `src/graph/languages/` (peers: `cypher` and
`fluent`), and (c) shared query primitives live under `src/graph/core/`.
Every `.rs` is ≤ 2,500 lines (hard cap, enforced by
`tests/test_phase7_parity.py::test_god_file_gate`); `mod.rs` files are
re-exports + module docs only.

## TL;DR

- kglite offers three storage modes — `memory`, `mapped`, `disk` — that all speak the same Python API
- internally they share most code through a `GraphRead` trait that's gradually absorbing storage-touching surface area
- the PyO3 boundary is the only place that pattern-matches on the concrete backend enum; internal consumers talk to the trait
- the core product is **memory** mode. `mapped` is the 1M–30M niche where RAM is tight; `disk` is Wikidata-scale (100M+)

## Current storage layers

```
┌──────────────────────────────────────────────────────────────────┐
│ PyO3 boundary (src/graph/pyapi/kg_methods.rs + transaction.rs +  │
│                result_view.rs + pymethods_*.rs)                  │
│   Dispatches on storage="..." at construction                    │
└───────┬───────────────────────┬───────────────────────┬──────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│ GraphBackend::   │   │ GraphBackend::   │   │ GraphBackend::     │
│ Memory(          │   │ Mapped(          │   │ Disk(              │
│   MemoryGraph)   │   │   MappedGraph)   │   │   Box<DiskGraph>)  │
│                  │   │   (distinct      │   │                    │
│                  │   │    struct, P5)   │   │                    │
└────────┬─────────┘   └────────┬─────────┘   └──────────┬─────────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│ petgraph         │   │ petgraph         │   │ CSR edges +        │
│ StableDiGraph    │   │ StableDiGraph    │   │ mmap'd columns     │
│                  │   │                  │   │                    │
│ NodeData.        │   │ NodeData.        │   │ DiskGraph owns     │
│ properties →     │   │ properties →     │   │ its own property   │
│ PropertyStorage  │   │ PropertyStorage  │   │ storage (separate  │
│ (Map | Compact   │   │ ::Columnar with  │   │ from the petgraph  │
│  | heap-Col.)    │   │ mmap_store set   │   │ path entirely)     │
└──────────────────┘   └──────────────────┘   └────────────────────┘
```

### Trait layer (Phase 0.3 → Phase 3)

`GraphRead` (Phase 0.3, expanded in Phase 1, GAT-converted in Phase 3)
covers reads — counts, per-node property access, iteration, neighbour
lookup, backend identity, edge-level accessors (edges, edge_references,
edge_weight, edge_indices, find_edge, edges_connecting, edge_weights),
and the disk-only helpers the hot Cypher path needs (peer histograms,
edge-count caches, connection-type source-node indexes).

**Iterator methods use GATs.** Phase 3 replaced the concrete return
types of `node_indices`, `edges_directed`, `neighbors_directed`,
`edge_references`, `edges`, `edge_indices`, and `edges_connecting` with
generic associated types (`Self::NodeIndicesIter<'_>`, etc.). This
makes the trait non-object-safe — `&dyn GraphRead` no longer compiles.
The payoff lands in Phase 5, when per-backend `impl GraphRead for
MemoryGraph` and `impl GraphRead for DiskGraph` return their native
iterator types without going through the `GraphBackend` enum match.

`GraphWrite: GraphRead` (Phase 2) covers mutations:

```rust
pub trait GraphWrite: GraphRead {
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData>;
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData>;
    fn add_node(&mut self, data: NodeData) -> NodeIndex;
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData>;
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex;
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData>;

    /// Disk-only: persist a new columnar row_id back to the slot.
    /// No-op on memory/mapped.
    fn update_row_id(&mut self, _node_idx: NodeIndex, _row_id: u32) {}
}
```

Implemented today for `GraphBackend` in `schema.rs` as a 4-arm
dispatcher (`Memory` / `Mapped` / `Disk` / `Recording`). Per-backend
`impl GraphRead` / `impl GraphWrite` land in `src/graph/storage/impls.rs`
(Phase 5) — `MappedGraph` was promoted from a type alias to a distinct
struct at the same time, and each backend now owns its own trait impl.

### Recording backend (Phase 6)

`GraphBackend::Recording(Box<RecordingGraph<GraphBackend>>)` is a
validation wrapper that forwards every `GraphRead` call to an inner
backend while logging the method name. Rust-only (no Python
constructor reaches it); the cross-mode parity matrix for it lives in
`src/graph/storage/recording.rs::tests`. Proves the refactor is
actually open/closed — adding a new backend is a 3-src-file change
(own file, the enum, the dispatcher).

### Transactions stay on DirGraph (Phase 2 decision)

Transactions (OCC `version` counter, `read_only` / `schema_locked`
flags) live on the concrete `DirGraph`, not on any trait. Rationale:

- No backend has per-backend OCC state; `version` is a `DirGraph`
  field incremented by mutations.
- `schema_locked` and validation helpers in `schema_validation.rs`
  operate on DirGraph metadata maps (`node_type_metadata`,
  `type_schemas`, `schema_definition`) — none of which belong to
  `GraphBackend`.
- Adding a `GraphTransaction` trait would only duplicate what
  `DirGraph::begin` / `commit` already provide; PyO3 txn boundaries
  in `src/graph/pyapi/transaction.rs` take `&mut self` on the full DirGraph.

This keeps the `storage/` trait layer focused on per-backend data
operations, not cross-cutting bookkeeping.

## Target structure

Destination layout for `src/graph/` after Phase 8 (updated 2026-04-17):

```
src/graph/
├── mod.rs                 # Struct defs, private helpers, CoW selection utilities. No #[pymethods].
│
├── storage/               # Trait layer + per-backend subfolders
│   ├── mod.rs             # GraphRead / GraphWrite traits + GraphBackend enum
│   ├── interner.rs        # StringInterner + InternedKey
│   ├── impls.rs           # Shared trait-impl helpers
│   ├── lookups.rs
│   ├── recording.rs       # RecordingGraph<G> validation wrapper
│   ├── type_build_meta.rs
│   ├── memory/            # Heap-resident backend
│   ├── mapped/            # mmap-Columnar backend
│   └── disk/              # CSR + mmap backend
│
├── core/                  # Shared query primitives
│   ├── pattern_matching/       # Pattern AST + parser + PatternExecutor
│   │   ├── mod.rs
│   │   ├── pattern.rs
│   │   ├── parser.rs
│   │   └── matcher.rs
│   ├── filtering_methods.rs
│   ├── traversal_methods.rs
│   ├── graph_iterators.rs
│   ├── data_retrieval.rs
│   ├── calculations.rs
│   ├── value_operations.rs
│   └── statistics_methods.rs
│
├── languages/             # Query-interface languages
│   ├── mod.rs
│   ├── cypher/            # parser/, planner/, executor/, window.rs, result_view.rs
│   └── fluent/            # Scaffolding only — chain state lives in pyapi/kg_fluent.rs
│                          # (reserved for future Rust-side extraction of a second
│                          # peer language like SPARQL or GraphQL).
│
├── algorithms/            # PageRank, centrality, components, clustering, vector
├── introspection/         # describe(), schema(), debug, bug_report (split into several submodules)
├── io/                    # Load / save / ntriples / export
├── features/              # spatial / temporal / timeseries / equations
├── mutation/              # Batch / maintain / validate / subgraph
│
└── pyapi/                 # ALL #[pymethods] impl blocks live here
    ├── mod.rs             # Module registration
    ├── kg_core.rs         # KnowledgeGraph: construction, properties, cypher()
    ├── kg_mutation.rs     # KnowledgeGraph: add_nodes, add_connections, conflict handling
    ├── kg_introspection.rs  # KnowledgeGraph: describe, schema, find, source, context
    ├── kg_fluent.rs       # KnowledgeGraph: select/where/traverse chain methods
    ├── transaction.rs     # Transaction #[pyclass]
    ├── result_view.rs     # ResultView + ResultIter #[pyclass]
    ├── pymethods_algorithms.rs
    ├── pymethods_export.rs
    ├── pymethods_indexes.rs
    ├── pymethods_spatial.rs
    ├── pymethods_timeseries.rs
    └── pymethods_vector.rs
```

## Rules for new storage code

1. **Add to the trait first.** A new read operation should be a trait method in `src/graph/storage/mod.rs`, implemented per-backend. Don't add inherent methods to `GraphBackend` and expect consumers to match on the variant — that's the layering we're getting rid of.
2. **Delete as you go.** The PR that introduces the trait-based path is the same PR that deletes the old enum-match code. No `#[deprecated]` shims.
3. **`&impl GraphRead` everywhere; no `&dyn GraphRead`.** Phase 3 added GATs to every iterator method on `GraphRead`, which makes the trait non-object-safe. All consumers take `&impl GraphRead` (monomorphised). Two methods that need type erasure for backend-specific fast paths (`iter_peers_filtered`, `edge_endpoint_keys`) explicitly return `Box<dyn Iterator<…> + 'a>` and stay non-GAT.
4. **Iterator-returning methods must use GATs.** When adding a new iteration point to `GraphRead`, declare an associated type (`type FooIter<'a>: Iterator<Item = …> where Self: 'a;`) and return `Self::FooIter<'_>` from the method. This sets up the trait for future per-backend impls (Phase 5) without requiring call-site rewrites.
5. **In-memory performance is sacred.** If an optimisation helps mapped or disk at the cost of memory, find a mode-specific way. Never regress the core product.
6. **No god files.** Soft cap 1500 lines per `.rs`; hard cap 2500 (enforced in Phase 7). `mod.rs` files are re-exports + module docs only — no `impl` blocks, no functions > 20 lines.

## Open questions tracked in `todo.md`

- Does `Value::String` become `Cow<'static, str>` / `Arc<str>`?
  → Deferred to post-0.8.0; touches every value-creation site.
- Does `Transaction` become a trait?
  → Decided in Phase 2: **no** — transactions stay on `DirGraph` (see
    "Transactions stay on DirGraph" above).
- When does `languages/fluent/` get a Rust-side implementation?
  → When a second peer language appears, or when a non-Python fluent
    caller materialises. Today `kg_fluent.rs` in `pyapi/` is the single
    implementation site.
