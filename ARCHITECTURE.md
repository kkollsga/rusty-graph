# KGLite — architecture (as of Phase 0 of the 0.8.0 refactor)

This document is the living spec for how storage is layered in kglite.
It's updated as each phase of the 0.8.0 refactor lands. See `todo.md`
for the full plan.

## TL;DR

- kglite offers three storage modes — `memory`, `mapped`, `disk` — that all speak the same Python API
- internally they share most code through a `GraphRead` trait that's gradually absorbing storage-touching surface area
- the PyO3 boundary is the only place that pattern-matches on the concrete backend enum; internal consumers talk to the trait
- the core product is **memory** mode. `mapped` is the 1M–30M niche where RAM is tight; `disk` is Wikidata-scale (100M+)

## Current storage layers

```
┌──────────────────────────────────────────────────────────────────┐
│ PyO3 boundary (src/graph/mod.rs — KnowledgeGraph #[pymethods])   │
│   Dispatches on storage="..." at construction                    │
└───────┬───────────────────────┬───────────────────────┬──────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│ GraphBackend::   │   │ GraphBackend::   │   │ GraphBackend::     │
│ Memory(          │   │ Mapped(          │   │ Disk(              │
│   MemoryGraph)   │   │   MappedGraph    │   │   Box<DiskGraph>)  │
│                  │   │   = MemoryGraph  │   │                    │
│                  │   │   (alias today)  │   │                    │
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

### Trait layer (Phase 0.3 → Phase 2)

`GraphRead` (Phase 0.3, expanded in Phase 1) covers reads — counts,
per-node property access, iteration, neighbour lookup, backend
identity, and the disk-only helpers the hot Cypher path needs (peer
histograms, edge-count caches, connection-type source-node indexes).

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

Implemented today for `GraphBackend` in `schema.rs`. Per-backend impls
arrive when the `MappedGraph` type alias is promoted to a distinct
struct (still deferred at end of Phase 2 — no write path needed
backend-specific divergence).

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
  in `src/graph/mod.rs` take `&mut self` on the full DirGraph.

This keeps the `storage/` trait layer focused on per-backend data
operations, not cross-cutting bookkeeping.

## Target structure

Destination layout for `src/graph/` at end of 0.8.0 (see `todo.md`
for the full migration plan — files move to their target subdir
during the phase that already touches them):

```
src/graph/
├── mod.rs                 # Re-exports + short module doc. No logic.
├── kg.rs                  # KnowledgeGraph struct + core #[pymethods]
│
├── storage/               # Trait layer + per-backend subfolders
│   ├── mod.rs             # GraphRead / GraphWrite / GraphTraverse traits + GraphBackend enum
│   ├── schema.rs          # Shared schema types
│   ├── interner.rs        # StringInterner + InternedKey
│   ├── memory/            # Heap-resident backend
│   ├── mapped/            # mmap-Columnar backend
│   └── disk/              # CSR + mmap backend
│
├── cypher/                # Unchanged (already a subdir)
├── query/                 # Shared execution (pattern matching, filters, traversal)
├── algorithms/            # PageRank, centrality, components, clustering, vector
├── introspection/         # describe(), schema(), debug, bug_report
├── io/                    # Load / save / ntriples / export
├── features/              # spatial / temporal / timeseries / equations
├── mutation/              # Batch / maintain / validate / subgraph
└── pyapi/                 # #[pymethods] blocks at the edge
```

## Rules for new storage code

1. **Add to the trait first.** A new read operation should be a trait method in `src/graph/storage/mod.rs`, implemented per-backend. Don't add inherent methods to `GraphBackend` and expect consumers to match on the variant — that's the layering we're getting rid of.
2. **Delete as you go.** The PR that introduces the trait-based path is the same PR that deletes the old enum-match code. No `#[deprecated]` shims.
3. **`&impl GraphRead` in hot loops; `&dyn GraphRead` at boundaries.** Monomorphisation for tight scan code (Cypher executor, algorithm inner loops). Trait objects where the API ergonomics matter more than the vtable cost (boundary helpers, collections of heterogeneous graphs).
4. **In-memory performance is sacred.** If an optimisation helps mapped or disk at the cost of memory, find a mode-specific way. Never regress the core product.
5. **No god files.** Soft cap 1500 lines per `.rs`; hard cap 2500 (enforced in Phase 7). `mod.rs` files are re-exports + module docs only — no `impl` blocks, no functions > 20 lines.

## Open questions tracked in `todo.md`

- When does `MappedGraph` stop being a type alias for `MemoryGraph`?
  → Phase 1 if the backends need distinct trait impls; later if they don't
- Does `Value::String` become `Cow<'static, str>` / `Arc<str>`?
  → Deferred; touches everything, not required for the refactor
- Does `Transaction` become a trait?
  → Decided in Phase 2: **no** — transactions stay on `DirGraph` (see
    "Transactions stay on DirGraph" above)
