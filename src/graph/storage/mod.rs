//! Storage-backend types and (future) `GraphRead` / `GraphWrite` traits.
//!
//! Anchor for the 0.8.0 storage-architecture refactor (see `todo.md`).
//! The long-term target is a single `GraphRead` trait every backend
//! implements, with the only enum match on
//! [`crate::graph::schema::GraphBackend`] living at the PyO3 boundary.
//!
//! In this increment (Phase 0.2) the wrapper is intentionally thin:
//!
//! - [`MemoryGraph`] is a newtype around `petgraph::StableDiGraph`
//!   that `Deref`s to the inner graph, so existing `g.method()` call
//!   sites compile unchanged.
//! - [`MappedGraph`] is **currently a type alias** for
//!   [`MemoryGraph`]. The two `GraphBackend` variants `Memory` and
//!   `Mapped` hold the same inner type, which lets match arms write
//!   `Self::Memory(g) | Self::Mapped(g) => g.method()` — zero branch
//!   duplication today. The distinguishing behaviour (mapped-mode
//!   spills columnar properties to mmap on build) lives on the
//!   parent `DirGraph` for now.
//!
//! Phase 0.3+ promotes `MappedGraph` to a distinct struct when the
//! two backends' impls need to diverge (e.g. when each owns its own
//! column store). At that point per-backend `GraphRead` impls start
//! making sense. Today they'd be identical boilerplate.
//!
//! Rule for new storage operations: add the method to a trait in this
//! module first, implement per-backend, and delegate from
//! `GraphBackend` — never the other way around.

use crate::datatypes::Value;
use crate::graph::graph_iterators::{GraphEdges, GraphNeighbors, GraphNodeIndices};
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use petgraph::Direction;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────
// GraphRead — unified read interface over storage backends
// ──────────────────────────────────────────────────────────────────────────

/// Read-side interface shared by every storage backend.
///
/// Phase 0.3 seeded this trait with counts + single-property reads +
/// zero-alloc string equality. Phase 1 expands it to cover iteration,
/// neighbour lookup, backend-kind predicates, and the disk-only helpers
/// the hot Cypher path needs (peer histograms, edge-count caches,
/// connection-type source-node indexes). After Phase 1 every read-path
/// file migrates off inherent `GraphBackend::*` methods.
///
/// Implemented today for [`crate::graph::schema::GraphBackend`];
/// per-backend impls arrive later when `MappedGraph` stops being a
/// type alias and the three backends start diverging.
///
/// Dispatch guidance for consumers:
/// - **hot loops** — take `&impl GraphRead` so the backend is
///   monomorphised and the dispatch cost disappears
/// - **boundary code / object-safe containers** — take
///   `&dyn GraphRead`, trading the vtable cost for simpler API shapes
///
/// ### Disk-only helpers
///
/// Methods such as [`GraphRead::sources_for_conn_type_bounded`],
/// [`GraphRead::lookup_peer_counts`], and [`GraphRead::iter_peers_filtered`]
/// have meaningful implementations only on the disk backend (they read
/// from persistent indexes built at `.kgl` load time). Memory and mapped
/// backends return `None` / fall back via `edges_directed`. The
/// `Option` / fallback contract is preserved from the pre-refactor
/// inherent methods so callers do not need to change their handling.
#[allow(dead_code)] // phase-1 consumers migrate file-by-file after this PR
pub trait GraphRead {
    // ─────────────── counts / backend identity ───────────────

    /// Total live node count across all types.
    fn node_count(&self) -> usize;

    /// Total live edge count.
    fn edge_count(&self) -> usize;

    /// Upper bound on node indices (petgraph `node_bound`). May exceed
    /// [`GraphRead::node_count`] when nodes have been removed from a
    /// `StableDiGraph` without vacuuming.
    fn node_bound(&self) -> usize;

    /// `true` for heap-resident [`GraphBackend::Memory`].
    fn is_memory(&self) -> bool;

    /// `true` for the mmap-Columnar [`GraphBackend::Mapped`] variant.
    fn is_mapped(&self) -> bool;

    /// `true` for disk-backed [`GraphBackend::Disk`] (CSR + mmap columns).
    fn is_disk(&self) -> bool;

    // ─────────────── per-node reads ───────────────

    /// Node type key for a given index. `None` if the node has been removed.
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey>;

    /// Borrow the full NodeData. **Escape hatch** — prefer granular reads
    /// ([`GraphRead::get_node_property`], [`GraphRead::get_node_id`], etc.)
    /// in hot loops. On the disk backend, materialises NodeData through
    /// the per-query arena, which is cheap per-call but accumulates if
    /// called many times without [`GraphRead::reset_arenas`].
    fn node_data(&self, idx: NodeIndex) -> Option<&NodeData>;

    /// Read a single property without full NodeData materialisation.
    /// Used by the hot WHERE-scan path. Returns `None` if the property
    /// is missing or set to `Value::Null`.
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value>;

    /// Read the node id (handles mapped-mode sentinel values).
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value>;

    /// Read the node title (handles mapped-mode sentinel values).
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value>;

    /// Zero-allocation string-equality check for a property against `target`.
    /// Skips the `Value::String(owned)` materialisation that `get_node_property`
    /// would do on mapped graphs. Used by the Cypher executor to short-circuit
    /// `WHERE n.strProp = 'literal'` scans.
    ///
    /// Returns:
    /// - `None` — property is missing or null for this row
    /// - `Some(true)` — stored value equals `target` byte-for-byte
    /// - `Some(false)` — stored value is present but differs
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool>;

    // ─────────────── iteration ───────────────

    /// Iterator over all live node indices.
    fn node_indices(&self) -> GraphNodeIndices<'_>;

    // ─────────────── per-node edges / neighbours ───────────────

    /// Directed edges incident to `idx` (yielded as [`GraphEdgeRef`]).
    fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> GraphEdges<'_>;

    /// Like [`GraphRead::edges_directed`] but the disk backend can
    /// pre-filter by connection type, skipping EdgeData materialisation
    /// for non-matching edges. Memory/mapped callers still post-filter.
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> GraphEdges<'_>;

    /// `(source, target)` endpoints for an edge, without materialising
    /// EdgeData. `None` if the edge has been removed.
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)>;

    /// Iterate edge endpoint metadata without materialising EdgeData.
    /// Yields `(source, target, connection_type)` for every live edge.
    /// On the disk backend this reads mmap'd `edge_endpoints` directly
    /// (zero heap allocation per edge).
    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a>;

    /// Neighbours reached via an edge in `dir`.
    fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> GraphNeighbors<'_>;

    /// Neighbours reached via an edge in either direction.
    fn neighbors_undirected(&self, idx: NodeIndex) -> GraphNeighbors<'_>;

    // ─────────────── disk-only helpers (Option / fallback contract) ─────

    /// Source nodes with outgoing edges of a given connection type,
    /// read from the disk inverted index. `None` on memory/mapped or on
    /// older disk graphs without this index.
    ///
    /// `max` caps the number of sources returned to avoid eager
    /// allocations when the pattern executor will truncate downstream.
    fn sources_for_conn_type_bounded(
        &self,
        _conn_type: InternedKey,
        _max: Option<usize>,
    ) -> Option<Vec<u32>> {
        None
    }

    /// Per-peer edge count for a connection type, read from the
    /// histogram cache on the disk backend. `None` on memory/mapped or
    /// on older disk graphs (caller falls back to
    /// [`GraphRead::count_edges_grouped_by_peer`]).
    fn lookup_peer_counts(&self, _conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        None
    }

    /// Count edges of a connection type grouped by peer node, via a full
    /// scan. Every backend implements this — disk uses sequential CSR
    /// I/O; memory/mapped iterate petgraph edges.
    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: Direction,
        deadline: Option<Instant>,
    ) -> Result<HashMap<u32, i64>, String>;

    /// Count edges from/to `node` matching optional connection-type and
    /// peer-node-type filters. On disk uses sorted-CSR binary search
    /// (O(log D + matching)); on memory/mapped iterates without
    /// allocation.
    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<Instant>,
    ) -> Result<usize, String>;

    /// Peer-iteration fast path used by the Cypher edge-no-variable
    /// optimisation. Yields `(peer, edge_idx)` pairs **without**
    /// materialising EdgeData — on disk this halves I/O on Wikidata-scale
    /// graphs.
    ///
    /// Default implementation falls back to [`GraphRead::edges_directed`]
    /// + post-filter. The disk backend overrides with a direct CSR walk.
    fn iter_peers_filtered<'a>(
        &'a self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<u64>,
    ) -> Box<dyn Iterator<Item = (NodeIndex, EdgeIndex)> + 'a> {
        let iter = self.edges_directed(node, dir).filter_map(move |er| {
            if let Some(want) = conn_type {
                if er.weight().connection_type.as_u64() != want {
                    return None;
                }
            }
            let peer = match dir {
                Direction::Outgoing => er.target(),
                Direction::Incoming => er.source(),
            };
            Some((peer, er.id()))
        });
        Box::new(iter)
    }

    /// Reset per-query materialisation arenas. No-op on memory/mapped;
    /// frees NodeData / EdgeData allocated during the previous query on
    /// the disk backend. Called between Cypher queries to cap memory.
    fn reset_arenas(&self) {}
}

// ──────────────────────────────────────────────────────────────────────────
// GraphWrite — unified mutation interface over storage backends
// ──────────────────────────────────────────────────────────────────────────

/// Write-side interface shared by every storage backend.
///
/// Phase 2 of the 0.8.0 refactor (see `todo.md`). Pulls together the
/// mutation methods that were inherent on
/// [`crate::graph::schema::GraphBackend`] so write-path files can
/// dispatch through the trait instead of matching on the backend
/// variant.
///
/// Transaction bookkeeping (OCC `version`, `read_only`,
/// `schema_locked`) lives on [`crate::graph::schema::DirGraph`], not
/// on this trait — no backend has its own OCC state, and validation
/// against the schema metadata sits architecturally above storage.
/// Documented decision: keep transactions on DirGraph.
///
/// Dispatch guidance matches [`GraphRead`]: `&mut impl GraphWrite`
/// in hot mutation loops, `&mut dyn GraphWrite` at wider API
/// boundaries.
#[allow(dead_code)] // phase-2 consumers migrate file-by-file after this PR
pub trait GraphWrite: GraphRead {
    /// Mutable borrow of the full NodeData. Escape hatch for property
    /// mutation — prefer higher-level helpers (`NodeData::set_property`,
    /// `NodeData::remove_property`) where available.
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData>;

    /// Mutable borrow of the full EdgeData.
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData>;

    /// Insert a new node, returning its assigned index.
    fn add_node(&mut self, data: NodeData) -> NodeIndex;

    /// Remove a node, returning its NodeData if present. On the disk
    /// backend this writes a tombstone; on memory/mapped the
    /// StableDiGraph entry is removed in-place.
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData>;

    /// Insert a directed edge from `a` to `b`.
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex;

    /// Remove an edge, returning its EdgeData if present.
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData>;

    /// Disk-only: after a columnar-properties row is materialised for a
    /// newly-added node, persist the per-type `row_id` back to the
    /// disk slot so subsequent reads find the correct columnar row.
    /// No-op on memory/mapped (their slot storage carries no separate
    /// row_id field). Invariant: callers must invoke this only after
    /// they have already assigned `PropertyStorage::Columnar { row_id }`
    /// to the node's `NodeData`; otherwise disk reads will drift.
    fn update_row_id(&mut self, _node_idx: NodeIndex, _row_id: u32) {}
}

// ──────────────────────────────────────────────────────────────────────────
// Newtype backends
// ──────────────────────────────────────────────────────────────────────────

/// Heap-resident in-memory graph backend. Wraps `StableDiGraph` and
/// `Deref`s to it so existing petgraph call sites compile unchanged.
#[derive(Clone, Debug, Default)]
pub struct MemoryGraph(pub(crate) StableDiGraph<NodeData, EdgeData>);

/// Memory-mapped in-memory graph backend — *currently* a type alias
/// for [`MemoryGraph`]. See module docs.
pub type MappedGraph = MemoryGraph;

impl MemoryGraph {
    #[inline]
    pub fn new() -> Self {
        Self(StableDiGraph::new())
    }

    #[inline]
    #[allow(dead_code)]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self(StableDiGraph::with_capacity(nodes, edges))
    }
}

impl Deref for MemoryGraph {
    type Target = StableDiGraph<NodeData, EdgeData>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MemoryGraph {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Serialize as the inner StableDiGraph so the on-disk binary format
// is unchanged between this refactor and pre-refactor code.
impl serde::Serialize for MemoryGraph {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(ser)
    }
}

impl<'de> serde::Deserialize<'de> for MemoryGraph {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        StableDiGraph::deserialize(de).map(MemoryGraph)
    }
}
