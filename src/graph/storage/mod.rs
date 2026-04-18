//! Storage-backend types and `GraphRead` / `GraphWrite` traits.
//!
//! Anchor for the 0.8.0 storage-architecture refactor.
//! Every backend implements [`GraphRead`] / [`GraphWrite`] directly;
//! the [`crate::graph::schema::GraphBackend`] enum is a dumb dispatcher.
//! Per-backend trait impls live in [`crate::graph::storage::impls`].
//!
//! - [`MemoryGraph`] — heap-resident, petgraph `StableDiGraph`.
//! - [`MappedGraph`] — mmap-columnar-spill variant (Phase 5 promoted
//!   this from a type alias to a distinct struct so its trait impls
//!   can diverge from memory's once the column ownership differs).
//! - [`crate::graph::storage::disk::disk_graph::DiskGraph`] — CSR + mmap
//!   columns.
//!
//! Rule for new storage operations: add the method to [`GraphRead`] or
//! [`GraphWrite`] first, implement per-backend, and let the
//! `GraphBackend` dispatcher route to them — never the other way.

pub mod backend;
pub mod disk;
pub mod interner;
pub mod lookups;
pub mod mapped;
pub mod memory;
pub mod type_build_meta;

use crate::datatypes::Value;
use crate::graph::core::graph_iterators::GraphEdgeRef;
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
/// Phase 0.3 seeded this trait with counts + single-property reads.
/// Phase 1 expanded it to cover iteration, neighbour lookup, backend-kind
/// predicates, and disk-only helpers. Phase 3 converted iterator-returning
/// methods to GATs (associated types with lifetime parameters) and promoted
/// the remaining inherent edge accessors (`edges`, `edge_references`,
/// `edge_weight`, `edge_indices`, `find_edge`, `edges_connecting`,
/// `edge_weights`) onto the trait.
///
/// Implemented today for [`crate::graph::schema::GraphBackend`];
/// per-backend impls (`MemoryGraph`, `DiskGraph`) will land in Phase 5
/// alongside the columnar cleanup that lets them diverge meaningfully.
///
/// ### GATs and object-safety
///
/// The iterator methods use generic associated types (e.g.
/// [`GraphRead::EdgesIter`]). This makes the trait **non-object-safe**:
/// `&dyn GraphRead` does not compile. All consumers take `&impl GraphRead`
/// (monomorphised) instead. Two methods that need type erasure for
/// backend-specific fast paths (`iter_peers_filtered`, `edge_endpoint_keys`)
/// return `Box<dyn Iterator<…> + 'a>` explicitly and stay non-GAT; they
/// would otherwise require a second associated type per method.
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
    // ─────────────── generic associated types ───────────────

    /// Iterator over all live node indices.
    type NodeIndicesIter<'a>: Iterator<Item = NodeIndex>
    where
        Self: 'a;

    /// Iterator over all live edge indices.
    type EdgeIndicesIter<'a>: Iterator<Item = EdgeIndex>
    where
        Self: 'a;

    /// Iterator over edges incident to a node (directed).
    type EdgesIter<'a>: Iterator<Item = GraphEdgeRef<'a>>
    where
        Self: 'a;

    /// Iterator over all edges in the graph (yielded as `GraphEdgeRef`).
    type EdgeReferencesIter<'a>: Iterator<Item = GraphEdgeRef<'a>>
    where
        Self: 'a;

    /// Iterator over edges connecting a given pair of nodes.
    type EdgesConnectingIter<'a>: Iterator<Item = GraphEdgeRef<'a>>
    where
        Self: 'a;

    /// Iterator over neighbour node indices.
    type NeighborsIter<'a>: Iterator<Item = NodeIndex>
    where
        Self: 'a;
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
    fn is_memory(&self) -> bool {
        false
    }

    /// `true` for the mmap-Columnar [`GraphBackend::Mapped`] variant.
    fn is_mapped(&self) -> bool {
        false
    }

    /// `true` for disk-backed [`GraphBackend::Disk`] (CSR + mmap columns).
    fn is_disk(&self) -> bool {
        false
    }

    // ─────────────── per-node reads ───────────────

    /// Node type key for a given index. `None` if the node has been removed.
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey>;

    /// Borrow the full NodeData. **Escape hatch** — prefer granular reads
    /// ([`GraphRead::get_node_property`], [`GraphRead::get_node_id`], etc.)
    /// in hot loops. On the disk backend, materialises NodeData through
    /// the per-query arena, which is cheap per-call but accumulates if
    /// called many times without [`GraphRead::reset_arenas`].
    ///
    /// Named `node_weight` for consistency with petgraph's `StableDiGraph`
    /// primitive, which is the heap-backed implementation of this method.
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData>;

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
    fn node_indices(&self) -> Self::NodeIndicesIter<'_>;

    /// Iterator over all live edge indices.
    fn edge_indices(&self) -> Self::EdgeIndicesIter<'_>;

    /// Iterator over every live edge in the graph, yielding
    /// [`GraphEdgeRef`] with materialised `EdgeData`.
    fn edge_references(&self) -> Self::EdgeReferencesIter<'_>;

    /// Iterator over every live edge's weight (EdgeData). Boxed because
    /// petgraph's underlying `edge_weights` returns an opaque
    /// `impl Iterator` that can't be named as a GAT associated type.
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a>;

    // ─────────────── per-node edges / neighbours ───────────────

    /// Directed edges incident to `idx` (yielded as [`GraphEdgeRef`]).
    fn edges_directed(&self, idx: NodeIndex, dir: Direction) -> Self::EdgesIter<'_>;

    /// Default-direction edges (outgoing) incident to `idx` — matches
    /// petgraph's `StableDiGraph::edges`.
    fn edges(&self, idx: NodeIndex) -> Self::EdgesIter<'_>;

    /// Like [`GraphRead::edges_directed`] but the disk backend can
    /// pre-filter by connection type, skipping EdgeData materialisation
    /// for non-matching edges. Memory/mapped callers still post-filter.
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> Self::EdgesIter<'_>;

    /// Iterator over edges directly connecting `a` → `b`.
    fn edges_connecting(&self, a: NodeIndex, b: NodeIndex) -> Self::EdgesConnectingIter<'_>;

    /// Borrow a single edge's weight.
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData>;

    /// First edge index from `a` to `b`, if one exists.
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex>;

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
    fn neighbors_directed(&self, idx: NodeIndex, dir: Direction) -> Self::NeighborsIter<'_>;

    /// Neighbours reached via an edge in either direction.
    fn neighbors_undirected(&self, idx: NodeIndex) -> Self::NeighborsIter<'_>;

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
/// Phase 2 of the 0.8.0 refactor. Pulls together the
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
/// Dispatch guidance: `&mut impl GraphWrite` everywhere. Because
/// `GraphWrite: GraphRead` and `GraphRead` is non-object-safe (GAT
/// iterators — see [`GraphRead`] docs), `&mut dyn GraphWrite` also
/// does not compile. All mutation consumers take `&mut impl GraphWrite`.
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

/// Memory-mapped in-memory graph backend — Phase 5 promoted this to a
/// distinct struct (previously a type alias for [`MemoryGraph`]) so
/// per-backend trait impls can diverge. Today its shape matches
/// `MemoryGraph` but the `GraphBackend::Mapped` variant carries its
/// own identity, letting the column-store ownership split cleanly.
#[derive(Clone, Debug, Default)]
pub struct MappedGraph(pub(crate) StableDiGraph<NodeData, EdgeData>);

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

    /// Borrow the inner `StableDiGraph`. Shared with [`MappedGraph`]
    /// for match arms that need the heap backend's petgraph view.
    #[inline]
    pub fn inner(&self) -> &StableDiGraph<NodeData, EdgeData> {
        &self.0
    }

    /// Mutable borrow of the inner `StableDiGraph`.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut StableDiGraph<NodeData, EdgeData> {
        &mut self.0
    }
}

impl MappedGraph {
    #[inline]
    pub fn new() -> Self {
        Self(StableDiGraph::new())
    }

    #[inline]
    #[allow(dead_code)]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self(StableDiGraph::with_capacity(nodes, edges))
    }

    /// Borrow the inner `StableDiGraph`. Shared with [`MemoryGraph`]
    /// for match arms that need the heap backend's petgraph view.
    #[inline]
    pub fn inner(&self) -> &StableDiGraph<NodeData, EdgeData> {
        &self.0
    }

    /// Mutable borrow of the inner `StableDiGraph`.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut StableDiGraph<NodeData, EdgeData> {
        &mut self.0
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

impl Deref for MappedGraph {
    type Target = StableDiGraph<NodeData, EdgeData>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MappedGraph {
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

impl serde::Serialize for MappedGraph {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(ser)
    }
}

impl<'de> serde::Deserialize<'de> for MappedGraph {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        StableDiGraph::deserialize(de).map(MappedGraph)
    }
}

pub mod impls;
pub mod recording;

// Phase-6 recording backend — re-exported so downstream consumers (and
// the Phase-6 parity test) can construct it without reaching into
// `storage::recording::`. DO NOT REMOVE despite unused-import warnings;
// `tests/test_phase6_parity.py::test_recording_graph_symbol_exported`
// asserts this exact line survives.
#[allow(unused_imports)]
pub use recording::RecordingGraph;
