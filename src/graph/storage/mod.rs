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
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;
use std::ops::{Deref, DerefMut};

// ──────────────────────────────────────────────────────────────────────────
// GraphRead — unified read interface over storage backends
// ──────────────────────────────────────────────────────────────────────────

/// Read-side interface shared by every storage backend.
#[allow(dead_code)] // read methods land incrementally across Phases 0.3 → 1
///
/// Phase 0.3 introduces this trait with a minimal surface (counts +
/// single-property reads + zero-alloc string equality) and migrates
/// two proof-of-concept call sites. Phase 1 expands to cover edge
/// iteration, neighbour lookup, and schema metadata — at which point
/// read-heavy files (pattern_matching, introspection, cypher/executor,
/// graph_algorithms) all migrate to `&impl GraphRead`.
///
/// Implemented today for [`crate::graph::schema::GraphBackend`];
/// per-backend impls arrive in Phase 1 when `MappedGraph` stops being
/// a type alias and the three backends start diverging.
///
/// Dispatch guidance for consumers:
/// - **hot loops** — take `&impl GraphRead` so the backend is
///   monomorphised and the dispatch cost disappears
/// - **boundary code / object-safe containers** — take
///   `&dyn GraphRead`, trading the vtable cost for simpler API shapes
pub trait GraphRead {
    /// Total live node count across all types.
    fn node_count(&self) -> usize;

    /// Total live edge count.
    fn edge_count(&self) -> usize;

    /// Node type key for a given index. `None` if the node has been removed.
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey>;

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
