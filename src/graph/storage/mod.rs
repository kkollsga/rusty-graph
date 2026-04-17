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

use crate::graph::schema::{EdgeData, NodeData};
use petgraph::stable_graph::StableDiGraph;
use std::ops::{Deref, DerefMut};

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
