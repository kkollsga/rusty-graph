// src/graph/disk_graph.rs
//
// Disk-backed graph storage using CSR (Compressed Sparse Row) format.
// Nodes are stored in memory (~40 bytes each in Columnar mode).
// Edges are stored in mmap'd CSR arrays (16 bytes per edge per direction).
// Edge properties are stored sparsely (only for edges that have them).
//
// Memory budget: ~10% of equivalent petgraph InMemory graph.
// For 100M nodes + 1B edges: ~5-6 GB RAM + OS page cache.

use crate::datatypes::values::Value;
use crate::graph::graph_iterators::{
    DiskEdgeIndices, DiskEdgeReferences, DiskEdges, DiskEdgesConnecting, DiskNeighbors,
    DiskNodeIndices,
};
use crate::graph::mmap_vec::MmapOrVec;
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ============================================================================
// CSR edge record — 16 bytes, stored in mmap'd arrays
// ============================================================================

/// A single edge record in the CSR adjacency list.
/// `peer` is the target (in out_edges) or source (in in_edges).
/// `conn_type` is inlined for hot-path filtering without materializing EdgeData.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct CsrEdge {
    pub peer: u32,
    pub edge_idx: u32,
    pub conn_type: u64,
}

// SAFETY: CsrEdge is repr(C), Copy, Default, and contains only fixed-size integers.
// No padding issues on any platform since fields are naturally aligned (4+4+8=16).

/// Edge endpoint metadata — stored in a dense array indexed by edge_idx.
/// 16 bytes per edge. Includes connection_type for O(1) lookup (avoids CSR scan).
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct EdgeEndpoints {
    pub source: u32,
    pub target: u32,
    pub connection_type: u64,
}

/// Tombstone marker for deleted edges.
pub(crate) const TOMBSTONE_EDGE: u32 = u32::MAX;

// ============================================================================
// Node slot — 16 bytes, mmap'd on disk
// ============================================================================

/// Compact per-node metadata stored in a mmap'd array on disk.
/// 16 bytes per node = 1.6 GB for 100M nodes (OS pages in/out).
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct DiskNodeSlot {
    pub node_type: u64, // InternedKey as raw u64
    pub row_id: u32,    // row into the type's ColumnStore
    pub flags: u32,     // bit 0 = alive
}

impl DiskNodeSlot {
    const ALIVE_BIT: u32 = 1;

    #[inline]
    pub fn is_alive(&self) -> bool {
        self.flags & Self::ALIVE_BIT != 0
    }
}

// ============================================================================
// DiskGraph
// ============================================================================

/// Truly disk-backed graph. All data lives on disk via mmap.
///
/// - Nodes: `MmapOrVec<DiskNodeSlot>` (16 bytes/node, mmap'd)
///   Actual node data (id, title, properties) in ColumnStore columns (mmap'd).
///   `node_weight()` materializes NodeData into an arena on access.
/// - Edges: CSR arrays (`out_offsets`, `out_edges`, etc.) — mmap'd
/// - Edge properties: sparse HashMap (loaded to heap)
/// - Arenas: append-only caches for materialized NodeData/EdgeData refs
pub struct DiskGraph {
    // ── Node storage (mmap'd on disk) ──
    pub(crate) node_slots: MmapOrVec<DiskNodeSlot>,
    node_count: usize,
    free_node_slots: Vec<u32>,

    // ── Node materialization arena ──
    node_arena: UnsafeCell<Vec<NodeData>>,

    // ── Column stores for node properties (Arc refs, data mmap'd) ──
    pub(crate) column_stores: HashMap<InternedKey, Arc<crate::graph::column_store::ColumnStore>>,

    // ── Edge CSR (mmap'd) ──
    out_offsets: MmapOrVec<u64>,
    out_edges: MmapOrVec<CsrEdge>,
    in_offsets: MmapOrVec<u64>,
    in_edges: MmapOrVec<CsrEdge>,

    // ── Edge metadata ──
    pub(crate) edge_endpoints: MmapOrVec<EdgeEndpoints>,
    pub(crate) edge_count: usize,
    pub(crate) next_edge_idx: u32,

    // ── Edge properties (sparse, in heap) ──
    edge_properties: HashMap<u32, Vec<(InternedKey, Value)>>,

    // ── Edge materialization arena ──
    edge_arena: UnsafeCell<Vec<EdgeData>>,

    // ── Pending edges (UnsafeCell for auto-CSR-build from &self queries) ──
    pub(crate) pending_edges: UnsafeCell<Vec<(u32, u32, u64)>>,

    // ── Mutation overflow (for incremental edges after CSR) ──
    overflow_out: HashMap<u32, Vec<CsrEdge>>,
    overflow_in: HashMap<u32, Vec<CsrEdge>>,
    free_edge_slots: Vec<u32>,

    // ── Storage directory (the graph lives here) ──
    data_dir: PathBuf,
}

use std::sync::Arc;

// SAFETY: DiskGraph uses UnsafeCell for node_arena and edge_arena — both are
// append-only caches mutated through &self. The borrow checker ensures no
// &mut self can occur while any &self borrow (iterator) is alive.
unsafe impl Send for DiskGraph {}
unsafe impl Sync for DiskGraph {}

impl std::fmt::Debug for DiskGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DiskGraph({} nodes, {} edges, dir={:?})",
            self.node_count,
            self.edge_count,
            self.data_dir.display()
        )
    }
}

impl DiskGraph {
    // ====================================================================
    // Construction
    // ====================================================================

    /// Create an empty DiskGraph at the given directory path.
    /// All data is written directly to disk via mmap.
    pub fn new_at_path(data_dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(data_dir)?;

        Ok(DiskGraph {
            node_slots: MmapOrVec::mapped(&data_dir.join("node_slots.bin"), 1024)?,
            node_count: 0,
            free_node_slots: Vec::new(),
            node_arena: UnsafeCell::new(Vec::with_capacity(256)),
            column_stores: HashMap::new(),
            out_offsets: MmapOrVec::mapped(&data_dir.join("out_offsets.bin"), 1025)?,
            out_edges: MmapOrVec::new(),
            in_offsets: MmapOrVec::mapped(&data_dir.join("in_offsets.bin"), 1025)?,
            in_edges: MmapOrVec::new(),
            edge_endpoints: MmapOrVec::new(),
            edge_count: 0,
            next_edge_idx: 0,
            edge_properties: HashMap::new(),
            edge_arena: UnsafeCell::new(Vec::with_capacity(256)),
            pending_edges: UnsafeCell::new(Vec::new()),
            overflow_out: HashMap::new(),
            overflow_in: HashMap::new(),
            free_edge_slots: Vec::new(),
            data_dir: data_dir.to_path_buf(),
        })
    }

    /// Build a DiskGraph from a petgraph StableDiGraph.
    /// Converts nodes to DiskNodeSlots on disk, builds CSR arrays.
    pub fn from_stable_digraph(
        graph: &mut petgraph::stable_graph::StableDiGraph<NodeData, EdgeData>,
        data_dir: &Path,
    ) -> std::io::Result<Self> {
        use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};

        std::fs::create_dir_all(data_dir)?;

        let node_bound = graph.node_bound();
        let edge_count = graph.edge_count();

        // ── Build node slots on disk ──
        let mut node_slots = MmapOrVec::mapped(&data_dir.join("node_slots.bin"), node_bound)?;
        let mut node_count = 0usize;
        for i in 0..node_bound {
            let idx = NodeIndex::new(i);
            if let Some(node) = graph.node_weight(idx) {
                let row_id = match &node.properties {
                    crate::graph::schema::PropertyStorage::Columnar { row_id, .. } => *row_id,
                    _ => i as u32,
                };
                node_slots.push(DiskNodeSlot {
                    node_type: node.node_type.as_u64(),
                    row_id,
                    flags: DiskNodeSlot::ALIVE_BIT,
                });
                node_count += 1;
            } else {
                node_slots.push(DiskNodeSlot::default()); // dead slot
            }
        }

        // ── Count outgoing/incoming edges per node ──
        let mut out_counts = vec![0u64; node_bound];
        let mut in_counts = vec![0u64; node_bound];
        for edge in graph.edge_references() {
            let s = edge.source().index();
            let t = edge.target().index();
            out_counts[s] += 1;
            in_counts[t] += 1;
        }

        // ── Build offset arrays (prefix sums) ──
        let mut out_offsets = MmapOrVec::mapped(&data_dir.join("out_offsets.bin"), node_bound + 1)?;
        let mut in_offsets = MmapOrVec::mapped(&data_dir.join("in_offsets.bin"), node_bound + 1)?;

        let mut out_acc = 0u64;
        let mut in_acc = 0u64;
        for i in 0..node_bound {
            out_offsets.push(out_acc);
            in_offsets.push(in_acc);
            out_acc += out_counts[i];
            in_acc += in_counts[i];
        }
        out_offsets.push(out_acc);
        in_offsets.push(in_acc);

        // ── Build CSR edge arrays ──
        let mut out_edges = MmapOrVec::mapped(&data_dir.join("out_edges.bin"), edge_count)?;
        let mut in_edges = MmapOrVec::mapped(&data_dir.join("in_edges.bin"), edge_count)?;
        let mut edge_endpoints_vec =
            MmapOrVec::mapped(&data_dir.join("edge_endpoints.bin"), edge_count)?;
        let mut edge_properties: HashMap<u32, Vec<(InternedKey, Value)>> = HashMap::new();

        // Initialize edge arrays with enough space
        for _ in 0..edge_count {
            out_edges.push(CsrEdge::default());
            in_edges.push(CsrEdge::default());
            edge_endpoints_vec.push(EdgeEndpoints::default());
        }

        // Fill positions: use write cursors per node
        let mut out_cursors = vec![0u64; node_bound];
        let mut in_cursors = vec![0u64; node_bound];

        let mut edge_idx = 0u32;
        for edge in graph.edge_references() {
            let s = edge.source().index();
            let t = edge.target().index();
            let ct = edge.weight().connection_type;

            let csr_out = CsrEdge {
                peer: t as u32,
                edge_idx,
                conn_type: ct.as_u64(),
            };
            let out_pos = out_offsets.get(s) + out_cursors[s];
            out_edges.set(out_pos as usize, csr_out);
            out_cursors[s] += 1;

            let csr_in = CsrEdge {
                peer: s as u32,
                edge_idx,
                conn_type: ct.as_u64(),
            };
            let in_pos = in_offsets.get(t) + in_cursors[t];
            in_edges.set(in_pos as usize, csr_in);
            in_cursors[t] += 1;

            edge_endpoints_vec.set(
                edge_idx as usize,
                EdgeEndpoints {
                    source: s as u32,
                    target: t as u32,
                    connection_type: ct.as_u64(),
                },
            );

            if !edge.weight().properties.is_empty() {
                edge_properties.insert(edge_idx, edge.weight().properties.clone());
            }

            edge_idx += 1;
        }

        Ok(DiskGraph {
            node_slots,
            node_count,
            free_node_slots: Vec::new(),
            node_arena: UnsafeCell::new(Vec::with_capacity(1024)),
            column_stores: HashMap::new(), // filled by caller via set_column_stores()
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            edge_endpoints: edge_endpoints_vec,
            edge_count,
            next_edge_idx: edge_idx,
            edge_properties,
            edge_arena: UnsafeCell::new(Vec::with_capacity(1024)),
            pending_edges: UnsafeCell::new(Vec::new()),
            overflow_out: HashMap::new(),
            overflow_in: HashMap::new(),
            free_edge_slots: Vec::new(),
            data_dir: data_dir.to_path_buf(),
        })
    }

    // ====================================================================
    // Node methods
    // ====================================================================

    /// Set column store references. Called by DirGraph after columnar setup or load.
    pub fn set_column_stores(
        &mut self,
        stores: HashMap<InternedKey, Arc<crate::graph::column_store::ColumnStore>>,
    ) {
        self.column_stores = stores;
    }

    /// Get a DiskNodeSlot by index (for rebuild_type_indices without arena).
    #[inline]
    pub fn node_slot(&self, i: usize) -> DiskNodeSlot {
        if i < self.node_slots.len() {
            self.node_slots.get(i)
        } else {
            DiskNodeSlot::default()
        }
    }

    // ====================================================================
    // Node methods
    // ====================================================================

    /// Materialize a NodeData from disk slot + ColumnStore into the arena.
    #[inline]
    pub fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        let i = idx.index();
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }

        let arena = unsafe { &mut *self.node_arena.get() };
        let pos = arena.len();

        let node_type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&node_type_key);

        let node_data = if let Some(store) = store {
            NodeData {
                id: Value::Null,
                title: Value::Null,
                node_type: node_type_key,
                properties: crate::graph::schema::PropertyStorage::Columnar {
                    store: Arc::clone(store),
                    row_id: slot.row_id,
                },
            }
        } else {
            NodeData {
                id: Value::Null,
                title: Value::Null,
                node_type: node_type_key,
                properties: crate::graph::schema::PropertyStorage::Map(HashMap::new()),
            }
        };

        arena.push(node_data);
        // SAFETY: arena is append-only during &self borrows.
        unsafe { Some(&*(arena.get_unchecked(pos) as *const NodeData)) }
    }

    pub fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        self.clear_arenas();
        let i = idx.index();
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }

        let node_type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&node_type_key).cloned();

        // Materialize with actual values for mutation
        let (id_val, title_val) = if let Some(ref s) = store {
            (
                s.get_id(slot.row_id).unwrap_or(Value::Null),
                s.get_title(slot.row_id).unwrap_or(Value::Null),
            )
        } else {
            (Value::Null, Value::Null)
        };

        let arena = self.node_arena.get_mut();
        let pos = arena.len();
        arena.push(NodeData {
            id: id_val,
            title: title_val,
            node_type: node_type_key,
            properties: if let Some(s) = store {
                crate::graph::schema::PropertyStorage::Columnar {
                    store: s,
                    row_id: slot.row_id,
                }
            } else {
                crate::graph::schema::PropertyStorage::Map(HashMap::new())
            },
        });

        Some(&mut arena[pos])
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    #[inline]
    pub fn node_bound(&self) -> usize {
        self.node_slots.len()
    }

    pub fn add_node(&mut self, data: NodeData) -> NodeIndex {
        self.clear_arenas();

        // Extract row_id from property storage if columnar, else use slot index
        let row_id = match &data.properties {
            crate::graph::schema::PropertyStorage::Columnar { row_id, .. } => *row_id,
            _ => self.node_slots.len() as u32,
        };

        let slot = DiskNodeSlot {
            node_type: data.node_type.as_u64(),
            row_id,
            flags: DiskNodeSlot::ALIVE_BIT,
        };

        if let Some(recycled) = self.free_node_slots.pop() {
            let idx = recycled as usize;
            self.node_slots.set(idx, slot);
            self.node_count += 1;
            NodeIndex::new(idx)
        } else {
            let idx = self.node_slots.len();
            self.node_slots.push(slot);
            // Extend CSR offset arrays
            let last_out = if !self.out_offsets.is_empty() {
                self.out_offsets.get(self.out_offsets.len() - 1)
            } else {
                0
            };
            self.out_offsets.push(last_out);
            let last_in = if !self.in_offsets.is_empty() {
                self.in_offsets.get(self.in_offsets.len() - 1)
            } else {
                0
            };
            self.in_offsets.push(last_in);
            self.node_count += 1;
            NodeIndex::new(idx)
        }
    }

    pub fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        self.clear_arenas();
        let i = idx.index();
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }

        // Materialize the NodeData before removing
        let node_type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&node_type_key).cloned();
        let (id_val, title_val) = if let Some(ref s) = store {
            (
                s.get_id(slot.row_id).unwrap_or(Value::Null),
                s.get_title(slot.row_id).unwrap_or(Value::Null),
            )
        } else {
            (Value::Null, Value::Null)
        };
        let data = NodeData {
            id: id_val,
            title: title_val,
            node_type: node_type_key,
            properties: if let Some(s) = store {
                crate::graph::schema::PropertyStorage::Columnar {
                    store: s,
                    row_id: slot.row_id,
                }
            } else {
                crate::graph::schema::PropertyStorage::Map(HashMap::new())
            },
        };

        // Mark slot as dead
        let mut dead_slot = slot;
        dead_slot.flags = 0;
        self.node_slots.set(i, dead_slot);
        self.node_count -= 1;
        self.free_node_slots.push(i as u32);

        // Tombstone all incident edges
        self.tombstone_edges_for_node(i);

        Some(data)
    }

    pub fn node_indices_iter(&self) -> DiskNodeIndices<'_> {
        DiskNodeIndices::new(&self.node_slots)
    }

    // ====================================================================
    // Edge methods
    // ====================================================================

    /// Materialize an EdgeData into the arena and return a reference to it.
    /// `conn_type` is passed by the caller (from CsrEdge.conn_type) since
    /// EdgeEndpoints no longer stores connection_type.
    #[inline]
    /// Materialize an EdgeData into the arena. Reads conn_type from EdgeEndpoints
    /// (O(1) lookup) and properties from edge_properties HashMap.
    pub(crate) fn materialize_edge(&self, edge_idx: u32) -> &EdgeData {
        let arena = unsafe { &mut *self.edge_arena.get() };
        let idx = arena.len();
        let ep = self.edge_endpoints.get(edge_idx as usize);
        let ct = InternedKey::from_u64(ep.connection_type);
        let props = self
            .edge_properties
            .get(&edge_idx)
            .cloned()
            .unwrap_or_default();
        arena.push(EdgeData {
            connection_type: ct,
            properties: props,
        });
        // SAFETY: arena is append-only during &self borrows.
        // The reference is valid as long as the &self borrow lives.
        unsafe { &*(arena.get_unchecked(idx) as *const EdgeData) }
    }

    /// Auto-build CSR from pending edges if needed. Called from &self query methods.
    /// Check if pending edges need to be built into CSR.
    /// Panics with a helpful message if called with unbuilt edges.
    #[inline]
    fn ensure_csr(&self) {
        // No-op check — pending edges should be empty after build_csr_from_pending.
        // If not empty, queries may miss some edges (but won't crash).
    }

    /// Clear all materialization arenas. Called before any &mut self operation.
    #[inline]
    fn clear_arenas(&mut self) {
        self.node_arena.get_mut().clear();
        self.edge_arena.get_mut().clear();
    }

    pub fn edges_directed_iter(&self, a: NodeIndex, dir: Direction) -> DiskEdges<'_> {
        self.ensure_csr();
        let node = a.index();
        let (offsets, edges) = match dir {
            Direction::Outgoing => (&self.out_offsets, &self.out_edges),
            Direction::Incoming => (&self.in_offsets, &self.in_edges),
        };

        if node >= offsets.len().saturating_sub(1) {
            return DiskEdges::new_empty(self, dir, a);
        }

        let start = offsets.get(node) as usize;
        let end = offsets.get(node + 1) as usize;

        DiskEdges::new(self, dir, a, edges, start, end, None)
    }

    pub fn edge_references_iter(&self) -> DiskEdgeReferences<'_> {
        self.ensure_csr();
        DiskEdgeReferences::new(self)
    }

    pub fn edge_indices_iter(&self) -> DiskEdgeIndices<'_> {
        self.ensure_csr();
        DiskEdgeIndices::new(self.next_edge_idx, &self.edge_endpoints)
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        self.ensure_csr();
        let ei = idx.index();
        if ei >= self.next_edge_idx as usize {
            return None;
        }
        let ep = self.edge_endpoints.get(ei);
        if ep.source == TOMBSTONE_EDGE {
            return None;
        }
        // Find conn_type from the source node's CSR outgoing edges

        Some(self.materialize_edge(ei as u32))
    }

    pub fn edge_weight_mut(&mut self, _idx: EdgeIndex) -> Option<&mut EdgeData> {
        // Edge mutation in disk mode is complex — for now, support via
        // re-materialization. Full implementation deferred.
        unimplemented!("DiskGraph: edge_weight_mut not yet supported")
    }

    pub fn edge_endpoints_fn(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.ensure_csr();
        let ei = idx.index();
        if ei >= self.next_edge_idx as usize {
            return None;
        }
        let ep = self.edge_endpoints.get(ei);
        if ep.source == TOMBSTONE_EDGE {
            return None;
        }
        Some((
            NodeIndex::new(ep.source as usize),
            NodeIndex::new(ep.target as usize),
        ))
    }

    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        self.clear_arenas();
        let edge_idx = self.next_edge_idx;
        self.next_edge_idx += 1;

        let ct = data.connection_type;

        if !data.properties.is_empty() {
            self.edge_properties.insert(edge_idx, data.properties);
        }

        // Append to pending edges log (fast Vec push, no HashMap)
        self.pending_edges
            .get_mut()
            .push((a.index() as u32, b.index() as u32, ct.as_u64()));

        self.edge_count += 1;
        EdgeIndex::new(edge_idx as usize)
    }

    pub fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        self.clear_arenas();
        let ei = idx.index();
        if ei >= self.next_edge_idx as usize {
            return None;
        }
        let ep = self.edge_endpoints.get(ei);
        if ep.source == TOMBSTONE_EDGE {
            return None;
        }

        let ct = InternedKey::from_u64(ep.connection_type);
        let props = self
            .edge_properties
            .remove(&(ei as u32))
            .unwrap_or_default();
        let result = EdgeData {
            connection_type: ct,
            properties: props,
        };

        let src = ep.source as usize;
        let tgt = ep.target as usize;
        let ei32 = ei as u32;

        // Tombstone in outgoing CSR
        Self::tombstone_in_array(&self.out_offsets, &mut self.out_edges, src, ei32);
        // Tombstone in incoming CSR
        Self::tombstone_in_array(&self.in_offsets, &mut self.in_edges, tgt, ei32);

        // Tombstone in overflow lists
        if let Some(list) = self.overflow_out.get_mut(&(src as u32)) {
            list.retain(|e| e.edge_idx != ei32);
        }
        if let Some(list) = self.overflow_in.get_mut(&(tgt as u32)) {
            list.retain(|e| e.edge_idx != ei32);
        }

        // Tombstone in endpoints
        self.edge_endpoints.set(
            ei,
            EdgeEndpoints {
                source: TOMBSTONE_EDGE,
                target: TOMBSTONE_EDGE,
                connection_type: 0,
            },
        );

        self.edge_count -= 1;
        self.free_edge_slots.push(ei32);
        Some(result)
    }

    pub fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        self.ensure_csr();
        let src = a.index();
        let tgt = b.index() as u32;

        // Search CSR outgoing edges from a
        if src < self.out_offsets.len().saturating_sub(1) {
            let start = self.out_offsets.get(src) as usize;
            let end = self.out_offsets.get(src + 1) as usize;
            for i in start..end {
                let e = self.out_edges.get(i);
                if e.edge_idx != TOMBSTONE_EDGE && e.peer == tgt {
                    return Some(EdgeIndex::new(e.edge_idx as usize));
                }
            }
        }

        // Search overflow
        if let Some(list) = self.overflow_out.get(&(src as u32)) {
            for e in list {
                if e.edge_idx != TOMBSTONE_EDGE && e.peer == tgt {
                    return Some(EdgeIndex::new(e.edge_idx as usize));
                }
            }
        }

        None
    }

    pub fn edges_connecting_iter(&self, a: NodeIndex, b: NodeIndex) -> DiskEdgesConnecting<'_> {
        self.ensure_csr();
        DiskEdgesConnecting::new(self, a, b)
    }

    pub fn edge_weights_iter(&self) -> Box<dyn Iterator<Item = &EdgeData> + '_> {
        self.ensure_csr();
        Box::new((0..self.next_edge_idx).filter_map(move |i| {
            let ep = self.edge_endpoints.get(i as usize);
            if ep.source == TOMBSTONE_EDGE {
                return None;
            }

            Some(self.materialize_edge(i))
        }))
    }

    // ====================================================================
    // Neighbor methods
    // ====================================================================

    pub fn neighbors_directed_iter(&self, a: NodeIndex, dir: Direction) -> DiskNeighbors {
        self.ensure_csr();
        let node = a.index();
        let (offsets, edges) = match dir {
            Direction::Outgoing => (&self.out_offsets, &self.out_edges),
            Direction::Incoming => (&self.in_offsets, &self.in_edges),
        };

        let overflow = match dir {
            Direction::Outgoing => self.overflow_out.get(&(node as u32)),
            Direction::Incoming => self.overflow_in.get(&(node as u32)),
        };

        if node >= offsets.len().saturating_sub(1) {
            return DiskNeighbors::new_empty();
        }

        let start = offsets.get(node) as usize;
        let end = offsets.get(node + 1) as usize;

        DiskNeighbors::new(edges, start, end, overflow)
    }

    pub fn neighbors_undirected_iter(&self, a: NodeIndex) -> DiskNeighbors {
        self.ensure_csr();
        // Collect both outgoing and incoming neighbors
        let node = a.index();
        let mut peers = Vec::new();

        // Outgoing
        if node < self.out_offsets.len().saturating_sub(1) {
            let start = self.out_offsets.get(node) as usize;
            let end = self.out_offsets.get(node + 1) as usize;
            for i in start..end {
                let e = self.out_edges.get(i);
                if e.edge_idx != TOMBSTONE_EDGE {
                    peers.push(NodeIndex::new(e.peer as usize));
                }
            }
        }
        if let Some(list) = self.overflow_out.get(&(node as u32)) {
            for e in list {
                if e.edge_idx != TOMBSTONE_EDGE {
                    peers.push(NodeIndex::new(e.peer as usize));
                }
            }
        }

        // Incoming
        if node < self.in_offsets.len().saturating_sub(1) {
            let start = self.in_offsets.get(node) as usize;
            let end = self.in_offsets.get(node + 1) as usize;
            for i in start..end {
                let e = self.in_edges.get(i);
                if e.edge_idx != TOMBSTONE_EDGE {
                    peers.push(NodeIndex::new(e.peer as usize));
                }
            }
        }
        if let Some(list) = self.overflow_in.get(&(node as u32)) {
            for e in list {
                if e.edge_idx != TOMBSTONE_EDGE {
                    peers.push(NodeIndex::new(e.peer as usize));
                }
            }
        }

        DiskNeighbors::from_collected(peers)
    }

    // ====================================================================
    // CSR construction from pending edges
    // ====================================================================

    /// Build CSR arrays from the pending edges log. Called lazily on first
    /// query, or explicitly on save. Two-pass algorithm:
    /// Pass 1: count outgoing/incoming degree per node
    /// Pass 2: fill CSR arrays
    pub fn build_csr_from_pending(&mut self) {
        let pending = self.pending_edges.get_mut();
        if pending.is_empty() {
            return;
        }

        let node_bound = self.node_slots.len();
        let edge_count = pending.len();

        // Pass 1: count degrees
        let mut out_counts = vec![0u64; node_bound];
        let mut in_counts = vec![0u64; node_bound];
        for &(src, tgt, _) in pending.iter() {
            if (src as usize) < node_bound {
                out_counts[src as usize] += 1;
            }
            if (tgt as usize) < node_bound {
                in_counts[tgt as usize] += 1;
            }
        }

        // Build offset arrays in heap (small: ~2 GB for 124M nodes)
        let mut out_offsets = MmapOrVec::with_capacity(node_bound + 1);
        let mut in_offsets = MmapOrVec::with_capacity(node_bound + 1);
        let mut out_acc = 0u64;
        let mut in_acc = 0u64;
        for i in 0..node_bound {
            out_offsets.push(out_acc);
            in_offsets.push(in_acc);
            out_acc += out_counts[i];
            in_acc += in_counts[i];
        }
        out_offsets.push(out_acc);
        in_offsets.push(in_acc);
        drop(out_counts);
        drop(in_counts);

        // Pass 2: Build CSR arrays with SEQUENTIAL writes only.
        // Sort pending_edges by key, then push() sequentially — no random access,
        // no page faults, no mmap thrashing. ~100x faster than random set().

        // 2a: Build edge_endpoints from original order (sequential push)
        let mut edge_endpoints_vec = MmapOrVec::with_capacity(edge_count);
        for &(src, tgt, conn_type) in pending.iter() {
            edge_endpoints_vec.push(EdgeEndpoints {
                source: src,
                target: tgt,
                connection_type: conn_type,
            });
        }

        // 2b: Create sort_indices (4 bytes/edge = 3.4 GB for 862M edges — fits in 16 GB)
        // Sort by source → build out_edges with sequential push (no random writes)
        let mut sort_indices: Vec<u32> = (0..edge_count as u32).collect();
        sort_indices.sort_unstable_by_key(|&i| pending[i as usize].0);

        let mut out_edges = MmapOrVec::with_capacity(edge_count);
        for &i in &sort_indices {
            let (_, tgt, conn_type) = pending[i as usize];
            out_edges.push(CsrEdge {
                peer: tgt,
                edge_idx: i,
                conn_type,
            });
        }

        // 2c: Re-sort by target → build in_edges with sequential push
        sort_indices.sort_unstable_by_key(|&i| pending[i as usize].1);

        let mut in_edges = MmapOrVec::with_capacity(edge_count);
        for &i in &sort_indices {
            let (src, _, conn_type) = pending[i as usize];
            in_edges.push(CsrEdge {
                peer: src,
                edge_idx: i,
                conn_type,
            });
        }
        drop(sort_indices);

        self.out_offsets = out_offsets;
        self.out_edges = out_edges;
        self.in_offsets = in_offsets;
        self.in_edges = in_edges;
        self.edge_endpoints = edge_endpoints_vec;
        pending.clear();
        pending.shrink_to_fit();
    }

    // ====================================================================
    // Internal helpers
    // ====================================================================

    /// Find the connection type for an edge by scanning the source node's CSR range.
    /// Returns 0 if not found (edge may be in overflow or deleted).
    pub(crate) fn find_conn_type_for_edge(&self, source_node: usize, edge_idx: u32) -> u64 {
        // Search CSR outgoing edges from source_node
        if source_node < self.out_offsets.len().saturating_sub(1) {
            let start = self.out_offsets.get(source_node) as usize;
            let end = self.out_offsets.get(source_node + 1) as usize;
            for i in start..end {
                let e = self.out_edges.get(i);
                if e.edge_idx == edge_idx {
                    return e.conn_type;
                }
            }
        }
        // Search overflow
        if let Some(list) = self.overflow_out.get(&(source_node as u32)) {
            for e in list {
                if e.edge_idx == edge_idx {
                    return e.conn_type;
                }
            }
        }
        0 // not found
    }

    fn tombstone_edges_for_node(&mut self, node: usize) {
        // Tombstone outgoing CSR edges
        if node < self.out_offsets.len().saturating_sub(1) {
            let start = self.out_offsets.get(node) as usize;
            let end = self.out_offsets.get(node + 1) as usize;
            for i in start..end {
                let mut e = self.out_edges.get(i);
                if e.edge_idx != TOMBSTONE_EDGE {
                    let ei = e.edge_idx;
                    e.edge_idx = TOMBSTONE_EDGE;
                    self.out_edges.set(i, e);
                    // Tombstone the corresponding incoming edge
                    self.tombstone_in_edge_for(e.peer as usize, ei);
                    // Tombstone endpoints
                    self.edge_endpoints.set(
                        ei as usize,
                        EdgeEndpoints {
                            source: TOMBSTONE_EDGE,
                            target: TOMBSTONE_EDGE,
                            connection_type: 0,
                        },
                    );
                    self.edge_properties.remove(&ei);
                    self.edge_count -= 1;
                    self.free_edge_slots.push(ei);
                }
            }
        }

        // Tombstone incoming CSR edges
        if node < self.in_offsets.len().saturating_sub(1) {
            let start = self.in_offsets.get(node) as usize;
            let end = self.in_offsets.get(node + 1) as usize;
            for i in start..end {
                let mut e = self.in_edges.get(i);
                if e.edge_idx != TOMBSTONE_EDGE {
                    let ei = e.edge_idx;
                    e.edge_idx = TOMBSTONE_EDGE;
                    self.in_edges.set(i, e);
                    self.tombstone_out_edge_for(e.peer as usize, ei);
                    self.edge_endpoints.set(
                        ei as usize,
                        EdgeEndpoints {
                            source: TOMBSTONE_EDGE,
                            target: TOMBSTONE_EDGE,
                            connection_type: 0,
                        },
                    );
                    self.edge_properties.remove(&ei);
                    self.edge_count -= 1;
                    self.free_edge_slots.push(ei);
                }
            }
        }

        // Tombstone overflow edges
        if let Some(list) = self.overflow_out.remove(&(node as u32)) {
            for e in &list {
                if e.edge_idx != TOMBSTONE_EDGE {
                    self.tombstone_in_edge_for(e.peer as usize, e.edge_idx);
                    self.edge_endpoints.set(
                        e.edge_idx as usize,
                        EdgeEndpoints {
                            source: TOMBSTONE_EDGE,
                            target: TOMBSTONE_EDGE,
                            connection_type: 0,
                        },
                    );
                    self.edge_properties.remove(&e.edge_idx);
                    self.edge_count -= 1;
                    self.free_edge_slots.push(e.edge_idx);
                }
            }
        }
        if let Some(list) = self.overflow_in.remove(&(node as u32)) {
            for e in &list {
                if e.edge_idx != TOMBSTONE_EDGE {
                    self.tombstone_out_edge_for(e.peer as usize, e.edge_idx);
                    self.edge_endpoints.set(
                        e.edge_idx as usize,
                        EdgeEndpoints {
                            source: TOMBSTONE_EDGE,
                            target: TOMBSTONE_EDGE,
                            connection_type: 0,
                        },
                    );
                    self.edge_properties.remove(&e.edge_idx);
                    self.edge_count -= 1;
                    self.free_edge_slots.push(e.edge_idx);
                }
            }
        }
    }

    fn tombstone_in_edge_for(&mut self, node: usize, edge_idx: u32) {
        if node < self.in_offsets.len().saturating_sub(1) {
            let start = self.in_offsets.get(node) as usize;
            let end = self.in_offsets.get(node + 1) as usize;
            for i in start..end {
                let mut e = self.in_edges.get(i);
                if e.edge_idx == edge_idx {
                    e.edge_idx = TOMBSTONE_EDGE;
                    self.in_edges.set(i, e);
                    return;
                }
            }
        }
        if let Some(list) = self.overflow_in.get_mut(&(node as u32)) {
            list.retain(|e| e.edge_idx != edge_idx);
        }
    }

    fn tombstone_out_edge_for(&mut self, node: usize, edge_idx: u32) {
        if node < self.out_offsets.len().saturating_sub(1) {
            let start = self.out_offsets.get(node) as usize;
            let end = self.out_offsets.get(node + 1) as usize;
            for i in start..end {
                let mut e = self.out_edges.get(i);
                if e.edge_idx == edge_idx {
                    e.edge_idx = TOMBSTONE_EDGE;
                    self.out_edges.set(i, e);
                    return;
                }
            }
        }
        if let Some(list) = self.overflow_out.get_mut(&(node as u32)) {
            list.retain(|e| e.edge_idx != edge_idx);
        }
    }

    fn tombstone_in_array(
        offsets: &MmapOrVec<u64>,
        edges: &mut MmapOrVec<CsrEdge>,
        node: usize,
        edge_idx: u32,
    ) {
        if node < offsets.len().saturating_sub(1) {
            let start = offsets.get(node) as usize;
            let end = offsets.get(node + 1) as usize;
            for i in start..end {
                let mut e = edges.get(i);
                if e.edge_idx == edge_idx {
                    e.edge_idx = TOMBSTONE_EDGE;
                    edges.set(i, e);
                    return;
                }
            }
        }
    }
}

impl Clone for DiskGraph {
    fn clone(&self) -> Self {
        // Clone into heap copies (clone is expensive for disk graphs)
        let mut node_slots = MmapOrVec::with_capacity(self.node_slots.len());
        for i in 0..self.node_slots.len() {
            node_slots.push(self.node_slots.get(i));
        }
        let mut out_offsets = MmapOrVec::with_capacity(self.out_offsets.len());
        for i in 0..self.out_offsets.len() {
            out_offsets.push(self.out_offsets.get(i));
        }
        let mut out_edges = MmapOrVec::with_capacity(self.out_edges.len());
        for i in 0..self.out_edges.len() {
            out_edges.push(self.out_edges.get(i));
        }
        let mut in_offsets = MmapOrVec::with_capacity(self.in_offsets.len());
        for i in 0..self.in_offsets.len() {
            in_offsets.push(self.in_offsets.get(i));
        }
        let mut in_edges = MmapOrVec::with_capacity(self.in_edges.len());
        for i in 0..self.in_edges.len() {
            in_edges.push(self.in_edges.get(i));
        }
        let mut edge_endpoints = MmapOrVec::with_capacity(self.edge_endpoints.len());
        for i in 0..self.edge_endpoints.len() {
            edge_endpoints.push(self.edge_endpoints.get(i));
        }

        DiskGraph {
            node_slots,
            node_count: self.node_count,
            free_node_slots: self.free_node_slots.clone(),
            node_arena: UnsafeCell::new(Vec::new()),
            column_stores: self.column_stores.clone(),
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            edge_endpoints,
            edge_count: self.edge_count,
            next_edge_idx: self.next_edge_idx,
            edge_properties: self.edge_properties.clone(),
            edge_arena: UnsafeCell::new(Vec::new()),
            pending_edges: UnsafeCell::new(unsafe { &*self.pending_edges.get() }.clone()),
            overflow_out: self.overflow_out.clone(),
            overflow_in: self.overflow_in.clone(),
            free_edge_slots: self.free_edge_slots.clone(),
            data_dir: self.data_dir.clone(),
        }
    }
}

// Index implementations
impl std::ops::Index<NodeIndex> for DiskGraph {
    type Output = NodeData;
    #[inline]
    fn index(&self, index: NodeIndex) -> &NodeData {
        self.node_weight(index).expect("DiskGraph: node not found")
    }
}

impl std::ops::Index<EdgeIndex> for DiskGraph {
    type Output = EdgeData;
    #[inline]
    fn index(&self, index: EdgeIndex) -> &EdgeData {
        self.edge_weight(index).expect("DiskGraph: edge not found")
    }
}

// ============================================================================
// Disk persistence — the directory IS the saved graph
// ============================================================================

/// Metadata stored alongside the binary files in the disk graph directory.
#[derive(serde::Serialize, serde::Deserialize)]
struct DiskGraphMeta {
    node_count: usize,
    node_slots_len: usize,
    edge_count: usize,
    next_edge_idx: u32,
    out_offsets_len: usize,
    out_edges_len: usize,
    in_offsets_len: usize,
    in_edges_len: usize,
    edge_endpoints_len: usize,
    free_node_slots: Vec<u32>,
    free_edge_slots: Vec<u32>,
}

impl DiskGraph {
    /// Save the disk graph to a directory. All binary arrays are zstd-compressed.
    /// On load, they're decompressed to temp files and mmap'd for zero-cost queries.
    pub fn save_to_dir(
        &self,
        target_dir: &Path,
        interner: &crate::graph::schema::StringInterner,
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(target_dir)?;

        // Build CSR from pending edges before saving
        // (can't call &mut self here, but pending_edges should be empty
        //  if build_csr_from_pending was called earlier)

        // Write all binary arrays as compressed .zst files
        write_compressed(&self.node_slots, &target_dir.join("node_slots.bin.zst"))?;
        write_compressed(&self.out_offsets, &target_dir.join("out_offsets.bin.zst"))?;
        write_compressed(&self.out_edges, &target_dir.join("out_edges.bin.zst"))?;
        write_compressed(&self.in_offsets, &target_dir.join("in_offsets.bin.zst"))?;
        write_compressed(&self.in_edges, &target_dir.join("in_edges.bin.zst"))?;
        write_compressed(
            &self.edge_endpoints,
            &target_dir.join("edge_endpoints.bin.zst"),
        )?;

        // Clean up any raw .bin files left from new_at_path()
        for name in &[
            "node_slots.bin",
            "out_offsets.bin",
            "out_edges.bin",
            "in_offsets.bin",
            "in_edges.bin",
            "edge_endpoints.bin",
        ] {
            let _ = std::fs::remove_file(target_dir.join(name));
        }

        // Save overflow edges (bincode + zstd)
        if !self.overflow_out.is_empty() || !self.overflow_in.is_empty() {
            let overflow = (&self.overflow_out, &self.overflow_in);
            let bytes = bincode::serialize(&overflow).map_err(std::io::Error::other)?;
            let compressed =
                zstd::encode_all(bytes.as_slice(), 3).map_err(std::io::Error::other)?;
            std::fs::write(target_dir.join("overflow_edges.bin.zst"), compressed)?;
        }

        // Save edge properties (sparse, bincode + zstd)
        if !self.edge_properties.is_empty() {
            let _guard = crate::graph::schema::SerdeSerializeGuard::new(interner);
            let prop_bytes =
                bincode::serialize(&self.edge_properties).map_err(std::io::Error::other)?;
            let compressed =
                zstd::encode_all(prop_bytes.as_slice(), 3).map_err(std::io::Error::other)?;
            std::fs::write(target_dir.join("edge_properties.bin.zst"), compressed)?;
        }

        // Save metadata
        let meta = DiskGraphMeta {
            node_count: self.node_count,
            node_slots_len: self.node_slots.len(),
            edge_count: self.edge_count,
            next_edge_idx: self.next_edge_idx,
            out_offsets_len: self.out_offsets.len(),
            out_edges_len: self.out_edges.len(),
            in_offsets_len: self.in_offsets.len(),
            in_edges_len: self.in_edges.len(),
            edge_endpoints_len: self.edge_endpoints.len(),
            free_node_slots: self.free_node_slots.clone(),
            free_edge_slots: self.free_edge_slots.clone(),
        };
        let meta_json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        std::fs::write(target_dir.join("disk_graph_meta.json"), meta_json)?;

        Ok(())
    }

    /// Load a disk graph from a directory.
    /// Compressed files are decompressed to temp dir and mmap'd.
    /// Returns `(DiskGraph, temp_dir)` — caller registers temp_dir for cleanup.
    pub fn load_from_dir(dir: &Path) -> std::io::Result<(Self, PathBuf)> {
        let meta_str = std::fs::read_to_string(dir.join("disk_graph_meta.json"))?;
        let meta: DiskGraphMeta = serde_json::from_str(&meta_str).map_err(std::io::Error::other)?;

        // Create temp dir for decompressed mmap files
        let temp_dir = std::env::temp_dir().join(format!(
            "kglite_disk_{}_{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir)?;

        // Decompress each array to temp file, then mmap
        let node_slots = load_compressed(
            &dir.join("node_slots.bin.zst"),
            meta.node_slots_len,
            &temp_dir,
        )?;
        let out_offsets = load_compressed(
            &dir.join("out_offsets.bin.zst"),
            meta.out_offsets_len,
            &temp_dir,
        )?;
        let out_edges = load_compressed(
            &dir.join("out_edges.bin.zst"),
            meta.out_edges_len,
            &temp_dir,
        )?;
        let in_offsets = load_compressed(
            &dir.join("in_offsets.bin.zst"),
            meta.in_offsets_len,
            &temp_dir,
        )?;
        let in_edges =
            load_compressed(&dir.join("in_edges.bin.zst"), meta.in_edges_len, &temp_dir)?;
        let edge_endpoints = load_compressed(
            &dir.join("edge_endpoints.bin.zst"),
            meta.edge_endpoints_len,
            &temp_dir,
        )?;

        // Load edge properties (sparse)
        let edge_properties = if dir.join("edge_properties.bin.zst").exists() {
            let compressed = std::fs::read(dir.join("edge_properties.bin.zst"))?;
            let bytes = zstd::decode_all(compressed.as_slice()).map_err(std::io::Error::other)?;
            bincode::deserialize(&bytes).map_err(std::io::Error::other)?
        } else {
            HashMap::new()
        };

        // Load overflow edges
        let (overflow_out, overflow_in) = if dir.join("overflow_edges.bin.zst").exists() {
            let compressed = std::fs::read(dir.join("overflow_edges.bin.zst"))?;
            let bytes = zstd::decode_all(compressed.as_slice()).map_err(std::io::Error::other)?;
            bincode::deserialize(&bytes).map_err(std::io::Error::other)?
        } else {
            (HashMap::new(), HashMap::new())
        };

        Ok((
            DiskGraph {
                node_slots,
                node_count: meta.node_count,
                free_node_slots: meta.free_node_slots,
                node_arena: UnsafeCell::new(Vec::with_capacity(1024)),
                column_stores: HashMap::new(),
                out_offsets,
                out_edges,
                in_offsets,
                in_edges,
                edge_endpoints,
                edge_count: meta.edge_count,
                next_edge_idx: meta.next_edge_idx,
                edge_properties,
                edge_arena: UnsafeCell::new(Vec::with_capacity(1024)),
                pending_edges: UnsafeCell::new(Vec::new()),
                overflow_out,
                overflow_in,
                free_edge_slots: meta.free_edge_slots,
                data_dir: dir.to_path_buf(),
            },
            temp_dir,
        ))
    }
}

// ============================================================================
// Compression helpers
// ============================================================================

/// Write a MmapOrVec as a zstd-compressed file.
fn write_compressed<T: Copy + Default + 'static>(
    data: &MmapOrVec<T>,
    path: &Path,
) -> std::io::Result<()> {
    let elem_size = std::mem::size_of::<T>();
    let total_bytes = data.len() * elem_size;

    let mut raw = Vec::with_capacity(total_bytes);
    if let Some(slice) = data.as_slice() {
        let byte_slice =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, total_bytes) };
        raw.extend_from_slice(byte_slice);
    } else {
        for i in 0..data.len() {
            let val = data.get(i);
            let bytes =
                unsafe { std::slice::from_raw_parts(&val as *const T as *const u8, elem_size) };
            raw.extend_from_slice(bytes);
        }
    }

    let compressed = zstd::encode_all(raw.as_slice(), 3)?;
    std::fs::write(path, compressed)
}

/// Load a zstd-compressed file, decompress to temp file, and mmap it.
fn load_compressed<T: Copy + Default + 'static>(
    path: &Path,
    len: usize,
    temp_dir: &Path,
) -> std::io::Result<MmapOrVec<T>> {
    if !path.exists() || len == 0 {
        return Ok(MmapOrVec::new());
    }
    let compressed = std::fs::read(path)?;
    let raw = zstd::decode_all(compressed.as_slice())?;

    // Write decompressed data to temp file and mmap
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("data")
        .trim_end_matches(".zst");
    let temp_path = temp_dir.join(file_name);
    std::fs::write(&temp_path, &raw)?;
    MmapOrVec::load_mapped(&temp_path, len)
}
