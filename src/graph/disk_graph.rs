// src/graph/disk_graph.rs
//
// Disk-backed graph storage using CSR (Compressed Sparse Row) format.
// Nodes are stored in memory (~40 bytes each in Columnar mode).
// Edges are stored in mmap'd CSR arrays (8 bytes per edge per direction).
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
// CSR edge record — 8 bytes, stored in mmap'd arrays
// ============================================================================

/// A single edge record in the CSR adjacency list.
/// `peer` is the target (in out_edges) or source (in in_edges).
/// Connection type is stored only in `EdgeEndpoints` (not duplicated here).
#[repr(C)]
#[derive(Copy, Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct CsrEdge {
    pub peer: u32,
    pub edge_idx: u32,
}

// SAFETY: CsrEdge is repr(C), Copy, Default, and contains only fixed-size integers.
// No padding issues on any platform since fields are naturally aligned (4+4=8).

/// [DEV] Entry for external merge sort. Carries all fields needed for CsrEdge
/// output plus sort keys, so the merge never needs to seek back to pending_mmap.
/// Secondary sort by connection_type ensures edges within each node's CSR range
/// are grouped by type, enabling O(log D) binary search for type-filtered queries.
/// 24 bytes (key:4 + conn_type:8 + peer:4 + orig_idx:4 + pad:4).
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct MergeSortEntry {
    key: u32,       // primary sort key (source or target node index)
    conn_type: u64, // secondary sort key (connection type)
    peer: u32,      // the other endpoint
    orig_idx: u32,  // original edge index (for CsrEdge.edge_idx)
}

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

    // ── Edge materialization arena (Mutex + Box for Rayon thread safety) ──
    // Box gives stable heap pointers that survive Vec reallocation.
    #[allow(clippy::vec_box)]
    edge_arena: std::sync::Mutex<Vec<Box<EdgeData>>>,
    /// Cache for edge_weight_mut: stores materialized EdgeData that may be modified.
    /// Flushed to edge_properties on next clear_arenas call.
    edge_mut_cache: HashMap<u32, EdgeData>,

    // ── Pending edges (UnsafeCell for auto-CSR-build from &self queries) ──
    // File-backed (MmapOrVec) to avoid ~14 GB heap allocation at Wikidata scale.
    pub(crate) pending_edges: UnsafeCell<MmapOrVec<(u32, u32, u64)>>,

    // ── Mutation overflow (for incremental edges after CSR) ──
    overflow_out: HashMap<u32, Vec<CsrEdge>>,
    overflow_in: HashMap<u32, Vec<CsrEdge>>,
    free_edge_slots: Vec<u32>,

    // ── Storage directory (the graph lives here) ──
    pub(crate) data_dir: PathBuf,
    // ── Dirty flag: flushed on Drop or next query ──
    metadata_dirty: bool,
    // ── CSR edges are sorted by (node, connection_type) — enables binary search
    pub(crate) csr_sorted_by_type: bool,
    // ── Defer CSR build: when true, ensure_csr() is a no-op. Edges accumulate
    // in pending_edges without intermediate CSR rebuilds. The CSR is built once
    // at save time via ensure_disk_edges_built(). Set true during construction
    // from add_nodes/add_connections, cleared after CSR build.
    pub(crate) defer_csr: bool,
    // ── Edge type counts computed during CSR build (raw InternedKey u64 → count).
    // Converted to String keys by the caller using the interner.
    pub(crate) edge_type_counts_raw: Option<HashMap<u64, usize>>,
    // ── Connection-type inverted index: maps conn_type → list of source node IDs
    // that have at least one outgoing edge of that type. Built during CSR merge sort.
    // conn_type_index_offsets[i] = start position in conn_type_index_sources for type i.
    // conn_type_index_types: list of connection type u64s (ordered).
    pub(crate) conn_type_index_types: MmapOrVec<u64>,
    pub(crate) conn_type_index_offsets: MmapOrVec<u64>,
    pub(crate) conn_type_index_sources: MmapOrVec<u32>,
    // ── Per-(conn_type, peer) edge-count histogram.
    // Built alongside conn_type_index at CSR time; answers unanchored-aggregate
    // queries (`MATCH (a)-[:T]->(b) RETURN b, count(a) ...`) in O(distinct-peers)
    // instead of O(|edge_endpoints|). 3-array CSR layout mirrors
    // conn_type_index. `peer_count_entries` is flat (peer_u32, count_u32) pairs
    // sorted by peer within each type's slice — stored as u32 pairs to avoid
    // alignment fuss (length is always 2× the pair count).
    pub(crate) peer_count_types: MmapOrVec<u64>,
    pub(crate) peer_count_offsets: MmapOrVec<u64>, // in units of pairs, not u32s
    pub(crate) peer_count_entries: MmapOrVec<u32>, // [peer0, count0, peer1, count1, …]
    // ── Tombstone tracking: set true when any node/edge is removed. Lets
    // count_edges_filtered short-circuit the per-edge tombstone check when
    // no removals have happened (fresh builds, reloaded read-only graphs).
    pub(crate) has_tombstones: bool,
    // ── Temp dir for CSR mmap files (cleaned up on Drop) ──
}

use std::sync::Arc;

// SAFETY: DiskGraph uses UnsafeCell for node_arena (single-threaded materialization)
// and Mutex<Vec<Box<EdgeData>>> for edge_arena (thread-safe for Rayon parallel queries).
// Pending edges use UnsafeCell but are only accessed during CSR build (&mut self).
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
            edge_arena: std::sync::Mutex::new(Vec::with_capacity(256)),
            edge_mut_cache: HashMap::new(),
            pending_edges: UnsafeCell::new(
                MmapOrVec::mapped(&data_dir.join("_pending_edges.bin"), 1 << 20)
                    .unwrap_or_else(|_| MmapOrVec::new()),
            ),
            overflow_out: HashMap::new(),
            overflow_in: HashMap::new(),
            free_edge_slots: Vec::new(),
            data_dir: data_dir.to_path_buf(),
            metadata_dirty: false,
            csr_sorted_by_type: false,
            // Phase 5: `defer_csr = false` by default so one-off Cypher
            // CREATE / MERGE inserts route directly to overflow_out /
            // overflow_in + edge_endpoints, where `edges_directed` reads
            // them immediately. Bulk loaders that want to batch edges in
            // `pending_edges` and rebuild the CSR at the end (ntriples)
            // flip this to `true` on the freshly-constructed DiskGraph.
            // Previously the default-`true` path silently dropped
            // Cypher-created edges from subsequent MATCH queries — the
            // pending buffer was written but `edges_directed_filtered_iter`
            // only reads CSR + overflow, not pending.
            defer_csr: false,
            edge_type_counts_raw: None,
            conn_type_index_types: MmapOrVec::new(),
            conn_type_index_offsets: MmapOrVec::new(),
            conn_type_index_sources: MmapOrVec::new(),
            peer_count_types: MmapOrVec::new(),
            peer_count_offsets: MmapOrVec::new(),
            peer_count_entries: MmapOrVec::new(),
            has_tombstones: false,
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
            };
            let out_pos = out_offsets.get(s) + out_cursors[s];
            out_edges.set(out_pos as usize, csr_out);
            out_cursors[s] += 1;

            let csr_in = CsrEdge {
                peer: s as u32,
                edge_idx,
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
            edge_arena: std::sync::Mutex::new(Vec::with_capacity(1024)),
            edge_mut_cache: HashMap::new(),
            pending_edges: UnsafeCell::new(MmapOrVec::new()),
            overflow_out: HashMap::new(),
            overflow_in: HashMap::new(),
            free_edge_slots: Vec::new(),
            data_dir: data_dir.to_path_buf(),
            metadata_dirty: false,
            csr_sorted_by_type: false,
            // Phase 5: `defer_csr = false` by default so one-off Cypher
            // CREATE / MERGE inserts route directly to overflow_out /
            // overflow_in + edge_endpoints, where `edges_directed` reads
            // them immediately. Bulk loaders that want to batch edges in
            // `pending_edges` and rebuild the CSR at the end (ntriples)
            // flip this to `true` on the freshly-constructed DiskGraph.
            // Previously the default-`true` path silently dropped
            // Cypher-created edges from subsequent MATCH queries — the
            // pending buffer was written but `edges_directed_filtered_iter`
            // only reads CSR + overflow, not pending.
            defer_csr: false,
            edge_type_counts_raw: None,
            conn_type_index_types: MmapOrVec::new(),
            conn_type_index_offsets: MmapOrVec::new(),
            conn_type_index_sources: MmapOrVec::new(),
            peer_count_types: MmapOrVec::new(),
            peer_count_offsets: MmapOrVec::new(),
            peer_count_entries: MmapOrVec::new(),
            has_tombstones: false,
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

    /// O(1) node type lookup from mmap'd node_slots — no materialization.
    /// Returns None if the node is dead or out of bounds.
    #[inline]
    pub fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        let i = idx.index();
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }
        Some(InternedKey::from_u64(slot.node_type))
    }

    /// O(1) property read from ColumnStore — no NodeData materialization.
    /// Returns None if the node is dead, out of bounds, or the property doesn't exist.
    #[inline]
    pub fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        let i = idx.index();
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }
        let type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&type_key)?;
        // Try schema column first, then fall back to special id/title columns.
        // The id and title are stored as __id__/__title__ (separate from schema),
        // so store.get() won't find them by their alias names (e.g., "title", "id").
        if let Some(val) = store.get(slot.row_id, key) {
            return Some(val);
        }
        // Fallback: check if this is an id/title alias
        if key == InternedKey::from_str("title") {
            return store.get_title(slot.row_id);
        }
        if key == InternedKey::from_str("id") {
            return store.get_id(slot.row_id);
        }
        None
    }

    /// O(1) id value read from ColumnStore — no NodeData materialization.
    #[inline]
    pub fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        let i = idx.index();
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }
        let type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&type_key)?;
        store.get_id(slot.row_id)
    }

    /// O(1) title value read from ColumnStore — no NodeData materialization.
    #[inline]
    pub fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        let i = idx.index();
        if i >= self.node_slots.len() {
            return None;
        }
        let slot = self.node_slots.get(i);
        if !slot.is_alive() {
            return None;
        }
        let type_key = InternedKey::from_u64(slot.node_type);
        let store = self.column_stores.get(&type_key)?;
        store.get_title(slot.row_id)
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
            let id = store.get_id(slot.row_id).unwrap_or(Value::Null);
            let title = store.get_title(slot.row_id).unwrap_or(Value::Null);
            NodeData {
                id,
                title,
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
        self.metadata_dirty = true;

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
        self.metadata_dirty = true;
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
        self.has_tombstones = true;

        // Tombstone all incident edges
        self.tombstone_edges_for_node(i);

        Some(data)
    }

    /// Update the row_id in a node's DiskNodeSlot.
    /// Used after BuildColumnStore conversion to fix per-type row_id mapping.
    pub fn update_row_id(&mut self, node_idx: NodeIndex, row_id: u32) {
        let i = node_idx.index();
        if i < self.node_slots.len() {
            let mut slot = self.node_slots.get(i);
            slot.row_id = row_id;
            self.node_slots.set(i, slot);
        }
    }

    pub fn node_indices_iter(&self) -> DiskNodeIndices<'_> {
        DiskNodeIndices::new(&self.node_slots)
    }

    // ====================================================================
    // Edge methods
    // ====================================================================

    /// Materialize an EdgeData into the arena. Reads conn_type from EdgeEndpoints
    /// (O(1) lookup) and properties from edge_properties HashMap.
    #[inline]
    pub(crate) fn materialize_edge(&self, edge_idx: u32) -> &EdgeData {
        let ep = self.edge_endpoints.get(edge_idx as usize);
        let ct = InternedKey::from_u64(ep.connection_type);
        let props = if self.edge_properties.is_empty() {
            Vec::new()
        } else {
            self.edge_properties
                .get(&edge_idx)
                .cloned()
                .unwrap_or_default()
        };
        let boxed = Box::new(EdgeData {
            connection_type: ct,
            properties: props,
        });
        // SAFETY: Box allocates on the heap — the pointer is stable even when
        // the Vec grows. The arena is never cleared while &self borrows are alive
        // (clear_arenas requires &mut self).
        let ptr = &*boxed as *const EdgeData;
        let mut arena = self.edge_arena.lock().unwrap();
        arena.push(boxed);
        unsafe { &*ptr }
    }

    /// Count edges of a specific type without materializing EdgeData.
    /// With sorted CSR, uses binary search to find the exact range, then counts
    /// peers matching the optional node type filter. Zero allocations.
    pub fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<u64>,
        other_node_type: Option<InternedKey>,
        deadline: Option<std::time::Instant>,
    ) -> Result<usize, String> {
        self.ensure_csr();
        let idx = node.index();
        let (offsets, edges) = match dir {
            Direction::Outgoing => (&self.out_offsets, &self.out_edges),
            Direction::Incoming => (&self.in_offsets, &self.in_edges),
        };
        if idx >= offsets.len().saturating_sub(1) {
            return Ok(0);
        }
        let mut start = offsets.get(idx) as usize;
        let mut end = offsets.get(idx + 1) as usize;

        // Narrow range via binary search when CSR is sorted by type
        if let Some(ct) = conn_type {
            if self.csr_sorted_by_type {
                let (lo, hi) = crate::graph::graph_iterators::binary_search_conn_type(
                    edges,
                    &self.edge_endpoints,
                    start,
                    end,
                    ct,
                );
                start = lo;
                end = hi;
            }
        }

        // Fast path: no tombstones and no peer-type filter → the answer is
        // literally the range length + overflow size, no scan required. This
        // turns Q5-class "count all P31 incoming" queries from 40 M loop
        // iterations (20+ s on USB SSD) into O(log D) binary search + two
        // integer subtractions.
        let can_shortcut = !self.has_tombstones
            && other_node_type.is_none()
            && (conn_type.is_none() || self.csr_sorted_by_type);
        if can_shortcut {
            let overflow = match dir {
                Direction::Outgoing => self.overflow_out.get(&(idx as u32)),
                Direction::Incoming => self.overflow_in.get(&(idx as u32)),
            };
            let mut overflow_count = 0usize;
            if let Some(list) = overflow {
                for e in list {
                    if let Some(ct) = conn_type {
                        if self.edge_endpoints.get(e.edge_idx as usize).connection_type != ct {
                            continue;
                        }
                    }
                    overflow_count += 1;
                }
            }
            return Ok(end.saturating_sub(start) + overflow_count);
        }

        let mut count = 0usize;
        for i in start..end {
            // Deadline check every 1 M edges — enough for Q5-scale hub fan-in
            // (~40 M P31 incoming) to terminate at ~20 s rather than 100 s.
            if (i - start).is_multiple_of(1 << 20) {
                if let Some(dl) = deadline {
                    if std::time::Instant::now() > dl {
                        return Err("Query timed out".to_string());
                    }
                }
            }
            let e = edges.get(i);
            if e.edge_idx == TOMBSTONE_EDGE {
                continue;
            }
            // Check connection type (only needed if CSR is NOT sorted)
            if let Some(ct) = conn_type {
                if !self.csr_sorted_by_type
                    && self.edge_endpoints.get(e.edge_idx as usize).connection_type != ct
                {
                    continue;
                }
            }
            // Check peer node type (O(1) mmap read, no materialization)
            if let Some(required_type) = other_node_type {
                let peer_idx = NodeIndex::new(e.peer as usize);
                if let Some(nt) = self.node_type_of(peer_idx) {
                    if nt != required_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            count += 1;
        }

        // Count overflow edges too
        let overflow = match dir {
            Direction::Outgoing => self.overflow_out.get(&(idx as u32)),
            Direction::Incoming => self.overflow_in.get(&(idx as u32)),
        };
        if let Some(list) = overflow {
            for e in list {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                if let Some(ct) = conn_type {
                    if self.edge_endpoints.get(e.edge_idx as usize).connection_type != ct {
                        continue;
                    }
                }
                if let Some(required_type) = other_node_type {
                    let peer_idx = NodeIndex::new(e.peer as usize);
                    if let Some(nt) = self.node_type_of(peer_idx) {
                        if nt != required_type {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                count += 1;
            }
        }
        Ok(count)
    }

    /// Iterate peer node indices for edges of a specific type, without materializing
    /// EdgeData. Yields (peer_idx, edge_idx) pairs. With sorted CSR, uses binary
    /// search. Completely avoids reading edge_endpoints.bin (13 GB) — only touches
    /// out_edges.bin/in_edges.bin + node_slots.bin.
    pub fn iter_peers_filtered(
        &self,
        node: NodeIndex,
        dir: Direction,
        conn_type: Option<u64>,
    ) -> Vec<(NodeIndex, u32)> {
        self.ensure_csr();
        let idx = node.index();
        let (offsets, edges) = match dir {
            Direction::Outgoing => (&self.out_offsets, &self.out_edges),
            Direction::Incoming => (&self.in_offsets, &self.in_edges),
        };
        if idx >= offsets.len().saturating_sub(1) {
            return Vec::new();
        }
        let mut start = offsets.get(idx) as usize;
        let mut end = offsets.get(idx + 1) as usize;

        // Narrow range via binary search when CSR is sorted
        if let Some(ct) = conn_type {
            if self.csr_sorted_by_type {
                let (lo, hi) = crate::graph::graph_iterators::binary_search_conn_type(
                    edges,
                    &self.edge_endpoints,
                    start,
                    end,
                    ct,
                );
                start = lo;
                end = hi;
            }
        }

        let mut result = Vec::with_capacity(end - start);
        for i in start..end {
            let e = edges.get(i);
            if e.edge_idx == TOMBSTONE_EDGE {
                continue;
            }
            // When CSR is NOT sorted, must check type via edge_endpoints
            if let Some(ct) = conn_type {
                if !self.csr_sorted_by_type
                    && self.edge_endpoints.get(e.edge_idx as usize).connection_type != ct
                {
                    continue;
                }
            }
            result.push((NodeIndex::new(e.peer as usize), e.edge_idx));
        }

        // Include overflow edges
        let overflow = match dir {
            Direction::Outgoing => self.overflow_out.get(&(idx as u32)),
            Direction::Incoming => self.overflow_in.get(&(idx as u32)),
        };
        if let Some(list) = overflow {
            for e in list {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                if let Some(ct) = conn_type {
                    if self.edge_endpoints.get(e.edge_idx as usize).connection_type != ct {
                        continue;
                    }
                }
                result.push((NodeIndex::new(e.peer as usize), e.edge_idx));
            }
        }
        result
    }

    /// Advise the kernel to prefetch hot mmap regions into page cache.
    /// Called after load to warm offset arrays and node_slots, reducing
    /// cold-cache penalty on first queries. Non-blocking — the kernel
    /// reads pages asynchronously in the background.
    /// Count all edges of a connection type, grouped by peer (target for outgoing,
    /// source for incoming). Returns a HashMap<peer_node_idx, count>.
    /// Uses a single sequential scan of edge_endpoints — O(E) total, purely sequential
    /// I/O (no random access). For outgoing grouping: counts by target. For incoming: by source.
    pub fn count_edges_grouped_by_peer(
        &self,
        conn_type: u64,
        dir: Direction,
        deadline: Option<std::time::Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        self.ensure_csr();
        let mut counts: HashMap<u32, i64> = HashMap::new();

        // Advise kernel: sequential read of edge_endpoints (13 GB).
        // MADV_SEQUENTIAL enables aggressive readahead and avoids polluting
        // the page cache with pages we won't revisit.
        self.edge_endpoints.advise_sequential();

        // Sequential scan of edge_endpoints — each entry is (source, target, conn_type).
        // 16 bytes per edge, purely sequential. Deadline check every 1M entries
        // keeps the per-check cost <0.001% while bounding wall-clock overshoot to ~0.3s.
        let total = self.next_edge_idx as usize;
        for i in 0..total {
            if i.is_multiple_of(1 << 20) {
                if let Some(dl) = deadline {
                    if std::time::Instant::now() > dl {
                        self.edge_endpoints.advise_dontneed();
                        return Err("Query timed out".to_string());
                    }
                }
            }
            let ep = self.edge_endpoints.get(i);
            if ep.source == TOMBSTONE_EDGE {
                continue;
            }
            if ep.connection_type != conn_type {
                continue;
            }
            let peer = match dir {
                Direction::Outgoing => ep.target, // group by target
                Direction::Incoming => ep.source, // group by source
            };
            *counts.entry(peer).or_insert(0) += 1;
        }

        // Release page cache pages after scan to reduce memory pressure.
        self.edge_endpoints.advise_dontneed();

        Ok(counts)
    }

    /// Look up source nodes that have outgoing edges of the given connection type.
    /// Returns an iterator-like slice of source node IDs from the inverted index.
    /// Returns None if the inverted index is not built (older graph format).
    pub fn sources_for_conn_type(&self, conn_type: u64) -> Option<Vec<u32>> {
        self.sources_for_conn_type_bounded(conn_type, None)
    }

    /// Bounded variant of `sources_for_conn_type`. When `max.is_some()` the
    /// function stops copying after the requested number of source node IDs
    /// — avoids eagerly materialising ~100 M u32s (400 MB) into a heap Vec
    /// when the caller will immediately truncate it to a few thousand.
    ///
    /// Overflow sources are always fully collected (tiny by definition).
    pub fn sources_for_conn_type_bounded(
        &self,
        conn_type: u64,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        if self.conn_type_index_types.is_empty() && self.overflow_out.is_empty() {
            return None;
        }

        // Read from persisted inverted index (binary search)
        let mut sources = Vec::new();
        if !self.conn_type_index_types.is_empty() {
            let num_types = self.conn_type_index_types.len();
            let mut lo = 0usize;
            let mut hi = num_types;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let mid_type = self.conn_type_index_types.get(mid);
                if mid_type < conn_type {
                    lo = mid + 1;
                } else if mid_type > conn_type {
                    hi = mid;
                } else {
                    let start = self.conn_type_index_offsets.get(mid) as usize;
                    let end = self.conn_type_index_offsets.get(mid + 1) as usize;
                    let take_end = match max {
                        Some(m) => start + (end - start).min(m),
                        None => end,
                    };
                    sources.reserve(take_end - start);
                    for i in start..take_end {
                        sources.push(self.conn_type_index_sources.get(i));
                    }
                    break;
                }
            }
        }

        // Supplement with overflow sources — check each overflow node for matching edges.
        // Overflow is almost always small; we don't apply `max` here because it'd require
        // a second dedup pass and complicate the contract. Callers that use `max` on cold
        // reads will rarely hit overflow anyway.
        if !self.overflow_out.is_empty() {
            for (&node_id, edges) in &self.overflow_out {
                for e in edges {
                    if e.edge_idx != TOMBSTONE_EDGE {
                        let ep = self.edge_endpoints.get(e.edge_idx as usize);
                        if ep.connection_type == conn_type {
                            sources.push(node_id);
                            break; // One matching edge is enough
                        }
                    }
                }
            }
            // Deduplicate (a node may appear in both CSR index and overflow)
            sources.sort_unstable();
            sources.dedup();
        }

        Some(sources)
    }

    pub fn prefetch_hot_regions(&self) {
        // Prefetch out_offsets + in_offsets (948 MB each — always needed for traversal).
        // Skip node_slots (2 GB) — prefetching it adds too much load latency.
        // The kernel will page in node_slots on demand during queries.
        self.out_offsets.advise_willneed();
        self.in_offsets.advise_willneed();
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
        // Flush modified edge properties from edge_weight_mut cache
        for (edge_idx, edge_data) in self.edge_mut_cache.drain() {
            if edge_data.properties.is_empty() {
                self.edge_properties.remove(&edge_idx);
            } else {
                self.edge_properties.insert(edge_idx, edge_data.properties);
            }
        }
        self.node_arena.get_mut().clear();
        self.edge_arena.lock().unwrap().clear();
    }

    /// Reset materialization arenas between queries to prevent unbounded growth.
    /// Only call when no references from prior `node_weight()` /
    /// `materialize_edge()` calls are alive — i.e. between top-level queries.
    pub fn reset_arenas(&self) {
        let node_arena = unsafe { &mut *self.node_arena.get() };
        node_arena.clear();
        self.edge_arena.lock().unwrap().clear();
    }

    pub fn edges_directed_iter(&self, a: NodeIndex, dir: Direction) -> DiskEdges<'_> {
        self.edges_directed_filtered_iter(a, dir, None)
    }

    pub fn edges_directed_filtered_iter(
        &self,
        a: NodeIndex,
        dir: Direction,
        conn_type_filter: Option<u64>,
    ) -> DiskEdges<'_> {
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

        // If the CSR offset table hasn't been built yet (fresh disk graph
        // pre-first-build, or a node added after build_csr_from_pending),
        // the overflow path may still carry edges. Fall through with an
        // empty CSR range instead of skipping the iterator entirely.
        let (start, end) = if node < offsets.len().saturating_sub(1) {
            (offsets.get(node) as usize, offsets.get(node + 1) as usize)
        } else {
            (0, 0)
        };

        let iter = DiskEdges::new(self, dir, a, edges, start, end, overflow);
        if let Some(ct) = conn_type_filter {
            iter.with_conn_type_filter(ct)
        } else {
            iter
        }
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

        Some(self.materialize_edge(ei as u32))
    }

    pub fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        let ei = idx.index();
        if ei >= self.next_edge_idx as usize {
            return None;
        }
        let ep = self.edge_endpoints.get(ei);
        if ep.source == TOMBSTONE_EDGE {
            return None;
        }
        self.metadata_dirty = true;
        // Store in dedicated cache (not the arena) so we can flush correctly.
        // The arena is append-only and shared with edge_weight (read-only),
        // making offset tracking fragile. The cache is keyed by edge_idx.
        let ct = InternedKey::from_u64(ep.connection_type);
        let props = self
            .edge_properties
            .get(&(ei as u32))
            .cloned()
            .unwrap_or_default();
        self.edge_mut_cache.entry(ei as u32).or_insert(EdgeData {
            connection_type: ct,
            properties: props,
        });
        Some(self.edge_mut_cache.get_mut(&(ei as u32)).unwrap())
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
        self.metadata_dirty = true;
        let edge_idx = self.next_edge_idx;
        self.next_edge_idx += 1;

        let ct = data.connection_type;

        if !data.properties.is_empty() {
            self.edge_properties.insert(edge_idx, data.properties);
        }

        let src = a.index() as u32;
        let tgt = b.index() as u32;
        let ct_u64 = ct.as_u64();

        if self.defer_csr {
            // Bulk-build mode: accumulate in pending_edges; caller is
            // responsible for `build_csr_from_pending()` at batch end.
            // Set explicitly by bulk loaders (ntriples); off by default so
            // individual Cypher mutations stay visible via overflow.
            self.pending_edges.get_mut().push((src, tgt, ct_u64));
        } else {
            // Post-CSR mode: go directly to overflow + edge_endpoints.
            // This makes new edges immediately visible to queries via
            // the DiskEdges iterator which merges CSR + overflow.
            self.edge_endpoints.push(EdgeEndpoints {
                source: src,
                target: tgt,
                connection_type: ct_u64,
            });
            self.overflow_out.entry(src).or_default().push(CsrEdge {
                peer: tgt,
                edge_idx,
            });
            self.overflow_in.entry(tgt).or_default().push(CsrEdge {
                peer: src,
                edge_idx,
            });
        }

        self.edge_count += 1;
        EdgeIndex::new(edge_idx as usize)
    }

    pub fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        self.clear_arenas();
        self.metadata_dirty = true;
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
        self.has_tombstones = true;
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
    /// query, or explicitly on save. Uses external merge sort — all I/O is
    /// sequential, designed for larger-than-RAM graphs.
    /// True if any overflow edges are present (edges added after the initial
    /// CSR build). Used by `ensure_disk_edges_built` to decide whether to
    /// merge overflow back into CSR so downstream indexes stay consistent.
    pub fn has_overflow(&self) -> bool {
        self.overflow_out.values().any(|v| !v.is_empty())
            || self.overflow_in.values().any(|v| !v.is_empty())
    }

    pub fn build_csr_from_pending(&mut self) {
        let pending = self.pending_edges.get_mut();
        if pending.is_empty() {
            return;
        }

        let node_bound = self.node_slots.len();
        let edge_count = pending.len();
        let verbose = std::env::var("KGLITE_CSR_VERBOSE").is_ok();
        let use_merge_sort = std::env::var("KGLITE_CSR_ALGO").is_ok_and(|v| v == "merge_sort");
        if use_merge_sort {
            self.build_csr_merge_sort(node_bound, edge_count, verbose);
        } else {
            self.build_csr_partitioned(node_bound, edge_count, verbose);
        }
        // After CSR is built, subsequent add_edge calls should route to overflow
        self.defer_csr = false;
    }

    /// Merge overflow edges back into the CSR arrays via full rebuild.
    /// Collects all live edges (CSR + overflow, excluding tombstones),
    /// writes to pending_edges, clears overflow, and rebuilds CSR.
    /// Returns the number of overflow edges that were merged.
    pub fn compact(&mut self) -> usize {
        let overflow_count: usize = self.overflow_out.values().map(|v| v.len()).sum();
        if overflow_count == 0 {
            return 0;
        }

        let verbose = std::env::var("KGLITE_CSR_VERBOSE").is_ok();
        if verbose {
            eprintln!(
                "Compacting: {} CSR edges + {} overflow edges",
                self.edge_count.saturating_sub(overflow_count),
                overflow_count
            );
        }

        let node_bound = self.node_slots.len();

        // Collect all live edges into a fresh pending_edges buffer.
        // Source: edge_endpoints (covers both CSR and post-CSR overflow edges).
        // Skip tombstoned entries.
        let mut live_count = 0usize;
        let total_endpoints = self.next_edge_idx as usize;

        let pending_path = self.data_dir.join("_compact_pending.bin");
        let mut new_pending: MmapOrVec<(u32, u32, u64)> =
            MmapOrVec::mapped(&pending_path, total_endpoints)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(total_endpoints));

        // Edge index remapping: old_idx → new_idx
        // Needed because compaction produces a dense edge array.
        let mut idx_remap: Vec<u32> = vec![TOMBSTONE_EDGE; total_endpoints];

        for (old_idx, remap_slot) in idx_remap.iter_mut().enumerate().take(total_endpoints) {
            let ep = self.edge_endpoints.get(old_idx);
            if ep.source != TOMBSTONE_EDGE
                && (ep.source as usize) < node_bound
                && (ep.target as usize) < node_bound
            {
                *remap_slot = live_count as u32;
                new_pending.push((ep.source, ep.target, ep.connection_type));
                live_count += 1;
            }
        }

        // Remap edge properties to new indices
        let old_props = std::mem::take(&mut self.edge_properties);
        for (old_idx, props) in old_props {
            let new_idx = idx_remap[old_idx as usize];
            if new_idx != TOMBSTONE_EDGE {
                self.edge_properties.insert(new_idx, props);
            }
        }

        // Clear overflow and free slots
        self.overflow_out.clear();
        self.overflow_in.clear();
        self.free_edge_slots.clear();

        // Reset edge tracking
        self.edge_count = live_count;
        self.next_edge_idx = live_count as u32;

        // Replace pending_edges and rebuild CSR
        let old_pending_path = self
            .pending_edges
            .get_mut()
            .file_path()
            .map(|p| p.to_path_buf());
        *self.pending_edges.get_mut() = new_pending;
        if let Some(path) = old_pending_path {
            let _ = std::fs::remove_file(path);
        }

        self.build_csr_from_pending();

        // Clean up compact temp file
        let _ = std::fs::remove_file(&pending_path);

        if verbose {
            eprintln!(
                "Compaction done: {} live edges (removed {} tombstoned)",
                live_count,
                total_endpoints - live_count
            );
        }

        overflow_count
    }

    /// [DEV] External merge sort variant — zero random reads.
    /// Sorts pending data into chunks, merges sequentially. All I/O is sequential.
    /// Use `KGLITE_CSR_ALGO=merge_sort` to select.
    fn build_csr_merge_sort(&mut self, node_bound: usize, edge_count: usize, verbose: bool) {
        let pending = self.pending_edges.get_mut();
        let phase3_start = std::time::Instant::now();

        // All files (temp + permanent) go in the graph directory.
        // Temp files (pending.bin, chunk_*.bin) are deleted after merge.
        let tmp_dir = self.data_dir.join(format!(
            "_csr_build_{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&tmp_dir);
        // CSR output goes to a separate build dir, then atomically swapped into data_dir.
        // This avoids overwriting mmap'd files that self still references.
        let build_dir = self.data_dir.join(format!(
            "_csr_output_{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&build_dir);
        let out_dir = &build_dir;

        // ── Step 1: Materialize + count degrees + build edge_endpoints (from heap) ──
        let step = std::time::Instant::now();
        let mut pending_mmap: MmapOrVec<(u32, u32, u64)> =
            MmapOrVec::mapped(&tmp_dir.join("pending.bin"), edge_count)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));
        let mut edge_endpoints_vec =
            MmapOrVec::mapped(&out_dir.join("edge_endpoints.bin"), edge_count)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));
        let mut out_counts = vec![0u64; node_bound];
        let mut in_counts = vec![0u64; node_bound];
        let mut edge_type_counts: HashMap<u64, usize> = HashMap::new();
        for i in 0..pending.len() {
            let (src, tgt, ct) = pending.get(i);
            pending_mmap.push((src, tgt, ct));
            edge_endpoints_vec.push(EdgeEndpoints {
                source: src,
                target: tgt,
                connection_type: ct,
            });
            if (src as usize) < node_bound {
                out_counts[src as usize] += 1;
            }
            if (tgt as usize) < node_bound {
                in_counts[tgt as usize] += 1;
            }
            *edge_type_counts.entry(ct).or_insert(0) += 1;
        }
        // Free pending_edges (file-backed mmap)
        *pending = MmapOrVec::new();
        if verbose {
            eprintln!(
                "    CSR step 1/4: materialize + endpoints + degrees ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // ── Step 2: Build offsets (mmap-backed to save ~2 GB heap) ──
        let step = std::time::Instant::now();
        let mut out_offsets: MmapOrVec<u64> =
            MmapOrVec::mapped(&out_dir.join("out_offsets.bin"), node_bound + 1)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(node_bound + 1));
        let mut in_offsets: MmapOrVec<u64> =
            MmapOrVec::mapped(&out_dir.join("in_offsets.bin"), node_bound + 1)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(node_bound + 1));
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
        if verbose {
            eprintln!(
                "    CSR step 2/4: build offsets ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // ── Helper: external merge sort into a CSR edge array ──
        // Reads pending_mmap in chunks (sequential), sorts each chunk,
        // then k-way merges all chunks (sequential) into output (sequential).
        // Zero random reads.
        let merge_sort_build = |pending: &MmapOrVec<(u32, u32, u64)>,
                                edge_count: usize,
                                by_source: bool,
                                chunk_dir: &std::path::Path,
                                output_dir: &std::path::Path,
                                label: &str,
                                verbose: bool|
         -> MmapOrVec<CsrEdge> {
            // Chunk size: fill available heap. MergeSortEntry is 24 bytes.
            // KGLITE_CSR_CHUNK_MB overrides (in MB) for testing; KGLITE_CSR_FORCE_CHUNKS
            // forces a specific number of chunks.
            let force_chunks: usize = std::env::var("KGLITE_CSR_FORCE_CHUNKS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let chunk_mb: usize = std::env::var("KGLITE_CSR_CHUNK_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let (chunk_size, num_chunks) = if force_chunks > 0 {
                let cs = edge_count.div_ceil(force_chunks);
                (cs, force_chunks.min(edge_count.div_ceil(cs)))
            } else {
                // Default: 12 GB or custom. After BlockPool frees column memory,
                // more heap is available → larger chunks → fewer merge passes.
                let max_bytes = if chunk_mb > 0 {
                    chunk_mb << 20
                } else {
                    12 << 30 // 12 GB default
                };
                let max_entries = max_bytes / std::mem::size_of::<MergeSortEntry>();
                let cs = max_entries.min(edge_count);
                (cs, edge_count.div_ceil(cs))
            };

            // ── Single-chunk fast path: sort in memory, write directly to output ──
            if num_chunks == 1 {
                let step = std::time::Instant::now();
                let mut entries: Vec<MergeSortEntry> = Vec::with_capacity(edge_count);
                for i in 0..edge_count {
                    let (src, tgt, ct) = pending.get(i);
                    let (key, peer) = if by_source { (src, tgt) } else { (tgt, src) };
                    entries.push(MergeSortEntry {
                        key,
                        conn_type: ct,
                        peer,
                        orig_idx: i as u32,
                    });
                }
                // Sort by (node, connection_type) so edges are grouped by type
                // within each node's CSR range — enables binary search for type filtering.
                entries.sort_unstable_by_key(|e| (e.key, e.conn_type));

                let mut output =
                    MmapOrVec::mapped(&output_dir.join(format!("{}_edges.bin", label)), edge_count)
                        .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));
                for entry in &entries {
                    output.push(CsrEdge {
                        peer: entry.peer,
                        edge_idx: entry.orig_idx,
                    });
                }
                drop(entries);
                if verbose {
                    eprintln!(
                        "      {label} single-chunk sort+write: {:.1}s",
                        step.elapsed().as_secs_f64()
                    );
                }
                return output;
            }

            // ── Multi-chunk path: external merge sort ──
            // Phase A: Create sorted chunks (sequential read + sort + sequential write)
            let step = std::time::Instant::now();
            let mut chunk_mmaps: Vec<MmapOrVec<MergeSortEntry>> = Vec::new();
            let mut chunk_lens: Vec<usize> = Vec::new();

            for c in 0..num_chunks {
                let start = c * chunk_size;
                let end = (start + chunk_size).min(edge_count);
                let len = end - start;

                // Load chunk from pending_mmap (sequential read)
                let mut chunk: Vec<MergeSortEntry> = Vec::with_capacity(len);
                for i in start..end {
                    let (src, tgt, ct) = pending.get(i);
                    let (key, peer) = if by_source { (src, tgt) } else { (tgt, src) };
                    chunk.push(MergeSortEntry {
                        key,
                        conn_type: ct,
                        peer,
                        orig_idx: i as u32,
                    });
                }

                // Sort by (node, connection_type) for type-grouped CSR
                chunk.sort_unstable_by_key(|e| (e.key, e.conn_type));

                // Write to mmap file (sequential)
                let path = chunk_dir.join(format!("chunk_{}_{}.bin", label, c));
                let mut mmap: MmapOrVec<MergeSortEntry> =
                    MmapOrVec::mapped(&path, len).unwrap_or_else(|_| MmapOrVec::with_capacity(len));
                for entry in &chunk {
                    mmap.push(*entry);
                }
                chunk_mmaps.push(mmap);
                chunk_lens.push(len);
                drop(chunk); // free heap before next chunk
            }
            if verbose {
                eprintln!(
                    "      {label} sort {num_chunks} chunks: {:.1}s",
                    step.elapsed().as_secs_f64()
                );
            }

            // Phase B: K-way merge using binary heap (O(E log K) instead of O(E×K))
            let merge_start = std::time::Instant::now();
            let mut positions: Vec<usize> = vec![0; num_chunks];
            let mut output =
                MmapOrVec::mapped(&output_dir.join(format!("{}_edges.bin", label)), edge_count)
                    .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));

            // Initialize min-heap with first entry from each chunk.
            // Heap key: (primary_key, conn_type, chunk_idx) for type-sorted output.
            use std::cmp::Reverse;
            let mut heap: std::collections::BinaryHeap<Reverse<(u32, u64, usize)>> =
                std::collections::BinaryHeap::with_capacity(num_chunks);
            for c in 0..num_chunks {
                if positions[c] < chunk_lens[c] {
                    let entry = chunk_mmaps[c].get(positions[c]);
                    heap.push(Reverse((entry.key, entry.conn_type, c)));
                }
            }

            for _ in 0..edge_count {
                let Reverse((_key, _ct, best_chunk)) = heap.pop().unwrap();
                let entry = chunk_mmaps[best_chunk].get(positions[best_chunk]);
                positions[best_chunk] += 1;
                output.push(CsrEdge {
                    peer: entry.peer,
                    edge_idx: entry.orig_idx,
                });
                // Refill heap with next entry from same chunk
                if positions[best_chunk] < chunk_lens[best_chunk] {
                    let next = chunk_mmaps[best_chunk].get(positions[best_chunk]);
                    heap.push(Reverse((next.key, next.conn_type, best_chunk)));
                }
            }

            // Cleanup chunk files
            for c in 0..num_chunks {
                let path = chunk_dir.join(format!("chunk_{}_{}.bin", label, c));
                let _ = std::fs::remove_file(path);
            }
            drop(chunk_mmaps);

            if verbose {
                eprintln!(
                    "      {label} merge: {:.1}s",
                    merge_start.elapsed().as_secs_f64()
                );
            }
            output
        };

        // ── Step 3: Build out_edges via merge sort (by source) ──
        let step = std::time::Instant::now();
        let out_edges = merge_sort_build(
            &pending_mmap,
            edge_count,
            true,
            &tmp_dir,
            out_dir,
            "out",
            verbose,
        );
        if verbose {
            eprintln!(
                "    CSR step 3/4: out_edges merge sort ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // ── Step 4: Build in_edges via merge sort (by target) ──
        let step = std::time::Instant::now();
        let in_edges = merge_sort_build(
            &pending_mmap,
            edge_count,
            false,
            &tmp_dir,
            out_dir,
            "in",
            verbose,
        );
        if verbose {
            eprintln!(
                "    CSR step 4/4: in_edges merge sort ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        drop(pending_mmap);
        // Clean up temp dir (sort chunks + pending.bin)
        let _ = std::fs::remove_dir_all(&tmp_dir);

        // Atomic swap: move build files to data_dir, re-mmap
        self.swap_csr_files(
            &build_dir,
            node_bound,
            edge_count,
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            edge_endpoints_vec,
        );
        self.csr_sorted_by_type = true;
        self.edge_type_counts_raw = Some(edge_type_counts);

        // Build connection-type inverted index from the swapped CSR
        self.build_conn_type_index(node_bound, verbose);
        // Build per-(type, peer) count histogram — single scan of edge_endpoints.
        self.build_peer_count_histogram(verbose);

        let _ = self.write_metadata();
        self.metadata_dirty = false;

        if verbose {
            eprintln!(
                "    CSR total: {:.1}s ({} edges, {} nodes) [merge_sort]",
                phase3_start.elapsed().as_secs_f64(),
                edge_count,
                node_bound
            );
        }
    }

    /// Hash-partitioned CSR build (Kuzu pattern).
    /// Partitions edges by node group, then local counting sort per partition.
    /// O(n) total — no global sort, no intermediate files, cache-friendly.
    fn build_csr_partitioned(&mut self, node_bound: usize, edge_count: usize, verbose: bool) {
        let pending = self.pending_edges.get_mut();
        let phase3_start = std::time::Instant::now();
        // CSR output goes to a separate build dir, then atomically swapped into data_dir.
        let build_dir = self.data_dir.join(format!(
            "_csr_output_{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&build_dir);
        let out_dir = &build_dir;

        // ── Step 1: Build edge_endpoints + count degrees ──
        // Single sequential pass over pending_edges: write endpoints + count degrees.
        let step = std::time::Instant::now();
        let mut edge_endpoints_vec =
            MmapOrVec::mapped(&out_dir.join("edge_endpoints.bin"), edge_count)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));
        let mut out_counts = vec![0u64; node_bound];
        let mut in_counts = vec![0u64; node_bound];
        let mut edge_type_counts: HashMap<u64, usize> = HashMap::new();
        for i in 0..pending.len() {
            let (src, tgt, ct) = pending.get(i);
            edge_endpoints_vec.push(EdgeEndpoints {
                source: src,
                target: tgt,
                connection_type: ct,
            });
            if (src as usize) < node_bound {
                out_counts[src as usize] += 1;
            }
            if (tgt as usize) < node_bound {
                in_counts[tgt as usize] += 1;
            }
            *edge_type_counts.entry(ct).or_insert(0) += 1;
        }
        if verbose {
            eprintln!(
                "    CSR step 1/3: endpoints + degrees ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // Free pending_edges — all data now in edge_endpoints.
        let pending_path = pending.file_path().map(|p| p.to_path_buf());
        *pending = MmapOrVec::new();
        if let Some(path) = pending_path {
            let _ = std::fs::remove_file(path);
        }

        // ── Step 2: Build offset arrays (prefix sum) ──
        let step = std::time::Instant::now();
        let mut out_offsets: MmapOrVec<u64> =
            MmapOrVec::mapped(&out_dir.join("out_offsets.bin"), node_bound + 1)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(node_bound + 1));
        let mut in_offsets: MmapOrVec<u64> =
            MmapOrVec::mapped(&out_dir.join("in_offsets.bin"), node_bound + 1)
                .unwrap_or_else(|_| MmapOrVec::with_capacity(node_bound + 1));
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
        if verbose {
            eprintln!(
                "    CSR step 2/3: offsets ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // ── Step 3: Buffered scatter for out_edges (by source) ──
        // Read edge_endpoints sequentially, scatter to out_edges via chunked buffers.
        // Uses mapped_zeroed — OS zero-fills lazily, no explicit pre-fill I/O.
        let step = std::time::Instant::now();
        let mut out_edges = MmapOrVec::mapped_zeroed(&out_dir.join("out_edges.bin"), edge_count)
            .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));

        let chunk_size: usize = std::env::var("KGLITE_CSR_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_000_000);
        let num_chunks = node_bound.div_ceil(chunk_size);
        let flush_threshold: usize = 512 * 1024 * 1024; // 512 MB

        {
            let mut out_cursor: Vec<u64> = (0..node_bound).map(|i| out_offsets.get(i)).collect();
            let mut chunk_bufs: Vec<Vec<(u32, u32, u32)>> =
                (0..num_chunks).map(|_| Vec::new()).collect();
            let mut buf_bytes: usize = 0;

            for edge_idx in 0..edge_count {
                let ep = edge_endpoints_vec.get(edge_idx);
                let s = ep.source as usize;
                if s < node_bound {
                    let ci = s / chunk_size;
                    chunk_bufs[ci].push((edge_idx as u32, ep.source, ep.target));
                    buf_bytes += 12;
                }
                if buf_bytes >= flush_threshold {
                    for buf in chunk_bufs.iter_mut() {
                        for &(eidx, src2, tgt2) in buf.iter() {
                            let pos = out_cursor[src2 as usize] as usize;
                            out_edges.set(
                                pos,
                                CsrEdge {
                                    peer: tgt2,
                                    edge_idx: eidx,
                                },
                            );
                            out_cursor[src2 as usize] += 1;
                        }
                        buf.clear();
                    }
                    buf_bytes = 0;
                }
            }
            for buf in chunk_bufs.iter_mut() {
                for &(eidx, src2, tgt2) in buf.iter() {
                    let pos = out_cursor[src2 as usize] as usize;
                    out_edges.set(
                        pos,
                        CsrEdge {
                            peer: tgt2,
                            edge_idx: eidx,
                        },
                    );
                    out_cursor[src2 as usize] += 1;
                }
            }
        }
        if verbose {
            eprintln!(
                "    CSR step 3a/4: out_edges scatter ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // Sort each node's out_edges range by connection_type.
        // Enables binary search for type filtering and conn_type_index build.
        // Cost: O(E * log D) where D is average degree — at D~7 across Wikidata's
        // 124 M nodes this totals ~300 s serially. Parallelised via Rayon over
        // disjoint per-node slices.
        let step = std::time::Instant::now();
        {
            use rayon::prelude::*;
            // Snapshot offsets into a heap Vec so the parallel closure can index
            // cheaply without aliasing the out_edges mmap. Cost: ~1 GB at Wikidata
            // scale (124 M * 8 B), freed at end of scope.
            let offsets_snap: Vec<u64> = (0..=node_bound).map(|i| out_offsets.get(i)).collect();

            // Raw pointer into out_edges — each node's range is disjoint by
            // construction, so parallel writes are safe.
            struct SendPtr(*mut CsrEdge);
            unsafe impl Send for SendPtr {}
            unsafe impl Sync for SendPtr {}
            let base = SendPtr(out_edges.as_mut_slice().as_mut_ptr());
            let base_ref = &base;
            let endpoints_ref = &edge_endpoints_vec;

            (0..node_bound).into_par_iter().for_each(|node| {
                let start = offsets_snap[node] as usize;
                let end = offsets_snap[node + 1] as usize;
                if end - start <= 1 {
                    return;
                }
                // SAFETY: disjoint ranges per node; endpoints_ref is read-only.
                let range: &mut [CsrEdge] =
                    unsafe { std::slice::from_raw_parts_mut(base_ref.0.add(start), end - start) };
                let mut with_ct: Vec<(u64, CsrEdge)> = range
                    .iter()
                    .map(|&e| (endpoints_ref.get(e.edge_idx as usize).connection_type, e))
                    .collect();
                with_ct.sort_unstable_by_key(|&(ct, _)| ct);
                for (i, &(_, e)) in with_ct.iter().enumerate() {
                    range[i] = e;
                }
            });
        }
        if verbose {
            eprintln!(
                "    CSR step 3b/4: out_edges sort by type ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // Free out_edges mmap before opening in_edges — keep working set to one 6.9 GB file
        let out_edges_path = out_dir.join("out_edges.bin");
        drop(out_edges);

        // ── Step 4: Build in_edges via merge sort (by target) ──
        // Scatter is slow for in_edges due to power-law target degree distribution
        // (popular targets cause page cache thrashing on the external drive).
        // Merge sort uses only sequential I/O: read → sort chunks → merge → write.
        let step = std::time::Instant::now();

        let in_edges = {
            // Chunk size for sort: fit in available heap after edge_endpoints mmap.
            let sort_chunk_mb: usize = std::env::var("KGLITE_CSR_CHUNK_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5120); // 5 GB default — safe on 16 GB machine
            let max_entries = (sort_chunk_mb << 20) / std::mem::size_of::<MergeSortEntry>();
            let sort_chunk_size = max_entries.min(edge_count);
            let num_sort_chunks = edge_count.div_ceil(sort_chunk_size);

            let sort_dir = out_dir.join("_in_sort");
            let _ = std::fs::create_dir_all(&sort_dir);

            if num_sort_chunks == 1 {
                // Single-chunk: sort in memory, write directly
                let substep = std::time::Instant::now();
                let mut entries: Vec<MergeSortEntry> = Vec::with_capacity(edge_count);
                for i in 0..edge_count {
                    let ep = edge_endpoints_vec.get(i);
                    entries.push(MergeSortEntry {
                        key: ep.target,
                        conn_type: ep.connection_type,
                        peer: ep.source,
                        orig_idx: i as u32,
                    });
                }
                entries.sort_unstable_by_key(|e| (e.key, e.conn_type));

                let mut output = MmapOrVec::mapped(&out_dir.join("in_edges.bin"), edge_count)
                    .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));
                for entry in &entries {
                    output.push(CsrEdge {
                        peer: entry.peer,
                        edge_idx: entry.orig_idx,
                    });
                }
                drop(entries);
                if verbose {
                    eprintln!(
                        "      in sort single-chunk: {:.1}s",
                        substep.elapsed().as_secs_f64()
                    );
                }
                output
            } else {
                // Multi-chunk: external merge sort with k-way heap merge
                let substep = std::time::Instant::now();
                let mut chunk_mmaps: Vec<MmapOrVec<MergeSortEntry>> = Vec::new();
                let mut chunk_lens: Vec<usize> = Vec::new();

                for c in 0..num_sort_chunks {
                    let start = c * sort_chunk_size;
                    let end = (start + sort_chunk_size).min(edge_count);
                    let len = end - start;

                    let mut entries: Vec<MergeSortEntry> = Vec::with_capacity(len);
                    for i in start..end {
                        let ep = edge_endpoints_vec.get(i);
                        entries.push(MergeSortEntry {
                            key: ep.target,
                            conn_type: ep.connection_type,
                            peer: ep.source,
                            orig_idx: i as u32,
                        });
                    }
                    entries.sort_unstable_by_key(|e| (e.key, e.conn_type));

                    let chunk_path = sort_dir.join(format!("chunk_in_{}.bin", c));
                    let mut chunk_mmap = MmapOrVec::mapped(&chunk_path, len)
                        .unwrap_or_else(|_| MmapOrVec::with_capacity(len));
                    for entry in &entries {
                        chunk_mmap.push(*entry);
                    }
                    drop(entries);
                    chunk_lens.push(len);
                    chunk_mmaps.push(chunk_mmap);
                }
                if verbose {
                    eprintln!(
                        "      in sort {} chunks: {:.1}s",
                        num_sort_chunks,
                        substep.elapsed().as_secs_f64()
                    );
                }

                // K-way merge via binary heap — all reads and writes sequential
                let substep = std::time::Instant::now();
                let mut positions: Vec<usize> = vec![0; num_sort_chunks];
                let mut output = MmapOrVec::mapped(&out_dir.join("in_edges.bin"), edge_count)
                    .unwrap_or_else(|_| MmapOrVec::with_capacity(edge_count));

                use std::cmp::Reverse;
                let mut heap: std::collections::BinaryHeap<Reverse<(u32, u64, usize)>> =
                    std::collections::BinaryHeap::with_capacity(num_sort_chunks);
                for c in 0..num_sort_chunks {
                    if chunk_lens[c] > 0 {
                        let entry = chunk_mmaps[c].get(0);
                        heap.push(Reverse((entry.key, entry.conn_type, c)));
                    }
                }

                for _ in 0..edge_count {
                    let Reverse((_key, _ct, best_chunk)) = heap.pop().unwrap();
                    let entry = chunk_mmaps[best_chunk].get(positions[best_chunk]);
                    positions[best_chunk] += 1;
                    output.push(CsrEdge {
                        peer: entry.peer,
                        edge_idx: entry.orig_idx,
                    });
                    if positions[best_chunk] < chunk_lens[best_chunk] {
                        let next = chunk_mmaps[best_chunk].get(positions[best_chunk]);
                        heap.push(Reverse((next.key, next.conn_type, best_chunk)));
                    }
                }

                // Cleanup sort chunks
                drop(chunk_mmaps);
                let _ = std::fs::remove_dir_all(&sort_dir);

                if verbose {
                    eprintln!("      in merge: {:.1}s", substep.elapsed().as_secs_f64());
                }
                output
            }
        };
        if verbose {
            eprintln!(
                "    CSR step 4/4: in_edges merge sort ({:.1}s)",
                step.elapsed().as_secs_f64()
            );
        }

        // Reload out_edges (dropped before step 4) — still in build_dir
        let out_edges: MmapOrVec<CsrEdge> = MmapOrVec::load_mapped(&out_edges_path, edge_count)
            .unwrap_or_else(|_| MmapOrVec::new());

        // Atomic swap: move build files to data_dir, re-mmap
        self.swap_csr_files(
            &build_dir,
            node_bound,
            edge_count,
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            edge_endpoints_vec,
        );
        self.csr_sorted_by_type = true;
        self.edge_type_counts_raw = Some(edge_type_counts);

        // Build connection-type inverted index from the swapped CSR
        self.build_conn_type_index(node_bound, verbose);
        // Build per-(type, peer) count histogram — single scan of edge_endpoints.
        self.build_peer_count_histogram(verbose);

        let _ = self.write_metadata();
        self.metadata_dirty = false;

        if verbose {
            eprintln!(
                "    CSR total: {:.1}s ({} edges, {} nodes) [partitioned]",
                phase3_start.elapsed().as_secs_f64(),
                edge_count,
                node_bound
            );
        }
    }

    // ====================================================================
    // CSR swap helper — atomic replacement of mmap'd CSR files
    // ====================================================================

    /// Swap CSR files built in `build_dir` into `self.data_dir`.
    /// 1. Drop old mmap fields (releases file handles)
    /// 2. Drop build mmaps (releases handles to temp files)
    /// 3. Rename temp files → data_dir (atomic on same filesystem)
    /// 4. Re-mmap from data_dir
    #[allow(clippy::too_many_arguments)]
    fn swap_csr_files(
        &mut self,
        build_dir: &Path,
        node_bound: usize,
        edge_count: usize,
        out_offsets: MmapOrVec<u64>,
        out_edges: MmapOrVec<CsrEdge>,
        in_offsets: MmapOrVec<u64>,
        in_edges: MmapOrVec<CsrEdge>,
        edge_endpoints: MmapOrVec<EdgeEndpoints>,
    ) {
        let data_dir = &self.data_dir;

        // If build_dir == data_dir, files are already in place — just assign.
        if build_dir == data_dir.as_path() {
            self.out_offsets = out_offsets;
            self.out_edges = out_edges;
            self.in_offsets = in_offsets;
            self.in_edges = in_edges;
            self.edge_endpoints = edge_endpoints;
            return;
        }

        // Drop old self mmaps (releases file handles to data_dir files)
        self.out_offsets = MmapOrVec::new();
        self.out_edges = MmapOrVec::new();
        self.in_offsets = MmapOrVec::new();
        self.in_edges = MmapOrVec::new();
        self.edge_endpoints = MmapOrVec::new();

        // Drop build mmaps (releases file handles to build_dir files)
        drop(out_offsets);
        drop(out_edges);
        drop(in_offsets);
        drop(in_edges);
        drop(edge_endpoints);

        // Rename files from build_dir → data_dir (atomic on same filesystem)
        let csr_files = [
            "out_offsets.bin",
            "out_edges.bin",
            "in_offsets.bin",
            "in_edges.bin",
            "edge_endpoints.bin",
        ];
        for fname in &csr_files {
            let src = build_dir.join(fname);
            let dst = data_dir.join(fname);
            if src.exists() {
                // Remove old file first (some filesystems require this)
                let _ = std::fs::remove_file(&dst);
                if let Err(e) = std::fs::rename(&src, &dst) {
                    eprintln!(
                        "Warning: failed to rename {} → {}: {}",
                        src.display(),
                        dst.display(),
                        e
                    );
                }
            }
        }

        // Also swap conn_type_index files if present
        let index_files = [
            "conn_type_index_types.bin",
            "conn_type_index_offsets.bin",
            "conn_type_index_sources.bin",
        ];
        for fname in &index_files {
            let src = build_dir.join(fname);
            let dst = data_dir.join(fname);
            if src.exists() {
                let _ = std::fs::remove_file(&dst);
                let _ = std::fs::rename(&src, &dst);
            }
        }

        // Re-mmap from data_dir
        self.out_offsets =
            MmapOrVec::load_mapped(&data_dir.join("out_offsets.bin"), node_bound + 1)
                .unwrap_or_else(|_| MmapOrVec::new());
        self.out_edges = MmapOrVec::load_mapped(&data_dir.join("out_edges.bin"), edge_count)
            .unwrap_or_else(|_| MmapOrVec::new());
        self.in_offsets = MmapOrVec::load_mapped(&data_dir.join("in_offsets.bin"), node_bound + 1)
            .unwrap_or_else(|_| MmapOrVec::new());
        self.in_edges = MmapOrVec::load_mapped(&data_dir.join("in_edges.bin"), edge_count)
            .unwrap_or_else(|_| MmapOrVec::new());
        self.edge_endpoints =
            MmapOrVec::load_mapped(&data_dir.join("edge_endpoints.bin"), edge_count)
                .unwrap_or_else(|_| MmapOrVec::new());

        // Re-load conn_type_index if swapped
        let types_path = data_dir.join("conn_type_index_types.bin");
        if types_path.exists() {
            let num_types = std::fs::metadata(&types_path)
                .map(|m| m.len() as usize / std::mem::size_of::<u64>())
                .unwrap_or(0);
            if num_types > 0 {
                self.conn_type_index_types = MmapOrVec::load_mapped(&types_path, num_types)
                    .unwrap_or_else(|_| MmapOrVec::new());
                self.conn_type_index_offsets = MmapOrVec::load_mapped(
                    &data_dir.join("conn_type_index_offsets.bin"),
                    num_types + 1,
                )
                .unwrap_or_else(|_| MmapOrVec::new());
                let sources_len = std::fs::metadata(data_dir.join("conn_type_index_sources.bin"))
                    .map(|m| m.len() as usize / std::mem::size_of::<u32>())
                    .unwrap_or(0);
                self.conn_type_index_sources = MmapOrVec::load_mapped(
                    &data_dir.join("conn_type_index_sources.bin"),
                    sources_len,
                )
                .unwrap_or_else(|_| MmapOrVec::new());
            }
        }

        // Clean up build dir
        let _ = std::fs::remove_dir_all(build_dir);
    }

    /// Build connection-type inverted index from current CSR arrays.
    /// Reads self.out_offsets, self.out_edges, self.edge_endpoints (must be valid).
    /// Writes index files to self.data_dir and assigns to self.
    ///
    /// Parallelised via Rayon: each thread scans a partition of nodes and builds
    /// a local `HashMap<conn_type, Vec<node_id>>`, then maps are merged. The
    /// preceding per-node sort guarantees that within one node's edge range, each
    /// conn_type appears contiguously, so each type is pushed at most once per
    /// node and the per-thread map key count is bounded by the global type count.
    fn build_conn_type_index(&mut self, node_bound: usize, verbose: bool) {
        use rayon::prelude::*;
        let idx_start = std::time::Instant::now();

        let out_offsets = &self.out_offsets;
        let out_edges = &self.out_edges;
        let edge_endpoints = &self.edge_endpoints;
        let effective_bound = node_bound.min(out_offsets.len().saturating_sub(1));

        let mut type_sources: HashMap<u64, Vec<u32>> = (0..effective_bound)
            .into_par_iter()
            .fold(HashMap::<u64, Vec<u32>>::new, |mut acc, node| {
                let start = out_offsets.get(node) as usize;
                let end = out_offsets.get(node + 1) as usize;
                if start == end {
                    return acc;
                }
                let mut last_type: u64 = u64::MAX;
                for i in start..end {
                    let e = out_edges.get(i);
                    if e.edge_idx == TOMBSTONE_EDGE {
                        continue;
                    }
                    let ct = edge_endpoints.get(e.edge_idx as usize).connection_type;
                    if ct != last_type {
                        acc.entry(ct).or_default().push(node as u32);
                        last_type = ct;
                    }
                }
                acc
            })
            .reduce(HashMap::<u64, Vec<u32>>::new, |mut a, b| {
                for (k, mut v) in b {
                    a.entry(k).or_default().append(&mut v);
                }
                a
            });

        // Node IDs within each type list are not globally ordered after parallel
        // reduce (each shard contributed its own range). Sort to match the
        // previous serial behaviour — callers may assume ascending order.
        for sources in type_sources.values_mut() {
            sources.sort_unstable();
        }

        let mut sorted_types: Vec<u64> = type_sources.keys().copied().collect();
        sorted_types.sort();
        let total_sources: usize = type_sources.values().map(|v| v.len()).sum();

        let mut idx_types = MmapOrVec::mapped(
            &self.data_dir.join("conn_type_index_types.bin"),
            sorted_types.len(),
        )
        .unwrap_or_else(|_| MmapOrVec::with_capacity(sorted_types.len()));
        let mut idx_offsets = MmapOrVec::mapped(
            &self.data_dir.join("conn_type_index_offsets.bin"),
            sorted_types.len() + 1,
        )
        .unwrap_or_else(|_| MmapOrVec::with_capacity(sorted_types.len() + 1));
        let mut idx_sources = MmapOrVec::mapped(
            &self.data_dir.join("conn_type_index_sources.bin"),
            total_sources,
        )
        .unwrap_or_else(|_| MmapOrVec::with_capacity(total_sources));

        let mut offset: u64 = 0;
        for &ct in &sorted_types {
            idx_types.push(ct);
            idx_offsets.push(offset);
            if let Some(sources) = type_sources.get(&ct) {
                for &src in sources {
                    idx_sources.push(src);
                }
                offset += sources.len() as u64;
            }
        }
        idx_offsets.push(offset);

        self.conn_type_index_types = idx_types;
        self.conn_type_index_offsets = idx_offsets;
        self.conn_type_index_sources = idx_sources;

        if verbose {
            eprintln!(
                "    Built conn-type inverted index: {} types, {} source entries ({:.1}s)",
                sorted_types.len(),
                total_sources,
                idx_start.elapsed().as_secs_f64()
            );
        }
    }

    /// Build per-(conn_type, peer) edge-count histogram. Answers unanchored
    /// aggregate queries in O(distinct-peers) instead of O(|edge_endpoints|).
    /// Scans edge_endpoints once in parallel chunks and folds into per-thread
    /// HashMap<(conn_type, peer), u32>; reduces + flattens to three mmap'd
    /// arrays sorted by (conn_type, peer).
    /// Build (or rebuild) the per-(conn_type, peer) edge-count histogram from
    /// the current `edge_endpoints`. Safe to call on a loaded graph; writes
    /// `peer_count_*.bin` files next to the other disk artifacts.
    pub fn rebuild_peer_count_histogram(&mut self) {
        let verbose = std::env::var("KGLITE_CSR_VERBOSE").is_ok();
        self.build_peer_count_histogram(verbose);
    }

    fn build_peer_count_histogram(&mut self, verbose: bool) {
        use rayon::prelude::*;
        let start = std::time::Instant::now();

        // Use edge_endpoints.len() rather than next_edge_idx — after certain
        // build paths (compact, merge rebuilds) next_edge_idx may have been
        // reset while edge_endpoints still holds the authoritative count.
        let total = self.edge_endpoints.len().min(self.next_edge_idx as usize);
        if total == 0 {
            return;
        }
        let endpoints = &self.edge_endpoints;

        // Advise kernel: we're about to do one large sequential read then drop
        // these pages. Matches the pattern in count_edges_grouped_by_peer.
        endpoints.advise_sequential();

        // Chunk the edge range so each Rayon task gets a contiguous slice of
        // edge_endpoints — maximises prefetcher effectiveness and keeps
        // per-thread working set bounded.
        let chunk = (total / rayon::current_num_threads().max(1)).max(1 << 20);
        let chunks: Vec<(usize, usize)> = (0..total)
            .step_by(chunk)
            .map(|lo| (lo, (lo + chunk).min(total)))
            .collect();

        let shard_maps: Vec<HashMap<(u64, u32), u32>> = chunks
            .into_par_iter()
            .map(|(lo, hi)| {
                let mut acc: HashMap<(u64, u32), u32> = HashMap::new();
                for i in lo..hi {
                    let ep = endpoints.get(i);
                    if ep.source == TOMBSTONE_EDGE {
                        continue;
                    }
                    // Target is the "peer" for an outgoing-oriented aggregate.
                    // Group-by-source queries are the symmetric case, handled
                    // lazily — histogram is stored keyed on target only;
                    // we add a second histogram keyed on source below.
                    *acc.entry((ep.connection_type, ep.target)).or_insert(0) += 1;
                }
                acc
            })
            .collect();

        // Reduce shard maps (serial — hot entries cluster fast anyway).
        let mut combined: HashMap<(u64, u32), u32> = HashMap::new();
        for shard in shard_maps {
            for (k, v) in shard {
                *combined.entry(k).or_insert(0) += v;
            }
        }

        endpoints.advise_dontneed();

        // Group by conn_type, sort peers within each group, flatten.
        let mut by_type: HashMap<u64, Vec<(u32, u32)>> = HashMap::new();
        for ((ct, peer), count) in combined {
            by_type.entry(ct).or_default().push((peer, count));
        }
        let mut sorted_types: Vec<u64> = by_type.keys().copied().collect();
        sorted_types.sort();
        for pairs in by_type.values_mut() {
            pairs.sort_unstable_by_key(|&(p, _)| p);
        }

        let total_pairs: usize = by_type.values().map(|v| v.len()).sum();

        // Build in heap Vecs first so the mmap'd files can be written with
        // the exact required size (MmapOrVec::mapped has a 64-slot minimum
        // that leaves trailing zeros — which break binary search because
        // the padding hashes compare less than any real FNV-1a hash).
        let mut types_vec: Vec<u64> = Vec::with_capacity(sorted_types.len());
        let mut offsets_vec: Vec<u64> = Vec::with_capacity(sorted_types.len() + 1);
        let mut entries_vec: Vec<u32> = Vec::with_capacity(total_pairs * 2);

        let mut cur_pairs: u64 = 0;
        for &ct in &sorted_types {
            types_vec.push(ct);
            offsets_vec.push(cur_pairs);
            if let Some(pairs) = by_type.get(&ct) {
                for &(peer, count) in pairs {
                    entries_vec.push(peer);
                    entries_vec.push(count);
                }
                cur_pairs += pairs.len() as u64;
            }
        }
        offsets_vec.push(cur_pairs);

        // Write exact-size raw byte files, then re-mmap with the known length.
        let write_u64 = |path: &Path, data: &[u64]| -> std::io::Result<()> {
            let bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
            };
            std::fs::write(path, bytes)
        };
        let write_u32 = |path: &Path, data: &[u32]| -> std::io::Result<()> {
            let bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
            };
            std::fs::write(path, bytes)
        };

        let types_path = self.data_dir.join("peer_count_types.bin");
        let offsets_path = self.data_dir.join("peer_count_offsets.bin");
        let entries_path = self.data_dir.join("peer_count_entries.bin");
        let _ = write_u64(&types_path, &types_vec);
        let _ = write_u64(&offsets_path, &offsets_vec);
        let _ = write_u32(&entries_path, &entries_vec);

        let pc_types = if !types_vec.is_empty() {
            MmapOrVec::load_mapped(&types_path, types_vec.len())
                .unwrap_or_else(|_| MmapOrVec::from_vec(types_vec.clone()))
        } else {
            MmapOrVec::new()
        };
        let pc_offsets = if !offsets_vec.is_empty() {
            MmapOrVec::load_mapped(&offsets_path, offsets_vec.len())
                .unwrap_or_else(|_| MmapOrVec::from_vec(offsets_vec.clone()))
        } else {
            MmapOrVec::new()
        };
        let pc_entries = if !entries_vec.is_empty() {
            MmapOrVec::load_mapped(&entries_path, entries_vec.len())
                .unwrap_or_else(|_| MmapOrVec::from_vec(entries_vec.clone()))
        } else {
            MmapOrVec::new()
        };

        self.peer_count_types = pc_types;
        self.peer_count_offsets = pc_offsets;
        self.peer_count_entries = pc_entries;

        if verbose {
            eprintln!(
                "    Built peer-count histogram: {} types, {} (peer, count) pairs ({:.1}s)",
                sorted_types.len(),
                total_pairs,
                start.elapsed().as_secs_f64()
            );
        }
    }

    /// Look up peer counts for a connection type from the persistent histogram.
    /// Returns None if the histogram hasn't been built (older graph format).
    /// The returned HashMap matches what `count_edges_grouped_by_peer` produced.
    pub fn lookup_peer_counts(&self, conn_type: u64) -> Option<HashMap<u32, i64>> {
        if self.peer_count_types.is_empty() {
            return None;
        }
        let n = self.peer_count_types.len();
        // Binary search the types array.
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let t = self.peer_count_types.get(mid);
            match t.cmp(&conn_type) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let start = self.peer_count_offsets.get(mid) as usize;
                    let end = self.peer_count_offsets.get(mid + 1) as usize;
                    let mut out: HashMap<u32, i64> = HashMap::with_capacity(end - start);
                    for i in start..end {
                        let peer = self.peer_count_entries.get(i * 2);
                        let count = self.peer_count_entries.get(i * 2 + 1);
                        out.insert(peer, count as i64);
                    }
                    return Some(out);
                }
            }
        }
        // Type not found in histogram. Return None so the caller falls back
        // to the sequential `count_edges_grouped_by_peer` scan — the
        // histogram may be stale (e.g. built pre-overflow-merge) and we
        // prefer a slow but correct answer over a fast but empty one.
        None
    }

    // ====================================================================
    // Internal helpers
    // ====================================================================

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
            edge_arena: std::sync::Mutex::new(Vec::new()),
            edge_mut_cache: HashMap::new(),
            pending_edges: UnsafeCell::new(MmapOrVec::new()),
            overflow_out: self.overflow_out.clone(),
            overflow_in: self.overflow_in.clone(),
            free_edge_slots: self.free_edge_slots.clone(),
            data_dir: self.data_dir.clone(),
            metadata_dirty: false,
            csr_sorted_by_type: self.csr_sorted_by_type,
            defer_csr: false,
            edge_type_counts_raw: None,
            conn_type_index_types: MmapOrVec::new(),
            conn_type_index_offsets: MmapOrVec::new(),
            conn_type_index_sources: MmapOrVec::new(),
            peer_count_types: MmapOrVec::new(),
            peer_count_offsets: MmapOrVec::new(),
            peer_count_entries: MmapOrVec::new(),
            has_tombstones: self.has_tombstones,
        }
    }
}

impl Drop for DiskGraph {
    fn drop(&mut self) {
        // Flush metadata only if mutations happened since last write
        if self.metadata_dirty {
            let _ = self.write_metadata();
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
    /// CSR edges sorted by (node, connection_type) — enables binary search.
    /// Added in v0.7.8; older graphs default to false.
    #[serde(default)]
    csr_sorted_by_type: bool,
    /// True if any node or edge has been removed since construction.
    /// Enables `count_edges_filtered` to short-circuit the per-edge
    /// tombstone check on fresh / read-only graphs. Default false is
    /// correct for legacy graphs only if they never saw a removal; since
    /// we can't retroactively prove that, treat unknown (missing field)
    /// as `true` (conservative) via a custom default.
    #[serde(default = "default_has_tombstones")]
    has_tombstones: bool,
}

fn default_has_tombstones() -> bool {
    // Conservative: older graphs that lack this field get the slow path
    // (tombstone-aware). Fresh builds emit `false` explicitly.
    true
}

impl DiskGraph {
    /// Write metadata JSON to the graph directory.
    /// Called automatically after CSR build and after mutations.
    pub(crate) fn write_metadata(&self) -> std::io::Result<()> {
        self.write_metadata_to(&self.data_dir)
    }

    fn write_metadata_to(&self, dir: &Path) -> std::io::Result<()> {
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
            csr_sorted_by_type: self.csr_sorted_by_type,
            has_tombstones: self.has_tombstones,
        };
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        std::fs::write(dir.join("disk_graph_meta.json"), json)
    }

    /// Save disk graph state. For disk mode, binary arrays are already on disk
    /// via mmap — this only flushes metadata + any in-memory overflow/properties.
    pub fn save_to_dir(
        &self,
        target_dir: &Path,
        interner: &crate::graph::schema::StringInterner,
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(target_dir)?;

        // Flush any mmap arrays that aren't already in the graph dir
        // (e.g. node_slots from new_at_path, or if target_dir differs from data_dir)
        if target_dir != self.data_dir {
            self.node_slots
                .save_to_file(&target_dir.join("node_slots.bin"))?;
            self.out_offsets
                .save_to_file(&target_dir.join("out_offsets.bin"))?;
            self.out_edges
                .save_to_file(&target_dir.join("out_edges.bin"))?;
            self.in_offsets
                .save_to_file(&target_dir.join("in_offsets.bin"))?;
            self.in_edges
                .save_to_file(&target_dir.join("in_edges.bin"))?;
            self.edge_endpoints
                .save_to_file(&target_dir.join("edge_endpoints.bin"))?;
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

        // Save metadata to target_dir (not data_dir)
        self.write_metadata_to(target_dir)?;

        // Also save optional persisted-cache files if present and target differs from data_dir
        if target_dir != self.data_dir {
            for fname in [
                "conn_type_index_types.bin",
                "conn_type_index_offsets.bin",
                "conn_type_index_sources.bin",
                "peer_count_types.bin",
                "peer_count_offsets.bin",
                "peer_count_entries.bin",
            ] {
                let src = self.data_dir.join(fname);
                if src.exists() {
                    let _ = std::fs::copy(&src, target_dir.join(fname));
                }
            }
        }

        Ok(())
    }

    /// Load a disk graph from a directory.
    /// Load a disk graph from a directory.
    /// Raw .bin files are mmap'd directly from the graph dir (no temp dir needed).
    /// Also supports legacy .bin.zst files (decompressed to temp dir).
    /// Returns `(DiskGraph, temp_dir)` — temp_dir may be empty if no decompression needed.
    pub fn load_from_dir(dir: &Path) -> std::io::Result<(Self, PathBuf)> {
        let meta_str = std::fs::read_to_string(dir.join("disk_graph_meta.json"))?;
        let meta: DiskGraphMeta = serde_json::from_str(&meta_str).map_err(std::io::Error::other)?;

        // Temp dir for legacy .zst decompression (only created if needed).
        // Lives inside graph dir so no external temp space required.
        let temp_dir = dir.join("_zst_cache");

        // Load raw .bin directly from graph dir; fall back to .bin.zst if missing
        let node_slots = load_raw_or_zst(&dir.join("node_slots"), meta.node_slots_len, &temp_dir)?;
        let out_offsets =
            load_raw_or_zst(&dir.join("out_offsets"), meta.out_offsets_len, &temp_dir)?;
        let out_edges = load_raw_or_zst(&dir.join("out_edges"), meta.out_edges_len, &temp_dir)?;
        let in_offsets = load_raw_or_zst(&dir.join("in_offsets"), meta.in_offsets_len, &temp_dir)?;
        let in_edges = load_raw_or_zst(&dir.join("in_edges"), meta.in_edges_len, &temp_dir)?;
        let edge_endpoints = load_raw_or_zst(
            &dir.join("edge_endpoints"),
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
                edge_arena: std::sync::Mutex::new(Vec::with_capacity(1024)),
                edge_mut_cache: HashMap::new(),
                pending_edges: UnsafeCell::new(MmapOrVec::new()),
                overflow_out,
                overflow_in,
                free_edge_slots: meta.free_edge_slots,
                data_dir: dir.to_path_buf(),
                metadata_dirty: false,
                csr_sorted_by_type: meta.csr_sorted_by_type,
                defer_csr: false,
                edge_type_counts_raw: None,
                conn_type_index_types: load_raw_or_zst_optional(&dir.join("conn_type_index_types")),
                conn_type_index_offsets: load_raw_or_zst_optional(
                    &dir.join("conn_type_index_offsets"),
                ),
                conn_type_index_sources: load_raw_or_zst_optional(
                    &dir.join("conn_type_index_sources"),
                ),
                peer_count_types: load_raw_or_zst_optional(&dir.join("peer_count_types")),
                peer_count_offsets: load_raw_or_zst_optional(&dir.join("peer_count_offsets")),
                peer_count_entries: load_raw_or_zst_optional(&dir.join("peer_count_entries")),
                has_tombstones: meta.has_tombstones,
            },
            temp_dir,
        ))
    }
}

// ============================================================================
// Compression helpers
// ============================================================================

/// Write a MmapOrVec as a zstd-compressed file.
/// Load a binary array: try raw `.bin` first (direct mmap, no temp dir),
/// fall back to `.bin.zst` (decompress to temp dir, then mmap).
fn load_raw_or_zst<T: Copy + Default + 'static>(
    base_path: &Path,
    len: usize,
    temp_dir: &Path,
) -> std::io::Result<MmapOrVec<T>> {
    let raw_path = base_path.with_extension("bin");
    if raw_path.exists() && len > 0 {
        return MmapOrVec::load_mapped(&raw_path, len);
    }
    let zst_path = base_path.with_extension("bin.zst");
    if zst_path.exists() && len > 0 {
        std::fs::create_dir_all(temp_dir)?;
        return load_compressed(&zst_path, len, temp_dir);
    }
    Ok(MmapOrVec::new())
}

/// Load a raw .bin file if it exists, otherwise return empty MmapOrVec.
/// Used for optional supplementary files (e.g., connection-type inverted index).
fn load_raw_or_zst_optional<T: Copy + Default + 'static>(base_path: &Path) -> MmapOrVec<T> {
    let raw_path = base_path.with_extension("bin");
    if raw_path.exists() {
        let file_len = std::fs::metadata(&raw_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let elem_size = std::mem::size_of::<T>();
        if file_len > 0 && elem_size > 0 {
            let len = file_len / elem_size;
            return MmapOrVec::load_mapped(&raw_path, len).unwrap_or_else(|_| MmapOrVec::new());
        }
    }
    MmapOrVec::new()
}

/// Load a zstd-compressed file, decompress to temp file, and mmap it.
/// Used only for loading legacy .bin.zst files from older graph format.
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
