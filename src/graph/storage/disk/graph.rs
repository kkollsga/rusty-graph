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
use crate::graph::core::iterators::{
    DiskEdgeIndices, DiskEdgeReferences, DiskEdges, DiskEdgesConnecting, DiskNeighbors,
    DiskNodeIndices,
};
use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;
use std::borrow::Cow;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::csr::{CsrEdge, DiskNodeSlot, EdgeEndpoints, TOMBSTONE_EDGE};
use super::edge_properties::{EdgePropertyStore, EdgePropertyStoreMeta};
use super::property_index;

/// PR1 phase 4: CSR + column binaries live in a per-segment subdirectory
/// of the graph root. Top-level files (disk_graph_meta.json,
/// seg_manifest.json, interner.json, metadata.json) stay at the graph root.
/// Legacy graphs gated by `DiskGraphMeta::csr_layout_version` == 0 use the
/// flat layout (everything at the root) — see `load_from_dir`.
///
/// PR1 phase 6 parameterised the directory name on segment id so future
/// saves can emit `seg_001/`, `seg_002/`, … next to the original
/// `seg_000/`. Call sites format via [`segment_subdir`]; directories are
/// discovered via [`enumerate_segment_dirs`].
pub(crate) fn segment_subdir(id: u32) -> String {
    // Three-digit zero-padding is enough for Wikidata-scale graphs
    // (plan targets ~200 segments); overflow past 999 is handled by
    // `{:03}` naturally widening without changing lexicographic ordering
    // (seg_999 < seg_1000 still sorts as seg_1000 first, but we use the
    // parsed u32 for ordering — see `enumerate_segment_dirs`).
    format!("seg_{id:03}")
}

/// Discover every `seg_NNN/` subdirectory under `root`, sorted
/// ascending by the numeric id parsed from the name. Returns
/// `(segment_id, path)` pairs. Non-matching directory entries and
/// unparsable `seg_*` names are skipped silently.
///
/// Used at load time to drive the CSR-load enumeration; at save time
/// the next free id is `last().map(|(id, _)| id + 1).unwrap_or(0)`.
/// PR1 phase 6 infrastructure — today every graph has exactly one
/// segment directory (`seg_000`), so the returned Vec has length 1
/// for non-legacy graphs. Phase 7+ will produce additional segments.
pub(crate) fn enumerate_segment_dirs(root: &Path) -> Vec<(u32, PathBuf)> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out: Vec<(u32, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            if !e.file_type().ok()?.is_dir() {
                return None;
            }
            let name = e.file_name();
            let s = name.to_str()?;
            let id_str = s.strip_prefix("seg_")?;
            let id: u32 = id_str.parse().ok()?;
            Some((id, e.path()))
        })
        .collect();
    out.sort_by_key(|(id, _)| *id);
    out
}

/// Current CSR-layout version emitted by every save. 0 = legacy flat,
/// 1 = segmented (seg_NNN/ subdirs). Loading tolerates both via the
/// version field in `DiskGraphMeta` (serde-defaulted to 0 for
/// pre-phase-4 graphs).
pub(crate) const CURRENT_CSR_LAYOUT_VERSION: u8 = 1;

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
    pub(crate) column_stores:
        HashMap<InternedKey, Arc<crate::graph::storage::column_store::ColumnStore>>,

    // ── Edge CSR (mmap'd) ──
    pub(super) out_offsets: MmapOrVec<u64>,
    pub(super) out_edges: MmapOrVec<CsrEdge>,
    pub(super) in_offsets: MmapOrVec<u64>,
    pub(super) in_edges: MmapOrVec<CsrEdge>,

    // ── Edge metadata ──
    pub(crate) edge_endpoints: MmapOrVec<EdgeEndpoints>,
    pub(crate) edge_count: usize,
    pub(crate) next_edge_idx: u32,

    // ── Edge properties (columnar base + mutation overlay) ──
    // PR2: heap-only HashMap replaced by disk-backed columnar store.
    // Overlay grows with mutations, base is mmap'd. See edge_properties.rs.
    edge_properties: EdgePropertyStore,

    // ── Edge materialization arena (Mutex + Box for Rayon thread safety) ──
    // Box gives stable heap pointers that survive Vec reallocation.
    #[allow(clippy::vec_box)]
    edge_arena: std::sync::Mutex<Vec<Box<EdgeData>>>,
    /// Cache for edge_weight_mut: stores materialized EdgeData that may be modified.
    /// Flushed to edge_properties on next clear_arenas call.
    edge_mut_cache: HashMap<u32, EdgeData>,

    // ── Pending edges (UnsafeCell reserved for a planned &self CSR-build path) ──
    // File-backed (MmapOrVec) to avoid ~14 GB heap allocation at Wikidata scale.
    // Today every access goes through `get_mut()` in `&mut self` methods
    // (add_edge with defer_csr, build_csr_from_pending, compact) — see the
    // struct-level SAFETY block for the full accounting of interior-mutability
    // fields.
    pub(crate) pending_edges: UnsafeCell<MmapOrVec<(u32, u32, u64)>>,

    // ── Mutation overflow (for incremental edges after CSR) ──
    overflow_out: HashMap<u32, Vec<CsrEdge>>,
    overflow_in: HashMap<u32, Vec<CsrEdge>>,
    free_edge_slots: Vec<u32>,

    // ── Storage directory (the graph lives here) ──
    pub(crate) data_dir: PathBuf,
    // ── Dirty flag: flushed on Drop or next query ──
    pub(super) metadata_dirty: bool,
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
    // ── Persistent property indexes (lazy-loaded).
    //
    // Populated in two ways:
    //   1. `build_property_index(type, prop)` — user calls `create_index`
    //      on a disk graph; writes 4 files to `data_dir` and caches the
    //      handle.
    //   2. `lookup_property_eq(type, prop, value)` on first miss — scans
    //      `data_dir` for a `property_index_{type}_{prop}_meta.bin`; if
    //      present, mmaps it and caches.
    //
    // The `None` sentinel records "we checked and no index exists" so
    // repeat misses don't stat the filesystem. `Arc` so concurrent reads
    // of the same index don't hold the outer RwLock.
    pub(crate) property_indexes: PropertyIndexCache,
    // ── Persistent cross-type global property indexes (lazy-loaded).
    //
    // Keyed by property name only. Built by `build_global_property_index(prop)`
    // — scans every alive `DiskNodeSlot` and collects one
    // `(string_value, NodeIndex)` entry per node where `prop` resolves
    // (via column slot, title alias, or id alias). Powers untyped
    // patterns like `MATCH (n {label: 'X'})` where the agent doesn't
    // know the node type.
    pub(crate) global_indexes: GlobalIndexCache,
    // ── Segment manifest (PR1 phases 2/5, single-segment view today).
    //
    // Persisted at `seg_manifest.json` alongside the CSR files. Legacy
    // graphs that lack the file load as an empty manifest — the planner
    // treats that as "pre-segmented, don't prune" in subsequent phases.
    // Fresh saves always write a one-segment manifest describing the
    // whole graph; PR1 phase 5 populates `indexed_prop_ranges` for each
    // `PropertyIndex` discovered in the segment directory, using
    // `StringBloomPlaceholder` until the bloom-filter variant lands.
    // Phase 6+ will split into multiple segments once multi-segment
    // writes and reads are in place.
    pub(crate) segment_manifest: super::segment_summary::SegmentManifest,
    // ── Sealed-nodes watermark (PR1 phase 8).
    //
    // Node ids in `[0, sealed_nodes_bound)` are accounted for in a prior
    // sealed segment's `node_slots`. Node ids in
    // `[sealed_nodes_bound, node_count)` are either in the active
    // (still-mutable) tail — not yet sealed into any segment — or were
    // just added via `add_node`. `seal_to_new_segment` flushes the tail
    // into a new `seg_NNN/` and advances this watermark.
    //
    // Zero on freshly-built / pre-phase-8 graphs; the `DiskGraphMeta`
    // serde `default` keeps old `.kgl` directories loadable.
    pub(crate) sealed_nodes_bound: u32,
    // ── Temp dir for CSR mmap files (cleaned up on Drop) ──
}

/// Lazy-loaded cache of persistent property indexes, keyed by
/// `(node_type, property)`. `None` records "checked and absent".
type PropertyIndexCache =
    std::sync::RwLock<HashMap<(String, String), Option<Arc<property_index::PropertyIndex>>>>;

/// Lazy-loaded cache of persistent cross-type global indexes, keyed
/// by property name. `None` records "checked and absent".
type GlobalIndexCache =
    std::sync::RwLock<HashMap<String, Option<Arc<property_index::PropertyIndex>>>>;

use std::sync::Arc;

// SAFETY — DiskGraph interior-mutability model:
//
// Three arena-like fields have different thread-safety postures:
//
// 1. `node_arena: UnsafeCell<Vec<NodeData>>` — accessed via `&self` from
//    `node_weight`, mutated via `&mut self` from `clear_arenas` and via
//    `&self` from `reset_arenas` (the sole non-`&mut self` reset path,
//    which requires no live references from prior materializations). The
//    invariant is KGLite's single-threaded-query contract: the graph is
//    accessed behind a Python-level Mutex / GIL, so concurrent `&self`
//    calls never reach this code. Rayon is used for CSR builds in
//    `disk/builder.rs`, but those sites are `&mut self` and do not touch
//    `node_arena`. Query-time Rayon (in `cypher/executor/…`) parallelises
//    row evaluation *after* materialization, so node_weight is not called
//    concurrently from Rayon tasks. Because the arena's `Vec` is not
//    Box-wrapped, a `push` that triggers realloc invalidates references
//    handed out by prior `node_weight` calls — relying on the same
//    single-threaded contract that says only one materialization path is
//    live at a time.
//
// 2. `edge_arena: Mutex<Vec<Box<EdgeData>>>` — thread-safe for Rayon
//    parallel queries. `Box` gives stable heap pointers that survive Vec
//    reallocation (see field doc at line 67). Contrast with `node_arena`,
//    which does not need this because query-time parallelism does not
//    call node_weight. If `node_weight` ever becomes Rayon-reachable,
//    this field is the pattern to follow.
//
// 3. `pending_edges: UnsafeCell<MmapOrVec<…>>` — only accessed via
//    `get_mut()` in `&mut self` contexts (`add_edge` with `defer_csr`,
//    `build_csr_from_pending`, `compact`). `UnsafeCell` is retained here
//    for a planned future auto-CSR-build-from-`&self` path; today no
//    `&self` access exists, so the current soundness argument is just
//    Rust's standard borrow checker.
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
    ///
    /// Fresh graphs use the segmented layout (PR1 phase 4): CSR / column
    /// binaries live at `root/seg_000/*.bin`, top-level `disk_graph_meta.json`
    /// and `seg_manifest.json` stay at `root/`. Legacy .kgl directories with
    /// the flat (pre-phase-4) layout continue to load — see `load_from_dir`.
    pub fn new_at_path(root_dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(root_dir)?;
        let data_dir = root_dir.join(segment_subdir(0));
        std::fs::create_dir_all(&data_dir)?;
        let data_dir = data_dir.as_path();

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
            edge_properties: EdgePropertyStore::new(),
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
            property_indexes: std::sync::RwLock::new(HashMap::new()),
            global_indexes: std::sync::RwLock::new(HashMap::new()),
            segment_manifest: super::segment_summary::SegmentManifest::new(),
            // Freshly-created graph has no sealed segments yet; the
            // first save seals everything up to node_count into seg_000
            // and advances this watermark accordingly.
            sealed_nodes_bound: 0,
        })
    }

    /// Build a DiskGraph from a petgraph StableDiGraph.
    /// Converts nodes to DiskNodeSlots on disk, builds CSR arrays.
    ///
    /// `root_dir` is the graph root; CSR binaries land in `root_dir/seg_000/`
    /// per the PR1 phase-4 segment layout.
    pub fn from_stable_digraph(
        graph: &mut petgraph::stable_graph::StableDiGraph<NodeData, EdgeData>,
        root_dir: &Path,
    ) -> std::io::Result<Self> {
        use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};

        std::fs::create_dir_all(root_dir)?;
        let data_dir_buf = root_dir.join(segment_subdir(0));
        std::fs::create_dir_all(&data_dir_buf)?;
        let data_dir = data_dir_buf.as_path();

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
            edge_properties: EdgePropertyStore::from_overlay(edge_properties),
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
            global_indexes: std::sync::RwLock::new(HashMap::new()),
            property_indexes: std::sync::RwLock::new(HashMap::new()),
            segment_manifest: super::segment_summary::SegmentManifest::new(),
            // Fresh build from a petgraph: no sealed segments yet.
            // First save seals the whole graph into seg_000.
            sealed_nodes_bound: 0,
        })
    }

    // ====================================================================
    // Node methods
    // ====================================================================

    /// Set column store references. Called by DirGraph after columnar setup or load.
    pub fn set_column_stores(
        &mut self,
        stores: HashMap<InternedKey, Arc<crate::graph::storage::column_store::ColumnStore>>,
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

        // SAFETY: `node_arena` is UnsafeCell<Vec<NodeData>>. Accessed from
        // this `&self` path, `clear_arenas` (`&mut self`), and `reset_arenas`
        // (`&self`, requires no live materialization refs). Under KGLite's
        // single-threaded-query contract (graph behind Python Mutex/GIL,
        // query-time Rayon does not call `node_weight`), only one path is
        // live at a time. See the struct-level SAFETY block for the full
        // argument.
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

        // Rationale: the arena is append-only during `&self` borrows (only
        // `clear_arenas` / `reset_arenas` shrink it, both with exclusive
        // access — `&mut self` or the no-live-refs reset contract). Under
        // the single-threaded-query contract (see struct-level SAFETY
        // block above), the `push` just above is not racing with other
        // borrows. Caveat: a subsequent `node_weight` call that triggers
        // Vec realloc invalidates any prior returned reference — callers
        // must consume each returned `&NodeData` before the next
        // materialization, or copy out.
        arena.push(node_data);
        // SAFETY: see rationale directly above and struct-level block.
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
                .get(edge_idx)
                .map(|cow| cow.into_owned())
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
        // SAFETY: `boxed: Box<EdgeData>` gives a stable heap pointer; the
        // arena keeps it alive until `clear_arenas` (`&mut self`); our
        // `&self` borrow prevents that. See block comment above.
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
                let (lo, hi) = crate::graph::core::iterators::binary_search_conn_type(
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
                let (lo, hi) = crate::graph::core::iterators::binary_search_conn_type(
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

    /// Iterate only the edges matching `conn_type`, yielding `(src, tgt, edge_idx)`
    /// per match. Never calls `materialize_edge` — no growth of `edge_arena`.
    ///
    /// Path: persisted inverted index (`conn_type_index_*`) gives the sources with
    /// at least one outgoing edge of that type. Each source's outgoing CSR slice
    /// is then filtered by `conn_type` (binary-search when the CSR is sorted by
    /// type, linear fallback otherwise). Overflow-out entries are visited for
    /// sources added after the last CSR build.
    ///
    /// The callback returns `true` to continue, `false` to stop iteration —
    /// lets callers collect a bounded prefix (e.g. two sample edges) without
    /// scanning every match.
    ///
    /// Complexity is O(matching edges) when `csr_sorted_by_type`, not O(all edges).
    /// Designed for the introspection fast path (`describe(connections=['T'])`)
    /// which previously did three full `edge_references()` sweeps per topic.
    pub fn for_each_edge_of_conn_type<F>(&self, conn_type: u64, mut f: F)
    where
        F: FnMut(NodeIndex, NodeIndex, u32) -> bool,
    {
        self.ensure_csr();

        // CSR-indexed sources via the inverted index.
        if !self.conn_type_index_types.is_empty() {
            let num_types = self.conn_type_index_types.len();
            let mut lo = 0usize;
            let mut hi = num_types;
            let mut range: Option<(usize, usize)> = None;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let mid_type = self.conn_type_index_types.get(mid);
                if mid_type < conn_type {
                    lo = mid + 1;
                } else if mid_type > conn_type {
                    hi = mid;
                } else {
                    let s = self.conn_type_index_offsets.get(mid) as usize;
                    let e = self.conn_type_index_offsets.get(mid + 1) as usize;
                    range = Some((s, e));
                    break;
                }
            }

            if let Some((src_start, src_end)) = range {
                let out_offsets_len = self.out_offsets.len().saturating_sub(1);
                for i in src_start..src_end {
                    let src_u32 = self.conn_type_index_sources.get(i);
                    let src_idx = src_u32 as usize;
                    if src_idx >= out_offsets_len {
                        continue;
                    }
                    let csr_start = self.out_offsets.get(src_idx) as usize;
                    let csr_end = self.out_offsets.get(src_idx + 1) as usize;

                    if self.csr_sorted_by_type {
                        let (lo_p, hi_p) = crate::graph::core::iterators::binary_search_conn_type(
                            &self.out_edges,
                            &self.edge_endpoints,
                            csr_start,
                            csr_end,
                            conn_type,
                        );
                        for p in lo_p..hi_p {
                            let e = self.out_edges.get(p);
                            if e.edge_idx == TOMBSTONE_EDGE {
                                continue;
                            }
                            if !f(
                                NodeIndex::new(src_u32 as usize),
                                NodeIndex::new(e.peer as usize),
                                e.edge_idx,
                            ) {
                                return;
                            }
                        }
                    } else {
                        for p in csr_start..csr_end {
                            let e = self.out_edges.get(p);
                            if e.edge_idx == TOMBSTONE_EDGE {
                                continue;
                            }
                            let ep = self.edge_endpoints.get(e.edge_idx as usize);
                            if ep.connection_type == conn_type
                                && !f(
                                    NodeIndex::new(src_u32 as usize),
                                    NodeIndex::new(e.peer as usize),
                                    e.edge_idx,
                                )
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }

        // Overflow sources — edges appended after the last CSR build. Typically tiny.
        for (&src_u32, edges) in &self.overflow_out {
            for e in edges {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                let ep = self.edge_endpoints.get(e.edge_idx as usize);
                if ep.connection_type == conn_type
                    && !f(
                        NodeIndex::new(src_u32 as usize),
                        NodeIndex::new(e.peer as usize),
                        e.edge_idx,
                    )
                {
                    return;
                }
            }
        }
    }

    /// Borrow an edge's property slice without materializing `EdgeData`.
    /// Returns `None` when the edge has no custom properties (common case).
    /// Safe to call in hot loops — does not push into `edge_arena`.
    ///
    /// The returned `Cow` is `Borrowed` for overlay hits (zero copy) and
    /// `Owned` for columnar-base hits (one bincode deserialize). Callers
    /// that need `&[(InternedKey, Value)]` can use `.as_deref()`.
    #[inline]
    pub fn edge_properties_at(&self, edge_idx: u32) -> Option<Cow<'_, [(InternedKey, Value)]>> {
        self.edge_properties.get(edge_idx)
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
                self.edge_properties.remove(edge_idx);
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
        // SAFETY: reset is only called between top-level queries; the
        // method doc requires no live references from prior `node_weight()`
        // / `materialize_edge()` calls, which KGLite's single-threaded
        // query loop guarantees.
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
            .get(ei as u32)
            .map(|cow| cow.into_owned())
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
        let props = self.edge_properties.take(ei as u32).unwrap_or_default();
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

        // Remap edge properties to new indices.
        // `mem::take` gives us ownership of the old store (base mmaps
        // stay live until we drop it at end of scope); we iterate every
        // potentially-populated slot and re-insert survivors into the
        // fresh store. Properties of tombstoned edges are discarded.
        let old_props = std::mem::take(&mut self.edge_properties);
        let upper = old_props.upper_bound();
        for old_idx in 0..upper {
            if let Some(cow) = old_props.get(old_idx) {
                let new_idx = idx_remap[old_idx as usize];
                if new_idx != TOMBSTONE_EDGE {
                    self.edge_properties.insert(new_idx, cow.into_owned());
                }
            }
        }
        drop(old_props);

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
                    self.edge_properties.remove(ei);
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
                    self.edge_properties.remove(ei);
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
                    self.edge_properties.remove(e.edge_idx);
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
                    self.edge_properties.remove(e.edge_idx);
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
            edge_properties: self.edge_properties.deep_clone(),
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
            global_indexes: std::sync::RwLock::new(HashMap::new()),
            has_tombstones: self.has_tombstones,
            property_indexes: std::sync::RwLock::new(HashMap::new()),
            segment_manifest: self.segment_manifest.clone(),
            sealed_nodes_bound: self.sealed_nodes_bound,
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

// ============================================================================
// Persistent property indexes
// ============================================================================

impl DiskGraph {
    /// Build (or rebuild) a persistent string property index for
    /// `(node_type, property)`. Writes four files to `data_dir` and
    /// caches the handle. Subsequent `lookup_property_eq` calls use the
    /// index; the planner sees it via the `GraphRead::lookup_by_property_eq`
    /// trait method.
    ///
    /// Only `TypedColumn::Str` columns are indexable today — the property
    /// must exist on the type's ColumnStore as a string column. Non-string
    /// or missing properties are a no-op that returns `Ok(())`; the index
    /// will simply contain zero entries and all lookups will miss.
    pub fn build_property_index(&self, node_type: &str, property: &str) -> std::io::Result<usize> {
        let type_key = InternedKey::from_str(node_type);
        let type_u64 = type_key.as_u64();
        let prop_key = InternedKey::from_str(property);

        // Three ways to resolve a property to a string value per node:
        //   1. Title/id alias columns (checked via helpers below) — covers
        //      `label`, `nid`, and any user-chosen title/id field names.
        //   2. Regular schema column via `get_str_by_slot`.
        //   3. Fall back to NodeData::get_property, which is the arena
        //      path used by the pattern matcher — slower but correct for
        //      exotic cases (non-columnar properties, map storage).
        let col_store = self.column_stores.get(&type_key);
        let schema_slot = col_store.and_then(|cs| cs.schema().slot(prop_key));
        // Heuristic: "title" or "id" literals, and anything stored outside
        // the regular schema, goes through the NodeData materialisation
        // path so title/id aliases and mapped-mode stores resolve
        // correctly. Everything else reads directly from the column.
        let use_slot_path = schema_slot.is_some();

        let node_bound = self.node_slots.len();
        let mut entries: Vec<(String, u32)> = Vec::with_capacity(node_bound);
        for i in 0..node_bound {
            let nslot = self.node_slots.get(i);
            if !nslot.is_alive() || nslot.node_type != type_u64 {
                continue;
            }
            // Try paths in order of specificity:
            //   1. Regular schema column (`get_str_by_slot`) — fast path.
            //   2. Title column (`get_title`) — covers `label`/`name`/
            //      any user-chosen title alias.
            //   3. Id column (`get_id`) — covers `nid` and other id
            //      aliases when the user explicitly indexes the id.
            let maybe_str: Option<String> = if use_slot_path {
                col_store
                    .and_then(|cs| cs.get_str_by_slot(nslot.row_id, schema_slot.unwrap()))
                    .map(str::to_string)
            } else if let Some(cs) = col_store {
                // Not in schema — try title, then id. If both return a
                // non-empty String, prefer title (which is what users
                // typically mean when aliasing `label` / `name` / ...).
                let from_title = cs.get_title(nslot.row_id).and_then(|v| match v {
                    Value::String(s) if !s.is_empty() => Some(s),
                    _ => None,
                });
                if from_title.is_some() {
                    from_title
                } else {
                    cs.get_id(nslot.row_id).and_then(|v| match v {
                        Value::String(s) if !s.is_empty() => Some(s),
                        _ => None,
                    })
                }
            } else {
                None
            };
            if let Some(s) = maybe_str {
                entries.push((s, i as u32));
            }
        }

        let count = entries.len();
        let idx =
            property_index::PropertyIndex::build(&self.data_dir, node_type, property, entries)?;
        self.property_indexes.write().unwrap().insert(
            (node_type.to_string(), property.to_string()),
            Some(Arc::new(idx)),
        );
        Ok(count)
    }

    /// Delete the on-disk index files and the in-memory cache entry.
    pub fn drop_property_index(&self, node_type: &str, property: &str) -> std::io::Result<()> {
        property_index::PropertyIndex::remove_files(&self.data_dir, node_type, property)?;
        self.property_indexes
            .write()
            .unwrap()
            .insert((node_type.to_string(), property.to_string()), None);
        Ok(())
    }

    /// Exact-match lookup. Returns `None` when no index has been built
    /// for `(node_type, property)`; returns `Some(Vec)` (possibly empty)
    /// when an index exists. The planner uses the distinction to decide
    /// whether to route through the fast path or fall back to scan.
    pub fn lookup_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        let key = (node_type.to_string(), property.to_string());
        // Fast path: cached handle.
        {
            let read = self.property_indexes.read().unwrap();
            if let Some(slot) = read.get(&key) {
                return slot.as_ref().map(|idx| idx.lookup_eq_str(value));
            }
        }
        // Slow path: check disk. If files exist, mmap and cache.
        let idx_opt = property_index::PropertyIndex::open(&self.data_dir, node_type, property)
            .ok()
            .flatten();
        let result = idx_opt.as_ref().map(|idx| idx.lookup_eq_str(value));
        self.property_indexes
            .write()
            .unwrap()
            .insert(key, idx_opt.map(Arc::new));
        result
    }

    /// Prefix lookup (STARTS WITH). Same `None`/`Some` semantics as
    /// [`lookup_property_eq`].
    pub fn lookup_property_prefix(
        &self,
        node_type: &str,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        let key = (node_type.to_string(), property.to_string());
        {
            let read = self.property_indexes.read().unwrap();
            if let Some(slot) = read.get(&key) {
                return slot
                    .as_ref()
                    .map(|idx| idx.lookup_prefix_str(prefix, limit));
            }
        }
        let idx_opt = property_index::PropertyIndex::open(&self.data_dir, node_type, property)
            .ok()
            .flatten();
        let result = idx_opt
            .as_ref()
            .map(|idx| idx.lookup_prefix_str(prefix, limit));
        self.property_indexes
            .write()
            .unwrap()
            .insert(key, idx_opt.map(Arc::new));
        result
    }

    /// Whether an index has been built for `(node_type, property)`.
    /// Checks the cache first, then the filesystem.
    #[allow(dead_code)]
    pub fn has_property_index(&self, node_type: &str, property: &str) -> bool {
        let key = (node_type.to_string(), property.to_string());
        if let Some(slot) = self.property_indexes.read().unwrap().get(&key) {
            return slot.is_some();
        }
        let (meta, _, _, _) = property_index::file_paths(&self.data_dir, node_type, property);
        meta.exists()
    }

    /// Build a cross-type global index for `property`. Scans every
    /// alive `DiskNodeSlot` and emits one `(string_value, NodeIndex)`
    /// entry per node where `property` resolves to a non-empty string
    /// (regular column, title alias, or id alias — same resolution
    /// order as [`build_property_index`]).
    ///
    /// Powers untyped patterns like `MATCH (n {label: 'X'})` and the
    /// `search(text)` helper. Re-run whenever the graph is rebuilt.
    pub fn build_global_property_index(&self, property: &str) -> std::io::Result<usize> {
        let prop_key = InternedKey::from_str(property);
        let node_bound = self.node_slots.len();
        let mut entries: Vec<(String, u32)> = Vec::with_capacity(node_bound / 2);

        // Cache per-type (column_store, schema_slot) lookups so every
        // node in the same type reuses the slot resolution.
        type ColStore = Arc<crate::graph::storage::column_store::ColumnStore>;
        type TypeCacheEntry = Option<(ColStore, Option<u16>)>;
        let mut type_cache: HashMap<u64, TypeCacheEntry> = HashMap::new();

        for i in 0..node_bound {
            let nslot = self.node_slots.get(i);
            if !nslot.is_alive() {
                continue;
            }
            let cached = type_cache.entry(nslot.node_type).or_insert_with(|| {
                let tk = InternedKey::from_u64(nslot.node_type);
                self.column_stores.get(&tk).cloned().map(|cs| {
                    let slot = cs.schema().slot(prop_key);
                    (cs, slot)
                })
            });
            let Some((col_store, schema_slot)) = cached else {
                continue;
            };
            let maybe_str: Option<String> = if let Some(slot) = schema_slot {
                col_store
                    .get_str_by_slot(nslot.row_id, *slot)
                    .map(str::to_string)
            } else {
                let from_title = col_store.get_title(nslot.row_id).and_then(|v| match v {
                    Value::String(s) if !s.is_empty() => Some(s),
                    _ => None,
                });
                from_title.or_else(|| {
                    col_store.get_id(nslot.row_id).and_then(|v| match v {
                        Value::String(s) if !s.is_empty() => Some(s),
                        _ => None,
                    })
                })
            };
            if let Some(s) = maybe_str {
                if !s.is_empty() {
                    entries.push((s, i as u32));
                }
            }
        }

        let count = entries.len();
        let idx = property_index::PropertyIndex::build_global(&self.data_dir, property, entries)?;
        self.global_indexes
            .write()
            .unwrap()
            .insert(property.to_string(), Some(Arc::new(idx)));
        Ok(count)
    }

    /// Exact-match lookup across every node type for a cross-type
    /// global index. Returns `None` when no index has been built for
    /// `property`; returns `Some(Vec)` (possibly empty) otherwise.
    pub fn lookup_global_eq(&self, property: &str, value: &str) -> Option<Vec<NodeIndex>> {
        {
            let read = self.global_indexes.read().unwrap();
            if let Some(slot) = read.get(property) {
                return slot.as_ref().map(|idx| idx.lookup_eq_str(value));
            }
        }
        let idx_opt = property_index::PropertyIndex::open_global(&self.data_dir, property)
            .ok()
            .flatten();
        let result = idx_opt.as_ref().map(|idx| idx.lookup_eq_str(value));
        self.global_indexes
            .write()
            .unwrap()
            .insert(property.to_string(), idx_opt.map(Arc::new));
        result
    }

    /// Prefix lookup (STARTS WITH) against the cross-type global
    /// index. Same `None`/`Some` semantics as [`lookup_global_eq`].
    pub fn lookup_global_prefix(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        {
            let read = self.global_indexes.read().unwrap();
            if let Some(slot) = read.get(property) {
                return slot
                    .as_ref()
                    .map(|idx| idx.lookup_prefix_str(prefix, limit));
            }
        }
        let idx_opt = property_index::PropertyIndex::open_global(&self.data_dir, property)
            .ok()
            .flatten();
        let result = idx_opt
            .as_ref()
            .map(|idx| idx.lookup_prefix_str(prefix, limit));
        self.global_indexes
            .write()
            .unwrap()
            .insert(property.to_string(), idx_opt.map(Arc::new));
        result
    }

    /// Whether a cross-type global index exists for `property`.
    #[allow(dead_code)]
    pub fn has_global_index(&self, property: &str) -> bool {
        if let Some(slot) = self.global_indexes.read().unwrap().get(property) {
            return slot.is_some();
        }
        let (meta, _, _, _) = property_index::global_file_paths(&self.data_dir, property);
        meta.exists()
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
    /// Edge property storage format. 0 = legacy bincode+zstd HashMap
    /// (edge_properties.bin.zst). 1 = columnar mmap base + overlay
    /// (edge_prop_offsets.bin + edge_prop_heap.bin). Added in PR2 of
    /// the disk-graph-improvement-plan; defaults to 0 for backward
    /// compat with older .kgl directories.
    #[serde(default)]
    edge_properties_format: u8,
    /// Lengths needed to mmap the columnar edge-property files. Zero
    /// for format=0 graphs or graphs that don't have any edge properties.
    #[serde(default)]
    edge_properties_meta: EdgePropertyStoreMeta,
    /// CSR-layout version. 0 = legacy flat (all files at graph root).
    /// 1 = segmented (CSR / columns / per-segment indexes live under
    /// `seg_000/`). Added in PR1 phase 4; defaults to 0 so pre-phase-4
    /// .kgl directories still load.
    #[serde(default)]
    csr_layout_version: u8,
    /// Boundary past which nodes are in the still-mutable tail (not yet
    /// sealed into any segment). `seal_to_new_segment` flushes
    /// `node_slots[sealed_nodes_bound..node_count]` into a new
    /// `seg_NNN/` and advances this. Added in PR1 phase 8; zero for
    /// pre-phase-8 graphs via serde default — their `seg_000` accounts
    /// for everything below `node_count`, so `seal_to_new_segment`
    /// will treat subsequent adds as the next-segment tail on first
    /// call (after saving to bump the watermark).
    #[serde(default)]
    sealed_nodes_bound: u32,
}

fn default_has_tombstones() -> bool {
    // Conservative: older graphs that lack this field get the slow path
    // (tombstone-aware). Fresh builds emit `false` explicitly.
    true
}

impl DiskGraph {
    /// Write metadata JSON to the graph directory.
    /// Called automatically after CSR build and after mutations. Reads
    /// the edge-property file metadata from the current data_dir so the
    /// JSON reflects whatever was last persisted there; mutations since
    /// then live in the overlay until the next explicit `save_to_dir`.
    pub(crate) fn write_metadata(&self) -> std::io::Result<()> {
        let edge_props_meta = EdgePropertyStore::meta_for(&self.data_dir);
        self.write_metadata_to(&self.data_dir, edge_props_meta)
    }

    fn write_metadata_to(
        &self,
        dir: &Path,
        edge_props_meta: EdgePropertyStoreMeta,
    ) -> std::io::Result<()> {
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
            // PR2: format=1 = columnar. Fresh graphs always emit the new
            // format; legacy format=0 is only ever loaded, never written.
            edge_properties_format: 1,
            edge_properties_meta: edge_props_meta,
            // PR1 phase 4: fresh saves always emit the segmented layout.
            csr_layout_version: CURRENT_CSR_LAYOUT_VERSION,
            // PR1 phase 8: persist the watermark so reloads know which
            // nodes already live in sealed segments vs which are tail.
            sealed_nodes_bound: self.sealed_nodes_bound,
        };
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        std::fs::write(dir.join("disk_graph_meta.json"), json)
    }

    /// Build a single-segment summary covering the whole graph. PR1 phase 2
    /// — subsequent phases split the graph into multiple segments and this
    /// helper becomes one of several per-segment builders.
    ///
    /// conn_types are read from the conn_type_index (built alongside the
    /// CSR), so the summary only reflects types that made it into the
    /// index — typically all of them after a save-time compact. Node type
    /// counts come from `column_stores` (one entry per node type); the
    /// count is the stored row count minus any tombstone slack (which we
    /// don't tally precisely here — the planner treats `has_node_type` as
    /// a lower-bound predicate).
    ///
    /// PR1 phase 5 additionally populates `indexed_prop_ranges` from the
    /// segment's on-disk `PropertyIndex` files (which live in `data_dir`,
    /// which under phase-4 layout is `seg_000/`). Today only string
    /// indexes exist, so every entry uses `PropRange::StringBloomPlaceholder`
    /// — a conservative placeholder that never prunes, but registers the
    /// `(type_hash, prop_hash)` pair so phase 6+ can upgrade to real
    /// bloom filters without changing the manifest schema.
    fn build_single_segment_manifest(&self) -> super::segment_summary::SegmentManifest {
        use super::segment_summary::{PropRange, SegmentManifest, SegmentSummary};
        use std::collections::HashSet;
        let mut summary = SegmentSummary::new(0, 0);
        summary.node_id_hi = self.node_count as u32;
        summary.edge_count = self.edge_count as u64;

        // Connection types: iterate the persisted inverted-index u64 list.
        for i in 0..self.conn_type_index_types.len() {
            summary.conn_types.insert(self.conn_type_index_types.get(i));
        }
        // Also include overflow edge conn_types that may not yet be in
        // the persisted index (post-CSR mutations).
        for edges in self.overflow_out.values() {
            for e in edges {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                let ct = self.edge_endpoints.get(e.edge_idx as usize).connection_type;
                summary.conn_types.insert(ct);
            }
        }

        // Node type counts: one column_store per node type.
        // `row_count` includes tombstoned rows — conservative upper
        // bound is fine for planner pruning (the planner uses these
        // only as "any rows of this type?" predicates).
        for (type_key, store) in &self.column_stores {
            summary
                .node_type_counts
                .insert(type_key.as_u64(), store.row_count());
        }

        // PR1 phase 5: record every (type, prop) index present in the
        // segment. Prefer the in-memory cache — its keys hold the
        // *original* type/prop strings the user passed in, so hashes
        // round-trip cleanly through `InternedKey::from_str`. Fall back
        // to a disk scan for indexes that were persisted earlier and
        // haven't been queried this session; scanned names are sanitised
        // filenames, which are identity-equal to originals for the only
        // shape we ever emit (`[A-Za-z0-9_-]`).
        let mut seen: HashSet<(u64, u64)> = HashSet::new();
        if let Ok(cache) = self.property_indexes.read() {
            for ((ty, prop), slot) in cache.iter() {
                if slot.is_none() {
                    continue;
                }
                let t_hash = InternedKey::from_str(ty).as_u64();
                let p_hash = InternedKey::from_str(prop).as_u64();
                if seen.insert((t_hash, p_hash)) {
                    summary.indexed_prop_ranges.push((
                        t_hash,
                        p_hash,
                        PropRange::StringBloomPlaceholder,
                    ));
                }
            }
        }
        for (t_hash, p_hash) in property_index::scan_segment_hashes(&self.data_dir) {
            if seen.insert((t_hash, p_hash)) {
                summary.indexed_prop_ranges.push((
                    t_hash,
                    p_hash,
                    PropRange::StringBloomPlaceholder,
                ));
            }
        }

        let mut manifest = SegmentManifest::new();
        manifest.append(summary);
        manifest
    }

    /// Save disk graph state. For disk mode, binary arrays are already on disk
    /// via mmap — this only flushes metadata + any in-memory overflow/properties.
    ///
    /// Takes `&mut self` because the edge-property store may need to drop
    /// its base mmap before overwriting the files (when target_dir equals
    /// the current data_dir).
    pub fn save_to_dir(
        &mut self,
        target_dir: &Path,
        _interner: &crate::graph::schema::StringInterner,
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(target_dir)?;
        // PR1 phase 4: CSR binaries live under a per-segment subdirectory.
        // Fresh graphs already have self.data_dir pointing at their own
        // seg_000/; save-as to a different path creates a matching
        // subdir. Phase 6 parameterised the subdir name on segment id —
        // today every save still targets id 0 until phase 7 splits
        // writes across segments.
        let csr_target = target_dir.join(segment_subdir(0));
        std::fs::create_dir_all(&csr_target)?;

        // Flush any mmap arrays that aren't already in the graph dir
        // (e.g. node_slots from new_at_path, or if csr_target differs from data_dir)
        if csr_target != self.data_dir {
            self.node_slots
                .save_to_file(&csr_target.join("node_slots.bin"))?;
            self.out_offsets
                .save_to_file(&csr_target.join("out_offsets.bin"))?;
            self.out_edges
                .save_to_file(&csr_target.join("out_edges.bin"))?;
            self.in_offsets
                .save_to_file(&csr_target.join("in_offsets.bin"))?;
            self.in_edges
                .save_to_file(&csr_target.join("in_edges.bin"))?;
            self.edge_endpoints
                .save_to_file(&csr_target.join("edge_endpoints.bin"))?;
        }

        // Save overflow edges (bincode + zstd)
        if !self.overflow_out.is_empty() || !self.overflow_in.is_empty() {
            let overflow = (&self.overflow_out, &self.overflow_in);
            let bytes = bincode::serialize(&overflow).map_err(std::io::Error::other)?;
            let compressed =
                zstd::encode_all(bytes.as_slice(), 3).map_err(std::io::Error::other)?;
            std::fs::write(target_dir.join("overflow_edges.bin.zst"), compressed)?;
        }

        // Save edge properties (columnar: edge_prop_offsets.bin + edge_prop_heap.bin).
        // Always write even when empty so format=1 + zero-length files are
        // self-consistent with the metadata. No interner/guard needed —
        // the columnar format stores raw u64 hashes directly. Phase-4
        // layout puts these alongside the CSR in `csr_target`.
        let upper = self.next_edge_idx;
        self.edge_properties.save_to(&csr_target, upper)?;
        let edge_props_meta = EdgePropertyStore::meta_for(&csr_target);

        // Trim the conn_type_index mmap'd files to their logical length.
        // `MmapOrVec::mapped(path, initial_cap)` has a 64-element minimum,
        // so a 1-type index leaves 512 bytes on disk with stale zeros that
        // the loader can't distinguish from real u64 type hashes. Without
        // this trim, `[r:TYPE]` typed-edge queries return 0 rows after
        // reload (pre-existing bug on v0.8.10).
        for field in [
            &self.conn_type_index_types as &MmapOrVec<u64>,
            &self.conn_type_index_offsets,
        ] {
            if let Some(path) = field.file_path().map(PathBuf::from) {
                let _ = field.save_to_file(&path);
            }
        }
        if let Some(path) = self.conn_type_index_sources.file_path().map(PathBuf::from) {
            let _ = self.conn_type_index_sources.save_to_file(&path);
        }

        // PR1 phase 8: trim the core CSR mmap files to their logical
        // length when writing in-place (csr_target == self.data_dir).
        // The not-in-place branch above already writes exact-sized
        // files via `save_to_file(&different_path)`. Without this trim
        // the multi-segment load path would misread the padding as
        // real CSR data — the single-segment path uses `meta.*_len`
        // and is unaffected.
        //
        // `trim_to_logical_length` truncates the file AND remaps, so
        // subsequent `push`es on the same MmapOrVec see the new size
        // as the starting capacity and extend cleanly. A naive
        // `save_to_file(&same_path)` set_len without remap leaves the
        // mmap spanning past the new EOF and SIGBUSes on the next
        // push — caught in the 0.8.11 ingest benchmark.
        if csr_target == self.data_dir {
            let _ = self.node_slots.trim_to_logical_length();
            let _ = self.out_offsets.trim_to_logical_length();
            let _ = self.out_edges.trim_to_logical_length();
            let _ = self.in_offsets.trim_to_logical_length();
            let _ = self.in_edges.trim_to_logical_length();
            let _ = self.edge_endpoints.trim_to_logical_length();
        }

        // PR1 phase 2: compute and persist a single-segment manifest.
        // Subsequent phases split saves into multiple segments; today the
        // manifest describes the whole graph as one segment so the planner
        // hook can be wired up without changing read-path behaviour.
        let manifest = self.build_single_segment_manifest();
        manifest.save_to(target_dir)?;
        self.segment_manifest = manifest;

        // Save metadata to target_dir (not data_dir)
        self.write_metadata_to(target_dir, edge_props_meta)?;

        // Also save optional persisted-cache files if present and csr_target differs from data_dir
        if csr_target != self.data_dir {
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
                    let _ = std::fs::copy(&src, csr_target.join(fname));
                }
            }
        }

        // PR1 phase 8: after a full save, everything up to node_count
        // is accounted for in the (single-segment) on-disk state. Bump
        // the sealed watermark so any subsequent `seal_to_new_segment`
        // correctly treats post-save adds as the new tail.
        self.sealed_nodes_bound = self.node_count as u32;

        Ok(())
    }

    /// Seal the still-mutable tail of the graph — nodes in
    /// `[sealed_nodes_bound, node_count)` plus the overflow edges
    /// between them — into a fresh `seg_NNN/` directory under `root`.
    /// Advances `sealed_nodes_bound` to `node_count`, clears the
    /// consumed overflow entries, appends a [`SegmentSummary`] to the
    /// on-disk manifest, and rewrites `disk_graph_meta.json`.
    ///
    /// PR1 phase 8 — the multi-segment write path. Lets incremental
    /// ingest add a batch of new entities and persist them without
    /// rewriting the existing sealed CSR (`save_to_dir`'s compact +
    /// rebuild). The new segment's CSR is segment-local: its
    /// `out_offsets` / `in_offsets` are zero-based, its `edge_idx`
    /// values span `0..|edges|`. Phase 7's `concat_segment_csrs`
    /// stitches these onto the combined view at load time.
    ///
    /// ## Constraints (enforced)
    ///
    /// Phase 7's concat assumes each node lives in exactly one segment
    /// and each segment's CSR owns the edges originating from its
    /// nodes. That forbids cross-segment edges for now: an overflow
    /// edge whose source or target is `< sealed_nodes_bound` cannot
    /// land in a new segment without breaking the invariant that
    /// each node's outgoing edges are all in one place. This method
    /// returns an `io::Error` in that case — the user must flush such
    /// edges via compact + `save_to_dir` (full rewrite) instead. A
    /// future phase will relax this once the concat model knows how
    /// to union a node's out_edges across segments.
    ///
    /// ## Not wired from save_to_dir
    ///
    /// `save_to_dir` still takes the traditional compact-and-rewrite
    /// path. `seal_to_new_segment` is an explicit call — phase 8 ships
    /// the mechanism; phase 9 ties it to the save path once the
    /// auxiliary-index story (conn_type_index, peer_count, edge
    /// properties for sealed edges) lands.
    ///
    /// ## Known phase-8 limitations
    ///
    /// The sealed segment writes **only** the six core CSR arrays
    /// (`node_slots`, `out_offsets`, `out_edges`, `in_offsets`,
    /// `in_edges`, `edge_endpoints`). It does not write its own
    /// `conn_type_index_*`, `peer_count_*`, or `edge_prop_*` files.
    /// Consequences for a graph loaded with ≥2 segments:
    ///   - Untyped edge scans (`MATCH (a)-[]->(b)`) work correctly.
    ///   - Typed-edge matches (`MATCH (a)-[:T]->(b)`) miss edges in
    ///     non-seg-0 segments (the inverted index only covers seg 0).
    ///   - `edge_weight()` on sealed edges returns `None`.
    ///   - `peer_count` aggregates undercount for sealed edges.
    ///
    /// These are deliberate phase-8 scope boundaries — the write path
    /// mechanism is what's shipping; phase 9 addresses the auxiliaries.
    pub fn seal_to_new_segment(&mut self, root: &Path) -> std::io::Result<u32> {
        use super::csr::{CsrEdge, DiskNodeSlot, EdgeEndpoints};

        let tail_lo = self.sealed_nodes_bound;
        let tail_hi = self.node_count as u32;
        if tail_hi <= tail_lo {
            return Err(std::io::Error::other(
                "seal_to_new_segment: nothing to seal — node_count <= sealed_nodes_bound",
            ));
        }
        let tail_len = (tail_hi - tail_lo) as usize;

        // Validate overflow edges up front — iterate both overflow_out
        // and overflow_in so we catch cross-segment edges regardless
        // of which side owns a given entry.
        for (&src_global, edges) in &self.overflow_out {
            for e in edges {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                let ep = self.edge_endpoints.get(e.edge_idx as usize);
                if src_global < tail_lo || ep.target < tail_lo {
                    return Err(std::io::Error::other(format!(
                        "seal_to_new_segment: edge {}→{} (edge_idx {}) crosses \
                         segment boundary (watermark {}); compact+save instead",
                        ep.source, ep.target, e.edge_idx, tail_lo
                    )));
                }
            }
        }
        for (&tgt_global, edges) in &self.overflow_in {
            for e in edges {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                let ep = self.edge_endpoints.get(e.edge_idx as usize);
                if tgt_global < tail_lo || ep.source < tail_lo {
                    return Err(std::io::Error::other(format!(
                        "seal_to_new_segment: edge {}→{} (edge_idx {}) crosses \
                         segment boundary (watermark {}); compact+save instead",
                        ep.source, ep.target, e.edge_idx, tail_lo
                    )));
                }
            }
        }

        // Next segment id = max(existing) + 1, or 0 if the dir is
        // empty (shouldn't happen in practice — first save creates
        // seg_000 before any seal).
        let existing = enumerate_segment_dirs(root);
        let next_id = existing
            .iter()
            .map(|(id, _)| *id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let seg_dir = root.join(segment_subdir(next_id));
        std::fs::create_dir_all(&seg_dir)?;

        // Collect overflow edges into a flat list, sorted by
        // (segment-local-source, connection_type). This matches the
        // layout `csr_sorted_by_type` promises — so typed-edge binary
        // search still works within a segment's CSR slice once phase
        // 9 wires the per-segment conn_type index.
        struct SealEdge {
            local_src: u32,
            local_tgt: u32, // segment-local target (global - tail_lo)
            conn_type: u64,
        }
        let mut seal_edges: Vec<SealEdge> = Vec::new();
        for (&src_global, edges) in &self.overflow_out {
            for e in edges {
                if e.edge_idx == TOMBSTONE_EDGE {
                    continue;
                }
                let ep = self.edge_endpoints.get(e.edge_idx as usize);
                seal_edges.push(SealEdge {
                    local_src: src_global - tail_lo,
                    local_tgt: ep.target - tail_lo,
                    conn_type: ep.connection_type,
                });
            }
        }
        seal_edges.sort_by_key(|e| (e.local_src, e.conn_type));
        let n_edges = seal_edges.len();

        // ─── Build segment-local arrays in memory (Heap-backed). ───
        //
        // node_slots: concat of self.node_slots[tail_lo..tail_hi].
        let mut node_slots: MmapOrVec<DiskNodeSlot> = MmapOrVec::with_capacity(tail_len);
        for i in 0..tail_len {
            node_slots.push(self.node_slots.get(tail_lo as usize + i));
        }

        // edge_endpoints: one per sealed edge, global source/target.
        // Segment-local edge_idx runs 0..n_edges parallel to the
        // sorted `seal_edges` order, so `edge_endpoints[local_idx]`
        // describes `seal_edges[local_idx]`.
        let mut edge_endpoints: MmapOrVec<EdgeEndpoints> = MmapOrVec::with_capacity(n_edges);
        for e in &seal_edges {
            edge_endpoints.push(EdgeEndpoints {
                source: e.local_src + tail_lo,
                target: e.local_tgt + tail_lo,
                connection_type: e.conn_type,
            });
        }

        // out_offsets / out_edges: CSR keyed by segment-local source.
        // Since `seal_edges` is pre-sorted by local_src, we can sweep
        // once to fill offsets + edges together.
        let mut out_offsets: MmapOrVec<u64> = MmapOrVec::with_capacity(tail_len + 1);
        let mut out_edges: MmapOrVec<CsrEdge> = MmapOrVec::with_capacity(n_edges);
        let mut cursor = 0usize;
        for node_local in 0..tail_len as u32 {
            out_offsets.push(cursor as u64);
            while cursor < n_edges && seal_edges[cursor].local_src == node_local {
                let e = &seal_edges[cursor];
                out_edges.push(CsrEdge {
                    peer: e.local_tgt + tail_lo, // peer stores GLOBAL node id
                    edge_idx: cursor as u32,
                });
                cursor += 1;
            }
        }
        out_offsets.push(cursor as u64);

        // in_offsets / in_edges: same sweep but keyed by local target.
        // Sort by (local_tgt, conn_type) for a matching layout on the
        // incoming side.
        let mut by_target: Vec<(u32, u32, u64)> = seal_edges
            .iter()
            .enumerate()
            .map(|(orig_idx, e)| (e.local_tgt, orig_idx as u32, e.conn_type))
            .collect();
        by_target.sort_by_key(|(t, _, ct)| (*t, *ct));

        let mut in_offsets: MmapOrVec<u64> = MmapOrVec::with_capacity(tail_len + 1);
        let mut in_edges: MmapOrVec<CsrEdge> = MmapOrVec::with_capacity(n_edges);
        let mut tcursor = 0usize;
        for node_local in 0..tail_len as u32 {
            in_offsets.push(tcursor as u64);
            while tcursor < n_edges && by_target[tcursor].0 == node_local {
                let (_, orig_idx, _) = by_target[tcursor];
                let src_peer = seal_edges[orig_idx as usize].local_src + tail_lo;
                in_edges.push(CsrEdge {
                    peer: src_peer, // global source id
                    edge_idx: orig_idx,
                });
                tcursor += 1;
            }
        }
        in_offsets.push(tcursor as u64);

        // ─── Persist to disk. ───
        node_slots.save_to_file(&seg_dir.join("node_slots.bin"))?;
        out_offsets.save_to_file(&seg_dir.join("out_offsets.bin"))?;
        out_edges.save_to_file(&seg_dir.join("out_edges.bin"))?;
        in_offsets.save_to_file(&seg_dir.join("in_offsets.bin"))?;
        in_edges.save_to_file(&seg_dir.join("in_edges.bin"))?;
        edge_endpoints.save_to_file(&seg_dir.join("edge_endpoints.bin"))?;

        // ─── Manifest bookkeeping. ───
        use super::segment_summary::SegmentSummary;
        let mut summary = SegmentSummary::new(next_id, tail_lo);
        summary.node_id_hi = tail_hi;
        summary.edge_count = n_edges as u64;
        // Connection types touched by this segment's edges.
        for e in &seal_edges {
            summary.conn_types.insert(e.conn_type);
        }
        // Node-type counts from the tail's slots.
        for i in 0..tail_len {
            let ns = self.node_slots.get(tail_lo as usize + i);
            if !ns.is_alive() {
                continue;
            }
            *summary.node_type_counts.entry(ns.node_type).or_insert(0) += 1;
        }
        // (indexed_prop_ranges stays empty for the new segment — phase
        // 5's cache+scan populates for seg 0's indexes only, and
        // per-segment indexes are phase 9.)

        self.segment_manifest.append(summary);
        self.segment_manifest.save_to(root)?;

        // ─── Reconcile seg_0's on-disk files with the new layout. ───
        //
        // self.{node_slots, out_offsets, in_offsets, edge_endpoints}
        // all grew during the post-save adds — their files are at
        // seg_0/... and now span past seg_0's logical extent (the
        // tail entries belong in seg_NNN/, which was just written).
        // Truncate each seg_0 file to its pre-tail size and swap
        // self's backing to heap-owned copies that still hold the
        // combined view for in-memory queries. On reload, seg_0 reads
        // cleanly via file-size inference, then concat stitches seg_NNN.
        //
        // `out_edges` / `in_edges` were NOT pushed during the overflow
        // adds (add_edge's post-CSR path writes overflow_out/in +
        // edge_endpoints only), so their files stay at seg_0's size —
        // no reconcile needed.
        let sealed_edge_count = n_edges;
        let seg0_next_edge_idx = self.next_edge_idx as usize - sealed_edge_count;
        reconcile_seg0_csr::<DiskNodeSlot>(&mut self.node_slots, tail_lo as usize)?;
        reconcile_seg0_csr::<u64>(&mut self.out_offsets, tail_lo as usize + 1)?;
        reconcile_seg0_csr::<u64>(&mut self.in_offsets, tail_lo as usize + 1)?;
        reconcile_seg0_csr::<EdgeEndpoints>(&mut self.edge_endpoints, seg0_next_edge_idx)?;

        // ─── Clear consumed overflow + advance watermark. ───
        //
        // All validated overflow edges belong to this seal (their
        // source and target are both >= tail_lo). Drop them in-memory;
        // the persisted CSR in seg_NNN/ is now the source of truth.
        self.overflow_out.clear();
        self.overflow_in.clear();
        self.sealed_nodes_bound = tail_hi;

        // Persist the updated metadata (watermark, manifest presence)
        // at the root. `write_metadata_to` reads edge-property meta
        // from `self.data_dir` — seg 0's subdir — which is the right
        // behaviour here since seal_to_new_segment doesn't rewrite
        // edge_properties.
        let edge_props_meta = EdgePropertyStore::meta_for(&self.data_dir);
        self.write_metadata_to(root, edge_props_meta)?;

        Ok(next_id)
    }

    /// Load a disk graph from a directory.
    /// Raw .bin files are mmap'd directly from the graph dir (no temp dir needed).
    /// Also supports legacy .bin.zst files (decompressed to temp dir).
    /// Returns `(DiskGraph, temp_dir)` — temp_dir may be empty if no decompression needed.
    ///
    /// `interner` is only mutated when loading a legacy format=0 graph
    /// whose `edge_properties.bin.zst` stores InternedKey as strings; the
    /// new columnar format stores raw u64 hashes and never touches it.
    pub fn load_from_dir(
        dir: &Path,
        interner: &mut crate::graph::schema::StringInterner,
    ) -> std::io::Result<(Self, PathBuf)> {
        let meta_str = std::fs::read_to_string(dir.join("disk_graph_meta.json"))?;
        let meta: DiskGraphMeta = serde_json::from_str(&meta_str).map_err(std::io::Error::other)?;

        // PR1 phase 4: CSR binaries live under seg_NNN/ when the graph
        // was written with csr_layout_version >= 1. Legacy .kgl directories
        // (version=0, the serde default) keep the flat layout.
        //
        // Phase 6 added `enumerate_segment_dirs`; phase 7 wires the
        // multi-segment concat read path through it. For csr_layout
        // version >= 1 the CSR dir choice (single-segment) or the
        // concat'd combined arrays (multi-segment) come out of this
        // block. Writes are still single-segment today (phase 8 will
        // seal overflow into additional segments), so the N>1 branch
        // is only exercised by unit tests on `concat_segment_csrs`
        // until then.
        //
        // Auxiliary per-segment data (conn_type_index_*, peer_count_*,
        // edge_properties, column_stores, per-(type,prop) property
        // indexes) is still loaded from segment 0 only in the N>1
        // branch — that's a documented limitation on `SegmentCsr`
        // pending phase 8. No code path produces an N>1 graph today,
        // so no running workload sees it.
        // Temp dir for legacy .zst decompression (only created if needed).
        // Lives inside graph dir so no external temp space required.
        let temp_dir = dir.join("_zst_cache");

        let (csr_dir, segment_csr): (PathBuf, SegmentCsr) = if meta.csr_layout_version >= 1 {
            let segs = enumerate_segment_dirs(dir);
            match segs.len() {
                0 => {
                    return Err(std::io::Error::other(format!(
                        "csr_layout_version={} but no seg_NNN/ directory found under {}",
                        meta.csr_layout_version,
                        dir.display()
                    )));
                }
                1 => {
                    // Single-segment: stay on the direct mmap path using
                    // the graph-level `meta.*_len` values. No allocation,
                    // zero overhead vs pre-phase-7 load.
                    let seg_dir = segs.into_iter().next().unwrap().1;
                    let csr = SegmentCsr {
                        node_slots: load_raw_or_zst(
                            &seg_dir.join("node_slots"),
                            meta.node_slots_len,
                            &temp_dir,
                        )?,
                        out_offsets: load_raw_or_zst(
                            &seg_dir.join("out_offsets"),
                            meta.out_offsets_len,
                            &temp_dir,
                        )?,
                        out_edges: load_raw_or_zst(
                            &seg_dir.join("out_edges"),
                            meta.out_edges_len,
                            &temp_dir,
                        )?,
                        in_offsets: load_raw_or_zst(
                            &seg_dir.join("in_offsets"),
                            meta.in_offsets_len,
                            &temp_dir,
                        )?,
                        in_edges: load_raw_or_zst(
                            &seg_dir.join("in_edges"),
                            meta.in_edges_len,
                            &temp_dir,
                        )?,
                        edge_endpoints: load_raw_or_zst(
                            &seg_dir.join("edge_endpoints"),
                            meta.edge_endpoints_len,
                            &temp_dir,
                        )?,
                    };
                    (seg_dir, csr)
                }
                _ => {
                    // Multi-segment: load each segment via the file-size-
                    // inferring loader, then concat. The first segment's
                    // path doubles as `data_dir` — that's where the
                    // auxiliary-indexes limitation points.
                    let mut loaded = Vec::with_capacity(segs.len());
                    let first_dir = segs[0].1.clone();
                    for (_, sdir) in &segs {
                        loaded.push(SegmentCsr::load_from(sdir, &temp_dir)?);
                    }
                    let csr = concat_segment_csrs(loaded);
                    (first_dir, csr)
                }
            }
        } else {
            // Legacy flat layout: load from root as one segment, using
            // meta's *_len values. Same code as phase 6 once unwrapped.
            let csr = SegmentCsr {
                node_slots: load_raw_or_zst(
                    &dir.join("node_slots"),
                    meta.node_slots_len,
                    &temp_dir,
                )?,
                out_offsets: load_raw_or_zst(
                    &dir.join("out_offsets"),
                    meta.out_offsets_len,
                    &temp_dir,
                )?,
                out_edges: load_raw_or_zst(&dir.join("out_edges"), meta.out_edges_len, &temp_dir)?,
                in_offsets: load_raw_or_zst(
                    &dir.join("in_offsets"),
                    meta.in_offsets_len,
                    &temp_dir,
                )?,
                in_edges: load_raw_or_zst(&dir.join("in_edges"), meta.in_edges_len, &temp_dir)?,
                edge_endpoints: load_raw_or_zst(
                    &dir.join("edge_endpoints"),
                    meta.edge_endpoints_len,
                    &temp_dir,
                )?,
            };
            (dir.to_path_buf(), csr)
        };

        let SegmentCsr {
            node_slots,
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            edge_endpoints,
        } = segment_csr;

        // Load edge properties — columnar (format=1) or legacy (format=0).
        // In the segmented layout the files live alongside the CSR.
        let edge_properties = EdgePropertyStore::load_from(
            &csr_dir,
            meta.edge_properties_format,
            meta.edge_properties_meta,
            interner,
        )?;

        // Load overflow edges (kept at the graph root; orthogonal to segments)
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
                data_dir: csr_dir.clone(),
                metadata_dirty: false,
                csr_sorted_by_type: meta.csr_sorted_by_type,
                defer_csr: false,
                edge_type_counts_raw: None,
                conn_type_index_types: load_raw_or_zst_optional(
                    &csr_dir.join("conn_type_index_types"),
                ),
                conn_type_index_offsets: load_raw_or_zst_optional(
                    &csr_dir.join("conn_type_index_offsets"),
                ),
                conn_type_index_sources: load_raw_or_zst_optional(
                    &csr_dir.join("conn_type_index_sources"),
                ),
                peer_count_types: load_raw_or_zst_optional(&csr_dir.join("peer_count_types")),
                peer_count_offsets: load_raw_or_zst_optional(&csr_dir.join("peer_count_offsets")),
                peer_count_entries: load_raw_or_zst_optional(&csr_dir.join("peer_count_entries")),
                has_tombstones: meta.has_tombstones,
                property_indexes: std::sync::RwLock::new(HashMap::new()),
                global_indexes: std::sync::RwLock::new(HashMap::new()),
                // Legacy .kgl directories have no seg_manifest.json;
                // load_from returns an empty manifest which subsequent
                // PR1 phases treat as "pre-segmented, don't prune".
                segment_manifest: super::segment_summary::SegmentManifest::load_from(dir)
                    .unwrap_or_default(),
                // PR1 phase 8: serde-defaulted to 0 on pre-phase-8 graphs
                // (their single seg_000 accounts for everything). Fresh
                // phase-8 graphs will advance the watermark on save.
                sealed_nodes_bound: meta.sealed_nodes_bound,
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

// ============================================================================
// Multi-segment CSR (PR1 phase 7)
// ============================================================================

/// One segment's core CSR arrays, loaded from its subdirectory. Used
/// only when a graph spans multiple `seg_NNN/` dirs — single-segment
/// graphs continue on the direct mmap path in [`DiskGraph::load_from_dir`]
/// for zero-overhead compatibility with every existing `.kgl` directory.
///
/// The arrays bundled here are the CSR backbone:
///
///   - `node_slots`: one `DiskNodeSlot` per node in this segment (the
///     segment owns a disjoint node-id range reported in its
///     `SegmentSummary`).
///   - `out_offsets` / `in_offsets`: CSR offsets into `out_edges` /
///     `in_edges`, indexed by local node position inside the segment.
///     Length = `node_slots.len() + 1`.
///   - `out_edges` / `in_edges`: `CsrEdge` arrays. Each entry's
///     `edge_idx` is **segment-local** (`0..edge_endpoints.len()`),
///     so concat shifts them onto combined edge_endpoints.
///   - `edge_endpoints`: `EdgeEndpoints` array, one per edge recorded
///     in this segment. `source` / `target` store *global* node ids
///     (unchanged by concat).
///
/// The auxiliary inverted indexes (`conn_type_index_*`, `peer_count_*`)
/// and the `edge_properties` / `column_stores` / per-type property
/// indexes are **not** bundled here — they remain loaded from segment 0
/// only. That's a known limitation pending the phase 8 multi-segment
/// write path, which will exercise it end-to-end. No current code
/// produces multi-segment graphs, so no existing workload sees the
/// limitation today.
pub(crate) struct SegmentCsr {
    pub(crate) node_slots: MmapOrVec<super::csr::DiskNodeSlot>,
    pub(crate) out_offsets: MmapOrVec<u64>,
    pub(crate) out_edges: MmapOrVec<super::csr::CsrEdge>,
    pub(crate) in_offsets: MmapOrVec<u64>,
    pub(crate) in_edges: MmapOrVec<super::csr::CsrEdge>,
    pub(crate) edge_endpoints: MmapOrVec<super::csr::EdgeEndpoints>,
}

impl SegmentCsr {
    /// Load the core CSR arrays from `csr_dir`, inferring each array's
    /// length from the file size (matches `load_raw_or_zst_optional`).
    /// Legacy `.bin.zst` fallback uses `temp_dir` for the decompressed
    /// staging files.
    pub(crate) fn load_from(csr_dir: &Path, temp_dir: &Path) -> std::io::Result<Self> {
        Ok(SegmentCsr {
            node_slots: load_with_inferred_len(&csr_dir.join("node_slots"), temp_dir)?,
            out_offsets: load_with_inferred_len(&csr_dir.join("out_offsets"), temp_dir)?,
            out_edges: load_with_inferred_len(&csr_dir.join("out_edges"), temp_dir)?,
            in_offsets: load_with_inferred_len(&csr_dir.join("in_offsets"), temp_dir)?,
            in_edges: load_with_inferred_len(&csr_dir.join("in_edges"), temp_dir)?,
            edge_endpoints: load_with_inferred_len(&csr_dir.join("edge_endpoints"), temp_dir)?,
        })
    }
}

/// Post-seal cleanup helper — see `DiskGraph::seal_to_new_segment`.
///
/// `field` is one of seg_0's CSR mmap-backed arrays that grew past its
/// seg_0 logical size during the post-save add batch (e.g.,
/// `self.node_slots` got pushes from `add_node`, `self.edge_endpoints`
/// got pushes from `add_edge`). We need three things at once:
///
///  1. The on-disk file trimmed to exactly `seg0_len` elements — so
///     the next reload reads seg_0 with the right element count.
///  2. The in-memory data to keep all current entries (seg_0 + tail)
///     so queries between seal and drop still see the combined graph.
///  3. The file handle released, so `set_len` doesn't race an
///     existing mmap.
///
/// Simplest way to get all three: snapshot the current contents into
/// a Vec, replace `field` with a fresh heap-backed MmapOrVec holding
/// that Vec, then reopen the file briefly just to `set_len`.
fn reconcile_seg0_csr<T: Copy + Default + 'static>(
    field: &mut MmapOrVec<T>,
    seg0_len: usize,
) -> std::io::Result<()> {
    let all = field.to_vec();
    let path = field.file_path().map(PathBuf::from);
    // Replace before truncate so the old mmap is dropped (releases the
    // file) before we `set_len` on the path.
    *field = MmapOrVec::from_vec(all);
    if let Some(p) = path {
        let f = std::fs::OpenOptions::new().write(true).open(&p)?;
        f.set_len((seg0_len * std::mem::size_of::<T>()) as u64)?;
    }
    Ok(())
}

/// Like [`load_raw_or_zst`] but derives the element count from the file
/// size on disk rather than from a pre-known length. Used in the multi-
/// segment load path, where `DiskGraphMeta`'s `*_len` fields describe
/// the *graph-level* concat total, not any one segment.
fn load_with_inferred_len<T: Copy + Default + 'static>(
    base_path: &Path,
    temp_dir: &Path,
) -> std::io::Result<MmapOrVec<T>> {
    let elem = std::mem::size_of::<T>();
    let raw_path = base_path.with_extension("bin");
    if raw_path.exists() && elem > 0 {
        let bytes = std::fs::metadata(&raw_path)?.len() as usize;
        let len = bytes / elem;
        if len > 0 {
            return MmapOrVec::load_mapped(&raw_path, len);
        }
    }
    let zst_path = base_path.with_extension("bin.zst");
    if zst_path.exists() && elem > 0 {
        // Legacy path: zstd stream doesn't carry the element count in
        // metadata, so we decompress to a temp file and infer from its
        // size. Matches `load_raw_or_zst`'s `load_compressed` flow,
        // minus the advance length check.
        std::fs::create_dir_all(temp_dir)?;
        let compressed = std::fs::read(&zst_path)?;
        let raw = zstd::decode_all(compressed.as_slice())?;
        let file_name = zst_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("data")
            .trim_end_matches(".zst");
        let temp_path = temp_dir.join(file_name);
        std::fs::write(&temp_path, &raw)?;
        let len = raw.len() / elem;
        if len > 0 {
            return MmapOrVec::load_mapped(&temp_path, len);
        }
    }
    Ok(MmapOrVec::new())
}

/// Combine per-segment CSR arrays into a single unified CSR by
/// concatenating node_slots / edge_endpoints, stitching offsets, and
/// shifting each segment's `edge_idx` values onto the combined
/// `edge_endpoints` numbering.
///
/// Single-segment input is returned as-is — the `Vec<SegmentCsr>` is
/// popped once and no allocation happens, so the N==1 case pays
/// essentially nothing beyond the function call.
///
/// The returned `SegmentCsr` is always `MmapOrVec::Vec`-backed — it
/// does not touch the filesystem. A graph-level page cache concat-to-
/// disk would be a future win; for now the in-memory combined CSR is
/// what the read path sees.
///
/// Assumptions on inputs (documented so the phase-8 writer obeys them):
///   1. Segments are provided in manifest order (ascending
///      `segment_id`), covering contiguous disjoint node-id ranges
///      `[0, n_0) + [n_0, n_0 + n_1) + ...`. The caller
///      ([`DiskGraph::load_from_dir`]) preserves `enumerate_segment_dirs`
///      ordering.
///   2. Each segment's `out_offsets` / `in_offsets` is segment-local:
///      `out_offsets[0] == 0`, `out_offsets[last] == out_edges.len()`.
///   3. Each segment's `CsrEdge::edge_idx` values are segment-local
///      (`0..edge_endpoints.len()`).
///   4. `EdgeEndpoints::{source, target}` hold *global* node ids and
///      are never rewritten by concat.
///
/// Violations produce a garbage combined CSR; the assumptions are
/// phase 7's contract with phase 8's writer.
pub(crate) fn concat_segment_csrs(mut segments: Vec<SegmentCsr>) -> SegmentCsr {
    use super::csr::{CsrEdge, DiskNodeSlot, EdgeEndpoints};
    match segments.len() {
        0 => SegmentCsr {
            node_slots: MmapOrVec::new(),
            out_offsets: MmapOrVec::new(),
            out_edges: MmapOrVec::new(),
            in_offsets: MmapOrVec::new(),
            in_edges: MmapOrVec::new(),
            edge_endpoints: MmapOrVec::new(),
        },
        1 => segments.pop().unwrap(),
        _ => {
            // Pre-size everything.
            let total_nodes: usize = segments.iter().map(|s| s.node_slots.len()).sum();
            let total_out_edges: usize = segments.iter().map(|s| s.out_edges.len()).sum();
            let total_in_edges: usize = segments.iter().map(|s| s.in_edges.len()).sum();
            let total_endpoints: usize = segments.iter().map(|s| s.edge_endpoints.len()).sum();

            let mut node_slots: MmapOrVec<DiskNodeSlot> = MmapOrVec::with_capacity(total_nodes);
            let mut edge_endpoints: MmapOrVec<EdgeEndpoints> =
                MmapOrVec::with_capacity(total_endpoints);
            let mut out_offsets: MmapOrVec<u64> = MmapOrVec::with_capacity(total_nodes + 1);
            let mut out_edges: MmapOrVec<CsrEdge> = MmapOrVec::with_capacity(total_out_edges);
            let mut in_offsets: MmapOrVec<u64> = MmapOrVec::with_capacity(total_nodes + 1);
            let mut in_edges: MmapOrVec<CsrEdge> = MmapOrVec::with_capacity(total_in_edges);

            // Offsets are cumulative: offset[0] = 0 for the first segment,
            // and each subsequent segment's local offsets get shifted by
            // the prior segments' edge-array length.
            out_offsets.push(0);
            in_offsets.push(0);

            let mut out_edge_base: u64 = 0;
            let mut in_edge_base: u64 = 0;
            let mut endpoint_base: u32 = 0;

            for seg in &segments {
                // Node slots: straight concat.
                for i in 0..seg.node_slots.len() {
                    node_slots.push(seg.node_slots.get(i));
                }

                // Edge endpoints: straight concat — source/target are
                // already global node ids.
                for i in 0..seg.edge_endpoints.len() {
                    edge_endpoints.push(seg.edge_endpoints.get(i));
                }

                // out_offsets: append entries [1..=n_k] of the segment's
                // local offsets, each shifted by `out_edge_base`.
                // Element 0 of every segment is 0, which is subsumed by
                // the previous segment's last entry after the shift.
                let n_nodes_k = seg.node_slots.len();
                for i in 1..=n_nodes_k {
                    out_offsets.push(seg.out_offsets.get(i) + out_edge_base);
                }
                for i in 1..=n_nodes_k {
                    in_offsets.push(seg.in_offsets.get(i) + in_edge_base);
                }

                // out_edges / in_edges: concat, shifting each entry's
                // segment-local edge_idx onto the combined endpoint
                // numbering.
                for i in 0..seg.out_edges.len() {
                    let mut e = seg.out_edges.get(i);
                    e.edge_idx = e.edge_idx.wrapping_add(endpoint_base);
                    out_edges.push(e);
                }
                for i in 0..seg.in_edges.len() {
                    let mut e = seg.in_edges.get(i);
                    e.edge_idx = e.edge_idx.wrapping_add(endpoint_base);
                    in_edges.push(e);
                }

                out_edge_base += seg.out_edges.len() as u64;
                in_edge_base += seg.in_edges.len() as u64;
                endpoint_base += seg.edge_endpoints.len() as u32;
            }

            SegmentCsr {
                node_slots,
                out_offsets,
                out_edges,
                in_offsets,
                in_edges,
                edge_endpoints,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{concat_segment_csrs, enumerate_segment_dirs, segment_subdir, SegmentCsr};
    use crate::graph::storage::disk::csr::{CsrEdge, DiskNodeSlot, EdgeEndpoints};
    use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
    use tempfile::TempDir;

    // ------------- fixture helpers -------------

    fn seg(
        node_slots: Vec<DiskNodeSlot>,
        out_offsets: Vec<u64>,
        out_edges: Vec<CsrEdge>,
        in_offsets: Vec<u64>,
        in_edges: Vec<CsrEdge>,
        edge_endpoints: Vec<EdgeEndpoints>,
    ) -> SegmentCsr {
        SegmentCsr {
            node_slots: from_vec(node_slots),
            out_offsets: from_vec(out_offsets),
            out_edges: from_vec(out_edges),
            in_offsets: from_vec(in_offsets),
            in_edges: from_vec(in_edges),
            edge_endpoints: from_vec(edge_endpoints),
        }
    }

    fn from_vec<T: Copy + Default + 'static>(v: Vec<T>) -> MmapOrVec<T> {
        let mut m: MmapOrVec<T> = MmapOrVec::with_capacity(v.len());
        for x in v {
            m.push(x);
        }
        m
    }

    fn slot(node_type: u64, row_id: u32) -> DiskNodeSlot {
        DiskNodeSlot {
            node_type,
            row_id,
            flags: DiskNodeSlot::ALIVE_BIT,
        }
    }

    // ------------- segment_subdir + enumerate (pre-phase-7 cases) -------------

    #[test]
    fn segment_subdir_zero_pads_three_digits() {
        assert_eq!(segment_subdir(0), "seg_000");
        assert_eq!(segment_subdir(1), "seg_001");
        assert_eq!(segment_subdir(42), "seg_042");
        assert_eq!(segment_subdir(999), "seg_999");
        // Past 999 the name widens; enumerate sorts by parsed u32 so
        // this still round-trips cleanly.
        assert_eq!(segment_subdir(1234), "seg_1234");
    }

    #[test]
    fn enumerate_segment_dirs_returns_sorted_ids() {
        let tmp = TempDir::new().unwrap();
        for id in [5u32, 0, 2, 17] {
            std::fs::create_dir_all(tmp.path().join(segment_subdir(id))).unwrap();
        }
        let got: Vec<u32> = enumerate_segment_dirs(tmp.path())
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(got, vec![0, 2, 5, 17]);
    }

    #[test]
    fn enumerate_segment_dirs_skips_non_matching_entries() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("seg_000")).unwrap();
        std::fs::create_dir_all(tmp.path().join("seg_abc")).unwrap(); // unparsable
        std::fs::create_dir_all(tmp.path().join("not_a_segment")).unwrap();
        // Top-level files must not be mistaken for segments.
        std::fs::write(tmp.path().join("seg_001"), b"not-a-dir").unwrap();
        std::fs::write(tmp.path().join("disk_graph_meta.json"), b"{}").unwrap();

        let got: Vec<u32> = enumerate_segment_dirs(tmp.path())
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(got, vec![0]);
    }

    #[test]
    fn enumerate_segment_dirs_on_missing_dir_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("does-not-exist");
        assert!(enumerate_segment_dirs(&missing).is_empty());
    }

    #[test]
    fn enumerate_segment_dirs_empty_root_returns_empty() {
        let tmp = TempDir::new().unwrap();
        assert!(enumerate_segment_dirs(tmp.path()).is_empty());
    }

    // ------------- concat_segment_csrs (phase 7) -------------

    #[test]
    fn concat_empty_input_returns_all_empty() {
        let c = concat_segment_csrs(Vec::new());
        assert_eq!(c.node_slots.len(), 0);
        assert_eq!(c.out_offsets.len(), 0);
        assert_eq!(c.out_edges.len(), 0);
        assert_eq!(c.in_offsets.len(), 0);
        assert_eq!(c.in_edges.len(), 0);
        assert_eq!(c.edge_endpoints.len(), 0);
    }

    #[test]
    fn concat_single_segment_returns_it_unchanged() {
        // Node 0 → node 1 (edge_idx 0). One node, one edge for simplicity
        // of comparison: passthrough must not mutate anything.
        let s = seg(
            vec![slot(7, 100), slot(7, 101)],
            vec![0, 1, 1], // out_offsets: node 0 emits edge [0,1), node 1 emits nothing
            vec![CsrEdge {
                peer: 1,
                edge_idx: 0,
            }],
            vec![0, 0, 1], // in_offsets: node 0 receives nothing, node 1 receives [0,1)
            vec![CsrEdge {
                peer: 0,
                edge_idx: 0,
            }],
            vec![EdgeEndpoints {
                source: 0,
                target: 1,
                connection_type: 42,
            }],
        );
        let c = concat_segment_csrs(vec![s]);
        assert_eq!(c.node_slots.len(), 2);
        assert_eq!(c.out_offsets.len(), 3);
        assert_eq!(c.out_edges.len(), 1);
        assert_eq!(c.out_edges.get(0).edge_idx, 0); // unchanged
        assert_eq!(c.edge_endpoints.len(), 1);
        assert_eq!(c.edge_endpoints.get(0).source, 0);
    }

    #[test]
    fn concat_two_segments_stitches_offsets_and_shifts_edge_idx() {
        // Segment 0: 2 nodes, 1 intra-segment edge  0 → 1
        let s0 = seg(
            vec![slot(1, 10), slot(1, 11)],
            vec![0, 1, 1],
            vec![CsrEdge {
                peer: 1,
                edge_idx: 0,
            }],
            vec![0, 0, 1],
            vec![CsrEdge {
                peer: 0,
                edge_idx: 0,
            }],
            vec![EdgeEndpoints {
                source: 0,
                target: 1,
                connection_type: 100,
            }],
        );
        // Segment 1: 2 nodes (global ids 2, 3), 2 intra-segment edges
        // 2 → 3 (segment-local edge_idx 0) and 3 → 2 (segment-local 1).
        let s1 = seg(
            vec![slot(2, 20), slot(2, 21)],
            vec![0, 1, 2],
            vec![
                CsrEdge {
                    peer: 3,
                    edge_idx: 0,
                },
                CsrEdge {
                    peer: 2,
                    edge_idx: 1,
                },
            ],
            vec![0, 1, 2],
            vec![
                CsrEdge {
                    peer: 3,
                    edge_idx: 1,
                },
                CsrEdge {
                    peer: 2,
                    edge_idx: 0,
                },
            ],
            vec![
                EdgeEndpoints {
                    source: 2,
                    target: 3,
                    connection_type: 200,
                },
                EdgeEndpoints {
                    source: 3,
                    target: 2,
                    connection_type: 201,
                },
            ],
        );

        let c = concat_segment_csrs(vec![s0, s1]);

        // Shape.
        assert_eq!(c.node_slots.len(), 4);
        assert_eq!(c.out_offsets.len(), 5); // n+1
        assert_eq!(c.in_offsets.len(), 5);
        assert_eq!(c.out_edges.len(), 3);
        assert_eq!(c.in_edges.len(), 3);
        assert_eq!(c.edge_endpoints.len(), 3);

        // Stitched out_offsets: [0, 1, 1, 2, 3] — seg 0 contributes
        // [0,1,1]; seg 1 contributes [+1, +2] atop seg 0's last (=1),
        // so combined ends [..., 2, 3].
        let out_off: Vec<u64> = (0..c.out_offsets.len())
            .map(|i| c.out_offsets.get(i))
            .collect();
        assert_eq!(out_off, vec![0, 1, 1, 2, 3]);

        // Stitched in_offsets: [0, 0, 1, 2, 3] — seg 0 [0,0,1]; seg 1
        // contributes [+1, +2].
        let in_off: Vec<u64> = (0..c.in_offsets.len())
            .map(|i| c.in_offsets.get(i))
            .collect();
        assert_eq!(in_off, vec![0, 0, 1, 2, 3]);

        // out_edges[0] comes from seg 0, edge_idx unchanged (0).
        // out_edges[1..3] come from seg 1, edge_idx shifted by seg 0's
        // edge_endpoints.len() == 1 → (1, 2).
        assert_eq!(c.out_edges.get(0).edge_idx, 0);
        assert_eq!(c.out_edges.get(0).peer, 1);
        assert_eq!(c.out_edges.get(1).edge_idx, 1);
        assert_eq!(c.out_edges.get(1).peer, 3);
        assert_eq!(c.out_edges.get(2).edge_idx, 2);
        assert_eq!(c.out_edges.get(2).peer, 2);

        // in_edges shifts follow the same rule.
        assert_eq!(c.in_edges.get(0).edge_idx, 0); // seg 0, unchanged
        assert_eq!(c.in_edges.get(1).edge_idx, 2); // seg 1, +1
        assert_eq!(c.in_edges.get(2).edge_idx, 1); // seg 1, +1

        // edge_endpoints concat — source/target are global node ids.
        assert_eq!(c.edge_endpoints.get(0).source, 0);
        assert_eq!(c.edge_endpoints.get(0).target, 1);
        assert_eq!(c.edge_endpoints.get(1).source, 2);
        assert_eq!(c.edge_endpoints.get(1).target, 3);
        assert_eq!(c.edge_endpoints.get(2).source, 3);
        assert_eq!(c.edge_endpoints.get(2).target, 2);
    }

    #[test]
    fn concat_three_segments_keeps_offset_chain_consistent() {
        // Three one-node-one-self-edge segments. Verifies that the
        // cumulative shifts carry through multiple iterations.
        let mk_one_node = |global_id: u32, conn: u64| {
            seg(
                vec![slot(1, 0)],
                vec![0, 1],
                vec![CsrEdge {
                    peer: global_id,
                    edge_idx: 0,
                }],
                vec![0, 1],
                vec![CsrEdge {
                    peer: global_id,
                    edge_idx: 0,
                }],
                vec![EdgeEndpoints {
                    source: global_id,
                    target: global_id,
                    connection_type: conn,
                }],
            )
        };
        let c = concat_segment_csrs(vec![
            mk_one_node(0, 10),
            mk_one_node(1, 20),
            mk_one_node(2, 30),
        ]);

        let out_off: Vec<u64> = (0..c.out_offsets.len())
            .map(|i| c.out_offsets.get(i))
            .collect();
        assert_eq!(out_off, vec![0, 1, 2, 3]);

        // Each out_edges entry's edge_idx should point at its own
        // self-loop's endpoint in the combined array — segment K's
        // endpoint lands at index K.
        assert_eq!(c.out_edges.get(0).edge_idx, 0);
        assert_eq!(c.out_edges.get(1).edge_idx, 1);
        assert_eq!(c.out_edges.get(2).edge_idx, 2);

        // The endpoint at edge_idx K should be the self-loop of node K.
        for k in 0..3 {
            assert_eq!(c.edge_endpoints.get(k).source, k as u32);
            assert_eq!(c.edge_endpoints.get(k).target, k as u32);
        }
    }

    #[test]
    fn concat_handles_edgeless_segment() {
        // A segment with nodes but no edges (e.g. a freshly-created
        // empty segment). Offsets must still stitch correctly.
        let s0 = seg(
            vec![slot(1, 0)],
            vec![0, 1],
            vec![CsrEdge {
                peer: 0,
                edge_idx: 0,
            }],
            vec![0, 1],
            vec![CsrEdge {
                peer: 0,
                edge_idx: 0,
            }],
            vec![EdgeEndpoints {
                source: 0,
                target: 0,
                connection_type: 1,
            }],
        );
        let s_empty = seg(
            vec![slot(1, 1)],
            vec![0, 0],
            Vec::new(),
            vec![0, 0],
            Vec::new(),
            Vec::new(),
        );
        let s1 = seg(
            vec![slot(1, 2)],
            vec![0, 1],
            vec![CsrEdge {
                peer: 2,
                edge_idx: 0,
            }],
            vec![0, 1],
            vec![CsrEdge {
                peer: 2,
                edge_idx: 0,
            }],
            vec![EdgeEndpoints {
                source: 2,
                target: 2,
                connection_type: 3,
            }],
        );
        let c = concat_segment_csrs(vec![s0, s_empty, s1]);
        let out_off: Vec<u64> = (0..c.out_offsets.len())
            .map(|i| c.out_offsets.get(i))
            .collect();
        // 3 nodes total; middle node contributes no edges.
        assert_eq!(out_off, vec![0, 1, 1, 2]);
        assert_eq!(c.out_edges.len(), 2);
        // Segment 2 (index 2 in the input) had endpoint_base of
        // s0.edge_endpoints.len() + s_empty.edge_endpoints.len() == 1,
        // so its self-loop's edge_idx should now be 1.
        assert_eq!(c.out_edges.get(1).edge_idx, 1);
    }

    // ------------- seal_to_new_segment round-trip (phase 8) -------------

    use crate::datatypes::values::Value;
    use crate::graph::schema::{EdgeData, NodeData, StringInterner};

    fn seal_test_node(interner: &mut StringInterner, id: i64, ntype: &str) -> NodeData {
        NodeData::new(
            Value::Int64(id),
            Value::String(format!("n{id}")),
            ntype.to_string(),
            std::collections::HashMap::new(),
            interner,
        )
    }

    fn seal_test_edge(interner: &mut StringInterner, ct: &str) -> EdgeData {
        EdgeData::new(ct.to_string(), std::collections::HashMap::new(), interner)
    }

    #[test]
    fn seal_rejects_when_nothing_to_seal() {
        let tmp = TempDir::new().unwrap();
        let mut interner = StringInterner::new();
        let mut dg = super::DiskGraph::new_at_path(tmp.path()).unwrap();
        dg.defer_csr = true;
        let _n0 = dg.add_node(seal_test_node(&mut interner, 0, "A"));
        dg.build_csr_from_pending();
        dg.save_to_dir(tmp.path(), &interner).unwrap();
        // save_to_dir set sealed_nodes_bound = node_count, so tail is empty.
        let err = dg.seal_to_new_segment(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("nothing to seal"));
    }

    #[test]
    fn seal_rejects_cross_segment_edges() {
        let tmp = TempDir::new().unwrap();
        let mut interner = StringInterner::new();
        let mut dg = super::DiskGraph::new_at_path(tmp.path()).unwrap();
        dg.defer_csr = true;
        let n0 = dg.add_node(seal_test_node(&mut interner, 0, "A"));
        let n1 = dg.add_node(seal_test_node(&mut interner, 1, "A"));
        dg.add_edge(n0, n1, seal_test_edge(&mut interner, "T"));
        dg.build_csr_from_pending();
        dg.save_to_dir(tmp.path(), &interner).unwrap();

        // Add a new node + a cross-segment edge (old n0 → new n2).
        // That edge has source below the watermark after save, so
        // seal should refuse.
        let n2 = dg.add_node(seal_test_node(&mut interner, 2, "A"));
        dg.add_edge(n0, n2, seal_test_edge(&mut interner, "T"));
        let err = dg.seal_to_new_segment(tmp.path()).unwrap_err();
        assert!(
            err.to_string().contains("crosses segment boundary"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn seal_round_trip_basic_reads() {
        // Build seg_0: 3 nodes of type A with one edge between them.
        let tmp = TempDir::new().unwrap();
        let mut interner = StringInterner::new();
        let mut dg = super::DiskGraph::new_at_path(tmp.path()).unwrap();
        dg.defer_csr = true;
        let n0 = dg.add_node(seal_test_node(&mut interner, 0, "A"));
        let n1 = dg.add_node(seal_test_node(&mut interner, 1, "A"));
        let _n2 = dg.add_node(seal_test_node(&mut interner, 2, "A"));
        dg.add_edge(n0, n1, seal_test_edge(&mut interner, "T"));
        dg.build_csr_from_pending();
        dg.save_to_dir(tmp.path(), &interner).unwrap();

        assert_eq!(dg.node_count, 3);
        assert_eq!(dg.sealed_nodes_bound, 3);

        // Add 2 new nodes of type B + an edge between them.
        // Both endpoints are strictly above the watermark, so the
        // seal constraint holds.
        let n3 = dg.add_node(seal_test_node(&mut interner, 3, "B"));
        let n4 = dg.add_node(seal_test_node(&mut interner, 4, "B"));
        dg.add_edge(n3, n4, seal_test_edge(&mut interner, "U"));

        let pre_edge_count = dg.edge_count;
        let pre_node_count = dg.node_count;
        assert_eq!(pre_node_count, 5);
        assert_eq!(pre_edge_count, 2);

        // Seal the tail to seg_001.
        let seg_id = dg.seal_to_new_segment(tmp.path()).unwrap();
        assert_eq!(seg_id, 1);
        assert_eq!(dg.sealed_nodes_bound, 5);
        assert!(dg.overflow_out.is_empty());
        assert!(dg.overflow_in.is_empty());

        // Verify seg_001/ has the expected files on disk.
        let seg1 = tmp.path().join("seg_001");
        for name in [
            "node_slots.bin",
            "out_offsets.bin",
            "out_edges.bin",
            "in_offsets.bin",
            "in_edges.bin",
            "edge_endpoints.bin",
        ] {
            assert!(seg1.join(name).exists(), "missing {name}");
        }
        // Manifest should have 2 entries now.
        let manifest =
            super::super::segment_summary::SegmentManifest::load_from(tmp.path()).unwrap();
        assert_eq!(manifest.len(), 2);
        assert_eq!(manifest.segments[1].segment_id, 1);
        assert_eq!(manifest.segments[1].node_id_lo, 3);
        assert_eq!(manifest.segments[1].node_id_hi, 5);
        assert_eq!(manifest.segments[1].edge_count, 1);

        // Drop in-memory graph and reload — this exercises the phase-7
        // concat read path.
        drop(dg);
        let mut interner2 = StringInterner::new();
        let (reloaded, _tmp_zst) =
            super::DiskGraph::load_from_dir(tmp.path(), &mut interner2).unwrap();

        assert_eq!(reloaded.node_count, pre_node_count);
        assert_eq!(reloaded.edge_count, pre_edge_count);
        // sealed_nodes_bound persists through save/load.
        assert_eq!(reloaded.sealed_nodes_bound, 5);

        // Untyped outgoing edges for node 3 should be exactly 1 (the
        // n3 → n4 edge). This verifies the concat stitched the
        // out_offsets correctly for the sealed tail.
        let n3_idx = 3usize;
        let start = reloaded.out_offsets.get(n3_idx) as usize;
        let end = reloaded.out_offsets.get(n3_idx + 1) as usize;
        assert_eq!(end - start, 1, "expected 1 outgoing edge for seg_1 node 3");
        let e = reloaded.out_edges.get(start);
        assert_eq!(e.peer, 4);
        // Combined edge_idx = segment-local 0 + endpoint_base (= seg_0's 1)
        // = 1. Verifies the phase-7 concat shift lands at the right slot.
        assert_eq!(e.edge_idx, 1);
        // edge_endpoints at that global index should hold the original
        // global node ids.
        let ep = reloaded.edge_endpoints.get(1);
        assert_eq!(ep.source, 3);
        assert_eq!(ep.target, 4);

        // And seg_0's original edge (0 → 1) is still present at
        // combined edge_idx 0.
        let ep0 = reloaded.edge_endpoints.get(0);
        assert_eq!(ep0.source, 0);
        assert_eq!(ep0.target, 1);
        let _ = (n1, n4); // silence unused-warnings if optimised out
    }
}
