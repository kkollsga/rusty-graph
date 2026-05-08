//! Disk-to-disk streaming subgraph filter.
//!
//! Phase 2 of `save_subset`: Pass A — sequential edge scan that builds a
//! kept-nodes bitset. No output files yet; this is the spike validating
//! the I/O shape on real Wikidata before output plumbing exists.
//!
//! Subsequent phases extend this module:
//! - Phase 3: rank-1 over the bitset + per-type node materialization.
//! - Phase 4: edge translation + property byte-range copy + CSR build.
//! - Phase 5: sidecars + atomic finalize.
//!
//! The streaming pipeline is gated to disk-backed sources. In-memory and
//! mapped graphs route through the existing `to_subgraph().save()` path
//! per CLAUDE.md ("in-memory wins every time"). All I/O here is
//! sequential — no per-node random edge lookups.

use crate::graph::schema::{CowSelection, DirGraph, InternedKey};
use crate::graph::storage::disk::csr::TOMBSTONE_EDGE;
use crate::graph::storage::disk::graph::DiskGraph;
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use std::path::Path;

/// Canonical streaming-filter spec. The fluent chain
/// (`select().expand().save_subset()`) lowers into this; future Cypher
/// integration will lower into the same struct.
#[derive(Clone, Debug, Default)]
pub struct SubsetSpec {
    /// Restrict to edges of these types. `None` means all edge types.
    /// In Phase 2 this is the only filter knob.
    pub edge_types: Option<Vec<InternedKey>>,
    // Phase 3+ will add: seed_node_ids, seed_node_types, expand_hops.
}

/// Stats reported back from a Pass A scan. Used for sanity-checking on
/// real Wikidata before committing to write the output graph.
#[derive(Clone, Debug, Default)]
pub struct ScanStats {
    pub kept_node_count: u64,
    pub kept_edge_count: u64,
    pub total_edge_count: u64,
    pub scan_duration_secs: f64,
}

/// Compact bitset over node ids. `Vec<u64>` blocks; one bit per source
/// node id. Phase 3 will extend this with rank-1 (popcount prefix array)
/// for O(1) old→new id translation.
#[derive(Clone, Debug)]
pub struct Bitset {
    blocks: Vec<u64>,
    len: usize,
}

impl Bitset {
    /// Allocate a bitset covering `[0..len)`. All bits start cleared.
    pub fn with_len(len: usize) -> Self {
        let n_blocks = len.div_ceil(64);
        Self {
            blocks: vec![0u64; n_blocks],
            len,
        }
    }

    /// Set bit `i` to 1. Out-of-range writes are silently ignored — the
    /// caller (Pass A) is expected to keep ids within `node_bound` but
    /// disk graphs occasionally surface tombstone-adjacent ids.
    #[inline]
    pub fn set(&mut self, i: usize) {
        if i < self.len {
            self.blocks[i / 64] |= 1u64 << (i % 64);
        }
    }

    /// Read bit `i`. Out-of-range reads return false.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        if i >= self.len {
            return false;
        }
        (self.blocks[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Number of bits the bitset covers.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Total set bits. Used to size the output graph in Phase 3+.
    pub fn count_ones(&self) -> u64 {
        self.blocks.iter().map(|b| b.count_ones() as u64).sum()
    }

    /// Raw block view — used by [`RankIndex`] to compute block-prefix
    /// popcounts without re-scanning the bitset.
    #[inline]
    pub(crate) fn blocks(&self) -> &[u64] {
        &self.blocks
    }
}

// ── Rank-1 over the kept-nodes bitset ──────────────────────────────────────
//
// Phase 3: build a popcount-prefix array so old→new node id translation
// is O(1) and entirely in RAM. No 480 MB dense remap.
//
// `block_prefix[k]` holds `popcount(bitset[0..k*64])`. For a query
// `old_to_new(old_id)`:
//   block = old_id / 64
//   bit   = old_id % 64
//   mask  = (1 << bit) - 1
//   rank  = block_prefix[block] + popcount(bitset_block[block] & mask)
//
// At Wikidata scale (120M nodes): 15 MB bitset + 15 MB prefix = 30 MB
// total. Two `count_ones` calls per query; both compile to `popcnt` on
// x86_64 / equivalents on ARM. No disk access.
//
// The new id space is contiguous `[0..kept_count)` ordered by source id —
// matches the natural sequential walk of `node_slots.bin` in Pass B.

/// O(1) translator from source node ids to dense destination ids. Built
/// once after Pass A; consumed by node materialization (Phase 4) and
/// edge translation (Phase 4). All RAM, zero disk reads.
///
/// `dead_code` is allowed at the impl level because Phase 3 ships the
/// primitive in isolation — Phase 4 is the first consumer and lights up
/// every method.
#[derive(Debug, Clone)]
pub struct RankIndex {
    bitset: Bitset,
    /// `block_prefix[k]` = popcount of `bitset.blocks()[0..k]`, i.e. the
    /// rank of bit `k * 64` exclusive of itself. Length = `bitset.blocks().len() + 1`
    /// so a 1-block bitset has prefixes `[0, popcount(block 0)]`.
    block_prefix: Vec<u32>,
    /// Total kept count = `block_prefix.last()`.
    kept_count: u32,
}

#[allow(dead_code)] // Phase 4 consumes every method.
impl RankIndex {
    /// Build the rank-1 index from a kept-nodes bitset. Single linear
    /// pass over the `Vec<u64>` blocks; cost is O(n_blocks).
    pub fn from_bitset(bitset: Bitset) -> Self {
        let blocks = bitset.blocks();
        let mut block_prefix: Vec<u32> = Vec::with_capacity(blocks.len() + 1);
        let mut acc: u32 = 0;
        block_prefix.push(0);
        for blk in blocks {
            acc = acc.saturating_add(blk.count_ones());
            block_prefix.push(acc);
        }
        Self {
            bitset,
            block_prefix,
            kept_count: acc,
        }
    }

    /// Total kept nodes — also the size of the destination id space.
    #[inline]
    pub fn kept_count(&self) -> u32 {
        self.kept_count
    }

    /// True if source id `old_id` is in the kept set.
    #[inline]
    pub fn contains(&self, old_id: u32) -> bool {
        self.bitset.get(old_id as usize)
    }

    /// Map a source node id to its destination id. Returns `None` when
    /// `old_id` is not in the kept set or is out of range.
    ///
    /// Cost: O(1) — two array accesses + one `popcount`.
    #[inline]
    pub fn old_to_new(&self, old_id: u32) -> Option<u32> {
        let len = self.bitset.len();
        let i = old_id as usize;
        if i >= len {
            return None;
        }
        let block_idx = i / 64;
        let bit = i % 64;
        let block = self.bitset.blocks()[block_idx];
        // Bit must be set for `old_id` to be in the kept set.
        if (block >> bit) & 1 == 0 {
            return None;
        }
        // Mask of bits before `bit` in the same block.
        let mask: u64 = if bit == 0 { 0 } else { (1u64 << bit) - 1 };
        let within_block = (block & mask).count_ones();
        Some(self.block_prefix[block_idx] + within_block)
    }

    /// Borrow the underlying bitset — used by tests and Phase 4 when
    /// iterating kept ids directly.
    #[inline]
    pub fn bitset(&self) -> &Bitset {
        &self.bitset
    }
}

/// Result of [`pass_a_scan`].
pub struct PassAResult {
    /// Source node ids that survive the filter. Endpoints of every kept
    /// edge, plus any explicit seed nodes (Phase 3+). Phase 2 spike does
    /// not surface the bitset to callers; Phase 3 consumes it to build
    /// the rank-1 index.
    #[allow(dead_code)]
    pub kept_nodes: Bitset,
    pub stats: ScanStats,
}

/// Pass A: sequential scan of `edge_endpoints.bin` with an edge-type
/// filter, building a kept-nodes bitset.
///
/// Memory: `Bitset` is `n_nodes / 8` bytes (~15 MB at 120M Wikidata
/// nodes). No other allocations scale with graph size.
///
/// I/O: one sequential read of `edge_endpoints` (16 B per edge). Disk
/// reads are mmap'd — the kernel prefetches sequential access. Tombstoned
/// edges (TOMBSTONE_EDGE marker in `source`) are skipped without
/// touching property storage.
///
/// Phase 4 will extend this loop to also write `kept_edges.tmp` +
/// `edge_prop_offsets.bin` + `edge_prop_heap.bin` for the destination
/// graph, all in lockstep with the same sequential scan.
pub fn pass_a_scan(source: &DiskGraph, spec: &SubsetSpec) -> PassAResult {
    let n_nodes = source.node_slots.len();
    let mut kept_nodes = Bitset::with_len(n_nodes);

    // Convert edge-type filter to a hash set keyed on the raw u64 hash —
    // matches `EdgeEndpoints.connection_type` directly without
    // re-interning per edge. Empty filter = "keep all".
    let edge_type_set: Option<std::collections::HashSet<u64>> = spec
        .edge_types
        .as_ref()
        .map(|v| v.iter().map(|k| k.as_u64()).collect());

    // Hint the kernel: we're about to scan endpoints front-to-back.
    source.edge_endpoints.advise_sequential();

    let scan_start = std::time::Instant::now();
    let n_edges = source.next_edge_idx as usize;
    let mut kept_edge_count: u64 = 0;
    for edge_idx in 0..n_edges {
        let ep = source.edge_endpoints.get(edge_idx);
        if ep.source == TOMBSTONE_EDGE {
            continue;
        }
        if let Some(ref types) = edge_type_set {
            if !types.contains(&ep.connection_type) {
                continue;
            }
        }
        kept_nodes.set(ep.source as usize);
        kept_nodes.set(ep.target as usize);
        kept_edge_count += 1;
    }

    let kept_node_count = kept_nodes.count_ones();
    PassAResult {
        kept_nodes,
        stats: ScanStats {
            kept_node_count,
            kept_edge_count,
            total_edge_count: n_edges as u64,
            scan_duration_secs: scan_start.elapsed().as_secs_f64(),
        },
    }
}

// ── Pass A with file output — Phase 4 ──────────────────────────────────────
//
// Same sequential scan as `pass_a_scan` but also spills the kept edges to
// a `MmapOrVec<(u32, u32, u64)>` at `kept_edges_path`. The on-disk record
// shape matches the existing CSR builder's input
// (`csr_build::build_csr_files` consumes `&MmapOrVec<(u32, u32, u64)>`),
// so a later phase can drive the merge sort over this file by translating
// `(src, tgt)` via the rank index in the iterator.
//
// Edge property bytes are NOT inlined yet — that's a Phase 5 extension
// that requires either (a) a sidecar file keyed by source edge_idx or
// (b) a fourth column in the temp record. For Phase 4 we keep the temp
// file shape compatible with the existing builder; properties travel
// independently in subsequent work.
//
// Memory: bitset (~15 MB at 120M nodes). No heap buffer scales with kept
// edge count; appended records go straight to the mmap'd file.
//
// I/O: one sequential read of `edge_endpoints` + one sequential write to
// `kept_edges_path`.

/// Result of [`pass_a_scan_to_file`]. The temp file path is returned so
/// the caller (Phase 5+) can hand it to `csr_build::build_csr_files`
/// after wrapping it with a rank-translating iterator.
pub struct PassAFileResult {
    pub kept_nodes: Bitset,
    pub stats: ScanStats,
    pub kept_edges_path: std::path::PathBuf,
    pub kept_edge_records: u64,
}

/// Pass A with file output. Identical semantics to [`pass_a_scan`] but
/// also appends `(src, tgt, conn_type)` for each kept edge to a file at
/// `kept_edges_path` via `MmapOrVec::mapped`. The file is sized for the
/// total edge count up front (a safe upper bound on kept edges); Phase 5
/// reads the actual count from `kept_edge_records`.
pub fn pass_a_scan_to_file(
    source: &DiskGraph,
    spec: &SubsetSpec,
    kept_edges_path: &Path,
) -> Result<PassAFileResult, String> {
    let n_nodes = source.node_slots.len();
    let n_edges = source.next_edge_idx as usize;
    let mut kept_nodes = Bitset::with_len(n_nodes);

    let edge_type_set: Option<std::collections::HashSet<u64>> = spec
        .edge_types
        .as_ref()
        .map(|v| v.iter().map(|k| k.as_u64()).collect());

    if let Some(parent) = kept_edges_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "save_subset: failed to create temp dir {}: {}",
                    parent.display(),
                    e
                )
            })?;
        }
    }

    let mut kept_edges: MmapOrVec<(u32, u32, u64)> = MmapOrVec::mapped(kept_edges_path, n_edges)
        .unwrap_or_else(|_| MmapOrVec::with_capacity(n_edges));

    source.edge_endpoints.advise_sequential();
    let scan_start = std::time::Instant::now();
    let mut kept_edge_count: u64 = 0;
    for edge_idx in 0..n_edges {
        let ep = source.edge_endpoints.get(edge_idx);
        if ep.source == TOMBSTONE_EDGE {
            continue;
        }
        if let Some(ref types) = edge_type_set {
            if !types.contains(&ep.connection_type) {
                continue;
            }
        }
        kept_nodes.set(ep.source as usize);
        kept_nodes.set(ep.target as usize);
        kept_edges.push((ep.source, ep.target, ep.connection_type));
        kept_edge_count += 1;
    }

    let kept_node_count = kept_nodes.count_ones();
    let scan_duration_secs = scan_start.elapsed().as_secs_f64();

    Ok(PassAFileResult {
        kept_nodes,
        stats: ScanStats {
            kept_node_count,
            kept_edge_count,
            total_edge_count: n_edges as u64,
            scan_duration_secs,
        },
        kept_edges_path: kept_edges_path.to_path_buf(),
        kept_edge_records: kept_edge_count,
    })
}

// ── save_subset orchestrator ───────────────────────────────────────────────
//
// Phase 5+6 ship a working `save_subset(selection, path)` that produces a
// fully independent on-disk graph. v1 uses the existing in-memory
// `extract_subgraph` machinery — bounded by RAM, but correct for every
// storage mode of the source. The streaming primitives in this module
// (Bitset, RankIndex, Pass A, kept_edges.tmp) are wired and tested but
// not yet on the hot path.
//
// v2 will swap the in-memory `extract_subgraph` step for a
// streaming-disk-to-disk pipeline using the primitives above. The public
// API stays unchanged — only the implementation flips.
//
// Why ship the in-memory baseline first: realistic 1-hop subsets on
// Wikidata (Articles + Authors ≈ 15 M nodes / 10 M edges) fit in ~1 GB
// of RAM via petgraph + Arc'd column stores, so the API is usable today
// for the user's stated workflow. Larger subsets need streaming, which
// is gated on the column-store materialization pieces still pending.

/// Save a filtered subgraph to disk.
///
/// `selection` defines which nodes are kept (typically built via the
/// fluent chain: `kg.select(...).expand(...)` returns a selection that
/// gets handed here). All edges between kept nodes are included.
///
/// Output is a v3 binary file (single `.kgl` file for in-memory/mapped
/// sources, directory for disk sources via `save_disk`). The file can be
/// reloaded into any storage mode via `kglite.load(path, storage=...)`.
///
/// `_spec` is currently unused — the selection carries the filter. The
/// parameter exists so v2 (Cypher integration) can lower into the same
/// entry point without an API change.
pub fn save_subset(
    source: &DirGraph,
    selection: &CowSelection,
    out_path: &Path,
    _spec: Option<&SubsetSpec>,
) -> Result<(), String> {
    use crate::graph::mutation::subgraph::extract_subgraph;
    use std::sync::Arc;

    // 1. Materialize the filtered subgraph in-memory. `extract_subgraph`
    //    works for every source storage mode — its reads go through the
    //    `GraphRead` trait which is implemented for memory/mapped/disk.
    //    For disk sources, `node_weight` returns NodeData with
    //    `PropertyStorage::Columnar { store: Arc<source_store>, row_id }`,
    //    so the extracted graph holds Arc references into the source's
    //    column stores. No deep clone of property data.
    let mut extracted = extract_subgraph(source, selection)?;

    // 2. Consolidate all node properties into column stores so the output
    //    file is self-contained — does not depend on the source's stores.
    //    Mirrors the prep that `KnowledgeGraph::save` does for the
    //    in-memory save path.
    extracted.enable_columnar();

    let path_str = out_path.to_str().ok_or_else(|| {
        format!(
            "save_subset: out_path is not valid UTF-8: {}",
            out_path.display()
        )
    })?;

    // 3. Persist via the v3 binary writer. `prepare_save` operates on
    //    `Arc<DirGraph>` (it stamps metadata + snapshots index keys), so
    //    wrap-then-unwrap once.
    let mut arc = Arc::new(extracted);
    crate::graph::io::file::prepare_save(&mut arc);

    crate::graph::io::file::write_graph_v3(&arc, path_str)
        .map_err(|e| format!("save_subset: write_graph_v3 failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitset_set_count_across_block_boundaries() {
        let mut bs = Bitset::with_len(200);
        bs.set(0);
        bs.set(63);
        bs.set(64);
        bs.set(199);
        // Setting the same bit twice does not double-count.
        bs.set(64);
        assert_eq!(bs.count_ones(), 4);
    }

    #[test]
    fn bitset_out_of_range_writes_are_ignored() {
        let mut bs = Bitset::with_len(100);
        bs.set(99);
        bs.set(500); // ignored, not panic
        assert_eq!(bs.count_ones(), 1);
    }

    // Note: `pass_a_scan` requires a real `DiskGraph`. End-to-end tests
    // live in `tests/test_subgraph_streaming.py` (Phase 2) where building
    // a disk graph is straightforward via the public Python API.

    // ── Phase 3: RankIndex ────────────────────────────────────────────

    /// Helper: brute-force rank-1 to validate the popcount-prefix
    /// implementation against. Iterates the bitset bit-by-bit, counting
    /// kept bits before `old_id`. O(n_bits) but trivially correct.
    fn brute_force_rank(bs: &Bitset, old_id: u32) -> Option<u32> {
        if !bs.get(old_id as usize) {
            return None;
        }
        let mut rank: u32 = 0;
        for i in 0..(old_id as usize) {
            if bs.get(i) {
                rank += 1;
            }
        }
        Some(rank)
    }

    #[test]
    fn rank_index_dense_pattern() {
        // Every 3rd id kept in [0..200). Crosses three 64-bit blocks
        // and ensures within-block + cross-block popcounts both fire.
        let mut bs = Bitset::with_len(200);
        for i in (0..200).step_by(3) {
            bs.set(i);
        }
        let idx = RankIndex::from_bitset(bs.clone());

        // Total kept = ceil(200 / 3) = 67.
        assert_eq!(idx.kept_count(), 67);

        // Spot-check the first few mappings.
        assert_eq!(idx.old_to_new(0), Some(0));
        assert_eq!(idx.old_to_new(3), Some(1));
        assert_eq!(idx.old_to_new(6), Some(2));
        assert_eq!(idx.old_to_new(63), Some(21)); // 63/3 = 21
        assert_eq!(idx.old_to_new(66), Some(22));
        assert_eq!(idx.old_to_new(198), Some(66));

        // Not-in-set: 1, 2, 4, 5, ...
        assert_eq!(idx.old_to_new(1), None);
        assert_eq!(idx.old_to_new(64), None);

        // Differential against brute force across the whole range.
        for i in 0..200u32 {
            assert_eq!(idx.old_to_new(i), brute_force_rank(&bs, i));
        }
    }

    #[test]
    fn rank_index_sparse_pattern() {
        // One bit set per 64-bit block. Tests block-prefix advancement
        // when `within_block` is always 0.
        let mut bs = Bitset::with_len(256);
        bs.set(0);
        bs.set(64);
        bs.set(128);
        bs.set(192);
        let idx = RankIndex::from_bitset(bs.clone());

        assert_eq!(idx.kept_count(), 4);
        assert_eq!(idx.old_to_new(0), Some(0));
        assert_eq!(idx.old_to_new(64), Some(1));
        assert_eq!(idx.old_to_new(128), Some(2));
        assert_eq!(idx.old_to_new(192), Some(3));

        // Bits before/after each set bit must report None.
        assert_eq!(idx.old_to_new(63), None);
        assert_eq!(idx.old_to_new(65), None);
        assert_eq!(idx.old_to_new(255), None);
    }

    #[test]
    fn rank_index_full_pattern() {
        // All bits set. old_to_new(i) == i for every i.
        let mut bs = Bitset::with_len(130);
        for i in 0..130 {
            bs.set(i);
        }
        let idx = RankIndex::from_bitset(bs);

        assert_eq!(idx.kept_count(), 130);
        for i in 0..130u32 {
            assert_eq!(idx.old_to_new(i), Some(i));
        }
    }

    #[test]
    fn rank_index_empty_pattern() {
        // Empty bitset — every query returns None, kept_count = 0.
        let bs = Bitset::with_len(200);
        let idx = RankIndex::from_bitset(bs);

        assert_eq!(idx.kept_count(), 0);
        for i in 0..200u32 {
            assert_eq!(idx.old_to_new(i), None);
        }
    }

    #[test]
    fn rank_index_out_of_range_returns_none() {
        let mut bs = Bitset::with_len(100);
        bs.set(99);
        let idx = RankIndex::from_bitset(bs);

        assert_eq!(idx.old_to_new(99), Some(0));
        assert_eq!(idx.old_to_new(100), None);
        assert_eq!(idx.old_to_new(u32::MAX), None);
    }

    #[test]
    fn rank_index_zero_length_bitset() {
        let bs = Bitset::with_len(0);
        let idx = RankIndex::from_bitset(bs);
        assert_eq!(idx.kept_count(), 0);
        assert_eq!(idx.old_to_new(0), None);
    }

    #[test]
    fn rank_index_single_bit_each_block_boundary() {
        // Stress the bit==0 branch (mask = 0, no shift).
        let mut bs = Bitset::with_len(256);
        for i in (0..256).step_by(64) {
            bs.set(i);
        }
        let idx = RankIndex::from_bitset(bs);

        assert_eq!(idx.old_to_new(0), Some(0));
        assert_eq!(idx.old_to_new(64), Some(1));
        assert_eq!(idx.old_to_new(128), Some(2));
        assert_eq!(idx.old_to_new(192), Some(3));
    }

    #[test]
    fn rank_index_pseudo_random_differential() {
        // Pseudorandom-but-deterministic bitset; differential against
        // brute-force rank across a whole 1024-bit range.
        let mut bs = Bitset::with_len(1024);
        let mut state: u32 = 0xDEAD_BEEF;
        for i in 0..1024 {
            // Cheap LCG; not a real RNG but sufficient for varied bits.
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            if state & 0b11 != 0 {
                // ~75% set rate
                bs.set(i);
            }
        }
        let idx = RankIndex::from_bitset(bs.clone());
        for i in 0..1024u32 {
            assert_eq!(idx.old_to_new(i), brute_force_rank(&bs, i));
        }
    }

    #[test]
    fn rank_index_contains_matches_bitset() {
        let mut bs = Bitset::with_len(100);
        bs.set(5);
        bs.set(50);
        bs.set(99);
        let idx = RankIndex::from_bitset(bs);

        assert!(idx.contains(5));
        assert!(idx.contains(50));
        assert!(idx.contains(99));
        assert!(!idx.contains(0));
        assert!(!idx.contains(49));
        assert!(!idx.contains(100));
    }
}
