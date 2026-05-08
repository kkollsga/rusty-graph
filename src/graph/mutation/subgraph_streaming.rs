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

use crate::graph::schema::InternedKey;
use crate::graph::storage::disk::csr::TOMBSTONE_EDGE;
use crate::graph::storage::disk::graph::DiskGraph;

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

    /// Total set bits. Used to size the output graph in Phase 3+.
    pub fn count_ones(&self) -> u64 {
        self.blocks.iter().map(|b| b.count_ones() as u64).sum()
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
}
