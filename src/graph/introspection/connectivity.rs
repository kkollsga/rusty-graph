//! Type connectivity triples — derived statistics about which types
//! connect to which, via which connection type.

use crate::graph::schema::{ConnectivityTriple, DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use std::collections::{HashMap, HashSet};

use super::{NeighborConnection, NeighborsSchema};

// ── Type connectivity ──────────────────────────────────────────────────────

/// Compute type connectivity triples via a single O(E) pass.
///
/// Uses InternedKey-based aggregation (no string allocation during scan)
/// and sequential iteration for cache-friendly I/O on disk graphs.
/// Resolves keys to strings only at the end.
///
/// **Hot path (861M+ edges on Wikidata).** Goes through the closure-based
/// [`crate::graph::schema::GraphBackend::for_each_edge_endpoint_key`] rather
/// than the boxed-iterator [`GraphRead::edge_endpoint_keys`] to avoid 863M
/// virtual `.next()` calls. `node_type_of` stays on the boxed enum method
/// because rustc's static-dispatch inlines it cleanly on a `&GraphBackend`.
pub fn compute_type_connectivity(graph: &DirGraph) -> Vec<ConnectivityTriple> {
    // Aggregate with InternedKey tuples — no string allocation per edge.
    let mut counts: HashMap<(InternedKey, InternedKey, InternedKey), usize> = HashMap::new();
    let backend = &graph.graph;

    backend.for_each_edge_endpoint_key(|src_idx, tgt_idx, conn_key| {
        let src_key = backend.node_type_of(src_idx);
        let tgt_key = backend.node_type_of(tgt_idx);
        if let (Some(sk), Some(tk)) = (src_key, tgt_key) {
            *counts.entry((sk, conn_key, tk)).or_insert(0) += 1;
        }
    });

    // Resolve to strings once at the end
    let mut triples: Vec<ConnectivityTriple> = counts
        .into_iter()
        .map(|((sk, ck, tk), count)| ConnectivityTriple {
            src: graph.interner.resolve(sk).to_string(),
            conn: graph.interner.resolve(ck).to_string(),
            tgt: graph.interner.resolve(tk).to_string(),
            count,
        })
        .collect();
    triples.sort_by_key(|t| std::cmp::Reverse(t.count));
    triples
}

/// Build NeighborsSchema for a specific type from pre-computed triples.
/// O(triples) linear scan — use `TypeConnectivityIndex` for O(1) lookups.
pub fn neighbors_from_triples(triples: &[ConnectivityTriple], node_type: &str) -> NeighborsSchema {
    let mut outgoing: Vec<NeighborConnection> = Vec::new();
    let mut incoming: Vec<NeighborConnection> = Vec::new();

    for t in triples {
        if t.src == node_type {
            outgoing.push(NeighborConnection {
                connection_type: t.conn.clone(),
                other_type: t.tgt.clone(),
                count: t.count,
            });
        }
        if t.tgt == node_type {
            incoming.push(NeighborConnection {
                connection_type: t.conn.clone(),
                other_type: t.src.clone(),
                count: t.count,
            });
        }
    }

    outgoing.sort_by_key(|o| std::cmp::Reverse(o.count));
    incoming.sort_by_key(|i| std::cmp::Reverse(i.count));

    NeighborsSchema { outgoing, incoming }
}

/// Pre-indexed type connectivity for O(1) neighbor lookups.
/// Built once from triples, used for all describe operations in a session.
pub struct TypeConnectivityIndex {
    /// type_name → (outgoing, incoming) neighbor connections, sorted by count desc.
    index: HashMap<String, NeighborsSchema>,
}

impl TypeConnectivityIndex {
    /// Build index from flat triples. O(triples) one-time cost.
    pub fn from_triples(triples: &[ConnectivityTriple]) -> Self {
        let mut out_map: HashMap<String, Vec<NeighborConnection>> = HashMap::new();
        let mut in_map: HashMap<String, Vec<NeighborConnection>> = HashMap::new();

        for t in triples {
            out_map
                .entry(t.src.clone())
                .or_default()
                .push(NeighborConnection {
                    connection_type: t.conn.clone(),
                    other_type: t.tgt.clone(),
                    count: t.count,
                });
            in_map
                .entry(t.tgt.clone())
                .or_default()
                .push(NeighborConnection {
                    connection_type: t.conn.clone(),
                    other_type: t.src.clone(),
                    count: t.count,
                });
        }

        // Merge into NeighborsSchema per type, sort by count desc.
        // Collect all type names first (owned) to avoid borrow conflicts.
        let all_types: HashSet<String> = out_map.keys().chain(in_map.keys()).cloned().collect();

        let mut index = HashMap::with_capacity(all_types.len());
        for nt in all_types {
            let mut outgoing = out_map.remove(&nt).unwrap_or_default();
            outgoing.sort_by_key(|o| std::cmp::Reverse(o.count));
            let mut incoming = in_map.remove(&nt).unwrap_or_default();
            incoming.sort_by_key(|i| std::cmp::Reverse(i.count));
            index.insert(nt, NeighborsSchema { outgoing, incoming });
        }

        TypeConnectivityIndex { index }
    }

    /// O(1) lookup of neighbors for a type.
    pub fn get(&self, node_type: &str) -> NeighborsSchema {
        self.index
            .get(node_type)
            .cloned()
            .unwrap_or(NeighborsSchema {
                outgoing: Vec::new(),
                incoming: Vec::new(),
            })
    }
}

/// Derived edge statistics from type connectivity triples.
pub struct DerivedEdgeStats {
    /// Edge type → count.
    pub counts: HashMap<String, usize>,
    /// Edge type → (source_types, target_types).
    pub endpoints: HashMap<String, (HashSet<String>, HashSet<String>)>,
}

/// Derive edge type counts + endpoint types from type connectivity triples.
/// Avoids separate O(E) scans for these derived data.
pub fn derive_edge_counts_from_triples(triples: &[ConnectivityTriple]) -> DerivedEdgeStats {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut endpoints: HashMap<String, (HashSet<String>, HashSet<String>)> = HashMap::new();

    for t in triples {
        *counts.entry(t.conn.clone()).or_insert(0) += t.count;
        let entry = endpoints
            .entry(t.conn.clone())
            .or_insert_with(|| (HashSet::new(), HashSet::new()));
        entry.0.insert(t.src.clone());
        entry.1.insert(t.tgt.clone());
    }

    DerivedEdgeStats { counts, endpoints }
}
