//! Cypher query optimizer.
//!
//! Split (Phase 9):
//! - [`join_order`] — pattern-start node selection, selectivity-based reordering
//! - [`index_selection`] — predicate pushdown into MATCH, equality/comparison helpers
//! - [`cost_model`] — predicate / expression cost heuristics
//! - [`simplification`] — fold_or_to_in, push LIMIT/DISTINCT, rewrite_text_score
//! - [`fusion`] — multi-clause fusion (MATCH+RETURN+AGG, top-K, …)

use super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;
use crate::graph::schema::DirGraph;
use std::collections::HashMap;

pub mod cost_model;
pub mod fusion;
pub mod index_selection;
pub mod join_order;
pub mod schema_check;
pub mod simplification;

use cost_model::reorder_predicates_by_cost;
use fusion::{
    fuse_anchored_edge_count, fuse_count_short_circuits, fuse_match_return_aggregate,
    fuse_match_with_aggregate, fuse_match_with_aggregate_top_k, fuse_node_scan_aggregate,
    fuse_node_scan_top_k, fuse_optional_match_aggregate, fuse_order_by_top_k, fuse_spatial_join,
    fuse_vector_score_order_limit, mark_return_lazy_eligible,
};
use index_selection::push_where_into_match;
use join_order::{optimize_pattern_start_node, reorder_match_clauses, reorder_match_patterns};
use simplification::{
    desugar_multi_match_return_aggregate, fold_or_to_in, fold_pass_through_with,
    push_distinct_into_match, push_limit_into_match,
};

/// Annotate the top-level query's terminal RETURN with `lazy_eligible`
/// when no downstream operator forces row materialisation. Called once
/// after `optimize`, never recursively, so nested UNION arms don't get
/// marked (their results pass through the union machinery, which expects
/// fully evaluated rows).
pub fn mark_lazy_eligibility(query: &mut CypherQuery) {
    // Don't mark when the top-level query contains a UNION — the union
    // machinery merges materialised rows.
    if query.clauses.iter().any(|c| matches!(c, Clause::Union(_))) {
        return;
    }
    // Don't mark for mutation queries — CREATE/SET/DELETE/REMOVE/MERGE go
    // through `execute_mutable`, which doesn't read the lazy descriptor
    // and would produce empty rows.
    if query.clauses.iter().any(|c| {
        matches!(
            c,
            Clause::Create(_)
                | Clause::Set(_)
                | Clause::Delete(_)
                | Clause::Remove(_)
                | Clause::Merge(_)
        )
    }) {
        return;
    }
    mark_return_lazy_eligible(query);
}

pub fn optimize(query: &mut CypherQuery, graph: &DirGraph, params: &HashMap<String, Value>) {
    // Recursively optimize nested queries (e.g., UNION right-arm)
    optimize_nested_queries(query, graph, params);
    push_where_into_match(query, params);
    fold_or_to_in(query);
    push_where_into_match(query, params); // second pass: push newly-created IN predicates
                                          // Strip pass-through WITH clauses that block downstream fusion. Runs
                                          // before the cross-clause MATCH reorder so the latter sees a
                                          // contiguous Match-Match span when a `WITH p` sat between them.
    fold_pass_through_with(query);
    // Rewrite `Match-Match-Return(group, agg)` into `Match-Match-With
    // (group, agg)-Return(project)` so the existing aggregate-fusion +
    // top-K pipeline picks it up.
    desugar_multi_match_return_aggregate(query);
    fuse_spatial_join(query, graph);
    // Cross-clause MATCH ordering uses connection-type total counts as an
    // O(1) cost proxy. Runs before pattern_start_node so reversal sees
    // the post-reorder clause sequence and tracks `bound_vars` correctly.
    reorder_match_clauses(query, graph);
    optimize_pattern_start_node(query, graph);
    reorder_match_patterns(query, graph);
    push_limit_into_match(query, graph);
    push_distinct_into_match(query);
    fuse_anchored_edge_count(query, graph);
    fuse_count_short_circuits(query);
    fuse_optional_match_aggregate(query);
    fuse_match_return_aggregate(query);
    fuse_match_with_aggregate(query);
    // Run top-K absorption AFTER fuse_match_with_aggregate (which produces
    // FusedMatchWithAggregate) but BEFORE fuse_order_by_top_k (which would
    // otherwise consume the downstream RETURN+ORDER BY+LIMIT into a
    // FusedOrderByTopK that no longer matches our pattern).
    fuse_match_with_aggregate_top_k(query);
    fuse_node_scan_aggregate(query);
    fuse_node_scan_top_k(query);
    fuse_vector_score_order_limit(query);
    fuse_order_by_top_k(query);
    reorder_predicates_by_cost(query);
    mark_fast_var_length_paths(query);
    mark_skip_target_type_check(query, graph);
}

/// Recursively optimize queries nested inside UNION clauses.
fn optimize_nested_queries(
    query: &mut CypherQuery,
    graph: &DirGraph,
    params: &HashMap<String, Value>,
) {
    for clause in &mut query.clauses {
        if let Clause::Union(ref mut u) = clause {
            optimize(&mut u.query, graph, params);
        }
    }
}

/// Mark variable-length edges that don't need path tracking.
///
/// When a MATCH clause has no path assignments (`p = ...`) and the edge
/// has no named variable (`[r:T*1..N]`), the full path vector is never
/// read downstream.  Setting `needs_path_info = false` lets the pattern
/// executor use a fast BFS with global dedup instead of tracking every path.
fn mark_fast_var_length_paths(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        let mc = match clause {
            Clause::Match(mc) | Clause::OptionalMatch(mc) => mc,
            _ => continue,
        };

        // If there are path assignments, path info is needed for all patterns
        if !mc.path_assignments.is_empty() {
            continue;
        }

        for pattern in &mut mc.patterns {
            for element in &mut pattern.elements {
                if let PatternElement::Edge(ep) = element {
                    if ep.var_length.is_some() && ep.variable.is_none() {
                        ep.needs_path_info = false;
                    }
                }
            }
        }
    }
}

/// Skip node type checks when the connection type metadata guarantees the target type.
///
/// For a pattern like `(a:Person)-[:AUTHORED]->(b:Paper)`, if `AUTHORED` edges
/// only ever connect Person→Paper, then checking `node_weight(target).node_type`
/// in the BFS inner loop is redundant. This saves one `StableDiGraph` slab
/// dereference per visited node.
fn mark_skip_target_type_check(query: &mut CypherQuery, graph: &DirGraph) {
    use crate::graph::core::pattern_matching::EdgeDirection;

    for clause in &mut query.clauses {
        let mc = match clause {
            Clause::Match(mc) | Clause::OptionalMatch(mc) => mc,
            _ => continue,
        };

        for pattern in &mut mc.patterns {
            let elements = &mut pattern.elements;
            // Walk elements in triples: Node, Edge, Node
            let len = elements.len();
            for i in 0..len {
                if i + 2 >= len {
                    break;
                }
                // Extract edge and target node info without overlapping borrows
                let (conn_type, direction, target_node_type) = {
                    let edge = match &elements[i + 1] {
                        PatternElement::Edge(ep) => ep,
                        _ => continue,
                    };
                    let target = match &elements[i + 2] {
                        PatternElement::Node(np) => np,
                        _ => continue,
                    };
                    match (&edge.connection_type, edge.direction, &target.node_type) {
                        (Some(ct), dir, Some(nt)) => (ct.clone(), dir, nt.clone()),
                        _ => continue,
                    }
                };

                // Look up connection type metadata
                if let Some(info) = graph.connection_type_metadata.get(&conn_type) {
                    let guaranteed = match direction {
                        EdgeDirection::Outgoing => {
                            info.target_types.len() == 1
                                && info.target_types.contains(&target_node_type)
                        }
                        EdgeDirection::Incoming => {
                            info.source_types.len() == 1
                                && info.source_types.contains(&target_node_type)
                        }
                        EdgeDirection::Both => false, // can't guarantee for bidirectional
                    };
                    if guaranteed {
                        if let PatternElement::Edge(ep) = &mut elements[i + 1] {
                            ep.skip_target_type_check = true;
                        }
                    }
                }
            }
        }
    }
}

// Historical note: the fusion docstrings for `FusedCountAll`,
// `FusedCountByType`, `FusedCountEdgesByType`, and
// `FusedCountAnchoredEdges` moved to their respective fuse functions in
// `src/graph/languages/cypher/planner/fusion.rs` during the Phase 9
// split. See those functions for the current prose.

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "planner_tests.rs"]
mod tests;
