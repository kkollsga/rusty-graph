//! Join-order optimisation — pick pattern-start nodes, reorder MATCH
//! patterns by estimated selectivity.

use super::super::ast::*;
use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher};
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;

pub(super) fn optimize_pattern_start_node(query: &mut CypherQuery, graph: &DirGraph) {
    use crate::graph::core::pattern_matching::EdgeDirection;

    for clause in &mut query.clauses {
        let (patterns, path_assignments) = match clause {
            Clause::Match(m) => (&mut m.patterns, &m.path_assignments),
            Clause::OptionalMatch(m) => (&mut m.patterns, &m.path_assignments),
            _ => continue,
        };
        for (pi, pattern) in patterns.iter_mut().enumerate() {
            if pattern.elements.len() < 3 {
                continue;
            }
            // Don't reverse patterns with path assignments — breaks path semantics
            if path_assignments.iter().any(|pa| pa.pattern_index == pi) {
                continue;
            }

            let first_node = match &pattern.elements[0] {
                PatternElement::Node(np) => np,
                _ => continue,
            };
            let last_node = match pattern.elements.last() {
                Some(PatternElement::Node(np)) => np,
                _ => continue,
            };

            // Don't reverse if any edge is undirected or variable-length
            let has_unsupported_edge = pattern.elements.iter().any(|elem| {
                if let PatternElement::Edge(ep) = elem {
                    ep.direction == EdgeDirection::Both || ep.var_length.is_some()
                } else {
                    false
                }
            });
            if has_unsupported_edge {
                continue;
            }

            let first_sel = estimate_node_selectivity(first_node, graph);
            let last_sel = estimate_node_selectivity(last_node, graph);

            // Only reverse if last node is significantly more selective (5× threshold).
            // A 5x advantage already saves 80% of expansion work.
            if last_sel * 5 >= first_sel {
                continue;
            }

            // Reverse: flip element order and flip each edge direction
            pattern.elements.reverse();
            for elem in &mut pattern.elements {
                if let PatternElement::Edge(ep) = elem {
                    ep.direction = match ep.direction {
                        EdgeDirection::Outgoing => EdgeDirection::Incoming,
                        EdgeDirection::Incoming => EdgeDirection::Outgoing,
                        EdgeDirection::Both => EdgeDirection::Both,
                    };
                }
            }
        }
    }
}

/// Estimate the number of candidate nodes for a node pattern.
/// Lower = more selective = better as start node.
pub(super) fn estimate_node_selectivity(
    np: &crate::graph::core::pattern_matching::NodePattern,
    graph: &DirGraph,
) -> usize {
    let type_count = np
        .node_type
        .as_ref()
        .and_then(|t| graph.type_indices.get(t))
        .map(|idx| idx.len())
        .unwrap_or(GraphRead::node_count(&graph.graph));

    match &np.properties {
        None => type_count,
        Some(props) if props.is_empty() => type_count,
        Some(props) => {
            // {id: X} is always selectivity 1 regardless of type
            for (prop, matcher) in props {
                if prop == "id" {
                    match matcher {
                        PropertyMatcher::Equals(_) | PropertyMatcher::EqualsParam(_) => return 1,
                        PropertyMatcher::In(vals) => return vals.len(),
                        _ => {}
                    }
                }
            }
            // Check if any property has equality on an indexed field
            if let Some(ref nt) = np.node_type {
                for (prop, matcher) in props {
                    match matcher {
                        PropertyMatcher::Equals(val) => {
                            let key = (nt.clone(), prop.clone());
                            if graph.property_indices.contains_key(&key) {
                                if let Some(results) = graph.lookup_by_index(nt, prop, val) {
                                    return results.len().max(1);
                                }
                                return 1;
                            }
                        }
                        PropertyMatcher::In(vals) => return vals.len(),
                        _ => {}
                    }
                }
            }
            // Heuristic: equality filters on string properties typically have many
            // distinct values (~100x reduction each). Range/comparison filters are
            // gentler (~10x). Multiple filters multiply the reduction.
            let eq_count = props
                .values()
                .filter(|m| {
                    matches!(
                        m,
                        PropertyMatcher::Equals(_) | PropertyMatcher::EqualsParam(_)
                    )
                })
                .count();
            let other_count = props.len() - eq_count;
            let mut est = type_count;
            for _ in 0..eq_count {
                est /= 100;
            }
            for _ in 0..other_count {
                est /= 10;
            }
            est.max(1)
        }
    }
}

/// Reorder patterns within a MATCH clause so the most selective pattern runs first.
///
/// For `MATCH (n)-[:P31]->({id:6256}), (n)-[:P30]->({id:46})`, the pattern with
/// the more selective start node should execute first to minimize the number of
/// rows passed to subsequent patterns via shared-variable join.
///
/// Estimates selectivity by looking at the first node of each pattern (after
/// start-node optimization has already picked the best direction).
pub(super) fn reorder_match_patterns(query: &mut CypherQuery, graph: &DirGraph) {
    for clause in &mut query.clauses {
        let mc = match clause {
            Clause::Match(mc) => mc,
            _ => continue,
        };
        if mc.patterns.len() < 2 {
            continue;
        }
        // Don't reorder if there are path assignments — indices would break
        if !mc.path_assignments.is_empty() {
            continue;
        }
        // Estimate selectivity for each pattern based on its start node
        let mut pattern_scores: Vec<(usize, usize)> = mc
            .patterns
            .iter()
            .enumerate()
            .map(|(i, pat)| {
                let sel = if let Some(PatternElement::Node(np)) = pat.elements.first() {
                    estimate_node_selectivity(np, graph)
                } else {
                    usize::MAX
                };
                (i, sel)
            })
            .collect();

        // Sort by selectivity (lower = more selective = should go first)
        pattern_scores.sort_by_key(|&(_, sel)| sel);

        // Only reorder if the order actually changes
        let already_ordered = pattern_scores
            .iter()
            .enumerate()
            .all(|(pos, &(idx, _))| pos == idx);
        if already_ordered {
            continue;
        }

        // Rebuild patterns in selectivity order
        let old_patterns = std::mem::take(&mut mc.patterns);
        mc.patterns = pattern_scores
            .iter()
            .map(|&(idx, _)| old_patterns[idx].clone())
            .collect();
    }
}
