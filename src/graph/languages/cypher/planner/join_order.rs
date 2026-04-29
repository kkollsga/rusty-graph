//! Join-order optimisation — pick pattern-start nodes, reorder MATCH
//! patterns by estimated selectivity.

use super::super::ast::*;
use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher};
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use std::collections::HashSet;

pub(super) fn optimize_pattern_start_node(query: &mut CypherQuery, graph: &DirGraph) {
    use crate::graph::core::pattern_matching::EdgeDirection;

    // Track variables bound by earlier clauses so an unconstrained pattern
    // node like `(p)` in M2 — which will be pre-bound at runtime — is treated
    // as effectively-anchored (selectivity 1). Without this, the planner sees
    // `(p)-[:T]->(target:Type)` and reverses it because `(p)` is statically
    // unconstrained → looks worst-case, even though it'll resolve to a single
    // pre-bound NodeIndex when the executor reaches this clause.
    let mut bound_vars: HashSet<String> = HashSet::new();

    for clause in &mut query.clauses {
        let (patterns, path_assignments) = match clause {
            Clause::Match(m) => (&mut m.patterns, &m.path_assignments),
            Clause::OptionalMatch(m) => (&mut m.patterns, &m.path_assignments),
            // Other clauses don't introduce node bindings the optimizer cares
            // about; advance without modifying patterns.
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

            // Reversing is safe for undirected and variable-length edges:
            // - `Both` flips to `Both` (identity).
            // - Var-length without path assignment is symmetric — `(a)-[*1..3]-(b)`
            //   is `(b)-[*1..3]-(a)` and `(a)-[*1..3]->(b)` reversed yields
            //   `(b)<-[*1..3]-(a)` (same edges traversed in reverse). Path-bound
            //   patterns are protected by the `path_assignments` check above.
            // No early-exit needed for direction/var-length anymore.

            let first_sel = estimate_node_selectivity_in_context(first_node, graph, &bound_vars);
            let last_sel = estimate_node_selectivity_in_context(last_node, graph, &bound_vars);

            // Only reverse if last node is significantly more selective (5× threshold).
            // A 5x advantage already saves 80% of expansion work. `saturating_mul`
            // because unconstrained nodes report `usize::MAX` and would otherwise
            // overflow.
            if last_sel.saturating_mul(5) >= first_sel {
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

        // Accumulate node variables introduced by this clause's patterns so
        // subsequent clauses see them as bound.
        for pattern in patterns.iter() {
            for elem in &pattern.elements {
                if let PatternElement::Node(np) = elem {
                    if let Some(ref v) = np.variable {
                        bound_vars.insert(v.clone());
                    }
                }
            }
        }
    }
}

/// Selectivity estimate that knows about variables bound by earlier clauses.
/// A pre-bound node resolves to a single NodeIndex at runtime, so its
/// effective candidate count is 1 — the most selective possible.
fn estimate_node_selectivity_in_context(
    np: &crate::graph::core::pattern_matching::NodePattern,
    graph: &DirGraph,
    bound_vars: &HashSet<String>,
) -> usize {
    if let Some(ref v) = np.variable {
        if bound_vars.contains(v) {
            return 1;
        }
    }
    estimate_node_selectivity(np, graph)
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

    // Unconstrained nodes (no type, no properties) match every node in the
    // graph — they represent the *worst* possible start node. Returning
    // `usize::MAX` ensures the optimizer never picks an unconstrained node
    // over a constrained one, regardless of how the graph is populated.
    // (On a freshly-created graph, `type_count = 0`, which would otherwise
    // make unconstrained nodes look maximally selective.)
    let unconstrained = np.node_type.is_none();
    // Floor typed-no-property and empty-property branches at 1 so they never
    // beat a legitimately-anchored node (`{id: X}` returns 1, pre-bound vars
    // also map to 1 in the in-context estimator). Without the floor, a typed
    // node on an empty graph reports 0 and the optimizer reverses toward it
    // even when the other end is a bound anchor.
    match &np.properties {
        None if unconstrained => usize::MAX,
        None => type_count.max(1),
        Some(props) if props.is_empty() && unconstrained => usize::MAX,
        Some(props) if props.is_empty() => type_count.max(1),
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
    let mut bound_vars: HashSet<String> = HashSet::new();

    for clause in &mut query.clauses {
        let mc = match clause {
            Clause::Match(mc) => mc,
            Clause::OptionalMatch(mc) => {
                // OPTIONAL MATCH still binds vars for downstream clauses;
                // accumulate but don't reorder OPTIONAL MATCH patterns.
                for pat in mc.patterns.iter() {
                    for elem in &pat.elements {
                        if let PatternElement::Node(np) = elem {
                            if let Some(ref v) = np.variable {
                                bound_vars.insert(v.clone());
                            }
                        }
                    }
                }
                continue;
            }
            _ => continue,
        };
        if mc.patterns.len() < 2 || !mc.path_assignments.is_empty() {
            // Still accumulate vars even when not reordering.
            for pat in mc.patterns.iter() {
                for elem in &pat.elements {
                    if let PatternElement::Node(np) = elem {
                        if let Some(ref v) = np.variable {
                            bound_vars.insert(v.clone());
                        }
                    }
                }
            }
            continue;
        }
        // Estimate selectivity for each pattern based on its start node,
        // accounting for variables already bound by prior clauses.
        let mut pattern_scores: Vec<(usize, usize)> = mc
            .patterns
            .iter()
            .enumerate()
            .map(|(i, pat)| {
                let sel = if let Some(PatternElement::Node(np)) = pat.elements.first() {
                    estimate_node_selectivity_in_context(np, graph, &bound_vars)
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
        if !already_ordered {
            let old_patterns = std::mem::take(&mut mc.patterns);
            mc.patterns = pattern_scores
                .iter()
                .map(|&(idx, _)| old_patterns[idx].clone())
                .collect();
        }

        // Accumulate vars from this clause's (possibly reordered) patterns.
        for pat in mc.patterns.iter() {
            for elem in &pat.elements {
                if let PatternElement::Node(np) = elem {
                    if let Some(ref v) = np.variable {
                        bound_vars.insert(v.clone());
                    }
                }
            }
        }
    }
}
