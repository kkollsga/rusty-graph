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
    fuse_match_with_aggregate, fuse_node_scan_aggregate, fuse_node_scan_top_k,
    fuse_optional_match_aggregate, fuse_order_by_top_k, fuse_spatial_join,
    fuse_vector_score_order_limit,
};
use index_selection::push_where_into_match;
use join_order::{optimize_pattern_start_node, reorder_match_patterns};
use simplification::{fold_or_to_in, push_distinct_into_match, push_limit_into_match};

pub fn optimize(query: &mut CypherQuery, graph: &DirGraph, params: &HashMap<String, Value>) {
    // Recursively optimize nested queries (e.g., UNION right-arm)
    optimize_nested_queries(query, graph, params);
    push_where_into_match(query, params);
    fold_or_to_in(query);
    push_where_into_match(query, params); // second pass: push newly-created IN predicates
    fuse_spatial_join(query, graph);
    optimize_pattern_start_node(query, graph);
    reorder_match_patterns(query, graph);
    push_limit_into_match(query, graph);
    push_distinct_into_match(query);
    fuse_anchored_edge_count(query, graph);
    fuse_count_short_circuits(query);
    fuse_optional_match_aggregate(query);
    fuse_match_return_aggregate(query);
    fuse_match_with_aggregate(query);
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
mod tests {
    use super::*;
    use crate::graph::core::pattern_matching::PropertyMatcher;
    use crate::graph::languages::cypher::parser::parse_cypher;

    #[test]
    fn test_predicate_pushdown_simple() {
        let mut query = parse_cypher("MATCH (n:Person) WHERE n.age = 30 RETURN n").unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        // WHERE is kept as a safety net even when all predicates are pushed
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[2], Clause::Return(_)));

        // The MATCH pattern should now have {age: 30} as a property
        if let Clause::Match(m) = &query.clauses[0] {
            if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
                assert!(np.properties.is_some());
                let props = np.properties.as_ref().unwrap();
                assert!(props.contains_key("age"));
            } else {
                panic!("Expected node pattern");
            }
        }
    }

    #[test]
    fn test_predicate_pushdown_partial() {
        let mut query =
            parse_cypher("MATCH (n:Person) WHERE n.age = 30 AND n.score > 100 RETURN n").unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        // Both n.age = 30 and n.score > 100 should be pushed into MATCH
        // WHERE is kept as a safety net
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

        if let Clause::Match(m) = &query.clauses[0] {
            if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
                let props = np.properties.as_ref().unwrap();
                assert!(matches!(
                    props.get("age"),
                    Some(PropertyMatcher::Equals(Value::Int64(30)))
                ));
                assert!(matches!(
                    props.get("score"),
                    Some(PropertyMatcher::GreaterThan(Value::Int64(100)))
                ));
            }
        }
    }

    #[test]
    fn test_comparison_pushdown() {
        let mut query = parse_cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        // Comparison should be pushed into MATCH, WHERE kept as safety net
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

        if let Clause::Match(m) = &query.clauses[0] {
            if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
                let props = np.properties.as_ref().unwrap();
                assert!(matches!(
                    props.get("age"),
                    Some(PropertyMatcher::GreaterThan(Value::Int64(30)))
                ));
            }
        }
    }

    #[test]
    fn test_no_pushdown_for_not_equals() {
        let mut query = parse_cypher("MATCH (n:Person) WHERE n.age <> 30 RETURN n").unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        // NotEquals should NOT be pushed - WHERE should remain
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN
    }

    #[test]
    fn test_predicate_pushdown_parameter() {
        let mut query = parse_cypher("MATCH (n:Person) WHERE n.name = $name RETURN n").unwrap();

        let graph = DirGraph::new();
        let mut params = HashMap::new();
        params.insert("name".to_string(), Value::String("Alice".to_string()));
        optimize(&mut query, &graph, &params);

        // Parameter resolved and pushed; WHERE kept as safety net
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

        // The MATCH pattern should now have {name: 'Alice'} as a property
        if let Clause::Match(m) = &query.clauses[0] {
            if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
                assert!(np.properties.is_some());
                let props = np.properties.as_ref().unwrap();
                assert!(props.contains_key("name"));
                assert!(matches!(
                    props.get("name"),
                    Some(PropertyMatcher::Equals(Value::String(s))) if s == "Alice"
                ));
            } else {
                panic!("Expected node pattern");
            }
        }
    }

    #[test]
    fn test_predicate_pushdown_parameter_partial() {
        let mut query =
            parse_cypher("MATCH (n:Person) WHERE n.name = $name AND n.age > $min_age RETURN n")
                .unwrap();

        let graph = DirGraph::new();
        let mut params = HashMap::new();
        params.insert("name".to_string(), Value::String("Alice".to_string()));
        params.insert("min_age".to_string(), Value::Int64(25));
        optimize(&mut query, &graph, &params);

        // Both should be pushed: n.name = $name (equality) and n.age > $min_age (comparison)
        // WHERE kept as safety net
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

        if let Clause::Match(m) = &query.clauses[0] {
            if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
                let props = np.properties.as_ref().unwrap();
                assert!(matches!(
                    props.get("name"),
                    Some(PropertyMatcher::Equals(Value::String(s))) if s == "Alice"
                ));
                assert!(matches!(
                    props.get("age"),
                    Some(PropertyMatcher::GreaterThan(Value::Int64(25)))
                ));
            }
        }
    }

    #[test]
    fn test_comparison_range_merge() {
        let mut query =
            parse_cypher("MATCH (n:Paper) WHERE n.year >= 2015 AND n.year <= 2022 RETURN n")
                .unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        // Both comparisons should be merged into a Range matcher; WHERE kept
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

        if let Clause::Match(m) = &query.clauses[0] {
            if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
                let props = np.properties.as_ref().unwrap();
                assert!(matches!(
                    props.get("year"),
                    Some(PropertyMatcher::Range {
                        lower: Value::Int64(2015),
                        lower_inclusive: true,
                        upper: Value::Int64(2022),
                        upper_inclusive: true,
                    })
                ));
            }
        }
    }

    #[test]
    fn test_correlated_nodeprop_pushdown() {
        // The classic shape from sodir-prospect build:
        //   MATCH (a:A) MATCH (b:B) WHERE b.x = a.y
        // should push { x: EqualsNodeProp { var: "a", prop: "y" } } onto B.
        let mut query =
            parse_cypher("MATCH (a:A) MATCH (b:B) WHERE b.x = a.y RETURN a.id, b.id").unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        // Locate the second MATCH (matching on B)
        let b_match = query
            .clauses
            .iter()
            .filter_map(|c| match c {
                Clause::Match(m) => Some(m),
                _ => None,
            })
            .find(|m| {
                matches!(
                    &m.patterns[0].elements[0],
                    PatternElement::Node(np) if np.node_type.as_deref() == Some("B")
                )
            })
            .expect("expected second MATCH on B");

        if let PatternElement::Node(np) = &b_match.patterns[0].elements[0] {
            let props = np.properties.as_ref().expect("expected props on b");
            match props.get("x") {
                Some(PropertyMatcher::EqualsNodeProp { var, prop }) => {
                    assert_eq!(var, "a");
                    assert_eq!(prop, "y");
                }
                other => panic!("expected EqualsNodeProp on b.x, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_correlated_nodeprop_reversed_sides() {
        // Reversed: a.y = b.x (the cur_var b appears on the right).
        let mut query =
            parse_cypher("MATCH (a:A) MATCH (b:B) WHERE a.y = b.x RETURN a.id, b.id").unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        let b_match = query
            .clauses
            .iter()
            .filter_map(|c| match c {
                Clause::Match(m) => Some(m),
                _ => None,
            })
            .find(|m| {
                matches!(
                    &m.patterns[0].elements[0],
                    PatternElement::Node(np) if np.node_type.as_deref() == Some("B")
                )
            })
            .unwrap();

        if let PatternElement::Node(np) = &b_match.patterns[0].elements[0] {
            let props = np.properties.as_ref().unwrap();
            assert!(matches!(
                props.get("x"),
                Some(PropertyMatcher::EqualsNodeProp { var, prop })
                    if var == "a" && prop == "y"
            ));
        }
    }

    #[test]
    fn test_scalar_var_pushdown_from_unwind() {
        // WHERE s.title = fname where fname comes from UNWIND should become
        // an EqualsVar matcher on the pattern.
        let mut query = parse_cypher(
            "UNWIND ['x','y'] AS fname MATCH (s:Strat) WHERE s.title = fname RETURN s.id",
        )
        .unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        let s_match = query
            .clauses
            .iter()
            .filter_map(|c| match c {
                Clause::Match(m) => Some(m),
                _ => None,
            })
            .next()
            .unwrap();

        if let PatternElement::Node(np) = &s_match.patterns[0].elements[0] {
            let props = np.properties.as_ref().unwrap();
            assert!(matches!(
                props.get("title"),
                Some(PropertyMatcher::EqualsVar(n)) if n == "fname"
            ));
        }
    }

    #[test]
    fn test_no_pushdown_when_both_vars_in_same_match() {
        // Within the same MATCH, a.y = b.x is handled by the pattern
        // executor's shared-var join. We must NOT rewrite it as an
        // EqualsNodeProp (which assumes prior-bound node).
        let mut query =
            parse_cypher("MATCH (a:A), (b:B) WHERE a.y = b.x RETURN a.id, b.id").unwrap();

        let graph = DirGraph::new();
        let params = HashMap::new();
        optimize(&mut query, &graph, &params);

        for clause in &query.clauses {
            if let Clause::Match(m) = clause {
                for pat in &m.patterns {
                    for el in &pat.elements {
                        if let PatternElement::Node(np) = el {
                            if let Some(props) = &np.properties {
                                for m in props.values() {
                                    assert!(
                                        !matches!(m, PropertyMatcher::EqualsNodeProp { .. }),
                                        "same-MATCH correlated equality must not be rewritten"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
