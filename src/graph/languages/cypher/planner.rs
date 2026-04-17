// src/graph/cypher/planner.rs
// Query optimizer: predicate pushdown, index hints, limit pushdown

use super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher};
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use std::collections::HashMap;

/// Optimize a parsed Cypher query before execution.
/// Accepts query parameters so that `WHERE n.prop = $param` can be pushed
/// into MATCH patterns the same way literal equalities are.
pub fn optimize(query: &mut CypherQuery, graph: &DirGraph, params: &HashMap<String, Value>) {
    // Recursively optimize nested queries (e.g., UNION right-arm)
    optimize_nested_queries(query, graph, params);
    push_where_into_match(query, params);
    fold_or_to_in(query);
    push_where_into_match(query, params); // second pass: push newly-created IN predicates
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

/// Fuse MATCH + RETURN into O(1)/O(types) count short-circuits.
///
/// Detects three patterns and replaces them with specialized fused clauses:
/// - `MATCH (n) RETURN count(n)` → `FusedCountAll` (O(1))
/// - `MATCH (n) RETURN n.type, count(n)` → `FusedCountByType` (O(types))
/// - `MATCH ()-[r]->() RETURN type(r), count(*)` → `FusedCountEdgesByType` (O(E) single pass)
///
/// Any trailing ORDER BY / LIMIT clauses are left in place since they
/// operate on the tiny fused result set.
/// Fuse `MATCH (var)-[r:TYPE?]->({id: V}) RETURN count(var)` (and the three
/// symmetric variants) into `FusedCountAnchoredEdges` — O(log D) offset
/// arithmetic on the anchored node's CSR range. When `has_tombstones` is
/// false the executor takes a pure subtraction; otherwise it still walks the
/// narrowed range.
///
/// Conservative: if the anchor's `{id: V}` literal doesn't resolve to any
/// known node (e.g. typo, or a value from a different graph), skip fusion and
/// let the normal executor return 0 via the usual path. That preserves
/// Cypher semantics.
fn fuse_anchored_edge_count(query: &mut CypherQuery, graph: &DirGraph) {
    use crate::graph::core::pattern_matching::{EdgeDirection, PropertyMatcher};

    if query.clauses.len() < 2 {
        return;
    }
    let is_match_return = matches!(
        (&query.clauses[0], &query.clauses[1]),
        (Clause::Match(_), Clause::Return(_))
    );
    if !is_match_return {
        return;
    }
    let match_clause = if let Clause::Match(m) = &query.clauses[0] {
        m
    } else {
        return;
    };
    let return_clause = if let Clause::Return(r) = &query.clauses[1] {
        r
    } else {
        return;
    };
    if return_clause.distinct || return_clause.having.is_some() {
        return;
    }
    if match_clause.patterns.len() != 1 || !match_clause.path_assignments.is_empty() {
        return;
    }
    let pat = &match_clause.patterns[0];
    if pat.elements.len() != 3 {
        return;
    }

    let src_node = match &pat.elements[0] {
        PatternElement::Node(np) => np,
        _ => return,
    };
    let edge = match &pat.elements[1] {
        PatternElement::Edge(ep) => ep,
        _ => return,
    };
    let tgt_node = match &pat.elements[2] {
        PatternElement::Node(np) => np,
        _ => return,
    };

    if edge.properties.is_some() || edge.var_length.is_some() {
        return;
    }
    if edge.direction == EdgeDirection::Both {
        return;
    }

    // Helper: does the node look like a pure `{id: VAL}` literal anchor —
    // no type, no variable, exactly one property keyed `id` with a literal
    // Equals matcher? Returns the id value on match.
    let as_anchor_id = |np: &crate::graph::core::pattern_matching::NodePattern| -> Option<Value> {
        if np.node_type.is_some() || np.variable.is_some() {
            return None;
        }
        let props = np.properties.as_ref()?;
        if props.len() != 1 {
            return None;
        }
        if let Some(PropertyMatcher::Equals(val)) = props.get("id") {
            Some(val.clone())
        } else {
            None
        }
    };
    // Helper: the other side is a named variable with no type/property filter.
    fn as_pure_var(np: &crate::graph::core::pattern_matching::NodePattern) -> Option<&String> {
        if np.node_type.is_some() || np.properties.is_some() {
            return None;
        }
        np.variable.as_ref()
    }

    let (var_name, anchor_val, anchor_dir) = match (as_pure_var(src_node), as_anchor_id(tgt_node)) {
        (Some(v), Some(id)) => {
            // var -[edge]-> {id: V}
            // anchor is the TARGET; traverse from anchor in the opposite dir.
            let dir = match edge.direction {
                EdgeDirection::Outgoing => petgraph::Direction::Incoming,
                EdgeDirection::Incoming => petgraph::Direction::Outgoing,
                EdgeDirection::Both => return,
            };
            (v, id, dir)
        }
        _ => match (as_anchor_id(src_node), as_pure_var(tgt_node)) {
            (Some(id), Some(v)) => {
                // {id: V} -[edge]-> var
                let dir = match edge.direction {
                    EdgeDirection::Outgoing => petgraph::Direction::Outgoing,
                    EdgeDirection::Incoming => petgraph::Direction::Incoming,
                    EdgeDirection::Both => return,
                };
                (v, id, dir)
            }
            _ => return,
        },
    };

    // RETURN must be exactly one item, which is count(var) or count(*).
    if return_clause.items.len() != 1 {
        return;
    }
    if !is_count_of_var_or_star(&return_clause.items[0].expression, Some(var_name)) {
        return;
    }

    // Resolve the anchor across node types. O(types) HashMap lookups; at
    // typical schema sizes this is negligible, and on Wikidata-scale (~88 k
    // types) we still only do one `HashMap::get` per type.
    let mut resolved: Option<petgraph::graph::NodeIndex> = None;
    for node_type in graph.type_indices.keys() {
        if let Some(idx) = graph.lookup_by_id_readonly(node_type, &anchor_val) {
            resolved = Some(idx);
            break;
        }
    }
    let anchor_idx = match resolved {
        Some(idx) => idx.index() as u32,
        None => return, // anchor not found — leave unfused, normal path returns 0
    };

    let alias = return_item_column_name(&return_clause.items[0]);
    let edge_type = edge.connection_type.clone();

    query.clauses.drain(0..2);
    query.clauses.insert(
        0,
        Clause::FusedCountAnchoredEdges {
            anchor_idx,
            anchor_direction: anchor_dir,
            edge_type,
            alias,
        },
    );
}

fn fuse_count_short_circuits(query: &mut CypherQuery) {
    use crate::graph::core::pattern_matching::EdgeDirection;

    if query.clauses.len() < 2 {
        return;
    }

    // First two clauses must be Match + Return
    let is_match_return = matches!(
        (&query.clauses[0], &query.clauses[1]),
        (Clause::Match(_), Clause::Return(_))
    );
    if !is_match_return {
        return;
    }

    let match_clause = if let Clause::Match(m) = &query.clauses[0] {
        m
    } else {
        return;
    };
    let return_clause = if let Clause::Return(r) = &query.clauses[1] {
        r
    } else {
        return;
    };

    // No DISTINCT on RETURN
    if return_clause.distinct {
        return;
    }

    // Must have exactly 1 pattern
    if match_clause.patterns.len() != 1 {
        return;
    }
    let pat = &match_clause.patterns[0];

    // ---- Pattern A: MATCH (n) RETURN count(n) / count(*) ----
    //   Also handles: MATCH (n:Type) RETURN count(n)  → FusedCountTypedNode
    if pat.elements.len() == 1 {
        let node = match &pat.elements[0] {
            PatternElement::Node(np) => np,
            _ => return,
        };
        // Cannot short-circuit with property filters
        if node.properties.is_some() {
            return;
        }

        let node_var = node.variable.as_deref();

        // Typed node count: MATCH (n:Type) RETURN count(n)
        if let Some(ref node_type) = node.node_type {
            if return_clause.items.len() == 1
                && is_count_of_var_or_star(&return_clause.items[0].expression, node_var)
            {
                let alias = return_item_column_name(&return_clause.items[0]);
                let nt = node_type.clone();
                query.clauses.drain(0..2);
                query.clauses.insert(
                    0,
                    Clause::FusedCountTypedNode {
                        node_type: nt,
                        alias,
                    },
                );
            }
            return;
        }

        if return_clause.items.len() == 1 {
            // Single item: must be count(var) or count(*)
            let item = &return_clause.items[0];
            if !is_count_of_var_or_star(&item.expression, node_var) {
                return;
            }
            let alias = return_item_column_name(item);
            // Replace Match + Return with FusedCountAll; keep trailing clauses
            query.clauses.drain(0..2);
            query.clauses.insert(0, Clause::FusedCountAll { alias });
            return;
        }

        if return_clause.items.len() == 2 {
            // Two items: one must be n.type / labels(n), the other count(var) / count(*)
            let (type_idx, count_idx) = identify_type_count_pair(&return_clause.items, node_var);
            if let Some((ti, ci)) = type_idx.zip(count_idx) {
                let type_alias = return_item_column_name(&return_clause.items[ti]);
                let count_alias = return_item_column_name(&return_clause.items[ci]);
                query.clauses.drain(0..2);
                query.clauses.insert(
                    0,
                    Clause::FusedCountByType {
                        type_alias,
                        count_alias,
                    },
                );
                return;
            }
        }
        return;
    }

    // ---- Pattern C: MATCH ()-[r]->() RETURN type(r), count(*) ----
    //   Also handles: MATCH ()-[r:Type]->() RETURN count(*)  → FusedCountTypedEdge
    if pat.elements.len() == 3 {
        let src_node = match &pat.elements[0] {
            PatternElement::Node(np) => np,
            _ => return,
        };
        let edge = match &pat.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return,
        };
        let tgt_node = match &pat.elements[2] {
            PatternElement::Node(np) => np,
            _ => return,
        };

        // Both nodes must be anonymous/unfiltered
        if src_node.node_type.is_some()
            || src_node.properties.is_some()
            || tgt_node.node_type.is_some()
            || tgt_node.properties.is_some()
        {
            return;
        }

        // Edge must have no property filters or var_length, and must be directed
        if edge.properties.is_some()
            || edge.var_length.is_some()
            || edge.direction == EdgeDirection::Both
        {
            return;
        }

        let edge_var = edge.variable.as_deref();

        // Sub-pattern C1: Typed edge count — MATCH ()-[r:Type]->() RETURN count(*)
        if let Some(ref edge_type) = edge.connection_type {
            if return_clause.items.len() == 1
                && is_count_of_var_or_star(&return_clause.items[0].expression, edge_var)
            {
                let alias = return_item_column_name(&return_clause.items[0]);
                let et = edge_type.clone();
                query.clauses.drain(0..2);
                query.clauses.insert(
                    0,
                    Clause::FusedCountTypedEdge {
                        edge_type: et,
                        alias,
                    },
                );
            }
            return;
        }

        // Sub-pattern C2: Untyped edge count by type — MATCH ()-[r]->() RETURN type(r), count(*)
        if return_clause.items.len() != 2 {
            return;
        }

        // Identify type(r) and count(*) / count(r)
        let (type_idx, count_idx) = identify_edge_type_count_pair(&return_clause.items, edge_var);
        if let Some((ti, ci)) = type_idx.zip(count_idx) {
            let type_alias = return_item_column_name(&return_clause.items[ti]);
            let count_alias = return_item_column_name(&return_clause.items[ci]);
            query.clauses.drain(0..2);
            query.clauses.insert(
                0,
                Clause::FusedCountEdgesByType {
                    type_alias,
                    count_alias,
                },
            );
        }
    }
}

/// Check if an expression is `count(var)`, `count(*)`, or `count()` matching the given variable.
fn is_count_of_var_or_star(expr: &Expression, node_var: Option<&str>) -> bool {
    if let Expression::FunctionCall {
        name,
        args,
        distinct,
    } = expr
    {
        if name != "count" || *distinct {
            return false;
        }
        if args.len() == 1 {
            return match &args[0] {
                Expression::Star => true,
                Expression::Variable(v) => node_var.is_some_and(|nv| v == nv),
                _ => false,
            };
        }
    }
    false
}

/// For `RETURN n.type, count(n)` — identify which item is the type accessor and which is the count.
/// Returns (type_item_index, count_item_index) or (None, None) if pattern doesn't match.
fn identify_type_count_pair(
    items: &[ReturnItem],
    node_var: Option<&str>,
) -> (Option<usize>, Option<usize>) {
    let mut type_idx = None;
    let mut count_idx = None;

    for (i, item) in items.iter().enumerate() {
        if is_count_of_var_or_star(&item.expression, node_var) {
            count_idx = Some(i);
        } else if is_node_type_accessor(&item.expression, node_var) {
            type_idx = Some(i);
        }
    }
    (type_idx, count_idx)
}

/// Check if expression is `n.type`, `n.node_type`, `n.label`, or `labels(n)`.
fn is_node_type_accessor(expr: &Expression, node_var: Option<&str>) -> bool {
    match expr {
        Expression::PropertyAccess { variable, property } => {
            let is_type_prop = matches!(property.as_str(), "type" | "node_type" | "label");
            is_type_prop && node_var.is_some_and(|nv| variable == nv)
        }
        Expression::FunctionCall { name, args, .. } => {
            if name == "labels" && args.len() == 1 {
                if let Expression::Variable(v) = &args[0] {
                    return node_var.is_some_and(|nv| v == nv);
                }
            }
            false
        }
        _ => false,
    }
}

/// For `RETURN type(r), count(*)` — identify edge type function and count.
fn identify_edge_type_count_pair(
    items: &[ReturnItem],
    edge_var: Option<&str>,
) -> (Option<usize>, Option<usize>) {
    let mut type_idx = None;
    let mut count_idx = None;

    for (i, item) in items.iter().enumerate() {
        if is_count_of_var_or_star(&item.expression, edge_var) {
            count_idx = Some(i);
        } else if is_edge_type_function(&item.expression, edge_var) {
            type_idx = Some(i);
        }
    }
    (type_idx, count_idx)
}

/// Check if expression is `type(r)`.
fn is_edge_type_function(expr: &Expression, edge_var: Option<&str>) -> bool {
    if let Expression::FunctionCall { name, args, .. } = expr {
        if name == "type" && args.len() == 1 {
            if let Expression::Variable(v) = &args[0] {
                return edge_var.is_some_and(|ev| v == ev);
            }
        }
    }
    false
}

/// Push simple equality predicates from WHERE into MATCH pattern properties.
/// This enables the pattern executor to filter during matching rather than after.
///
/// Fold OR chains of equalities on the same variable.property into IN predicates.
///
/// Example: `WHERE n.name = 'A' OR n.name = 'B' OR n.name = 'C'`
/// Becomes: `WHERE n.name IN ['A', 'B', 'C']`
///
/// This enables predicate pushdown into MATCH patterns and index acceleration.
/// Must run BEFORE `push_where_into_match`.
fn fold_or_to_in(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        if let Clause::Where(ref mut w) = clause {
            w.predicate = fold_or_to_in_pred(&w.predicate);
        }
    }
}

/// Recursively fold OR chains of same-property equalities into IN predicates.
fn fold_or_to_in_pred(pred: &Predicate) -> Predicate {
    match pred {
        Predicate::Or(_, _) => {
            // Collect all OR-chained equality comparisons
            let mut equalities: Vec<(String, String, Expression)> = Vec::new();
            let mut other_preds: Vec<Predicate> = Vec::new();
            collect_or_equalities(pred, &mut equalities, &mut other_preds);

            // Group equalities by (variable, property)
            let mut groups: std::collections::HashMap<(String, String), Vec<Expression>> =
                std::collections::HashMap::new();
            for (var, prop, val_expr) in equalities {
                groups.entry((var, prop)).or_default().push(val_expr);
            }

            // Build result predicates
            let mut result_preds: Vec<Predicate> = Vec::new();

            // Convert groups with 2+ equalities into IN predicates
            for ((var, prop), values) in groups {
                if values.len() >= 2 {
                    result_preds.push(Predicate::In {
                        expr: Expression::PropertyAccess {
                            variable: var,
                            property: prop,
                        },
                        list: values,
                    });
                } else {
                    // Single equality — keep as comparison
                    result_preds.push(Predicate::Comparison {
                        left: Expression::PropertyAccess {
                            variable: var,
                            property: prop,
                        },
                        operator: ComparisonOp::Equals,
                        right: values.into_iter().next().unwrap(),
                    });
                }
            }

            // Add back non-equality predicates (recursively folded)
            for p in other_preds {
                result_preds.push(fold_or_to_in_pred(&p));
            }

            // Combine with OR
            if result_preds.len() == 1 {
                result_preds.pop().unwrap()
            } else {
                let mut combined = result_preds.pop().unwrap();
                for p in result_preds.into_iter().rev() {
                    combined = Predicate::Or(Box::new(p), Box::new(combined));
                }
                combined
            }
        }
        Predicate::And(l, r) => Predicate::And(
            Box::new(fold_or_to_in_pred(l)),
            Box::new(fold_or_to_in_pred(r)),
        ),
        Predicate::Not(inner) => Predicate::Not(Box::new(fold_or_to_in_pred(inner))),
        other => other.clone(),
    }
}

/// Collect equalities from an OR chain. Non-equality predicates go to `others`.
fn collect_or_equalities(
    pred: &Predicate,
    equalities: &mut Vec<(String, String, Expression)>,
    others: &mut Vec<Predicate>,
) {
    match pred {
        Predicate::Or(left, right) => {
            collect_or_equalities(left, equalities, others);
            collect_or_equalities(right, equalities, others);
        }
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals,
            right,
        } => {
            if let Expression::PropertyAccess { variable, property } = left {
                if matches!(right, Expression::Literal(_) | Expression::Parameter(_)) {
                    equalities.push((variable.clone(), property.clone(), right.clone()));
                    return;
                }
            }
            if let Expression::PropertyAccess { variable, property } = right {
                if matches!(left, Expression::Literal(_) | Expression::Parameter(_)) {
                    equalities.push((variable.clone(), property.clone(), left.clone()));
                    return;
                }
            }
            others.push(pred.clone());
        }
        other => {
            others.push(other.clone());
        }
    }
}

/// Example: MATCH (n:Person) WHERE n.age = 30
/// Becomes: MATCH (n:Person {age: 30}) (WHERE removed if fully consumed)
///
/// Also handles parameterized equalities:
/// MATCH (n:Person) WHERE n.age = $min_age  (with params = {min_age: 30})
/// Becomes: MATCH (n:Person {age: 30})
fn push_where_into_match(query: &mut CypherQuery, params: &HashMap<String, Value>) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        let can_push = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Where(_)) | (Clause::OptionalMatch(_), Clause::Where(_))
        );

        if !can_push {
            i += 1;
            continue;
        }

        // Extract the WHERE predicate
        let where_pred = if let Clause::Where(w) = &query.clauses[i + 1] {
            w.predicate.clone()
        } else {
            i += 1;
            continue;
        };

        // Collect variables defined in the MATCH/OPTIONAL MATCH patterns
        let match_vars: Vec<(String, Option<String>)> = match &query.clauses[i] {
            Clause::Match(m) => collect_pattern_variables(&m.patterns),
            Clause::OptionalMatch(m) => collect_pattern_variables(&m.patterns),
            _ => {
                i += 1;
                continue;
            }
        };

        // Split predicate into pushable conditions and remainder
        let (pushable, pushable_in, pushable_cmp, remaining) =
            extract_pushable_equalities(&where_pred, &match_vars, params);

        // Apply pushable conditions to MATCH/OPTIONAL MATCH patterns
        if !pushable.is_empty() || !pushable_in.is_empty() || !pushable_cmp.is_empty() {
            let patterns = match &mut query.clauses[i] {
                Clause::Match(ref mut m) => &mut m.patterns,
                Clause::OptionalMatch(ref mut m) => &mut m.patterns,
                _ => {
                    i += 1;
                    continue;
                }
            };
            for (var_name, property, value) in &pushable {
                apply_property_to_patterns(patterns, var_name, property, value.clone());
            }
            for (var_name, property, values) in pushable_in {
                apply_in_property_to_patterns(patterns, &var_name, &property, values);
            }
            for (var_name, property, op, value) in pushable_cmp {
                apply_comparison_to_patterns(patterns, &var_name, &property, op, value);
            }

            // Update WHERE clause with remaining predicates.
            // When all predicates are pushed into the pattern, keep the WHERE
            // clause as-is so it acts as a safety-net filter. The pushed
            // predicates provide fast-path filtering in the pattern matcher,
            // but the WHERE clause must survive for correctness (e.g. when
            // fuse_match_return_aggregate rejects patterns with properties).
            if let Some(pred) = remaining {
                query.clauses[i + 1] = Clause::Where(WhereClause { predicate: pred });
            }
        }

        i += 1;
    }
}

/// Push LIMIT into MATCH when there's no ORDER BY/aggregation between them.
/// Reverse pattern direction when a later node has a more selective filter
/// than the first node, so the pattern executor starts from fewer candidates.
///
/// Example: `(d:CourtDecision)-[:CITES]->(s)-[:SECTION_OF]->(l:Law {korttittel: 'X'})`
/// → reversed to `(l:Law {korttittel: 'X'})<-[:SECTION_OF]-(s)<-[:CITES]-(d:CourtDecision)`
///
/// Must run AFTER `push_where_into_match` (so equality predicates are already in the pattern).
fn optimize_pattern_start_node(query: &mut CypherQuery, graph: &DirGraph) {
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
fn estimate_node_selectivity(
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
fn reorder_match_patterns(query: &mut CypherQuery, graph: &DirGraph) {
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

/// This allows the pattern executor to stop early via max_matches.
///
/// Safe when: MATCH → [WHERE] → RETURN → LIMIT, with RETURN having no aggregation
/// or DISTINCT. The LIMIT clause is removed from the pipeline and its value is
/// stored in `MatchClause.limit_hint`, which `execute_match` passes to PatternExecutor.
///
/// When WHERE is present between MATCH and RETURN, the LIMIT is still pushed into
/// MATCH as an overcommit hint — the expansion produces at most `limit * 100` candidates,
/// which WHERE then filters. This prevents O(N) expansion on large types even when
/// WHERE prevents exact early termination.
fn push_limit_into_match(query: &mut CypherQuery, _graph: &DirGraph) {
    if query.clauses.len() < 3 {
        return;
    }
    let mut i = 0;
    while i + 2 < query.clauses.len() {
        // Look for MATCH → RETURN → LIMIT  or  MATCH → WHERE → RETURN → LIMIT
        let (has_where, return_offset, limit_offset) = if i + 3 < query.clauses.len()
            && matches!(&query.clauses[i], Clause::Match(_))
            && matches!(&query.clauses[i + 1], Clause::Where(_))
            && matches!(&query.clauses[i + 2], Clause::Return(_))
            && matches!(&query.clauses[i + 3], Clause::Limit(_))
        {
            (true, i + 2, i + 3)
        } else if matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Return(_))
        ) && i + 2 < query.clauses.len()
            && matches!(&query.clauses[i + 2], Clause::Limit(_))
        {
            (false, i + 1, i + 2)
        } else {
            i += 1;
            continue;
        };

        // Safety check: RETURN must have no aggregation, no DISTINCT, no window functions
        let safe = if let Clause::Return(r) = &query.clauses[return_offset] {
            !r.distinct
                && !r
                    .items
                    .iter()
                    .any(|item| super::executor::is_aggregate_expression(&item.expression))
                && !r
                    .items
                    .iter()
                    .any(|item| super::ast::is_window_expression(&item.expression))
        } else {
            false
        };
        if !safe {
            i += 1;
            continue;
        }

        // Extract LIMIT value — must be a literal positive integer
        let limit_val = if let Clause::Limit(l) = &query.clauses[limit_offset] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => Some(*n as usize),
                _ => None,
            }
        } else {
            None
        };
        let Some(limit) = limit_val else {
            i += 1;
            continue;
        };

        // All checks passed: push limit hint into MATCH.
        // The executor fuses WHERE into MATCH (inline evaluation during expansion),
        // so the hint can be exact in both cases. LIMIT clause is removed since
        // the executor stops after finding exactly `limit` matching rows.
        if has_where {
            if let Clause::Match(ref mut m) = query.clauses[i] {
                m.limit_hint = Some(limit);
            }
            query.clauses.remove(limit_offset); // Safe: executor enforces exact limit
        } else {
            if let Clause::Match(ref mut m) = query.clauses[i] {
                m.limit_hint = Some(limit);
            }
            query.clauses.remove(limit_offset); // Remove LIMIT — hint is exact
        }
    }
}

/// Push DISTINCT hint into MATCH when RETURN DISTINCT references a single node variable.
///
/// When all RETURN DISTINCT expressions depend on a single node variable
/// (e.g., `RETURN DISTINCT c2.id` or `RETURN DISTINCT c2.id, c2.name`),
/// the executor can pre-deduplicate pattern matches by that variable's NodeIndex
/// during the MATCH phase, avoiding creation of duplicate ResultRows.
///
/// Detects patterns: MATCH → [WHERE] → RETURN DISTINCT
fn push_distinct_into_match(query: &mut CypherQuery) {
    // Find MATCH + RETURN DISTINCT (with optional WHERE in between)
    for i in 0..query.clauses.len() {
        let match_idx = match &query.clauses[i] {
            Clause::Match(_) => i,
            _ => continue,
        };

        // Find the RETURN clause (skip optional WHERE)
        let return_idx = if match_idx + 1 < query.clauses.len() {
            match &query.clauses[match_idx + 1] {
                Clause::Return(_) => match_idx + 1,
                Clause::Where(_) if match_idx + 2 < query.clauses.len() => {
                    if matches!(&query.clauses[match_idx + 2], Clause::Return(_)) {
                        match_idx + 2
                    } else {
                        continue;
                    }
                }
                _ => continue,
            }
        } else {
            continue;
        };

        // Check: RETURN must be DISTINCT, no aggregation
        let distinct_var = if let Clause::Return(r) = &query.clauses[return_idx] {
            if !r.distinct {
                continue;
            }
            if r.items
                .iter()
                .any(|item| super::executor::is_aggregate_expression(&item.expression))
            {
                continue;
            }
            // All return items must reference a single node variable
            let mut var: Option<&str> = None;
            let mut all_same = true;
            for item in &r.items {
                let v = match &item.expression {
                    Expression::PropertyAccess { variable, .. } => variable.as_str(),
                    Expression::Variable(v) => v.as_str(),
                    _ => {
                        all_same = false;
                        break;
                    }
                };
                match var {
                    None => var = Some(v),
                    Some(prev) if prev == v => {}
                    _ => {
                        all_same = false;
                        break;
                    }
                }
            }
            if all_same {
                var.map(String::from)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(dv) = distinct_var {
            // Verify the variable is a node variable in the MATCH pattern
            if let Clause::Match(ref mc) = &query.clauses[match_idx] {
                let is_node_var = mc.patterns.iter().any(|p| {
                    p.elements.iter().any(|e| {
                        if let crate::graph::core::pattern_matching::PatternElement::Node(np) = e {
                            np.variable.as_deref() == Some(dv.as_str())
                        } else {
                            false
                        }
                    })
                });
                if !is_node_var {
                    continue;
                }
            }
            // Set the hint
            if let Clause::Match(ref mut mc) = query.clauses[match_idx] {
                mc.distinct_node_hint = Some(dv);
            }
        }
    }
}

/// Fuse consecutive OPTIONAL MATCH + WITH (containing count aggregation) into
/// a single `FusedOptionalMatchAggregate` clause. This avoids materializing
/// N×M intermediate rows when only the count is needed.
///
/// Criteria for fusion:
/// 1. `clauses[i]` is `OptionalMatch` and `clauses[i+1]` is `With`
/// 2. The WITH has at least one `count(variable)` aggregate
/// 3. All non-aggregate items in the WITH are simple variable pass-throughs
/// 4. ALL count aggregate variables come from THIS OPTIONAL MATCH pattern
///    (not from earlier OPTIONAL MATCHes — otherwise the fused execution would
///    assign a single match_count to all count columns, producing wrong results)
/// 5. The count aggregates do NOT use DISTINCT (the fused fast-path counts raw
///    matches and cannot perform deduplication)
fn fuse_optional_match_aggregate(query: &mut CypherQuery) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // Note: unlike fuse_match_*_aggregate, this fused executor correctly
        // iterates over existing rows from prior clauses, so no i > 0 guard needed.
        let can_fuse = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::OptionalMatch(_), Clause::With(_))
                | (Clause::OptionalMatch(_), Clause::Return(_))
        );

        if !can_fuse {
            i += 1;
            continue;
        }

        // Check that the WITH/RETURN contains count() aggregation and simple pass-through group keys
        let fusable = match &query.clauses[i + 1] {
            Clause::With(w) => is_fusable_with_clause(w),
            Clause::Return(r) => is_fusable_return_clause(r),
            _ => false,
        };

        if !fusable {
            i += 1;
            continue;
        }

        // Collect variables defined in the OPTIONAL MATCH pattern
        let opt_match_vars: std::collections::HashSet<String> =
            if let Clause::OptionalMatch(m) = &query.clauses[i] {
                collect_pattern_variables(&m.patterns)
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect()
            } else {
                i += 1;
                continue;
            };

        // Verify ALL count aggregate variables come from THIS OPTIONAL MATCH,
        // and none use DISTINCT (which the fused path cannot handle)
        let items = match &query.clauses[i + 1] {
            Clause::With(w) => &w.items,
            Clause::Return(r) => &r.items,
            _ => {
                i += 1;
                continue;
            }
        };
        let all_counts_local = items.iter().all(|item| {
            if let Expression::FunctionCall {
                name,
                args,
                distinct,
            } = &item.expression
            {
                if name == "count" {
                    // Reject DISTINCT — fused path can't deduplicate
                    if *distinct {
                        return false;
                    }
                    // count(*) is always fine
                    if args.len() == 1 && matches!(args[0], Expression::Star) {
                        return true;
                    }
                    // count(var) — var must come from this OPTIONAL MATCH
                    if let Some(Expression::Variable(var)) = args.first() {
                        return opt_match_vars.contains(var);
                    }
                    // count(expr) — not a simple variable, bail
                    return false;
                }
            }
            true // non-aggregate items (group keys) are fine
        });

        if !all_counts_local {
            i += 1;
            continue;
        }

        // Extract both clauses and replace with fused variant.
        // Convert Return → With for the fused representation.
        let with_clause = match query.clauses.remove(i + 1) {
            Clause::With(w) => w,
            Clause::Return(r) => WithClause {
                items: r.items,
                distinct: r.distinct,
                where_clause: r.having.map(|pred| WhereClause { predicate: pred }),
            },
            _ => unreachable!(),
        };
        let match_clause = if let Clause::OptionalMatch(m) = query.clauses.remove(i) {
            m
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedOptionalMatchAggregate {
                match_clause,
                with_clause,
            },
        );

        i += 1;
    }
}

/// Check if a WITH clause is eligible for fusion with an OPTIONAL MATCH.
/// Must have: simple variable group keys + count() aggregates only.
fn is_fusable_with_clause(with: &WithClause) -> bool {
    use super::executor::is_aggregate_expression;

    let mut has_count = false;

    for item in &with.items {
        if is_aggregate_expression(&item.expression) {
            // Only fuse for count() — not sum/collect/avg etc.
            match &item.expression {
                Expression::FunctionCall { name, .. } if name == "count" => {
                    has_count = true;
                }
                _ => return false, // Non-count aggregate → bail
            }
        } else {
            // Group key must be a simple variable pass-through
            if !matches!(&item.expression, Expression::Variable(_)) {
                return false;
            }
        }
    }

    has_count
}

/// Check if a RETURN clause is eligible for fusion with an OPTIONAL MATCH.
/// Same as `is_fusable_with_clause` but allows PropertyAccess group keys
/// (RETURN items can be `l.korttittel`, not just bare `l`).
fn is_fusable_return_clause(ret: &ReturnClause) -> bool {
    use super::executor::is_aggregate_expression;

    let mut has_count = false;

    for item in &ret.items {
        if is_aggregate_expression(&item.expression) {
            // Only fuse for count() — not sum/collect/avg etc.
            match &item.expression {
                Expression::FunctionCall { name, .. } if name == "count" => {
                    has_count = true;
                }
                _ => return false, // Non-count aggregate → bail
            }
        } else {
            // Group key must be a simple variable or property access
            if !matches!(
                &item.expression,
                Expression::Variable(_) | Expression::PropertyAccess { .. }
            ) {
                return false;
            }
        }
    }

    has_count
}

/// Fuse MATCH (node-edge-node) + RETURN (group-by + count) into a single
/// pass that counts edges directly per node instead of materializing all rows.
///
/// Criteria for fusion:
/// 1. `clauses[i]` is `Match` with exactly 1 pattern of 3 elements (node-edge-node)
/// 2. `clauses[i+1]` is `Return` with at least one `count()` aggregate
/// 3. All non-aggregate RETURN items are PropertyAccess on the first node variable
/// 4. All `count()` args reference the second node variable (or `*`)
/// 5. No DISTINCT on count, no property filters on edge or second node
///    (required by `try_count_simple_pattern`)
fn fuse_match_return_aggregate(query: &mut CypherQuery) {
    use super::executor::is_aggregate_expression;

    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // Only fuse when the MATCH is the first clause — a non-first MATCH
        // depends on the pipeline state from prior clauses, which the fused
        // path would ignore.
        if i > 0 {
            i += 1;
            continue;
        }
        let can_fuse = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Return(_))
        );
        if !can_fuse {
            i += 1;
            continue;
        }

        // Check MATCH: exactly 1 pattern with 3 or 5 elements
        let (first_var, second_var, edge_has_props) = if let Clause::Match(m) = &query.clauses[i] {
            let n_elems = m.patterns[0].elements.len();
            if m.patterns.len() != 1 || (n_elems != 3 && n_elems != 5) {
                i += 1;
                continue;
            }
            let pat = &m.patterns[0];
            let first_var = match &pat.elements[0] {
                PatternElement::Node(np) => np.variable.clone(),
                _ => {
                    i += 1;
                    continue;
                }
            };
            let edge_has_props = match &pat.elements[1] {
                PatternElement::Edge(ep) => ep.properties.is_some() || ep.var_length.is_some(),
                _ => {
                    i += 1;
                    continue;
                }
            };

            if n_elems == 5 {
                // 5-element: (a)-[e1]->(b)<-[e2]-(c)
                // Middle node (elements[2]) must have no properties
                let mid_has_props = match &pat.elements[2] {
                    PatternElement::Node(np) => np.properties.is_some(),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let edge2_has_props = match &pat.elements[3] {
                    PatternElement::Edge(ep) => ep.properties.is_some() || ep.var_length.is_some(),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let (last_var, last_has_props) = match &pat.elements[4] {
                    PatternElement::Node(np) => (np.variable.clone(), np.properties.is_some()),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                if mid_has_props || edge2_has_props || last_has_props {
                    i += 1;
                    continue;
                }
                (first_var, last_var, edge_has_props)
            } else {
                // 3-element: (a)-[e]->(b)
                let second_var = match &pat.elements[2] {
                    PatternElement::Node(np) => np.variable.clone(),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                (first_var, second_var, edge_has_props)
            }
        } else {
            i += 1;
            continue;
        };

        // Edge property filters and variable-length edges require the full executor.
        // Node property filters on the second (unbound) node are allowed — the
        // counting loop checks them inline via columnar access.
        if edge_has_props {
            i += 1;
            continue;
        }

        // At least one of first_var / second_var must be named
        if first_var.is_none() && second_var.is_none() {
            i += 1;
            continue;
        }

        // Check RETURN: must have count() aggregate + group-by on one node variable.
        // Determine which variable is the group key (first or second).
        //
        // HAVING is allowed and carried through on the ReturnClause — the fused
        // executor applies it post-aggregation against the small group-by map
        // instead of against the materialised edge-row set.
        let fusable = if let Clause::Return(r) = &query.clauses[i + 1] {
            if r.distinct {
                false
            } else {
                let mut has_count = false;
                let mut all_valid = true;
                let mut group_var: Option<&str> = None;
                let mut count_var_ok = true;

                // First pass: identify which variable group-by items reference
                for item in &r.items {
                    if !is_aggregate_expression(&item.expression) {
                        let refs_var = match &item.expression {
                            Expression::PropertyAccess { variable, .. } => Some(variable.as_str()),
                            Expression::Variable(v) => Some(v.as_str()),
                            _ => None,
                        };
                        match refs_var {
                            Some(v) => {
                                if group_var.is_none() {
                                    group_var = Some(v);
                                } else if group_var != Some(v) {
                                    // Group-by references multiple variables — can't fuse
                                    all_valid = false;
                                    break;
                                }
                            }
                            None => {
                                all_valid = false;
                                break;
                            }
                        }
                    }
                }

                // group_var must be either first_var or second_var
                if all_valid {
                    if let Some(gv) = group_var {
                        let is_first = first_var.as_deref() == Some(gv);
                        let is_second = second_var.as_deref() == Some(gv);
                        if !is_first && !is_second {
                            all_valid = false;
                        }
                    } else {
                        all_valid = false; // no group keys found
                    }
                }

                // Second pass: check count() aggregates
                if all_valid {
                    let other_var = if group_var == first_var.as_deref() {
                        &second_var
                    } else {
                        &first_var
                    };
                    for item in &r.items {
                        if is_aggregate_expression(&item.expression) {
                            match &item.expression {
                                Expression::FunctionCall {
                                    name,
                                    args,
                                    distinct,
                                } if name == "count" => {
                                    if *distinct {
                                        count_var_ok = false;
                                        break;
                                    }
                                    // count(*) is fine
                                    if args.len() == 1 && matches!(args[0], Expression::Star) {
                                        has_count = true;
                                        continue;
                                    }
                                    // count(var) — var must be the OTHER node
                                    if let Some(Expression::Variable(var)) = args.first() {
                                        if other_var.as_deref() == Some(var.as_str()) {
                                            has_count = true;
                                            continue;
                                        }
                                    }
                                    count_var_ok = false;
                                    break;
                                }
                                _ => {
                                    count_var_ok = false;
                                    break;
                                }
                            }
                        }
                    }
                }

                has_count && all_valid && count_var_ok
            }
        } else {
            false
        };

        if !fusable {
            i += 1;
            continue;
        }

        // All checks passed — fuse MATCH + RETURN
        let return_clause = if let Clause::Return(r) = query.clauses.remove(i + 1) {
            r
        } else {
            unreachable!()
        };
        let match_clause = if let Clause::Match(m) = query.clauses.remove(i) {
            m
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedMatchReturnAggregate {
                match_clause,
                return_clause,
                top_k: None,
            },
        );

        i += 1;
    }

    // Second pass: absorb ORDER BY + LIMIT into FusedMatchReturnAggregate
    fuse_aggregate_order_limit(query);
}

/// Absorb ORDER BY + LIMIT into a preceding FusedMatchReturnAggregate.
/// When the sort key is the count aggregate, uses a BinaryHeap to find
/// top-k instead of materializing all rows then sorting.
fn fuse_aggregate_order_limit(query: &mut CypherQuery) {
    use super::executor::is_aggregate_expression;

    let mut i = 0;
    while i + 2 < query.clauses.len() {
        let is_pattern = matches!(
            (
                &query.clauses[i],
                &query.clauses[i + 1],
                &query.clauses[i + 2]
            ),
            (
                Clause::FusedMatchReturnAggregate { .. },
                Clause::OrderBy(_),
                Clause::Limit(_)
            )
        );
        if !is_pattern {
            i += 1;
            continue;
        }

        // Skip fusion when HAVING is present. HAVING must apply on the full
        // aggregated set BEFORE any top-K; absorbing ORDER BY + LIMIT here
        // would flip that order and drop entries that should've passed.
        if let Clause::FusedMatchReturnAggregate { return_clause, .. } = &query.clauses[i] {
            if return_clause.having.is_some() {
                i += 1;
                continue;
            }
        }

        // Extract ORDER BY sort key and LIMIT value
        let (sort_expr_idx, descending) = if let Clause::OrderBy(ob) = &query.clauses[i + 1] {
            if ob.items.len() != 1 {
                i += 1;
                continue; // multi-key sort — bail
            }
            let sort_item = &ob.items[0];
            // Find which RETURN item the sort key references
            if let Clause::FusedMatchReturnAggregate { return_clause, .. } = &query.clauses[i] {
                let mut found_idx = None;
                for (ri, item) in return_clause.items.iter().enumerate() {
                    // Match by alias or by expression
                    let matches_alias =
                        item.alias
                            .as_ref()
                            .is_some_and(|a| match &sort_item.expression {
                                Expression::Variable(v) => v == a,
                                _ => false,
                            });
                    if matches_alias && is_aggregate_expression(&item.expression) {
                        found_idx = Some(ri);
                        break;
                    }
                }
                match found_idx {
                    Some(idx) => (idx, !sort_item.ascending),
                    None => {
                        i += 1;
                        continue;
                    }
                }
            } else {
                i += 1;
                continue;
            }
        } else {
            i += 1;
            continue;
        };

        let limit = if let Clause::Limit(l) = &query.clauses[i + 2] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => *n as usize,
                _ => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        // Absorb ORDER BY + LIMIT into the fused aggregate
        query.clauses.remove(i + 2); // remove LIMIT
        query.clauses.remove(i + 1); // remove ORDER BY
        if let Clause::FusedMatchReturnAggregate { top_k, .. } = &mut query.clauses[i] {
            *top_k = Some((sort_expr_idx, descending, limit));
        }

        i += 1;
    }
}

/// Fuse MATCH (n:Type) [WHERE pred] RETURN group_keys, agg_funcs(...)
/// into a single-pass node scan with inline aggregation.
///
/// Instead of: MATCH creates 20k ResultRows → RETURN groups and aggregates them
/// Fused: iterate nodes directly, evaluate group keys and aggregates from node properties.
fn fuse_node_scan_aggregate(query: &mut CypherQuery) {
    use super::executor::is_aggregate_expression;

    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // Only fuse when the MATCH is the first clause — a non-first MATCH
        // depends on the pipeline state from prior clauses, which the fused
        // path would ignore.
        if i > 0 {
            i += 1;
            continue;
        }
        // Find MATCH + [WHERE] + RETURN pattern
        let match_idx = i;
        if !matches!(&query.clauses[match_idx], Clause::Match(_)) {
            i += 1;
            continue;
        }

        // Check for optional WHERE clause between MATCH and RETURN
        let (where_idx, return_idx) = if i + 2 < query.clauses.len()
            && matches!(&query.clauses[i + 1], Clause::Where(_))
            && matches!(&query.clauses[i + 2], Clause::Return(_))
        {
            (Some(i + 1), i + 2)
        } else if matches!(&query.clauses[i + 1], Clause::Return(_)) {
            (None, i + 1)
        } else {
            i += 1;
            continue;
        };

        // Validate MATCH: single pattern, single node element (no edges).
        // Pushed-down properties (e.g. {city: 'Oslo'}) are allowed — the executor
        // evaluates them inline via PatternExecutor::node_matches_properties_pub().
        // This enables streaming aggregation for queries like:
        //   MATCH (n:Entity) WHERE n.population > 1M RETURN n.continent, count(n)
        let is_single_node = if let Clause::Match(mc) = &query.clauses[match_idx] {
            mc.patterns.len() == 1
                && mc.patterns[0].elements.len() == 1
                && matches!(mc.patterns[0].elements[0], PatternElement::Node(_))
                && mc.path_assignments.is_empty()
        } else {
            false
        };
        if !is_single_node {
            i += 1;
            continue;
        }

        // Validate RETURN: must have supported aggregation (count/sum/avg/min/max only)
        let has_supported_agg = if let Clause::Return(r) = &query.clauses[return_idx] {
            let has_any_agg = r
                .items
                .iter()
                .any(|item| is_aggregate_expression(&item.expression));
            let all_supported = r.items.iter().all(|item| {
                if !is_aggregate_expression(&item.expression) {
                    return true; // group key — OK
                }
                match &item.expression {
                    Expression::FunctionCall { name, distinct, .. } => {
                        if *distinct {
                            return false; // DISTINCT not supported inline
                        }
                        matches!(
                            name.to_lowercase().as_str(),
                            "count" | "sum" | "avg" | "mean" | "average" | "min" | "max"
                        )
                    }
                    _ => false,
                }
            });
            has_any_agg && all_supported
        } else {
            false
        };
        if !has_supported_agg {
            i += 1;
            continue;
        }

        // All checks passed — fuse
        let where_predicate = if let Some(wi) = where_idx {
            if let Clause::Where(w) = query.clauses.remove(wi) {
                // return_idx shifted by 1 after remove
                Some(w.predicate)
            } else {
                None
            }
        } else {
            None
        };

        // Recalculate return_idx after potential WHERE removal
        let ret_idx = if where_idx.is_some() {
            return_idx - 1
        } else {
            return_idx
        };

        let return_clause = if let Clause::Return(r) = query.clauses.remove(ret_idx) {
            r
        } else {
            unreachable!()
        };
        let match_clause = if let Clause::Match(mc) = query.clauses.remove(match_idx) {
            mc
        } else {
            unreachable!()
        };

        query.clauses.insert(
            match_idx,
            Clause::FusedNodeScanAggregate {
                match_clause,
                where_predicate,
                return_clause,
            },
        );

        i += 1;
    }
}

/// Fuse MATCH (node-edge-node) + WITH (group-by + count) into a single
/// pass that counts edges directly per node. Same criteria as
/// `fuse_match_return_aggregate` but targets WITH clauses so the pipeline
/// can continue (e.g., out-degree histogram: WITH p, count(cited) → RETURN).
fn fuse_match_with_aggregate(query: &mut CypherQuery) {
    use super::executor::is_aggregate_expression;

    let mut i = 0;
    while i + 1 < query.clauses.len() {
        // Only fuse when the MATCH is the first clause — a non-first MATCH
        // depends on the pipeline state from prior clauses, which the fused
        // path would ignore.
        if i > 0 {
            i += 1;
            continue;
        }
        let can_fuse = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::With(_))
        );
        if !can_fuse {
            i += 1;
            continue;
        }

        // Check MATCH: exactly 1 pattern with 3 elements (node-edge-node)
        let (first_var, second_var, edge_has_props, second_has_props) =
            if let Clause::Match(m) = &query.clauses[i] {
                if m.patterns.len() != 1 || m.patterns[0].elements.len() != 3 {
                    i += 1;
                    continue;
                }
                let pat = &m.patterns[0];
                let first_var = match &pat.elements[0] {
                    PatternElement::Node(np) => np.variable.clone(),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let edge_has_props = match &pat.elements[1] {
                    PatternElement::Edge(ep) => ep.properties.is_some() || ep.var_length.is_some(),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let (second_var, second_has_props) = match &pat.elements[2] {
                    PatternElement::Node(np) => (np.variable.clone(), np.properties.is_some()),
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                (first_var, second_var, edge_has_props, second_has_props)
            } else {
                i += 1;
                continue;
            };

        if edge_has_props || second_has_props {
            i += 1;
            continue;
        }
        if first_var.is_none() && second_var.is_none() {
            i += 1;
            continue;
        }

        // Check WITH: must have count() aggregate + group-by on one node variable
        let fusable = if let Clause::With(w) = &query.clauses[i + 1] {
            if w.distinct {
                false
            } else {
                let mut has_count = false;
                let mut all_valid = true;
                let mut group_var: Option<&str> = None;
                let mut count_var_ok = true;

                for item in &w.items {
                    if !is_aggregate_expression(&item.expression) {
                        let refs_var = match &item.expression {
                            Expression::Variable(v) => Some(v.as_str()),
                            _ => None,
                        };
                        match refs_var {
                            Some(v) => {
                                if group_var.is_none() {
                                    group_var = Some(v);
                                } else if group_var != Some(v) {
                                    all_valid = false;
                                    break;
                                }
                            }
                            None => {
                                all_valid = false;
                                break;
                            }
                        }
                    }
                }

                // group_var must be either first_var or second_var
                if all_valid {
                    if let Some(gv) = group_var {
                        let is_first = first_var.as_deref() == Some(gv);
                        let is_second = second_var.as_deref() == Some(gv);
                        if !is_first && !is_second {
                            all_valid = false;
                        }
                    } else {
                        all_valid = false;
                    }
                }

                // Check count() aggregates reference the OTHER node variable
                if all_valid {
                    let other_var = if group_var == first_var.as_deref() {
                        &second_var
                    } else {
                        &first_var
                    };
                    for item in &w.items {
                        if is_aggregate_expression(&item.expression) {
                            match &item.expression {
                                Expression::FunctionCall {
                                    name,
                                    args,
                                    distinct,
                                } if name == "count" => {
                                    if *distinct {
                                        count_var_ok = false;
                                        break;
                                    }
                                    if args.len() == 1 && matches!(args[0], Expression::Star) {
                                        has_count = true;
                                        continue;
                                    }
                                    if let Some(Expression::Variable(var)) = args.first() {
                                        if other_var.as_deref() == Some(var.as_str()) {
                                            has_count = true;
                                            continue;
                                        }
                                    }
                                    count_var_ok = false;
                                    break;
                                }
                                _ => {
                                    count_var_ok = false;
                                    break;
                                }
                            }
                        }
                    }
                }

                has_count && all_valid && count_var_ok
            }
        } else {
            false
        };

        if !fusable {
            i += 1;
            continue;
        }

        // All checks passed — fuse MATCH + WITH
        let with_clause = if let Clause::With(w) = query.clauses.remove(i + 1) {
            w
        } else {
            unreachable!()
        };
        let match_clause = if let Clause::Match(m) = query.clauses.remove(i) {
            m
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedMatchWithAggregate {
                match_clause,
                with_clause,
            },
        );

        i += 1;
    }
}

/// Collect variable names and their node types from patterns
fn collect_pattern_variables(
    patterns: &[crate::graph::core::pattern_matching::Pattern],
) -> Vec<(String, Option<String>)> {
    let mut vars = Vec::new();
    for pattern in patterns {
        for element in &pattern.elements {
            if let PatternElement::Node(np) = element {
                if let Some(ref var) = np.variable {
                    vars.push((var.clone(), np.node_type.clone()));
                }
            }
        }
    }
    vars
}

/// (equality_conditions, in_conditions, comparison_conditions, remaining_predicate)
type PushableResult = (
    Vec<(String, String, Value)>,
    Vec<(String, String, Vec<Value>)>,
    Vec<(String, String, ComparisonOp, Value)>,
    Option<Predicate>,
);

/// Extract pushable predicates from a WHERE clause into MATCH patterns.
/// Returns (equality_conditions, in_conditions, comparison_conditions, remaining_predicate).
///
/// Pushes conditions of the form:
/// - `variable.property = literal_value` (equality)
/// - `variable.property = $param` (equality with param)
/// - `variable.property IN [literal, ...]` (IN list)
/// - `variable.property > literal_value` (and >=, <, <=)
///
/// The variable must be defined in MATCH.
fn extract_pushable_equalities(
    pred: &Predicate,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
) -> PushableResult {
    let mut pushable = Vec::new();
    let mut pushable_in = Vec::new();
    let mut pushable_cmp = Vec::new();
    let remaining = extract_from_predicate(
        pred,
        match_vars,
        params,
        &mut pushable,
        &mut pushable_in,
        &mut pushable_cmp,
    );
    (pushable, pushable_in, pushable_cmp, remaining)
}

/// Recursively extract pushable predicates from a predicate tree.
/// Returns the remaining predicate (None if fully consumed).
fn extract_from_predicate(
    pred: &Predicate,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
    pushable: &mut Vec<(String, String, Value)>,
    pushable_in: &mut Vec<(String, String, Vec<Value>)>,
    pushable_cmp: &mut Vec<(String, String, ComparisonOp, Value)>,
) -> Option<Predicate> {
    match pred {
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals,
            right,
        } => {
            // Check if this is variable.property = literal or variable.property = $param
            if let Some((var, prop, val)) = try_extract_equality(left, right, match_vars, params) {
                pushable.push((var, prop, val));
                None // Fully consumed
            } else {
                Some(pred.clone()) // Keep as-is
            }
        }
        Predicate::Comparison {
            left,
            operator:
                op @ (ComparisonOp::GreaterThan
                | ComparisonOp::GreaterThanEq
                | ComparisonOp::LessThan
                | ComparisonOp::LessThanEq),
            right,
        } => {
            if let Some((var, prop, op, val)) =
                try_extract_comparison(left, right, *op, match_vars, params)
            {
                pushable_cmp.push((var, prop, op, val));
                None
            } else {
                Some(pred.clone())
            }
        }
        Predicate::In { expr, list } => {
            // Push variable.property IN [literal, ...] into MATCH pattern
            if let Expression::PropertyAccess { variable, property } = expr {
                if match_vars.iter().any(|(v, _)| v == variable) {
                    let all_literals: Option<Vec<Value>> = list
                        .iter()
                        .map(|item| {
                            if let Expression::Literal(val) = item {
                                Some(val.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    if let Some(values) = all_literals {
                        pushable_in.push((variable.clone(), property.clone(), values));
                        return None; // Fully consumed
                    }
                }
            }
            Some(pred.clone())
        }
        Predicate::And(left, right) => {
            let left_remaining = extract_from_predicate(
                left,
                match_vars,
                params,
                pushable,
                pushable_in,
                pushable_cmp,
            );
            let right_remaining = extract_from_predicate(
                right,
                match_vars,
                params,
                pushable,
                pushable_in,
                pushable_cmp,
            );

            match (left_remaining, right_remaining) {
                (None, None) => None,
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (Some(l), Some(r)) => Some(Predicate::And(Box::new(l), Box::new(r))),
            }
        }
        // Other predicate types can't be pushed
        _ => Some(pred.clone()),
    }
}

/// Try to extract a simple equality: variable.property = literal_or_param
fn try_extract_equality(
    left: &Expression,
    right: &Expression,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
) -> Option<(String, String, Value)> {
    // Left is property access, right is literal
    if let (Expression::PropertyAccess { variable, property }, Expression::Literal(val)) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), val.clone()));
        }
    }

    // Right is property access, left is literal (commutative)
    if let (Expression::Literal(val), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), val.clone()));
        }
    }

    // Left is property access, right is parameter (resolve from params)
    if let (Expression::PropertyAccess { variable, property }, Expression::Parameter(name)) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), val.clone()));
            }
        }
    }

    // Right is property access, left is parameter (commutative)
    if let (Expression::Parameter(name), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), val.clone()));
            }
        }
    }

    // id(variable) = literal → treat as variable.id = literal
    // This enables O(1) lookup via lookup_by_id instead of full scan.
    if let (Expression::FunctionCall { name, args, .. }, Expression::Literal(val)) = (left, right) {
        if name == "id" {
            if let Some(Expression::Variable(var)) = args.first() {
                if match_vars.iter().any(|(v, _)| v == var) {
                    return Some((var.clone(), "id".to_string(), val.clone()));
                }
            }
        }
    }
    // Commutative: literal = id(variable)
    if let (Expression::Literal(val), Expression::FunctionCall { name, args, .. }) = (left, right) {
        if name == "id" {
            if let Some(Expression::Variable(var)) = args.first() {
                if match_vars.iter().any(|(v, _)| v == var) {
                    return Some((var.clone(), "id".to_string(), val.clone()));
                }
            }
        }
    }

    None
}

/// Try to extract a comparison: variable.property OP literal_or_param
/// When the literal is on the left (e.g. `30 < n.age`), reverse the operator
/// so it becomes `n.age > 30`.
fn try_extract_comparison(
    left: &Expression,
    right: &Expression,
    op: ComparisonOp,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
) -> Option<(String, String, ComparisonOp, Value)> {
    // Left is property access, right is literal: variable.property OP literal
    if let (Expression::PropertyAccess { variable, property }, Expression::Literal(val)) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), op, val.clone()));
        }
    }

    // Right is property access, left is literal: literal OP variable.property → reverse
    if let (Expression::Literal(val), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            let reversed = match op {
                ComparisonOp::GreaterThan => ComparisonOp::LessThan,
                ComparisonOp::GreaterThanEq => ComparisonOp::LessThanEq,
                ComparisonOp::LessThan => ComparisonOp::GreaterThan,
                ComparisonOp::LessThanEq => ComparisonOp::GreaterThanEq,
                other => other,
            };
            return Some((variable.clone(), property.clone(), reversed, val.clone()));
        }
    }

    // Left is property access, right is parameter
    if let (Expression::PropertyAccess { variable, property }, Expression::Parameter(name)) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), op, val.clone()));
            }
        }
    }

    // Right is property access, left is parameter → reverse
    if let (Expression::Parameter(name), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                let reversed = match op {
                    ComparisonOp::GreaterThan => ComparisonOp::LessThan,
                    ComparisonOp::GreaterThanEq => ComparisonOp::LessThanEq,
                    ComparisonOp::LessThan => ComparisonOp::GreaterThan,
                    ComparisonOp::LessThanEq => ComparisonOp::GreaterThanEq,
                    other => other,
                };
                return Some((variable.clone(), property.clone(), reversed, val.clone()));
            }
        }
    }

    None
}

/// Apply a comparison condition to the matching node pattern in MATCH.
/// If the same property already has a comparison matcher (e.g. `year >= 2015`
/// followed by `year <= 2022`), merge them into a `Range` matcher.
fn apply_comparison_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    op: ComparisonOp,
    value: Value,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    // Check if there's already a comparison on this property to merge
                    if let Some(existing) = props.get(property) {
                        if let Some(merged) = merge_comparison(existing, op, &value) {
                            props.insert(property.to_string(), merged);
                            return;
                        }
                    }
                    let matcher = match op {
                        ComparisonOp::GreaterThan => PropertyMatcher::GreaterThan(value),
                        ComparisonOp::GreaterThanEq => PropertyMatcher::GreaterOrEqual(value),
                        ComparisonOp::LessThan => PropertyMatcher::LessThan(value),
                        ComparisonOp::LessThanEq => PropertyMatcher::LessOrEqual(value),
                        _ => return,
                    };
                    props.insert(property.to_string(), matcher);
                    return;
                }
            }
        }
    }
}

/// Merge two comparison matchers on the same property into a Range.
/// E.g. existing `>= 2015` + new `<= 2022` → `Range { 2015..=2022 }`.
fn merge_comparison(
    existing: &PropertyMatcher,
    new_op: ComparisonOp,
    new_val: &Value,
) -> Option<PropertyMatcher> {
    // Extract the existing bound direction
    let (existing_lower, existing_val, existing_inclusive) = match existing {
        PropertyMatcher::GreaterThan(v) => (true, v, false),
        PropertyMatcher::GreaterOrEqual(v) => (true, v, true),
        PropertyMatcher::LessThan(v) => (false, v, false),
        PropertyMatcher::LessOrEqual(v) => (false, v, true),
        _ => return None,
    };

    // Determine the new bound direction
    let (new_lower, new_inclusive) = match new_op {
        ComparisonOp::GreaterThan => (true, false),
        ComparisonOp::GreaterThanEq => (true, true),
        ComparisonOp::LessThan => (false, false),
        ComparisonOp::LessThanEq => (false, true),
        _ => return None,
    };

    // Can only merge opposite directions (lower + upper)
    if existing_lower == new_lower {
        return None; // Both are lower or both are upper — can't merge cleanly
    }

    if existing_lower {
        // existing is lower bound, new is upper bound
        Some(PropertyMatcher::Range {
            lower: existing_val.clone(),
            lower_inclusive: existing_inclusive,
            upper: new_val.clone(),
            upper_inclusive: new_inclusive,
        })
    } else {
        // existing is upper bound, new is lower bound
        Some(PropertyMatcher::Range {
            lower: new_val.clone(),
            lower_inclusive: new_inclusive,
            upper: existing_val.clone(),
            upper_inclusive: existing_inclusive,
        })
    }
}

/// Apply a property equality condition to the matching node pattern in MATCH
fn apply_property_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    value: Value,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    // Don't overwrite an existing matcher (e.g. IN or Range)
                    props
                        .entry(property.to_string())
                        .or_insert(PropertyMatcher::Equals(value));
                    return;
                }
            }
        }
    }
}

/// Apply an IN-list property condition to the matching node pattern in MATCH
fn apply_in_property_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    values: Vec<Value>,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    props.insert(property.to_string(), PropertyMatcher::In(values));
                    return;
                }
            }
        }
    }
}

// ============================================================================
// Fused RETURN + ORDER BY + LIMIT for vector_score
// ============================================================================

/// Fuse MATCH (n:Type) [WHERE ...] RETURN expr ORDER BY expr LIMIT k into a
/// single-pass node scan with inline top-K selection. Avoids materializing all
/// rows — scans nodes directly, evaluates sort key per node, maintains K-element
/// heap. RETURN expressions are only evaluated for the K winners.
///
/// Pattern: MATCH (single node) [WHERE] RETURN (no agg, no distinct) ORDER BY LIMIT
fn fuse_node_scan_top_k(query: &mut CypherQuery) {
    use super::executor::is_aggregate_expression;

    // Need at least MATCH + RETURN + ORDER BY + LIMIT (4 clauses)
    // or MATCH + WHERE + RETURN + ORDER BY + LIMIT (5 clauses)
    if query.clauses.len() < 4 {
        return;
    }

    let mut i = 0;
    while i + 3 < query.clauses.len() {
        // Only fuse first-clause MATCH
        if i > 0 {
            i += 1;
            continue;
        }

        // Detect: MATCH [WHERE] RETURN ORDER_BY LIMIT
        let (match_idx, where_idx, return_idx, orderby_idx, limit_idx) =
            if matches!(&query.clauses[i], Clause::Match(_))
                && matches!(&query.clauses[i + 1], Clause::Where(_))
                && i + 4 < query.clauses.len()
                && matches!(&query.clauses[i + 2], Clause::Return(_))
                && matches!(&query.clauses[i + 3], Clause::OrderBy(_))
                && matches!(&query.clauses[i + 4], Clause::Limit(_))
            {
                (i, Some(i + 1), i + 2, i + 3, i + 4)
            } else if matches!(&query.clauses[i], Clause::Match(_))
                && matches!(&query.clauses[i + 1], Clause::Return(_))
                && matches!(&query.clauses[i + 2], Clause::OrderBy(_))
                && matches!(&query.clauses[i + 3], Clause::Limit(_))
            {
                (i, None, i + 1, i + 2, i + 3)
            } else {
                i += 1;
                continue;
            };

        // MATCH must be single pattern, single node, no edges
        let is_single_node = if let Clause::Match(mc) = &query.clauses[match_idx] {
            mc.patterns.len() == 1
                && mc.patterns[0].elements.len() == 1
                && matches!(
                    mc.patterns[0].elements[0],
                    crate::graph::core::pattern_matching::PatternElement::Node(_)
                )
                && mc.path_assignments.is_empty()
        } else {
            false
        };
        if !is_single_node {
            i += 1;
            continue;
        }

        // RETURN must have no aggregation, no DISTINCT, and no function calls
        // (function calls like ts_sum need special evaluation context)
        let return_ok = if let Clause::Return(r) = &query.clauses[return_idx] {
            !r.distinct
                && !r
                    .items
                    .iter()
                    .any(|item| is_aggregate_expression(&item.expression))
                && !r
                    .items
                    .iter()
                    .any(|item| matches!(item.expression, Expression::FunctionCall { .. }))
        } else {
            false
        };
        if !return_ok {
            i += 1;
            continue;
        }

        // ORDER BY must have exactly 1 sort item
        let sort_info = if let Clause::OrderBy(o) = &query.clauses[orderby_idx] {
            if o.items.len() == 1 {
                Some((o.items[0].expression.clone(), !o.items[0].ascending))
            } else {
                None
            }
        } else {
            None
        };
        let Some((sort_expr, descending)) = sort_info else {
            i += 1;
            continue;
        };

        // LIMIT must be positive literal integer
        let limit_val = if let Clause::Limit(l) = &query.clauses[limit_idx] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => Some(*n as usize),
                _ => None,
            }
        } else {
            None
        };
        let Some(limit) = limit_val else {
            i += 1;
            continue;
        };

        // All checks passed — fuse
        // Remove clauses from back to front to preserve indices
        query.clauses.remove(limit_idx);
        query.clauses.remove(orderby_idx);
        let return_clause = if let Clause::Return(r) = query.clauses.remove(return_idx) {
            r
        } else {
            unreachable!()
        };
        let where_predicate = if let Some(wi) = where_idx {
            if let Clause::Where(w) = query.clauses.remove(wi) {
                Some(w.predicate)
            } else {
                None
            }
        } else {
            None
        };
        let match_clause = if let Clause::Match(mc) = query.clauses.remove(match_idx) {
            mc
        } else {
            unreachable!()
        };

        query.clauses.insert(
            match_idx,
            Clause::FusedNodeScanTopK {
                match_clause,
                where_predicate,
                return_clause,
                sort_expression: sort_expr,
                descending,
                limit,
            },
        );

        i += 1;
    }
}

/// Detect `RETURN ... vector_score(...) AS s ... ORDER BY s DESC LIMIT k`
/// and replace with a fused clause that uses a min-heap (O(n log k) vs O(n log n))
/// and projects RETURN expressions only for the k surviving rows.
fn fuse_vector_score_order_limit(query: &mut CypherQuery) {
    use super::executor::is_aggregate_expression;

    if query.clauses.len() < 3 {
        return;
    }

    let mut i = 0;
    while i + 2 < query.clauses.len() {
        // Check for RETURN + ORDER BY + LIMIT pattern
        let is_pattern = matches!(
            (
                &query.clauses[i],
                &query.clauses[i + 1],
                &query.clauses[i + 2]
            ),
            (Clause::Return(_), Clause::OrderBy(_), Clause::Limit(_))
        );
        if !is_pattern {
            i += 1;
            continue;
        }

        // Extract references for analysis (before removing)
        let (score_idx, alias) = if let Clause::Return(r) = &query.clauses[i] {
            // Don't fuse if RETURN has aggregation or DISTINCT
            if r.distinct
                || r.items
                    .iter()
                    .any(|item| is_aggregate_expression(&item.expression))
            {
                i += 1;
                continue;
            }
            // Find the vector_score item
            let found = r.items.iter().enumerate().find(|(_, item)| {
                matches!(
                    &item.expression,
                    Expression::FunctionCall { name, .. }
                        if name == "vector_score"
                )
            });
            match found {
                Some((idx, item)) => {
                    let col = return_item_column_name(item);
                    (idx, col)
                }
                None => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        // Check ORDER BY references the score alias and has exactly one item
        let descending = if let Clause::OrderBy(o) = &query.clauses[i + 1] {
            if o.items.len() != 1 {
                i += 1;
                continue;
            }
            let sort_name = match &o.items[0].expression {
                Expression::Variable(v) => v.clone(),
                other => expression_to_column_name(other),
            };
            if sort_name != alias {
                i += 1;
                continue;
            }
            !o.items[0].ascending
        } else {
            i += 1;
            continue;
        };

        // Extract LIMIT value (must be a literal non-negative integer)
        let limit = if let Clause::Limit(l) = &query.clauses[i + 2] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => *n as usize,
                _ => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        // All checks passed — fuse the three clauses
        query.clauses.remove(i + 2); // LIMIT
        query.clauses.remove(i + 1); // ORDER BY
        let return_clause = if let Clause::Return(r) = query.clauses.remove(i) {
            r
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedVectorScoreTopK {
                return_clause,
                score_item_index: score_idx,
                descending,
                limit,
            },
        );

        i += 1;
    }
}

/// Column name for a return item (mirrors executor's return_item_column_name).
fn return_item_column_name(item: &ReturnItem) -> String {
    if let Some(ref alias) = item.alias {
        alias.clone()
    } else {
        expression_to_column_name(&item.expression)
    }
}

/// Simple expression-to-string for column name matching in the planner.
fn expression_to_column_name(expr: &Expression) -> String {
    match expr {
        Expression::Variable(name) => name.clone(),
        Expression::PropertyAccess { variable, property } => format!("{}.{}", variable, property),
        Expression::FunctionCall { name, args, .. } => {
            let args_str: Vec<String> = args.iter().map(expression_to_column_name).collect();
            format!("{}({})", name, args_str.join(", "))
        }
        _ => format!("{:?}", expr),
    }
}

// ============================================================================
// General Top-K ORDER BY LIMIT Fusion
// ============================================================================

/// Fuse RETURN + ORDER BY + LIMIT into a single top-k heap pass.
/// Generalizes `fuse_vector_score_order_limit` to any numeric sort expression.
/// Runs after the vector_score-specific pass so it only handles non-vector_score cases.
fn fuse_order_by_top_k(query: &mut CypherQuery) {
    if query.clauses.len() < 3 {
        return;
    }

    let mut i = 0;
    while i + 2 < query.clauses.len() {
        // Check for RETURN + ORDER BY + LIMIT pattern
        let is_pattern = matches!(
            (
                &query.clauses[i],
                &query.clauses[i + 1],
                &query.clauses[i + 2]
            ),
            (Clause::Return(_), Clause::OrderBy(_), Clause::Limit(_))
        );
        if !is_pattern {
            i += 1;
            continue;
        }

        // Note: SKIP before LIMIT (RETURN, ORDER BY, SKIP, LIMIT) is already handled:
        // the pattern match above requires clauses[i+2] to be Limit, so SKIP at i+2 won't match.

        let (score_idx, sort_expression) = if let Clause::Return(r) = &query.clauses[i] {
            // Don't fuse if RETURN has DISTINCT
            if r.distinct {
                i += 1;
                continue;
            }
            // Don't fuse if any RETURN item has aggregation
            if r.items
                .iter()
                .any(|item| super::executor::is_aggregate_expression(&item.expression))
            {
                i += 1;
                continue;
            }
            // Don't fuse if any RETURN item has window functions —
            // window functions need the full result set to compute
            // partitions/ranks, which is incompatible with the per-row
            // scoring in FusedOrderByTopK.
            if r.items
                .iter()
                .any(|item| matches!(item.expression, Expression::WindowFunction { .. }))
            {
                i += 1;
                continue;
            }
            // Find which RETURN item the ORDER BY references
            let order_info = if let Clause::OrderBy(o) = &query.clauses[i + 1] {
                if o.items.len() != 1 {
                    i += 1;
                    continue;
                }
                let order_alias = match &o.items[0].expression {
                    Expression::Variable(v) => v.clone(),
                    other => expression_to_column_name(other),
                };
                // Try matching a RETURN item
                let found = r
                    .items
                    .iter()
                    .enumerate()
                    .find(|(_, item)| return_item_column_name(item) == order_alias);
                match found {
                    Some((idx, _)) => (idx, None), // sort key is RETURN item
                    None => {
                        // Sort key not in RETURN — store expression directly
                        (0, Some(o.items[0].expression.clone()))
                    }
                }
            } else {
                i += 1;
                continue;
            };
            order_info
        } else {
            i += 1;
            continue;
        };
        // Extract ORDER BY direction
        let descending = if let Clause::OrderBy(o) = &query.clauses[i + 1] {
            !o.items[0].ascending
        } else {
            i += 1;
            continue;
        };

        // Extract LIMIT (must be positive integer literal)
        let limit = if let Clause::Limit(l) = &query.clauses[i + 2] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => *n as usize,
                _ => {
                    i += 1;
                    continue;
                }
            }
        } else {
            i += 1;
            continue;
        };

        // All checks passed — fuse the three clauses
        query.clauses.remove(i + 2); // LIMIT
        query.clauses.remove(i + 1); // ORDER BY
        let return_clause = if let Clause::Return(r) = query.clauses.remove(i) {
            r
        } else {
            unreachable!()
        };

        query.clauses.insert(
            i,
            Clause::FusedOrderByTopK {
                return_clause,
                score_item_index: score_idx,
                descending,
                limit,
                sort_expression,
            },
        );

        i += 1;
    }
}

// ============================================================================
// Predicate Cost-Based Reordering
// ============================================================================

/// Reorder AND/OR children in WHERE predicates so cheaper predicates
/// are evaluated first (enabling short-circuit evaluation).
fn reorder_predicates_by_cost(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        if let Clause::Where(ref mut w) = clause {
            w.predicate = reorder_predicate(std::mem::replace(
                &mut w.predicate,
                Predicate::IsNull(Expression::Literal(Value::Null)),
            ));
        }
    }
}

/// Recursively reorder a predicate tree by estimated cost.
fn reorder_predicate(pred: Predicate) -> Predicate {
    match pred {
        Predicate::And(left, right) => {
            let left = reorder_predicate(*left);
            let right = reorder_predicate(*right);
            // Put cheaper predicate first for short-circuit
            if estimate_predicate_cost(&right) < estimate_predicate_cost(&left) {
                Predicate::And(Box::new(right), Box::new(left))
            } else {
                Predicate::And(Box::new(left), Box::new(right))
            }
        }
        Predicate::Or(left, right) => {
            let left = reorder_predicate(*left);
            let right = reorder_predicate(*right);
            // Put cheaper predicate first for short-circuit
            if estimate_predicate_cost(&right) < estimate_predicate_cost(&left) {
                Predicate::Or(Box::new(right), Box::new(left))
            } else {
                Predicate::Or(Box::new(left), Box::new(right))
            }
        }
        Predicate::Xor(left, right) => {
            let left = reorder_predicate(*left);
            let right = reorder_predicate(*right);
            Predicate::Xor(Box::new(left), Box::new(right))
        }
        Predicate::Not(inner) => Predicate::Not(Box::new(reorder_predicate(*inner))),
        other => other,
    }
}

/// Estimate the relative cost of evaluating a predicate.
fn estimate_predicate_cost(pred: &Predicate) -> u32 {
    match pred {
        Predicate::Comparison { left, right, .. } => {
            estimate_expression_cost(left) + estimate_expression_cost(right) + 1
        }
        Predicate::And(l, r) | Predicate::Or(l, r) | Predicate::Xor(l, r) => {
            estimate_predicate_cost(l) + estimate_predicate_cost(r)
        }
        Predicate::Not(inner) => estimate_predicate_cost(inner) + 1,
        Predicate::IsNull(_) | Predicate::IsNotNull(_) => 2,
        Predicate::In { list, .. } => 3 + list.len() as u32,
        Predicate::InLiteralSet { values, .. } => 2 + (values.len() > 16) as u32, // HashSet is O(1)
        Predicate::StartsWith { .. } | Predicate::EndsWith { .. } | Predicate::Contains { .. } => 5,
        Predicate::Exists { .. } => 100, // Pattern existence checks are expensive
        Predicate::InExpression { .. } => 10,
        Predicate::LabelCheck { .. } => 1, // Single node-type string compare
    }
}

/// Estimate the relative cost of evaluating an expression.
fn estimate_expression_cost(expr: &Expression) -> u32 {
    match expr {
        Expression::Literal(_) => 1,
        Expression::Parameter(_) => 1,
        Expression::PropertyAccess { .. } => 2,
        Expression::Variable(_) => 1,
        Expression::Star => 1,
        Expression::FunctionCall { name, args, .. } => {
            let base = match name.as_str() {
                "point" => 3,
                "distance" => 10,
                "contains" => 50,
                "intersects" => 60,
                "centroid" => 30,
                "area" => 40,
                "perimeter" => 40,
                "latitude" | "longitude" => 2,
                "tostring" | "tointeger" | "tofloat" | "toboolean" => 2,
                "size" | "length" | "type" | "id" => 2,
                "tolower" | "toupper" | "trim" | "ltrim" | "rtrim" => 3,
                "substring" | "replace" | "split" => 5,
                "abs" | "ceil" | "ceiling" | "floor" | "round" | "sqrt" | "sign" => 2,
                "vector_score" => 200, // Embedding lookup + similarity computation
                "valid_at" | "valid_during" => 5, // 2 property lookups + 2 comparisons
                _ => 5,                // Unknown functions get moderate cost
            };
            let arg_cost: u32 = args.iter().map(estimate_expression_cost).sum();
            base + arg_cost
        }
        Expression::Add(l, r)
        | Expression::Subtract(l, r)
        | Expression::Multiply(l, r)
        | Expression::Divide(l, r)
        | Expression::Modulo(l, r) => estimate_expression_cost(l) + estimate_expression_cost(r) + 1,
        Expression::Negate(inner) => estimate_expression_cost(inner) + 1,
        Expression::ListLiteral(items) => {
            items.iter().map(estimate_expression_cost).sum::<u32>() + 1
        }
        Expression::Case {
            when_clauses,
            else_expr,
            ..
        } => {
            let clause_cost: u32 = when_clauses
                .iter()
                .map(|(_, e)| estimate_expression_cost(e) + 2)
                .sum();
            clause_cost
                + else_expr
                    .as_ref()
                    .map_or(0, |e| estimate_expression_cost(e))
        }
        Expression::IndexAccess { expr, index } => {
            estimate_expression_cost(expr) + estimate_expression_cost(index) + 1
        }
        Expression::ListSlice { expr, start, end } => {
            estimate_expression_cost(expr)
                + start.as_ref().map_or(0, |s| estimate_expression_cost(s))
                + end.as_ref().map_or(0, |e| estimate_expression_cost(e))
                + 1
        }
        Expression::PredicateExpr(_) => 3,
        Expression::ExprPropertyAccess { expr, .. } => estimate_expression_cost(expr) + 1,
        _ => 5, // ListComprehension, MapProjection
    }
}

// ============================================================================
// text_score → vector_score AST Rewrite
// ============================================================================

/// Collected texts that the caller must embed before execution.
/// Each entry is `(param_name, query_text)` — the caller embeds the text and
/// inserts the resulting vector into the params map under `param_name`.
pub struct TextScoreRewrite {
    pub texts_to_embed: Vec<(String, String)>,
}

/// Walk the AST and rewrite all `text_score(node, col, query_text)` calls
/// to `vector_score(node, col_emb, $__ts_N)`.
///
/// The text argument can be a string literal or a `$parameter` (resolved from
/// `params`).  Returns the collected texts so the caller can embed them and
/// inject the resulting vectors into the params map before optimization.
pub fn rewrite_text_score(
    query: &mut CypherQuery,
    params: &HashMap<String, Value>,
) -> Result<TextScoreRewrite, String> {
    let mut collector = TextScoreCollector {
        counter: 0,
        texts_to_embed: Vec::new(),
    };

    for clause in &mut query.clauses {
        match clause {
            Clause::Return(r) => {
                for item in &mut r.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
            }
            Clause::Where(w) => {
                collector.rewrite_pred(&mut w.predicate, params)?;
            }
            Clause::With(w) => {
                for item in &mut w.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
                if let Some(ref mut wh) = w.where_clause {
                    collector.rewrite_pred(&mut wh.predicate, params)?;
                }
            }
            Clause::OrderBy(o) => {
                for item in &mut o.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
            }
            Clause::Unwind(u) => {
                collector.rewrite_expr(&mut u.expression, params)?;
            }
            Clause::Delete(d) => {
                for expr in &mut d.expressions {
                    collector.rewrite_expr(expr, params)?;
                }
            }
            Clause::Set(s) => {
                for item in &mut s.items {
                    if let SetItem::Property {
                        ref mut expression, ..
                    } = item
                    {
                        collector.rewrite_expr(expression, params)?;
                    }
                }
            }
            Clause::Create(c) => {
                for pattern in &mut c.patterns {
                    for element in &mut pattern.elements {
                        match element {
                            CreateElement::Node(n) => {
                                for (_, expr) in &mut n.properties {
                                    collector.rewrite_expr(expr, params)?;
                                }
                            }
                            CreateElement::Edge(e) => {
                                for (_, expr) in &mut e.properties {
                                    collector.rewrite_expr(expr, params)?;
                                }
                            }
                        }
                    }
                }
            }
            Clause::Merge(m) => {
                for element in &mut m.pattern.elements {
                    match element {
                        CreateElement::Node(n) => {
                            for (_, expr) in &mut n.properties {
                                collector.rewrite_expr(expr, params)?;
                            }
                        }
                        CreateElement::Edge(e) => {
                            for (_, expr) in &mut e.properties {
                                collector.rewrite_expr(expr, params)?;
                            }
                        }
                    }
                }
                if let Some(ref mut items) = m.on_create {
                    for item in items {
                        if let SetItem::Property {
                            ref mut expression, ..
                        } = item
                        {
                            collector.rewrite_expr(expression, params)?;
                        }
                    }
                }
                if let Some(ref mut items) = m.on_match {
                    for item in items {
                        if let SetItem::Property {
                            ref mut expression, ..
                        } = item
                        {
                            collector.rewrite_expr(expression, params)?;
                        }
                    }
                }
            }
            Clause::Skip(s) => {
                collector.rewrite_expr(&mut s.count, params)?;
            }
            Clause::Limit(l) => {
                collector.rewrite_expr(&mut l.count, params)?;
            }
            // Match/OptionalMatch: patterns only, no function call expressions
            // Remove: no expressions
            // Fused clauses: don't exist yet (created by optimize, which runs after rewrite)
            _ => {}
        }
    }

    Ok(TextScoreRewrite {
        texts_to_embed: collector.texts_to_embed,
    })
}

struct TextScoreCollector {
    counter: usize,
    texts_to_embed: Vec<(String, String)>,
}

impl TextScoreCollector {
    /// Rewrite an expression in-place.  Turns `text_score(...)` into `vector_score(...)`.
    fn rewrite_expr(
        &mut self,
        expr: &mut Expression,
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        match expr {
            Expression::FunctionCall { name, args, .. } if name == "text_score" => {
                if args.len() != 3 && args.len() != 4 {
                    return Err(
                        "text_score() requires 3 arguments: (node, text_column, query_text) \
                         with optional 4th metric argument"
                            .into(),
                    );
                }

                // arg[1]: text column — must be a string literal
                let col_name =
                    match &args[1] {
                        Expression::Literal(Value::String(s)) => s.clone(),
                        _ => return Err(
                            "text_score(): second argument must be a string literal column name"
                                .into(),
                        ),
                    };

                // arg[2]: query text — string literal or $param
                let query_text = match &args[2] {
                    Expression::Literal(Value::String(s)) => s.clone(),
                    Expression::Parameter(param_name) => match params.get(param_name.as_str()) {
                        Some(Value::String(s)) => s.clone(),
                        Some(_) => {
                            return Err(format!(
                                "text_score(): parameter ${} must be a string",
                                param_name
                            ))
                        }
                        None => {
                            return Err(format!(
                                "text_score(): parameter ${} not found",
                                param_name
                            ))
                        }
                    },
                    _ => {
                        return Err(
                            "text_score(): third argument must be a string literal or $parameter"
                                .into(),
                        )
                    }
                };

                // Deduplicate: reuse param if same query text already collected
                let param_name = if let Some((existing, _)) =
                    self.texts_to_embed.iter().find(|(_, t)| t == &query_text)
                {
                    existing.clone()
                } else {
                    let pname = format!("__ts_{}", self.counter);
                    self.counter += 1;
                    self.texts_to_embed.push((pname.clone(), query_text));
                    pname
                };

                // Rewrite: text_score(n, 'summary', ...) → vector_score(n, 'summary_emb', $__ts_N)
                *name = "vector_score".to_string();
                args[1] = Expression::Literal(Value::String(format!("{}_emb", col_name)));
                args[2] = Expression::Parameter(param_name);

                Ok(())
            }
            Expression::FunctionCall { args, .. } => {
                for arg in args.iter_mut() {
                    self.rewrite_expr(arg, params)?;
                }
                Ok(())
            }
            Expression::Add(l, r)
            | Expression::Subtract(l, r)
            | Expression::Multiply(l, r)
            | Expression::Divide(l, r)
            | Expression::Modulo(l, r)
            | Expression::Concat(l, r) => {
                self.rewrite_expr(l, params)?;
                self.rewrite_expr(r, params)?;
                Ok(())
            }
            Expression::Negate(inner) => self.rewrite_expr(inner, params),
            Expression::ListLiteral(items) => {
                for item in items.iter_mut() {
                    self.rewrite_expr(item, params)?;
                }
                Ok(())
            }
            Expression::Case {
                operand,
                when_clauses,
                else_expr,
            } => {
                if let Some(op) = operand {
                    self.rewrite_expr(op, params)?;
                }
                for (cond, result) in when_clauses.iter_mut() {
                    match cond {
                        CaseCondition::Expression(e) => self.rewrite_expr(e, params)?,
                        CaseCondition::Predicate(p) => self.rewrite_pred(p, params)?,
                    }
                    self.rewrite_expr(result, params)?;
                }
                if let Some(el) = else_expr {
                    self.rewrite_expr(el, params)?;
                }
                Ok(())
            }
            Expression::IndexAccess { expr, index } => {
                self.rewrite_expr(expr, params)?;
                self.rewrite_expr(index, params)?;
                Ok(())
            }
            Expression::ListSlice { expr, start, end } => {
                self.rewrite_expr(expr, params)?;
                if let Some(s) = start {
                    self.rewrite_expr(s, params)?;
                }
                if let Some(e) = end {
                    self.rewrite_expr(e, params)?;
                }
                Ok(())
            }
            Expression::ListComprehension {
                list_expr,
                filter,
                map_expr,
                ..
            } => {
                self.rewrite_expr(list_expr, params)?;
                if let Some(f) = filter {
                    self.rewrite_pred(f, params)?;
                }
                if let Some(m) = map_expr {
                    self.rewrite_expr(m, params)?;
                }
                Ok(())
            }
            Expression::MapProjection { items, .. } => {
                for item in items.iter_mut() {
                    if let MapProjectionItem::Alias { expr, .. } = item {
                        self.rewrite_expr(expr, params)?;
                    }
                }
                Ok(())
            }
            Expression::MapLiteral(entries) => {
                for (_, expr) in entries.iter_mut() {
                    self.rewrite_expr(expr, params)?;
                }
                Ok(())
            }
            // Leaf nodes
            Expression::PropertyAccess { .. }
            | Expression::Variable(_)
            | Expression::Literal(_)
            | Expression::Parameter(_)
            | Expression::Star => Ok(()),
            Expression::IsNull(inner) | Expression::IsNotNull(inner) => {
                self.rewrite_expr(inner, params)
            }
            Expression::QuantifiedList {
                list_expr, filter, ..
            } => {
                self.rewrite_expr(list_expr, params)?;
                self.rewrite_pred(filter, params)?;
                Ok(())
            }
            Expression::WindowFunction {
                partition_by,
                order_by,
                ..
            } => {
                for expr in partition_by.iter_mut() {
                    self.rewrite_expr(expr, params)?;
                }
                for item in order_by.iter_mut() {
                    self.rewrite_expr(&mut item.expression, params)?;
                }
                Ok(())
            }
            Expression::PredicateExpr(pred) => self.rewrite_pred(pred, params),
            Expression::ExprPropertyAccess { expr, .. } => self.rewrite_expr(expr, params),
        }
    }

    /// Rewrite predicates in-place (for WHERE clauses).
    fn rewrite_pred(
        &mut self,
        pred: &mut Predicate,
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        match pred {
            Predicate::Comparison { left, right, .. } => {
                self.rewrite_expr(left, params)?;
                self.rewrite_expr(right, params)?;
                Ok(())
            }
            Predicate::And(l, r) | Predicate::Or(l, r) | Predicate::Xor(l, r) => {
                self.rewrite_pred(l, params)?;
                self.rewrite_pred(r, params)?;
                Ok(())
            }
            Predicate::Not(inner) => self.rewrite_pred(inner, params),
            Predicate::IsNull(e) | Predicate::IsNotNull(e) => self.rewrite_expr(e, params),
            Predicate::In { expr, list } => {
                self.rewrite_expr(expr, params)?;
                for item in list.iter_mut() {
                    self.rewrite_expr(item, params)?;
                }
                Ok(())
            }
            Predicate::InLiteralSet { expr, .. } => self.rewrite_expr(expr, params),
            Predicate::StartsWith { expr, pattern }
            | Predicate::EndsWith { expr, pattern }
            | Predicate::Contains { expr, pattern } => {
                self.rewrite_expr(expr, params)?;
                self.rewrite_expr(pattern, params)?;
                Ok(())
            }
            Predicate::Exists { .. } => Ok(()),
            Predicate::InExpression { expr, list_expr } => {
                self.rewrite_expr(expr, params)?;
                self.rewrite_expr(list_expr, params)?;
                Ok(())
            }
            Predicate::LabelCheck { .. } => Ok(()),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
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
}
