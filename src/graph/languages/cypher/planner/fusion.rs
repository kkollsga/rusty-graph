//! Multi-clause fusion passes — rewrite MATCH+RETURN+AGG, top-K, ORDER BY+LIMIT
//! into specialised physical plans.

use super::super::ast::*;
use super::index_selection::collect_pattern_variables;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;
use crate::graph::schema::DirGraph;

pub(super) fn fuse_anchored_edge_count(query: &mut CypherQuery, graph: &DirGraph) {
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

pub(super) fn fuse_count_short_circuits(query: &mut CypherQuery) {
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
pub(super) fn is_count_of_var_or_star(expr: &Expression, node_var: Option<&str>) -> bool {
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
pub(super) fn identify_type_count_pair(
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
pub(super) fn is_node_type_accessor(expr: &Expression, node_var: Option<&str>) -> bool {
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
pub(super) fn identify_edge_type_count_pair(
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
pub(super) fn is_edge_type_function(expr: &Expression, edge_var: Option<&str>) -> bool {
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
pub(super) fn fuse_optional_match_aggregate(query: &mut CypherQuery) {
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

        // Collect variables defined in the OPTIONAL MATCH pattern *first* —
        // we need this both to validate the count() args and to reject
        // group-key PropertyAccess on OPTIONAL-bound variables. The fused
        // executor evaluates group keys against the *source* row (before
        // OPTIONAL MATCH expansion), so `pet.name` where `pet` only exists
        // post-OPTIONAL would always be NULL — silently wrong.
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

        // Check that the WITH/RETURN contains count() aggregation and simple pass-through group keys
        let fusable = match &query.clauses[i + 1] {
            Clause::With(w) => is_fusable_with_clause(w),
            Clause::Return(r) => is_fusable_return_clause(r, &opt_match_vars),
            _ => false,
        };

        if !fusable {
            i += 1;
            continue;
        }

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
pub(super) fn is_fusable_with_clause(with: &WithClause) -> bool {
    use super::super::ast::is_aggregate_expression;

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
/// (RETURN items can be `l.korttittel`, not just bare `l`) — *except* when
/// the PropertyAccess targets a variable that's only bound by the OPTIONAL
/// MATCH itself. The fused executor evaluates group keys against the source
/// row (pre-OPTIONAL-MATCH), so `pet.name` where `pet` only exists post-
/// OPTIONAL would always resolve to NULL — silently merging all rows into
/// one wrong group.
pub(super) fn is_fusable_return_clause(
    ret: &ReturnClause,
    opt_match_vars: &std::collections::HashSet<String>,
) -> bool {
    use super::super::ast::is_aggregate_expression;

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
            // Group key must be a simple variable or PropertyAccess on a
            // variable bound *before* the OPTIONAL MATCH.
            match &item.expression {
                Expression::Variable(_) => {}
                Expression::PropertyAccess { variable, .. } => {
                    if opt_match_vars.contains(variable) {
                        return false;
                    }
                }
                _ => return false,
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
pub(super) fn fuse_match_return_aggregate(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

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
pub(super) fn fuse_aggregate_order_limit(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

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
pub(super) fn fuse_node_scan_aggregate(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

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
pub(super) fn fuse_match_with_aggregate(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

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

// ============================================================================
// Fused RETURN + ORDER BY + LIMIT for vector_score
// ============================================================================

/// Fuse MATCH (n:Type) [WHERE ...] RETURN expr ORDER BY expr LIMIT k into a
/// single-pass node scan with inline top-K selection. Avoids materializing all
/// rows — scans nodes directly, evaluates sort key per node, maintains K-element
/// heap. RETURN expressions are only evaluated for the K winners.
///
/// Pattern: MATCH (single node) [WHERE] RETURN (no agg, no distinct) ORDER BY LIMIT
pub(super) fn fuse_node_scan_top_k(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

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
pub(super) fn fuse_vector_score_order_limit(query: &mut CypherQuery) {
    use super::super::ast::is_aggregate_expression;

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
pub(super) fn return_item_column_name(item: &ReturnItem) -> String {
    if let Some(ref alias) = item.alias {
        alias.clone()
    } else {
        expression_to_column_name(&item.expression)
    }
}

/// Simple expression-to-string for column name matching in the planner.
pub(super) fn expression_to_column_name(expr: &Expression) -> String {
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
pub(super) fn fuse_order_by_top_k(query: &mut CypherQuery) {
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
                .any(|item| super::super::ast::is_aggregate_expression(&item.expression))
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
// Spatial-join fusion: MATCH (s:A), (w:B) WHERE contains(s, w) [AND rest]
// ============================================================================

/// Try to strip `contains(var, var)` from a predicate, returning
/// (container_var, probe_var, remainder_predicate).
/// Returns `None` if the predicate doesn't match the required shape.
///
/// Matches these AST shapes (see where_clause.rs `try_extract_contains_filter`):
///   - `contains(a, b) <> false` (parser truthy wrapper) → primary case
///   - `contains(a, b) <> false AND rest` or `rest AND contains(...)` → with remainder
///
/// Does NOT match: `NOT contains(...)`, `contains(a, point(…))` (constant point),
/// disjunctions, or any non-variable first/second arg.
fn extract_spatial_join_contains(pred: &Predicate) -> Option<(String, String, Option<Predicate>)> {
    match pred {
        Predicate::Comparison {
            left,
            operator: ComparisonOp::NotEquals,
            right: Expression::Literal(Value::Boolean(false)),
        } => {
            let (c, p) = extract_contains_call_vars(left)?;
            Some((c, p, None))
        }
        Predicate::And(l, r) => {
            if let Some((c, p, None)) = extract_spatial_join_contains(l) {
                return Some((c, p, Some((**r).clone())));
            }
            if let Some((c, p, None)) = extract_spatial_join_contains(r) {
                return Some((c, p, Some((**l).clone())));
            }
            None
        }
        _ => None,
    }
}

/// Match a `contains(Variable, Variable)` function call expression.
fn extract_contains_call_vars(expr: &Expression) -> Option<(String, String)> {
    if let Expression::FunctionCall { name, args, .. } = expr {
        if name != "contains" || args.len() != 2 {
            return None;
        }
        let c = match &args[0] {
            Expression::Variable(n) => n.clone(),
            _ => return None,
        };
        let p = match &args[1] {
            Expression::Variable(n) => n.clone(),
            _ => return None,
        };
        if c == p {
            return None;
        }
        Some((c, p))
    } else {
        None
    }
}

/// Rewrite `MATCH (s:A), (w:B) WHERE contains(s, w) [AND rest]` into
/// `Clause::SpatialJoin { ..., remainder: rest }`.
///
/// Preconditions (all must hold or the rewrite is skipped):
/// - Adjacent `Match` + `Where` clauses with no intervening skipped fusion.
/// - Two disjoint single-node patterns, each with `variable` and `node_type`,
///   no edges, no path assignments, no limit/distinct hints.
/// - WHERE predicate matches `contains(var, var) <> false` (parser's truthy
///   wrapper), possibly ANDed with a remainder. No NOT, no OR, no constant point.
/// - The two contains() variables bind to the two MATCH patterns (in either order).
/// - Container type has `SpatialConfig::geometry`; probe type has
///   `SpatialConfig::location`.
pub(super) fn fuse_spatial_join(query: &mut CypherQuery, graph: &DirGraph) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        let eligible = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Where(_))
        );
        if !eligible {
            i += 1;
            continue;
        }

        let (p0_var, p0_type, p1_var, p1_type) = {
            let mc = match &query.clauses[i] {
                Clause::Match(m) => m,
                _ => unreachable!(),
            };
            if mc.patterns.len() != 2
                || !mc.path_assignments.is_empty()
                || mc.limit_hint.is_some()
                || mc.distinct_node_hint.is_some()
            {
                i += 1;
                continue;
            }
            let extract = |pat: &crate::graph::core::pattern_matching::Pattern| {
                if pat.elements.len() != 1 {
                    return None;
                }
                match &pat.elements[0] {
                    PatternElement::Node(np) => {
                        // Require a variable and a typed label. Properties are
                        // allowed (planner still lets us prune per-type); other
                        // fields must be defaults.
                        let v = np.variable.as_ref()?.clone();
                        let t = np.node_type.as_ref()?.clone();
                        Some((v, t))
                    }
                    _ => None,
                }
            };
            let (v0, t0) = match extract(&mc.patterns[0]) {
                Some(x) => x,
                None => {
                    i += 1;
                    continue;
                }
            };
            let (v1, t1) = match extract(&mc.patterns[1]) {
                Some(x) => x,
                None => {
                    i += 1;
                    continue;
                }
            };
            (v0, t0, v1, t1)
        };

        let (container_var, probe_var, remainder) = {
            let w = match &query.clauses[i + 1] {
                Clause::Where(w) => w,
                _ => unreachable!(),
            };
            match extract_spatial_join_contains(&w.predicate) {
                Some(x) => x,
                None => {
                    i += 1;
                    continue;
                }
            }
        };

        // Resolve which MATCH pattern each variable came from, and the types.
        let (container_type, probe_type) = if container_var == p0_var && probe_var == p1_var {
            (p0_type.clone(), p1_type.clone())
        } else if container_var == p1_var && probe_var == p0_var {
            (p1_type.clone(), p0_type.clone())
        } else {
            i += 1;
            continue;
        };

        // Schema gate: container needs geometry, probe needs location.
        let container_ok = graph
            .get_spatial_config(&container_type)
            .is_some_and(|c| c.geometry.is_some());
        let probe_ok = graph
            .get_spatial_config(&probe_type)
            .is_some_and(|c| c.location.is_some());
        if !(container_ok && probe_ok) {
            i += 1;
            continue;
        }

        // Commit rewrite: replace Match+Where with a single SpatialJoin clause.
        query.clauses.remove(i + 1);
        query.clauses[i] = Clause::SpatialJoin {
            container_var,
            probe_var,
            container_type,
            probe_type,
            remainder,
        };

        i += 1;
    }
}

#[cfg(test)]
mod spatial_join_tests {
    use super::*;
    use crate::graph::languages::cypher::parser::parse_cypher;
    use crate::graph::schema::{DirGraph, SpatialConfig};

    fn graph_with_spatial() -> DirGraph {
        let mut g = DirGraph::new();
        g.spatial_configs.insert(
            "Area".into(),
            SpatialConfig {
                geometry: Some("geom".into()),
                location: None,
                points: Default::default(),
                shapes: Default::default(),
            },
        );
        g.spatial_configs.insert(
            "City".into(),
            SpatialConfig {
                geometry: None,
                location: Some(("lat".into(), "lon".into())),
                points: Default::default(),
                shapes: Default::default(),
            },
        );
        g
    }

    #[test]
    fn rewrites_canonical_two_pattern_contains() {
        let mut q =
            parse_cypher("MATCH (a:Area), (c:City) WHERE contains(a, c) RETURN a.name, c.name")
                .unwrap();
        fuse_spatial_join(&mut q, &graph_with_spatial());
        assert!(matches!(&q.clauses[0], Clause::SpatialJoin { .. }));
        // WHERE consumed, RETURN remains
        assert_eq!(q.clauses.len(), 2);
        assert!(matches!(&q.clauses[1], Clause::Return(_)));
    }

    #[test]
    fn rewrites_with_and_remainder() {
        let mut q = parse_cypher(
            "MATCH (a:Area), (c:City) WHERE contains(a, c) AND c.name = 'x' RETURN a.name",
        )
        .unwrap();
        fuse_spatial_join(&mut q, &graph_with_spatial());
        if let Clause::SpatialJoin { remainder, .. } = &q.clauses[0] {
            assert!(remainder.is_some(), "remainder should carry c.name = 'x'");
        } else {
            panic!("expected SpatialJoin, got {:?}", q.clauses[0]);
        }
    }

    #[test]
    fn skips_constant_point() {
        let mut q =
            parse_cypher("MATCH (a:Area), (c:City) WHERE contains(a, point(60.0, 10.0)) RETURN a")
                .unwrap();
        fuse_spatial_join(&mut q, &graph_with_spatial());
        assert!(
            !q.clauses
                .iter()
                .any(|c| matches!(c, Clause::SpatialJoin { .. })),
            "constant-point case must not rewrite"
        );
    }

    #[test]
    fn skips_negated_contains() {
        let mut q =
            parse_cypher("MATCH (a:Area), (c:City) WHERE NOT contains(a, c) RETURN a").unwrap();
        fuse_spatial_join(&mut q, &graph_with_spatial());
        assert!(
            !q.clauses
                .iter()
                .any(|c| matches!(c, Clause::SpatialJoin { .. })),
            "NOT contains must fall back to existing path"
        );
    }

    #[test]
    fn skips_three_patterns() {
        let mut q =
            parse_cypher("MATCH (a:Area), (b:Area), (c:City) WHERE contains(a, c) RETURN a")
                .unwrap();
        fuse_spatial_join(&mut q, &graph_with_spatial());
        assert!(
            !q.clauses
                .iter()
                .any(|c| matches!(c, Clause::SpatialJoin { .. })),
            "three-pattern MATCH must not rewrite"
        );
    }

    #[test]
    fn skips_without_spatial_config() {
        let mut q = parse_cypher("MATCH (a:Foo), (c:Bar) WHERE contains(a, c) RETURN a").unwrap();
        fuse_spatial_join(&mut q, &DirGraph::new());
        assert!(
            !q.clauses
                .iter()
                .any(|c| matches!(c, Clause::SpatialJoin { .. })),
            "types without SpatialConfig must not rewrite"
        );
    }

    #[test]
    fn skips_edge_pattern() {
        let mut q =
            parse_cypher("MATCH (a:Area)-[:R]->(x), (c:City) WHERE contains(a, c) RETURN a")
                .unwrap();
        fuse_spatial_join(&mut q, &graph_with_spatial());
        assert!(
            !q.clauses
                .iter()
                .any(|c| matches!(c, Clause::SpatialJoin { .. })),
            "patterns with edges must not rewrite"
        );
    }
}
