//! Structural-validator rule procedures exposed via Cypher `CALL`.
//!
//! Six procedures replace the legacy Python rule-pack stack from 0.8.16–
//! 0.8.18. Each walks the graph directly via [`GraphRead`] and emits
//! [`ResultRow`]s with node bindings — no recursion through the executor,
//! no YAML parse, no `g.cypher()` round-trip per rule. The rows flow into
//! surrounding Cypher (`WHERE` / `ORDER BY` / `LIMIT` / `RETURN`) the
//! same way [`super::call_clause::CypherExecutor::centrality_to_rows`]
//! does for `pagerank`.
//!
//! Naming follows the existing flat-name convention (`pagerank`,
//! `cluster`, `connected_components`); rule procedures are dispatched
//! by the `match` statement in `call_clause::execute_call`.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;
use petgraph::Direction;

use super::super::ast::YieldItem;
use super::super::result::ResultRow;
use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;

/// `CALL orphan_node({type: 'Wellbore'}) YIELD node`
///
/// Yields one row per node of the requested type that has zero edges in
/// any direction. Useful as a baseline integrity check — orphans are
/// almost always ingest artifacts.
pub(super) fn execute_orphan_node(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let node_type = require_string_param(params, "type", "orphan_node")?;
    let yield_var = require_node_yield(yield_items, "orphan_node", "node")?;
    let nodes = type_indices(graph, &node_type)?;

    let mut rows = Vec::with_capacity(0);
    for &nidx in nodes {
        if graph
            .graph
            .edges_directed(nidx, Direction::Outgoing)
            .next()
            .is_some()
        {
            continue;
        }
        if graph
            .graph
            .edges_directed(nidx, Direction::Incoming)
            .next()
            .is_some()
        {
            continue;
        }
        rows.push(make_node_row(&yield_var, nidx));
    }
    Ok(rows)
}

/// `CALL self_loop({type: 'Person', edge: 'KNOWS'}) YIELD node`
///
/// Yields one row per node of `type` that has an outgoing `edge` whose
/// target is itself. Always a data error in tree-shaped hierarchies;
/// occasionally legitimate for self-referential domain edges.
pub(super) fn execute_self_loop(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let node_type = require_string_param(params, "type", "self_loop")?;
    let edge_type = require_string_param(params, "edge", "self_loop")?;
    let yield_var = require_node_yield(yield_items, "self_loop", "node")?;
    let nodes = type_indices(graph, &node_type)?;
    let edge_key = InternedKey::from_str(&edge_type);

    let mut rows = Vec::new();
    for &nidx in nodes {
        let hit = graph
            .graph
            .edges_directed(nidx, Direction::Outgoing)
            .any(|er| er.target() == nidx && er.weight().connection_type == edge_key);
        if hit {
            rows.push(make_node_row(&yield_var, nidx));
        }
    }
    Ok(rows)
}

/// `CALL cycle_2step({type: 'Person', edge: 'KNOWS'}) YIELD node_a, node_b`
///
/// Yields one row per `(a, b)` pair where `a -[edge]-> b -[edge]-> a`,
/// `a` and `b` are both of `type`, and `a < b` (deduplicates the pair).
/// Variables are named `node_a` / `node_b` to avoid clashing with the
/// Cypher reserved word `END` (CASE expressions).
pub(super) fn execute_cycle_2step(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let node_type = require_string_param(params, "type", "cycle_2step")?;
    let edge_type = require_string_param(params, "edge", "cycle_2step")?;
    let (start_var, end_var) =
        require_two_node_yields(yield_items, "cycle_2step", "node_a", "node_b")?;
    let nodes = type_indices(graph, &node_type)?;
    let node_set: std::collections::HashSet<NodeIndex> = nodes.iter().copied().collect();
    let edge_key = InternedKey::from_str(&edge_type);

    let mut rows = Vec::new();
    for &a in nodes {
        for er_a in graph.graph.edges_directed(a, Direction::Outgoing) {
            if er_a.weight().connection_type != edge_key {
                continue;
            }
            let b = er_a.target();
            if a >= b {
                continue;
            }
            if !node_set.contains(&b) {
                continue;
            }
            let returns = graph
                .graph
                .edges_directed(b, Direction::Outgoing)
                .any(|er_b| er_b.target() == a && er_b.weight().connection_type == edge_key);
            if returns {
                let mut row = ResultRow::new();
                row.node_bindings.insert(start_var.clone(), a);
                row.node_bindings.insert(end_var.clone(), b);
                rows.push(row);
            }
        }
    }
    Ok(rows)
}

/// `CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node`
///
/// Yields one row per node of `type` that has **no outgoing** edge of
/// `edge`. Refuses to execute when the graph's actual schema for `edge`
/// makes `type` the target rather than the source — flips that case to a
/// `DirectionMismatch` error suggesting `missing_inbound_edge`.
pub(super) fn execute_missing_required_edge(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let node_type = require_string_param(params, "type", "missing_required_edge")?;
    let edge_type = require_string_param(params, "edge", "missing_required_edge")?;
    let yield_var = require_node_yield(yield_items, "missing_required_edge", "node")?;
    validate_direction(
        graph,
        &node_type,
        &edge_type,
        OUTBOUND,
        "missing_required_edge",
    )?;
    let nodes = type_indices(graph, &node_type)?;
    let edge_key = InternedKey::from_str(&edge_type);

    let mut rows = Vec::new();
    for &nidx in nodes {
        let has_edge = has_edge_of_type(graph, nidx, Direction::Outgoing, &edge_type, edge_key);
        if !has_edge {
            rows.push(make_node_row(&yield_var, nidx));
        }
    }
    Ok(rows)
}

/// `CALL missing_inbound_edge({type: 'Discovery', edge: 'IN_DISCOVERY'}) YIELD node`
///
/// Mirror of [`execute_missing_required_edge`]: yields nodes of `type`
/// with no **incoming** edge of `edge`.
pub(super) fn execute_missing_inbound_edge(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let node_type = require_string_param(params, "type", "missing_inbound_edge")?;
    let edge_type = require_string_param(params, "edge", "missing_inbound_edge")?;
    let yield_var = require_node_yield(yield_items, "missing_inbound_edge", "node")?;
    validate_direction(
        graph,
        &node_type,
        &edge_type,
        INBOUND,
        "missing_inbound_edge",
    )?;
    let nodes = type_indices(graph, &node_type)?;
    let edge_key = InternedKey::from_str(&edge_type);

    let mut rows = Vec::new();
    for &nidx in nodes {
        let has_edge = has_edge_of_type(graph, nidx, Direction::Incoming, &edge_type, edge_key);
        if !has_edge {
            rows.push(make_node_row(&yield_var, nidx));
        }
    }
    Ok(rows)
}

/// `CALL duplicate_title({type: 'Prospect'}) YIELD node`
///
/// Yields one row per node of `type` whose title is shared with at least
/// one other node of the same type. Agents can aggregate downstream:
///
/// ```cypher
/// CALL duplicate_title({type: 'Prospect'}) YIELD node
/// WITH node.title AS t, collect(node) AS dups
/// WHERE size(dups) > 1
/// RETURN t, size(dups) AS count
/// ORDER BY count DESC LIMIT 20
/// ```
pub(super) fn execute_duplicate_title(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let node_type = require_string_param(params, "type", "duplicate_title")?;
    let yield_var = require_node_yield(yield_items, "duplicate_title", "node")?;
    let nodes = type_indices(graph, &node_type)?;

    // Pass 1 — count titles
    let mut counts: HashMap<String, u32> = HashMap::with_capacity(nodes.len());
    for &nidx in nodes {
        if let Some(title) = title_of(graph, nidx) {
            *counts.entry(title).or_insert(0) += 1;
        }
    }
    // Pass 2 — emit nodes whose title count > 1
    let mut rows = Vec::new();
    for &nidx in nodes {
        if let Some(title) = title_of(graph, nidx) {
            if counts.get(&title).copied().unwrap_or(0) > 1 {
                rows.push(make_node_row(&yield_var, nidx));
            }
        }
    }
    Ok(rows)
}

/// `CALL inverse_violation({rel_a: 'parent_of', rel_b: 'child_of'}) YIELD a, b`
///
/// Yields one row per `(a, b)` pair where `(a)-[rel_a]->(b)` exists but
/// the inverse `(b)-[rel_b]->(a)` does not. Useful when two relations
/// are declared as logical inverses (parent_of/child_of, manages/works_for,
/// cites/cited_by) and you want to find unidirectional cases.
pub(super) fn execute_inverse_violation(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let rel_a = require_string_param(params, "rel_a", "inverse_violation")?;
    let rel_b = require_string_param(params, "rel_b", "inverse_violation")?;
    let (a_var, b_var) = require_two_node_yields(yield_items, "inverse_violation", "a", "b")?;
    let key_a = InternedKey::from_str(&rel_a);
    let key_b = InternedKey::from_str(&rel_b);

    let mut rows = Vec::new();
    for a in graph.graph.node_indices() {
        for er in graph.graph.edges_directed(a, Direction::Outgoing) {
            if er.weight().connection_type != key_a {
                continue;
            }
            let b = er.target();
            let has_inverse = graph
                .graph
                .edges_directed(b, Direction::Outgoing)
                .any(|er2| er2.target() == a && er2.weight().connection_type == key_b);
            if !has_inverse {
                let mut row = ResultRow::new();
                row.node_bindings.insert(a_var.clone(), a);
                row.node_bindings.insert(b_var.clone(), b);
                rows.push(row);
            }
        }
    }
    Ok(rows)
}

/// `CALL transitivity_violation({rel: 'subClassOf'}) YIELD a, b, c`
///
/// For every `(a)-[rel]->(b)-[rel]->(c)` chain, yields the triple when no
/// direct `(a)-[rel]->(c)` edge exists. Generalizes the OCTF subclass-fold
/// audit pattern. Pairs with `cycle_2step` for the cycle case.
pub(super) fn execute_transitivity_violation(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let rel = require_string_param(params, "rel", "transitivity_violation")?;
    let a_var = require_node_yield(yield_items, "transitivity_violation", "a")?;
    let b_var = require_node_yield(yield_items, "transitivity_violation", "b")?;
    let c_var = require_node_yield(yield_items, "transitivity_violation", "c")?;
    let key = InternedKey::from_str(&rel);

    let mut rows = Vec::new();
    for a in graph.graph.node_indices() {
        // Collect direct successors of a (used for the closure check).
        let direct_a: std::collections::HashSet<NodeIndex> = graph
            .graph
            .edges_directed(a, Direction::Outgoing)
            .filter(|er| er.weight().connection_type == key)
            .map(|er| er.target())
            .collect();
        for &b in &direct_a {
            for er_b in graph.graph.edges_directed(b, Direction::Outgoing) {
                if er_b.weight().connection_type != key {
                    continue;
                }
                let c = er_b.target();
                if c == a || c == b {
                    continue;
                }
                if !direct_a.contains(&c) {
                    let mut row = ResultRow::new();
                    row.node_bindings.insert(a_var.clone(), a);
                    row.node_bindings.insert(b_var.clone(), b);
                    row.node_bindings.insert(c_var.clone(), c);
                    rows.push(row);
                }
            }
        }
    }
    Ok(rows)
}

/// `CALL cardinality_violation({type: 'Wellbore', edge: 'IN_LICENCE', min: 1, max: 1}) YIELD node, count`
///
/// Yields nodes of `type` whose outgoing edge count of `edge` falls
/// outside `[min, max]`. Either bound is optional (`min` defaults to 0,
/// `max` defaults to no upper limit). Setting `max: 1` catches functional-
/// property violations; setting `min: 1` catches missing-required-edge.
pub(super) fn execute_cardinality_violation(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let node_type = require_string_param(params, "type", "cardinality_violation")?;
    let edge_type = require_string_param(params, "edge", "cardinality_violation")?;
    let min_count = call_param_i64(params, "min", 0).max(0) as usize;
    let max_count = call_param_opt_i64(params, "max").map(|v| v.max(0) as usize);
    let node_var = require_node_yield(yield_items, "cardinality_violation", "node")?;
    let count_var = require_scalar_yield(yield_items, "cardinality_violation", "count")?;
    let nodes = type_indices(graph, &node_type)?;
    let edge_key = InternedKey::from_str(&edge_type);

    let mut rows = Vec::new();
    for &nidx in nodes {
        let count = graph
            .graph
            .edges_directed(nidx, Direction::Outgoing)
            .filter(|er| er.weight().connection_type == edge_key)
            .count();
        let too_few = count < min_count;
        let too_many = max_count.is_some_and(|m| count > m);
        if too_few || too_many {
            let mut row = ResultRow::new();
            row.node_bindings.insert(node_var.clone(), nidx);
            row.projected
                .insert(count_var.clone(), Value::Int64(count as i64));
            rows.push(row);
        }
    }
    Ok(rows)
}

/// `CALL type_domain_violation({edge: 'CITES', expected_source: 'Case'}) YIELD source, target`
///
/// Yields edges of `edge` whose source node is not of `expected_source`
/// type. Useful as a post-load schema check.
pub(super) fn execute_type_domain_violation(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let edge_type = require_string_param(params, "edge", "type_domain_violation")?;
    let expected = require_string_param(params, "expected_source", "type_domain_violation")?;
    let src_var = require_node_yield(yield_items, "type_domain_violation", "source")?;
    let tgt_var = require_node_yield(yield_items, "type_domain_violation", "target")?;
    let key = InternedKey::from_str(&edge_type);

    let mut rows = Vec::new();
    for er in graph.graph.edge_references() {
        if er.weight().connection_type != key {
            continue;
        }
        let src = er.source();
        let actual_type = match graph.graph.node_weight(src) {
            Some(n) => n.node_type_str(&graph.interner),
            None => continue,
        };
        if actual_type != expected {
            let mut row = ResultRow::new();
            row.node_bindings.insert(src_var.clone(), src);
            row.node_bindings.insert(tgt_var.clone(), er.target());
            rows.push(row);
        }
    }
    Ok(rows)
}

/// `CALL type_range_violation({edge: 'CITES', expected_target: 'Case'}) YIELD source, target`
///
/// Mirror of [`execute_type_domain_violation`]: yields edges whose
/// target node is not of `expected_target` type.
pub(super) fn execute_type_range_violation(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let edge_type = require_string_param(params, "edge", "type_range_violation")?;
    let expected = require_string_param(params, "expected_target", "type_range_violation")?;
    let src_var = require_node_yield(yield_items, "type_range_violation", "source")?;
    let tgt_var = require_node_yield(yield_items, "type_range_violation", "target")?;
    let key = InternedKey::from_str(&edge_type);

    let mut rows = Vec::new();
    for er in graph.graph.edge_references() {
        if er.weight().connection_type != key {
            continue;
        }
        let tgt = er.target();
        let actual_type = match graph.graph.node_weight(tgt) {
            Some(n) => n.node_type_str(&graph.interner),
            None => continue,
        };
        if actual_type != expected {
            let mut row = ResultRow::new();
            row.node_bindings.insert(src_var.clone(), er.source());
            row.node_bindings.insert(tgt_var.clone(), tgt);
            rows.push(row);
        }
    }
    Ok(rows)
}

/// `CALL parallel_edges({edge: 'CITES'}) YIELD a, b, count`
///
/// Yields one row per `(a, b)` pair connected by more than one edge of
/// the same `edge` type. Almost always a load-time bug — duplicate rows
/// in the source CSV, or an upsert path that didn't dedupe.
pub(super) fn execute_parallel_edges(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let edge_type = require_string_param(params, "edge", "parallel_edges")?;
    let (a_var, b_var) = require_two_node_yields(yield_items, "parallel_edges", "a", "b")?;
    let count_var = require_scalar_yield(yield_items, "parallel_edges", "count")?;
    let key = InternedKey::from_str(&edge_type);

    let mut counts: HashMap<(NodeIndex, NodeIndex), u32> = HashMap::new();
    for er in graph.graph.edge_references() {
        if er.weight().connection_type == key {
            *counts.entry((er.source(), er.target())).or_insert(0) += 1;
        }
    }
    let mut rows = Vec::new();
    for ((a, b), c) in counts {
        if c > 1 {
            let mut row = ResultRow::new();
            row.node_bindings.insert(a_var.clone(), a);
            row.node_bindings.insert(b_var.clone(), b);
            row.projected
                .insert(count_var.clone(), Value::Int64(c as i64));
            rows.push(row);
        }
    }
    Ok(rows)
}

/// `CALL null_property({type: 'Person', property: 'email'}) YIELD node`
///
/// Yields nodes of `type` where `property` is absent or null. Pairs with
/// `missing_required_edge` for the property side of the integrity story.
pub(super) fn execute_null_property(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let node_type = require_string_param(params, "type", "null_property")?;
    let property = require_string_param(params, "property", "null_property")?;
    let yield_var = require_node_yield(yield_items, "null_property", "node")?;
    let nodes = type_indices(graph, &node_type)?;

    let mut rows = Vec::new();
    for &nidx in nodes {
        let node = match graph.graph.node_weight(nidx) {
            Some(n) => n,
            None => continue,
        };
        let val = node.get_property_value(&property);
        let is_null = match val {
            None => true,
            Some(Value::Null) => true,
            Some(Value::String(ref s)) if s.is_empty() => true,
            _ => false,
        };
        if is_null {
            rows.push(make_node_row(&yield_var, nidx));
        }
    }
    Ok(rows)
}

/// Per-edge-type presence check. The disk backend's
/// `edges_directed_filtered` may use an inverted-index fast path that
/// can miss specific (node, edge_type) combinations on sparsely-built
/// disk graphs — iterating the unfiltered edge stream and matching
/// against the interned key is correct in all storage modes and the
/// cost is identical for nodes with low fan-out.
fn has_edge_of_type(
    graph: &DirGraph,
    nidx: NodeIndex,
    dir: Direction,
    _edge_type_name: &str,
    edge_type_key: InternedKey,
) -> bool {
    graph
        .graph
        .edges_directed(nidx, dir)
        .any(|er| er.weight().connection_type == edge_type_key)
}

fn title_of(graph: &DirGraph, nidx: NodeIndex) -> Option<String> {
    let nd = graph.graph.node_weight(nidx)?;
    match nd.title().as_ref() {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        other => Some(format!("{other:?}")),
    }
}

// ─────────────────────────── helpers ────────────────────────────────

const OUTBOUND: bool = true;
const INBOUND: bool = false;

fn require_string_param(
    params: &HashMap<String, Value>,
    key: &str,
    proc: &str,
) -> Result<String, String> {
    match params.get(key) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(format!(
            "CALL {proc}: parameter '{key}' must be a string, got {other:?}"
        )),
        None => Err(format!(
            "CALL {proc}: missing required parameter '{key}'. \
             Use map syntax — e.g. CALL {proc}({{{key}: 'X'}})."
        )),
    }
}

fn require_node_yield(
    yield_items: &[YieldItem],
    proc: &str,
    expected: &str,
) -> Result<String, String> {
    if yield_items.iter().any(|y| y.name == expected) {
        // Use alias when present so the binding is reachable downstream
        // by the name the agent picked.
        let item = yield_items.iter().find(|y| y.name == expected).unwrap();
        Ok(item.alias.clone().unwrap_or_else(|| expected.to_string()))
    } else {
        Err(format!(
            "CALL {proc}: must YIELD '{expected}'. Got {got:?}.",
            got = yield_items
                .iter()
                .map(|y| y.name.as_str())
                .collect::<Vec<_>>(),
        ))
    }
}

fn require_two_node_yields(
    yield_items: &[YieldItem],
    proc: &str,
    a: &str,
    b: &str,
) -> Result<(String, String), String> {
    let av = require_node_yield(yield_items, proc, a)?;
    let bv = require_node_yield(yield_items, proc, b)?;
    Ok((av, bv))
}

/// Validate that `expected` appears in `yield_items` and return the alias
/// the caller assigned (or the original name). Used for non-node yields
/// (e.g. `count`, `values`) that flow into `row.projected` rather than
/// `row.node_bindings`.
fn require_scalar_yield(
    yield_items: &[YieldItem],
    proc: &str,
    expected: &str,
) -> Result<String, String> {
    require_node_yield(yield_items, proc, expected)
}

fn call_param_i64(params: &HashMap<String, Value>, key: &str, default: i64) -> i64 {
    match params.get(key) {
        Some(Value::Int64(v)) => *v,
        Some(Value::Float64(v)) => *v as i64,
        _ => default,
    }
}

fn call_param_opt_i64(params: &HashMap<String, Value>, key: &str) -> Option<i64> {
    match params.get(key) {
        Some(Value::Int64(v)) => Some(*v),
        Some(Value::Float64(v)) => Some(*v as i64),
        _ => None,
    }
}

fn type_indices<'a>(graph: &'a DirGraph, node_type: &str) -> Result<&'a Vec<NodeIndex>, String> {
    graph
        .type_indices
        .get(node_type)
        .ok_or_else(|| format!("Type '{node_type}' has no nodes in this graph"))
}

fn make_node_row(yield_var: &str, nidx: NodeIndex) -> ResultRow {
    let mut row = ResultRow::new();
    row.node_bindings.insert(yield_var.to_string(), nidx);
    row
}

fn validate_direction(
    graph: &DirGraph,
    node_type: &str,
    edge_type: &str,
    outbound: bool,
    proc: &str,
) -> Result<(), String> {
    let info = match graph.connection_type_metadata.get(edge_type) {
        Some(i) => i,
        None => return Ok(()), // edge unknown → graph might be partial; trust the author
    };
    let on_correct_side = if outbound {
        info.source_types.contains(node_type)
    } else {
        info.target_types.contains(node_type)
    };
    if on_correct_side || info.source_types.is_empty() || info.target_types.is_empty() {
        // Allow when empty source/target_types — pre-rename graphs may not
        // have populated metadata even though the edge is real.
        return Ok(());
    }

    let mut sources: Vec<&String> = info.source_types.iter().collect();
    let mut targets: Vec<&String> = info.target_types.iter().collect();
    sources.sort();
    targets.sort();

    if outbound {
        if info.target_types.contains(node_type) {
            Err(format!(
                "DirectionMismatch in CALL {proc}: '{edge_type}' flows {sources:?} → \
                 {targets:?} — '{node_type}' is on the target side. \
                 Use missing_inbound_edge with the same parameters instead."
            ))
        } else {
            Err(format!(
                "DirectionMismatch in CALL {proc}: '{edge_type}' flows {sources:?} → \
                 {targets:?} — '{node_type}' isn't a source type for this edge. \
                 Set type to one of {sources:?}, or pick a different edge."
            ))
        }
    } else if info.source_types.contains(node_type) {
        Err(format!(
            "DirectionMismatch in CALL {proc}: '{edge_type}' flows {sources:?} → \
             {targets:?} — '{node_type}' is on the source side. \
             Use missing_required_edge with the same parameters instead."
        ))
    } else {
        Err(format!(
            "DirectionMismatch in CALL {proc}: '{edge_type}' flows {sources:?} → \
             {targets:?} — '{node_type}' isn't a target type for this edge. \
             Set type to one of {targets:?}, or pick a different edge."
        ))
    }
}
