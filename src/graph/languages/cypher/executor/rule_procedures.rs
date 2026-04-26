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
