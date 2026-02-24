// src/graph/cypher/mod.rs
// Cypher query language implementation for kglite
//
// Architecture:
//   Query String -> Tokenizer -> Parser -> AST -> Planner -> Executor -> Result
//
// The MATCH clause delegates pattern parsing to pattern_matching::parse_pattern()
// WHERE/RETURN/ORDER BY etc. are handled by the Cypher-level parser and executor.

pub mod ast;
pub mod executor;
pub mod parser;
pub mod planner;
pub mod py_convert;
pub mod result;
pub mod result_view;
pub mod tokenizer;

// Re-exports for convenience
pub use executor::{execute_mutable, is_mutation_query, CypherExecutor};
pub use parser::parse_cypher;
pub use planner::{optimize, rewrite_text_score};
pub use result_view::{ResultIter, ResultView};

use crate::datatypes::values::Value;
use crate::graph::schema::DirGraph;

use ast::*;

/// Estimate the number of rows a MATCH clause will produce based on type_indices.
fn estimate_match_rows(m: &MatchClause, graph: &DirGraph) -> Option<usize> {
    let types = collect_node_types(m);
    if types.is_empty() {
        // Untyped scan — total node count
        Some(graph.graph.node_count())
    } else {
        // Use the smallest type's count as the estimate (join selectivity heuristic)
        types
            .iter()
            .map(|t| graph.type_indices.get(t.as_str()).map_or(0, |v| v.len()))
            .min()
    }
}

/// Generate a human-readable query plan string from a parsed (and optimized) query.
/// Includes cardinality estimates when a graph reference is available.
/// Note: `generate_explain_result` is preferred for the Python API (returns structured data).
#[allow(dead_code)]
pub fn generate_explain_plan(query: &CypherQuery, graph: &DirGraph) -> String {
    let mut lines = Vec::new();
    lines.push("Query Plan:".to_string());

    for (i, clause) in query.clauses.iter().enumerate() {
        let step = i + 1;
        let (desc, est) = match clause {
            Clause::Match(m) => {
                let est = estimate_match_rows(m, graph);
                if m.path_assignments.iter().any(|pa| pa.is_shortest_path) {
                    ("ShortestPathScan (MATCH shortestPath)".to_string(), None)
                } else {
                    let types = collect_node_types(m);
                    if types.is_empty() {
                        ("NodeScan (MATCH)".to_string(), est)
                    } else {
                        (format!("NodeScan (MATCH) :{}", types.join(", :")), est)
                    }
                }
            }
            Clause::OptionalMatch(m) => {
                let est = estimate_match_rows(m, graph);
                let types = collect_node_types(m);
                if types.is_empty() {
                    ("OptionalExpand (OPTIONAL MATCH)".to_string(), est)
                } else {
                    (
                        format!("OptionalExpand (OPTIONAL MATCH) :{}", types.join(", :")),
                        est,
                    )
                }
            }
            Clause::Where(_) => ("Filter (WHERE)".to_string(), None),
            Clause::Return(r) => {
                let cols: Vec<String> = r
                    .items
                    .iter()
                    .map(|item| {
                        item.alias
                            .clone()
                            .unwrap_or_else(|| format_expr(&item.expression))
                    })
                    .collect();
                (format!("Projection (RETURN) [{}]", cols.join(", ")), None)
            }
            Clause::With(w) => {
                let has_agg = w
                    .items
                    .iter()
                    .any(|item| executor::is_aggregate_expression(&item.expression));
                if has_agg {
                    let groups: Vec<String> = w
                        .items
                        .iter()
                        .filter(|item| !executor::is_aggregate_expression(&item.expression))
                        .map(|item| format_expr(&item.expression))
                        .collect();
                    let aggs: Vec<String> = w
                        .items
                        .iter()
                        .filter(|item| executor::is_aggregate_expression(&item.expression))
                        .map(|item| {
                            item.alias
                                .clone()
                                .unwrap_or_else(|| format_expr(&item.expression))
                        })
                        .collect();
                    (
                        format!(
                            "Aggregate (WITH) group=[{}] aggs=[{}]",
                            groups.join(", "),
                            aggs.join(", ")
                        ),
                        None,
                    )
                } else {
                    ("Projection (WITH)".to_string(), None)
                }
            }
            Clause::OrderBy(_) => ("Sort (ORDER BY)".to_string(), None),
            Clause::Limit(_) => ("Limit (LIMIT)".to_string(), None),
            Clause::Skip(_) => ("Skip (SKIP)".to_string(), None),
            Clause::Unwind(_) => ("Unwind (UNWIND)".to_string(), None),
            Clause::Union(_) => ("Union (UNION)".to_string(), None),
            Clause::Create(_) => ("Create (CREATE)".to_string(), None),
            Clause::Set(_) => ("Mutate (SET)".to_string(), None),
            Clause::Delete(d) => {
                if d.detach {
                    ("DetachDelete (DETACH DELETE)".to_string(), None)
                } else {
                    ("Delete (DELETE)".to_string(), None)
                }
            }
            Clause::Remove(_) => ("Remove (REMOVE)".to_string(), None),
            Clause::Merge(_) => ("Merge (MERGE)".to_string(), None),
            Clause::Call(c) => {
                let yields: Vec<&str> = c.yield_items.iter().map(|y| y.name.as_str()).collect();
                (
                    format!(
                        "ProcedureCall (CALL {}) YIELD [{}]",
                        c.procedure_name,
                        yields.join(", ")
                    ),
                    None,
                )
            }
            Clause::FusedMatchReturnAggregate { .. } => (
                "FusedMatchReturnAggregate (optimized MATCH + count)".to_string(),
                Some(1),
            ),
            Clause::FusedOptionalMatchAggregate { .. } => (
                "FusedOptionalMatchAggregate (optimized OPTIONAL MATCH + count)".to_string(),
                Some(1),
            ),
            Clause::FusedVectorScoreTopK { limit, .. } => (
                format!(
                    "FusedVectorScoreTopK (optimized RETURN+ORDER BY+LIMIT, k={})",
                    limit
                ),
                Some(*limit),
            ),
            Clause::FusedOrderByTopK { limit, .. } => (
                format!(
                    "FusedOrderByTopK (optimized RETURN+ORDER BY+LIMIT, k={})",
                    limit
                ),
                Some(*limit),
            ),
            Clause::FusedCountAll { .. } => (
                "FusedCountAll (optimized MATCH (n) RETURN count(n))".to_string(),
                Some(1),
            ),
            Clause::FusedCountByType { .. } => {
                let est = graph.type_indices.len();
                (
                    "FusedCountByType (optimized MATCH (n) RETURN n.type, count(n))".to_string(),
                    Some(est),
                )
            }
            Clause::FusedCountEdgesByType { .. } => (
                "FusedCountEdgesByType (optimized MATCH ()-[r]->() RETURN type(r), count(*))"
                    .to_string(),
                None,
            ),
            Clause::FusedCountTypedNode { node_type, .. } => {
                let count = graph
                    .type_indices
                    .get(node_type.as_str())
                    .map_or(0, |v| v.len());
                (
                    format!(
                        "FusedCountTypedNode (optimized MATCH (n:{}) RETURN count(n))",
                        node_type
                    ),
                    Some(count.min(1)),
                )
            }
            Clause::FusedCountTypedEdge { edge_type, .. } => (
                format!(
                    "FusedCountTypedEdge (optimized MATCH ()-[r:{}]->() RETURN count(*))",
                    edge_type
                ),
                Some(1),
            ),
        };
        if let Some(est) = est {
            lines.push(format!("  {}. {} (~{} rows)", step, desc, est));
        } else {
            lines.push(format!("  {}. {}", step, desc));
        }
    }

    // Note applied optimizations
    let has_opt_fusion = query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::FusedOptionalMatchAggregate { .. }));
    let opt_fusion_count = query
        .clauses
        .iter()
        .filter(|c| matches!(c, Clause::FusedOptionalMatchAggregate { .. }))
        .count();
    let has_topk_fusion = query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::FusedVectorScoreTopK { .. }));
    let has_general_topk = query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::FusedOrderByTopK { .. }));

    let mut opts = Vec::new();
    if has_opt_fusion {
        opts.push(format!("optional_match_fusion={}", opt_fusion_count));
    }
    if has_topk_fusion {
        opts.push("vector_score_topk_fusion".to_string());
    }
    if has_general_topk {
        opts.push("order_by_topk_fusion".to_string());
    }

    if !opts.is_empty() {
        lines.push(format!("Optimizations: {}", opts.join(", ")));
    }

    lines.join("\n")
}

/// Generate a structured query plan as a CypherResult with columns
/// [step, operation, estimated_rows].
pub fn generate_explain_result(query: &CypherQuery, graph: &DirGraph) -> result::CypherResult {
    let mut rows = Vec::new();

    for (i, clause) in query.clauses.iter().enumerate() {
        let step = (i + 1) as i64;
        let operation = executor::clause_display_name(clause);
        let est = match clause {
            Clause::Match(m) | Clause::OptionalMatch(m) => estimate_match_rows(m, graph)
                .map(|e| Value::Int64(e as i64))
                .unwrap_or(Value::Null),
            Clause::FusedCountAll { .. }
            | Clause::FusedMatchReturnAggregate { .. }
            | Clause::FusedOptionalMatchAggregate { .. }
            | Clause::FusedCountTypedEdge { .. } => Value::Int64(1),
            Clause::FusedCountTypedNode { node_type, .. } => {
                let n = graph
                    .type_indices
                    .get(node_type.as_str())
                    .map_or(0, |v| v.len());
                Value::Int64(n.min(1) as i64)
            }
            Clause::FusedCountByType { .. } => Value::Int64(graph.type_indices.len() as i64),
            Clause::FusedVectorScoreTopK { limit, .. } | Clause::FusedOrderByTopK { limit, .. } => {
                Value::Int64(*limit as i64)
            }
            _ => Value::Null,
        };

        rows.push(vec![Value::Int64(step), Value::String(operation), est]);
    }

    // Add optimizations as a final metadata row
    let mut optimizations = Vec::new();
    for clause in &query.clauses {
        match clause {
            Clause::FusedOptionalMatchAggregate { .. } => {
                optimizations.push("optional_match_fusion");
            }
            Clause::FusedVectorScoreTopK { .. } => {
                optimizations.push("vector_score_topk_fusion");
            }
            Clause::FusedOrderByTopK { .. } => {
                optimizations.push("order_by_topk_fusion");
            }
            Clause::FusedCountAll { .. }
            | Clause::FusedCountByType { .. }
            | Clause::FusedCountEdgesByType { .. }
            | Clause::FusedCountTypedNode { .. }
            | Clause::FusedCountTypedEdge { .. }
            | Clause::FusedMatchReturnAggregate { .. } => {
                optimizations.push("count_fusion");
            }
            _ => {}
        }
    }

    result::CypherResult {
        columns: vec!["step".into(), "operation".into(), "estimated_rows".into()],
        rows,
        stats: None,
        profile: None,
    }
}

/// Collect node types from a MatchClause's patterns.
fn collect_node_types(m: &MatchClause) -> Vec<String> {
    use crate::graph::pattern_matching::PatternElement;
    let mut types = Vec::new();
    for pattern in &m.patterns {
        for element in &pattern.elements {
            if let PatternElement::Node(np) = element {
                if let Some(ref t) = np.node_type {
                    types.push(t.clone());
                }
            }
        }
    }
    types
}

/// Format an expression for EXPLAIN output in a readable way.
#[allow(dead_code)]
fn format_expr(expr: &Expression) -> String {
    match expr {
        Expression::Variable(v) => v.clone(),
        Expression::PropertyAccess { variable, property } => format!("{}.{}", variable, property),
        Expression::Literal(v) => format!("{:?}", v),
        Expression::FunctionCall {
            name,
            args,
            distinct,
        } => {
            let arg_strs: Vec<String> = args.iter().map(format_expr).collect();
            let dist = if *distinct { "DISTINCT " } else { "" };
            format!("{}({}{})", name, dist, arg_strs.join(", "))
        }
        Expression::Star => "*".to_string(),
        _ => format!("{:?}", expr),
    }
}
