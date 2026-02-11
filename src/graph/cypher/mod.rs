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
pub mod tokenizer;

// Re-exports for convenience
pub use executor::{execute_mutable, is_mutation_query, CypherExecutor};
pub use parser::parse_cypher;
pub use planner::optimize;

use ast::*;

/// Generate a human-readable query plan from a parsed (and optimized) query.
/// Returns the plan as a formatted string without executing the query.
pub fn generate_explain_plan(query: &CypherQuery) -> String {
    let mut lines = Vec::new();
    lines.push("Query Plan:".to_string());

    for (i, clause) in query.clauses.iter().enumerate() {
        let step = i + 1;
        let desc = match clause {
            Clause::Match(m) => {
                if m.path_assignments.iter().any(|pa| pa.is_shortest_path) {
                    "ShortestPathScan (MATCH shortestPath)".to_string()
                } else {
                    let types = collect_node_types(m);
                    if types.is_empty() {
                        "NodeScan (MATCH)".to_string()
                    } else {
                        format!("NodeScan (MATCH) :{}", types.join(", :"))
                    }
                }
            }
            Clause::OptionalMatch(m) => {
                let types = collect_node_types(m);
                if types.is_empty() {
                    "OptionalExpand (OPTIONAL MATCH)".to_string()
                } else {
                    format!("OptionalExpand (OPTIONAL MATCH) :{}", types.join(", :"))
                }
            }
            Clause::Where(_) => "Filter (WHERE)".to_string(),
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
                format!("Projection (RETURN) [{}]", cols.join(", "))
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
                    format!(
                        "Aggregate (WITH) group=[{}] aggs=[{}]",
                        groups.join(", "),
                        aggs.join(", ")
                    )
                } else {
                    "Projection (WITH)".to_string()
                }
            }
            Clause::OrderBy(_) => "Sort (ORDER BY)".to_string(),
            Clause::Limit(_) => "Limit (LIMIT)".to_string(),
            Clause::Skip(_) => "Skip (SKIP)".to_string(),
            Clause::Unwind(_) => "Unwind (UNWIND)".to_string(),
            Clause::Union(_) => "Union (UNION)".to_string(),
            Clause::Create(_) => "Create (CREATE)".to_string(),
            Clause::Set(_) => "Mutate (SET)".to_string(),
            Clause::Delete(d) => {
                if d.detach {
                    "DetachDelete (DETACH DELETE)".to_string()
                } else {
                    "Delete (DELETE)".to_string()
                }
            }
            Clause::Remove(_) => "Remove (REMOVE)".to_string(),
            Clause::Merge(_) => "Merge (MERGE)".to_string(),
            Clause::FusedOptionalMatchAggregate { .. } => {
                "FusedOptionalMatchAggregate (optimized OPTIONAL MATCH + count)".to_string()
            }
        };
        lines.push(format!("  {}. {}", step, desc));
    }

    // Note applied optimizations
    let has_fusion = query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::FusedOptionalMatchAggregate { .. }));
    let fusion_count = query
        .clauses
        .iter()
        .filter(|c| matches!(c, Clause::FusedOptionalMatchAggregate { .. }))
        .count();

    if has_fusion {
        lines.push(format!(
            "Optimizations: optional_match_fusion={}",
            fusion_count
        ));
    }

    lines.join("\n")
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
