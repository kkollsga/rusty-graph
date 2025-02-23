// src/graph/calculations.rs

use super::statistics_methods::get_parent_child_pairs;
use super::equation_parser::{Parser, Evaluator, Expr};
use super::maintain_graph;
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, CurrentSelection, NodeData};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

pub enum EvaluationResult {
    Stored(()),
    Computed(Vec<StatResult>)
}

#[derive(Debug)]
pub struct StatResult {
    pub node_idx: Option<NodeIndex>,     // Add direct node reference
    pub parent_idx: Option<NodeIndex>,   // Keep parent for aggregations
    pub parent_title: Option<String>,
    pub value: Option<f64>,
    pub error: Option<String>,
}

pub fn process_equation(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    expression: &str,
    level_index: Option<usize>,
    store_as: Option<&str>,
) -> Result<EvaluationResult, String> {
    let results = evaluate_equation(graph, selection, expression, level_index);
    
    if let Some(target_property) = store_as {
        let nodes: Vec<(Option<NodeIndex>, Value)> = results.iter()
            .filter_map(|result| {
                result.value.map(|v| {
                    // For individual nodes, use node_idx, for aggregations use parent_idx
                    let idx = result.node_idx.or(result.parent_idx);
                    (idx, Value::Float64(v))
                })
            })
            .collect();
            
        maintain_graph::update_node_properties(graph, &nodes, target_property)?;
        Ok(EvaluationResult::Stored(()))
    } else {
        Ok(EvaluationResult::Computed(results))
    }
}

pub fn evaluate_equation(
    graph: &DirGraph,
    selection: &CurrentSelection,
    expression: &str,
    level_index: Option<usize>,
) -> Vec<StatResult> {
    let parsed_expr = match Parser::parse_expression(expression) {
        Ok(expr) => expr,
        Err(err) => {
            return vec![StatResult {
                node_idx: None,
                parent_idx: None,
                parent_title: None,
                value: None,
                error: Some(format!("Failed to parse expression: {}", err)),
            }];
        }
    };

    let is_aggregation = has_aggregation(&parsed_expr);

    if is_aggregation {
        // Use parent-child pairs for aggregation
        let pairs = get_parent_child_pairs(selection, level_index);
        
        pairs.iter()
            .flat_map(|pair| {
                let child_nodes: Vec<(NodeIndex, NodeData, HashMap<String, f64>)> = pair.children.iter()
                    .map(|&node_idx| {
                        graph.get_node(node_idx)
                            .map(|node| (node_idx, node.clone(), convert_node_to_object(node).unwrap_or_default()))
                    })
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_default();

                if child_nodes.is_empty() {
                    return vec![StatResult {
                        node_idx: None,
                        parent_idx: pair.parent,
                        parent_title: Some("No valid nodes".to_string()),
                        value: None,
                        error: Some("No valid nodes found for evaluation".to_string()),
                    }];
                }

                let parent_title = pair.parent.and_then(|idx| {
                    graph.get_node(idx)
                        .and_then(|node| node.get_field("title"))
                        .and_then(|v| v.as_string())
                });

                let objects: Vec<HashMap<String, f64>> = child_nodes.into_iter()
                    .map(|(_, _, obj)| obj)
                    .collect();

                vec![match Evaluator::evaluate(&parsed_expr, &objects) {
                    Ok(value) => StatResult {
                        node_idx: None,
                        parent_idx: pair.parent,
                        parent_title,
                        value: Some(value),
                        error: None,
                    },
                    Err(err) => StatResult {
                        node_idx: None,
                        parent_idx: pair.parent,
                        parent_title,
                        value: None,
                        error: Some(err),
                    },
                }]
            })
            .collect()
    } else {
        // For non-aggregation, work directly with nodes in the current level
        let effective_index = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
        let level = match selection.get_level(effective_index) {
            Some(l) => l,
            None => return vec![],
        };

        let nodes = level.get_all_nodes();

        nodes.iter()
            .filter_map(|&node_idx| {
                graph.get_node(node_idx).map(|node| {
                    let title = node.get_field("title")
                        .and_then(|v| v.as_string());
                    let obj = convert_node_to_object(node).unwrap_or_default();

                    match Evaluator::evaluate(&parsed_expr, &[obj]) {
                        Ok(value) => {
                            StatResult {
                                node_idx: Some(node_idx),  // Include the node_idx for direct updates
                                parent_idx: None,
                                parent_title: title,
                                value: Some(value),
                                error: None,
                            }
                        },
                        Err(err) => StatResult {
                            node_idx: Some(node_idx),  // Include the node_idx even for errors
                            parent_idx: None,
                            parent_title: title,
                            value: None,
                            error: Some(err),
                        },
                    }
                })
            })
            .collect()
    }
}

fn has_aggregation(expr: &Expr) -> bool {
    match expr {
        Expr::Aggregate(_, _) => true,
        Expr::Add(left, right) => has_aggregation(left) || has_aggregation(right),
        Expr::Subtract(left, right) => has_aggregation(left) || has_aggregation(right),
        Expr::Multiply(left, right) => has_aggregation(left) || has_aggregation(right),
        Expr::Divide(left, right) => has_aggregation(left) || has_aggregation(right),
        _ => false,
    }
}

fn convert_node_to_object(node: &NodeData) -> Option<HashMap<String, f64>> {
    let mut object = HashMap::new();
    
    match node {
        NodeData::Regular { properties, .. } | NodeData::Schema { properties, .. } => {
            for (key, value) in properties {
                if let Some(num_value) = match value {
                    Value::Int64(i) => Some(*i as f64),
                    Value::Float64(f) => Some(*f),
                    Value::UniqueId(u) => Some(*u as f64),
                    Value::String(s) => s.parse::<f64>().ok(),
                    _ => None,
                } {
                    object.insert(key.clone(), num_value);
                }
            }
        }
    }
    
    if object.is_empty() {
        None
    } else {
        Some(object)
    }
}

pub fn count_nodes_in_level(
    selection: &CurrentSelection,
    level_index: Option<usize>,
) -> usize {
    let effective_index = match level_index {
        Some(idx) => idx,
        None => selection.get_level_count().saturating_sub(1)
    };

    let level = selection.get_level(effective_index)
        .expect("Level should exist");  // Safe due to saturating_sub
    
    level.get_all_nodes().len()
}