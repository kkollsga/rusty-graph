// src/graph/custom_equation.rs

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
    pub parent_idx: Option<petgraph::graph::NodeIndex>,  // Added field
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
        // Check if this is an aggregation expression
        let is_aggregation = results.first()
            .map(|r| r.parent_idx.is_some() && r.value.is_some() && r.error.is_none())
            .unwrap_or(false);

        if is_aggregation {
            store_parent_results(graph, &results, target_property)?;
        } else {
            store_child_results(graph, selection, level_index, &results, target_property)?;
        }
        Ok(EvaluationResult::Stored(()))
    } else {
        Ok(EvaluationResult::Computed(results))
    }
}

fn store_parent_results(
    graph: &mut DirGraph,
    results: &[StatResult],
    property: &str,
) -> Result<(), String> {
    let nodes: Vec<(Option<NodeIndex>, Value)> = results.iter()
        .filter_map(|result| {
            result.value.map(|v| (
                result.parent_idx,
                Value::Float64(v)
            ))
        })
        .collect();

    maintain_graph::update_node_properties(graph, &nodes, property)
}

fn store_child_results(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    level_index: Option<usize>,
    results: &[StatResult],
    property: &str,
) -> Result<(), String> {
    for result in results {
        if let Some(value) = result.value {
            maintain_graph::update_selected_node_properties(
                graph,
                selection,
                level_index,
                property,
                Value::Float64(value)
            )?;
        }
    }
    Ok(())
}

pub fn evaluate_equation(
    graph: &DirGraph,
    selection: &CurrentSelection,
    expression: &str,
    level_index: Option<usize>,
) -> Vec<StatResult> {
    // Parse the expression once at the start
    let parsed_expr = match Parser::parse_expression(expression) {
        Ok(expr) => expr,
        Err(err) => {
            return vec![StatResult {
                parent_idx: None,
                parent_title: None,
                value: None,
                error: Some(format!("Failed to parse expression: {}", err)),
            }];
        }
    };

    // Check if this is an aggregation expression
    let is_aggregation = has_aggregation(&parsed_expr);
    let pairs = get_parent_child_pairs(selection, level_index);
    
    pairs.iter()
        .flat_map(|pair| {
            // Convert node data to objects for evaluation
            let child_nodes: Vec<(NodeIndex, NodeData, HashMap<String, f64>)> = pair.children.iter()
                .filter_map(|&node_idx| {
                    graph.get_node(node_idx)
                        .and_then(|node| {
                            convert_node_to_object(node)
                                .map(|obj| (node_idx, node.clone(), obj))
                        })
                })
                .collect();

            if child_nodes.is_empty() {
                return vec![StatResult {
                    parent_idx: pair.parent,
                    parent_title: Some("No valid nodes".to_string()),
                    value: None,
                    error: Some("No valid nodes found for evaluation".to_string()),
                }];
            }

            if is_aggregation {
                // For aggregation, compute one result per parent
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
                        parent_idx: pair.parent,
                        parent_title,
                        value: Some(value),
                        error: None,
                    },
                    Err(err) => StatResult {
                        parent_idx: pair.parent,
                        parent_title,
                        value: None,
                        error: Some(err),
                    },
                }]
            } else {
                // For direct calculations, compute one result per child
                child_nodes.into_iter()
                    .map(|(_, node, obj)| {  // Changed from (node_idx, node, obj)
                        let title = node.get_field("title")
                            .and_then(|v| v.as_string());

                        match Evaluator::evaluate(&parsed_expr, &[obj]) {
                            Ok(value) => StatResult {
                                parent_idx: None,  // For direct calculations, we don't use parent
                                parent_title: title,
                                value: Some(value),
                                error: None,
                            },
                            Err(err) => StatResult {
                                parent_idx: None,
                                parent_title: title,
                                value: None,
                                error: Some(err),
                            },
                        }
                    })
                    .collect()
            }
        })
        .collect()
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