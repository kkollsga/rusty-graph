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
    pub node_idx: Option<NodeIndex>,
    pub parent_idx: Option<NodeIndex>,
    pub parent_title: Option<String>,
    pub value: Value,
    pub error_msg: Option<String>,  // Added error field
}

pub fn process_equation(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    expression: &str,
    level_index: Option<usize>,
    store_as: Option<&str>,
) -> Result<EvaluationResult, String> {
    // Parse the expression first
    let parsed_expr = match Parser::parse_expression(expression) {
        Ok(expr) => expr,
        Err(err) => return Err(format!("Failed to parse expression: {}. Check for syntax errors or case sensitivity in function names (use 'sum', not 'SUM').", err)),
    };
    
    let is_aggregation = has_aggregation(&parsed_expr);
    let results = evaluate_equation(graph, selection, &parsed_expr, level_index);
    
    // If we don't need to store results, just return them directly
    if store_as.is_none() {
        if results.is_empty() {
            return Err("No results from calculation".to_string());
        }
        
        return Ok(EvaluationResult::Computed(results));
    }
    
    // Only proceed with node updating logic if we need to store results
    let target_property = store_as.unwrap();
    
    // Determine where to store results based on whether there's aggregation
    let effective_level_index = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    
    // Prepare a Vec to hold valid nodes for update
    let mut nodes_to_update: Vec<(Option<NodeIndex>, Value)> = Vec::new();
    
    if is_aggregation {
        // For aggregation - get actual parent nodes from the selection
        for result in &results {
            if let Some(parent_idx) = result.parent_idx {
                // Verify the parent node exists in the graph
                if graph.get_node(parent_idx).is_some() {
                    nodes_to_update.push((Some(parent_idx), result.value.clone()));
                }
            }
        }
    } else {
        // For non-aggregation - get actual child nodes from the selection
        if let Some(level) = selection.get_level(effective_level_index) {
            // Create HashMap from node indices to results
            let result_map: HashMap<NodeIndex, &StatResult> = results.iter()
                .filter_map(|r| r.node_idx.map(|idx| (idx, r)))
                .collect();
            
            // Get all node indices directly from the current level
            for node_idx in level.get_all_nodes() {
                // Direct HashMap lookup instead of linear search
                if let Some(&result) = result_map.get(&node_idx) {
                    // Verify node exists in the graph - IMPORTANT: Must check here
                    if graph.get_node(node_idx).is_some() {
                        nodes_to_update.push((Some(node_idx), result.value.clone()));
                    }
                }
            }
        }
    }
    
    // Check if we found any valid nodes to update
    if nodes_to_update.is_empty() {
        return Err(format!(
            "No valid nodes found to store '{}'. Selection level: {}, Aggregation: {}", 
            target_property, effective_level_index, is_aggregation
        ));
    }
    
    // Update the node properties with verified node indices
    maintain_graph::update_node_properties(graph, &nodes_to_update, target_property)?;
    Ok(EvaluationResult::Stored(()))
}


// Modified evaluate_equation to take a parsed expression directly
pub fn evaluate_equation(
    graph: &DirGraph,
    selection: &CurrentSelection,
    parsed_expr: &Expr,
    level_index: Option<usize>,
) -> Vec<StatResult> {
    let is_aggregation = has_aggregation(parsed_expr);

    if is_aggregation {
        let pairs = get_parent_child_pairs(selection, level_index);
        
        // IMPROVEMENT #2: Cache parent titles to avoid redundant lookups
        let parent_titles: HashMap<NodeIndex, Option<String>> = pairs.iter()
            .filter_map(|pair| pair.parent.map(|idx| (
                idx, 
                graph.get_node(idx)
                    .and_then(|node| node.get_field("title"))
                    .and_then(|v| v.as_string())
            )))
            .collect();
        
        pairs.iter()
            .map(|pair| {
                let child_nodes: Vec<(NodeIndex, NodeData, HashMap<String, Value>)> = pair.children.iter()
                    .filter_map(|&node_idx| {
                        graph.get_node(node_idx)
                            .map(|node| (node_idx, node.clone(), convert_node_to_object(node)))
                    })
                    .collect();

                if child_nodes.is_empty() {
                    return StatResult {
                        node_idx: None,
                        parent_idx: pair.parent,
                        // Use cached parent title instead of looking it up again
                        parent_title: pair.parent.and_then(|idx| parent_titles.get(&idx).cloned().flatten()),
                        value: Value::Null,
                        error_msg: Some("No valid nodes found".to_string()),
                    };
                }

                let objects: Vec<HashMap<String, Value>> = child_nodes.into_iter()
                    .map(|(_, _, obj)| obj)
                    .collect();

                match Evaluator::evaluate(parsed_expr, &objects) {
                    Ok(value) => StatResult {
                        node_idx: None,
                        parent_idx: pair.parent,
                        // Use cached parent title
                        parent_title: pair.parent.and_then(|idx| parent_titles.get(&idx).cloned().flatten()),
                        value,
                        error_msg: None,
                    },
                    Err(err) => StatResult {
                        node_idx: None,
                        parent_idx: pair.parent,
                        // Use cached parent title
                        parent_title: pair.parent.and_then(|idx| parent_titles.get(&idx).cloned().flatten()),
                        value: Value::Null,
                        error_msg: Some(err),
                    },
                }
            })
            .collect()
    } else {
        let effective_index = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
        let level = match selection.get_level(effective_index) {
            Some(l) => l,
            None => return vec![],
        };

        let nodes = level.get_all_nodes();

        nodes.iter()
            .map(|&node_idx| {
                match graph.get_node(node_idx) {
                    Some(node) => {
                        let title = node.get_field("title")
                            .and_then(|v| v.as_string());
                        let obj = convert_node_to_object(node);
                
                        match Evaluator::evaluate(parsed_expr, &[obj]) {
                            Ok(value) => StatResult {
                                node_idx: Some(node_idx),
                                parent_idx: None,
                                parent_title: title,
                                value,
                                error_msg: None,
                            },
                            Err(err) => {
                                StatResult {
                                    node_idx: Some(node_idx),
                                    parent_idx: None,
                                    parent_title: title,
                                    value: Value::Null,
                                    error_msg: Some(err),
                                }
                            }
                        }
                    },
                    None => StatResult {
                        node_idx: Some(node_idx),
                        parent_idx: None,
                        parent_title: None,
                        value: Value::Null,
                        error_msg: Some("Node not found".to_string()),
                    },
                }
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

fn convert_node_to_object(node: &NodeData) -> HashMap<String, Value> {
    let mut object = HashMap::new();
    
    match node {
        NodeData::Regular { properties, .. } | NodeData::Schema { properties, .. } => {
            // Process all properties
            for (key, value) in properties {
                match value {
                    Value::Int64(_) | Value::Float64(_) | Value::UniqueId(_) => {
                        object.insert(key.clone(), value.clone());
                    }
                    Value::Null => {
                        object.insert(key.clone(), Value::Null);
                    }
                    Value::String(s) => {
                        // Try to parse as number
                        if let Ok(num) = s.parse::<f64>() {
                            object.insert(key.clone(), Value::Float64(num));
                        } else {
                            // Include the string value too
                            object.insert(key.clone(), value.clone());
                        }
                    }
                    _ => {
                        // Include all other value types
                        object.insert(key.clone(), value.clone());
                    }
                }
            }
        }
    }
    
    object
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
        .expect("Level should exist");
    
    level.get_all_nodes().len()
}

pub fn count_nodes_by_parent(
    graph: &DirGraph,
    selection: &CurrentSelection,
    level_index: Option<usize>,
) -> Vec<StatResult> {
    let pairs = get_parent_child_pairs(selection, level_index);
    
    pairs.iter()
        .map(|pair| {
            StatResult {
                node_idx: None,
                parent_idx: pair.parent,
                parent_title: pair.parent.and_then(|idx| {
                    graph.get_node(idx)
                        .and_then(|node| node.get_field("title"))
                        .and_then(|v| v.as_string())
                }),
                value: Value::Int64(pair.children.len() as i64),
                error_msg: None,
            }
        })
        .collect()
}

pub fn store_count_results(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    level_index: Option<usize>,
    group_by_parent: bool,
    target_property: &str,
) -> Result<(), String> {
    let mut nodes_to_update: Vec<(Option<NodeIndex>, Value)> = Vec::new();
    
    if group_by_parent {
        // For grouped counting, store count for each parent
        let counts = count_nodes_by_parent(graph, selection, level_index);
        
        for result in &counts {
            if let Some(parent_idx) = result.parent_idx {
                // Verify the parent node exists in the graph
                if graph.get_node(parent_idx).is_some() {
                    nodes_to_update.push((Some(parent_idx), result.value.clone()));
                }
            }
        }
    } else {
        // For flat counting, store same count for all nodes in level
        let count = count_nodes_in_level(selection, level_index);
        let effective_index = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
        
        if let Some(level) = selection.get_level(effective_index) {
            // Apply the count to each node in the level
            for node_idx in level.get_all_nodes() {
                if graph.get_node(node_idx).is_some() {
                    nodes_to_update.push((Some(node_idx), Value::Int64(count as i64)));
                }
            }
        } else {
            return Err(format!("No valid level found at index {}", effective_index));
        }
    }
    
    // Check if we found any valid nodes to update
    if nodes_to_update.is_empty() {
        return Err(format!(
            "No valid nodes found to store '{}' count values.", target_property
        ));
    }
    
    // Use the optimized batch update (which no longer checks existence)
    maintain_graph::update_node_properties(graph, &nodes_to_update, target_property)
}