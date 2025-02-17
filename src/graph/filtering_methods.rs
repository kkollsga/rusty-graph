// src/graph/filtering_methods.rs
use std::collections::HashMap;
use petgraph::graph::NodeIndex;
use crate::datatypes::values::{Value, FilterCondition};
use crate::graph::schema::{DirGraph, CurrentSelection};

fn matches_condition(value: &Value, condition: &FilterCondition) -> bool {
    match condition {
        FilterCondition::Equals(target) => value == target,
        FilterCondition::NotEquals(target) => value != target,
        FilterCondition::GreaterThan(target) => compare_values(value, target) == Some(std::cmp::Ordering::Greater),
        FilterCondition::GreaterThanEquals(target) => {
            matches!(compare_values(value, target), 
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal))
        },
        FilterCondition::LessThan(target) => compare_values(value, target) == Some(std::cmp::Ordering::Less),
        FilterCondition::LessThanEquals(target) => {
            matches!(compare_values(value, target), 
                Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal))
        },
        FilterCondition::In(targets) => targets.contains(value),
    }
}

fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Null, Value::Null) => Some(std::cmp::Ordering::Equal),
        (Value::Null, _) => Some(std::cmp::Ordering::Less),
        (_, Value::Null) => Some(std::cmp::Ordering::Greater),
        
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        (Value::UniqueId(a), Value::UniqueId(b)) => Some(a.cmp(b)),
        (Value::DateTime(a), Value::DateTime(b)) => Some(a.cmp(b)),
        (Value::Boolean(a), Value::Boolean(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

// Core operations that work with Vec<NodeIndex>
fn filter_nodes_by_conditions(
    graph: &DirGraph,
    nodes: Vec<NodeIndex>,
    conditions: &HashMap<String, FilterCondition>
) -> Vec<NodeIndex> {
    // Special case for type-only filter
    if conditions.len() == 1 {
        if let Some((key, condition)) = conditions.iter().next() {
            if key == "type" {
                if let FilterCondition::Equals(Value::String(type_value)) = condition {
                    if let Some(type_nodes) = graph.type_indices.get(type_value) {
                        return nodes.into_iter()
                            .filter(|node| type_nodes.contains(node))
                            .collect();
                    }
                }
            }
        }
    }

    // Standard filtering
    nodes.into_iter()
        .filter(|&idx| {
            if let Some(node) = graph.get_node(idx) {
                conditions.iter().all(|(key, condition)| {
                    if let Some(value) = node.get_field(key) {
                        matches_condition(&value, condition)
                    } else {
                        false
                    }
                })
            } else {
                false
            }
        })
        .collect()
}

fn sort_nodes_by_fields(
    graph: &DirGraph,
    mut nodes: Vec<NodeIndex>,
    sort_fields: &[(String, bool)]
) -> Vec<NodeIndex> {
    nodes.sort_by(|&a, &b| {
        for (field, ascending) in sort_fields {
            if let (Some(node_a), Some(node_b)) = (graph.get_node(a), graph.get_node(b)) {
                if let (Some(val_a), Some(val_b)) = (node_a.get_field(field), node_b.get_field(field)) {
                    if let Some(ordering) = compare_values(&val_a, &val_b) {
                        return if *ascending { ordering } else { ordering.reverse() };
                    }
                }
            }
        }
        std::cmp::Ordering::Equal
    });
    nodes
}

fn limit_nodes(nodes: Vec<NodeIndex>, max_nodes: usize) -> Vec<NodeIndex> {
    nodes.into_iter().take(max_nodes).collect()
}

// Main processing function for raw node operations
pub fn process_nodes(
    graph: &DirGraph,
    nodes: Vec<NodeIndex>,
    conditions: Option<HashMap<String, FilterCondition>>,
    sort_fields: Option<Vec<(String, bool)>>,
    max_nodes: Option<usize>
) -> Vec<NodeIndex> {
    let mut result = nodes;
    
    if let Some(conditions) = conditions {
        result = filter_nodes_by_conditions(graph, result, &conditions);
    }
    
    if let Some(fields) = sort_fields.as_ref() {
        result = sort_nodes_by_fields(graph, result, fields);
    }
    
    if let Some(max) = max_nodes {
        result = limit_nodes(result, max);
    }
    
    result
}

// Public interface functions that work with CurrentSelection

pub fn filter_nodes(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    conditions: HashMap<String, FilterCondition>,
    sort_fields: Option<Vec<(String, bool)>>,
    max_nodes: Option<usize>
) -> Result<(), String> {
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection.get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    if current_index == 0 {
        level.selections.clear();
    }

    // Handle empty selections
    if level.selections.is_empty() {
        // Check for type-only filter first
        if conditions.len() == 1 {
            if let Some((key, condition)) = conditions.iter().next() {
                if key == "type" {
                    if let FilterCondition::Equals(Value::String(type_value)) = condition {
                        if let Some(type_nodes) = graph.type_indices.get(type_value) {
                            // Use type indices directly for empty selections
                            let nodes = type_nodes.clone();
                            
                            // Apply sort if needed
                            let processed = if let Some(fields) = sort_fields {
                                sort_nodes_by_fields(graph, nodes, &fields)
                            } else {
                                nodes
                            };
                            
                            // Apply limit if needed
                            let processed = if let Some(max) = max_nodes {
                                limit_nodes(processed, max)
                            } else {
                                processed
                            };
                            
                            if !processed.is_empty() {
                                level.add_selection(None, processed);
                                level.add_filter(conditions);
                                return Ok(());
                            }
                        }
                        // If type exists but no nodes found, return early
                        return Ok(());
                    }
                }
            }
        }

        // Fallback to regular processing for non-type filters
        let all_nodes: Vec<NodeIndex> = graph.graph.node_indices()
            .filter(|&idx| graph.get_node(idx).map_or(false, |n| n.is_regular()))
            .collect();
            
        let processed = process_nodes(
            graph,
            all_nodes,
            Some(conditions.clone()),
            sort_fields.clone(),
            max_nodes
        );
        
        if !processed.is_empty() {
            level.add_selection(None, processed);
        }
    } else {
        // Process existing selections normally
        let new_selections: Vec<_> = level.selections.iter()
            .map(|(parent, nodes)| {
                let processed = process_nodes(
                    graph,
                    nodes.clone(),
                    Some(conditions.clone()),
                    sort_fields.clone(),
                    max_nodes
                );
                (*parent, processed)
            })
            .filter(|(_, nodes)| !nodes.is_empty())
            .collect();

        level.selections = new_selections;
    }

    level.add_filter(conditions);
    if let Some(fields) = sort_fields {
        level.add_sort(fields);
    }

    Ok(())
}

pub fn sort_nodes(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    sort_fields: Vec<(String, bool)>
) -> Result<(), String> {
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection.get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    if level.selections.is_empty() {
        let all_nodes: Vec<NodeIndex> = graph.graph.node_indices()
            .filter(|&idx| graph.get_node(idx).map_or(false, |n| n.is_regular()))
            .collect();
            
        let sorted = sort_nodes_by_fields(graph, all_nodes, &sort_fields);
        if !sorted.is_empty() {
            level.add_selection(None, sorted);
        }
    } else {
        let new_selections: Vec<_> = level.selections.iter()
            .map(|(parent, nodes)| {
                let sorted = sort_nodes_by_fields(graph, nodes.clone(), &sort_fields);
                (*parent, sorted)
            })
            .filter(|(_, nodes)| !nodes.is_empty())
            .collect();

        level.selections = new_selections;
    }

    level.add_sort(sort_fields);
    Ok(())
}

pub fn limit_nodes_per_group(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    max_per_group: usize,
) -> Result<(), String> {
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection.get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    if level.selections.is_empty() {
        let all_nodes: Vec<NodeIndex> = graph.graph.node_indices()
            .filter(|&idx| graph.get_node(idx).map_or(false, |n| n.is_regular()))
            .collect();
            
        let limited = limit_nodes(all_nodes, max_per_group);
        if !limited.is_empty() {
            level.add_selection(None, limited);
        }
    } else {
        let new_selections: Vec<_> = level.selections.iter()
            .map(|(parent, nodes)| {
                let limited = limit_nodes(nodes.clone(), max_per_group);
                (*parent, limited)
            })
            .filter(|(_, nodes)| !nodes.is_empty())
            .collect();

        level.selections = new_selections;
    }

    Ok(())
}