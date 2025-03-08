// src/graph/traversal_methods.rs
use std::collections::{HashSet, HashMap};
use petgraph::visit::EdgeRef;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use crate::graph::schema::{DirGraph, CurrentSelection, SelectionOperation};
use crate::datatypes::values::FilterCondition;
use crate::graph::filtering_methods;
use crate::datatypes::values::Value;

pub fn make_traversal(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    connection_type: String,
    level_index: Option<usize>,
    direction: Option<String>,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
    new_level: Option<bool>,
) -> Result<(), String> {
    // Validate connection type exists
    if !graph.has_connection_type(&connection_type) {
        return Err(format!("Connection type '{}' does not exist in graph", connection_type));
    }

    // First get the source level index
    let source_level_index = level_index.unwrap_or_else(|| 
        selection.get_level_count().saturating_sub(1)
    );
    
    let create_new_level = new_level.unwrap_or(true);

    // Get source level
    let source_level = selection.get_level(source_level_index)
        .ok_or_else(|| "No valid source level found for traversal".to_string())?;

    // Collect all necessary data from source level
    let parents: Vec<NodeIndex> = if create_new_level {
        source_level.get_all_nodes()
    } else {
        source_level.selections.keys()
            .filter_map(|k| *k)
            .collect()
    };

    // Early empty check
    if parents.is_empty() {
        return Err("No source nodes available for traversal".to_string());
    }

    // Create a mapping of parent nodes to their source nodes
    let source_nodes_map: HashMap<NodeIndex, Vec<NodeIndex>> = if create_new_level {
        parents.iter()
            .map(|&parent| (parent, vec![parent]))
            .collect()
    } else {
        source_level.selections.iter()
            .filter_map(|(parent, children)| {
                parent.map(|p| (p, children.clone()))
            })
            .collect()
    };

    // Now we can safely modify the selection
    if create_new_level {
        selection.add_level();
    }

    let target_level_index = if create_new_level {
        selection.get_level_count() - 1
    } else {
        source_level_index
    };

    // Set up traversal directions
    let directions = match direction.as_deref() {
        Some("incoming") => vec![Direction::Incoming],
        Some("outgoing") => vec![Direction::Outgoing],
        Some(d) => return Err(format!("Invalid direction: {}. Must be 'incoming' or 'outgoing'", d)),
        None => vec![Direction::Outgoing, Direction::Incoming],
    };

    // Get and initialize target level
    let level = selection.get_level_mut(target_level_index)
        .ok_or_else(|| "Failed to access target selection level".to_string())?;

    // Set up operation
    let operation = SelectionOperation::Traverse {
        connection_type: connection_type.clone(),
        direction: direction.clone(),
        max_nodes,
    };
    level.operations = vec![operation];

    // Define an empty vector to use when no source nodes exist
    let empty_vec: Vec<NodeIndex> = Vec::new();

    // Process each parent node once
    for &parent in &parents {
        // Use a reference to an existing empty vector to avoid temporary lifetime issues
        let source_nodes = source_nodes_map.get(&parent).unwrap_or(&empty_vec);
        
        if !create_new_level {
            // Clear existing selection for this parent
            level.selections.entry(Some(parent)).or_default().clear();
        }
        
        // Collect all targets for this parent in one pass
        let mut targets = HashSet::new();
        
        // Optimize direction handling by collecting all nodes at once
        for &source_node in source_nodes {
            // Process outgoing edges if needed
            if directions.contains(&Direction::Outgoing) {
                graph.graph.edges_directed(source_node, Direction::Outgoing)
                    .filter(|edge| edge.weight().connection_type == connection_type)
                    .for_each(|edge| {
                        targets.insert(edge.target());
                    });
            }
            
            // Process incoming edges if needed
            if directions.contains(&Direction::Incoming) {
                graph.graph.edges_directed(source_node, Direction::Incoming)
                    .filter(|edge| edge.weight().connection_type == connection_type)
                    .for_each(|edge| {
                        targets.insert(edge.source());
                    });
            }
        }
        
        // Convert to Vec for processing
        let target_vec: Vec<NodeIndex> = targets.into_iter().collect();
        
        // Apply filtering and sorting in one pass
        let processed_nodes = filtering_methods::process_nodes(
            graph,
            target_vec,
            filter_target,
            sort_target,
            max_nodes
        );
        
        // Add the processed nodes to the selection
        level.add_selection(Some(parent), processed_nodes);
    }

    Ok(())
}

pub struct ChildPropertyGroup {
    pub parent_idx: NodeIndex,
    pub parent_title: String,
    pub values: Vec<String>
}

pub fn get_children_properties(
    graph: &DirGraph,
    selection: &CurrentSelection,
    property: &str
) -> Vec<ChildPropertyGroup> {
    let mut result = Vec::new();
    
    // Get the current level index
    let level_index = selection.get_level_count().saturating_sub(1);
    
    // Get all parents with their children
    if let Some(level) = selection.get_level(level_index) {
        for (&parent_opt, children) in &level.selections {
            if let Some(parent) = parent_opt {
                // Get parent title
                let parent_title = if let Some(node) = graph.get_node(parent) {
                    match node.get_field("title") {
                        Some(Value::String(s)) => s.clone(),
                        _ => format!("node_{}", parent.index())
                    }
                } else {
                    format!("node_{}", parent.index())
                };
                
                // For each parent, collect property values from children
                let mut values_list = Vec::new();
                
                for &child_idx in children {
                    if let Some(node) = graph.get_node(child_idx) {
                        let value = match node.get_field(property) {
                            Some(Value::String(s)) => s.clone(),
                            Some(Value::Int64(i)) => i.to_string(),
                            Some(Value::Float64(f)) => f.to_string(),
                            Some(Value::Boolean(b)) => b.to_string(),
                            Some(Value::UniqueId(u)) => u.to_string(),
                            Some(Value::DateTime(d)) => d.format("%Y-%m-%d").to_string(),
                            Some(Value::Null) => "null".to_string(),
                            None => continue,
                        };
                        
                        values_list.push(value);
                    }
                }
                
                result.push(ChildPropertyGroup {
                    parent_idx: parent,
                    parent_title,
                    values: values_list
                });
            }
        }
    }
    
    result
}

pub fn format_for_storage(
    property_groups: &[ChildPropertyGroup],
    max_length: Option<usize>
) -> Vec<(Option<NodeIndex>, Value)> {
    property_groups.iter()
        .map(|group| {
            // Join into a comma-separated list
            let list_value = group.values.join(", ");
            
            // Truncate if a max length is specified
            let final_value = if let Some(max) = max_length {
                if list_value.len() > max {
                    format!("{}...", &list_value[..max.saturating_sub(3)])
                } else {
                    list_value
                }
            } else {
                list_value
            };
            
            (Some(group.parent_idx), Value::String(final_value))
        })
        .collect()
}

pub fn format_for_dictionary(
    property_groups: &[ChildPropertyGroup],
    max_length: Option<usize>
) -> Vec<(String, String)> {
    property_groups.iter()
        .map(|group| {
            // Join into a comma-separated list
            let list_value = group.values.join(", ");
            
            // Truncate if a max length is specified
            let final_value = if let Some(max) = max_length {
                if list_value.len() > max {
                    format!("{}...", &list_value[..max.saturating_sub(3)])
                } else {
                    list_value
                }
            } else {
                list_value
            };
            
            (group.parent_title.clone(), final_value)
        })
        .collect()
}