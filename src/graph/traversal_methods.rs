// src/graph/traversal_methods.rs
use std::collections::{HashSet, HashMap};
use std::iter::FromIterator;
use petgraph::visit::EdgeRef;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use crate::graph::schema::{DirGraph, CurrentSelection, SelectionOperation};
use crate::datatypes::values::FilterCondition;
use crate::graph::filtering_methods;
use crate::datatypes::values::Value;

fn get_connected_targets(
    graph: &DirGraph,
    source_node: NodeIndex,
    directions: &[Direction],
    connection_type: &str,
    targets: &[NodeIndex],
) -> Vec<NodeIndex> {
    targets.iter()
        .filter(|&&target| {
            directions.iter().any(|dir| {
                graph.graph.edges_directed(source_node, *dir)
                    .any(|edge| {
                        edge.weight().connection_type == connection_type &&
                        (match dir {
                            Direction::Outgoing => edge.target(),
                            Direction::Incoming => edge.source(),
                        }) == target
                    })
            })
        })
        .cloned()
        .collect()
}

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

    // Collect all necessary data from source level before any modifications
    let (parents, source_nodes_map) = {
        let source_level = selection.get_level(source_level_index)
            .ok_or_else(|| "No valid source level found for traversal".to_string())?;

        let parents: Vec<NodeIndex> = if create_new_level {
            source_level.get_all_nodes()
        } else {
            source_level.selections.keys()
                .filter_map(|k| *k)
                .collect()
        };

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

        (parents, source_nodes_map)
    };

    // Early empty check
    if parents.is_empty() {
        return Err("No source nodes available for traversal".to_string());
    }

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

    if !create_new_level {
        for parent in &parents {
            level.selections.entry(Some(*parent)).or_default().clear();
        }
    }

    // Set up operation
    let operation = SelectionOperation::Traverse {
        connection_type: connection_type.clone(),
        direction: direction.clone(),
        max_nodes,
    };
    level.operations = vec![operation];

    let mut all_targets: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();
    let mut seen_targets = HashSet::new();
    let empty_vec = Vec::new();

    // Process each parent and their source nodes
    for &parent in &parents {
        let source_nodes = source_nodes_map.get(&parent).unwrap_or(&empty_vec);
        
        for &source_node in source_nodes {
            for dir in &directions {
                let matching_edges = graph.graph.edges_directed(source_node, *dir)
                    .filter(|edge| edge.weight().connection_type == connection_type);

                for edge in matching_edges {
                    let target = match dir {
                        Direction::Outgoing => edge.target(),
                        Direction::Incoming => edge.source(),
                    };
                    
                    if seen_targets.insert(target) {
                        all_targets.entry(parent)
                            .or_insert_with(HashSet::new)
                            .insert(target);
                    }
                }
            }
        }

        // Always add a selection for the parent, even if empty
        if !all_targets.contains_key(&parent) {
            level.add_selection(Some(parent), Vec::new());
        }
    }

    // Process all collected targets
    for &parent in &parents {
        if let Some(targets) = all_targets.get(&parent) {
            let target_vec = Vec::from_iter(targets.iter().cloned());
            
            if create_new_level {
                // Process all nodes together for new levels
                let mut processed_nodes = filtering_methods::process_nodes(
                    graph,
                    target_vec,
                    filter_target,
                    sort_target,
                    None
                );
                
                if let Some(max) = max_nodes {
                    processed_nodes.truncate(max);
                }
                
                level.add_selection(Some(parent), processed_nodes);
            } else {
                // Process per source node for existing levels
                let source_nodes = source_nodes_map.get(&parent).unwrap_or(&empty_vec);
                let mut final_nodes = Vec::new();
                
                for &source_node in source_nodes {
                    let source_targets = get_connected_targets(
                        graph,
                        source_node,
                        &directions,
                        &connection_type,
                        &target_vec
                    );
                    
                    let processed_nodes = filtering_methods::process_nodes(
                        graph,
                        source_targets,
                        filter_target,
                        sort_target,
                        max_nodes  // Apply max_nodes per source node
                    );
                    
                    final_nodes.extend(processed_nodes);
                }
                
                level.add_selection(Some(parent), final_nodes);
            }
        } else if !create_new_level {
            // Preserve existing selection if no new targets
            level.add_selection(Some(parent), 
                level.selections.get(&Some(parent))
                    .map(|existing| existing.clone())
                    .unwrap_or_default()
            );
        }
    }

    Ok(())
}

pub fn get_children_properties(
    graph: &DirGraph,
    selection: &CurrentSelection,
    property: &str
) -> Vec<(NodeIndex, Vec<String>)> {
    let mut result = Vec::new();
    
    // Get the current level index
    let level_index = selection.get_level_count().saturating_sub(1);
    
    // Get all parents with their children
    if let Some(level) = selection.get_level(level_index) {
        for (&parent_opt, children) in &level.selections {
            if let Some(parent) = parent_opt {
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
                
                result.push((parent, values_list));
            }
        }
    }
    
    result
}

pub fn format_children_properties_for_storage(
    property_lists: &[(NodeIndex, Vec<String>)],
    max_length: Option<usize>
) -> Vec<(Option<NodeIndex>, Value)> {
    property_lists.iter()
        .map(|(parent, values)| {
            // Join into a comma-separated list
            let list_value = values.join(", ");
            
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
            
            (Some(*parent), Value::String(final_value))
        })
        .collect()
}