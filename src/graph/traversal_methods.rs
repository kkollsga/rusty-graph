// src/graph/traversal_methods.rs
use std::collections::{HashSet, HashMap};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use petgraph::graph::NodeIndex;
use crate::graph::schema::{DirGraph, CurrentSelection, SelectionOperation};
use crate::datatypes::values::{Value, FilterCondition};
use crate::graph::filtering_methods;

pub fn make_traversal(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    connection_type: String,
    level_index: Option<usize>,
    direction: Option<String>,
    filter_conditions: Option<HashMap<String, FilterCondition>>,
    sort_fields: Option<Vec<(String, bool)>>,
    max_nodes: Option<usize>,
    new_level: Option<bool>,
) -> Result<(), String> {
    let source_level_index = level_index.unwrap_or_else(|| 
        selection.get_level_count().saturating_sub(1)
    );
    
    let create_new_level = new_level.unwrap_or(true);
    if create_new_level {
        selection.add_level();
    }
    
    let target_level_index = if create_new_level {
        selection.get_level_count() - 1
    } else {
        source_level_index
    };

    let source_level = selection.get_level(source_level_index)
        .ok_or_else(|| "No valid source level found for traversal".to_string())?;
    
    let source_nodes = source_level.get_all_nodes();
    
    // Store the parent mapping for nodes in the current level
    let parent_mapping: HashMap<_, _> = source_level.selections.iter()
        .flat_map(|(parent, children)| children.iter().map(|child| (*child, *parent)))
        .collect();

    let directions = match direction.as_deref() {
        Some("incoming") => vec![Direction::Incoming],
        Some("outgoing") => vec![Direction::Outgoing],
        Some(d) => return Err(format!("Invalid direction: {}. Must be 'incoming' or 'outgoing'", d)),
        None => vec![Direction::Outgoing, Direction::Incoming],
    };

    let level = selection.get_level_mut(target_level_index)
        .ok_or_else(|| "Failed to access target selection level".to_string())?;

    if !create_new_level {
        level.selections.clear();
    }

    level.operations.push(SelectionOperation::Traverse {
        connection_type: connection_type.clone(),
        direction: direction.clone(),
        max_nodes,
    });

    // Process each source node
    for &source_node in &source_nodes {
        let mut target_nodes = Vec::new();
        
        // Try each direction
        'direction_loop: for dir in &directions {
            let matching_edges = graph.graph.edges_directed(source_node, *dir)
                .filter(|edge| edge.weight().connection_type == connection_type);

            // Collect all target nodes for this direction
            for edge in matching_edges {
                let target = match dir {
                    Direction::Outgoing => edge.target(),
                    Direction::Incoming => edge.source(),
                };
                target_nodes.push(target);
            }

            // If we found nodes in this direction and no specific direction was requested,
            // we can skip checking the other direction
            if !target_nodes.is_empty() && direction.is_none() {
                break 'direction_loop;
            }
        }

        // If we found any target nodes, process them using the filtering API
        if !target_nodes.is_empty() {
            let processed_nodes = filtering_methods::process_nodes(
                graph,
                target_nodes,
                filter_conditions.clone(),
                sort_fields.clone(),
                max_nodes
            );

            // Only add selection if we still have nodes after processing
            if !processed_nodes.is_empty() {
                let parent = if create_new_level {
                    Some(source_node)
                } else {
                    parent_mapping.get(&source_node).copied().flatten()
                };

                // Group nodes by parent when not creating a new level
                if !create_new_level {
                    if let Some(p) = parent {
                        // Update or create the selection for this parent
                        let existing_nodes = level.selections.iter()
                            .find(|(existing_parent, _)| *existing_parent == Some(p))
                            .map(|(_, nodes)| nodes.clone())
                            .unwrap_or_default();
                        let mut all_nodes = existing_nodes;
                        all_nodes.extend(processed_nodes);
                        level.selections.retain(|(existing_parent, _)| *existing_parent != Some(p));
                        level.add_selection(Some(p), all_nodes);
                    }
                } else {
                    level.add_selection(parent, processed_nodes);
                }
            }
        }
    }

    // Check if any traversals were found
    if selection.get_level(target_level_index)
        .map_or(true, |level| level.is_empty()) {
        // Only clear if we created a new level
        if create_new_level {
            selection.clear();
        }
        return Err(format!("No valid traversals found for connection type: {}", connection_type));
    }

    Ok(())
}