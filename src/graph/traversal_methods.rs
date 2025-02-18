use std::collections::{HashSet, HashMap};
use std::iter::FromIterator;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use crate::graph::schema::{DirGraph, CurrentSelection, SelectionOperation};
use crate::datatypes::values::FilterCondition;
use crate::graph::filtering_methods;

pub fn make_traversal(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    connection_type: String,
    level_index: Option<usize>,
    direction: Option<String>,
    filter_conditions: Option<&HashMap<String, FilterCondition>>,
    sort_fields: Option<&Vec<(String, bool)>>,
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
    
    // Get source level and collect all necessary data from it
    let create_new_level = new_level.unwrap_or(true);
    let (source_nodes, parent_mapping) = {
        let source_level = selection.get_level(source_level_index)
            .ok_or_else(|| "No valid source level found for traversal".to_string())?;
        
        let nodes = source_level.get_all_nodes();
        
        let mapping = if !create_new_level {
            source_level.selections.iter()
                .flat_map(|(parent, children)| 
                    children.iter().map(|child| (*child, *parent)))
                .collect()
        } else {
            HashMap::new()
        };
        
        (nodes, mapping)
    };

    // Early empty check
    if source_nodes.is_empty() {
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
        level.selections.clear();
    }

    // Set up operation
    let operation = SelectionOperation::Traverse {
        connection_type: connection_type.clone(),
        direction: direction.clone(),
        max_nodes,
    };
    level.operations = vec![operation];

    // Batch collection of target nodes
    let mut all_targets = HashMap::new(); // parent -> HashSet<target>
    let mut seen_targets = HashSet::new();

    // Process each source node and collect all targets
    for &source_node in &source_nodes {
        let parent = if create_new_level {
            Some(source_node)
        } else {
            parent_mapping.get(&source_node).copied().flatten()
        };

        if let Some(parent_node) = parent {
            let mut found_targets = false;
            
            for dir in &directions {
                let matching_edges = graph.graph.edges_directed(source_node, *dir)
                    .filter(|edge| edge.weight().connection_type == connection_type);

                for edge in matching_edges {
                    found_targets = true;
                    let target = match dir {
                        Direction::Outgoing => edge.target(),
                        Direction::Incoming => edge.source(),
                    };
                    
                    if seen_targets.insert(target) {
                        all_targets.entry(parent_node)
                            .or_insert_with(HashSet::new)
                            .insert(target);
                    }
                }
            }
        }
    }

    // Process all collected targets at once
    if !all_targets.is_empty() {
        for (parent, targets) in all_targets {
            let target_vec = Vec::from_iter(targets);
            let processed_nodes = filtering_methods::process_nodes(
                graph,
                target_vec,
                filter_conditions,      // Already Option<&HashMap>
                sort_fields,           // Already Option<&Vec>
                max_nodes
            );
    
            if !processed_nodes.is_empty() {
                level.add_selection(Some(parent), processed_nodes);
            }
        }
    }

    // Check if any traversals were found
    if level.is_empty() {
        if create_new_level {
            selection.clear();
        }
        return Err(format!("No valid traversals found for connection type: {}", connection_type));
    }

    Ok(())
}