// src/graph/traversal_methods.rs
use crate::datatypes::values::FilterCondition;
use crate::datatypes::values::Value;
use crate::graph::filtering_methods;
use crate::graph::schema::{CurrentSelection, DirGraph, SelectionOperation};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

/// Check if edge properties match all given filter conditions
fn edge_matches_conditions(
    properties: &HashMap<String, Value>,
    conditions: &HashMap<String, FilterCondition>,
) -> bool {
    conditions.iter().all(|(field, condition)| {
        match properties.get(field) {
            Some(value) => filtering_methods::matches_condition(value, condition),
            None => {
                // Missing field is treated as null
                matches!(condition, FilterCondition::IsNull)
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub fn make_traversal(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    connection_type: String,
    level_index: Option<usize>,
    direction: Option<String>,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    filter_connection: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
    new_level: Option<bool>,
) -> Result<(), String> {
    // Validate connection type exists
    if !graph.has_connection_type(&connection_type) {
        return Err(format!(
            "Connection type '{}' does not exist in graph",
            connection_type
        ));
    }

    // First get the source level index
    let source_level_index =
        level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));

    let create_new_level = new_level.unwrap_or(true);

    // Get source level
    let source_level = selection
        .get_level(source_level_index)
        .ok_or_else(|| "No valid source level found for traversal".to_string())?;

    // Early empty check
    if source_level.is_empty() {
        return Err("No source nodes available for traversal".to_string());
    }

    // Set up traversal directions
    let dir = match direction.as_deref() {
        Some("incoming") => Some(Direction::Incoming),
        Some("outgoing") => Some(Direction::Outgoing),
        Some(d) => {
            return Err(format!(
                "Invalid direction: {}. Must be 'incoming' or 'outgoing'",
                d
            ))
        }
        None => None, // Both directions
    };

    // FAST PATH: No filtering, sorting, or limits - optimized for common case
    let use_fast_path = filter_target.is_none()
        && filter_connection.is_none()
        && sort_target.is_none()
        && max_nodes.is_none()
        && create_new_level;

    if use_fast_path {
        return make_traversal_fast(graph, selection, &connection_type, source_level_index, dir);
    }

    // SLOW PATH: Full processing with filtering/sorting/limits
    make_traversal_full(
        graph,
        selection,
        connection_type,
        source_level_index,
        dir,
        filter_target,
        filter_connection,
        sort_target,
        max_nodes,
        create_new_level,
    )
}

/// Fast traversal path for the common case: no filtering, no sorting, no limits.
/// Avoids HashMap overhead by collecting all targets directly.
fn make_traversal_fast(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    connection_type: &str,
    source_level_index: usize,
    direction: Option<Direction>,
) -> Result<(), String> {
    // Get source nodes using iterator to avoid allocation
    let source_level = selection
        .get_level(source_level_index)
        .ok_or_else(|| "No valid source level found for traversal".to_string())?;

    // Collect source nodes (we need this twice - once for iteration, once for the parent map)
    let source_nodes: Vec<NodeIndex> = source_level.iter_node_indices().collect();

    // Create new level
    selection.add_level();
    let target_level_index = selection.get_level_count() - 1;

    // Pre-allocate targets HashSet with estimated capacity
    let mut all_targets_per_parent: HashMap<NodeIndex, Vec<NodeIndex>> =
        HashMap::with_capacity(source_nodes.len());

    // Process each source node
    for &source_node in &source_nodes {
        let mut targets: HashSet<NodeIndex> = HashSet::new();

        // Process edges based on direction
        match direction {
            Some(Direction::Outgoing) => {
                for edge in graph.graph.edges_directed(source_node, Direction::Outgoing) {
                    if edge.weight().connection_type == connection_type {
                        targets.insert(edge.target());
                    }
                }
            }
            Some(Direction::Incoming) => {
                for edge in graph.graph.edges_directed(source_node, Direction::Incoming) {
                    if edge.weight().connection_type == connection_type {
                        targets.insert(edge.source());
                    }
                }
            }
            None => {
                // Both directions
                for edge in graph.graph.edges_directed(source_node, Direction::Outgoing) {
                    if edge.weight().connection_type == connection_type {
                        targets.insert(edge.target());
                    }
                }
                for edge in graph.graph.edges_directed(source_node, Direction::Incoming) {
                    if edge.weight().connection_type == connection_type {
                        targets.insert(edge.source());
                    }
                }
            }
        }

        // Store targets for this parent
        if !targets.is_empty() {
            all_targets_per_parent.insert(source_node, targets.into_iter().collect());
        }
    }

    // Get target level and populate it
    let level = selection
        .get_level_mut(target_level_index)
        .ok_or_else(|| "Failed to access target selection level".to_string())?;

    // Set up operation
    level.operations = vec![SelectionOperation::Traverse {
        connection_type: connection_type.to_string(),
        direction: direction.map(|d| {
            if d == Direction::Incoming {
                "incoming"
            } else {
                "outgoing"
            }
            .to_string()
        }),
        max_nodes: None,
    }];

    // Add all parent->children mappings
    for (parent, children) in all_targets_per_parent {
        level.add_selection(Some(parent), children);
    }

    Ok(())
}

/// Full traversal path with filtering, sorting, and limits support.
#[allow(clippy::too_many_arguments)]
fn make_traversal_full(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    connection_type: String,
    source_level_index: usize,
    direction: Option<Direction>,
    filter_target: Option<&HashMap<String, FilterCondition>>,
    filter_connection: Option<&HashMap<String, FilterCondition>>,
    sort_target: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
    create_new_level: bool,
) -> Result<(), String> {
    // Get source level
    let source_level = selection
        .get_level(source_level_index)
        .ok_or_else(|| "No valid source level found for traversal".to_string())?;

    // Collect all necessary data from source level
    let parents: Vec<NodeIndex> = if create_new_level {
        source_level.iter_node_indices().collect()
    } else {
        source_level.selections.keys().filter_map(|k| *k).collect()
    };

    // Create a mapping of parent nodes to their source nodes
    let source_nodes_map: HashMap<NodeIndex, Vec<NodeIndex>> = if create_new_level {
        parents
            .iter()
            .map(|&parent| (parent, vec![parent]))
            .collect()
    } else {
        source_level
            .selections
            .iter()
            .filter_map(|(parent, children)| parent.map(|p| (p, children.clone())))
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

    // Get and initialize target level
    let level = selection
        .get_level_mut(target_level_index)
        .ok_or_else(|| "Failed to access target selection level".to_string())?;

    // Set up operation
    let operation = SelectionOperation::Traverse {
        connection_type: connection_type.clone(),
        direction: direction.map(|d| {
            if d == Direction::Incoming {
                "incoming"
            } else {
                "outgoing"
            }
            .to_string()
        }),
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

        // Process edges based on direction
        for &source_node in source_nodes {
            match direction {
                Some(Direction::Outgoing) => {
                    for edge in graph.graph.edges_directed(source_node, Direction::Outgoing) {
                        if edge.weight().connection_type == connection_type {
                            if let Some(conn_filter) = filter_connection {
                                if !edge_matches_conditions(&edge.weight().properties, conn_filter)
                                {
                                    continue;
                                }
                            }
                            targets.insert(edge.target());
                        }
                    }
                }
                Some(Direction::Incoming) => {
                    for edge in graph.graph.edges_directed(source_node, Direction::Incoming) {
                        if edge.weight().connection_type == connection_type {
                            if let Some(conn_filter) = filter_connection {
                                if !edge_matches_conditions(&edge.weight().properties, conn_filter)
                                {
                                    continue;
                                }
                            }
                            targets.insert(edge.source());
                        }
                    }
                }
                None => {
                    // Both directions
                    for edge in graph.graph.edges_directed(source_node, Direction::Outgoing) {
                        if edge.weight().connection_type == connection_type {
                            if let Some(conn_filter) = filter_connection {
                                if !edge_matches_conditions(&edge.weight().properties, conn_filter)
                                {
                                    continue;
                                }
                            }
                            targets.insert(edge.target());
                        }
                    }
                    for edge in graph.graph.edges_directed(source_node, Direction::Incoming) {
                        if edge.weight().connection_type == connection_type {
                            if let Some(conn_filter) = filter_connection {
                                if !edge_matches_conditions(&edge.weight().properties, conn_filter)
                                {
                                    continue;
                                }
                            }
                            targets.insert(edge.source());
                        }
                    }
                }
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
            max_nodes,
        );

        // Add the processed nodes to the selection
        level.add_selection(Some(parent), processed_nodes);
    }

    Ok(())
}

pub struct ChildPropertyGroup {
    pub parent_idx: NodeIndex,
    pub parent_title: String,
    pub values: Vec<String>,
}

pub fn get_children_properties(
    graph: &DirGraph,
    selection: &CurrentSelection,
    property: &str,
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
                    match node.get_field_ref("title") {
                        Some(Value::String(s)) => s.clone(),
                        _ => format!("node_{}", parent.index()),
                    }
                } else {
                    format!("node_{}", parent.index())
                };

                // For each parent, collect property values from children
                let mut values_list = Vec::new();

                for &child_idx in children {
                    if let Some(node) = graph.get_node(child_idx) {
                        let value = match node.get_field_ref(property) {
                            Some(Value::String(s)) => s.clone(),
                            Some(Value::Int64(i)) => i.to_string(),
                            Some(Value::Float64(f)) => f.to_string(),
                            Some(Value::Boolean(b)) => b.to_string(),
                            Some(Value::UniqueId(u)) => u.to_string(),
                            Some(Value::DateTime(d)) => d.format("%Y-%m-%d").to_string(),
                            Some(Value::Point { lat, lon }) => {
                                format!("point({}, {})", lat, lon)
                            }
                            Some(Value::Null) => "null".to_string(),
                            Some(Value::NodeRef(idx)) => format!("node#{}", idx),
                            None => continue,
                        };

                        values_list.push(value);
                    }
                }

                result.push(ChildPropertyGroup {
                    parent_idx: parent,
                    parent_title,
                    values: values_list,
                });
            }
        }
    }

    result
}

/// Helper to format a list of values with optional truncation
fn format_property_list(values: &[String], max_length: Option<usize>) -> String {
    let joined = values.join(", ");
    match max_length {
        Some(max) if joined.len() > max => {
            format!("{}...", &joined[..max.saturating_sub(3)])
        }
        _ => joined,
    }
}

pub fn format_for_storage(
    property_groups: &[ChildPropertyGroup],
    max_length: Option<usize>,
) -> Vec<(Option<NodeIndex>, Value)> {
    property_groups
        .iter()
        .map(|group| {
            let formatted = format_property_list(&group.values, max_length);
            (Some(group.parent_idx), Value::String(formatted))
        })
        .collect()
}

pub fn format_for_dictionary(
    property_groups: &[ChildPropertyGroup],
    max_length: Option<usize>,
) -> Vec<(String, String)> {
    property_groups
        .iter()
        .map(|group| {
            let formatted = format_property_list(&group.values, max_length);
            (group.parent_title.clone(), formatted)
        })
        .collect()
}
