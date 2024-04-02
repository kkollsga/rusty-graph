use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use crate::data_types::AttributeValue; 
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::schema::{Node, Relation};

/// Retrieves nodes by their unique ID, with an optional node_type filter
pub fn get_nodes(
    graph: &mut DiGraph<Node, Relation>,
    attribute_key: &str, 
    attribute_value: &str, 
    filter_node_type: Option<&str>
) -> Vec<usize> {
    graph.node_indices().filter_map(|node_index| {
        if let Some(node) = graph.node_weight(node_index) {
            match node {
                Node::StandardNode { node_type, unique_id, attributes, title } => {
                    // Apply node_type filter if provided
                    if let Some(filter_type) = filter_node_type {
                        if node_type != filter_type {
                            return None;
                        }
                    }

                    // Check if the node matches the specified attribute
                    let matches = match attribute_key {
                        "unique_id" => unique_id == attribute_value,
                        "title" => title.as_deref() == Some(attribute_value),
                        _ => attributes.get(attribute_key).map_or(false, |v| v.to_string() == attribute_value),
                    };

                    if matches {
                        Some(node_index.index())  // Return the index of the matching node
                    } else {
                        None
                    }
                },
                // Handle other node variants if necessary
                _ => None,
            }
        } else {
            None
        }
    }).collect()
}

/// Retrieves relationships for specified nodes
pub fn get_relationships(
    graph: &mut DiGraph<Node, Relation>,
    py: Python, 
    indices: Vec<usize>
) -> PyResult<PyObject> {
    let mut incoming_relations = Vec::new();
    let mut outgoing_relations = Vec::new();

    for index in indices {
        let node_index = NodeIndex::new(index);

        // Iterate over incoming and outgoing edges and collect unique relation types
        for direction in &[Direction::Incoming, Direction::Outgoing] {
            for edge in graph.edges_directed(node_index, *direction) {
                let relation_type = &edge.weight().relation_type;
                match direction {
                    Direction::Incoming => {
                        if !incoming_relations.contains(relation_type) {
                            incoming_relations.push(relation_type.clone());
                        }
                    },
                    Direction::Outgoing => {
                        if !outgoing_relations.contains(relation_type) {
                            outgoing_relations.push(relation_type.clone());
                        }
                    },
                }
            }
        }
    }

    // Prepare the Python dictionary with consolidated lists
    let result = PyDict::new(py);
    result.set_item("incoming", incoming_relations)?;
    result.set_item("outgoing", outgoing_relations)?;

    Ok(result.into())
}


pub fn traverse_nodes(
    graph: &DiGraph<Node, Relation>,
    indices: Vec<usize>,
    relationship_type: String,
    is_incoming: bool,
    sort_attribute: Option<&str>,
    ascending: Option<bool>,
    max_relations: Option<usize>,
) -> Vec<usize> {
    let mut final_nodes: Vec<usize> = Vec::new();
    let direction = if is_incoming { Direction::Incoming } else { Direction::Outgoing };

    for index in indices {
        let node_index = NodeIndex::new(index);
        let mut nodes_with_attrs: Vec<(usize, Option<AttributeValue>)> = Vec::new();

        for edge in graph.edges_directed(node_index, direction).filter(|edge| edge.weight().relation_type == relationship_type) {
            let target_node_index = if is_incoming { edge.source() } else { edge.target() };
            let target_node = graph.node_weight(target_node_index).expect("Node must exist");

            if let Node::StandardNode { attributes, .. } = target_node {
                let attr_value = sort_attribute.and_then(|attr| attributes.get(attr).cloned());
                nodes_with_attrs.push((target_node_index.index(), attr_value));
            }
        }

        // If sorting is enabled, sort the nodes based on the attribute value.
        let mut sorted_or_filtered_nodes = if let Some(_attr) = sort_attribute {
            sort_nodes_by_attribute(nodes_with_attrs, ascending.unwrap_or(true))
        } else {
            // If sorting is not enabled, simply collect the node indices without sorting.
            nodes_with_attrs.into_iter().map(|(idx, _)| idx).collect::<Vec<_>>()
        };

        // Limit the number of nodes based on `max_relations` after sorting or filtering.
        if let Some(max) = max_relations {
            sorted_or_filtered_nodes.truncate(max);
        }

        final_nodes.extend(sorted_or_filtered_nodes);
    }

    final_nodes
}

fn sort_nodes_by_attribute(nodes_with_attrs: Vec<(usize, Option<AttributeValue>)>, ascending: bool) -> Vec<usize> {
    let mut sorted_nodes = nodes_with_attrs;

    // Sort based on the attribute value, handling different types of AttributeValue
    sorted_nodes.sort_by(|a, b| {
        match (&a.1, &b.1) {
            (Some(AttributeValue::Int(a_val)), Some(AttributeValue::Int(b_val))) => a_val.cmp(b_val),
            (Some(AttributeValue::Float(a_val)), Some(AttributeValue::Float(b_val))) => a_val.partial_cmp(b_val).unwrap_or(std::cmp::Ordering::Equal),
            (Some(AttributeValue::DateTime(a_val)), Some(AttributeValue::DateTime(b_val))) => a_val.cmp(b_val),
            (Some(AttributeValue::String(a_val)), Some(AttributeValue::String(b_val))) => a_val.cmp(b_val),
            _ => std::cmp::Ordering::Equal, // If no attribute or non-comparable types, consider them equal
        }
    });

    // Reverse the order if descending
    if !ascending {
        sorted_nodes.reverse();
    }

    // Return the sorted node indices
    sorted_nodes.into_iter().map(|(idx, _)| idx).collect()
}