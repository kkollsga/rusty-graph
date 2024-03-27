use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
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
        let node_index = petgraph::graph::NodeIndex::new(index);

        // Iterate over incoming and outgoing edges and collect unique relation types
        for direction in &[petgraph::Direction::Incoming, petgraph::Direction::Outgoing] {
            for edge in graph.edges_directed(node_index, *direction) {
                let relation_type = &edge.weight().relation_type;
                match direction {
                    petgraph::Direction::Incoming => {
                        if !incoming_relations.contains(relation_type) {
                            incoming_relations.push(relation_type.clone());
                        }
                    },
                    petgraph::Direction::Outgoing => {
                        if !outgoing_relations.contains(relation_type) {
                            outgoing_relations.push(relation_type.clone());
                        }
                    },
                    _ => {}
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

/// Traverses nodes in a specified direction based on relationship type
pub fn traverse_nodes(
    graph: &DiGraph<Node, Relation>,
    indices: Vec<usize>,
    relationship_type: String,
    is_incoming: bool,
    sort_attribute: Option<&str>, 
    ascending: Option<bool>, 
    max_relations: Option<usize>
) -> Vec<usize> {
    let mut related_nodes = Vec::new();
    let direction = if is_incoming { petgraph::Direction::Incoming } else { petgraph::Direction::Outgoing };

    for index in indices {
        let node_index = petgraph::graph::NodeIndex::new(index);
        let mut edges: Vec<_> = graph.edges_directed(node_index, direction)
            .filter(|edge| edge.weight().relation_type == relationship_type)
            .collect();

        // Optional sorting based on a specified attribute
        if let Some(attr) = sort_attribute {
            edges.sort_by(|a, b| {
                let a_attr_value = a.weight().attributes.as_ref()
                    .and_then(|attrs| attrs.get(attr))
                    .map(|v| v.to_string());  // Keep this as a String

                let b_attr_value = b.weight().attributes.as_ref()
                    .and_then(|attrs| attrs.get(attr))
                    .map(|v| v.to_string());  // Keep this as a String

                let order = ascending.unwrap_or(true);
                if order {
                    a_attr_value.cmp(&b_attr_value)
                } else {
                    b_attr_value.cmp(&a_attr_value)
                }
            });
        }

        // Process edges up to the max_relations limit (if specified)
        for edge in edges.iter().take(max_relations.unwrap_or(usize::MAX)) {
            let related_node_index = if is_incoming { edge.source() } else { edge.target() };
            // Ensure we don't add duplicates
            if !related_nodes.contains(&related_node_index.index()) {
                related_nodes.push(related_node_index.index());
            }
        }
    }

    related_nodes
}
