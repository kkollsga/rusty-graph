use pyo3::prelude::*;
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use pyo3::types::PyDict;
use pyo3::PyResult;
use crate::node::Node;
use crate::relation::Relation;
use std::collections::HashSet;

/// Method to retrieve nodes by their unique ID, with an optional node_type filter
pub fn get_nodes(
    graph: &mut DiGraph<Node, Relation>,
    attribute_key: &str, 
    attribute_value: &str, 
    filter_node_type: Option<&str>
) -> Vec<usize> {
    graph.node_indices().filter_map(|node_index| {
        let node = &graph[node_index];
        
        // Apply node_type filter if provided
        if let Some(filter_type) = filter_node_type {
            if &node.node_type != filter_type {
                return None;
            }
        }
        // Check if the node matches the specified attribute
        let matches = match attribute_key {
            "unique_id" => &node.unique_id == attribute_value,
            "title" => node.title.as_deref() == Some(attribute_value),  // Adjusted comparison
            _ => node.attributes.get(attribute_key).map(String::as_str) == Some(attribute_value),
        };
        if matches {
            Some(node_index.index())  // Return the index of the matching node
        } else {
            None
        }
    }).collect()
}

pub fn get_relationships(
    graph: &mut DiGraph<Node, Relation>,
    py: Python, 
    indices: Vec<usize>
) -> PyResult<PyObject> {
    let mut incoming_relations = Vec::new();
    let mut outgoing_relations = Vec::new();

    for index in indices {
        let node_index = petgraph::graph::NodeIndex::new(index);

        // Iterate over incoming edges
        for edge in graph.edges_directed(node_index, petgraph::Direction::Incoming) {
            let relation_type = &edge.weight().relation_type;
            // Use `relation_type` to identify unique relationships
            if !incoming_relations.contains(relation_type) {
                incoming_relations.push(relation_type.clone());
            }
        }
        // Iterate over outgoing edges
        for edge in graph.edges_directed(node_index, petgraph::Direction::Outgoing) {
            let relation_type = &edge.weight().relation_type;
            // Use `relation_type` to identify unique relationships
            if !outgoing_relations.contains(relation_type) {
                outgoing_relations.push(relation_type.clone());
            }
        }
    }

    // Prepare the Python dictionary with consolidated lists
    let result = PyDict::new(py);
    result.set_item("incoming", incoming_relations)?;
    result.set_item("outgoing", outgoing_relations)?;

    Ok(result.into())
}

// Adjusted private method to use a boolean flag for direction
pub fn traverse_nodes(
    graph: &DiGraph<Node, Relation>,
    indices: Vec<usize>, 
    relationship_type: String, 
    is_incoming: bool
) -> Vec<usize> {
    let mut related_nodes_set = HashSet::new(); // Use a HashSet to ensure uniqueness
    let direction = if is_incoming {
        petgraph::Direction::Incoming
    } else {
        petgraph::Direction::Outgoing
    };
    for index in indices {
        let node_index = petgraph::graph::NodeIndex::new(index);
        let edges = graph.edges_directed(node_index, direction)
            .filter(|edge| edge.weight().relation_type == relationship_type);

        for edge in edges {
            let related_node_index = if is_incoming {
                edge.source()
            } else {
                edge.target()
            };

            // Add the index to the HashSet, which automatically ensures uniqueness
            related_nodes_set.insert(related_node_index.index());
        }
    }
    // Convert the HashSet to a Vec before returning
    related_nodes_set.into_iter().collect()
}