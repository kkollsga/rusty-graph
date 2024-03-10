use pyo3::prelude::*;
use pyo3::types::PyList;
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use crate::node::Node;
use crate::relation::Relation;

pub fn add_relationships(
    graph: &mut DiGraph<Node, Relation>,
    data: &PyList,  // 2D list where each inner list represents a row
    columns: Vec<String>,  // Column header names
    relationship_name: String,  // Configuration items directly in the function call
    left_node_type: String,
    left_unique_id_field: String,
    right_node_type: String,
    right_unique_id_field: String,
    conflict_handling: String,  // Default value handled inside function if necessary
) -> PyResult<Vec<(usize, usize)>> {
    let mut indices = Vec::new();

    // Create lookup tables for left and right nodes
    let mut left_node_lookup = HashMap::new();
    let mut right_node_lookup = HashMap::new();
    // Populate the lookup tables by filtering nodes based on type
    for index in graph.node_indices() {
        let node = &graph[index];
        if node.node_type == left_node_type {
            left_node_lookup.insert(node.unique_id.clone(), index);
        } else if node.node_type == right_node_type {
            right_node_lookup.insert(node.unique_id.clone(), index);
        }
    }

    // Find indices for unique ID fields
    let left_unique_id_index = columns.iter().position(|col| col == &left_unique_id_field)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("'{}' column not found", left_unique_id_field)
        ))?;
    let right_unique_id_index = columns.iter().position(|col| col == &right_unique_id_field)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("'{}' column not found", right_unique_id_field)
        ))?;

    // Determine attribute indexes, excluding left and right unique ID indexes
    let attribute_indexes: Vec<usize> = columns.iter().enumerate()
        .filter_map(|(index, _)| {
            if index != left_unique_id_index && index != right_unique_id_index {
                Some(index)
            } else {
                None
            }
        })
        .collect();

    for py_row in data {
        let row: Vec<String> = py_row.extract()?;
        // Process left_unique_id
        let left_unique_id = row[left_unique_id_index].parse::<f64>()
            .map(|num| num.trunc().to_string())  // Convert to integer string if parse is successful
            .unwrap_or_else(|_| row[left_unique_id_index].clone());  // Keep original string if parse fails

        // Process right_unique_id
        let right_unique_id = row[right_unique_id_index].parse::<f64>()
            .map(|num| num.trunc().to_string())  // Convert to integer string if parse is successful
            .unwrap_or_else(|_| row[right_unique_id_index].clone());  // Keep original string if parse fails

        // Find or create left node
        let left_node_index = *left_node_lookup.entry(left_unique_id.clone())
            .or_insert_with(|| {
                // Create a new Node by passing string slices instead of owned String objects
                let node = Node::new(&left_node_type, &left_unique_id, HashMap::new());
                graph.add_node(node)
            });
        // Find or create right node
        let right_node_index = *right_node_lookup.entry(right_unique_id.clone())
            .or_insert_with(|| {
                // Same here, pass string slices to the new Node
                let node = Node::new(&right_node_type, &right_unique_id, HashMap::new());
                graph.add_node(node)
            });
        

        // Construct relation attributes using the attribute_indexes
        let attributes: HashMap<String, String> = attribute_indexes.iter()
            .map(|&index| {
                // For each attribute index, get the column name and value from the row
                let attribute_name = columns[index].clone();
                let attribute_value = row[index].clone();
                (attribute_name, attribute_value)
            })
            .collect();

        match conflict_handling.as_str() {
            "update" => {
                if let Some(edge_id) = graph.find_edge(left_node_index, right_node_index) {
                    let existing_relation = &mut graph[edge_id];
                    for (key, value) in attributes.iter() {
                        existing_relation.attributes.insert(key.clone(), value.clone());
                    }
                } else {
                    graph.add_edge(left_node_index, right_node_index, Relation {
                        relation_type: relationship_name.clone(),
                        attributes,
                    });
                }
            },
            "replace" => {
                let edges: Vec<_> = graph.edges_connecting(left_node_index, right_node_index).map(|e| e.id()).collect();
                for edge_id in edges {
                    graph.remove_edge(edge_id);
                }
                graph.add_edge(left_node_index, right_node_index, Relation {
                    relation_type: relationship_name.clone(),
                    attributes,
                });
            },
            "skip" => {
                if graph.find_edge(left_node_index, right_node_index).is_none() {
                    graph.add_edge(left_node_index, right_node_index, Relation {
                        relation_type: relationship_name.clone(),
                        attributes,
                    });
                }
            },
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid 'conflict_handling' value")),
        }

        indices.push((left_node_index.index(), right_node_index.index()));
    }

    Ok(indices)
}
