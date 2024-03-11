use pyo3::prelude::*;
use pyo3::types::PyList;
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use crate::node::Node;
use crate::relation::Relation;
pub fn add_nodes(
    graph: &mut DiGraph<Node, Relation>,
    data: &PyList,  // 2D list where each inner list represents a row
    columns: Vec<String>,  // Column header names
    node_type: String,
    unique_id_field: String,
    node_title_field: String,
    conflict_handling: String,
) -> PyResult<Vec<usize>> {
    let mut indices = Vec::new();
    // Initialize indices as None
    let mut unique_id_index: Option<usize> = None;
    let mut node_title_index: Option<usize> = None;
    let mut attribute_indexes: Vec<usize> = Vec::new();
    // Iterate once through columns to set unique id indices and attribute indices
    for (index, col_name) in columns.iter().enumerate() {
        match col_name.as_str() {
            _ if col_name == &unique_id_field => unique_id_index = Some(index),
            _ if col_name == &node_title_field => node_title_index = Some(index),
            _ => attribute_indexes.push(index), // Any column that is not a unique id is an attribute
        }
    }
    // Ensure we found both unique ID columns
    let unique_id_index = unique_id_index.ok_or_else(|| 
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("'{}' column not found", unique_id_field)))?;
    let node_title_index = node_title_index.ok_or_else(|| 
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("'{}' column not found", node_title_field)))?;
    

    for py_row in data {
        let row: Vec<String> = py_row.extract()?;
        // Attempt to parse the unique_id as a float and truncate it if successful; otherwise, keep the original string
        let unique_id = row[unique_id_index].parse::<f64>()
            .map(|num| num.trunc().to_string())  // Convert to integer string if parse is successful
            .unwrap_or_else(|_| row[unique_id_index].clone());  // Keep original string if parse fails
        let node_title = &row[node_title_index];
    
        // Construct relation attributes using the attribute_indexes
        let attributes: HashMap<String, String> = attribute_indexes.iter()
            .map(|&index| (columns[index].clone(), row[index].clone()))
            .collect();

        // Search for an existing node with the given unique_id and node_type
        let existing_node_index = graph.node_indices().find(|&i| {
            graph[i].node_type == node_type && graph[i].unique_id == unique_id
        });

        match existing_node_index {
            Some(node_index) => match conflict_handling.as_str() {
                "replace" => {
                    // Replace the existing node with a new one
                    graph[node_index] = Node::new(&node_type, &unique_id, &node_title, attributes);
                    indices.push(node_index.index());
                },
                "update" => {
                    // Update the existing node's attributes
                    let node = &mut graph[node_index];
                    node.attributes.extend(attributes);
                    indices.push(node_index.index());
                },
                "skip" => {
                    // Skip adding a new node if it already exists
                    indices.push(node_index.index());
                },
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid conflict_handling value"
                )),
            },
            None => {
                // Add a new node if it does not exist
                let node = Node::new(&node_type, &unique_id, &node_title, attributes);
                let index = graph.add_node(node);
                indices.push(index.index());
            },
        }
    }

    Ok(indices)
}