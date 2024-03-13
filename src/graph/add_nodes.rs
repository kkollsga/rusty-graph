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
    node_title_field: Option<String>,
    conflict_handling: Option<String>,
) -> PyResult<Vec<usize>> {
    let mut indices = Vec::new();
    // Default handling for optional parameters
    let conflict_handling = conflict_handling.unwrap_or_else(|| "update".to_string());
    // Initialize indices as None
    let mut unique_id_index: Option<usize> = None;
    let mut node_title_index: Option<usize> = None;
    let mut attribute_indexes: Vec<usize> = Vec::new();
    // Iterate once through columns to set unique id indices and attribute indices
    for (index, col_name) in columns.iter().enumerate() {
        if col_name == &unique_id_field {
            unique_id_index = Some(index);
        } else if node_title_field.as_ref().map_or(false, |ntf| col_name == ntf) {
            node_title_index = Some(index);
        } else {
            attribute_indexes.push(index); // Any column that is not a unique id or title is an attribute
        }
    }
    // Ensure we found both unique ID columns
    let unique_id_index = unique_id_index.ok_or_else(|| 
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("'{}' column not found", unique_id_field)))?;
    

    for py_row in data {
        let row: Vec<String> = py_row.extract()?;
        // Attempt to parse the unique_id as a float and truncate it if successful; otherwise, keep the original string
        let unique_id = row[unique_id_index].parse::<f64>()
            .map(|num| num.trunc().to_string())  // Convert to integer string if parse is successful
            .unwrap_or_else(|_| row[unique_id_index].clone());  // Keep original string if parse fails
        // Get the node title if the title index is defined, otherwise use None
        let node_title = node_title_index.map(|index| row[index].clone());
    
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
                    graph[node_index] = Node::new(&node_type, &unique_id, attributes, node_title.as_deref());
                    indices.push(node_index.index());
                },
                "update" => {
                    let node = &mut graph[node_index];
                    node.attributes.extend(attributes);
                    indices.push(node_index.index());
                },
                "skip" => {
                    indices.push(node_index.index());
                },
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid conflict_handling value"
                )),
            },
            None => {
                let node = Node::new(&node_type, &unique_id, attributes, node_title.as_deref());
                let index = graph.add_node(node);
                indices.push(index.index());
            },
        }
    }

    Ok(indices)
}