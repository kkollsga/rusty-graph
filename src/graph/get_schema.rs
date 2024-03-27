use petgraph::graph::DiGraph;
use std::collections::{HashMap, hash_map::Entry};
use crate::schema::{Node, Relation};  // Import the Node enum
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Updates or retrieves the schema (DataTypeNode) from the graph
///
/// # Arguments
///
/// * `graph` - The graph object containing all nodes and relations
/// * `data_type` - The type of the data as a string (e.g., "Node" or "Relation")
/// * `name` - The name of the DataTypeNode
/// * `columns` - Optional list of columns to update in the DataTypeNode
/// * `column_types` - Optional mapping of column names to their data types
pub fn update_or_retrieve_schema(
    graph: &mut DiGraph<Node, Relation>,
    data_type: &str,
    name: &str,
    columns: Option<Vec<String>>,
    column_types: Option<HashMap<String, String>>,
) -> PyResult<HashMap<String, String>> {
    // Find the existing DataTypeNode or create a new one if it doesn't exist
    let data_type_node_index = match graph.node_indices().find(|&i| {
        if let Node::DataTypeNode { data_type: dt, name: n, .. } = &graph[i] {
            return dt.as_str() == data_type && n == name;
        }
        false
    }) {
        Some(index) => index,
        None => {
            let new_data_type_node = Node::new_data_type(data_type, name, HashMap::new());
            graph.add_node(new_data_type_node)
        },
    };

    // If columns are provided, update the DataTypeNode's attributes
    if let Some(cols) = columns {
        if let Node::DataTypeNode { attributes: attr, .. } = &mut graph[data_type_node_index] {
            for column in cols.iter() {
                let column_data_type = column_types
                    .as_ref()
                    .and_then(|ct| ct.get(column))
                    .unwrap_or(&"String".to_string())
                    .clone();

                match attr.entry(column.clone()) {
                    Entry::Occupied(entry) if entry.get() != &column_data_type => {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "Data type conflict for attribute '{}': existing type '{}', new type '{}'",
                            column,
                            entry.get(),
                            column_data_type
                        )));
                    },
                    Entry::Vacant(entry) => {
                        entry.insert(column_data_type);
                    },
                    _ => (),
                }
            }
        }
    }

    // Return the updated attributes HashMap
    if let Node::DataTypeNode { attributes: attr, .. } = &graph[data_type_node_index] {
        Ok(attr.clone())
    } else {
        Err(PyErr::new::<PyValueError, _>("Failed to retrieve or update DataTypeNode"))
    }
}

pub fn retrieve_schema(
    graph: &DiGraph<Node, Relation>,  // Use immutable borrow
    data_type: &str,
    name: &str,
) -> PyResult<HashMap<String, String>> {
    // Find the existing DataTypeNode
    let data_type_node_index = graph.node_indices().find(|&i| {
        if let Node::DataTypeNode { data_type: dt, name: n, .. } = &graph[i] {
            return dt.as_str() == data_type && n == name;
        }
        false
    }).ok_or_else(|| {
        PyErr::new::<PyValueError, _>(format!(
            "DataTypeNode with data_type '{}' and name '{}' not found",
            data_type, name
        ))
    })?;

    // Return the attributes HashMap if the DataTypeNode is found
    if let Node::DataTypeNode { attributes: attr, .. } = &graph[data_type_node_index] {
        Ok(attr.clone())
    } else {
        Err(PyErr::new::<PyValueError, _>("Failed to retrieve DataTypeNode"))
    }
}