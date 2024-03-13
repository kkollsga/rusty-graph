use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use crate::node::Node;
use crate::relation::Relation;
use crate::utils::{convert_column, DataType};
pub fn add_nodes(
    graph: &mut DiGraph<Node, Relation>,
    data: &PyList,  // 2D list where each inner list represents a row
    columns: Vec<String>,  // Column header names
    node_type: String,
    unique_id_field: String,
    node_title_field: Option<String>,
    conflict_handling: Option<String>,
    column_types: Option<&PyDict>,
) -> PyResult<Vec<usize>> {
    let mut indices = Vec::new();
    let conflict_handling = conflict_handling.unwrap_or_else(|| "update".to_string());
    // Find indices for unique ID and node title fields
    let unique_id_index = columns.iter().position(|c| c == &unique_id_field)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("'{}' column not found", unique_id_field)))?;
    let node_title_index = node_title_field.as_ref().and_then(|ntf| columns.iter().position(|c| c == ntf));

    // Indices for attributes (excluding unique ID and node title fields)
    let attribute_indexes: Vec<usize> = columns.iter().enumerate()
        .filter_map(|(i, _)| {
            if i != unique_id_index && node_title_index != Some(i) { Some(i) } else { None }
        })
        .collect();

    // Parse column types mapping from Python dictionary
    let column_types_map: HashMap<String, DataType> = column_types.map_or_else(HashMap::new, |ct| {
        ct.into_iter().filter_map(|(k, v)| {
            let key = k.extract::<String>().ok()?;
            let value = v.extract::<String>().ok()?;
            let data_type = match value.as_str() {
                "int" => DataType::Int,
                "float" => DataType::Float,
                "datetime" => DataType::DateTime,
                _ => return None,
            };
            Some((key, data_type))
        }).collect()
    });

    // Step 1: Extract each column into a Rust vector of strings
    let mut columns_data: HashMap<String, Vec<String>> = HashMap::new();
    for (index, column_name) in columns.iter().enumerate() {
        let py_column: &PyList = data.get_item(index)?.extract()?;
        let column_data: Vec<String> = py_column.into_iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        columns_data.insert(column_name.clone(), column_data);
    }

    // Step 2 & 3: Convert the data in corresponding columns
    for (column_name, data_type) in column_types_map.iter() {
        if let Some(column_data) = columns_data.get_mut(column_name) {
            let converted_column = convert_column(column_data.clone(), data_type.clone())?;
            *column_data = converted_column; // Update the original column data with the converted values
        }
    }
    // Determine the number of rows
    let num_rows = data.get_item(unique_id_index)?.extract::<&PyList>()?.len();
    // Iterate over each row
    // Iterate over each row
    for row_index in 0..num_rows {
        // Extract unique ID and node title for the current row
        let unique_id: String = columns_data.get(&unique_id_field)
            .and_then(|col| col.get(row_index))
            .cloned()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Unique ID column value missing"))?;

        let node_title: Option<String> = node_title_index
            .and_then(|index| columns.get(index))
            .and_then(|title_col| columns_data.get(title_col))
            .and_then(|col| col.get(row_index))
            .cloned();

        // Construct relation attributes for the current row using attribute_indexes
        let attributes: HashMap<String, String> = attribute_indexes.iter()
            .filter_map(|&index| {
                let column_name = &columns[index];
                columns_data.get(column_name)
                    .and_then(|col| col.get(row_index))
                    .map(|value| (column_name.clone(), value.clone()))
            })
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
