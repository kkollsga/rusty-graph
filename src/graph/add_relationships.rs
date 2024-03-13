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
    relationship_type: String,  // Configuration items directly in the function call
    source_type: String,
    source_id_field: String,
    target_type: String,
    target_id_field: String,
    source_title_field: Option<String>,
    target_title_field: Option<String>,
    conflict_handling: Option<String>,  // Default value handled inside function if necessary
) -> PyResult<Vec<(usize, usize)>> {
    let mut indices = Vec::new();
    // Default handling for optional parameters
    let conflict_handling = conflict_handling.unwrap_or_else(|| "update".to_string());

    // Step 1: Extract each column into a Rust vector of strings
    let mut columns_data: HashMap<String, Vec<String>> = HashMap::new();
    for (index, column_name) in columns.iter().enumerate() {
        let py_column: &PyList = data.get_item(index)?.extract()?;
        let column_data: Vec<String> = py_column.into_iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?;
        columns_data.insert(column_name.clone(), column_data);
    }

    // Create lookup tables for left and right nodes
    let mut source_node_lookup = HashMap::new();
    let mut target_node_lookup = HashMap::new();
    // Populate the lookup tables by filtering nodes based on type
    for index in graph.node_indices() {
        let node = &graph[index];
        if node.node_type == source_type {
            source_node_lookup.insert(node.unique_id.clone(), index);
        } else if node.node_type == target_type {
            target_node_lookup.insert(node.unique_id.clone(), index);
        }
    }

    // Determine the number of rows (all columns should have the same length)
    let num_rows = columns_data.values().next().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("No data provided"))?.len();

    // Iterate over each row
    for row_index in 0..num_rows {
        // Extract source and target IDs for the current row
        let source_unique_id: String = columns_data.get(&source_id_field)
            .and_then(|col| col.get(row_index))
            .cloned()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Source ID column '{}' value missing", source_id_field)))?;
        // Optionally extract source and target titles
        let source_title = source_title_field
            .as_ref()  // Convert Option<String> to Option<&String>
            .and_then(|title_field| columns_data.get(title_field))  // Use the dereferenced title_field
            .and_then(|col| col.get(row_index))
            .cloned();
        // Find or create left node
        let source_node_index = *source_node_lookup.entry(source_unique_id.clone())
            .or_insert_with(|| {
                // Create a new Node by passing string slices instead of owned String objects
                let node = Node::new(&source_type, &source_unique_id, HashMap::new(), source_title.as_deref());
                graph.add_node(node)
            });

        let target_unique_id: String = columns_data.get(&target_id_field)
            .and_then(|col| col.get(row_index))
            .cloned()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Target ID column '{}' value missing", target_id_field)))?;
        let target_title = target_title_field
            .as_ref()  // Convert Option<String> to Option<&String>
            .and_then(|title_field| columns_data.get(title_field))  // Use the dereferenced title_field
            .and_then(|col| col.get(row_index))
            .cloned();
        // Find or create target node
        let target_node_index = *target_node_lookup.entry(target_unique_id.clone())
            .or_insert_with(|| {
                // Same here, pass string slices to the new Node
                let node = Node::new(&target_type, &target_unique_id, HashMap::new(), target_title.as_deref());
                graph.add_node(node)
            });
        
        // Construct relation attributes using the attribute_indexes (excluding source and target ID fields)
        let attributes: HashMap<String, String> = columns.iter().enumerate()
            .filter_map(|(_, column_name)| {
                if column_name != &source_id_field && column_name != &target_id_field {
                    columns_data.get(column_name)
                        .and_then(|col| col.get(row_index))
                        .map(|value| (column_name.clone(), value.clone()))
                } else {
                    None
                }
            })
            .collect();

        match conflict_handling.as_str() {
            "update" => {
                if let Some(edge_id) = graph.find_edge(source_node_index, target_node_index) {
                    let existing_relation = &mut graph[edge_id];
                    for (key, value) in attributes.iter() {
                        existing_relation.attributes.insert(key.clone(), value.clone());
                    }
                } else {
                    graph.add_edge(source_node_index, target_node_index, Relation {
                        relation_type: relationship_type.clone(),
                        attributes,
                    });
                }
            },
            "replace" => {
                let edges: Vec<_> = graph.edges_connecting(source_node_index, target_node_index).map(|e| e.id()).collect();
                for edge_id in edges {
                    graph.remove_edge(edge_id);
                }
                graph.add_edge(source_node_index, target_node_index, Relation {
                    relation_type: relationship_type.clone(),
                    attributes,
                });
            },
            "skip" => {
                if graph.find_edge(source_node_index, target_node_index).is_none() {
                    graph.add_edge(source_node_index, target_node_index, Relation {
                        relation_type: relationship_type.clone(),
                        attributes,
                    });
                }
            },
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid 'conflict_handling' value")),
        }

        indices.push((source_node_index.index(), target_node_index.index()));
    }

    Ok(indices)
}
