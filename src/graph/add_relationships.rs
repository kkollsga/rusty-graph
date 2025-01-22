use pyo3::prelude::*;
use pyo3::types::PyList;
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use crate::schema::{Node, Relation};

pub fn add_relationships(
    graph: &mut DiGraph<Node, Relation>,
    data: &PyList,  // 2D list where each inner list represents a row
    columns: Vec<String>,  // Column header names
    relationship_type: String,  // Configuration items directly in the function call
    source_type: String,
    source_id_field: i32,
    target_type: String,
    target_id_field: i32,
    source_title_field: Option<String>,
    target_title_field: Option<String>,
) -> PyResult<Vec<(usize, usize)>> {
    let mut indices = Vec::new();

    // Create lookup tables for source and target nodes
    let mut source_node_lookup: HashMap<i32, petgraph::graph::NodeIndex> = HashMap::new();
    let mut target_node_lookup: HashMap<i32, petgraph::graph::NodeIndex> = HashMap::new();

    // Populate the lookup tables by filtering nodes based on type
    for index in graph.node_indices() {
        if let Some(node) = graph.node_weight(index) {
            match node {
                Node::StandardNode { node_type, unique_id, .. } => {
                    if node_type == &source_type {
                        source_node_lookup.insert(*unique_id, index);
                    } else if node_type == &target_type {
                        target_node_lookup.insert(*unique_id, index);
                    }
                },
                _ => {}
            }
        }
    }

    // Iterate over each row in the data
    for row in data.iter() {
        let row: Vec<&PyAny> = row.extract()?;
        let mut source_unique_id: Option<i32> = None;
        let mut target_unique_id: Option<i32> = None;
        let mut source_title: Option<String> = None;
        let mut target_title: Option<String> = None;

        // Process each column in the row
        for (col_index, column_name) in columns.iter().enumerate() {
            let item = row.get(col_index).unwrap();

            // Handle source ID field
            if let Ok(col_num) = column_name.parse::<i32>() {
                if col_num == source_id_field {
                    source_unique_id = Some(match item.extract::<f64>() {
                        Ok(float_val) => float_val as i32,
                        Err(_) => match item.extract::<i32>() {
                            Ok(int_val) => int_val,
                            Err(_) => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Source ID must be a number"
                            ))
                        }
                    });
                    continue;
                }
                if col_num == target_id_field {
                    target_unique_id = Some(match item.extract::<f64>() {
                        Ok(float_val) => float_val as i32,
                        Err(_) => match item.extract::<i32>() {
                            Ok(int_val) => int_val,
                            Err(_) => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Target ID must be a number"
                            ))
                        }
                    });
                    continue;
                }
            }

            // Handle title fields
            if let Some(ref title_field) = source_title_field {
                if column_name == title_field {
                    source_title = Some(item.extract()?);
                    continue;
                }
            }
            if let Some(ref title_field) = target_title_field {
                if column_name == title_field {
                    target_title = Some(item.extract()?);
                    continue;
                }
            }
        }

        // Extract the IDs we found, returning error if not found
        let source_unique_id = source_unique_id.ok_or_else(|| 
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Source ID field not found")
        )?;
        let target_unique_id = target_unique_id.ok_or_else(|| 
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Target ID field not found")
        )?;

        // Find or create source and target nodes - now passing owned values
        let source_node_index = find_or_create_node(graph, &source_type, source_unique_id, source_title, &mut source_node_lookup);
        let target_node_index = find_or_create_node(graph, &target_type, target_unique_id, target_title, &mut target_node_lookup);

        // Construct and add the relationship
        let relation = Relation::new(&relationship_type, None);
        let _edge = graph.add_edge(source_node_index, target_node_index, relation);

        indices.push((source_node_index.index(), target_node_index.index()));
    }

    Ok(indices)
}

// Helper function to find or create a node - now takes owned i32
fn find_or_create_node(
    graph: &mut DiGraph<Node, Relation>,
    node_type: &str,
    unique_id: i32,
    title: Option<String>,
    node_lookup: &mut HashMap<i32, petgraph::graph::NodeIndex>,
) -> petgraph::graph::NodeIndex {
    if let Some(&index) = node_lookup.get(&unique_id) {
        index
    } else {
        let new_node = Node::new(node_type, unique_id, None, title.as_deref());
        let index = graph.add_node(new_node);
        node_lookup.insert(unique_id, index);
        index
    }
}