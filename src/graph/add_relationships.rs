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
    source_id_field: String,
    target_type: String,
    target_id_field: String,
    source_title_field: Option<String>,
    target_title_field: Option<String>,
    conflict_handling: Option<String>,  // Default value handled inside function if necessary
) -> PyResult<Vec<(usize, usize)>> {
    let mut indices = Vec::new();
    // let conflict_handling = conflict_handling.unwrap_or_else(|| "update".to_string());

    // Create lookup tables for source and target nodes
    let mut source_node_lookup = HashMap::new();
    let mut target_node_lookup = HashMap::new();

    // Populate the lookup tables by filtering nodes based on type
    for index in graph.node_indices() {
        if let Some(node) = graph.node_weight(index) {
            match node {
                Node::StandardNode { node_type, unique_id, .. } => {
                    if node_type == &source_type {
                        source_node_lookup.insert(unique_id.clone(), index);
                    } else if node_type == &target_type {
                        target_node_lookup.insert(unique_id.clone(), index);
                    }
                },
                // Optionally handle DataTypeNode or other variants if needed
                _ => {}
            }
        }
    }

    // Iterate over each row in the data
    for row in data.iter() {
        let row: Vec<&PyAny> = row.extract()?;
        let row_data: HashMap<_, _> = columns.iter().zip(row.iter()).collect();

        let source_unique_id = row_data.get(&source_id_field)
            .and_then(|&item| item.extract::<String>().ok())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Source ID column '{}' value missing", source_id_field)))?;

        let target_unique_id = row_data.get(&target_id_field)
            .and_then(|&item| item.extract::<String>().ok())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Target ID column '{}' value missing", target_id_field)))?;

        // Optionally extract source and target titles
        let source_title = source_title_field.as_ref().and_then(|field| row_data.get(field).and_then(|&item| item.extract::<String>().ok()));
        let target_title = target_title_field.as_ref().and_then(|field| row_data.get(field).and_then(|&item| item.extract::<String>().ok()));

        // Find or create source and target nodes
        let source_node_index = find_or_create_node(graph, &source_type, &source_unique_id, source_title.clone(), &mut source_node_lookup);
        let target_node_index = find_or_create_node(graph, &target_type, &target_unique_id, target_title.clone(), &mut target_node_lookup);

        // Construct and add the relationship
        let relation = Relation::new(&relationship_type, None);  // Construct a Relation instance, attributes can be added as needed
        let edge = graph.add_edge(source_node_index, target_node_index, relation);

        indices.push((source_node_index.index(), target_node_index.index()));
    }

    Ok(indices)
}

// Helper function to find or create a node
fn find_or_create_node(
    graph: &mut DiGraph<Node, Relation>,
    node_type: &str,
    unique_id: &str,
    title: Option<String>,
    node_lookup: &mut HashMap<String, petgraph::graph::NodeIndex>,  // Note: Changed to mutable reference
) -> petgraph::graph::NodeIndex {
    // Try to get the node index from the lookup table
    if let Some(index) = node_lookup.get(unique_id) {
        *index  // If found, return a cloned value of the reference
    } else {
        // If not found, create a new node and add it to the graph
        let new_node = Node::new(node_type, unique_id, None, title.as_deref());  // Ensure this matches your Node creation logic
        let index = graph.add_node(new_node);
        
        // Insert the new node's index into the lookup table for future reference
        node_lookup.insert(unique_id.to_string(), index);
        
        index  // Return the new node's index
    }
}
