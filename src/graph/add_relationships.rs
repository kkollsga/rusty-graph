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

    // Find indices for unique ID fields
    let source_unique_id_index = columns.iter().position(|col| col == &source_id_field)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("'{}' column not found", source_id_field)
        ))?;
    let target_unique_id_index = columns.iter().position(|col| col == &target_id_field)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("'{}' column not found", target_id_field)
        ))?;
    
    // Optionally find indices for title fields
    let source_title_index = source_title_field
        .as_ref()
        .and_then(|field| columns.iter().position(|col| col == field));
    let target_title_index = target_title_field
        .as_ref()
        .and_then(|field| columns.iter().position(|col| col == field));

    // Determine attribute indexes, excluding left and right unique ID indexes
    let attribute_indexes: Vec<usize> = columns.iter().enumerate()
        .filter_map(|(index, _)| {
            if index != source_unique_id_index && index != target_unique_id_index {
                Some(index)
            } else {
                None
            }
        })
        .collect();

    for py_row in data {
        let row: Vec<String> = py_row.extract()?;
        // Process source_unique_id
        let source_unique_id = row[source_unique_id_index].parse::<f64>()
            .map(|num| num.trunc().to_string())  // Convert to integer string if parse is successful
            .unwrap_or_else(|_| row[source_unique_id_index].clone());  // Keep original string if parse fails
        // Process target_unique_id
        let target_unique_id = row[target_unique_id_index].parse::<f64>()
            .map(|num| num.trunc().to_string())  // Convert to integer string if parse is successful
            .unwrap_or_else(|_| row[target_unique_id_index].clone());  // Keep original string if parse fails

        let source_title = source_title_index.map(|index| row[index].clone());
        let target_title = target_title_index.map(|index| row[index].clone());
        
        // Find or create left node
        let source_node_index = *source_node_lookup.entry(source_unique_id.clone())
            .or_insert_with(|| {
                // Create a new Node by passing string slices instead of owned String objects
                let node = Node::new(&source_type, &source_unique_id, HashMap::new(), source_title.as_deref());
                graph.add_node(node)
            });
        // Find or create target node
        let target_node_index = *target_node_lookup.entry(target_unique_id.clone())
            .or_insert_with(|| {
                // Same here, pass string slices to the new Node
                let node = Node::new(&target_type, &target_unique_id, HashMap::new(), target_title.as_deref());
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
