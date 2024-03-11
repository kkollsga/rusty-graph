use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::node::Node;
use crate::relation::Relation;
use petgraph::graph::DiGraph;
use std::collections::HashMap;

mod add_nodes;
mod add_relationships;

#[pyclass]
pub struct KnowledgeGraph {
    pub graph: DiGraph<Node, Relation>,
}

#[pymethods]
impl KnowledgeGraph {
    #[new]
    pub fn new() -> Self {
        KnowledgeGraph {
            graph: DiGraph::new(),
        }
    }

    // Method to add a single node
    pub fn add_node(&mut self, node_type: String, unique_id: String, node_title: String, attributes: HashMap<String, String>) -> usize {
        let node = Node::new(&node_type, &unique_id, &node_title, attributes);
        let index = self.graph.add_node(node);
        index.index() // Convert NodeIndex to usize before returning
    }

    // Wrap the add_nodes function from the methods module
    pub fn add_nodes(
        &mut self,
        data: &PyList, // 2D list of string data
        columns: Vec<String>, // Column names
        node_type: String,
        unique_id_field: String,
        node_title_field: String,
        conflict_handling: String,
    ) -> PyResult<Vec<usize>> {
        add_nodes::add_nodes(
            &mut self.graph, 
            data,
            columns,
            node_type,
            unique_id_field,
            node_title_field,
            conflict_handling,
        ) // Call the standalone function
    }

    // Within the KnowledgeGraph impl block
    pub fn add_relationships(
        &mut self,
        data: &PyList, // 2D list of string data
        columns: Vec<String>, // Column names
        relationship_type: String,
        source_type: String,
        source_id_field: String,
        source_title_field: String,
        target_type: String,        
        target_id_field: String,
        target_title_field: String,
        conflict_handling: String,
    ) -> PyResult<Vec<(usize, usize)>> {
        // Call the updated add_relationships function with the new parameters
        add_relationships::add_relationships(
            &mut self.graph,
            data,
            columns,
            relationship_type,
            source_type,
            source_id_field,
            source_title_field,
            target_type,            
            target_id_field,
            target_title_field,
            conflict_handling,
        )
    }
    
    // Method to retrieve nodes by their unique ID, with an optional node_type filter
    pub fn get_nodes(&self, attribute_key: &str, attribute_value: &str, filter_node_type: Option<&str>) -> Vec<usize> {
        self.graph.node_indices().filter_map(|node_index| {
            let node = &self.graph[node_index];
            
            // Apply node_type filter if provided
            if let Some(filter_type) = filter_node_type {
                if &node.node_type != filter_type {
                    return None;
                }
            }
    
            // Check if the node matches the specified attribute
            let matches = match attribute_key {
                "unique_id" => &node.unique_id == attribute_value,
                "title" => &node.title == attribute_value,
                _ => node.attributes.get(attribute_key).map(String::as_str) == Some(attribute_value),
            };
    
            if matches {
                Some(node_index.index())  // Return the index of the matching node
            } else {
                None
            }
        }).collect()
    }
    
    
    

    // Additional methods as needed...
}
