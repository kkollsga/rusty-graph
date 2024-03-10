use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::node::Node;
use crate::relation::Relation;
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
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
    pub fn add_node(&mut self, node_type: String, unique_id: String, attributes: HashMap<String, String>) -> usize {
        let node = Node::new(&node_type, &unique_id, attributes);
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
        conflict_handling: String,
    ) -> PyResult<Vec<usize>> {
        add_nodes::add_nodes(
            &mut self.graph, 
            data,
            columns,
            node_type,
            unique_id_field,
            conflict_handling,
        ) // Call the standalone function
    }

    // Within the KnowledgeGraph impl block
    pub fn add_relationships(
        &mut self,
        data: &PyList, // 2D list of string data
        columns: Vec<String>, // Column names
        relationship_name: String,
        left_node_type: String,
        left_unique_id_field: String,
        right_node_type: String,        
        right_unique_id_field: String,
        conflict_handling: String,
    ) -> PyResult<Vec<(usize, usize)>> {
        // Call the updated add_relationships function with the new parameters
        add_relationships::add_relationships(
            &mut self.graph,
            data,
            columns,
            relationship_name,
            left_node_type,
            left_unique_id_field,
            right_node_type,            
            right_unique_id_field,
            conflict_handling,
        )
    }
    
    // Method to retrieve nodes by their unique ID, with an optional node_type filter
    pub fn get_nodes_by_id(&self, unique_id: &str, filter_node_type: Option<&str>) -> Vec<HashMap<String, Vec<HashMap<String, String>>>> {
        self.graph.node_indices().filter_map(|node_index| {
            let node = &self.graph[node_index];
    
            // Apply node_type filter if provided
            if let Some(filter_type) = filter_node_type {
                if &node.node_type != filter_type {
                    return None;
                }
            }
    
            if &node.unique_id == unique_id {
                // Node attributes including node_type and unique_id
                let mut attributes = HashMap::new();
                attributes.insert("node_type".to_string(), node.node_type.clone());
                attributes.insert("unique_id".to_string(), node.unique_id.clone());
                for (key, value) in &node.attributes {
                    attributes.insert(key.clone(), value.clone());
                }
    
                // Outgoing relationships
                let relationships: Vec<HashMap<String, String>> = self.graph.edges(node_index).map(|edge| {
                    let mut relationship_data = HashMap::new();
                    relationship_data.insert("relation_type".to_string(), edge.weight().relation_type.clone());
                    for (key, value) in &edge.weight().attributes {
                        relationship_data.insert(key.clone(), value.clone());
                    }
                    let target_node_index = edge.target();
                    let target_node = &self.graph[target_node_index];
                    relationship_data.insert("target_unique_id".to_string(), target_node.unique_id.clone());
                    relationship_data.insert("target_type".to_string(), target_node.node_type.clone());
                    relationship_data
                }).collect();
    
                let mut node_data = HashMap::new();
                node_data.insert("attributes".to_string(), vec![attributes]);
                node_data.insert("outgoing_connections".to_string(), relationships);
    
                Some(node_data)
            } else {
                None
            }
        }).collect()
    }
    


    // Additional methods as needed...
}
