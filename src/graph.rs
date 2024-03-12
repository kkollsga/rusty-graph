use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use pyo3::PyResult;
use crate::node::Node;
use crate::relation::Relation;
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};

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
    
    pub fn get_relationships(&self, py: Python, indices: Vec<usize>) -> PyResult<PyObject> {
        let mut incoming_relations = Vec::new();
        let mut outgoing_relations = Vec::new();

        for index in indices {
            let node_index = petgraph::graph::NodeIndex::new(index);

            // Iterate over incoming edges
            for edge in self.graph.edges_directed(node_index, petgraph::Direction::Incoming) {
                let relation_type = &edge.weight().relation_type;
                // Use `relation_type` to identify unique relationships
                if !incoming_relations.contains(relation_type) {
                    incoming_relations.push(relation_type.clone());
                }
            }

            // Iterate over outgoing edges
            for edge in self.graph.edges_directed(node_index, petgraph::Direction::Outgoing) {
                let relation_type = &edge.weight().relation_type;
                // Use `relation_type` to identify unique relationships
                if !outgoing_relations.contains(relation_type) {
                    outgoing_relations.push(relation_type.clone());
                }
            }
        }

        // Prepare the Python dictionary with consolidated lists
        let result = PyDict::new(py);
        result.set_item("incoming", incoming_relations)?;
        result.set_item("outgoing", outgoing_relations)?;

        Ok(result.into())
    }
    
    pub fn traverse_incoming(&self, indices: Vec<usize>, relationship_type: String) -> Vec<usize> {
        self.traverse_nodes(indices, relationship_type, true)
    }

    // Public method for traversing outgoing relationships
    pub fn traverse_outgoing(&self, indices: Vec<usize>, relationship_type: String) -> Vec<usize> {
        self.traverse_nodes(indices, relationship_type, false)
    }

    // Adjusted private method to use a boolean flag for direction
    fn traverse_nodes(&self, indices: Vec<usize>, relationship_type: String, is_incoming: bool) -> Vec<usize> {
        let mut related_nodes_set = HashSet::new(); // Use a HashSet to ensure uniqueness
        let direction = if is_incoming {
            petgraph::Direction::Incoming
        } else {
            petgraph::Direction::Outgoing
        };
    
        for index in indices {
            let node_index = petgraph::graph::NodeIndex::new(index);
            let edges = self.graph.edges_directed(node_index, direction)
                .filter(|edge| edge.weight().relation_type == relationship_type);
    
            for edge in edges {
                let related_node_index = if is_incoming {
                    edge.source()
                } else {
                    edge.target()
                };
    
                // Add the index to the HashSet, which automatically ensures uniqueness
                related_nodes_set.insert(related_node_index.index());
            }
        }
    
        // Convert the HashSet to a Vec before returning
        related_nodes_set.into_iter().collect()
    }

    pub fn get_node_attributes(
        &self,
        py: Python,
        indices: Vec<usize>,
        specified_attributes: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let result = PyDict::new(py);

        for index in indices {
            let node_index = petgraph::graph::NodeIndex::new(index);
            if let Some(node) = self.graph.node_weight(node_index) {
                let node_attributes = PyDict::new(py);

                match &specified_attributes {
                    Some(attrs) => {
                        // Check for default attributes and specified custom attributes
                        for attr in attrs {
                            match attr.as_str() {
                                "node_type" => node_attributes.set_item("node_type", &node.node_type)?,
                                "unique_id" => node_attributes.set_item("unique_id", &node.unique_id)?,
                                "title" => node_attributes.set_item("title", &node.title)?,
                                _ => {
                                    if let Some(value) = node.attributes.get(attr) {
                                        node_attributes.set_item(attr, value)?;
                                    }
                                }
                            }
                        }
                    }
                    None => {
                        // Include all attributes if none are specified
                        node_attributes.set_item("node_type", &node.node_type)?;
                        node_attributes.set_item("unique_id", &node.unique_id)?;
                        node_attributes.set_item("title", &node.title)?;

                        for (key, value) in &node.attributes {
                            node_attributes.set_item(key, value)?;
                        }
                    }
                }

                result.set_item(index, node_attributes)?;
            }
        }

        Ok(result.into())
    }

    // Additional methods as needed...
}
