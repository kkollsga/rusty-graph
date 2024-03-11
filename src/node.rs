use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone)]
pub struct Node {
    pub node_type: String, // Add node_type field
    pub unique_id: String,
    pub title: String,
    pub attributes: HashMap<String, String>,
}

#[pymethods]
impl Node {
    #[new]
    pub fn new(node_type: &str, unique_id: &str, node_title: &str, attributes: HashMap<String, String>) -> Self {
        Node {
            node_type: node_type.to_string(),
            unique_id: unique_id.to_string(),
            title: node_title.to_string(),
            attributes,
        }
    }
}