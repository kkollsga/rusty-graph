use std::collections::HashMap;

pub struct Node {
    pub node_type: String, // Add node_type field
    pub unique_id: String,
    pub attributes: HashMap<String, String>,
    pub title: Option<String>,
}

impl Node {
    pub fn new(node_type: &str, unique_id: &str, attributes: HashMap<String, String>, node_title: Option<&str>) -> Self {
        Node {
            node_type: node_type.to_string(),
            unique_id: unique_id.to_string(),
            attributes,
            title: node_title.map(|t| t.to_string()),  // Convert Option<&str> to Option<String> 
        }
    }
}