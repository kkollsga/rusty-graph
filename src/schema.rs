// In schema.rs

use crate::data_types::AttributeValue;
use std::collections::HashMap;

// Node structure definition
pub enum Node {
    StandardNode {
        node_type: String,
        unique_id: String,
        attributes: HashMap<String, AttributeValue>,
        title: Option<String>,
    },
    DataTypeNode {
        data_type: String,  // 'Node' or 'Relation'
        name: String,
        attributes: HashMap<String, String>,  // Attribute name to data type ('Int', 'Float', etc.)
    },
    // Add other variants as needed
}

impl Node {
    // Implement constructor methods for each variant if needed
    pub fn new(node_type: &str, unique_id: &str, attributes: Option<HashMap<String, AttributeValue>>, node_title: Option<&str>) -> Self {
        Node::StandardNode {
            node_type: node_type.to_string(),
            unique_id: unique_id.to_string(),
            attributes: attributes.unwrap_or_else(HashMap::new),
            title: node_title.map(|t| t.to_string()),
        }
    }

    pub fn new_data_type(data_type: &str, name: &str, attributes: HashMap<String, String>) -> Self {
        Node::DataTypeNode {
            data_type: data_type.to_string(),
            name: name.to_string(),
            attributes,
        }
    }
}

// Relation structure definition
pub struct Relation {
    pub relation_type: String,
    pub attributes: Option<HashMap<String, AttributeValue>>,  // Now an Option
}

impl Relation {
    // Adjust the constructor to accept an Option for attributes
    pub fn new(name: &str, attributes: Option<HashMap<String, AttributeValue>>) -> Self {
        Relation {
            relation_type: name.to_string(),
            attributes,  // Directly passed as an Option
        }
    }
}