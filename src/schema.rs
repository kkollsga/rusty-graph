use crate::data_types::AttributeValue;
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum Node {
    StandardNode {
        node_type: String,
        unique_id: i32,
        attributes: HashMap<String, AttributeValue>,
        title: Option<String>,
    },
    DataTypeNode {
        data_type: String,  // 'Node' or 'Relation'
        name: String,
        attributes: HashMap<String, String>,  // Attribute name to data type ('Int', 'Float', etc.)
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Relation {
    pub relation_type: String,
    pub attributes: Option<HashMap<String, AttributeValue>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeTypeStats {
    pub title: String,
    pub graph_id: String,
    pub attributes: HashMap<String, AttributeMetadata>,
    pub occurrences: usize,
    pub relationships: RelationshipMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AttributeMetadata {
    pub data_type: String,
    pub nullable: bool,
    pub unique_values: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RelationshipMetadata {
    pub incoming_types: HashSet<String>,
    pub outgoing_types: HashSet<String>,
    pub connected_node_types: HashSet<String>,
}

impl Node {
    pub fn new(node_type: &str, unique_id: i32, attributes: Option<HashMap<String, AttributeValue>>, node_title: Option<&str>) -> Self {
        Node::StandardNode {
            node_type: node_type.to_string(),
            unique_id,
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

impl Relation {
    pub fn new(name: &str, attributes: Option<HashMap<String, AttributeValue>>) -> Self {
        Relation {
            relation_type: name.to_string(),
            attributes,
        }
    }
}