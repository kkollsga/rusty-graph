// src/graph/lookups.rs
use std::collections::HashMap;
use petgraph::graph::NodeIndex;
use serde::{Serialize, Deserialize};
use crate::datatypes::Value;
use super::schema::{Graph, NodeData};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeLookup {
    uid_to_index: HashMap<Value, NodeIndex>,
    title_to_index: HashMap<Value, NodeIndex>,
    node_type: String,
}

impl TypeLookup {
    pub fn new(graph: &Graph, node_type: String) -> Result<Self, String> {
        if node_type.is_empty() {
            return Err("Node type cannot be empty".to_string());
        }

        let mut uid_to_index = HashMap::new();
        let mut title_to_index = HashMap::new();

        // Single pass through the graph
        for i in graph.node_indices() {
            if let Some(node_data) = graph.node_weight(i) {
                match node_data {
                    NodeData::Regular { node_type: nt, id, title, .. } if nt == &node_type => {
                        uid_to_index.insert(id.clone(), i);
                        title_to_index.insert(title.clone(), i);
                    },
                    NodeData::Schema { title, .. } if node_type == "SchemaNode" => {
                        uid_to_index.insert(title.clone(), i);
                        title_to_index.insert(title.clone(), i);
                    },
                    _ => {}
                }
            }
        }

        Ok(TypeLookup {
            uid_to_index,
            title_to_index,
            node_type,
        })
    }

    pub fn check_uid(&self, uid: &Value) -> Option<NodeIndex> {
        self.uid_to_index.get(uid).copied()
    }

    pub fn check_title(&self, title: &Value) -> Option<NodeIndex> {
        self.title_to_index.get(title).copied()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedTypeLookup {
    source_uid_to_index: HashMap<Value, NodeIndex>,
    target_uid_to_index: HashMap<Value, NodeIndex>,
    source_type: String,
    target_type: String,
    same_type: bool,  // Flag to indicate if source and target types are the same
}

impl CombinedTypeLookup {
    pub fn new(graph: &Graph, source_type: String, target_type: String) -> Result<Self, String> {
        if source_type.is_empty() || target_type.is_empty() {
            return Err("Node types cannot be empty".to_string());
        }

        // Check if source and target types are the same
        let same_type = source_type == target_type;
        
        let mut source_uid_to_index = HashMap::new();
        let target_uid_to_index: HashMap<Value, NodeIndex>;

        // First pass: populate source_uid_to_index
        for idx in graph.node_indices() {
            if let Some(node_data) = graph.node_weight(idx) {
                match node_data {
                    NodeData::Regular { node_type, id, .. } => {
                        if node_type == &source_type {
                            source_uid_to_index.insert(id.clone(), idx);
                        }
                    },
                    NodeData::Schema { node_type, title, .. } if node_type == "SchemaNode" => {
                        if source_type == "SchemaNode" {
                            source_uid_to_index.insert(title.clone(), idx);
                        }
                    },
                    _ => {}
                }
            }
        }

        // Performance optimization: If types are the same, reuse the source map
        if same_type {
            target_uid_to_index = source_uid_to_index.clone();
        } else {
            // Different types - create a separate target map
            let mut target_map = HashMap::new();
            for idx in graph.node_indices() {
                if let Some(node_data) = graph.node_weight(idx) {
                    match node_data {
                        NodeData::Regular { node_type, id, .. } => {
                            if node_type == &target_type {
                                target_map.insert(id.clone(), idx);
                            }
                        },
                        NodeData::Schema { node_type, title, .. } if node_type == "SchemaNode" => {
                            if target_type == "SchemaNode" {
                                target_map.insert(title.clone(), idx);
                            }
                        },
                        _ => {}
                    }
                }
            }
            target_uid_to_index = target_map;
        }

        Ok(CombinedTypeLookup {
            source_uid_to_index,
            target_uid_to_index,
            source_type,
            target_type,
            same_type,
        })
    }

    pub fn check_source(&self, uid: &Value) -> Option<NodeIndex> {
        self.source_uid_to_index.get(uid).copied()
    }

    pub fn check_target(&self, uid: &Value) -> Option<NodeIndex> {
        self.target_uid_to_index.get(uid).copied()
    }

    pub fn get_source_type(&self) -> &str {
        &self.source_type
    }

    pub fn get_target_type(&self) -> &str {
        &self.target_type
    }
}