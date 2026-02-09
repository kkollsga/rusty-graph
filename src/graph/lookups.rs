// src/graph/lookups.rs
use super::schema::Graph;
use crate::datatypes::Value;
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
                if node_data.node_type == node_type {
                    uid_to_index.insert(node_data.id.clone(), i);
                    title_to_index.insert(node_data.title.clone(), i);
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
        // First try direct lookup
        if let Some(idx) = self.uid_to_index.get(uid).copied() {
            return Some(idx);
        }

        // Handle type mismatches between Int64 and UniqueId
        // Python integers become Int64, but unique IDs are stored as UniqueId
        match uid {
            Value::Int64(i) => {
                // Try as UniqueId if the value fits
                if *i >= 0 && *i <= u32::MAX as i64 {
                    self.uid_to_index.get(&Value::UniqueId(*i as u32)).copied()
                } else {
                    None
                }
            }
            Value::UniqueId(u) => {
                // Try as Int64
                self.uid_to_index.get(&Value::Int64(*u as i64)).copied()
            }
            _ => None,
        }
    }

    pub fn check_title(&self, title: &Value) -> Option<NodeIndex> {
        self.title_to_index.get(title).copied()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedTypeLookup {
    source_uid_to_index: HashMap<Value, NodeIndex>,
    /// Only populated when source and target types differ (None when same_type is true)
    target_uid_to_index: Option<HashMap<Value, NodeIndex>>,
    source_type: String,
    target_type: String,
    same_type: bool,
}

impl CombinedTypeLookup {
    pub fn new(graph: &Graph, source_type: String, target_type: String) -> Result<Self, String> {
        if source_type.is_empty() || target_type.is_empty() {
            return Err("Node types cannot be empty".to_string());
        }

        let same_type = source_type == target_type;
        let mut source_uid_to_index = HashMap::new();
        let mut target_uid_to_index_map: Option<HashMap<Value, NodeIndex>> = if same_type {
            None // Don't allocate separate map when types are the same
        } else {
            Some(HashMap::new())
        };

        // Single pass through graph - collect both source and target if different types
        for idx in graph.node_indices() {
            if let Some(node_data) = graph.node_weight(idx) {
                if node_data.node_type == source_type {
                    source_uid_to_index.insert(node_data.id.clone(), idx);
                }
                // Also collect target type in same pass (if different from source)
                if let Some(ref mut target_map) = target_uid_to_index_map {
                    if node_data.node_type == target_type {
                        target_map.insert(node_data.id.clone(), idx);
                    }
                }
            }
        }

        Ok(CombinedTypeLookup {
            source_uid_to_index,
            target_uid_to_index: target_uid_to_index_map,
            source_type,
            target_type,
            same_type,
        })
    }

    pub fn check_source(&self, uid: &Value) -> Option<NodeIndex> {
        Self::lookup_with_type_fallback(&self.source_uid_to_index, uid)
    }

    pub fn check_target(&self, uid: &Value) -> Option<NodeIndex> {
        // Reuse source map when types are the same (avoids clone)
        let map = self
            .target_uid_to_index
            .as_ref()
            .unwrap_or(&self.source_uid_to_index);
        Self::lookup_with_type_fallback(map, uid)
    }

    /// Helper function to handle Int64/UniqueId type mismatches during lookup
    fn lookup_with_type_fallback(
        map: &HashMap<Value, NodeIndex>,
        uid: &Value,
    ) -> Option<NodeIndex> {
        // First try direct lookup
        if let Some(idx) = map.get(uid).copied() {
            return Some(idx);
        }

        // Handle type mismatches between Int64 and UniqueId
        match uid {
            Value::Int64(i) => {
                if *i >= 0 && *i <= u32::MAX as i64 {
                    map.get(&Value::UniqueId(*i as u32)).copied()
                } else {
                    None
                }
            }
            Value::UniqueId(u) => map.get(&Value::Int64(*u as i64)).copied(),
            _ => None,
        }
    }

    pub fn get_source_type(&self) -> &str {
        &self.source_type
    }

    pub fn get_target_type(&self) -> &str {
        &self.target_type
    }
}
