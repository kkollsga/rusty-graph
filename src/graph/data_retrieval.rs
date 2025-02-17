// src/graph/data_retrieval.rs
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, CurrentSelection, NodeInfo, NodeData};  // Added NodeData
use petgraph::graph::NodeIndex;

#[derive(Debug)]
pub struct LevelNodes {
    pub parent_title: String,
    pub parent_id: Option<Value>,
    pub parent_idx: Option<NodeIndex>,
    pub parent_type: Option<String>,
    pub nodes: Vec<NodeInfo>,
}


pub fn get_nodes(
    graph: &DirGraph,
    selection: &CurrentSelection,
    level_index: Option<usize>,
    indices: Option<&[usize]>
) -> Vec<LevelNodes> {
    // If specific indices are provided, do direct lookup
    if let Some(idx) = indices {
        let mut direct_nodes = Vec::new();
        for &index in idx {
            if let Some(node_idx) = NodeIndex::new(index).into() {
                if let Some(node) = graph.get_node(node_idx) {
                    if let Some(node_info) = node.to_node_info() {
                        direct_nodes.push(node_info);
                    }
                }
            }
        }
        
        if !direct_nodes.is_empty() {
            return vec![LevelNodes {
                parent_title: "Direct Lookup".to_string(),
                parent_id: None,
                parent_idx: None,
                parent_type: None,
                nodes: direct_nodes,
            }];
        }
        return Vec::new();
    }
    
    // Original selection-based logic
    let level_idx = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    let mut result = Vec::new();
    
    if let Some(level) = selection.get_level(level_idx) {
        for (parent, children) in level.iter_groups() {
            let mut nodes = Vec::new();
            
            for &child_idx in children {
                if let Some(node) = graph.get_node(child_idx) {
                    if let Some(node_info) = node.to_node_info() {
                        nodes.push(node_info);
                    }
                }
            }
            
            if !nodes.is_empty() {
                let (parent_title, parent_id, parent_type) = match parent {
                    Some(p) => {
                        if let Some(node) = graph.get_node(*p) {
                            (
                                node.get_field("title")
                                    .and_then(|v| match v {
                                        Value::String(s) => Some(s),
                                        _ => None
                                    })
                                    .unwrap_or_else(|| "Unknown".to_string()),
                                node.get_field("id"),
                                match node {
                                    NodeData::Regular { node_type, .. } => Some(node_type.clone()),
                                    _ => None
                                }
                            )
                        } else {
                            ("Unknown".to_string(), None, None)
                        }
                    },
                    None => ("Root".to_string(), None, None),
                };
                
                result.push(LevelNodes {
                    parent_title,
                    parent_id,
                    parent_idx: parent.map(|p| p),
                    parent_type,
                    nodes,
                });
            }
        }
    }
    result
}


#[derive(Debug)]
pub struct LevelValues {
    pub parent_title: String,
    pub values: Vec<Vec<Value>>,  // Modified to store multiple values per node
}

pub fn get_property_values(
    graph: &DirGraph,
    selection: &CurrentSelection,
    level_index: Option<usize>,
    properties: &[&str],
    indices: Option<&[usize]>
) -> Vec<LevelValues> {
    let level_idx = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    let mut result = Vec::new();
    
    if let Some(level) = selection.get_level(level_idx) {
        for (parent, children) in level.iter_groups() {
            let filtered_children: Vec<NodeIndex> = match indices {
                Some(idx) => children.iter()
                    .filter(|&c| idx.contains(&c.index()))
                    .cloned()
                    .collect(),
                None => children.clone(),
            };
            
            if filtered_children.is_empty() {
                continue;
            }
            
            let values: Vec<Vec<Value>> = filtered_children.iter()
                .map(|&idx| {
                    properties.iter()
                        .map(|&prop| {
                            graph.get_node(idx)
                                .and_then(|node| node.get_field(prop))
                                .map(|value| value.clone())
                                .unwrap_or(Value::Null)
                        })
                        .collect()
                })
                .collect();
                
            if !values.is_empty() {
                let parent_title = match parent {
                    Some(p) => {
                        if let Some(node) = graph.get_node(*p) {
                            if let Some(Value::String(title)) = node.get_field("title") {
                                title
                            } else {
                                "Unknown".to_string()
                            }
                        } else {
                            "Unknown".to_string()
                        }
                    },
                    None => "Root".to_string(),
                };
                
                result.push(LevelValues {
                    parent_title,
                    values,
                });
            }
        }
    }
    result
}

#[derive(Debug)]
pub struct UniqueValues {
    pub parent_title: String,
    pub values: Vec<Value>,  // Single vec of unique values
}

// Add these new functions
pub fn get_unique_values(
    graph: &DirGraph,
    selection: &CurrentSelection,
    property: &str,
    level_index: Option<usize>,
    group_by_parent: bool,
    indices: Option<&[usize]>
) -> Vec<UniqueValues> {
    let level_idx = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    let mut result = Vec::new();
    
    if let Some(level) = selection.get_level(level_idx) {
        if group_by_parent {
            // Process each parent-child group separately
            for (parent, children) in level.iter_groups() {
                let filtered_children: Vec<NodeIndex> = match indices {
                    Some(idx) => children.iter()
                        .filter(|&c| idx.contains(&c.index()))
                        .cloned()
                        .collect(),
                    None => children.clone(),
                };
                
                if filtered_children.is_empty() {
                    continue;
                }
                
                let mut unique_values = std::collections::HashSet::new();
                
                // Collect unique values for this parent's children
                for &idx in &filtered_children {
                    if let Some(node) = graph.get_node(idx) {
                        if let Some(value) = node.get_field(property) {
                            unique_values.insert(value.clone());
                        }
                    }
                }
                
                if !unique_values.is_empty() {
                    let parent_title = match parent {
                        Some(p) => {
                            if let Some(node) = graph.get_node(*p) {
                                if let Some(Value::String(title)) = node.get_field("title") {
                                    title
                                } else {
                                    "Unknown".to_string()
                                }
                            } else {
                                "Unknown".to_string()
                            }
                        },
                        None => "Root".to_string(),
                    };
                    
                    result.push(UniqueValues {
                        parent_title,
                        values: unique_values.into_iter().collect(),
                    });
                }
            }
        } else {
            // Process all children together
            let mut all_unique_values = std::collections::HashSet::new();
            
            for (_, children) in level.iter_groups() {
                let filtered_children: Vec<NodeIndex> = match indices {
                    Some(idx) => children.iter()
                        .filter(|&c| idx.contains(&c.index()))
                        .cloned()
                        .collect(),
                    None => children.clone(),
                };
                
                for &idx in &filtered_children {
                    if let Some(node) = graph.get_node(idx) {
                        if let Some(value) = node.get_field(property) {
                            all_unique_values.insert(value.clone());
                        }
                    }
                }
            }
            
            if !all_unique_values.is_empty() {
                result.push(UniqueValues {
                    parent_title: "All".to_string(),
                    values: all_unique_values.into_iter().collect(),
                });
            }
        }
    }
    
    result
}