// src/graph/data_retrieval.rs
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, CurrentSelection, NodeInfo, NodeData};  // Added NodeData
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use petgraph::visit::EdgeRef;

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
            
            // Always create a LevelNodes entry for the parent, even if nodes is empty
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
        // Handle direct lookup if indices provided
        let nodes = if let Some(idx) = indices {
            // Convert indices to NodeIndex, filtering out invalid ones
            idx.iter()
                .filter_map(|&i| NodeIndex::new(i).into())
                .collect::<Vec<_>>()
        } else {
            level.get_all_nodes()
        };

        // If using direct indices, create a single level
        let groups = if indices.is_some() {
            vec![(None, nodes)]
        } else {
            level.iter_groups()
                .map(|(p, c)| (p.clone(), c.clone()))
                .collect()
        };

        for (parent, children) in groups {
            let values: Vec<Vec<Value>> = children.iter()
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
                let parent_title = if indices.is_some() {
                    "Direct Lookup".to_string()
                } else {
                    match parent {
                        Some(p) => {
                            if let Some(node) = graph.get_node(p) {
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
                    }
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
        // Handle direct lookup if indices provided
        let nodes = if let Some(idx) = indices {
            // Convert indices to NodeIndex, filtering out invalid ones
            idx.iter()
                .filter_map(|&i| NodeIndex::new(i).into())
                .collect::<Vec<_>>()
        } else {
            level.get_all_nodes()
        };

        if group_by_parent {
            // If using direct indices, create a single level
            let groups = if indices.is_some() {
                vec![(None, nodes)]
            } else {
                level.iter_groups()
                    .map(|(p, c)| (p.clone(), c.clone()))
                    .collect()
            };

            for (parent, children) in groups {
                let mut unique_values = std::collections::HashSet::new();
                
                for &idx in &children {
                    if let Some(node) = graph.get_node(idx) {
                        if let Some(value) = node.get_field(property) {
                            unique_values.insert(value.clone());
                        }
                    }
                }
                
                if !unique_values.is_empty() {
                    let parent_title = if indices.is_some() {
                        "Direct Lookup".to_string()
                    } else {
                        match parent {
                            Some(p) => {
                                if let Some(node) = graph.get_node(p) {
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
                        }
                    };
                    
                    result.push(UniqueValues {
                        parent_title,
                        values: unique_values.into_iter().collect(),
                    });
                }
            }
        } else {
            // Process all nodes together
            let mut all_unique_values = std::collections::HashSet::new();
            
            for &idx in &nodes {
                if let Some(node) = graph.get_node(idx) {
                    if let Some(value) = node.get_field(property) {
                        all_unique_values.insert(value.clone());
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

#[derive(Debug)]
pub struct ConnectionInfo {
    pub node_id: Value,
    pub node_title: String,
    pub node_type: String,
    pub incoming: Vec<(String, Value, Value, HashMap<String, Value>)>, // (type, id, title, props)
    pub outgoing: Vec<(String, Value, Value, HashMap<String, Value>)>, // (type, id, title, props)
}

#[derive(Debug)]
pub struct LevelConnections {
    pub parent_title: String,
    pub parent_id: Option<Value>,
    pub parent_idx: Option<NodeIndex>,
    pub parent_type: Option<String>,
    pub connections: Vec<ConnectionInfo>,
}

pub fn get_connections(
    graph: &DirGraph,
    selection: &CurrentSelection,
    level_index: Option<usize>,
    indices: Option<&[usize]>
) -> Vec<LevelConnections> {
    let level_idx = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    let mut result = Vec::new();
    
    if let Some(level) = selection.get_level(level_idx) {
        // Handle direct lookup if indices provided
        let nodes = if let Some(idx) = indices {
            idx.iter()
                .filter_map(|&i| NodeIndex::new(i).into())
                .collect::<Vec<_>>()
        } else {
            level.get_all_nodes()
        };

        // If using direct indices, create a single level
        let groups = if indices.is_some() {
            vec![(None, nodes)]
        } else {
            level.iter_groups()
                .map(|(p, c)| (p.clone(), c.clone()))
                .collect()
        };

        for (parent, children) in groups {
            let mut level_connections = Vec::new();
            
            for node_idx in children {
                if let Some(NodeData::Regular { id, title, node_type, .. }) = graph.get_node(node_idx) {
                    let title_str = match title {
                        Value::String(ref s) => s.clone(),
                        _ => "Unknown".to_string(),
                    };

                    let mut incoming = Vec::new();
                    let mut outgoing = Vec::new();

                    // Collect incoming connections
                    for edge_ref in graph.graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                        if let Some(source_node) = graph.get_node(edge_ref.source()) {
                            let edge_data = edge_ref.weight();
                            incoming.push((
                                edge_data.connection_type.clone(),
                                source_node.get_field("id").unwrap_or(Value::Null),
                                source_node.get_field("title").unwrap_or(Value::Null),
                                edge_data.properties.clone(),
                            ));
                        }
                    }

                    // Collect outgoing connections
                    for edge_ref in graph.graph.edges_directed(node_idx, petgraph::Direction::Outgoing) {
                        if let Some(target_node) = graph.get_node(edge_ref.target()) {
                            let edge_data = edge_ref.weight();
                            outgoing.push((
                                edge_data.connection_type.clone(),
                                target_node.get_field("id").unwrap_or(Value::Null),
                                target_node.get_field("title").unwrap_or(Value::Null),
                                edge_data.properties.clone(),
                            ));
                        }
                    }

                    if !incoming.is_empty() || !outgoing.is_empty() {
                        level_connections.push(ConnectionInfo {
                            node_id: id.clone(),
                            node_title: title_str,
                            node_type: node_type.clone(),
                            incoming,
                            outgoing,
                        });
                    }
                }
            }

            if !level_connections.is_empty() {
                let (parent_title, parent_id, parent_type) = if indices.is_some() {
                    ("Direct Lookup".to_string(), None, None)
                } else {
                    match parent {
                        Some(p) => {
                            if let Some(node) = graph.get_node(p) {
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
                    }
                };

                result.push(LevelConnections {
                    parent_title,
                    parent_id,
                    parent_idx: parent.map(|p| p),
                    parent_type,
                    connections: level_connections,
                });
            }
        }
    }
    result
}