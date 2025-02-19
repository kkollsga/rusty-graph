// src/grap/data_retrieval.rs
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, CurrentSelection, NodeInfo, NodeData};
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

#[derive(Debug)]
pub struct LevelValues {
    pub parent_title: String,
    pub values: Vec<Vec<Value>>,
}

#[derive(Debug)]
pub struct UniqueValues {
    pub parent_title: String,
    pub values: Vec<Value>,
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
            
            // Always create an entry for the parent, even if nodes is empty
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
            
            // Always create values vector, even if empty
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
                
            // Get parent title even if there are no children
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
            
            // Always add to result, even with empty values
            result.push(LevelValues {
                parent_title,
                values,
            });
        }
    }
    result
}

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
                
                let mut unique_values = std::collections::HashSet::new();
                
                // Collect unique values for this parent's children
                for &idx in &filtered_children {
                    if let Some(node) = graph.get_node(idx) {
                        if let Some(value) = node.get_field(property) {
                            unique_values.insert(value.clone());
                        }
                    }
                }
                
                // Always include parent in result, even if no unique values
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
            
            // Always add result even if no unique values found
            result.push(UniqueValues {
                parent_title: "All".to_string(),
                values: all_unique_values.into_iter().collect(),
            });
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

            // Always add the parent level to result, even if no connections found
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
    result
}