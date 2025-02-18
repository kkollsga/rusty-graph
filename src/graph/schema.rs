// src/graph/schema.rs
use std::collections::HashMap;
use petgraph::graph::{DiGraph, NodeIndex, EdgeIndex};
use serde::{Serialize, Deserialize};
use crate::datatypes::values::{Value, FilterCondition};

#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub id: Value,
    pub title: Value,
    pub node_type: String,
    pub properties: HashMap<String, Value>,
}

#[derive(Clone, Debug)]
pub enum SelectionOperation {
    Filter(HashMap<String, FilterCondition>),
    Sort(Vec<(String, bool)>),  // (field_name, ascending)
    Traverse {
        connection_type: String,
        direction: Option<String>,
        max_nodes: Option<usize>,
    },
}

#[derive(Clone, Debug)]
pub struct SelectionLevel {
    pub selections: HashMap<Option<NodeIndex>, Vec<NodeIndex>>, // parent_idx -> selected_children
    pub operations: Vec<SelectionOperation>,
}

impl SelectionLevel {
    pub fn new() -> Self {
        SelectionLevel {
            selections: HashMap::new(),
            operations: Vec::new(),
        }
    }

    pub fn add_filter(&mut self, conditions: HashMap<String, FilterCondition>) {
        self.operations.push(SelectionOperation::Filter(conditions));
    }

    pub fn add_sort(&mut self, sort_fields: Vec<(String, bool)>) {
        self.operations.push(SelectionOperation::Sort(sort_fields));
    }

    pub fn add_selection(&mut self, parent: Option<NodeIndex>, children: Vec<NodeIndex>) {
        self.selections.insert(parent, children);
    }

    pub fn get_all_nodes(&self) -> Vec<NodeIndex> {
        self.selections.values()
            .flat_map(|children| children.iter().cloned())
            .collect()
    }

    pub fn get_children(&self, parent: NodeIndex) -> Option<&Vec<NodeIndex>> {
        self.selections.get(&Some(parent))
    }

    pub fn get_parent_nodes(&self) -> Vec<NodeIndex> {
        self.selections.keys()
            .filter_map(|opt_idx| *opt_idx)
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.selections.is_empty()
    }

    pub fn iter_groups(&self) -> impl Iterator<Item = (&Option<NodeIndex>, &Vec<NodeIndex>)> {
        self.selections.iter()
    }
}

#[derive(Clone, Default)]
pub struct CurrentSelection {
    levels: Vec<SelectionLevel>,
    current_level: usize,
}

impl CurrentSelection {
    pub fn new() -> Self {
        let mut selection = CurrentSelection {
            levels: Vec::new(),
            current_level: 0,
        };
        selection.add_level(); // Always start with an initial level
        selection
    }

    pub fn add_level(&mut self) {
        // No need to pass level index
        self.levels.push(SelectionLevel::new());
        self.current_level = self.levels.len() - 1;
    }

    pub fn clear(&mut self) {
        self.levels.clear();
        self.current_level = 0;
        self.add_level(); // Ensure we always have at least one level after clearing
    }

    pub fn get_level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn get_level(&self, index: usize) -> Option<&SelectionLevel> {
        self.levels.get(index)
    }

    pub fn get_level_mut(&mut self, index: usize) -> Option<&mut SelectionLevel> {
        self.levels.get_mut(index)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DirGraph {
    pub(crate) graph: Graph,
    pub(crate) type_indices: HashMap<String, Vec<NodeIndex>>,
}

impl DirGraph {
    pub fn new() -> Self {
        DirGraph {
            graph: Graph::new(),
            type_indices: HashMap::new(),
        }
    }

    pub fn has_connection_type(&self, connection_type: &str) -> bool {
        // Look for a SchemaNode that represents this connection type
        self.graph.node_weights().any(|node| {
            match node {
                NodeData::Schema { node_type, title, .. } => {
                    node_type == "SchemaNode" && 
                    matches!(title, Value::String(t) if t == connection_type)
                },
                _ => false
            }
        })
    }
    
    pub fn get_node(&self, index: NodeIndex) -> Option<&NodeData> {
        self.graph.node_weight(index)
    }

    pub fn get_node_mut(&mut self, index: NodeIndex) -> Option<&mut NodeData> {
        self.graph.node_weight_mut(index)
    }

    pub fn _get_connection(&self, index: EdgeIndex) -> Option<&EdgeData> {
        self.graph.edge_weight(index)
    }

    pub fn _get_connection_mut(&mut self, index: EdgeIndex) -> Option<&mut EdgeData> {
        self.graph.edge_weight_mut(index)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeData {
    Regular {
        id: Value,
        title: Value,
        node_type: String,
        properties: HashMap<String, Value>,
    },
    Schema {
        id: Value,
        title: Value,
        node_type: String,
        properties: HashMap<String, Value>,
    },
}

impl NodeData {
    pub fn new(id: Value, title: Value, node_type: String, properties: HashMap<String, Value>) -> Self {
        NodeData::Regular {
            id,
            title,
            node_type,
            properties,
        }
    }
    pub fn get_field(&self, field: &str) -> Option<Value> {
        match self {
            NodeData::Regular { id, title, node_type, properties } => {
                match field {
                    "id" => Some(id.clone()),
                    "title" => Some(title.clone()),
                    "type" => Some(Value::String(node_type.clone())),
                    _ => properties.get(field).cloned()
                }
            },
            NodeData::Schema { .. } => None
        }
    }

    pub fn is_regular(&self) -> bool {
        matches!(self, NodeData::Regular { .. })
    }

    pub fn to_node_info(&self) -> Option<NodeInfo> {
        match self {
            NodeData::Regular { id, title, node_type, properties } => Some(NodeInfo {
                id: id.clone(),
                title: title.clone(),
                node_type: node_type.clone(),
                properties: properties.clone(),
            }),
            NodeData::Schema { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    pub connection_type: String,
    pub properties: HashMap<String, Value>,
}

impl EdgeData {
    pub fn new(connection_type: String, properties: HashMap<String, Value>) -> Self {
        EdgeData {
            connection_type,
            properties,
        }
    }
}

pub type Graph = DiGraph<NodeData, EdgeData>;