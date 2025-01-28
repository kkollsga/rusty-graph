use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use crate::schema::{Node, Relation};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct TraversalLevel {
    pub nodes: Vec<usize>,
    pub parent_node: Option<usize>,
    pub relationship_type: Option<String>,
    pub node_relationships: HashMap<usize, Vec<usize>>, // Maps parent nodes to their child nodes
}

#[derive(Debug, Clone)]
pub struct TraversalContext {
    pub levels: Vec<TraversalLevel>
}

impl TraversalContext {
    pub fn new_base() -> Self {
        TraversalContext {
            levels: Vec::new()
        }
    }

    pub fn new_with_nodes(nodes: Vec<usize>) -> Self {
        let mut context = Self::new_base();
        context.add_level(nodes, None, None);
        context
    }

    pub fn new_with_traversal(nodes: Vec<usize>, parent: Option<usize>, rel_type: String) -> Self {
        let mut context = Self::new_base();
        context.add_level(nodes, parent, Some(rel_type));
        context
    }

    pub fn add_level(&mut self, nodes: Vec<usize>, parent: Option<usize>, rel_type: Option<String>) {
        let node_relationships = HashMap::new();
        self.levels.push(TraversalLevel {
            nodes,
            parent_node: parent,
            relationship_type: rel_type,
            node_relationships,
        });
    }

    pub fn add_relationships(&mut self, relationships: HashMap<usize, Vec<usize>>) {
        if let Some(last_level) = self.levels.last_mut() {
            // First collect all unique target nodes
            let all_nodes: HashSet<usize> = relationships
                .iter()
                .flat_map(|(_source, targets)| targets)
                .copied()
                .collect();
                
            // Move relationships into the level
            last_level.node_relationships = relationships;
            
            // Update nodes with the collected targets
            last_level.nodes = all_nodes.into_iter().collect();
        }
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty() || self.levels.iter().all(|level| level.nodes.is_empty())
    }

    pub fn current_nodes(&self) -> Vec<usize> {
        self.levels.last()
            .map(|level| level.nodes.clone())
            .unwrap_or_default()
    }
}

pub fn traverse_relationships(
    graph: &DiGraph<Node, Relation>,
    start_nodes: &[usize],
    rel_type: String,
    direction: Option<String>,
) -> PyResult<HashMap<usize, Vec<usize>>> {
    
    let dir = match direction.as_deref() {
        Some("incoming") => Direction::Outgoing,
        Some("outgoing") => Direction::Incoming,
        Some("both") => Direction::Outgoing,
        Some(d) => return Err(PyValueError::new_err(format!("Invalid direction: {}", d))),
        None => Direction::Incoming,
    };
    
    let mut relationships = HashMap::new();
    let handle_both = direction.is_none() || direction.as_deref() == Some("both");
    
    for &node_idx in start_nodes {
        let node_index = NodeIndex::new(node_idx);
        let mut node_traversals = Vec::new();
        
        // Handle primary direction
        let edges = graph.edges_directed(node_index, dir);
        for edge in edges {
            if edge.weight().relation_type == rel_type {
                let target_idx = match dir {
                    Direction::Incoming => edge.source().index(),
                    Direction::Outgoing => edge.target().index(),
                };
                if !node_traversals.contains(&target_idx) {
                    node_traversals.push(target_idx);
                }
            }
        }
        
        if handle_both {
            let opposite_dir = match dir {
                Direction::Incoming => Direction::Outgoing,
                Direction::Outgoing => Direction::Incoming,
            };
            
            let opposite_edges = graph.edges_directed(node_index, opposite_dir);
            for edge in opposite_edges {
                if edge.weight().relation_type == rel_type {
                    let target_idx = match opposite_dir {
                        Direction::Incoming => edge.source().index(),
                        Direction::Outgoing => edge.target().index(),
                    };
                    if !node_traversals.contains(&target_idx) {
                        node_traversals.push(target_idx);
                    }
                }
            }
        }
        
        if !node_traversals.is_empty() {
            relationships.insert(node_idx, node_traversals);
        }
    }
    Ok(relationships)
}

pub fn process_node_data(
    graph: &DiGraph<Node, Relation>,
    context: &TraversalContext,
    value_key: &str,
) -> PyResult<Vec<HashMap<String, PyObject>>> {
    let py = unsafe { Python::assume_gil_acquired() };
    let mut result = Vec::new();

    // Helper function to get node value
    fn get_node_value(graph: &DiGraph<Node, Relation>, node_idx: usize, value_key: &str) -> Option<String> {
        if let Some(Node::StandardNode { title, unique_id, attributes, .. }) = graph.node_weight(NodeIndex::new(node_idx)) {
            match value_key {
                "title" => title.clone(),
                "id" => Some(unique_id.to_string()),
                _ => attributes.get(value_key).map(|v| v.to_string()),
            }
        } else {
            None
        }
    }

    // Helper function to process node and its children recursively
    fn process_node_recursively(
        graph: &DiGraph<Node, Relation>,
        node_idx: usize,
        value_key: &str,
        current_level: usize,
        levels: &[TraversalLevel],
        py: Python,
    ) -> PyResult<Option<HashMap<String, PyObject>>> {
        // Get current node's value
        if let Some(value) = get_node_value(graph, node_idx, value_key) {
            let mut node_data = HashMap::new();
            node_data.insert(value_key.to_string(), value.into_py(py));

            // If there are more levels to process
            if current_level + 1 < levels.len() {
                // Get the next level's relationships
                let next_level = &levels[current_level + 1];
                
                // If this node has children in the next level
                if let Some(child_nodes) = next_level.node_relationships.get(&node_idx) {
                    // Process each child recursively
                    let mut child_data = Vec::new();
                    for &child_idx in child_nodes {
                        if let Some(child_result) = process_node_recursively(
                            graph,
                            child_idx,
                            value_key,
                            current_level + 1,
                            levels,
                            py
                        )? {
                            child_data.push(child_result);
                        }
                    }
                    
                    if !child_data.is_empty() {
                        node_data.insert("traversals".to_string(), child_data.into_py(py));
                    }
                }
            }
            
            Ok(Some(node_data))
        } else {
            Ok(None)
        }
    }

    // Process each root node
    if let Some(first_level) = context.levels.first() {
        for &node_idx in &first_level.nodes {
            if let Some(node_data) = process_node_recursively(
                graph,
                node_idx,
                value_key,
                0,
                &context.levels,
                py
            )? {
                result.push(node_data);
            }
        }
    }

    Ok(result)
}