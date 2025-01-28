use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use crate::schema::{Node, Relation};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TraversalContext {
    pub nodes: Vec<usize>,
    pub traversal_type: TraversalType,
    pub node_relationships: Option<HashMap<usize, Vec<usize>>>, // Maps original nodes to their traversed nodes
}

#[derive(Debug, Clone)]
pub enum TraversalType {
    Simple,
    Relationship {
        rel_type: String,
        direction: Direction,
    },
}

impl TraversalContext {
    pub fn new_simple(nodes: Vec<usize>) -> Self {
        TraversalContext {
            nodes,
            traversal_type: TraversalType::Simple,
            node_relationships: None,
        }
    }

    pub fn new_relationship(nodes: Vec<usize>, rel_type: String, direction: Direction, relationships: HashMap<usize, Vec<usize>>) -> Self {
        TraversalContext {
            nodes,
            traversal_type: TraversalType::Relationship {
                rel_type,
                direction,
            },
            node_relationships: Some(relationships),
        }
    }
}

pub fn traverse_relationships(
    graph: &DiGraph<Node, Relation>,
    start_nodes: &[usize],
    rel_type: String,
    direction: Option<String>,
) -> PyResult<(Vec<usize>, HashMap<usize, Vec<usize>>)> {
    let dir = match direction.as_deref() {
        Some("incoming") => Direction::Incoming,
        Some("outgoing") => Direction::Outgoing,
        Some("both") => Direction::Outgoing,
        Some(d) => return Err(PyValueError::new_err(format!("Invalid direction: {}", d))),
        None => Direction::Outgoing,
    };
    
    let mut result = Vec::new();
    let mut relationships = HashMap::new();
    let handle_both = direction.as_deref() == Some("both");
    
    for &node_idx in start_nodes {
        let node_index = NodeIndex::new(node_idx);
        let mut node_traversals = Vec::new();
        
        // For each class, look for students with relationships to it
        let edges_to_check = graph.edges_directed(node_index, Direction::Incoming);
        for edge in edges_to_check {
            if edge.weight().relation_type == rel_type {
                let source = edge.source().index();
                if !result.contains(&source) {
                    result.push(source);
                }
                node_traversals.push(source);
            }
        }
        
        if !node_traversals.is_empty() {
            relationships.insert(node_idx, node_traversals);
        }
    }
    
    Ok((result, relationships))
}