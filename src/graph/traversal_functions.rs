use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::{HashMap, HashSet};
use pyo3::types::PyDict;

use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue;
use crate::graph::query_functions;

#[derive(Debug, Clone)]
pub struct TraversalLevel {
    pub nodes: Vec<usize>,
    #[allow(dead_code)]
    pub parent_node: Option<usize>,
    #[allow(dead_code)]
    pub relationship_type: Option<String>,
    pub node_relationships: HashMap<usize, Vec<usize>>, // Maps parent nodes to their child nodes
}

#[derive(Debug, Clone)]
pub struct TraversalContext {
    pub levels: Vec<TraversalLevel>,
    pub results: Option<PyObject>
}

impl TraversalContext {
    pub fn new_base() -> Self {
        TraversalContext {
            levels: Vec::new(),
            results: None
        }
    }

    pub fn new_with_nodes(nodes: Vec<usize>) -> Self {
        let mut context = Self::new_base();
        context.add_level(nodes, None, None);
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
    filter: Option<&PyDict>,
    sort: Option<&PyDict>,
    max_traversals: Option<usize>,  // New parameter
) -> PyResult<HashMap<usize, Vec<usize>>> {
    
    let check_both = direction.is_none() || direction.as_deref() == Some("both");
    
    let mut relationships = HashMap::new();
    
    for &node_idx in start_nodes {
        let node_index = NodeIndex::new(node_idx);
        let mut node_traversals = Vec::new();
        
        // Helper closure to collect target nodes from edges in a given direction
        let mut collect_targets = |dir: Direction| {
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
        };

        // Process edges based on direction
        if check_both {
            collect_targets(Direction::Incoming);
            collect_targets(Direction::Outgoing);
        } else {
            match direction.as_deref() {
                Some("incoming") => collect_targets(Direction::Incoming),
                Some("outgoing") => collect_targets(Direction::Outgoing),
                Some(d) => return Err(PyValueError::new_err(format!("Invalid direction: {}", d))),
                None => unreachable!(), // This case is handled by check_both
            }
        }
        
        // Apply filter to all collected nodes only if filter is provided and not empty
        if let Some(filter_dict) = filter {
            if !filter_dict.is_empty() {
                let filtered_nodes = query_functions::filter_nodes(graph, Some(node_traversals), filter_dict)?;
                node_traversals = filtered_nodes;
            }
        }

        // Apply sorting if provided
        if let Some(sort_dict) = sort {
            if !sort_dict.is_empty() {
                node_traversals = query_functions::sort_nodes(graph, node_traversals, |&a, &b| {
                    for (key, value) in sort_dict.iter() {
                        let key = match key.extract::<String>() {
                            Ok(k) => k,
                            Err(_) => continue,
                        };
                        
                        // Handle both direct values and condition dictionaries
                        let ascending = if let Ok(dict) = value.extract::<&PyDict>() {
                            // Fixed extraction of order parameter
                            match dict.get_item("order") {
                                Ok(Some(v)) => v.extract::<bool>().unwrap_or(true),
                                _ => true,
                            }
                        } else {
                            true
                        };

                        let compare_values = |idx: usize| {
                            let node = graph.node_weight(petgraph::graph::NodeIndex::new(idx));
                            match node {
                                Some(Node::StandardNode { attributes, .. }) => {
                                    match key.as_str() {
                                        "title" => node.and_then(|n| match n {
                                            Node::StandardNode { title, .. } => title.clone().map(AttributeValue::String),
                                            _ => None
                                        }),
                                        _ => attributes.get(&key).cloned()
                                    }
                                },
                                _ => None
                            }
                        };

                        let a_val = compare_values(a);
                        let b_val = compare_values(b);

                        let ordering = match (a_val, b_val) {
                            (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
                            (Some(_), None) => std::cmp::Ordering::Less,
                            (None, Some(_)) => std::cmp::Ordering::Greater,
                            (None, None) => std::cmp::Ordering::Equal,
                        };

                        if ordering != std::cmp::Ordering::Equal {
                            return if ascending { ordering } else { ordering.reverse() };
                        }
                    }
                    std::cmp::Ordering::Equal
                });
            }
        }
        
        // Apply max_traversals limit if provided
        if let Some(limit) = max_traversals {
            if limit > 0 {  // Protect against zero limit
                node_traversals.truncate(limit);
            }
        }
        
        if !node_traversals.is_empty() {
            relationships.insert(node_idx, node_traversals);
        }
    }
    Ok(relationships)
}

pub fn process_attributes_levels(
    graph: &DiGraph<Node, Relation>,
    context: &TraversalContext,
    attributes: Option<Vec<String>>,
    max_results: Option<usize>,
) -> PyResult<PyObject> {
    let py = unsafe { Python::assume_gil_acquired() };

    // Get nodes from context, or return empty list if no context
    let mut nodes = if context.levels.is_empty() {
        Vec::new()
    } else {
        context.levels[0].nodes.clone()
    };

    // Apply max_results if specified
    if let Some(limit) = max_results {
        if limit == 0 {
            return Err(PyValueError::new_err("max_results must be positive"));
        }
        nodes.truncate(limit);
    }

    let mut result = Vec::new();
    
    for &root_idx in &nodes {
        if let Some(node) = graph.node_weight(NodeIndex::new(root_idx)) {
            if let Ok(mut root_data) = node.to_node_data(root_idx, py, attributes.as_deref()) {
                // Add traversal data only if we have more levels AND relationships exist
                if context.levels.len() > 1 && 
                   context.levels[1].node_relationships.contains_key(&root_idx) &&
                   !context.levels[1].node_relationships[&root_idx].is_empty() {
                    
                    let mut children = Vec::new();
                    
                    if let Some(child_nodes) = context.levels[1].node_relationships.get(&root_idx) {
                        for &child_idx in child_nodes {
                            if let Some(child_node) = graph.node_weight(NodeIndex::new(child_idx)) {
                                if let Ok(mut child_data) = child_node.to_node_data(child_idx, py, attributes.as_deref()) {
                                    // Add grandchildren only if they exist
                                    if context.levels.len() > 2 && 
                                       context.levels[2].node_relationships.contains_key(&child_idx) &&
                                       !context.levels[2].node_relationships[&child_idx].is_empty() {
                                        
                                        let mut grandchildren = Vec::new();
                                        if let Some(grandchild_nodes) = context.levels[2].node_relationships.get(&child_idx) {
                                            for &grandchild_idx in grandchild_nodes {
                                                if let Some(grandchild_node) = graph.node_weight(NodeIndex::new(grandchild_idx)) {
                                                    if let Ok(grandchild_data) = grandchild_node.to_node_data(grandchild_idx, py, attributes.as_deref()) {
                                                        grandchildren.push(grandchild_data);
                                                    }
                                                }
                                            }
                                        }
                                        
                                        if !grandchildren.is_empty() {
                                            child_data.traversals = Some(grandchildren);
                                        }
                                    }
                                    children.push(child_data);
                                }
                            }
                        }
                        
                        if !children.is_empty() {
                            root_data.traversals = Some(children);
                        }
                    }
                }
                
                result.push(Node::node_data_to_py(&root_data, py)?);
            }
        }
    }

    Ok(result.into_py(py))
}

pub fn process_traversal_levels(
    graph: &DiGraph<Node, Relation>,
    context: &TraversalContext,
    selector: &str,
    max_results: Option<usize>,
) -> PyResult<PyObject> {
    let py = unsafe { Python::assume_gil_acquired() };

    if context.levels.is_empty() {
        return Ok(PyDict::new(py).into());
    }

    // Helper function to get node value
    let get_value = |node_idx: usize| -> Option<String> {
        if let Some(Node::StandardNode { title, unique_id, attributes, .. }) = graph.node_weight(NodeIndex::new(node_idx)) {
            match selector {
                "title" => title.clone(),
                "unique_id" => Some(unique_id.to_string()),
                "graph_index" => Some(node_idx.to_string()),
                _ => attributes.get(selector).map(|v| v.to_string())
            }
        } else {
            None
        }
    };

    // Single level case: return a list
    if context.levels.len() == 1 {
        let mut nodes = context.levels[0].nodes.clone();
        if let Some(limit) = max_results {
            if limit == 0 {
                return Err(PyValueError::new_err("max_results must be positive"));
            }
            nodes.truncate(limit);
        }

        let mut values = Vec::new();
        for node_idx in nodes {
            if let Some(value) = get_value(node_idx) {
                values.push(value);
            }
        }
        return Ok(values.into_py(py));
    }

    let result = PyDict::new(py);

    // Process root nodes
    for &root_idx in &context.levels[0].nodes {
        if let Some(root_value) = get_value(root_idx) {
            // For single traversal, return dictionary with lists
            if context.levels.len() == 2 {
                let mut children = Vec::new();
                if let Some(child_nodes) = context.levels[1].node_relationships.get(&root_idx) {
                    for &child_idx in child_nodes {
                        if let Some(child_value) = get_value(child_idx) {
                            children.push(child_value);
                        }
                    }
                }
                result.set_item(root_value, children)?;
            } 
            // For multiple traversals
            else {
                let second_level = PyDict::new(py);
                if let Some(child_nodes) = context.levels[1].node_relationships.get(&root_idx) {
                    for &child_idx in child_nodes {
                        if let Some(child_value) = get_value(child_idx) {
                            // Always create a list for the final level
                            let mut final_level = Vec::new();
                            if let Some(grandchild_nodes) = context.levels[2].node_relationships.get(&child_idx) {
                                for &grandchild_idx in grandchild_nodes {
                                    if let Some(grandchild_value) = get_value(grandchild_idx) {
                                        final_level.push(grandchild_value);
                                    }
                                }
                            }
                            second_level.set_item(child_value, final_level)?;
                        }
                    }
                }
                result.set_item(root_value, second_level)?;
            }
        }
    }

    Ok(result.into())
}

pub fn count_traversal_levels(
    context: &TraversalContext,
    max_results: Option<usize>,
) -> PyResult<PyObject> {
    let py = unsafe { Python::assume_gil_acquired() };

    if context.levels.is_empty() {
        return Ok(0_usize.into_py(py));
    }

    // Single level case: return total count
    if context.levels.len() == 1 {
        let mut count = context.levels[0].nodes.len();
        if let Some(limit) = max_results {
            if limit == 0 {
                return Err(PyValueError::new_err("max_results must be positive"));
            }
            count = count.min(limit);
        }
        return Ok(count.into_py(py));
    }

    let result = PyDict::new(py);

    // Process root nodes
    for &root_idx in &context.levels[0].nodes {
        // For single traversal, return dictionary with counts
        if context.levels.len() == 2 {
            let count = context.levels[1]
                .node_relationships
                .get(&root_idx)
                .map_or(0, |children| children.len());
            result.set_item(root_idx.to_string(), count)?;
        } 
        // For multiple traversals
        else {
            let second_level = PyDict::new(py);
            if let Some(child_nodes) = context.levels[1].node_relationships.get(&root_idx) {
                for &child_idx in child_nodes {
                    let final_count = context.levels[2]
                        .node_relationships
                        .get(&child_idx)
                        .map_or(0, |grandchildren| grandchildren.len());
                    second_level.set_item(child_idx.to_string(), final_count)?;
                }
            }
            result.set_item(root_idx.to_string(), second_level)?;
        }
    }

    Ok(result.into())
}