use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::{HashMap, HashSet};
use pyo3::types::PyDict;

use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue;
use crate::graph::add_nodes::update_node_attribute;
use crate::graph::get_schema::update_or_retrieve_schema;

#[derive(Debug, Clone)]
pub struct TraversalLevel {
    pub nodes: Vec<usize>,
    pub parent_node: Option<usize>,
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
                // Add traversal data if we have more levels
                if context.levels.len() > 1 {
                    if let Some(child_nodes) = context.levels[1].node_relationships.get(&root_idx) {
                        let mut children = Vec::new();
                        
                        for &child_idx in child_nodes {
                            if let Some(child_node) = graph.node_weight(NodeIndex::new(child_idx)) {
                                if let Ok(mut child_data) = child_node.to_node_data(child_idx, py, attributes.as_deref()) {
                                    // Add grandchildren if they exist
                                    if context.levels.len() > 2 {
                                        if let Some(grandchild_nodes) = context.levels[2].node_relationships.get(&child_idx) {
                                            for &grandchild_idx in grandchild_nodes {
                                                if let Some(grandchild_node) = graph.node_weight(NodeIndex::new(grandchild_idx)) {
                                                    if let Ok(grandchild_data) = grandchild_node.to_node_data(grandchild_idx, py, attributes.as_deref()) {
                                                        child_data.traversals.push(grandchild_data);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    children.push(child_data);
                                }
                            }
                        }
                        
                        root_data.traversals = children;
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

pub fn store_traversal_values(
    graph: &mut DiGraph<Node, Relation>,
    context: &TraversalContext,
    attribute_name: &str,
    store_values: &PyObject,
) -> PyResult<()> {
    let py = unsafe { Python::assume_gil_acquired() };
    
    println!("Store function called with attribute_name: {}", attribute_name);
    
    if context.levels.is_empty() {
        println!("Context levels empty, returning");
        return Ok(());
    }

    if context.levels.len() == 1 {
        println!("Only one context level, returning");
        return Ok(());
    }

    let dict = store_values.downcast::<PyDict>(py)?;
    println!("Store values: {:?}", dict.to_string());
    
    let mut node_types_to_update = HashSet::new();
    
    for (key, value) in dict.iter() {
        let parent_idx = key.extract::<String>()?.parse::<usize>()
            .map_err(|_| PyValueError::new_err("Invalid node index"))?;
            
        println!("Processing node index: {}", parent_idx);
            
        if let Ok(count) = value.extract::<usize>() {
            println!("Found count value: {}", count);
            
            // Debug node before update
            if let Node::StandardNode { node_type, attributes, .. } = &graph[NodeIndex::new(parent_idx)] {
                println!("Before update - Node type: {}, Attributes: {:?}", node_type, attributes);
                node_types_to_update.insert(node_type.clone());
            }
            
            match update_node_attribute(
                graph,
                parent_idx,
                attribute_name,
                AttributeValue::Int(count as i32)
            ) {
                Ok(_) => println!("Successfully updated attribute"),
                Err(e) => println!("Error updating attribute: {:?}", e),
            }
            
            // Debug node after update
            if let Node::StandardNode { attributes, .. } = &graph[NodeIndex::new(parent_idx)] {
                println!("After update - Attributes: {:?}", attributes);
            }
            
        } else if let Ok(inner_dict) = value.downcast::<PyDict>() {
            println!("Processing inner dictionary");
            for (inner_key, inner_value) in inner_dict.iter() {
                let child_idx = inner_key.extract::<String>()?.parse::<usize>()
                    .map_err(|_| PyValueError::new_err("Invalid node index"))?;
                let count = inner_value.extract::<usize>()?;
                
                println!("Processing child node index: {} with count: {}", child_idx, count);
                
                // Debug node before update
                if let Node::StandardNode { node_type, attributes, .. } = &graph[NodeIndex::new(child_idx)] {
                    println!("Before update - Node type: {}, Attributes: {:?}", node_type, attributes);
                    node_types_to_update.insert(node_type.clone());
                }
                
                match update_node_attribute(
                    graph,
                    child_idx,
                    attribute_name,
                    AttributeValue::Int(count as i32)
                ) {
                    Ok(_) => println!("Successfully updated child attribute"),
                    Err(e) => println!("Error updating child attribute: {:?}", e),
                }
                
                // Debug node after update
                if let Node::StandardNode { attributes, .. } = &graph[NodeIndex::new(child_idx)] {
                    println!("After update - Attributes: {:?}", attributes);
                }
            }
        }
    }
    
    println!("Node types to update: {:?}", node_types_to_update);
    
    for node_type in node_types_to_update {
        let mut schema_types = HashMap::new();
        schema_types.insert(attribute_name.to_string(), "Int".to_string());
        
        match update_or_retrieve_schema(
            graph,
            "Node",
            &node_type,
            Some(vec![attribute_name.to_string()]),
            Some(schema_types)
        ) {
            Ok(_) => println!("Successfully updated schema for type: {}", node_type),
            Err(e) => println!("Error updating schema: {:?}", e),
        }
    }
    
    println!("Store function completed");
    Ok(())
}