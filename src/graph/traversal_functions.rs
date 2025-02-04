use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::{HashMap, HashSet};
use serde_json::Value as JsonValue;
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
    pub selection_params: Option<Vec<HashMap<String, JsonValue>>>,  // Changed to support array of dicts
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
        context.add_level(nodes, None, false);  // Remove the extra None argument
        context
    }

    pub fn add_level(
        &mut self,
        nodes: Vec<usize>,
        selection_params: Option<Vec<HashMap<String, JsonValue>>>,
        new_level: bool,
    ) {
        let level = if new_level {
            0
        } else {
            self.levels.last().map_or(0, |_| self.levels.len())
        };

        // Clear subsequent levels if this is a new traversal sequence
        if new_level {
            self.levels.clear();
        }

        let parent_node = if level == 0 {
            None
        } else {
            self.levels.last().and_then(|l| l.nodes.first()).copied()
        };

        let node_relationships = HashMap::new();
        self.levels.push(TraversalLevel {
            nodes,
            parent_node,
            selection_params,
            node_relationships,
        });
    }

    pub fn add_relationships(&mut self, relationships: HashMap<usize, Vec<usize>>) {
        if let Some(last_level) = self.levels.last_mut() {
            // Only update nodes if they're empty
            if last_level.nodes.is_empty() {
                let all_nodes: HashSet<usize> = relationships
                    .iter() // Use iter() instead of consuming relationships
                    .flat_map(|(_source, targets)| targets)
                    .copied()
                    .collect();
                last_level.nodes = all_nodes.into_iter().collect();
            }
            last_level.node_relationships = relationships;
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
    max_traversals: Option<usize>,
) -> PyResult<HashMap<usize, Vec<usize>>> {
    // Define which directions to check based on input
    let directions_to_check = match direction.as_deref() {
        Some("incoming") => vec![Direction::Incoming],
        Some("outgoing") => vec![Direction::Outgoing],
        Some(d) => return Err(PyValueError::new_err(format!("Invalid direction: {}", d))),
        None => vec![Direction::Incoming, Direction::Outgoing],
    };
    
    let mut relationships = HashMap::new();
    
    for &node_idx in start_nodes {
        let node_index = NodeIndex::new(node_idx);
        let mut node_traversals = Vec::new();
        
        // Check each specified direction
        for &dir in &directions_to_check {
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
    
    // Validate relationships before returning
    if relationships.is_empty() && !start_nodes.is_empty() {
        relationships = start_nodes
            .iter()
            .map(|&node| (node, Vec::new()))
            .collect();
    }
    
    Ok(relationships)
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

pub fn process_node_levels(
    graph: &DiGraph<Node, Relation>,
    context: &TraversalContext,
    include_attributes: bool,
    include_calculations: bool,
    include_connections: bool,
    only_return: Option<&[String]>,
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
            let mut node_data = HashMap::new();
            
            // Add base fields
            if let Node::StandardNode { node_type, unique_id, title, .. } = node {
                node_data.insert("node_type".to_string(), node_type.clone().into_py(py));
                node_data.insert("unique_id".to_string(), (*unique_id).into_py(py));
                if let Some(t) = title {
                    node_data.insert("title".to_string(), t.clone().into_py(py));
                }
            }

            // Add attributes if requested
            if include_attributes {
                if let Node::StandardNode { attributes, .. } = node {
                    let mut py_attrs: HashMap<String, PyObject> = HashMap::new();
                    for (k, v) in attributes {  // Direct iteration since attributes is not Optional
                        match v.to_python_object(py, None) {
                            Ok(py_value) => {
                                py_attrs.insert(k.clone(), py_value);
                            },
                            Err(e) => {
                                eprintln!("Warning: Failed to convert attribute {}: {}", k, e);
                                continue;
                            }
                        }
                    }
                    if !py_attrs.is_empty() {
                        node_data.insert("attributes".to_string(), py_attrs.into_py(py));
                    }
                }
            }

            // Add calculations if requested
            if include_calculations {
                if let Node::StandardNode { calculations, .. } = node {
                    if let Some(calcs) = calculations {  // Need Option handling for calculations
                        let mut py_calcs: HashMap<String, PyObject> = HashMap::new();
                        for (k, v) in calcs {
                            match v.to_python_object(py, None) {
                                Ok(py_value) => {
                                    py_calcs.insert(k.clone(), py_value);
                                },
                                Err(e) => {
                                    eprintln!("Warning: Failed to convert calculation {}: {}", k, e);
                                    continue;
                                }
                            }
                        }
                        if !py_calcs.is_empty() {
                            node_data.insert("calculations".to_string(), py_calcs.into_py(py));
                        }
                    }
                }
            }

            // Add connections if requested
            if include_connections {
                let node_idx = NodeIndex::new(root_idx);
                let mut incoming = Vec::new();
                let mut outgoing = Vec::new();

                // Process incoming edges
                for edge_ref in graph.edges_directed(node_idx, Direction::Incoming) {
                    if let Some(source_node) = graph.node_weight(edge_ref.source()) {
                        let mut connection = HashMap::new();
                        connection.insert("type".to_string(), edge_ref.weight().relation_type.clone());
                        if let Node::StandardNode { title, unique_id, .. } = source_node {
                            connection.insert("source_title".to_string(), title.clone().unwrap_or_default());
                            connection.insert("source_id".to_string(), unique_id.to_string());
                            connection.insert("source_idx".to_string(), edge_ref.source().index().to_string());
                        }
                        incoming.push(connection);
                    }
                }

                // Process outgoing edges
                for edge_ref in graph.edges_directed(node_idx, Direction::Outgoing) {
                    if let Some(target_node) = graph.node_weight(edge_ref.target()) {
                        let mut connection = HashMap::new();
                        connection.insert("type".to_string(), edge_ref.weight().relation_type.clone());
                        if let Node::StandardNode { title, unique_id, .. } = target_node {
                            connection.insert("target_title".to_string(), title.clone().unwrap_or_default());
                            connection.insert("target_id".to_string(), unique_id.to_string());
                            connection.insert("target_idx".to_string(), edge_ref.target().index().to_string());
                        }
                        outgoing.push(connection);
                    }
                }

                if !incoming.is_empty() {
                    node_data.insert("incoming_connections".to_string(), incoming.into_py(py));
                }
                if !outgoing.is_empty() {
                    node_data.insert("outgoing_connections".to_string(), outgoing.into_py(py));
                }
            }

            // Handle traversal data
            if context.levels.len() > 1 && 
               context.levels[1].node_relationships.contains_key(&root_idx) &&
               !context.levels[1].node_relationships[&root_idx].is_empty() {
                
                let mut traversal_data = Vec::new();
                
                if let Some(child_nodes) = context.levels[1].node_relationships.get(&root_idx) {
                    for &child_idx in child_nodes {
                        if let Some(child_node) = graph.node_weight(NodeIndex::new(child_idx)) {
                            // Recursively process child nodes with same settings
                            let mut child_context = TraversalContext::new_base();
                            child_context.add_level(vec![child_idx], None, false);
                            
                            if let Ok(child_result) = process_node_levels(
                                graph,
                                &child_context,
                                include_attributes,
                                include_calculations,
                                include_connections,
                                None,
                                None
                            ) {
                                if let Ok(child_list) = <Vec<PyObject>>::extract(child_result.as_ref(py)) {
                                    if !child_list.is_empty() {
                                        traversal_data.extend(child_list);
                                    }
                                }
                            }
                        }
                    }
                }
                
                if !traversal_data.is_empty() {
                    node_data.insert("traversals".to_string(), traversal_data.into_py(py));
                }
            }

            // Handle only_return if specified
            if let Some(fields) = only_return {
                let mut filtered_data = HashMap::new();
                for field in fields {
                    match node_data.get(field) {
                        Some(value) => { filtered_data.insert(field.to_string(), value.clone()); },
                        None => {
                            // Check in attributes
                            if let Some(attrs) = node_data.get("attributes") {
                                if let Ok(attrs_dict) = attrs.extract::<&PyDict>(py) {
                                    if let Ok(Some(value)) = attrs_dict.get_item(field) {
                                        filtered_data.insert(field.to_string(), value.into_py(py));
                                        continue;
                                    }
                                }
                            }
                            // Check in calculations
                            if let Some(calcs) = node_data.get("calculations") {
                                if let Ok(calcs_dict) = calcs.extract::<&PyDict>(py) {
                                    if let Ok(Some(value)) = calcs_dict.get_item(field) {
                                        filtered_data.insert(field.to_string(), value.into_py(py));
                                    }
                                }
                            }
                        }
                    }
                }
                result.push(filtered_data.into_py(py));
            } else {
                result.push(node_data.into_py(py));
            }
        }
    }

    // Return single object if only_return is specified and we have exactly one result
    if only_return.is_some() && result.len() == 1 {
        Ok(result[0].clone())
    } else {
        Ok(result.into_py(py))
    }
}