use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict}; 
use pyo3::exceptions::PyValueError;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue;
use crate::graph::types::DataInput;
use std::collections::HashMap;

pub fn filter_nodes(
    graph: &DiGraph<Node, Relation>,
    indices: Option<Vec<usize>>,
    filter_dict: &PyDict,
) -> PyResult<Vec<usize>> {
    let mut result = Vec::new();
    let nodes_to_check = match indices {
        Some(idx) => idx.into_iter().map(NodeIndex::new).collect::<Vec<_>>(),
        None => graph.node_indices().collect(),
    };

    let mut filters = HashMap::new();
    for (key, value) in filter_dict.iter() {
        let key = key.extract::<String>()?;
        let value = value.extract::<String>()?;
        filters.insert(key, value);
    }

    for idx in nodes_to_check {
        if let Some(Node::StandardNode { node_type, unique_id, attributes, title }) = graph.node_weight(idx) {
            let mut matches = true;

            for (key, value) in &filters {
                let matches_filter = match key.as_str() {
                    "type" | "node_type" => node_type == value,
                    "title" => title.as_ref().map_or(false, |t| t == value),
                    "unique_id" => unique_id.to_string() == *value,
                    _ => attributes.get(key).map_or(false, |attr| match attr {
                        AttributeValue::String(s) => s == value,
                        AttributeValue::Int(i) => i.to_string() == *value,
                        AttributeValue::Float(f) => f.to_string() == *value,
                        AttributeValue::DateTime(dt) => dt.to_string() == *value,
                    }),
                };

                if !matches_filter {
                    matches = false;
                    break;
                }
            }

            if matches {
                result.push(idx.index());
            }
        }
    }

    Ok(result)
}

pub fn get_simplified_relationships(
    graph: &DiGraph<Node, Relation>,
    indices: Vec<usize>,
) -> PyResult<Vec<HashMap<String, PyObject>>> {
    let py = unsafe { Python::assume_gil_acquired() };
    let mut result = Vec::new();

    for idx in indices {
        let node_idx = NodeIndex::new(idx);
        if let Some(Node::StandardNode { node_type, title, .. }) = graph.node_weight(node_idx) {
            let mut node_data = HashMap::new();
            
            // Add base node info
            node_data.insert("type".to_string(), node_type.clone().into_py(py));
            node_data.insert("title".to_string(), title.clone().unwrap_or_default().into_py(py));
            
            // Process incoming relationships
            let incoming: Vec<HashMap<String, String>> = graph.edges_directed(node_idx, Direction::Incoming)
                .filter_map(|edge| {
                    if let Some(Node::StandardNode { node_type, title, .. }) = graph.node_weight(edge.source()) {
                        let mut rel = HashMap::new();
                        rel.insert("type".to_string(), node_type.clone());
                        rel.insert("title".to_string(), title.clone().unwrap_or_default());
                        rel.insert("relationship_type".to_string(), edge.weight().relation_type.clone());
                        Some(rel)
                    } else {
                        None
                    }
                })
                .collect();
            
            // Process outgoing relationships
            let outgoing: Vec<HashMap<String, String>> = graph.edges_directed(node_idx, Direction::Outgoing)
                .filter_map(|edge| {
                    if let Some(Node::StandardNode { node_type, title, .. }) = graph.node_weight(edge.target()) {
                        let mut rel = HashMap::new();
                        rel.insert("type".to_string(), node_type.clone());
                        rel.insert("title".to_string(), title.clone().unwrap_or_default());
                        rel.insert("relationship_type".to_string(), edge.weight().relation_type.clone());
                        Some(rel)
                    } else {
                        None
                    }
                })
                .collect();

            // Only add relationships if they exist
            if !incoming.is_empty() {
                node_data.insert("incoming".to_string(), incoming.into_py(py));
            }
            if !outgoing.is_empty() {
                node_data.insert("outgoing".to_string(), outgoing.into_py(py));
            }
            
            result.push(node_data);
        }
    }

    Ok(result)
}

pub fn sort_nodes(
    graph: &DiGraph<Node, Relation>, 
    mut nodes: Vec<usize>,
    sort_fn: impl Fn(&usize, &usize) -> std::cmp::Ordering
) -> Vec<usize> {
    if nodes.is_empty() {
        nodes = graph.node_indices().map(|n| n.index()).collect();
    }
    nodes.sort_by(|a, b| sort_fn(a, b));
    nodes
}

pub fn extract_dataframe_content(df: &PyAny) -> PyResult<DataInput> {
    let values = match df.call_method0("to_numpy") {
        Ok(v) => v,
        Err(e) => return Err(PyErr::new::<PyValueError, _>(
            format!("Failed to convert DataFrame to numpy array: {}", e)
        )),
    };
    
    let values_list = match values.call_method0("tolist") {
        Ok(v) => v,
        Err(e) => return Err(PyErr::new::<PyValueError, _>(
            format!("Failed to convert numpy array to list: {}", e)
        )),
    };

    let columns = match df.getattr("columns")?.call_method0("tolist") {
        Ok(v) => v.extract()?,
        Err(e) => return Err(PyErr::new::<PyValueError, _>(
            format!("Failed to get columns: {}", e)
        )),
    };
    
    Ok(DataInput {
        data: values_list.downcast::<PyList>()?.into(),
        columns
    })
}