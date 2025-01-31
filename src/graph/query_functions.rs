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
use std::cmp::Ordering;

pub fn filter_nodes(
    graph: &DiGraph<Node, Relation>,
    indices: Option<Vec<usize>>,
    filter_dict: &PyDict,
) -> PyResult<Vec<usize>> {
    let py = unsafe { Python::assume_gil_acquired() };
    let mut result = Vec::new();
    let nodes_to_check = match indices {
        Some(idx) => idx.into_iter().map(NodeIndex::new).collect::<Vec<_>>(),
        None => graph.node_indices().collect(),
    };

    let mut filters = HashMap::new();
    for (key, value) in filter_dict.iter() {
        let key = key.extract::<String>()?;
        // Try to extract as PyDict first (new format), fall back to direct value (old format)
        let conditions = if let Ok(dict) = value.extract::<&PyDict>() {
            dict.into()
        } else {
            // For direct values, create a dict with equality comparison
            let simple_condition = PyDict::new(py);
            simple_condition.set_item("==", value)?;
            simple_condition
        };
        filters.insert(key, conditions);
    }

    for idx in nodes_to_check {
        if let Some(Node::StandardNode { node_type, unique_id, attributes, title }) = graph.node_weight(idx) {
            let mut matches = true;

            // Check each filter attribute
            for (key, conditions) in &filters {
                let node_value = match key.as_str() {
                    "type" | "node_type" => Some(AttributeValue::String(node_type.clone())),
                    "title" => title.clone().map(AttributeValue::String),
                    "unique_id" => Some(AttributeValue::Int(*unique_id)),
                    _ => attributes.get(key).cloned(),
                };

                if let Some(node_value) = node_value {
                    // Check all conditions for this attribute
                    for (op, filter_value) in conditions.iter() {
                        let op = op.extract::<String>()?;
                        
                        // Check if the filter value is a list
                        let mut condition_matches = false;
                        
                        if let Ok(value_list) = filter_value.extract::<Vec<PyObject>>() {
                            // If it's a list, any match makes the condition true
                            for value in value_list {
                                let comparison_result = compare_values(&node_value, &value.into_py(py))?;
                                if matches_operator(&comparison_result, &op) {
                                    condition_matches = true;
                                    break;
                                }
                            }
                        } else {
                            // Single value comparison
                            let comparison_result = compare_values(&node_value, &filter_value.into_py(py))?;
                            condition_matches = matches_operator(&comparison_result, &op);
                        }

                        if !condition_matches {
                            matches = false;
                            break;
                        }
                    }
                } else {
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

// Helper function to match operators
fn matches_operator(comparison: &Ordering, op: &str) -> bool {
    match op {
        "==" | "=" => *comparison == Ordering::Equal,
        "!=" | "<>" => *comparison != Ordering::Equal,
        ">=" | "=>" => *comparison != Ordering::Less,
        "<=" | "=<" => *comparison != Ordering::Greater,
        ">" => *comparison == Ordering::Greater,
        "<" => *comparison == Ordering::Less,
        _ => false,
    }
}

fn compare_values(node_value: &AttributeValue, filter_value: &PyObject) -> PyResult<Ordering> {
    let py = unsafe { Python::assume_gil_acquired() };
    
    match node_value {
        AttributeValue::String(s) => {
            let filter_str = filter_value.extract::<String>(py)?;
            Ok(s.cmp(&filter_str))
        }
        AttributeValue::Int(i) => {
            if let Ok(filter_int) = filter_value.extract::<i32>(py) {
                Ok(i.cmp(&filter_int))
            } else if let Ok(filter_float) = filter_value.extract::<f64>(py) {
                let node_float = *i as f64;
                compare_floats(node_float, filter_float)
            } else {
                Err(PyValueError::new_err("Invalid comparison value type"))
            }
        }
        AttributeValue::Float(f) => {
            if let Ok(filter_float) = filter_value.extract::<f64>(py) {
                compare_floats(*f, filter_float)
            } else if let Ok(filter_int) = filter_value.extract::<i32>(py) {
                compare_floats(*f, filter_int as f64)
            } else {
                Err(PyValueError::new_err("Invalid comparison value type"))
            }
        }
        AttributeValue::DateTime(dt) => {
            let filter_str = filter_value.extract::<String>(py)?;
            Ok(dt.to_string().cmp(&filter_str))
        }
    }
}

fn compare_floats(a: f64, b: f64) -> PyResult<Ordering> {
    if a.is_nan() || b.is_nan() {
        return Err(PyValueError::new_err("Cannot compare NaN values"));
    }
    
    match a.partial_cmp(&b) {
        Some(ordering) => Ok(ordering),
        None => Err(PyValueError::new_err("Invalid float comparison"))
    }
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