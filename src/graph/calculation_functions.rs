use petgraph::graph::{DiGraph, NodeIndex};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::PyValueError;
use std::collections::{HashMap, HashSet};

use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue;
use crate::graph::get_schema::update_or_retrieve_schema;
use crate::graph::TraversalContext;

fn get_float_value(value: &AttributeValue) -> Option<f64> {
    match value {
        AttributeValue::Int(i) => Some(*i as f64),
        AttributeValue::Float(f) => Some(*f),
        _ => None,
    }
}

pub fn calculate_aggregate(
    graph: &DiGraph<Node, Relation>,
    context: &TraversalContext,
    attribute: &str,
    operation: &str,
    quantile: Option<f64>,
    max_results: Option<usize>,
) -> PyResult<PyObject> {
    let py = unsafe { Python::assume_gil_acquired() };

    println!("Starting calculate_aggregate");
    println!("Attribute: {}, Operation: {}", attribute, operation);

    if context.levels.is_empty() {
        println!("Context levels empty, returning None");
        return Ok(py.None());
    }

    let last_level = context.levels.last().unwrap();
    println!("Last level nodes count: {}", last_level.nodes.len());
    println!("Last level relationships count: {}", last_level.node_relationships.len());
    
    // If we have relationships, calculate per parent's children
    if !last_level.node_relationships.is_empty() {
        println!("Processing relationships path");
        let result = PyDict::new(py);
        
        let parent_nodes: Vec<_> = if let Some(limit) = max_results {
            println!("Applying max_results limit: {}", limit);
            last_level.node_relationships.keys().take(limit).collect()
        } else {
            last_level.node_relationships.keys().collect()
        };
        println!("Number of parent nodes to process: {}", parent_nodes.len());

        for &parent_idx in parent_nodes {
            println!("\nProcessing parent node: {}", parent_idx);
            let mut values: Vec<f64> = Vec::new();
            
            if let Some(children) = last_level.node_relationships.get(&parent_idx) {
                println!("Found {} children for parent {}", children.len(), parent_idx);
                for &child_idx in children {
                    println!("Processing child node: {}", child_idx);
                    if let Some(Node::StandardNode { attributes, .. }) = graph.node_weight(NodeIndex::new(child_idx)) {
                        println!("Found attributes for child {}", child_idx);
                        if let Some(value) = attributes.get(attribute) {
                            println!("Found attribute '{}' with value: {:?}", attribute, value);
                            if let Some(num) = get_float_value(value) {
                                println!("Converted to float: {}", num);
                                values.push(num);
                            } else {
                                println!("Could not convert value to float");
                            }
                        } else {
                            println!("Attribute '{}' not found in child {}", attribute, child_idx);
                        }
                    } else {
                        println!("Could not find node or node is not StandardNode: {}", child_idx);
                    }
                }
            }
            
            println!("Collected {} values for parent {}", values.len(), parent_idx);
            let aggregate_value = calculate_result(&values, operation, quantile)?;
            println!("Calculated result for parent {}: {:?}", parent_idx, aggregate_value);
            result.set_item(parent_idx.to_string(), aggregate_value.map_or(py.None(), |v| v.into_py(py)))?;
        }
        
        println!("Returning results dict for relationships");
        Ok(result.into())
    } else {
        println!("Processing direct nodes path");
        let nodes = if let Some(limit) = max_results {
            println!("Applying max_results limit: {}", limit);
            last_level.nodes.iter().take(limit).copied().collect::<Vec<_>>()
        } else {
            last_level.nodes.clone()
        };
        println!("Processing {} nodes", nodes.len());

        let mut values: Vec<f64> = Vec::new();
        for node_idx in nodes {
            println!("\nProcessing node: {}", node_idx);
            if let Some(Node::StandardNode { attributes, .. }) = graph.node_weight(NodeIndex::new(node_idx)) {
                println!("Found attributes for node {}", node_idx);
                if let Some(value) = attributes.get(attribute) {
                    println!("Found attribute '{}' with value: {:?}", attribute, value);
                    if let Some(num) = get_float_value(value) {
                        println!("Converted to float: {}", num);
                        values.push(num);
                    } else {
                        println!("Could not convert value to float");
                    }
                } else {
                    println!("Attribute '{}' not found in node {}", attribute, node_idx);
                }
            } else {
                println!("Could not find node or node is not StandardNode: {}", node_idx);
            }
        }

        println!("Collected {} values total", values.len());
        let result = calculate_result(&values, operation, quantile)?;
        println!("Final result: {:?}", result);
        Ok(result.map_or(py.None(), |v| v.into_py(py)))
    }
}

// Helper function to calculate the result based on operation
fn calculate_result(values: &[f64], operation: &str, quantile: Option<f64>) -> PyResult<Option<f64>> {
    if values.is_empty() {
        return Ok(None);
    }

    let result = match operation {
        "sum" => Some(values.iter().sum()),
        "avg" => Some(values.iter().sum::<f64>() / values.len() as f64),
        "max" => Some(values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
        "min" => Some(values.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
        "median" => Some(calculate_median(values)),
        "mode" => Some(calculate_mode(values)),
        "std" => Some(calculate_std(values)),
        "var" => Some(calculate_variance(values)),
        "quantile" => {
            let q = quantile.unwrap_or(0.5);
            if q < 0.0 || q > 1.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Quantile must be between 0 and 1"
                ));
            }
            Some(calculate_quantile(values, q))
        },
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid operation: {}", operation)
        )),
    };

    Ok(result)
}

fn calculate_median(values: &[f64]) -> f64 {
    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let len = sorted_values.len();
    if len % 2 == 0 {
        (sorted_values[len/2 - 1] + sorted_values[len/2]) / 2.0
    } else {
        sorted_values[len/2]
    }
}

fn calculate_mode(values: &[f64]) -> f64 {
    let precision = 1e10;
    let mut counts = std::collections::HashMap::new();
    
    for &value in values {
        let key = (value * precision).round() as i64;
        *counts.entry(key).or_insert(0) += 1;
    }
    
    counts.into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(key, _)| (key as f64) / precision)
        .unwrap_or(0.0)
}

fn calculate_variance(values: &[f64]) -> f64 {
    let len = values.len() as f64;
    let mean = values.iter().sum::<f64>() / len;
    values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / len
}

fn calculate_std(values: &[f64]) -> f64 {
    let variance = calculate_variance(values);
    let std = variance.sqrt();
    std
}

fn calculate_quantile(values: &[f64], q: f64) -> f64 {
    if q < 0.0 || q > 1.0 {
        return 0.0;
    }
    
    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let pos = (sorted_values.len() as f64 - 1.0) * q;
    let floor = pos.floor() as usize;
    let ceil = pos.ceil() as usize;
    
    if floor == ceil {
        sorted_values[floor]
    } else {
        let weight = pos - floor as f64;
        (1.0 - weight) * sorted_values[floor] + weight * sorted_values[ceil]
    }
}

pub fn store_calculation_values(
    graph: &mut DiGraph<Node, Relation>,
    _context: &TraversalContext,
    calculation_name: &str,
    store_values: &PyObject,
) -> PyResult<()> {
    let py = unsafe { Python::assume_gil_acquired() };
    
    let dict = store_values.downcast::<PyDict>(py)?;
    let mut node_types_to_update = HashSet::new();
    
    for (key, value) in dict.iter() {
        let node_idx = key.extract::<String>()?.parse::<usize>()
            .map_err(|_| PyValueError::new_err("Invalid node index"))?;
            
        // Get the node's type for schema updates
        if let Some(Node::StandardNode { node_type, calculations, .. }) = graph.node_weight_mut(NodeIndex::new(node_idx)) {
            node_types_to_update.insert(node_type.clone());
            
            // Initialize calculations if None
            if calculations.is_none() {
                *calculations = Some(HashMap::new());
            }
            
            if let Some(calcs) = calculations {
                let attr_value = if let Ok(num) = value.extract::<f64>() {
                    AttributeValue::Float(num)
                } else if let Ok(num) = value.extract::<i32>() {
                    AttributeValue::Int(num)
                } else {
                    return Err(PyValueError::new_err("Unsupported value type for calculation"));
                };
                
                calcs.insert(calculation_name.to_string(), attr_value);
            }
        }
    }
    
    // Update schema for affected node types
    for node_type in node_types_to_update {
        let mut schema_types = HashMap::new();
        schema_types.insert(calculation_name.to_string(), "Float".to_string());
        
        update_or_retrieve_schema(
            graph,
            "Node",
            &node_type,
            Some(schema_types)
        )?;
    }
    Ok(())
}

pub fn process_calculation_levels(
    graph: &DiGraph<Node, Relation>,
    context: &TraversalContext,
    calculation_names: Option<Vec<String>>,
    max_results: Option<usize>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let mut result = Vec::new();
        
        for level in &context.levels {
            let mut level_dict = HashMap::new();
            
            for &node_idx in &level.nodes {
                if let Some(max) = max_results {
                    if node_idx >= max {
                        continue;
                    }
                }
                
                if let Some(Node::StandardNode { calculations, .. }) = graph.node_weight(NodeIndex::new(node_idx)) {
                    if let Some(calcs) = calculations {
                        let mut node_calcs = HashMap::new();
                        
                        match &calculation_names {
                            Some(names) => {
                                for name in names {
                                    if let Some(value) = calcs.get(name) {
                                        node_calcs.insert(name.clone(), value.to_python_object(py, None)?);
                                    }
                                }
                            },
                            None => {
                                for (name, value) in calcs {
                                    node_calcs.insert(name.clone(), value.to_python_object(py, None)?);
                                }
                            }
                        }
                        
                        if !node_calcs.is_empty() {
                            level_dict.insert(node_idx.to_string(), node_calcs);
                        }
                    }
                }
            }
            
            if !level_dict.is_empty() {
                result.push(level_dict);
            }
        }
        
        Ok(result.into_py(py))
    })
}