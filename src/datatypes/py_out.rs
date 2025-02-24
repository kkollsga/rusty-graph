// src/datatypes/py_out.rs
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use super::values::Value;
use crate::graph::calculations::StatResult;
use crate::graph::schema::NodeInfo;
use crate::graph::statistics_methods::PropertyStats;
use crate::graph::data_retrieval::{LevelNodes, LevelValues, LevelConnections, UniqueValues};

pub fn nodeinfo_to_pydict(py: Python, node: &NodeInfo) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", node.id.to_object(py))?;
    dict.set_item("title", node.title.to_object(py))?;
    dict.set_item("type", node.node_type.to_object(py))?;

    let props = PyDict::new_bound(py);
    for (k, v) in &node.properties {
        props.set_item(k, value_to_py(py, v)?)?;
    }
    dict.set_item("properties", props)?;

    Ok(dict.into())
}

pub fn value_to_py(py: Python, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::String(s) => Ok(s.clone().into_py(py)),
        Value::Float64(f) => Ok(f.into_py(py)),
        Value::Int64(i) => Ok(i.into_py(py)),
        Value::Boolean(b) => Ok(b.into_py(py)),
        Value::UniqueId(u) => Ok(u.into_py(py)),
        Value::DateTime(d) => Ok(d.format("%Y-%m-%d").to_string().into_py(py)),
        Value::Null => Ok(py.None()),
    }
}

pub fn convert_stats_for_python(stats: Vec<PropertyStats>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        
        let parent_idx = PyList::empty_bound(py);
        let parent_type = PyList::empty_bound(py);
        let parent_title = PyList::empty_bound(py);
        let parent_id = PyList::empty_bound(py);
        let property_name = PyList::empty_bound(py);
        let value_type = PyList::empty_bound(py);
        let children = PyList::empty_bound(py);
        let count = PyList::empty_bound(py);
        let valid_count = PyList::empty_bound(py);
        let sum_val = PyList::empty_bound(py);
        let avg = PyList::empty_bound(py);
        let min_val = PyList::empty_bound(py);
        let max_val = PyList::empty_bound(py);

        for stat in stats {
            parent_idx.append(stat.parent_idx.map(|idx| idx.index().into_py(py))
                                           .unwrap_or_else(|| py.None()))?;
            parent_type.append(stat.parent_type.unwrap_or_default())?;
            parent_title.append(stat.parent_title.map_or_else(|| py.None(), |v| value_to_py(py, &v).unwrap()))?;
            parent_id.append(stat.parent_id.map_or_else(|| py.None(), |v| value_to_py(py, &v).unwrap()))?;
            property_name.append(stat.property_name)?;
            value_type.append(stat.value_type)?;
            children.append(stat.children)?;
            count.append(stat.count)?;
            valid_count.append(stat.valid_count)?;
            
            if stat.is_numeric {
                sum_val.append(stat.sum.map(|v| v.into_py(py)).unwrap_or_else(|| py.None()))?;
                avg.append(stat.avg.map(|v| v.into_py(py)).unwrap_or_else(|| py.None()))?;
                min_val.append(stat.min.map(|v| v.into_py(py)).unwrap_or_else(|| py.None()))?;
                max_val.append(stat.max.map(|v| v.into_py(py)).unwrap_or_else(|| py.None()))?;
            } else {
                sum_val.append(py.None())?;
                avg.append(py.None())?;
                min_val.append(py.None())?;
                max_val.append(py.None())?;
            }
        }

        dict.set_item("parent_idx", parent_idx)?;
        dict.set_item("parent_type", parent_type)?;
        dict.set_item("parent_title", parent_title)?;
        dict.set_item("parent_id", parent_id)?;
        dict.set_item("property", property_name)?;
        dict.set_item("value_type", value_type)?;
        dict.set_item("children_count", children)?;
        dict.set_item("property_count", count)?;
        dict.set_item("valid_count", valid_count)?;
        dict.set_item("sum", sum_val)?;
        dict.set_item("avg", avg)?;
        dict.set_item("min", min_val)?;
        dict.set_item("max", max_val)?;
        
        Ok(dict.into())
    })
}

pub fn level_nodes_to_pydict(
    py: Python,
    level_nodes: &[LevelNodes],
    parent_key: Option<&str>,
    parent_info: Option<bool>
) -> PyResult<PyObject> {
    let result = PyDict::new_bound(py);
    let mut seen_keys = std::collections::HashMap::new();
    
    for group in level_nodes {
        let base_key = match parent_key {
            Some("idx") => {
                if let Some(idx) = group.parent_idx {
                    format!("{}", idx.index())
                } else {
                    String::from("no_idx")
                }
            },
            Some("id") => {
                if let Some(ref id) = group.parent_id {
                    match id {
                        Value::String(s) => s.clone(),
                        Value::Int64(i) => i.to_string(),
                        Value::Float64(f) => f.to_string(),
                        Value::UniqueId(u) => u.to_string(),
                        _ => format!("{:?}", id)
                    }
                } else {
                    String::from("no_id")
                }
            },
            _ => {
                if !group.parent_title.is_empty() {
                    group.parent_title.clone()
                } else {
                    String::from("no_title")
                }
            },
        };

        let key = {
            let count = seen_keys.entry(base_key.clone()).or_insert(0);
            *count += 1;
            
            if *count > 1 {
                format!("{}_{}", base_key, count)
            } else {
                base_key
            }
        };

        let value = if parent_info.unwrap_or(false) && group.parent_idx.is_some() {
            let parent_dict = PyDict::new_bound(py);
            
            if let Some(ref id) = group.parent_id {
                parent_dict.set_item("id", value_to_py(py, id)?)?;
            }
            parent_dict.set_item("title", &group.parent_title)?;
            if let Some(ref type_str) = group.parent_type {
                parent_dict.set_item("type", type_str)?;
            }
            
            let nodes: Vec<PyObject> = group.nodes.iter()
                .map(|node| nodeinfo_to_pydict(py, node))
                .collect::<PyResult<_>>()?;
            parent_dict.set_item("children", nodes)?;
            
            parent_dict.into()
        } else {
            let nodes: Vec<PyObject> = group.nodes.iter()
                .map(|node| nodeinfo_to_pydict(py, node))
                .collect::<PyResult<_>>()?;
            nodes.into_py(py)
        };
        
        result.set_item(key, value)?;
    }
    
    Ok(result.into())
}

pub fn level_values_to_pydict(py: Python, level_values: &[LevelValues]) -> PyResult<PyObject> {
    let result = PyDict::new_bound(py);
    
    for group in level_values {
        let values: Vec<PyObject> = group.values.iter()
            .map(|vec_values| {
                let tuple_values: Vec<PyObject> = vec_values.iter()
                    .map(|v| value_to_py(py, v))
                    .collect::<PyResult<_>>()?;
                Ok(PyTuple::new_bound(py, &tuple_values).into())
            })
            .collect::<PyResult<_>>()?;
            
        result.set_item(&group.parent_title, values)?;
    }
    
    Ok(result.into())
}

pub fn level_single_values_to_pydict(py: Python, level_values: &[LevelValues]) -> PyResult<PyObject> {
    let result = PyDict::new_bound(py);
    
    for group in level_values {
        let values: Vec<PyObject> = group.values.iter()
            .map(|vec_values| {
                value_to_py(py, &vec_values[0])
            })
            .collect::<PyResult<_>>()?;
            
        result.set_item(&group.parent_title, values)?;
    }
    
    Ok(result.into())
}

pub fn level_connections_to_pydict(
    py: Python,
    connections: &[LevelConnections],
    parent_info: Option<bool>
) -> PyResult<PyObject> {
    let result = PyDict::new_bound(py);
    
    for level in connections {
        let group_dict = PyDict::new_bound(py);
        
        if parent_info.unwrap_or(false) {
            if let Some(parent_id) = &level.parent_id {
                group_dict.set_item("parent_id", parent_id.to_object(py))?;
            }
            if let Some(parent_type) = &level.parent_type {
                group_dict.set_item("parent_type", parent_type)?;
            }
            if let Some(parent_idx) = &level.parent_idx {
                group_dict.set_item("parent_idx", parent_idx.index())?;
            }
        }
        
        let connections_dict = PyDict::new_bound(py);
        for conn in &level.connections {
            let node_dict = PyDict::new_bound(py);
            node_dict.set_item("node_id", conn.node_id.to_object(py))?;
            node_dict.set_item("node_type", &conn.node_type)?;
            
            let incoming_dict = PyDict::new_bound(py);
            for (conn_type, id, title, conn_props, node_props) in &conn.incoming {
                if !incoming_dict.contains(conn_type)? {
                    incoming_dict.set_item(conn_type, PyDict::new_bound(py))?;
                }
                
                let conn_type_item = incoming_dict.get_item(conn_type)?;
                let conn_type_any = conn_type_item.unwrap();
                let conn_type_dict = conn_type_any.downcast::<PyDict>()?;
                
                let node_info = PyDict::new_bound(py);
                node_info.set_item("node_id", id.to_object(py))?;
                node_info.set_item("connection_properties", conn_props)?;
                if let Some(props) = node_props {
                    node_info.set_item("node_properties", props)?;
                }
                
                match title {
                    Value::String(t) => conn_type_dict.set_item(t, node_info)?,
                    _ => conn_type_dict.set_item("Unknown", node_info)?,
                }
            }
            node_dict.set_item("incoming", incoming_dict)?;
            
            let outgoing_dict = PyDict::new_bound(py);
            for (conn_type, id, title, conn_props, node_props) in &conn.outgoing {
                if !outgoing_dict.contains(conn_type)? {
                    outgoing_dict.set_item(conn_type, PyDict::new_bound(py))?;
                }
                
                let conn_type_item = outgoing_dict.get_item(conn_type)?;
                let conn_type_any = conn_type_item.unwrap();
                let conn_type_dict = conn_type_any.downcast::<PyDict>()?;
                
                let node_info = PyDict::new_bound(py);
                node_info.set_item("node_id", id.to_object(py))?;
                node_info.set_item("connection_properties", conn_props)?;
                if let Some(props) = node_props {
                    node_info.set_item("node_properties", props)?;
                }
                
                match title {
                    Value::String(t) => conn_type_dict.set_item(t, node_info)?,
                    _ => conn_type_dict.set_item("Unknown", node_info)?,
                }
            }
            node_dict.set_item("outgoing", outgoing_dict)?;
            
            connections_dict.set_item(&conn.node_title, node_dict)?;
        }
        group_dict.set_item("connections", connections_dict)?;
        
        result.set_item(&level.parent_title, group_dict)?;
    }
    
    Ok(result.to_object(py))
}

pub fn level_unique_values_to_pydict(py: Python, values: &[UniqueValues]) -> PyResult<PyObject> {
    let result = PyDict::new_bound(py);
    for unique_values in values {
        let py_values: Vec<PyObject> = unique_values.values.iter()
            .map(|v| v.to_object(py))
            .collect();
        result.set_item(&unique_values.parent_title, PyList::new_bound(py, &py_values))?;
    }
    Ok(result.to_object(py))
}

pub fn convert_computation_results_for_python(results: Vec<StatResult>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        
        // Convert and insert each result within the GIL scope
        for result in results {
            let key = result.parent_title.unwrap_or_else(|| "node".to_string());
            
            match result.error_msg {
                Some(error) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(error));
                },
                None => {
                    // For non-error cases, handle the value
                    match result.value {
                        Value::Null => continue,  // Skip null values
                        Value::Int64(i) => dict.set_item(key, i)?,
                        Value::Float64(f) => dict.set_item(key, f)?,
                        Value::UniqueId(u) => dict.set_item(key, u)?,
                        // Other types should have been converted to numeric during evaluation
                        _ => continue,
                    }
                }
            }
        }

        // If we just have a single value under 'node', return the value directly
        if dict.len() == 1 {
            if let Some(value) = dict.get_item("node")? {
                return Ok(value.into());
            }
        }

        Ok(dict.into())
    })
}