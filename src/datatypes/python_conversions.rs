// src/datatypes/python_conversions.rs
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use super::values::{DataFrame, ColumnType, ColumnData, Value, FilterCondition};
use super::type_conversions::{to_u32, to_i64, to_f64, to_datetime, to_bool};
use crate::graph::schema::NodeInfo;
use crate::graph::statistics_methods::PropertyStats;
use crate::graph::data_retrieval::{LevelNodes, LevelValues, LevelConnections};

pub fn pydict_to_filter_conditions(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, FilterCondition>> {
    dict.iter()
        .map(|(k, v)| {
            let key = k.extract::<String>()?;
            let condition = match v.downcast::<PyDict>() {
                // Handle operator-based condition
                Ok(op_dict) => {
                    let (op, val) = op_dict.iter().next()
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Empty operator dictionary"
                        ))?;

                    let op_str = op.extract::<String>()?;
                    parse_operator_condition(&op_str, &val)?
                },
                // Handle direct value match (equivalent to ==)
                Err(_) => FilterCondition::Equals(py_value_to_value(&v)?),
            };
            
            Ok((key, condition))
        })
        .collect()
}

fn parse_operator_condition(op: &str, val: &Bound<'_, PyAny>) -> PyResult<FilterCondition> {
    match op {
        "==" => {
            // Check if the value is a list
            if val.downcast::<PyList>().is_ok() {
                // If it's a list, treat it as an IN condition
                parse_in_condition(val)
            } else {
                // Otherwise, normal equality
                Ok(FilterCondition::Equals(py_value_to_value(val)?))
            }
        },
        "!=" => Ok(FilterCondition::NotEquals(py_value_to_value(val)?)),
        ">" => Ok(FilterCondition::GreaterThan(py_value_to_value(val)?)),
        ">=" => Ok(FilterCondition::GreaterThanEquals(py_value_to_value(val)?)),
        "<" => Ok(FilterCondition::LessThan(py_value_to_value(val)?)),
        "<=" => Ok(FilterCondition::LessThanEquals(py_value_to_value(val)?)),
        "in" => parse_in_condition(val),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unsupported operator: {}", op)
        )),
    }
}

fn parse_in_condition(val: &Bound<'_, PyAny>) -> PyResult<FilterCondition> {
    match val.downcast::<PyList>() {
        Ok(list) => {
            let values: PyResult<Vec<Value>> = list.iter()
                .map(|item| py_value_to_value(&item))
                .collect();
            Ok(FilterCondition::In(values?))
        },
        Err(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "'in' operator requires a list value"
        )),
    }
}

fn convert_pandas_series(series: &Bound<'_, PyAny>, col_type: ColumnType) -> PyResult<ColumnData> {
    let length = series.len()?;

    match col_type {
        ColumnType::UniqueId => {
            let mut vec = Vec::with_capacity(length);
            for i in 0..length {
                let value = series.get_item(i)?;
                // Simply push None for null values, regardless of whether it's a unique ID
                vec.push(to_u32(&value));
            }
            Ok(ColumnData::UniqueId(vec))
        },
        // Rest of the match arms remain the same...
        ColumnType::Int64 => {
            let mut vec = Vec::with_capacity(length);
            for i in 0..length {
                vec.push(to_i64(&series.get_item(i)?));
            }
            Ok(ColumnData::Int64(vec))
        },
        ColumnType::Float64 => {
            let mut vec = Vec::with_capacity(length);
            for i in 0..length {
                vec.push(to_f64(&series.get_item(i)?));
            }
            Ok(ColumnData::Float64(vec))
        },
        ColumnType::String => {
            let mut vec = Vec::with_capacity(length);
            for i in 0..length {
                let value = series.get_item(i)?;
                vec.push(if value.is_none() {
                    None
                } else {
                    value.str().ok().map(|s| s.to_string())
                });
            }
            Ok(ColumnData::String(vec))
        },
        ColumnType::Boolean => {
            let mut vec = Vec::with_capacity(length);
            for i in 0..length {
                vec.push(to_bool(&series.get_item(i)?));
            }
            Ok(ColumnData::Boolean(vec))
        },
        ColumnType::DateTime => {
            let mut vec = Vec::with_capacity(length);
            for i in 0..length {
                let value = series.get_item(i)?;
                vec.push(to_datetime(&value));
            }
            Ok(ColumnData::DateTime(vec))
        },
    }
}

pub fn pandas_to_dataframe(
    df: &Bound<'_, PyAny>,
    unique_id_fields: &[String],
    columns: Option<&[String]>,
) -> PyResult<DataFrame> {
    let df_columns = df.getattr("columns")?;
    let column_names: Vec<String> = df_columns.extract()?;

    let filtered_column_names = match columns {
        Some(cols) => {
            column_names.into_iter().filter(|col_name| cols.contains(col_name)).collect()
        },
        None => column_names,
    };

    let mut df_out = DataFrame::new(Vec::new());

    Python::with_gil(|_py| {
        for col_name in &filtered_column_names {
            let series = df.getattr(col_name.as_str())?;
            let dtype = series.getattr("dtype")?;
            let type_str = dtype.str()?.to_string();

            let col_type = if unique_id_fields.contains(col_name) {
                ColumnType::UniqueId
            } else {
                match type_str.as_str() {
                    "int64" | "int32" | "int16" | "int8" => ColumnType::Int64,
                    "float64" | "float32" => ColumnType::Float64,
                    "bool" | "boolean" => ColumnType::Boolean,
                    s if s.starts_with("datetime64") => ColumnType::DateTime,
                    "object" | "string" => ColumnType::String,
                    _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Unsupported column type {} for column {}", type_str, col_name)
                    )),
                }
            };

            let data = convert_pandas_series(&series, col_type.clone())?;
            df_out.add_column(col_name.clone(), col_type, data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        }
        Ok(())
    })?;

    Ok(df_out)
}



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

pub fn ensure_columns(
    columns: Option<&Bound<'_, PyList>>,
    unique_id_field: &str,
    node_title_field: &Option<String>,
) -> PyResult<Option<Vec<String>>> {
    match columns {
        Some(cols) => {
            let mut cols_vec: Vec<String> = cols.iter()
                .map(|item| item.extract::<String>())
                .collect::<PyResult<_>>()?;
            if !cols_vec.contains(&unique_id_field.to_string()) {
                cols_vec.push(unique_id_field.to_string());
            }
            if let Some(title_field) = node_title_field {
                if !cols_vec.contains(title_field) {
                    cols_vec.push(title_field.clone());
                }
            }
            Ok(Some(cols_vec))
        },
        None => Ok(None),
    }
}

pub fn convert_stats_for_python(
    stats: Vec<PropertyStats>
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        
        // Initialize lists for each column
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

        // Fill lists with data
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

        // Add all lists to the dictionary
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

pub fn parse_sort_fields(
    sort_spec: &Bound<'_, PyAny>, 
    ascending: Option<bool>
) -> PyResult<Vec<(String, bool)>> {
    // Single field with explicit direction
    if let Some(asc) = ascending {
        return match sort_spec.extract::<String>() {
            Ok(field) => Ok(vec![(field, asc)]),
            Err(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "When providing direction, sort_spec must be a string"
            ))
        };
    }
    
    parse_complex_sort_spec(sort_spec)
}

pub fn parse_complex_sort_spec(sort_spec: &Bound<'_, PyAny>) -> PyResult<Vec<(String, bool)>> {
    // Single field (ascending)
    if let Ok(field) = sort_spec.extract::<String>() {
        return Ok(vec![(field, true)]);
    }
    
    // Single (field, direction) tuple
    if let Ok((field, ascending)) = sort_spec.extract::<(String, bool)>() {
        return Ok(vec![(field, ascending)]);
    }
    
    // List of (field, direction) tuples
    let sort_list = sort_spec.downcast::<PyList>()?;
    let sort_fields: Vec<(String, bool)> = sort_list.iter()
        .map(|item| item.extract::<(String, bool)>())
        .collect::<PyResult<_>>()?;
    
    if sort_fields.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Sort specification cannot be empty"
        ));
    }
    
    Ok(sort_fields)
}


pub fn py_value_to_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }

    if let Ok(s) = value.extract::<String>() {
        Ok(Value::String(s))
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(Value::Float64(f))
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(Value::Int64(i))
    } else if let Ok(i) = value.extract::<i32>() {
        Ok(Value::Int64(i64::from(i)))
    } else if let Ok(b) = value.extract::<bool>() {
        Ok(Value::Boolean(b))
    } else if let Ok(u) = value.extract::<u32>() {
        Ok(Value::UniqueId(u))
    } else {
        Ok(Value::Null)
    }
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
        // Get base key based on preference
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

        // Handle duplicate keys by adding numeric suffix
        let key = {
            let count = seen_keys.entry(base_key.clone()).or_insert(0);
            *count += 1;
            
            if *count > 1 {
                format!("{}_{}", base_key, count)
            } else {
                base_key
            }
        };

        // Create dictionary value based on parent_info parameter
        let value = if parent_info.unwrap_or(false) && group.parent_idx.is_some() {
            // Include full parent node info as dictionary with children
            let parent_dict = PyDict::new_bound(py);
            
            // Add parent node information
            if let Some(ref id) = group.parent_id {
                parent_dict.set_item("id", value_to_py(py, id)?)?;
            }
            parent_dict.set_item("title", &group.parent_title)?;
            if let Some(ref type_str) = group.parent_type {
                parent_dict.set_item("type", type_str)?;
            }
            
            // Add children list
            let nodes: Vec<PyObject> = group.nodes.iter()
                .map(|node| nodeinfo_to_pydict(py, node))
                .collect::<PyResult<_>>()?;
            parent_dict.set_item("children", nodes)?;
            
            parent_dict.into()
        } else {
            // Just include children list
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
                // For single values, we know we only have one value per vector
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
            
            // Process incoming connections
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
            
            // Process outgoing connections
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