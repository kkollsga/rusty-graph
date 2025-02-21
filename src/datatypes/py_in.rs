use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use super::values::{DataFrame, ColumnType, ColumnData, Value, FilterCondition};
use super::type_conversions::{to_u32, to_i64, to_f64, to_datetime, to_bool};
use crate::graph::node_calculations::StatMethod;


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
            if val.downcast::<PyList>().is_ok() {
                parse_in_condition(val)
            } else {
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
                vec.push(to_u32(&value));
            }
            Ok(ColumnData::UniqueId(vec))
        },
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

pub fn parse_sort_fields(
    sort_spec: &Bound<'_, PyAny>, 
    ascending: Option<bool>
) -> PyResult<Vec<(String, bool)>> {
    // Handle single field with explicit direction
    if let Some(asc) = ascending {
        let field = sort_spec.extract::<String>()?;
        return Ok(vec![(field, asc)]);
    }
    
    // Handle single string
    if let Ok(field) = sort_spec.extract::<String>() {
        return Ok(vec![(field, true)]);
    }
    
    // Handle single tuple
    if let Ok((field, ascending)) = sort_spec.extract::<(String, bool)>() {
        return Ok(vec![(field, ascending)]);
    }
    
    // Handle list of sort specifications
    let sort_list = sort_spec.downcast::<PyList>()?;
    if sort_list.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Sort specification cannot be empty"));
    }

    // Process each item in the list
    sort_list.iter()
        .map(|item| {
            if let Ok(field) = item.extract::<String>() {
                Ok((field, true))
            } else if let Ok((field, ascending)) = item.extract::<(String, bool)>() {
                Ok((field, ascending))
            } else if let Ok(list) = item.downcast::<PyList>() {
                if list.len() != 2 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("List specification must have exactly 2 elements"));
                }
                Ok((
                    list.get_item(0)?.extract::<String>()?,
                    list.get_item(1)?.extract::<bool>()?
                ))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid sort specification"))
            }
        })
        .collect()
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

pub fn parse_stat_method(method: &str) -> PyResult<StatMethod> {
    match method.to_lowercase().as_str() {
        "sum" => Ok(StatMethod::Sum),
        "average" | "avg" | "mean" => Ok(StatMethod::Average),
        "min" | "minimum" => Ok(StatMethod::Min),
        "max" | "maximum" => Ok(StatMethod::Max),
        "count" => Ok(StatMethod::Count),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unsupported statistical method: {}", method)
        )),
    }
}