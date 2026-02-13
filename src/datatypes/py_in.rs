// src/datatypes/py_in.rs
use super::type_conversions::{to_bool, to_datetime, to_f64, to_i64, to_u32};
use super::values::{ColumnData, ColumnType, DataFrame, FilterCondition, Value};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use std::collections::HashMap;

pub fn pydict_to_filter_conditions(
    dict: &Bound<'_, PyDict>,
) -> PyResult<HashMap<String, FilterCondition>> {
    dict.iter()
        .map(|(k, v)| {
            let key = k.extract::<String>()?;
            let condition = match v.cast::<PyDict>() {
                // Handle operator-based condition
                Ok(op_dict) => {
                    let (op, val) = op_dict.iter().next().ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty operator dictionary")
                    })?;

                    let op_str = op.extract::<String>()?;
                    parse_operator_condition(&op_str, &val)?
                }
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
            if val.cast::<PyList>().is_ok() {
                parse_in_condition(val)
            } else {
                Ok(FilterCondition::Equals(py_value_to_value(val)?))
            }
        }
        "!=" => Ok(FilterCondition::NotEquals(py_value_to_value(val)?)),
        ">" => Ok(FilterCondition::GreaterThan(py_value_to_value(val)?)),
        ">=" => Ok(FilterCondition::GreaterThanEquals(py_value_to_value(val)?)),
        "<" => Ok(FilterCondition::LessThan(py_value_to_value(val)?)),
        "<=" => Ok(FilterCondition::LessThanEquals(py_value_to_value(val)?)),
        "in" => parse_in_condition(val),
        "between" => parse_between_condition(val),
        "is_null" => Ok(FilterCondition::IsNull),
        "is_not_null" => Ok(FilterCondition::IsNotNull),
        "contains" => Ok(FilterCondition::Contains(py_value_to_value(val)?)),
        "starts_with" => Ok(FilterCondition::StartsWith(py_value_to_value(val)?)),
        "ends_with" => Ok(FilterCondition::EndsWith(py_value_to_value(val)?)),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported operator: {}",
            op
        ))),
    }
}

fn parse_between_condition(val: &Bound<'_, PyAny>) -> PyResult<FilterCondition> {
    match val.cast::<PyList>() {
        Ok(list) => {
            let items: Vec<_> = list.iter().collect();
            if items.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "'between' operator requires a list of exactly 2 values [min, max]",
                ));
            }
            let min = py_value_to_value(&items[0])?;
            let max = py_value_to_value(&items[1])?;
            Ok(FilterCondition::Between(min, max))
        }
        Err(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "'between' operator requires a list of [min, max] values",
        )),
    }
}

fn parse_in_condition(val: &Bound<'_, PyAny>) -> PyResult<FilterCondition> {
    match val.cast::<PyList>() {
        Ok(list) => {
            let values: PyResult<Vec<Value>> =
                list.iter().map(|item| py_value_to_value(&item)).collect();
            Ok(FilterCondition::In(values?))
        }
        Err(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "'in' operator requires a list value",
        )),
    }
}

fn convert_pandas_series(series: &Bound<'_, PyAny>, col_type: ColumnType) -> PyResult<ColumnData> {
    let length = series.len()?;

    // Get the null mask from pandas — this correctly handles None, np.nan, pd.NA, pd.NaT
    // regardless of pandas version or dtype backend (object, string[python], string[pyarrow], etc.)
    let null_mask: Vec<bool> = series
        .call_method0("isna")?
        .call_method0("tolist")?
        .extract()?;

    match col_type {
        ColumnType::UniqueId => {
            let mut vec = Vec::with_capacity(length);
            for (i, &is_null) in null_mask.iter().enumerate() {
                if is_null {
                    vec.push(None);
                } else {
                    let value = series.get_item(i)?;
                    vec.push(to_u32(&value));
                }
            }
            Ok(ColumnData::UniqueId(vec))
        }
        ColumnType::Int64 => {
            let mut vec = Vec::with_capacity(length);
            for (i, &is_null) in null_mask.iter().enumerate() {
                if is_null {
                    vec.push(None);
                } else {
                    let value = series.get_item(i)?;
                    vec.push(to_i64(&value));
                }
            }
            Ok(ColumnData::Int64(vec))
        }
        ColumnType::Float64 => {
            let mut vec = Vec::with_capacity(length);
            for (i, &is_null) in null_mask.iter().enumerate() {
                if is_null {
                    vec.push(None);
                } else {
                    let value = series.get_item(i)?;
                    vec.push(to_f64(&value));
                }
            }
            Ok(ColumnData::Float64(vec))
        }
        ColumnType::String => {
            let mut vec = Vec::with_capacity(length);
            for (i, &is_null) in null_mask.iter().enumerate() {
                if is_null {
                    vec.push(None);
                } else {
                    let value = series.get_item(i)?;
                    vec.push(value.str().ok().map(|s| s.to_string()));
                }
            }
            Ok(ColumnData::String(vec))
        }
        ColumnType::Boolean => {
            let mut vec = Vec::with_capacity(length);
            for (i, &is_null) in null_mask.iter().enumerate() {
                if is_null {
                    vec.push(None);
                } else {
                    let value = series.get_item(i)?;
                    vec.push(to_bool(&value));
                }
            }
            Ok(ColumnData::Boolean(vec))
        }
        ColumnType::DateTime => {
            let mut vec = Vec::with_capacity(length);
            for (i, &is_null) in null_mask.iter().enumerate() {
                if is_null {
                    vec.push(None);
                } else {
                    let value = series.get_item(i)?;
                    vec.push(to_datetime(&value));
                }
            }
            Ok(ColumnData::DateTime(vec))
        }
    }
}

pub fn pandas_to_dataframe(
    df: &Bound<'_, PyAny>,
    unique_id_fields: &[String],
    column_names: &[String],
    column_types: Option<&Bound<'_, PyDict>>,
) -> PyResult<DataFrame> {
    // Reset index to ensure contiguous positional access.
    // Handles filtered/deduped DataFrames with non-contiguous indexes.
    let kwargs = pyo3::types::PyDict::new(df.py());
    kwargs.set_item("drop", true)?;
    let df = df.call_method("reset_index", (), Some(&kwargs))?;

    let df_columns = df.getattr("columns")?;
    let all_column_names: Vec<String> = df_columns.extract()?;

    let mut df_out = DataFrame::new(Vec::new());

    Python::attach(|_py| {
        for col_name in column_names {
            // Skip if column doesn't exist in the dataframe
            if !all_column_names.contains(col_name) {
                continue;
            }

            // Use bracket notation df[col_name] instead of df.col_name to avoid
            // conflicts with DataFrame attributes like 'shape', 'index', 'columns', etc.
            let series = df.get_item(col_name).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Failed to access column '{}': {}. If using a reserved name like 'shape', 'index', or 'columns', please rename your column.", col_name, e)
                )
            })?;

            // Determine column type - check custom type mapping first
            let col_type = if let Some(type_dict) = column_types {
                // Get item returns Result<Option<...>, ...> so we need to handle both
                let maybe_type_value = type_dict.get_item(col_name)?;

                if let Some(type_value) = maybe_type_value {
                    // Extract type string from the dictionary
                    let type_str = type_value.extract::<String>()?;

                    // Parse the requested type
                    match type_str.to_lowercase().as_str() {
                        "uniqueid" | "unique_id" | "id" => ColumnType::UniqueId,
                        "int" | "int64" | "integer" => ColumnType::Int64,
                        "float" | "float64" | "double" => ColumnType::Float64,
                        "str" | "string" | "text" => ColumnType::String,
                        "bool" | "boolean" => ColumnType::Boolean,
                        "date" | "datetime" | "timestamp" => ColumnType::DateTime,
                        _ => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "Unsupported column type '{}' specified for column '{}'",
                                type_str, col_name
                            )))
                        }
                    }
                } else if unique_id_fields.contains(col_name) {
                    // Auto-detect: string/object columns keep String type rather than
                    // forcing UniqueId (which silently drops non-numeric string IDs)
                    let detected = determine_column_type(&series, col_name)?;
                    match detected {
                        ColumnType::String => ColumnType::String,
                        _ => ColumnType::UniqueId,
                    }
                } else {
                    // Fall back to auto-detection if not in custom mapping
                    determine_column_type(&series, col_name)?
                }
            } else if unique_id_fields.contains(col_name) {
                // Auto-detect: string/object columns keep String type
                let detected = determine_column_type(&series, col_name)?;
                match detected {
                    ColumnType::String => ColumnType::String,
                    _ => ColumnType::UniqueId,
                }
            } else {
                // No custom mapping provided, use auto-detection
                determine_column_type(&series, col_name)?
            };

            let data = convert_pandas_series(&series, col_type.clone())?;
            df_out
                .add_column(col_name.clone(), col_type, data)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        }
        Ok(())
    })?;

    Ok(df_out)
}

// Helper function to determine column type from pandas series
fn determine_column_type(series: &Bound<'_, PyAny>, col_name: &str) -> PyResult<ColumnType> {
    let dtype = series.getattr("dtype").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Could not determine type for column '{}'. The data may not be a valid pandas Series. \
                If your column is named 'shape', 'index', 'columns', 'values', or 'dtype', \
                please rename it as these conflict with pandas DataFrame attributes.",
            col_name
        ))
    })?;
    let type_str = dtype.str()?.to_string();

    match type_str.as_str() {
        "int64" | "int32" | "int16" | "int8" | "Int64" | "Int32" | "Int16" | "Int8" => {
            Ok(ColumnType::Int64)
        }
        "float64" | "float32" | "Float64" | "Float32" => Ok(ColumnType::Float64),
        "bool" | "boolean" => Ok(ColumnType::Boolean),
        s if s.starts_with("datetime64") => Ok(ColumnType::DateTime),
        "object" | "string" | "str" | "string[python]" | "string[pyarrow]" => {
            Ok(ColumnType::String)
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unsupported column type '{}' for column '{}'. Supported types: int, float, bool, datetime, string/object.", type_str, col_name)
        )),
    }
}

pub fn ensure_columns(
    df_columns: &[String],
    default_columns: &[&str],
    columns_to_include: Option<&Bound<'_, PyList>>,
    columns_to_skip: Option<&Bound<'_, PyList>>,
    enforce_columns: Option<bool>,
) -> PyResult<Vec<String>> {
    // Process columns_to_skip if present
    let skip_cols = match columns_to_skip {
        Some(list) => list
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?,
        None => Vec::new(),
    };

    // Process user-specified columns if present
    if let Some(include_list) = columns_to_include {
        let mut result: Vec<String> = include_list
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<_>>()?;

        // Always add default columns
        for &field in default_columns {
            if !result.contains(&field.to_string()) && !skip_cols.contains(&field.to_string()) {
                result.push(field.to_string());
            }
        }

        // Filter out any skipped columns
        result.retain(|col| !skip_cols.contains(col));

        Ok(result)
    } else if enforce_columns.unwrap_or(false) {
        // If enforce_columns is true and no columns specified, use only default columns
        let mut result = Vec::new();

        for &field in default_columns {
            if !skip_cols.contains(&field.to_string()) {
                result.push(field.to_string());
            }
        }

        Ok(result)
    } else {
        // Use all dataframe columns except skipped ones
        let mut result = df_columns
            .iter()
            .filter(|col| !skip_cols.contains(col))
            .cloned()
            .collect::<Vec<String>>();

        // Ensure default columns are included
        for &field in default_columns {
            if !result.contains(&field.to_string()) && !skip_cols.contains(&field.to_string()) {
                result.push(field.to_string());
            }
        }

        Ok(result)
    }
}

pub fn parse_sort_fields(
    sort_spec: &Bound<'_, PyAny>,
    ascending: Option<bool>,
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
    let sort_list = sort_spec.cast::<PyList>()?;
    if sort_list.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Sort specification cannot be empty",
        ));
    }

    // Process each item in the list
    sort_list
        .iter()
        .map(|item| {
            if let Ok(field) = item.extract::<String>() {
                Ok((field, true))
            } else if let Ok((field, ascending)) = item.extract::<(String, bool)>() {
                Ok((field, ascending))
            } else if let Ok(list) = item.cast::<PyList>() {
                if list.len() != 2 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "List specification must have exactly 2 elements",
                    ));
                }
                Ok((
                    list.get_item(0)?.extract::<String>()?,
                    list.get_item(1)?.extract::<bool>()?,
                ))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid sort specification",
                ))
            }
        })
        .collect()
}

pub fn py_value_to_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }

    // Check bool FIRST - Python bool is a subclass of int, so must check before numeric
    if value.is_instance_of::<pyo3::types::PyBool>() {
        if let Ok(b) = value.extract::<bool>() {
            return Ok(Value::Boolean(b));
        }
    }

    // Try numeric types before string (they fail fast without allocation)
    if let Ok(i) = value.extract::<i64>() {
        return Ok(Value::Int64(i));
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(Value::Float64(f));
    }

    // String extraction is expensive (allocates), so try last among common types
    if let Ok(s) = value.extract::<String>() {
        return Ok(Value::String(s));
    }

    // Fallback for other types
    if let Ok(u) = value.extract::<u32>() {
        return Ok(Value::UniqueId(u));
    }

    // Handle lists (e.g. embedding vectors for Cypher params) — serialize as JSON string
    if let Ok(list) = value.cast::<pyo3::types::PyList>() {
        let items: Vec<String> = list
            .iter()
            .map(|item| {
                if let Ok(f) = item.extract::<f64>() {
                    format!("{}", f)
                } else if let Ok(s) = item.extract::<String>() {
                    format!("\"{}\"", s)
                } else if let Ok(i) = item.extract::<i64>() {
                    format!("{}", i)
                } else {
                    "null".to_string()
                }
            })
            .collect();
        return Ok(Value::String(format!("[{}]", items.join(", "))));
    }

    Ok(Value::Null)
}
