// src/graph/cypher/py_convert.rs
// Convert CypherResult to Python objects

use super::result::CypherResult;
use crate::datatypes::py_out;
use crate::datatypes::values::Value;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;

/// Recursively convert a serde_json::Value to a Python object.
fn json_value_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => b.into_py_any(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py_any(py)
            } else if let Some(f) = n.as_f64() {
                f.into_py_any(py)
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => s.clone().into_py_any(py),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_value_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_value_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// Convert a Cypher Value to Python, parsing JSON-formatted strings
/// (e.g. from nodes(p), relationships(p), collect(), list comprehensions, map projections)
/// into native Python lists/dicts.
///
/// Uses Rust's serde_json instead of Python's json module to avoid
/// repeated py.import("json") calls that are expensive (~10-50µs each) and require the GIL.
fn cypher_value_to_py(py: Python<'_>, val: &Value) -> PyResult<Py<PyAny>> {
    if let Value::String(s) = val {
        if (s.starts_with('[') && s.ends_with(']')) || (s.starts_with('{') && s.ends_with('}')) {
            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(s) {
                return json_value_to_py(py, &json_val);
            }
        }
    }
    py_out::value_to_py(py, val)
}

/// Convert a CypherResult to a pandas DataFrame
pub fn cypher_result_to_dataframe(py: Python<'_>, result: &CypherResult) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    let col_order = PyList::empty(py);

    // Pre-create Python strings for column names (reused across rows)
    let col_keys: Vec<Py<PyAny>> = result
        .columns
        .iter()
        .map(|col| col.clone().into_py_any(py))
        .collect::<PyResult<_>>()?;

    // Build columnar dict-of-lists
    for (i, key) in col_keys.iter().enumerate() {
        let col_list = PyList::empty(py);
        for row in &result.rows {
            if let Some(val) = row.get(i) {
                col_list.append(cypher_value_to_py(py, val)?)?;
            } else {
                col_list.append(py.None())?;
            }
        }
        dict.set_item(key, col_list)?;
        col_order.append(key)?;
    }

    let pd = py.import("pandas")?;

    if result.rows.is_empty() {
        let kwargs = PyDict::new(py);
        kwargs.set_item("columns", col_order)?;
        return pd
            .call_method("DataFrame", (), Some(&kwargs))
            .map(|df| df.unbind());
    }

    let kwargs = PyDict::new(py);
    kwargs.set_item("columns", col_order)?;
    pd.call_method("DataFrame", (dict,), Some(&kwargs))
        .map(|df| df.unbind())
}

/// Convert a CypherResult to a Python list of row dicts
pub fn cypher_result_to_py(py: Python<'_>, result: &CypherResult) -> PyResult<Py<PyAny>> {
    // Pre-create Python strings for column names (created once, reused per row)
    let col_keys: Vec<Py<PyAny>> = result
        .columns
        .iter()
        .map(|col| col.clone().into_py_any(py))
        .collect::<PyResult<_>>()?;

    let rows_list = PyList::empty(py);
    for row in &result.rows {
        let row_dict = PyDict::new(py);
        for (i, key) in col_keys.iter().enumerate() {
            if let Some(val) = row.get(i) {
                row_dict.set_item(key, cypher_value_to_py(py, val)?)?;
            } else {
                row_dict.set_item(key, py.None())?;
            }
        }
        rows_list.append(row_dict)?;
    }
    Ok(rows_list.into_any().unbind())
}

fn stats_to_py<'py>(
    py: Python<'py>,
    stats: &super::result::MutationStats,
) -> PyResult<Bound<'py, PyDict>> {
    let stats_dict = PyDict::new(py);
    stats_dict.set_item("nodes_created", stats.nodes_created)?;
    stats_dict.set_item("relationships_created", stats.relationships_created)?;
    stats_dict.set_item("properties_set", stats.properties_set)?;
    stats_dict.set_item("nodes_deleted", stats.nodes_deleted)?;
    stats_dict.set_item("relationships_deleted", stats.relationships_deleted)?;
    stats_dict.set_item("properties_removed", stats.properties_removed)?;
    Ok(stats_dict)
}

/// Convert a CypherResult to Python, including mutation stats when present.
/// - Read queries → list[dict]
/// - Mutations without RETURN → {'stats': {...}}
/// - Mutations with RETURN → {'rows': list[dict], 'stats': {...}}
pub fn cypher_result_to_py_auto(py: Python<'_>, result: &CypherResult) -> PyResult<Py<PyAny>> {
    match &result.stats {
        Some(stats) if result.rows.is_empty() => {
            // Mutation with no RETURN → {'stats': {...}}
            let dict = PyDict::new(py);
            dict.set_item("stats", stats_to_py(py, stats)?)?;
            Ok(dict.into_any().unbind())
        }
        Some(stats) => {
            // Mutation WITH RETURN → {'rows': [...], 'stats': {...}}
            let dict = PyDict::new(py);
            dict.set_item("rows", cypher_result_to_py(py, result)?)?;
            dict.set_item("stats", stats_to_py(py, stats)?)?;
            Ok(dict.into_any().unbind())
        }
        None => {
            // Read query → bare list[dict]
            cypher_result_to_py(py, result)
        }
    }
}

// ========================================================================
// Pre-processed result conversion (for GIL-free JSON pre-parsing)
// ========================================================================

/// A value that may contain pre-parsed JSON for efficient Python conversion.
/// Used in the read-only cypher path where JSON parsing happens during py.detach()
/// (GIL released) instead of during Python conversion (GIL held).
pub enum PreProcessedValue {
    /// Plain value — convert directly via py_out::value_to_py
    Plain(Value),
    /// JSON already parsed during GIL-free phase
    ParsedJson(serde_json::Value),
}

/// Pre-parse JSON strings in a CypherResult. Runs without the GIL (pure Rust).
/// Scans all string values and parses those that look like JSON arrays/objects.
pub fn preprocess_result(result: &CypherResult) -> Vec<Vec<PreProcessedValue>> {
    result
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|val| {
                    if let Value::String(s) = val {
                        if (s.starts_with('[') && s.ends_with(']'))
                            || (s.starts_with('{') && s.ends_with('}'))
                        {
                            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(s) {
                                return PreProcessedValue::ParsedJson(json_val);
                            }
                        }
                    }
                    PreProcessedValue::Plain(val.clone())
                })
                .collect()
        })
        .collect()
}

/// Convert a pre-processed value to Python.
fn preprocessed_value_to_py(py: Python<'_>, pv: &PreProcessedValue) -> PyResult<Py<PyAny>> {
    match pv {
        PreProcessedValue::Plain(v) => py_out::value_to_py(py, v),
        PreProcessedValue::ParsedJson(jv) => json_value_to_py(py, jv),
    }
}

/// Convert pre-processed rows to a Python list of row dicts.
/// Column keys are pre-created once and reused.
pub fn preprocessed_result_to_py(
    py: Python<'_>,
    columns: &[String],
    rows: &[Vec<PreProcessedValue>],
) -> PyResult<Py<PyAny>> {
    let col_keys: Vec<Py<PyAny>> = columns
        .iter()
        .map(|col| col.clone().into_py_any(py))
        .collect::<PyResult<_>>()?;

    let rows_list = PyList::empty(py);
    for row in rows {
        let row_dict = PyDict::new(py);
        for (i, key) in col_keys.iter().enumerate() {
            if let Some(pv) = row.get(i) {
                row_dict.set_item(key, preprocessed_value_to_py(py, pv)?)?;
            } else {
                row_dict.set_item(key, py.None())?;
            }
        }
        rows_list.append(row_dict)?;
    }
    Ok(rows_list.into_any().unbind())
}

/// Convert pre-processed rows to a pandas DataFrame.
pub fn preprocessed_result_to_dataframe(
    py: Python<'_>,
    columns: &[String],
    rows: &[Vec<PreProcessedValue>],
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    let col_order = PyList::empty(py);

    let col_keys: Vec<Py<PyAny>> = columns
        .iter()
        .map(|col| col.clone().into_py_any(py))
        .collect::<PyResult<_>>()?;

    for (i, key) in col_keys.iter().enumerate() {
        let col_list = PyList::empty(py);
        for row in rows {
            if let Some(pv) = row.get(i) {
                col_list.append(preprocessed_value_to_py(py, pv)?)?;
            } else {
                col_list.append(py.None())?;
            }
        }
        dict.set_item(key, col_list)?;
        col_order.append(key)?;
    }

    let pd = py.import("pandas")?;

    if rows.is_empty() {
        let kwargs = PyDict::new(py);
        kwargs.set_item("columns", col_order)?;
        return pd
            .call_method("DataFrame", (), Some(&kwargs))
            .map(|df| df.unbind());
    }

    let kwargs = PyDict::new(py);
    kwargs.set_item("columns", col_order)?;
    pd.call_method("DataFrame", (dict,), Some(&kwargs))
        .map(|df| df.unbind())
}

/// Dispatch for pre-processed results: handles stats + read-only paths.
pub fn preprocessed_result_to_py_auto(
    py: Python<'_>,
    result: &CypherResult,
    columns: &[String],
    rows: &[Vec<PreProcessedValue>],
) -> PyResult<Py<PyAny>> {
    match &result.stats {
        Some(stats) if result.rows.is_empty() => {
            let dict = PyDict::new(py);
            dict.set_item("stats", stats_to_py(py, stats)?)?;
            Ok(dict.into_any().unbind())
        }
        Some(stats) => {
            let dict = PyDict::new(py);
            dict.set_item("rows", preprocessed_result_to_py(py, columns, rows)?)?;
            dict.set_item("stats", stats_to_py(py, stats)?)?;
            Ok(dict.into_any().unbind())
        }
        None => preprocessed_result_to_py(py, columns, rows),
    }
}
