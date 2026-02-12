// src/graph/cypher/py_convert.rs
// Convert pre-processed result data to Python objects.
// Used by ResultView for lazy conversion and by to_df=True direct paths.

use crate::datatypes::py_out;
use crate::datatypes::values::Value;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;

// ========================================================================
// JSON → Python conversion
// ========================================================================

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

// ========================================================================
// PreProcessedValue — the core data type for ResultView rows
// ========================================================================

/// A value that may contain pre-parsed JSON for efficient Python conversion.
/// Used by ResultView to store row data in Rust and convert lazily on access.
#[derive(Clone)]
pub enum PreProcessedValue {
    /// Plain value — convert directly via py_out::value_to_py
    Plain(Value),
    /// JSON already parsed during GIL-free phase
    ParsedJson(serde_json::Value),
}

/// Convert a pre-processed value to a Python object.
pub fn preprocessed_value_to_py(py: Python<'_>, pv: &PreProcessedValue) -> PyResult<Py<PyAny>> {
    match pv {
        PreProcessedValue::Plain(v) => py_out::value_to_py(py, v),
        PreProcessedValue::ParsedJson(jv) => json_value_to_py(py, jv),
    }
}

// ========================================================================
// Pre-processing: Value → PreProcessedValue (runs without GIL)
// ========================================================================

/// Pre-parse JSON strings in owned Value rows. Takes ownership (zero cloning for non-JSON values).
/// Runs in pure Rust without the GIL — used in py.detach blocks and ResultView constructors.
pub fn preprocess_values_owned(rows: Vec<Vec<Value>>) -> Vec<Vec<PreProcessedValue>> {
    use rayon::prelude::*;

    fn preprocess_row(row: Vec<Value>) -> Vec<PreProcessedValue> {
        row.into_iter()
            .map(|val| {
                if let Value::String(ref s) = val {
                    if (s.starts_with('[') && s.ends_with(']'))
                        || (s.starts_with('{') && s.ends_with('}'))
                    {
                        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(s) {
                            return PreProcessedValue::ParsedJson(json_val);
                        }
                    }
                }
                PreProcessedValue::Plain(val)
            })
            .collect()
    }

    if rows.len() >= 256 {
        rows.into_par_iter().map(preprocess_row).collect()
    } else {
        rows.into_iter().map(preprocess_row).collect()
    }
}

// ========================================================================
// Stats conversion
// ========================================================================

/// Convert MutationStats to a Python dict.
pub fn stats_to_py<'py>(
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

// ========================================================================
// DataFrame conversion (used by to_df=True shortcut and ResultView::to_df)
// ========================================================================

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
