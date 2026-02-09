// src/graph/cypher/py_convert.rs
// Convert CypherResult to Python objects

use super::result::CypherResult;
use crate::datatypes::py_out;
use crate::datatypes::values::Value;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Convert a Cypher Value to Python, parsing JSON-formatted list strings
/// (e.g. from nodes(p), relationships(p), collect(), list comprehensions)
/// into native Python lists.
fn cypher_value_to_py(py: Python<'_>, val: &Value) -> PyResult<Py<PyAny>> {
    if let Value::String(s) = val {
        if s.starts_with('[') && s.ends_with(']') {
            // Try to parse as JSON array → Python list
            let json = py.import("json")?;
            if let Ok(parsed) = json.call_method1("loads", (s.as_str(),)) {
                return Ok(parsed.unbind());
            }
        }
    }
    py_out::value_to_py(py, val)
}

/// Convert a CypherResult to a pandas DataFrame
pub fn cypher_result_to_dataframe(py: Python<'_>, result: &CypherResult) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    let col_order = PyList::empty(py);

    // Build columnar dict-of-lists
    for (i, col) in result.columns.iter().enumerate() {
        let col_list = PyList::empty(py);
        for row in &result.rows {
            if let Some(val) = row.get(i) {
                col_list.append(cypher_value_to_py(py, val)?)?;
            } else {
                col_list.append(py.None())?;
            }
        }
        dict.set_item(col, col_list)?;
        col_order.append(col)?;
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

/// Convert a CypherResult to a Python dict with 'columns' and 'rows' keys
pub fn cypher_result_to_py(py: Python<'_>, result: &CypherResult) -> PyResult<Py<PyAny>> {
    let result_dict = PyDict::new(py);

    // Columns list
    let cols = PyList::empty(py);
    for col in &result.columns {
        cols.append(col)?;
    }
    result_dict.set_item("columns", cols)?;

    // Rows as list of dicts
    let rows_list = PyList::empty(py);
    for row in &result.rows {
        let row_dict = PyDict::new(py);
        for (i, col) in result.columns.iter().enumerate() {
            if let Some(val) = row.get(i) {
                row_dict.set_item(col, cypher_value_to_py(py, val)?)?;
            } else {
                row_dict.set_item(col, py.None())?;
            }
        }
        rows_list.append(row_dict)?;
    }
    result_dict.set_item("rows", rows_list)?;

    // Add mutation stats if present
    if let Some(ref stats) = result.stats {
        let stats_dict = PyDict::new(py);
        stats_dict.set_item("nodes_created", stats.nodes_created)?;
        stats_dict.set_item("relationships_created", stats.relationships_created)?;
        stats_dict.set_item("properties_set", stats.properties_set)?;
        stats_dict.set_item("nodes_deleted", stats.nodes_deleted)?;
        stats_dict.set_item("relationships_deleted", stats.relationships_deleted)?;
        stats_dict.set_item("properties_removed", stats.properties_removed)?;
        result_dict.set_item("stats", stats_dict)?;
    }

    Ok(result_dict.into_any().unbind())
}
