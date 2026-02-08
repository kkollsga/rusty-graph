// src/graph/cypher/py_convert.rs
// Convert CypherResult to Python objects

use super::result::CypherResult;
use crate::datatypes::py_out;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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
                row_dict.set_item(col, py_out::value_to_py(py, val)?)?;
            } else {
                row_dict.set_item(col, py.None())?;
            }
        }
        rows_list.append(row_dict)?;
    }
    result_dict.set_item("rows", rows_list)?;

    Ok(result_dict.into_any().unbind())
}
