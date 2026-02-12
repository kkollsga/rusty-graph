// src/graph/cypher/result_view.rs
// Lazy ResultView — Polars-style result container.
// Data stays in Rust and converts to Python only on access.

use super::py_convert::{
    preprocess_values_owned, preprocessed_result_to_dataframe, preprocessed_value_to_py,
    stats_to_py, PreProcessedValue,
};
use super::result::{CypherResult, MutationStats};
use crate::datatypes::values::Value;
use crate::graph::graph_algorithms::CentralityResult;
use crate::graph::schema::{DirGraph, NodeData};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use std::collections::HashSet;

/// Lazy result view — data stays in Rust until accessed from Python.
///
/// Returned by `cypher()`, centrality methods, `get_nodes()`, and `sample()`.
/// Supports `len()`, indexing, iteration, `to_list()`, and `to_df()`.
#[pyclass(name = "ResultView")]
pub struct ResultView {
    columns: Vec<String>,
    rows: Vec<Vec<PreProcessedValue>>,
    stats: Option<MutationStats>,
}

// ========================================================================
// Rust-only constructors (not exposed to Python)
// ========================================================================

impl ResultView {
    /// Cypher read path: data already preprocessed during py.detach (GIL-free).
    /// O(1) — just moves owned data into the struct.
    pub fn from_preprocessed(
        columns: Vec<String>,
        rows: Vec<Vec<PreProcessedValue>>,
        stats: Option<MutationStats>,
    ) -> Self {
        ResultView {
            columns,
            rows,
            stats,
        }
    }

    /// Cypher mutation path + Transaction: takes a CypherResult and preprocesses values.
    pub fn from_cypher_result(result: CypherResult) -> Self {
        let rows = preprocess_values_owned(result.rows);
        ResultView {
            columns: result.columns,
            rows,
            stats: result.stats,
        }
    }

    /// Centrality methods: resolves node_idx → NodeData lookups, builds rows.
    /// Pure Rust, no GIL needed.
    pub fn from_centrality(
        graph: &DirGraph,
        results: Vec<CentralityResult>,
        top_k: Option<usize>,
    ) -> Self {
        let limit = top_k.unwrap_or(results.len());
        let columns = vec!["type".into(), "title".into(), "id".into(), "score".into()];

        let rows: Vec<Vec<PreProcessedValue>> = results
            .into_iter()
            .take(limit)
            .filter_map(|r| {
                graph.get_node(r.node_idx).map(|node| {
                    vec![
                        PreProcessedValue::Plain(Value::String(node.node_type.clone())),
                        PreProcessedValue::Plain(node.title.clone()),
                        PreProcessedValue::Plain(node.id.clone()),
                        PreProcessedValue::Plain(Value::Float64(r.score)),
                    ]
                })
            })
            .collect();

        ResultView {
            columns,
            rows,
            stats: None,
        }
    }

    /// get_nodes / sample: collects all nodes, computes property key union for columns.
    /// Pure Rust, no GIL needed.
    pub fn from_nodes<'a>(nodes: impl Iterator<Item = &'a NodeData>) -> Self {
        let nodes_vec: Vec<&NodeData> = nodes.collect();

        // Compute union of property keys (preserving insertion order via seen set)
        let mut prop_keys: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for node in &nodes_vec {
            for key in node.properties.keys() {
                if seen.insert(key.clone()) {
                    prop_keys.push(key.clone());
                }
            }
        }
        prop_keys.sort();

        let mut columns = vec!["type".into(), "title".into(), "id".into()];
        columns.extend(prop_keys.iter().cloned());

        let rows: Vec<Vec<PreProcessedValue>> = nodes_vec
            .iter()
            .map(|node| {
                let mut row = vec![
                    PreProcessedValue::Plain(Value::String(node.node_type.clone())),
                    PreProcessedValue::Plain(node.title.clone()),
                    PreProcessedValue::Plain(node.id.clone()),
                ];
                for key in &prop_keys {
                    row.push(PreProcessedValue::Plain(
                        node.properties.get(key).cloned().unwrap_or(Value::Null),
                    ));
                }
                row
            })
            .collect();

        ResultView {
            columns,
            rows,
            stats: None,
        }
    }

    /// Convert a single row to a Python dict. Used by __getitem__ and __iter__.
    fn row_to_py(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        let row = &self.rows[index];
        let dict = PyDict::new(py);
        for (i, col) in self.columns.iter().enumerate() {
            if let Some(pv) = row.get(i) {
                dict.set_item(col, preprocessed_value_to_py(py, pv)?)?;
            } else {
                dict.set_item(col, py.None())?;
            }
        }
        Ok(dict.into_any().unbind())
    }
}

// ========================================================================
// Python protocol
// ========================================================================

#[pymethods]
impl ResultView {
    fn __len__(&self) -> usize {
        self.rows.len()
    }

    fn __bool__(&self) -> bool {
        !self.rows.is_empty()
    }

    fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<Py<PyAny>> {
        let len = self.rows.len() as isize;
        let actual = if idx < 0 { len + idx } else { idx };
        if actual < 0 || actual >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of range for ResultView with {} rows",
                idx,
                self.rows.len()
            )));
        }
        self.row_to_py(py, actual as usize)
    }

    fn __iter__(slf: Py<Self>) -> ResultIter {
        ResultIter {
            view: slf,
            index: 0,
        }
    }

    fn __repr__(&self) -> String {
        let cols: Vec<&str> = self.columns.iter().map(|s| s.as_str()).collect();
        format!("ResultView({} rows, columns={:?})", self.rows.len(), cols)
    }

    /// Column names as a list of strings.
    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.columns.clone().into_py_any(py)
    }

    /// Mutation statistics, or None for read queries.
    #[getter]
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.stats {
            Some(s) => stats_to_py(py, s).map(|d| d.into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    /// Convert all rows to a Python list of dicts (full materialization).
    fn to_list(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = pyo3::types::PyList::empty(py);
        for i in 0..self.rows.len() {
            list.append(self.row_to_py(py, i)?)?;
        }
        Ok(list.into_any().unbind())
    }

    /// Return a new ResultView with the first n rows (default 5).
    #[pyo3(signature = (n=5))]
    fn head(&self, n: usize) -> Self {
        let take = n.min(self.rows.len());
        ResultView {
            columns: self.columns.clone(),
            rows: self.rows[..take].to_vec(),
            stats: None,
        }
    }

    /// Return a new ResultView with the last n rows (default 5).
    #[pyo3(signature = (n=5))]
    fn tail(&self, n: usize) -> Self {
        let len = self.rows.len();
        let start = len.saturating_sub(n);
        ResultView {
            columns: self.columns.clone(),
            rows: self.rows[start..].to_vec(),
            stats: None,
        }
    }

    /// Convert to a pandas DataFrame.
    fn to_df(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        preprocessed_result_to_dataframe(py, &self.columns, &self.rows)
    }
}

// ========================================================================
// ResultIter — lazy iterator over ResultView rows
// ========================================================================

/// Iterator for ResultView. Converts one row per __next__ call.
#[pyclass(name = "ResultIter")]
pub struct ResultIter {
    view: Py<ResultView>,
    index: usize,
}

#[pymethods]
impl ResultIter {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let view = self.view.borrow(py);
        if self.index >= view.rows.len() {
            return Ok(None);
        }
        let result = view.row_to_py(py, self.index)?;
        self.index += 1;
        Ok(Some(result))
    }
}
