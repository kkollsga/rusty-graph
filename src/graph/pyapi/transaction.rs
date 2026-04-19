//! `Transaction` `#[pyclass]` + its `#[pymethods]`.
//!
//! Moved out of `graph::mod.rs` in Phase 8.

use crate::datatypes::py_in;
use crate::datatypes::values::Value;
use crate::graph::languages::cypher;
use crate::graph::schema::{CowSelection, DirGraph};
use crate::graph::KnowledgeGraph;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{Bound, IntoPyObjectExt};
use std::collections::HashMap;
use std::sync::Arc;

/// Mutable working copy during a transaction.
///
/// Created by `graph.begin()`, provides a separate `DirGraph` that can be
/// modified without affecting the original. Call `commit()` to apply changes
/// back, or let it drop to discard.
///
/// ## Isolation semantics
///
/// - **Snapshot isolation**: `begin()` clones the entire `DirGraph` (via
///   `Arc` deep-copy). The transaction sees a frozen snapshot of the graph
///   at the moment `begin()` was called.
/// - **Write isolation**: mutations inside the transaction (via `cypher()`,
///   `add_nodes()`, etc.) modify only the working copy. The original graph
///   is untouched until `commit()`.
/// - **Commit**: `commit()` replaces the owner's `Arc<DirGraph>` with the
///   transaction's working copy. This is an atomic pointer swap — other
///   Python references to the `KnowledgeGraph` will see the new state on
///   their next operation.
/// - **No concurrent-transaction guarantees**: if two transactions are
///   created from the same graph, each gets an independent snapshot.
///   Whichever commits last wins (last-writer-wins). There is no conflict
///   detection or merge — the second commit silently overwrites the first.
/// - **No read-snapshot across transactions**: reads on the original graph
///   while a transaction is open will see the pre-transaction state. After
///   commit, they see the post-transaction state.
#[pyclass]
pub struct Transaction {
    /// Back-reference to the owning KnowledgeGraph (for commit)
    pub(crate) owner: Py<KnowledgeGraph>,
    /// Mutable working copy of the graph — `None` after commit/rollback
    pub(crate) working: Option<DirGraph>,
    /// Whether commit() was called
    pub(crate) committed: bool,
    /// Read-only transactions hold an Arc snapshot instead of a mutable clone
    pub(crate) read_only: bool,
    /// Arc snapshot for read-only transactions (O(1) to create, zero memory overhead)
    pub(crate) snapshot: Option<Arc<DirGraph>>,
    /// Graph version at `begin()` time — used for optimistic concurrency control
    pub(crate) base_version: u64,
    /// Optional transaction-level deadline — all operations fail after this instant
    pub(crate) deadline: Option<std::time::Instant>,
}

#[pymethods]
impl Transaction {
    /// Execute a Cypher query within this transaction.
    ///
    /// Mutations are applied to the transaction's working copy, not the original graph.
    /// Read queries also operate on the working copy (seeing uncommitted changes).
    ///
    /// Args:
    ///     query: A Cypher query string.
    ///     params: Optional dict of query parameters.
    ///     to_df: If True, return a pandas DataFrame instead of list of dicts.
    ///
    /// Returns:
    ///     Query results (same format as KnowledgeGraph.cypher).
    /// Whether this is a read-only transaction.
    #[getter]
    fn is_read_only(&self) -> bool {
        self.read_only
    }

    #[pyo3(signature = (query, params=None, to_df=false, timeout_ms=None))]
    fn cypher(
        &mut self,
        py: Python<'_>,
        query: &str,
        params: Option<&Bound<'_, PyDict>>,
        to_df: bool,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        // Check transaction-level deadline first
        if let Some(tx_deadline) = self.deadline {
            if std::time::Instant::now() >= tx_deadline {
                return Err(PyErr::new::<pyo3::exceptions::PyTimeoutError, _>(
                    "Transaction timed out",
                ));
            }
        }

        // Merge per-query timeout with transaction deadline (use the earlier one).
        // timeout_ms == 0 is the documented escape hatch: "no per-query deadline"
        // (the transaction-level deadline still applies if set).
        let effective_timeout_ms = match timeout_ms {
            Some(0) => None,
            Some(ms) => Some(ms),
            None => {
                // Fall through to the graph's backend-aware default.
                let graph = self.snapshot.as_deref().or(self.working.as_ref());
                graph.and_then(super::kg_core::backend_default_timeout_ms)
            }
        };
        let query_deadline = effective_timeout_ms
            .map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
        let deadline = match (self.deadline, query_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        // Convert params
        let param_map: HashMap<String, Value> = match params {
            Some(d) => {
                let mut map = HashMap::new();
                for (k, v) in d.iter() {
                    let key: String = k.extract()?;
                    let val = py_in::py_value_to_value(&v)?;
                    map.insert(key, val);
                }
                map
            }
            None => HashMap::new(),
        };

        if self.read_only {
            // Read-only transaction: execute against Arc snapshot
            let graph = self.snapshot.as_ref().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Read-only transaction already closed",
                )
            })?;

            let mut parsed = cypher::parse_cypher(query).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Cypher parse error: {}",
                    e
                ))
            })?;
            cypher::validate_schema(&parsed, graph).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Schema error: {}", e))
            })?;
            cypher::optimize(&mut parsed, graph, &param_map);

            if parsed.explain {
                let result = cypher::generate_explain_result(&parsed, graph);
                let view = crate::graph::pyapi::result_view::ResultView::from_cypher_result(result);
                return Py::new(py, view).map(|v| v.into_any());
            }

            if cypher::is_mutation_query(&parsed) {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Read-only transaction does not support mutations \
                     (CREATE, SET, DELETE, REMOVE, MERGE). Use begin() for read-write.",
                ));
            }

            let output_csv = parsed.output_format == cypher::OutputFormat::Csv;

            let executor = cypher::CypherExecutor::with_params(graph, &param_map, deadline);
            let result = executor.execute(&parsed).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Cypher execution error: {}",
                    e
                ))
            })?;

            if output_csv {
                result.to_csv().into_py_any(py)
            } else if to_df {
                let preprocessed = cypher::py_convert::preprocess_values_owned(result.rows);
                cypher::py_convert::preprocessed_result_to_dataframe(
                    py,
                    &result.columns,
                    &preprocessed,
                )
            } else {
                let view = crate::graph::pyapi::result_view::ResultView::from_cypher_result(result);
                Py::new(py, view).map(|v| v.into_any())
            }
        } else {
            // Read-write transaction: execute against mutable working copy
            let working = self.working.as_mut().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Transaction already committed or rolled back",
                )
            })?;

            let mut parsed = cypher::parse_cypher(query).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Cypher parse error: {}",
                    e
                ))
            })?;
            cypher::validate_schema(&parsed, working).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Schema error: {}", e))
            })?;
            cypher::optimize(&mut parsed, working, &param_map);

            if parsed.explain {
                let result = cypher::generate_explain_result(&parsed, working);
                let view = crate::graph::pyapi::result_view::ResultView::from_cypher_result(result);
                return Py::new(py, view).map(|v| v.into_any());
            }

            let output_csv = parsed.output_format == cypher::OutputFormat::Csv;

            let result = if cypher::is_mutation_query(&parsed) {
                cypher::execute_mutable(working, &parsed, param_map, deadline).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Cypher execution error: {}",
                        e
                    ))
                })?
            } else {
                let executor = cypher::CypherExecutor::with_params(working, &param_map, deadline);
                executor.execute(&parsed).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Cypher execution error: {}",
                        e
                    ))
                })?
            };

            if output_csv {
                result.to_csv().into_py_any(py)
            } else if to_df {
                let preprocessed = cypher::py_convert::preprocess_values_owned(result.rows);
                cypher::py_convert::preprocessed_result_to_dataframe(
                    py,
                    &result.columns,
                    &preprocessed,
                )
            } else {
                let view = crate::graph::pyapi::result_view::ResultView::from_cypher_result(result);
                Py::new(py, view).map(|v| v.into_any())
            }
        }
    }

    /// Commit the transaction — apply all changes to the original graph.
    ///
    /// For read-only transactions, this is a no-op.
    /// After commit, the transaction cannot be used again.
    fn commit(&mut self) -> PyResult<()> {
        if self.read_only {
            // Read-only: just release the snapshot
            self.snapshot = None;
            self.committed = true;
            return Ok(());
        }

        let working = self.working.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Transaction already committed or rolled back",
            )
        })?;

        // Optimistic concurrency control: check version hasn't changed
        let current_version = Python::attach(|py| {
            let kg = self.owner.borrow(py);
            kg.inner.version
        });
        if current_version != self.base_version {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Transaction conflict: graph was modified since begin(). \
                 Retry the transaction.",
            ));
        }

        Python::attach(|py| {
            let mut kg = self.owner.borrow_mut(py);
            let mut working = working;
            working.version = current_version + 1;
            kg.inner = Arc::new(working);
            kg.selection = CowSelection::new();
        });

        self.committed = true;
        Ok(())
    }

    /// Roll back the transaction — discard all changes.
    ///
    /// After rollback, the transaction cannot be used again.
    fn rollback(&mut self) -> PyResult<()> {
        if self.read_only {
            if self.snapshot.is_none() {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Transaction already committed or rolled back",
                ));
            }
            self.snapshot = None;
            return Ok(());
        }
        if self.working.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Transaction already committed or rolled back",
            ));
        }
        self.working = None;
        Ok(())
    }

    /// Context manager entry — returns self.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Context manager exit — commits on success, rolls back on exception.
    fn __exit__(
        &mut self,
        exc_type: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_val: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_tb: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<bool> {
        let is_active = if self.read_only {
            self.snapshot.is_some()
        } else {
            self.working.is_some()
        };

        if !is_active {
            // Already committed or rolled back
            return Ok(false);
        }

        if exc_type.is_some() {
            // Exception occurred — rollback
            self.working = None;
            self.snapshot = None;
        } else {
            // No exception — commit
            self.commit()?;
        }

        // Return false = don't suppress exception
        Ok(false)
    }
}
