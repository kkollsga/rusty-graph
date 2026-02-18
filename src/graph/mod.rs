// src/graph/mod.rs
use crate::datatypes::values::{FilterCondition, Value};
use crate::datatypes::{py_in, py_out};
use crate::graph::calculations::StatResult;
use crate::graph::reporting::{OperationReport, OperationReports};
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, NodeIndexable};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::{Bound, IntoPyObjectExt};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub mod batch_operations;
pub mod calculations;
pub mod cypher;
pub mod data_retrieval;
pub mod debugging;
pub mod equation_parser;
pub mod export;
pub mod filtering_methods;
pub mod graph_algorithms;
pub mod introspection;
pub mod io_operations;
pub mod lookups;
pub mod maintain_graph;
pub mod pattern_matching;
pub mod reporting;
pub mod schema;
pub mod schema_validation;
pub mod set_operations;
pub mod spatial;
pub mod statistics_methods;
pub mod subgraph;
pub mod traversal_methods;
pub mod value_operations;
pub mod vector_search;

mod pymethods_algorithms;
mod pymethods_export;
mod pymethods_indexes;
mod pymethods_spatial;
mod pymethods_vector;

use schema::{
    ConnectionSchemaDefinition, CowSelection, DirGraph, NodeSchemaDefinition, PlanStep,
    SchemaDefinition,
};

/// Embedding column data extracted from a DataFrame: `[(column_name, [(node_id, embedding)])]`
type EmbeddingColumnData = Vec<(String, Vec<(Value, Vec<f32>)>)>;

/// Main knowledge graph type exposed to Python via PyO3.
///
/// Wraps a `DirGraph` behind an `Arc` for cheap cloning (read-heavy workloads).
/// All read methods take `&self`; mutations use `Arc::make_mut` for copy-on-write.
/// Supports Cypher queries, property filtering, traversals, graph algorithms,
/// and code entity exploration methods (`find`, `source`, `context`, `toc`).
#[pyclass]
pub struct KnowledgeGraph {
    inner: Arc<DirGraph>,
    selection: CowSelection, // Using Cow wrapper for copy-on-write semantics
    reports: OperationReports,
    last_mutation_stats: Option<cypher::result::MutationStats>,
    /// Registered Python embedding model (not serialized — re-set after load).
    embedder: Option<Py<PyAny>>,
}

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
    owner: Py<KnowledgeGraph>,
    /// Mutable working copy of the graph — `None` after commit/rollback
    working: Option<DirGraph>,
    /// Whether commit() was called
    committed: bool,
}

impl Clone for KnowledgeGraph {
    fn clone(&self) -> Self {
        KnowledgeGraph {
            inner: Arc::clone(&self.inner),
            selection: self.selection.clone(), // Arc clone - O(1), shares data
            reports: self.reports.clone(),
            last_mutation_stats: self.last_mutation_stats.clone(),
            embedder: Python::attach(|py| self.embedder.as_ref().map(|m| m.clone_ref(py))),
        }
    }
}

/// Error message shown when embed_texts/search_text is called without set_embedder().
const EMBEDDER_SKELETON_MSG: &str = "\
No embedding model registered. Call g.set_embedder(model) first.

Your model must implement:

    class MyEmbedder:
        dimension: int  # vector dimensionality (e.g. 384)

        def embed(self, texts: list[str]) -> list[list[float]]:
            # Return one vector per input text
            ...

Example with sentence-transformers:

    from sentence_transformers import SentenceTransformer

    class Embedder:
        def __init__(self, model_name=\"all-MiniLM-L6-v2\"):
            self._model = SentenceTransformer(model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()

        def embed(self, texts: list[str]) -> list[list[float]]:
            return self._model.encode(texts).tolist()

    g.set_embedder(Embedder())";

impl KnowledgeGraph {
    fn add_report(&mut self, report: OperationReport) -> usize {
        self.reports.add_report(report)
    }

    /// Get the registered embedder or return a helpful error with a skeleton.
    fn get_embedder_or_error<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.embedder {
            Some(model) => Ok(model.bind(py).clone()),
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                EMBEDDER_SKELETON_MSG,
            )),
        }
    }

    /// Call `model.load()` if the method exists (optional lifecycle hook).
    /// Errors propagate — if load() fails, the caller should not proceed.
    fn try_load_embedder(model: &Bound<'_, PyAny>) -> PyResult<()> {
        if model.hasattr("load")? {
            model.call_method0("load")?;
        }
        Ok(())
    }

    /// Call `model.unload()` if the method exists (optional lifecycle hook).
    /// Best-effort: errors are silently ignored since this is cleanup.
    fn try_unload_embedder(model: &Bound<'_, PyAny>) {
        if model.hasattr("unload").unwrap_or(false) {
            let _ = model.call_method0("unload");
        }
    }

    /// Code entity node types used by find/context/source.
    const CODE_TYPES: &[&str] = &[
        "Function",
        "Struct",
        "Class",
        "Enum",
        "Trait",
        "Protocol",
        "Interface",
        "Module",
        "Constant",
    ];

    /// Resolve a name (or qualified_name) to a single code entity NodeIndex.
    ///
    /// Returns:
    /// - `Ok(Ok(idx))` — uniquely resolved
    /// - `Ok(Err(matches))` — ambiguous (>1) or not found (0)
    fn resolve_code_entity(
        &self,
        name: &str,
        node_type: Option<&str>,
    ) -> (Option<NodeIndex>, Vec<(NodeIndex, schema::NodeInfo)>) {
        let name_val = Value::String(name.to_string());
        let types_to_search: Vec<&str> = match node_type {
            Some(nt) => vec![nt],
            None => Self::CODE_TYPES.to_vec(),
        };

        // Try qualified_name (stored as "id") exact match first
        for nt in &types_to_search {
            if let Some(indices) = self.inner.type_indices.get(*nt) {
                for &idx in indices {
                    if let Some(node) = self.inner.get_node(idx) {
                        if node.id == name_val {
                            return (Some(idx), Vec::new());
                        }
                    }
                }
            }
        }

        // Try qualified_name suffix match (e.g. "CypherExecutor::execute_single_clause"
        // matches "crate::graph::cypher::executor::CypherExecutor::execute_single_clause")
        if name.contains("::") {
            let suffix = format!("::{}", name);
            let mut matches: Vec<(NodeIndex, schema::NodeInfo)> = Vec::new();
            for nt in &types_to_search {
                if let Some(indices) = self.inner.type_indices.get(*nt) {
                    for &idx in indices {
                        if let Some(node) = self.inner.get_node(idx) {
                            if let Value::String(qn) = &node.id {
                                if qn.ends_with(&suffix) {
                                    matches.push((idx, node.to_node_info()));
                                }
                            }
                        }
                    }
                }
            }
            if matches.len() == 1 {
                return (Some(matches[0].0), matches);
            } else if !matches.is_empty() {
                return (None, matches);
            }
        }

        // Fall back to name/title search
        let mut matches: Vec<(NodeIndex, schema::NodeInfo)> = Vec::new();
        for nt in &types_to_search {
            if let Some(indices) = self.inner.type_indices.get(*nt) {
                for &idx in indices {
                    if let Some(node) = self.inner.get_node(idx) {
                        let name_match = node
                            .get_field_ref("name")
                            .map(|v| v == &name_val)
                            .unwrap_or(false)
                            || node
                                .get_field_ref("title")
                                .map(|v| v == &name_val)
                                .unwrap_or(false);
                        if name_match {
                            matches.push((idx, node.to_node_info()));
                        }
                    }
                }
            }
        }

        if matches.len() == 1 {
            (Some(matches[0].0), matches)
        } else {
            (None, matches)
        }
    }

    /// Build a source-location dict for a single name.
    fn source_one(&self, py: Python, name: &str, node_type: Option<&str>) -> PyResult<Py<PyAny>> {
        let (resolved, matches) = self.resolve_code_entity(name, node_type);

        let target_idx = match resolved {
            Some(idx) => idx,
            None => {
                let dict = PyDict::new(py);
                dict.set_item("name", name)?;
                if matches.is_empty() {
                    dict.set_item("error", format!("Node not found: {}", name))?;
                } else {
                    dict.set_item("ambiguous", true)?;
                    let match_list = PyList::empty(py);
                    for (_, info) in &matches {
                        let d = py_out::nodeinfo_to_pydict(py, info)?;
                        match_list.append(d)?;
                    }
                    dict.set_item("matches", match_list)?;
                }
                return Ok(dict.into());
            }
        };

        let node = self
            .inner
            .get_node(target_idx)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Node disappeared"))?;

        let dict = PyDict::new(py);
        dict.set_item("type", node.get_node_type_ref())?;
        dict.set_item("name", py_out::value_to_py(py, &node.title)?)?;
        dict.set_item("qualified_name", py_out::value_to_py(py, &node.id)?)?;

        if let Some(v) = node.get_field_ref("file_path") {
            dict.set_item("file_path", py_out::value_to_py(py, v)?)?;
        }
        if let Some(v) = node.get_field_ref("line_number") {
            dict.set_item("line_number", py_out::value_to_py(py, v)?)?;
        }
        if let Some(v) = node.get_field_ref("end_line") {
            dict.set_item("end_line", py_out::value_to_py(py, v)?)?;
        }
        if let (Some(Value::Int64(start)), Some(Value::Int64(end))) = (
            node.get_field_ref("line_number"),
            node.get_field_ref("end_line"),
        ) {
            dict.set_item("line_count", end - start + 1)?;
        }
        if let Some(v) = node.get_field_ref("signature") {
            dict.set_item("signature", py_out::value_to_py(py, v)?)?;
        }

        Ok(dict.into())
    }

    /// Check if a node's field value contains the given lowercase string (case-insensitive).
    fn field_contains_ci(node: &schema::NodeData, field: &str, needle_lower: &str) -> bool {
        node.get_field_ref(field)
            .and_then(|v| match v {
                Value::String(s) => Some(s.to_lowercase().contains(needle_lower)),
                _ => None,
            })
            .unwrap_or(false)
    }

    /// Check if a node's field value starts with the given lowercase string (case-insensitive).
    fn field_starts_with_ci(node: &schema::NodeData, field: &str, prefix_lower: &str) -> bool {
        node.get_field_ref(field)
            .and_then(|v| match v {
                Value::String(s) => Some(s.to_lowercase().starts_with(prefix_lower)),
                _ => None,
            })
            .unwrap_or(false)
    }
}

/// Helper function to get a mutable DirGraph from Arc.
/// Uses Arc::make_mut which clones only if there are other references,
/// otherwise gives a mutable reference in place. Callers mutate the graph
/// through the returned reference — no extraction/replacement needed.
///
/// WARNING: If other Arc references exist (e.g., a ResultView still in Python
/// scope, or a cloned KnowledgeGraph), this will deep-clone the entire DirGraph
/// including all nodes, edges, and indices. In read-heavy workloads this is fine,
/// but be aware that a lingering reference can cause unexpected memory spikes on mutation.
pub(super) fn get_graph_mut(arc: &mut Arc<DirGraph>) -> &mut DirGraph {
    Arc::make_mut(arc)
}

/// Lightweight centrality result conversion: returns {title: score} dict.
/// Creates ONE Python dict instead of N dicts — returns {title: score} format.
/// ~3-4x faster PyO3 serialization for large graphs.
pub(super) fn centrality_results_to_py_dict(
    py: Python<'_>,
    graph: &DirGraph,
    results: Vec<graph_algorithms::CentralityResult>,
    top_k: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let limit = top_k.unwrap_or(results.len());
    let scores_dict = PyDict::new(py);

    for result in results.into_iter().take(limit) {
        if let Some(node) = graph.get_node(result.node_idx) {
            let id_py = py_out::value_to_py(py, &node.id)?;
            scores_dict.set_item(id_py, result.score)?;
        }
    }

    Ok(scores_dict.into())
}

/// Convert centrality results to a pandas DataFrame with columns:
/// type, title, id, score — sorted by score descending.
pub(super) fn centrality_results_to_dataframe(
    py: Python<'_>,
    graph: &DirGraph,
    results: Vec<graph_algorithms::CentralityResult>,
    top_k: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let limit = top_k.unwrap_or(results.len());

    let mut types: Vec<&str> = Vec::with_capacity(limit);
    let mut titles: Vec<String> = Vec::with_capacity(limit);
    let mut ids: Vec<Py<PyAny>> = Vec::with_capacity(limit);
    let mut scores: Vec<f64> = Vec::with_capacity(limit);

    for result in results.into_iter().take(limit) {
        if let Some(node) = graph.get_node(result.node_idx) {
            types.push(&node.node_type);
            let title_str = match &node.title {
                Value::String(s) => s.clone(),
                _ => String::new(),
            };
            titles.push(title_str);
            ids.push(py_out::value_to_py(py, &node.id)?);
            scores.push(result.score);
        }
    }

    let pd = py.import("pandas")?;
    let data = PyDict::new(py);
    data.set_item("type", PyList::new(py, &types)?)?;
    data.set_item("title", PyList::new(py, &titles)?)?;
    data.set_item("id", PyList::new(py, &ids)?)?;
    data.set_item("score", PyList::new(py, &scores)?)?;

    let df = pd.call_method1("DataFrame", (data,))?;
    Ok(df.unbind())
}

/// Helper to convert community detection results to Python dict.
/// Accesses node data directly and uses interned keys for faster dict construction.
pub(super) fn community_results_to_py(
    py: Python<'_>,
    graph: &DirGraph,
    result: graph_algorithms::CommunityResult,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    // Pre-intern keys
    let key_type = pyo3::intern!(py, "type");
    let key_title = pyo3::intern!(py, "title");
    let key_id = pyo3::intern!(py, "id");

    // Group nodes by community
    let communities = PyDict::new(py);
    let mut grouped: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
    for a in &result.assignments {
        grouped.entry(a.community_id).or_default().push(a.node_idx);
    }

    for (comm_id, members) in &grouped {
        let member_list = PyList::empty(py);
        for &node_idx in members {
            if let Some(node) = graph.get_node(node_idx) {
                let node_dict = PyDict::new(py);
                node_dict.set_item(key_type, &node.node_type)?;
                let title_str = match &node.title {
                    Value::String(s) => s.as_str(),
                    _ => "",
                };
                node_dict.set_item(key_title, title_str)?;
                node_dict.set_item(key_id, py_out::value_to_py(py, &node.id)?)?;
                member_list.append(node_dict)?;
            }
        }
        communities.set_item(comm_id, member_list)?;
    }

    dict.set_item("communities", communities)?;
    dict.set_item("modularity", result.modularity)?;
    dict.set_item("num_communities", result.num_communities)?;

    Ok(dict.into())
}

#[pymethods]
impl KnowledgeGraph {
    #[new]
    fn new() -> Self {
        KnowledgeGraph {
            inner: Arc::new(DirGraph::new()),
            selection: CowSelection::new(),
            reports: OperationReports::new(),
            last_mutation_stats: None,
            embedder: None,
        }
    }

    /// Add nodes from a pandas DataFrame.
    ///
    /// Args:
    ///     data: DataFrame containing node data.
    ///     node_type: Label for this set of nodes (e.g. 'Person').
    ///     unique_id_field: Column used as unique identifier. String and integer IDs
    ///         are auto-detected from the DataFrame dtype.
    ///     node_title_field: Column used as display title. Defaults to unique_id_field.
    ///     columns: Whitelist of columns to include. None = all.
    ///     conflict_handling: 'update' (default), 'replace', 'skip', or 'preserve'.
    ///     skip_columns: Columns to exclude from properties.
    ///     column_types: Override column type detection: {'col': 'string'|'integer'|'float'|'datetime'|'uniqueid'}.
    ///
    /// Returns:
    ///     dict with 'nodes_created', 'nodes_updated', 'nodes_skipped',
    ///     'processing_time_ms', 'has_errors', and optionally 'errors'.
    #[pyo3(signature = (data, node_type, unique_id_field, node_title_field=None, columns=None, conflict_handling=None, skip_columns=None, column_types=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_nodes(
        &mut self,
        data: &Bound<'_, PyAny>,
        node_type: String,
        unique_id_field: String,
        node_title_field: Option<String>,
        columns: Option<&Bound<'_, PyList>>,
        conflict_handling: Option<String>,
        skip_columns: Option<&Bound<'_, PyList>>,
        column_types: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        // Detect embedding columns from column_types before DataFrame conversion
        let mut embedding_columns: Vec<String> = Vec::new();
        if let Some(type_dict) = column_types {
            for (key, value) in type_dict.iter() {
                let col_name: String = key.extract()?;
                let type_str: String = value.extract()?;
                if type_str.to_lowercase() == "embedding" {
                    embedding_columns.push(col_name);
                }
            }
        }

        // Get all columns from the dataframe
        let df_cols = data.getattr("columns")?;
        let all_columns: Vec<String> = df_cols.extract()?;

        // Create default columns array
        let mut default_cols = vec![unique_id_field.as_str()];
        if let Some(ref title_field) = node_title_field {
            default_cols.push(title_field);
        }

        // Use enforce_columns=false for add_nodes
        let enforce_columns = Some(false);

        // Get the filtered columns
        let mut column_list = py_in::ensure_columns(
            &all_columns,
            &default_cols,
            columns,
            skip_columns,
            enforce_columns,
        )?;

        // Remove embedding columns from the regular column list
        if !embedding_columns.is_empty() {
            column_list.retain(|c| !embedding_columns.contains(c));
        }

        // Extract embedding data before DataFrame conversion
        let embedding_data: EmbeddingColumnData = if !embedding_columns.is_empty() {
            let id_series = data.get_item(&unique_id_field)?;
            let nrows: usize = data.getattr("shape")?.get_item(0)?.extract()?;
            let mut result = Vec::new();

            for emb_col in &embedding_columns {
                let series = data.get_item(emb_col)?;
                let mut pairs = Vec::with_capacity(nrows);

                for i in 0..nrows {
                    let id_val = py_in::py_value_to_value(&id_series.get_item(i)?)?;
                    let emb_val: Vec<f32> = series.get_item(i)?.extract()?;
                    pairs.push((id_val, emb_val));
                }

                result.push((emb_col.clone(), pairs));
            }

            result
        } else {
            Vec::new()
        };

        let df_result = py_in::pandas_to_dataframe(
            data,
            std::slice::from_ref(&unique_id_field),
            &column_list,
            column_types,
        )?;

        let graph = get_graph_mut(&mut self.inner);

        let result = maintain_graph::add_nodes(
            graph,
            df_result,
            node_type.clone(),
            unique_id_field,
            node_title_field,
            conflict_handling,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Store embeddings for the created nodes
        if !embedding_data.is_empty() {
            graph.build_id_index(&node_type);
            for (emb_col, pairs) in &embedding_data {
                let dimension = pairs.first().map(|(_, v)| v.len()).unwrap_or(0);
                if dimension == 0 {
                    continue;
                }

                let store_key = if emb_col.ends_with("_emb") {
                    emb_col.clone()
                } else {
                    format!("{}_emb", emb_col)
                };

                let mut store = schema::EmbeddingStore::new(dimension);
                store.data.reserve(pairs.len() * dimension);
                for (id_val, vec) in pairs {
                    if vec.len() != dimension {
                        continue; // skip mismatched dimensions
                    }
                    if let Some(node_idx) = graph.lookup_by_id(&node_type, id_val) {
                        store.set_embedding(node_idx.index(), vec);
                    }
                }
                if store.len() > 0 {
                    graph
                        .embeddings
                        .insert((node_type.clone(), store_key), store);
                }
            }
        }

        self.selection.clear();

        // Store the report
        self.add_report(OperationReport::NodeOperation(result.clone()));

        // Convert the report to a Python dictionary
        Python::attach(|py| {
            let report_dict = PyDict::new(py);
            report_dict.set_item("operation", &result.operation_type)?;
            report_dict.set_item("timestamp", result.timestamp.to_rfc3339())?;
            report_dict.set_item("nodes_created", result.nodes_created)?;
            report_dict.set_item("nodes_updated", result.nodes_updated)?;
            report_dict.set_item("nodes_skipped", result.nodes_skipped)?;
            report_dict.set_item("processing_time_ms", result.processing_time_ms)?;

            // has_errors is true when there are errors OR rows were skipped
            let has_errors = !result.errors.is_empty() || result.nodes_skipped > 0;
            if !result.errors.is_empty() {
                report_dict.set_item("errors", &result.errors)?;
            }
            report_dict.set_item("has_errors", has_errors)?;

            // Emit Python warning if rows were skipped
            if result.nodes_skipped > 0 {
                let total = result.nodes_created + result.nodes_updated + result.nodes_skipped;
                let detail = result.errors.join("; ");
                let msg = std::ffi::CString::new(format!(
                    "add_nodes: {} of {} rows skipped. {}",
                    result.nodes_skipped, total, detail
                ))
                .unwrap_or_default();
                let _ = PyErr::warn(
                    py,
                    py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
                    msg.as_c_str(),
                    1,
                );
            }

            Ok(report_dict.into())
        })
    }

    /// Add connections (edges) from a pandas DataFrame.
    ///
    /// Args:
    ///     data: DataFrame containing connection data.
    ///     connection_type: Label for this connection type (e.g. 'KNOWS').
    ///     source_type: Node type of the source nodes.
    ///     source_id_field: Column containing source node IDs.
    ///     target_type: Node type of the target nodes.
    ///     target_id_field: Column containing target node IDs.
    ///     source_title_field: Optional column to update source node titles.
    ///     target_title_field: Optional column to update target node titles.
    ///     columns: Whitelist of columns to include as edge properties.
    ///     skip_columns: Columns to exclude from edge properties.
    ///     conflict_handling: 'update' (default), 'replace', 'skip', or 'preserve'.
    ///     column_types: Override column type detection.
    ///
    /// Returns:
    ///     dict with 'connections_created', 'connections_skipped',
    ///     'processing_time_ms', 'has_errors', and optionally 'errors'.
    #[pyo3(signature = (data, connection_type, source_type, source_id_field, target_type, target_id_field, source_title_field=None, target_title_field=None, columns=None, skip_columns=None, conflict_handling=None, column_types=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_connections(
        &mut self,
        data: &Bound<'_, PyAny>,
        connection_type: String,
        source_type: String,
        source_id_field: String,
        target_type: String,
        target_id_field: String,
        source_title_field: Option<String>,
        target_title_field: Option<String>,
        columns: Option<&Bound<'_, PyList>>,
        skip_columns: Option<&Bound<'_, PyList>>,
        conflict_handling: Option<String>,
        column_types: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        // Get all columns from the dataframe
        let df_cols = data.getattr("columns")?;
        let all_columns: Vec<String> = df_cols.extract()?;

        // Create default columns array
        let mut default_cols = vec![source_id_field.as_str(), target_id_field.as_str()];
        if let Some(ref src_title) = source_title_field {
            default_cols.push(src_title);
        }
        if let Some(ref tgt_title) = target_title_field {
            default_cols.push(tgt_title);
        }

        // Use enforce_columns=true for add_connections
        let enforce_columns = Some(true);

        // Get the filtered columns
        let column_list = py_in::ensure_columns(
            &all_columns,
            &default_cols,
            columns,
            skip_columns,
            enforce_columns,
        )?;

        let df_result = py_in::pandas_to_dataframe(
            data,
            &[source_id_field.clone(), target_id_field.clone()],
            &column_list,
            column_types,
        )?;

        let graph = get_graph_mut(&mut self.inner);

        let result = maintain_graph::add_connections(
            graph,
            df_result,
            connection_type,
            source_type,
            source_id_field,
            target_type,
            target_id_field,
            source_title_field,
            target_title_field,
            conflict_handling,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        self.selection.clear();

        // Store the report
        self.add_report(OperationReport::ConnectionOperation(result.clone()));

        // Convert the report to a Python dictionary
        Python::attach(|py| {
            let report_dict = PyDict::new(py);
            report_dict.set_item("operation", &result.operation_type)?;
            report_dict.set_item("timestamp", result.timestamp.to_rfc3339())?;
            report_dict.set_item("connections_created", result.connections_created)?;
            report_dict.set_item("connections_skipped", result.connections_skipped)?;
            report_dict.set_item("property_fields_tracked", result.property_fields_tracked)?;
            report_dict.set_item("processing_time_ms", result.processing_time_ms)?;

            // has_errors is true when there are errors OR connections were skipped
            let has_errors = !result.errors.is_empty() || result.connections_skipped > 0;
            if !result.errors.is_empty() {
                report_dict.set_item("errors", &result.errors)?;
            }
            report_dict.set_item("has_errors", has_errors)?;

            // Emit Python warning if rows were skipped
            if result.connections_skipped > 0 {
                let total = result.connections_created + result.connections_skipped;
                let detail = result.errors.join("; ");
                let msg = std::ffi::CString::new(format!(
                    "add_connections: {} of {} rows skipped. {}",
                    result.connections_skipped, total, detail
                ))
                .unwrap_or_default();
                let _ = PyErr::warn(
                    py,
                    py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
                    msg.as_c_str(),
                    1,
                );
            }

            Ok(report_dict.into())
        })
    }

    // ========================================================================
    // Connector API Methods (Bulk Loading)
    // ========================================================================

    /// Get the set of node types that exist in the graph.
    ///
    /// Returns:
    ///     List of node type names (excludes internal SchemaNode type)
    ///
    /// Example:
    ///     ```python
    ///     graph.add_nodes(df, 'Person', 'id', 'name')
    ///     graph.add_nodes(df2, 'Company', 'id', 'name')
    ///     print(graph.node_types)  # ['Person', 'Company']
    ///     ```
    #[getter]
    fn node_types(&self) -> Vec<String> {
        self.inner.get_node_types()
    }

    /// Add multiple node types at once from a list of node specifications.
    ///
    /// This enables bulk loading of nodes from data sources that provide
    /// standardized node specifications.
    ///
    /// Args:
    ///     nodes: List of dicts, each containing:
    ///         - 'node_type': str - The type/label for these nodes
    ///         - 'unique_id_field': str - Column name for unique ID
    ///         - 'node_title_field': str - Column name for display title
    ///         - 'data': DataFrame - The node data
    ///
    /// Returns:
    ///     Dict mapping node_type to count of nodes added
    ///
    /// Example:
    ///     ```python
    ///     nodes = [
    ///         {'node_type': 'Person', 'unique_id_field': 'id',
    ///          'node_title_field': 'name', 'data': people_df},
    ///         {'node_type': 'Company', 'unique_id_field': 'id',
    ///          'node_title_field': 'name', 'data': companies_df},
    ///     ]
    ///     stats = graph.add_nodes_bulk(nodes)
    ///     # {'Person': 100, 'Company': 50}
    ///     ```
    fn add_nodes_bulk(&mut self, py: Python<'_>, nodes: &Bound<'_, PyList>) -> PyResult<Py<PyAny>> {
        let result_dict = PyDict::new(py);

        for item in nodes.iter() {
            let spec = item.cast::<PyDict>()?;

            let node_type: String = spec
                .get_item("node_type")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'node_type' in node spec",
                    )
                })?
                .extract()?;
            let unique_id_field: String = spec
                .get_item("unique_id_field")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'unique_id_field' in node spec",
                    )
                })?
                .extract()?;
            let node_title_field: String = spec
                .get_item("node_title_field")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'node_title_field' in node spec",
                    )
                })?
                .extract()?;
            let data = spec.get_item("data")?.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'data' in node spec")
            })?;

            // Get columns from dataframe
            let df_cols = data.getattr("columns")?;
            let all_columns: Vec<String> = df_cols.extract()?;

            let df_result = py_in::pandas_to_dataframe(
                &data,
                std::slice::from_ref(&unique_id_field),
                &all_columns,
                None,
            )?;

            let graph = get_graph_mut(&mut self.inner);

            let report = maintain_graph::add_nodes(
                graph,
                df_result,
                node_type.clone(),
                unique_id_field,
                Some(node_title_field),
                None,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

            result_dict.set_item(&node_type, report.nodes_created + report.nodes_updated)?;
        }

        self.selection.clear();
        Ok(result_dict.into())
    }

    /// Add multiple connection types at once from a list of connection specifications.
    ///
    /// This enables bulk loading of connections from data sources that provide
    /// standardized connection specifications with 'source_id' and 'target_id' columns.
    ///
    /// Args:
    ///     connections: List of dicts, each containing:
    ///         - 'source_type': str - Node type of source nodes
    ///         - 'target_type': str - Node type of target nodes
    ///         - 'connection_name': str - The connection/edge type
    ///         - 'data': DataFrame - Must have 'source_id' and 'target_id' columns
    ///
    /// Returns:
    ///     Dict mapping connection_name to count of connections added
    ///
    /// Example:
    ///     ```python
    ///     connections = [
    ///         {'source_type': 'Person', 'target_type': 'Company',
    ///          'connection_name': 'WORKS_AT', 'data': works_df},
    ///         {'source_type': 'Person', 'target_type': 'Person',
    ///          'connection_name': 'KNOWS', 'data': knows_df},
    ///     ]
    ///     stats = graph.add_connections_bulk(connections)
    ///     # {'WORKS_AT': 500, 'KNOWS': 1200}
    ///     ```
    fn add_connections_bulk(
        &mut self,
        py: Python<'_>,
        connections: &Bound<'_, PyList>,
    ) -> PyResult<Py<PyAny>> {
        self.add_connections_internal(py, connections, false)
    }

    /// Add connections, automatically filtering to only those where
    /// both source and target node types exist in the graph.
    ///
    /// This enables data sources to provide ALL possible connections,
    /// and kglite selects only the valid ones based on loaded node types.
    ///
    /// Args:
    ///     connections: List of dicts, each containing:
    ///         - 'source_type': str - Node type of source nodes
    ///         - 'target_type': str - Node type of target nodes
    ///         - 'connection_name': str - The connection/edge type
    ///         - 'data': DataFrame - Must have 'source_id' and 'target_id' columns
    ///
    /// Returns:
    ///     Dict mapping connection_name to count of connections added
    ///     (only includes connections that were actually loaded)
    ///
    /// Example:
    ///     ```python
    ///     # Data source provides all possible connections
    ///     all_connections = data_source.get_all_connections()
    ///
    ///     # Graph only has Person and Company loaded
    ///     # This will skip connections involving other node types
    ///     stats = graph.add_connections_from_source(all_connections)
    ///     ```
    fn add_connections_from_source(
        &mut self,
        py: Python<'_>,
        connections: &Bound<'_, PyList>,
    ) -> PyResult<Py<PyAny>> {
        self.add_connections_internal(py, connections, true)
    }

    /// Internal helper for bulk connection loading
    fn add_connections_internal(
        &mut self,
        py: Python<'_>,
        connections: &Bound<'_, PyList>,
        filter_to_loaded: bool,
    ) -> PyResult<Py<PyAny>> {
        let result_dict = PyDict::new(py);
        let loaded_types: std::collections::HashSet<String> = if filter_to_loaded {
            self.inner.get_node_types().into_iter().collect()
        } else {
            std::collections::HashSet::new()
        };

        for item in connections.iter() {
            let spec = item.cast::<PyDict>()?;

            let source_type: String = spec
                .get_item("source_type")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'source_type' in connection spec",
                    )
                })?
                .extract()?;
            let target_type: String = spec
                .get_item("target_type")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'target_type' in connection spec",
                    )
                })?
                .extract()?;
            let connection_name: String = spec
                .get_item("connection_name")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'connection_name' in connection spec",
                    )
                })?
                .extract()?;
            let data = spec.get_item("data")?.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'data' in connection spec")
            })?;

            // Skip if filtering and types not loaded
            if filter_to_loaded
                && (!loaded_types.contains(&source_type) || !loaded_types.contains(&target_type))
            {
                continue;
            }

            // Standardized column names for connector API
            let source_id_field = "source_id".to_string();
            let target_id_field = "target_id".to_string();

            // Get columns from dataframe
            let df_cols = data.getattr("columns")?;
            let all_columns: Vec<String> = df_cols.extract()?;

            // Verify required columns exist
            if !all_columns.contains(&source_id_field) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Connection spec for '{}' missing required 'source_id' column. Available: [{}]",
                    connection_name,
                    all_columns.join(", ")
                )));
            }
            if !all_columns.contains(&target_id_field) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Connection spec for '{}' missing required 'target_id' column. Available: [{}]",
                    connection_name,
                    all_columns.join(", ")
                )));
            }

            let df_result = py_in::pandas_to_dataframe(
                &data,
                &[source_id_field.clone(), target_id_field.clone()],
                &all_columns,
                None,
            )?;

            let graph = get_graph_mut(&mut self.inner);

            let report = maintain_graph::add_connections(
                graph,
                df_result,
                connection_name.clone(),
                source_type,
                source_id_field,
                target_type,
                target_id_field,
                None, // source_title_field
                None, // target_title_field
                None, // conflict_handling
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

            result_dict.set_item(&connection_name, report.connections_created)?;
        }

        self.selection.clear();
        Ok(result_dict.into())
    }

    #[pyo3(signature = (node_type, sort=None, max_nodes=None))]
    fn type_filter(
        &mut self,
        node_type: String,
        sort: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>,
    ) -> PyResult<Self> {
        let mut new_kg = self.clone();

        // Record plan step: estimate based on type index
        let estimated = self
            .inner
            .type_indices
            .get(&node_type)
            .map(|v| v.len())
            .unwrap_or(0);
        new_kg.selection.clear_execution_plan(); // Start fresh plan

        let mut conditions = HashMap::new();
        conditions.insert(
            "type".to_string(),
            FilterCondition::Equals(Value::String(node_type.clone())),
        );

        let sort_fields = if let Some(spec) = sort {
            match spec.extract::<String>() {
                Ok(field) => Some(vec![(field, true)]),
                Err(_) => Some(py_in::parse_sort_fields(spec, None)?),
            }
        } else {
            None
        };

        filtering_methods::filter_nodes(
            &self.inner,
            &mut new_kg.selection,
            conditions,
            sort_fields,
            max_nodes,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Record actual result
        let actual = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);
        new_kg.selection.add_plan_step(
            PlanStep::new("TYPE_FILTER", Some(&node_type), estimated).with_actual_rows(actual),
        );

        Ok(new_kg)
    }

    #[pyo3(signature = (conditions, sort=None, max_nodes=None))]
    fn filter(
        &mut self,
        conditions: &Bound<'_, PyDict>,
        sort: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>,
    ) -> PyResult<Self> {
        let mut new_kg = self.clone();

        // Estimate based on current selection
        let estimated = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        let filter_conditions = py_in::pydict_to_filter_conditions(conditions)?;
        let sort_fields = match sort {
            Some(spec) => Some(py_in::parse_sort_fields(spec, None)?),
            None => None,
        };

        filtering_methods::filter_nodes(
            &self.inner,
            &mut new_kg.selection,
            filter_conditions,
            sort_fields,
            max_nodes,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Record actual result
        let actual = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);
        new_kg
            .selection
            .add_plan_step(PlanStep::new("FILTER", None, estimated).with_actual_rows(actual));

        Ok(new_kg)
    }

    #[pyo3(signature = (include_orphans=None, sort=None, max_nodes=None))]
    fn filter_orphans(
        &mut self,
        include_orphans: Option<bool>,
        sort: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>,
    ) -> PyResult<Self> {
        let mut new_kg = self.clone();
        let include = include_orphans.unwrap_or(true);

        let sort_fields = if let Some(spec) = sort {
            Some(py_in::parse_sort_fields(spec, None)?)
        } else {
            None
        };

        filtering_methods::filter_orphan_nodes(
            &self.inner,
            &mut new_kg.selection,
            include,
            sort_fields.as_ref(),
            max_nodes,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        Ok(new_kg)
    }

    #[pyo3(signature = (sort, ascending=None))]
    fn sort(&mut self, sort: &Bound<'_, PyAny>, ascending: Option<bool>) -> PyResult<Self> {
        let mut new_kg = self.clone();
        let sort_fields = py_in::parse_sort_fields(sort, ascending)?;

        filtering_methods::sort_nodes(&self.inner, &mut new_kg.selection, sort_fields)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(new_kg)
    }

    fn max_nodes(&mut self, max_per_group: usize) -> PyResult<Self> {
        let mut new_kg = self.clone();
        filtering_methods::limit_nodes_per_group(&self.inner, &mut new_kg.selection, max_per_group)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        Ok(new_kg)
    }

    /// Filter nodes that are valid at a specific date
    ///
    /// This is a convenience method for temporal queries. It filters nodes where:
    /// - date_from_field <= date <= date_to_field
    ///
    /// Default field names are 'date_from' and 'date_to' if not specified.
    #[pyo3(signature = (date, date_from_field=None, date_to_field=None))]
    fn valid_at(
        &mut self,
        date: &str,
        date_from_field: Option<&str>,
        date_to_field: Option<&str>,
    ) -> PyResult<Self> {
        let from_field = date_from_field.unwrap_or("date_from");
        let to_field = date_to_field.unwrap_or("date_to");

        let mut new_kg = self.clone();

        // Estimate based on current selection
        let estimated = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        // Build compound filter: date_from <= date AND date_to >= date
        let mut conditions = HashMap::new();
        conditions.insert(
            from_field.to_string(),
            FilterCondition::LessThanEquals(Value::String(date.to_string())),
        );
        conditions.insert(
            to_field.to_string(),
            FilterCondition::GreaterThanEquals(Value::String(date.to_string())),
        );

        filtering_methods::filter_nodes(&self.inner, &mut new_kg.selection, conditions, None, None)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Record actual result
        let actual = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);
        new_kg
            .selection
            .add_plan_step(PlanStep::new("VALID_AT", None, estimated).with_actual_rows(actual));

        Ok(new_kg)
    }

    /// Filter nodes that are valid during a date range
    ///
    /// This filters nodes where their validity period overlaps with the given range:
    /// - date_from_field <= end_date AND date_to_field >= start_date
    ///
    /// Default field names are 'date_from' and 'date_to' if not specified.
    #[pyo3(signature = (start_date, end_date, date_from_field=None, date_to_field=None))]
    fn valid_during(
        &mut self,
        start_date: &str,
        end_date: &str,
        date_from_field: Option<&str>,
        date_to_field: Option<&str>,
    ) -> PyResult<Self> {
        let from_field = date_from_field.unwrap_or("date_from");
        let to_field = date_to_field.unwrap_or("date_to");

        let mut new_kg = self.clone();

        // Estimate based on current selection
        let estimated = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        // Build compound filter for overlapping ranges:
        // node.date_from <= end_date AND node.date_to >= start_date
        let mut conditions = HashMap::new();
        conditions.insert(
            from_field.to_string(),
            FilterCondition::LessThanEquals(Value::String(end_date.to_string())),
        );
        conditions.insert(
            to_field.to_string(),
            FilterCondition::GreaterThanEquals(Value::String(start_date.to_string())),
        );

        filtering_methods::filter_nodes(&self.inner, &mut new_kg.selection, conditions, None, None)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Record actual result
        let actual = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);
        new_kg
            .selection
            .add_plan_step(PlanStep::new("VALID_DURING", None, estimated).with_actual_rows(actual));

        Ok(new_kg)
    }

    /// Update properties on all currently selected nodes
    ///
    /// This allows batch updating of properties on nodes matching the current selection.
    /// Returns a dictionary containing:
    ///   - 'graph': A new KnowledgeGraph with the updated nodes (original is unchanged)
    ///   - 'nodes_updated': Number of nodes that were updated
    ///   - 'report_index': Index of the operation report
    ///
    /// Example:
    ///     ```python
    ///     result = graph.type_filter('Discovery').filter({'year': {'>=': 2020}}).update({
    ///         'is_recent': True
    ///     })
    ///     graph = result['graph']  # Use the returned graph with updates
    ///     print(f"Updated {result['nodes_updated']} nodes")
    ///     ```
    ///
    /// Args:
    ///     properties: Dictionary of property names and values to set
    ///     keep_selection: If True, preserve the current selection in the returned graph
    ///
    /// Returns:
    ///     Dictionary with 'graph' (KnowledgeGraph), 'nodes_updated' (int), 'report_index' (int)
    #[pyo3(signature = (properties, keep_selection=None))]
    fn update(
        &mut self,
        properties: &Bound<'_, PyDict>,
        keep_selection: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        // Get the current level's nodes
        let current_index = self.selection.get_level_count().saturating_sub(1);
        let level = self.selection.get_level(current_index).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("No active selection level")
        })?;

        let nodes = level.get_all_nodes();
        if nodes.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No nodes selected for update",
            ));
        }

        // Pre-extract Python values before mutating the graph
        let mut parsed_properties: Vec<(String, Value)> = Vec::new();
        for (key, value) in properties.iter() {
            let property_name: String = key.extract().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Property names must be strings")
            })?;
            let property_value = py_in::py_value_to_value(&value)?;
            parsed_properties.push((property_name, property_value));
        }

        // Now mutate the graph — no ? operators from here to Arc creation
        let graph = get_graph_mut(&mut self.inner);

        let mut total_updated = 0;
        let mut errors = Vec::new();

        for (property_name, property_value) in &parsed_properties {
            let node_values: Vec<(Option<petgraph::graph::NodeIndex>, Value)> = nodes
                .iter()
                .map(|&idx| (Some(idx), property_value.clone()))
                .collect();

            match maintain_graph::update_node_properties(graph, &node_values, property_name) {
                Ok(report) => {
                    total_updated += report.nodes_updated;
                    errors.extend(report.errors);
                }
                Err(e) => {
                    errors.push(format!(
                        "Error updating property '{}': {}",
                        property_name, e
                    ));
                }
            }
        }

        // Create the result KnowledgeGraph (clone the Arc for the new graph)
        let mut new_kg = KnowledgeGraph {
            inner: self.inner.clone(),
            selection: if keep_selection.unwrap_or(false) {
                self.selection.clone()
            } else {
                CowSelection::new()
            },
            reports: self.reports.clone(),
            last_mutation_stats: None,
            embedder: Python::attach(|py| self.embedder.as_ref().map(|m| m.clone_ref(py))),
        };

        // Create and add a report
        let report = reporting::NodeOperationReport {
            operation_type: "update".to_string(),
            timestamp: chrono::Utc::now(),
            nodes_created: 0,
            nodes_updated: total_updated,
            nodes_skipped: 0,
            processing_time_ms: 0.0, // Could track this if needed
            errors,
        };

        let report_index = new_kg.add_report(OperationReport::NodeOperation(report));

        // Return the new KnowledgeGraph and the report
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("graph", Py::new(py, new_kg)?.into_any())?;
            dict.set_item("nodes_updated", total_updated)?;
            dict.set_item("report_index", report_index)?;
            Ok(dict.into())
        })
    }

    #[pyo3(signature = (max_nodes=None, indices=None, parent_type=None, parent_info=None,
                         flatten_single_parent=true))]
    fn get_nodes(
        &self,
        max_nodes: Option<usize>,
        indices: Option<Vec<usize>>,
        parent_type: Option<&str>,
        parent_info: Option<bool>,
        flatten_single_parent: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        // Fast path: when we just want a flat list of nodes without grouping,
        // skip the intermediate NodeInfo clone and convert directly from NodeData.
        // This avoids cloning the properties HashMap for each node.
        //
        // We only use fast path when selection is not empty, because empty selection
        // with no query operations means "return all nodes" which requires different logic.
        let selection_has_nodes = self.selection.current_node_count() > 0;
        let has_query_operations = !self.selection.get_execution_plan().is_empty();

        let use_fast_path = indices.is_none()
            && parent_type.is_none()
            && !parent_info.unwrap_or(false)
            && flatten_single_parent.unwrap_or(true)
            && (selection_has_nodes || has_query_operations);

        if use_fast_path {
            let max = max_nodes.unwrap_or(usize::MAX);
            let nodes: Vec<&schema::NodeData> = self
                .selection
                .current_node_indices()
                .filter_map(|idx| self.inner.get_node(idx))
                .take(max)
                .collect();
            let view = cypher::ResultView::from_nodes(nodes.into_iter());
            return Python::attach(|py| Py::new(py, view).map(|v| v.into_any()));
        }

        // Full path: handles grouping by parent, filtering, etc.
        let nodes = data_retrieval::get_nodes(
            &self.inner,
            &self.selection,
            None,
            indices.as_deref(),
            max_nodes,
        );
        Python::attach(|py| {
            py_out::level_nodes_to_pydict(
                py,
                &nodes,
                parent_type,
                parent_info,
                flatten_single_parent,
            )
        })
    }

    /// Export the current selection as a pandas DataFrame.
    ///
    /// Each node becomes a row with columns for title, type, id, and all properties.
    /// Nodes of different types may have different properties — missing values become None.
    #[pyo3(signature = (*, include_type=true, include_id=true))]
    fn to_df(&self, py: Python<'_>, include_type: bool, include_id: bool) -> PyResult<Py<PyAny>> {
        // Collect nodes from the current selection
        let mut nodes_data: Vec<(&str, &Value, &Value, &HashMap<String, Value>)> = Vec::new();
        let mut prop_keys: Vec<String> = Vec::new();
        let mut prop_keys_seen: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for node_idx in self.selection.current_node_indices() {
            if let Some(node) = self.inner.get_node(node_idx) {
                for key in node.properties.keys() {
                    if prop_keys_seen.insert(key.clone()) {
                        prop_keys.push(key.clone());
                    }
                }
                nodes_data.push((&node.node_type, &node.id, &node.title, &node.properties));
            }
        }

        prop_keys.sort();

        // Build columnar dict-of-lists
        let n = nodes_data.len();
        let title_col = PyList::empty(py);
        let type_col = if include_type {
            Some(PyList::empty(py))
        } else {
            None
        };
        let id_col = if include_id {
            Some(PyList::empty(py))
        } else {
            None
        };

        // Pre-create property column lists
        let prop_cols: Vec<pyo3::Bound<'_, PyList>> =
            prop_keys.iter().map(|_| PyList::empty(py)).collect();

        for (node_type, id, title, properties) in &nodes_data {
            title_col.append(py_out::value_to_py(py, title)?)?;
            if let Some(ref tc) = type_col {
                tc.append(*node_type)?;
            }
            if let Some(ref ic) = id_col {
                ic.append(py_out::value_to_py(py, id)?)?;
            }
            for (j, key) in prop_keys.iter().enumerate() {
                let val = properties.get(key).unwrap_or(&Value::Null);
                prop_cols[j].append(py_out::value_to_py(py, val)?)?;
            }
        }

        // Build the dict with ordered columns: type, title, id, ...properties
        let dict = PyDict::new(py);
        let columns = PyList::empty(py);

        if let Some(tc) = type_col {
            dict.set_item("type", tc)?;
            columns.append("type")?;
        }
        dict.set_item("title", title_col)?;
        columns.append("title")?;
        if let Some(ic) = id_col {
            dict.set_item("id", ic)?;
            columns.append("id")?;
        }
        for (j, key) in prop_keys.iter().enumerate() {
            dict.set_item(key, &prop_cols[j])?;
            columns.append(key)?;
        }

        let pd = py.import("pandas")?;

        if n == 0 {
            return pd.call_method0("DataFrame").map(|df| df.unbind());
        }

        // Create DataFrame with column order preserved
        let kwargs = PyDict::new(py);
        kwargs.set_item("columns", columns)?;
        let df = pd.call_method("DataFrame", (dict,), Some(&kwargs))?;
        Ok(df.unbind())
    }

    /// Returns the count of nodes in the current selection without materialization.
    /// If no selection/filter has been applied, returns the total graph node count.
    /// Much faster than get_nodes() when you only need the count.
    ///
    /// Example:
    ///     ```python
    ///     count = graph.node_count()           # total nodes in graph
    ///     count = graph.type_filter('User').node_count()  # filtered count
    ///     ```
    fn node_count(&self) -> usize {
        if self.selection.has_active_selection() {
            self.selection.current_node_count()
        } else {
            self.inner.graph.node_count()
        }
    }

    /// Returns the raw node indices in the current selection.
    /// Much faster than get_nodes() when you only need indices for further processing.
    ///
    /// Example:
    ///     ```python
    ///     indices = graph.type_filter('User').indices()
    ///     ```
    fn indices(&self) -> Vec<usize> {
        self.selection
            .current_node_indices()
            .map(|idx| idx.index())
            .collect()
    }

    /// Returns only id, title, and type for nodes - no other properties.
    /// Much faster than get_nodes() when you only need basic identification.
    ///
    /// Returns:
    ///     List of dicts with 'id', 'title', and 'type' keys only.
    ///
    /// Example:
    ///     ```python
    ///     ids = graph.type_filter('User').get_ids()
    ///     ```
    fn get_ids(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let key_type = pyo3::intern!(py, "type");
            let key_title = pyo3::intern!(py, "title");
            let key_id = pyo3::intern!(py, "id");
            let result = PyList::empty(py);

            for node_idx in self.selection.current_node_indices() {
                if let Some(node) = self.inner.get_node(node_idx) {
                    let dict = PyDict::new(py);
                    dict.set_item(key_type, &node.node_type)?;
                    dict.set_item(key_title, py_out::value_to_py(py, &node.title)?)?;
                    dict.set_item(key_id, py_out::value_to_py(py, &node.id)?)?;
                    result.append(dict)?;
                }
            }

            Ok(result.into())
        })
    }

    /// Returns just the raw ID values from the current selection as a flat list.
    /// This is the lightest possible output when you only need ID values.
    ///
    /// Much faster than get_nodes() or even get_ids() since it:
    /// - Skips title and type extraction
    /// - Returns raw values without dict wrapping
    /// - Minimal Python object creation
    ///
    /// Returns:
    ///     List of ID values (int, str, or whatever type the IDs are)
    ///
    /// Example:
    ///     ```python
    ///     # Get just the user IDs
    ///     user_ids = graph.type_filter('User').id_values()
    ///     # Returns: [1, 2, 3, 4, 5, ...]
    ///     ```
    fn id_values(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let result = PyList::empty(py);

            for node_idx in self.selection.current_node_indices() {
                if let Some(node) = self.inner.get_node(node_idx) {
                    result.append(py_out::value_to_py(py, &node.id)?)?;
                }
            }

            Ok(result.into())
        })
    }

    /// Look up a single node by its type and ID value. O(1) after first call.
    ///
    /// This is much faster than type_filter().filter() for single-node lookups
    /// because it uses a hash index instead of scanning all nodes.
    ///
    /// Args:
    ///     node_type: The type of node to look up (e.g., "User", "Product")
    ///     node_id: The ID value of the node
    ///
    /// Returns:
    ///     Dict with all node properties, or None if not found
    ///
    /// Example:
    ///     ```python
    ///     user = graph.get_node_by_id("User", 38870)
    ///     ```
    #[pyo3(signature = (node_type, node_id))]
    fn get_node_by_id(
        &mut self,
        node_type: &str,
        node_id: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        // Convert Python value to Rust Value
        let id_value = py_in::py_value_to_value(node_id)?;

        // Get mutable access to build index if needed
        let graph = Arc::make_mut(&mut self.inner);

        // This will build the index lazily if not already built
        let node_idx = match graph.lookup_by_id(node_type, &id_value) {
            Some(idx) => idx,
            None => return Ok(None),
        };

        // Get the node data
        let node = match graph.get_node(node_idx) {
            Some(n) => n,
            None => return Ok(None),
        };

        // Convert to Python dict
        let node_info = node.to_node_info();
        Python::attach(|py| {
            let dict = py_out::nodeinfo_to_pydict(py, &node_info)?;
            Ok(Some(dict))
        })
    }

    // ========================================================================
    // Code Entity Search Methods
    // ========================================================================

    /// Find code entities by name, with disambiguation context.
    ///
    /// Searches across code entity node types (Function, Struct, Class, Enum,
    /// Trait, Protocol, Interface, Module, Constant) for nodes matching the
    /// given name or qualified_name.
    ///
    /// Args:
    ///     name: Entity name to search for (e.g. "execute", "KnowledgeGraph")
    ///     node_type: Optional filter — only search this node type
    ///         (e.g. "Function", "Struct")
    ///
    /// Returns:
    ///     List of dicts, each containing: type, name, qualified_name,
    ///     file_path, line_number, and optionally signature and visibility
    ///
    /// Example:
    ///     ```python
    ///     results = graph.find("execute")
    ///     results = graph.find("KnowledgeGraph", node_type="Struct")
    ///     ```
    #[pyo3(signature = (name, node_type=None, match_type=None))]
    fn find(
        &self,
        name: &str,
        node_type: Option<&str>,
        match_type: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let match_type = match_type.unwrap_or("exact");
        let name_lower = name.to_lowercase();
        let types_to_search: Vec<&str> = match node_type {
            Some(nt) => vec![nt],
            None => Self::CODE_TYPES.to_vec(),
        };

        let mut results: Vec<schema::NodeInfo> = Vec::new();
        for nt in &types_to_search {
            if let Some(indices) = self.inner.type_indices.get(*nt) {
                for &idx in indices {
                    if let Some(node) = self.inner.get_node(idx) {
                        let matches = match match_type {
                            "contains" => {
                                Self::field_contains_ci(node, "name", &name_lower)
                                    || Self::field_contains_ci(node, "title", &name_lower)
                            }
                            "starts_with" => {
                                Self::field_starts_with_ci(node, "name", &name_lower)
                                    || Self::field_starts_with_ci(node, "title", &name_lower)
                            }
                            _ => {
                                // "exact" (default)
                                let name_val = Value::String(name.to_string());
                                node.get_field_ref("name")
                                    .map(|v| v == &name_val)
                                    .unwrap_or(false)
                                    || node
                                        .get_field_ref("title")
                                        .map(|v| v == &name_val)
                                        .unwrap_or(false)
                            }
                        };
                        if matches {
                            results.push(node.to_node_info());
                        }
                    }
                }
            }
        }

        Python::attach(|py| {
            let list = PyList::empty(py);
            for node_info in &results {
                let dict = py_out::nodeinfo_to_pydict(py, node_info)?;
                list.append(dict)?;
            }
            Ok(list.into_any().unbind())
        })
    }

    /// Get the source location of one or more code entities.
    ///
    /// Resolves names or qualified names to code entities and returns
    /// file paths and line ranges. Accepts a single string or a list.
    ///
    /// Args:
    ///     name: Entity name, qualified name, or list of names.
    ///     node_type: Optional node type hint ("Function", "Struct", etc.)
    ///
    /// Returns:
    ///     Single name: dict with file_path, line_number, end_line, line_count,
    ///         name, qualified_name, type, signature.
    ///     List of names: list of dicts (one per name).
    ///     Ambiguous names return {"name": ..., "ambiguous": true, "matches": [...]}.
    ///     Unknown names return {"name": ..., "error": "Node not found: ..."}.
    ///
    /// Example:
    ///     ```python
    ///     loc = graph.source("execute_single_clause")
    ///     locs = graph.source(["KnowledgeGraph", "build", "execute"])
    ///     ```
    #[pyo3(signature = (name, node_type=None))]
    fn source(&self, name: &Bound<'_, PyAny>, node_type: Option<&str>) -> PyResult<Py<PyAny>> {
        // Check if name is a list/sequence of strings
        if let Ok(list) = name.cast::<PyList>() {
            let names: Vec<String> = list.extract()?;
            return Python::attach(|py| {
                let result = PyList::empty(py);
                for n in &names {
                    let dict = self.source_one(py, n, node_type)?;
                    result.append(dict)?;
                }
                Ok(result.into_any().unbind())
            });
        }

        // Single string
        let name_str: String = name.extract()?;
        Python::attach(|py| self.source_one(py, &name_str, node_type))
    }

    /// Get the full neighborhood of a code entity.
    ///
    /// Returns the node's properties and all related entities grouped by
    /// relationship type. If the name is ambiguous (matches multiple nodes),
    /// returns the matches so you can refine with a qualified name.
    ///
    /// Args:
    ///     name: Entity name (e.g. "build") or qualified name
    ///         (e.g. "kglite.code_tree.builder.build")
    ///     node_type: Optional node type hint ("Function", "Struct", etc.)
    ///     hops: Max traversal depth for multi-hop neighbors (default 1)
    ///
    /// Returns:
    ///     Dict with "node" (properties), "defined_in" (file path), and
    ///     relationship groups (e.g. "HAS_METHOD", "CALLS", "CALLED_BY")
    ///
    /// Example:
    ///     ```python
    ///     ctx = graph.context("KnowledgeGraph")
    ///     ctx = graph.context("kglite.code_tree.builder.build", hops=2)
    ///     ```
    #[pyo3(signature = (name, node_type=None, hops=None))]
    fn context(
        &self,
        name: &str,
        node_type: Option<&str>,
        hops: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let hops = hops.unwrap_or(1);

        let (resolved, matches) = self.resolve_code_entity(name, node_type);

        let target_idx = match resolved {
            Some(idx) => idx,
            None => {
                return Python::attach(|py| {
                    let dict = PyDict::new(py);
                    if matches.is_empty() {
                        dict.set_item("error", format!("Node not found: {}", name))?;
                    } else {
                        dict.set_item("ambiguous", true)?;
                        let match_list = PyList::empty(py);
                        for (_, info) in &matches {
                            let d = py_out::nodeinfo_to_pydict(py, info)?;
                            match_list.append(d)?;
                        }
                        dict.set_item("matches", match_list)?;
                    }
                    Ok(dict.into_any().unbind())
                });
            }
        };

        let target_node = self
            .inner
            .get_node(target_idx)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Node disappeared"))?;

        // Phase 2: Build result dict
        Python::attach(|py| {
            let result = PyDict::new(py);

            // Node properties
            let node_info = target_node.to_node_info();
            let node_dict = py_out::nodeinfo_to_pydict(py, &node_info)?;
            result.set_item("node", &node_dict)?;

            // defined_in (file_path shortcut)
            if let Some(Value::String(fp)) = target_node.get_field_ref("file_path") {
                result.set_item("defined_in", fp)?;
            }

            // Phase 3: Collect neighbors, grouped by edge type
            // For hops > 1, do BFS expansion
            let neighbor_indices = if hops <= 1 {
                // Direct neighbors only
                let mut neighbors = HashSet::new();
                for edge in self
                    .inner
                    .graph
                    .edges_directed(target_idx, petgraph::Direction::Outgoing)
                {
                    neighbors.insert(edge.target());
                }
                for edge in self
                    .inner
                    .graph
                    .edges_directed(target_idx, petgraph::Direction::Incoming)
                {
                    neighbors.insert(edge.source());
                }
                neighbors
            } else {
                // BFS expansion for N hops
                let mut visited = HashSet::new();
                visited.insert(target_idx);
                let mut frontier = HashSet::new();
                frontier.insert(target_idx);

                for _ in 0..hops {
                    let mut next_frontier = HashSet::new();
                    for &node in &frontier {
                        for neighbor in self.inner.graph.neighbors_undirected(node) {
                            if visited.insert(neighbor) {
                                next_frontier.insert(neighbor);
                            }
                        }
                    }
                    if next_frontier.is_empty() {
                        break;
                    }
                    frontier = next_frontier;
                }
                visited.remove(&target_idx);
                visited
            };

            // Group outgoing edges by type
            let mut outgoing_groups: HashMap<String, Vec<NodeIndex>> = HashMap::new();
            let mut incoming_groups: HashMap<String, Vec<NodeIndex>> = HashMap::new();

            for edge in self
                .inner
                .graph
                .edges_directed(target_idx, petgraph::Direction::Outgoing)
            {
                let edge_type = &edge.weight().connection_type;
                let target = edge.target();
                if hops <= 1 || neighbor_indices.contains(&target) {
                    outgoing_groups
                        .entry(edge_type.clone())
                        .or_default()
                        .push(target);
                }
            }

            for edge in self
                .inner
                .graph
                .edges_directed(target_idx, petgraph::Direction::Incoming)
            {
                let edge_type = &edge.weight().connection_type;
                let source = edge.source();
                if hops <= 1 || neighbor_indices.contains(&source) {
                    incoming_groups
                        .entry(edge_type.clone())
                        .or_default()
                        .push(source);
                }
            }

            // For multi-hop: also collect edges between neighbor nodes
            if hops > 1 {
                for &n_idx in &neighbor_indices {
                    for edge in self
                        .inner
                        .graph
                        .edges_directed(n_idx, petgraph::Direction::Outgoing)
                    {
                        let t = edge.target();
                        if t != target_idx && neighbor_indices.contains(&t) {
                            let edge_type = &edge.weight().connection_type;
                            outgoing_groups
                                .entry(edge_type.clone())
                                .or_default()
                                .push(t);
                        }
                    }
                }
            }

            // Convert outgoing groups to Python
            for (edge_type, indices) in &outgoing_groups {
                let list = PyList::empty(py);
                let mut seen = HashSet::new();
                for &idx in indices {
                    if !seen.insert(idx) {
                        continue; // deduplicate
                    }
                    if let Some(node) = self.inner.get_node(idx) {
                        let info = node.to_node_info();
                        let d = py_out::nodeinfo_to_pydict(py, &info)?;
                        list.append(d)?;
                    }
                }
                result.set_item(edge_type.as_str(), list)?;
            }

            // Convert incoming groups to Python (prefix with "incoming_" to avoid collision)
            for (edge_type, indices) in &incoming_groups {
                let key = if outgoing_groups.contains_key(edge_type) {
                    format!("incoming_{}", edge_type)
                } else {
                    // Use a readable reverse name for common patterns
                    match edge_type.as_str() {
                        "CALLS" => "called_by".to_string(),
                        "HAS_METHOD" => "method_of".to_string(),
                        "DEFINES" => "defined_by".to_string(),
                        "USES_TYPE" => "used_by".to_string(),
                        "IMPLEMENTS" => "implemented_by".to_string(),
                        "EXTENDS" => "extended_by".to_string(),
                        _ => format!("incoming_{}", edge_type),
                    }
                };
                let list = PyList::empty(py);
                let mut seen = HashSet::new();
                for &idx in indices {
                    if !seen.insert(idx) {
                        continue;
                    }
                    if let Some(node) = self.inner.get_node(idx) {
                        let info = node.to_node_info();
                        let d = py_out::nodeinfo_to_pydict(py, &info)?;
                        list.append(d)?;
                    }
                }
                result.set_item(key.as_str(), list)?;
            }

            Ok(result.into_any().unbind())
        })
    }

    /// Get a table of contents for a file — all code entities defined in it.
    ///
    /// Returns entities sorted by line_number with a type summary.
    ///
    /// Args:
    ///     file_path: Path of the file (the File node's path).
    ///
    /// Returns:
    ///     Dict with "file" (path), "entities" (list of dicts sorted by
    ///     line_number, each with type, name, qualified_name, line_number,
    ///     end_line, and optionally signature), and "summary" (type -> count).
    ///     Returns {"error": "..."} if file not found.
    ///
    /// Example:
    ///     ```python
    ///     toc = graph.toc("src/graph/mod.rs")
    ///     ```
    #[pyo3(signature = (file_path))]
    fn toc(&self, file_path: &str) -> PyResult<Py<PyAny>> {
        let file_id = Value::String(file_path.to_string());

        // Find the File node by its id (path)
        let file_idx = if let Some(indices) = self.inner.type_indices.get("File") {
            indices
                .iter()
                .find(|&&idx| {
                    self.inner
                        .get_node(idx)
                        .map(|n| n.id == file_id)
                        .unwrap_or(false)
                })
                .copied()
        } else {
            None
        };

        let file_idx = match file_idx {
            Some(idx) => idx,
            None => {
                return Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("error", format!("File not found: {}", file_path))?;
                    Ok(dict.into_any().unbind())
                });
            }
        };

        // Collect all entities connected via outgoing DEFINES edges
        // (type, name, qualified_name, line_number, end_line, signature)
        let mut entities: Vec<(String, String, String, i64, i64, Option<String>)> = Vec::new();

        for edge in self
            .inner
            .graph
            .edges_directed(file_idx, petgraph::Direction::Outgoing)
        {
            if edge.weight().connection_type != "DEFINES" {
                continue;
            }
            if let Some(node) = self.inner.get_node(edge.target()) {
                let node_type = node.get_node_type_ref().to_string();
                let name = match &node.title {
                    Value::String(s) => s.clone(),
                    _ => String::new(),
                };
                let qname = match &node.id {
                    Value::String(s) => s.clone(),
                    _ => String::new(),
                };
                let line = match node.get_field_ref("line_number") {
                    Some(Value::Int64(n)) => *n,
                    _ => 0,
                };
                let end = match node.get_field_ref("end_line") {
                    Some(Value::Int64(n)) => *n,
                    _ => 0,
                };
                let sig = match node.get_field_ref("signature") {
                    Some(Value::String(s)) => Some(s.clone()),
                    _ => None,
                };
                entities.push((node_type, name, qname, line, end, sig));
            }
        }

        // Sort by line_number
        entities.sort_by_key(|e| e.3);

        // Build summary: type -> count
        let mut summary: HashMap<String, usize> = HashMap::new();
        for e in &entities {
            *summary.entry(e.0.clone()).or_insert(0) += 1;
        }

        Python::attach(|py| {
            let result = PyDict::new(py);
            result.set_item("file", file_path)?;

            let entity_list = PyList::empty(py);
            for (etype, name, qname, line, end, sig) in &entities {
                let d = PyDict::new(py);
                d.set_item("type", etype)?;
                d.set_item("name", name)?;
                d.set_item("qualified_name", qname)?;
                d.set_item("line_number", line)?;
                d.set_item("end_line", end)?;
                if let Some(s) = sig {
                    d.set_item("signature", s)?;
                }
                entity_list.append(d)?;
            }
            result.set_item("entities", entity_list)?;

            let summary_dict = PyDict::new(py);
            let mut sorted_summary: Vec<_> = summary.iter().collect();
            sorted_summary.sort_by_key(|(k, _)| (*k).clone());
            for (k, v) in sorted_summary {
                summary_dict.set_item(k.as_str(), v)?;
            }
            result.set_item("summary", summary_dict)?;

            Ok(result.into_any().unbind())
        })
    }

    /// Build ID indices for specified node types for faster get_node_by_id lookups.
    ///
    /// Call this after loading a graph if you plan to do many ID lookups.
    /// Indices are built lazily anyway, but this pre-builds them.
    ///
    /// Args:
    ///     node_types: List of node types to index. If None, indexes all types.
    ///
    /// Example:
    ///     ```python
    ///     graph.build_id_indices(["User", "Product"])
    ///     ```
    #[pyo3(signature = (node_types=None))]
    fn build_id_indices(&mut self, node_types: Option<Vec<String>>) {
        let graph = Arc::make_mut(&mut self.inner);

        match node_types {
            Some(types) => {
                for node_type in types {
                    graph.build_id_index(&node_type);
                }
            }
            None => {
                // Build for all existing types
                let types: Vec<String> = graph.type_indices.keys().cloned().collect();
                for node_type in types {
                    graph.build_id_index(&node_type);
                }
            }
        }
    }

    /// Rebuild all indexes from the current graph state.
    ///
    /// Reconstructs type_indices, property_indices, and composite_indices by
    /// scanning all live nodes. Clears lazy caches (id_indices, connection_types)
    /// so they rebuild on next access.
    ///
    /// Use after bulk mutations (especially Cypher DELETE/REMOVE) to ensure
    /// index consistency.
    ///
    /// Example:
    ///     ```python
    ///     graph.reindex()
    ///     ```
    fn reindex(&mut self) {
        let graph = Arc::make_mut(&mut self.inner);
        graph.reindex();
    }

    /// Compact the graph by removing tombstones left by node/edge deletions.
    ///
    /// With StableDiGraph, deletions leave holes in the internal storage.
    /// Over time, this wastes memory and degrades iteration performance.
    /// vacuum() rebuilds the graph with contiguous indices, then rebuilds all indexes.
    ///
    /// **Important**: This resets the current selection since node indices change.
    /// Call this between query chains, not in the middle of one.
    ///
    /// Returns:
    ///     dict: Statistics about the compaction:
    ///         - 'nodes_remapped': Number of nodes that were remapped
    ///         - 'tombstones_removed': Number of tombstone slots reclaimed
    ///
    /// Example:
    ///     ```python
    ///     info = graph.graph_info()
    ///     if info['fragmentation_ratio'] > 0.3:
    ///         result = graph.vacuum()
    ///         print(f"Reclaimed {result['tombstones_removed']} slots")
    ///     ```
    fn vacuum(&mut self) -> PyResult<Py<PyAny>> {
        let graph = get_graph_mut(&mut self.inner);

        let tombstones_before = graph.graph.node_bound() - graph.graph.node_count();
        let old_to_new = graph.vacuum();
        let nodes_remapped = old_to_new.len();

        // Reset selection — indices have changed
        if nodes_remapped > 0 {
            self.selection = CowSelection::new();
        }

        Python::attach(|py| {
            let result = PyDict::new(py);
            result.set_item("nodes_remapped", nodes_remapped)?;
            result.set_item("tombstones_removed", tombstones_before)?;
            Ok(result.into())
        })
    }

    /// Get diagnostic information about graph storage health.
    ///
    /// Returns a dictionary with storage metrics useful for deciding when
    /// to call vacuum() or reindex().
    ///
    /// Returns:
    ///     dict: Graph health metrics:
    ///         - 'node_count': Number of live nodes
    ///         - 'node_capacity': Upper bound of node indices (includes tombstones)
    ///         - 'node_tombstones': Number of wasted slots from deletions
    ///         - 'edge_count': Number of live edges
    ///         - 'fragmentation_ratio': Ratio of wasted storage (0.0 = clean)
    ///         - 'type_count': Number of distinct node types
    ///         - 'property_index_count': Number of single-property indexes
    ///         - 'composite_index_count': Number of composite indexes
    ///
    /// Example:
    ///     ```python
    ///     info = graph.graph_info()
    ///     if info['fragmentation_ratio'] > 0.3:
    ///         graph.vacuum()
    ///     ```
    fn graph_info(&self) -> PyResult<Py<PyAny>> {
        let info = self.inner.graph_info();
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("node_count", info.node_count)?;
            dict.set_item("node_capacity", info.node_capacity)?;
            dict.set_item("node_tombstones", info.node_tombstones)?;
            dict.set_item("edge_count", info.edge_count)?;
            dict.set_item("fragmentation_ratio", info.fragmentation_ratio)?;
            dict.set_item("type_count", info.type_count)?;
            dict.set_item("property_index_count", info.property_index_count)?;
            dict.set_item("composite_index_count", info.composite_index_count)?;
            dict.set_item("format_version", self.inner.save_metadata.format_version)?;
            dict.set_item("library_version", &self.inner.save_metadata.library_version)?;
            Ok(dict.into())
        })
    }

    /// Configure automatic vacuum after DELETE operations.
    ///
    /// When enabled, the graph automatically compacts itself after Cypher DELETE
    /// operations if the fragmentation ratio exceeds the threshold and there are
    /// more than 100 tombstones.
    ///
    /// Args:
    ///     threshold: A float between 0.0 and 1.0, or None to disable.
    ///         Default is 0.3 (30% fragmentation triggers vacuum).
    ///         Set to None to disable auto-vacuum entirely.
    ///
    /// Example:
    ///     ```python
    ///     graph.set_auto_vacuum(0.2)   # more aggressive — vacuum at 20% fragmentation
    ///     graph.set_auto_vacuum(None)  # disable auto-vacuum
    ///     graph.set_auto_vacuum(0.3)   # restore default
    ///     ```
    #[pyo3(signature = (threshold))]
    fn set_auto_vacuum(&mut self, threshold: Option<f64>) -> PyResult<()> {
        if let Some(t) = threshold {
            if !(0.0..=1.0).contains(&t) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "threshold must be between 0.0 and 1.0, or None to disable",
                ));
            }
        }
        let graph = get_graph_mut(&mut self.inner);
        graph.auto_vacuum_threshold = threshold;
        Ok(())
    }

    /// Returns a dict of {node_type: count} using the type index (O(type_count)).
    fn node_type_counts(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            for (node_type, indices) in &self.inner.type_indices {
                dict.set_item(node_type, indices.len())?;
            }
            Ok(dict.into())
        })
    }

    #[pyo3(signature = (indices=None, parent_info=None, include_node_properties=None,
                        flatten_single_parent=true))]
    fn get_connections(
        &self,
        indices: Option<Vec<usize>>,
        parent_info: Option<bool>,
        include_node_properties: Option<bool>,
        flatten_single_parent: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let connections = data_retrieval::get_connections(
            &self.inner,
            &self.selection,
            None,
            indices.as_deref(),
            include_node_properties.unwrap_or(true),
        );
        Python::attach(|py| {
            py_out::level_connections_to_pydict(
                py,
                &connections,
                parent_info,
                flatten_single_parent,
            )
        })
    }

    #[pyo3(signature = (max_nodes=None, indices=None, flatten_single_parent=None))]
    fn get_titles(
        &self,
        max_nodes: Option<usize>,
        indices: Option<Vec<usize>>,
        flatten_single_parent: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let values = data_retrieval::get_property_values(
            &self.inner,
            &self.selection,
            None,
            &["title"],
            indices.as_deref(),
            max_nodes,
        );
        Python::attach(|py| {
            py_out::level_single_values_to_pydict(py, &values, flatten_single_parent)
        })
    }

    /// Returns a string representation of the query execution plan.
    ///
    /// Shows each operation in the query chain with estimated and actual row counts.
    /// Example output: "TYPE_FILTER Prospect (6775 nodes) -> TRAVERSE HAS_ESTIMATE (10954 nodes)"
    fn explain(&self) -> PyResult<String> {
        let plan = self.selection.get_execution_plan();
        if plan.is_empty() {
            return Ok("No query operations recorded".to_string());
        }

        let steps: Vec<String> = plan
            .iter()
            .map(|step| {
                let type_info = step
                    .node_type
                    .as_ref()
                    .map(|t| format!(" {}", t))
                    .unwrap_or_default();
                let rows = step.actual_rows.unwrap_or(step.estimated_rows);
                format!("{}{} ({} nodes)", step.operation, type_info, rows)
            })
            .collect();

        Ok(steps.join(" -> "))
    }

    #[pyo3(signature = (properties, max_nodes=None, indices=None, flatten_single_parent=None))]
    fn get_properties(
        &self,
        properties: Vec<String>,
        max_nodes: Option<usize>,
        indices: Option<Vec<usize>>,
        flatten_single_parent: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let property_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
        let values = data_retrieval::get_property_values(
            &self.inner,
            &self.selection,
            None,
            &property_refs,
            indices.as_deref(),
            max_nodes,
        );
        Python::attach(|py| py_out::level_values_to_pydict(py, &values, flatten_single_parent))
    }

    #[allow(clippy::too_many_arguments)]
    fn unique_values(
        &mut self,
        property: String,
        group_by_parent: Option<bool>,
        level_index: Option<usize>,
        indices: Option<Vec<usize>>,
        store_as: Option<&str>,
        max_length: Option<usize>,
        keep_selection: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let values = data_retrieval::get_unique_values(
            &self.inner,
            &self.selection,
            &property,
            level_index,
            group_by_parent.unwrap_or(true),
            indices.as_deref(),
        );

        if let Some(target_property) = store_as {
            let nodes = data_retrieval::format_unique_values_for_storage(&values, max_length);

            let graph = get_graph_mut(&mut self.inner);

            maintain_graph::update_node_properties(graph, &nodes, target_property)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

            if !keep_selection.unwrap_or(false) {
                self.selection.clear();
            }

            Python::attach(|py| Ok(Py::new(py, self.clone())?.into_any()))
        } else {
            Python::attach(|py| py_out::level_unique_values_to_pydict(py, &values))
        }
    }

    #[pyo3(signature = (connection_type, level_index=None, direction=None, filter_target=None, filter_connection=None, sort_target=None, max_nodes=None, new_level=None))]
    #[allow(clippy::too_many_arguments)]
    fn traverse(
        &mut self,
        connection_type: String,
        level_index: Option<usize>,
        direction: Option<String>,
        filter_target: Option<&Bound<'_, PyDict>>,
        filter_connection: Option<&Bound<'_, PyDict>>,
        sort_target: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>,
        new_level: Option<bool>,
    ) -> PyResult<Self> {
        let mut new_kg = self.clone();

        // Estimate based on current selection (source nodes) - use node_count() to avoid allocation
        let estimated = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        let conditions = if let Some(cond) = filter_target {
            Some(py_in::pydict_to_filter_conditions(cond)?)
        } else {
            None
        };

        let conn_conditions = if let Some(cond) = filter_connection {
            Some(py_in::pydict_to_filter_conditions(cond)?)
        } else {
            None
        };

        let sort_fields = if let Some(spec) = sort_target {
            Some(py_in::parse_sort_fields(spec, None)?)
        } else {
            None
        };

        traversal_methods::make_traversal(
            &self.inner,
            &mut new_kg.selection,
            connection_type.clone(),
            level_index,
            direction,
            conditions.as_ref(),
            conn_conditions.as_ref(),
            sort_fields.as_ref(),
            max_nodes,
            new_level,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Record actual result - use node_count() to avoid allocation
        let actual = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);
        new_kg.selection.add_plan_step(
            PlanStep::new("TRAVERSE", Some(&connection_type), estimated).with_actual_rows(actual),
        );

        Ok(new_kg)
    }

    fn selection_to_new_connections(
        &mut self,
        connection_type: String,
        keep_selection: Option<bool>,
        conflict_handling: Option<String>,
    ) -> PyResult<Self> {
        let graph = get_graph_mut(&mut self.inner);

        let result = maintain_graph::selection_to_new_connections(
            graph,
            &self.selection,
            connection_type,
            conflict_handling,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let mut new_kg = KnowledgeGraph {
            inner: self.inner.clone(),
            selection: if keep_selection.unwrap_or(false) {
                self.selection.clone()
            } else {
                CowSelection::new()
            },
            reports: self.reports.clone(), // Copy over existing reports
            last_mutation_stats: None,
            embedder: Python::attach(|py| self.embedder.as_ref().map(|m| m.clone_ref(py))),
        };

        // Store the report in the new graph
        new_kg.add_report(OperationReport::ConnectionOperation(result));

        // Just return the new KnowledgeGraph
        Ok(new_kg)
    }

    #[pyo3(signature = (property=None, filter=None, sort=None, max_nodes=None, store_as=None, max_length=None, keep_selection=None))]
    #[allow(clippy::too_many_arguments)]
    fn children_properties_to_list(
        &mut self,
        property: Option<&str>,
        filter: Option<&Bound<'_, PyDict>>,
        sort: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>,
        store_as: Option<&str>,
        max_length: Option<usize>,
        keep_selection: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let property_name = property.unwrap_or("title");

        // Apply filtering and sorting if needed
        let mut filtered_kg = self.clone();

        if let Some(filter_dict) = filter {
            let conditions = py_in::pydict_to_filter_conditions(filter_dict)?;
            let sort_fields = match sort {
                Some(spec) => Some(py_in::parse_sort_fields(spec, None)?),
                None => None,
            };

            filtering_methods::filter_nodes(
                &self.inner,
                &mut filtered_kg.selection,
                conditions,
                sort_fields,
                max_nodes,
            )
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        } else if let Some(spec) = sort {
            let sort_fields = py_in::parse_sort_fields(spec, None)?;

            filtering_methods::sort_nodes(&self.inner, &mut filtered_kg.selection, sort_fields)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

            if let Some(max) = max_nodes {
                filtering_methods::limit_nodes_per_group(
                    &self.inner,
                    &mut filtered_kg.selection,
                    max,
                )
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            }
        } else if let Some(max) = max_nodes {
            filtering_methods::limit_nodes_per_group(&self.inner, &mut filtered_kg.selection, max)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        }

        // Generate the property lists with titles already included
        let property_groups = traversal_methods::get_children_properties(
            &filtered_kg.inner,
            &filtered_kg.selection,
            property_name,
        );

        // If store_as is not provided, return the properties as a dictionary
        if store_as.is_none() {
            // Format for dictionary display
            let dict_pairs = traversal_methods::format_for_dictionary(&property_groups, max_length);

            return Python::attach(|py| py_out::string_pairs_to_pydict(py, &dict_pairs));
        }

        // Format for storage
        let nodes = traversal_methods::format_for_storage(&property_groups, max_length);

        let graph = get_graph_mut(&mut self.inner);

        let result = maintain_graph::update_node_properties(graph, &nodes, store_as.unwrap())
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let mut new_kg = KnowledgeGraph {
            inner: self.inner.clone(),
            selection: if keep_selection.unwrap_or(false) {
                self.selection.clone()
            } else {
                CowSelection::new()
            },
            reports: self.reports.clone(),
            last_mutation_stats: None,
            embedder: Python::attach(|py| self.embedder.as_ref().map(|m| m.clone_ref(py))),
        };

        // Store the report
        new_kg.add_report(OperationReport::NodeOperation(result));

        // Return the updated graph (no report in return value)
        Python::attach(|py| Ok(Py::new(py, new_kg)?.into_any()))
    }

    #[pyo3(signature = (property, level_index=None))]
    fn statistics(&self, property: &str, level_index: Option<usize>) -> PyResult<Py<PyAny>> {
        let pairs = statistics_methods::get_parent_child_pairs(&self.selection, level_index);
        let stats = statistics_methods::calculate_property_stats(&self.inner, &pairs, property);
        py_out::convert_stats_for_python(stats)
    }

    #[pyo3(signature = (expression, level_index=None, store_as=None, keep_selection=None, aggregate_connections=None))]
    fn calculate(
        &mut self,
        expression: &str,
        level_index: Option<usize>,
        store_as: Option<&str>,
        keep_selection: Option<bool>,
        aggregate_connections: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        // If we're storing results, we'll need a mutable graph
        if let Some(target_property) = store_as {
            let graph = get_graph_mut(&mut self.inner);

            let process_result = calculations::process_equation(
                graph,
                &self.selection,
                expression,
                level_index,
                Some(target_property),
                aggregate_connections,
            );

            match process_result {
                Ok(calculations::EvaluationResult::Stored(report)) => {
                    let mut new_kg = KnowledgeGraph {
                        inner: self.inner.clone(),
                        selection: if keep_selection.unwrap_or(false) {
                            self.selection.clone()
                        } else {
                            CowSelection::new()
                        },
                        reports: self.reports.clone(), // Copy existing reports
                        last_mutation_stats: None,
                        embedder: Python::attach(|py| {
                            self.embedder.as_ref().map(|m| m.clone_ref(py))
                        }),
                    };

                    // Store the calculation report
                    new_kg.add_report(OperationReport::CalculationOperation(report));

                    Python::attach(|py| Ok(Py::new(py, new_kg)?.into_any()))
                }
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unexpected result type when storing calculation result",
                )),
                Err(e) => {
                    let error_msg = format!("Error evaluating expression '{}': {}", expression, e);
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(error_msg))
                }
            }
        } else {
            // Just computing without storing - no need to modify graph
            let process_result = calculations::process_equation(
                &mut (*self.inner).clone(), // Create a temporary clone just for calculation
                &self.selection,
                expression,
                level_index,
                None,
                aggregate_connections,
            );

            // Handle regular errors with descriptive messages
            match process_result {
                Ok(calculations::EvaluationResult::Computed(results)) => {
                    // Check for errors
                    let error_count = results.iter().filter(|r| r.error_msg.is_some()).count();
                    if error_count == results.len() && !results.is_empty() {
                        if let Some(first_error) = results.iter().find(|r| r.error_msg.is_some()) {
                            if let Some(error_text) = &first_error.error_msg {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    format!(
                                        "Error in calculation '{}': {}",
                                        expression, error_text
                                    ),
                                ));
                            }
                        }
                    }

                    // Filter out results with errors
                    let valid_results: Vec<StatResult> = results
                        .into_iter()
                        .filter(|r| r.error_msg.is_none())
                        .collect();

                    if valid_results.is_empty() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "No valid results found for expression '{}'",
                            expression
                        )));
                    }

                    py_out::convert_computation_results_for_python(valid_results)
                }
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unexpected result type when computing",
                )),
                Err(e) => {
                    let error_msg = format!("Error evaluating expression '{}': {}", expression, e);
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(error_msg))
                }
            }
        }
    }

    #[pyo3(signature = (level_index=None, group_by_parent=None, store_as=None, keep_selection=None))]
    fn count(
        &mut self,
        level_index: Option<usize>,
        group_by_parent: Option<bool>,
        store_as: Option<&str>,
        keep_selection: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        // Default to grouping by parent if we have a nested structure
        let has_multiple_levels = self.selection.get_level_count() > 1;
        // Use the provided group_by_parent if given, otherwise default based on structure
        let use_grouping = group_by_parent.unwrap_or(has_multiple_levels);

        if let Some(target_property) = store_as {
            let graph = get_graph_mut(&mut self.inner);

            let result = match calculations::store_count_results(
                graph,
                &self.selection,
                level_index,
                use_grouping,
                target_property,
            ) {
                Ok(report) => report,
                Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
            };

            let mut new_kg = KnowledgeGraph {
                inner: self.inner.clone(),
                selection: if keep_selection.unwrap_or(false) {
                    self.selection.clone()
                } else {
                    CowSelection::new()
                },
                reports: self.reports.clone(), // Copy existing reports
                last_mutation_stats: None,
                embedder: Python::attach(|py| self.embedder.as_ref().map(|m| m.clone_ref(py))),
            };

            // Add the report
            new_kg.add_report(OperationReport::CalculationOperation(result));

            Python::attach(|py| Ok(Py::new(py, new_kg)?.into_any()))
        } else if use_grouping {
            // Return counts grouped by parent
            let counts =
                calculations::count_nodes_by_parent(&self.inner, &self.selection, level_index);
            py_out::convert_computation_results_for_python(counts)
        } else {
            // Simple flat count
            let count = calculations::count_nodes_in_level(&self.selection, level_index);
            Python::attach(|py| count.into_py_any(py))
        }
    }

    fn get_schema(&self) -> PyResult<String> {
        let schema_string = debugging::get_schema_string(&self.inner);
        Ok(schema_string)
    }

    /// Return a minimal XML string describing this graph for AI agents.
    fn agent_describe(&self) -> PyResult<String> {
        Ok(introspection::compute_agent_description(&self.inner))
    }

    fn get_selection(&self) -> PyResult<String> {
        Ok(debugging::get_selection_string(
            &self.inner,
            &self.selection,
        ))
    }

    // ================================================================
    // Schema Introspection
    // ================================================================

    /// Return a full schema overview of the graph.
    fn schema(&self) -> PyResult<Py<PyAny>> {
        let overview = introspection::compute_schema(&self.inner);
        Python::attach(|py| {
            let result = PyDict::new(py);

            // node_types
            let node_types_dict = PyDict::new(py);
            for (nt, info) in &overview.node_types {
                let type_dict = PyDict::new(py);
                type_dict.set_item("count", info.count)?;
                let props_dict = PyDict::new(py);
                for (k, v) in &info.properties {
                    props_dict.set_item(k.as_str(), v.as_str())?;
                }
                type_dict.set_item("properties", props_dict)?;
                node_types_dict.set_item(nt.as_str(), type_dict)?;
            }
            result.set_item("node_types", node_types_dict)?;

            // connection_types
            let conn_dict = PyDict::new(py);
            for ct in &overview.connection_types {
                let ct_dict = PyDict::new(py);
                ct_dict.set_item("count", ct.count)?;
                ct_dict.set_item("source_types", &ct.source_types)?;
                ct_dict.set_item("target_types", &ct.target_types)?;
                conn_dict.set_item(ct.connection_type.as_str(), ct_dict)?;
            }
            result.set_item("connection_types", conn_dict)?;

            result.set_item("indexes", &overview.indexes)?;
            result.set_item("node_count", overview.node_count)?;
            result.set_item("edge_count", overview.edge_count)?;

            Ok(result.into())
        })
    }

    /// Return all connection types with counts and endpoint type sets.
    #[pyo3(name = "connection_types")]
    fn connection_types_info(&self) -> PyResult<Py<PyAny>> {
        let stats = introspection::compute_connection_type_stats(&self.inner);
        Python::attach(|py| {
            let result_list = PyList::empty(py);
            for ct in &stats {
                let ct_dict = PyDict::new(py);
                ct_dict.set_item("type", ct.connection_type.as_str())?;
                ct_dict.set_item("count", ct.count)?;
                ct_dict.set_item("source_types", &ct.source_types)?;
                ct_dict.set_item("target_types", &ct.target_types)?;
                result_list.append(ct_dict)?;
            }
            Ok(result_list.into())
        })
    }

    /// Return property statistics for a node type.
    #[pyo3(signature = (node_type, max_values=20))]
    fn properties(&self, node_type: &str, max_values: usize) -> PyResult<Py<PyAny>> {
        let stats = introspection::compute_property_stats(&self.inner, node_type, max_values)
            .map_err(PyErr::new::<pyo3::exceptions::PyKeyError, _>)?;
        Python::attach(|py| {
            let result = PyDict::new(py);
            for prop in &stats {
                let prop_dict = PyDict::new(py);
                prop_dict.set_item("type", prop.type_string.as_str())?;
                prop_dict.set_item("non_null", prop.non_null)?;
                prop_dict.set_item("unique", prop.unique)?;
                if let Some(ref vals) = prop.values {
                    let py_vals = PyList::empty(py);
                    for v in vals {
                        py_vals.append(py_out::value_to_py(py, v)?)?;
                    }
                    prop_dict.set_item("values", py_vals)?;
                }
                result.set_item(prop.property_name.as_str(), prop_dict)?;
            }
            Ok(result.into())
        })
    }

    /// Return connection topology for a node type (outgoing and incoming).
    fn neighbors_schema(&self, node_type: &str) -> PyResult<Py<PyAny>> {
        let ns = introspection::compute_neighbors_schema(&self.inner, node_type)
            .map_err(PyErr::new::<pyo3::exceptions::PyKeyError, _>)?;
        Python::attach(|py| {
            let result = PyDict::new(py);

            let out_list = PyList::empty(py);
            for nc in &ns.outgoing {
                let d = PyDict::new(py);
                d.set_item("connection_type", nc.connection_type.as_str())?;
                d.set_item("target_type", nc.other_type.as_str())?;
                d.set_item("count", nc.count)?;
                out_list.append(d)?;
            }
            result.set_item("outgoing", out_list)?;

            let in_list = PyList::empty(py);
            for nc in &ns.incoming {
                let d = PyDict::new(py);
                d.set_item("connection_type", nc.connection_type.as_str())?;
                d.set_item("source_type", nc.other_type.as_str())?;
                d.set_item("count", nc.count)?;
                in_list.append(d)?;
            }
            result.set_item("incoming", in_list)?;

            Ok(result.into())
        })
    }

    /// Return a quick sample of nodes for a given type.
    #[pyo3(signature = (node_type, n=5))]
    fn sample(&self, node_type: &str, n: usize) -> PyResult<Py<PyAny>> {
        let nodes = introspection::compute_sample(&self.inner, node_type, n)
            .map_err(PyErr::new::<pyo3::exceptions::PyKeyError, _>)?;
        let view = cypher::ResultView::from_nodes(nodes.into_iter());
        Python::attach(|py| Py::new(py, view).map(|v| v.into_any()))
    }

    /// Return a unified list of all indexes (single-property and composite).
    fn indexes(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let result_list = PyList::empty(py);

            for (node_type, property) in self.inner.property_indices.keys() {
                let d = PyDict::new(py);
                d.set_item("node_type", node_type.as_str())?;
                d.set_item("property", property.as_str())?;
                d.set_item("type", "equality")?;
                result_list.append(d)?;
            }

            for (node_type, properties) in self.inner.composite_indices.keys() {
                let d = PyDict::new(py);
                d.set_item("node_type", node_type.as_str())?;
                d.set_item("properties", properties)?;
                d.set_item("type", "composite")?;
                result_list.append(d)?;
            }

            Ok(result_list.into())
        })
    }

    fn clear(&mut self) -> PyResult<()> {
        self.selection.clear();
        Ok(())
    }

    fn save(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        // Prep phase (quick): stamp metadata, snapshot index keys
        io_operations::prepare_save(&mut self.inner);
        // Heavy phase: serialize, compress, write — release GIL for other Python threads
        let inner = self.inner.clone();
        let path_owned = path.to_string();
        py.detach(move || io_operations::write_graph_to_file(&inner, &path_owned))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))
    }

    /// Get the most recent operation report as a Python dictionary
    fn get_last_report(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            if let Some(report) = self.reports.get_last_report() {
                match report {
                    OperationReport::NodeOperation(node_report) => {
                        let report_dict = PyDict::new(py);
                        report_dict.set_item("operation", &node_report.operation_type)?;
                        report_dict.set_item("timestamp", node_report.timestamp.to_rfc3339())?;
                        report_dict.set_item("nodes_created", node_report.nodes_created)?;
                        report_dict.set_item("nodes_updated", node_report.nodes_updated)?;
                        report_dict.set_item("nodes_skipped", node_report.nodes_skipped)?;
                        report_dict
                            .set_item("processing_time_ms", node_report.processing_time_ms)?;

                        // Add errors array if there are any
                        if !node_report.errors.is_empty() {
                            report_dict.set_item("errors", &node_report.errors)?;
                            report_dict.set_item("has_errors", true)?;
                        } else {
                            report_dict.set_item("has_errors", false)?;
                        }

                        Ok(report_dict.into())
                    }
                    OperationReport::ConnectionOperation(conn_report) => {
                        let report_dict = PyDict::new(py);
                        report_dict.set_item("operation", &conn_report.operation_type)?;
                        report_dict.set_item("timestamp", conn_report.timestamp.to_rfc3339())?;
                        report_dict
                            .set_item("connections_created", conn_report.connections_created)?;
                        report_dict
                            .set_item("connections_skipped", conn_report.connections_skipped)?;
                        report_dict.set_item(
                            "property_fields_tracked",
                            conn_report.property_fields_tracked,
                        )?;
                        report_dict
                            .set_item("processing_time_ms", conn_report.processing_time_ms)?;

                        // Add errors array if there are any
                        if !conn_report.errors.is_empty() {
                            report_dict.set_item("errors", &conn_report.errors)?;
                            report_dict.set_item("has_errors", true)?;
                        } else {
                            report_dict.set_item("has_errors", false)?;
                        }

                        Ok(report_dict.into())
                    }
                    OperationReport::CalculationOperation(calc_report) => {
                        let report_dict = PyDict::new(py);
                        report_dict.set_item("operation", &calc_report.operation_type)?;
                        report_dict.set_item("timestamp", calc_report.timestamp.to_rfc3339())?;
                        report_dict.set_item("expression", &calc_report.expression)?;
                        report_dict.set_item("nodes_processed", calc_report.nodes_processed)?;
                        report_dict.set_item("nodes_updated", calc_report.nodes_updated)?;
                        report_dict.set_item("nodes_with_errors", calc_report.nodes_with_errors)?;
                        report_dict
                            .set_item("processing_time_ms", calc_report.processing_time_ms)?;
                        report_dict.set_item("is_aggregation", calc_report.is_aggregation)?;

                        // Add errors array if there are any
                        if !calc_report.errors.is_empty() {
                            report_dict.set_item("errors", &calc_report.errors)?;
                            report_dict.set_item("has_errors", true)?;
                        } else {
                            report_dict.set_item("has_errors", false)?;
                        }

                        Ok(report_dict.into())
                    }
                }
            } else {
                let empty_dict = PyDict::new(py);
                Ok(empty_dict.into())
            }
        })
    }

    /// Get the last operation index (a sequential ID of operations performed)
    fn get_operation_index(&self) -> usize {
        self.reports.get_last_operation_index()
    }

    /// Get all report history as a list of dictionaries
    fn get_report_history(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            // Create an empty list with PyList::empty
            let report_list = PyList::empty(py);

            for report in self.reports.get_all_reports() {
                let report_dict = match report {
                    OperationReport::NodeOperation(node_report) => {
                        let dict = PyDict::new(py);
                        dict.set_item("operation", &node_report.operation_type)?;
                        dict.set_item("timestamp", node_report.timestamp.to_rfc3339())?;
                        dict.set_item("nodes_created", node_report.nodes_created)?;
                        dict.set_item("nodes_updated", node_report.nodes_updated)?;
                        dict.set_item("nodes_skipped", node_report.nodes_skipped)?;
                        dict.set_item("processing_time_ms", node_report.processing_time_ms)?;

                        // Add errors array if there are any
                        if !node_report.errors.is_empty() {
                            dict.set_item("errors", &node_report.errors)?;
                            dict.set_item("has_errors", true)?;
                        } else {
                            dict.set_item("has_errors", false)?;
                        }

                        dict
                    }
                    OperationReport::ConnectionOperation(conn_report) => {
                        let dict = PyDict::new(py);
                        dict.set_item("operation", &conn_report.operation_type)?;
                        dict.set_item("timestamp", conn_report.timestamp.to_rfc3339())?;
                        dict.set_item("connections_created", conn_report.connections_created)?;
                        dict.set_item("connections_skipped", conn_report.connections_skipped)?;
                        dict.set_item(
                            "property_fields_tracked",
                            conn_report.property_fields_tracked,
                        )?;
                        dict.set_item("processing_time_ms", conn_report.processing_time_ms)?;

                        // Add errors array if there are any
                        if !conn_report.errors.is_empty() {
                            dict.set_item("errors", &conn_report.errors)?;
                            dict.set_item("has_errors", true)?;
                        } else {
                            dict.set_item("has_errors", false)?;
                        }

                        dict
                    }
                    OperationReport::CalculationOperation(calc_report) => {
                        let dict = PyDict::new(py);
                        dict.set_item("operation", &calc_report.operation_type)?;
                        dict.set_item("timestamp", calc_report.timestamp.to_rfc3339())?;
                        dict.set_item("expression", &calc_report.expression)?;
                        dict.set_item("nodes_processed", calc_report.nodes_processed)?;
                        dict.set_item("nodes_updated", calc_report.nodes_updated)?;
                        dict.set_item("nodes_with_errors", calc_report.nodes_with_errors)?;
                        dict.set_item("processing_time_ms", calc_report.processing_time_ms)?;
                        dict.set_item("is_aggregation", calc_report.is_aggregation)?;

                        // Add errors array if there are any
                        if !calc_report.errors.is_empty() {
                            dict.set_item("errors", &calc_report.errors)?;
                            dict.set_item("has_errors", true)?;
                        } else {
                            dict.set_item("has_errors", false)?;
                        }

                        dict
                    }
                };
                report_list.append(report_dict)?;
            }
            Ok(report_list.into())
        })
    }

    /// Perform union of two selections - combines all nodes from both selections
    /// Returns a new KnowledgeGraph with the union of both selections
    fn union(&self, other: &Self) -> PyResult<Self> {
        let mut new_kg = self.clone();
        set_operations::union_selections(&mut new_kg.selection, &other.selection)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(new_kg)
    }

    /// Perform intersection of two selections - keeps only nodes present in both
    /// Returns a new KnowledgeGraph with only nodes that exist in both selections
    fn intersection(&self, other: &Self) -> PyResult<Self> {
        let mut new_kg = self.clone();
        set_operations::intersection_selections(&mut new_kg.selection, &other.selection)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(new_kg)
    }

    /// Perform difference of two selections - keeps nodes in self but not in other
    /// Returns a new KnowledgeGraph with nodes from self that are not in other
    fn difference(&self, other: &Self) -> PyResult<Self> {
        let mut new_kg = self.clone();
        set_operations::difference_selections(&mut new_kg.selection, &other.selection)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(new_kg)
    }

    /// Perform symmetric difference of two selections - keeps nodes in either but not both
    /// Returns a new KnowledgeGraph with nodes that are in exactly one of the selections
    fn symmetric_difference(&self, other: &Self) -> PyResult<Self> {
        let mut new_kg = self.clone();
        set_operations::symmetric_difference_selections(&mut new_kg.selection, &other.selection)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        Ok(new_kg)
    }

    // ========================================================================
    // Schema Definition & Validation Methods
    // ========================================================================

    /// Define the expected schema for the graph
    ///
    /// Args:
    ///     schema_dict: A dictionary defining the schema with the following structure:
    ///         {
    ///             'nodes': {
    ///                 'NodeType': {
    ///                     'required': ['field1', 'field2'],  # Required fields
    ///                     'optional': ['field3'],            # Optional fields (for documentation)
    ///                     'types': {'field1': 'string', 'field2': 'integer'}  # Field types
    ///                 }
    ///             },
    ///             'connections': {
    ///                 'CONNECTION_TYPE': {
    ///                     'source': 'SourceNodeType',
    ///                     'target': 'TargetNodeType',
    ///                     'cardinality': 'one-to-many',  # Optional
    ///                     'required_properties': ['prop1'],  # Optional
    ///                     'property_types': {'prop1': 'float'}  # Optional
    ///                 }
    ///             }
    ///         }
    ///
    /// Returns:
    ///     Self with schema defined
    fn define_schema(&mut self, schema_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut schema = SchemaDefinition::new();

        // Parse node schemas
        if let Some(nodes_dict) = schema_dict.get_item("nodes")? {
            if let Ok(nodes) = nodes_dict.cast::<PyDict>() {
                for (node_type_key, node_schema_val) in nodes.iter() {
                    let node_type: String = node_type_key.extract()?;
                    let node_schema_dict = node_schema_val.cast::<PyDict>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                            "Schema for node type '{}' must be a dictionary",
                            node_type
                        ))
                    })?;

                    let mut node_schema = NodeSchemaDefinition::default();

                    // Parse required fields
                    if let Some(required) = node_schema_dict.get_item("required")? {
                        node_schema.required_fields = required.extract::<Vec<String>>()?;
                    }

                    // Parse optional fields
                    if let Some(optional) = node_schema_dict.get_item("optional")? {
                        node_schema.optional_fields = optional.extract::<Vec<String>>()?;
                    }

                    // Parse field types
                    if let Some(types) = node_schema_dict.get_item("types")? {
                        let types_dict = types.cast::<PyDict>().map_err(|_| {
                            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "types must be a dictionary",
                            )
                        })?;
                        for (field, type_val) in types_dict.iter() {
                            node_schema
                                .field_types
                                .insert(field.extract::<String>()?, type_val.extract::<String>()?);
                        }
                    }

                    schema.add_node_schema(node_type, node_schema);
                }
            }
        }

        // Parse connection schemas
        if let Some(connections_dict) = schema_dict.get_item("connections")? {
            if let Ok(connections) = connections_dict.cast::<PyDict>() {
                for (conn_type_key, conn_schema_val) in connections.iter() {
                    let conn_type: String = conn_type_key.extract()?;
                    let conn_schema_dict = conn_schema_val.cast::<PyDict>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                            "Schema for connection type '{}' must be a dictionary",
                            conn_type
                        ))
                    })?;

                    let source_type: String = conn_schema_dict
                        .get_item("source")?
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                                "Connection '{}' missing required 'source' field",
                                conn_type
                            ))
                        })?
                        .extract()?;

                    let target_type: String = conn_schema_dict
                        .get_item("target")?
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                                "Connection '{}' missing required 'target' field",
                                conn_type
                            ))
                        })?
                        .extract()?;

                    let mut conn_schema = ConnectionSchemaDefinition {
                        source_type,
                        target_type,
                        cardinality: None,
                        required_properties: Vec::new(),
                        property_types: HashMap::new(),
                    };

                    // Parse optional cardinality
                    if let Some(cardinality) = conn_schema_dict.get_item("cardinality")? {
                        conn_schema.cardinality = Some(cardinality.extract::<String>()?);
                    }

                    // Parse required_properties
                    if let Some(required_props) =
                        conn_schema_dict.get_item("required_properties")?
                    {
                        conn_schema.required_properties =
                            required_props.extract::<Vec<String>>()?;
                    }

                    // Parse property_types
                    if let Some(prop_types) = conn_schema_dict.get_item("property_types")? {
                        let types_dict = prop_types.cast::<PyDict>().map_err(|_| {
                            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "property_types must be a dictionary",
                            )
                        })?;
                        for (field, type_val) in types_dict.iter() {
                            conn_schema
                                .property_types
                                .insert(field.extract::<String>()?, type_val.extract::<String>()?);
                        }
                    }

                    schema.add_connection_schema(conn_type, conn_schema);
                }
            }
        }

        get_graph_mut(&mut self.inner).set_schema(schema);

        Ok(self.clone())
    }

    /// Validate the graph against the defined schema
    ///
    /// Args:
    ///     strict: If True, reports node/connection types that exist in the graph
    ///             but are not defined in the schema. Default is False.
    ///
    /// Returns:
    ///     A list of validation error dictionaries. Empty list means validation passed.
    ///     Each error dict contains:
    ///         - 'error_type': Type of error (e.g., 'missing_required_field', 'type_mismatch')
    ///         - 'message': Human-readable error message
    ///         - Additional fields depending on error type
    #[pyo3(signature = (strict=None))]
    fn validate_schema(&self, py: Python<'_>, strict: Option<bool>) -> PyResult<Py<PyAny>> {
        let schema = self.inner.get_schema().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No schema defined. Call define_schema() first.",
            )
        })?;

        let errors =
            schema_validation::validate_graph(&self.inner, schema, strict.unwrap_or(false));

        // Convert errors to Python list of dicts
        let result = PyList::empty(py);
        for error in errors {
            let error_dict = PyDict::new(py);

            match &error {
                schema::ValidationError::MissingRequiredField {
                    node_type,
                    node_title,
                    field,
                } => {
                    error_dict.set_item("error_type", "missing_required_field")?;
                    error_dict.set_item("node_type", node_type)?;
                    error_dict.set_item("node_title", node_title)?;
                    error_dict.set_item("field", field)?;
                }
                schema::ValidationError::TypeMismatch {
                    node_type,
                    node_title,
                    field,
                    expected_type,
                    actual_type,
                } => {
                    error_dict.set_item("error_type", "type_mismatch")?;
                    error_dict.set_item("node_type", node_type)?;
                    error_dict.set_item("node_title", node_title)?;
                    error_dict.set_item("field", field)?;
                    error_dict.set_item("expected_type", expected_type)?;
                    error_dict.set_item("actual_type", actual_type)?;
                }
                schema::ValidationError::InvalidConnectionEndpoint {
                    connection_type,
                    expected_source,
                    expected_target,
                    actual_source,
                    actual_target,
                } => {
                    error_dict.set_item("error_type", "invalid_connection_endpoint")?;
                    error_dict.set_item("connection_type", connection_type)?;
                    error_dict.set_item("expected_source", expected_source)?;
                    error_dict.set_item("expected_target", expected_target)?;
                    error_dict.set_item("actual_source", actual_source)?;
                    error_dict.set_item("actual_target", actual_target)?;
                }
                schema::ValidationError::MissingConnectionProperty {
                    connection_type,
                    source_title,
                    target_title,
                    property,
                } => {
                    error_dict.set_item("error_type", "missing_connection_property")?;
                    error_dict.set_item("connection_type", connection_type)?;
                    error_dict.set_item("source_title", source_title)?;
                    error_dict.set_item("target_title", target_title)?;
                    error_dict.set_item("property", property)?;
                }
                schema::ValidationError::UndefinedNodeType { node_type, count } => {
                    error_dict.set_item("error_type", "undefined_node_type")?;
                    error_dict.set_item("node_type", node_type)?;
                    error_dict.set_item("count", count)?;
                }
                schema::ValidationError::UndefinedConnectionType {
                    connection_type,
                    count,
                } => {
                    error_dict.set_item("error_type", "undefined_connection_type")?;
                    error_dict.set_item("connection_type", connection_type)?;
                    error_dict.set_item("count", count)?;
                }
            }

            error_dict.set_item("message", error.to_string())?;
            result.append(error_dict)?;
        }

        Ok(result.into())
    }

    /// Check if a schema has been defined for this graph
    fn has_schema(&self) -> bool {
        self.inner.get_schema().is_some()
    }

    /// Clear the schema definition from the graph
    fn clear_schema(&mut self) -> PyResult<Self> {
        get_graph_mut(&mut self.inner).clear_schema();
        Ok(self.clone())
    }

    /// Get the current schema definition as a dictionary
    fn get_schema_definition(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let schema = match self.inner.get_schema() {
            Some(s) => s,
            None => return Ok(py.None()),
        };

        let result = PyDict::new(py);

        // Convert node schemas
        let nodes_dict = PyDict::new(py);
        for (node_type, node_schema) in &schema.node_schemas {
            let schema_dict = PyDict::new(py);
            schema_dict.set_item("required", &node_schema.required_fields)?;
            schema_dict.set_item("optional", &node_schema.optional_fields)?;

            let types_dict = PyDict::new(py);
            for (field, field_type) in &node_schema.field_types {
                types_dict.set_item(field, field_type)?;
            }
            schema_dict.set_item("types", types_dict)?;

            nodes_dict.set_item(node_type, schema_dict)?;
        }
        result.set_item("nodes", nodes_dict)?;

        // Convert connection schemas
        let connections_dict = PyDict::new(py);
        for (conn_type, conn_schema) in &schema.connection_schemas {
            let schema_dict = PyDict::new(py);
            schema_dict.set_item("source", &conn_schema.source_type)?;
            schema_dict.set_item("target", &conn_schema.target_type)?;

            if let Some(cardinality) = &conn_schema.cardinality {
                schema_dict.set_item("cardinality", cardinality)?;
            }

            if !conn_schema.required_properties.is_empty() {
                schema_dict.set_item("required_properties", &conn_schema.required_properties)?;
            }

            if !conn_schema.property_types.is_empty() {
                let types_dict = PyDict::new(py);
                for (prop, prop_type) in &conn_schema.property_types {
                    types_dict.set_item(prop, prop_type)?;
                }
                schema_dict.set_item("property_types", types_dict)?;
            }

            connections_dict.set_item(conn_type, schema_dict)?;
        }
        result.set_item("connections", connections_dict)?;

        Ok(result.into())
    }

    // ========================================================================
    // Pattern Matching Methods
    // ========================================================================

    /// Match a Cypher-like pattern against the graph.
    ///
    /// Supports patterns like:
    /// - Simple node: `(p:Person)`
    /// - Single hop: `(p:Person)-[:KNOWS]->(f:Person)`
    /// - Multi-hop: `(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)`
    /// - Property filters: `(p:Person {name: "Alice"})`
    /// - Edge filters: `(a)-[:KNOWS {since: 2020}]->(b)`
    /// - Bidirectional: `(a)-[:KNOWS]-(b)` (matches both directions)
    /// - Incoming: `(a)<-[:KNOWS]-(b)` (matches edges from b to a)
    ///
    /// Syntax:
    /// - Node: `(variable:Type {property: value})`
    /// - Edge: `-[:TYPE {property: value}]->` or `<-[:TYPE]-` or `-[:TYPE]-`
    /// - Variable and type are optional: `()`, `(:Type)`, `(var)`
    ///
    /// Args:
    ///     pattern: The Cypher-like pattern string
    ///     max_matches: Maximum number of matches to return (default: unlimited)
    ///
    /// Returns:
    ///     A list of match dictionaries. Each match contains bindings for
    ///     named variables in the pattern. Node bindings have 'type', 'title',
    ///     'id', and 'properties'. Edge bindings have 'source', 'target',
    ///     'connection_type', and 'properties'.
    ///
    /// Example:
    ///     ```python
    ///     # Find all plays with their prospects
    ///     matches = graph.match_pattern('(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)')
    ///     for m in matches:
    ///         print(f"Play: {m['p']['title']}, Prospect: {m['pr']['title']}")
    ///
    ///     # Find discoveries from specific prospects
    ///     matches = graph.match_pattern(
    ///         '(pr:Prospect {status: "Active"})-[:BECAME_DISCOVERY]->(d:Discovery)'
    ///     )
    ///
    ///     # Limit results
    ///     top_10 = graph.match_pattern('(p:Person)-[:KNOWS]->(f:Person)', max_matches=10)
    ///     ```
    #[pyo3(signature = (pattern, max_matches=None))]
    fn match_pattern(
        &self,
        py: Python<'_>,
        pattern: &str,
        max_matches: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        // Parse the pattern
        let parsed = pattern_matching::parse_pattern(pattern).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Pattern syntax error: {}", e))
        })?;

        // Execute the pattern
        let executor = pattern_matching::PatternExecutor::new(&self.inner, max_matches);
        let matches = executor.execute(&parsed).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Pattern execution error: {}",
                e
            ))
        })?;

        // Convert matches to Python
        py_out::pattern_matches_to_pylist(py, &matches)
    }

    /// Execute a Cypher query against the graph.
    ///
    /// Supports MATCH, WHERE, RETURN, ORDER BY, LIMIT, SKIP, WITH,
    /// OPTIONAL MATCH, UNWIND, UNION, and aggregation functions
    /// (count, sum, avg, min, max, collect, std).
    ///
    /// The MATCH clause uses the same pattern syntax as match_pattern().
    /// WHERE supports AND/OR/NOT, comparisons (=, <>, <, <=, >, >=),
    /// IS NULL, IS NOT NULL, IN, STARTS WITH, ENDS WITH, CONTAINS.
    /// RETURN supports property access (n.prop), aliases (AS), aggregation,
    /// and DISTINCT.
    ///
    /// Args:
    ///     query: The Cypher query string
    ///
    /// Returns:
    ///     A dict with 'columns' (list of column names) and 'rows'
    ///     (list of row dicts mapping column name to value).
    ///
    /// Example:
    ///     ```python
    ///     result = graph.cypher('''
    ///         MATCH (p:Person)-[:KNOWS]->(f:Person)
    ///         WHERE p.age > 25
    ///         RETURN p.name AS person, count(f) AS friends
    ///         ORDER BY friends DESC
    ///         LIMIT 10
    ///     ''')
    ///     for row in result:
    ///         print(f"{row['person']}: {row['friends']} friends")
    ///     ```
    #[pyo3(signature = (query, *, to_df=false, params=None, timeout_ms=None))]
    fn cypher(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        query: &str,
        to_df: bool,
        params: Option<&Bound<'_, PyDict>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        // Parse the Cypher query (no borrow needed)
        let mut parsed = cypher::parse_cypher(query).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Cypher syntax error: {}", e))
        })?;

        // Convert params dict to HashMap<String, Value> (before optimize so pushdown can resolve params)
        let mut param_map = if let Some(params_dict) = params {
            let mut map = std::collections::HashMap::new();
            for (key, val) in params_dict.iter() {
                let key_str: String = key.extract()?;
                let value = py_in::py_value_to_value(&val)?;
                map.insert(key_str, value);
            }
            map
        } else {
            std::collections::HashMap::new()
        };

        // Rewrite text_score() → vector_score() and collect texts to embed
        let rewrite = cypher::rewrite_text_score(&mut parsed, &param_map)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Embed collected query texts if any (skip for EXPLAIN)
        if !rewrite.texts_to_embed.is_empty() && !parsed.explain {
            let this = slf.borrow();
            let model = match &this.embedder {
                Some(m) => m.bind(py).clone(),
                None => {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "text_score() requires a registered embedding model. \
                         Call g.set_embedder(model) first.",
                    ))
                }
            };
            Self::try_load_embedder(&model)?;

            let texts: Vec<&str> = rewrite
                .texts_to_embed
                .iter()
                .map(|(_, t)| t.as_str())
                .collect();
            let py_texts = PyList::new(py, &texts)?;
            let embed_result = model.call_method1("embed", (py_texts,));
            Self::try_unload_embedder(&model);
            let embeddings_result = embed_result?;
            let embeddings: Vec<Vec<f32>> = embeddings_result.extract().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "model.embed() must return list[list[float]]",
                )
            })?;

            if embeddings.len() != texts.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "text_score: model.embed() returned {} vectors for {} texts",
                    embeddings.len(),
                    texts.len()
                )));
            }

            for (i, (param_name, _)) in rewrite.texts_to_embed.iter().enumerate() {
                let json = format!(
                    "[{}]",
                    embeddings[i]
                        .iter()
                        .map(|f| f.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                param_map.insert(param_name.clone(), Value::String(json));
            }
        }

        // Optimize (predicate pushdown, etc.) — needs shared borrow of graph
        {
            let this = slf.borrow();
            cypher::optimize(&mut parsed, &this.inner, &param_map);
        }

        // EXPLAIN: return query plan as string without executing
        if parsed.explain {
            let plan = cypher::generate_explain_plan(&parsed);
            return Ok(plan.into_pyobject(py)?.into_any().unbind());
        }

        if cypher::is_mutation_query(&parsed) {
            // Mutation path: needs exclusive borrow
            let mut this = slf.borrow_mut();
            let graph = get_graph_mut(&mut this.inner);
            let result =
                cypher::execute_mutable(graph, &parsed, param_map, deadline).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Cypher execution error: {}",
                        e
                    ))
                })?;
            // Auto-vacuum after deletions
            if let Some(ref stats) = result.stats {
                if (stats.nodes_deleted > 0 || stats.relationships_deleted > 0)
                    && graph.check_auto_vacuum()
                {
                    this.selection = schema::CowSelection::new();
                }
            }
            // Store mutation stats
            if let Some(ref stats) = result.stats {
                this.last_mutation_stats = Some(stats.clone());
            }
            // Convert to Python
            if to_df {
                let preprocessed = cypher::py_convert::preprocess_values_owned(result.rows);
                cypher::py_convert::preprocessed_result_to_dataframe(
                    py,
                    &result.columns,
                    &preprocessed,
                )
            } else {
                let view = cypher::ResultView::from_cypher_result(result);
                Py::new(py, view).map(|v| v.into_any())
            }
        } else {
            // Read-only path: clone Arc, release borrow, then execute without GIL
            let inner = {
                let this = slf.borrow();
                this.inner.clone()
            };
            let result = {
                let executor = cypher::CypherExecutor::with_params(&inner, &param_map, deadline);
                py.detach(|| executor.execute(&parsed))
            }
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Cypher execution error: {}",
                    e
                ))
            })?;
            let columns = result.columns;
            let stats = result.stats;
            let preprocessed = cypher::py_convert::preprocess_values_owned(result.rows);
            if to_df {
                cypher::py_convert::preprocessed_result_to_dataframe(py, &columns, &preprocessed)
            } else {
                let view = cypher::ResultView::from_preprocessed(columns, preprocessed, stats);
                Py::new(py, view).map(|v| v.into_any())
            }
        }
    }

    /// Mutation statistics from the last Cypher mutation query (CREATE/SET/DELETE/REMOVE/MERGE).
    ///
    /// Returns None if no mutation has been executed yet.
    #[getter]
    fn last_mutation_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.last_mutation_stats {
            Some(stats) => {
                let dict = PyDict::new(py);
                dict.set_item("nodes_created", stats.nodes_created)?;
                dict.set_item("relationships_created", stats.relationships_created)?;
                dict.set_item("properties_set", stats.properties_set)?;
                dict.set_item("nodes_deleted", stats.nodes_deleted)?;
                dict.set_item("relationships_deleted", stats.relationships_deleted)?;
                dict.set_item("properties_removed", stats.properties_removed)?;
                Ok(dict.into())
            }
            None => Ok(py.None()),
        }
    }

    // ========================================================================
    // Transaction Support
    // ========================================================================

    /// Begin a transaction — returns a Transaction object with a working copy of the graph.
    ///
    /// Creates a snapshot of the current graph state. All mutations within the
    /// transaction are isolated until ``commit()`` is called. If the transaction
    /// is rolled back (or dropped without committing), no changes are applied.
    ///
    /// **Note:** the snapshot is a full deep-clone of the graph, so creating a
    /// transaction on a very large graph has a one-time memory cost proportional
    /// to graph size. Embeddings are *not* cloned (they live outside `DirGraph`).
    ///
    /// Can also be used as a context manager:
    ///
    /// Example:
    ///     ```python
    ///     with graph.begin() as tx:
    ///         tx.cypher("CREATE (n:Person {name: 'Alice', age: 30})")
    ///         tx.cypher("CREATE (n:Person {name: 'Bob', age: 25})")
    ///         # auto-commits on success, auto-rollbacks on exception
    ///     ```
    fn begin(slf: Py<Self>) -> PyResult<Transaction> {
        let working = Python::attach(|py| {
            let kg = slf.borrow(py);
            (*kg.inner).clone()
        });
        Ok(Transaction {
            owner: slf,
            working: Some(working),
            committed: false,
        })
    }

}

// ============================================================================
// Transaction Implementation
// ============================================================================

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
    #[pyo3(signature = (query, params=None, to_df=false, timeout_ms=None))]
    fn cypher(
        &mut self,
        py: Python<'_>,
        query: &str,
        params: Option<&Bound<'_, PyDict>>,
        to_df: bool,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        let working = self.working.as_mut().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Transaction already committed or rolled back",
            )
        })?;

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

        // Parse and optimize
        let mut parsed = cypher::parse_cypher(query).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Cypher parse error: {}", e))
        })?;
        cypher::optimize(&mut parsed, working, &param_map);

        // Execute
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

        if to_df {
            let preprocessed = cypher::py_convert::preprocess_values_owned(result.rows);
            cypher::py_convert::preprocessed_result_to_dataframe(py, &result.columns, &preprocessed)
        } else {
            let view = cypher::ResultView::from_cypher_result(result);
            Py::new(py, view).map(|v| v.into_any())
        }
    }

    /// Commit the transaction — apply all changes to the original graph.
    ///
    /// After commit, the transaction cannot be used again.
    fn commit(&mut self) -> PyResult<()> {
        let working = self.working.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Transaction already committed or rolled back",
            )
        })?;

        Python::attach(|py| {
            let mut kg = self.owner.borrow_mut(py);
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
        if self.working.is_none() {
            // Already committed or rolled back
            return Ok(false);
        }

        if exc_type.is_some() {
            // Exception occurred — rollback
            self.working = None;
        } else {
            // No exception — commit
            self.commit()?;
        }

        // Return false = don't suppress exception
        Ok(false)
    }
}
