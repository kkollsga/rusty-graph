// Embedding / Vector Search #[pymethods] — extracted from mod.rs

use crate::datatypes::{py_in, py_out};
use petgraph::graph::NodeIndex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;
use std::sync::Arc;

use super::schema;
use super::{io_operations, vector_search, KnowledgeGraph};

#[pymethods]
impl KnowledgeGraph {
    // ========================================================================
    // Embedding / Vector Search Methods
    // ========================================================================

    /// Store embeddings for nodes of the given type.
    ///
    /// Args:
    ///     node_type: The node type (e.g. 'Article')
    ///     text_column: Source column name (e.g. 'summary'). Stored as '{text_column}_emb'.
    ///     embeddings: Dict mapping node IDs to embedding vectors (list of floats)
    ///
    /// Returns:
    ///     dict: {'embeddings_stored': int, 'dimension': int, 'skipped': int}
    fn set_embeddings(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        embeddings: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        let g = Arc::make_mut(&mut self.inner);
        let embedding_property = format!("{}_emb", text_column);

        // Validate node type exists
        if !g.type_indices.contains_key(node_type) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Node type '{}' does not exist in the graph",
                node_type
            )));
        }

        // Validate source column exists (skip for empty dicts)
        if !embeddings.is_empty() {
            let is_builtin = matches!(text_column, "id" | "title" | "type");
            if !is_builtin {
                let has_property = g
                    .type_indices
                    .get(node_type)
                    .map(|indices| {
                        indices.iter().any(|&idx| {
                            g.graph
                                .node_weight(idx)
                                .map(|n| n.properties.contains_key(text_column))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);
                if !has_property {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Source column '{}' not found on any '{}' node. \
                         set_embeddings() expects the text column name \
                         (e.g. 'summary'), not the embedding store name.",
                        text_column, node_type
                    )));
                }
            }
        }

        // Build ID index for this node type if not already built
        g.build_id_index(node_type);

        let mut dimension: Option<usize> = None;
        let mut entries: Vec<(NodeIndex, Vec<f32>)> = Vec::new();
        let mut skipped = 0usize;

        for (key, value) in embeddings.iter() {
            // Convert key to Value for ID lookup
            let id = py_in::py_value_to_value(&key)?;

            // Look up node by ID
            let node_idx = match g.lookup_by_id(node_type, &id) {
                Some(idx) => idx,
                None => {
                    skipped += 1;
                    continue;
                }
            };

            // Convert embedding to Vec<f32>
            let vec: Vec<f32> = value.extract()?;

            // Validate/set dimension
            match dimension {
                None => dimension = Some(vec.len()),
                Some(d) => {
                    if vec.len() != d {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Inconsistent embedding dimensions: expected {} but got {}",
                            d,
                            vec.len()
                        )));
                    }
                }
            }

            entries.push((node_idx, vec));
        }

        let dim = match dimension {
            Some(d) => d,
            None => {
                let result = PyDict::new(py);
                result.set_item("embeddings_stored", 0)?;
                result.set_item("dimension", 0)?;
                result.set_item("skipped", skipped)?;
                return Ok(result.into());
            }
        };

        // Create or replace the EmbeddingStore
        let mut store = schema::EmbeddingStore::new(dim);
        store.data.reserve(entries.len() * dim);
        for (node_idx, vec) in &entries {
            store.set_embedding(node_idx.index(), vec);
        }

        let stored = store.len();
        g.embeddings
            .insert((node_type.to_string(), embedding_property), store);

        let result = PyDict::new(py);
        result.set_item("embeddings_stored", stored)?;
        result.set_item("dimension", dim)?;
        result.set_item("skipped", skipped)?;
        Ok(result.into())
    }

    /// Vector similarity search within the current selection.
    ///
    /// Args:
    ///     text_column: Source column name (e.g. 'summary'). Resolves to '{text_column}_emb'.
    ///     query_vector: The query embedding vector (list of floats)
    ///     top_k: Number of results to return (default 10)
    ///     metric: Distance metric - 'cosine' (default), 'dot_product', or 'euclidean'
    ///     to_df: If True, return a pandas DataFrame instead of list of dicts
    #[pyo3(signature = (text_column, query_vector, top_k=None, metric=None, to_df=None))]
    fn vector_search(
        &self,
        py: Python<'_>,
        text_column: &str,
        query_vector: Vec<f32>,
        top_k: Option<usize>,
        metric: Option<&str>,
        to_df: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let top_k = top_k.unwrap_or(10);
        let metric = match metric.unwrap_or("cosine") {
            "cosine" => vector_search::DistanceMetric::Cosine,
            "dot_product" => vector_search::DistanceMetric::DotProduct,
            "euclidean" => vector_search::DistanceMetric::Euclidean,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown metric '{}'. Use 'cosine', 'dot_product', or 'euclidean'.",
                    other
                )));
            }
        };

        let embedding_property = format!("{}_emb", text_column);
        // Release GIL during heavy vector similarity computation
        let inner = self.inner.clone();
        let selection = self.selection.clone();
        let results = py
            .detach(|| {
                vector_search::vector_search(
                    &inner,
                    &selection,
                    &embedding_property,
                    &query_vector,
                    top_k,
                    metric,
                )
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        if to_df.unwrap_or(false) {
            // Build DataFrame via pandas
            let pandas = py.import("pandas")?;
            let records: Vec<Py<PyAny>> = results
                .iter()
                .filter_map(|r| {
                    self.inner.graph.node_weight(r.node_idx).map(|node| {
                        let dict = PyDict::new(py);
                        let _ = dict.set_item("id", py_out::value_to_py(py, &node.id).ok());
                        let _ = dict.set_item("title", py_out::value_to_py(py, &node.title).ok());
                        let _ = dict.set_item("type", &node.node_type);
                        let _ = dict.set_item("score", r.score);
                        for (k, v) in &node.properties {
                            let _ = dict.set_item(k, py_out::value_to_py(py, v).ok());
                        }
                        dict.into()
                    })
                })
                .collect();
            let py_list = PyList::new(py, &records)?;
            let df = pandas.call_method1("DataFrame", (py_list,))?;
            return df.into_py_any(py);
        }

        // Return as list of dicts
        let py_list = PyList::empty(py);
        for r in &results {
            if let Some(node) = self.inner.graph.node_weight(r.node_idx) {
                let dict = PyDict::new(py);
                dict.set_item("id", py_out::value_to_py(py, &node.id)?)?;
                dict.set_item("title", py_out::value_to_py(py, &node.title)?)?;
                dict.set_item("type", &node.node_type)?;
                dict.set_item("score", r.score)?;
                for (k, v) in &node.properties {
                    dict.set_item(k, py_out::value_to_py(py, v)?)?;
                }
                py_list.append(dict)?;
            }
        }

        py_list.into_py_any(py)
    }

    /// List all embedding stores in the graph.
    ///
    /// Returns:
    ///     List of dicts with 'node_type', 'text_column', 'dimension', 'count'.
    fn list_embeddings(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let py_list = PyList::empty(py);
        for ((node_type, store_name), store) in &self.inner.embeddings {
            let text_column = store_name
                .strip_suffix("_emb")
                .unwrap_or(store_name.as_str());
            let dict = PyDict::new(py);
            dict.set_item("node_type", node_type)?;
            dict.set_item("text_column", text_column)?;
            dict.set_item("dimension", store.dimension)?;
            dict.set_item("count", store.len())?;
            py_list.append(dict)?;
        }
        py_list.into_py_any(py)
    }

    /// Remove an embedding store.
    ///
    /// Args:
    ///     node_type: The node type
    ///     text_column: Source column name (e.g. 'summary')
    fn remove_embeddings(&mut self, node_type: &str, text_column: &str) -> PyResult<()> {
        let g = Arc::make_mut(&mut self.inner);
        let key = (node_type.to_string(), format!("{}_emb", text_column));
        g.embeddings.remove(&key);
        Ok(())
    }

    /// Export embeddings to a standalone .kgle file.
    ///
    /// Exported embeddings are keyed by node ID, so they survive graph rebuilds.
    ///
    /// Args:
    ///     path: File path to write (typically ending in .kgle)
    ///     node_types: Optional filter. Can be:
    ///         - None: export all embeddings
    ///         - list[str]: export all embedding stores for these node types
    ///         - dict[str, list[str]]: export specific (node_type -> [text_columns]) pairs.
    ///           An empty list means all properties for that type.
    ///
    /// Returns:
    ///     Dict with 'stores' (int) and 'embeddings' (int) counts.
    #[pyo3(signature = (path, node_types=None))]
    fn export_embeddings(
        &self,
        py: Python<'_>,
        path: &str,
        node_types: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let filter = match &node_types {
            None => None,
            Some(obj) => {
                if let Ok(list) = obj.cast::<PyList>() {
                    let types: Vec<String> = list.extract()?;
                    Some(io_operations::EmbeddingExportFilter::Types(types))
                } else if let Ok(dict) = obj.cast::<PyDict>() {
                    let mut map: HashMap<String, Vec<String>> = HashMap::new();
                    for (k, v) in dict.iter() {
                        let key: String = k.extract()?;
                        let vals: Vec<String> = v.extract()?;
                        map.insert(key, vals);
                    }
                    Some(io_operations::EmbeddingExportFilter::TypeProperties(map))
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "node_types must be a list of strings or a dict of {str: list[str]}",
                    ));
                }
            }
        };

        let inner = self.inner.clone();
        let path_owned = path.to_string();
        let stats = py
            .detach(move || {
                io_operations::export_embeddings_to_file(&inner, &path_owned, filter.as_ref())
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        let result = PyDict::new(py);
        result.set_item("stores", stats.stores)?;
        result.set_item("embeddings", stats.embeddings)?;
        result.into_py_any(py)
    }

    /// Import embeddings from a .kgle file.
    ///
    /// Matches embeddings to nodes by (node_type, node_id). Embeddings whose
    /// node ID doesn't exist in the current graph are silently skipped.
    ///
    /// Args:
    ///     path: Path to a .kgle file previously created by export_embeddings.
    ///
    /// Returns:
    ///     Dict with 'stores' (int), 'imported' (int), and 'skipped' (int) counts.
    fn import_embeddings(&mut self, py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
        let g = Arc::make_mut(&mut self.inner);
        let stats = io_operations::import_embeddings_from_file(g, path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        let result = PyDict::new(py);
        result.set_item("stores", stats.stores)?;
        result.set_item("imported", stats.imported)?;
        result.set_item("skipped", stats.skipped)?;
        result.into_py_any(py)
    }

    /// Retrieve embeddings for nodes.
    ///
    /// Can be called in two ways:
    ///   - ``embeddings(node_type, text_column)`` — returns all embeddings of that type
    ///   - ``embeddings(text_column)`` — returns embeddings for the current selection
    ///
    /// Args:
    ///     text_column: Source column name (e.g. 'summary'). Resolves to '{text_column}_emb'.
    ///
    /// Returns:
    ///     Dict mapping node IDs to embedding vectors (list of floats).
    #[pyo3(signature = (node_type_or_text_column, text_column=None))]
    fn embeddings(
        &self,
        py: Python<'_>,
        node_type_or_text_column: &str,
        text_column: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let result = PyDict::new(py);

        // Two-arg form: embeddings(node_type, text_column)
        if let Some(col) = text_column {
            let key = (node_type_or_text_column.to_string(), format!("{}_emb", col));
            let store = match self.inner.embeddings.get(&key) {
                Some(s) => s,
                None => return result.into_py_any(py),
            };

            for (&node_index, &_slot) in &store.node_to_slot {
                if let Some(embedding) = store.get_embedding(node_index) {
                    if let Some(node) = self.inner.graph.node_weight(NodeIndex::new(node_index)) {
                        let py_id = py_out::value_to_py(py, &node.id)?;
                        let py_vec = PyList::new(py, embedding)?;
                        result.set_item(py_id, py_vec)?;
                    }
                }
            }

            return result.into_py_any(py);
        }

        // One-arg form: embeddings(text_column) — selection-based
        let col = node_type_or_text_column;

        let level_count = self.selection.get_level_count();
        if level_count == 0 {
            return result.into_py_any(py);
        }

        let nodes: Vec<NodeIndex> = self
            .selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default();

        for node_idx in &nodes {
            let node = match self.inner.graph.node_weight(*node_idx) {
                Some(n) => n,
                None => continue,
            };

            let key = (node.node_type.clone(), format!("{}_emb", col));
            let store = match self.inner.embeddings.get(&key) {
                Some(s) => s,
                None => continue,
            };

            if let Some(embedding) = store.get_embedding(node_idx.index()) {
                let py_id = py_out::value_to_py(py, &node.id)?;
                let py_vec = PyList::new(py, embedding)?;
                result.set_item(py_id, py_vec)?;
            }
        }

        result.into_py_any(py)
    }

    /// Retrieve a single node's embedding vector.
    ///
    /// Args:
    ///     node_type: The node type (e.g. 'Article').
    ///     text_column: Source column name (e.g. 'summary').
    ///     node_id: The node ID to look up.
    ///
    /// Returns:
    ///     The embedding vector as a list of floats, or None if not found.
    fn embedding(
        &self,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        node_id: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let id = py_in::py_value_to_value(node_id)?;

        let node_idx = match self.inner.lookup_by_id_readonly(node_type, &id) {
            Some(idx) => idx,
            None => return Ok(py.None()),
        };

        let key = (node_type.to_string(), format!("{}_emb", text_column));
        let store = match self.inner.embeddings.get(&key) {
            Some(s) => s,
            None => return Ok(py.None()),
        };

        match store.get_embedding(node_idx.index()) {
            Some(embedding) => {
                let py_vec = PyList::new(py, embedding)?;
                py_vec.into_py_any(py)
            }
            None => Ok(py.None()),
        }
    }

    // ========================================================================
    // Text-Level Embedding API
    // ========================================================================

    /// Register an embedding model on the graph.
    ///
    /// The model must have:
    /// - ``dimension: int`` — the embedding vector size
    /// - ``embed(texts: list[str]) -> list[list[float]]`` — batch embedding method
    ///
    /// After calling this, ``embed_texts()`` and ``search_text()`` use the
    /// registered model automatically.  The model is **not** serialized —
    /// call ``set_embedder()`` again after ``load()``.
    #[pyo3(signature = (model,))]
    fn set_embedder(&mut self, py: Python<'_>, model: Py<PyAny>) -> PyResult<()> {
        let bound = model.bind(py);
        bound.getattr("dimension").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
                "model must have a 'dimension' attribute (int)",
            )
        })?;
        bound.getattr("embed").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyAttributeError, _>("model must have an 'embed' method")
        })?;
        self.embedder = Some(model);
        Ok(())
    }

    /// Embed a text column for all nodes of a given type.
    ///
    /// Uses the model registered via ``set_embedder()``.  Reads each node's
    /// ``text_column`` property, calls ``model.embed()`` in batches, and stores
    /// the resulting vectors as ``{text_column}_emb``.  Nodes with missing or
    /// non-string text values are skipped.
    ///
    /// Args:
    ///     node_type: The node type to embed (e.g. ``'Article'``).
    ///     text_column: The node property containing text to embed.
    ///     batch_size: Number of texts per ``model.embed()`` call (default 256).
    ///     show_progress: Show a tqdm progress bar (default ``True``).
    ///         Requires ``tqdm`` to be installed; silently falls back to no
    ///         progress bar if it is not available.
    ///     replace: If ``True``, re-embed all nodes even if they already have
    ///         embeddings.  Default ``False`` (skip nodes with existing embeddings).
    ///
    /// Returns:
    ///     Dict with ``embedded``, ``skipped``, ``skipped_existing``, and ``dimension``.
    #[pyo3(signature = (node_type, text_column, batch_size=None, show_progress=None, replace=None))]
    fn embed_texts(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        text_column: &str,
        batch_size: Option<usize>,
        show_progress: Option<bool>,
        replace: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let model = self.get_embedder_or_error(py)?;
        let embedding_property = format!("{}_emb", text_column);
        let batch_size = batch_size.unwrap_or(256);
        let replace = replace.unwrap_or(false);

        // Load model if it has a load() lifecycle method
        Self::try_load_embedder(&model)?;

        // Get model dimension
        let dimension: usize = match model.getattr("dimension").and_then(|d| d.extract()) {
            Ok(dim) => dim,
            Err(_) => {
                Self::try_unload_embedder(&model);
                return Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
                    "model must have an int 'dimension' attribute",
                ));
            }
        };

        // Collect (node_index, text) for nodes that need embedding
        let mut node_texts: Vec<(NodeIndex, String)> = Vec::new();
        let mut skipped = 0usize;
        let mut skipped_existing = 0usize;

        let emb_key = (node_type.to_string(), embedding_property.clone());
        let existing_store = if replace {
            None
        } else {
            self.inner.embeddings.get(&emb_key)
        };

        let node_indices: Vec<NodeIndex> = self
            .inner
            .type_indices
            .get(node_type)
            .cloned()
            .unwrap_or_default();

        for &node_idx in &node_indices {
            if let Some(node) = self.inner.graph.node_weight(node_idx) {
                match node.properties.get(text_column) {
                    Some(crate::datatypes::values::Value::String(s)) if !s.is_empty() => {
                        // Skip nodes that already have an embedding
                        if existing_store
                            .map(|s| s.get_embedding(node_idx.index()).is_some())
                            .unwrap_or(false)
                        {
                            skipped_existing += 1;
                        } else {
                            node_texts.push((node_idx, s.clone()));
                        }
                    }
                    _ => {
                        skipped += 1;
                    }
                }
            }
        }

        if node_texts.is_empty() {
            Self::try_unload_embedder(&model);
            let result = PyDict::new(py);
            result.set_item("embedded", 0)?;
            result.set_item("skipped", skipped)?;
            result.set_item("skipped_existing", skipped_existing)?;
            result.set_item("dimension", dimension)?;
            return Ok(result.into());
        }

        // Clone existing store or create new — we'll merge new embeddings into it
        let mut store = match existing_store {
            Some(s) => s.clone(),
            None => schema::EmbeddingStore::new(dimension),
        };
        store.data.reserve(node_texts.len() * dimension);

        // Try to create a tqdm progress bar (if tqdm is installed and show_progress != false)
        let progress_bar = if show_progress.unwrap_or(true) {
            py.import("tqdm.auto")
                .or_else(|_| py.import("tqdm"))
                .ok()
                .and_then(|tqdm_mod| {
                    let kwargs = PyDict::new(py);
                    let _ = kwargs.set_item("total", node_texts.len());
                    let _ =
                        kwargs.set_item("desc", format!("Embedding {}.{}", node_type, text_column));
                    let _ = kwargs.set_item("unit", "text");
                    tqdm_mod.call_method("tqdm", (), Some(&kwargs)).ok()
                })
        } else {
            None
        };

        for batch in node_texts.chunks(batch_size) {
            let texts: Vec<&str> = batch.iter().map(|(_, t)| t.as_str()).collect();
            let py_texts = PyList::new(py, &texts)?;

            let embeddings_result = match model.call_method1("embed", (py_texts,)) {
                Ok(r) => r,
                Err(e) => {
                    if let Some(ref bar) = progress_bar {
                        let _ = bar.call_method0("close");
                    }
                    Self::try_unload_embedder(&model);
                    return Err(e);
                }
            };
            let embeddings: Vec<Vec<f32>> = match embeddings_result.extract() {
                Ok(v) => v,
                Err(_) => {
                    if let Some(ref bar) = progress_bar {
                        let _ = bar.call_method0("close");
                    }
                    Self::try_unload_embedder(&model);
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "model.embed() must return list[list[float]]",
                    ));
                }
            };

            if embeddings.len() != batch.len() {
                if let Some(ref bar) = progress_bar {
                    let _ = bar.call_method0("close");
                }
                Self::try_unload_embedder(&model);
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "model.embed() returned {} vectors for {} texts",
                    embeddings.len(),
                    batch.len()
                )));
            }

            for (i, vec) in embeddings.iter().enumerate() {
                if vec.len() != dimension {
                    if let Some(ref bar) = progress_bar {
                        let _ = bar.call_method0("close");
                    }
                    Self::try_unload_embedder(&model);
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "model.embed() returned vector of dimension {} (expected {})",
                        vec.len(),
                        dimension
                    )));
                }
                store.set_embedding(batch[i].0.index(), vec);
            }

            // Update progress bar
            if let Some(ref bar) = progress_bar {
                let _ = bar.call_method1("update", (batch.len(),));
            }
        }

        // Close progress bar
        if let Some(ref bar) = progress_bar {
            let _ = bar.call_method0("close");
        }

        // Unload model after embedding is complete
        Self::try_unload_embedder(&model);

        let embedded = node_texts.len();
        let g = Arc::make_mut(&mut self.inner);
        g.embeddings.insert(emb_key, store);

        let result = PyDict::new(py);
        result.set_item("embedded", embedded)?;
        result.set_item("skipped", skipped)?;
        result.set_item("skipped_existing", skipped_existing)?;
        result.set_item("dimension", dimension)?;
        Ok(result.into())
    }

    /// Search embeddings using a text query.
    ///
    /// Uses the model registered via ``set_embedder()`` to embed the query,
    /// then performs vector search within the current selection.  The user
    /// refers to the text column name (e.g. ``"summary"``); the graph
    /// resolves it to ``"summary_emb"`` internally.
    ///
    /// Args:
    ///     text_column: Text column whose embeddings to search (e.g. ``'summary'``).
    ///     query: The text query to search for.
    ///     top_k: Number of results to return (default 10).
    ///     metric: Distance metric (default ``'cosine'``).
    ///     to_df: If True, return a pandas DataFrame.
    ///
    /// Returns:
    ///     Same format as ``vector_search()`` — list of dicts or DataFrame.
    #[pyo3(signature = (text_column, query, top_k=None, metric=None, to_df=None))]
    fn search_text(
        &self,
        py: Python<'_>,
        text_column: &str,
        query: &str,
        top_k: Option<usize>,
        metric: Option<&str>,
        to_df: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let model = self.get_embedder_or_error(py)?;

        // Load model if it has a load() lifecycle method
        Self::try_load_embedder(&model)?;

        // Embed the query text, then unload regardless of success/failure
        let py_texts = PyList::new(py, [query])?;
        let embed_result = model.call_method1("embed", (py_texts,));
        Self::try_unload_embedder(&model);
        let embeddings_result = embed_result?;

        let embeddings: Vec<Vec<f32>> = embeddings_result.extract().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "model.embed() must return list[list[float]]",
            )
        })?;

        if embeddings.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "model.embed() returned an empty list",
            ));
        }

        let query_vector = embeddings.into_iter().next().unwrap();

        // Delegate to existing vector_search
        self.vector_search(py, text_column, query_vector, top_k, metric, to_df)
    }
}
