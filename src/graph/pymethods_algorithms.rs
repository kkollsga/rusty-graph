// Graph Algorithms #[pymethods] — extracted from mod.rs

use crate::datatypes::values::Value;
use crate::datatypes::{py_in, py_out};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

use super::reporting::OperationReports;
use super::schema::{CowSelection, PlanStep};
use super::{
    centrality_results_to_dataframe, centrality_results_to_py_dict, community_results_to_py,
    cypher, graph_algorithms, lookups, subgraph, KnowledgeGraph, TemporalContext,
};

#[pymethods]
impl KnowledgeGraph {
    // ========================================================================
    // Graph Algorithms: Path Finding & Connectivity
    // ========================================================================

    /// Find the shortest path between two nodes.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     A dictionary with:
    ///         - 'path': List of node info dicts along the path
    ///         - 'connections': List of connection types between nodes
    ///         - 'length': Number of hops in the path
    ///     Returns None if no path exists.
    #[pyo3(signature = (source_type, source_id, target_type, target_id, connection_types=None, via_types=None, timeout_ms=None))]
    #[allow(clippy::too_many_arguments)]
    fn shortest_path(
        &self,
        py: Python<'_>,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
        connection_types: Option<Vec<String>>,
        via_types: Option<Vec<String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        // Look up source node
        let source_lookup = lookups::TypeLookup::new(&self.inner.graph, source_type.to_string())
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = source_lookup.check_uid(&source_value).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Source node with id {:?} not found in type '{}'",
                source_value, source_type
            ))
        })?;

        // Look up target node
        let target_lookup = if target_type == source_type {
            source_lookup
        } else {
            lookups::TypeLookup::new(&self.inner.graph, target_type.to_string())
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?
        };

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = target_lookup.check_uid(&target_value).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Target node with id {:?} not found in type '{}'",
                target_value, target_type
            ))
        })?;

        // Find shortest path
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
        let result = graph_algorithms::shortest_path(
            &self.inner,
            source_idx,
            target_idx,
            connection_types.as_deref(),
            via_types.as_deref(),
            deadline,
        );

        match result {
            Some(path_result) => {
                let result_dict = PyDict::new(py);

                // Build path info list
                let path_list = PyList::empty(py);
                for &node_idx in &path_result.path {
                    if let Some(info) = graph_algorithms::get_node_info(&self.inner, node_idx) {
                        let node_dict = PyDict::new(py);
                        node_dict.set_item("type", &info.node_type)?;
                        node_dict.set_item("title", &info.title)?;
                        node_dict.set_item("id", py_out::value_to_py(py, &info.id)?)?;
                        path_list.append(node_dict)?;
                    }
                }
                result_dict.set_item("path", path_list)?;

                // Build connections list
                let connections =
                    graph_algorithms::get_path_connections(&self.inner, &path_result.path);
                let conn_list = PyList::empty(py);
                for conn in connections {
                    match conn {
                        Some(c) => conn_list.append(&c)?,
                        None => conn_list.append(py.None())?,
                    }
                }
                result_dict.set_item("connections", conn_list)?;
                result_dict.set_item("length", path_result.cost)?;

                Ok(result_dict.into())
            }
            None => Ok(py.None()),
        }
    }

    /// Get just the length (hop count) of the shortest path between two nodes.
    ///
    /// This is a lightweight version of shortest_path() that avoids materializing
    /// node data, making it much faster when you only need the distance.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     The number of hops (edges) in the shortest path, or None if no path exists.
    fn shortest_path_length(
        &self,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
    ) -> PyResult<Option<usize>> {
        // Use O(1) direct lookup from id_indices (populated during add_nodes)
        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = self
            .inner
            .lookup_by_id_normalized(source_type, &source_value)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Source node with id {:?} not found in type '{}'",
                    source_value, source_type
                ))
            })?;

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = self
            .inner
            .lookup_by_id_normalized(target_type, &target_value)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Target node with id {:?} not found in type '{}'",
                    target_value, target_type
                ))
            })?;

        // Find shortest path cost only (no path reconstruction — faster)
        Ok(graph_algorithms::shortest_path_cost(
            &self.inner,
            source_idx,
            target_idx,
        ))
    }

    /// Batch shortest path lengths — computes distances for multiple pairs at once.
    ///
    /// Much faster than calling shortest_path_length in a loop because it:
    /// 1. Builds the adjacency list once (amortized across all pairs)
    /// 2. Reuses the visited-tracking allocation between queries
    ///
    /// Args:
    ///     node_type: The node type for all source/target nodes
    ///     pairs: List of (source_id, target_id) tuples
    ///
    /// Returns:
    ///     List of distances (None where no path exists), same order as input pairs.
    fn shortest_path_lengths_batch(
        &self,
        py: Python<'_>,
        node_type: &str,
        pairs: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>,
    ) -> PyResult<Py<PyAny>> {
        // Resolve all node indices up front
        let mut index_pairs = Vec::with_capacity(pairs.len());
        for (src_py, tgt_py) in &pairs {
            let src_val = py_in::py_value_to_value(src_py)?;
            let src_idx = self
                .inner
                .lookup_by_id_normalized(node_type, &src_val)
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Node with id {:?} not found in type '{}'",
                        src_val, node_type
                    ))
                })?;

            let tgt_val = py_in::py_value_to_value(tgt_py)?;
            let tgt_idx = self
                .inner
                .lookup_by_id_normalized(node_type, &tgt_val)
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Node with id {:?} not found in type '{}'",
                        tgt_val, node_type
                    ))
                })?;

            index_pairs.push((src_idx, tgt_idx));
        }

        let results = graph_algorithms::shortest_path_cost_batch(&self.inner, &index_pairs);

        let result_list = PyList::empty(py);
        for result in results {
            match result {
                Some(cost) => result_list.append(cost)?,
                None => result_list.append(py.None())?,
            }
        }

        Ok(result_list.into())
    }

    /// Get just the node IDs along the shortest path between two nodes.
    ///
    /// This is a lightweight version of shortest_path() that returns only the
    /// node IDs without full node info, making it faster when you don't need
    /// titles, types, or connection info.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     A list of node IDs along the path, or None if no path exists.
    #[pyo3(signature = (source_type, source_id, target_type, target_id, connection_types=None, via_types=None, timeout_ms=None))]
    #[allow(clippy::too_many_arguments)]
    fn shortest_path_ids(
        &self,
        py: Python<'_>,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
        connection_types: Option<Vec<String>>,
        via_types: Option<Vec<String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        // Use O(1) direct lookup from id_indices (populated during add_nodes)
        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = self
            .inner
            .lookup_by_id_normalized(source_type, &source_value)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Source node with id {:?} not found in type '{}'",
                    source_value, source_type
                ))
            })?;

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = self
            .inner
            .lookup_by_id_normalized(target_type, &target_value)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Target node with id {:?} not found in type '{}'",
                    target_value, target_type
                ))
            })?;

        // Find shortest path
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
        match graph_algorithms::shortest_path(
            &self.inner,
            source_idx,
            target_idx,
            connection_types.as_deref(),
            via_types.as_deref(),
            deadline,
        ) {
            Some(path_result) => {
                // Extract just the IDs - no PyDict creation per node
                let ids: Vec<Py<PyAny>> = path_result
                    .path
                    .iter()
                    .filter_map(|&idx| {
                        self.inner
                            .get_node(idx)
                            .and_then(|node| node.get_field_ref("id"))
                            .map(|id| py_out::value_to_py(py, id).unwrap_or_else(|_| py.None()))
                    })
                    .collect();
                Ok(PyList::new(py, ids)?.into())
            }
            None => Ok(py.None()),
        }
    }

    /// Get just the raw graph indices along the shortest path between two nodes.
    ///
    /// This is the fastest path query - returns only integer indices without
    /// any node data lookup. Use this when you need maximum performance and
    /// will look up node data separately if needed.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     A list of integer node indices along the path, or None if no path exists.
    #[pyo3(signature = (source_type, source_id, target_type, target_id, connection_types=None, via_types=None, timeout_ms=None))]
    #[allow(clippy::too_many_arguments)]
    fn shortest_path_indices(
        &self,
        py: Python<'_>,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
        connection_types: Option<Vec<String>>,
        via_types: Option<Vec<String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        // Use O(1) direct lookup from id_indices (populated during add_nodes)
        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = self
            .inner
            .lookup_by_id_normalized(source_type, &source_value)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Source node with id {:?} not found in type '{}'",
                    source_value, source_type
                ))
            })?;

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = self
            .inner
            .lookup_by_id_normalized(target_type, &target_value)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Target node with id {:?} not found in type '{}'",
                    target_value, target_type
                ))
            })?;

        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        // Find shortest path and return raw indices
        match graph_algorithms::shortest_path(
            &self.inner,
            source_idx,
            target_idx,
            connection_types.as_deref(),
            via_types.as_deref(),
            deadline,
        ) {
            Some(path_result) => {
                let indices: Vec<usize> = path_result.path.iter().map(|idx| idx.index()).collect();
                Ok(PyList::new(py, indices)?.into())
            }
            None => Ok(py.None()),
        }
    }

    /// Find all paths between two nodes up to a maximum number of hops.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///     max_hops: Maximum path length to search (default: 5)
    ///     max_results: Stop after finding this many paths (default: unlimited).
    ///                  Use this to prevent OOM on dense graphs.
    ///
    /// Returns:
    ///     A list of path dictionaries, each with 'path', 'connections', and 'length'
    #[pyo3(signature = (source_type, source_id, target_type, target_id, max_hops=None, max_results=None, connection_types=None, via_types=None, timeout_ms=None))]
    #[allow(clippy::too_many_arguments)]
    fn all_paths(
        &self,
        py: Python<'_>,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
        max_hops: Option<usize>,
        max_results: Option<usize>,
        connection_types: Option<Vec<String>>,
        via_types: Option<Vec<String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let max_hops = max_hops.unwrap_or(5);

        // Look up source node
        let source_lookup = lookups::TypeLookup::new(&self.inner.graph, source_type.to_string())
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = source_lookup.check_uid(&source_value).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Source node with id {:?} not found in type '{}'",
                source_value, source_type
            ))
        })?;

        // Look up target node
        let target_lookup = if target_type == source_type {
            source_lookup
        } else {
            lookups::TypeLookup::new(&self.inner.graph, target_type.to_string())
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?
        };

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = target_lookup.check_uid(&target_value).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Target node with id {:?} not found in type '{}'",
                target_value, target_type
            ))
        })?;

        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        // Find all paths
        let paths = graph_algorithms::all_paths(
            &self.inner,
            source_idx,
            target_idx,
            max_hops,
            max_results,
            connection_types.as_deref(),
            via_types.as_deref(),
            deadline,
        );

        // Convert to Python output
        let result_list = PyList::empty(py);
        for path in paths {
            let path_dict = PyDict::new(py);

            // Build path info list
            let path_list = PyList::empty(py);
            for &node_idx in &path {
                if let Some(info) = graph_algorithms::get_node_info(&self.inner, node_idx) {
                    let node_dict = PyDict::new(py);
                    node_dict.set_item("type", &info.node_type)?;
                    node_dict.set_item("title", &info.title)?;
                    node_dict.set_item("id", py_out::value_to_py(py, &info.id)?)?;
                    path_list.append(node_dict)?;
                }
            }
            path_dict.set_item("path", path_list)?;

            // Build connections list
            let connections = graph_algorithms::get_path_connections(&self.inner, &path);
            let conn_list = PyList::empty(py);
            for conn in connections {
                match conn {
                    Some(c) => conn_list.append(&c)?,
                    None => conn_list.append(py.None())?,
                }
            }
            path_dict.set_item("connections", conn_list)?;
            path_dict.set_item("length", path.len().saturating_sub(1))?;

            result_list.append(path_dict)?;
        }

        Ok(result_list.into())
    }

    /// Find all connected components in the graph.
    ///
    /// Args:
    ///     weak: If True (default), find weakly connected components (treating graph as undirected).
    ///           If False, find strongly connected components (respecting edge direction).
    ///
    /// Returns:
    ///     A list of components, each component is a list of node info dicts.
    ///     Components are sorted by size (largest first).
    #[pyo3(signature = (weak=None, titles_only=None))]
    fn connected_components(
        &self,
        py: Python<'_>,
        weak: Option<bool>,
        titles_only: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let weak = weak.unwrap_or(true);

        let components = if weak {
            graph_algorithms::weakly_connected_components(&self.inner)
        } else {
            graph_algorithms::connected_components(&self.inner)
        };

        if titles_only.unwrap_or(false) {
            // Lightweight: return [[title1, title2, ...], [...], ...]
            // Creates only string objects — no PyDicts per node
            let result_list = PyList::empty(py);
            for component in components {
                let comp_list = PyList::empty(py);
                for &node_idx in &component {
                    if let Some(node) = self.inner.get_node(node_idx) {
                        let title_str = match &node.title {
                            Value::String(s) => s.as_str(),
                            _ => "",
                        };
                        comp_list.append(title_str)?;
                    }
                }
                result_list.append(comp_list)?;
            }
            Ok(result_list.into())
        } else {
            // Full: return [[{type, title, id}, ...], [...], ...]
            let key_type = pyo3::intern!(py, "type");
            let key_title = pyo3::intern!(py, "title");
            let key_id = pyo3::intern!(py, "id");

            let result_list = PyList::empty(py);
            for component in components {
                let comp_list = PyList::empty(py);
                for &node_idx in &component {
                    if let Some(node) = self.inner.get_node(node_idx) {
                        let node_dict = PyDict::new(py);
                        node_dict.set_item(key_type, &node.node_type)?;
                        let title_str = match &node.title {
                            Value::String(s) => s.as_str(),
                            _ => "",
                        };
                        node_dict.set_item(key_title, title_str)?;
                        node_dict.set_item(key_id, py_out::value_to_py(py, &node.id)?)?;
                        comp_list.append(node_dict)?;
                    }
                }
                result_list.append(comp_list)?;
            }
            Ok(result_list.into())
        }
    }

    /// Check if two nodes are connected (directly or indirectly).
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     True if the nodes are connected, False otherwise
    fn are_connected(
        &self,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        // Look up source node
        let source_lookup = lookups::TypeLookup::new(&self.inner.graph, source_type.to_string())
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = source_lookup.check_uid(&source_value).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Source node with id {:?} not found in type '{}'",
                source_value, source_type
            ))
        })?;

        // Look up target node
        let target_lookup = if target_type == source_type {
            source_lookup
        } else {
            lookups::TypeLookup::new(&self.inner.graph, target_type.to_string())
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?
        };

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = target_lookup.check_uid(&target_value).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Target node with id {:?} not found in type '{}'",
                target_value, target_type
            ))
        })?;

        Ok(graph_algorithms::are_connected(
            &self.inner,
            source_idx,
            target_idx,
        ))
    }

    /// Get the degree (number of connections) for nodes in the current selection.
    ///
    /// Returns:
    ///     A dictionary mapping node titles to their degree counts
    fn degrees(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result_dict = PyDict::new(py);

        let level_count = self.selection.get_level_count();
        if level_count == 0 {
            return Ok(result_dict.into());
        }

        let level = self.selection.get_level(level_count - 1).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No selection level")
        })?;

        for node_idx in level.iter_node_indices() {
            if let Some(info) = graph_algorithms::get_node_info(&self.inner, node_idx) {
                let degree = graph_algorithms::node_degree(&self.inner, node_idx);
                result_dict.set_item(&info.title, degree)?;
            }
        }

        Ok(result_dict.into())
    }

    // ========================================================================
    // Centrality Algorithms
    // ========================================================================

    /// Calculate betweenness centrality for nodes in the graph.
    ///
    /// Betweenness centrality measures how often a node lies on the shortest path
    /// between other pairs of nodes. Higher values indicate nodes that are more
    /// important as "bridges" in the network.
    ///
    /// Args:
    ///     normalized: If True, normalize scores to [0, 1] range (default: True)
    ///     sample_size: Optional number of source nodes to sample for faster computation
    ///                  on large graphs. If None, uses all nodes.
    ///     top_k: Return only the top K nodes by centrality (default: all)
    ///
    /// Returns:
    ///     A list of dicts with 'node_type', 'title', 'id', and 'score' keys,
    ///     sorted by score descending.
    ///
    /// Example:
    ///     ```python
    ///     # Find the most central nodes (bridges)
    ///     central_nodes = graph.betweenness_centrality(top_k=10)
    ///     for node in central_nodes:
    ///         print(f"{node['title']}: {node['score']:.4f}")
    ///     ```
    #[pyo3(signature = (normalized=None, sample_size=None, connection_types=None, top_k=None, as_dict=None, timeout_ms=None, to_df=None))]
    #[allow(clippy::too_many_arguments)]
    fn betweenness_centrality(
        &self,
        py: Python<'_>,
        normalized: Option<bool>,
        sample_size: Option<usize>,
        connection_types: Option<Vec<String>>,
        top_k: Option<usize>,
        as_dict: Option<bool>,
        timeout_ms: Option<u64>,
        to_df: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let normalized = normalized.unwrap_or(true);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        let inner = Arc::clone(&self.inner);
        let results = py.detach(move || {
            graph_algorithms::betweenness_centrality(
                &inner,
                normalized,
                sample_size,
                connection_types.as_deref(),
                deadline,
            )
        });

        if to_df.unwrap_or(false) {
            centrality_results_to_dataframe(py, &self.inner, results, top_k)
        } else if as_dict.unwrap_or(false) {
            centrality_results_to_py_dict(py, &self.inner, results, top_k)
        } else {
            {
                let view = cypher::ResultView::from_centrality(&self.inner, results, top_k);
                Py::new(py, view).map(|v| v.into_any())
            }
        }
    }

    /// Calculate PageRank centrality for nodes in the graph.
    ///
    /// PageRank measures the importance of nodes based on the structure of
    /// incoming links. Originally developed by Google for ranking web pages.
    ///
    /// Args:
    ///     damping_factor: Probability of following a link (default: 0.85)
    ///     max_iterations: Maximum number of iterations (default: 100)
    ///     tolerance: Convergence threshold (default: 1e-6)
    ///     top_k: Return only the top K nodes by centrality (default: all)
    ///
    /// Returns:
    ///     A list of dicts with 'node_type', 'title', 'id', and 'score' keys,
    ///     sorted by score descending.
    ///
    /// Example:
    ///     ```python
    ///     # Find the most important nodes by PageRank
    ///     important_nodes = graph.pagerank(top_k=10)
    ///     for node in important_nodes:
    ///         print(f"{node['title']}: {node['score']:.6f}")
    ///     ```
    #[pyo3(signature = (damping_factor=None, max_iterations=None, tolerance=None, connection_types=None, top_k=None, as_dict=None, timeout_ms=None, to_df=None))]
    #[allow(clippy::too_many_arguments)]
    fn pagerank(
        &self,
        py: Python<'_>,
        damping_factor: Option<f64>,
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
        connection_types: Option<Vec<String>>,
        top_k: Option<usize>,
        as_dict: Option<bool>,
        timeout_ms: Option<u64>,
        to_df: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let damping = damping_factor.unwrap_or(0.85);
        let max_iter = max_iterations.unwrap_or(100);
        let tol = tolerance.unwrap_or(1e-6);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        let inner = Arc::clone(&self.inner);
        let results = py.detach(move || {
            graph_algorithms::pagerank(
                &inner,
                damping,
                max_iter,
                tol,
                connection_types.as_deref(),
                deadline,
            )
        });

        if to_df.unwrap_or(false) {
            centrality_results_to_dataframe(py, &self.inner, results, top_k)
        } else if as_dict.unwrap_or(false) {
            centrality_results_to_py_dict(py, &self.inner, results, top_k)
        } else {
            {
                let view = cypher::ResultView::from_centrality(&self.inner, results, top_k);
                Py::new(py, view).map(|v| v.into_any())
            }
        }
    }

    /// Calculate degree centrality for nodes in the graph.
    ///
    /// Degree centrality simply counts the number of connections each node has.
    /// This is the simplest centrality measure but often effective.
    ///
    /// Args:
    ///     normalized: If True, normalize by (n-1) for values in [0, 1] (default: True)
    ///     top_k: Return only the top K nodes by centrality (default: all)
    ///
    /// Returns:
    ///     A list of dicts with 'node_type', 'title', 'id', and 'score' keys,
    ///     sorted by score descending.
    ///
    /// Example:
    ///     ```python
    ///     # Find the most connected nodes
    ///     connected_nodes = graph.degree_centrality(top_k=10)
    ///     ```
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (normalized=None, connection_types=None, top_k=None, as_dict=None, timeout_ms=None, to_df=None))]
    fn degree_centrality(
        &self,
        py: Python<'_>,
        normalized: Option<bool>,
        connection_types: Option<Vec<String>>,
        top_k: Option<usize>,
        as_dict: Option<bool>,
        timeout_ms: Option<u64>,
        to_df: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let normalized = normalized.unwrap_or(true);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        let inner = Arc::clone(&self.inner);
        let results = py.detach(move || {
            graph_algorithms::degree_centrality(
                &inner,
                normalized,
                connection_types.as_deref(),
                deadline,
            )
        });

        if to_df.unwrap_or(false) {
            centrality_results_to_dataframe(py, &self.inner, results, top_k)
        } else if as_dict.unwrap_or(false) {
            centrality_results_to_py_dict(py, &self.inner, results, top_k)
        } else {
            {
                let view = cypher::ResultView::from_centrality(&self.inner, results, top_k);
                Py::new(py, view).map(|v| v.into_any())
            }
        }
    }

    /// Calculate closeness centrality for nodes in the graph.
    ///
    /// Closeness centrality measures how close a node is to all other nodes,
    /// based on the sum of shortest path distances. Higher values mean the
    /// node can reach other nodes more quickly.
    ///
    /// Args:
    ///     normalized: If True, adjust for disconnected components (default: True)
    ///     top_k: Return only the top K nodes by centrality (default: all)
    ///
    /// Returns:
    ///     A list of dicts with 'node_type', 'title', 'id', and 'score' keys,
    ///     sorted by score descending.
    ///
    /// Example:
    ///     ```python
    ///     # Find nodes that are "closest" to all others
    ///     close_nodes = graph.closeness_centrality(top_k=10)
    ///     ```
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (normalized=None, connection_types=None, top_k=None, as_dict=None, timeout_ms=None, to_df=None))]
    fn closeness_centrality(
        &self,
        py: Python<'_>,
        normalized: Option<bool>,
        connection_types: Option<Vec<String>>,
        top_k: Option<usize>,
        as_dict: Option<bool>,
        timeout_ms: Option<u64>,
        to_df: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let normalized = normalized.unwrap_or(true);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));

        let inner = Arc::clone(&self.inner);
        let results = py.detach(move || {
            graph_algorithms::closeness_centrality(
                &inner,
                normalized,
                connection_types.as_deref(),
                deadline,
            )
        });

        if to_df.unwrap_or(false) {
            centrality_results_to_dataframe(py, &self.inner, results, top_k)
        } else if as_dict.unwrap_or(false) {
            centrality_results_to_py_dict(py, &self.inner, results, top_k)
        } else {
            {
                let view = cypher::ResultView::from_centrality(&self.inner, results, top_k);
                Py::new(py, view).map(|v| v.into_any())
            }
        }
    }

    // ========================================================================
    // Community Detection
    // ========================================================================

    /// Detect communities using the Louvain modularity optimization algorithm.
    ///
    /// Args:
    ///     weight_property: Edge property to use as weight (default: all edges weight 1.0)
    ///     resolution: Resolution parameter (default: 1.0). Higher values produce more communities.
    ///
    /// Returns:
    ///     dict with 'communities' (dict mapping community_id -> list of node dicts),
    ///     'modularity' (float), and 'num_communities' (int)
    #[pyo3(signature = (weight_property=None, resolution=None, connection_types=None, timeout_ms=None))]
    fn louvain_communities(
        &self,
        py: Python<'_>,
        weight_property: Option<String>,
        resolution: Option<f64>,
        connection_types: Option<Vec<String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let res = resolution.unwrap_or(1.0);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
        let result = graph_algorithms::louvain_communities(
            &self.inner,
            weight_property.as_deref(),
            res,
            connection_types.as_deref(),
            deadline,
        );
        community_results_to_py(py, &self.inner, result)
    }

    /// Detect communities using label propagation.
    ///
    /// Each node adopts the most frequent label among its neighbors until convergence.
    ///
    /// Args:
    ///     max_iterations: Maximum iterations before stopping (default: 100)
    ///
    /// Returns:
    ///     dict with 'communities' (dict mapping community_id -> list of node dicts),
    ///     'modularity' (float), and 'num_communities' (int)
    #[pyo3(signature = (max_iterations=None, connection_types=None, timeout_ms=None))]
    fn label_propagation(
        &self,
        py: Python<'_>,
        max_iterations: Option<usize>,
        connection_types: Option<Vec<String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let max_iter = max_iterations.unwrap_or(100);
        let deadline =
            timeout_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
        let result = graph_algorithms::label_propagation(
            &self.inner,
            max_iter,
            connection_types.as_deref(),
            deadline,
        );
        community_results_to_py(py, &self.inner, result)
    }

    // ========================================================================
    // Subgraph Extraction Methods
    // ========================================================================

    /// Expand the current selection by N hops.
    ///
    /// This performs a breadth-first expansion from all currently selected nodes,
    /// including all nodes within the specified number of hops. The expansion
    /// considers edges in both directions (undirected).
    ///
    /// Args:
    ///     hops: Number of hops to expand (default: 1)
    ///
    /// Returns:
    ///     A new KnowledgeGraph with the expanded selection
    ///
    /// Example:
    ///     ```python
    ///     # Start with a single field and expand to include connected nodes
    ///     expanded = graph.select('Field').where({'name': 'EKOFISK'}).expand(hops=2)
    ///     ```
    #[pyo3(signature = (hops=None))]
    fn expand(&mut self, hops: Option<usize>) -> PyResult<Self> {
        let hops = hops.unwrap_or(1);
        let mut new_kg = self.clone();

        // Record plan step - use node_count() to avoid allocation
        let estimated = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        subgraph::expand_selection(&self.inner, &mut new_kg.selection, hops)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Record actual result - use node_count() to avoid allocation
        let actual = new_kg
            .selection
            .get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);
        new_kg
            .selection
            .add_plan_step(PlanStep::new("EXPAND", None, estimated).with_actual_rows(actual));

        Ok(new_kg)
    }

    /// Extract the currently selected nodes into a new independent subgraph.
    ///
    /// Creates a new KnowledgeGraph containing only the nodes in the current
    /// selection and all edges that connect those nodes. The new graph is
    /// completely independent from the original.
    ///
    /// Returns:
    ///     A new KnowledgeGraph containing only the selected nodes and their
    ///     connecting edges
    ///
    /// Example:
    ///     ```python
    ///     # Extract a subgraph of a specific region
    ///     subgraph = (
    ///         graph.select('Field')
    ///         .where({'region': 'North Sea'})
    ///         .expand(hops=2)
    ///         .to_subgraph()
    ///     )
    ///     # Save the subgraph for later use
    ///     subgraph.save('north_sea_region.kgl')
    ///     ```
    fn to_subgraph(&self) -> PyResult<Self> {
        let extracted = subgraph::extract_subgraph(&self.inner, &self.selection)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        Ok(KnowledgeGraph {
            inner: Arc::new(extracted),
            selection: CowSelection::new(),
            reports: OperationReports::new(), // Fresh reports for new graph
            last_mutation_stats: None,
            embedder: None,
            temporal_context: TemporalContext::default(),
        })
    }

    /// Get statistics about the subgraph that would be extracted.
    ///
    /// Returns information about what would be included in a subgraph extraction
    /// without actually creating the subgraph. Useful for understanding the
    /// scope of an extraction before committing to it.
    ///
    /// Returns:
    ///     A dictionary with:
    ///         - 'node_count': Total number of nodes
    ///         - 'edge_count': Total number of edges
    ///         - 'node_types': Dict of node type -> count
    ///         - 'connection_types': Dict of connection type -> count
    fn subgraph_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stats = subgraph::get_subgraph_stats(&self.inner, &self.selection)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let result_dict = PyDict::new(py);
        result_dict.set_item("node_count", stats.node_count)?;
        result_dict.set_item("edge_count", stats.edge_count)?;

        let node_types_dict = PyDict::new(py);
        for (node_type, count) in &stats.node_types {
            node_types_dict.set_item(node_type, count)?;
        }
        result_dict.set_item("node_types", node_types_dict)?;

        let conn_types_dict = PyDict::new(py);
        for (conn_type, count) in &stats.connection_types {
            conn_types_dict.set_item(conn_type, count)?;
        }
        result_dict.set_item("connection_types", conn_types_dict)?;

        Ok(result_dict.into())
    }
}
