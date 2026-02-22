// src/graph/pymethods_timeseries.rs
//
// PyO3 methods for timeseries operations on KnowledgeGraph.

use super::timeseries::{self, NodeTimeseries, TimeseriesConfig};
use super::{get_graph_mut, KnowledgeGraph};
use crate::datatypes::py_in;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

#[pymethods]
impl KnowledgeGraph {
    // ── Config ────────────────────────────────────────────────────────────

    /// Configure timeseries metadata for a node type.
    ///
    /// `resolution` declares the time granularity: "year", "month", or "day".
    /// `channels` lists known channel names (informational).
    /// `units` maps channel names to unit strings (e.g. {"oil": "MSm3"}).
    /// `bin_type` describes what values represent: "total", "mean", or "sample".
    #[pyo3(signature = (node_type, *, resolution, channels=None, units=None, bin_type=None))]
    fn set_timeseries(
        &mut self,
        node_type: String,
        resolution: String,
        channels: Option<Vec<String>>,
        units: Option<HashMap<String, String>>,
        bin_type: Option<String>,
    ) -> PyResult<()> {
        timeseries::validate_resolution(&resolution)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let graph = get_graph_mut(&mut self.inner);
        graph.timeseries_configs.insert(
            node_type,
            TimeseriesConfig {
                resolution,
                channels: channels.unwrap_or_default(),
                units: units.unwrap_or_default(),
                bin_type,
            },
        );
        Ok(())
    }

    /// Get timeseries configuration for a node type, or all types.
    #[pyo3(signature = (node_type=None))]
    fn get_timeseries_config(
        &self,
        py: Python<'_>,
        node_type: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let graph = &self.inner;
        if let Some(nt) = node_type {
            match graph.timeseries_configs.get(&nt) {
                Some(config) => ts_config_to_py(py, config),
                None => Ok(py.None()),
            }
        } else {
            if graph.timeseries_configs.is_empty() {
                return Ok(py.None());
            }
            let result = PyDict::new(py);
            for (nt, config) in &graph.timeseries_configs {
                result.set_item(nt, ts_config_to_py(py, config)?)?;
            }
            Ok(result.into_any().unbind())
        }
    }

    // ── Per-node data ─────────────────────────────────────────────────────

    /// Set the sorted time index for a specific node.
    ///
    /// If the node already has a timeseries, this replaces its time index
    /// and clears all channels (since they would no longer align).
    /// Key depth must match the resolution set via `set_timeseries()`.
    #[pyo3(signature = (node_id, keys))]
    fn set_time_index(&mut self, node_id: &Bound<'_, PyAny>, keys: Vec<Vec<i64>>) -> PyResult<()> {
        timeseries::validate_keys_sorted(&keys)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let graph = get_graph_mut(&mut self.inner);
        let id_val = py_in::py_value_to_value(node_id)?;
        let node_idx = find_node_by_id(graph, &id_val)?;

        // Validate key depth against resolution if config exists
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let nt = node.node_type.clone();
            if let Some(config) = graph.timeseries_configs.get(&nt) {
                if let Ok(expected_depth) = timeseries::resolution_depth(&config.resolution) {
                    if let Some(first_key) = keys.first() {
                        if first_key.len() != expected_depth {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "Key depth {} does not match resolution '{}' (expected {})",
                                first_key.len(),
                                config.resolution,
                                expected_depth
                            )));
                        }
                    }
                }
            }
        }

        graph.timeseries_store.insert(
            node_idx.index(),
            NodeTimeseries {
                keys,
                channels: HashMap::new(),
            },
        );

        Ok(())
    }

    /// Add a timeseries channel to a node.
    ///
    /// The node must already have a time index set (via `set_time_index` or `add_timeseries`).
    /// The values length must match the time index length.
    #[pyo3(signature = (node_id, channel_name, values))]
    fn add_ts_channel(
        &mut self,
        node_id: &Bound<'_, PyAny>,
        channel_name: String,
        values: Vec<f64>,
    ) -> PyResult<()> {
        let graph = get_graph_mut(&mut self.inner);
        let id_val = py_in::py_value_to_value(node_id)?;
        let node_idx = find_node_by_id(graph, &id_val)?;

        let ts = graph
            .timeseries_store
            .get_mut(&node_idx.index())
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Node has no time index. Call set_time_index() first.",
                )
            })?;

        timeseries::validate_channel_length(ts.keys.len(), values.len(), &channel_name)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Update config channels list
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let nt = node.node_type.clone();
            if let Some(config) = graph.timeseries_configs.get_mut(&nt) {
                if !config.channels.contains(&channel_name) {
                    config.channels.push(channel_name.clone());
                }
            }
        }

        ts.channels.insert(channel_name, values);
        Ok(())
    }

    /// Bulk-load timeseries data from a DataFrame.
    ///
    /// Groups rows by `fk`, sorts by `time_key`, and attaches the resulting
    /// timeseries to matching nodes (found by node ID).
    ///
    /// `resolution` is required if `set_timeseries()` has not been called for
    /// this node type. The number of `time_key` columns must match the resolution depth.
    ///
    /// `channels` accepts either:
    /// - a list of column names (used as channel names): `["oil", "gas"]`
    /// - a dict mapping channel names to column names: `{"oil": "prfOilCol"}`
    ///
    /// `units` optionally maps channel names to unit strings, merged into config.
    #[pyo3(signature = (node_type, *, data, fk, time_key, channels, resolution=None, units=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_timeseries(
        &mut self,
        py: Python<'_>,
        node_type: String,
        data: &Bound<'_, PyAny>,
        fk: String,
        time_key: Vec<String>,
        channels: &Bound<'_, PyAny>,
        resolution: Option<String>,
        units: Option<HashMap<String, String>>,
    ) -> PyResult<Py<PyAny>> {
        // Resolve resolution: parameter > existing config > error
        let graph_ref = &self.inner;
        let resolved_resolution = if let Some(r) = resolution {
            timeseries::validate_resolution(&r)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            r
        } else if let Some(config) = graph_ref.timeseries_configs.get(&node_type) {
            config.resolution.clone()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No resolution specified and no existing config. \
                 Pass resolution= or call set_timeseries() first.",
            ));
        };

        // Validate time_key column count matches resolution
        let expected_depth = timeseries::resolution_depth(&resolved_resolution)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        if time_key.len() != expected_depth {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Resolution '{}' requires {} time_key columns, got {}",
                resolved_resolution,
                expected_depth,
                time_key.len()
            )));
        }

        // Parse channels: dict or list
        let channel_map: Vec<(String, String)> = if let Ok(d) = data
            .py()
            .import("builtins")?
            .call_method1("isinstance", (channels, data.py().get_type::<PyDict>()))
            .and_then(|r| r.extract::<bool>())
        {
            if d {
                let dict = channels.cast::<PyDict>()?;
                dict.iter()
                    .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<String>()?)))
                    .collect::<PyResult<Vec<_>>>()?
            } else {
                let list: Vec<String> = channels.extract()?;
                list.into_iter().map(|c| (c.clone(), c)).collect()
            }
        } else {
            let list: Vec<String> = channels.extract()?;
            list.into_iter().map(|c| (c.clone(), c)).collect()
        };

        // Extract columns from DataFrame as Python lists
        let fk_col: Vec<Py<PyAny>> = data.get_item(&fk)?.call_method0("tolist")?.extract()?;

        let mut time_cols: Vec<Vec<i64>> = Vec::with_capacity(time_key.len());
        for tk in &time_key {
            let col: Vec<i64> = data.get_item(tk)?.call_method0("tolist")?.extract()?;
            time_cols.push(col);
        }

        let mut value_cols: Vec<(String, Vec<f64>)> = Vec::with_capacity(channel_map.len());
        for (ch_name, col_name) in &channel_map {
            let col: Vec<f64> = data.get_item(col_name)?.call_method0("tolist")?.extract()?;
            value_cols.push((ch_name.clone(), col));
        }

        let n_rows = fk_col.len();

        // Group row indices by FK value
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, fk_val) in fk_col.iter().enumerate() {
            let key = fk_val.bind(py).str()?.to_string();
            groups.entry(key).or_default().push(i);
        }

        // Build ID index for this node type
        let graph = get_graph_mut(&mut self.inner);
        graph.build_id_index(&node_type);

        let mut nodes_loaded = 0usize;
        let mut total_records = 0usize;

        for (fk_str, row_indices) in &groups {
            // Look up node by FK value (try as string, int, or unique_id)
            let node_idx = {
                let id_str = crate::datatypes::values::Value::String(fk_str.clone());
                if let Some(idx) = graph.lookup_by_id_normalized(&node_type, &id_str) {
                    idx
                } else if let Ok(n) = fk_str.parse::<i64>() {
                    let id_int = crate::datatypes::values::Value::Int64(n);
                    if let Some(idx) = graph.lookup_by_id_normalized(&node_type, &id_int) {
                        idx
                    } else {
                        continue; // No matching node
                    }
                } else {
                    continue; // No matching node
                }
            };

            // Sort row indices by time key
            let mut sorted_rows: Vec<usize> = row_indices.clone();
            sorted_rows.sort_by(|&a, &b| {
                for tc in &time_cols {
                    match tc[a].cmp(&tc[b]) {
                        std::cmp::Ordering::Equal => continue,
                        other => return other,
                    }
                }
                std::cmp::Ordering::Equal
            });

            // Build composite keys
            let keys: Vec<Vec<i64>> = sorted_rows
                .iter()
                .map(|&ri| time_cols.iter().map(|tc| tc[ri]).collect())
                .collect();

            // Build channels
            let mut channels_data: HashMap<String, Vec<f64>> = HashMap::new();
            for (ch_name, col) in &value_cols {
                let values: Vec<f64> = sorted_rows.iter().map(|&ri| col[ri]).collect();
                channels_data.insert(ch_name.clone(), values);
            }

            graph.timeseries_store.insert(
                node_idx.index(),
                NodeTimeseries {
                    keys,
                    channels: channels_data,
                },
            );

            nodes_loaded += 1;
            total_records += sorted_rows.len();
        }

        // Auto-configure / update TimeseriesConfig
        let channel_names: Vec<String> = channel_map.iter().map(|(n, _)| n.clone()).collect();
        let existing = graph.timeseries_configs.get(&node_type);
        let mut merged_channels = existing.map(|c| c.channels.clone()).unwrap_or_default();
        for ch in &channel_names {
            if !merged_channels.contains(ch) {
                merged_channels.push(ch.clone());
            }
        }
        let mut merged_units = existing.map(|c| c.units.clone()).unwrap_or_default();
        if let Some(u) = units {
            for (k, v) in u {
                merged_units.insert(k, v);
            }
        }
        let bin_type = existing.and_then(|c| c.bin_type.clone());

        graph.timeseries_configs.insert(
            node_type,
            TimeseriesConfig {
                resolution: resolved_resolution,
                channels: merged_channels,
                units: merged_units,
                bin_type,
            },
        );

        // Return summary dict
        let result = PyDict::new(py);
        result.set_item("nodes_loaded", nodes_loaded)?;
        result.set_item("total_records", total_records)?;
        result.set_item("total_rows", n_rows)?;
        Ok(result.into_any().unbind())
    }

    // ── Retrieval ─────────────────────────────────────────────────────────

    /// Extract timeseries data for a node.
    ///
    /// Range arguments `start`/`end` are date strings (e.g. "2020", "2020-2").
    /// If `channel` is given, returns `{keys: [...], values: [...]}`.
    /// Otherwise returns `{keys: [...], channels: {name: [...], ...}}`.
    /// Returns None if the node has no timeseries data.
    #[pyo3(signature = (node_id, channel=None, start=None, end=None))]
    fn get_timeseries(
        &self,
        py: Python<'_>,
        node_id: &Bound<'_, PyAny>,
        channel: Option<String>,
        start: Option<String>,
        end: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let graph = &self.inner;
        let id_val = py_in::py_value_to_value(node_id)?;
        let node_idx = find_node_by_id_readonly(graph, &id_val)?;

        let ts = match graph.get_node_timeseries(node_idx.index()) {
            Some(ts) => ts,
            None => return Ok(py.None()),
        };

        // Parse date strings and compute range
        let start_key = start
            .as_ref()
            .map(|s| timeseries::parse_date_string(s))
            .transpose()
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let end_key = end
            .as_ref()
            .map(|s| timeseries::parse_date_string(s))
            .transpose()
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let key_depth = ts.keys.first().map(|k| k.len()).unwrap_or(1);
        let (lo, hi) = timeseries::find_range(
            &ts.keys,
            start_key.as_deref(),
            end_key.as_deref(),
            key_depth,
        );
        let keys_slice = &ts.keys[lo..hi];

        let result = PyDict::new(py);
        let keys_py = PyList::new(py, keys_slice.iter().map(|k| PyList::new(py, k).unwrap()))?;
        result.set_item("keys", keys_py)?;

        if let Some(ch_name) = channel {
            match ts.channels.get(&ch_name) {
                Some(values) => {
                    let values_py = PyList::new(py, &values[lo..hi])?;
                    result.set_item("values", values_py)?;
                }
                None => {
                    return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                        "Channel '{}' not found",
                        ch_name
                    )));
                }
            }
        } else {
            let channels_dict = PyDict::new(py);
            for (name, values) in &ts.channels {
                let values_py = PyList::new(py, &values[lo..hi])?;
                channels_dict.set_item(name, values_py)?;
            }
            result.set_item("channels", channels_dict)?;
        }

        Ok(result.into_any().unbind())
    }

    /// Get the time index for a node, or None.
    #[pyo3(signature = (node_id,))]
    fn get_time_index(&self, py: Python<'_>, node_id: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let graph = &self.inner;
        let id_val = py_in::py_value_to_value(node_id)?;
        let node_idx = find_node_by_id_readonly(graph, &id_val)?;

        match graph.get_node_timeseries(node_idx.index()) {
            Some(ts) => {
                let keys_py = PyList::new(py, ts.keys.iter().map(|k| PyList::new(py, k).unwrap()))?;
                Ok(keys_py.into_any().unbind())
            }
            None => Ok(py.None()),
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Convert a TimeseriesConfig to a Python dict.
fn ts_config_to_py(py: Python<'_>, config: &TimeseriesConfig) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("resolution", &config.resolution)?;
    if !config.channels.is_empty() {
        let channels = PyList::new(py, &config.channels)?;
        dict.set_item("channels", channels)?;
    }
    if !config.units.is_empty() {
        let units_dict = PyDict::new(py);
        for (k, v) in &config.units {
            units_dict.set_item(k, v)?;
        }
        dict.set_item("units", units_dict)?;
    }
    if let Some(bt) = &config.bin_type {
        dict.set_item("bin_type", bt)?;
    }
    Ok(dict.into_any().unbind())
}

use super::schema::DirGraph;
use crate::datatypes::values::Value;
use petgraph::graph::NodeIndex;

/// Find a node by its ID value, scanning all type indices (mutable).
fn find_node_by_id(graph: &mut DirGraph, id: &Value) -> PyResult<NodeIndex> {
    let types: Vec<String> = graph.type_indices.keys().cloned().collect();
    for nt in &types {
        graph.build_id_index(nt);
        if let Some(idx) = graph.lookup_by_id_normalized(nt, id) {
            return Ok(idx);
        }
    }
    Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
        "No node found with id {:?}",
        id
    )))
}

/// Find a node by its ID value, scanning all type indices (read-only).
fn find_node_by_id_readonly(graph: &DirGraph, id: &Value) -> PyResult<NodeIndex> {
    for nt in graph.type_indices.keys() {
        if let Some(idx) = graph.lookup_by_id_normalized(nt, id) {
            return Ok(idx);
        }
    }
    Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
        "No node found with id {:?}",
        id
    )))
}
