// src/graph/cypher/result_view.rs
// Lazy ResultView — Polars-style result container.
// Data stays in Rust and converts to Python only on access.

use super::py_convert::{
    preprocess_values_owned, preprocessed_result_to_dataframe, preprocessed_value_to_py,
    stats_to_py, PreProcessedValue,
};
use super::result::{ClauseStats, CypherResult, MutationStats};
use crate::datatypes::values::Value;
use crate::graph::graph_algorithms::CentralityResult;
use crate::graph::schema::{DirGraph, NodeData};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};
use pyo3::IntoPyObjectExt;
use std::collections::HashSet;

/// A single connection summary for display: connection type, target type, id, title.
#[derive(Clone)]
struct ConnectionSummary {
    connection_type: String,
    target_type: String,
    target_id: String,
    target_title: String,
    outgoing: bool, // true = this node → target, false = target → this node
}

/// Connection summaries for a single node.
#[derive(Clone, Default)]
struct NodeConnections {
    connections: Vec<ConnectionSummary>,
}

/// Lazy result view — data stays in Rust until accessed from Python.
///
/// Returned by `cypher()`, centrality methods, `collect()`, and `sample()`.
/// Supports `len()`, indexing, iteration, `to_list()`, and `to_df()`.
#[pyclass(name = "ResultView")]
pub struct ResultView {
    columns: Vec<String>,
    rows: Vec<Vec<PreProcessedValue>>,
    stats: Option<MutationStats>,
    profile: Option<Vec<ClauseStats>>,
    /// Per-row connection summaries (only populated for node-based results).
    node_connections: Option<Vec<NodeConnections>>,
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
        profile: Option<Vec<ClauseStats>>,
    ) -> Self {
        ResultView {
            columns,
            rows,
            stats,
            profile,
            node_connections: None,
        }
    }

    /// Cypher mutation path + Transaction: takes a CypherResult and preprocesses values.
    pub fn from_cypher_result(result: CypherResult) -> Self {
        let rows = preprocess_values_owned(result.rows);
        ResultView {
            columns: result.columns,
            rows,
            stats: result.stats,
            profile: result.profile,
            node_connections: None,
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
            profile: None,
            node_connections: None,
        }
    }

    /// collect / sample with graph access: nodes + connection summaries.
    pub fn from_nodes_with_graph(
        graph: &DirGraph,
        node_indices: &[petgraph::graph::NodeIndex],
    ) -> Self {
        use crate::datatypes::values::format_value;

        let nodes_vec: Vec<&NodeData> = node_indices
            .iter()
            .filter_map(|&idx| graph.get_node(idx))
            .collect();

        // Compute union of property keys
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

        // Gather connection summaries per node
        let node_connections: Vec<NodeConnections> = node_indices
            .iter()
            .map(|&idx| {
                let mut conns = Vec::new();

                // Outgoing: this node → target
                for edge in graph.graph.edges_directed(idx, Direction::Outgoing) {
                    let target_idx = edge.target();
                    if let Some(target) = graph.get_node(target_idx) {
                        conns.push(ConnectionSummary {
                            connection_type: edge.weight().connection_type.clone(),
                            target_type: target.node_type.clone(),
                            target_id: format_value(&target.id),
                            target_title: format_value(&target.title),
                            outgoing: true,
                        });
                    }
                }

                // Incoming: source → this node
                for edge in graph.graph.edges_directed(idx, Direction::Incoming) {
                    let source_idx = edge.source();
                    if let Some(source) = graph.get_node(source_idx) {
                        conns.push(ConnectionSummary {
                            connection_type: edge.weight().connection_type.clone(),
                            target_type: source.node_type.clone(),
                            target_id: format_value(&source.id),
                            target_title: format_value(&source.title),
                            outgoing: false,
                        });
                    }
                }

                NodeConnections { connections: conns }
            })
            .collect();

        ResultView {
            columns,
            rows,
            stats: None,
            profile: None,
            node_connections: Some(node_connections),
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

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // String key access — dict-like interface for 'columns' and 'rows'
        if let Ok(skey) = key.extract::<String>() {
            match skey.as_str() {
                "columns" => return self.columns(py),
                "rows" => {
                    let rows: Vec<Py<PyAny>> = (0..self.rows.len())
                        .map(|i| self.row_to_py(py, i))
                        .collect::<Result<_, _>>()?;
                    return rows.into_py_any(py);
                }
                _ => {
                    return Err(pyo3::exceptions::PyKeyError::new_err(skey));
                }
            }
        }
        if let Ok(idx) = key.extract::<isize>() {
            // Integer indexing — returns a single row as dict
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
        } else if let Ok(slice) = key.cast::<PySlice>() {
            // Slice indexing — returns a new ResultView
            let len = self.rows.len();
            let indices = slice.indices(len as isize)?;
            let mut sliced_rows = Vec::new();
            let mut i = indices.start;
            while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                if i >= 0 && (i as usize) < len {
                    sliced_rows.push(self.rows[i as usize].clone());
                }
                i += indices.step;
            }
            Py::new(
                py,
                ResultView {
                    columns: self.columns.clone(),
                    rows: sliced_rows,
                    stats: None,
                    profile: None,
                    node_connections: None,
                },
            )
            .map(|v| v.into_any())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "indices must be integers, slices, or string keys ('columns', 'rows')",
            ))
        }
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

    fn __str__(&self) -> String {
        if self.rows.len() <= 3 {
            format_result_view_multiline(
                &self.columns,
                &self.rows,
                self.node_connections.as_deref(),
            )
        } else {
            self.__repr__()
        }
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

    /// PROFILE execution statistics, or None for non-profiled queries.
    /// Returns a list of dicts with keys: clause, rows_in, rows_out, elapsed_us.
    #[getter]
    fn profile(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.profile {
            Some(steps) => {
                let list = pyo3::types::PyList::empty(py);
                for step in steps {
                    let dict = PyDict::new(py);
                    dict.set_item("clause", &step.clause_name)?;
                    dict.set_item("rows_in", step.rows_in)?;
                    dict.set_item("rows_out", step.rows_out)?;
                    dict.set_item("elapsed_us", step.elapsed_us)?;
                    list.append(dict)?;
                }
                Ok(list.into_any().unbind())
            }
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
            profile: None,
            node_connections: self.node_connections.as_ref().map(|nc| nc[..take].to_vec()),
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
            profile: None,
            node_connections: self
                .node_connections
                .as_ref()
                .map(|nc| nc[start..].to_vec()),
        }
    }

    /// Convert to a pandas DataFrame.
    fn to_df(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        preprocessed_result_to_dataframe(py, &self.columns, &self.rows)
    }

    /// Convert to a GeoDataFrame with a geometry column parsed from WKT.
    ///
    /// Materializes the data as a pandas DataFrame, then converts the
    /// specified WKT string column into shapely geometries and returns
    /// a geopandas GeoDataFrame.
    ///
    /// Args:
    ///     geometry_column: Name of the column containing WKT strings (default: 'geometry')
    ///     crs: Coordinate reference system (e.g. 'EPSG:4326'), or None
    ///
    /// Returns:
    ///     A geopandas GeoDataFrame
    #[pyo3(signature = (geometry_column="geometry", crs=None))]
    fn to_gdf(
        &self,
        py: Python<'_>,
        geometry_column: &str,
        crs: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let df = preprocessed_result_to_dataframe(py, &self.columns, &self.rows)?;

        let gpd = py.import("geopandas").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyImportError, _>(
                "geopandas is required for to_gdf(). Install it with: pip install geopandas",
            )
        })?;

        // gpd.GeoSeries.from_wkt(df[geometry_column])
        let geo_series_cls = gpd.getattr("GeoSeries")?;
        let wkt_col = df.call_method1(py, "__getitem__", (geometry_column,))?;
        let geo_series = geo_series_cls.call_method1("from_wkt", (wkt_col,))?;

        // df[geometry_column] = geo_series
        df.call_method1(py, "__setitem__", (geometry_column, geo_series))?;

        // gpd.GeoDataFrame(df, geometry=geometry_column, crs=crs)
        let kwargs = PyDict::new(py);
        kwargs.set_item("geometry", geometry_column)?;
        if let Some(crs_val) = crs {
            kwargs.set_item("crs", crs_val)?;
        }
        let gdf_cls = gpd.getattr("GeoDataFrame")?;
        let gdf = gdf_cls.call((df,), Some(&kwargs))?;
        Ok(gdf.unbind())
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

// ========================================================================
// Pretty-print formatting for ResultView
// ========================================================================

fn format_preprocessed_value(pv: &PreProcessedValue) -> String {
    match pv {
        PreProcessedValue::Plain(v) => crate::datatypes::values::format_value(v),
        PreProcessedValue::ParsedJson(jv) => {
            // Compact JSON string
            serde_json::to_string(jv).unwrap_or_else(|_| "???".to_string())
        }
    }
}

/// Truncate a string in the middle if it exceeds `max_len`, keeping both ends visible.
fn truncate_middle(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    let keep = (max_len - 5) / 2; // 5 chars for " ... "
    format!("{} ... {}", &s[..keep], &s[s.len() - keep..])
}

fn format_result_view_multiline(
    columns: &[String],
    rows: &[Vec<PreProcessedValue>],
    node_connections: Option<&[NodeConnections]>,
) -> String {
    if rows.is_empty() {
        return "(empty result)".to_string();
    }

    // Find the widest column name for alignment
    let key_width = columns.iter().map(|c| c.len()).max().unwrap_or(0);

    let mut buf = String::with_capacity(rows.len() * 300);

    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            buf.push('\n');
        }

        // Properties
        for (j, val) in row.iter().enumerate() {
            if j < columns.len() {
                let s = format_preprocessed_value(val);
                let display = truncate_middle(&s, 80);
                buf.push_str(&format!(
                    "  {:width$}  {}\n",
                    columns[j],
                    display,
                    width = key_width
                ));
            }
        }

        // Connection summaries
        if let Some(all_conns) = node_connections {
            if let Some(nc) = all_conns.get(i) {
                if !nc.connections.is_empty() {
                    buf.push_str(&format!("  {:width$}\n", "───", width = key_width + 4));
                    for c in &nc.connections {
                        if c.outgoing {
                            buf.push_str(&format!(
                                "  {:width$}  <> --{}--> {}({}, {})\n",
                                "",
                                c.connection_type,
                                c.target_type,
                                c.target_id,
                                c.target_title,
                                width = key_width,
                            ));
                        } else {
                            buf.push_str(&format!(
                                "  {:width$}  {}({}, {}) --{}--> <>\n",
                                "",
                                c.target_type,
                                c.target_id,
                                c.target_title,
                                c.connection_type,
                                width = key_width,
                            ));
                        }
                    }
                }
            }
        }
    }

    buf
}
