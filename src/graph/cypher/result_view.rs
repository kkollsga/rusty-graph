// src/graph/cypher/result_view.rs
// Lazy ResultView — Polars-style result container.
// Data stays in Rust and converts to Python only on access.

use super::py_convert::{
    preprocess_values_owned, preprocessed_result_to_dataframe, preprocessed_value_to_py,
    stats_to_py, PreProcessedValue,
};
use super::result::{ClauseStats, CypherResult, MutationStats};
use crate::datatypes::values::Value;
use crate::graph::algorithms::graph_algorithms::CentralityResult;
use crate::graph::schema::{DirGraph, NodeData};
use crate::graph::storage::GraphRead;
use petgraph::Direction;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};
use pyo3::IntoPyObjectExt;
use std::borrow::Cow;
use std::collections::HashSet;

/// A single connection summary for display: connection type, target type, id, title.
#[derive(Clone)]
#[allow(dead_code)]
struct ConnectionSummary {
    connection_type: String,
    target_type: String,
    target_id: String,
    target_title: String,
    outgoing: bool, // true = this node → target, false = target → this node
}

/// Connection summaries for a single node.
#[derive(Clone, Default)]
#[allow(dead_code)]
struct NodeConnections {
    connections: Vec<ConnectionSummary>,
}

/// Lazy result container — data stays in Rust until you access it.
///
/// Returned by ``cypher()``, centrality methods, ``collect()`` (flat),
/// and ``sample()``.
///
/// Data is only converted to Python objects when you actually access rows
/// (via iteration, indexing, ``to_list()``, or ``to_df()``).  This makes
/// ``cypher()`` calls fast even for large result sets — the cost is
/// deferred to when you consume the data.
///
/// Quick reference::
///
///     r = g.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age")
///
///     len(r)           # row count (O(1), no conversion)
///     bool(r)          # True if non-empty
///     r[0]             # single row as dict  {'n.name': 'Alice', 'n.age': 30}
///     r[-1]            # last row
///     r[1:3]           # slice → new ResultView
///     r.columns        # column names
///     r.head(5)        # first 5 rows → new ResultView
///     r.tail(5)        # last 5 rows → new ResultView
///     r.to_list()      # all rows as list[dict]
///     r.to_df()        # pandas DataFrame
///     r.to_gdf()       # GeoDataFrame (requires geopandas)
///     r.stats          # mutation stats (CREATE/SET/DELETE only)
///     r.profile        # PROFILE stats (only with "PROFILE MATCH ...")
///
///     for row in r:    # iterate rows as dicts (one at a time)
///         print(row)
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
                        PreProcessedValue::Plain(Value::String(
                            node.node_type_str(&graph.interner).to_string(),
                        )),
                        PreProcessedValue::Plain(node.title().into_owned()),
                        PreProcessedValue::Plain(node.id().into_owned()),
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

    /// Discover property keys by scanning all nodes (fallback path).
    fn discover_property_keys(
        nodes: &[&NodeData],
        interner: &crate::graph::schema::StringInterner,
    ) -> Vec<String> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut keys: Vec<String> = Vec::new();
        for node in nodes {
            for key in node.property_keys(interner) {
                if seen.insert(key) {
                    keys.push(key.to_string());
                }
            }
        }
        keys.sort();
        keys
    }

    /// collect / sample with graph access: nodes + connection summaries.
    pub fn from_nodes_with_graph(
        graph: &DirGraph,
        node_indices: &[petgraph::graph::NodeIndex],
        temporal_context: &super::super::TemporalContext,
    ) -> Self {
        use crate::datatypes::values::format_value;
        use crate::graph::temporal;

        let nodes_vec: Vec<&NodeData> = node_indices
            .iter()
            .filter_map(|&idx| graph.get_node(idx))
            .collect();

        // Compute union of property keys.
        // Fast path: if all nodes share a type, use TypeSchema (O(1) key discovery).
        let prop_keys: Vec<String> = if nodes_vec.len() > 50 {
            let first_type = nodes_vec[0].node_type;
            let all_same_type = nodes_vec.iter().all(|n| n.node_type == first_type);
            if all_same_type {
                let first_type_str = graph.interner.resolve(first_type);
                if let Some(schema) = graph.type_schemas.get(first_type_str) {
                    let mut keys: Vec<String> = schema
                        .iter()
                        .filter_map(|(_, ik)| graph.interner.try_resolve(ik).map(|s| s.to_string()))
                        .collect();
                    keys.sort();
                    keys
                } else {
                    Self::discover_property_keys(&nodes_vec, &graph.interner)
                }
            } else {
                Self::discover_property_keys(&nodes_vec, &graph.interner)
            }
        } else {
            Self::discover_property_keys(&nodes_vec, &graph.interner)
        };

        let mut columns = vec!["type".into(), "title".into(), "id".into()];
        columns.extend(prop_keys.iter().cloned());

        let rows: Vec<Vec<PreProcessedValue>> = nodes_vec
            .iter()
            .map(|node| {
                let mut row = vec![
                    PreProcessedValue::Plain(Value::String(
                        node.node_type_str(&graph.interner).to_string(),
                    )),
                    PreProcessedValue::Plain(node.title().into_owned()),
                    PreProcessedValue::Plain(node.id().into_owned()),
                ];
                for key in &prop_keys {
                    row.push(PreProcessedValue::Plain(
                        node.get_property(key)
                            .map(Cow::into_owned)
                            .unwrap_or(Value::Null),
                    ));
                }
                row
            })
            .collect();

        // Resolve temporal context to a concrete ref_date or range for edge filtering
        use super::super::TemporalContext;
        let is_all = matches!(temporal_context, TemporalContext::All);
        let ref_date = match temporal_context {
            TemporalContext::Today => Some(chrono::Local::now().date_naive()),
            TemporalContext::At(d) => Some(*d),
            _ => None,
        };
        let range_dates = match temporal_context {
            TemporalContext::During(s, e) => Some((*s, *e)),
            _ => None,
        };

        // Cap connections per node for display purposes
        const MAX_CONNS_PER_NODE: usize = 50;

        // Inline helper: check if edge passes temporal filter.
        let edge_temporal_ok = |edge_data: &crate::graph::schema::EdgeData| -> bool {
            if is_all {
                return true;
            }
            let ct_str = edge_data.connection_type_str(&graph.interner);
            if let Some(configs) = graph.temporal_edge_configs.get(ct_str) {
                if let Some(d) = &ref_date {
                    return temporal::is_temporally_valid_multi(&edge_data.properties, configs, d);
                }
                if let Some((s, e)) = &range_dates {
                    return temporal::overlaps_range_multi(&edge_data.properties, configs, s, e);
                }
            }
            true
        };

        // Gather connection summaries per node, filtering temporal connections
        let g = &graph.graph;
        let node_connections: Vec<NodeConnections> = node_indices
            .iter()
            .map(|&idx| {
                let mut conns = Vec::with_capacity(16);

                // Outgoing: this node → target
                for edge in g.edges_directed(idx, Direction::Outgoing) {
                    if conns.len() >= MAX_CONNS_PER_NODE {
                        break;
                    }
                    if !edge_temporal_ok(edge.weight()) {
                        continue;
                    }
                    let target_idx = edge.target();
                    if let Some(target) = graph.get_node(target_idx) {
                        conns.push(ConnectionSummary {
                            connection_type: edge
                                .weight()
                                .connection_type_str(&graph.interner)
                                .to_string(),
                            target_type: target.node_type_str(&graph.interner).to_string(),
                            target_id: format_value(&target.id),
                            target_title: format_value(&target.title),
                            outgoing: true,
                        });
                    }
                }

                // Incoming: source → this node
                for edge in g.edges_directed(idx, Direction::Incoming) {
                    if conns.len() >= MAX_CONNS_PER_NODE {
                        break;
                    }
                    if !edge_temporal_ok(edge.weight()) {
                        continue;
                    }
                    let source_idx = edge.source();
                    if let Some(source) = graph.get_node(source_idx) {
                        conns.push(ConnectionSummary {
                            connection_type: edge
                                .weight()
                                .connection_type_str(&graph.interner)
                                .to_string(),
                            target_type: source.node_type_str(&graph.interner).to_string(),
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
        format_table(&self.columns, &self.rows)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Column names as a list of strings.
    ///
    /// Example::
    ///
    ///     r = g.cypher("MATCH (n) RETURN n.name, n.age")
    ///     r.columns   # ['n.name', 'n.age']
    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.columns.clone().into_py_any(py)
    }

    /// Mutation statistics (CREATE/SET/DELETE queries), or None for reads.
    ///
    /// Returns a dict with keys like ``nodes_created``, ``properties_set``,
    /// ``relationships_created``, etc.
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

    /// Materialize all rows as a list of dicts.
    ///
    /// Example::
    ///
    ///     r.to_list()  # [{'name': 'Alice', 'age': 30}, ...]
    fn to_list(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = pyo3::types::PyList::empty(py);
        for i in 0..self.rows.len() {
            list.append(self.row_to_py(py, i)?)?;
        }
        Ok(list.into_any().unbind())
    }

    /// First *n* rows as a new ResultView (default 5). Data stays lazy.
    ///
    /// Example::
    ///
    ///     r.head()     # first 5 rows
    ///     r.head(10)   # first 10 rows
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

    /// Last *n* rows as a new ResultView (default 5). Data stays lazy.
    ///
    /// Example::
    ///
    ///     r.tail()     # last 5 rows
    ///     r.tail(10)   # last 10 rows
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

    /// Materialize as a pandas DataFrame.
    ///
    /// Example::
    ///
    ///     df = r.to_df()
    ///     df.plot(x='year', y='count')
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

/// Format a ResultView as a Polars-style table.
///
/// Shows `shape: (rows, cols)` header, a bordered table with column names,
/// and for large results shows the first and last rows with `…` in between.
fn format_table(columns: &[String], rows: &[Vec<PreProcessedValue>]) -> String {
    if rows.is_empty() {
        return format!("shape: (0, {})\n(empty)", columns.len());
    }

    let n = rows.len();
    let max_col_width = 30;
    let max_display_rows = 20;

    // Decide which rows to show
    let (show_head, show_tail, truncated) = if n <= max_display_rows {
        (n, 0, false)
    } else {
        (10, 5, true)
    };

    // Format all visible cell values
    let mut formatted: Vec<Vec<String>> = Vec::new();
    for row in rows.iter().take(show_head) {
        formatted.push(
            row.iter()
                .map(|v| truncate_middle(&format_preprocessed_value(v), max_col_width))
                .collect(),
        );
    }
    if truncated {
        for row in rows.iter().skip(n - show_tail) {
            formatted.push(
                row.iter()
                    .map(|v| truncate_middle(&format_preprocessed_value(v), max_col_width))
                    .collect(),
            );
        }
    }

    // Compute column widths (header vs data)
    let num_cols = columns.len();
    let mut widths: Vec<usize> = columns.iter().map(|c| c.len()).collect();
    for row in &formatted {
        for (j, cell) in row.iter().enumerate() {
            if j < num_cols {
                widths[j] = widths[j].max(cell.len());
            }
        }
    }
    if truncated {
        // Ensure columns are wide enough for "…"
        for w in &mut widths {
            *w = (*w).max(1);
        }
    }

    let mut buf = String::with_capacity(n * 100);

    // Shape header
    buf.push_str(&format!("shape: ({}, {})\n", n, num_cols));

    // Top border: ┌──────┬──────┐
    buf.push('┌');
    for (j, w) in widths.iter().enumerate() {
        if j > 0 {
            buf.push('┬');
        }
        for _ in 0..(w + 2) {
            buf.push('─');
        }
    }
    buf.push_str("┐\n");

    // Header row: │ col1 ┆ col2 │
    buf.push('│');
    for (j, col) in columns.iter().enumerate() {
        if j > 0 {
            buf.push_str(" ┆");
        }
        buf.push_str(&format!(" {:width$}", col, width = widths[j]));
    }
    buf.push_str(" │\n");

    // Separator: ╞══════╪══════╡
    buf.push('╞');
    for (j, w) in widths.iter().enumerate() {
        if j > 0 {
            buf.push('╪');
        }
        for _ in 0..(w + 2) {
            buf.push('═');
        }
    }
    buf.push_str("╡\n");

    // Data rows (head)
    for row in &formatted[..show_head] {
        buf.push('│');
        for (j, w) in widths.iter().enumerate() {
            if j > 0 {
                buf.push_str(" ┆");
            }
            let cell = row.get(j).map(|s| s.as_str()).unwrap_or("");
            buf.push_str(&format!(" {:width$}", cell, width = *w));
        }
        buf.push_str(" │\n");
    }

    // Truncation row: │ …    ┆ …    │
    if truncated {
        buf.push('│');
        for (j, w) in widths.iter().enumerate() {
            if j > 0 {
                buf.push_str(" ┆");
            }
            buf.push_str(&format!(" {:width$}", "…", width = *w));
        }
        buf.push_str(" │\n");

        // Tail rows
        for row in &formatted[show_head..] {
            buf.push('│');
            for (j, w) in widths.iter().enumerate() {
                if j > 0 {
                    buf.push_str(" ┆");
                }
                let cell = row.get(j).map(|s| s.as_str()).unwrap_or("");
                buf.push_str(&format!(" {:width$}", cell, width = *w));
            }
            buf.push_str(" │\n");
        }
    }

    // Bottom border: └──────┴──────┘
    buf.push('└');
    for (j, w) in widths.iter().enumerate() {
        if j > 0 {
            buf.push('┴');
        }
        for _ in 0..(w + 2) {
            buf.push('─');
        }
    }
    buf.push_str("┘\n");

    buf
}

/// Truncate a string in the middle if it exceeds `max_len`, keeping both ends visible.
fn truncate_middle(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    let keep = (max_len - 5) / 2; // 5 chars for " ... "
    format!("{} ... {}", &s[..keep], &s[s.len() - keep..])
}

#[allow(dead_code)]
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
                                "  {:width$}  ◆ --{}--> {}({}, {})\n",
                                "",
                                c.connection_type,
                                c.target_type,
                                c.target_id,
                                c.target_title,
                                width = key_width,
                            ));
                        } else {
                            buf.push_str(&format!(
                                "  {:width$}  {}({}, {}) --{}--> ◆\n",
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
