//! KGLite-specific MCP tools: `cypher_query`, `graph_overview`, `save_graph`.
//!
//! All three close over a [`GraphState`] holding the active
//! `KnowledgeGraph` Python object behind an `Arc<RwLock<…>>`. Wired
//! into the framework's tool router via `ToolRoute::new_dyn` so they
//! sit alongside the built-in source / GitHub / python tools.

use std::path::Path;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use mcp_server::python::json_to_py;
use mcp_server::server::McpServer;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use serde::{Deserialize, Serialize};

const NO_GRAPH: &str =
    "No active graph. Pass --graph X.kgl, or activate one via repo_management('org/repo').";

/// Shared active-graph state. Cloning is cheap (Arc).
#[derive(Clone, Default)]
pub struct GraphState {
    inner: Arc<RwLock<Option<ActiveGraph>>>,
}

#[derive(Debug)]
struct ActiveGraph {
    py_obj: Py<PyAny>,
    source_path: Option<std::path::PathBuf>,
}

impl GraphState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_kgl(&self, path: &Path) -> Result<()> {
        let py_obj = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let kglite = py.import("kglite")?;
            let g = kglite.call_method1("load", (path.to_string_lossy().as_ref(),))?;
            Ok(g.unbind())
        })?;
        *self.inner.write().unwrap() = Some(ActiveGraph {
            py_obj,
            source_path: Some(path.to_path_buf()),
        });
        Ok(())
    }

    pub fn build_code_tree(&self, dir: &Path) -> Result<()> {
        let py_obj = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let ct = py.import("kglite.code_tree")?;
            let g = ct.call_method1("build", (dir.to_string_lossy().as_ref(),))?;
            Ok(g.unbind())
        })?;
        *self.inner.write().unwrap() = Some(ActiveGraph {
            py_obj,
            source_path: None,
        });
        Ok(())
    }

    pub fn bind_embedder(&self, embedder: Py<PyAny>) -> Result<()> {
        let guard = self.inner.read().unwrap();
        let Some(active) = guard.as_ref() else {
            tracing::warn!("embedder loaded before any graph is active; binding deferred");
            return Ok(());
        };
        Python::attach(|py| -> PyResult<()> {
            active
                .py_obj
                .call_method1(py, "set_embedder", (embedder,))?;
            Ok(())
        })
        .context("graph.set_embedder failed")?;
        Ok(())
    }

    pub fn schema(&self) -> Option<(u64, u64)> {
        let guard = self.inner.read().unwrap();
        let active = guard.as_ref()?;
        Python::attach(|py| -> PyResult<(u64, u64)> {
            let s = active.py_obj.call_method0(py, "schema")?;
            let s = s.bind(py);
            Ok((
                s.get_item("node_count")?.extract()?,
                s.get_item("edge_count")?.extract()?,
            ))
        })
        .ok()
    }

    /// Run a closure against the active graph if one is loaded, or
    /// return the standard "no graph" message otherwise. This is the
    /// state-injection contract used by every kglite tool registered
    /// via `McpServer::register_typed_tool`.
    fn with_active<F>(&self, f: F) -> String
    where
        F: FnOnce(&ActiveGraph) -> String,
    {
        let guard = self.inner.read().unwrap();
        match guard.as_ref() {
            Some(active) => f(active),
            None => NO_GRAPH.to_string(),
        }
    }

    /// Resolve a code-entity qualified name to its source location via
    /// `graph.source(qualified_name, node_type)`. Used by the
    /// `read_code_source` tool to bridge the qualified-name → file path
    /// lookup that the framework's `read_source` can't do on its own.
    /// Returns the resolved `(file_path, line_number, end_line)` or an
    /// error message describing what went wrong (no graph, name not
    /// found, ambiguous match, missing line metadata).
    pub fn source_lookup(
        &self,
        qualified_name: &str,
        node_type: Option<&str>,
    ) -> Result<crate::code_source::SourceLookup, String> {
        let guard = self.inner.read().unwrap();
        let Some(active) = guard.as_ref() else {
            return Err(NO_GRAPH.to_string());
        };
        Python::attach(|py| -> Result<crate::code_source::SourceLookup, String> {
            let args =
                pyo3::types::PyTuple::new(py, [qualified_name]).map_err(|e| e.to_string())?;
            let kwargs = PyDict::new(py);
            if let Some(nt) = node_type {
                kwargs
                    .set_item("node_type", nt)
                    .map_err(|e| e.to_string())?;
            }
            let result = active
                .py_obj
                .call_method(py, "source", args, Some(&kwargs))
                .map_err(|e| format!("graph.source({qualified_name:?}) failed: {e}"))?;
            let dict = result
                .bind(py)
                .cast::<PyDict>()
                .map_err(|_| format!("graph.source({qualified_name:?}) did not return a dict"))?
                .clone();
            // `{"error": "..."}` or `{"ambiguous": true, "matches": [...]}` —
            // surface either with the body shape so the agent gets the
            // actual graph message rather than a generic error.
            if let Ok(Some(err)) = dict.get_item("error") {
                return Err(err.extract::<String>().unwrap_or_else(|_| {
                    format!("graph.source({qualified_name:?}) returned an error")
                }));
            }
            if let Ok(Some(amb)) = dict.get_item("ambiguous") {
                if amb.extract::<bool>().unwrap_or(false) {
                    let matches: String = dict
                        .get_item("matches")
                        .ok()
                        .flatten()
                        .and_then(|m| m.repr().ok().map(|r| r.to_string()))
                        .unwrap_or_default();
                    return Err(format!(
                        "ambiguous qualified_name {qualified_name:?}; matches: {matches}. \
                         Pass `node_type` to narrow."
                    ));
                }
            }
            let file_path: String = dict
                .get_item("file_path")
                .ok()
                .flatten()
                .ok_or_else(|| format!("graph.source({qualified_name:?}) returned no file_path"))?
                .extract()
                .map_err(|e| format!("file_path extraction failed: {e}"))?;
            let line_number: usize = dict
                .get_item("line_number")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<usize>().ok())
                .unwrap_or(1);
            let end_line: usize = dict
                .get_item("end_line")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<usize>().ok())
                .unwrap_or(line_number);
            Ok(crate::code_source::SourceLookup {
                file_path,
                line_number,
                end_line,
            })
        })
    }

    /// Run a parameterised Cypher template against the active graph.
    /// Used by the YAML-declared `tools[].cypher` registration path
    /// (see `crate::cypher_tools::register_cypher_tools`). The agent's
    /// argument map is forwarded as `params=...` to `graph.cypher`,
    /// which substitutes `$name` placeholders in the template.
    ///
    /// Returns the rendered tool body (or the standard "no graph"
    /// message / a Cypher error string).
    pub fn run_cypher_template(
        &self,
        template: &str,
        args: &serde_json::Map<String, serde_json::Value>,
    ) -> String {
        let guard = self.inner.read().unwrap();
        let Some(active) = guard.as_ref() else {
            return NO_GRAPH.to_string();
        };
        Python::attach(|py| -> PyResult<String> {
            let kwargs = PyDict::new(py);
            let params = PyDict::new(py);
            for (k, v) in args {
                params.set_item(k, json_to_py(py, v)?)?;
            }
            kwargs.set_item("params", params)?;
            let result = active
                .py_obj
                .call_method(py, "cypher", (template,), Some(&kwargs))?;
            format_cypher_result(py, &result)
        })
        .unwrap_or_else(|e| format!("Cypher error: {e}"))
    }
}

/// Format a `g.cypher(...)` Python return value into a tool body string.
/// String returns (CSV / EXPLAIN) pass through; iterable returns are
/// repr'd row-by-row with a 15-row inline cap, matching `cypher_query`.
fn format_cypher_result(py: Python<'_>, result: &Py<PyAny>) -> PyResult<String> {
    let bound = result.bind(py);
    if let Ok(s) = bound.extract::<String>() {
        return Ok(s);
    }
    let len: usize = bound
        .call_method0("__len__")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(0);
    if len == 0 {
        return Ok("No results.".to_string());
    }
    let header = if len > 15 {
        format!("{len} row(s) (showing first 15):\n")
    } else {
        format!("{len} row(s):\n")
    };
    let mut out = header;
    for (i, row) in bound.try_iter()?.enumerate() {
        if i >= 15 {
            break;
        }
        out.push_str(&row?.repr()?.to_string());
        out.push('\n');
    }
    Ok(out)
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct CypherArgs {
    /// Cypher query string. Append `FORMAT CSV` for CSV-encoded output.
    pub query: String,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct OverviewArgs {
    /// Drill into specific node types (e.g. `["Person", "Document"]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub types: Option<Vec<String>>,
    /// `true` for all connection types; or `["CALLS"]` for a deep-dive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connections: Option<serde_json::Value>,
    /// `true` for the Cypher language reference; or `["MATCH","WHERE"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cypher: Option<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct SaveGraphArgs {}

/// Register cypher_query / graph_overview / save_graph on the supplied
/// server. State is shared by Arc, so a graph swap on `state` lands
/// instantly on the next tool call.
pub fn register(server: &mut McpServer, state: GraphState) {
    let s = state.clone();
    server.register_typed_tool::<CypherArgs, _>(
        "cypher_query",
        "Run a Cypher query against the active knowledge graph. Returns up to 15 rows \
         inline; append FORMAT CSV to export full results to a CSV string.",
        move |args| s.with_active(|g| run_cypher(g, &args.query)),
    );
    let s = state.clone();
    server.register_typed_tool::<OverviewArgs, _>(
        "graph_overview",
        "Inspect the active graph's schema. With no args returns the inventory; pass \
         types=[...] / connections=true|[...] / cypher=true|[...] for drill-down.",
        move |args| s.with_active(|g| run_overview(g, &args)),
    );
    let s = state;
    server.register_typed_tool::<SaveGraphArgs, _>(
        "save_graph",
        "Persist the active graph to its source .kgl file (single-graph mode only).",
        move |_| s.with_active(run_save),
    );
}

fn run_cypher(graph: &ActiveGraph, query: &str) -> String {
    Python::attach(|py| -> PyResult<String> {
        let result = graph.py_obj.call_method1(py, "cypher", (query,))?;
        format_cypher_result(py, &result)
    })
    .unwrap_or_else(|e| format!("Cypher error: {e}"))
}

fn run_overview(graph: &ActiveGraph, args: &OverviewArgs) -> String {
    Python::attach(|py| -> PyResult<String> {
        let kwargs = PyDict::new(py);
        if let Some(types) = &args.types {
            kwargs.set_item("types", types)?;
        }
        if let Some(c) = &args.connections {
            kwargs.set_item("connections", json_to_py(py, c)?)?;
        }
        if let Some(c) = &args.cypher {
            kwargs.set_item("cypher", json_to_py(py, c)?)?;
        }
        let result = graph
            .py_obj
            .call_method(py, "describe", PyTuple::empty(py), Some(&kwargs))?;
        result.extract::<String>(py)
    })
    .unwrap_or_else(|e| format!("graph_overview error: {e}"))
}

fn run_save(graph: &ActiveGraph) -> String {
    let Some(path) = graph.source_path.as_ref() else {
        return "save_graph requires --graph mode (no source path bound).".to_string();
    };
    let path_str = path.to_string_lossy().into_owned();
    Python::attach(|py| -> PyResult<String> {
        graph
            .py_obj
            .call_method1(py, "save", (path_str.as_str(),))?;
        let s = graph.py_obj.call_method0(py, "schema")?;
        let s = s.bind(py);
        let nodes: u64 = s.get_item("node_count")?.extract()?;
        let edges: u64 = s.get_item("edge_count")?.extract()?;
        Ok(format!("Saved {path_str} ({nodes} nodes, {edges} edges)."))
    })
    .unwrap_or_else(|e| format!("save_graph error: {e}"))
}
