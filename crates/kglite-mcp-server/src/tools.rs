//! KGLite-specific MCP tools: `cypher_query`, `graph_overview`, `save_graph`.
//!
//! All three close over a [`GraphState`] holding the active
//! [`kglite::api::KnowledgeGraph`] behind an `Arc<RwLock<…>>`. Wired
//! into the framework's tool router via `register_typed_tool` so they
//! sit alongside the built-in source / GitHub tools.
//!
//! 0.9.18: rewritten against the pure-Rust `kglite::api` surface.
//! There is no `Python::attach` anywhere in this module — the binary
//! has no libpython link at all.

use std::path::Path;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use kglite::api::cypher;
use kglite::api::{
    compute_description, compute_schema, load_file, ConnectionDetail, CypherDetail, Embedder,
    FluentDetail, KnowledgeGraph, Value,
};
use mcp_methods::server::McpServer;
use serde::{Deserialize, Serialize};

const NO_GRAPH: &str =
    "No active graph. Pass --graph X.kgl, or activate one via repo_management('org/repo').";

/// Shared active-graph state. Cloning is cheap (Arc).
#[derive(Clone, Default)]
pub struct GraphState {
    inner: Arc<RwLock<Option<ActiveGraph>>>,
}

struct ActiveGraph {
    kg: KnowledgeGraph,
    source_path: Option<std::path::PathBuf>,
}

impl GraphState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_kgl(&self, path: &Path) -> Result<()> {
        let kg = load_file(&path.to_string_lossy())
            .map_err(|e| anyhow::anyhow!("kglite::load_file failed: {}", e))?;
        *self.inner.write().unwrap() = Some(ActiveGraph {
            kg,
            source_path: Some(path.to_path_buf()),
        });
        Ok(())
    }

    pub fn build_code_tree(&self, dir: &Path) -> Result<()> {
        let kg = kglite::api::build_code_tree(dir, false, true, None, None)
            .map_err(|e| anyhow::anyhow!("kglite::build_code_tree failed: {}", e))?;
        *self.inner.write().unwrap() = Some(ActiveGraph {
            kg,
            source_path: None,
        });
        Ok(())
    }

    pub fn bind_embedder(&self, embedder: Arc<dyn Embedder>) -> Result<()> {
        let mut guard = self.inner.write().unwrap();
        let Some(active) = guard.as_mut() else {
            tracing::warn!("embedder loaded before any graph is active; binding deferred");
            return Ok(());
        };
        active.kg.set_embedder_native(embedder);
        Ok(())
    }

    pub fn schema(&self) -> Option<(u64, u64)> {
        let guard = self.inner.read().unwrap();
        let active = guard.as_ref()?;
        let overview = compute_schema(active.kg.dir());
        Some((overview.node_count as u64, overview.edge_count as u64))
    }

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
    /// `KnowledgeGraph::source_location`. Used by the `read_code_source`
    /// tool to bridge the qualified-name → file path lookup.
    pub fn source_lookup(
        &self,
        qualified_name: &str,
        node_type: Option<&str>,
    ) -> Result<crate::code_source::SourceLookup, String> {
        let guard = self.inner.read().unwrap();
        let Some(active) = guard.as_ref() else {
            return Err(NO_GRAPH.to_string());
        };
        match active.kg.source_location(qualified_name, node_type) {
            kglite::api::SourceLookup::Found(loc) => {
                let file_path = loc.file_path.ok_or_else(|| {
                    format!("graph.source({qualified_name:?}) returned no file_path")
                })?;
                let line_number = loc.line_number.unwrap_or(1).max(1) as usize;
                let end_line = loc.end_line.unwrap_or(loc.line_number.unwrap_or(1)).max(1) as usize;
                Ok(crate::code_source::SourceLookup {
                    file_path,
                    line_number,
                    end_line,
                })
            }
            kglite::api::SourceLookup::Ambiguous(matches) => Err(format!(
                "ambiguous qualified_name {qualified_name:?}; matches: {matches:?}. \
                 Pass `node_type` to narrow."
            )),
            kglite::api::SourceLookup::NotFound => Err(format!(
                "graph.source({qualified_name:?}) returned no match. \
                 Try passing `node_type` or using a different qualified name."
            )),
        }
    }

    /// Run a parameterised Cypher template against the active graph.
    /// Used by the YAML-declared `tools[].cypher` registration path
    /// (see [`crate::cypher_tools::register_cypher_tools`]).
    pub fn run_cypher_template(
        &self,
        template: &str,
        args: &serde_json::Map<String, serde_json::Value>,
    ) -> String {
        let guard = self.inner.read().unwrap();
        let Some(active) = guard.as_ref() else {
            return NO_GRAPH.to_string();
        };
        let mut params = std::collections::HashMap::new();
        for (k, v) in args {
            params.insert(k.clone(), json_to_value(v));
        }
        match run_cypher_inner(&active.kg, template, params) {
            Ok(body) => body,
            Err(e) => format!("Cypher error: {e}"),
        }
    }
}

/// Convert a `serde_json::Value` into a Cypher param `Value`. Mirrors
/// the Python boundary's `py_value_to_value` for the JSON subset.
fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int64(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float64(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        // Arrays and objects flow through as JSON-serialised strings; the
        // Cypher engine doesn't have a first-class list/map Value variant
        // at the param boundary, so this matches the existing behaviour
        // for non-scalar JSON inputs from MCP tool calls.
        other => Value::String(other.to_string()),
    }
}

/// Run a Cypher query against the given KnowledgeGraph snapshot. Picks
/// between read and write paths based on `is_mutation_query`; on success
/// returns the rendered tool body (CSV when `FORMAT CSV` is in the
/// query, inline 15-row preview otherwise).
fn run_cypher_inner(
    kg: &KnowledgeGraph,
    query: &str,
    params: std::collections::HashMap<String, Value>,
) -> Result<String, String> {
    let mut parsed =
        cypher::parse_cypher(query).map_err(|e| format!("Cypher syntax error: {e}"))?;
    let mut params = params;

    let rewrite = cypher::rewrite_text_score(&mut parsed, &params)?;
    if !rewrite.texts_to_embed.is_empty() && !parsed.explain {
        let model = kg
            .embedder()
            .ok_or_else(|| {
                "text_score() requires a registered embedding model. \
                 Configure `extensions.embedder:` in the manifest."
                    .to_string()
            })?
            .clone();
        model.load()?;
        let texts: Vec<String> = rewrite
            .texts_to_embed
            .iter()
            .map(|(_, t)| t.clone())
            .collect();
        let embeddings = model.embed(&texts);
        model.unload();
        let embeddings = embeddings?;
        if embeddings.len() != texts.len() {
            return Err(format!(
                "text_score: model.embed() returned {} vectors for {} texts",
                embeddings.len(),
                texts.len()
            ));
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
            params.insert(param_name.clone(), Value::String(json));
        }
    }

    cypher::planner::optimize_with_disabled(
        &mut parsed,
        kg.dir(),
        &params,
        cypher::planner::empty_disabled_set(),
    );
    cypher::mark_lazy_eligibility(&mut parsed);

    let output_csv = parsed.output_format == cypher::OutputFormat::Csv;

    let result = if cypher::is_mutation_query(&parsed) {
        // Mutation requires &mut DirGraph; the shim holds an
        // Arc<DirGraph> snapshot via the KnowledgeGraph, so we cannot
        // mutate it here without a write lock. The shim today only
        // exposes mutating queries when explicitly invoked through
        // `cypher_query`, and the surrounding RwLock<Option<ActiveGraph>>
        // is held read-only across the call. For 0.9.18 we forbid
        // mutations through the MCP tool surface — agents that need to
        // edit the graph use a CLI shell instead.
        return Err(
            "mutation Cypher (CREATE/SET/DELETE/REMOVE/MERGE) is not allowed through \
             the MCP cypher_query tool. Use the kglite CLI for graph edits."
                .to_string(),
        );
    } else {
        let executor = cypher::CypherExecutor::with_params(kg.dir(), &params, None);
        executor
            .execute(&parsed)
            .map_err(|e| format!("Cypher execution error: {e}"))?
    };

    if output_csv {
        Ok(result.to_csv())
    } else {
        Ok(format_cypher_inline(&result))
    }
}

/// Render a CypherResult as an inline 15-row preview (header + repr per
/// row). Matches the format the pre-0.9.18 Python shim produced via
/// `format_cypher_result`.
fn format_cypher_inline(result: &cypher::CypherResult) -> String {
    let len = result.rows.len();
    if len == 0 {
        return "No results.".to_string();
    }
    let header = if len > 15 {
        format!("{len} row(s) (showing first 15):\n")
    } else {
        format!("{len} row(s):\n")
    };
    let mut out = header;
    out.push_str(&result.columns.join("\t"));
    out.push('\n');
    for row in result.rows.iter().take(15) {
        for (i, val) in row.iter().enumerate() {
            if i > 0 {
                out.push('\t');
            }
            push_value_repr(&mut out, val);
        }
        out.push('\n');
    }
    out
}

fn push_value_repr(out: &mut String, val: &Value) {
    use std::fmt::Write;
    match val {
        Value::Null => out.push_str("null"),
        Value::String(s) => {
            let _ = write!(out, "{s:?}");
        }
        Value::Int64(n) => {
            let _ = write!(out, "{n}");
        }
        Value::Float64(f) => {
            let _ = write!(out, "{f}");
        }
        Value::Boolean(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::UniqueId(u) => {
            let _ = write!(out, "{u}");
        }
        Value::DateTime(d) => out.push_str(&d.format("%Y-%m-%d").to_string()),
        Value::Point { lat, lon } => {
            let _ = write!(out, "POINT({lon} {lat})");
        }
        Value::Duration {
            months,
            days,
            seconds,
        } => {
            let _ = write!(out, "duration(M={months}, D={days}, S={seconds})");
        }
        Value::NodeRef(idx) => {
            let _ = write!(out, "node[{idx}]");
        }
    }
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

/// Builtins toggled by the manifest's `builtins:` block.
#[derive(Clone, Copy, Debug, Default)]
pub struct Builtins {
    pub save_graph: bool,
    pub temp_cleanup_on_overview: bool,
}

pub fn register(server: &mut McpServer, state: GraphState, builtins: Builtins) {
    let s = state.clone();
    server.register_typed_tool::<CypherArgs, _>(
        "cypher_query",
        "Run a Cypher query against the active knowledge graph. Returns up to 15 rows \
         inline; append FORMAT CSV to export full results to a CSV string.",
        move |args| s.with_active(|g| run_cypher_tool(g, &args.query)),
    );
    let s = state.clone();
    let cleanup_temp = builtins.temp_cleanup_on_overview;
    server.register_typed_tool::<OverviewArgs, _>(
        "graph_overview",
        "Inspect the active graph's schema. With no args returns the inventory; pass \
         types=[...] / connections=true|[...] / cypher=true|[...] for drill-down.",
        move |args| {
            if cleanup_temp
                && args.types.is_none()
                && args.connections.is_none()
                && args.cypher.is_none()
            {
                wipe_temp_dir();
            }
            s.with_active(|g| run_overview(g, &args))
        },
    );
    if builtins.save_graph {
        let s = state;
        server.register_typed_tool::<SaveGraphArgs, _>(
            "save_graph",
            "Persist the active graph to its source .kgl file (single-graph mode only).",
            move |_| s.with_active(run_save),
        );
    }
}

fn wipe_temp_dir() {
    let dir = std::path::Path::new("temp");
    if !dir.is_dir() {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!(error = %e, "temp_cleanup: read_dir failed");
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let res = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        if let Err(e) = res {
            tracing::debug!(path = %path.display(), error = %e, "temp_cleanup: remove failed");
        }
    }
}

fn run_cypher_tool(graph: &ActiveGraph, query: &str) -> String {
    match run_cypher_inner(&graph.kg, query, std::collections::HashMap::new()) {
        Ok(s) => s,
        Err(e) => format!("Cypher error: {e}"),
    }
}

fn run_overview(graph: &ActiveGraph, args: &OverviewArgs) -> String {
    let conn = parse_connection_detail(args.connections.as_ref());
    let cy = parse_cypher_detail(args.cypher.as_ref());
    let fluent = FluentDetail::Off;
    match compute_description(
        graph.kg.dir(),
        args.types.as_deref(),
        &conn,
        &cy,
        &fluent,
        None,
        None,
        None,
    ) {
        Ok(s) => s,
        Err(e) => format!("graph_overview error: {e}"),
    }
}

fn parse_connection_detail(v: Option<&serde_json::Value>) -> ConnectionDetail {
    use serde_json::Value;
    match v {
        None | Some(Value::Null) => ConnectionDetail::Off,
        Some(Value::Bool(false)) => ConnectionDetail::Off,
        Some(Value::Bool(true)) => ConnectionDetail::Overview,
        Some(Value::Array(items)) => {
            let names: Vec<String> = items
                .iter()
                .filter_map(|i| i.as_str().map(String::from))
                .collect();
            if names.is_empty() {
                ConnectionDetail::Overview
            } else {
                ConnectionDetail::Topics(names)
            }
        }
        Some(_) => ConnectionDetail::Overview,
    }
}

fn parse_cypher_detail(v: Option<&serde_json::Value>) -> CypherDetail {
    use serde_json::Value;
    match v {
        None | Some(Value::Null) => CypherDetail::Off,
        Some(Value::Bool(false)) => CypherDetail::Off,
        Some(Value::Bool(true)) => CypherDetail::Overview,
        Some(Value::Array(items)) => {
            let names: Vec<String> = items
                .iter()
                .filter_map(|i| i.as_str().map(String::from))
                .collect();
            if names.is_empty() {
                CypherDetail::Overview
            } else {
                CypherDetail::Topics(names)
            }
        }
        Some(_) => CypherDetail::Overview,
    }
}

fn run_save(graph: &ActiveGraph) -> String {
    let Some(path) = graph.source_path.as_ref() else {
        return "save_graph requires --graph mode (no source path bound).".to_string();
    };
    let path_str = path.to_string_lossy().into_owned();
    // save_disk needs &mut DirGraph; we hold an Arc<DirGraph>. Clone the
    // Arc and unwrap to get exclusive access (cheap when the Arc is
    // already unique, otherwise we do a deep clone of the storage —
    // acceptable for save since it's an explicit operator action).
    let mut dir_arc = graph.kg.dir().clone();
    let dir = std::sync::Arc::make_mut(&mut dir_arc);
    match dir.save_disk(&path_str) {
        Ok(()) => {
            let overview = compute_schema(dir);
            format!(
                "Saved {path_str} ({} nodes, {} edges).",
                overview.node_count, overview.edge_count
            )
        }
        Err(e) => format!("save_graph error: {e}"),
    }
}
