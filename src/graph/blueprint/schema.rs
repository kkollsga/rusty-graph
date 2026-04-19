//! Serde types for the blueprint JSON schema.
//!
//! See docs/guides/blueprints.md for the user-facing spec. These structs
//! are lenient: unknown fields are allowed and missing fields default to
//! empty where sensible, matching the behaviour of the old Python loader.

use indexmap::IndexMap;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Default)]
pub struct Blueprint {
    #[serde(default)]
    pub settings: Settings,
    /// Node specs, in blueprint-JSON order. Iteration order matters because
    /// the FK-edge phase writes parallel edges on the *first* call per
    /// connection type (then dedupes on subsequent calls). Alphabetical
    /// order would produce different edge counts than the Python loader.
    #[serde(default)]
    pub nodes: IndexMap<String, NodeSpec>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Settings {
    #[serde(default, alias = "root")]
    pub input_root: Option<String>,
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default, alias = "output")]
    pub output_file: Option<String>,
}

impl Settings {
    /// Compute the absolute output path from `output_path` + `output_file`,
    /// falling back to `input_root / output_file`. Returns None if no output
    /// was configured.
    pub fn resolved_output(&self, input_root: &std::path::Path) -> Option<PathBuf> {
        let output_file = self.output_file.as_ref()?;
        let base = match &self.output_path {
            Some(p) => std::path::PathBuf::from(p),
            None => input_root.to_path_buf(),
        };
        Some(base.join(output_file))
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct NodeSpec {
    #[serde(default)]
    pub csv: Option<String>,
    #[serde(default)]
    pub pk: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub parent_fk: Option<String>,
    #[serde(default)]
    pub properties: IndexMap<String, String>,
    #[serde(default)]
    pub skipped: Vec<String>,
    #[serde(default)]
    pub filter: IndexMap<String, serde_json::Value>,
    #[serde(default)]
    pub connections: Connections,
    #[serde(default)]
    pub sub_nodes: IndexMap<String, NodeSpec>,
    #[serde(default)]
    pub timeseries: Option<TimeseriesSpec>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Connections {
    #[serde(default)]
    pub fk_edges: IndexMap<String, FkEdge>,
    #[serde(default)]
    pub junction_edges: IndexMap<String, JunctionEdge>,
}

#[derive(Debug, Deserialize)]
pub struct FkEdge {
    pub target: String,
    pub fk: String,
}

#[derive(Debug, Deserialize)]
pub struct JunctionEdge {
    pub csv: String,
    pub source_fk: String,
    pub target: String,
    pub target_fk: String,
    #[serde(default)]
    pub properties: Vec<String>,
    #[serde(default)]
    pub property_types: IndexMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TimeKey {
    Single(String),
    Composite(IndexMap<String, String>),
}

#[derive(Debug, Deserialize)]
pub struct TimeseriesSpec {
    pub time_key: TimeKey,
    #[serde(default)]
    pub channels: IndexMap<String, String>,
    #[serde(default)]
    pub resolution: Option<String>,
    #[serde(default)]
    pub units: IndexMap<String, String>,
}

/// Load a blueprint from a file path.
pub fn load_blueprint_file(path: &std::path::Path) -> Result<Blueprint, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("Blueprint file not found: {}: {}", path.display(), e))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("Invalid blueprint JSON: {}", e))
}
