// src/graph/schema.rs
use crate::datatypes::values::{FilterCondition, Value};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::NodeIndexable;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Spatial configuration for a node type. Declares which properties hold
/// spatial data (lat/lon pairs, WKT geometries) and enables auto-resolution
/// in Cypher `distance(a, b)` and fluent API methods.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SpatialConfig {
    /// Primary lat/lon location: (lat_field, lon_field). At most one per type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<(String, String)>,
    /// Primary WKT geometry field name. At most one per type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<String>,
    /// Named lat/lon points: name → (lat_field, lon_field). Zero or more.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub points: HashMap<String, (String, String)>,
    /// Named WKT shape fields: name → field_name. Zero or more.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub shapes: HashMap<String, String>,
}

/// Lightweight snapshot of a node's data: id, title, type, and properties.
/// Used as the return type for node queries and traversals.
#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub id: Value,
    pub title: Value,
    pub node_type: String,
    pub properties: HashMap<String, Value>,
}

/// Records a filtering, sorting, or traversal operation applied to a selection.
/// Used by `explain()` to show the query execution plan.
#[derive(Clone, Debug)]
pub enum SelectionOperation {
    Filter(HashMap<String, FilterCondition>),
    Sort(Vec<(String, bool)>), // (field_name, ascending)
    Traverse {
        connection_type: String,
        direction: Option<String>,
        max_nodes: Option<usize>,
    },
    Custom(String), // For operations that don't fit other categories
}

/// A single level in the selection hierarchy — holds node sets grouped
/// by their parent (for traversals) and tracks applied operations.
#[derive(Clone, Debug)]
pub struct SelectionLevel {
    pub selections: HashMap<Option<NodeIndex>, Vec<NodeIndex>>, // parent_idx -> selected_children
    pub operations: Vec<SelectionOperation>,
}

impl SelectionLevel {
    pub fn new() -> Self {
        SelectionLevel {
            selections: HashMap::new(),
            operations: Vec::new(),
        }
    }

    pub fn add_selection(&mut self, parent: Option<NodeIndex>, children: Vec<NodeIndex>) {
        self.selections.insert(parent, children);
    }

    pub fn get_all_nodes(&self) -> Vec<NodeIndex> {
        self.selections
            .values()
            .flat_map(|children| children.iter().copied())
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.selections.is_empty()
    }

    pub fn iter_groups(&self) -> impl Iterator<Item = (&Option<NodeIndex>, &Vec<NodeIndex>)> {
        self.selections.iter()
    }

    /// Returns an iterator over all node indices without allocating a Vec.
    /// Use this instead of get_all_nodes() when you only need to iterate or count.
    pub fn iter_node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.selections
            .values()
            .flat_map(|children| children.iter().copied())
    }

    /// Returns the total count of nodes without allocating a Vec.
    /// More efficient than get_all_nodes().len() for just getting the count.
    pub fn node_count(&self) -> usize {
        self.selections.values().map(|v| v.len()).sum()
    }
}

/// Represents a single step in the query execution plan
#[derive(Clone, Debug)]
pub struct PlanStep {
    pub operation: String,
    pub node_type: Option<String>,
    pub estimated_rows: usize,
    pub actual_rows: Option<usize>,
}

impl PlanStep {
    pub fn new(operation: &str, node_type: Option<&str>, estimated_rows: usize) -> Self {
        PlanStep {
            operation: operation.to_string(),
            node_type: node_type.map(|s| s.to_string()),
            estimated_rows,
            actual_rows: None,
        }
    }

    pub fn with_actual_rows(mut self, actual: usize) -> Self {
        self.actual_rows = Some(actual);
        self
    }
}

/// Tracks the current selection state across a chain of query operations
/// (type_filter → filter → traverse → ...). Supports nested levels for
/// parent-child traversals and records execution plan steps for `explain()`.
#[derive(Clone, Default)]
pub struct CurrentSelection {
    levels: Vec<SelectionLevel>,
    current_level: usize,
    execution_plan: Vec<PlanStep>,
}

impl CurrentSelection {
    pub fn new() -> Self {
        let mut selection = CurrentSelection {
            levels: Vec::new(),
            current_level: 0,
            execution_plan: Vec::new(),
        };
        selection.add_level(); // Always start with an initial level
        selection
    }

    pub fn add_level(&mut self) {
        // No need to pass level index
        self.levels.push(SelectionLevel::new());
        self.current_level = self.levels.len() - 1;
    }

    pub fn clear(&mut self) {
        self.levels.clear();
        self.current_level = 0;
        self.execution_plan.clear();
        self.add_level(); // Ensure we always have at least one level after clearing
    }

    /// Add a step to the execution plan
    pub fn add_plan_step(&mut self, step: PlanStep) {
        self.execution_plan.push(step);
    }

    /// Get the execution plan steps
    pub fn get_execution_plan(&self) -> &[PlanStep] {
        &self.execution_plan
    }

    /// Clear just the execution plan (for fresh queries)
    pub fn clear_execution_plan(&mut self) {
        self.execution_plan.clear();
    }

    pub fn get_level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn get_level(&self, index: usize) -> Option<&SelectionLevel> {
        self.levels.get(index)
    }

    pub fn get_level_mut(&mut self, index: usize) -> Option<&mut SelectionLevel> {
        self.levels.get_mut(index)
    }

    /// Returns the node count for the current (most recent) level without allocation.
    pub fn current_node_count(&self) -> usize {
        self.levels.last().map(|l| l.node_count()).unwrap_or(0)
    }

    /// Returns true if any filtering/selection operation has been applied to the current level.
    /// Used to distinguish "no filter applied" (pristine state) from "filter returned 0 results".
    pub fn has_active_selection(&self) -> bool {
        self.levels
            .last()
            .map(|l| !l.operations.is_empty())
            .unwrap_or(false)
    }

    /// Returns an iterator over node indices in the current (most recent) level.
    pub fn current_node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.levels
            .last()
            .into_iter()
            .flat_map(|l| l.iter_node_indices())
    }

    /// Returns the node type of the first node in the current selection, if any.
    /// Used by spatial auto-resolution to look up SpatialConfig.
    pub fn first_node_type(&self, graph: &DirGraph) -> Option<String> {
        self.current_node_indices()
            .next()
            .and_then(|idx| graph.graph.node_weight(idx))
            .map(|node| node.node_type.clone())
    }
}

/// Copy-on-write wrapper for CurrentSelection.
/// Avoids cloning the selection on every method call when the selection isn't modified.
/// Implements Deref/DerefMut for transparent usage where CurrentSelection is expected.
#[derive(Clone, Default)]
pub struct CowSelection {
    inner: Arc<CurrentSelection>,
}

impl CowSelection {
    pub fn new() -> Self {
        CowSelection {
            inner: Arc::new(CurrentSelection::new()),
        }
    }

    /// Check if we have exclusive ownership (no cloning needed for mutation).
    #[inline]
    #[allow(dead_code)]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }
}

// Implement Deref for transparent read access
impl std::ops::Deref for CowSelection {
    type Target = CurrentSelection;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// Implement DerefMut for copy-on-write mutation
impl std::ops::DerefMut for CowSelection {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.inner)
    }
}

/// Key for single-property indexes: (node_type, property_name)
pub type IndexKey = (String, String);

/// Key for composite indexes: (node_type, property_names)
pub type CompositeIndexKey = (String, Vec<String>);

/// Composite value key: tuple of values for multi-field lookup
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompositeValue(pub Vec<Value>);

/// Metadata stamped into saved files for version tracking and portability warnings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SaveMetadata {
    /// Format version — incremented when DirGraph layout changes.
    /// 0 = files saved before this field existed (via serde default).
    /// 1 = first versioned format.
    pub format_version: u32,
    /// Library version at save time, e.g. "0.4.7".
    pub library_version: String,
}

impl SaveMetadata {
    pub fn current() -> Self {
        SaveMetadata {
            format_version: 2,
            library_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Metadata about a connection type: which node types it connects and what properties it carries.
#[derive(Debug, Clone, Serialize, Default)]
pub struct ConnectionTypeInfo {
    pub source_types: HashSet<String>,
    pub target_types: HashSet<String>,
    /// property_name → type_string (e.g. "weight" → "Float64")
    pub property_types: HashMap<String, String>,
}

/// Custom deserializer to handle both old format (source_type/target_type as single strings)
/// and new format (source_types/target_types as sets).
impl<'de> Deserialize<'de> for ConnectionTypeInfo {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Legacy {
            source_type: Option<String>,
            target_type: Option<String>,
            #[serde(default)]
            source_types: Option<HashSet<String>>,
            #[serde(default)]
            target_types: Option<HashSet<String>>,
            #[serde(default)]
            property_types: HashMap<String, String>,
        }

        let legacy = Legacy::deserialize(deserializer)?;
        let source_types = legacy.source_types.unwrap_or_else(|| {
            legacy
                .source_type
                .map(|s| HashSet::from([s]))
                .unwrap_or_default()
        });
        let target_types = legacy.target_types.unwrap_or_else(|| {
            legacy
                .target_type
                .map(|s| HashSet::from([s]))
                .unwrap_or_default()
        });
        Ok(ConnectionTypeInfo {
            source_types,
            target_types,
            property_types: legacy.property_types,
        })
    }
}

/// Contiguous columnar storage for f32 embeddings associated with a (node_type, property_name).
/// All vectors in one store share the same dimensionality.
/// The flat Vec<f32> layout enables SIMD-friendly linear scans during vector search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingStore {
    /// Embedding dimensionality (e.g. 384, 768, 1536)
    pub dimension: usize,
    /// Contiguous f32 buffer: embedding i occupies data[i*dimension..(i+1)*dimension]
    pub data: Vec<f32>,
    /// Maps NodeIndex.index() -> slot position in the contiguous buffer
    pub node_to_slot: HashMap<usize, usize>,
    /// Reverse map: slot -> NodeIndex.index(), needed for returning results
    pub slot_to_node: Vec<usize>,
}

impl EmbeddingStore {
    pub fn new(dimension: usize) -> Self {
        EmbeddingStore {
            dimension,
            data: Vec::new(),
            node_to_slot: HashMap::new(),
            slot_to_node: Vec::new(),
        }
    }

    /// Add or replace an embedding for a node. Returns the slot index.
    pub fn set_embedding(&mut self, node_index: usize, embedding: &[f32]) -> usize {
        if let Some(&slot) = self.node_to_slot.get(&node_index) {
            // Replace existing embedding in-place
            let start = slot * self.dimension;
            self.data[start..start + self.dimension].copy_from_slice(embedding);
            slot
        } else {
            // Append new embedding
            let slot = self.slot_to_node.len();
            self.node_to_slot.insert(node_index, slot);
            self.slot_to_node.push(node_index);
            self.data.extend_from_slice(embedding);
            slot
        }
    }

    /// Get the embedding slice for a node (by NodeIndex.index()).
    #[inline]
    pub fn get_embedding(&self, node_index: usize) -> Option<&[f32]> {
        self.node_to_slot.get(&node_index).map(|&slot| {
            let start = slot * self.dimension;
            &self.data[start..start + self.dimension]
        })
    }

    /// Number of stored embeddings.
    #[inline]
    pub fn len(&self) -> usize {
        self.slot_to_node.len()
    }
}

/// Core graph storage: a directed graph (petgraph `StableDiGraph`) with fast
/// type-based indexing and optional property/composite/range/spatial indexes.
///
/// Fields include `type_indices` for O(1) node-type lookup, `property_indices`
/// for indexed equality filters, connection-type metadata, schema definitions,
/// and optional embedding stores for vector similarity search.
#[derive(Clone, Serialize, Deserialize)]
pub struct DirGraph {
    pub(crate) graph: Graph,
    /// Skipped during serialization — rebuilt from graph on load via `rebuild_type_indices()`.
    #[serde(skip)]
    pub(crate) type_indices: HashMap<String, Vec<NodeIndex>>,
    /// Optional schema definition for validation
    #[serde(default)]
    pub(crate) schema_definition: Option<SchemaDefinition>,
    /// Single-property indexes for fast lookups: (node_type, property) -> value -> [node_indices]
    /// Skipped during serialization — rebuilt from `property_index_keys` on load.
    #[serde(skip)]
    pub(crate) property_indices: HashMap<IndexKey, HashMap<Value, Vec<NodeIndex>>>,
    /// Composite indexes for multi-field queries: (node_type, [properties]) -> composite_value -> [node_indices]
    /// Skipped during serialization — rebuilt from `composite_index_keys` on load.
    #[serde(skip)]
    pub(crate) composite_indices:
        HashMap<CompositeIndexKey, HashMap<CompositeValue, Vec<NodeIndex>>>,
    /// Persisted list of property index keys so indexes can be rebuilt on load
    #[serde(default)]
    pub(crate) property_index_keys: Vec<IndexKey>,
    /// Persisted list of composite index keys so indexes can be rebuilt on load
    #[serde(default)]
    pub(crate) composite_index_keys: Vec<CompositeIndexKey>,
    /// B-Tree range indexes for ordered lookups: (node_type, property) -> BTreeMap<Value, [NodeIndex]>
    /// Skipped during serialization — rebuilt from `range_index_keys` on load.
    #[serde(skip)]
    pub(crate) range_indices: HashMap<IndexKey, std::collections::BTreeMap<Value, Vec<NodeIndex>>>,
    /// Persisted list of range index keys so indexes can be rebuilt on load
    #[serde(default)]
    pub(crate) range_index_keys: Vec<IndexKey>,
    /// Fast O(1) lookup by node ID: node_type -> (id_value -> NodeIndex)
    /// Lazily built on first use for each node type, skipped during serialization
    #[serde(skip)]
    pub(crate) id_indices: HashMap<String, HashMap<Value, NodeIndex>>,
    /// Fast O(1) lookup for connection types. Populated on first edge access.
    #[serde(skip)]
    pub(crate) connection_types: std::collections::HashSet<String>,
    /// Node type metadata: node_type → { property_name → type_string }
    /// Replaces SchemaNode graph nodes — persisted via serde/bincode.
    #[serde(default)]
    pub(crate) node_type_metadata: HashMap<String, HashMap<String, String>>,
    /// Connection type metadata: connection_type → ConnectionTypeInfo
    /// Replaces SchemaNode graph nodes for connections — persisted via serde/bincode.
    #[serde(default)]
    pub(crate) connection_type_metadata: HashMap<String, ConnectionTypeInfo>,
    /// Version and library info stamped at save time.
    /// Old files without this field deserialize to SaveMetadata::default() (format_version=0).
    #[serde(default)]
    pub(crate) save_metadata: SaveMetadata,
    /// Original ID field name per node type (e.g. "Person" → "npdid").
    /// Stored when the user-supplied unique_id_field differs from "id".
    /// Used for alias resolution: querying by original column name maps to the `id` field.
    #[serde(default)]
    pub(crate) id_field_aliases: HashMap<String, String>,
    /// Original title field name per node type (e.g. "Person" → "prospect_name").
    /// Stored when the user-supplied node_title_field differs from "title".
    /// Used for alias resolution: querying by original column name maps to the `title` field.
    #[serde(default)]
    pub(crate) title_field_aliases: HashMap<String, String>,
    /// Auto-vacuum threshold: if Some(t), vacuum() is triggered automatically after
    /// DELETE operations when fragmentation_ratio exceeds t and tombstones > 100.
    /// Default: Some(0.3). Set to None to disable.
    #[serde(default = "default_auto_vacuum_threshold")]
    pub(crate) auto_vacuum_threshold: Option<f64>,
    /// Spatial configuration per node type: type_name → SpatialConfig.
    /// Declares which properties hold lat/lon or WKT data for auto-resolution.
    #[serde(default)]
    pub(crate) spatial_configs: HashMap<String, SpatialConfig>,
    /// Graph-level WKT geometry cache — persists across queries.
    /// Uses Arc<Geometry> to avoid cloning heavy geometry objects.
    /// RwLock allows concurrent reads from parallel row evaluation.
    #[serde(skip)]
    pub(crate) wkt_cache: Arc<RwLock<HashMap<String, Arc<geo::Geometry<f64>>>>>,
    /// Lazy edge-type count cache — avoids O(E) rescan for FusedCountEdgesByType.
    /// Invalidated on edge mutations (add/remove).
    #[serde(skip)]
    pub(crate) edge_type_counts_cache: Arc<RwLock<Option<HashMap<String, usize>>>>,
    /// Columnar embedding storage: (node_type, property_name) -> EmbeddingStore.
    /// Stored separately from NodeData.properties — invisible to normal node API.
    /// Persisted as a separate section in v2 .kgl files.
    #[serde(skip)]
    pub(crate) embeddings: HashMap<(String, String), EmbeddingStore>,
    /// Timeseries configuration per node type: type_name → TimeseriesConfig.
    /// Declares composite key labels and known channels for auto-resolution.
    #[serde(default)]
    pub(crate) timeseries_configs: HashMap<String, super::timeseries::TimeseriesConfig>,
    /// Per-node timeseries storage: NodeIndex.index() → NodeTimeseries.
    /// Stored separately from NodeData.properties (like embeddings).
    /// Persisted as a separate section in v2 .kgl files.
    #[serde(skip)]
    pub(crate) timeseries_store: HashMap<usize, super::timeseries::NodeTimeseries>,
    /// If true, Cypher mutations (CREATE, SET, DELETE, REMOVE, MERGE) are rejected
    /// and agent_describe() omits mutation documentation.
    #[serde(skip)]
    pub(crate) read_only: bool,
}

fn default_auto_vacuum_threshold() -> Option<f64> {
    Some(0.3)
}

impl DirGraph {
    pub fn new() -> Self {
        DirGraph {
            graph: Graph::new(),
            type_indices: HashMap::new(),
            schema_definition: None,
            property_indices: HashMap::new(),
            composite_indices: HashMap::new(),
            property_index_keys: Vec::new(),
            composite_index_keys: Vec::new(),
            range_indices: HashMap::new(),
            range_index_keys: Vec::new(),
            id_indices: HashMap::new(),
            connection_types: std::collections::HashSet::new(),
            node_type_metadata: HashMap::new(),
            connection_type_metadata: HashMap::new(),
            save_metadata: SaveMetadata::current(),
            id_field_aliases: HashMap::new(),
            title_field_aliases: HashMap::new(),
            auto_vacuum_threshold: default_auto_vacuum_threshold(),
            spatial_configs: HashMap::new(),
            wkt_cache: Arc::new(RwLock::new(HashMap::new())),
            edge_type_counts_cache: Arc::new(RwLock::new(None)),
            embeddings: HashMap::new(),
            timeseries_configs: HashMap::new(),
            timeseries_store: HashMap::new(),
            read_only: false,
        }
    }

    /// Create a DirGraph from a pre-existing graph (used by v2 loader).
    /// All metadata fields start empty and are populated by the caller.
    pub fn from_graph(graph: Graph) -> Self {
        DirGraph {
            graph,
            type_indices: HashMap::new(),
            schema_definition: None,
            property_indices: HashMap::new(),
            composite_indices: HashMap::new(),
            property_index_keys: Vec::new(),
            composite_index_keys: Vec::new(),
            range_indices: HashMap::new(),
            range_index_keys: Vec::new(),
            id_indices: HashMap::new(),
            connection_types: std::collections::HashSet::new(),
            node_type_metadata: HashMap::new(),
            connection_type_metadata: HashMap::new(),
            save_metadata: SaveMetadata::default(),
            id_field_aliases: HashMap::new(),
            title_field_aliases: HashMap::new(),
            auto_vacuum_threshold: default_auto_vacuum_threshold(),
            spatial_configs: HashMap::new(),
            wkt_cache: Arc::new(RwLock::new(HashMap::new())),
            edge_type_counts_cache: Arc::new(RwLock::new(None)),
            embeddings: HashMap::new(),
            timeseries_configs: HashMap::new(),
            timeseries_store: HashMap::new(),
            read_only: false,
        }
    }

    /// Look up spatial config for a node type.
    pub fn get_spatial_config(&self, node_type: &str) -> Option<&SpatialConfig> {
        self.spatial_configs.get(node_type)
    }

    /// Look up timeseries data for a specific node by its index.
    pub fn get_node_timeseries(
        &self,
        node_index: usize,
    ) -> Option<&super::timeseries::NodeTimeseries> {
        self.timeseries_store.get(&node_index)
    }

    /// Look up an embedding store by `(&str, &str)` without allocating owned Strings.
    /// Falls back to a linear scan of the embeddings map (typically 1-3 entries).
    #[inline]
    pub fn embedding_store(&self, node_type: &str, prop_name: &str) -> Option<&EmbeddingStore> {
        // Embedding maps are tiny (usually 1-5 entries), so linear scan beats allocation
        self.embeddings
            .iter()
            .find(|((nt, pn), _)| nt == node_type && pn == prop_name)
            .map(|(_, store)| store)
    }

    /// Build the ID index for a specific node type.
    /// Called lazily on first lookup for that type.
    pub fn build_id_index(&mut self, node_type: &str) {
        if self.id_indices.contains_key(node_type) {
            return; // Already built
        }

        let mut index = HashMap::new();

        if let Some(node_indices) = self.type_indices.get(node_type) {
            for &node_idx in node_indices {
                if let Some(node) = self.graph.node_weight(node_idx) {
                    index.insert(node.id.clone(), node_idx);
                }
            }
        }

        self.id_indices.insert(node_type.to_string(), index);
    }

    /// Look up a node by type and ID value. O(1) after index is built.
    /// Builds the index lazily if not already built.
    /// Handles type normalization: Python int may come as Int64 but be stored as UniqueId.
    pub fn lookup_by_id(&mut self, node_type: &str, id: &Value) -> Option<NodeIndex> {
        // Build index if needed
        if !self.id_indices.contains_key(node_type) {
            self.build_id_index(node_type);
        }

        self.lookup_by_id_normalized(node_type, id)
    }

    /// Look up a node by type and ID value without building index.
    /// Use this for read-only access when index already exists.
    /// Handles type normalization for integer types.
    #[allow(dead_code)]
    pub fn lookup_by_id_readonly(&self, node_type: &str, id: &Value) -> Option<NodeIndex> {
        self.lookup_by_id_normalized(node_type, id)
    }

    /// Lookup node by ID with automatic type normalization.
    /// This handles the Python-Rust type mismatch where Python int -> Int64 but
    /// DataFrame unique_id columns store as UniqueId(u32).
    ///
    /// Falls back to a linear scan of type_indices if the id_index hasn't been
    /// built yet (e.g., after DELETE invalidates id_indices).
    pub fn lookup_by_id_normalized(&self, node_type: &str, id: &Value) -> Option<NodeIndex> {
        if let Some(type_index) = self.id_indices.get(node_type) {
            // Try direct lookup first
            if let Some(&idx) = type_index.get(id) {
                return Some(idx);
            }

            // If direct lookup fails, try alternative integer representations
            let result = match id {
                Value::Int64(i) => {
                    if *i >= 0 && *i <= u32::MAX as i64 {
                        type_index.get(&Value::UniqueId(*i as u32)).copied()
                    } else {
                        None
                    }
                }
                Value::UniqueId(u) => type_index.get(&Value::Int64(*u as i64)).copied(),
                Value::Float64(f) => {
                    if f.fract() == 0.0 {
                        let i = *f as i64;
                        if let Some(&idx) = type_index.get(&Value::Int64(i)) {
                            return Some(idx);
                        }
                        if i >= 0 && i <= u32::MAX as i64 {
                            return type_index.get(&Value::UniqueId(i as u32)).copied();
                        }
                    }
                    None
                }
                _ => None,
            };
            if result.is_some() {
                return result;
            }
        }

        // Fallback: linear scan through type_indices when id_index is missing
        // (e.g., after DELETE invalidates id_indices for this type)
        if let Some(node_indices) = self.type_indices.get(node_type) {
            for &node_idx in node_indices {
                if let Some(node) = self.graph.node_weight(node_idx) {
                    let node_id = &node.id;
                    if node_id == id {
                        return Some(node_idx);
                    }
                    // Normalize: check Int64 ↔ UniqueId
                    match (id, node_id) {
                        (Value::Int64(i), Value::UniqueId(u)) => {
                            if *i >= 0 && *i as u32 == *u {
                                return Some(node_idx);
                            }
                        }
                        (Value::UniqueId(u), Value::Int64(i)) => {
                            if *i >= 0 && *u == *i as u32 {
                                return Some(node_idx);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }

    /// Invalidate the ID index for a node type (call when nodes are added/removed)
    #[allow(dead_code)]
    pub fn invalidate_id_index(&mut self, node_type: &str) {
        self.id_indices.remove(node_type);
    }

    /// Clear all ID indices (call after bulk operations)
    #[allow(dead_code)]
    pub fn clear_id_indices(&mut self) {
        self.id_indices.clear();
    }

    /// Set the schema definition for this graph
    pub fn set_schema(&mut self, schema: SchemaDefinition) {
        self.schema_definition = Some(schema);
    }

    /// Get the schema definition if one is set
    pub fn get_schema(&self) -> Option<&SchemaDefinition> {
        self.schema_definition.as_ref()
    }

    /// Clear the schema definition
    pub fn clear_schema(&mut self) {
        self.schema_definition = None;
    }

    pub fn has_connection_type(&self, connection_type: &str) -> bool {
        // Fast path: check the connection_types cache (O(1))
        if !self.connection_types.is_empty() {
            return self.connection_types.contains(connection_type);
        }
        // Fallback: check metadata
        self.connection_type_metadata.contains_key(connection_type)
    }

    /// Register a connection type for O(1) lookups.
    /// Called when edges are added to the graph.
    pub fn register_connection_type(&mut self, connection_type: String) {
        self.connection_types.insert(connection_type);
    }

    /// Build the connection types cache.
    /// Called after deserialization or when cache is needed.
    /// Fast path: populate from connection_type_metadata (O(types), no edge scan).
    /// Fallback: scan all edges (O(edges)) if metadata is empty.
    pub fn build_connection_types_cache(&mut self) {
        if !self.connection_types.is_empty() {
            return; // Already built
        }

        // Fast path: metadata is serialized — use it instead of scanning edges
        if !self.connection_type_metadata.is_empty() {
            for key in self.connection_type_metadata.keys() {
                self.connection_types.insert(key.clone());
            }
            return;
        }

        // Fallback: scan all edges (pre-metadata graphs)
        for edge in self.graph.edge_weights() {
            self.connection_types.insert(edge.connection_type.clone());
        }
    }

    /// Compute edge counts grouped by connection type. Lazily cached.
    pub fn get_edge_type_counts(&self) -> HashMap<String, usize> {
        // Fast path: return cached result
        {
            let read = self.edge_type_counts_cache.read().unwrap();
            if let Some(ref cached) = *read {
                return cached.clone();
            }
        }
        // Slow path: compute O(E) and cache
        let mut counts: HashMap<String, usize> = HashMap::new();
        for edge in self.graph.edge_weights() {
            *counts.entry(edge.connection_type.clone()).or_insert(0) += 1;
        }
        let mut write = self.edge_type_counts_cache.write().unwrap();
        *write = Some(counts.clone());
        counts
    }

    /// Invalidate the edge type count cache (call after edge mutations).
    pub(crate) fn invalidate_edge_type_counts_cache(&self) {
        *self.edge_type_counts_cache.write().unwrap() = None;
    }

    // ========================================================================
    // Type Metadata Methods (replaces SchemaNode graph nodes)
    // ========================================================================

    /// Get metadata for a node type (property names → type strings).
    pub fn get_node_type_metadata(&self, node_type: &str) -> Option<&HashMap<String, String>> {
        self.node_type_metadata.get(node_type)
    }

    /// Get metadata for a connection type.
    #[allow(dead_code)]
    pub fn get_connection_type_info(&self, conn_type: &str) -> Option<&ConnectionTypeInfo> {
        self.connection_type_metadata.get(conn_type)
    }

    /// Upsert node type metadata — merges new property types into existing.
    pub fn upsert_node_type_metadata(&mut self, node_type: &str, props: HashMap<String, String>) {
        let entry = self
            .node_type_metadata
            .entry(node_type.to_string())
            .or_default();
        for (k, v) in props {
            entry.insert(k, v);
        }
    }

    /// Upsert connection type metadata — merges property types and accumulates type pairs.
    pub fn upsert_connection_type_metadata(
        &mut self,
        conn_type: &str,
        source_type: &str,
        target_type: &str,
        prop_types: HashMap<String, String>,
    ) {
        let entry = self
            .connection_type_metadata
            .entry(conn_type.to_string())
            .or_insert_with(|| ConnectionTypeInfo {
                source_types: HashSet::new(),
                target_types: HashSet::new(),
                property_types: HashMap::new(),
            });
        entry.source_types.insert(source_type.to_string());
        entry.target_types.insert(target_type.to_string());
        for (k, v) in prop_types {
            entry.property_types.insert(k, v);
        }
    }

    pub fn has_node_type(&self, node_type: &str) -> bool {
        self.type_indices.contains_key(node_type) || self.node_type_metadata.contains_key(node_type)
    }

    /// Get all node types that exist in the graph.
    pub fn get_node_types(&self) -> Vec<String> {
        let mut types: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Get types from type_indices
        for node_type in self.type_indices.keys() {
            types.insert(node_type.clone());
        }

        // Also include types from metadata (may have metadata but no live nodes)
        for node_type in self.node_type_metadata.keys() {
            types.insert(node_type.clone());
        }

        types.into_iter().collect()
    }

    /// Resolve a property name through field aliases.
    /// If the property matches the original ID or title field name for this node type,
    /// returns the canonical name ("id" or "title"). Otherwise returns the property unchanged.
    pub fn resolve_alias<'a>(&'a self, node_type: &str, property: &'a str) -> &'a str {
        if self.id_field_aliases.is_empty() && self.title_field_aliases.is_empty() {
            return property;
        }
        if let Some(alias) = self.id_field_aliases.get(node_type) {
            if alias == property {
                return "id";
            }
        }
        if let Some(alias) = self.title_field_aliases.get(node_type) {
            if alias == property {
                return "title";
            }
        }
        property
    }

    pub fn get_node(&self, index: NodeIndex) -> Option<&NodeData> {
        self.graph.node_weight(index)
    }

    pub fn get_node_mut(&mut self, index: NodeIndex) -> Option<&mut NodeData> {
        self.graph.node_weight_mut(index)
    }

    pub fn _get_connection(&self, index: EdgeIndex) -> Option<&EdgeData> {
        self.graph.edge_weight(index)
    }

    pub fn _get_connection_mut(&mut self, index: EdgeIndex) -> Option<&mut EdgeData> {
        self.graph.edge_weight_mut(index)
    }

    // ========================================================================
    // Index Management Methods
    // ========================================================================

    /// Create an index on a property for a specific node type.
    /// Returns the number of entries indexed.
    pub fn create_index(&mut self, node_type: &str, property: &str) -> usize {
        let key = (node_type.to_string(), property.to_string());

        // Build the index
        let mut index: HashMap<Value, Vec<NodeIndex>> = HashMap::new();

        if let Some(node_indices) = self.type_indices.get(node_type) {
            for &idx in node_indices {
                if let Some(node) = self.graph.node_weight(idx) {
                    if let Some(value) = node.properties.get(property) {
                        index.entry(value.clone()).or_default().push(idx);
                    }
                }
            }
        }

        let count = index.len();
        self.property_indices.insert(key, index);
        count
    }

    /// Drop an index on a property for a specific node type.
    /// Returns true if the index existed and was removed.
    pub fn drop_index(&mut self, node_type: &str, property: &str) -> bool {
        let key = (node_type.to_string(), property.to_string());
        self.property_indices.remove(&key).is_some()
    }

    /// Check if an index exists for a given node type and property.
    pub fn has_index(&self, node_type: &str, property: &str) -> bool {
        let key = (node_type.to_string(), property.to_string());
        self.property_indices.contains_key(&key)
    }

    /// Get all existing indexes as a list of (node_type, property) tuples.
    pub fn list_indexes(&self) -> Vec<(String, String)> {
        self.property_indices.keys().cloned().collect()
    }

    /// Look up nodes by property value using an index.
    /// Returns None if no index exists, otherwise returns matching node indices.
    pub fn lookup_by_index(
        &self,
        node_type: &str,
        property: &str,
        value: &Value,
    ) -> Option<Vec<NodeIndex>> {
        let key = (node_type.to_string(), property.to_string());
        self.property_indices
            .get(&key)
            .and_then(|idx| idx.get(value))
            .cloned()
    }

    /// Get statistics about an index.
    pub fn get_index_stats(&self, node_type: &str, property: &str) -> Option<IndexStats> {
        let key = (node_type.to_string(), property.to_string());
        self.property_indices.get(&key).map(|idx| {
            let total_entries: usize = idx.values().map(|v| v.len()).sum();
            IndexStats {
                unique_values: idx.len(),
                total_entries,
                avg_entries_per_value: if idx.is_empty() {
                    0.0
                } else {
                    total_entries as f64 / idx.len() as f64
                },
            }
        })
    }

    // ========================================================================
    // Range Index Methods (B-Tree)
    // ========================================================================

    /// Create a range index (B-Tree) on a property for a specific node type.
    /// Enables efficient range queries (>, >=, <, <=, BETWEEN).
    /// Returns the number of unique values indexed.
    pub fn create_range_index(&mut self, node_type: &str, property: &str) -> usize {
        let key = (node_type.to_string(), property.to_string());
        let mut index: std::collections::BTreeMap<Value, Vec<NodeIndex>> =
            std::collections::BTreeMap::new();

        if let Some(node_indices) = self.type_indices.get(node_type) {
            for &idx in node_indices {
                if let Some(node) = self.graph.node_weight(idx) {
                    if let Some(value) = node.properties.get(property) {
                        index.entry(value.clone()).or_default().push(idx);
                    }
                }
            }
        }

        let count = index.len();
        self.range_indices.insert(key, index);
        count
    }

    /// Drop a range index. Returns true if it existed.
    pub fn drop_range_index(&mut self, node_type: &str, property: &str) -> bool {
        let key = (node_type.to_string(), property.to_string());
        self.range_indices.remove(&key).is_some()
    }

    /// Check if a range index exists.
    #[allow(dead_code)]
    pub fn has_range_index(&self, node_type: &str, property: &str) -> bool {
        let key = (node_type.to_string(), property.to_string());
        self.range_indices.contains_key(&key)
    }

    /// Range lookup: returns node indices where property value falls in the given range.
    pub fn lookup_range(
        &self,
        node_type: &str,
        property: &str,
        lower: std::ops::Bound<&Value>,
        upper: std::ops::Bound<&Value>,
    ) -> Option<Vec<NodeIndex>> {
        let key = (node_type.to_string(), property.to_string());
        self.range_indices.get(&key).map(|btree| {
            btree
                .range((lower, upper))
                .flat_map(|(_, indices)| indices.iter().copied())
                .collect()
        })
    }

    // ========================================================================
    // Composite Index Methods
    // ========================================================================

    /// Create a composite index on multiple properties for a specific node type.
    /// Composite indexes enable efficient lookups on multiple fields at once.
    ///
    /// Returns the number of unique value combinations indexed.
    ///
    /// Example: create_composite_index("Person", &["city", "age"]) allows efficient
    /// queries like filter({'city': 'Oslo', 'age': 30}).
    pub fn create_composite_index(&mut self, node_type: &str, properties: &[&str]) -> usize {
        let key = (
            node_type.to_string(),
            properties.iter().map(|s| s.to_string()).collect(),
        );

        // Build the composite index
        let mut index: HashMap<CompositeValue, Vec<NodeIndex>> = HashMap::new();

        if let Some(node_indices) = self.type_indices.get(node_type) {
            for &idx in node_indices {
                if let Some(node) = self.graph.node_weight(idx) {
                    // Extract values for all properties in order
                    let values: Vec<Value> = properties
                        .iter()
                        .map(|p| node.properties.get(*p).cloned().unwrap_or(Value::Null))
                        .collect();

                    // Only index if at least one value is non-null
                    if values.iter().any(|v| !matches!(v, Value::Null)) {
                        index.entry(CompositeValue(values)).or_default().push(idx);
                    }
                }
            }
        }

        let count = index.len();
        self.composite_indices.insert(key, index);
        count
    }

    /// Drop a composite index.
    /// Returns true if the index existed and was removed.
    pub fn drop_composite_index(&mut self, node_type: &str, properties: &[String]) -> bool {
        let key = (node_type.to_string(), properties.to_vec());
        self.composite_indices.remove(&key).is_some()
    }

    /// Check if a composite index exists.
    pub fn has_composite_index(&self, node_type: &str, properties: &[String]) -> bool {
        let key = (node_type.to_string(), properties.to_vec());
        self.composite_indices.contains_key(&key)
    }

    /// Get all existing composite indexes.
    pub fn list_composite_indexes(&self) -> Vec<(String, Vec<String>)> {
        self.composite_indices.keys().cloned().collect()
    }

    /// Look up nodes by composite values using a composite index.
    /// Properties must match the order used when creating the index.
    pub fn lookup_by_composite_index(
        &self,
        node_type: &str,
        properties: &[String],
        values: &[Value],
    ) -> Option<Vec<NodeIndex>> {
        let key = (node_type.to_string(), properties.to_vec());
        let composite_value = CompositeValue(values.to_vec());

        self.composite_indices
            .get(&key)
            .and_then(|idx| idx.get(&composite_value))
            .cloned()
    }

    /// Get statistics about a composite index.
    pub fn get_composite_index_stats(
        &self,
        node_type: &str,
        properties: &[String],
    ) -> Option<IndexStats> {
        let key = (node_type.to_string(), properties.to_vec());
        self.composite_indices.get(&key).map(|idx| {
            let total_entries: usize = idx.values().map(|v| v.len()).sum();
            IndexStats {
                unique_values: idx.len(),
                total_entries,
                avg_entries_per_value: if idx.is_empty() {
                    0.0
                } else {
                    total_entries as f64 / idx.len() as f64
                },
            }
        })
    }

    /// Find a composite index that can be used for a given set of filter properties.
    /// Returns the index key and whether all filter properties are covered.
    pub fn find_matching_composite_index(
        &self,
        node_type: &str,
        filter_properties: &[String],
    ) -> Option<(CompositeIndexKey, bool)> {
        // Sort filter properties for comparison
        let mut sorted_filter: Vec<String> = filter_properties.to_vec();
        sorted_filter.sort();

        for key in self.composite_indices.keys() {
            if key.0 == node_type {
                let mut sorted_index: Vec<String> = key.1.clone();
                sorted_index.sort();

                // Check if index properties are a subset of or equal to filter properties
                // For exact match, the index must cover exactly the filter fields
                if sorted_index == sorted_filter {
                    return Some((key.clone(), true)); // Exact match
                }

                // Check if index is a prefix of filter (can be used for partial filtering)
                if sorted_filter.starts_with(&sorted_index)
                    || sorted_index.iter().all(|p| sorted_filter.contains(p))
                {
                    return Some((key.clone(), false)); // Partial match
                }
            }
        }
        None
    }

    // ========================================================================
    // Incremental Index Maintenance (called by Cypher mutations)
    // ========================================================================

    /// Update property, composite, and range indices after a new node is added.
    /// Only updates indices that already exist for this node_type.
    pub fn update_property_indices_for_add(&mut self, node_type: &str, node_idx: NodeIndex) {
        // Collect single-property index updates (immutable borrow of self.graph)
        let prop_updates: Vec<(IndexKey, Value)> = {
            let node = match self.graph.node_weight(node_idx) {
                Some(n) => n,
                None => return,
            };
            self.property_indices
                .keys()
                .chain(self.range_indices.keys())
                .filter(|(nt, _)| nt == node_type)
                .filter_map(|key| {
                    node.properties
                        .get(&key.1)
                        .map(|v| (key.clone(), v.clone()))
                })
                .collect()
        };
        for (key, value) in &prop_updates {
            if let Some(value_map) = self.property_indices.get_mut(key) {
                value_map.entry(value.clone()).or_default().push(node_idx);
            }
            if let Some(btree) = self.range_indices.get_mut(key) {
                btree.entry(value.clone()).or_default().push(node_idx);
            }
        }

        // Collect composite index updates
        let comp_updates: Vec<(CompositeIndexKey, CompositeValue)> = {
            let node = match self.graph.node_weight(node_idx) {
                Some(n) => n,
                None => return,
            };
            self.composite_indices
                .keys()
                .filter(|(nt, _)| nt == node_type)
                .filter_map(|key| {
                    let vals: Vec<Value> = key
                        .1
                        .iter()
                        .map(|p| node.properties.get(p).cloned().unwrap_or(Value::Null))
                        .collect();
                    if vals.iter().any(|v| !matches!(v, Value::Null)) {
                        Some((key.clone(), CompositeValue(vals)))
                    } else {
                        None
                    }
                })
                .collect()
        };
        for (key, comp_val) in comp_updates {
            if let Some(comp_map) = self.composite_indices.get_mut(&key) {
                comp_map.entry(comp_val).or_default().push(node_idx);
            }
        }
    }

    /// Update property, range, and composite indices after a property value is changed.
    /// Removes node from the old value bucket and adds to the new value bucket.
    pub fn update_property_indices_for_set(
        &mut self,
        node_type: &str,
        node_idx: NodeIndex,
        property: &str,
        old_value: Option<&Value>,
        new_value: &Value,
    ) {
        let key = (node_type.to_string(), property.to_string());
        // Update hash index
        if let Some(value_map) = self.property_indices.get_mut(&key) {
            if let Some(old_val) = old_value {
                if let Some(indices) = value_map.get_mut(old_val) {
                    indices.retain(|&idx| idx != node_idx);
                    if indices.is_empty() {
                        value_map.remove(old_val);
                    }
                }
            }
            value_map
                .entry(new_value.clone())
                .or_default()
                .push(node_idx);
        }
        // Update range index
        if let Some(btree) = self.range_indices.get_mut(&key) {
            if let Some(old_val) = old_value {
                if let Some(indices) = btree.get_mut(old_val) {
                    indices.retain(|&idx| idx != node_idx);
                    if indices.is_empty() {
                        btree.remove(old_val);
                    }
                }
            }
            btree.entry(new_value.clone()).or_default().push(node_idx);
        }

        // Update any composite indices that include this property
        self.update_composite_indices_for_property_change(node_type, node_idx, property);
    }

    /// Update property, range, and composite indices after a property is removed.
    pub fn update_property_indices_for_remove(
        &mut self,
        node_type: &str,
        node_idx: NodeIndex,
        property: &str,
        old_value: &Value,
    ) {
        let key = (node_type.to_string(), property.to_string());
        if let Some(value_map) = self.property_indices.get_mut(&key) {
            if let Some(indices) = value_map.get_mut(old_value) {
                indices.retain(|&idx| idx != node_idx);
                if indices.is_empty() {
                    value_map.remove(old_value);
                }
            }
        }
        if let Some(btree) = self.range_indices.get_mut(&key) {
            if let Some(indices) = btree.get_mut(old_value) {
                indices.retain(|&idx| idx != node_idx);
                if indices.is_empty() {
                    btree.remove(old_value);
                }
            }
        }

        // Update any composite indices that include this property
        self.update_composite_indices_for_property_change(node_type, node_idx, property);
    }

    /// Re-index a single node in all composite indices that include the changed property.
    /// Reads current node properties to build the new composite value.
    fn update_composite_indices_for_property_change(
        &mut self,
        node_type: &str,
        node_idx: NodeIndex,
        changed_property: &str,
    ) {
        let comp_keys: Vec<CompositeIndexKey> = self
            .composite_indices
            .keys()
            .filter(|(nt, props)| nt == node_type && props.contains(&changed_property.to_string()))
            .cloned()
            .collect();

        if comp_keys.is_empty() {
            return;
        }

        // Read current node properties once
        let current_props: HashMap<String, Value> = match self.graph.node_weight(node_idx) {
            Some(node) => node.properties.clone(),
            None => return,
        };

        for key in comp_keys {
            if let Some(comp_map) = self.composite_indices.get_mut(&key) {
                // Remove node from all existing composite buckets
                for indices in comp_map.values_mut() {
                    indices.retain(|&idx| idx != node_idx);
                }
                // Remove empty buckets
                comp_map.retain(|_, v| !v.is_empty());

                // Build new composite value from current properties
                let new_values: Vec<Value> = key
                    .1
                    .iter()
                    .map(|p| current_props.get(p).cloned().unwrap_or(Value::Null))
                    .collect();
                if new_values.iter().any(|v| !matches!(v, Value::Null)) {
                    comp_map
                        .entry(CompositeValue(new_values))
                        .or_default()
                        .push(node_idx);
                }
            }
        }
    }

    // ========================================================================
    // Serialization helpers
    // ========================================================================

    /// Snapshot which property/composite indexes exist so they survive serialization.
    /// Called automatically before save.
    pub fn populate_index_keys(&mut self) {
        self.property_index_keys = self.property_indices.keys().cloned().collect();
        self.composite_index_keys = self.composite_indices.keys().cloned().collect();
        self.range_index_keys = self.range_indices.keys().cloned().collect();
    }

    /// Rebuild property and composite indexes from the persisted key lists.
    /// Called automatically after load.
    pub fn rebuild_indices_from_keys(&mut self) {
        let prop_keys: Vec<IndexKey> = std::mem::take(&mut self.property_index_keys);
        for (node_type, property) in &prop_keys {
            self.create_index(node_type, property);
        }
        self.property_index_keys = prop_keys;

        let comp_keys: Vec<CompositeIndexKey> = std::mem::take(&mut self.composite_index_keys);
        for (node_type, properties) in &comp_keys {
            let prop_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
            self.create_composite_index(node_type, &prop_refs);
        }
        self.composite_index_keys = comp_keys;

        let range_keys: Vec<IndexKey> = std::mem::take(&mut self.range_index_keys);
        for (node_type, property) in &range_keys {
            self.create_range_index(node_type, property);
        }
        self.range_index_keys = range_keys;
    }

    // ========================================================================
    // Graph Maintenance: reindex, vacuum, graph_info
    // ========================================================================

    /// Rebuild all indexes from the current graph state.
    ///
    /// Reconstructs type_indices, property_indices, and composite_indices by
    /// scanning all live nodes. Clears lazy caches (id_indices, connection_types)
    /// so they rebuild on next access.
    ///
    /// Use after bulk mutations to ensure index consistency, or when you suspect
    /// indexes have drifted from the actual graph state.
    /// Rebuild type_indices from the live graph.
    /// Called after deserialization (type_indices is `#[serde(skip)]`) and by `reindex()`.
    pub fn rebuild_type_indices(&mut self) {
        let type_count = self.node_type_metadata.len().max(4);
        let avg_per_type = self.graph.node_count() / type_count.max(1);
        let mut new_type_indices: HashMap<String, Vec<NodeIndex>> =
            HashMap::with_capacity(type_count);
        for node_idx in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(node_idx) {
                new_type_indices
                    .entry(node.node_type.clone())
                    .or_insert_with(|| Vec::with_capacity(avg_per_type))
                    .push(node_idx);
            }
        }
        self.type_indices = new_type_indices;
    }

    pub fn reindex(&mut self) {
        // 1. Rebuild type_indices from scratch
        self.rebuild_type_indices();

        // 2. Clear lazy caches — they'll rebuild on next access
        self.id_indices.clear();
        self.connection_types.clear();

        // 3. Rebuild existing property_indices (preserve which indexes exist)
        let property_keys: Vec<IndexKey> = self.property_indices.keys().cloned().collect();
        for (node_type, property) in property_keys {
            self.create_index(&node_type, &property);
        }

        // 4. Rebuild existing composite_indices (preserve which indexes exist)
        let composite_keys: Vec<CompositeIndexKey> =
            self.composite_indices.keys().cloned().collect();
        for (node_type, properties) in composite_keys {
            let prop_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
            self.create_composite_index(&node_type, &prop_refs);
        }

        // 5. Rebuild existing range_indices (preserve which indexes exist)
        let range_keys: Vec<IndexKey> = self.range_indices.keys().cloned().collect();
        for (node_type, property) in range_keys {
            self.create_range_index(&node_type, &property);
        }
    }

    /// Compact the graph by removing tombstones left by deleted nodes/edges.
    ///
    /// With StableDiGraph, deletions leave holes (tombstones) in the internal
    /// storage. Over time, this wastes memory and degrades iteration performance.
    /// vacuum() rebuilds the graph with contiguous indices, then rebuilds all indexes.
    ///
    /// Returns a mapping from old NodeIndex → new NodeIndex so callers can
    /// update any external references (e.g., selections).
    ///
    /// No-op if there are no tombstones (node_count == node_bound).
    pub fn vacuum(&mut self) -> HashMap<NodeIndex, NodeIndex> {
        let old_node_count = self.graph.node_count();
        let old_node_bound = self.graph.node_bound();

        // No tombstones — nothing to compact
        if old_node_count == old_node_bound {
            return HashMap::new();
        }

        // Build new graph with contiguous indices
        let mut new_graph = StableDiGraph::with_capacity(old_node_count, self.graph.edge_count());
        let mut old_to_new: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(old_node_count);

        // Copy all live nodes, recording index mapping
        for old_idx in self.graph.node_indices() {
            let node_data = self.graph[old_idx].clone();
            let new_idx = new_graph.add_node(node_data);
            old_to_new.insert(old_idx, new_idx);
        }

        // Copy all live edges with remapped endpoints
        for old_edge_idx in self.graph.edge_indices() {
            if let Some((src, tgt)) = self.graph.edge_endpoints(old_edge_idx) {
                let edge_data = self.graph[old_edge_idx].clone();
                let new_src = old_to_new[&src];
                let new_tgt = old_to_new[&tgt];
                new_graph.add_edge(new_src, new_tgt, edge_data);
            }
        }

        // Replace graph storage
        self.graph = new_graph;

        // Remap embedding stores to use new node indices
        for store in self.embeddings.values_mut() {
            let mut new_node_to_slot = HashMap::with_capacity(store.node_to_slot.len());
            let mut new_slot_to_node = Vec::with_capacity(store.slot_to_node.len());
            let mut new_data = Vec::with_capacity(store.data.len());

            for (&old_node_raw, &slot) in &store.node_to_slot {
                let old_idx = NodeIndex::new(old_node_raw);
                if let Some(&new_idx) = old_to_new.get(&old_idx) {
                    let new_slot = new_slot_to_node.len();
                    new_node_to_slot.insert(new_idx.index(), new_slot);
                    new_slot_to_node.push(new_idx.index());
                    let start = slot * store.dimension;
                    let end = start + store.dimension;
                    new_data.extend_from_slice(&store.data[start..end]);
                }
                // Deleted nodes (not in old_to_new) are dropped
            }

            store.node_to_slot = new_node_to_slot;
            store.slot_to_node = new_slot_to_node;
            store.data = new_data;
        }

        // Rebuild all indexes from the compacted graph
        self.reindex();

        old_to_new
    }

    /// Check if auto-vacuum should run and trigger it if so.
    ///
    /// Called after DELETE operations. Only vacuums if:
    /// - `auto_vacuum_threshold` is Some(threshold)
    /// - Tombstones exceed 100 (avoid overhead on tiny graphs)
    /// - `fragmentation_ratio` exceeds the threshold
    ///
    /// Returns true if vacuum was triggered.
    pub fn check_auto_vacuum(&mut self) -> bool {
        let threshold = match self.auto_vacuum_threshold {
            Some(t) => t,
            None => return false,
        };

        let node_count = self.graph.node_count();
        let node_bound = self.graph.node_bound();
        let tombstones = node_bound - node_count;

        if tombstones <= 100 {
            return false;
        }

        let ratio = tombstones as f64 / node_bound as f64;
        if ratio > threshold {
            self.vacuum();
            true
        } else {
            false
        }
    }

    /// Return diagnostic information about graph storage health.
    ///
    /// Useful for deciding when to call vacuum():
    /// - `tombstones` > 0 means deleted nodes left holes
    /// - `fragmentation_ratio` approaching 1.0 means most storage is wasted
    /// - A ratio above 0.3 is a good threshold for calling vacuum()
    pub fn graph_info(&self) -> GraphInfo {
        let node_count = self.graph.node_count();
        let node_bound = self.graph.node_bound();
        let edge_count = self.graph.edge_count();
        let node_tombstones = node_bound - node_count;

        GraphInfo {
            node_count,
            node_capacity: node_bound,
            node_tombstones,
            edge_count,
            fragmentation_ratio: if node_bound == 0 {
                0.0
            } else {
                node_tombstones as f64 / node_bound as f64
            },
            type_count: self.type_indices.len(),
            property_index_count: self.property_indices.len(),
            composite_index_count: self.composite_indices.len(),
        }
    }
}

/// Statistics about a property index
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub unique_values: usize,
    pub total_entries: usize,
    pub avg_entries_per_value: f64,
}

/// Diagnostic information about graph storage health.
#[derive(Debug, Clone)]
pub struct GraphInfo {
    /// Number of live nodes in the graph
    pub node_count: usize,
    /// Upper bound of node indices (includes tombstones from deletions)
    pub node_capacity: usize,
    /// Number of tombstone slots (node_capacity - node_count)
    pub node_tombstones: usize,
    /// Number of live edges in the graph
    pub edge_count: usize,
    /// Ratio of wasted storage (0.0 = clean, approaching 1.0 = heavily fragmented)
    pub fragmentation_ratio: f64,
    /// Number of distinct node types
    pub type_count: usize,
    /// Number of single-property indexes
    pub property_index_count: usize,
    /// Number of composite indexes
    pub composite_index_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeData {
    pub id: Value,
    pub title: Value,
    pub node_type: String,
    pub properties: HashMap<String, Value>,
}

impl NodeData {
    pub fn new(
        id: Value,
        title: Value,
        node_type: String,
        properties: HashMap<String, Value>,
    ) -> Self {
        NodeData {
            id,
            title,
            node_type,
            properties,
        }
    }

    /// Returns a reference to the field value without cloning.
    /// Use this for read-only access to avoid allocation overhead.
    /// Note: "type" field is not supported here as it requires Value::String allocation.
    /// Use get_node_type_ref() for the node type.
    #[inline]
    pub fn get_field_ref(&self, field: &str) -> Option<&Value> {
        match field {
            "id" => Some(&self.id),
            "title" => Some(&self.title),
            _ => self.properties.get(field),
        }
    }

    /// Returns the node type as a string reference without allocation.
    #[inline]
    pub fn get_node_type_ref(&self) -> &str {
        self.node_type.as_str()
    }

    pub fn to_node_info(&self) -> NodeInfo {
        NodeInfo {
            id: self.id.clone(),
            title: self.title.clone(),
            node_type: self.node_type.clone(),
            properties: self.properties.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    pub connection_type: String,
    pub properties: HashMap<String, Value>,
}

impl EdgeData {
    pub fn new(connection_type: String, properties: HashMap<String, Value>) -> Self {
        EdgeData {
            connection_type,
            properties,
        }
    }
}

pub type Graph = StableDiGraph<NodeData, EdgeData>;

// ============================================================================
// Schema Definition & Validation Types
// ============================================================================

/// Defines the expected schema for a node type
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeSchemaDefinition {
    /// Fields that must be present on all nodes of this type
    pub required_fields: Vec<String>,
    /// Fields that may be present (for documentation purposes)
    pub optional_fields: Vec<String>,
    /// Expected types for fields: "string", "integer", "float", "boolean", "datetime"
    pub field_types: HashMap<String, String>,
}

/// Defines the expected schema for a connection type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionSchemaDefinition {
    /// The source node type for this connection
    pub source_type: String,
    /// The target node type for this connection
    pub target_type: String,
    /// Optional cardinality constraint: "one-to-one", "one-to-many", "many-to-one", "many-to-many"
    pub cardinality: Option<String>,
    /// Required properties on the connection
    pub required_properties: Vec<String>,
    /// Expected types for connection properties
    pub property_types: HashMap<String, String>,
}

/// Complete schema definition for the graph
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchemaDefinition {
    /// Schema definitions for each node type
    pub node_schemas: HashMap<String, NodeSchemaDefinition>,
    /// Schema definitions for each connection type
    pub connection_schemas: HashMap<String, ConnectionSchemaDefinition>,
}

impl SchemaDefinition {
    pub fn new() -> Self {
        SchemaDefinition {
            node_schemas: HashMap::new(),
            connection_schemas: HashMap::new(),
        }
    }

    /// Add a node type schema
    pub fn add_node_schema(&mut self, node_type: String, schema: NodeSchemaDefinition) {
        self.node_schemas.insert(node_type, schema);
    }

    /// Add a connection type schema
    pub fn add_connection_schema(
        &mut self,
        connection_type: String,
        schema: ConnectionSchemaDefinition,
    ) {
        self.connection_schemas.insert(connection_type, schema);
    }
}

/// Represents a validation error found during schema validation
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// A required field is missing from a node
    MissingRequiredField {
        node_type: String,
        node_title: String,
        field: String,
    },
    /// A field has the wrong type
    TypeMismatch {
        node_type: String,
        node_title: String,
        field: String,
        expected_type: String,
        actual_type: String,
    },
    /// A connection has invalid source or target type
    InvalidConnectionEndpoint {
        connection_type: String,
        expected_source: String,
        expected_target: String,
        actual_source: String,
        actual_target: String,
    },
    /// A required property is missing from a connection
    MissingConnectionProperty {
        connection_type: String,
        source_title: String,
        target_title: String,
        property: String,
    },
    /// A node type exists in the graph but not in the schema
    UndefinedNodeType { node_type: String, count: usize },
    /// A connection type exists in the graph but not in the schema
    UndefinedConnectionType {
        connection_type: String,
        count: usize,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingRequiredField {
                node_type,
                node_title,
                field,
            } => {
                write!(
                    f,
                    "Missing required field '{}' on {} node '{}'",
                    field, node_type, node_title
                )
            }
            ValidationError::TypeMismatch {
                node_type,
                node_title,
                field,
                expected_type,
                actual_type,
            } => {
                write!(
                    f,
                    "Type mismatch on {} node '{}': field '{}' expected {}, got {}",
                    node_type, node_title, field, expected_type, actual_type
                )
            }
            ValidationError::InvalidConnectionEndpoint {
                connection_type,
                expected_source,
                expected_target,
                actual_source,
                actual_target,
            } => {
                write!(
                    f,
                    "Invalid connection '{}': expected {}->{}  but found {}->{}",
                    connection_type, expected_source, expected_target, actual_source, actual_target
                )
            }
            ValidationError::MissingConnectionProperty {
                connection_type,
                source_title,
                target_title,
                property,
            } => {
                write!(
                    f,
                    "Missing required property '{}' on {} connection from '{}' to '{}'",
                    property, connection_type, source_title, target_title
                )
            }
            ValidationError::UndefinedNodeType { node_type, count } => {
                write!(
                    f,
                    "Node type '{}' ({} nodes) exists in graph but not defined in schema",
                    node_type, count
                )
            }
            ValidationError::UndefinedConnectionType {
                connection_type,
                count,
            } => {
                write!(f, "Connection type '{}' ({} connections) exists in graph but not defined in schema", connection_type, count)
            }
        }
    }
}

#[cfg(test)]
mod maintenance_tests {
    use super::*;

    /// Helper: create a DirGraph with N Person nodes and edges between consecutive pairs
    fn make_test_graph(num_nodes: usize, num_edges: bool) -> DirGraph {
        let mut g = DirGraph::new();
        for i in 0..num_nodes {
            let mut props = HashMap::new();
            props.insert("age".to_string(), Value::Int64(20 + i as i64));
            let node = NodeData::new(
                Value::UniqueId(i as u32),
                Value::String(format!("Person_{}", i)),
                "Person".to_string(),
                props,
            );
            let idx = g.graph.add_node(node);
            g.type_indices
                .entry("Person".to_string())
                .or_default()
                .push(idx);
        }
        if num_edges {
            for i in 0..(num_nodes.saturating_sub(1)) {
                let src = NodeIndex::new(i);
                let tgt = NodeIndex::new(i + 1);
                g.graph
                    .add_edge(src, tgt, EdgeData::new("KNOWS".to_string(), HashMap::new()));
            }
        }
        g
    }

    #[test]
    fn test_graph_info_clean() {
        let g = make_test_graph(5, true);
        let info = g.graph_info();
        assert_eq!(info.node_count, 5);
        assert_eq!(info.node_capacity, 5);
        assert_eq!(info.node_tombstones, 0);
        assert_eq!(info.edge_count, 4);
        assert_eq!(info.fragmentation_ratio, 0.0);
        assert_eq!(info.type_count, 1);
    }

    #[test]
    fn test_graph_info_after_deletion() {
        let mut g = make_test_graph(5, false);
        // Delete node 2 — leaves a tombstone
        g.graph.remove_node(NodeIndex::new(2));
        let info = g.graph_info();
        assert_eq!(info.node_count, 4);
        assert_eq!(info.node_capacity, 5); // Still 5 slots
        assert_eq!(info.node_tombstones, 1);
        assert!(info.fragmentation_ratio > 0.19 && info.fragmentation_ratio < 0.21);
    }

    #[test]
    fn test_graph_info_empty() {
        let g = DirGraph::new();
        let info = g.graph_info();
        assert_eq!(info.node_count, 0);
        assert_eq!(info.node_capacity, 0);
        assert_eq!(info.fragmentation_ratio, 0.0);
    }

    #[test]
    fn test_reindex_rebuilds_type_indices() {
        let mut g = make_test_graph(5, false);

        // Manually corrupt type_indices (simulate drift)
        g.type_indices.clear();
        assert!(g.type_indices.is_empty());

        g.reindex();

        // type_indices should be rebuilt
        assert_eq!(g.type_indices.len(), 1);
        assert_eq!(g.type_indices["Person"].len(), 5);
    }

    #[test]
    fn test_reindex_rebuilds_property_indices() {
        let mut g = make_test_graph(5, false);

        // Create a property index
        g.create_index("Person", "age");
        assert!(g.has_index("Person", "age"));

        // Manually corrupt the property index
        g.property_indices
            .get_mut(&("Person".to_string(), "age".to_string()))
            .unwrap()
            .clear();

        g.reindex();

        // Property index should be rebuilt with correct data
        let stats = g.get_index_stats("Person", "age").unwrap();
        assert_eq!(stats.unique_values, 5); // ages 20..24
        assert_eq!(stats.total_entries, 5);
    }

    #[test]
    fn test_reindex_rebuilds_composite_indices() {
        let mut g = make_test_graph(5, false);
        g.create_composite_index("Person", &["age"]);
        assert!(g.has_composite_index("Person", &["age".to_string()]));

        // Corrupt composite index
        g.composite_indices.values_mut().for_each(|v| v.clear());

        g.reindex();

        let stats = g
            .get_composite_index_stats("Person", &["age".to_string()])
            .unwrap();
        assert_eq!(stats.unique_values, 5);
    }

    #[test]
    fn test_reindex_clears_id_indices() {
        let mut g = make_test_graph(3, false);
        g.build_id_index("Person");
        assert!(g.id_indices.contains_key("Person"));

        g.reindex();

        // id_indices should be cleared (lazy rebuild on next access)
        assert!(g.id_indices.is_empty());
    }

    #[test]
    fn test_reindex_after_deletion() {
        let mut g = make_test_graph(5, false);
        // Delete node 2
        g.graph.remove_node(NodeIndex::new(2));
        // type_indices still has the stale entry
        assert_eq!(g.type_indices["Person"].len(), 5);

        g.reindex();

        // Now type_indices should reflect only 4 live nodes
        assert_eq!(g.type_indices["Person"].len(), 4);
        // And none of them should be index 2
        assert!(!g.type_indices["Person"].contains(&NodeIndex::new(2)));
    }

    #[test]
    fn test_vacuum_noop_when_clean() {
        let mut g = make_test_graph(5, true);
        let mapping = g.vacuum();
        assert!(mapping.is_empty()); // No remapping needed
        assert_eq!(g.graph.node_count(), 5);
        assert_eq!(g.graph_info().node_tombstones, 0);
    }

    #[test]
    fn test_vacuum_compacts_after_deletion() {
        let mut g = make_test_graph(5, true);
        // Delete middle node (creates tombstone)
        g.graph.remove_node(NodeIndex::new(2));
        assert_eq!(g.graph.node_count(), 4);
        assert_eq!(g.graph_info().node_tombstones, 1);

        let mapping = g.vacuum();

        // After vacuum: no tombstones, indices are contiguous
        assert_eq!(g.graph.node_count(), 4);
        assert_eq!(g.graph_info().node_tombstones, 0);
        assert_eq!(g.graph_info().node_capacity, 4);

        // Mapping should have 4 entries (one for each surviving node)
        assert_eq!(mapping.len(), 4);
    }

    #[test]
    fn test_vacuum_preserves_node_data() {
        let mut g = make_test_graph(3, false);
        g.graph.remove_node(NodeIndex::new(1)); // Delete Person_1

        let mapping = g.vacuum();

        // Verify all surviving nodes are present with correct data
        let mut titles: Vec<String> = Vec::new();
        for idx in g.graph.node_indices() {
            if let Some(node) = g.graph.node_weight(idx) {
                if let Value::String(s) = &node.title {
                    titles.push(s.clone());
                }
            }
        }
        titles.sort();
        assert_eq!(titles, vec!["Person_0", "Person_2"]);
        assert_eq!(mapping.len(), 2);
    }

    #[test]
    fn test_vacuum_preserves_edges() {
        let mut g = make_test_graph(4, true);
        // Edges: 0→1, 1→2, 2→3
        // Delete node 0 (and its edge to 1)
        g.graph.remove_node(NodeIndex::new(0));
        // Remaining edges should be 1→2, 2→3

        let _mapping = g.vacuum();

        assert_eq!(g.graph.edge_count(), 2);
        assert_eq!(g.graph.node_count(), 3);
    }

    #[test]
    fn test_vacuum_rebuilds_type_indices() {
        let mut g = make_test_graph(5, false);
        g.graph.remove_node(NodeIndex::new(2));

        g.vacuum();

        // type_indices should point to valid, contiguous indices
        assert_eq!(g.type_indices["Person"].len(), 4);
        for &idx in &g.type_indices["Person"] {
            assert!(g.graph.node_weight(idx).is_some());
        }
    }

    #[test]
    fn test_vacuum_rebuilds_property_indices() {
        let mut g = make_test_graph(5, false);
        g.create_index("Person", "age");
        g.graph.remove_node(NodeIndex::new(2));

        g.vacuum();

        // Property index should still exist with correct entries
        assert!(g.has_index("Person", "age"));
        let stats = g.get_index_stats("Person", "age").unwrap();
        assert_eq!(stats.total_entries, 4); // 5 - 1 deleted
    }

    #[test]
    fn test_vacuum_heavy_fragmentation() {
        let mut g = make_test_graph(100, false);
        // Delete every other node — 50% fragmentation
        for i in (0..100).step_by(2) {
            g.graph.remove_node(NodeIndex::new(i));
        }
        assert_eq!(g.graph.node_count(), 50);
        let info = g.graph_info();
        assert!(info.fragmentation_ratio > 0.49);

        let mapping = g.vacuum();

        assert_eq!(mapping.len(), 50);
        assert_eq!(g.graph.node_count(), 50);
        assert_eq!(g.graph_info().node_tombstones, 0);
        assert_eq!(g.graph_info().fragmentation_ratio, 0.0);
    }

    // ========================================================================
    // Incremental Index Update Tests
    // ========================================================================

    #[test]
    fn test_update_property_indices_for_add() {
        let mut g = DirGraph::new();
        // Add a node and create an index
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
        ));
        g.type_indices
            .entry("Person".to_string())
            .or_default()
            .push(n0);
        g.create_index("Person", "city");

        // Add a second node and call the helper
        let mut props2 = HashMap::new();
        props2.insert("city".to_string(), Value::String("Bergen".to_string()));
        let n1 = g.graph.add_node(NodeData::new(
            Value::Int64(2),
            Value::String("Bob".to_string()),
            "Person".to_string(),
            props2,
        ));
        g.type_indices
            .entry("Person".to_string())
            .or_default()
            .push(n1);
        g.update_property_indices_for_add("Person", n1);

        // Verify index was updated
        let oslo = g.lookup_by_index("Person", "city", &Value::String("Oslo".to_string()));
        assert_eq!(oslo.unwrap().len(), 1);
        let bergen = g.lookup_by_index("Person", "city", &Value::String("Bergen".to_string()));
        let bergen = bergen.unwrap();
        assert_eq!(bergen.len(), 1);
        assert_eq!(bergen[0], n1);
    }

    #[test]
    fn test_update_property_indices_for_set() {
        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
        ));
        g.type_indices
            .entry("Person".to_string())
            .or_default()
            .push(n0);
        g.create_index("Person", "city");

        // Simulate SET n.city = 'Bergen'
        let old_val = Value::String("Oslo".to_string());
        let new_val = Value::String("Bergen".to_string());
        // Actually change the property on the node
        if let Some(node) = g.graph.node_weight_mut(n0) {
            node.properties.insert("city".to_string(), new_val.clone());
        }
        g.update_property_indices_for_set("Person", n0, "city", Some(&old_val), &new_val);

        // Verify: Oslo bucket should be empty, Bergen should have the node
        let oslo = g.lookup_by_index("Person", "city", &Value::String("Oslo".to_string()));
        assert!(oslo.is_none() || oslo.unwrap().is_empty());
        let bergen = g.lookup_by_index("Person", "city", &Value::String("Bergen".to_string()));
        assert_eq!(bergen.unwrap(), vec![n0]);
    }

    #[test]
    fn test_update_property_indices_for_remove() {
        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
        ));
        g.type_indices
            .entry("Person".to_string())
            .or_default()
            .push(n0);
        g.create_index("Person", "city");

        // Simulate REMOVE n.city
        let old_val = Value::String("Oslo".to_string());
        if let Some(node) = g.graph.node_weight_mut(n0) {
            node.properties.remove("city");
        }
        g.update_property_indices_for_remove("Person", n0, "city", &old_val);

        // Verify: Oslo bucket should be empty
        let oslo = g.lookup_by_index("Person", "city", &Value::String("Oslo".to_string()));
        assert!(oslo.is_none() || oslo.unwrap().is_empty());
    }

    #[test]
    fn test_update_composite_index_on_property_change() {
        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        props.insert("age".to_string(), Value::Int64(30));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
        ));
        g.type_indices
            .entry("Person".to_string())
            .or_default()
            .push(n0);
        g.create_composite_index("Person", &["city", "age"]);

        // Verify initial state
        let key = (
            "Person".to_string(),
            vec!["city".to_string(), "age".to_string()],
        );
        assert!(g.composite_indices.get(&key).unwrap().len() == 1);

        // Change city to Bergen
        let old_val = Value::String("Oslo".to_string());
        let new_val = Value::String("Bergen".to_string());
        if let Some(node) = g.graph.node_weight_mut(n0) {
            node.properties.insert("city".to_string(), new_val.clone());
        }
        g.update_property_indices_for_set("Person", n0, "city", Some(&old_val), &new_val);

        // Verify: old composite value gone, new one present
        let comp_map = g.composite_indices.get(&key).unwrap();
        let old_comp = CompositeValue(vec![Value::String("Oslo".to_string()), Value::Int64(30)]);
        let new_comp = CompositeValue(vec![Value::String("Bergen".to_string()), Value::Int64(30)]);
        assert!(!comp_map.contains_key(&old_comp) || comp_map.get(&old_comp).unwrap().is_empty());
        assert_eq!(comp_map.get(&new_comp).unwrap(), &vec![n0]);
    }

    #[test]
    fn test_no_update_when_no_index_exists() {
        let mut g = DirGraph::new();
        let mut props = HashMap::new();
        props.insert("city".to_string(), Value::String("Oslo".to_string()));
        let n0 = g.graph.add_node(NodeData::new(
            Value::Int64(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            props,
        ));
        g.type_indices
            .entry("Person".to_string())
            .or_default()
            .push(n0);
        // No index created — these should be no-ops without crash
        g.update_property_indices_for_add("Person", n0);
        g.update_property_indices_for_set(
            "Person",
            n0,
            "city",
            Some(&Value::String("Oslo".to_string())),
            &Value::String("Bergen".to_string()),
        );
        g.update_property_indices_for_remove(
            "Person",
            n0,
            "city",
            &Value::String("Oslo".to_string()),
        );
        assert!(g.property_indices.is_empty());
    }
}
