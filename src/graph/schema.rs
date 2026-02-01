// src/graph/schema.rs
use std::collections::HashMap;
use std::sync::Arc;
use petgraph::graph::{DiGraph, NodeIndex, EdgeIndex};
use serde::{Serialize, Deserialize};
use crate::datatypes::values::{Value, FilterCondition};

#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub id: Value,
    pub title: Value,
    pub node_type: String,
    pub properties: HashMap<String, Value>,
}

#[derive(Clone, Debug)]
pub enum SelectionOperation {
    Filter(HashMap<String, FilterCondition>),
    Sort(Vec<(String, bool)>),  // (field_name, ascending)
    Traverse {
        connection_type: String,
        direction: Option<String>,
        max_nodes: Option<usize>,
    },
    Custom(String),  // For operations that don't fit other categories
}

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
        self.selections.values()
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
        self.selections.values().flat_map(|children| children.iter().copied())
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
        self.levels.last()
            .map(|l| l.node_count())
            .unwrap_or(0)
    }

    /// Returns an iterator over node indices in the current (most recent) level.
    pub fn current_node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.levels.last()
            .into_iter()
            .flat_map(|l| l.iter_node_indices())
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

#[derive(Clone, Serialize, Deserialize)]
pub struct DirGraph {
    pub(crate) graph: Graph,
    pub(crate) type_indices: HashMap<String, Vec<NodeIndex>>,
    /// Optional schema definition for validation
    #[serde(default)]
    pub(crate) schema_definition: Option<SchemaDefinition>,
    /// Single-property indexes for fast lookups: (node_type, property) -> value -> [node_indices]
    #[serde(default)]
    pub(crate) property_indices: HashMap<IndexKey, HashMap<Value, Vec<NodeIndex>>>,
    /// Composite indexes for multi-field queries: (node_type, [properties]) -> composite_value -> [node_indices]
    #[serde(default)]
    pub(crate) composite_indices: HashMap<CompositeIndexKey, HashMap<CompositeValue, Vec<NodeIndex>>>,
    /// Fast O(1) lookup by node ID: node_type -> (id_value -> NodeIndex)
    /// Lazily built on first use for each node type, skipped during serialization
    #[serde(skip)]
    pub(crate) id_indices: HashMap<String, HashMap<Value, NodeIndex>>,
    /// Fast O(1) lookup for connection types. Populated on first edge access.
    #[serde(skip)]
    pub(crate) connection_types: std::collections::HashSet<String>,
}

impl DirGraph {
    pub fn new() -> Self {
        DirGraph {
            graph: Graph::new(),
            type_indices: HashMap::new(),
            schema_definition: None,
            property_indices: HashMap::new(),
            composite_indices: HashMap::new(),
            id_indices: HashMap::new(),
            connection_types: std::collections::HashSet::new(),
        }
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
                if let Some(NodeData::Regular { id, .. }) = self.graph.node_weight(node_idx) {
                    index.insert(id.clone(), node_idx);
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
    pub fn lookup_by_id_normalized(&self, node_type: &str, id: &Value) -> Option<NodeIndex> {
        let type_index = self.id_indices.get(node_type)?;

        // Try direct lookup first
        if let Some(&idx) = type_index.get(id) {
            return Some(idx);
        }

        // If direct lookup fails, try alternative integer representations
        match id {
            Value::Int64(i) => {
                // Int64 -> try UniqueId(u32) if value fits
                if *i >= 0 && *i <= u32::MAX as i64 {
                    type_index.get(&Value::UniqueId(*i as u32)).copied()
                } else {
                    None
                }
            }
            Value::UniqueId(u) => {
                // UniqueId -> try Int64
                type_index.get(&Value::Int64(*u as i64)).copied()
            }
            Value::Float64(f) => {
                // Float64 -> try Int64 then UniqueId if it's a whole number
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
            _ => None, // String, Boolean, DateTime, Null - no normalization
        }
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
        // Fast path: check the connection_types cache first (O(1))
        if !self.connection_types.is_empty() {
            return self.connection_types.contains(connection_type);
        }

        // Fallback: scan SchemaNodes (O(n) - only happens if cache not built yet)
        self.graph.node_weights().any(|node| {
            match node {
                NodeData::Schema { node_type, title, .. } => {
                    node_type == "SchemaNode" &&
                    matches!(title, Value::String(t) if t == connection_type)
                },
                _ => false
            }
        })
    }

    /// Register a connection type for O(1) lookups.
    /// Called when edges are added to the graph.
    pub fn register_connection_type(&mut self, connection_type: String) {
        self.connection_types.insert(connection_type);
    }

    /// Build the connection types cache by scanning all edges.
    /// Called lazily after deserialization or when cache is needed.
    pub fn build_connection_types_cache(&mut self) {
        if !self.connection_types.is_empty() {
            return; // Already built
        }

        for edge in self.graph.edge_weights() {
            self.connection_types.insert(edge.connection_type.clone());
        }
    }

    pub fn has_node_type(&self, node_type: &str) -> bool {
        // Check if we have any nodes of this type in our type indices
        self.type_indices.contains_key(node_type) ||
        // Also check for SchemaNodes that might represent this type
        self.graph.node_weights().any(|node| {
            match node {
                NodeData::Schema { node_type: nt, title, .. } => {
                    nt == "SchemaNode" &&
                    matches!(title, Value::String(t) if t == node_type)
                },
                _ => false
            }
        })
    }

    /// Get all node types that exist in the graph (excluding SchemaNode)
    pub fn get_node_types(&self) -> Vec<String> {
        let mut types: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Get types from type_indices (fast path)
        for node_type in self.type_indices.keys() {
            if node_type != "SchemaNode" {
                types.insert(node_type.clone());
            }
        }

        // Also scan for any types not in indices (fallback)
        for node in self.graph.node_weights() {
            if let NodeData::Regular { node_type, .. } = node {
                types.insert(node_type.clone());
            }
        }

        types.into_iter().collect()
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
                if let Some(NodeData::Regular { properties, .. }) = self.graph.node_weight(idx) {
                    if let Some(value) = properties.get(property) {
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
    pub fn lookup_by_index(&self, node_type: &str, property: &str, value: &Value) -> Option<Vec<NodeIndex>> {
        let key = (node_type.to_string(), property.to_string());
        self.property_indices.get(&key)
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
                avg_entries_per_value: if idx.is_empty() { 0.0 } else { total_entries as f64 / idx.len() as f64 },
            }
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
                if let Some(NodeData::Regular { properties: props, .. }) = self.graph.node_weight(idx) {
                    // Extract values for all properties in order
                    let values: Vec<Value> = properties.iter()
                        .map(|p| props.get(*p).cloned().unwrap_or(Value::Null))
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

        self.composite_indices.get(&key)
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
                avg_entries_per_value: if idx.is_empty() { 0.0 } else { total_entries as f64 / idx.len() as f64 },
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

        for (key, _) in &self.composite_indices {
            if key.0 == node_type {
                let mut sorted_index: Vec<String> = key.1.clone();
                sorted_index.sort();

                // Check if index properties are a subset of or equal to filter properties
                // For exact match, the index must cover exactly the filter fields
                if sorted_index == sorted_filter {
                    return Some((key.clone(), true)); // Exact match
                }

                // Check if index is a prefix of filter (can be used for partial filtering)
                if sorted_filter.starts_with(&sorted_index) ||
                   sorted_index.iter().all(|p| sorted_filter.contains(p)) {
                    return Some((key.clone(), false)); // Partial match
                }
            }
        }
        None
    }
}

/// Statistics about a property index
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub unique_values: usize,
    pub total_entries: usize,
    pub avg_entries_per_value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeData {
    Regular {
        id: Value,
        title: Value,
        node_type: String,
        properties: HashMap<String, Value>,
    },
    Schema {
        id: Value,
        title: Value,
        node_type: String,
        properties: HashMap<String, Value>,
    },
}

impl NodeData {
    pub fn new(id: Value, title: Value, node_type: String, properties: HashMap<String, Value>) -> Self {
        NodeData::Regular {
            id,
            title,
            node_type,
            properties,
        }
    }
    pub fn get_field(&self, field: &str) -> Option<Value> {
        match self {
            NodeData::Regular { id, title, node_type, properties } => {
                match field {
                    "id" => Some(id.clone()),
                    "title" => Some(title.clone()),
                    "type" => Some(Value::String(node_type.clone())),
                    _ => properties.get(field).cloned()
                }
            },
            NodeData::Schema { .. } => None
        }
    }

    pub fn is_regular(&self) -> bool {
        matches!(self, NodeData::Regular { .. })
    }

    pub fn to_node_info(&self) -> Option<NodeInfo> {
        match self {
            NodeData::Regular { id, title, node_type, properties } => Some(NodeInfo {
                id: id.clone(),
                title: title.clone(),
                node_type: node_type.clone(),
                properties: properties.clone(),
            }),
            NodeData::Schema { .. } => None,
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

pub type Graph = DiGraph<NodeData, EdgeData>;

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
    pub fn add_connection_schema(&mut self, connection_type: String, schema: ConnectionSchemaDefinition) {
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
    UndefinedNodeType {
        node_type: String,
        count: usize,
    },
    /// A connection type exists in the graph but not in the schema
    UndefinedConnectionType {
        connection_type: String,
        count: usize,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingRequiredField { node_type, node_title, field } => {
                write!(f, "Missing required field '{}' on {} node '{}'", field, node_type, node_title)
            }
            ValidationError::TypeMismatch { node_type, node_title, field, expected_type, actual_type } => {
                write!(f, "Type mismatch on {} node '{}': field '{}' expected {}, got {}",
                    node_type, node_title, field, expected_type, actual_type)
            }
            ValidationError::InvalidConnectionEndpoint { connection_type, expected_source, expected_target, actual_source, actual_target } => {
                write!(f, "Invalid connection '{}': expected {}->{}  but found {}->{}",
                    connection_type, expected_source, expected_target, actual_source, actual_target)
            }
            ValidationError::MissingConnectionProperty { connection_type, source_title, target_title, property } => {
                write!(f, "Missing required property '{}' on {} connection from '{}' to '{}'",
                    property, connection_type, source_title, target_title)
            }
            ValidationError::UndefinedNodeType { node_type, count } => {
                write!(f, "Node type '{}' ({} nodes) exists in graph but not defined in schema", node_type, count)
            }
            ValidationError::UndefinedConnectionType { connection_type, count } => {
                write!(f, "Connection type '{}' ({} connections) exists in graph but not defined in schema", connection_type, count)
            }
        }
    }
}