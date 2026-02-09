// src/graph/cypher/result.rs
// Result types for the Cypher query pipeline

use crate::datatypes::values::Value;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

// ============================================================================
// Pipeline Result Types
// ============================================================================

/// A single row in the pipeline result set.
/// During execution, rows carry lightweight NodeIndex references.
/// Properties are resolved on-demand (zero-copy via get_field_ref).
#[derive(Debug, Clone)]
pub struct ResultRow {
    /// Node variable bindings: variable_name -> NodeIndex
    pub node_bindings: HashMap<String, NodeIndex>,
    /// Edge variable bindings: variable_name -> (source_idx, target_idx, connection_type, properties)
    pub edge_bindings: HashMap<String, EdgeBinding>,
    /// Variable-length path bindings
    pub path_bindings: HashMap<String, PathBinding>,
    /// Projected values from WITH/RETURN
    pub projected: HashMap<String, Value>,
}

/// Lightweight edge binding
#[derive(Debug, Clone)]
pub struct EdgeBinding {
    #[allow(dead_code)]
    pub source: NodeIndex,
    #[allow(dead_code)]
    pub target: NodeIndex,
    pub connection_type: String,
    pub properties: HashMap<String, Value>,
}

/// Variable-length path binding
#[derive(Debug, Clone)]
pub struct PathBinding {
    #[allow(dead_code)]
    pub source: NodeIndex,
    #[allow(dead_code)]
    pub target: NodeIndex,
    pub hops: usize,
    #[allow(dead_code)]
    pub path: Vec<(NodeIndex, String)>,
}

impl ResultRow {
    pub fn new() -> Self {
        ResultRow {
            node_bindings: HashMap::new(),
            edge_bindings: HashMap::new(),
            path_bindings: HashMap::new(),
            projected: HashMap::new(),
        }
    }

    /// Pre-sized constructor to avoid HashMap reallocation.
    pub fn with_capacity(nodes: usize, edges: usize, projected: usize) -> Self {
        ResultRow {
            node_bindings: HashMap::with_capacity(nodes),
            edge_bindings: HashMap::with_capacity(edges),
            path_bindings: HashMap::new(),
            projected: HashMap::with_capacity(projected),
        }
    }

    /// Create a row with only projected values (for aggregation results)
    pub fn from_projected(projected: HashMap<String, Value>) -> Self {
        ResultRow {
            node_bindings: HashMap::new(),
            edge_bindings: HashMap::new(),
            path_bindings: HashMap::new(),
            projected,
        }
    }
}

/// The result set flowing through the pipeline
#[derive(Debug)]
pub struct ResultSet {
    pub rows: Vec<ResultRow>,
    /// Column names in output order (populated by RETURN)
    pub columns: Vec<String>,
}

impl ResultSet {
    pub fn new() -> Self {
        ResultSet {
            rows: Vec::new(),
            columns: Vec::new(),
        }
    }
}

// ============================================================================
// Final Output
// ============================================================================

/// Mutation statistics returned from CREATE/SET/DELETE queries
#[derive(Debug, Clone, Default)]
pub struct MutationStats {
    pub nodes_created: usize,
    pub relationships_created: usize,
    pub properties_set: usize,
    pub nodes_deleted: usize,
    pub relationships_deleted: usize,
    pub properties_removed: usize,
}

/// Final query result returned to Python
#[derive(Debug)]
pub struct CypherResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
    pub stats: Option<MutationStats>,
}

impl CypherResult {
    pub fn empty() -> Self {
        CypherResult {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: None,
        }
    }
}
