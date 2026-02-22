// src/graph/cypher/result.rs
// Result types for the Cypher query pipeline

use crate::datatypes::values::Value;
use petgraph::graph::{EdgeIndex, NodeIndex};
use std::collections::HashMap;

// ============================================================================
// Bindings — compact ordered map for small variable counts
// ============================================================================

/// Compact ordered map using `Vec<(String, V)>` with linear search.
/// Faster than HashMap for typical Cypher variable counts (1-8 entries):
/// no hashing overhead, cache-friendly sequential access, cheaper clone
/// (one Vec allocation vs HashMap bucket array + entries).
#[derive(Debug, Clone, Default)]
pub struct Bindings<V> {
    entries: Vec<(String, V)>,
}

impl<V> Bindings<V> {
    pub fn new() -> Self {
        Bindings {
            entries: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Bindings {
            entries: Vec::with_capacity(cap),
        }
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    #[allow(dead_code)]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        self.entries
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Upsert: update if key exists, push if not.
    pub fn insert(&mut self, key: String, val: V) {
        if let Some(entry) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            entry.1 = val;
        } else {
            self.entries.push((key, val));
        }
    }

    #[allow(dead_code)]
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.iter().map(|(k, _)| k)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove a key and return its value (move, no clone).
    pub fn remove(&mut self, key: &str) -> Option<V> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == key) {
            Some(self.entries.swap_remove(pos).1)
        } else {
            None
        }
    }

    /// Convert to HashMap for interop with pattern_matching pre_bindings.
    pub fn to_hashmap(&self) -> HashMap<String, V>
    where
        V: Clone,
    {
        self.entries
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

impl<V> IntoIterator for Bindings<V> {
    type Item = (String, V);
    type IntoIter = std::vec::IntoIter<(String, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a, V> IntoIterator for &'a Bindings<V> {
    type Item = &'a (String, V);
    type IntoIter = std::slice::Iter<'a, (String, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

// ============================================================================
// Pipeline Result Types
// ============================================================================

/// A single row in the pipeline result set.
/// During execution, rows carry lightweight NodeIndex/EdgeIndex references.
/// Properties are resolved on-demand from the graph.
#[derive(Debug, Clone)]
pub struct ResultRow {
    /// Node variable bindings: variable_name -> NodeIndex
    pub node_bindings: Bindings<NodeIndex>,
    /// Edge variable bindings: variable_name -> EdgeBinding (source, target, edge_index)
    pub edge_bindings: Bindings<EdgeBinding>,
    /// Variable-length path bindings
    pub path_bindings: Bindings<PathBinding>,
    /// Projected values from WITH/RETURN
    pub projected: Bindings<Value>,
}

/// Lightweight edge binding — stores only indices, no cloned data.
/// Edge properties are resolved on-demand from the graph via edge_index.
#[derive(Debug, Clone, Copy)]
pub struct EdgeBinding {
    pub source: NodeIndex,
    pub target: NodeIndex,
    pub edge_index: EdgeIndex,
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
            node_bindings: Bindings::new(),
            edge_bindings: Bindings::new(),
            path_bindings: Bindings::new(),
            projected: Bindings::new(),
        }
    }

    /// Pre-sized constructor to avoid reallocation.
    pub fn with_capacity(nodes: usize, edges: usize, projected: usize) -> Self {
        ResultRow {
            node_bindings: Bindings::with_capacity(nodes),
            edge_bindings: Bindings::with_capacity(edges),
            path_bindings: Bindings::new(),
            projected: Bindings::with_capacity(projected),
        }
    }

    /// Create a row with only projected values (for aggregation results)
    pub fn from_projected(projected: Bindings<Value>) -> Self {
        ResultRow {
            node_bindings: Bindings::new(),
            edge_bindings: Bindings::new(),
            path_bindings: Bindings::new(),
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
