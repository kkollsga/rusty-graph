// Pattern AST types for graph pattern matching.
//
// Supports patterns like: (p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use petgraph::graph::{EdgeIndex, NodeIndex};
use std::collections::HashMap;

// ============================================================================
// AST Types
// ============================================================================

/// A complete pattern to match against the graph
#[derive(Debug, Clone)]
pub struct Pattern {
    pub elements: Vec<PatternElement>,
}

/// Either a node or edge pattern
#[derive(Debug, Clone)]
pub enum PatternElement {
    Node(NodePattern),
    Edge(EdgePattern),
}

/// Pattern for matching nodes: (var:Type {prop: value})
#[derive(Debug, Clone)]
pub struct NodePattern {
    pub variable: Option<String>,
    pub node_type: Option<String>,
    pub properties: Option<HashMap<String, PropertyMatcher>>,
}

/// Pattern for matching edges: -[:TYPE {prop: value}]->
/// Supports variable-length paths with *min..max syntax:
/// - `*` or `*..` means 1 or more hops (default)
/// - `*2` means exactly 2 hops
/// - `*1..3` means 1 to 3 hops
/// - `*..5` means 1 to 5 hops
/// - `*2..` means 2 or more hops (up to default max)
#[derive(Debug, Clone)]
pub struct EdgePattern {
    pub variable: Option<String>,
    pub connection_type: Option<String>,
    /// Multiple allowed connection types from pipe syntax: `[:A|B|C]`.
    /// When set, an edge matches if its type equals ANY of these types.
    /// `connection_type` holds the first type for backward compatibility.
    pub connection_types: Option<Vec<String>>,
    pub direction: EdgeDirection,
    pub properties: Option<HashMap<String, PropertyMatcher>>,
    /// Variable-length path configuration: (min_hops, max_hops)
    /// None means exactly 1 hop (normal edge)
    pub var_length: Option<(usize, usize)>,
    /// When false, variable-length expansion skips path tracking and uses
    /// global BFS dedup.  Set by the query planner when the query doesn't
    /// reference path info (no `p = ...` assignment, no named edge variable).
    pub needs_path_info: bool,
    /// When true, the connection type metadata guarantees the target node
    /// matches the pattern's type, so the node_weight() lookup can be skipped.
    /// Set by the query planner when connection_type_metadata confirms a single
    /// target type (outgoing) or source type (incoming).
    pub skip_target_type_check: bool,
}

/// Direction of edge traversal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeDirection {
    Outgoing, // -[]->
    Incoming, // <-[]-
    Both,     // -[]-
}

/// Property value matcher
#[derive(Debug, Clone)]
pub enum PropertyMatcher {
    Equals(Value),
    /// Deferred parameter resolution: matched at execution time from params map
    EqualsParam(String),
    /// Deferred variable resolution: resolved against projected row values
    /// from WITH/UNWIND before pattern matching. Example:
    /// `WITH "Oslo" AS city MATCH (n:Person {city: city})`
    EqualsVar(String),
    /// Deferred node-property resolution: resolved against an already-bound
    /// node's property at row-execute time. Pushed by the planner from a
    /// correlated `WHERE cur.prop = prior.other_prop` so the pattern executor
    /// can pick an indexed lookup when `(cur_type, prop)` is indexed.
    EqualsNodeProp {
        var: String,
        prop: String,
    },
    /// IN-list matching: value must be one of these values.
    /// Pushed from `WHERE n.prop IN [v1, v2, ...]` by the planner.
    In(Vec<Value>),
    /// Comparison matchers: pushed from `WHERE n.prop > val` etc. by the planner.
    /// Enables filter pushdown into MATCH and range index acceleration.
    GreaterThan(Value),
    GreaterOrEqual(Value),
    LessThan(Value),
    LessOrEqual(Value),
    /// Combined range: both a lower and upper bound on the same property.
    /// Used when WHERE has e.g. `n.year >= 2015 AND n.year <= 2022`.
    /// Booleans indicate inclusive (true) vs exclusive (false).
    Range {
        lower: Value,
        lower_inclusive: bool,
        upper: Value,
        upper_inclusive: bool,
    },
}

// ============================================================================
// Match Results
// ============================================================================

/// A single pattern match with variable bindings.
/// Uses Vec instead of HashMap — patterns add 1-6 unique variables,
/// so linear search is faster than hashing and clone is a single memcpy.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub bindings: Vec<(String, MatchBinding)>,
}

/// A bound value (either node, edge, or variable-length path)
#[derive(Debug, Clone)]
pub enum MatchBinding {
    Node {
        #[allow(dead_code)]
        index: NodeIndex,
        node_type: String,
        title: String,
        id: Value,
        properties: HashMap<String, Value>,
    },
    /// Lightweight node reference — stores only NodeIndex (4 bytes).
    /// Used in Cypher executor path where node data is resolved on demand from graph.
    NodeRef(NodeIndex),
    Edge {
        source: NodeIndex,
        target: NodeIndex,
        edge_index: EdgeIndex,
        connection_type: InternedKey,
        properties: HashMap<String, Value>,
    },
    /// Variable-length path binding for patterns like -[:TYPE*1..3]->
    VariableLengthPath {
        source: NodeIndex,
        target: NodeIndex,
        hops: usize,
        /// Path as list of (node_index, connection_type) pairs
        path: Vec<(NodeIndex, InternedKey)>,
    },
}
