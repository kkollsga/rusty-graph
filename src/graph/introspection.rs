// src/graph/introspection.rs
//
// Schema introspection functions for exploring graph structure.
// All functions take &DirGraph and return Rust structs — PyO3 conversion in mod.rs.

use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, NodeData};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

// ── Return types ────────────────────────────────────────────────────────────

/// Statistics about a connection type: count, source/target node types.
pub struct ConnectionTypeStats {
    pub connection_type: String,
    pub count: usize,
    pub source_types: Vec<String>,
    pub target_types: Vec<String>,
}

/// Summary of a node type: count and property schemas with types.
pub struct NodeTypeOverview {
    pub count: usize,
    pub properties: HashMap<String, String>,
}

/// Complete schema summary: all node types, connection types, indexes, and totals.
pub struct SchemaOverview {
    pub node_types: Vec<(String, NodeTypeOverview)>,
    pub connection_types: Vec<ConnectionTypeStats>,
    pub indexes: Vec<String>,
    pub node_count: usize,
    pub edge_count: usize,
}

/// Per-property statistics: data type, non-null count, unique count, and optional value list.
pub struct PropertyStatInfo {
    pub property_name: String,
    pub type_string: String,
    pub non_null: usize,
    pub unique: usize,
    pub values: Option<Vec<Value>>,
}

/// A single neighbor connection: edge type, connected node type, and count.
pub struct NeighborConnection {
    pub connection_type: String,
    pub other_type: String,
    pub count: usize,
}

/// Grouped neighbor connections for a node type: incoming and outgoing edges.
pub struct NeighborsSchema {
    pub outgoing: Vec<NeighborConnection>,
    pub incoming: Vec<NeighborConnection>,
}

/// Level of Cypher documentation requested via `describe(cypher=...)`.
pub enum CypherDetail {
    /// No Cypher docs (default).
    Off,
    /// Tier 2: compact reference listing — all clauses, operators, functions, procedures.
    Overview,
    /// Tier 3: detailed docs with params and examples for specific topics.
    Topics(Vec<String>),
}

/// Level of connection documentation requested via `describe(connections=...)`.
pub enum ConnectionDetail {
    /// No standalone connection docs (default — connections shown in inventory).
    Off,
    /// Overview: all connection types with count, endpoints, property names.
    Overview,
    /// Deep-dive: specific connection types with per-pair counts, property stats, samples.
    Topics(Vec<String>),
}

// ── Describe helpers ────────────────────────────────────────────────────────

/// Capability flags for a node type (used by `describe()`).
struct TypeCapabilities {
    has_timeseries: bool,
    has_location: bool,
    has_geometry: bool,
    has_embeddings: bool,
}

impl TypeCapabilities {
    /// Format inline capability flags: "ts", "geo", "loc", "vec".
    fn flags_csv(&self) -> String {
        let mut flags = Vec::new();
        if self.has_timeseries {
            flags.push("ts");
        }
        if self.has_geometry {
            flags.push("geo");
        }
        if self.has_location && !self.has_geometry {
            flags.push("loc");
        }
        if self.has_embeddings {
            flags.push("vec");
        }
        flags.join(",")
    }

    /// Merge another type's capabilities into this one (for bubbling up).
    fn merge(&mut self, other: &TypeCapabilities) {
        self.has_timeseries |= other.has_timeseries;
        self.has_location |= other.has_location;
        self.has_geometry |= other.has_geometry;
        self.has_embeddings |= other.has_embeddings;
    }
}

/// Property complexity marker based on property count.
fn property_complexity(count: usize) -> &'static str {
    match count {
        0..=3 => "vl",
        4..=8 => "l",
        9..=15 => "m",
        16..=30 => "h",
        _ => "vh",
    }
}

/// Size tier for node count: vs (<10), s (10-99), m (100-999), l (1K-9999), vl (10K+).
fn size_tier(count: usize) -> &'static str {
    match count {
        0..=9 => "vs",
        10..=99 => "s",
        100..=999 => "m",
        1000..=9999 => "l",
        _ => "vl",
    }
}

/// Format a compact type descriptor: `Name[size,complexity,flags]` or `Name[size,complexity]`.
fn format_type_descriptor(
    name: &str,
    count: usize,
    prop_count: usize,
    caps: &TypeCapabilities,
) -> String {
    let size = size_tier(count);
    let complexity = property_complexity(prop_count);
    let flags = caps.flags_csv();
    if flags.is_empty() {
        format!("{}[{},{}]", xml_escape(name), size, complexity)
    } else {
        format!("{}[{},{},{}]", xml_escape(name), size, complexity, flags)
    }
}

/// Bubble capabilities from supporting types up to their parent core types.
fn bubble_capabilities(
    caps: &mut HashMap<String, TypeCapabilities>,
    parent_types: &HashMap<String, String>,
) {
    // Collect child caps first to avoid borrow issues
    let child_caps: Vec<(String, TypeCapabilities)> = parent_types
        .iter()
        .filter_map(|(child, parent)| {
            caps.get(child).map(|c| {
                (
                    parent.clone(),
                    TypeCapabilities {
                        has_timeseries: c.has_timeseries,
                        has_location: c.has_location,
                        has_geometry: c.has_geometry,
                        has_embeddings: c.has_embeddings,
                    },
                )
            })
        })
        .collect();
    for (parent, child_cap) in &child_caps {
        if let Some(parent_cap) = caps.get_mut(parent) {
            parent_cap.merge(child_cap);
        }
    }
}

/// Count supporting children per parent type.
fn children_counts(parent_types: &HashMap<String, String>) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for parent in parent_types.values() {
        *counts.entry(parent.clone()).or_insert(0) += 1;
    }
    counts
}

/// Detect capabilities for all node types in the graph.
fn compute_type_capabilities(graph: &DirGraph) -> HashMap<String, TypeCapabilities> {
    let mut caps: HashMap<String, TypeCapabilities> = HashMap::new();

    for node_type in graph.type_indices.keys() {
        let mut tc = TypeCapabilities {
            has_timeseries: false,
            has_location: false,
            has_geometry: false,
            has_embeddings: false,
        };

        // Timeseries
        tc.has_timeseries = graph.timeseries_configs.contains_key(node_type);

        // Spatial
        if let Some(sc) = graph.spatial_configs.get(node_type) {
            tc.has_location = sc.location.is_some() || !sc.points.is_empty();
            tc.has_geometry = sc.geometry.is_some() || !sc.shapes.is_empty();
        }

        // Also check metadata for point-type fields (no SpatialConfig set)
        if !tc.has_location {
            if let Some(meta) = graph.node_type_metadata.get(node_type) {
                tc.has_location = meta.values().any(|t| t.eq_ignore_ascii_case("point"));
            }
        }

        // Embeddings
        tc.has_embeddings = graph.embeddings.keys().any(|(nt, _)| nt == node_type);

        caps.insert(node_type.clone(), tc);
    }
    caps
}

// ── Core functions ──────────────────────────────────────────────────────────

/// Scan all edges once to compute per-connection-type stats.
pub fn compute_connection_type_stats(graph: &DirGraph) -> Vec<ConnectionTypeStats> {
    let mut stats: HashMap<String, (usize, HashSet<String>, HashSet<String>)> = HashMap::new();

    for edge_ref in graph.graph.edge_references() {
        let edge_data = edge_ref.weight();
        let entry = stats
            .entry(edge_data.connection_type.clone())
            .or_insert_with(|| (0, HashSet::new(), HashSet::new()));
        entry.0 += 1;

        if let Some(source_node) = graph.get_node(edge_ref.source()) {
            entry.1.insert(source_node.node_type.clone());
        }
        if let Some(target_node) = graph.get_node(edge_ref.target()) {
            entry.2.insert(target_node.node_type.clone());
        }
    }

    let mut result: Vec<ConnectionTypeStats> = stats
        .into_iter()
        .map(|(conn_type, (count, src_set, tgt_set))| {
            let mut source_types: Vec<String> = src_set.into_iter().collect();
            source_types.sort();
            let mut target_types: Vec<String> = tgt_set.into_iter().collect();
            target_types.sort();
            ConnectionTypeStats {
                connection_type: conn_type,
                count,
                source_types,
                target_types,
            }
        })
        .collect();
    result.sort_by(|a, b| a.connection_type.cmp(&b.connection_type));
    result
}

/// Full schema overview: node types, connection types, indexes, totals.
pub fn compute_schema(graph: &DirGraph) -> SchemaOverview {
    // Node types from type_indices
    let mut node_types: Vec<(String, NodeTypeOverview)> = graph
        .type_indices
        .iter()
        .map(|(nt, indices)| {
            let properties = graph
                .node_type_metadata
                .get(nt)
                .cloned()
                .unwrap_or_default();
            (
                nt.clone(),
                NodeTypeOverview {
                    count: indices.len(),
                    properties,
                },
            )
        })
        .collect();
    node_types.sort_by(|a, b| a.0.cmp(&b.0));

    // Connection types via edge scan
    let connection_types = compute_connection_type_stats(graph);

    // Indexes
    let mut indexes: Vec<String> = Vec::new();
    for (node_type, property) in graph.property_indices.keys() {
        indexes.push(format!("{}.{}", node_type, property));
    }
    for (node_type, properties) in graph.composite_indices.keys() {
        indexes.push(format!("{}.({})", node_type, properties.join(", ")));
    }
    for (node_type, property) in graph.range_indices.keys() {
        indexes.push(format!("{}.{} [range]", node_type, property));
    }
    indexes.sort();

    SchemaOverview {
        node_types,
        connection_types,
        indexes,
        node_count: graph.graph.node_count(),
        edge_count: graph.graph.edge_count(),
    }
}

fn is_null_value(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Float64(f) => f.is_nan(),
        _ => false,
    }
}

fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "str",
        Value::Int64(_) => "int",
        Value::Float64(_) => "float",
        Value::Boolean(_) => "bool",
        Value::DateTime(_) => "datetime",
        Value::UniqueId(_) => "uniqueid",
        Value::Point { .. } => "point",
        Value::Null => "unknown",
    }
}

/// Compact display string for a Value (used in agent description `vals` attributes).
/// Truncates long strings to keep output concise.
fn value_display_compact(v: &Value) -> String {
    match v {
        Value::String(s) => {
            if s.chars().count() > 40 {
                let truncated: String = s.chars().take(37).collect();
                format!("{}...", truncated)
            } else {
                s.clone()
            }
        }
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => format!("{}", f),
        Value::Boolean(b) => {
            if *b {
                "true"
            } else {
                "false"
            }
        }
        .to_string(),
        Value::DateTime(d) => d.to_string(),
        Value::UniqueId(u) => u.to_string(),
        Value::Point { lat, lon } => format!("({},{})", lat, lon),
        Value::Null => String::new(),
    }
}

/// Property stats for one node type. Scans all nodes of that type.
/// `max_values`: include `values` list when unique count ≤ this threshold (0 = never).
pub fn compute_property_stats(
    graph: &DirGraph,
    node_type: &str,
    max_values: usize,
) -> Result<Vec<PropertyStatInfo>, String> {
    let node_indices = graph
        .type_indices
        .get(node_type)
        .ok_or_else(|| format!("Node type '{}' not found", node_type))?;

    let total_nodes = node_indices.len();

    // First pass: discover all property names from actual nodes
    let mut all_props: HashSet<String> = HashSet::new();
    for &idx in node_indices {
        if let Some(node) = graph.get_node(idx) {
            for key in node.properties.keys() {
                all_props.insert(key.clone());
            }
        }
    }

    // Build ordered property list: built-ins first, then discovered from actual nodes
    // (metadata-only properties with no actual values are excluded)
    let mut property_names: Vec<String> =
        vec!["type".to_string(), "title".to_string(), "id".to_string()];
    // Add discovered properties (sorted for determinism)
    let mut discovered: Vec<String> = all_props.into_iter().collect();
    discovered.sort();
    property_names.extend(discovered);

    let mut results = Vec::new();

    for prop_name in &property_names {
        // Handle "type" specially — always same value, always non-null
        if prop_name == "type" {
            results.push(PropertyStatInfo {
                property_name: "type".to_string(),
                type_string: "str".to_string(),
                non_null: total_nodes,
                unique: 1,
                values: Some(vec![Value::String(node_type.to_string())]),
            });
            continue;
        }

        let mut non_null: usize = 0;
        let mut value_set: HashSet<Value> = HashSet::new();
        let mut first_type: Option<&'static str> = None;

        for &idx in node_indices {
            if let Some(node) = graph.get_node(idx) {
                let val = match prop_name.as_str() {
                    "id" => Some(&node.id),
                    "title" => Some(&node.title),
                    _ => node.properties.get(prop_name),
                };

                if let Some(v) = val {
                    if !is_null_value(v) {
                        non_null += 1;
                        value_set.insert(v.clone());
                        if first_type.is_none() {
                            first_type = Some(value_type_name(v));
                        }
                    }
                }
            }
        }

        // Determine type string: prefer metadata, fallback to inferred
        let type_string = graph
            .node_type_metadata
            .get(node_type)
            .and_then(|meta| meta.get(prop_name))
            .cloned()
            .unwrap_or_else(|| first_type.unwrap_or("unknown").to_string());

        let unique = value_set.len();
        let values = if max_values > 0 && unique <= max_values && unique > 0 {
            let mut vals: Vec<Value> = value_set.into_iter().collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Some(vals)
        } else {
            None
        };

        results.push(PropertyStatInfo {
            property_name: prop_name.clone(),
            type_string,
            non_null,
            unique,
            values,
        });
    }

    Ok(results)
}

/// Connection topology for one node type: outgoing and incoming grouped by (conn_type, other_type).
pub fn compute_neighbors_schema(
    graph: &DirGraph,
    node_type: &str,
) -> Result<NeighborsSchema, String> {
    let node_indices = graph
        .type_indices
        .get(node_type)
        .ok_or_else(|| format!("Node type '{}' not found", node_type))?;

    let mut outgoing: HashMap<(String, String), usize> = HashMap::new();
    let mut incoming: HashMap<(String, String), usize> = HashMap::new();

    for &node_idx in node_indices {
        for edge_ref in graph.graph.edges_directed(node_idx, Direction::Outgoing) {
            if let Some(target_node) = graph.get_node(edge_ref.target()) {
                let key = (
                    edge_ref.weight().connection_type.clone(),
                    target_node.node_type.clone(),
                );
                *outgoing.entry(key).or_insert(0) += 1;
            }
        }
        for edge_ref in graph.graph.edges_directed(node_idx, Direction::Incoming) {
            if let Some(source_node) = graph.get_node(edge_ref.source()) {
                let key = (
                    edge_ref.weight().connection_type.clone(),
                    source_node.node_type.clone(),
                );
                *incoming.entry(key).or_insert(0) += 1;
            }
        }
    }

    let mut outgoing_list: Vec<NeighborConnection> = outgoing
        .into_iter()
        .map(|((ct, ot), count)| NeighborConnection {
            connection_type: ct,
            other_type: ot,
            count,
        })
        .collect();
    outgoing_list.sort_by(|a, b| {
        (&a.connection_type, &a.other_type).cmp(&(&b.connection_type, &b.other_type))
    });

    let mut incoming_list: Vec<NeighborConnection> = incoming
        .into_iter()
        .map(|((ct, ot), count)| NeighborConnection {
            connection_type: ct,
            other_type: ot,
            count,
        })
        .collect();
    incoming_list.sort_by(|a, b| {
        (&a.connection_type, &a.other_type).cmp(&(&b.connection_type, &b.other_type))
    });

    Ok(NeighborsSchema {
        outgoing: outgoing_list,
        incoming: incoming_list,
    })
}

/// Return first N nodes of a type for quick inspection.
pub fn compute_sample<'a>(
    graph: &'a DirGraph,
    node_type: &str,
    n: usize,
) -> Result<Vec<&'a NodeData>, String> {
    let node_indices = graph
        .type_indices
        .get(node_type)
        .ok_or_else(|| format!("Node type '{}' not found", node_type))?;

    let mut result = Vec::with_capacity(n.min(node_indices.len()));
    for &idx in node_indices.iter().take(n) {
        if let Some(node) = graph.get_node(idx) {
            result.push(node);
        }
    }
    Ok(result)
}

// ── Describe: shared XML writers ────────────────────────────────────────────

/// Write the `<conventions>` element.
fn write_conventions(xml: &mut String, caps: &HashMap<String, TypeCapabilities>) {
    let mut specials: Vec<&str> = Vec::new();
    if caps.values().any(|c| c.has_location) {
        specials.push("location");
    }
    if caps.values().any(|c| c.has_geometry) {
        specials.push("geometry");
    }
    if caps.values().any(|c| c.has_timeseries) {
        specials.push("timeseries");
    }
    if caps.values().any(|c| c.has_embeddings) {
        specials.push("embeddings");
    }
    if specials.is_empty() {
        xml.push_str("  <conventions>All nodes have .id and .title</conventions>\n");
    } else {
        xml.push_str(&format!(
            "  <conventions>All nodes have .id and .title. Some have: {}</conventions>\n",
            specials.join(", ")
        ));
    }
}

/// Write a `<read-only>` element when the graph is in read-only mode.
fn write_read_only_notice(xml: &mut String, graph: &DirGraph) {
    if graph.read_only {
        xml.push_str(
            "  <read-only>Cypher mutations disabled: CREATE, SET, DELETE, REMOVE, MERGE</read-only>\n",
        );
    }
}

/// Write the `<connections>` element from global edge stats.
/// When `parent_types` is non-empty, filter out connections where ALL source types
/// are supporting children of the target type (the implicit OF_* pattern).
fn write_connection_map(xml: &mut String, graph: &DirGraph) {
    let conn_stats = compute_connection_type_stats(graph);
    let has_tiers = !graph.parent_types.is_empty();

    let filtered: Vec<&ConnectionTypeStats> = conn_stats
        .iter()
        .filter(|ct| {
            if !has_tiers {
                return true;
            }
            // Filter out connections where ALL sources are children of the single target
            if ct.target_types.len() == 1 {
                let target = &ct.target_types[0];
                let all_sources_are_children = ct.source_types.iter().all(|src| {
                    graph
                        .parent_types
                        .get(src)
                        .is_some_and(|parent| parent == target)
                });
                if all_sources_are_children {
                    return false;
                }
            }
            true
        })
        .collect();

    if filtered.is_empty() {
        xml.push_str("  <connections/>\n");
    } else {
        xml.push_str("  <connections>\n");
        for ct in &filtered {
            // When tiers are active, filter supporting types from source/target lists
            let sources: Vec<&str> = if has_tiers {
                ct.source_types
                    .iter()
                    .filter(|s| !graph.parent_types.contains_key(*s))
                    .map(|s| s.as_str())
                    .collect()
            } else {
                ct.source_types.iter().map(|s| s.as_str()).collect()
            };
            let targets: Vec<&str> = if has_tiers {
                ct.target_types
                    .iter()
                    .filter(|s| !graph.parent_types.contains_key(*s))
                    .map(|s| s.as_str())
                    .collect()
            } else {
                ct.target_types.iter().map(|s| s.as_str()).collect()
            };
            if sources.is_empty() || targets.is_empty() {
                continue;
            }
            xml.push_str(&format!(
                "    <conn type=\"{}\" count=\"{}\" from=\"{}\" to=\"{}\"/>\n",
                xml_escape(&ct.connection_type),
                ct.count,
                sources.join(","),
                targets.join(","),
            ));
        }
        xml.push_str("  </connections>\n");
    }
}

/// Compute property stats for edges of a given connection type.
fn compute_edge_property_stats(
    graph: &DirGraph,
    connection_type: &str,
    max_values: usize,
) -> Vec<PropertyStatInfo> {
    let mut all_props: HashSet<String> = HashSet::new();
    let mut total_edges: usize = 0;

    // First pass: discover property names
    for edge_ref in graph.graph.edge_references() {
        let ed = edge_ref.weight();
        if ed.connection_type == connection_type {
            total_edges += 1;
            for key in ed.properties.keys() {
                all_props.insert(key.clone());
            }
        }
    }

    if all_props.is_empty() {
        return Vec::new();
    }

    let mut prop_names: Vec<String> = all_props.into_iter().collect();
    prop_names.sort();

    let mut results = Vec::new();
    for prop_name in &prop_names {
        let mut non_null: usize = 0;
        let mut value_set: HashSet<Value> = HashSet::new();
        let mut first_type: Option<&'static str> = None;

        for edge_ref in graph.graph.edge_references() {
            let ed = edge_ref.weight();
            if ed.connection_type != connection_type {
                continue;
            }
            if let Some(v) = ed.properties.get(prop_name) {
                if !is_null_value(v) {
                    non_null += 1;
                    value_set.insert(v.clone());
                    if first_type.is_none() {
                        first_type = Some(value_type_name(v));
                    }
                }
            }
        }

        let unique = value_set.len();
        let values = if max_values > 0 && unique <= max_values && unique > 0 {
            let mut vals: Vec<Value> = value_set.into_iter().collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Some(vals)
        } else {
            None
        };

        results.push(PropertyStatInfo {
            property_name: prop_name.clone(),
            type_string: first_type.unwrap_or("unknown").to_string(),
            non_null,
            unique,
            values,
        });
    }
    let _ = total_edges; // used implicitly by context
    results
}

/// Connections overview: all connection types with count, endpoints, property names.
fn write_connections_overview(xml: &mut String, graph: &DirGraph) {
    let conn_stats = compute_connection_type_stats(graph);
    if conn_stats.is_empty() {
        xml.push_str("<connections/>\n");
        return;
    }
    xml.push_str("<connections>\n");
    for ct in &conn_stats {
        // Collect property names for this connection type
        let prop_names: Vec<String> = {
            let mut props: HashSet<String> = HashSet::new();
            for edge_ref in graph.graph.edge_references() {
                let ed = edge_ref.weight();
                if ed.connection_type == ct.connection_type {
                    for key in ed.properties.keys() {
                        props.insert(key.clone());
                    }
                }
            }
            let mut sorted: Vec<String> = props.into_iter().collect();
            sorted.sort();
            sorted
        };

        let props_attr = if prop_names.is_empty() {
            String::new()
        } else {
            format!(" properties=\"{}\"", xml_escape(&prop_names.join(",")))
        };

        xml.push_str(&format!(
            "  <conn type=\"{}\" count=\"{}\" from=\"{}\" to=\"{}\"{}/>\n",
            xml_escape(&ct.connection_type),
            ct.count,
            ct.source_types.join(","),
            ct.target_types.join(","),
            props_attr,
        ));
    }
    xml.push_str("</connections>\n");
}

/// Connections deep-dive: per-pair counts, property stats, sample edges.
fn write_connections_detail(
    xml: &mut String,
    graph: &DirGraph,
    topics: &[String],
) -> Result<(), String> {
    // Validate all connection types exist
    let conn_stats = compute_connection_type_stats(graph);
    let valid_types: HashSet<&str> = conn_stats
        .iter()
        .map(|c| c.connection_type.as_str())
        .collect();
    for topic in topics {
        if !valid_types.contains(topic.as_str()) {
            let mut available: Vec<&str> = valid_types.iter().copied().collect();
            available.sort();
            return Err(format!(
                "Connection type '{}' not found. Available: {}",
                topic,
                available.join(", ")
            ));
        }
    }

    xml.push_str("<connections>\n");
    for topic in topics {
        let ct = conn_stats
            .iter()
            .find(|c| c.connection_type == *topic)
            .unwrap();

        xml.push_str(&format!(
            "  <{} count=\"{}\">\n",
            xml_escape(&ct.connection_type),
            ct.count
        ));

        // Per source→target pair counts
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        for edge_ref in graph.graph.edge_references() {
            let ed = edge_ref.weight();
            if ed.connection_type != *topic {
                continue;
            }
            let src_type = graph
                .get_node(edge_ref.source())
                .map(|n| n.node_type.clone())
                .unwrap_or_default();
            let tgt_type = graph
                .get_node(edge_ref.target())
                .map(|n| n.node_type.clone())
                .unwrap_or_default();
            *pair_counts.entry((src_type, tgt_type)).or_insert(0) += 1;
        }
        let mut pairs: Vec<((String, String), usize)> = pair_counts.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));

        xml.push_str("    <endpoints>\n");
        for ((src, tgt), count) in &pairs {
            xml.push_str(&format!(
                "      <pair from=\"{}\" to=\"{}\" count=\"{}\"/>\n",
                xml_escape(src),
                xml_escape(tgt),
                count
            ));
        }
        xml.push_str("    </endpoints>\n");

        // Edge property stats
        let prop_stats = compute_edge_property_stats(graph, topic, 15);
        if !prop_stats.is_empty() {
            xml.push_str("    <properties>\n");
            for ps in &prop_stats {
                if ps.non_null == 0 {
                    continue;
                }
                let vals_attr = match &ps.values {
                    Some(vals) if !vals.is_empty() => {
                        let vals_str: Vec<String> =
                            vals.iter().map(value_display_compact).collect();
                        format!(" vals=\"{}\"", xml_escape(&vals_str.join("|")))
                    }
                    _ => String::new(),
                };
                xml.push_str(&format!(
                    "      <prop name=\"{}\" type=\"{}\" non_null=\"{}\" unique=\"{}\"{}/>\n",
                    xml_escape(&ps.property_name),
                    xml_escape(&ps.type_string),
                    ps.non_null,
                    ps.unique,
                    vals_attr,
                ));
            }
            xml.push_str("    </properties>\n");
        }

        // Sample edges (first 2)
        xml.push_str("    <samples>\n");
        let mut sample_count = 0;
        for edge_ref in graph.graph.edge_references() {
            let ed = edge_ref.weight();
            if ed.connection_type != *topic {
                continue;
            }
            if sample_count >= 2 {
                break;
            }
            let src_label = graph
                .get_node(edge_ref.source())
                .map(|n| format!("{}:{}", n.node_type, value_display_compact(&n.title)))
                .unwrap_or_default();
            let tgt_label = graph
                .get_node(edge_ref.target())
                .map(|n| format!("{}:{}", n.node_type, value_display_compact(&n.title)))
                .unwrap_or_default();

            let mut attrs = format!(
                "from=\"{}\" to=\"{}\"",
                xml_escape(&src_label),
                xml_escape(&tgt_label),
            );
            // Add up to 4 edge properties
            let mut prop_count = 0;
            let mut keys: Vec<&String> = ed.properties.keys().collect();
            keys.sort();
            for key in keys {
                if prop_count >= 4 {
                    break;
                }
                if let Some(v) = ed.properties.get(key) {
                    if !is_null_value(v) {
                        attrs.push_str(&format!(
                            " {}=\"{}\"",
                            xml_escape(key),
                            xml_escape(&value_display_compact(v))
                        ));
                        prop_count += 1;
                    }
                }
            }
            xml.push_str(&format!("      <edge {}/>\n", attrs));
            sample_count += 1;
        }
        xml.push_str("    </samples>\n");

        xml.push_str(&format!("  </{}>\n", xml_escape(&ct.connection_type)));
    }
    xml.push_str("</connections>\n");
    Ok(())
}

/// Write the `<extensions>` element — only sections the graph actually uses.
fn write_extensions(xml: &mut String, graph: &DirGraph) {
    let has_timeseries = !graph.timeseries_configs.is_empty();
    let has_spatial = !graph.spatial_configs.is_empty()
        || graph
            .node_type_metadata
            .values()
            .any(|props| props.values().any(|t| t.eq_ignore_ascii_case("point")));
    let has_embeddings = !graph.embeddings.is_empty();

    xml.push_str("  <extensions>\n");

    if has_timeseries {
        xml.push_str("    <timeseries hint=\"ts_avg(n.ch, start?, end?), ts_sum, ts_min, ts_max, ts_count, ts_first, ts_last, ts_delta, ts_at, ts_series — date args: 'YYYY', 'YYYY-M', 'YYYY-M-D' or DateTime properties. NaN skipped.\"/>\n");
    }
    if has_spatial {
        xml.push_str("    <spatial hint=\"distance(a,b)→m, contains(a,b), intersects(a,b), centroid(n), area(n)→m², perimeter(n)→m\"/>\n");
    }
    if has_embeddings {
        xml.push_str(
            "    <semantic hint=\"text_score(n, 'col', 'query text') — similarity 0..1\"/>\n",
        );
    }
    xml.push_str("    <algorithms hint=\"CALL pagerank/betweenness/degree/closeness/louvain/label_propagation/connected_components/cluster({params}) YIELD node, score|community|cluster\"/>\n");
    xml.push_str("    <cypher hint=\"Full Cypher with extensions: ||, =~, coalesce(), CALL cluster/pagerank/louvain/..., distance(), contains(). describe(cypher=True) for reference, describe(cypher=['topic']) for detailed docs.\"/>\n");
    if graph.graph.edge_count() > 0 {
        xml.push_str("    <connections hint=\"describe(connections=True) for all connection types, describe(connections=['TYPE']) for deep-dive with properties and samples.\"/>\n");
    }
    xml.push_str("    <bug_report hint=\"bug_report(query, result, expected, description) — file a Cypher bug report to reported_bugs.md.\"/>\n");
    xml.push_str("  </extensions>\n");
}

/// Tier 2: compact Cypher reference — all clauses, operators, functions, procedures.
/// No examples. Ends with hint to use tier 3.
fn write_cypher_overview(xml: &mut String) {
    xml.push_str("<cypher>\n");

    // Clauses
    xml.push_str("  <clauses>\n");
    xml.push_str("    <clause name=\"MATCH\">Pattern-match nodes and relationships. OPTIONAL MATCH for left-join semantics.</clause>\n");
    xml.push_str("    <clause name=\"WHERE\">Filter by predicate (comparison, null check, regex, string predicates).</clause>\n");
    xml.push_str("    <clause name=\"RETURN\">Project columns. Supports DISTINCT, aliases (AS), aggregations.</clause>\n");
    xml.push_str("    <clause name=\"WITH\">Intermediate projection, aggregation, and variable scoping.</clause>\n");
    xml.push_str("    <clause name=\"ORDER BY\">Sort results. Append DESC for descending. Combine with SKIP n, LIMIT n.</clause>\n");
    xml.push_str("    <clause name=\"UNWIND\">Expand a list into individual rows: UNWIND expr AS var.</clause>\n");
    xml.push_str(
        "    <clause name=\"UNION\">Combine result sets. UNION ALL keeps duplicates.</clause>\n",
    );
    xml.push_str("    <clause name=\"CASE\">Conditional expression: CASE WHEN cond THEN val ... ELSE val END.</clause>\n");
    xml.push_str(
        "    <clause name=\"CREATE\">Create nodes and relationships with properties.</clause>\n",
    );
    xml.push_str("    <clause name=\"SET\">Set or update node/relationship properties.</clause>\n");
    xml.push_str("    <clause name=\"DELETE\">Delete nodes/relationships. REMOVE to drop individual properties.</clause>\n");
    xml.push_str(
        "    <clause name=\"MERGE\">Match existing or create new (upsert pattern).</clause>\n",
    );
    xml.push_str("  </clauses>\n");

    // Operators
    xml.push_str("  <operators>\n");
    xml.push_str("    <group name=\"math\">+ - * /</group>\n");
    xml.push_str("    <group name=\"string\">|| (concatenation)</group>\n");
    xml.push_str("    <group name=\"comparison\">= &lt;&gt; &lt; &gt; &lt;= &gt;= IN</group>\n");
    xml.push_str("    <group name=\"logical\">AND OR NOT XOR</group>\n");
    xml.push_str("    <group name=\"null\">IS NULL, IS NOT NULL</group>\n");
    xml.push_str("    <group name=\"regex\">=~ 'pattern'</group>\n");
    xml.push_str("    <group name=\"predicates\">CONTAINS, STARTS WITH, ENDS WITH</group>\n");
    xml.push_str("  </operators>\n");

    // Functions
    xml.push_str("  <functions>\n");
    xml.push_str("    <group name=\"math\">abs, ceil, floor, round(x [,decimals]), sqrt, sign, toInteger, toFloat</group>\n");
    xml.push_str("    <group name=\"string\">toString, toUpper, toLower, trim, lTrim, rTrim, replace, substring, left, right, split, reverse</group>\n");
    xml.push_str(
        "    <group name=\"aggregate\">count, sum, avg, min, max, collect, stDev</group>\n",
    );
    xml.push_str(
        "    <group name=\"graph\">size, length, id, labels, type, coalesce, range, keys</group>\n",
    );
    xml.push_str("    <group name=\"spatial\">distance(a,b)→m, contains(a,b), intersects(a,b), centroid(n), area(n)→m², perimeter(n)→m</group>\n");
    xml.push_str("  </functions>\n");

    // Procedures
    xml.push_str("  <procedures>\n");
    xml.push_str("    <proc name=\"pagerank\" yields=\"node, score\">PageRank centrality for all nodes.</proc>\n");
    xml.push_str("    <proc name=\"betweenness\" yields=\"node, score\">Betweenness centrality for all nodes.</proc>\n");
    xml.push_str("    <proc name=\"degree\" yields=\"node, score\">Degree centrality for all nodes.</proc>\n");
    xml.push_str("    <proc name=\"closeness\" yields=\"node, score\">Closeness centrality for all nodes.</proc>\n");
    xml.push_str("    <proc name=\"louvain\" yields=\"node, community\">Community detection (Louvain algorithm).</proc>\n");
    xml.push_str("    <proc name=\"label_propagation\" yields=\"node, community\">Community detection (label propagation).</proc>\n");
    xml.push_str("    <proc name=\"connected_components\" yields=\"node, component\">Weakly connected components.</proc>\n");
    xml.push_str("    <proc name=\"cluster\" yields=\"node, cluster\">DBSCAN/K-means clustering on spatial or property data.</proc>\n");
    xml.push_str("  </procedures>\n");

    // Patterns
    xml.push_str("  <patterns>(n:Label), (n {prop: val}), (a)-[:TYPE]-&gt;(b), (a)-[:T*1..3]-&gt;(b), [x IN list WHERE pred | expr], n {.p1, .p2}</patterns>\n");

    xml.push_str("  <not_supported>CALL {} subqueries, FOREACH, CREATE INDEX, shortestPath(), variable-length weighted paths</not_supported>\n");
    xml.push_str("  <hint>Use describe(cypher=['MATCH','cluster','spatial',...]) for detailed docs with examples.</hint>\n");
    xml.push_str("</cypher>\n");
}

// ── Cypher tier 3: topic detail functions ──────────────────────────────────

const CYPHER_TOPIC_LIST: &str = "MATCH, WHERE, RETURN, WITH, ORDER BY, UNWIND, UNION, \
    CASE, CREATE, SET, DELETE, MERGE, operators, functions, patterns, spatial, \
    pagerank, betweenness, degree, closeness, louvain, \
    label_propagation, connected_components, cluster";

/// Tier 3: detailed Cypher docs for specific topics with params and examples.
fn write_cypher_topics(xml: &mut String, topics: &[String]) -> Result<(), String> {
    // Empty list → tier 2 overview
    if topics.is_empty() {
        write_cypher_overview(xml);
        return Ok(());
    }

    xml.push_str("<cypher>\n");
    for topic in topics {
        let key = topic.to_uppercase();
        match key.as_str() {
            "MATCH" => write_topic_match(xml),
            "WHERE" => write_topic_where(xml),
            "RETURN" => write_topic_return(xml),
            "WITH" => write_topic_with(xml),
            "ORDER BY" | "ORDERBY" | "ORDER_BY" => write_topic_order_by(xml),
            "UNWIND" => write_topic_unwind(xml),
            "UNION" => write_topic_union(xml),
            "CASE" => write_topic_case(xml),
            "CREATE" => write_topic_create(xml),
            "SET" => write_topic_set(xml),
            "DELETE" | "REMOVE" => write_topic_delete(xml),
            "MERGE" => write_topic_merge(xml),
            "OPERATORS" => write_topic_operators(xml),
            "FUNCTIONS" => write_topic_functions(xml),
            "PATTERNS" => write_topic_patterns(xml),
            "PAGERANK" => write_topic_pagerank(xml),
            "BETWEENNESS" => write_topic_betweenness(xml),
            "DEGREE" => write_topic_degree(xml),
            "CLOSENESS" => write_topic_closeness(xml),
            "LOUVAIN" => write_topic_louvain(xml),
            "LABEL_PROPAGATION" | "LABELPROPAGATION" => write_topic_label_propagation(xml),
            "CONNECTED_COMPONENTS" | "CONNECTEDCOMPONENTS" => {
                write_topic_connected_components(xml);
            }
            "CLUSTER" => write_topic_cluster(xml),
            "SPATIAL" => write_topic_spatial(xml),
            _ => {
                return Err(format!(
                    "Unknown Cypher topic '{}'. Available: {}",
                    topic, CYPHER_TOPIC_LIST
                ));
            }
        }
    }
    xml.push_str("</cypher>\n");
    Ok(())
}

fn write_topic_match(xml: &mut String) {
    xml.push_str("  <MATCH>\n");
    xml.push_str("    <desc>Pattern-match nodes and relationships. OPTIONAL MATCH returns nulls for non-matching patterns (left join).</desc>\n");
    xml.push_str("    <syntax>MATCH (n:Label {prop: val})-[r:TYPE]-&gt;(m)</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"all nodes of type\">MATCH (n:Field) RETURN n.name</ex>\n");
    xml.push_str("      <ex desc=\"with relationship\">MATCH (a:Person)-[:KNOWS]-&gt;(b) RETURN a.name, b.name</ex>\n");
    xml.push_str("      <ex desc=\"variable-length path\">MATCH (a)-[:KNOWS*1..3]-&gt;(b) RETURN a, b</ex>\n");
    xml.push_str("      <ex desc=\"inline property filter\">MATCH (n:Field {status: 'active'}) RETURN n</ex>\n");
    xml.push_str("      <ex desc=\"optional match\">MATCH (a:Field) OPTIONAL MATCH (a)-[:HAS]-&gt;(b:Well) RETURN a.name, b.name</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </MATCH>\n");
}

fn write_topic_where(xml: &mut String) {
    xml.push_str("  <WHERE>\n");
    xml.push_str("    <desc>Filter results by predicate. Supports comparison, null checks, regex, string predicates, boolean logic.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"comparison\">WHERE n.depth &gt; 3000</ex>\n");
    xml.push_str("      <ex desc=\"string contains\">WHERE n.name CONTAINS 'oil'</ex>\n");
    xml.push_str("      <ex desc=\"starts/ends with\">WHERE n.name STARTS WITH '35/'</ex>\n");
    xml.push_str("      <ex desc=\"regex\">WHERE n.name =~ '35/9-.*'</ex>\n");
    xml.push_str("      <ex desc=\"null check\">WHERE n.depth IS NOT NULL</ex>\n");
    xml.push_str("      <ex desc=\"IN list\">WHERE n.status IN ['active', 'planned']</ex>\n");
    xml.push_str("      <ex desc=\"boolean\">WHERE n.depth &gt; 1000 AND n.temp &lt; 100</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </WHERE>\n");
}

fn write_topic_return(xml: &mut String) {
    xml.push_str("  <RETURN>\n");
    xml.push_str("    <desc>Project columns to output. Supports DISTINCT, aliases (AS), expressions, aggregations.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">RETURN n.name, n.depth</ex>\n");
    xml.push_str("      <ex desc=\"alias\">RETURN n.name AS field_name</ex>\n");
    xml.push_str("      <ex desc=\"distinct\">RETURN DISTINCT n.status</ex>\n");
    xml.push_str(
        "      <ex desc=\"expression\">RETURN n.name || ' (' || n.status || ')' AS label</ex>\n",
    );
    xml.push_str("      <ex desc=\"aggregation\">RETURN n.status, count(*) AS n, collect(n.name) AS names</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </RETURN>\n");
}

fn write_topic_with(xml: &mut String) {
    xml.push_str("  <WITH>\n");
    xml.push_str("    <desc>Intermediate projection and aggregation. Creates a new scope — only variables listed in WITH are available in subsequent clauses.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"filter after aggregation\">MATCH (n:Field) WITH n.area AS area, count(*) AS c WHERE c &gt; 5 RETURN area, c</ex>\n");
    xml.push_str("      <ex desc=\"pipe between matches\">MATCH (a:Field) WITH a MATCH (a)-[:HAS]-&gt;(b) RETURN a.name, b.name</ex>\n");
    xml.push_str("      <ex desc=\"limit intermediate\">MATCH (n:Field) WITH n ORDER BY n.name LIMIT 10 RETURN n.name</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </WITH>\n");
}

fn write_topic_order_by(xml: &mut String) {
    xml.push_str("  <ORDER_BY>\n");
    xml.push_str("    <desc>Sort results. Default ascending; append DESC for descending. Combine with SKIP and LIMIT for pagination.</desc>\n");
    xml.push_str("    <syntax>ORDER BY expr [DESC] [SKIP n] [LIMIT n]</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"ascending\">ORDER BY n.name</ex>\n");
    xml.push_str("      <ex desc=\"descending\">ORDER BY n.depth DESC</ex>\n");
    xml.push_str("      <ex desc=\"pagination\">ORDER BY n.name SKIP 20 LIMIT 10</ex>\n");
    xml.push_str("      <ex desc=\"multi-key\">ORDER BY n.status, n.name DESC</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </ORDER_BY>\n");
}

fn write_topic_unwind(xml: &mut String) {
    xml.push_str("  <UNWIND>\n");
    xml.push_str("    <desc>Expand a list expression into individual rows. Each element becomes a new row bound to the alias.</desc>\n");
    xml.push_str("    <syntax>UNWIND expression AS variable</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"literal list\">UNWIND ['A','B','C'] AS x MATCH (n {code: x}) RETURN n</ex>\n");
    xml.push_str("      <ex desc=\"collected list\">MATCH (n:Field) WITH collect(n.name) AS names UNWIND names AS name RETURN name</ex>\n");
    xml.push_str("      <ex desc=\"range\">UNWIND range(1, 10) AS i RETURN i</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </UNWIND>\n");
}

fn write_topic_union(xml: &mut String) {
    xml.push_str("  <UNION>\n");
    xml.push_str("    <desc>Combine result sets from two queries. UNION removes duplicates; UNION ALL keeps all rows. Column names must match.</desc>\n");
    xml.push_str("    <syntax>query1 UNION [ALL] query2</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic union\">MATCH (a:Field) RETURN a.name AS name UNION MATCH (b:Discovery) RETURN b.name AS name</ex>\n");
    xml.push_str("      <ex desc=\"union all\">MATCH (a:Field) RETURN a.name AS name UNION ALL MATCH (b:Field) RETURN b.name AS name</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </UNION>\n");
}

fn write_topic_case(xml: &mut String) {
    xml.push_str("  <CASE>\n");
    xml.push_str("    <desc>Conditional expression. Two forms: simple (CASE expr WHEN val THEN ...) and generic (CASE WHEN cond THEN ...).</desc>\n");
    xml.push_str("    <syntax>CASE WHEN condition THEN value [WHEN ... THEN ...] [ELSE default] END</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"generic\">RETURN CASE WHEN n.depth &gt; 3000 THEN 'deep' WHEN n.depth &gt; 1000 THEN 'medium' ELSE 'shallow' END AS category</ex>\n");
    xml.push_str("      <ex desc=\"simple\">RETURN CASE n.status WHEN 'PRODUCING' THEN 'active' WHEN 'SHUT DOWN' THEN 'closed' ELSE 'other' END</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </CASE>\n");
}

fn write_topic_create(xml: &mut String) {
    xml.push_str("  <CREATE>\n");
    xml.push_str("    <desc>Create new nodes and relationships with properties.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"node\">CREATE (:Field {name: 'Troll', status: 'PRODUCING'})</ex>\n",
    );
    xml.push_str("      <ex desc=\"relationship\">MATCH (a:Field {name: 'Troll'}), (b:Company {name: 'Equinor'}) CREATE (a)-[:OPERATED_BY]-&gt;(b)</ex>\n");
    xml.push_str("      <ex desc=\"with properties\">MATCH (a:Field), (b:Well) WHERE a.name = b.field CREATE (b)-[:BELONGS_TO {since: 2020}]-&gt;(a)</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </CREATE>\n");
}

fn write_topic_set(xml: &mut String) {
    xml.push_str("  <SET>\n");
    xml.push_str("    <desc>Set or update properties on existing nodes/relationships.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"set property\">MATCH (n:Field {name: 'Troll'}) SET n.status = 'SHUT DOWN'</ex>\n");
    xml.push_str("      <ex desc=\"set multiple\">MATCH (n:Field {name: 'Troll'}) SET n.status = 'SHUT DOWN', n.end_year = 2025</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </SET>\n");
}

fn write_topic_delete(xml: &mut String) {
    xml.push_str("  <DELETE>\n");
    xml.push_str("    <desc>Delete nodes or relationships. REMOVE drops individual properties from a node.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"delete node\">MATCH (n:Field {name: 'Test'}) DELETE n</ex>\n");
    xml.push_str(
        "      <ex desc=\"delete relationship\">MATCH (a)-[r:OLD_REL]-&gt;(b) DELETE r</ex>\n",
    );
    xml.push_str("      <ex desc=\"remove property\">MATCH (n:Field {name: 'Troll'}) REMOVE n.temp_flag</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </DELETE>\n");
}

fn write_topic_merge(xml: &mut String) {
    xml.push_str("  <MERGE>\n");
    xml.push_str("    <desc>Match existing node/relationship or create if it doesn't exist (upsert). ON CREATE SET and ON MATCH SET for conditional property updates.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">MERGE (n:Field {name: 'Troll'})</ex>\n");
    xml.push_str("      <ex desc=\"on create\">MERGE (n:Field {name: 'Troll'}) ON CREATE SET n.created = 2025</ex>\n");
    xml.push_str("      <ex desc=\"on match\">MERGE (n:Field {name: 'Troll'}) ON MATCH SET n.updated = 2025</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </MERGE>\n");
}

fn write_topic_operators(xml: &mut String) {
    xml.push_str("  <operators>\n");
    xml.push_str("    <desc>All supported operators with semantics.</desc>\n");
    xml.push_str("    <group name=\"math\" desc=\"Arithmetic\">+ (add), - (subtract), * (multiply), / (divide)</group>\n");
    xml.push_str("    <group name=\"string\" desc=\"String concatenation\">|| — null propagates: 'a' || null = null. Auto-converts numbers: 'v' || 42 = 'v42'.</group>\n");
    xml.push_str("    <group name=\"comparison\" desc=\"Comparison\">= (equal), &lt;&gt; (not equal), &lt;, &gt;, &lt;=, &gt;=, IN (list membership)</group>\n");
    xml.push_str("    <group name=\"logical\" desc=\"Boolean\">AND, OR, NOT, XOR</group>\n");
    xml.push_str("    <group name=\"null\" desc=\"Null checks\">IS NULL, IS NOT NULL</group>\n");
    xml.push_str("    <group name=\"regex\" desc=\"Regex match\">=~ 'pattern' — Java-style regex, case-sensitive by default. Use (?i) for case-insensitive.</group>\n");
    xml.push_str("    <group name=\"predicates\" desc=\"String predicates\">CONTAINS, STARTS WITH, ENDS WITH — case-sensitive substring checks.</group>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"concat with number\">RETURN n.name || '-' || n.block AS label</ex>\n",
    );
    xml.push_str("      <ex desc=\"regex case-insensitive\">WHERE n.name =~ '(?i)troll.*'</ex>\n");
    xml.push_str("      <ex desc=\"IN list\">WHERE n.status IN ['PRODUCING', 'SHUT DOWN']</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </operators>\n");
}

fn write_topic_functions(xml: &mut String) {
    xml.push_str("  <functions>\n");
    xml.push_str("    <desc>All built-in functions grouped by category.</desc>\n");
    xml.push_str("    <group name=\"math\">abs(x), ceil(x)/ceiling(x), floor(x), round(x [,decimals]), sqrt(x), sign(x), toInteger(x)/toInt(x), toFloat(x)</group>\n");
    xml.push_str("    <group name=\"string\">toString(x), toUpper(s), toLower(s), trim(s), lTrim(s), rTrim(s), replace(s,from,to), substring(s,start[,len]), left(s,n), right(s,n), split(s,delim), reverse(s), size(s)</group>\n");
    xml.push_str("    <group name=\"aggregate\">count(*)/count(expr), sum(expr), avg(expr), min(expr), max(expr), collect(expr), stDev(expr)/std(expr)</group>\n");
    xml.push_str("    <group name=\"graph\">size(list), length(path), id(node), labels(node), type(rel), coalesce(expr,...) — first non-null, range(start,end[,step]), keys(node)</group>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"round precision\">RETURN round(n.depth / 1000.0, 1) AS depth_km</ex>\n",
    );
    xml.push_str("      <ex desc=\"coalesce\">RETURN coalesce(n.nickname, n.name) AS label</ex>\n");
    xml.push_str("      <ex desc=\"string\">RETURN toLower(n.name) AS lower_name</ex>\n");
    xml.push_str("      <ex desc=\"aggregate\">RETURN n.status, count(*) AS n, avg(n.depth) AS avg_depth</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </functions>\n");
}

fn write_topic_patterns(xml: &mut String) {
    xml.push_str("  <patterns>\n");
    xml.push_str("    <desc>Pattern syntax for matching graph structures.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"labeled node\">(n:Field)</ex>\n");
    xml.push_str("      <ex desc=\"inline properties\">(n:Field {status: 'active'})</ex>\n");
    xml.push_str("      <ex desc=\"directed relationship\">(a)-[:BELONGS_TO]-&gt;(b)</ex>\n");
    xml.push_str(
        "      <ex desc=\"variable-length\">(a)-[:KNOWS*1..3]-&gt;(b) — path length 1 to 3</ex>\n",
    );
    xml.push_str("      <ex desc=\"any relationship\">(a)--&gt;(b) or (a)-[r]-&gt;(b)</ex>\n");
    xml.push_str("      <ex desc=\"list comprehension\">[x IN collect(n.name) WHERE x STARTS WITH '35']</ex>\n");
    xml.push_str("      <ex desc=\"map projection\">n {.name, .status} — returns {name: ..., status: ...}</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </patterns>\n");
}

// ── Procedure deep-dive functions ──────────────────────────────────────────

fn write_topic_pagerank(xml: &mut String) {
    xml.push_str("  <pagerank>\n");
    xml.push_str("    <desc>Compute PageRank centrality for all nodes. Higher score = more influential.</desc>\n");
    xml.push_str("    <syntax>CALL pagerank({params}) YIELD node, score</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"damping_factor\" type=\"float\" default=\"0.85\">Probability of following a link vs random jump.</param>\n");
    xml.push_str("      <param name=\"max_iterations\" type=\"int\" default=\"100\">Convergence iteration limit.</param>\n");
    xml.push_str("      <param name=\"tolerance\" type=\"float\" default=\"1e-6\">Convergence threshold.</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL pagerank() YIELD node, score RETURN node.name, score ORDER BY score DESC LIMIT 10</ex>\n");
    xml.push_str("      <ex desc=\"filtered\">CALL pagerank({connection_types: 'CITES'}) YIELD node, score RETURN node.name, score ORDER BY score DESC</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </pagerank>\n");
}

fn write_topic_betweenness(xml: &mut String) {
    xml.push_str("  <betweenness>\n");
    xml.push_str("    <desc>Compute betweenness centrality. High score = node lies on many shortest paths (bridge/broker).</desc>\n");
    xml.push_str("    <syntax>CALL betweenness({params}) YIELD node, score</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"normalized\" type=\"bool\" default=\"true\">Normalize scores to 0..1 range.</param>\n");
    xml.push_str("      <param name=\"sample_size\" type=\"int\" optional=\"true\">Approximate by sampling N source nodes (faster for large graphs).</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL betweenness() YIELD node, score RETURN node.name, score ORDER BY score DESC LIMIT 10</ex>\n");
    xml.push_str("      <ex desc=\"sampled\">CALL betweenness({sample_size: 100}) YIELD node, score RETURN node.name, round(score, 4) ORDER BY score DESC</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </betweenness>\n");
}

fn write_topic_degree(xml: &mut String) {
    xml.push_str("  <degree>\n");
    xml.push_str("    <desc>Compute degree centrality (number of connections per node, optionally normalized).</desc>\n");
    xml.push_str("    <syntax>CALL degree({params}) YIELD node, score</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"normalized\" type=\"bool\" default=\"true\">Normalize by max possible degree.</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL degree() YIELD node, score RETURN node.name, score ORDER BY score DESC LIMIT 10</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </degree>\n");
}

fn write_topic_closeness(xml: &mut String) {
    xml.push_str("  <closeness>\n");
    xml.push_str("    <desc>Compute closeness centrality (inverse of average shortest path distance). High = close to all others.</desc>\n");
    xml.push_str("    <syntax>CALL closeness({params}) YIELD node, score</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"normalized\" type=\"bool\" default=\"true\">Normalize scores.</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL closeness() YIELD node, score RETURN node.name, score ORDER BY score DESC LIMIT 10</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </closeness>\n");
}

fn write_topic_louvain(xml: &mut String) {
    xml.push_str("  <louvain>\n");
    xml.push_str("    <desc>Community detection using the Louvain algorithm. Assigns each node a community ID.</desc>\n");
    xml.push_str("    <syntax>CALL louvain({params}) YIELD node, community</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"resolution\" type=\"float\" default=\"1.0\">Higher = more/smaller communities, lower = fewer/larger.</param>\n");
    xml.push_str("      <param name=\"weight_property\" type=\"string\" optional=\"true\">Edge property to use as weight.</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL louvain() YIELD node, community RETURN community, count(*) AS size, collect(node.name) AS members ORDER BY size DESC</ex>\n");
    xml.push_str("      <ex desc=\"high resolution\">CALL louvain({resolution: 2.0}) YIELD node, community RETURN community, count(*) AS size ORDER BY size DESC</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </louvain>\n");
}

fn write_topic_label_propagation(xml: &mut String) {
    xml.push_str("  <label_propagation>\n");
    xml.push_str("    <desc>Community detection using label propagation. Fast, non-deterministic. Each node adopts its neighbors' majority label.</desc>\n");
    xml.push_str("    <syntax>CALL label_propagation({params}) YIELD node, community</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"max_iterations\" type=\"int\" default=\"100\">Iteration limit.</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL label_propagation() YIELD node, community RETURN community, count(*) AS size ORDER BY size DESC</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </label_propagation>\n");
}

fn write_topic_connected_components(xml: &mut String) {
    xml.push_str("  <connected_components>\n");
    xml.push_str("    <desc>Find weakly connected components. Nodes in the same component can reach each other ignoring edge direction.</desc>\n");
    xml.push_str("    <syntax>CALL connected_components() YIELD node, component</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL connected_components() YIELD node, component RETURN component, count(*) AS size ORDER BY size DESC</ex>\n");
    xml.push_str("      <ex desc=\"find isolated\">CALL connected_components() YIELD node, component WITH component, count(*) AS size WHERE size = 1 RETURN count(*) AS isolated_nodes</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </connected_components>\n");
}

fn write_topic_cluster(xml: &mut String) {
    xml.push_str("  <cluster>\n");
    xml.push_str("    <desc>Cluster nodes using DBSCAN or K-means. Reads nodes from preceding MATCH clause.</desc>\n");
    xml.push_str("    <syntax>MATCH (n:Type) CALL cluster({params}) YIELD node, cluster RETURN ...</syntax>\n");
    xml.push_str("    <modes>\n");
    xml.push_str("      <spatial>Omit 'properties' — auto-detects lat/lon from set_spatial() config. Uses haversine distance. eps is in meters. Geometry centroids used as fallback for WKT types.</spatial>\n");
    xml.push_str("      <property>Specify properties: ['col1','col2'] — euclidean distance on numeric values. Use normalize: true when feature scales differ.</property>\n");
    xml.push_str("    </modes>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"method\" type=\"string\" default=\"dbscan\">'dbscan' or 'kmeans'.</param>\n");
    xml.push_str("      <param name=\"eps\" type=\"float\" default=\"0.5\">DBSCAN: max neighborhood distance. In meters for spatial mode.</param>\n");
    xml.push_str("      <param name=\"min_points\" type=\"int\" default=\"3\">DBSCAN: min neighbors to form a core point.</param>\n");
    xml.push_str(
        "      <param name=\"k\" type=\"int\" default=\"5\">K-means: number of clusters.</param>\n",
    );
    xml.push_str("      <param name=\"max_iterations\" type=\"int\" default=\"100\">K-means: iteration limit.</param>\n");
    xml.push_str("      <param name=\"normalize\" type=\"bool\" default=\"false\">Property mode: scale features to [0,1] before clustering.</param>\n");
    xml.push_str("      <param name=\"properties\" type=\"list\" optional=\"true\">Numeric property names for property mode. Omit for spatial mode.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <yields>node (the matched node), cluster (int — cluster ID; -1 = noise for DBSCAN)</yields>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"spatial DBSCAN\">MATCH (f:Field) CALL cluster({method: 'dbscan', eps: 50000, min_points: 2}) YIELD node, cluster RETURN cluster, count(*) AS n, collect(node.name) AS fields ORDER BY n DESC</ex>\n");
    xml.push_str("      <ex desc=\"property K-means\">MATCH (w:Well) CALL cluster({properties: ['depth', 'temperature'], method: 'kmeans', k: 3, normalize: true}) YIELD node, cluster RETURN cluster, collect(node.name) AS wells</ex>\n");
    xml.push_str("      <ex desc=\"spatial K-means\">MATCH (s:Station) CALL cluster({method: 'kmeans', k: 4}) YIELD node, cluster RETURN cluster, count(*) AS n</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </cluster>\n");
}

fn write_topic_spatial(xml: &mut String) {
    xml.push_str("  <spatial>\n");
    xml.push_str("    <desc>Spatial functions for geographic queries. Requires set_spatial() config on the node type (location or geometry). All distance/area/perimeter results are in meters.</desc>\n");
    xml.push_str("    <setup>Python: g.set_spatial('Field', location=('lat', 'lon')) or g.set_spatial('Area', geometry='wkt')</setup>\n");
    xml.push_str("    <functions>\n");
    xml.push_str("      <fn name=\"distance(a, b)\">Geodesic distance in meters between two spatial nodes. Returns Null if either node has no location.</fn>\n");
    xml.push_str("      <fn name=\"contains(a, b)\">True if geometry a fully contains geometry b (or point b).</fn>\n");
    xml.push_str("      <fn name=\"intersects(a, b)\">True if geometries a and b overlap.</fn>\n");
    xml.push_str(
        "      <fn name=\"centroid(n)\">Returns {lat, lon} centroid of node's geometry.</fn>\n",
    );
    xml.push_str("      <fn name=\"area(n)\">Area of node's geometry in m².</fn>\n");
    xml.push_str("      <fn name=\"perimeter(n)\">Perimeter of node's geometry in meters.</fn>\n");
    xml.push_str("    </functions>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"distance between nodes\">MATCH (a:Field {name: 'Troll'}), (b:Field {name: 'Ekofisk'}) RETURN distance(a, b) / 1000.0 AS km</ex>\n");
    xml.push_str("      <ex desc=\"nearest neighbors\">MATCH (a:Field {name: 'Troll'}), (b:Field) WHERE a &lt;&gt; b RETURN b.name, round(distance(a, b) / 1000.0, 1) AS km ORDER BY km LIMIT 5</ex>\n");
    xml.push_str("      <ex desc=\"contains check\">MATCH (area:Block), (w:Well) WHERE contains(area, w) RETURN area.name, collect(w.name) AS wells</ex>\n");
    xml.push_str("      <ex desc=\"area calculation\">MATCH (b:Block) RETURN b.name, round(area(b) / 1e6, 1) AS km2</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </spatial>\n");
}

/// Write full detail for a single node type: properties, connections,
/// timeseries/spatial/embedding config, and sample nodes.
fn write_type_detail(
    xml: &mut String,
    graph: &DirGraph,
    node_type: &str,
    caps: &TypeCapabilities,
    indent: &str,
) {
    let count = graph
        .type_indices
        .get(node_type)
        .map(|v| v.len())
        .unwrap_or(0);

    let mut alias_attrs = String::new();
    if let Some(id_alias) = graph.id_field_aliases.get(node_type) {
        alias_attrs.push_str(&format!(" id_alias=\"{}\"", xml_escape(id_alias)));
    }
    if let Some(title_alias) = graph.title_field_aliases.get(node_type) {
        alias_attrs.push_str(&format!(" title_alias=\"{}\"", xml_escape(title_alias)));
    }

    xml.push_str(&format!(
        "{}<type name=\"{}\" count=\"{}\"{}>\n",
        indent,
        xml_escape(node_type),
        count,
        alias_attrs
    ));

    // Properties (exclude builtins: type, title, id)
    if let Ok(stats) = compute_property_stats(graph, node_type, 15) {
        let filtered: Vec<&PropertyStatInfo> = stats
            .iter()
            .filter(|p| !matches!(p.property_name.as_str(), "type" | "title" | "id"))
            .filter(|p| p.non_null > 0)
            .collect();
        if !filtered.is_empty() {
            xml.push_str(&format!("{}  <properties>\n", indent));
            for prop in &filtered {
                let mut attrs = format!(
                    "name=\"{}\" type=\"{}\" unique=\"{}\"",
                    xml_escape(&prop.property_name),
                    xml_escape(&prop.type_string),
                    prop.unique
                );
                if let Some(ref vals) = prop.values {
                    if !vals.is_empty() {
                        let val_strs: Vec<String> =
                            vals.iter().map(value_display_compact).collect();
                        attrs.push_str(&format!(" vals=\"{}\"", xml_escape(&val_strs.join("|"))));
                    }
                }
                xml.push_str(&format!("{}    <prop {}/>\n", indent, attrs));
            }
            xml.push_str(&format!("{}  </properties>\n", indent));
        }
    }

    // Connections (neighbors)
    if let Ok(neighbors) = compute_neighbors_schema(graph, node_type) {
        if !neighbors.outgoing.is_empty() || !neighbors.incoming.is_empty() {
            xml.push_str(&format!("{}  <connections>\n", indent));
            for nc in &neighbors.outgoing {
                xml.push_str(&format!(
                    "{}    <out type=\"{}\" target=\"{}\" count=\"{}\"/>\n",
                    indent,
                    xml_escape(&nc.connection_type),
                    xml_escape(&nc.other_type),
                    nc.count
                ));
            }
            for nc in &neighbors.incoming {
                xml.push_str(&format!(
                    "{}    <in type=\"{}\" source=\"{}\" count=\"{}\"/>\n",
                    indent,
                    xml_escape(&nc.connection_type),
                    xml_escape(&nc.other_type),
                    nc.count
                ));
            }
            xml.push_str(&format!("{}  </connections>\n", indent));
        }
    }

    // Timeseries config
    if caps.has_timeseries {
        if let Some(config) = graph.timeseries_configs.get(node_type) {
            let mut attrs = format!("resolution=\"{}\"", xml_escape(&config.resolution));
            if !config.channels.is_empty() {
                attrs.push_str(&format!(
                    " channels=\"{}\"",
                    config
                        .channels
                        .iter()
                        .map(|c| xml_escape(c))
                        .collect::<Vec<_>>()
                        .join(",")
                ));
            }
            if !config.units.is_empty() {
                let units_str: Vec<String> = config
                    .units
                    .iter()
                    .map(|(k, v)| format!("{}={}", xml_escape(k), xml_escape(v)))
                    .collect();
                attrs.push_str(&format!(" units=\"{}\"", units_str.join(",")));
            }
            xml.push_str(&format!("{}  <timeseries {}/>\n", indent, attrs));
        }
    }

    // Spatial config
    if caps.has_location || caps.has_geometry {
        if let Some(config) = graph.spatial_configs.get(node_type) {
            let mut attrs = String::new();
            if let Some((lat, lon)) = &config.location {
                attrs.push_str(&format!(
                    "location=\"{},{}\"",
                    xml_escape(lat),
                    xml_escape(lon)
                ));
            }
            if let Some(geom) = &config.geometry {
                if !attrs.is_empty() {
                    attrs.push(' ');
                }
                attrs.push_str(&format!("geometry=\"{}\"", xml_escape(geom)));
            }
            if !attrs.is_empty() {
                xml.push_str(&format!("{}  <spatial {}/>\n", indent, attrs));
            }
        }
    }

    // Embedding config
    if caps.has_embeddings {
        for ((nt, prop_name), store) in &graph.embeddings {
            if nt == node_type {
                let text_col = prop_name.strip_suffix("_emb").unwrap_or(prop_name.as_str());
                xml.push_str(&format!(
                    "{}  <embeddings text_col=\"{}\" dim=\"{}\" count=\"{}\"/>\n",
                    indent,
                    xml_escape(text_col),
                    store.dimension,
                    store.len()
                ));
            }
        }
    }

    // Supporting children (if this is a core type with children)
    {
        let children: Vec<&String> = graph
            .parent_types
            .iter()
            .filter(|(_, parent)| parent.as_str() == node_type)
            .map(|(child, _)| child)
            .collect();
        if !children.is_empty() {
            let empty_caps = TypeCapabilities {
                has_timeseries: false,
                has_location: false,
                has_geometry: false,
                has_embeddings: false,
            };
            // Compute caps for children (direct, not bubbled)
            let child_caps = compute_type_capabilities(graph);
            let mut child_strs: Vec<(usize, String)> = children
                .iter()
                .map(|child| {
                    let count = graph.type_indices.get(*child).map(|v| v.len()).unwrap_or(0);
                    let prop_count = graph
                        .node_type_metadata
                        .get(*child)
                        .map(|m| m.len())
                        .unwrap_or(0);
                    let tc = child_caps.get(*child).unwrap_or(&empty_caps);
                    (count, format_type_descriptor(child, count, prop_count, tc))
                })
                .collect();
            child_strs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let strs: Vec<&str> = child_strs.iter().map(|(_, s)| s.as_str()).collect();
            xml.push_str(&format!(
                "{}  <supporting>{}</supporting>\n",
                indent,
                strs.join(", ")
            ));
        }
    }

    // Sample nodes (2 samples)
    if let Ok(samples) = compute_sample(graph, node_type, 2) {
        if !samples.is_empty() {
            xml.push_str(&format!("{}  <samples>\n", indent));
            for node in samples {
                let mut attrs = format!(
                    "id=\"{}\" title=\"{}\"",
                    xml_escape(&value_display_compact(&node.id)),
                    xml_escape(&value_display_compact(&node.title))
                );
                // Include up to 4 non-null custom properties
                let mut prop_count = 0;
                let mut sorted_props: Vec<(&String, &Value)> = node.properties.iter().collect();
                sorted_props.sort_by_key(|(k, _)| k.as_str());
                for (k, v) in sorted_props {
                    if !is_null_value(v) && prop_count < 4 {
                        attrs.push_str(&format!(
                            " {}=\"{}\"",
                            xml_escape(k),
                            xml_escape(&value_display_compact(v))
                        ));
                        prop_count += 1;
                    }
                }
                xml.push_str(&format!("{}    <node {}/>\n", indent, attrs));
            }
            xml.push_str(&format!("{}  </samples>\n", indent));
        }
    }

    xml.push_str(&format!("{}</type>\n", indent));
}

// ── Describe: builders ─────────────────────────────────────────────────────

/// Build inventory for complex graphs (>15 types): size bands with
/// complexity markers and capability flags.
fn build_inventory(graph: &DirGraph) -> String {
    let mut caps = compute_type_capabilities(graph);
    bubble_capabilities(&mut caps, &graph.parent_types);
    let child_counts = children_counts(&graph.parent_types);
    let has_tiers = !graph.parent_types.is_empty();
    let empty_caps = TypeCapabilities {
        has_timeseries: false,
        has_location: false,
        has_geometry: false,
        has_embeddings: false,
    };

    let mut xml = String::with_capacity(2048);

    xml.push_str(&format!(
        "<graph nodes=\"{}\" edges=\"{}\">\n",
        graph.graph.node_count(),
        graph.graph.edge_count()
    ));

    write_conventions(&mut xml, &caps);
    write_read_only_notice(&mut xml, graph);

    // Collect types: if tiers active, only core types; otherwise all types
    let mut entries: Vec<(String, usize, usize)> = graph
        .type_indices
        .iter()
        .filter(|(nt, _)| !has_tiers || !graph.parent_types.contains_key(*nt))
        .map(|(nt, indices)| {
            let prop_count = graph
                .node_type_metadata
                .get(nt)
                .map(|m| m.len())
                .unwrap_or(0);
            (nt.clone(), indices.len(), prop_count)
        })
        .collect();
    // Sort by count descending, then alphabetically
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let core_count = entries.len();
    let supporting_count = graph.parent_types.len();
    if has_tiers {
        xml.push_str(&format!(
            "  <types core=\"{}\" supporting=\"{}\">\n    ",
            core_count, supporting_count
        ));
    } else {
        xml.push_str(&format!("  <types count=\"{}\">\n    ", core_count));
    }

    let type_strs: Vec<String> = entries
        .iter()
        .map(|(nt, count, prop_count)| {
            let tc = caps.get(nt).unwrap_or(&empty_caps);
            let desc = format_type_descriptor(nt, *count, *prop_count, tc);
            let children = child_counts.get(nt).copied().unwrap_or(0);
            if children > 0 {
                format!("{} +{}", desc, children)
            } else {
                desc
            }
        })
        .collect();
    xml.push_str(&type_strs.join(", "));
    xml.push_str("\n  </types>\n");

    write_connection_map(&mut xml, graph);
    write_extensions(&mut xml, graph);

    xml.push_str(
        "  <hint>Use describe(types=['TypeName']) for properties, children, connections, and samples.</hint>\n",
    );
    xml.push_str("</graph>");
    xml
}

/// Build inventory with inline detail for simple graphs (≤15 types).
fn build_inventory_with_detail(graph: &DirGraph) -> String {
    let mut caps = compute_type_capabilities(graph);
    bubble_capabilities(&mut caps, &graph.parent_types);
    let mut xml = String::with_capacity(4096);

    xml.push_str(&format!(
        "<graph nodes=\"{}\" edges=\"{}\">\n",
        graph.graph.node_count(),
        graph.graph.edge_count()
    ));

    write_conventions(&mut xml, &caps);
    write_read_only_notice(&mut xml, graph);

    // Full detail for each type (core only if tiers active)
    let has_tiers = !graph.parent_types.is_empty();
    let mut type_names: Vec<&String> = graph
        .type_indices
        .keys()
        .filter(|nt| !has_tiers || !graph.parent_types.contains_key(*nt))
        .collect();
    type_names.sort();

    xml.push_str("  <types>\n");
    let empty_caps = TypeCapabilities {
        has_timeseries: false,
        has_location: false,
        has_geometry: false,
        has_embeddings: false,
    };
    for nt in type_names {
        let tc = caps.get(nt).unwrap_or(&empty_caps);
        write_type_detail(&mut xml, graph, nt, tc, "    ");
    }
    xml.push_str("  </types>\n");

    write_connection_map(&mut xml, graph);
    write_extensions(&mut xml, graph);

    xml.push_str("</graph>");
    xml
}

/// Build focused detail for specific requested types.
fn build_focused_detail(graph: &DirGraph, types: &[String]) -> Result<String, String> {
    // Validate all types exist
    for t in types {
        if !graph.type_indices.contains_key(t) {
            return Err(format!("Node type '{}' not found. Available: {}", t, {
                let mut names: Vec<&String> = graph.type_indices.keys().collect();
                names.sort();
                names
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            }));
        }
    }

    let caps = compute_type_capabilities(graph);
    let empty_caps = TypeCapabilities {
        has_timeseries: false,
        has_location: false,
        has_geometry: false,
        has_embeddings: false,
    };
    let mut xml = String::with_capacity(2048);
    xml.push_str("<graph>\n");
    write_read_only_notice(&mut xml, graph);

    for t in types {
        let tc = caps.get(t).unwrap_or(&empty_caps);
        write_type_detail(&mut xml, graph, t, tc, "  ");
    }

    xml.push_str("</graph>");
    Ok(xml)
}

// ── Describe: entry point ──────────────────────────────────────────────────

/// Build an XML description of the graph for AI agents (progressive disclosure).
///
/// Three independent axes:
/// - `types` → Node type deep-dive (None=inventory, Some=focused detail).
/// - `connections` → Connection type docs (Off=in inventory, Overview=all, Topics=specific).
/// - `cypher` → Cypher language reference (Off=hint, Overview=compact, Topics=detailed).
///
/// When `connections` or `cypher` is not Off, only those tracks are returned (no node inventory).
pub fn compute_description(
    graph: &DirGraph,
    types: Option<&[String]>,
    connections: &ConnectionDetail,
    cypher: &CypherDetail,
) -> Result<String, String> {
    // If connections or cypher is requested, return only those tracks
    let standalone =
        !matches!(connections, ConnectionDetail::Off) || !matches!(cypher, CypherDetail::Off);

    if standalone {
        let mut result = String::with_capacity(4096);
        match connections {
            ConnectionDetail::Off => {}
            ConnectionDetail::Overview => write_connections_overview(&mut result, graph),
            ConnectionDetail::Topics(ref topics) => {
                write_connections_detail(&mut result, graph, topics)?;
            }
        }
        match cypher {
            CypherDetail::Off => {}
            CypherDetail::Overview => write_cypher_overview(&mut result),
            CypherDetail::Topics(ref topics) => {
                write_cypher_topics(&mut result, topics)?;
            }
        }
        return Ok(result);
    }

    // Normal describe — inventory or focused detail
    let result = match types {
        Some(requested) if !requested.is_empty() => build_focused_detail(graph, requested)?,
        _ => {
            // Count core types only (exclude supporting types)
            let core_count = graph
                .type_indices
                .keys()
                .filter(|nt| !graph.parent_types.contains_key(*nt))
                .count();
            if core_count <= 15 {
                build_inventory_with_detail(graph)
            } else {
                build_inventory(graph)
            }
        }
    };
    Ok(result)
}

/// Minimal XML escaping for attribute values.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ── MCP quickstart ──────────────────────────────────────────────────────────

/// Return a self-contained XML quickstart for setting up a KGLite MCP server.
///
/// Static content — no graph instance needed.
pub fn mcp_quickstart() -> String {
    format!(
        r##"<mcp_quickstart version="{version}">

  <setup>
    <install>pip install kglite fastmcp</install>
    <server><![CDATA[
import kglite
from fastmcp import FastMCP

graph = kglite.load("your_graph.kgl")
mcp = FastMCP("my-graph", instructions="Knowledge graph. Call graph_overview first.")

@mcp.tool()
def graph_overview(
    types: list[str] | None = None,
    connections: bool | list[str] | None = None,
    cypher: bool | list[str] | None = None,
) -> str:
    """Get graph schema, connection details, or Cypher language reference.

    Three independent axes — call with no args first for the overview:
      graph_overview()                            — inventory of node types
      graph_overview(types=["Field"])             — property schemas, samples
      graph_overview(connections=True)            — all connection types overview
      graph_overview(connections=["BELONGS_TO"])  — deep-dive with properties
      graph_overview(cypher=True)                 — Cypher clauses, functions, procedures
      graph_overview(cypher=["cluster","MATCH"])  — detailed docs with examples"""
    return graph.describe(types=types, connections=connections, cypher=cypher)

@mcp.tool()
def cypher_query(query: str) -> str:
    """Run a Cypher query against the knowledge graph.

    Supports MATCH, WHERE, RETURN, ORDER BY, LIMIT, aggregations,
    path traversals, CREATE, SET, DELETE, and CALL procedures.
    Returns up to 200 rows as formatted text."""
    result = graph.cypher(query)
    if len(result) == 0:
        return "Query returned no results."
    rows = [str(dict(row)) for row in result[:200]]
    header = f"Returned {{len(result)}} row(s)"
    if len(result) > 200:
        header += " (showing first 200)"
    return header + ":\n" + "\n".join(rows)

@mcp.tool()
def bug_report(query: str, result: str, expected: str, description: str) -> str:
    """File a Cypher bug report to reported_bugs.md.

    Writes a timestamped, version-tagged entry (newest first).
    Use when a query returns incorrect or unexpected results."""
    return graph.bug_report(query, result, expected, description)

if __name__ == "__main__":
    mcp.run(transport="stdio")
]]></server>
  </setup>

  <core_tools desc="Essential — include all three in every MCP server">
    <tool name="graph_overview" method="graph.describe()" args="types, connections, cypher">
      Schema introspection with 3-tier progressive disclosure.
      The agent's entry point — always expose this.
    </tool>
    <tool name="cypher_query" method="graph.cypher()" args="query">
      Execute Cypher queries. MATCH/WHERE/RETURN/CREATE/SET/DELETE,
      aggregations, CALL procedures (pagerank, cluster, etc.).
    </tool>
    <tool name="bug_report" method="graph.bug_report()" args="query, result, expected, description">
      File bug reports to reported_bugs.md. Input is sanitised.
    </tool>
  </core_tools>

  <optional_tools desc="Add based on your use case">
    <tool name="find_entity" method="graph.find()" args="name, node_type?, match_type?">
      Search nodes by name. match_type: 'exact' (default), 'contains', 'starts_with'.
      Useful for code graphs where entities have qualified names.
    </tool>
    <tool name="read_source" method="graph.source()" args="names, node_type?">
      Resolve entity names to file paths and line ranges.
      Returns source code locations for code navigation.
    </tool>
    <tool name="entity_context" method="graph.context()" args="name, node_type?, hops?">
      Get neighborhood of a node — related entities within N hops.
      Good for understanding how entities connect.
    </tool>
    <tool name="file_toc" method="graph.toc()" args="file_path">
      Table of contents for a file — lists all entities sorted by line.
      Only relevant for code-tree graphs.
    </tool>
    <tool name="grep_source" custom="true">
      Text search across source files. Not built-in — implement with
      your own file-reading logic or expose graph.cypher() with
      CONTAINS/STARTS WITH/=~ for in-graph text search.
    </tool>
  </optional_tools>

  <register_with_claude>
    <claude_desktop desc="Add to Claude Desktop config">
      <file>~/Library/Application Support/Claude/claude_desktop_config.json</file>
      <config><![CDATA[
{{
  "mcpServers": {{
    "my-graph": {{
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }}
  }}
}}
]]></config>
    </claude_desktop>
    <claude_code desc="Add to Claude Code config">
      <file>.claude/settings.json (project) or ~/.claude/settings.json (global)</file>
      <config><![CDATA[
{{
  "mcpServers": {{
    "my-graph": {{
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }}
  }}
}}
]]></config>
    </claude_code>
    <note>Restart Claude after editing config. The server appears as an MCP tool provider.</note>
  </register_with_claude>

</mcp_quickstart>
"##,
        version = env!("CARGO_PKG_VERSION"),
    )
}
