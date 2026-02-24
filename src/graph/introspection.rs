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
                "    <conn type=\"{}\" from=\"{}\" to=\"{}\"/>\n",
                xml_escape(&ct.connection_type),
                sources.join(","),
                targets.join(","),
            ));
        }
        xml.push_str("  </connections>\n");
    }
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
    xml.push_str("    <cypher hint=\"MATCH, WHERE, RETURN, WITH, ORDER BY, SKIP, LIMIT, UNION [ALL], UNWIND, CASE, CREATE/SET/DELETE/MERGE. Use describe(cypher=true) for operators, functions, and predicates.\"/>\n");
    xml.push_str("  </extensions>\n");
}

/// Write the full `<cypher>` language reference (progressive disclosure level 2).
fn write_cypher_reference(xml: &mut String) {
    xml.push_str("<cypher>\n");
    xml.push_str("  <clauses>MATCH, OPTIONAL MATCH, WHERE, RETURN [DISTINCT], WITH, ORDER BY [DESC], SKIP, LIMIT, UNWIND expr AS var, UNION [ALL], CASE WHEN/THEN/ELSE/END, CREATE, SET, DELETE, REMOVE, MERGE</clauses>\n");
    xml.push_str("  <operators>\n");
    xml.push_str("    <math>+ - * /</math>\n");
    xml.push_str("    <string>|| (concatenation)</string>\n");
    xml.push_str("    <comparison>= &lt;&gt; &lt; &gt; &lt;= &gt;= IN</comparison>\n");
    xml.push_str("    <logical>AND OR NOT XOR</logical>\n");
    xml.push_str("    <null>IS NULL, IS NOT NULL</null>\n");
    xml.push_str("    <regex>=~ 'pattern'</regex>\n");
    xml.push_str("    <string_predicates>CONTAINS, STARTS WITH, ENDS WITH</string_predicates>\n");
    xml.push_str("  </operators>\n");
    xml.push_str("  <functions>\n");
    xml.push_str("    <math>abs, ceil, floor, round(x [,decimals]), sqrt, sign, log, exp, pow, pi, rand, toInteger, toFloat</math>\n");
    xml.push_str("    <string>toString, toUpper, toLower, trim, lTrim, rTrim, replace, substring, left, right, split, reverse, size</string>\n");
    xml.push_str("    <aggregate>count, sum, avg, min, max, collect, stDev</aggregate>\n");
    xml.push_str("    <list>size, head, tail, last, range, keys, labels, type</list>\n");
    xml.push_str("    <null>coalesce(expr, ...) — first non-null</null>\n");
    xml.push_str("  </functions>\n");
    xml.push_str("  <patterns>\n");
    xml.push_str(
        "    (n:Label), (n:Label {prop: val}), (a)-[:TYPE]-&gt;(b), (a)-[:TYPE*1..3]-&gt;(b),\n",
    );
    xml.push_str("    [x IN list WHERE pred | expr] — list comprehension,\n");
    xml.push_str("    n {.prop1, .prop2} — map projection\n");
    xml.push_str("  </patterns>\n");
    xml.push_str("  <examples>\n");
    xml.push_str("    <ex desc=\"string predicate\">WHERE n.name CONTAINS 'oil'</ex>\n");
    xml.push_str("    <ex desc=\"regex match\">WHERE n.name =~ '35/9-.*'</ex>\n");
    xml.push_str(
        "    <ex desc=\"null coalesce\">RETURN coalesce(n.nickname, n.name) AS label</ex>\n",
    );
    xml.push_str("    <ex desc=\"concat\">RETURN n.quadrant || '/' || n.block AS qb</ex>\n");
    xml.push_str("    <ex desc=\"union\">MATCH (a:Field) RETURN a.name UNION MATCH (b:Discovery) RETURN b.name</ex>\n");
    xml.push_str(
        "    <ex desc=\"unwind\">UNWIND ['A','B','C'] AS x MATCH (n {code: x}) RETURN n</ex>\n",
    );
    xml.push_str("    <ex desc=\"list comprehension\">[x IN collect(n.name) WHERE x STARTS WITH '35']</ex>\n");
    xml.push_str(
        "    <ex desc=\"case\">CASE WHEN n.depth &gt; 3000 THEN 'deep' ELSE 'shallow' END</ex>\n",
    );
    xml.push_str(
        "    <ex desc=\"round precision\">round(distance(a, b) / 1000.0, 1) AS dist_km</ex>\n",
    );
    xml.push_str("  </examples>\n");
    xml.push_str("</cypher>\n");
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
/// - `types = None` → Inventory mode. If ≤15 types, auto-inlines full detail.
/// - `types = Some(list)` → Focused detail for the requested types only.
/// - `cypher = true` → Append full Cypher language reference (operators, functions, examples).
pub fn compute_description(
    graph: &DirGraph,
    types: Option<&[String]>,
    cypher: bool,
) -> Result<String, String> {
    let mut result = match types {
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
    if cypher {
        result.push('\n');
        write_cypher_reference(&mut result);
    }
    Ok(result)
}

/// Minimal XML escaping for attribute values.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
