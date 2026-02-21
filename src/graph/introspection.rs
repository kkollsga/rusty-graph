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

// ── Agent description ──────────────────────────────────────────────────────

/// Static XML: code-entity exploration methods (only for code graphs).
const API_CODE_EXPLORATION: &str = r#"    <exploration hint="Use these FIRST to understand the codebase">
    <method sig="find(name, node_type=None, match_type=None)">Search code entities by name. match_type: 'exact' (default), 'contains' (substring), 'starts_with'. Returns type, qualified_name, file_path, line_number.</method>
    <method sig="source(name, node_type=None)">Get source location: file_path, line_number, end_line, signature. Resolves qualified names directly.</method>
    <method sig="context(name, node_type=None, hops=None)">Full neighborhood of an entity: properties + relationships grouped by edge type.</method>
    <method sig="toc(file_path)">Table of contents for a source file: all entities sorted by line_number.</method>
    </exploration>
"#;

/// Static XML: query tools (always present).
const API_QUERY: &str = r#"    <method sig="cypher(query, *, to_df=False, params=None)">Primary query interface. Returns ResultView (lazy) or DataFrame with to_df=True.</method>
    <method sig="schema()">Returns dict: node_types, connection_types, indexes, node_count, edge_count.</method>
    <method sig="properties(node_type, max_values=20)">Per-property stats: type, non_null, unique, values.</method>
    <method sig="sample(node_type, n=5)">Returns first N nodes as ResultView.</method>
    <method sig="save(path) / load(path)">Persist and reload the graph.</method>
  </api>
"#;

/// Static XML: Cypher reference — clauses through expressions (read-write mode).
const CYPHER_REF_BASE: &str = r#"  <cypher_ref>
    <clauses>MATCH, OPTIONAL MATCH, WHERE, RETURN, WITH, ORDER BY, SKIP, LIMIT, UNWIND, UNION, UNION ALL, CREATE, SET, DELETE, DETACH DELETE, REMOVE, MERGE, CALL...YIELD, EXPLAIN</clauses>
    <patterns>(n:Type), (n:Type {key: val}), (n {key: val}), (n), -[:REL]-&gt;, &lt;-[:REL]-, -[:REL]-, -[:REL*1..3]-&gt;, p = shortestPath(...)</patterns>
    <where>=, &lt;&gt;, &lt;, &gt;, &lt;=, &gt;=, =~ (regex), AND, OR, NOT, IS NULL, IS NOT NULL, IN [...], CONTAINS, STARTS WITH, ENDS WITH, EXISTS { pattern }, EXISTS(( pattern ))</where>
    <return>n.prop, r.prop, AS alias, DISTINCT, arithmetic (+, -, *, /), map projections n {.prop1, .prop2}</return>
    <aggregation>count(*), count(expr), sum, avg, min, max, collect, std</aggregation>
    <expressions>CASE WHEN...THEN...ELSE...END, $param, [x IN list WHERE ... | expr], list[index], list[start..end]</expressions>
"#;

/// Static XML: Cypher reference — clauses through expressions (read-only mode).
const CYPHER_REF_BASE_RO: &str = r#"  <cypher_ref mode="read-only">
    <clauses>MATCH, OPTIONAL MATCH, WHERE, RETURN, WITH, ORDER BY, SKIP, LIMIT, UNWIND, UNION, UNION ALL, CALL...YIELD, EXPLAIN</clauses>
    <patterns>(n:Type), (n:Type {key: val}), (n {key: val}), (n), -[:REL]-&gt;, &lt;-[:REL]-, -[:REL]-, -[:REL*1..3]-&gt;, p = shortestPath(...)</patterns>
    <where>=, &lt;&gt;, &lt;, &gt;, &lt;=, &gt;=, =~ (regex), AND, OR, NOT, IS NULL, IS NOT NULL, IN [...], CONTAINS, STARTS WITH, ENDS WITH, EXISTS { pattern }, EXISTS(( pattern ))</where>
    <return>n.prop, r.prop, AS alias, DISTINCT, arithmetic (+, -, *, /), map projections n {.prop1, .prop2}</return>
    <aggregation>count(*), count(expr), sum, avg, min, max, collect, std</aggregation>
    <expressions>CASE WHEN...THEN...ELSE...END, $param, [x IN list WHERE ... | expr], list[index], list[start..end]</expressions>
"#;

/// Static XML: Cypher mutations and unsupported features (read-write mode only).
const CYPHER_REF_MUTATIONS: &str = r#"    <mutations>CREATE (n:Label {props}), CREATE (a)-[:TYPE]-&gt;(b), SET n.prop = expr, DELETE, DETACH DELETE, REMOVE n.prop, MERGE...ON CREATE SET...ON MATCH SET</mutations>
    <not_supported>FOREACH, subqueries, SET n:Label, REMOVE n:Label, multi-label, list comprehension patterns</not_supported>
"#;

/// Minimal XML escaping for attribute values.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Build a minimal XML description of the graph for AI agents.
/// Only includes embedding and spatial sections when the graph actually uses those features.
pub fn compute_agent_description(graph: &DirGraph) -> String {
    let overview = compute_schema(graph);
    let has_embeddings = !graph.embeddings.is_empty();
    let has_spatial = !graph.spatial_configs.is_empty()
        || graph
            .node_type_metadata
            .values()
            .any(|props| props.values().any(|t| t.eq_ignore_ascii_case("point")));
    let has_code_entities = ["Function", "Struct", "Class", "Enum", "Trait"]
        .iter()
        .any(|t| graph.type_indices.contains_key(*t));

    let mut xml = String::with_capacity(4096);

    xml.push_str(&format!(
        "<kglite version=\"{}\" nodes=\"{}\" edges=\"{}\">\n",
        graph.save_metadata.library_version, overview.node_count, overview.edge_count
    ));

    // Quick start — actionable tips at the very top
    if has_code_entities {
        xml.push_str("  <quick_start>\n");
        xml.push_str("    <tip>Find code by name — faster than grep: MATCH (f:Function {name: 'foo'}) RETURN f.file_path, f.line_number, f.end_line</tip>\n");
        xml.push_str("    <tip>Read code by qualified name — no file path needed: read_source(qualified_name=\"MyClass.method\")</tip>\n");
        xml.push_str("    <tip>Trace call chains: MATCH (f:Function {name: 'x'})-[:CALLS*1..3]->(g) RETURN g.name, g.file_path</tip>\n");
        xml.push_str("    <tip>Node id IS the qualified_name (e.g. 'package.module.Class.method'). Use with find(), source(), context().</tip>\n");
        xml.push_str("    <tip>Use find(name, match_type='contains') for fuzzy search. Use toc(file_path) for file overview. Use context(name) for relationships.</tip>\n");
        xml.push_str("  </quick_start>\n");
    }

    // Node types
    if overview.node_types.is_empty() {
        xml.push_str("  <node_types/>\n");
    } else {
        xml.push_str("  <node_types>\n");
        for (nt, info) in &overview.node_types {
            let mut alias_attrs = String::new();
            if let Some(id_alias) = graph.id_field_aliases.get(nt) {
                alias_attrs.push_str(&format!(" id_alias=\"{}\"", xml_escape(id_alias)));
            }
            if let Some(title_alias) = graph.title_field_aliases.get(nt) {
                alias_attrs.push_str(&format!(" title_alias=\"{}\"", xml_escape(title_alias)));
            }

            if info.properties.is_empty() {
                xml.push_str(&format!(
                    "    <type name=\"{}\" count=\"{}\"{}/>\n",
                    xml_escape(nt),
                    info.count,
                    alias_attrs
                ));
            } else {
                xml.push_str(&format!(
                    "    <type name=\"{}\" count=\"{}\"{}>\n",
                    xml_escape(nt),
                    info.count,
                    alias_attrs
                ));
                let mut props: Vec<(&String, &String)> = info.properties.iter().collect();
                props.sort_by_key(|(k, _)| k.as_str());

                // Collect unique values per property for low-cardinality display
                const MAX_DISPLAY_VALS: usize = 15;
                let mut prop_values: HashMap<&str, Vec<String>> = HashMap::new();
                if let Some(indices) = graph.type_indices.get(nt) {
                    for (pname, ptype) in &props {
                        // Skip UniqueId — always unique, not useful for WHERE clauses
                        if ptype.eq_ignore_ascii_case("uniqueid") {
                            continue;
                        }
                        let mut unique_vals: HashSet<String> = HashSet::new();
                        let mut exceeded = false;
                        for &idx in indices {
                            if let Some(node) = graph.get_node(idx) {
                                let val = match pname.as_str() {
                                    "title" => Some(&node.title),
                                    _ => node.properties.get(*pname),
                                };
                                if let Some(v) = val {
                                    if !is_null_value(v) {
                                        unique_vals.insert(value_display_compact(v));
                                        if unique_vals.len() > MAX_DISPLAY_VALS {
                                            exceeded = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        if !exceeded && !unique_vals.is_empty() {
                            let mut sorted: Vec<String> = unique_vals.into_iter().collect();
                            sorted.sort();
                            prop_values.insert(pname.as_str(), sorted);
                        }
                    }
                }

                for (pname, ptype) in &props {
                    if let Some(vals) = prop_values.get(pname.as_str()) {
                        xml.push_str(&format!(
                            "      <prop name=\"{}\" type=\"{}\" vals=\"{}\"/>\n",
                            xml_escape(pname),
                            xml_escape(ptype),
                            xml_escape(&vals.join("|"))
                        ));
                    } else {
                        xml.push_str(&format!(
                            "      <prop name=\"{}\" type=\"{}\"/>\n",
                            xml_escape(pname),
                            xml_escape(ptype)
                        ));
                    }
                }
                xml.push_str("    </type>\n");
            }
        }
        xml.push_str("  </node_types>\n");
    }

    // Connections
    if overview.connection_types.is_empty() {
        xml.push_str("  <connections/>\n");
    } else {
        xml.push_str("  <connections>\n");
        for ct in &overview.connection_types {
            xml.push_str(&format!(
                "    <conn type=\"{}\" count=\"{}\" from=\"{}\" to=\"{}\"/>\n",
                xml_escape(&ct.connection_type),
                ct.count,
                ct.source_types.join(","),
                ct.target_types.join(","),
            ));
        }
        xml.push_str("  </connections>\n");

        // Workflow recipes for code graphs — group edges by use case
        if has_code_entities {
            xml.push_str(
                "  <workflows hint=\"Common query patterns using the edge types above\">\n",
            );
            xml.push_str("    <workflow name=\"Call graph\">CALLS edges link Function→Function. Use [:CALLS*1..N] for transitive callers/callees.\n      Example: MATCH (f:Function {name: 'main'})-[:CALLS*1..2]->(g) RETURN g.name, g.file_path</workflow>\n");
            xml.push_str("    <workflow name=\"Inheritance\">EXTENDS (class→class), IMPLEMENTS (class→trait/interface).\n      Example: MATCH (s)-[:IMPLEMENTS]->(t {name: 'Display'}) RETURN s.name, s.file_path</workflow>\n");
            xml.push_str("    <workflow name=\"File structure\">DEFINES (Module→entities), HAS_SOURCE (entity→File).\n      Example: MATCH (m:Module)-[:DEFINES]->(f:Function) WHERE m.name = 'utils' RETURN f.name</workflow>\n");
            xml.push_str("    <workflow name=\"Dependencies\">IMPORTS (Module→Module), DEPENDS_ON (Project→Dependency).\n      Example: MATCH (m:Module)-[:IMPORTS]->(dep) RETURN dep.name, count(*) ORDER BY count(*) DESC</workflow>\n");
            xml.push_str("  </workflows>\n");
        }
    }

    // Indexes
    if overview.indexes.is_empty() {
        xml.push_str("  <indexes/>\n");
    } else {
        xml.push_str("  <indexes>\n");
        for idx in &overview.indexes {
            xml.push_str(&format!("    <idx on=\"{}\"/>\n", xml_escape(idx)));
        }
        xml.push_str("  </indexes>\n");
    }

    // Embeddings (only when graph has embedding data)
    if has_embeddings {
        xml.push_str("  <embeddings>\n");
        let mut emb_keys: Vec<&(String, String)> = graph.embeddings.keys().collect();
        emb_keys.sort();
        for (node_type, prop_name) in emb_keys {
            if let Some(store) = graph
                .embeddings
                .get(&(node_type.clone(), prop_name.clone()))
            {
                // Show text column name (strip _emb suffix) for text_score() usage
                let text_col = prop_name.strip_suffix("_emb").unwrap_or(prop_name.as_str());
                xml.push_str(&format!(
                    "    <emb type=\"{}\" text_col=\"{}\" dim=\"{}\" count=\"{}\"/>\n",
                    xml_escape(node_type),
                    xml_escape(text_col),
                    store.dimension,
                    store.len()
                ));
            }
        }
        xml.push_str("  </embeddings>\n");
    }

    // Spatial config section — shows configured spatial types for distance auto-resolution
    if !graph.spatial_configs.is_empty() {
        xml.push_str("  <spatial hint=\"distance(a,b), contains(a,b), intersects(a,b), centroid(n), area(n), perimeter(n) — all auto-resolve via spatial config\">\n");
        let mut sorted_types: Vec<_> = graph.spatial_configs.iter().collect();
        sorted_types.sort_by_key(|(k, _)| k.as_str());
        for (node_type, config) in sorted_types {
            let mut attrs = format!("name=\"{}\"", xml_escape(node_type));
            if let Some((lat, lon)) = &config.location {
                attrs.push_str(&format!(
                    " location=\"{},{}\"",
                    xml_escape(lat),
                    xml_escape(lon)
                ));
            }
            if let Some(geom) = &config.geometry {
                attrs.push_str(&format!(" geometry=\"{}\"", xml_escape(geom)));
            }
            if config.points.is_empty() && config.shapes.is_empty() {
                xml.push_str(&format!("    <type {}/>\n", attrs));
            } else {
                xml.push_str(&format!("    <type {}>\n", attrs));
                let mut sorted_points: Vec<_> = config.points.iter().collect();
                sorted_points.sort_by_key(|(k, _)| k.as_str());
                for (name, (lat, lon)) in sorted_points {
                    xml.push_str(&format!(
                        "      <point name=\"{}\" fields=\"{},{}\"/>\n",
                        xml_escape(name),
                        xml_escape(lat),
                        xml_escape(lon)
                    ));
                }
                let mut sorted_shapes: Vec<_> = config.shapes.iter().collect();
                sorted_shapes.sort_by_key(|(k, _)| k.as_str());
                for (name, field) in sorted_shapes {
                    xml.push_str(&format!(
                        "      <shape name=\"{}\" field=\"{}\"/>\n",
                        xml_escape(name),
                        xml_escape(field)
                    ));
                }
                xml.push_str("    </type>\n");
            }
        }
        xml.push_str("  </spatial>\n");
    }

    // API methods — exploration (code graphs only) + query tools
    xml.push_str("  <api>\n");
    if has_code_entities {
        xml.push_str(API_CODE_EXPLORATION);
    }
    xml.push_str(API_QUERY);

    // Cypher reference (read-only mode omits mutation clauses)
    if graph.read_only {
        xml.push_str(CYPHER_REF_BASE_RO);
    } else {
        xml.push_str(CYPHER_REF_BASE);
    }

    // Functions list — conditionally include spatial and embedding functions
    let mut functions = String::from(
        "toUpper, toLower, toString, toInteger, toFloat, size, length, \
         type, id, labels, coalesce, nodes(p), relationships(p), \
         split, replace, substring, left, right, trim, ltrim, rtrim, reverse",
    );
    if has_spatial {
        functions.push_str(
            ", point, distance, contains, intersects, centroid, area, perimeter, \
             latitude, longitude",
        );
    }
    if has_embeddings {
        functions.push_str(", text_score");
    }
    xml.push_str(&format!("    <functions>{}</functions>\n", functions));

    if graph.read_only {
        xml.push_str(
            "    <not_supported>FOREACH, subqueries, SET n:Label, REMOVE n:Label, multi-label, \
             list comprehension patterns, CREATE, SET, DELETE, REMOVE, MERGE (read-only mode)</not_supported>\n",
        );
    } else {
        xml.push_str(CYPHER_REF_MUTATIONS);
    }

    // CALL procedures
    xml.push_str("    <procedures hint=\"CALL name({params}) YIELD columns — then use WHERE/RETURN/ORDER BY/LIMIT as usual\">\n");
    xml.push_str("      <proc name=\"pagerank\" params=\"damping_factor(0.85), max_iterations(100), tolerance(1e-6), connection_types\" yields=\"node, score\">Importance via incoming links.</proc>\n");
    xml.push_str("      <proc name=\"betweenness\" params=\"normalized(true), sample_size, connection_types\" yields=\"node, score\">Bridge nodes on shortest paths.</proc>\n");
    xml.push_str("      <proc name=\"degree\" params=\"normalized(true), connection_types\" yields=\"node, score\">Connection count per node.</proc>\n");
    xml.push_str("      <proc name=\"closeness\" params=\"normalized(true), connection_types\" yields=\"node, score\">Proximity to all other nodes.</proc>\n");
    xml.push_str("      <proc name=\"louvain\" params=\"resolution(1.0), weight_property, connection_types\" yields=\"node, community\">Community detection via modularity.</proc>\n");
    xml.push_str("      <proc name=\"label_propagation\" params=\"max_iterations(100), connection_types\" yields=\"node, community\">Community detection via label spread.</proc>\n");
    xml.push_str("      <proc name=\"connected_components\" params=\"\" yields=\"node, component\">Weakly connected component membership.</proc>\n");
    xml.push_str("    </procedures>\n");

    // Notes — technical details (workflow tips are in <quick_start>)
    xml.push_str("    <notes>\n");
    xml.push_str("      <note>Each node has exactly one type. labels(n) returns a string, not a list.</note>\n");
    xml.push_str("      <note>Built-in node fields: type, title, id. Access via n.type, n.title, n.id.</note>\n");
    xml.push_str("      <note>Label-optional MATCH: (n {name: 'x'}) searches all node types. Use n.type or labels(n) to see the type.</note>\n");
    xml.push_str("      <note>If a node type has id_alias/title_alias attributes, the original column name also works as a property accessor (e.g. n.npdid resolves to n.id).</note>\n");
    xml.push_str("      <note>Each cypher() call is atomic. Params via $param syntax. to_df=True returns a pandas DataFrame.</note>\n");
    xml.push_str(
        "      <note>CALL proc() YIELD node, score — node is a node binding (use node.title, node.type, etc. in RETURN/WHERE).</note>\n",
    );
    if has_embeddings {
        xml.push_str(
            "      <note>text_score(n, 'col', 'query') — semantic similarity (0..1). Use text_col value from &lt;emb&gt; as 'col'. Usable in WHERE/RETURN/ORDER BY.</note>\n",
        );
    }
    if has_code_entities {
        xml.push_str(
            "      <note>Qualified name formats — Rust: crate::module::Type::method, Python: package.module.Class.method, TS/JS: module.Class.method.</note>\n",
        );
    }
    xml.push_str("    </notes>\n");
    xml.push_str("  </cypher_ref>\n");
    xml.push_str("</kglite>");

    xml
}
