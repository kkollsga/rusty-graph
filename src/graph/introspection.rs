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

pub struct ConnectionTypeStats {
    pub connection_type: String,
    pub count: usize,
    pub source_types: Vec<String>,
    pub target_types: Vec<String>,
}

pub struct NodeTypeOverview {
    pub count: usize,
    pub properties: HashMap<String, String>,
}

pub struct SchemaOverview {
    pub node_types: Vec<(String, NodeTypeOverview)>,
    pub connection_types: Vec<ConnectionTypeStats>,
    pub indexes: Vec<String>,
    pub node_count: usize,
    pub edge_count: usize,
}

pub struct PropertyStatInfo {
    pub property_name: String,
    pub type_string: String,
    pub non_null: usize,
    pub unique: usize,
    pub values: Option<Vec<Value>>,
}

pub struct NeighborConnection {
    pub connection_type: String,
    pub other_type: String,
    pub count: usize,
}

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
        Value::Null => "unknown",
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

/// Static XML fragment: API methods and Cypher reference (never changes per graph).
const STATIC_XML: &str = r#"  <api>
    <method sig="cypher(query, *, to_df=False, params=None)">Primary query interface. Returns list[dict] or DataFrame.</method>
    <method sig="schema()">Returns dict: node_types, connection_types, indexes, node_count, edge_count.</method>
    <method sig="properties(node_type, max_values=20)">Per-property stats: type, non_null, unique, values.</method>
    <method sig="sample(node_type, n=5)">Returns first N nodes as list[dict].</method>
    <method sig="save(path) / load(path)">Persist and reload the graph.</method>
  </api>
  <cypher_ref>
    <clauses>MATCH, OPTIONAL MATCH, WHERE, RETURN, WITH, ORDER BY, SKIP, LIMIT, UNWIND, UNION, UNION ALL, CREATE, SET, DELETE, DETACH DELETE, REMOVE, MERGE, EXPLAIN</clauses>
    <patterns>(n:Type), (n:Type {key: val}), -[:REL]-&gt;, &lt;-[:REL]-, -[:REL]-, -[:REL*1..3]-&gt;, p = shortestPath(...)</patterns>
    <where>=, &lt;&gt;, &lt;, &gt;, &lt;=, &gt;=, =~ (regex), AND, OR, NOT, IS NULL, IS NOT NULL, IN [...], CONTAINS, STARTS WITH, ENDS WITH, EXISTS { pattern }, EXISTS(( pattern ))</where>
    <return>n.prop, r.prop, AS alias, DISTINCT, arithmetic (+, -, *, /), map projections n {.prop1, .prop2}</return>
    <aggregation>count(*), count(expr), sum, avg, min, max, collect, std</aggregation>
    <expressions>CASE WHEN...THEN...ELSE...END, $param, [x IN list WHERE ... | expr]</expressions>
    <functions>toUpper, toLower, toString, toInteger, toFloat, size, length, type, id, labels, coalesce, nodes(p), relationships(p)</functions>
    <mutations>CREATE (n:Label {props}), CREATE (a)-[:TYPE]-&gt;(b), SET n.prop = expr, DELETE, DETACH DELETE, REMOVE n.prop, MERGE...ON CREATE SET...ON MATCH SET</mutations>
    <not_supported>CALL/stored procedures, FOREACH, subqueries, SET n:Label, REMOVE n:Label, multi-label</not_supported>
    <notes>
      <note>Each node has exactly one type. labels(n) returns a string, not a list.</note>
      <note>Built-in node fields: type, title, id. Access via n.type, n.title, n.id.</note>
      <note>If a node type has id_alias/title_alias attributes, the original column name also works as a property accessor (e.g. n.npdid resolves to n.id).</note>
      <note>Each cypher() call is atomic. Params via $param syntax.</note>
      <note>to_df=True returns a pandas DataFrame instead of list[dict].</note>
    </notes>
  </cypher_ref>"#;

/// Minimal XML escaping for attribute values.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Build a minimal XML description of the graph for AI agents.
pub fn compute_agent_description(graph: &DirGraph) -> String {
    let overview = compute_schema(graph);
    let mut xml = String::with_capacity(2048 + STATIC_XML.len());

    xml.push_str(&format!(
        "<kglite nodes=\"{}\" edges=\"{}\">\n",
        overview.node_count, overview.edge_count
    ));

    // Node types
    if overview.node_types.is_empty() {
        xml.push_str("  <node_types/>\n");
    } else {
        xml.push_str("  <node_types>\n");
        for (nt, info) in &overview.node_types {
            // Build alias attributes if original field names differ from canonical
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
                for (pname, ptype) in props {
                    xml.push_str(&format!(
                        "      <prop name=\"{}\" type=\"{}\"/>\n",
                        xml_escape(pname),
                        xml_escape(ptype)
                    ));
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

    // Static sections
    xml.push_str(STATIC_XML);
    xml.push('\n');
    xml.push_str("</kglite>");

    xml
}
