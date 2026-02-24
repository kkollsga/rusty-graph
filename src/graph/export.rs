// src/graph/export.rs
//! Export graph data to various visualization formats

use crate::datatypes::values::Value;
use crate::graph::schema::{CurrentSelection, DirGraph};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Export the graph (or selection) to GraphML format.
///
/// GraphML is an XML-based format supported by many graph visualization tools
/// including Gephi, yEd, and Cytoscape.
pub fn to_graphml(
    graph: &DirGraph,
    selection: Option<&CurrentSelection>,
) -> Result<String, String> {
    let mut xml = String::with_capacity(64 * 1024); // Pre-allocate 64KB

    // XML header
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n");
    xml.push_str("         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
    xml.push_str("         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns\n");
    xml.push_str("         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n");

    // Define attribute keys for nodes
    xml.push_str(
        "  <key id=\"node_type\" for=\"node\" attr.name=\"type\" attr.type=\"string\"/>\n",
    );
    xml.push_str(
        "  <key id=\"node_title\" for=\"node\" attr.name=\"title\" attr.type=\"string\"/>\n",
    );
    xml.push_str("  <key id=\"node_id\" for=\"node\" attr.name=\"id\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"node_properties\" for=\"node\" attr.name=\"properties\" attr.type=\"string\"/>\n");

    // Define attribute keys for edges
    xml.push_str("  <key id=\"edge_type\" for=\"edge\" attr.name=\"connection_type\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"edge_properties\" for=\"edge\" attr.name=\"properties\" attr.type=\"string\"/>\n");

    xml.push_str("  <graph id=\"G\" edgedefault=\"directed\">\n");

    // Determine which nodes to export
    let node_indices: Vec<_> = if let Some(sel) = selection {
        let level_idx = sel.get_level_count().saturating_sub(1);
        if let Some(level) = sel.get_level(level_idx) {
            level.get_all_nodes()
        } else {
            graph.graph.node_indices().collect()
        }
    } else {
        graph.graph.node_indices().collect()
    };

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Export nodes
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_weight(idx) {
            xml.push_str(&format!("    <node id=\"n{}\">\n", idx.index()));
            xml.push_str(&format!(
                "      <data key=\"node_type\">{}</data>\n",
                escape_xml(&node.node_type)
            ));
            xml.push_str(&format!(
                "      <data key=\"node_title\">{}</data>\n",
                escape_xml(&value_to_string(&node.title))
            ));
            xml.push_str(&format!(
                "      <data key=\"node_id\">{}</data>\n",
                escape_xml(&value_to_string(&node.id))
            ));

            // Serialize properties as JSON
            if !node.properties.is_empty() {
                let props_json = properties_to_json(&node.properties);
                xml.push_str(&format!(
                    "      <data key=\"node_properties\">{}</data>\n",
                    escape_xml(&props_json)
                ));
            }

            xml.push_str("    </node>\n");
        }
    }

    // Export edges (only between selected nodes)
    let mut edge_id = 0;
    for &source_idx in &node_indices {
        for edge in graph.graph.edges(source_idx) {
            let target_idx = edge.target();

            // Only include edge if target is in selection
            if node_set.contains(&target_idx) {
                xml.push_str(&format!(
                    "    <edge id=\"e{}\" source=\"n{}\" target=\"n{}\">\n",
                    edge_id,
                    source_idx.index(),
                    target_idx.index()
                ));
                xml.push_str(&format!(
                    "      <data key=\"edge_type\">{}</data>\n",
                    escape_xml(&edge.weight().connection_type)
                ));

                if !edge.weight().properties.is_empty() {
                    let props_json = properties_to_json(&edge.weight().properties);
                    xml.push_str(&format!(
                        "      <data key=\"edge_properties\">{}</data>\n",
                        escape_xml(&props_json)
                    ));
                }

                xml.push_str("    </edge>\n");
                edge_id += 1;
            }
        }
    }

    xml.push_str("  </graph>\n");
    xml.push_str("</graphml>\n");

    Ok(xml)
}

/// Export the graph (or selection) to D3.js compatible JSON format.
///
/// This format is designed for use with D3.js force-directed graph visualizations.
/// The output is a JSON object with "nodes" and "links" arrays.
pub fn to_d3_json(
    graph: &DirGraph,
    selection: Option<&CurrentSelection>,
) -> Result<String, String> {
    // Determine which nodes to export
    let node_indices: Vec<_> = if let Some(sel) = selection {
        let level_idx = sel.get_level_count().saturating_sub(1);
        if let Some(level) = sel.get_level(level_idx) {
            level.get_all_nodes()
        } else {
            graph.graph.node_indices().collect()
        }
    } else {
        graph.graph.node_indices().collect()
    };

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Build index mapping (old index -> array position)
    let mut index_map: HashMap<usize, usize> = HashMap::with_capacity(node_indices.len());
    for (pos, &idx) in node_indices.iter().enumerate() {
        index_map.insert(idx.index(), pos);
    }

    // Build nodes array
    let mut nodes_json = Vec::with_capacity(node_indices.len());
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_weight(idx) {
            let mut obj = String::from("{");
            obj.push_str(&format!("\"id\":{},", json_value(&node.id)));
            obj.push_str(&format!("\"type\":{},", json_string(&node.node_type)));
            obj.push_str(&format!("\"title\":{}", json_value(&node.title)));

            // Add select properties (not all to keep output clean)
            for (key, value) in &node.properties {
                if key != "id" && key != "title" && key != "type" {
                    obj.push_str(&format!(",{}:{}", json_string(key), json_value(value)));
                }
            }

            obj.push('}');
            nodes_json.push(obj);
        }
    }

    // Build links array
    let mut links_json = Vec::new();
    for &source_idx in &node_indices {
        for edge in graph.graph.edges(source_idx) {
            let target_idx = edge.target();

            if node_set.contains(&target_idx) {
                if let (Some(&source_pos), Some(&target_pos)) = (
                    index_map.get(&source_idx.index()),
                    index_map.get(&target_idx.index()),
                ) {
                    let mut link = String::from("{");
                    link.push_str(&format!("\"source\":{},", source_pos));
                    link.push_str(&format!("\"target\":{},", target_pos));
                    link.push_str(&format!(
                        "\"type\":{}",
                        json_string(&edge.weight().connection_type)
                    ));

                    // Add edge properties
                    for (key, value) in &edge.weight().properties {
                        link.push_str(&format!(",{}:{}", json_string(key), json_value(value)));
                    }

                    link.push('}');
                    links_json.push(link);
                }
            }
        }
    }

    // Build final JSON
    let mut result = String::with_capacity(32 * 1024);
    result.push_str("{\n  \"nodes\": [\n    ");
    result.push_str(&nodes_json.join(",\n    "));
    result.push_str("\n  ],\n  \"links\": [\n    ");
    result.push_str(&links_json.join(",\n    "));
    result.push_str("\n  ]\n}");

    Ok(result)
}

/// Export to GEXF format (Gephi native format).
///
/// GEXF is the native format for Gephi and supports dynamic graphs,
/// hierarchies, and rich attribute types.
pub fn to_gexf(graph: &DirGraph, selection: Option<&CurrentSelection>) -> Result<String, String> {
    let mut xml = String::with_capacity(64 * 1024);

    // XML header
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<gexf xmlns=\"http://www.gexf.net/1.2draft\"\n");
    xml.push_str("      xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
    xml.push_str("      xsi:schemaLocation=\"http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd\"\n");
    xml.push_str("      version=\"1.2\">\n");
    xml.push_str("  <meta>\n");
    xml.push_str("    <creator>kglite</creator>\n");
    xml.push_str("    <description>Exported from KnowledgeGraph</description>\n");
    xml.push_str("  </meta>\n");
    xml.push_str("  <graph mode=\"static\" defaultedgetype=\"directed\">\n");

    // Define node attributes
    xml.push_str("    <attributes class=\"node\">\n");
    xml.push_str("      <attribute id=\"0\" title=\"type\" type=\"string\"/>\n");
    xml.push_str("      <attribute id=\"1\" title=\"title\" type=\"string\"/>\n");
    xml.push_str("    </attributes>\n");

    // Define edge attributes
    xml.push_str("    <attributes class=\"edge\">\n");
    xml.push_str("      <attribute id=\"0\" title=\"connection_type\" type=\"string\"/>\n");
    xml.push_str("    </attributes>\n");

    // Determine which nodes to export
    let node_indices: Vec<_> = if let Some(sel) = selection {
        let level_idx = sel.get_level_count().saturating_sub(1);
        if let Some(level) = sel.get_level(level_idx) {
            level.get_all_nodes()
        } else {
            graph.graph.node_indices().collect()
        }
    } else {
        graph.graph.node_indices().collect()
    };

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Export nodes
    xml.push_str("    <nodes>\n");
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_weight(idx) {
            let title_str = value_to_string(&node.title);
            xml.push_str(&format!(
                "      <node id=\"{}\" label=\"{}\">\n",
                idx.index(),
                escape_xml(&title_str)
            ));
            xml.push_str("        <attvalues>\n");
            xml.push_str(&format!(
                "          <attvalue for=\"0\" value=\"{}\"/>\n",
                escape_xml(&node.node_type)
            ));
            xml.push_str(&format!(
                "          <attvalue for=\"1\" value=\"{}\"/>\n",
                escape_xml(&title_str)
            ));
            xml.push_str("        </attvalues>\n");
            xml.push_str("      </node>\n");
        }
    }
    xml.push_str("    </nodes>\n");

    // Export edges
    xml.push_str("    <edges>\n");
    let mut edge_id = 0;
    for &source_idx in &node_indices {
        for edge in graph.graph.edges(source_idx) {
            let target_idx = edge.target();

            if node_set.contains(&target_idx) {
                xml.push_str(&format!(
                    "      <edge id=\"{}\" source=\"{}\" target=\"{}\">\n",
                    edge_id,
                    source_idx.index(),
                    target_idx.index()
                ));
                xml.push_str("        <attvalues>\n");
                xml.push_str(&format!(
                    "          <attvalue for=\"0\" value=\"{}\"/>\n",
                    escape_xml(&edge.weight().connection_type)
                ));
                xml.push_str("        </attvalues>\n");
                xml.push_str("      </edge>\n");
                edge_id += 1;
            }
        }
    }
    xml.push_str("    </edges>\n");

    xml.push_str("  </graph>\n");
    xml.push_str("</gexf>\n");

    Ok(xml)
}

/// Export to CSV format (nodes and edges as separate content).
///
/// Returns a tuple of (nodes_csv, edges_csv).
pub fn to_csv(
    graph: &DirGraph,
    selection: Option<&CurrentSelection>,
) -> Result<(String, String), String> {
    // Determine which nodes to export
    let node_indices: Vec<_> = if let Some(sel) = selection {
        let level_idx = sel.get_level_count().saturating_sub(1);
        if let Some(level) = sel.get_level(level_idx) {
            level.get_all_nodes()
        } else {
            graph.graph.node_indices().collect()
        }
    } else {
        graph.graph.node_indices().collect()
    };

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Build nodes CSV
    let mut nodes_csv = String::from("id,type,title\n");
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_weight(idx) {
            nodes_csv.push_str(&format!(
                "{},{},{}\n",
                idx.index(),
                escape_csv(&node.node_type),
                escape_csv(&value_to_string(&node.title))
            ));
        }
    }

    // Build edges CSV
    let mut edges_csv = String::from("source,target,type\n");
    for &source_idx in &node_indices {
        for edge in graph.graph.edges(source_idx) {
            let target_idx = edge.target();

            if node_set.contains(&target_idx) {
                edges_csv.push_str(&format!(
                    "{},{},{}\n",
                    source_idx.index(),
                    target_idx.index(),
                    escape_csv(&edge.weight().connection_type)
                ));
            }
        }
    }

    Ok((nodes_csv, edges_csv))
}

// Helper functions

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Int64(n) => n.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::DateTime(dt) => dt.to_string(),
        Value::UniqueId(id) => id.to_string(),
        Value::Point { lat, lon } => format!("point({}, {})", lat, lon),
        Value::Null => String::new(),
        Value::NodeRef(idx) => format!("node#{}", idx),
    }
}

fn json_string(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    )
}

fn json_value(value: &Value) -> String {
    match value {
        Value::String(s) => json_string(s),
        Value::Int64(n) => n.to_string(),
        Value::Float64(f) => {
            if f.is_nan() || f.is_infinite() {
                "null".to_string()
            } else {
                f.to_string()
            }
        }
        Value::Boolean(b) => b.to_string(),
        Value::DateTime(dt) => json_string(&dt.to_string()),
        Value::UniqueId(id) => id.to_string(),
        Value::Point { lat, lon } => format!("{{\"lat\":{},\"lon\":{}}}", lat, lon),
        Value::Null => "null".to_string(),
        Value::NodeRef(idx) => idx.to_string(),
    }
}

fn properties_to_json(properties: &HashMap<String, Value>) -> String {
    let pairs: Vec<String> = properties
        .iter()
        .map(|(k, v)| format!("{}:{}", json_string(k), json_value(v)))
        .collect();
    format!("{{{}}}", pairs.join(","))
}
