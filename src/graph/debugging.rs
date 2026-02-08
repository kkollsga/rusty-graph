// src/graph/debugging.rs
use crate::datatypes::values::Value;
use crate::graph::schema::{CurrentSelection, DirGraph, NodeData, SelectionOperation};

pub fn get_schema_string(graph: &DirGraph) -> String {
    let mut schema_string = String::from("Graph Schema:\n");
    let mut schema_nodes = Vec::new();

    for i in graph.graph.node_indices() {
        if let Some(node_data) = graph.get_node(i) {
            match node_data {
                NodeData::Regular { node_type, .. } | NodeData::Schema { node_type, .. } => {
                    if node_type == "SchemaNode" {
                        schema_nodes.push(i);
                    }
                }
            }
        }
    }

    if schema_nodes.is_empty() {
        schema_string.push_str("  No schema nodes found.\n");
        return schema_string;
    }

    for &node_index in &schema_nodes {
        if let Some(node_data) = graph.get_node(node_index) {
            match node_data {
                NodeData::Schema {
                    title, properties, ..
                } => {
                    if properties.contains_key("source_type")
                        && properties.contains_key("target_type")
                    {
                        // Connection schema
                        let source_type = properties
                            .get("source_type")
                            .and_then(|v| v.as_string())
                            .unwrap_or_else(|| "Unknown".to_string());

                        let target_type = properties
                            .get("target_type")
                            .and_then(|v| v.as_string())
                            .unwrap_or_else(|| "Unknown".to_string());

                        let connection_type = match title {
                            Value::String(s) => s.clone(),
                            _ => "Unknown".to_string(),
                        };

                        schema_string.push_str(&format!(
                            "  Connection Type: {} ({} -> {})\n",
                            connection_type, source_type, target_type
                        ));

                        // Filter out source_type and target_type from properties
                        let filtered_properties: Vec<(&String, &Value)> = properties
                            .iter()
                            .filter(|(k, _)| *k != "source_type" && *k != "target_type")
                            .collect();

                        if !filtered_properties.is_empty() {
                            schema_string.push_str("    Properties:\n");
                            for (key, value) in filtered_properties {
                                if let Value::String(type_name) = value {
                                    schema_string
                                        .push_str(&format!("      - {}: {}\n", key, type_name));
                                }
                            }
                        }
                    } else {
                        // Node schema
                        let node_type = match title {
                            Value::String(s) => s.clone(),
                            _ => "Unknown".to_string(),
                        };

                        schema_string.push_str(&format!("  Node Type: {}\n", node_type));

                        if !properties.is_empty() {
                            schema_string.push_str("    Properties:\n");
                            for (key, value) in properties {
                                if let Value::String(type_name) = value {
                                    schema_string
                                        .push_str(&format!("      - {}: {}\n", key, type_name));
                                }
                            }
                        }
                    }
                }
                _ => {} // Skip non-schema nodes
            }
            schema_string.push('\n');
        }
    }
    schema_string
}

pub fn get_selection_string(graph: &DirGraph, selection: &CurrentSelection) -> String {
    let mut output = String::from("Selection State:\n");
    if selection.get_level_count() == 0 {
        output.push_str("    No active selections\n");
        return output;
    }

    // Iterate through levels using level count and get_level
    for level_idx in 0..selection.get_level_count() {
        let level = selection
            .get_level(level_idx)
            .expect("Level should exist as we checked count");

        output.push_str(&format!("\nLevel {}:\n", level_idx));

        // Print operations that created this level
        if !level.operations.is_empty() {
            output.push_str("    Operations:\n");
            for op in &level.operations {
                match op {
                    SelectionOperation::Filter(conditions) => {
                        output.push_str("        Filter: ");
                        let conditions_str: Vec<String> = conditions
                            .iter()
                            .map(|(key, condition)| format!("{}={:?}", key, condition))
                            .collect();
                        output.push_str(&conditions_str.join(", "));
                        output.push('\n');
                    }
                    SelectionOperation::Sort(sort_fields) => {
                        output.push_str("        Sort: ");
                        let sorts_str: Vec<String> = sort_fields
                            .iter()
                            .map(|(field, ascending)| {
                                format!("{} {}", field, if *ascending { "↑" } else { "↓" })
                            })
                            .collect();
                        output.push_str(&sorts_str.join(", "));
                        output.push('\n');
                    }
                    SelectionOperation::Traverse {
                        connection_type,
                        direction,
                        max_nodes,
                    } => {
                        let mut traverse_parts = vec![format!("type={}", connection_type)];
                        if let Some(dir) = direction {
                            traverse_parts.push(format!("dir={}", dir));
                        }
                        if let Some(max) = max_nodes {
                            traverse_parts.push(format!("max={}", max));
                        }
                        output.push_str("        Traverse: ");
                        output.push_str(&traverse_parts.join(", "));
                        output.push('\n');
                    }
                    SelectionOperation::Custom(description) => {
                        output.push_str("        Custom: ");
                        output.push_str(description);
                        output.push('\n');
                    }
                }
            }
        }

        // Print selection structure
        output.push_str("    Selections:\n");
        for (parent, children) in level.iter_groups() {
            match parent {
                Some(parent_idx) => {
                    if let Some(parent_node) = graph.get_node(*parent_idx) {
                        match parent_node {
                            NodeData::Regular {
                                node_type, title, ..
                            }
                            | NodeData::Schema {
                                node_type, title, ..
                            } => {
                                let title_str = match title {
                                    Value::String(s) => s.clone(),
                                    _ => format!("{:?}", title),
                                };
                                output.push_str(&format!(
                                    "        Parent [{}]: {} - {}\n",
                                    parent_idx.index(),
                                    node_type,
                                    title_str
                                ));
                            }
                        }
                    }
                }
                None => output.push_str("        Root Selection\n"),
            }

            // Add children info with node type if available
            if !children.is_empty() {
                let first_child = graph.get_node(children[0]);
                let child_type = first_child.map(|node| match node {
                    NodeData::Regular { node_type, .. } | NodeData::Schema { node_type, .. } => {
                        node_type
                    }
                });

                let indices: Vec<String> = children
                    .iter()
                    .take(5)
                    .map(|idx| idx.index().to_string())
                    .collect();

                let indices_str = if children.len() > 5 {
                    format!("[{}, ...]", indices.join(", "))
                } else {
                    format!("[{}]", indices.join(", "))
                };

                output.push_str(&format!(
                    "            Children: {} {}{}\n",
                    children.len(),
                    child_type.map(|t| format!("({})", t)).unwrap_or_default(),
                    if children.len() <= 5 {
                        format!(": {}", indices_str)
                    } else {
                        format!(", e.g. {}", indices_str)
                    }
                ));
            }
        }
    }
    output
}
