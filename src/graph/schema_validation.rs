// src/graph/schema_validation.rs
//! Schema validation module for validating graph data against a defined schema.

use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, NodeSchemaDefinition, SchemaDefinition, ValidationError};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use std::collections::HashMap;

/// Validate the graph against the provided schema definition
pub fn validate_graph(
    graph: &DirGraph,
    schema: &SchemaDefinition,
    strict: bool, // If true, report undefined types as errors
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Validate nodes
    errors.extend(validate_nodes(graph, schema, strict));

    // Validate connections
    errors.extend(validate_connections(graph, schema, strict));

    errors
}

/// Validate all nodes against the schema
fn validate_nodes(
    graph: &DirGraph,
    schema: &SchemaDefinition,
    strict: bool,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Check each node type defined in schema
    for (node_type, node_schema) in &schema.node_schemas {
        if let Some(node_indices) = graph.type_indices.get(node_type) {
            for &node_idx in node_indices {
                if let Some(node) = graph.get_node(node_idx) {
                    errors.extend(validate_single_node(node, node_type, node_schema));
                }
            }
        }
    }

    // Check for undefined node types (strict mode)
    if strict {
        for (node_type, node_indices) in &graph.type_indices {
            if !schema.node_schemas.contains_key(node_type) {
                errors.push(ValidationError::UndefinedNodeType {
                    node_type: node_type.clone(),
                    count: node_indices.len(),
                });
            }
        }
    }

    errors
}

/// Validate a single node against its schema
fn validate_single_node(
    node: &crate::graph::schema::NodeData,
    node_type: &str,
    schema: &NodeSchemaDefinition,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let title = match &node.title {
        Value::String(s) => s.clone(),
        _ => format!("{:?}", node.title),
    };
    let properties = &node.properties;

    // Check required fields
    for required_field in &schema.required_fields {
        // Skip built-in fields that are always present
        if required_field == "id" || required_field == "title" || required_field == "type" {
            continue;
        }

        let has_field = properties
            .get(required_field)
            .map(|v| !matches!(v, Value::Null))
            .unwrap_or(false);

        if !has_field {
            errors.push(ValidationError::MissingRequiredField {
                node_type: node_type.to_string(),
                node_title: title.clone(),
                field: required_field.clone(),
            });
        }
    }

    // Check field types
    for (field, expected_type) in &schema.field_types {
        if let Some(value) = properties.get(field) {
            if !value_matches_type(value, expected_type) {
                errors.push(ValidationError::TypeMismatch {
                    node_type: node_type.to_string(),
                    node_title: title.clone(),
                    field: field.clone(),
                    expected_type: expected_type.clone(),
                    actual_type: get_value_type_name(value),
                });
            }
        }
    }

    errors
}

/// Validate all connections against the schema
fn validate_connections(
    graph: &DirGraph,
    schema: &SchemaDefinition,
    strict: bool,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let mut connection_type_counts: HashMap<String, usize> = HashMap::new();

    // Iterate through all edges
    for edge_ref in graph.graph.edge_references() {
        let edge_data = edge_ref.weight();
        let connection_type = &edge_data.connection_type;

        // Count connection types for strict mode check
        *connection_type_counts
            .entry(connection_type.clone())
            .or_insert(0) += 1;

        // If there's a schema for this connection type, validate it
        if let Some(conn_schema) = schema.connection_schemas.get(connection_type) {
            let source_idx = edge_ref.source();
            let target_idx = edge_ref.target();

            // Get source and target node info
            let (source_type, source_title) = get_node_info(graph, source_idx);
            let (target_type, target_title) = get_node_info(graph, target_idx);

            // Validate endpoint types
            if source_type != conn_schema.source_type || target_type != conn_schema.target_type {
                errors.push(ValidationError::InvalidConnectionEndpoint {
                    connection_type: connection_type.clone(),
                    expected_source: conn_schema.source_type.clone(),
                    expected_target: conn_schema.target_type.clone(),
                    actual_source: source_type,
                    actual_target: target_type,
                });
            }

            // Validate required properties
            for required_prop in &conn_schema.required_properties {
                let has_prop = edge_data
                    .properties
                    .get(required_prop)
                    .map(|v| !matches!(v, Value::Null))
                    .unwrap_or(false);

                if !has_prop {
                    errors.push(ValidationError::MissingConnectionProperty {
                        connection_type: connection_type.clone(),
                        source_title: source_title.clone(),
                        target_title: target_title.clone(),
                        property: required_prop.clone(),
                    });
                }
            }
        }
    }

    // Check for undefined connection types (strict mode)
    if strict {
        for (conn_type, count) in connection_type_counts {
            if !schema.connection_schemas.contains_key(&conn_type) {
                errors.push(ValidationError::UndefinedConnectionType {
                    connection_type: conn_type,
                    count,
                });
            }
        }
    }

    errors
}

/// Get node type and title from a node index
fn get_node_info(graph: &DirGraph, node_idx: petgraph::graph::NodeIndex) -> (String, String) {
    match graph.get_node(node_idx) {
        Some(node) => {
            let title_str = match &node.title {
                Value::String(s) => s.clone(),
                _ => format!("{:?}", node.title),
            };
            (node.node_type.clone(), title_str)
        }
        None => ("Unknown".to_string(), "Unknown".to_string()),
    }
}

/// Check if a value matches the expected type
fn value_matches_type(value: &Value, expected_type: &str) -> bool {
    match expected_type.to_lowercase().as_str() {
        "string" | "str" => matches!(value, Value::String(_)),
        "integer" | "int" | "i64" => matches!(value, Value::Int64(_) | Value::UniqueId(_)),
        "float" | "double" | "f64" | "number" => {
            matches!(
                value,
                Value::Float64(_) | Value::Int64(_) | Value::UniqueId(_)
            )
        }
        "boolean" | "bool" => matches!(value, Value::Boolean(_)),
        "datetime" | "date" => matches!(value, Value::DateTime(_)),
        "null" => matches!(value, Value::Null),
        "any" => true, // Any type is valid
        _ => true,     // Unknown types default to valid (permissive)
    }
}

/// Get a human-readable type name for a value
fn get_value_type_name(value: &Value) -> String {
    match value {
        Value::String(_) => "string".to_string(),
        Value::Int64(_) => "integer".to_string(),
        Value::Float64(_) => "float".to_string(),
        Value::Boolean(_) => "boolean".to_string(),
        Value::DateTime(_) => "datetime".to_string(),
        Value::UniqueId(_) => "integer".to_string(),
        Value::Point { .. } => "point".to_string(),
        Value::Null => "null".to_string(),
        Value::NodeRef(_) => "noderef".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_matches_type() {
        assert!(value_matches_type(
            &Value::String("test".to_string()),
            "string"
        ));
        assert!(value_matches_type(&Value::Int64(42), "integer"));
        assert!(value_matches_type(&Value::Float64(3.14), "float"));
        assert!(value_matches_type(&Value::Boolean(true), "boolean"));
        assert!(!value_matches_type(
            &Value::String("test".to_string()),
            "integer"
        ));
    }
}
