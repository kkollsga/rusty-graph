use petgraph::graph::DiGraph;
use petgraph::Direction;
use petgraph::visit::EdgeRef;  // Added for edge.source() and edge.target()
use std::collections::{HashMap, HashSet};
use crate::schema::{Node, Relation, NodeTypeStats, AttributeMetadata, RelationshipMetadata};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Updates or retrieves the schema (DataTypeNode) from the graph
///
/// # Arguments
///
/// * `graph` - The graph object containing all nodes and relations
/// * `data_type` - The type of the data as a string (e.g., "Node" or "Relation")
/// * `name` - The name of the DataTypeNode
/// * `columns` - Optional list of columns to update in the DataTypeNode
/// * `column_types` - Optional mapping of column names to their data types
pub fn update_or_retrieve_schema(
    graph: &mut DiGraph<Node, Relation>,
    data_type: &str,
    name: &str,
    types: Option<HashMap<String, String>>,
) -> PyResult<HashMap<String, String>> {
    let schema_node = graph.node_indices().find(|&i| match &graph[i] {
        Node::DataTypeNode { data_type: dt, name: nt, .. } => {
            dt == data_type && nt == name
        },
        _ => false
    });

    match schema_node {
        Some(idx) => {
            if let Some(new_types) = types {
                let mut curr_attrs = if let Node::DataTypeNode { attributes, .. } = &mut graph[idx] {
                    attributes.clone()
                } else {
                    HashMap::new()
                };
                
                // Merge new types with existing
                curr_attrs.extend(new_types);
                
                // Replace entire node to ensure update is atomic
                graph[idx] = Node::new_data_type(data_type, name, curr_attrs.clone());
                
                Ok(curr_attrs)
            } else if let Node::DataTypeNode { attributes, .. } = &graph[idx] {
                Ok(attributes.clone())
            } else {
                unreachable!()
            }
        },
        None => {
            let attributes = types.unwrap_or_default();
            let node = Node::new_data_type(data_type, name, attributes.clone());
            graph.add_node(node);
            Ok(attributes)
        }
    }
}

pub fn get_node_schemas(
    graph: &DiGraph<Node, Relation>
) -> PyResult<HashMap<String, NodeTypeStats>> {
    let mut schemas: HashMap<String, NodeTypeStats> = HashMap::new();
    let mut temp_nodes: HashMap<String, Vec<&Node>> = HashMap::new();

    // First pass: collect all nodes by type and find DataTypeNodes
    for node_idx in graph.node_indices() {
        match &graph[node_idx] {
            Node::StandardNode { node_type, .. } => {
                temp_nodes
                    .entry(node_type.clone())
                    .or_default()
                    .push(&graph[node_idx]);
            }
            Node::DataTypeNode { data_type, name, attributes } if data_type == "Node" => {
                schemas.insert(name.clone(), NodeTypeStats {
                    title: "String".to_string(),
                    graph_id: "String".to_string(),
                    attributes: attributes
                        .iter()
                        .map(|(k, v)| (k.clone(), AttributeMetadata {
                            data_type: v.clone(),
                            nullable: false,
                            unique_values: None,
                        }))
                        .collect(),
                    occurrences: 0,
                    relationships: RelationshipMetadata {
                        incoming_types: HashSet::new(),
                        outgoing_types: HashSet::new(),
                        connected_node_types: HashSet::new(),
                    },
                });
            }
            _ => {}
        }
    }

    // Second pass: analyze nodes and relationships
    for (node_type, nodes) in temp_nodes {
        if let Some(stats) = schemas.get_mut(&node_type) {
            stats.occurrences = nodes.len();

            let mut null_tracking: HashMap<String, bool> = HashMap::new();
            for node in nodes {
                if let Node::StandardNode { attributes, .. } = node {
                    for (attr_name, attr_value) in attributes {
                        if attr_value.is_null() {
                            null_tracking.insert(attr_name.clone(), true);
                        }
                    }
                }
            }

            for (attr_name, is_null) in null_tracking {
                if let Some(attr_metadata) = stats.attributes.get_mut(&attr_name) {
                    attr_metadata.nullable = is_null;
                }
            }

            for node_idx in graph.node_indices() {
                if let Node::StandardNode { node_type: nt, .. } = &graph[node_idx] {
                    if nt == &node_type {
                        for edge in graph.edges_directed(node_idx, Direction::Outgoing) {
                            if let Node::StandardNode { node_type: target_type, .. } = &graph[edge.target()] {
                                stats.relationships.outgoing_types.insert(edge.weight().relation_type.clone());
                                stats.relationships.connected_node_types.insert(target_type.clone());
                            }
                        }

                        for edge in graph.edges_directed(node_idx, Direction::Incoming) {
                            if let Node::StandardNode { node_type: source_type, .. } = &graph[edge.source()] {
                                stats.relationships.incoming_types.insert(edge.weight().relation_type.clone());
                                stats.relationships.connected_node_types.insert(source_type.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(schemas)
}

/// Gets the complete schema information for the graph
/// Currently returns node schemas, with relation schemas to be implemented
pub fn get_schema(
    py: Python,
    graph: &DiGraph<Node, Relation>,
) -> PyResult<PyObject> {
    let schema_dict = PyDict::new(py);
    
    // Get node schemas
    let node_schemas = get_node_schemas(graph)?;
    
    // Convert node_schemas to Python dictionary
    let node_dict = PyDict::new(py);
    for (node_type, stats) in node_schemas {
        let stats_dict = PyDict::new(py);
        
        stats_dict.set_item("title", stats.title)?;
        stats_dict.set_item("graph_id", stats.graph_id)?;
        stats_dict.set_item("occurrences", stats.occurrences)?;
        
        // Convert attributes
        let attr_dict = PyDict::new(py);
        for (attr_name, attr_meta) in stats.attributes {
            let meta_dict = PyDict::new(py);
            meta_dict.set_item("data_type", attr_meta.data_type)?;
            meta_dict.set_item("nullable", attr_meta.nullable)?;
            if let Some(unique_values) = attr_meta.unique_values {
                meta_dict.set_item("unique_values", unique_values)?;
            }
            attr_dict.set_item(attr_name, meta_dict)?;
        }
        stats_dict.set_item("attributes", attr_dict)?;
        
        // Convert relationships
        let rel_dict = PyDict::new(py);
        rel_dict.set_item("incoming_types", stats.relationships.incoming_types.into_iter().collect::<Vec<_>>())?;
        rel_dict.set_item("outgoing_types", stats.relationships.outgoing_types.into_iter().collect::<Vec<_>>())?;
        rel_dict.set_item("connected_node_types", stats.relationships.connected_node_types.into_iter().collect::<Vec<_>>())?;
        stats_dict.set_item("relationships", rel_dict)?;
        
        node_dict.set_item(node_type, stats_dict)?;
    }
    
    schema_dict.set_item("nodes", node_dict)?;
    // Placeholder for relations schema (to be implemented)
    schema_dict.set_item("relations", PyDict::new(py))?;
    
    Ok(schema_dict.into())
}