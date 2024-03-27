use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue;
use crate::graph::get_schema::retrieve_schema;

pub fn get_node_attributes(
    graph: &mut DiGraph<Node, Relation>,
    py: Python,
    indices: Vec<usize>,
    specified_attributes: Option<Vec<String>>,
    max_relations: Option<usize>,
) -> PyResult<PyObject> {
    let mut result_list = Vec::new();
    let max_relations = max_relations.unwrap_or(10);

    // Initialize an empty HashMap to cache the schemas
    let mut schemas: HashMap<String, HashMap<String, String>> = HashMap::new();

    // First pass: Populate the schemas HashMap for all node types in indices
    for index in &indices {
        let node_index = petgraph::graph::NodeIndex::new(*index);
        if let Some(Node::StandardNode { node_type, .. }) = graph.node_weight(node_index) {
            if !schemas.contains_key(node_type) {
                let schema = retrieve_schema(
                    graph,
                    "Node",
                    node_type,
                ).expect("Failed to fetch schema");
                schemas.insert(node_type.clone(), schema);
            }
        }
    }

    // Main loop: Process each node using the pre-fetched schemas
    for index in indices {
        let node_index = petgraph::graph::NodeIndex::new(index);
        if let Some(Node::StandardNode { node_type, unique_id, attributes, title }) = graph.node_weight(node_index) {
            let schema = schemas.get(node_type).expect("Schema should be present");

            let return_attributes = PyDict::new(py);
            return_attributes.set_item("graph_id", index)?;
            return_attributes.set_item("node_type", node_type)?;
            return_attributes.set_item("unique_id", unique_id)?;
            if let Some(t) = title {
                return_attributes.set_item("title", t)?;
            }

            // Extract and set attributes using the pre-fetched schema
            extract_and_set_attributes(
                py,
                return_attributes,
                attributes,
                schema,
                &specified_attributes,
            )?;

            // Incoming relations
            if specified_attributes.as_ref().map_or(true, |attrs| attrs.contains(&"incoming_relations".to_string())) {
                let incoming = graph.edges_directed(node_index, petgraph::Direction::Incoming)
                    .take(max_relations)
                    .filter_map(|edge| {
                        let source_node_index = edge.source();
                        if let Some(Node::StandardNode { unique_id, title, .. }) = graph.node_weight(source_node_index) {
                            let rel_dict = PyDict::new(py);
                            rel_dict.set_item("relation_type", &edge.weight().relation_type).unwrap();
                            rel_dict.set_item("source_index", source_node_index.index()).unwrap();
                            rel_dict.set_item("source_id", unique_id).unwrap();
                            if let Some(t) = title {
                                rel_dict.set_item("source_title", t).unwrap();
                            }
                            Some(rel_dict.to_object(py))
                        } else {
                            // Handle other node variants if necessary
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                return_attributes.set_item("incoming_relations", incoming)?;
            }

            // Outgoing relations
            if specified_attributes.as_ref().map_or(true, |attrs| attrs.contains(&"outgoing_relations".to_string())) {
                let outgoing = graph.edges_directed(node_index, petgraph::Direction::Outgoing)
                    .take(max_relations)
                    .filter_map(|edge| {
                        let target_node_index = edge.target();
                        if let Some(Node::StandardNode { unique_id, title, .. }) = graph.node_weight(target_node_index) {
                            let rel_dict = PyDict::new(py);
                            rel_dict.set_item("relation_type", &edge.weight().relation_type).unwrap();
                            rel_dict.set_item("target_index", target_node_index.index()).unwrap();
                            rel_dict.set_item("target_id", unique_id).unwrap();
                            if let Some(t) = title {
                                rel_dict.set_item("target_title", t).unwrap();
                            }
                            Some(rel_dict.to_object(py))
                        } else {
                            // Handle other node variants if necessary
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                return_attributes.set_item("outgoing_relations", outgoing)?;
            }

            
            // Add the node_attributes dict to the result_list
            result_list.push(return_attributes);
        }
    }

    // Convert the Vec of node attributes dicts to a PyList before returning
    Ok(PyList::new(py, &result_list).into())
}

fn extract_and_set_attributes(
    py: Python,
    return_attributes: &PyDict,
    attributes: &HashMap<String, AttributeValue>,
    schema: &HashMap<String, String>,
    specified_attributes: &Option<Vec<String>>,
) -> PyResult<()> {
    if let Some(attrs) = specified_attributes {
        for attr in attrs {
            if let Some(value) = attributes.get(attr) {
                let attr_value = value.to_python_object(py, schema.get(attr).map(String::as_str))?;
                return_attributes.set_item(attr, attr_value)?;
            }
        }
    } else {
        for (key, value) in attributes.iter() {
            let attr_value = value.to_python_object(py, schema.get(key).map(String::as_str))?;
            return_attributes.set_item(key, attr_value)?;
        }
    }
    Ok(())
}