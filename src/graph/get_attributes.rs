use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use crate::node::Node;
use crate::relation::Relation;

pub fn get_node_attributes(
    graph: &mut DiGraph<Node, Relation>, // Pass a reference to the KnowledgeGraph
    py: Python,
    indices: Vec<usize>,
    specified_attributes: Option<Vec<String>>,
    max_relations: Option<usize>,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    let max_relations = max_relations.unwrap_or(10); // Default to 10 if None

    for index in indices {
        let node_index = petgraph::graph::NodeIndex::new(index);
        if let Some(node) = graph.node_weight(node_index) {
            let node_attributes = PyDict::new(py);

            match &specified_attributes {
                Some(attrs) => {
                    // Check for default attributes and specified custom attributes
                    for attr in attrs {
                        match attr.as_str() {
                            "node_type" => node_attributes.set_item("node_type", &node.node_type)?,
                            "unique_id" => node_attributes.set_item("unique_id", &node.unique_id)?,
                            "title" => node_attributes.set_item("title", &node.title)?,
                            _ => {
                                if let Some(value) = node.attributes.get(attr) {
                                    node_attributes.set_item(attr, value)?;
                                }
                            }
                        }
                    }
                }
                None => {
                    // Include all attributes if none are specified
                    node_attributes.set_item("node_type", &node.node_type)?;
                    node_attributes.set_item("unique_id", &node.unique_id)?;
                    node_attributes.set_item("title", &node.title)?;

                    for (key, value) in &node.attributes {
                        node_attributes.set_item(key, value)?;
                    }
                }
            }

            // Handle incoming relationships
            let incoming = graph.edges_directed(node_index, petgraph::Direction::Incoming)
                .take(max_relations)
                .map(|edge| {
                    let source_node_index = edge.source();
                    let source_node = graph.node_weight(source_node_index).unwrap(); // Safe unwrap, since the edge exists
                    let rel_dict = PyDict::new(py);
                    rel_dict.set_item("relation_type", edge.weight().relation_type.clone()).unwrap();
                    rel_dict.set_item("source_index", source_node_index.index()).unwrap(); // Safe unwrap in example
                    rel_dict.set_item("source_id", &source_node.unique_id).unwrap();
                    rel_dict.set_item("source_title", &source_node.title).unwrap();
                    rel_dict
                })
                .collect::<Vec<_>>();

            // Handle outgoing relationships
            let outgoing = graph.edges_directed(node_index, petgraph::Direction::Outgoing)
                .take(max_relations)
                .map(|edge| {
                    let target_node_index = edge.target();
                    let target_node = graph.node_weight(target_node_index).unwrap(); // Safe unwrap, since the edge exists
                    let rel_dict = PyDict::new(py);
                    rel_dict.set_item("relation_type", edge.weight().relation_type.clone()).unwrap();
                    rel_dict.set_item("target_index", target_node_index.index()).unwrap(); // Safe unwrap in example
                    rel_dict.set_item("target_id", &target_node.unique_id).unwrap();
                    rel_dict.set_item("target_title", &target_node.title).unwrap();
                    rel_dict
                })
                .collect::<Vec<_>>();

            // Convert Rust Vecs to Python lists
            let py_incoming = PyList::new(py, &incoming);
            let py_outgoing = PyList::new(py, &outgoing);

            node_attributes.set_item("incoming_relationships", py_incoming)?;
            node_attributes.set_item("outgoing_relationships", py_outgoing)?;

            result.set_item(index, node_attributes)?;
        }
    }

    Ok(result.into())
}