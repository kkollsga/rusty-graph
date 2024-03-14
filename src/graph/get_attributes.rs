use pyo3::prelude::*;
use pyo3::types::PyDict;
use petgraph::graph::DiGraph;
use crate::node::Node;
use crate::relation::Relation;

pub fn get_node_attributes(
    graph: &mut DiGraph<Node, Relation>, // Pass a reference to the KnowledgeGraph
    py: Python,
    indices: Vec<usize>,
    specified_attributes: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);

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

            result.set_item(index, node_attributes)?;
        }
    }

    Ok(result.into())
}