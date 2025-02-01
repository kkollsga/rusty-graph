use crate::data_types::AttributeValue;
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use pyo3::prelude::*;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NodeData {
    pub node_type: String,
    pub unique_id: i32,
    pub graph_index: usize,
    pub title: String,
    pub attributes: HashMap<String, AttributeValue>,
    pub calculations: Option<HashMap<String, AttributeValue>>,  // Made optional
    pub traversals: Option<Vec<NodeData>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Node {
    StandardNode {
        node_type: String,
        unique_id: i32,
        attributes: HashMap<String, AttributeValue>,
        calculations: Option<HashMap<String, AttributeValue>>,  // Made optional
        title: Option<String>,
    },
    DataTypeNode {
        data_type: String, // 'Node' or 'Relation'
        name: String,
        attributes: HashMap<String, String>, // Attribute name to data type
        calculations: Option<HashMap<String, String>>, // Made optional
        metadata: HashMap<String, String>,  // For storing calculation metadata
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Relation {
    pub relation_type: String,
    pub attributes: Option<HashMap<String, AttributeValue>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NodeTypeStats {
    pub title: String,
    pub graph_id: String,
    pub attributes: HashMap<String, AttributeMetadata>,
    pub calculations: Option<HashMap<String, AttributeMetadata>>,  // Made optional
    pub occurrences: usize,
    pub relationships: RelationshipMetadata,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AttributeMetadata {
    pub data_type: String,
    pub nullable: bool,
    pub unique_values: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RelationshipMetadata {
    pub incoming_types: HashSet<String>,
    pub outgoing_types: HashSet<String>,
    pub connected_node_types: HashSet<String>,
}

impl Node {
    pub fn new(
        node_type: &str, 
        unique_id: i32, 
        attributes: Option<HashMap<String, AttributeValue>>, 
        node_title: Option<&str>
    ) -> Self {
        Node::StandardNode {
            node_type: node_type.to_string(),
            unique_id,
            attributes: attributes.unwrap_or_else(HashMap::new),
            calculations: None,  // Initialize as None
            title: node_title.map(|t| t.to_string()),
        }
    }

    pub fn new_data_type(
        data_type: &str, 
        name: &str, 
        attributes: HashMap<String, String>,
        calculations: Option<HashMap<String, String>>  // Already optional
    ) -> Self {
        Node::DataTypeNode {
            data_type: data_type.to_string(),
            name: name.to_string(),
            attributes,
            calculations,
            metadata: HashMap::new(),
        }
    }

    pub fn to_node_data(&self, graph_index: usize, _py: Python, filter_attributes: Option<&[String]>) -> PyResult<NodeData> {
        match self {
            Node::StandardNode { node_type, unique_id, attributes, calculations, title } => {
                let filtered_attrs = match filter_attributes {
                    Some(filter) => {
                        let mut filtered = HashMap::new();
                        for attr_name in filter {
                            if let Some(value) = attributes.get(attr_name) {
                                filtered.insert(attr_name.clone(), value.clone());
                            }
                        }
                        filtered
                    },
                    None => attributes.clone(),
                };

                Ok(NodeData {
                    unique_id: *unique_id,
                    title: title.clone().unwrap_or_default(),
                    graph_index,
                    attributes: filtered_attrs,
                    calculations: calculations.clone(),
                    node_type: node_type.clone(),
                    traversals: None,
                })
            },
            Node::DataTypeNode { .. } => {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Cannot convert DataTypeNode to NodeData"))
            }
        }
    }

    pub fn node_data_to_py(node_data: &NodeData, py: Python) -> PyResult<PyObject> {
        let mut dict = HashMap::new();
        
        dict.insert("node_type".to_string(), node_data.node_type.clone().into_py(py));
        dict.insert("unique_id".to_string(), node_data.unique_id.into_py(py));
        dict.insert("graph_index".to_string(), node_data.graph_index.into_py(py));
        dict.insert("title".to_string(), node_data.title.clone().into_py(py));
        
        // Convert attributes
        let py_attrs: HashMap<String, PyObject> = node_data.attributes.iter()
            .map(|(k, v)| Ok((k.clone(), v.to_python_object(py, None)?)))
            .collect::<PyResult<_>>()?;
        dict.insert("attributes".to_string(), py_attrs.into_py(py));
        
        // Convert calculations if they exist
        if let Some(ref calculations) = node_data.calculations {
            let py_calcs: HashMap<String, PyObject> = calculations.iter()
                .map(|(k, v)| Ok((k.clone(), v.to_python_object(py, None)?)))
                .collect::<PyResult<_>>()?;
            dict.insert("calculations".to_string(), py_calcs.into_py(py));
        }
        
        // Only include traversals if Some and non-empty
        if let Some(ref traversals) = node_data.traversals {
            if !traversals.is_empty() {
                let traversals: Vec<PyObject> = traversals.iter()
                    .map(|node| Self::node_data_to_py(node, py))
                    .collect::<PyResult<_>>()?;
                dict.insert("traversals".to_string(), traversals.into_py(py));
            }
        }
        
        Ok(dict.into_py(py))
    }
}

impl Relation {
    pub fn new(name: &str, attributes: Option<HashMap<String, AttributeValue>>) -> Self {
        Relation {
            relation_type: name.to_string(),
            attributes,
        }
    }
}