use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use crate::graph::get_schema::update_or_retrieve_schema;
use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue; 

// Function to handle node updating or creation based on conflict handling strategy
fn update_or_create_node(
    graph: &mut DiGraph<Node, Relation>,
    node_type: &String,
    unique_id: String,
    node_title: Option<String>,
    attributes: Option<HashMap<String, AttributeValue>>, // Now an Option
    conflict_handling: &String,
) -> usize {
    let existing_node_index = graph.node_indices().find(|&i| match &graph[i] {
        Node::StandardNode {
            node_type: nt,
            unique_id: uid,
            ..
        } => nt == node_type && uid == &unique_id,
        _ => false,
    });

    match existing_node_index {
        Some(node_index) => {
            match conflict_handling.as_str() {
                "replace" => {
                    // If replacing, create a new node with the provided attributes (which may be None)
                    graph[node_index] = Node::new(&node_type, &unique_id, attributes, node_title.as_deref());
                },
                "update" => {
                    if let Some(attrs) = attributes {
                        if let Node::StandardNode {
                            attributes: node_attrs,
                            ..
                        } = &mut graph[node_index]
                        {
                            for (key, value) in attrs {
                                node_attrs.insert(key, value);
                            }
                        }
                    }
                },
                "skip" => (),
                _ => panic!("Invalid conflict_handling value"),
            }
            node_index.index()
        },
        None => {
            // Create a new node with the provided attributes, which may be None
            let node = Node::new(&node_type, &unique_id, attributes, node_title.as_deref());
            graph.add_node(node).index()
        },
    }
}

// The simplified main function
pub fn add_nodes(
    graph: &mut DiGraph<Node, Relation>,
    data: &PyList, // Each item in this list is a sublist representing a single node's attributes
    columns: Vec<String>,
    node_type: String,
    unique_id_field: String,
    node_title_field: Option<String>,
    conflict_handling: Option<String>,
    column_types: Option<&PyDict>,
) -> PyResult<Vec<usize>> {
    let conflict_handling = conflict_handling.unwrap_or_else(|| "update".to_string());
    let mut indices = Vec::new();

    // Convert PyDict to HashMap for easier handling
    let column_types_map: HashMap<String, String> = column_types
        .map_or(Ok(HashMap::new()), |ct| ct.extract())?;

    // Update or retrieve the DataTypeNode schema once before processing the rows
    let schema = update_or_retrieve_schema(
        graph,
        "Node",
        &node_type,
        Some(columns.clone()),
        Some(column_types_map.clone())
    )?;

    for row in data.iter() {
        let row: Vec<&PyAny> = row.extract()?; // Extract the row as a list of PyAny references
        let mut attributes: HashMap<String, AttributeValue> = HashMap::new();
        let mut unique_id = String::new();
        let mut node_title: Option<String> = None;

        for (col_index, column_name) in columns.iter().enumerate() {
            let item = row.get(col_index).unwrap(); // Safe to use unwrap() due to the structure of the data

            if column_name == &unique_id_field {
                unique_id = item.extract()?;
                continue;
            }

            if node_title_field.as_deref() == Some(column_name.as_str()) {
                node_title = Some(item.extract()?);
                continue;
            }

            // Determine the attribute's data type from the schema and extract value accordingly
            let data_type = schema.get(column_name).map_or("String", String::as_str);
            let attribute_value = match data_type {
                "Int" => match item.extract::<i32>() {
                    Ok(value) => Ok(AttributeValue::Int(value)),
                    Err(_) => {
                        // Attempt to parse from String if direct extraction fails
                        item.extract::<String>()
                            .and_then(|s| s.parse::<i32>().map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("Failed to parse Int from String")))
                            .map(AttributeValue::Int)
                    }
                },
                "Float" => match item.extract::<f64>() {
                    Ok(value) => Ok(AttributeValue::Float(value)),
                    Err(_) => {
                        // Attempt to parse from String if direct extraction fails
                        item.extract::<String>()
                            .and_then(|s| s.parse::<f64>().map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("Failed to parse Float from String")))
                            .map(AttributeValue::Float)
                    }
                },
                "String" => item.extract::<String>().map(AttributeValue::String),
                // Extend cases for other data types like 'DateTime', 'Date', etc.
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Unsupported data type")),
            }?;

            attributes.insert(column_name.clone(), attribute_value);
        }

        // Create or update the node in the graph based on the conflict handling strategy
        let index = update_or_create_node(
            graph,
            &node_type,
            unique_id,
            node_title,
            Some(attributes),
            &conflict_handling,
        );

        indices.push(index);
    }

    Ok(indices)
}