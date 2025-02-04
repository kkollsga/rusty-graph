use pyo3::prelude::*;
use pyo3::types::PyAny;
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue;

fn parse_value_to_i32(item: &PyAny) -> Option<i32> {
    if let Ok(float_val) = item.extract::<f64>() {
        return Some(float_val as i32);
    }
    if let Ok(int_val) = item.extract::<i32>() {
        return Some(int_val);
    }
    if let Ok(s) = item.extract::<String>() {
        if let Ok(num) = s.parse::<i32>() {
            return Some(num);
        }
        if let Ok(num) = s.parse::<f64>() {
            return Some(num as i32);
        }
    }
    None
}

fn extract_attribute_value(item: &PyAny) -> Option<AttributeValue> {
    if let Ok(int_val) = item.extract::<i64>() {
        Some(AttributeValue::Int(int_val as i32))
    } else if let Ok(float_val) = item.extract::<f64>() {
        Some(AttributeValue::Float(float_val))
    } else if let Ok(str_val) = item.extract::<String>() {
        Some(AttributeValue::String(str_val))
    } else {
        None
    }
}

pub fn add_relationships(
    graph: &mut DiGraph<Node, Relation>,
    data: &PyAny,
    columns: Vec<String>,
    relationship_type: String,
    source_type: String,
    source_id_field: String,
    target_type: String,
    target_id_field: String,
    source_title_field: Option<String>,
    target_title_field: Option<String>,
    attribute_columns: Option<Vec<String>>,
    conflict_handling: Option<String>,
) -> PyResult<()> {  // Changed return type
    let conflict_handling = conflict_handling.unwrap_or_else(|| "skip".to_string());
    let data_input = crate::graph::KnowledgeGraph::process_input_data(data)?;
    let mut source_node_lookup: HashMap<i32, petgraph::graph::NodeIndex> = HashMap::new();
    let mut target_node_lookup: HashMap<i32, petgraph::graph::NodeIndex> = HashMap::new();
    let attribute_columns = attribute_columns.unwrap_or_default();

    // Validate attribute columns
    for col in &attribute_columns {
        if !columns.contains(col) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Attribute column '{}' not found in input data", col)
            ));
        }
    }

    // Build node lookups
    for index in graph.node_indices() {
        if let Some(node) = graph.node_weight(index) {
            match node {
                Node::StandardNode { node_type, unique_id, .. } => {
                    if node_type == &source_type {
                        source_node_lookup.insert(*unique_id, index);
                    } else if node_type == &target_type {
                        target_node_lookup.insert(*unique_id, index);
                    }
                },
                _ => {}
            }
        }
    }

    // Get GIL and convert Py<PyList> to PyList
    Python::with_gil(|py| {
        let data_list = data_input.data.as_ref(py);
        'row_loop: for row in data_list.iter() {
            let row: Vec<&PyAny> = match row.extract() {
                Ok(r) => r,
                Err(_) => {
                    println!("Skipping malformed relationship row");
                    continue 'row_loop;
                }
            };

            let mut source_unique_id: Option<i32> = None;
            let mut target_unique_id: Option<i32> = None;
            let mut source_title: Option<String> = None;
            let mut target_title: Option<String> = None;
            let mut attributes: HashMap<String, AttributeValue> = HashMap::new();

            for (col_index, column_name) in columns.iter().enumerate() {
                let item = match row.get(col_index) {
                    Some(i) => i,
                    None => {
                        println!("Skipping relationship row with missing columns");
                        continue 'row_loop;
                    }
                };

                if column_name == &source_id_field {
                    source_unique_id = parse_value_to_i32(item);
                    if source_unique_id.is_none() {
                        println!("Skipping row due to invalid source_id");
                        continue 'row_loop;
                    }
                } else if column_name == &target_id_field {
                    target_unique_id = parse_value_to_i32(item);
                    if target_unique_id.is_none() {
                        println!("Skipping row due to invalid target_id");
                        continue 'row_loop;
                    }
                } else if let Some(ref title_field) = source_title_field {
                    if column_name == title_field {
                        source_title = item.extract().ok();
                    }
                } else if let Some(ref title_field) = target_title_field {
                    if column_name == title_field {
                        target_title = item.extract().ok();
                    }
                }

                // Process attribute columns
                if attribute_columns.contains(column_name) {
                    if let Some(value) = extract_attribute_value(item) {
                        attributes.insert(column_name.clone(), value);
                    }
                }
            }

            let source_unique_id = source_unique_id.unwrap();
            let target_unique_id = target_unique_id.unwrap();

            let source_node_index = find_or_create_node(
                graph,
                &source_type,
                source_unique_id,
                source_title,
                &mut source_node_lookup
            );
            let target_node_index = find_or_create_node(
                graph,
                &target_type,
                target_unique_id,
                target_title,
                &mut target_node_lookup
            );

            match conflict_handling.as_str() {
                "skip" => {
                    if !graph.contains_edge(source_node_index, target_node_index) {
                        let relation = Relation::new(&relationship_type, Some(attributes));
                        let _edge = graph.add_edge(source_node_index, target_node_index, relation);
                    }
                },
                "replace" => {
                    if let Some(edge) = graph.find_edge(source_node_index, target_node_index) {
                        graph.remove_edge(edge);
                    }
                    let relation = Relation::new(&relationship_type, Some(attributes));
                    let _edge = graph.add_edge(source_node_index, target_node_index, relation);
                },
                "update" => {
                    if let Some(edge) = graph.find_edge(source_node_index, target_node_index) {
                        if let Some(existing_relation) = graph.edge_weight_mut(edge) {
                            if let Some(ref mut existing_attrs) = existing_relation.attributes {
                                existing_attrs.extend(attributes);
                            } else {
                                existing_relation.attributes = Some(attributes);
                            }
                        }
                    } else {
                        let relation = Relation::new(&relationship_type, Some(attributes));
                        let _edge = graph.add_edge(source_node_index, target_node_index, relation);
                    }
                },
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid conflict_handling value. Must be 'skip', 'replace', or 'update'"
                )),
            }
        }
        Ok(())
    })
}

fn find_or_create_node(
    graph: &mut DiGraph<Node, Relation>,
    node_type: &str,
    unique_id: i32,
    title: Option<String>,
    node_lookup: &mut HashMap<i32, petgraph::graph::NodeIndex>,
) -> petgraph::graph::NodeIndex {
    if let Some(&index) = node_lookup.get(&unique_id) {
        index
    } else {
        let new_node = Node::new(node_type, unique_id, None, title.as_deref());
        let index = graph.add_node(new_node);
        node_lookup.insert(unique_id, index);
        index
    }
}