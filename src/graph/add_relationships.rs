use pyo3::prelude::*;
use pyo3::types::PyList;
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use crate::schema::{Node, Relation};

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

pub fn add_relationships(
    graph: &mut DiGraph<Node, Relation>,
    data: &PyList,
    columns: Vec<String>,
    relationship_type: String,
    source_type: String,
    source_id_field: i32,
    target_type: String,
    target_id_field: i32,
    source_title_field: Option<String>,
    target_title_field: Option<String>,
) -> PyResult<Vec<(usize, usize)>> {
    let mut indices = Vec::new();
    let mut source_node_lookup: HashMap<i32, petgraph::graph::NodeIndex> = HashMap::new();
    let mut target_node_lookup: HashMap<i32, petgraph::graph::NodeIndex> = HashMap::new();

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

    'row_loop: for row in data.iter() {
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
    
        for (col_index, column_name) in columns.iter().enumerate() {
            let item = match row.get(col_index) {
                Some(i) => i,
                None => {
                    println!("Skipping relationship row with missing columns");
                    continue 'row_loop;
                }
            };
    
            if let Ok(col_num) = column_name.parse::<i32>() {
                if col_num == source_id_field {
                    source_unique_id = parse_value_to_i32(item);
                    if source_unique_id.is_none() {
                        println!("Skipping row due to invalid source_id");
                        continue 'row_loop;
                    }
                    continue;
                }
                if col_num == target_id_field {
                    target_unique_id = parse_value_to_i32(item);
                    if target_unique_id.is_none() {
                        println!("Skipping row due to invalid target_id");
                        continue 'row_loop;
                    }
                    continue;
                }
            }
    
            if let Some(ref title_field) = source_title_field {
                if column_name == title_field {
                    source_title = match item.extract() {
                        Ok(title) => Some(title),
                        Err(_) => {
                            println!("Invalid source title, setting to None");
                            None
                        }
                    };
                    continue;
                }
            }
            if let Some(ref title_field) = target_title_field {
                if column_name == title_field {
                    target_title = match item.extract() {
                        Ok(title) => Some(title),
                        Err(_) => {
                            println!("Invalid target title, setting to None");
                            None
                        }
                    };
                    continue;
                }
            }
        }
    
        let source_unique_id = source_unique_id.unwrap(); // Safe due to continue above
        let target_unique_id = target_unique_id.unwrap(); // Safe due to continue above
    
        if source_node_lookup.get(&source_unique_id).is_none() {
            println!("Source node {} not found, skipping relationship", source_unique_id);
            continue 'row_loop;
        }
        if target_node_lookup.get(&target_unique_id).is_none() {
            println!("Target node {} not found, skipping relationship", target_unique_id);
            continue 'row_loop;
        }
    
        let source_node_index = find_or_create_node(graph, &source_type, source_unique_id, source_title, &mut source_node_lookup);
        let target_node_index = find_or_create_node(graph, &target_type, target_unique_id, target_title, &mut target_node_lookup);
    
        let relation = Relation::new(&relationship_type, None);
        let _edge = graph.add_edge(source_node_index, target_node_index, relation);
    
        indices.push((source_node_index.index(), target_node_index.index()));
    }

    Ok(indices)
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