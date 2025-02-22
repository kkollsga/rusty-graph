// src/graph/custom_equation.rs

use super::node_calculations::StatResult;
use super::statistics_methods::get_parent_child_pairs;
use super::equation_parser::{Parser, Evaluator};
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, CurrentSelection, NodeData};
use std::collections::HashMap;

pub fn evaluate_equation(
    graph: &DirGraph,
    selection: &CurrentSelection,
    expression: &str,
    level_index: Option<usize>,
) -> Vec<StatResult> {
    // Parse the expression once at the start
    let parsed_expr = match Parser::parse_expression(expression) {
        Ok(expr) => expr,
        Err(err) => {
            return vec![StatResult {
                parent_idx: None,
                parent_title: None,
                value: None,
                error: Some(format!("Failed to parse expression: {}", err)),
            }];
        }
    };

    let pairs = get_parent_child_pairs(selection, level_index);
    
    pairs.iter()
        .map(|pair| {
            let mut result = StatResult {
                parent_idx: pair.parent,
                parent_title: pair.parent
                    .and_then(|idx| graph.get_node(idx))
                    .and_then(|n| n.get_field("title"))
                    .and_then(|v| v.as_string()),
                value: None,
                error: None,
            };

            // Convert node data to objects for evaluation
            let objects: Vec<HashMap<String, f64>> = pair.children.iter()
                .filter_map(|&node_idx| {
                    graph.get_node(node_idx)
                        .and_then(convert_node_to_object)
                })
                .collect();

            if objects.is_empty() {
                result.error = Some("No valid nodes found for evaluation".to_string());
                return result;
            }

            // Evaluate expression using parsed AST
            match Evaluator::evaluate(&parsed_expr, &objects) {
                Ok(value) => result.value = Some(value),
                Err(err) => result.error = Some(err),
            }

            result
        })
        .collect()
}

fn convert_node_to_object(node: &NodeData) -> Option<HashMap<String, f64>> {
    let mut object = HashMap::new();
    
    match node {
        NodeData::Regular { properties, .. } | NodeData::Schema { properties, .. } => {
            for (key, value) in properties {
                if let Some(num_value) = match value {
                    Value::Int64(i) => Some(*i as f64),
                    Value::Float64(f) => Some(*f),
                    Value::UniqueId(u) => Some(*u as f64),
                    Value::String(s) => s.parse::<f64>().ok(),
                    _ => None,
                } {
                    object.insert(key.clone(), num_value);
                }
            }
        }
    }
    
    if object.is_empty() {
        None
    } else {
        Some(object)
    }
}