use std::fmt;
use crate::graph::schema::{DirGraph, CurrentSelection, NodeData};
use crate::datatypes::values::Value;
use crate::graph::statistics_methods::get_parent_child_pairs;
use petgraph::graph::NodeIndex;

#[derive(Debug, Clone, PartialEq)]
pub enum StatMethod {
    Sum,
    Average,
    Min,
    Max,
    Count,
}

impl fmt::Display for StatMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatMethod::Sum => write!(f, "sum"),
            StatMethod::Average => write!(f, "average"),
            StatMethod::Min => write!(f, "min"),
            StatMethod::Max => write!(f, "max"),
            StatMethod::Count => write!(f, "count"),
        }
    }
}

#[derive(Debug)]
pub struct StatResult {
    pub parent_idx: Option<NodeIndex>,
    pub value: Option<f64>,
    pub error: Option<String>,
}

pub fn calculate_node_statistic(
    graph: &DirGraph,
    selection: &CurrentSelection,
    property: &str,
    method: StatMethod,
    level_index: Option<usize>,
) -> Vec<StatResult> {
    let pairs = get_parent_child_pairs(selection, level_index);
    
    pairs.iter()
        .map(|pair| {
            let mut values = Vec::new();
            
            // Collect valid numeric values from children
            for &node_idx in &pair.children {
                if let Some(node) = graph.get_node(node_idx) {
                    if let Some(value) = try_get_numeric_value(node, property) {
                        values.push(value);
                    }
                }
            }
            
            // Calculate the requested statistic
            let result = match method {
                StatMethod::Sum => {
                    if values.is_empty() {
                        None
                    } else {
                        Some(values.iter().copied().sum())
                    }
                },
                StatMethod::Average => {
                    if values.is_empty() {
                        None
                    } else {
                        Some(values.iter().copied().sum::<f64>() / values.len() as f64)
                    }
                },
                StatMethod::Min => values.iter().copied().fold(
                    None,
                    |min: Option<f64>, x: f64| Some(min.map_or(x, |y| y.min(x)))
                ),
                StatMethod::Max => values.iter().copied().fold(
                    None,
                    |max: Option<f64>, x: f64| Some(max.map_or(x, |y| y.max(x)))
                ),
                StatMethod::Count => Some(values.len() as f64),
            };

            StatResult {
                parent_idx: pair.parent,
                value: result,
                error: if values.is_empty() && method != StatMethod::Count {
                    Some(format!("No valid numeric values found for property '{}'", property))
                } else {
                    None
                },
            }
        })
        .collect()
}

fn try_get_numeric_value(node: &NodeData, property: &str) -> Option<f64> {
    match node {
        NodeData::Regular { properties, .. } | NodeData::Schema { properties, .. } => {
            properties.get(property).and_then(|value| match value {
                Value::Int64(i) => Some(*i as f64),
                Value::Float64(f) => Some(*f),
                Value::UniqueId(u) => Some(*u as f64),
                Value::String(s) => s.parse::<f64>().ok(),
                _ => None,
            })
        }
    }
}