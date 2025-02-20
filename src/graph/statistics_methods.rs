// src/graph/statistics_methods.rs
use petgraph::graph::NodeIndex;
use crate::graph::schema::{DirGraph, CurrentSelection, NodeData};
use crate::datatypes::values::Value;
use std::collections::HashSet;

#[derive(Debug)]
pub struct ParentChildPair {
    pub parent: Option<NodeIndex>,
    pub children: Vec<NodeIndex>,
}

pub fn get_parent_child_pairs(
    selection: &CurrentSelection,
    level_index: Option<usize>,
) -> Vec<ParentChildPair> {
    // If no level specified, use the deepest level
    let target_level = level_index.unwrap_or_else(|| 
        selection.get_level_count().saturating_sub(1)
    );

    // Return empty vec if level doesn't exist
    if target_level >= selection.get_level_count() {
        return Vec::new();
    }

    let level = selection.get_level(target_level)
        .expect("Level index was already checked");

    // If the level has no selections, return empty vec
    if level.is_empty() {
        return Vec::new();
    }

    // If we have parent-child pairs, return them
    if level.iter_groups().any(|(parent, _)| parent.is_some()) {
        level.iter_groups()
            .map(|(parent, children)| ParentChildPair {
                parent: *parent,
                children: children.clone(),
            })
            .collect()
    } else {
        // For root level or standalone selections, create a single pair with no parent
        vec![ParentChildPair {
            parent: None,
            children: level.get_all_nodes(),
        }]
    }
}

#[derive(Debug)]
pub struct PropertyStats {
    pub parent_idx: Option<NodeIndex>,
    pub parent_type: Option<String>,
    pub parent_title: Option<Value>,
    pub parent_id: Option<Value>,
    pub property_name: String,
    pub value_type: String,
    pub count: usize,
    pub children: usize,
    pub sum: Option<f64>,
    pub avg: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub valid_count: usize,
    pub is_numeric: bool,
}

impl PropertyStats {
    fn new(parent_idx: Option<NodeIndex>, graph: &DirGraph, property: &str) -> Self {
        let (parent_type, parent_title, parent_id) = parent_idx
            .and_then(|idx| graph.get_node(idx))
            .map(|node| match node {
                NodeData::Regular { node_type, title, id, .. } |
                NodeData::Schema { node_type, title, id, .. } => {
                    (Some(node_type.clone()), Some(title.clone()), Some(id.clone()))
                }
            })
            .unwrap_or((None, None, None));

        PropertyStats {
            parent_idx,
            parent_type,
            parent_title,
            parent_id,
            property_name: property.to_string(),
            value_type: "unknown".to_string(),
            count: 0,
            children: 0,
            sum: None,
            avg: None,
            min: None,
            max: None,
            valid_count: 0,
            is_numeric: false,
        }
    }

    fn finalize(&mut self) {
        if self.is_numeric {
            if let Some(sum) = self.sum {
                if self.valid_count > 0 {
                    self.avg = Some(sum / self.valid_count as f64);
                }
            }
        }
    }
}

pub fn calculate_property_stats(
    graph: &DirGraph,
    pairs: &[ParentChildPair],
    property: &str,
) -> Vec<PropertyStats> {
    pairs.iter()
        .map(|pair| {
            let mut stats = PropertyStats::new(pair.parent, graph, property);
            calculate_stats_for_nodes(graph, &pair.children, property, &mut stats);
            stats.finalize();
            stats
        })
        .collect()
}

fn calculate_stats_for_nodes(
    graph: &DirGraph,
    nodes: &[NodeIndex],
    property: &str,
    stats: &mut PropertyStats,
) {
    stats.count = nodes.len();
    stats.children = nodes.len();

    let mut found_numeric = false;
    let mut sum = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut valid_numeric_count = 0;
    let mut seen_types = HashSet::new();

    for &node_idx in nodes {
        if let Some(node) = graph.get_node(node_idx) {
            if let Some(value) = get_node_property(node, property) {
                match value {
                    Value::Null => continue,
                    Value::String(s) if s.is_empty() => continue,
                    _ => {
                        stats.valid_count += 1;
                        seen_types.insert(match value {
                            Value::String(_) => "string",
                            Value::Int64(_) => "int64",
                            Value::Float64(_) => "float64",
                            Value::Boolean(_) => "boolean",
                            Value::DateTime(_) => "datetime",
                            Value::UniqueId(_) => "unique_id",
                            Value::Null => "null",
                        });
                    }
                }

                if let Some(num) = try_convert_to_float(value) {
                    found_numeric = true;
                    sum += num;
                    min = min.min(num);
                    max = max.max(num);
                    valid_numeric_count += 1;
                }
            }
        }
    }

    // Set value type based on seen values
    stats.value_type = if seen_types.is_empty() {
        "null".to_string()
    } else if seen_types.len() == 1 {
        seen_types.into_iter().next().unwrap().to_string()
    } else {
        "mixed".to_string()
    };

    if found_numeric && valid_numeric_count > 0 {
        stats.is_numeric = true;
        stats.sum = Some(sum);
        stats.min = Some(min);
        stats.max = Some(max);
    } else {
        stats.is_numeric = false;
        stats.sum = None;
        stats.min = None;
        stats.max = None;
        stats.avg = None;
    }
}

fn try_convert_to_float(value: &Value) -> Option<f64> {
    match value {
        Value::Int64(i) => Some(*i as f64),
        Value::Float64(f) => Some(*f),
        Value::String(s) => s.parse::<f64>().ok(),
        Value::UniqueId(u) => Some(*u as f64),
        _ => None,
    }
}

fn get_node_property<'a>(node: &'a NodeData, property: &str) -> Option<&'a Value> {
    match node {
        NodeData::Regular { properties, .. } | NodeData::Schema { properties, .. } => {
            properties.get(property)
        }
    }
}