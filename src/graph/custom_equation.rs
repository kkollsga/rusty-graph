use rhai::{Engine, Scope, Dynamic};
use crate::graph::schema::{DirGraph, CurrentSelection, NodeData};
use crate::datatypes::values::Value;
use std::collections::HashMap;
use petgraph::graph::NodeIndex;

/// Result structure for equation evaluations, following similar pattern to StatResult
pub struct EquationResult {
    pub parent_title: Option<String>,
    pub parent_idx: Option<NodeIndex>,
    pub value: Option<f64>,
    pub error: Option<String>,
}

/// Evaluates a custom equation for each group in the current selection
/// 
/// # Arguments
/// * `graph` - The directed graph containing all nodes and their properties
/// * `selection` - Current selection state containing levels and groups
/// * `expression` - The equation to evaluate (using Rhai syntax)
/// * `level_index` - Optional specific level to evaluate, defaults to last level
/// 
/// # Returns
/// Vector of EquationResult containing evaluation results for each group
pub fn evaluate_equation(
    graph: &DirGraph,
    selection: &CurrentSelection,
    expression: &str,
    level_index: Option<usize>,
) -> Vec<EquationResult> {
    let level_idx = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    let mut results = Vec::new();

    if let Some(level) = selection.get_level(level_idx) {
        let mut engine = Engine::new();
        setup_engine(&mut engine);

        for (parent, children) in level.iter_groups() {
            let mut scope = Scope::new();
            let property_arrays = collect_property_arrays(graph, children);
            
            // Register arrays in scope
            for (key, values) in property_arrays {
                scope.push(key, Dynamic::from(values));
            }

            let parent_title = match parent {
                Some(p) => {
                    if let Some(node) = graph.get_node(*p) {
                        node.get_field("title")
                            .and_then(|v| match v {
                                Value::String(s) => Some(s),
                                _ => None
                            })
                    } else {
                        None
                    }
                },
                None => Some("Root".to_string()),
            };

            // Evaluate expression
            let result = match engine.eval_with_scope::<f64>(&mut scope, expression) {
                Ok(value) => EquationResult {
                    parent_idx: parent.map(|p| p),
                    parent_title,
                    value: Some(value),
                    error: None,
                },
                Err(e) => EquationResult {
                    parent_idx: parent.map(|p| p),
                    parent_title,
                    value: None,
                    error: Some(e.to_string()),
                },
            };

            results.push(result);
        }
    }

    results
}

/// Sets up the Rhai engine with common functions needed for calculations
fn setup_engine(engine: &mut Engine) {
    // Basic aggregation functions
    engine.register_fn("sum", |arr: Vec<f64>| arr.iter().sum::<f64>());
    engine.register_fn("avg", |arr: Vec<f64>| {
        if arr.is_empty() { 0.0 } else { arr.iter().sum::<f64>() / arr.len() as f64 }
    });
    engine.register_fn("max", |arr: Vec<f64>| {
        arr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    });
    engine.register_fn("min", |arr: Vec<f64>| {
        arr.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    });
    engine.register_fn("count", |arr: Vec<f64>| arr.len() as f64);

    // Statistical functions
    engine.register_fn("variance", |arr: Vec<f64>| {
        if arr.len() < 2 { return 0.0; }
        let n = arr.len() as f64;
        let mean = arr.iter().sum::<f64>() / n;
        arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    });
    
    engine.register_fn("std", |arr: Vec<f64>| {
        if arr.len() < 2 { return 0.0; }
        let n = arr.len() as f64;
        let mean = arr.iter().sum::<f64>() / n;
        (arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    });

    // Helper functions
    engine.register_fn("abs", |x: f64| x.abs());
    engine.register_fn("sqrt", |x: f64| x.sqrt());
    engine.register_fn("pow", |x: f64, y: f64| x.powf(y));
    engine.register_fn("round", |x: f64| x.round());
}

/// Collects property values from nodes into arrays for calculations
/// 
/// # Arguments
/// * `graph` - The graph containing the nodes
/// * `nodes` - List of node indices to collect properties from
/// 
/// # Returns
/// HashMap mapping property names to vectors of their values
fn collect_property_arrays(graph: &DirGraph, nodes: &[NodeIndex]) -> HashMap<String, Vec<f64>> {
    let mut property_arrays: HashMap<String, Vec<f64>> = HashMap::new();
    
    for &node_idx in nodes {
        if let Some(node) = graph.get_node(node_idx) {
            if let NodeData::Regular { properties, .. } = node {
                for (key, value) in properties {
                    let entry = property_arrays.entry(key.clone()).or_default();
                    match value {
                        Value::Float64(f) => entry.push(*f),  // Added dereferencing
                        Value::Int64(i) => entry.push(*i as f64),
                        _ => continue,
                    }
                }
            }
        }
    }

    property_arrays
}