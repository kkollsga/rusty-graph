// src/graph/calculations.rs
use super::statistics_methods::get_parent_child_pairs;
use super::equation_parser::{Parser, Evaluator, Expr, AggregateType};
use super::maintain_graph;
use super::lookups::TypeLookup;
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, CurrentSelection, NodeData};
use crate::graph::reporting::CalculationOperationReport; // Remove unused OperationReport import
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::time::Instant; // For timing operations

pub enum EvaluationResult {
    Stored(CalculationOperationReport),
    Computed(Vec<StatResult>)
}

#[derive(Debug)]
pub struct StatResult {
    pub node_idx: Option<NodeIndex>,
    pub parent_idx: Option<NodeIndex>,
    pub parent_title: Option<String>,
    pub value: Value,
    pub error_msg: Option<String>,
}

pub fn process_equation(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    expression: &str,
    level_index: Option<usize>,
    store_as: Option<&str>,
) -> Result<EvaluationResult, String> {
    // Start tracking time for reporting
    let start_time = Instant::now();
    
    // Track errors
    let mut errors = Vec::new();
    
    // Check for unknown aggregate function names
    if let Some(unknown_func) = extract_unknown_aggregate_function(expression) {
        let supported = AggregateType::get_supported_names().join(", ");
        let error_msg = format!(
            "Unknown aggregate function '{}'. Supported functions are: {}",
            unknown_func, supported
        );
        errors.push(error_msg.clone());
        return Err(error_msg);
    }
    
    // Parse the expression first
    let parsed_expr = match Parser::parse_expression(expression) {
        Ok(expr) => expr,
        Err(err) => {
            // Try to provide more context for why the parsing failed
            let error_msg = if expression.is_empty() {
                "Expression cannot be empty.".to_string()
            } else if expression.contains("(") && !expression.contains(")") {
                "Missing closing parenthesis in expression.".to_string()
            } else if !expression.contains("(") && expression.contains(")") {
                "Unexpected closing parenthesis in expression.".to_string()
            } else if !expression.contains("(") && is_likely_aggregate_name(expression) {
                format!(
                    "Function '{}' requires parentheses. Try '{}(property)' instead.", 
                    expression, expression
                )
            } else {
                format!("Failed to parse expression: {}. Check for syntax errors or case sensitivity in function names (use 'sum', not 'SUM').", err)
            };
            
            errors.push(error_msg.clone());
            return Err(error_msg);
        },
    };
    
    // Extract variables from the expression
    let variables = parsed_expr.extract_variables();
    
    // Check if selection is valid or empty
    if selection.get_level_count() == 0 {
        let error_msg = "No nodes selected. Please apply filters or traversals before calculating.".to_string();
        errors.push(error_msg.clone());
        return Err(error_msg);
    }
    
    // Additional check to see if the current level has any nodes
    let effective_level_index = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    let nodes_processed;
    if let Some(level) = selection.get_level(effective_level_index) {
        if level.get_all_nodes().is_empty() {
            let error_msg = format!(
                "No nodes found at level {}. Make sure your filters and traversals return data.", 
                effective_level_index
            );
            errors.push(error_msg.clone());
            return Err(error_msg);
        }
        
        // Define nodes_processed at first usage
        nodes_processed = level.get_all_nodes().len();
    } else {
        let error_msg = format!("Invalid level index: {}. Selection only has {} levels.", 
            effective_level_index, selection.get_level_count());
        errors.push(error_msg.clone());
        return Err(error_msg);
    }
    // If we have a selection, validate variables against schema
    if let Some(level) = selection.get_level(effective_level_index) {
        if !level.is_empty() {
            // Get a sample node to determine node type
            if let Some(nodes) = level.get_all_nodes().first() {
                if let Some(node) = graph.get_node(*nodes) {
                    match node {
                        NodeData::Regular { node_type, .. } => {
                            // Check if schema node exists for this type
                            let schema_lookup = match TypeLookup::new(&graph.graph, "SchemaNode".to_string()) {
                                Ok(lookup) => lookup,
                                Err(_) => {
                                    let error_msg = "Could not access schema information".to_string();
                                    errors.push(error_msg.clone());
                                    return Err(error_msg);
                                },
                            };
                            
                            let schema_title = Value::String(node_type.clone());
                            
                            if let Some(schema_idx) = schema_lookup.check_title(&schema_title) {
                                if let Some(NodeData::Schema { properties, .. }) = graph.get_node(schema_idx) {
                                    // Validate each variable against schema properties
                                    // Don't check reserved field names like 'id', 'title', 'type'
                                    for var in &variables {
                                        if var != "id" && var != "title" && var != "type" && !properties.contains_key(var) {
                                            let available = properties.keys().cloned().collect::<Vec<String>>().join(", ");
                                            let error_msg = format!(
                                                "Property '{}' does not exist on '{}' nodes. Available properties: {}", 
                                                var, node_type, available
                                            );
                                            errors.push(error_msg.clone());
                                            return Err(error_msg);
                                        }
                                    }
                                }
                            }
                        },
                        _ => {}, // Skip schema nodes
                    }
                }
            }
        }
    }
    
    let is_aggregation = has_aggregation(&parsed_expr);
    
    // When performing evaluation, we can use an immutable reference to graph
    let results = evaluate_equation(graph, selection, &parsed_expr, level_index);
    
    // Count nodes with errors for reporting
    let nodes_with_errors = results.iter().filter(|r| r.error_msg.is_some()).count();
    
    // Collect evaluation errors
    for result in &results {
        if let Some(error_msg) = &result.error_msg {
            let node_info = if let Some(title) = &result.parent_title {
                format!("Node '{}': ", title)
            } else {
                "".to_string()
            };
            errors.push(format!("{}Evaluation error: {}", node_info, error_msg));
        }
    }
    
    // If we don't need to store results, just return them directly
    if store_as.is_none() {
        if results.is_empty() {
            let error_msg = "No results from calculation. Check that your selection contains data.".to_string();
            errors.push(error_msg.clone());
            return Err(error_msg);
        }
        
        return Ok(EvaluationResult::Computed(results));
    }
    
    // Only proceed with node updating logic if we need to store results
    let target_property = store_as.unwrap();
    
    // Determine where to store results based on whether there's aggregation
    let effective_level_index = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    
    // Prepare a Vec to hold valid nodes for update
    let mut nodes_to_update: Vec<(Option<NodeIndex>, Value)> = Vec::new();
    
    if is_aggregation {
        // For aggregation - get actual parent nodes from the selection
        for result in &results {
            if let Some(parent_idx) = result.parent_idx {
                // Verify the parent node exists in the graph
                if graph.get_node(parent_idx).is_some() {
                    nodes_to_update.push((Some(parent_idx), result.value.clone()));
                }
            }
        }
    } else {
        // For non-aggregation - get actual child nodes from the selection
        if let Some(level) = selection.get_level(effective_level_index) {
            // Create HashMap from node indices to results
            let result_map: HashMap<NodeIndex, &StatResult> = results.iter()
                .filter_map(|r| r.node_idx.map(|idx| (idx, r)))
                .collect();
            
            // Get all node indices directly from the current level
            for node_idx in level.get_all_nodes() {
                // Direct HashMap lookup instead of linear search
                if let Some(&result) = result_map.get(&node_idx) {
                    // Verify node exists in the graph - IMPORTANT: Must check here
                    if graph.get_node(node_idx).is_some() {
                        nodes_to_update.push((Some(node_idx), result.value.clone()));
                    }
                }
            }
        }
    }
    
    // Check if we found any valid nodes to update
    if nodes_to_update.is_empty() {
        return Err(format!(
            "No valid nodes found to store '{}'. Selection level: {}, Aggregation: {}", 
            target_property, effective_level_index, is_aggregation
        ));
    }
    
    // Update the node properties with verified node indices
    let update_result = maintain_graph::update_node_properties(graph, &nodes_to_update, target_property)?;
    
    // Update nodes_updated from the result (now used)
    let nodes_updated = update_result.nodes_updated;
    
    // Calculate elapsed time for report
    let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    
    // Create the report - using all counters to avoid unused assignment warnings
    let mut report = CalculationOperationReport::new(
        "process_equation".to_string(),
        expression.to_string(),
        nodes_processed,
        nodes_updated,
        nodes_with_errors,
        elapsed_ms,
        is_aggregation
    );
    
    // Add errors if we found any
    if !errors.is_empty() {
        report = report.with_errors(errors);
    }
    
    Ok(EvaluationResult::Stored(report))
}

// Helper function to extract potentially unknown aggregate function name from expression
fn extract_unknown_aggregate_function(expression: &str) -> Option<String> {
    // Simple heuristic: if expression contains word(property) pattern but word is not a known aggregate
    let lowercase_expr = expression.to_lowercase();
    
    // Check for common patterns like "func(arg)" where func is not recognized
    let parts: Vec<&str> = lowercase_expr.split('(').collect();
    if parts.len() > 1 {
        let func_part = parts[0].trim();
        
        // Skip known functions
        if !is_known_aggregate(func_part) {
            // Check that it looks like a function name (alphanumeric)
            if func_part.chars().all(|c| c.is_alphanumeric() || c == '_') {
                return Some(func_part.to_string());
            }
        }
    }
    
    None
}

// Check if a name is a supported aggregate function
fn is_known_aggregate(name: &str) -> bool {
    AggregateType::from_string(name).is_some()
}

// Check if a string looks like it might be intended as an aggregate function name
fn is_likely_aggregate_name(name: &str) -> bool {
    let name = name.trim().to_lowercase();
    
    // Common aggregate function names people might try to use
    let common_aggregates = [
        "sum", "avg", "average", "mean", "median", "min", "max", "count", 
        "std", "stdev", "stddev", "var", "variance"
    ];
    
    common_aggregates.contains(&name.as_str())
}

// Modified evaluate_equation to take a parsed expression directly
// Now takes an immutable reference to graph since it only needs to read
pub fn evaluate_equation(
    graph: &DirGraph,
    selection: &CurrentSelection,
    parsed_expr: &Expr,
    level_index: Option<usize>,
) -> Vec<StatResult> {
    let is_aggregation = has_aggregation(parsed_expr);

    if is_aggregation {
        let pairs = get_parent_child_pairs(selection, level_index);
        
        // IMPROVEMENT #2: Cache parent titles to avoid redundant lookups
        let parent_titles: HashMap<NodeIndex, Option<String>> = pairs.iter()
            .filter_map(|pair| pair.parent.map(|idx| (
                idx, 
                graph.get_node(idx)
                    .and_then(|node| node.get_field("title"))
                    .and_then(|v| v.as_string())
            )))
            .collect();
        
        pairs.iter()
            .map(|pair| {
                let child_nodes: Vec<(NodeIndex, NodeData, HashMap<String, Value>)> = pair.children.iter()
                    .filter_map(|&node_idx| {
                        graph.get_node(node_idx)
                            .map(|node| (node_idx, node.clone(), convert_node_to_object(node)))
                    })
                    .collect();

                if child_nodes.is_empty() {
                    return StatResult {
                        node_idx: None,
                        parent_idx: pair.parent,
                        // Use cached parent title instead of looking it up again
                        parent_title: pair.parent.and_then(|idx| parent_titles.get(&idx).cloned().flatten()),
                        value: Value::Null,
                        error_msg: Some("No valid nodes found".to_string()),
                    };
                }

                let objects: Vec<HashMap<String, Value>> = child_nodes.into_iter()
                    .map(|(_, _, obj)| obj)
                    .collect();

                match Evaluator::evaluate(parsed_expr, &objects) {
                    Ok(value) => StatResult {
                        node_idx: None,
                        parent_idx: pair.parent,
                        // Use cached parent title
                        parent_title: pair.parent.and_then(|idx| parent_titles.get(&idx).cloned().flatten()),
                        value,
                        error_msg: None,
                    },
                    Err(err) => StatResult {
                        node_idx: None,
                        parent_idx: pair.parent,
                        // Use cached parent title
                        parent_title: pair.parent.and_then(|idx| parent_titles.get(&idx).cloned().flatten()),
                        value: Value::Null,
                        error_msg: Some(err),
                    },
                }
            })
            .collect()
    } else {
        let effective_index = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
        let level = match selection.get_level(effective_index) {
            Some(l) => l,
            None => return vec![],
        };

        let nodes = level.get_all_nodes();

        nodes.iter()
            .map(|&node_idx| {
                match graph.get_node(node_idx) {
                    Some(node) => {
                        let title = node.get_field("title")
                            .and_then(|v| v.as_string());
                        let obj = convert_node_to_object(node);
                
                        match Evaluator::evaluate(parsed_expr, &[obj]) {
                            Ok(value) => StatResult {
                                node_idx: Some(node_idx),
                                parent_idx: None,
                                parent_title: title,
                                value,
                                error_msg: None,
                            },
                            Err(err) => {
                                StatResult {
                                    node_idx: Some(node_idx),
                                    parent_idx: None,
                                    parent_title: title,
                                    value: Value::Null,
                                    error_msg: Some(err),
                                }
                            }
                        }
                    },
                    None => StatResult {
                        node_idx: Some(node_idx),
                        parent_idx: None,
                        parent_title: None,
                        value: Value::Null,
                        error_msg: Some("Node not found".to_string()),
                    },
                }
            })
            .collect()
    }
}

fn has_aggregation(expr: &Expr) -> bool {
    match expr {
        Expr::Aggregate(_, _) => true,
        Expr::Add(left, right) => has_aggregation(left) || has_aggregation(right),
        Expr::Subtract(left, right) => has_aggregation(left) || has_aggregation(right),
        Expr::Multiply(left, right) => has_aggregation(left) || has_aggregation(right),
        Expr::Divide(left, right) => has_aggregation(left) || has_aggregation(right),
        _ => false,
    }
}

fn convert_node_to_object(node: &NodeData) -> HashMap<String, Value> {
    let mut object = HashMap::new();
    
    match node {
        NodeData::Regular { properties, .. } | NodeData::Schema { properties, .. } => {
            // Process all properties
            for (key, value) in properties {
                match value {
                    Value::Int64(_) | Value::Float64(_) | Value::UniqueId(_) => {
                        object.insert(key.clone(), value.clone());
                    }
                    Value::Null => {
                        object.insert(key.clone(), Value::Null);
                    }
                    Value::String(s) => {
                        // Try to parse as number
                        if let Ok(num) = s.parse::<f64>() {
                            object.insert(key.clone(), Value::Float64(num));
                        } else {
                            // Include the string value too
                            object.insert(key.clone(), value.clone());
                        }
                    }
                    _ => {
                        // Include all other value types
                        object.insert(key.clone(), value.clone());
                    }
                }
            }
        }
    }
    
    object
}

pub fn count_nodes_in_level(
    selection: &CurrentSelection,
    level_index: Option<usize>,
) -> usize {
    let effective_index = match level_index {
        Some(idx) => idx,
        None => selection.get_level_count().saturating_sub(1)
    };

    let level = selection.get_level(effective_index)
        .expect("Level should exist");
    
    level.get_all_nodes().len()
}

pub fn count_nodes_by_parent(
    graph: &DirGraph,
    selection: &CurrentSelection,
    level_index: Option<usize>,
) -> Vec<StatResult> {
    let pairs = get_parent_child_pairs(selection, level_index);
    
    pairs.iter()
        .map(|pair| {
            StatResult {
                node_idx: None,
                parent_idx: pair.parent,
                parent_title: pair.parent.and_then(|idx| {
                    graph.get_node(idx)
                        .and_then(|node| node.get_field("title"))
                        .and_then(|v| v.as_string())
                }),
                value: Value::Int64(pair.children.len() as i64),
                error_msg: None,
            }
        })
        .collect()
}

pub fn store_count_results(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    level_index: Option<usize>,
    group_by_parent: bool,
    target_property: &str,
) -> Result<CalculationOperationReport, String> {
    // Track start time for reporting
    let start_time = std::time::Instant::now();
    
    // Track errors
    let mut errors = Vec::new();
    
    let mut nodes_to_update: Vec<(Option<NodeIndex>, Value)> = Vec::new();
    
    let nodes_processed;
    if group_by_parent {
        // For grouped counting, store count for each parent
        let counts = count_nodes_by_parent(graph, selection, level_index);
        nodes_processed = counts.len();
        
        for result in &counts {
            if let Some(parent_idx) = result.parent_idx {
                // Verify the parent node exists in the graph
                if graph.get_node(parent_idx).is_some() {
                    nodes_to_update.push((Some(parent_idx), result.value.clone()));
                } else {
                    errors.push(format!("Parent node index {:?} not found in graph", parent_idx));
                }
            }
        }
    } else {
        // For flat counting, store same count for all nodes in level
        let count = count_nodes_in_level(selection, level_index);
        let effective_index = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
        
        if let Some(level) = selection.get_level(effective_index) {
            let all_nodes = level.get_all_nodes();
            nodes_processed = all_nodes.len();
            
            // Apply the count to each node in the level
            for node_idx in all_nodes {
                if graph.get_node(node_idx).is_some() {
                    nodes_to_update.push((Some(node_idx), Value::Int64(count as i64)));
                } else {
                    errors.push(format!("Node index {:?} not found in graph", node_idx));
                }
            }
        } else {
            let error_msg = format!("No valid level found at index {}", effective_index);
            errors.push(error_msg.clone());
            return Err(error_msg);
        }
    }
    
    // Check if we found any valid nodes to update
    if nodes_to_update.is_empty() {
        let error_msg = format!(
            "No valid nodes found to store '{}' count values.", target_property
        );
        errors.push(error_msg.clone());
        return Err(error_msg);
    }
    
    // Use the optimized batch update (which now returns a NodeOperationReport)
    let update_result = match maintain_graph::update_node_properties(graph, &nodes_to_update, target_property) {
        Ok(result) => result,
        Err(e) => {
            errors.push(format!("Failed to update node properties: {}", e));
            return Err(format!("Failed to update node properties: {}", e));
        }
    };
    
    // Add any errors from the update operation
    for error in &update_result.errors {
        errors.push(error.clone());
    }
    
    // Calculate elapsed time
    let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    
    // Create the calculation report
    let mut report = CalculationOperationReport::new(
        "count".to_string(),
        format!("count({})", if level_index.is_some() { format!("level {}", level_index.unwrap()) } else { "current level".to_string() }),
        nodes_processed,
        update_result.nodes_updated,
        update_result.nodes_skipped,
        elapsed_ms,
        group_by_parent
    );
    
    // Add errors if we found any
    if !errors.is_empty() {
        report = report.with_errors(errors);
    }
    
    Ok(report)
}