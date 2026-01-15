// src/graph/filtering_methods.rs
use std::collections::{HashMap, HashSet};
use petgraph::graph::NodeIndex;
use crate::datatypes::values::{Value, FilterCondition};
use crate::graph::schema::{DirGraph, CurrentSelection, SelectionOperation};

/// Constant for the "type" field key used in type filtering
const TYPE_FIELD: &str = "type";


pub fn matches_condition(value: &Value, condition: &FilterCondition) -> bool {
    match condition {
        FilterCondition::Equals(target) => values_equal(value, target),
        FilterCondition::NotEquals(target) => !values_equal(value, target),
        FilterCondition::GreaterThan(target) => compare_values(value, target) == Some(std::cmp::Ordering::Greater),
        FilterCondition::GreaterThanEquals(target) => {
            matches!(compare_values(value, target),
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal))
        },
        FilterCondition::LessThan(target) => compare_values(value, target) == Some(std::cmp::Ordering::Less),
        FilterCondition::LessThanEquals(target) => {
            matches!(compare_values(value, target),
                Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal))
        },
        FilterCondition::In(targets) => targets.contains(value),
        FilterCondition::IsNull => matches!(value, Value::Null),
        FilterCondition::IsNotNull => !matches!(value, Value::Null),
    }
}

/// Check equality with cross-type numeric comparison support
fn values_equal(a: &Value, b: &Value) -> bool {
    // Direct equality check first
    if a == b {
        return true;
    }
    // Handle numeric cross-type comparison (Int64 vs Float64)
    match (a, b) {
        (Value::Int64(i), Value::Float64(f)) => (*i as f64) == *f,
        (Value::Float64(f), Value::Int64(i)) => *f == (*i as f64),
        _ => false,
    }
}

pub fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Null, Value::Null) => Some(std::cmp::Ordering::Equal),
        (Value::Null, _) => Some(std::cmp::Ordering::Less),
        (_, Value::Null) => Some(std::cmp::Ordering::Greater),

        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        (Value::UniqueId(a), Value::UniqueId(b)) => Some(a.cmp(b)),
        (Value::DateTime(a), Value::DateTime(b)) => Some(a.cmp(b)),
        (Value::Boolean(a), Value::Boolean(b)) => Some(a.cmp(b)),
        // Handle DateTime vs String comparison by parsing the string
        (Value::DateTime(date), Value::String(s)) => {
            parse_date_string(s).map(|parsed| date.cmp(&parsed))
        },
        (Value::String(s), Value::DateTime(date)) => {
            parse_date_string(s).map(|parsed| parsed.cmp(date))
        },
        _ => None,
    }
}

/// Parse a date string in common formats (ISO YYYY-MM-DD preferred)
fn parse_date_string(s: &str) -> Option<chrono::NaiveDate> {
    use chrono::NaiveDate;
    // ISO format (YYYY-MM-DD)
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(s, "%Y/%m/%d"))
        .or_else(|_| NaiveDate::parse_from_str(s, "%d-%m-%Y"))
        .or_else(|_| NaiveDate::parse_from_str(s, "%m/%d/%Y"))
        .ok()
}

// Optimized core operations
fn filter_nodes_by_conditions(
    graph: &DirGraph,
    nodes: Vec<NodeIndex>,
    conditions: &HashMap<String, FilterCondition>
) -> Vec<NodeIndex> {
    // Special case for type filter which we can optimize
    if conditions.len() == 1 {
        if let Some((key, FilterCondition::Equals(Value::String(type_value)))) = conditions.iter().next() {
            if key == TYPE_FIELD {
                if let Some(type_nodes) = graph.type_indices.get(type_value) {
                    // Use HashSet for O(1) lookups
                    let type_set: HashSet<_> = type_nodes.iter().collect();
                    return nodes.into_iter()
                        .filter(|node| type_set.contains(node))
                        .collect();
                }
                return Vec::new();
            }
        }
    }

    // Try to use property indexes for equality conditions (O(1) lookup)
    // Find nodes by their node_type first, then check for indexed properties
    let node_types: HashSet<String> = nodes.iter()
        .filter_map(|&idx| {
            graph.get_node(idx).and_then(|n| {
                if let crate::graph::schema::NodeData::Regular { node_type, .. } = n {
                    Some(node_type.clone())
                } else {
                    None
                }
            })
        })
        .collect();

    // Check if any equality condition has an index we can use
    for (property, condition) in conditions {
        if let FilterCondition::Equals(target_value) = condition {
            // Check if any of our node types has an index on this property
            for node_type in &node_types {
                if let Some(matching_nodes) = graph.lookup_by_index(node_type, property, target_value) {
                    // Found an index! Use it to narrow down candidates
                    let indexed_set: HashSet<_> = matching_nodes.iter().copied().collect();
                    let original_set: HashSet<_> = nodes.iter().copied().collect();

                    // Intersection of indexed results with input nodes
                    let candidates: Vec<_> = indexed_set.intersection(&original_set).copied().collect();

                    // If there are remaining conditions, filter further
                    let remaining_conditions: HashMap<_, _> = conditions.iter()
                        .filter(|(k, _)| *k != property)
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();

                    if remaining_conditions.is_empty() {
                        return candidates;
                    } else {
                        // Recursively filter with remaining conditions
                        return filter_nodes_by_conditions(graph, candidates, &remaining_conditions);
                    }
                }
            }
        }
    }

    // Cache field lookups for frequently accessed fields
    let estimated_cache_size = nodes.len() * conditions.len();
    let mut field_cache: HashMap<(NodeIndex, &str), Option<Value>> = HashMap::with_capacity(estimated_cache_size);

    nodes.into_iter()
        .filter(|&idx| {
            if let Some(node) = graph.get_node(idx) {
                conditions.iter().all(|(key, condition)| {
                    let value = field_cache
                        .entry((idx, key.as_str()))
                        .or_insert_with(|| node.get_field(key));

                    match value {
                        Some(v) => matches_condition(v, condition),
                        None => {
                            // Missing field is treated as null
                            matches!(condition, FilterCondition::IsNull)
                        }
                    }
                })
            } else {
                false
            }
        })
        .collect()
}

fn sort_nodes_by_fields(
    graph: &DirGraph,
    mut nodes: Vec<NodeIndex>,
    sort_fields: &[(String, bool)]
) -> Vec<NodeIndex> {
    // Pre-fetch and cache field values for all nodes
    let mut value_cache: HashMap<(NodeIndex, &str), Option<Value>> = HashMap::new();

    for &node_idx in &nodes {
        if let Some(node) = graph.get_node(node_idx) {
            for (field, _) in sort_fields {
                value_cache.insert(
                    (node_idx, field.as_str()),
                    node.get_field(field)
                );
            }
        }
    }

    nodes.sort_by(|&a, &b| {
        for (field, ascending) in sort_fields {
            let val_a = value_cache.get(&(a, field.as_str()));
            let val_b = value_cache.get(&(b, field.as_str()));
            
            match (val_a, val_b) {
                (Some(Some(va)), Some(Some(vb))) => {
                    if let Some(ordering) = compare_values(va, vb) {
                        return if *ascending { ordering } else { ordering.reverse() };
                    }
                }
                _ => continue,
            }
        }
        std::cmp::Ordering::Equal
    });
    
    nodes
}

// Optimized processing function
pub fn process_nodes(
    graph: &DirGraph,
    nodes: Vec<NodeIndex>,
    conditions: Option<&HashMap<String, FilterCondition>>,
    sort_fields: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>
) -> Vec<NodeIndex> {
    let mut result = if let Some(max) = max_nodes {
        Vec::with_capacity(max.min(nodes.len()))
    } else {
        Vec::with_capacity(nodes.len())
    };
    
    result.extend(nodes);
    
    if let Some(conditions) = conditions {
        result = filter_nodes_by_conditions(graph, result, conditions);
    }
    
    if let Some(fields) = sort_fields {
        result = sort_nodes_by_fields(graph, result, fields);
    }
    
    if let Some(max) = max_nodes {
        result.truncate(max);
    }
    
    result
}

// Optimized public interface functions
pub fn filter_nodes(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    conditions: HashMap<String, FilterCondition>,
    sort_fields: Option<Vec<(String, bool)>>,
    max_nodes: Option<usize>
) -> Result<(), String> {
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection.get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    // Note: We don't clear selections here to allow chaining filters.
    // Each filter operation builds on the previous selection.

    if level.selections.is_empty() {
        // Optimized type-only filter handling
        if conditions.len() == 1 {
            if let Some((key, FilterCondition::Equals(Value::String(type_value)))) = conditions.iter().next() {
                if key == TYPE_FIELD {
                    if let Some(type_nodes) = graph.type_indices.get(type_value) {
                        let processed = process_nodes(
                            graph,
                            type_nodes.clone(),
                            None,
                            sort_fields.as_ref(),
                            max_nodes
                        );
                        
                        if !processed.is_empty() {
                            level.add_selection(None, processed);
                            level.operations.push(SelectionOperation::Filter(conditions));
                            return Ok(());
                        }
                    }
                    return Ok(());
                }
            }
        }

        // Regular processing with capacity hint
        let estimated_capacity = graph.graph.node_count() / 2;
        let mut all_nodes = Vec::with_capacity(estimated_capacity);
        all_nodes.extend(
            graph.graph.node_indices()
                .filter(|&idx| graph.get_node(idx).is_some_and(|n| n.is_regular()))
        );
            
        let processed = process_nodes(
            graph,
            all_nodes,
            Some(&conditions),
            sort_fields.as_ref(),
            max_nodes
        );
        
        if !processed.is_empty() {
            level.add_selection(None, processed);
        }
    } else {
        // Process existing selections with HashMap
        let mut new_selections = HashMap::new();
        
        for (parent, children) in level.selections.iter() {
            let processed = process_nodes(
                graph,
                children.clone(),
                Some(&conditions),
                sort_fields.as_ref(),
                max_nodes
            );
            
            if !processed.is_empty() {
                new_selections.insert(*parent, processed);
            }
        }
        
        level.selections = new_selections;
    }

    level.operations.push(SelectionOperation::Filter(conditions));
    if let Some(fields) = sort_fields {
        level.operations.push(SelectionOperation::Sort(fields));
    }

    Ok(())
}

// Other public functions remain the same but use the optimized process_nodes
pub fn sort_nodes(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    sort_fields: Vec<(String, bool)>
) -> Result<(), String> {
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection.get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    if level.selections.is_empty() {
        let mut all_nodes = Vec::with_capacity(graph.graph.node_count() / 2);
        all_nodes.extend(
            graph.graph.node_indices()
                .filter(|&idx| graph.get_node(idx).is_some_and(|n| n.is_regular()))
        );
            
        let sorted = sort_nodes_by_fields(graph, all_nodes, &sort_fields);
        if !sorted.is_empty() {
            level.add_selection(None, sorted);
        }
    } else {
        let mut new_selections = HashMap::new();
        
        for (parent, children) in level.selections.iter() {
            let sorted = sort_nodes_by_fields(graph, children.clone(), &sort_fields);
            if !sorted.is_empty() {
                new_selections.insert(*parent, sorted);
            }
        }
        
        level.selections = new_selections;
    }

    level.operations.push(SelectionOperation::Sort(sort_fields));
    Ok(())
}

pub fn limit_nodes_per_group(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    max_per_group: usize,
) -> Result<(), String> {
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection.get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    if level.selections.is_empty() {
        let mut all_nodes = Vec::with_capacity(graph.graph.node_count().min(max_per_group));
        all_nodes.extend(
            graph.graph.node_indices()
                .filter(|&idx| graph.get_node(idx).is_some_and(|n| n.is_regular()))
                .take(max_per_group)
        );
            
        if !all_nodes.is_empty() {
            level.add_selection(None, all_nodes);
        }
    } else {
        let mut new_selections = HashMap::new();
        
        for (parent, children) in level.selections.iter() {
            let mut limited = children.clone();
            limited.truncate(max_per_group);
            if !limited.is_empty() {
                new_selections.insert(*parent, limited);
            }
        }
        
        level.selections = new_selections;
    }

    Ok(())
}

pub fn filter_orphan_nodes(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    include_orphans: bool,  // true to include orphans, false to exclude them
    sort_fields: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>
) -> Result<(), String> {
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection.get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    // Function to check if a node is an orphan (no connections)
    let is_orphan = |node_idx: NodeIndex| {
        // Check both incoming and outgoing edges
        graph.graph.neighbors_directed(node_idx, petgraph::Direction::Outgoing).count() == 0 &&
        graph.graph.neighbors_directed(node_idx, petgraph::Direction::Incoming).count() == 0
    };

    if level.selections.is_empty() {
        // Start with all regular nodes
        let nodes = graph.graph.node_indices()
            .filter(|&idx| graph.get_node(idx).is_some_and(|n| n.is_regular()))
            .filter(|&idx| include_orphans == is_orphan(idx))
            .collect::<Vec<_>>();
            
        // Apply sorting and max limit
        let processed = process_nodes(
            graph,
            nodes,
            None,
            sort_fields,
            max_nodes
        );
        
        if !processed.is_empty() {
            level.add_selection(None, processed);
        }
    } else {
        // Process existing selections
        let mut new_selections = HashMap::new();
        
        for (parent, children) in level.selections.iter() {
            // Filter children based on orphan status
            let filtered = children.iter()
                .filter(|&&idx| include_orphans == is_orphan(idx))
                .copied()
                .collect::<Vec<_>>();
            
            // Apply sorting and max limit
            let processed = process_nodes(
                graph,
                filtered,
                None,
                sort_fields,
                max_nodes
            );
            
            if !processed.is_empty() {
                new_selections.insert(*parent, processed);
            }
        }
        
        level.selections = new_selections;
    }

    // Record the operation in the selection history
    level.operations.push(SelectionOperation::Custom(format!("filter_orphans(include={})", include_orphans)));
    if let Some(fields) = sort_fields {
        level.operations.push(SelectionOperation::Sort(fields.clone()));
    }

    Ok(())
}