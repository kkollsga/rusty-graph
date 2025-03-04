use std::collections::{HashMap, HashSet};
use petgraph::graph::NodeIndex;
use crate::datatypes::values::{Value, FilterCondition};
use crate::graph::schema::{DirGraph, CurrentSelection, SelectionOperation};


pub fn matches_condition(value: &Value, condition: &FilterCondition) -> bool {
    match condition {
        FilterCondition::Equals(target) => value == target,
        FilterCondition::NotEquals(target) => value != target,
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
        _ => None,
    }
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
            if key == "type" {
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

    // Cache field lookups for frequently accessed fields
    let estimated_cache_size = nodes.len() * conditions.len();
    let mut field_cache: HashMap<(NodeIndex, &String), Option<Value>> = HashMap::with_capacity(estimated_cache_size);
    
    nodes.into_iter()
        .filter(|&idx| {
            if let Some(node) = graph.get_node(idx) {
                conditions.iter().all(|(key, condition)| {
                    let value = field_cache
                        .entry((idx, key))
                        .or_insert_with(|| node.get_field(key).map(|v| v.clone()));
                    
                    value.as_ref().map_or(false, |v| matches_condition(v, condition))
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
    let mut value_cache: HashMap<(NodeIndex, &String), Option<Value>> = HashMap::new();
    
    for &node_idx in &nodes {
        if let Some(node) = graph.get_node(node_idx) {
            for (field, _) in sort_fields {
                value_cache.insert(
                    (node_idx, field),
                    node.get_field(field).map(|v| v.clone())
                );
            }
        }
    }
    
    nodes.sort_by(|&a, &b| {
        for (field, ascending) in sort_fields {
            let val_a = value_cache.get(&(a, field));
            let val_b = value_cache.get(&(b, field));
            
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

    if current_index == 0 {
        level.selections.clear();
    }

    if level.selections.is_empty() {
        // Optimized type-only filter handling
        if conditions.len() == 1 {
            if let Some((key, FilterCondition::Equals(Value::String(type_value)))) = conditions.iter().next() {
                if key == "type" {
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
                .filter(|&idx| graph.get_node(idx).map_or(false, |n| n.is_regular()))
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
                .filter(|&idx| graph.get_node(idx).map_or(false, |n| n.is_regular()))
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
                .filter(|&idx| graph.get_node(idx).map_or(false, |n| n.is_regular()))
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
            .filter(|&idx| graph.get_node(idx).map_or(false, |n| n.is_regular()))
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
                .cloned()
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