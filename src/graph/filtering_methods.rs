// src/graph/filtering_methods.rs
use crate::datatypes::values::{FilterCondition, Value};
use crate::graph::schema::{CurrentSelection, DirGraph, SelectionOperation};
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

/// Constant for the "type" field key used in type filtering
const TYPE_FIELD: &str = "type";

pub fn matches_condition(value: &Value, condition: &FilterCondition) -> bool {
    match condition {
        FilterCondition::Equals(target) => values_equal(value, target),
        FilterCondition::NotEquals(target) => !values_equal(value, target),
        FilterCondition::GreaterThan(target) => {
            compare_values(value, target) == Some(std::cmp::Ordering::Greater)
        }
        FilterCondition::GreaterThanEquals(target) => {
            matches!(
                compare_values(value, target),
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
            )
        }
        FilterCondition::LessThan(target) => {
            compare_values(value, target) == Some(std::cmp::Ordering::Less)
        }
        FilterCondition::LessThanEquals(target) => {
            matches!(
                compare_values(value, target),
                Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
            )
        }
        FilterCondition::In(targets) => targets.iter().any(|t| values_equal(value, t)),
        FilterCondition::Between(min, max) => {
            // Inclusive range: min <= value <= max
            matches!(
                compare_values(value, min),
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
            ) && matches!(
                compare_values(value, max),
                Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
            )
        }
        FilterCondition::IsNull => matches!(value, Value::Null),
        FilterCondition::IsNotNull => !matches!(value, Value::Null),
        FilterCondition::Contains(target) => match (value, target) {
            (Value::String(s), Value::String(t)) => s.contains(t.as_str()),
            _ => false,
        },
        FilterCondition::StartsWith(target) => match (value, target) {
            (Value::String(s), Value::String(t)) => s.starts_with(t.as_str()),
            _ => false,
        },
        FilterCondition::EndsWith(target) => match (value, target) {
            (Value::String(s), Value::String(t)) => s.ends_with(t.as_str()),
            _ => false,
        },
    }
}

/// Check equality with cross-type numeric comparison support.
/// Handles Int64 <-> Float64 <-> UniqueId conversions to match Python's loose typing.
pub(crate) fn values_equal(a: &Value, b: &Value) -> bool {
    // Direct equality check first
    if a == b {
        return true;
    }
    // Handle numeric cross-type comparison
    match (a, b) {
        // Int64 <-> Float64
        (Value::Int64(i), Value::Float64(f)) => (*i as f64) == *f,
        (Value::Float64(f), Value::Int64(i)) => *f == (*i as f64),
        // UniqueId <-> Int64 (Python int may come as Int64 but be stored as UniqueId)
        (Value::UniqueId(u), Value::Int64(i)) => *i >= 0 && *u as i64 == *i,
        (Value::Int64(i), Value::UniqueId(u)) => *i >= 0 && *i == *u as i64,
        // UniqueId <-> Float64 (for completeness)
        (Value::UniqueId(u), Value::Float64(f)) => f.fract() == 0.0 && *u as f64 == *f,
        (Value::Float64(f), Value::UniqueId(u)) => f.fract() == 0.0 && *f == *u as f64,
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
        // UniqueId <-> Int64 cross-type comparison
        (Value::UniqueId(u), Value::Int64(i)) => (*u as i64).partial_cmp(i),
        (Value::Int64(i), Value::UniqueId(u)) => i.partial_cmp(&(*u as i64)),
        // UniqueId <-> Float64 cross-type comparison
        (Value::UniqueId(u), Value::Float64(f)) => (*u as f64).partial_cmp(f),
        (Value::Float64(f), Value::UniqueId(u)) => f.partial_cmp(&(*u as f64)),
        (Value::DateTime(a), Value::DateTime(b)) => Some(a.cmp(b)),
        (Value::Boolean(a), Value::Boolean(b)) => Some(a.cmp(b)),
        // Handle DateTime vs String comparison by parsing the string
        (Value::DateTime(date), Value::String(s)) => {
            parse_date_string(s).map(|parsed| date.cmp(&parsed))
        }
        (Value::String(s), Value::DateTime(date)) => {
            parse_date_string(s).map(|parsed| parsed.cmp(date))
        }
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
    conditions: &HashMap<String, FilterCondition>,
) -> Vec<NodeIndex> {
    // Special case for type filter which we can optimize
    if conditions.len() == 1 {
        if let Some((key, FilterCondition::Equals(Value::String(type_value)))) =
            conditions.iter().next()
        {
            if key == TYPE_FIELD {
                if let Some(type_nodes) = graph.type_indices.get(type_value) {
                    // Use HashSet for O(1) lookups
                    let type_set: HashSet<_> = type_nodes.iter().collect();
                    return nodes
                        .into_iter()
                        .filter(|node| type_set.contains(node))
                        .collect();
                }
                return Vec::new();
            }
        }
    }

    // Try to use property indexes for equality conditions (O(1) lookup)
    // Find nodes by their node_type first, then check for indexed properties
    let node_types: HashSet<String> = nodes
        .iter()
        .filter_map(|&idx| graph.get_node(idx).map(|n| n.node_type.clone()))
        .collect();

    // Collect equality conditions that could use a composite index
    let equality_conditions: Vec<(&String, &crate::datatypes::values::Value)> = conditions
        .iter()
        .filter_map(|(k, v)| {
            if let FilterCondition::Equals(val) = v {
                Some((k, val))
            } else {
                None
            }
        })
        .collect();

    // Try composite index first (if we have 2+ equality conditions)
    if equality_conditions.len() >= 2 {
        let eq_properties: Vec<String> = equality_conditions
            .iter()
            .map(|(k, _)| (*k).clone())
            .collect();

        for node_type in &node_types {
            if let Some((index_key, is_exact)) =
                graph.find_matching_composite_index(node_type, &eq_properties)
            {
                if is_exact {
                    // Exact match - we can use the composite index directly
                    // Build values in the same order as the index
                    let index_properties = &index_key.1;
                    let values: Vec<crate::datatypes::values::Value> = index_properties
                        .iter()
                        .map(|p| {
                            equality_conditions
                                .iter()
                                .find(|(k, _)| *k == p)
                                .map(|(_, v)| (*v).clone())
                                .unwrap_or(crate::datatypes::values::Value::Null)
                        })
                        .collect();

                    if let Some(matching_nodes) =
                        graph.lookup_by_composite_index(node_type, index_properties, &values)
                    {
                        // Found composite index match!
                        let indexed_set: HashSet<_> = matching_nodes.iter().copied().collect();
                        let original_set: HashSet<_> = nodes.iter().copied().collect();

                        // Intersection of indexed results with input nodes
                        let candidates: Vec<_> =
                            indexed_set.intersection(&original_set).copied().collect();

                        // Filter remaining non-equality conditions
                        let remaining_conditions: HashMap<_, _> = conditions
                            .iter()
                            .filter(|(k, v)| {
                                !matches!(v, FilterCondition::Equals(_))
                                    || !eq_properties.contains(k)
                            })
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();

                        if remaining_conditions.is_empty() {
                            return candidates;
                        } else {
                            return filter_nodes_by_conditions(
                                graph,
                                candidates,
                                &remaining_conditions,
                            );
                        }
                    }
                }
            }
        }
    }

    // Fall back to single-property index check
    for (property, condition) in conditions {
        if let FilterCondition::Equals(target_value) = condition {
            // Check if any of our node types has an index on this property
            for node_type in &node_types {
                if let Some(matching_nodes) =
                    graph.lookup_by_index(node_type, property, target_value)
                {
                    // Found an index! Use it to narrow down candidates
                    let indexed_set: HashSet<_> = matching_nodes.iter().copied().collect();
                    let original_set: HashSet<_> = nodes.iter().copied().collect();

                    // Intersection of indexed results with input nodes
                    let candidates: Vec<_> =
                        indexed_set.intersection(&original_set).copied().collect();

                    // If there are remaining conditions, filter further
                    let remaining_conditions: HashMap<_, _> = conditions
                        .iter()
                        .filter(|(k, _)| *k != property)
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();

                    if remaining_conditions.is_empty() {
                        return candidates;
                    } else {
                        // Recursively filter with remaining conditions
                        return filter_nodes_by_conditions(
                            graph,
                            candidates,
                            &remaining_conditions,
                        );
                    }
                }
            }
        }
    }

    // Cache field lookups for frequently accessed fields
    let estimated_cache_size = nodes.len() * conditions.len();
    let mut field_cache: HashMap<(NodeIndex, &str), Option<Value>> =
        HashMap::with_capacity(estimated_cache_size);

    nodes
        .into_iter()
        .filter(|&idx| {
            if let Some(node) = graph.get_node(idx) {
                conditions.iter().all(|(key, condition)| {
                    let value = field_cache
                        .entry((idx, key.as_str()))
                        .or_insert_with(|| node.get_field_ref(key).cloned());

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
    sort_fields: &[(String, bool)],
) -> Vec<NodeIndex> {
    // Pre-fetch and cache field values for all nodes
    let mut value_cache: HashMap<(NodeIndex, &str), Option<Value>> = HashMap::new();

    for &node_idx in &nodes {
        if let Some(node) = graph.get_node(node_idx) {
            for (field, _) in sort_fields {
                value_cache.insert(
                    (node_idx, field.as_str()),
                    node.get_field_ref(field).cloned(),
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
                        return if *ascending {
                            ordering
                        } else {
                            ordering.reverse()
                        };
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
    max_nodes: Option<usize>,
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
    max_nodes: Option<usize>,
) -> Result<(), String> {
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    // Note: We don't clear selections here to allow chaining filters.
    // Each filter operation builds on the previous selection.

    if level.selections.is_empty() {
        // Optimized type-only filter handling
        if conditions.len() == 1 {
            if let Some((key, FilterCondition::Equals(Value::String(type_value)))) =
                conditions.iter().next()
            {
                if key == TYPE_FIELD {
                    if let Some(type_nodes) = graph.type_indices.get(type_value) {
                        let processed = process_nodes(
                            graph,
                            type_nodes.clone(),
                            None,
                            sort_fields.as_ref(),
                            max_nodes,
                        );

                        if !processed.is_empty() {
                            level.add_selection(None, processed);
                        }
                    }
                    // Always record the filter operation (even if 0 nodes matched)
                    level
                        .operations
                        .push(SelectionOperation::Filter(conditions));
                    return Ok(());
                }
            }
        }

        // Regular processing with capacity hint
        let estimated_capacity = graph.graph.node_count() / 2;
        let mut all_nodes = Vec::with_capacity(estimated_capacity);
        all_nodes.extend(graph.graph.node_indices());

        let processed = process_nodes(
            graph,
            all_nodes,
            Some(&conditions),
            sort_fields.as_ref(),
            max_nodes,
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
                max_nodes,
            );

            if !processed.is_empty() {
                new_selections.insert(*parent, processed);
            }
        }

        level.selections = new_selections;
    }

    level
        .operations
        .push(SelectionOperation::Filter(conditions));
    if let Some(fields) = sort_fields {
        level.operations.push(SelectionOperation::Sort(fields));
    }

    Ok(())
}

// Other public functions remain the same but use the optimized process_nodes
pub fn sort_nodes(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    sort_fields: Vec<(String, bool)>,
) -> Result<(), String> {
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    if level.selections.is_empty() {
        let mut all_nodes = Vec::with_capacity(graph.graph.node_count() / 2);
        all_nodes.extend(graph.graph.node_indices());

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
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    if level.selections.is_empty() {
        let mut all_nodes = Vec::with_capacity(graph.graph.node_count().min(max_per_group));
        all_nodes.extend(graph.graph.node_indices().take(max_per_group));

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
    include_orphans: bool, // true to include orphans, false to exclude them
    sort_fields: Option<&Vec<(String, bool)>>,
    max_nodes: Option<usize>,
) -> Result<(), String> {
    let current_index = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level_mut(current_index)
        .ok_or_else(|| "No active selection level".to_string())?;

    // Function to check if a node is an orphan (no connections)
    let is_orphan = |node_idx: NodeIndex| {
        // Check both incoming and outgoing edges
        graph
            .graph
            .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
            .count()
            == 0
            && graph
                .graph
                .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                .count()
                == 0
    };

    if level.selections.is_empty() {
        // Start with all nodes
        let nodes = graph
            .graph
            .node_indices()
            .filter(|&idx| include_orphans == is_orphan(idx))
            .collect::<Vec<_>>();

        // Apply sorting and max limit
        let processed = process_nodes(graph, nodes, None, sort_fields, max_nodes);

        if !processed.is_empty() {
            level.add_selection(None, processed);
        }
    } else {
        // Process existing selections
        let mut new_selections = HashMap::new();

        for (parent, children) in level.selections.iter() {
            // Filter children based on orphan status
            let filtered = children
                .iter()
                .filter(|&&idx| include_orphans == is_orphan(idx))
                .copied()
                .collect::<Vec<_>>();

            // Apply sorting and max limit
            let processed = process_nodes(graph, filtered, None, sort_fields, max_nodes);

            if !processed.is_empty() {
                new_selections.insert(*parent, processed);
            }
        }

        level.selections = new_selections;
    }

    // Record the operation in the selection history
    level.operations.push(SelectionOperation::Custom(format!(
        "filter_orphans(include={})",
        include_orphans
    )));
    if let Some(fields) = sort_fields {
        level
            .operations
            .push(SelectionOperation::Sort(fields.clone()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::values::{FilterCondition, Value};
    use chrono::NaiveDate;

    // ========================================================================
    // values_equal — cross-type numeric comparisons
    // ========================================================================

    #[test]
    fn test_values_equal_same_type() {
        assert!(values_equal(&Value::Int64(5), &Value::Int64(5)));
        assert!(values_equal(&Value::Float64(3.14), &Value::Float64(3.14)));
        assert!(values_equal(
            &Value::String("abc".into()),
            &Value::String("abc".into())
        ));
        assert!(values_equal(&Value::Null, &Value::Null));
    }

    #[test]
    fn test_values_equal_int_float_crosstype() {
        assert!(values_equal(&Value::Int64(5), &Value::Float64(5.0)));
        assert!(values_equal(&Value::Float64(5.0), &Value::Int64(5)));
        assert!(!values_equal(&Value::Int64(5), &Value::Float64(5.1)));
    }

    #[test]
    fn test_values_equal_uniqueid_int() {
        assert!(values_equal(&Value::UniqueId(10), &Value::Int64(10)));
        assert!(values_equal(&Value::Int64(10), &Value::UniqueId(10)));
        assert!(!values_equal(&Value::UniqueId(10), &Value::Int64(11)));
    }

    #[test]
    fn test_values_equal_uniqueid_float() {
        assert!(values_equal(&Value::UniqueId(7), &Value::Float64(7.0)));
        assert!(!values_equal(&Value::UniqueId(7), &Value::Float64(7.5)));
    }

    #[test]
    fn test_values_equal_different_types() {
        assert!(!values_equal(&Value::Int64(1), &Value::String("1".into())));
        assert!(!values_equal(&Value::Boolean(true), &Value::Int64(1)));
    }

    // ========================================================================
    // compare_values — ordering
    // ========================================================================

    #[test]
    fn test_compare_values_integers() {
        assert_eq!(
            compare_values(&Value::Int64(1), &Value::Int64(2)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&Value::Int64(2), &Value::Int64(2)),
            Some(std::cmp::Ordering::Equal)
        );
        assert_eq!(
            compare_values(&Value::Int64(3), &Value::Int64(2)),
            Some(std::cmp::Ordering::Greater)
        );
    }

    #[test]
    fn test_compare_values_floats() {
        assert_eq!(
            compare_values(&Value::Float64(1.0), &Value::Float64(2.0)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&Value::Float64(2.0), &Value::Float64(2.0)),
            Some(std::cmp::Ordering::Equal)
        );
    }

    #[test]
    fn test_compare_values_cross_type_numeric() {
        assert_eq!(
            compare_values(&Value::Int64(1), &Value::Float64(2.5)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&Value::Float64(3.0), &Value::Int64(2)),
            Some(std::cmp::Ordering::Greater)
        );
    }

    #[test]
    fn test_compare_values_strings() {
        assert_eq!(
            compare_values(&Value::String("abc".into()), &Value::String("def".into())),
            Some(std::cmp::Ordering::Less)
        );
    }

    #[test]
    fn test_compare_values_null_ordering() {
        // Null < any non-null
        assert_eq!(
            compare_values(&Value::Null, &Value::Int64(0)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&Value::Int64(0), &Value::Null),
            Some(std::cmp::Ordering::Greater)
        );
        assert_eq!(
            compare_values(&Value::Null, &Value::Null),
            Some(std::cmp::Ordering::Equal)
        );
    }

    #[test]
    fn test_compare_values_incompatible_types() {
        assert_eq!(
            compare_values(&Value::String("a".into()), &Value::Int64(1)),
            None
        );
        assert_eq!(
            compare_values(&Value::Boolean(true), &Value::Float64(1.0)),
            None
        );
    }

    #[test]
    fn test_compare_values_datetime_vs_string() {
        let date = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
        let result = compare_values(&Value::DateTime(date), &Value::String("2024-06-15".into()));
        assert_eq!(result, Some(std::cmp::Ordering::Equal));

        let result = compare_values(&Value::DateTime(date), &Value::String("2024-01-01".into()));
        assert_eq!(result, Some(std::cmp::Ordering::Greater));
    }

    // ========================================================================
    // matches_condition — filter operators
    // ========================================================================

    #[test]
    fn test_matches_condition_equals() {
        assert!(matches_condition(
            &Value::Int64(5),
            &FilterCondition::Equals(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(5),
            &FilterCondition::Equals(Value::Int64(6))
        ));
    }

    #[test]
    fn test_matches_condition_not_equals() {
        assert!(matches_condition(
            &Value::Int64(5),
            &FilterCondition::NotEquals(Value::Int64(6))
        ));
        assert!(!matches_condition(
            &Value::Int64(5),
            &FilterCondition::NotEquals(Value::Int64(5))
        ));
    }

    #[test]
    fn test_matches_condition_greater_than() {
        assert!(matches_condition(
            &Value::Int64(10),
            &FilterCondition::GreaterThan(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(5),
            &FilterCondition::GreaterThan(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(3),
            &FilterCondition::GreaterThan(Value::Int64(5))
        ));
    }

    #[test]
    fn test_matches_condition_greater_than_equals() {
        assert!(matches_condition(
            &Value::Int64(10),
            &FilterCondition::GreaterThanEquals(Value::Int64(5))
        ));
        assert!(matches_condition(
            &Value::Int64(5),
            &FilterCondition::GreaterThanEquals(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(3),
            &FilterCondition::GreaterThanEquals(Value::Int64(5))
        ));
    }

    #[test]
    fn test_matches_condition_less_than() {
        assert!(matches_condition(
            &Value::Int64(3),
            &FilterCondition::LessThan(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(5),
            &FilterCondition::LessThan(Value::Int64(5))
        ));
    }

    #[test]
    fn test_matches_condition_less_than_equals() {
        assert!(matches_condition(
            &Value::Int64(3),
            &FilterCondition::LessThanEquals(Value::Int64(5))
        ));
        assert!(matches_condition(
            &Value::Int64(5),
            &FilterCondition::LessThanEquals(Value::Int64(5))
        ));
        assert!(!matches_condition(
            &Value::Int64(6),
            &FilterCondition::LessThanEquals(Value::Int64(5))
        ));
    }

    #[test]
    fn test_matches_condition_in() {
        let targets = vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)];
        assert!(matches_condition(
            &Value::Int64(2),
            &FilterCondition::In(targets.clone())
        ));
        assert!(!matches_condition(
            &Value::Int64(5),
            &FilterCondition::In(targets)
        ));
    }

    #[test]
    fn test_matches_condition_between() {
        assert!(matches_condition(
            &Value::Int64(5),
            &FilterCondition::Between(Value::Int64(1), Value::Int64(10))
        ));
        assert!(matches_condition(
            &Value::Int64(1),
            &FilterCondition::Between(Value::Int64(1), Value::Int64(10))
        )); // inclusive
        assert!(matches_condition(
            &Value::Int64(10),
            &FilterCondition::Between(Value::Int64(1), Value::Int64(10))
        )); // inclusive
        assert!(!matches_condition(
            &Value::Int64(0),
            &FilterCondition::Between(Value::Int64(1), Value::Int64(10))
        ));
        assert!(!matches_condition(
            &Value::Int64(11),
            &FilterCondition::Between(Value::Int64(1), Value::Int64(10))
        ));
    }

    #[test]
    fn test_matches_condition_is_null() {
        assert!(matches_condition(&Value::Null, &FilterCondition::IsNull));
        assert!(!matches_condition(
            &Value::Int64(0),
            &FilterCondition::IsNull
        ));
    }

    #[test]
    fn test_matches_condition_is_not_null() {
        assert!(matches_condition(
            &Value::Int64(0),
            &FilterCondition::IsNotNull
        ));
        assert!(!matches_condition(
            &Value::Null,
            &FilterCondition::IsNotNull
        ));
    }

    // ========================================================================
    // parse_date_string
    // ========================================================================

    // ========================================================================
    // matches_condition — string predicates
    // ========================================================================

    #[test]
    fn test_matches_condition_contains() {
        assert!(matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::Contains(Value::String("world".into()))
        ));
        assert!(matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::Contains(Value::String("hello".into()))
        ));
        assert!(!matches_condition(
            &Value::String("hello".into()),
            &FilterCondition::Contains(Value::String("world".into()))
        ));
        // Non-string values return false
        assert!(!matches_condition(
            &Value::Int64(42),
            &FilterCondition::Contains(Value::String("4".into()))
        ));
    }

    #[test]
    fn test_matches_condition_starts_with() {
        assert!(matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::StartsWith(Value::String("hello".into()))
        ));
        assert!(!matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::StartsWith(Value::String("world".into()))
        ));
        assert!(!matches_condition(
            &Value::Int64(42),
            &FilterCondition::StartsWith(Value::String("4".into()))
        ));
    }

    #[test]
    fn test_matches_condition_ends_with() {
        assert!(matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::EndsWith(Value::String("world".into()))
        ));
        assert!(!matches_condition(
            &Value::String("hello world".into()),
            &FilterCondition::EndsWith(Value::String("hello".into()))
        ));
        assert!(!matches_condition(
            &Value::Int64(42),
            &FilterCondition::EndsWith(Value::String("2".into()))
        ));
    }

    // ========================================================================
    // parse_date_string
    // ========================================================================

    #[test]
    fn test_parse_date_string_iso() {
        let result = parse_date_string("2024-06-15");
        assert_eq!(result, Some(NaiveDate::from_ymd_opt(2024, 6, 15).unwrap()));
    }

    #[test]
    fn test_parse_date_string_slash() {
        let result = parse_date_string("2024/06/15");
        assert_eq!(result, Some(NaiveDate::from_ymd_opt(2024, 6, 15).unwrap()));
    }

    #[test]
    fn test_parse_date_string_invalid() {
        assert_eq!(parse_date_string("not-a-date"), None);
        assert_eq!(parse_date_string(""), None);
    }
}
