// src/graph/temporal.rs
//
// Shared temporal validity helpers for filtering nodes and edges by date.

use crate::datatypes::values::Value;
use crate::graph::schema::{NodeData, TemporalConfig};
use chrono::NaiveDate;
use std::collections::HashMap;

/// Check if a set of properties is temporally valid at a reference date.
///
/// Valid when: valid_from <= reference AND (valid_to IS NULL OR valid_to >= reference)
///
/// Handles Value::DateTime(NaiveDate) and Value::Null (open-ended).
/// Missing properties are treated as unbounded (always valid on that side).
pub fn is_temporally_valid(
    properties: &HashMap<String, Value>,
    config: &TemporalConfig,
    reference: &NaiveDate,
) -> bool {
    // Check valid_from: must be <= reference (or missing/null = unbounded start)
    if let Some(from_val) = properties.get(&config.valid_from) {
        match from_val {
            Value::DateTime(d) => {
                if d > reference {
                    return false;
                }
            }
            Value::Null => {} // unbounded
            _ => {}           // non-date value, skip check
        }
    }

    // Check valid_to: must be >= reference (or missing/null = still active)
    if let Some(to_val) = properties.get(&config.valid_to) {
        match to_val {
            Value::DateTime(d) => {
                if d < reference {
                    return false;
                }
            }
            Value::Null => {} // unbounded (still active)
            _ => {}           // non-date value, skip check
        }
    }

    true
}

/// Check if a node is temporally valid at a reference date.
///
/// Uses `get_field_ref()` which checks id, title, and properties.
pub fn node_is_temporally_valid(
    node: &NodeData,
    config: &TemporalConfig,
    reference: &NaiveDate,
) -> bool {
    // Check valid_from
    if let Some(val) = node.get_field_ref(&config.valid_from) {
        match val {
            Value::DateTime(d) => {
                if d > reference {
                    return false;
                }
            }
            Value::Null => {}
            _ => {}
        }
    }

    // Check valid_to
    if let Some(val) = node.get_field_ref(&config.valid_to) {
        match val {
            Value::DateTime(d) => {
                if d < reference {
                    return false;
                }
            }
            Value::Null => {}
            _ => {}
        }
    }

    true
}

/// Check if a validity period overlaps a date range [start, end].
///
/// Overlap when: valid_from <= end AND (valid_to IS NULL OR valid_to >= start)
pub fn overlaps_range(
    properties: &HashMap<String, Value>,
    config: &TemporalConfig,
    start: &NaiveDate,
    end: &NaiveDate,
) -> bool {
    // Check valid_from <= end
    if let Some(from_val) = properties.get(&config.valid_from) {
        match from_val {
            Value::DateTime(d) => {
                if d > end {
                    return false;
                }
            }
            Value::Null => {}
            _ => {}
        }
    }

    // Check valid_to >= start
    if let Some(to_val) = properties.get(&config.valid_to) {
        match to_val {
            Value::DateTime(d) => {
                if d < start {
                    return false;
                }
            }
            Value::Null => {}
            _ => {}
        }
    }

    true
}

/// Check if a node's validity period overlaps a date range [start, end].
///
/// Uses `get_field_ref()` which checks id, title, and properties.
pub fn node_overlaps_range(
    node: &NodeData,
    config: &TemporalConfig,
    start: &NaiveDate,
    end: &NaiveDate,
) -> bool {
    // Check valid_from <= end
    if let Some(val) = node.get_field_ref(&config.valid_from) {
        match val {
            Value::DateTime(d) => {
                if d > end {
                    return false;
                }
            }
            Value::Null => {}
            _ => {}
        }
    }

    // Check valid_to >= start
    if let Some(val) = node.get_field_ref(&config.valid_to) {
        match val {
            Value::DateTime(d) => {
                if d < start {
                    return false;
                }
            }
            Value::Null => {} // unbounded (still active)
            _ => {}
        }
    }

    true
}

/// Check if edge properties pass ANY temporal config in the list.
///
/// For each config, checks if the valid_from or valid_to field exists on the edge.
/// If found, uses that config for the temporal validity check.
/// If no config's fields exist on the edge, returns true (non-temporal edge).
pub fn is_temporally_valid_multi(
    properties: &HashMap<String, Value>,
    configs: &[TemporalConfig],
    reference: &NaiveDate,
) -> bool {
    for config in configs {
        if properties.contains_key(&config.valid_from) || properties.contains_key(&config.valid_to)
        {
            return is_temporally_valid(properties, config, reference);
        }
    }
    true // no matching config = not temporal for this edge
}

/// Check if edge properties overlap a date range, trying multiple configs.
///
/// Same multi-config matching as `is_temporally_valid_multi`.
pub fn overlaps_range_multi(
    properties: &HashMap<String, Value>,
    configs: &[TemporalConfig],
    start: &NaiveDate,
    end: &NaiveDate,
) -> bool {
    for config in configs {
        if properties.contains_key(&config.valid_from) || properties.contains_key(&config.valid_to)
        {
            return overlaps_range(properties, config, start, end);
        }
    }
    true
}

/// Check if a node passes the given temporal context.
///
/// Dispatches to `node_is_temporally_valid` (Today/At) or `node_overlaps_range` (During).
/// Returns `true` for `All` (no filtering).
pub fn node_passes_context(
    node: &NodeData,
    config: &TemporalConfig,
    context: &super::TemporalContext,
) -> bool {
    use super::TemporalContext;
    match context {
        TemporalContext::All => true,
        TemporalContext::Today => {
            let today = chrono::Local::now().date_naive();
            node_is_temporally_valid(node, config, &today)
        }
        TemporalContext::At(d) => node_is_temporally_valid(node, config, d),
        TemporalContext::During(start, end) => node_overlaps_range(node, config, start, end),
    }
}
