//! Predicate pushdown into MATCH — equality/comparison extraction + application.

use super::super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher};
use std::collections::HashMap;

pub(super) fn push_where_into_match(query: &mut CypherQuery, params: &HashMap<String, Value>) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        let can_push = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Where(_)) | (Clause::OptionalMatch(_), Clause::Where(_))
        );

        if !can_push {
            i += 1;
            continue;
        }

        // Extract the WHERE predicate
        let where_pred = if let Clause::Where(w) = &query.clauses[i + 1] {
            w.predicate.clone()
        } else {
            i += 1;
            continue;
        };

        // Collect variables defined in the MATCH/OPTIONAL MATCH patterns
        let match_vars: Vec<(String, Option<String>)> = match &query.clauses[i] {
            Clause::Match(m) => collect_pattern_variables(&m.patterns),
            Clause::OptionalMatch(m) => collect_pattern_variables(&m.patterns),
            _ => {
                i += 1;
                continue;
            }
        };

        // Split predicate into pushable conditions and remainder
        let (pushable, pushable_in, pushable_cmp, remaining) =
            extract_pushable_equalities(&where_pred, &match_vars, params);

        // Apply pushable conditions to MATCH/OPTIONAL MATCH patterns
        if !pushable.is_empty() || !pushable_in.is_empty() || !pushable_cmp.is_empty() {
            let patterns = match &mut query.clauses[i] {
                Clause::Match(ref mut m) => &mut m.patterns,
                Clause::OptionalMatch(ref mut m) => &mut m.patterns,
                _ => {
                    i += 1;
                    continue;
                }
            };
            for (var_name, property, value) in &pushable {
                apply_property_to_patterns(patterns, var_name, property, value.clone());
            }
            for (var_name, property, values) in pushable_in {
                apply_in_property_to_patterns(patterns, &var_name, &property, values);
            }
            for (var_name, property, op, value) in pushable_cmp {
                apply_comparison_to_patterns(patterns, &var_name, &property, op, value);
            }

            // Update WHERE clause with remaining predicates.
            // When all predicates are pushed into the pattern, keep the WHERE
            // clause as-is so it acts as a safety-net filter. The pushed
            // predicates provide fast-path filtering in the pattern matcher,
            // but the WHERE clause must survive for correctness (e.g. when
            // fuse_match_return_aggregate rejects patterns with properties).
            if let Some(pred) = remaining {
                query.clauses[i + 1] = Clause::Where(WhereClause { predicate: pred });
            }
        }

        i += 1;
    }
}

/// Push LIMIT into MATCH when there's no ORDER BY/aggregation between them.
/// Reverse pattern direction when a later node has a more selective filter
/// than the first node, so the pattern executor starts from fewer candidates.
///
/// Example: `(d:CourtDecision)-[:CITES]->(s)-[:SECTION_OF]->(l:Law {korttittel: 'X'})`
/// → reversed to `(l:Law {korttittel: 'X'})<-[:SECTION_OF]-(s)<-[:CITES]-(d:CourtDecision)`
///
/// Must run AFTER `push_where_into_match` (so equality predicates are already in the pattern).
pub(super) fn collect_pattern_variables(
    patterns: &[crate::graph::core::pattern_matching::Pattern],
) -> Vec<(String, Option<String>)> {
    let mut vars = Vec::new();
    for pattern in patterns {
        for element in &pattern.elements {
            if let PatternElement::Node(np) = element {
                if let Some(ref var) = np.variable {
                    vars.push((var.clone(), np.node_type.clone()));
                }
            }
        }
    }
    vars
}

/// (equality_conditions, in_conditions, comparison_conditions, remaining_predicate)
type PushableResult = (
    Vec<(String, String, Value)>,
    Vec<(String, String, Vec<Value>)>,
    Vec<(String, String, ComparisonOp, Value)>,
    Option<Predicate>,
);

/// Extract pushable predicates from a WHERE clause into MATCH patterns.
/// Returns (equality_conditions, in_conditions, comparison_conditions, remaining_predicate).
///
/// Pushes conditions of the form:
/// - `variable.property = literal_value` (equality)
/// - `variable.property = $param` (equality with param)
/// - `variable.property IN [literal, ...]` (IN list)
/// - `variable.property > literal_value` (and >=, <, <=)
///
/// The variable must be defined in MATCH.
pub(super) fn extract_pushable_equalities(
    pred: &Predicate,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
) -> PushableResult {
    let mut pushable = Vec::new();
    let mut pushable_in = Vec::new();
    let mut pushable_cmp = Vec::new();
    let remaining = extract_from_predicate(
        pred,
        match_vars,
        params,
        &mut pushable,
        &mut pushable_in,
        &mut pushable_cmp,
    );
    (pushable, pushable_in, pushable_cmp, remaining)
}

/// Recursively extract pushable predicates from a predicate tree.
/// Returns the remaining predicate (None if fully consumed).
pub(super) fn extract_from_predicate(
    pred: &Predicate,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
    pushable: &mut Vec<(String, String, Value)>,
    pushable_in: &mut Vec<(String, String, Vec<Value>)>,
    pushable_cmp: &mut Vec<(String, String, ComparisonOp, Value)>,
) -> Option<Predicate> {
    match pred {
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals,
            right,
        } => {
            // Check if this is variable.property = literal or variable.property = $param
            if let Some((var, prop, val)) = try_extract_equality(left, right, match_vars, params) {
                pushable.push((var, prop, val));
                None // Fully consumed
            } else {
                Some(pred.clone()) // Keep as-is
            }
        }
        Predicate::Comparison {
            left,
            operator:
                op @ (ComparisonOp::GreaterThan
                | ComparisonOp::GreaterThanEq
                | ComparisonOp::LessThan
                | ComparisonOp::LessThanEq),
            right,
        } => {
            if let Some((var, prop, op, val)) =
                try_extract_comparison(left, right, *op, match_vars, params)
            {
                pushable_cmp.push((var, prop, op, val));
                None
            } else {
                Some(pred.clone())
            }
        }
        Predicate::In { expr, list } => {
            // Push variable.property IN [literal, ...] into MATCH pattern
            if let Expression::PropertyAccess { variable, property } = expr {
                if match_vars.iter().any(|(v, _)| v == variable) {
                    let all_literals: Option<Vec<Value>> = list
                        .iter()
                        .map(|item| {
                            if let Expression::Literal(val) = item {
                                Some(val.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    if let Some(values) = all_literals {
                        pushable_in.push((variable.clone(), property.clone(), values));
                        return None; // Fully consumed
                    }
                }
            }
            Some(pred.clone())
        }
        Predicate::And(left, right) => {
            let left_remaining = extract_from_predicate(
                left,
                match_vars,
                params,
                pushable,
                pushable_in,
                pushable_cmp,
            );
            let right_remaining = extract_from_predicate(
                right,
                match_vars,
                params,
                pushable,
                pushable_in,
                pushable_cmp,
            );

            match (left_remaining, right_remaining) {
                (None, None) => None,
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (Some(l), Some(r)) => Some(Predicate::And(Box::new(l), Box::new(r))),
            }
        }
        // Other predicate types can't be pushed
        _ => Some(pred.clone()),
    }
}

/// Try to extract a simple equality: variable.property = literal_or_param
pub(super) fn try_extract_equality(
    left: &Expression,
    right: &Expression,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
) -> Option<(String, String, Value)> {
    // Left is property access, right is literal
    if let (Expression::PropertyAccess { variable, property }, Expression::Literal(val)) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), val.clone()));
        }
    }

    // Right is property access, left is literal (commutative)
    if let (Expression::Literal(val), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), val.clone()));
        }
    }

    // Left is property access, right is parameter (resolve from params)
    if let (Expression::PropertyAccess { variable, property }, Expression::Parameter(name)) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), val.clone()));
            }
        }
    }

    // Right is property access, left is parameter (commutative)
    if let (Expression::Parameter(name), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), val.clone()));
            }
        }
    }

    // id(variable) = literal → treat as variable.id = literal
    // This enables O(1) lookup via lookup_by_id instead of full scan.
    if let (Expression::FunctionCall { name, args, .. }, Expression::Literal(val)) = (left, right) {
        if name == "id" {
            if let Some(Expression::Variable(var)) = args.first() {
                if match_vars.iter().any(|(v, _)| v == var) {
                    return Some((var.clone(), "id".to_string(), val.clone()));
                }
            }
        }
    }
    // Commutative: literal = id(variable)
    if let (Expression::Literal(val), Expression::FunctionCall { name, args, .. }) = (left, right) {
        if name == "id" {
            if let Some(Expression::Variable(var)) = args.first() {
                if match_vars.iter().any(|(v, _)| v == var) {
                    return Some((var.clone(), "id".to_string(), val.clone()));
                }
            }
        }
    }

    None
}

/// Try to extract a comparison: variable.property OP literal_or_param
/// When the literal is on the left (e.g. `30 < n.age`), reverse the operator
/// so it becomes `n.age > 30`.
pub(super) fn try_extract_comparison(
    left: &Expression,
    right: &Expression,
    op: ComparisonOp,
    match_vars: &[(String, Option<String>)],
    params: &HashMap<String, Value>,
) -> Option<(String, String, ComparisonOp, Value)> {
    // Left is property access, right is literal: variable.property OP literal
    if let (Expression::PropertyAccess { variable, property }, Expression::Literal(val)) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            return Some((variable.clone(), property.clone(), op, val.clone()));
        }
    }

    // Right is property access, left is literal: literal OP variable.property → reverse
    if let (Expression::Literal(val), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if match_vars.iter().any(|(v, _)| v == variable) {
            let reversed = match op {
                ComparisonOp::GreaterThan => ComparisonOp::LessThan,
                ComparisonOp::GreaterThanEq => ComparisonOp::LessThanEq,
                ComparisonOp::LessThan => ComparisonOp::GreaterThan,
                ComparisonOp::LessThanEq => ComparisonOp::GreaterThanEq,
                other => other,
            };
            return Some((variable.clone(), property.clone(), reversed, val.clone()));
        }
    }

    // Left is property access, right is parameter
    if let (Expression::PropertyAccess { variable, property }, Expression::Parameter(name)) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                return Some((variable.clone(), property.clone(), op, val.clone()));
            }
        }
    }

    // Right is property access, left is parameter → reverse
    if let (Expression::Parameter(name), Expression::PropertyAccess { variable, property }) =
        (left, right)
    {
        if let Some(val) = params.get(name.as_str()) {
            if match_vars.iter().any(|(v, _)| v == variable) {
                let reversed = match op {
                    ComparisonOp::GreaterThan => ComparisonOp::LessThan,
                    ComparisonOp::GreaterThanEq => ComparisonOp::LessThanEq,
                    ComparisonOp::LessThan => ComparisonOp::GreaterThan,
                    ComparisonOp::LessThanEq => ComparisonOp::GreaterThanEq,
                    other => other,
                };
                return Some((variable.clone(), property.clone(), reversed, val.clone()));
            }
        }
    }

    None
}

/// Apply a comparison condition to the matching node pattern in MATCH.
/// If the same property already has a comparison matcher (e.g. `year >= 2015`
/// followed by `year <= 2022`), merge them into a `Range` matcher.
pub(super) fn apply_comparison_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    op: ComparisonOp,
    value: Value,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    // Check if there's already a comparison on this property to merge
                    if let Some(existing) = props.get(property) {
                        if let Some(merged) = merge_comparison(existing, op, &value) {
                            props.insert(property.to_string(), merged);
                            return;
                        }
                    }
                    let matcher = match op {
                        ComparisonOp::GreaterThan => PropertyMatcher::GreaterThan(value),
                        ComparisonOp::GreaterThanEq => PropertyMatcher::GreaterOrEqual(value),
                        ComparisonOp::LessThan => PropertyMatcher::LessThan(value),
                        ComparisonOp::LessThanEq => PropertyMatcher::LessOrEqual(value),
                        _ => return,
                    };
                    props.insert(property.to_string(), matcher);
                    return;
                }
            }
        }
    }
}

/// Merge two comparison matchers on the same property into a Range.
/// E.g. existing `>= 2015` + new `<= 2022` → `Range { 2015..=2022 }`.
pub(super) fn merge_comparison(
    existing: &PropertyMatcher,
    new_op: ComparisonOp,
    new_val: &Value,
) -> Option<PropertyMatcher> {
    // Extract the existing bound direction
    let (existing_lower, existing_val, existing_inclusive) = match existing {
        PropertyMatcher::GreaterThan(v) => (true, v, false),
        PropertyMatcher::GreaterOrEqual(v) => (true, v, true),
        PropertyMatcher::LessThan(v) => (false, v, false),
        PropertyMatcher::LessOrEqual(v) => (false, v, true),
        _ => return None,
    };

    // Determine the new bound direction
    let (new_lower, new_inclusive) = match new_op {
        ComparisonOp::GreaterThan => (true, false),
        ComparisonOp::GreaterThanEq => (true, true),
        ComparisonOp::LessThan => (false, false),
        ComparisonOp::LessThanEq => (false, true),
        _ => return None,
    };

    // Can only merge opposite directions (lower + upper)
    if existing_lower == new_lower {
        return None; // Both are lower or both are upper — can't merge cleanly
    }

    if existing_lower {
        // existing is lower bound, new is upper bound
        Some(PropertyMatcher::Range {
            lower: existing_val.clone(),
            lower_inclusive: existing_inclusive,
            upper: new_val.clone(),
            upper_inclusive: new_inclusive,
        })
    } else {
        // existing is upper bound, new is lower bound
        Some(PropertyMatcher::Range {
            lower: new_val.clone(),
            lower_inclusive: new_inclusive,
            upper: existing_val.clone(),
            upper_inclusive: existing_inclusive,
        })
    }
}

/// Apply a property equality condition to the matching node pattern in MATCH
pub(super) fn apply_property_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    value: Value,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    // Don't overwrite an existing matcher (e.g. IN or Range)
                    props
                        .entry(property.to_string())
                        .or_insert(PropertyMatcher::Equals(value));
                    return;
                }
            }
        }
    }
}

/// Apply an IN-list property condition to the matching node pattern in MATCH
pub(super) fn apply_in_property_to_patterns(
    patterns: &mut [crate::graph::core::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    values: Vec<Value>,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    props.insert(property.to_string(), PropertyMatcher::In(values));
                    return;
                }
            }
        }
    }
}
