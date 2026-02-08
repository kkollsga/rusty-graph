// src/graph/cypher/planner.rs
// Query optimizer: predicate pushdown, index hints, limit pushdown

use super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::pattern_matching::{PatternElement, PropertyMatcher};
use crate::graph::schema::DirGraph;

/// Optimize a parsed Cypher query before execution
pub fn optimize(query: &mut CypherQuery, graph: &DirGraph) {
    push_where_into_match(query);
    push_limit_into_match(query, graph);
}

/// Push simple equality predicates from WHERE into MATCH pattern properties.
/// This enables the pattern executor to filter during matching rather than after.
///
/// Example: MATCH (n:Person) WHERE n.age = 30
/// Becomes: MATCH (n:Person {age: 30}) (WHERE removed if fully consumed)
fn push_where_into_match(query: &mut CypherQuery) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        let can_push = matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Where(_))
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

        // Collect variables defined in the MATCH patterns
        let match_vars: Vec<(String, Option<String>)> = if let Clause::Match(m) = &query.clauses[i]
        {
            collect_pattern_variables(&m.patterns)
        } else {
            i += 1;
            continue;
        };

        // Split predicate into pushable equality conditions and remainder
        let (pushable, remaining) = extract_pushable_equalities(&where_pred, &match_vars);

        // Apply pushable conditions to MATCH patterns
        if !pushable.is_empty() {
            if let Clause::Match(ref mut m) = query.clauses[i] {
                for (var_name, property, value) in &pushable {
                    apply_property_to_patterns(&mut m.patterns, var_name, property, value.clone());
                }
            }

            // Update or remove WHERE clause
            match remaining {
                Some(pred) => {
                    query.clauses[i + 1] = Clause::Where(WhereClause { predicate: pred });
                }
                None => {
                    query.clauses.remove(i + 1);
                    continue; // Don't increment, check the new i+1
                }
            }
        }

        i += 1;
    }
}

/// Push LIMIT into MATCH when there's no ORDER BY between them.
/// This allows the pattern executor to stop early via max_matches.
fn push_limit_into_match(_query: &mut CypherQuery, _graph: &DirGraph) {
    // This optimization is deferred - it requires passing hints to PatternExecutor
    // which needs additional infrastructure. For now, LIMIT is applied post-execution.
    // The framework is here for future implementation.
}

/// Collect variable names and their node types from patterns
fn collect_pattern_variables(
    patterns: &[crate::graph::pattern_matching::Pattern],
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

/// Extract simple equality predicates that can be pushed into MATCH patterns.
/// Returns (pushable_conditions, remaining_predicate).
///
/// Only pushes conditions of the form: variable.property = literal_value
/// where the variable is defined in MATCH.
fn extract_pushable_equalities(
    pred: &Predicate,
    match_vars: &[(String, Option<String>)],
) -> (Vec<(String, String, Value)>, Option<Predicate>) {
    let mut pushable = Vec::new();
    let remaining = extract_from_predicate(pred, match_vars, &mut pushable);
    (pushable, remaining)
}

/// Recursively extract pushable equalities from a predicate.
/// Returns the remaining predicate (None if fully consumed).
fn extract_from_predicate(
    pred: &Predicate,
    match_vars: &[(String, Option<String>)],
    pushable: &mut Vec<(String, String, Value)>,
) -> Option<Predicate> {
    match pred {
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals,
            right,
        } => {
            // Check if this is variable.property = literal
            if let Some((var, prop, val)) = try_extract_equality(left, right, match_vars) {
                pushable.push((var, prop, val));
                None // Fully consumed
            } else {
                Some(pred.clone()) // Keep as-is
            }
        }
        Predicate::And(left, right) => {
            let left_remaining = extract_from_predicate(left, match_vars, pushable);
            let right_remaining = extract_from_predicate(right, match_vars, pushable);

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

/// Try to extract a simple equality: variable.property = literal
fn try_extract_equality(
    left: &Expression,
    right: &Expression,
    match_vars: &[(String, Option<String>)],
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

    None
}

/// Apply a property equality condition to the matching node pattern in MATCH
fn apply_property_to_patterns(
    patterns: &mut [crate::graph::pattern_matching::Pattern],
    var_name: &str,
    property: &str,
    value: Value,
) {
    for pattern in patterns.iter_mut() {
        for element in &mut pattern.elements {
            if let PatternElement::Node(ref mut np) = element {
                if np.variable.as_deref() == Some(var_name) {
                    let props = np.properties.get_or_insert_with(Default::default);
                    props.insert(property.to_string(), PropertyMatcher::Equals(value));
                    return;
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::cypher::parser::parse_cypher;

    #[test]
    fn test_predicate_pushdown_simple() {
        let mut query = parse_cypher("MATCH (n:Person) WHERE n.age = 30 RETURN n").unwrap();

        let graph = DirGraph::new();
        optimize(&mut query, &graph);

        // WHERE should be removed (fully consumed)
        assert_eq!(query.clauses.len(), 2); // MATCH + RETURN
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Return(_)));

        // The MATCH pattern should now have {age: 30} as a property
        if let Clause::Match(m) = &query.clauses[0] {
            if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
                assert!(np.properties.is_some());
                let props = np.properties.as_ref().unwrap();
                assert!(props.contains_key("age"));
            } else {
                panic!("Expected node pattern");
            }
        }
    }

    #[test]
    fn test_predicate_pushdown_partial() {
        let mut query =
            parse_cypher("MATCH (n:Person) WHERE n.age = 30 AND n.score > 100 RETURN n").unwrap();

        let graph = DirGraph::new();
        optimize(&mut query, &graph);

        // n.age = 30 should be pushed, n.score > 100 should remain
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN
        assert!(matches!(&query.clauses[1], Clause::Where(_)));

        if let Clause::Where(w) = &query.clauses[1] {
            // Remaining should be n.score > 100
            assert!(matches!(
                &w.predicate,
                Predicate::Comparison {
                    operator: ComparisonOp::GreaterThan,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_no_pushdown_for_non_equality() {
        let mut query = parse_cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();

        let graph = DirGraph::new();
        optimize(&mut query, &graph);

        // No pushdown - WHERE should remain
        assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN
    }
}
