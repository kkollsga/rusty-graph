//! Schema validation pass — runs after parse, before optimize.
//!
//! Catches unknown property references inside pattern-literal syntax
//! (`{prop: value}`) before the executor commits to a scan. These are
//! unambiguously property names — a typo here can never be a virtual
//! column or computed alias, so rejecting produces zero false positives.
//!
//! Deliberately *does not* validate:
//! - Unknown node types in MATCH (`MATCH (n:Nonexistent)` legitimately
//!   returns zero rows and is a common existence-check idiom).
//! - Unknown connection types (same rationale).
//! - Property references in WHERE / RETURN expressions (virtual columns,
//!   timeseries sub-nodes, aliases can be legitimate `n.prop` accesses
//!   not present in `node_type_metadata`).
//!
//! Phase 3 will surface those as non-fatal warnings in `QueryDiagnostics`
//! with "did you mean?" hints — the agent sees the signal without
//! rejecting legitimate empty-result queries.

use super::super::ast::*;
use crate::graph::core::pattern_matching::{Pattern, PatternElement};
use crate::graph::mutation::validation::did_you_mean;
use crate::graph::schema::DirGraph;
use std::collections::HashMap;

/// Built-in fields valid on any node type — mirrors BUILTIN_FIELDS in
/// `mutation/validation.rs`. Listed explicitly so it's obvious what's
/// tolerated without a metadata entry.
const BUILTIN_FIELDS: &[&str] = &["id", "title", "name", "type"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaErrorKind {
    UnknownProperty,
}

#[derive(Debug, Clone)]
pub struct SchemaError {
    #[allow(dead_code)] // Test-only.
    pub kind: SchemaErrorKind,
    pub message: String,
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SchemaError {}

/// Validate a parsed Cypher query against the graph schema.
///
/// Runs in O(clauses) with O(1) lookups against `node_type_metadata`.
/// Returns early on the first violation.
///
/// Skips validation entirely when the graph has neither declared node
/// types (`node_type_metadata`) nor live nodes (`type_indices`) — that
/// state appears in tests and during initial construction.
pub fn validate_schema(query: &CypherQuery, graph: &DirGraph) -> Result<(), SchemaError> {
    // If the graph has no declared node types at all, there is nothing to
    // validate against — skip. This covers fresh graphs and construction
    // before any nodes/edges exist.
    if graph.node_type_metadata.is_empty() && graph.type_indices.is_empty() {
        return Ok(());
    }

    let mut var_types: HashMap<String, String> = HashMap::new();
    validate_query(query, graph, &mut var_types)
}

fn validate_query(
    query: &CypherQuery,
    graph: &DirGraph,
    var_types: &mut HashMap<String, String>,
) -> Result<(), SchemaError> {
    for clause in &query.clauses {
        validate_clause(clause, graph, var_types)?;
    }
    Ok(())
}

fn validate_clause(
    clause: &Clause,
    graph: &DirGraph,
    var_types: &mut HashMap<String, String>,
) -> Result<(), SchemaError> {
    match clause {
        Clause::Match(m) | Clause::OptionalMatch(m) => {
            for pattern in &m.patterns {
                validate_pattern(pattern, graph, var_types)?;
            }
        }
        Clause::Where(w) => {
            walk_predicate_for_nested_patterns(&w.predicate, graph, var_types)?;
        }
        Clause::With(w) => {
            for item in &w.items {
                if let Some(alias) = &item.alias {
                    var_types.remove(alias);
                }
            }
            if let Some(wc) = &w.where_clause {
                walk_predicate_for_nested_patterns(&wc.predicate, graph, var_types)?;
            }
        }
        Clause::Union(u) => {
            let mut inner_vars: HashMap<String, String> = HashMap::new();
            validate_query(&u.query, graph, &mut inner_vars)?;
        }
        Clause::Return(_) | Clause::OrderBy(_) | Clause::Unwind(_) => {}
        Clause::Create(_)
        | Clause::Set(_)
        | Clause::Delete(_)
        | Clause::Remove(_)
        | Clause::Merge(_)
        | Clause::Call(_) => {}
        _ => {}
    }
    Ok(())
}

fn validate_pattern(
    pattern: &Pattern,
    graph: &DirGraph,
    var_types: &mut HashMap<String, String>,
) -> Result<(), SchemaError> {
    for element in &pattern.elements {
        if let PatternElement::Node(np) = element {
            if let Some(ref node_type) = np.node_type {
                if let Some(ref var) = np.variable {
                    var_types.insert(var.clone(), node_type.clone());
                }
                if let Some(ref props) = np.properties {
                    for prop_name in props.keys() {
                        validate_property(node_type, prop_name, graph)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Descend the predicate to find nested `EXISTS { MATCH ... }` patterns
/// so their pattern-literal properties get validated too. Does not check
/// expression-level property accesses or label checks — those are
/// delivered as Phase 3 diagnostics instead.
fn walk_predicate_for_nested_patterns(
    predicate: &Predicate,
    graph: &DirGraph,
    var_types: &HashMap<String, String>,
) -> Result<(), SchemaError> {
    match predicate {
        Predicate::And(a, b) | Predicate::Or(a, b) | Predicate::Xor(a, b) => {
            walk_predicate_for_nested_patterns(a, graph, var_types)?;
            walk_predicate_for_nested_patterns(b, graph, var_types)?;
        }
        Predicate::Not(p) => walk_predicate_for_nested_patterns(p, graph, var_types)?,
        Predicate::Exists {
            patterns,
            where_clause,
        } => {
            let mut inner_vars = var_types.clone();
            for p in patterns {
                validate_pattern(p, graph, &mut inner_vars)?;
            }
            if let Some(w) = where_clause {
                walk_predicate_for_nested_patterns(w, graph, &inner_vars)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_property(node_type: &str, property: &str, graph: &DirGraph) -> Result<(), SchemaError> {
    if BUILTIN_FIELDS.contains(&property) {
        return Ok(());
    }
    let Some(type_props) = graph.node_type_metadata.get(node_type) else {
        // No declared property metadata for this type — skip rather than
        // false-positive on dynamically-typed or under-declared graphs.
        return Ok(());
    };
    if type_props.is_empty() || type_props.contains_key(property) {
        return Ok(());
    }
    let candidates: Vec<&str> = type_props.keys().map(|s| s.as_str()).collect();
    let hint = did_you_mean(property, &candidates);
    let mut sorted = candidates;
    sorted.sort();
    Err(SchemaError {
        kind: SchemaErrorKind::UnknownProperty,
        message: format!(
            "Unknown property '{}' on {}.{}\n  Valid properties: {}",
            property,
            node_type,
            hint,
            sorted.join(", ")
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::languages::cypher::parser::parse_cypher;

    fn graph_with_schema() -> DirGraph {
        let mut g = DirGraph::new();
        let mut person_props = HashMap::new();
        person_props.insert("age".to_string(), "int".to_string());
        person_props.insert("email".to_string(), "string".to_string());
        g.upsert_node_type_metadata("Person", person_props);
        let mut paper_props = HashMap::new();
        paper_props.insert("year".to_string(), "int".to_string());
        g.upsert_node_type_metadata("Paper", paper_props);
        g.upsert_connection_type_metadata("KNOWS", "Person", "Person", HashMap::new());
        g.upsert_connection_type_metadata("AUTHORED", "Person", "Paper", HashMap::new());
        g
    }

    #[test]
    fn validates_known_node_type() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person) RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn tolerates_unknown_node_type() {
        // `MATCH (n:Nonexistent) RETURN n` is valid Cypher — returns 0
        // rows. Phase 3 will surface a "did you mean?" hint via
        // diagnostics rather than rejecting.
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:person) RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn tolerates_unknown_connection_type() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (a:Person)-[:nonexistent]->(b:Person) RETURN a").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn rejects_unknown_property_in_pattern_literal() {
        // The `{agee: 30}` form is unambiguous — validate.
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person {agee: 30}) RETURN n").unwrap();
        let err = validate_schema(&q, &g).unwrap_err();
        assert_eq!(err.kind, SchemaErrorKind::UnknownProperty);
        assert!(err.message.contains("age"), "got: {}", err.message);
    }

    #[test]
    fn tolerates_unknown_property_in_where_expression() {
        // `n.prop` accesses in WHERE may reference virtual columns
        // (timeseries sub-nodes, computed aliases). Do not flag.
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person) WHERE n.birth_yr = 1900 RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn tolerates_unknown_property_in_return_expression() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person) RETURN n.agee").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn allows_builtin_fields_in_pattern_literal() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person {id: 1}) RETURN n.title").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn skips_validation_on_empty_schema() {
        let g = DirGraph::new();
        let q = parse_cypher("MATCH (n:Anything) RETURN n.whatever").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn validates_untyped_patterns_permissively() {
        // Untyped patterns (no :Label) are common and legal — we only
        // validate when the user has declared intent via a label.
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n) WHERE n.whatever = 1 RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn rejects_unknown_property_on_multi_hop_pattern_literal() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person {agee: 30}) RETURN a").unwrap();
        let err = validate_schema(&q, &g).unwrap_err();
        assert_eq!(err.kind, SchemaErrorKind::UnknownProperty);
    }

    #[test]
    fn allows_order_by_and_return_of_known_properties() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person) RETURN n.age ORDER BY n.email").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn tolerates_unknown_label_in_where_label_check() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n) WHERE n:person RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn validates_property_inside_exists_nested_pattern() {
        // Pattern literals inside EXISTS { MATCH ... } should be
        // validated too.
        let g = graph_with_schema();
        let q = parse_cypher(
            "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->(b:Person {agee: 1}) } RETURN a",
        )
        .unwrap();
        let err = validate_schema(&q, &g).unwrap_err();
        assert_eq!(err.kind, SchemaErrorKind::UnknownProperty);
    }
}
