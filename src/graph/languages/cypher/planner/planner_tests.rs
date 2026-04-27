use super::*;
use crate::graph::core::pattern_matching::PropertyMatcher;
use crate::graph::languages::cypher::parser::parse_cypher;

#[test]
fn test_predicate_pushdown_simple() {
    let mut query = parse_cypher("MATCH (n:Person) WHERE n.age = 30 RETURN n").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // WHERE is kept as a safety net even when all predicates are pushed
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN
    assert!(matches!(&query.clauses[0], Clause::Match(_)));
    assert!(matches!(&query.clauses[2], Clause::Return(_)));

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
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Both n.age = 30 and n.score > 100 should be pushed into MATCH
    // WHERE is kept as a safety net
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

    if let Clause::Match(m) = &query.clauses[0] {
        if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
            let props = np.properties.as_ref().unwrap();
            assert!(matches!(
                props.get("age"),
                Some(PropertyMatcher::Equals(Value::Int64(30)))
            ));
            assert!(matches!(
                props.get("score"),
                Some(PropertyMatcher::GreaterThan(Value::Int64(100)))
            ));
        }
    }
}

#[test]
fn test_comparison_pushdown() {
    let mut query = parse_cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Comparison should be pushed into MATCH, WHERE kept as safety net
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

    if let Clause::Match(m) = &query.clauses[0] {
        if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
            let props = np.properties.as_ref().unwrap();
            assert!(matches!(
                props.get("age"),
                Some(PropertyMatcher::GreaterThan(Value::Int64(30)))
            ));
        }
    }
}

#[test]
fn test_no_pushdown_for_not_equals() {
    let mut query = parse_cypher("MATCH (n:Person) WHERE n.age <> 30 RETURN n").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // NotEquals should NOT be pushed - WHERE should remain
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN
}

#[test]
fn test_predicate_pushdown_parameter() {
    let mut query = parse_cypher("MATCH (n:Person) WHERE n.name = $name RETURN n").unwrap();

    let graph = DirGraph::new();
    let mut params = HashMap::new();
    params.insert("name".to_string(), Value::String("Alice".to_string()));
    optimize(&mut query, &graph, &params);

    // Parameter resolved and pushed; WHERE kept as safety net
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

    // The MATCH pattern should now have {name: 'Alice'} as a property
    if let Clause::Match(m) = &query.clauses[0] {
        if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
            assert!(np.properties.is_some());
            let props = np.properties.as_ref().unwrap();
            assert!(props.contains_key("name"));
            assert!(matches!(
                props.get("name"),
                Some(PropertyMatcher::Equals(Value::String(s))) if s == "Alice"
            ));
        } else {
            panic!("Expected node pattern");
        }
    }
}

#[test]
fn test_predicate_pushdown_parameter_partial() {
    let mut query =
        parse_cypher("MATCH (n:Person) WHERE n.name = $name AND n.age > $min_age RETURN n")
            .unwrap();

    let graph = DirGraph::new();
    let mut params = HashMap::new();
    params.insert("name".to_string(), Value::String("Alice".to_string()));
    params.insert("min_age".to_string(), Value::Int64(25));
    optimize(&mut query, &graph, &params);

    // Both should be pushed: n.name = $name (equality) and n.age > $min_age (comparison)
    // WHERE kept as safety net
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

    if let Clause::Match(m) = &query.clauses[0] {
        if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
            let props = np.properties.as_ref().unwrap();
            assert!(matches!(
                props.get("name"),
                Some(PropertyMatcher::Equals(Value::String(s))) if s == "Alice"
            ));
            assert!(matches!(
                props.get("age"),
                Some(PropertyMatcher::GreaterThan(Value::Int64(25)))
            ));
        }
    }
}

#[test]
fn test_comparison_range_merge() {
    let mut query =
        parse_cypher("MATCH (n:Paper) WHERE n.year >= 2015 AND n.year <= 2022 RETURN n").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Both comparisons should be merged into a Range matcher; WHERE kept
    assert_eq!(query.clauses.len(), 3); // MATCH + WHERE + RETURN

    if let Clause::Match(m) = &query.clauses[0] {
        if let PatternElement::Node(np) = &m.patterns[0].elements[0] {
            let props = np.properties.as_ref().unwrap();
            assert!(matches!(
                props.get("year"),
                Some(PropertyMatcher::Range {
                    lower: Value::Int64(2015),
                    lower_inclusive: true,
                    upper: Value::Int64(2022),
                    upper_inclusive: true,
                })
            ));
        }
    }
}

#[test]
fn test_correlated_nodeprop_pushdown() {
    // The classic shape from sodir-prospect build:
    //   MATCH (a:A) MATCH (b:B) WHERE b.x = a.y
    // should push { x: EqualsNodeProp { var: "a", prop: "y" } } onto B.
    let mut query =
        parse_cypher("MATCH (a:A) MATCH (b:B) WHERE b.x = a.y RETURN a.id, b.id").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Locate the second MATCH (matching on B)
    let b_match = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .find(|m| {
            matches!(
                &m.patterns[0].elements[0],
                PatternElement::Node(np) if np.node_type.as_deref() == Some("B")
            )
        })
        .expect("expected second MATCH on B");

    if let PatternElement::Node(np) = &b_match.patterns[0].elements[0] {
        let props = np.properties.as_ref().expect("expected props on b");
        match props.get("x") {
            Some(PropertyMatcher::EqualsNodeProp { var, prop }) => {
                assert_eq!(var, "a");
                assert_eq!(prop, "y");
            }
            other => panic!("expected EqualsNodeProp on b.x, got {:?}", other),
        }
    }
}

#[test]
fn test_correlated_nodeprop_reversed_sides() {
    // Reversed: a.y = b.x (the cur_var b appears on the right).
    let mut query =
        parse_cypher("MATCH (a:A) MATCH (b:B) WHERE a.y = b.x RETURN a.id, b.id").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let b_match = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .find(|m| {
            matches!(
                &m.patterns[0].elements[0],
                PatternElement::Node(np) if np.node_type.as_deref() == Some("B")
            )
        })
        .unwrap();

    if let PatternElement::Node(np) = &b_match.patterns[0].elements[0] {
        let props = np.properties.as_ref().unwrap();
        assert!(matches!(
            props.get("x"),
            Some(PropertyMatcher::EqualsNodeProp { var, prop })
                if var == "a" && prop == "y"
        ));
    }
}

#[test]
fn test_scalar_var_pushdown_from_unwind() {
    // WHERE s.title = fname where fname comes from UNWIND should become
    // an EqualsVar matcher on the pattern.
    let mut query =
        parse_cypher("UNWIND ['x','y'] AS fname MATCH (s:Strat) WHERE s.title = fname RETURN s.id")
            .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let s_match = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .next()
        .unwrap();

    if let PatternElement::Node(np) = &s_match.patterns[0].elements[0] {
        let props = np.properties.as_ref().unwrap();
        assert!(matches!(
            props.get("title"),
            Some(PropertyMatcher::EqualsVar(n)) if n == "fname"
        ));
    }
}

#[test]
fn test_no_pushdown_when_both_vars_in_same_match() {
    // Within the same MATCH, a.y = b.x is handled by the pattern
    // executor's shared-var join. We must NOT rewrite it as an
    // EqualsNodeProp (which assumes prior-bound node).
    let mut query = parse_cypher("MATCH (a:A), (b:B) WHERE a.y = b.x RETURN a.id, b.id").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    for clause in &query.clauses {
        if let Clause::Match(m) = clause {
            for pat in &m.patterns {
                for el in &pat.elements {
                    if let PatternElement::Node(np) = el {
                        if let Some(props) = &np.properties {
                            for m in props.values() {
                                assert!(
                                    !matches!(m, PropertyMatcher::EqualsNodeProp { .. }),
                                    "same-MATCH correlated equality must not be rewritten"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
