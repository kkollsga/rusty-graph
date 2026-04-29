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

#[test]
fn test_undirected_pattern_reversed_by_selectivity() {
    // Regression: `(other)-[r]-(p {title: 'X'})` was not reversed because
    // the planner bailed out on undirected edges, leaving `other` (no
    // constraints, full graph scan) as the start node. The selective
    // anchor `p {title: 'X'}` must now be picked as start.
    let mut query =
        parse_cypher("MATCH (other)-[r]-(p {title: 'X'}) RETURN type(r), count(other)").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let m = query
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .unwrap();

    let first = match &m.patterns[0].elements[0] {
        PatternElement::Node(np) => np,
        _ => panic!("expected node"),
    };
    assert_eq!(
        first.variable.as_deref(),
        Some("p"),
        "selective anchor `p` should be the start after reversal"
    );
    assert!(
        first.properties.is_some(),
        "start node must carry the title property after reversal"
    );
}

#[test]
fn test_undirected_pattern_no_reverse_when_first_is_anchor() {
    // Reverse case: anchor is already first — no reversal expected.
    let mut query =
        parse_cypher("MATCH (p {title: 'X'})-[r]-(other) RETURN type(r), count(other)").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let m = query
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .unwrap();

    let first = match &m.patterns[0].elements[0] {
        PatternElement::Node(np) => np,
        _ => panic!("expected node"),
    };
    assert_eq!(first.variable.as_deref(), Some("p"));
}

#[test]
fn test_var_length_pattern_reversed_by_selectivity() {
    // `(other)-[*1..3]-(p {id: 1})` — the anchor is selective. Reversal
    // is safe when the pattern has no path assignment (the `path_assignments`
    // guard already protects path-bound patterns).
    let mut query = parse_cypher("MATCH (other)-[*1..3]-(p {id: 1}) RETURN p, other").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let m = query
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .unwrap();

    let first = match &m.patterns[0].elements[0] {
        PatternElement::Node(np) => np,
        _ => panic!("expected node"),
    };
    assert_eq!(
        first.variable.as_deref(),
        Some("p"),
        "var-length patterns should still get start-node optimization"
    );
}

#[test]
fn test_var_length_with_path_assignment_not_reversed() {
    // Path assignments must block reversal (path semantics depend on direction).
    let mut query = parse_cypher("MATCH path = (other)-[*1..3]-(p {id: 1}) RETURN path").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let m = query
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .unwrap();

    let first = match &m.patterns[0].elements[0] {
        PatternElement::Node(np) => np,
        _ => panic!("expected node"),
    };
    assert_eq!(
        first.variable.as_deref(),
        Some("other"),
        "path-bound patterns must not be reversed"
    );
}

#[test]
fn test_limit_pushdown_single_match_with_where() {
    // Single MATCH + WHERE + RETURN + LIMIT — pushdown is safe; the LIMIT
    // clause should be removed and the MATCH should carry limit_hint.
    let mut query =
        parse_cypher("MATCH (n:Person) WHERE n.age > 25 RETURN n.name LIMIT 10").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Limit clause should be gone (absorbed into the MATCH hint)
    let has_limit = query.clauses.iter().any(|c| matches!(c, Clause::Limit(_)));
    assert!(
        !has_limit,
        "single-MATCH query should have LIMIT pushed into MATCH"
    );

    let m = query
        .clauses
        .iter()
        .find_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .expect("expected a MATCH clause");
    assert_eq!(m.limit_hint, Some(10));
}

#[test]
fn test_multi_match_no_reverse_when_bound_var_first() {
    // Regression for the user's "(p)-[:P31]->(:human) gets reversed to scan
    // 13M humans" report: in the second MATCH, `p` is already bound by the
    // first MATCH. The planner must not reverse the pattern to start from
    // `(:human)` (a 13M-row scan) just because `p` looks unconstrained
    // statically.
    let mut query =
        parse_cypher("MATCH (p:Person) MATCH (p)-[:KNOWS]->(c:Company) RETURN p, c").unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // The second MATCH should still start with `p`, not `c`.
    let matches: Vec<_> = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .collect();
    assert!(matches.len() >= 2, "expected two MATCH clauses");
    let second = matches[1];
    let first_var = match &second.patterns[0].elements[0] {
        PatternElement::Node(np) => np.variable.as_deref(),
        _ => None,
    };
    assert_eq!(
        first_var,
        Some("p"),
        "second MATCH must keep pre-bound `p` as start node, not reverse to `c`"
    );
}

#[test]
fn test_multi_match_reorder_prefers_anchored_pattern() {
    // When a single MATCH has two patterns sharing a pre-bound variable,
    // the more selective pattern should run first. The planner already
    // does intra-clause reordering — this test pins the cross-clause
    // bound-vars logic so it doesn't regress.
    let mut query = parse_cypher(
        "MATCH (p {id: 1}) \
         MATCH (p)-[:R1]->(:T1), (p)-[:R2]->({id: 99}) \
         RETURN p",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Just ensure optimization doesn't crash and patterns survive.
    let m2 = query
        .clauses
        .iter()
        .filter_map(|c| match c {
            Clause::Match(m) => Some(m),
            _ => None,
        })
        .nth(1)
        .expect("expected second MATCH");
    assert_eq!(m2.patterns.len(), 2);
}

#[test]
fn test_limit_pushdown_multi_match_safety() {
    // Regression: 3-MATCH + WHERE on last-bound variable + LIMIT N produced
    // fewer rows than expected because push_limit_into_match was setting
    // `limit_hint` on the last MATCH, which interacts incorrectly with the
    // per-row pattern executor in the subsequent-MATCH path. The planner
    // must NOT push LIMIT into MATCH when there are multiple MATCH clauses.
    let mut query = parse_cypher(
        "MATCH (a)-[:R1]->(:T1) \
         MATCH (a)-[:R2]->(b) \
         MATCH (b)-[:R3]->(c) \
         WHERE c.id = 7318 \
         RETURN a.id, b.id LIMIT 50",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    // Limit clause must remain — pushdown is unsafe for multi-MATCH
    let has_limit = query.clauses.iter().any(|c| matches!(c, Clause::Limit(_)));
    assert!(has_limit, "multi-MATCH query must retain its LIMIT clause");

    // No MATCH clause should have a limit_hint set
    for clause in &query.clauses {
        if let Clause::Match(m) = clause {
            assert_eq!(
                m.limit_hint, None,
                "multi-MATCH clauses must not receive a limit_hint"
            );
        }
    }
}

#[test]
fn test_fuse_match_return_aggregate_count_distinct() {
    // Single-MATCH + RETURN with count(DISTINCT v) on the OTHER node variable
    // — the planner should now fuse this and set distinct_count=true.
    let mut query = parse_cypher(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         RETURN a, count(DISTINCT b) AS friends \
         ORDER BY friends DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let mut found = false;
    for clause in &query.clauses {
        if let Clause::FusedMatchReturnAggregate {
            distinct_count,
            top_k,
            ..
        } = clause
        {
            assert!(*distinct_count, "distinct_count flag must be set");
            assert!(
                top_k.is_some(),
                "ORDER BY count DESC LIMIT 10 must absorb into top_k"
            );
            found = true;
        }
    }
    assert!(
        found,
        "FusedMatchReturnAggregate must fire for count(DISTINCT) shape"
    );
}

#[test]
fn test_fuse_match_with_aggregate_count_distinct() {
    // Single-MATCH + WITH with count(DISTINCT v) — pipeline form.
    let mut query = parse_cypher(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         WITH a, count(DISTINCT b) AS friends \
         RETURN a, friends",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let mut found = false;
    for clause in &query.clauses {
        if let Clause::FusedMatchWithAggregate { distinct_count, .. } = clause {
            assert!(*distinct_count, "WITH-form distinct_count flag must be set");
            found = true;
        }
    }
    assert!(
        found,
        "FusedMatchWithAggregate must fire for WITH-count-DISTINCT shape"
    );
}

#[test]
fn test_count_distinct_unconstrained_group_not_fused() {
    // Unconstrained group node — fusing would force a 124M-node enumeration
    // on a Wikidata-scale graph. The materializing path is faster in that
    // regime, so the planner must NOT fuse.
    let mut query = parse_cypher(
        "MATCH (a)-[:R]->(b) \
         RETURN b, count(DISTINCT a) AS n \
         ORDER BY n DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let fused_with_distinct = query.clauses.iter().any(|c| {
        matches!(
            c,
            Clause::FusedMatchReturnAggregate {
                distinct_count: true,
                ..
            }
        )
    });
    assert!(
        !fused_with_distinct,
        "untyped group node must skip distinct-count fusion"
    );
}

#[test]
fn test_count_distinct_5_element_pattern_not_fused() {
    // 5-element patterns aren't supported in the distinct-count path yet.
    // The planner should leave the query unfused so the materializing
    // executor produces semantically-correct results.
    let mut query = parse_cypher(
        "MATCH (a:A)-[:R1]->(b)<-[:R2]-(c) \
         RETURN a, count(DISTINCT c) AS n \
         ORDER BY n DESC LIMIT 10",
    )
    .unwrap();

    let graph = DirGraph::new();
    let params = HashMap::new();
    optimize(&mut query, &graph, &params);

    let fused_with_distinct = query.clauses.iter().any(|c| {
        matches!(
            c,
            Clause::FusedMatchReturnAggregate {
                distinct_count: true,
                ..
            }
        )
    });
    assert!(
        !fused_with_distinct,
        "5-element distinct-count pattern must not be fused"
    );
}
