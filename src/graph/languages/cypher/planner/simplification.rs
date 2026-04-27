//! Rewriting simplifications — fold OR→IN, push LIMIT/DISTINCT into MATCH,
//! rewrite text_score.

use super::super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::schema::DirGraph;
use std::collections::HashMap;

pub(super) fn fold_or_to_in(query: &mut CypherQuery) {
    for clause in &mut query.clauses {
        if let Clause::Where(ref mut w) = clause {
            w.predicate = fold_or_to_in_pred(&w.predicate);
        }
    }
}

/// Recursively fold OR chains of same-property equalities into IN predicates.
pub(super) fn fold_or_to_in_pred(pred: &Predicate) -> Predicate {
    match pred {
        Predicate::Or(_, _) => {
            // Collect all OR-chained equality comparisons
            let mut equalities: Vec<(String, String, Expression)> = Vec::new();
            let mut other_preds: Vec<Predicate> = Vec::new();
            collect_or_equalities(pred, &mut equalities, &mut other_preds);

            // Group equalities by (variable, property)
            let mut groups: std::collections::HashMap<(String, String), Vec<Expression>> =
                std::collections::HashMap::new();
            for (var, prop, val_expr) in equalities {
                groups.entry((var, prop)).or_default().push(val_expr);
            }

            // Build result predicates
            let mut result_preds: Vec<Predicate> = Vec::new();

            // Convert groups with 2+ equalities into IN predicates
            for ((var, prop), values) in groups {
                if values.len() >= 2 {
                    result_preds.push(Predicate::In {
                        expr: Expression::PropertyAccess {
                            variable: var,
                            property: prop,
                        },
                        list: values,
                    });
                } else {
                    // Single equality — keep as comparison
                    result_preds.push(Predicate::Comparison {
                        left: Expression::PropertyAccess {
                            variable: var,
                            property: prop,
                        },
                        operator: ComparisonOp::Equals,
                        right: values.into_iter().next().unwrap(),
                    });
                }
            }

            // Add back non-equality predicates (recursively folded)
            for p in other_preds {
                result_preds.push(fold_or_to_in_pred(&p));
            }

            // Combine with OR
            if result_preds.len() == 1 {
                result_preds.pop().unwrap()
            } else {
                let mut combined = result_preds.pop().unwrap();
                for p in result_preds.into_iter().rev() {
                    combined = Predicate::Or(Box::new(p), Box::new(combined));
                }
                combined
            }
        }
        Predicate::And(l, r) => Predicate::And(
            Box::new(fold_or_to_in_pred(l)),
            Box::new(fold_or_to_in_pred(r)),
        ),
        Predicate::Not(inner) => Predicate::Not(Box::new(fold_or_to_in_pred(inner))),
        other => other.clone(),
    }
}

/// Collect equalities from an OR chain. Non-equality predicates go to `others`.
pub(super) fn collect_or_equalities(
    pred: &Predicate,
    equalities: &mut Vec<(String, String, Expression)>,
    others: &mut Vec<Predicate>,
) {
    match pred {
        Predicate::Or(left, right) => {
            collect_or_equalities(left, equalities, others);
            collect_or_equalities(right, equalities, others);
        }
        Predicate::Comparison {
            left,
            operator: ComparisonOp::Equals,
            right,
        } => {
            if let Expression::PropertyAccess { variable, property } = left {
                if matches!(right, Expression::Literal(_) | Expression::Parameter(_)) {
                    equalities.push((variable.clone(), property.clone(), right.clone()));
                    return;
                }
            }
            if let Expression::PropertyAccess { variable, property } = right {
                if matches!(left, Expression::Literal(_) | Expression::Parameter(_)) {
                    equalities.push((variable.clone(), property.clone(), left.clone()));
                    return;
                }
            }
            others.push(pred.clone());
        }
        other => {
            others.push(other.clone());
        }
    }
}

/// Example: MATCH (n:Person) WHERE n.age = 30
/// Becomes: MATCH (n:Person {age: 30}) (WHERE removed if fully consumed)
///
/// Also handles parameterized equalities:
/// MATCH (n:Person) WHERE n.age = $min_age  (with params = {min_age: 30})
/// Becomes: MATCH (n:Person {age: 30})
pub(super) fn push_limit_into_match(query: &mut CypherQuery, _graph: &DirGraph) {
    if query.clauses.len() < 3 {
        return;
    }
    let mut i = 0;
    while i + 2 < query.clauses.len() {
        // Look for MATCH → RETURN → LIMIT  or  MATCH → WHERE → RETURN → LIMIT
        let (has_where, return_offset, limit_offset) = if i + 3 < query.clauses.len()
            && matches!(&query.clauses[i], Clause::Match(_))
            && matches!(&query.clauses[i + 1], Clause::Where(_))
            && matches!(&query.clauses[i + 2], Clause::Return(_))
            && matches!(&query.clauses[i + 3], Clause::Limit(_))
        {
            (true, i + 2, i + 3)
        } else if matches!(
            (&query.clauses[i], &query.clauses[i + 1]),
            (Clause::Match(_), Clause::Return(_))
        ) && i + 2 < query.clauses.len()
            && matches!(&query.clauses[i + 2], Clause::Limit(_))
        {
            (false, i + 1, i + 2)
        } else {
            i += 1;
            continue;
        };

        // Safety check: RETURN must have no aggregation, no DISTINCT, no window functions
        let safe = if let Clause::Return(r) = &query.clauses[return_offset] {
            !r.distinct
                && !r
                    .items
                    .iter()
                    .any(|item| super::super::ast::is_aggregate_expression(&item.expression))
                && !r
                    .items
                    .iter()
                    .any(|item| super::super::ast::is_window_expression(&item.expression))
        } else {
            false
        };
        if !safe {
            i += 1;
            continue;
        }

        // Extract LIMIT value — must be a literal positive integer
        let limit_val = if let Clause::Limit(l) = &query.clauses[limit_offset] {
            match &l.count {
                Expression::Literal(Value::Int64(n)) if *n > 0 => Some(*n as usize),
                _ => None,
            }
        } else {
            None
        };
        let Some(limit) = limit_val else {
            i += 1;
            continue;
        };

        // All checks passed: push limit hint into MATCH.
        // The executor fuses WHERE into MATCH (inline evaluation during expansion),
        // so the hint can be exact in both cases. LIMIT clause is removed since
        // the executor stops after finding exactly `limit` matching rows.
        if has_where {
            if let Clause::Match(ref mut m) = query.clauses[i] {
                m.limit_hint = Some(limit);
            }
            query.clauses.remove(limit_offset); // Safe: executor enforces exact limit
        } else {
            if let Clause::Match(ref mut m) = query.clauses[i] {
                m.limit_hint = Some(limit);
            }
            query.clauses.remove(limit_offset); // Remove LIMIT — hint is exact
        }
    }
}

/// Push DISTINCT hint into MATCH when RETURN DISTINCT references a single node variable.
///
/// When all RETURN DISTINCT expressions depend on a single node variable
/// (e.g., `RETURN DISTINCT c2.id` or `RETURN DISTINCT c2.id, c2.name`),
/// the executor can pre-deduplicate pattern matches by that variable's NodeIndex
/// during the MATCH phase, avoiding creation of duplicate ResultRows.
///
/// Detects patterns: MATCH → [WHERE] → RETURN DISTINCT
pub(super) fn push_distinct_into_match(query: &mut CypherQuery) {
    // Find MATCH + RETURN DISTINCT (with optional WHERE in between)
    for i in 0..query.clauses.len() {
        let match_idx = match &query.clauses[i] {
            Clause::Match(_) => i,
            _ => continue,
        };

        // Find the RETURN clause (skip optional WHERE)
        let return_idx = if match_idx + 1 < query.clauses.len() {
            match &query.clauses[match_idx + 1] {
                Clause::Return(_) => match_idx + 1,
                Clause::Where(_) if match_idx + 2 < query.clauses.len() => {
                    if matches!(&query.clauses[match_idx + 2], Clause::Return(_)) {
                        match_idx + 2
                    } else {
                        continue;
                    }
                }
                _ => continue,
            }
        } else {
            continue;
        };

        // Check: RETURN must be DISTINCT, no aggregation
        let distinct_var = if let Clause::Return(r) = &query.clauses[return_idx] {
            if !r.distinct {
                continue;
            }
            if r.items
                .iter()
                .any(|item| super::super::ast::is_aggregate_expression(&item.expression))
            {
                continue;
            }
            // All return items must reference a single node variable
            let mut var: Option<&str> = None;
            let mut all_same = true;
            for item in &r.items {
                let v = match &item.expression {
                    Expression::PropertyAccess { variable, .. } => variable.as_str(),
                    Expression::Variable(v) => v.as_str(),
                    _ => {
                        all_same = false;
                        break;
                    }
                };
                match var {
                    None => var = Some(v),
                    Some(prev) if prev == v => {}
                    _ => {
                        all_same = false;
                        break;
                    }
                }
            }
            if all_same {
                var.map(String::from)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(dv) = distinct_var {
            // Verify the variable is a node variable in the MATCH pattern
            if let Clause::Match(ref mc) = &query.clauses[match_idx] {
                let is_node_var = mc.patterns.iter().any(|p| {
                    p.elements.iter().any(|e| {
                        if let crate::graph::core::pattern_matching::PatternElement::Node(np) = e {
                            np.variable.as_deref() == Some(dv.as_str())
                        } else {
                            false
                        }
                    })
                });
                if !is_node_var {
                    continue;
                }
            }
            // Set the hint
            if let Clause::Match(ref mut mc) = query.clauses[match_idx] {
                mc.distinct_node_hint = Some(dv);
            }
        }
    }
}

// ============================================================================
// text_score → vector_score AST Rewrite
// ============================================================================

/// Collected texts that the caller must embed before execution.
///
/// Each entry is `(param_name, query_text)` — the caller embeds the text and
/// inserts the resulting vector into the params map under `param_name`.
pub struct TextScoreRewrite {
    pub texts_to_embed: Vec<(String, String)>,
}

/// Walk the AST and rewrite all `text_score(node, col, query_text)` calls
/// to `vector_score(node, col_emb, $__ts_N)`.
///
/// The text argument can be a string literal or a `$parameter` (resolved from
/// `params`).  Returns the collected texts so the caller can embed them and
/// inject the resulting vectors into the params map before optimization.
pub fn rewrite_text_score(
    query: &mut CypherQuery,
    params: &HashMap<String, Value>,
) -> Result<TextScoreRewrite, String> {
    let mut collector = TextScoreCollector {
        counter: 0,
        texts_to_embed: Vec::new(),
    };

    for clause in &mut query.clauses {
        match clause {
            Clause::Return(r) => {
                for item in &mut r.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
            }
            Clause::Where(w) => {
                collector.rewrite_pred(&mut w.predicate, params)?;
            }
            Clause::With(w) => {
                for item in &mut w.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
                if let Some(ref mut wh) = w.where_clause {
                    collector.rewrite_pred(&mut wh.predicate, params)?;
                }
            }
            Clause::OrderBy(o) => {
                for item in &mut o.items {
                    collector.rewrite_expr(&mut item.expression, params)?;
                }
            }
            Clause::Unwind(u) => {
                collector.rewrite_expr(&mut u.expression, params)?;
            }
            Clause::Delete(d) => {
                for expr in &mut d.expressions {
                    collector.rewrite_expr(expr, params)?;
                }
            }
            Clause::Set(s) => {
                for item in &mut s.items {
                    if let SetItem::Property {
                        ref mut expression, ..
                    } = item
                    {
                        collector.rewrite_expr(expression, params)?;
                    }
                }
            }
            Clause::Create(c) => {
                for pattern in &mut c.patterns {
                    for element in &mut pattern.elements {
                        match element {
                            CreateElement::Node(n) => {
                                for (_, expr) in &mut n.properties {
                                    collector.rewrite_expr(expr, params)?;
                                }
                            }
                            CreateElement::Edge(e) => {
                                for (_, expr) in &mut e.properties {
                                    collector.rewrite_expr(expr, params)?;
                                }
                            }
                        }
                    }
                }
            }
            Clause::Merge(m) => {
                for element in &mut m.pattern.elements {
                    match element {
                        CreateElement::Node(n) => {
                            for (_, expr) in &mut n.properties {
                                collector.rewrite_expr(expr, params)?;
                            }
                        }
                        CreateElement::Edge(e) => {
                            for (_, expr) in &mut e.properties {
                                collector.rewrite_expr(expr, params)?;
                            }
                        }
                    }
                }
                if let Some(ref mut items) = m.on_create {
                    for item in items {
                        if let SetItem::Property {
                            ref mut expression, ..
                        } = item
                        {
                            collector.rewrite_expr(expression, params)?;
                        }
                    }
                }
                if let Some(ref mut items) = m.on_match {
                    for item in items {
                        if let SetItem::Property {
                            ref mut expression, ..
                        } = item
                        {
                            collector.rewrite_expr(expression, params)?;
                        }
                    }
                }
            }
            Clause::Skip(s) => {
                collector.rewrite_expr(&mut s.count, params)?;
            }
            Clause::Limit(l) => {
                collector.rewrite_expr(&mut l.count, params)?;
            }
            // Match/OptionalMatch: patterns only, no function call expressions
            // Remove: no expressions
            // Fused clauses: don't exist yet (created by optimize, which runs after rewrite)
            _ => {}
        }
    }

    Ok(TextScoreRewrite {
        texts_to_embed: collector.texts_to_embed,
    })
}

struct TextScoreCollector {
    counter: usize,
    texts_to_embed: Vec<(String, String)>,
}

impl TextScoreCollector {
    /// Rewrite an expression in-place.  Turns `text_score(...)` into `vector_score(...)`.
    fn rewrite_expr(
        &mut self,
        expr: &mut Expression,
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        match expr {
            Expression::FunctionCall { name, args, .. } if name == "text_score" => {
                if args.len() != 3 && args.len() != 4 {
                    return Err(
                        "text_score() requires 3 arguments: (node, text_column, query_text) \
                         with optional 4th metric argument"
                            .into(),
                    );
                }

                // arg[1]: text column — must be a string literal
                let col_name =
                    match &args[1] {
                        Expression::Literal(Value::String(s)) => s.clone(),
                        _ => return Err(
                            "text_score(): second argument must be a string literal column name"
                                .into(),
                        ),
                    };

                // arg[2]: query text — string literal or $param
                let query_text = match &args[2] {
                    Expression::Literal(Value::String(s)) => s.clone(),
                    Expression::Parameter(param_name) => match params.get(param_name.as_str()) {
                        Some(Value::String(s)) => s.clone(),
                        Some(_) => {
                            return Err(format!(
                                "text_score(): parameter ${} must be a string",
                                param_name
                            ))
                        }
                        None => {
                            return Err(format!(
                                "text_score(): parameter ${} not found",
                                param_name
                            ))
                        }
                    },
                    _ => {
                        return Err(
                            "text_score(): third argument must be a string literal or $parameter"
                                .into(),
                        )
                    }
                };

                // Deduplicate: reuse param if same query text already collected
                let param_name = if let Some((existing, _)) =
                    self.texts_to_embed.iter().find(|(_, t)| t == &query_text)
                {
                    existing.clone()
                } else {
                    let pname = format!("__ts_{}", self.counter);
                    self.counter += 1;
                    self.texts_to_embed.push((pname.clone(), query_text));
                    pname
                };

                // Rewrite: text_score(n, 'summary', ...) → vector_score(n, 'summary_emb', $__ts_N)
                *name = "vector_score".to_string();
                args[1] = Expression::Literal(Value::String(format!("{}_emb", col_name)));
                args[2] = Expression::Parameter(param_name);

                Ok(())
            }
            Expression::FunctionCall { args, .. } => {
                for arg in args.iter_mut() {
                    self.rewrite_expr(arg, params)?;
                }
                Ok(())
            }
            Expression::Add(l, r)
            | Expression::Subtract(l, r)
            | Expression::Multiply(l, r)
            | Expression::Divide(l, r)
            | Expression::Modulo(l, r)
            | Expression::Concat(l, r) => {
                self.rewrite_expr(l, params)?;
                self.rewrite_expr(r, params)?;
                Ok(())
            }
            Expression::Negate(inner) => self.rewrite_expr(inner, params),
            Expression::ListLiteral(items) => {
                for item in items.iter_mut() {
                    self.rewrite_expr(item, params)?;
                }
                Ok(())
            }
            Expression::Case {
                operand,
                when_clauses,
                else_expr,
            } => {
                if let Some(op) = operand {
                    self.rewrite_expr(op, params)?;
                }
                for (cond, result) in when_clauses.iter_mut() {
                    match cond {
                        CaseCondition::Expression(e) => self.rewrite_expr(e, params)?,
                        CaseCondition::Predicate(p) => self.rewrite_pred(p, params)?,
                    }
                    self.rewrite_expr(result, params)?;
                }
                if let Some(el) = else_expr {
                    self.rewrite_expr(el, params)?;
                }
                Ok(())
            }
            Expression::IndexAccess { expr, index } => {
                self.rewrite_expr(expr, params)?;
                self.rewrite_expr(index, params)?;
                Ok(())
            }
            Expression::ListSlice { expr, start, end } => {
                self.rewrite_expr(expr, params)?;
                if let Some(s) = start {
                    self.rewrite_expr(s, params)?;
                }
                if let Some(e) = end {
                    self.rewrite_expr(e, params)?;
                }
                Ok(())
            }
            Expression::ListComprehension {
                list_expr,
                filter,
                map_expr,
                ..
            } => {
                self.rewrite_expr(list_expr, params)?;
                if let Some(f) = filter {
                    self.rewrite_pred(f, params)?;
                }
                if let Some(m) = map_expr {
                    self.rewrite_expr(m, params)?;
                }
                Ok(())
            }
            Expression::MapProjection { items, .. } => {
                for item in items.iter_mut() {
                    if let MapProjectionItem::Alias { expr, .. } = item {
                        self.rewrite_expr(expr, params)?;
                    }
                }
                Ok(())
            }
            Expression::MapLiteral(entries) => {
                for (_, expr) in entries.iter_mut() {
                    self.rewrite_expr(expr, params)?;
                }
                Ok(())
            }
            // Leaf nodes
            Expression::PropertyAccess { .. }
            | Expression::Variable(_)
            | Expression::Literal(_)
            | Expression::Parameter(_)
            | Expression::Star => Ok(()),
            Expression::IsNull(inner) | Expression::IsNotNull(inner) => {
                self.rewrite_expr(inner, params)
            }
            Expression::QuantifiedList {
                list_expr, filter, ..
            } => {
                self.rewrite_expr(list_expr, params)?;
                self.rewrite_pred(filter, params)?;
                Ok(())
            }
            Expression::WindowFunction {
                partition_by,
                order_by,
                ..
            } => {
                for expr in partition_by.iter_mut() {
                    self.rewrite_expr(expr, params)?;
                }
                for item in order_by.iter_mut() {
                    self.rewrite_expr(&mut item.expression, params)?;
                }
                Ok(())
            }
            Expression::PredicateExpr(pred) => self.rewrite_pred(pred, params),
            Expression::ExprPropertyAccess { expr, .. } => self.rewrite_expr(expr, params),
            Expression::CountSubquery { where_clause, .. } => {
                // Patterns don't carry text_score() calls; only the
                // optional WHERE predicate might. Rewrite it if present.
                if let Some(pred) = where_clause.as_deref_mut() {
                    self.rewrite_pred(pred, params)?;
                }
                Ok(())
            }
            Expression::Reduce {
                init,
                list_expr,
                body,
                ..
            } => {
                self.rewrite_expr(init, params)?;
                self.rewrite_expr(list_expr, params)?;
                self.rewrite_expr(body, params)?;
                Ok(())
            }
        }
    }

    /// Rewrite predicates in-place (for WHERE clauses).
    fn rewrite_pred(
        &mut self,
        pred: &mut Predicate,
        params: &HashMap<String, Value>,
    ) -> Result<(), String> {
        match pred {
            Predicate::Comparison { left, right, .. } => {
                self.rewrite_expr(left, params)?;
                self.rewrite_expr(right, params)?;
                Ok(())
            }
            Predicate::And(l, r) | Predicate::Or(l, r) | Predicate::Xor(l, r) => {
                self.rewrite_pred(l, params)?;
                self.rewrite_pred(r, params)?;
                Ok(())
            }
            Predicate::Not(inner) => self.rewrite_pred(inner, params),
            Predicate::IsNull(e) | Predicate::IsNotNull(e) => self.rewrite_expr(e, params),
            Predicate::In { expr, list } => {
                self.rewrite_expr(expr, params)?;
                for item in list.iter_mut() {
                    self.rewrite_expr(item, params)?;
                }
                Ok(())
            }
            Predicate::InLiteralSet { expr, .. } => self.rewrite_expr(expr, params),
            Predicate::StartsWith { expr, pattern }
            | Predicate::EndsWith { expr, pattern }
            | Predicate::Contains { expr, pattern } => {
                self.rewrite_expr(expr, params)?;
                self.rewrite_expr(pattern, params)?;
                Ok(())
            }
            Predicate::Exists { .. } => Ok(()),
            Predicate::InExpression { expr, list_expr } => {
                self.rewrite_expr(expr, params)?;
                self.rewrite_expr(list_expr, params)?;
                Ok(())
            }
            Predicate::LabelCheck { .. } => Ok(()),
        }
    }
}
