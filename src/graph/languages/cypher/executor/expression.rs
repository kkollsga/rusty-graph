//! Cypher executor — expression methods.

use super::super::ast::*;
use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::algorithms::vector as vs;
use crate::graph::storage::GraphRead;
use geo::BoundingRect;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::sync::Arc;

impl<'a> CypherExecutor<'a> {
    pub(crate) fn evaluate_expression(
        &self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<Value, String> {
        match expr {
            Expression::PropertyAccess { variable, property } => {
                self.resolve_property(variable, property, row)
            }
            Expression::Variable(name) => {
                // Check projected values first (from WITH)
                if let Some(val) = row.projected.get(name) {
                    return Ok(val.clone());
                }
                // For node variables, return a NodeRef (preserves identity
                // through collect → index → WITH → property-access)
                if let Some(&idx) = row.node_bindings.get(name) {
                    return Ok(Value::NodeRef(idx.index() as u32));
                }
                // Edge variable — return connection_type as representative value
                if let Some(edge) = row.edge_bindings.get(name) {
                    if let Some(edge_data) = {
                        let g = &self.graph.graph;
                        g.edge_weight(edge.edge_index)
                    } {
                        return Ok(Value::String(
                            edge_data
                                .connection_type_str(&self.graph.interner)
                                .to_string(),
                        ));
                    }
                }
                // Path variable — return hops count
                if let Some(path) = row.path_bindings.get(name) {
                    return Ok(Value::Int64(path.hops as i64));
                }
                // Variable might be unbound (OPTIONAL MATCH null)
                Ok(Value::Null)
            }
            Expression::Literal(val) => Ok(val.clone()),
            Expression::Star => Ok(Value::Int64(1)), // For count(*)
            Expression::Add(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_add(&l, &r))
            }
            Expression::Subtract(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_sub(&l, &r))
            }
            Expression::Multiply(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_mul(&l, &r))
            }
            Expression::Divide(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_div(&l, &r))
            }
            Expression::Modulo(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(arithmetic_mod(&l, &r))
            }
            Expression::Concat(left, right) => {
                let l = self.evaluate_expression(left, row)?;
                let r = self.evaluate_expression(right, row)?;
                Ok(crate::graph::core::value_operations::string_concat(&l, &r))
            }
            Expression::Negate(inner) => {
                let val = self.evaluate_expression(inner, row)?;
                Ok(arithmetic_negate(&val))
            }
            Expression::FunctionCall { name, args, .. } => {
                // HAVING context: aggregate function calls reference pre-computed
                // projected values. `count(m)` in HAVING resolves to the matching
                // column in the row (stored under alias or under its expression
                // string — augment_rows_with_aggregate_keys ensures both forms
                // are present before HAVING is evaluated).
                if is_aggregate_expression(expr) {
                    let col_key = expression_to_string(expr);
                    if let Some(val) = row.projected.get(&col_key) {
                        return Ok(val.clone());
                    }
                }
                // Non-aggregate functions evaluated per-row
                self.evaluate_scalar_function(name, args, row)
            }
            Expression::ListLiteral(items) => {
                // Evaluate each item - for now represent as string
                let values: Result<Vec<Value>, String> = items
                    .iter()
                    .map(|item| self.evaluate_expression(item, row))
                    .collect();
                let vals = values?;
                let formatted: Vec<String> = vals.iter().map(format_value_json).collect();
                Ok(Value::String(format!("[{}]", formatted.join(", "))))
            }
            Expression::Case {
                operand,
                when_clauses,
                else_expr,
            } => self.evaluate_case(operand.as_deref(), when_clauses, else_expr.as_deref(), row),
            Expression::Parameter(name) => self
                .params
                .get(name)
                .cloned()
                .ok_or_else(|| format!("Missing parameter: ${}", name)),
            Expression::ListComprehension {
                variable,
                list_expr,
                filter,
                map_expr,
            } => {
                // Special handling for nodes(p) / relationships(p): extract structured
                // data directly from path bindings so property access works correctly.
                // Without this, nodes(p) returns a JSON string that parse_list_value
                // cannot split correctly (commas inside JSON objects).
                if let Expression::FunctionCall { name, args, .. } = list_expr.as_ref() {
                    let fn_name = name.as_str();
                    if fn_name == "nodes" || fn_name == "relationships" || fn_name == "rels" {
                        if let Some(Expression::Variable(path_var)) = args.first() {
                            if let Some(path) = row.path_bindings.get(path_var) {
                                let path = path.clone();
                                return if fn_name == "nodes" {
                                    self.list_comp_nodes(variable, &path, filter, map_expr, row)
                                } else {
                                    self.list_comp_relationships(
                                        variable, &path, filter, map_expr, row,
                                    )
                                };
                            }
                        }
                    }
                }

                // Default path: evaluate and parse list value
                let list_val = self.evaluate_expression(list_expr, row)?;
                let items = parse_list_value(&list_val);

                let mut results = Vec::new();
                for item in items {
                    // Create a temporary row with the variable bound
                    let mut temp_row = row.clone();
                    temp_row.projected.insert(variable.clone(), item.clone());

                    // Apply filter if present
                    if let Some(ref pred) = filter {
                        if !self.evaluate_predicate(pred, &temp_row)? {
                            continue;
                        }
                    }

                    // Apply map expression or use the item itself
                    let result = if let Some(ref expr) = map_expr {
                        self.evaluate_expression(expr, &temp_row)?
                    } else {
                        item
                    };

                    results.push(format_value_json(&result));
                }

                Ok(Value::String(format!("[{}]", results.join(", "))))
            }

            Expression::MapProjection { variable, items } => {
                // Look up the node from bindings
                if let Some(&node_idx) = row.node_bindings.get(variable.as_str()) {
                    if let Some(node) = self.graph.graph.node_weight(node_idx) {
                        let mut props = Vec::new();
                        for item in items {
                            match item {
                                MapProjectionItem::Property(prop) => {
                                    let val = resolve_node_property(node, prop, self.graph);
                                    props.push(format!(
                                        "{}: {}",
                                        format_value_json(&Value::String(prop.clone())),
                                        format_value_json(&val)
                                    ));
                                }
                                MapProjectionItem::AllProperties => {
                                    // Include standard fields first
                                    for &builtin in &["title", "id", "type"] {
                                        let val = resolve_node_property(node, builtin, self.graph);
                                        if !matches!(val, Value::Null) {
                                            props.push(format!(
                                                "{}: {}",
                                                format_value_json(&Value::String(
                                                    builtin.to_string()
                                                )),
                                                format_value_json(&val)
                                            ));
                                        }
                                    }
                                    // Then all user-defined properties
                                    for key in node.property_keys(&self.graph.interner) {
                                        let val = resolve_node_property(node, key, self.graph);
                                        props.push(format!(
                                            "{}: {}",
                                            format_value_json(&Value::String(key.to_string())),
                                            format_value_json(&val)
                                        ));
                                    }
                                }
                                MapProjectionItem::Alias { key, expr } => {
                                    let val = self.evaluate_expression(expr, row)?;
                                    props.push(format!(
                                        "{}: {}",
                                        format_value_json(&Value::String(key.clone())),
                                        format_value_json(&val)
                                    ));
                                }
                            }
                        }
                        return Ok(Value::String(format!("{{{}}}", props.join(", "))));
                    }
                }
                Ok(Value::Null)
            }

            Expression::MapLiteral(entries) => {
                let mut props = Vec::new();
                for (key, expr) in entries {
                    let val = self.evaluate_expression(expr, row)?;
                    props.push(format!(
                        "{}: {}",
                        format_value_json(&Value::String(key.clone())),
                        format_value_json(&val)
                    ));
                }
                Ok(Value::String(format!("{{{}}}", props.join(", "))))
            }

            Expression::IndexAccess { expr, index } => {
                // Fast path: labels(n)[0] — bypass JSON round-trip
                if let Expression::FunctionCall { name, args, .. } = expr.as_ref() {
                    if name == "labels" {
                        if let Some(Expression::Variable(var)) = args.first() {
                            if let Expression::Literal(Value::Int64(lit_idx)) = index.as_ref() {
                                if *lit_idx == 0 {
                                    if let Some(&node_idx) = row.node_bindings.get(var.as_str()) {
                                        if let Some(node) = self.graph.graph.node_weight(node_idx) {
                                            return Ok(Value::String(
                                                node.get_node_type_ref(&self.graph.interner)
                                                    .to_string(),
                                            ));
                                        }
                                    }
                                }
                                return Ok(Value::Null);
                            }
                        }
                    }
                }

                let list_val = self.evaluate_expression(expr, row)?;
                let idx_val = self.evaluate_expression(index, row)?;

                let idx = match &idx_val {
                    Value::Int64(i) => *i,
                    Value::Float64(f) => *f as i64,
                    _ => return Err(format!("Index must be an integer, got {:?}", idx_val)),
                };

                // Parse the list (JSON-formatted string like "[\"Person\"]" or "[1, 2, 3]")
                let items = parse_list_value(&list_val);

                // Support negative indexing
                let len = items.len() as i64;
                let actual_idx = if idx < 0 { len + idx } else { idx };

                if actual_idx >= 0 && (actual_idx as usize) < items.len() {
                    Ok(items[actual_idx as usize].clone())
                } else {
                    Ok(Value::Null)
                }
            }
            Expression::ListSlice { expr, start, end } => {
                let list_val = self.evaluate_expression(expr, row)?;
                let items = parse_list_value(&list_val);
                let len = items.len() as i64;

                // Resolve start index (default 0), clamp to [0, len]
                let s = if let Some(se) = start {
                    let v = self.evaluate_expression(se, row)?;
                    match v {
                        Value::Int64(i) => {
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        Value::Float64(f) => {
                            let i = f as i64;
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        _ => return Err(format!("Slice start must be integer, got {:?}", v)),
                    }
                } else {
                    0
                };

                // Resolve end index (default len), clamp to [0, len]
                let e = if let Some(ee) = end {
                    let v = self.evaluate_expression(ee, row)?;
                    match v {
                        Value::Int64(i) => {
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        Value::Float64(f) => {
                            let i = f as i64;
                            let i = if i < 0 { len + i } else { i };
                            i.clamp(0, len) as usize
                        }
                        _ => return Err(format!("Slice end must be integer, got {:?}", v)),
                    }
                } else {
                    len as usize
                };

                if s >= e {
                    Ok(Value::String("[]".to_string()))
                } else {
                    let sliced = &items[s..e];
                    let formatted: Vec<String> = sliced.iter().map(format_value_json).collect();
                    Ok(Value::String(format!("[{}]", formatted.join(", "))))
                }
            }
            Expression::IsNull(inner) => {
                let val = self.evaluate_expression(inner, row)?;
                Ok(Value::Boolean(matches!(val, Value::Null)))
            }
            Expression::IsNotNull(inner) => {
                let val = self.evaluate_expression(inner, row)?;
                Ok(Value::Boolean(!matches!(val, Value::Null)))
            }
            Expression::QuantifiedList {
                quantifier,
                variable,
                list_expr,
                filter,
            } => {
                let list_val = self.evaluate_expression(list_expr, row)?;
                let items = parse_list_value(&list_val);

                let result = match quantifier {
                    ListQuantifier::Any => {
                        let mut found = false;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if self.evaluate_predicate(filter, &temp_row)? {
                                found = true;
                                break;
                            }
                        }
                        found
                    }
                    ListQuantifier::All => {
                        let mut all_pass = true;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if !self.evaluate_predicate(filter, &temp_row)? {
                                all_pass = false;
                                break;
                            }
                        }
                        all_pass
                    }
                    ListQuantifier::None => {
                        let mut none_pass = true;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if self.evaluate_predicate(filter, &temp_row)? {
                                none_pass = false;
                                break;
                            }
                        }
                        none_pass
                    }
                    ListQuantifier::Single => {
                        let mut count = 0;
                        for item in items {
                            let mut temp_row = row.clone();
                            temp_row.projected.insert(variable.clone(), item);
                            if self.evaluate_predicate(filter, &temp_row)? {
                                count += 1;
                                if count > 1 {
                                    break;
                                }
                            }
                        }
                        count == 1
                    }
                };
                Ok(Value::Boolean(result))
            }
            Expression::Reduce {
                accumulator,
                init,
                variable,
                list_expr,
                body,
            } => {
                let mut acc = self.evaluate_expression(init, row)?;
                let list_val = self.evaluate_expression(list_expr, row)?;
                let items = parse_list_value(&list_val);
                for item in items {
                    let mut temp_row = row.clone();
                    temp_row.projected.insert(accumulator.clone(), acc.clone());
                    temp_row.projected.insert(variable.clone(), item);
                    acc = self.evaluate_expression(body, &temp_row)?;
                }
                Ok(acc)
            }
            Expression::WindowFunction { .. } => {
                // Window functions are evaluated in a separate pass (apply_window_functions),
                // not per-row. If we reach here, the value should already be in projected bindings.
                Err("Window function must appear in RETURN/WITH clause".into())
            }
            Expression::PredicateExpr(pred) => {
                // Evaluate predicate as an expression (e.g. RETURN n.name STARTS WITH 'A').
                // For comparisons, implement three-valued logic: if either operand
                // is null, return Null instead of false.
                match pred.as_ref() {
                    Predicate::Comparison {
                        left,
                        operator,
                        right,
                    } => {
                        let left_val = self.evaluate_expression(left, row)?;
                        let right_val = self.evaluate_expression(right, row)?;
                        if matches!(left_val, Value::Null) || matches!(right_val, Value::Null) {
                            Ok(Value::Null)
                        } else {
                            match evaluate_comparison(
                                &left_val,
                                operator,
                                &right_val,
                                Some(&self.regex_cache),
                            ) {
                                Ok(b) => Ok(Value::Boolean(b)),
                                Err(_) => Ok(Value::Null),
                            }
                        }
                    }
                    _ => match self.evaluate_predicate(pred, row) {
                        Ok(b) => Ok(Value::Boolean(b)),
                        Err(_) => Ok(Value::Null),
                    },
                }
            }
            Expression::ExprPropertyAccess { expr, property } => {
                let val = self.evaluate_expression(expr, row)?;
                match &val {
                    Value::String(s) => {
                        // Try to parse as date string (YYYY-MM-DD) for .year/.month/.day
                        if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
                            use chrono::Datelike;
                            match property.as_str() {
                                "year" => return Ok(Value::Int64(date.year() as i64)),
                                "month" => return Ok(Value::Int64(date.month() as i64)),
                                "day" => return Ok(Value::Int64(date.day() as i64)),
                                _ => {}
                            }
                        }
                        // Try ISO datetime format
                        if let Ok(dt) =
                            chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
                        {
                            use chrono::Datelike;
                            match property.as_str() {
                                "year" => return Ok(Value::Int64(dt.year() as i64)),
                                "month" => return Ok(Value::Int64(dt.month() as i64)),
                                "day" => return Ok(Value::Int64(dt.day() as i64)),
                                _ => {}
                            }
                        }
                        Ok(Value::Null)
                    }
                    Value::DateTime(date) => {
                        use chrono::Datelike;
                        match property.as_str() {
                            "year" => Ok(Value::Int64(date.year() as i64)),
                            "month" => Ok(Value::Int64(date.month() as i64)),
                            "day" => Ok(Value::Int64(date.day() as i64)),
                            _ => Ok(Value::Null),
                        }
                    }
                    _ => Ok(Value::Null),
                }
            }
            Expression::CountSubquery {
                patterns,
                where_clause,
            } => {
                // Evaluate each pattern scoped to the outer row's
                // bindings; sum the match counts. Mirrors the
                // `Predicate::Exists` execution in `where_clause.rs`
                // but counts (not short-circuits).
                use crate::graph::core::pattern_matching::PatternExecutor;
                let mut total = 0i64;
                for pattern in patterns {
                    let resolved;
                    let pat = if Self::pattern_has_vars(pattern) {
                        resolved = self.resolve_pattern_vars(pattern, row);
                        &resolved
                    } else {
                        pattern
                    };
                    let executor = PatternExecutor::with_bindings_and_params(
                        self.graph,
                        None,
                        &row.node_bindings,
                        self.params,
                    )
                    .set_deadline(self.deadline);
                    let matches = executor.execute(pat)?;
                    let count = if let Some(ref where_pred) = where_clause {
                        matches
                            .iter()
                            .filter(|m| {
                                if !self.bindings_compatible(row, m) {
                                    return false;
                                }
                                let mut combined = row.clone();
                                self.merge_match_into_row(&mut combined, m);
                                self.evaluate_predicate(where_pred, &combined)
                                    .unwrap_or(false)
                            })
                            .count()
                    } else {
                        matches
                            .iter()
                            .filter(|m| self.bindings_compatible(row, m))
                            .count()
                    };
                    total += count as i64;
                }
                Ok(Value::Int64(total))
            }
        }
    }

    /// List comprehension over nodes(p): bind each path node as a node_binding
    /// so that property access (n.name, n.type, etc.) resolves correctly.
    pub(super) fn list_comp_nodes(
        &self,
        variable: &str,
        path: &PathBinding,
        filter: &Option<Box<Predicate>>,
        map_expr: &Option<Box<Expression>>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        let mut node_indices = vec![path.source];
        for (node_idx, _) in &path.path {
            node_indices.push(*node_idx);
        }

        let mut results = Vec::new();
        for node_idx in node_indices {
            let mut temp_row = row.clone();
            temp_row
                .node_bindings
                .insert(variable.to_string(), node_idx);

            if let Some(ref pred) = filter {
                if !self.evaluate_predicate(pred, &temp_row)? {
                    continue;
                }
            }

            let result = if let Some(ref expr) = map_expr {
                self.evaluate_expression(expr, &temp_row)?
            } else {
                // No map expression — serialize node as JSON dict (backward compatible)
                if let Some(node) = self.graph.graph.node_weight(node_idx) {
                    let mut props = Vec::new();
                    props.push(format!("\"id\": {}", format_value_compact(&node.id())));
                    props.push(format!(
                        "\"title\": \"{}\"",
                        format_value_compact(&node.title()).replace('"', "\\\"")
                    ));
                    props.push(format!(
                        "\"type\": \"{}\"",
                        node.node_type_str(&self.graph.interner)
                    ));
                    Value::String(format!("{{{}}}", props.join(", ")))
                } else {
                    Value::Null
                }
            };

            results.push(format_value_json(&result));
        }
        Ok(Value::String(format!("[{}]", results.join(", "))))
    }

    /// List comprehension over relationships(p): bind each relationship type as a projected value.
    pub(super) fn list_comp_relationships(
        &self,
        variable: &str,
        path: &PathBinding,
        filter: &Option<Box<Predicate>>,
        map_expr: &Option<Box<Expression>>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        let mut results = Vec::new();
        for (_, conn_type) in &path.path {
            let mut temp_row = row.clone();
            temp_row
                .projected
                .insert(variable.to_string(), Value::String(conn_type.clone()));

            if let Some(ref pred) = filter {
                if !self.evaluate_predicate(pred, &temp_row)? {
                    continue;
                }
            }

            let result = if let Some(ref expr) = map_expr {
                self.evaluate_expression(expr, &temp_row)?
            } else {
                Value::String(conn_type.clone())
            };

            results.push(format_value_json(&result));
        }
        Ok(Value::String(format!("[{}]", results.join(", "))))
    }

    /// Evaluate a CASE expression
    pub(super) fn evaluate_case(
        &self,
        operand: Option<&Expression>,
        when_clauses: &[(CaseCondition, Expression)],
        else_expr: Option<&Expression>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        if let Some(operand_expr) = operand {
            // Simple form: CASE expr WHEN val THEN result ...
            let operand_val = self.evaluate_expression(operand_expr, row)?;
            for (condition, result) in when_clauses {
                if let CaseCondition::Expression(cond_expr) = condition {
                    let cond_val = self.evaluate_expression(cond_expr, row)?;
                    if crate::graph::core::filtering::values_equal(&operand_val, &cond_val) {
                        return self.evaluate_expression(result, row);
                    }
                }
            }
        } else {
            // Generic form: CASE WHEN predicate THEN result ...
            for (condition, result) in when_clauses {
                if let CaseCondition::Predicate(pred) = condition {
                    if self.evaluate_predicate(pred, row)? {
                        return self.evaluate_expression(result, row);
                    }
                }
            }
        }

        // No match — evaluate ELSE or return null
        if let Some(else_e) = else_expr {
            self.evaluate_expression(else_e, row)
        } else {
            Ok(Value::Null)
        }
    }

    /// Unified spatial argument resolver. Returns Point or Geometry depending
    /// on what the expression/value resolves to.
    ///
    /// Resolve a spatial argument from its expression, using a per-node cache
    /// that ensures each NodeIndex is resolved at most once per query execution.
    ///
    /// `prefer_geometry`: When true, Variable resolution prefers geometry config
    /// over location (for contains/intersects/centroid/area/perimeter).
    /// When false, prefers location → Point (for distance).
    /// PropertyAccess always resolves based on the explicit property name.
    pub(super) fn resolve_spatial(
        &self,
        expr: &Expression,
        row: &ResultRow,
        prefer_geometry: bool,
    ) -> Result<Option<ResolvedSpatial>, String> {
        match expr {
            // Fast path: Variable bound to a node → resolve from per-node cache
            Expression::Variable(name) => {
                if let Some(&idx) = row.node_bindings.get(name) {
                    self.ensure_node_spatial_cached(idx);
                    let cache = self.spatial_node_cache.read().unwrap();
                    if let Some(cached) = cache.get(&idx.index()) {
                        return Ok(Self::pick_from_node_cache(cached, prefer_geometry));
                    }
                }
                // Not a node binding — evaluate and check value
                let val = self.evaluate_expression(expr, row)?;
                self.resolve_spatial_from_value(&val)
            }
            // Fast path: PropertyAccess on a node → resolve from per-node cache
            Expression::PropertyAccess { variable, property } => {
                if let Some(&idx) = row.node_bindings.get(variable) {
                    self.ensure_node_spatial_cached(idx);
                    let cache = self.spatial_node_cache.read().unwrap();
                    if let Some(cached) = cache.get(&idx.index()) {
                        if let Some(result) = Self::pick_property_from_node_cache(cached, property)
                        {
                            return Ok(Some(result));
                        }
                    }
                }
                // Fallback: evaluate and check value
                let val = self.evaluate_expression(expr, row)?;
                self.resolve_spatial_from_value(&val)
            }
            // Any other expression: evaluate first, then check if spatial
            _ => {
                let val = self.evaluate_expression(expr, row)?;
                self.resolve_spatial_from_value(&val)
            }
        }
    }

    /// Resolve a Cypher argument to a [`geo::Geometry`]. Accepts:
    /// - WKT string literal or property — parsed via the WKT crate
    /// - Node variable / property access whose spatial config produces a Geometry
    /// - Point variable / property — converted to a Geometry::Point
    ///
    /// Returns `Ok(None)` when the input is null.
    pub(super) fn geom_arg(
        &self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<Option<geo::Geometry<f64>>, String> {
        // Try the spatial cache first (handles node/property variables).
        if let Ok(Some(resolved)) = self.resolve_spatial(expr, row, true) {
            return match resolved {
                ResolvedSpatial::Geometry(g, _) => Ok(Some((*g).clone())),
                ResolvedSpatial::Point(lat, lon) => {
                    Ok(Some(geo::Geometry::Point(geo::Point::new(lon, lat))))
                }
            };
        }
        // Otherwise: evaluate the expression and try to parse a WKT string.
        let val = self.evaluate_expression(expr, row)?;
        match val {
            Value::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    return Ok(None);
                }
                Ok(Some(crate::graph::features::spatial::parse_wkt(trimmed)?))
            }
            Value::Point { lat, lon } => Ok(Some(geo::Geometry::Point(geo::Point::new(lon, lat)))),
            Value::Null => Ok(None),
            _ => Err("expected a WKT string, Point, or spatial node/property".into()),
        }
    }

    /// Ensure that the per-node spatial cache entry exists for the given NodeIndex.
    /// Populates geometry+bbox, location, named shapes, and named points on first access.
    #[inline]
    pub(super) fn ensure_node_spatial_cached(&self, idx: NodeIndex) {
        let idx_raw = idx.index();
        {
            let cache = self.spatial_node_cache.read().unwrap();
            if cache.contains_key(&idx_raw) {
                return;
            }
        }
        let data = self.build_node_spatial_data(idx);
        self.spatial_node_cache
            .write()
            .unwrap()
            .insert(idx_raw, data);
    }

    /// Build the full spatial data for a node: geometry+bbox, location, named shapes/points.
    /// Returns None if the node has no spatial config.
    pub(super) fn build_node_spatial_data(&self, idx: NodeIndex) -> Option<NodeSpatialData> {
        let node = self.graph.graph.node_weight(idx)?;
        let config = self
            .graph
            .get_spatial_config(node.node_type_str(&self.graph.interner))?;

        // Primary geometry + bounding box
        let geometry = config.geometry.as_ref().and_then(|geom_f| {
            if let Some(Value::String(wkt)) = node.get_property(geom_f).as_deref() {
                if let Ok(geom) = self.parse_wkt_cached(wkt) {
                    let bbox = geom.bounding_rect();
                    return Some((geom, bbox));
                }
            }
            None
        });

        // Primary location
        let location = config.location.as_ref().and_then(|(lat_f, lon_f)| {
            let lat = node
                .get_property(lat_f)
                .as_deref()
                .and_then(crate::graph::core::value_operations::value_to_f64)?;
            let lon = node
                .get_property(lon_f)
                .as_deref()
                .and_then(crate::graph::core::value_operations::value_to_f64)?;
            Some((lat, lon))
        });

        // Named shapes
        let mut shapes = HashMap::new();
        for (name, field) in &config.shapes {
            if let Some(Value::String(wkt)) = node.get_property(field).as_deref() {
                if let Ok(geom) = self.parse_wkt_cached(wkt) {
                    let bbox = geom.bounding_rect();
                    shapes.insert(name.clone(), (geom, bbox));
                }
            }
        }

        // Named points
        let mut points = HashMap::new();
        for (name, (lat_f, lon_f)) in &config.points {
            if let (Some(lat), Some(lon)) = (
                node.get_property(lat_f)
                    .as_deref()
                    .and_then(crate::graph::core::value_operations::value_to_f64),
                node.get_property(lon_f)
                    .as_deref()
                    .and_then(crate::graph::core::value_operations::value_to_f64),
            ) {
                points.insert(name.clone(), (lat, lon));
            }
        }

        Some(NodeSpatialData {
            geometry,
            location,
            shapes,
            points,
        })
    }

    /// Pick the right spatial value from cached node data based on preference.
    #[inline]
    pub(super) fn pick_from_node_cache(
        data: &Option<NodeSpatialData>,
        prefer_geometry: bool,
    ) -> Option<ResolvedSpatial> {
        let data = data.as_ref()?;
        if prefer_geometry {
            // Prefer geometry → Geometry; fallback to location → Point
            if let Some((geom, bbox)) = &data.geometry {
                return Some(ResolvedSpatial::Geometry(Arc::clone(geom), *bbox));
            }
            if let Some((lat, lon)) = data.location {
                return Some(ResolvedSpatial::Point(lat, lon));
            }
        } else {
            // Prefer location → Point; fallback to geometry centroid → Point
            if let Some((lat, lon)) = data.location {
                return Some(ResolvedSpatial::Point(lat, lon));
            }
            if let Some((geom, _bbox)) = &data.geometry {
                if let Ok((lat, lon)) = crate::graph::features::spatial::geometry_centroid(geom) {
                    return Some(ResolvedSpatial::Point(lat, lon));
                }
            }
        }
        None
    }

    /// Pick a specific property from cached node data (for PropertyAccess resolution).
    #[inline]
    pub(super) fn pick_property_from_node_cache(
        data: &Option<NodeSpatialData>,
        property: &str,
    ) -> Option<ResolvedSpatial> {
        let data = data.as_ref()?;
        // Named shapes
        if let Some((geom, bbox)) = data.shapes.get(property) {
            return Some(ResolvedSpatial::Geometry(Arc::clone(geom), *bbox));
        }
        // Named points
        if let Some((lat, lon)) = data.points.get(property) {
            return Some(ResolvedSpatial::Point(*lat, *lon));
        }
        // "geometry" → primary geometry
        if property == "geometry" {
            if let Some((geom, bbox)) = &data.geometry {
                return Some(ResolvedSpatial::Geometry(Arc::clone(geom), *bbox));
            }
        }
        // "location" → primary location
        if property == "location" {
            if let Some((lat, lon)) = data.location {
                return Some(ResolvedSpatial::Point(lat, lon));
            }
        }
        None
    }

    /// Try to resolve a pre-evaluated value as spatial (Point or WKT geometry).
    #[inline]
    pub(super) fn resolve_spatial_from_value(
        &self,
        val: &Value,
    ) -> Result<Option<ResolvedSpatial>, String> {
        if let Value::Point { lat, lon } = val {
            return Ok(Some(ResolvedSpatial::Point(*lat, *lon)));
        }
        if let Value::String(s) = val {
            if let Ok(geom) = self.parse_wkt_cached(s) {
                let bbox = geom.bounding_rect();
                return Ok(Some(ResolvedSpatial::Geometry(geom, bbox)));
            }
        }
        Ok(None)
    }

    /// Resolve property access: variable.property
    /// Uses zero-copy get_field_ref when possible
    pub(super) fn resolve_property(
        &self,
        variable: &str,
        property: &str,
        row: &ResultRow,
    ) -> Result<Value, String> {
        // Check node bindings first — these carry full property data
        // and must take priority over projected scalars (e.g. after WITH)
        if let Some(&idx) = row.node_bindings.get(variable) {
            // Disk-graph fast path: resolve properties via direct column access
            // without full NodeData materialization. Saves arena allocation +
            // id/title reads per access. In-memory graphs use node_weight()
            // which is a cheap pointer chase.
            if self.graph.graph.is_disk() {
                if let Some(type_key) = self.graph.graph.node_type_of(idx) {
                    let type_str = self.graph.interner.try_resolve(type_key).unwrap_or("?");
                    let resolved = self.graph.resolve_alias(type_str, property);
                    match resolved {
                        "id" => {
                            return Ok(self.graph.graph.get_node_id(idx).unwrap_or(Value::Null))
                        }
                        "title" | "name" => {
                            return Ok(self.graph.graph.get_node_title(idx).unwrap_or(Value::Null))
                        }
                        "type" | "node_type" | "label" => {
                            return Ok(Value::String(type_str.to_string()))
                        }
                        _ => {
                            let key = crate::graph::schema::InternedKey::from_str(resolved);
                            if let Some(val) = self.graph.graph.get_node_property(idx, key) {
                                return Ok(val);
                            }
                            // Fall through to full materialization for spatial
                            // virtual properties (location, geometry, etc.)
                            if self.graph.get_spatial_config(type_str).is_some() {
                                if let Some(node) = self.graph.graph.node_weight(idx) {
                                    return Ok(resolve_node_property(node, property, self.graph));
                                }
                            }
                            return Ok(Value::Null);
                        }
                    }
                }
                return Ok(Value::Null);
            }

            // In-memory path: node_weight() is a cheap pointer chase
            if let Some(node) = self.graph.graph.node_weight(idx) {
                return Ok(resolve_node_property(node, property, self.graph));
            }
            return Ok(Value::Null);
        }

        // Edge variable
        if let Some(edge) = row.edge_bindings.get(variable) {
            return Ok(resolve_edge_property(self.graph, edge, property));
        }

        // Path variable
        if let Some(path) = row.path_bindings.get(variable) {
            return match property {
                "length" | "hops" => Ok(Value::Int64(path.hops as i64)),
                _ => Ok(Value::Null),
            };
        }

        // Fall back to projected values (scalar aliases from WITH)
        if let Some(val) = row.projected.get(variable) {
            // NodeRef in projected → resolve the actual node property
            if let Value::NodeRef(idx) = val {
                let node_idx = petgraph::graph::NodeIndex::new(*idx as usize);
                if let Some(node) = self.graph.graph.node_weight(node_idx) {
                    return Ok(resolve_node_property(node, property, self.graph));
                }
                return Ok(Value::Null);
            }
            // DateTime property accessors: .year, .month, .day
            if let Value::DateTime(date) = val {
                use chrono::Datelike;
                return Ok(match property {
                    "year" => Value::Int64(date.year() as i64),
                    "month" => Value::Int64(date.month() as i64),
                    "day" => Value::Int64(date.day() as i64),
                    _ => Value::Null,
                });
            }
            return Ok(val.clone());
        }

        // Variable not found - might be OPTIONAL MATCH null
        Ok(Value::Null)
    }

    /// Parse a WKT string, using the graph-level cache to avoid redundant parsing.
    /// Returns Arc<Geometry> — cheap to clone (just a refcount bump).
    pub(super) fn parse_wkt_cached(&self, wkt: &str) -> Result<Arc<geo::Geometry<f64>>, String> {
        // Fast path: read lock for cache hit
        {
            let cache = self.graph.wkt_cache.read().unwrap();
            if let Some(geom) = cache.get(wkt) {
                return Ok(Arc::clone(geom));
            }
        }
        // Slow path: parse + write lock
        let geom = Arc::new(crate::graph::features::spatial::parse_wkt(wkt)?);
        {
            let mut cache = self.graph.wkt_cache.write().unwrap();
            cache.insert(wkt.to_string(), Arc::clone(&geom));
        }
        Ok(geom)
    }

    /// Evaluate scalar (non-aggregate) functions
    pub(super) fn evaluate_scalar_function(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        match name {
            "toupper" | "touppercase" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => Ok(Value::String(s.to_uppercase())),
                    _ => Ok(Value::Null),
                }
            }
            "tolower" | "tolowercase" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => Ok(Value::String(s.to_lowercase())),
                    _ => Ok(Value::Null),
                }
            }
            "tostring" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(Value::String(format_value_compact(&val)))
            }
            "tointeger" | "toint" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(to_integer(&val))
            }
            "tofloat" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(to_float(&val))
            }
            "date" => {
                if args.len() != 1 {
                    return Err("date() requires 1 argument: date('2020-01-15')".into());
                }
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => {
                        // Return Null on invalid input instead of crashing (BUG-09)
                        match crate::graph::features::timeseries::parse_date_query(&s) {
                            Ok((d, _)) => Ok(Value::DateTime(d)),
                            Err(_) => Ok(Value::Null),
                        }
                    }
                    Value::DateTime(_) => Ok(val),
                    Value::Null => Ok(Value::Null),
                    _ => Err(format!("date() argument must be a string, got {:?}", val)),
                }
            }
            "datetime" => {
                if args.len() != 1 {
                    return Err(
                        "datetime() requires 1 argument: datetime('2024-03-15T10:30:00')".into(),
                    );
                }
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => {
                        // Try parsing as ISO datetime with T separator
                        if s.contains('T') {
                            let date_part = s.split('T').next().unwrap_or("");
                            match crate::graph::features::timeseries::parse_date_query(date_part) {
                                Ok((d, _)) => Ok(Value::DateTime(d)),
                                Err(_) => Ok(Value::Null),
                            }
                        } else {
                            // Fallback: try as plain date
                            match crate::graph::features::timeseries::parse_date_query(&s) {
                                Ok((d, _)) => Ok(Value::DateTime(d)),
                                Err(_) => Ok(Value::Null),
                            }
                        }
                    }
                    Value::DateTime(_) => Ok(val),
                    Value::Null => Ok(Value::Null),
                    _ => Err(format!(
                        "datetime() argument must be a string, got {:?}",
                        val
                    )),
                }
            }
            "date_diff" | "datediff" => {
                if args.len() != 2 {
                    return Err("date_diff() requires 2 date arguments".into());
                }
                let a = self.evaluate_expression(&args[0], row)?;
                let b = self.evaluate_expression(&args[1], row)?;
                match (&a, &b) {
                    (Value::DateTime(d1), Value::DateTime(d2)) => {
                        Ok(Value::Int64((*d1 - *d2).num_days()))
                    }
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    _ => Err("date_diff() arguments must be dates".into()),
                }
            }
            "size" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => {
                        // Lists are stored as JSON-like strings; count elements
                        if s.starts_with('[') && s.ends_with(']') {
                            let items = parse_list_value(&Value::String(s));
                            Ok(Value::Int64(items.len() as i64))
                        } else {
                            Ok(Value::Int64(s.len() as i64))
                        }
                    }
                    _ => Ok(Value::Null),
                }
            }
            "length" => {
                // length(p) for paths, length(s) for strings, length(list) for lists
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        return Ok(Value::Int64(path.hops as i64));
                    }
                }
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => {
                        if s.starts_with('[') && s.ends_with(']') {
                            let items = parse_list_value(&Value::String(s));
                            Ok(Value::Int64(items.len() as i64))
                        } else {
                            Ok(Value::Int64(s.len() as i64))
                        }
                    }
                    _ => Ok(Value::Null),
                }
            }
            "nodes" => {
                // nodes(p) returns list of node dicts in a path (source + intermediates + target)
                // Path format is normalized: path.path excludes source, source is in path.source
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        let mut entries = Vec::new();
                        let mut node_indices = vec![path.source];
                        for (node_idx, _) in &path.path {
                            node_indices.push(*node_idx);
                        }
                        for node_idx in &node_indices {
                            if let Some(node) = self.graph.graph.node_weight(*node_idx) {
                                let mut props = Vec::new();
                                props.push(format!("\"id\": {}", format_value_compact(&node.id())));
                                props.push(format!(
                                    "\"title\": \"{}\"",
                                    format_value_compact(&node.title()).replace('"', "\\\"")
                                ));
                                props.push(format!(
                                    "\"type\": \"{}\"",
                                    node.node_type_str(&self.graph.interner)
                                ));
                                entries.push(format!("{{{}}}", props.join(", ")));
                            }
                        }
                        return Ok(Value::String(format!("[{}]", entries.join(", "))));
                    }
                }
                Ok(Value::Null)
            }
            "relationships" | "rels" => {
                // relationships(p) returns list of relationship types in a path (JSON array)
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        let mut rel_strs = Vec::new();
                        for (_, conn_type) in &path.path {
                            if !conn_type.is_empty() {
                                rel_strs.push(format!("\"{}\"", conn_type));
                            }
                        }
                        return Ok(Value::String(format!("[{}]", rel_strs.join(", "))));
                    }
                }
                Ok(Value::Null)
            }
            "type" => {
                // type(r) returns the relationship type
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(edge) = row.edge_bindings.get(var) {
                        if let Some(edge_data) = {
                            let g = &self.graph.graph;
                            g.edge_weight(edge.edge_index)
                        } {
                            return Ok(Value::String(
                                edge_data
                                    .connection_type_str(&self.graph.interner)
                                    .to_string(),
                            ));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "id" => {
                // id(n) returns the node id
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            return Ok(resolve_node_property(node, "id", self.graph));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "labels" => {
                // labels(n) returns list of node labels (as JSON list)
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            let node_type = node.get_node_type_ref(&self.graph.interner);
                            return Ok(Value::String(format!(
                                "[\"{}\"]",
                                node_type.replace('\\', "\\\\").replace('"', "\\\"")
                            )));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "keys" => {
                // keys(n) or keys(r) — return property names as a JSON list
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            let mut keys: Vec<&str> = vec!["id", "title", "type"];
                            keys.extend(node.property_keys(&self.graph.interner));
                            keys.sort();
                            return Ok(Value::String(format!(
                                "[{}]",
                                keys.iter()
                                    .map(|k| format!("\"{}\"", k))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )));
                        }
                    }
                    if let Some(edge) = row.edge_bindings.get(var) {
                        if let Some(edge_data) = {
                            let g = &self.graph.graph;
                            g.edge_weight(edge.edge_index)
                        } {
                            let mut keys: Vec<&str> = vec!["type"];
                            keys.extend(edge_data.property_keys(&self.graph.interner));
                            keys.sort();
                            return Ok(Value::String(format!(
                                "[{}]",
                                keys.iter()
                                    .map(|k| format!("\"{}\"", k))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "coalesce" => {
                // coalesce(expr1, expr2, ...) returns first non-null
                for arg in args {
                    let val = self.evaluate_expression(arg, row)?;
                    if !matches!(val, Value::Null) {
                        return Ok(val);
                    }
                }
                Ok(Value::Null)
            }
            "properties" => {
                // properties(n) / properties(r) → JSON-formatted property map
                if args.len() != 1 {
                    return Err("properties() requires 1 argument: a node or relationship".into());
                }
                if let Expression::Variable(var) = &args[0] {
                    if let Some(&idx) = row.node_bindings.get(var.as_str()) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            let mut props: Vec<String> = Vec::new();
                            for &builtin in &["id", "title", "type"] {
                                let val = resolve_node_property(node, builtin, self.graph);
                                if !matches!(val, Value::Null) {
                                    props.push(format!(
                                        "{}: {}",
                                        format_value_json(&Value::String(builtin.to_string())),
                                        format_value_json(&val)
                                    ));
                                }
                            }
                            for key in node.property_keys(&self.graph.interner) {
                                let val = resolve_node_property(node, key, self.graph);
                                props.push(format!(
                                    "{}: {}",
                                    format_value_json(&Value::String(key.to_string())),
                                    format_value_json(&val)
                                ));
                            }
                            return Ok(Value::String(format!("{{{}}}", props.join(", "))));
                        }
                    }
                    if let Some(edge) = row.edge_bindings.get(var.as_str()) {
                        if let Some(edge_data) = {
                            let g = &self.graph.graph;
                            g.edge_weight(edge.edge_index)
                        } {
                            let mut props: Vec<String> = Vec::new();
                            props.push(format!(
                                "{}: {}",
                                format_value_json(&Value::String("type".to_string())),
                                format_value_json(&Value::String(
                                    edge_data
                                        .connection_type_str(&self.graph.interner)
                                        .to_string()
                                ))
                            ));
                            for key in edge_data.property_keys(&self.graph.interner) {
                                if let Some(val) = edge_data.get_property(key) {
                                    props.push(format!(
                                        "{}: {}",
                                        format_value_json(&Value::String(key.to_string())),
                                        format_value_json(val)
                                    ));
                                }
                            }
                            return Ok(Value::String(format!("{{{}}}", props.join(", "))));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "start_node" | "startnode" => {
                // start_node(r) → source node of the bound relationship
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(edge) = row.edge_bindings.get(var.as_str()) {
                        return Ok(Value::NodeRef(edge.source.index() as u32));
                    }
                }
                Ok(Value::Null)
            }
            "end_node" | "endnode" => {
                // end_node(r) → target node of the bound relationship
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(edge) = row.edge_bindings.get(var.as_str()) {
                        return Ok(Value::NodeRef(edge.target.index() as u32));
                    }
                }
                Ok(Value::Null)
            }
            // ── Text predicates (0.8.20) ──────────────────────────────
            "text_edit_distance" => {
                if args.len() != 2 {
                    return Err("text_edit_distance() requires 2 arguments".into());
                }
                let a = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let b = coerce_to_string(self.evaluate_expression(&args[1], row)?);
                match (&a, &b) {
                    (Value::String(s1), Value::String(s2)) => {
                        Ok(Value::Int64(levenshtein(s1, s2) as i64))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "text_normalize" => {
                if args.len() != 1 {
                    return Err("text_normalize() requires 1 argument".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => {
                        let mut out = String::with_capacity(s.len());
                        let mut last_space = true;
                        for c in s.chars() {
                            if c.is_alphanumeric() {
                                for lc in c.to_lowercase() {
                                    out.push(lc);
                                }
                                last_space = false;
                            } else if c.is_whitespace() && !last_space {
                                out.push(' ');
                                last_space = true;
                            }
                            // punctuation: drop
                        }
                        Ok(Value::String(out.trim().to_string()))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "text_jaccard" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(
                        "text_jaccard() requires 2-3 arguments: (a, b [, separator])".into(),
                    );
                }
                let a = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let b = coerce_to_string(self.evaluate_expression(&args[1], row)?);
                let sep = if args.len() == 3 {
                    match self.evaluate_expression(&args[2], row)? {
                        Value::String(s) => Some(s),
                        _ => return Err("text_jaccard(): separator must be a string".into()),
                    }
                } else {
                    None
                };
                match (&a, &b) {
                    (Value::String(s1), Value::String(s2)) => {
                        let tokenize = |s: &str| -> std::collections::HashSet<String> {
                            match &sep {
                                Some(d) => s.split(d.as_str()).map(|t| t.to_string()).collect(),
                                None => s.split_whitespace().map(|t| t.to_string()).collect(),
                            }
                        };
                        let set_a = tokenize(s1);
                        let set_b = tokenize(s2);
                        if set_a.is_empty() && set_b.is_empty() {
                            return Ok(Value::Float64(1.0));
                        }
                        let inter = set_a.intersection(&set_b).count() as f64;
                        let union = set_a.union(&set_b).count() as f64;
                        Ok(Value::Float64(inter / union))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "text_ngrams" => {
                if args.len() != 2 {
                    return Err("text_ngrams() requires 2 arguments: (string, n)".into());
                }
                let s_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let n_val = self.evaluate_expression(&args[1], row)?;
                match (&s_val, &n_val) {
                    (Value::String(s), Value::Int64(n)) => {
                        let n = *n as usize;
                        if n == 0 {
                            return Err("text_ngrams(): n must be ≥ 1".into());
                        }
                        let chars: Vec<char> = s.chars().collect();
                        let mut grams: Vec<String> = Vec::new();
                        if chars.len() >= n {
                            for i in 0..=chars.len() - n {
                                let gram: String = chars[i..i + n].iter().collect();
                                grams.push(format!(
                                    "\"{}\"",
                                    gram.replace('\\', "\\\\").replace('"', "\\\"")
                                ));
                            }
                        }
                        Ok(Value::String(format!("[{}]", grams.join(", "))))
                    }
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    _ => Ok(Value::Null),
                }
            }
            "text_contains_any" => {
                if args.is_empty() {
                    return Err("text_contains_any() requires at least 1 argument".into());
                }
                let s_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let s = match &s_val {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Value::Null),
                };
                // Accept either a literal list as second arg, or variadic remaining args.
                if args.len() == 2 {
                    let list_val = self.evaluate_expression(&args[1], row)?;
                    if let Value::String(ref ls) = list_val {
                        if ls.starts_with('[') && ls.ends_with(']') {
                            let needles = parse_list_value(&list_val);
                            for needle in needles {
                                if let Value::String(n) = needle {
                                    if s.contains(n.as_str()) {
                                        return Ok(Value::Boolean(true));
                                    }
                                }
                            }
                            return Ok(Value::Boolean(false));
                        }
                        if s.contains(ls.as_str()) {
                            return Ok(Value::Boolean(true));
                        }
                        return Ok(Value::Boolean(false));
                    }
                }
                for arg in &args[1..] {
                    let needle = self.evaluate_expression(arg, row)?;
                    if let Value::String(n) = needle {
                        if s.contains(n.as_str()) {
                            return Ok(Value::Boolean(true));
                        }
                    }
                }
                Ok(Value::Boolean(false))
            }
            "text_starts_with_any" => {
                if args.is_empty() {
                    return Err("text_starts_with_any() requires at least 1 argument".into());
                }
                let s_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let s = match &s_val {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Value::Null),
                };
                if args.len() == 2 {
                    let list_val = self.evaluate_expression(&args[1], row)?;
                    if let Value::String(ref ls) = list_val {
                        if ls.starts_with('[') && ls.ends_with(']') {
                            let prefixes = parse_list_value(&list_val);
                            for prefix in prefixes {
                                if let Value::String(p) = prefix {
                                    if s.starts_with(p.as_str()) {
                                        return Ok(Value::Boolean(true));
                                    }
                                }
                            }
                            return Ok(Value::Boolean(false));
                        }
                        if s.starts_with(ls.as_str()) {
                            return Ok(Value::Boolean(true));
                        }
                        return Ok(Value::Boolean(false));
                    }
                }
                for arg in &args[1..] {
                    let prefix = self.evaluate_expression(arg, row)?;
                    if let Value::String(p) = prefix {
                        if s.starts_with(p.as_str()) {
                            return Ok(Value::Boolean(true));
                        }
                    }
                }
                Ok(Value::Boolean(false))
            }
            // ── String functions ──────────────────────────────────
            "split" => {
                if args.len() != 2 {
                    return Err("split() requires 2 arguments: string, delimiter".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let delim_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &delim_val) {
                    (Value::String(s), Value::String(delim)) => {
                        let parts: Vec<String> = s
                            .split(delim.as_str())
                            .map(|p| {
                                format!("\"{}\"", p.replace('\\', "\\\\").replace('"', "\\\""))
                            })
                            .collect();
                        Ok(Value::String(format!("[{}]", parts.join(", "))))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "replace" => {
                if args.len() != 3 {
                    return Err(
                        "replace() requires 3 arguments: string, search, replacement".into(),
                    );
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let search_val = self.evaluate_expression(&args[1], row)?;
                let replace_val = self.evaluate_expression(&args[2], row)?;
                match (&str_val, &search_val, &replace_val) {
                    (Value::String(s), Value::String(search), Value::String(replacement)) => Ok(
                        Value::String(s.replace(search.as_str(), replacement.as_str())),
                    ),
                    _ => Ok(Value::Null),
                }
            }
            "substring" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(
                        "substring() requires 2-3 arguments: string, start [, length]".into(),
                    );
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let start_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &start_val) {
                    (Value::String(s), Value::Int64(start)) => {
                        let start_idx = (*start).max(0) as usize;
                        let substr: String = if args.len() == 3 {
                            let len_val = self.evaluate_expression(&args[2], row)?;
                            match len_val {
                                Value::Int64(len) => {
                                    let take = (len).max(0) as usize;
                                    s.chars().skip(start_idx).take(take).collect()
                                }
                                _ => return Ok(Value::Null),
                            }
                        } else {
                            s.chars().skip(start_idx).collect()
                        };
                        Ok(Value::String(substr))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "left" => {
                if args.len() != 2 {
                    return Err("left() requires 2 arguments: string, length".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let len_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &len_val) {
                    (Value::String(s), Value::Int64(len)) => {
                        let result: String = s.chars().take(*len as usize).collect();
                        Ok(Value::String(result))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "right" => {
                if args.len() != 2 {
                    return Err("right() requires 2 arguments: string, length".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let len_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &len_val) {
                    (Value::String(s), Value::Int64(len)) => {
                        let char_count = s.chars().count();
                        let skip = char_count.saturating_sub(*len as usize);
                        let result: String = s.chars().skip(skip).collect();
                        Ok(Value::String(result))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "trim" | "btrim" => {
                if args.len() != 1 {
                    return Err("trim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            "ltrim" => {
                if args.len() != 1 {
                    return Err("ltrim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim_start().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            "rtrim" => {
                if args.len() != 1 {
                    return Err("rtrim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim_end().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            "reverse" => {
                if args.len() != 1 {
                    return Err("reverse() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.chars().rev().collect())),
                    _ => Ok(Value::Null),
                }
            }
            // ── List functions ────────────────────────────────────
            "head" => {
                if args.len() != 1 {
                    return Err("head() requires 1 argument".into());
                }
                let val = self.evaluate_expression(&args[0], row)?;
                let items = parse_list_value(&val);
                Ok(items.into_iter().next().unwrap_or(Value::Null))
            }
            "last" => {
                if args.len() != 1 {
                    return Err("last() requires 1 argument".into());
                }
                let val = self.evaluate_expression(&args[0], row)?;
                let items = parse_list_value(&val);
                Ok(items.into_iter().last().unwrap_or(Value::Null))
            }
            // ── Spatial functions ─────────────────────────────────
            "point" => {
                if args.len() != 2 {
                    return Err("point() requires 2 arguments: lat, lon".into());
                }
                let lat = crate::graph::core::value_operations::value_to_f64(
                    &self.evaluate_expression(&args[0], row)?,
                )
                .ok_or("point(): lat must be numeric")?;
                let lon = crate::graph::core::value_operations::value_to_f64(
                    &self.evaluate_expression(&args[1], row)?,
                )
                .ok_or("point(): lon must be numeric")?;
                Ok(Value::Point { lat, lon })
            }
            "distance" => match args.len() {
                2 => {
                    // Resolve via spatial config — prefer_geometry=false so bare
                    // variables resolve as Points; explicit .geometry resolves as Geometry
                    let r1 = self.resolve_spatial(&args[0], row, false)?;
                    let r2 = self.resolve_spatial(&args[1], row, false)?;
                    match (r1, r2) {
                        (
                            Some(ResolvedSpatial::Point(lat1, lon1)),
                            Some(ResolvedSpatial::Point(lat2, lon2)),
                        ) => Ok(Value::Float64(
                            crate::graph::features::spatial::geodesic_distance(
                                lat1, lon1, lat2, lon2,
                            ),
                        )),
                        (
                            Some(ResolvedSpatial::Point(lat, lon)),
                            Some(ResolvedSpatial::Geometry(g, _)),
                        )
                        | (
                            Some(ResolvedSpatial::Geometry(g, _)),
                            Some(ResolvedSpatial::Point(lat, lon)),
                        ) => Ok(Value::Float64(
                            crate::graph::features::spatial::point_to_geometry_distance_m(
                                lat, lon, &g,
                            )?,
                        )),
                        (
                            Some(ResolvedSpatial::Geometry(g1, _)),
                            Some(ResolvedSpatial::Geometry(g2, _)),
                        ) => Ok(Value::Float64(
                            crate::graph::features::spatial::geometry_to_geometry_distance_m(
                                &g1, &g2,
                            )?,
                        )),
                        // One or both sides have no spatial data (e.g. node
                        // exists but geometry field is NULL) → propagate Null
                        // so WHERE distance(a, b) < X simply filters them out.
                        _ => Ok(Value::Null),
                    }
                }
                4 => {
                    let lat1 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[0], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    let lon1 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[1], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    let lat2 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[2], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    let lon2 = crate::graph::core::value_operations::value_to_f64(
                        &self.evaluate_expression(&args[3], row)?,
                    )
                    .ok_or("distance(): args must be numeric")?;
                    Ok(Value::Float64(
                        crate::graph::features::spatial::geodesic_distance(lat1, lon1, lat2, lon2),
                    ))
                }
                _ => Err(
                    "distance() requires 2 (Point, Point) or 4 (lat1, lon1, lat2, lon2) arguments"
                        .into(),
                ),
            },
            // ── Node-aware spatial functions ──────────────────────────
            "contains" => {
                if args.len() != 2 {
                    return Err("contains() requires 2 arguments".into());
                }
                // Arg 1: must be a geometry (the container)
                let resolved1 = self.resolve_spatial(&args[0], row, true)?.ok_or(
                    "contains(): first arg must resolve to a geometry (node, WKT string, or named shape)",
                )?;
                let (geom, bbox1) = match &resolved1 {
                    ResolvedSpatial::Geometry(g, bbox) => (g, bbox),
                    ResolvedSpatial::Point(_, _) => {
                        return Err("contains(): first arg must be a geometry, not a point".into());
                    }
                };
                // Arg 2: prefer point for the contained item (point-in-polygon)
                let resolved2 = self
                    .resolve_spatial(&args[1], row, false)?
                    .ok_or("contains(): second arg must resolve to a point or geometry")?;

                match &resolved2 {
                    ResolvedSpatial::Point(lat, lon) => {
                        // Bbox pre-filter: if the point is outside the container's bbox,
                        // it cannot be inside the polygon. This is O(1) vs O(n_vertices).
                        if let Some(bb) = bbox1 {
                            let pt = geo::Coord { x: *lon, y: *lat };
                            if !bb.min().x.le(&pt.x)
                                || !bb.max().x.ge(&pt.x)
                                || !bb.min().y.le(&pt.y)
                                || !bb.max().y.ge(&pt.y)
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        let pt = geo::Point::new(*lon, *lat);
                        Ok(Value::Boolean(
                            crate::graph::features::spatial::geometry_contains_point(geom, &pt),
                        ))
                    }
                    ResolvedSpatial::Geometry(g2, bbox2) => {
                        // Bbox pre-filter: if bboxes don't overlap, containment is impossible
                        if let (Some(bb1), Some(bb2)) = (bbox1, bbox2) {
                            if bb1.max().x < bb2.min().x
                                || bb2.max().x < bb1.min().x
                                || bb1.max().y < bb2.min().y
                                || bb2.max().y < bb1.min().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        Ok(Value::Boolean(
                            crate::graph::features::spatial::geometry_contains_geometry(geom, g2),
                        ))
                    }
                }
            }
            "intersects" => {
                if args.len() != 2 {
                    return Err("intersects() requires 2 arguments".into());
                }
                let r1 = self.resolve_spatial(&args[0], row, true)?.ok_or(
                    "intersects(): args must resolve to geometries or nodes with spatial config",
                )?;
                let r2 = self.resolve_spatial(&args[1], row, true)?.ok_or(
                    "intersects(): args must resolve to geometries or nodes with spatial config",
                )?;
                // Dispatch without cloning — use Arc references where possible
                let result = match (&r1, &r2) {
                    (
                        ResolvedSpatial::Geometry(g1, bbox1),
                        ResolvedSpatial::Geometry(g2, bbox2),
                    ) => {
                        // Bbox pre-filter: if bboxes don't overlap, no intersection possible
                        if let (Some(bb1), Some(bb2)) = (bbox1, bbox2) {
                            if bb1.max().x < bb2.min().x
                                || bb2.max().x < bb1.min().x
                                || bb1.max().y < bb2.min().y
                                || bb2.max().y < bb1.min().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        crate::graph::features::spatial::geometries_intersect(g1, g2)
                    }
                    (ResolvedSpatial::Point(lat, lon), ResolvedSpatial::Geometry(g, bbox)) => {
                        // Bbox pre-filter for point-vs-geometry
                        if let Some(bb) = bbox {
                            if *lon < bb.min().x
                                || *lon > bb.max().x
                                || *lat < bb.min().y
                                || *lat > bb.max().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        let pt = geo::Geometry::Point(geo::Point::new(*lon, *lat));
                        crate::graph::features::spatial::geometries_intersect(&pt, g)
                    }
                    (ResolvedSpatial::Geometry(g, bbox), ResolvedSpatial::Point(lat, lon)) => {
                        if let Some(bb) = bbox {
                            if *lon < bb.min().x
                                || *lon > bb.max().x
                                || *lat < bb.min().y
                                || *lat > bb.max().y
                            {
                                return Ok(Value::Boolean(false));
                            }
                        }
                        let pt = geo::Geometry::Point(geo::Point::new(*lon, *lat));
                        crate::graph::features::spatial::geometries_intersect(g, &pt)
                    }
                    (ResolvedSpatial::Point(lat1, lon1), ResolvedSpatial::Point(lat2, lon2)) => {
                        lat1 == lat2 && lon1 == lon2
                    }
                };
                Ok(Value::Boolean(result))
            }
            "centroid" => {
                if args.len() != 1 {
                    return Err("centroid() requires 1 argument".into());
                }
                let resolved = self.resolve_spatial(&args[0], row, true)?.ok_or(
                    "centroid(): arg must resolve to a geometry (node, WKT string, or named shape)",
                )?;
                match &resolved {
                    ResolvedSpatial::Point(lat, lon) => Ok(Value::Point {
                        lat: *lat,
                        lon: *lon,
                    }),
                    ResolvedSpatial::Geometry(g, _) => {
                        let (lat, lon) = crate::graph::features::spatial::geometry_centroid(g)?;
                        Ok(Value::Point { lat, lon })
                    }
                }
            }
            "area" => {
                if args.len() != 1 {
                    return Err("area() requires 1 argument".into());
                }
                let resolved = self
                    .resolve_spatial(&args[0], row, true)?
                    .ok_or("area(): arg must resolve to a polygon geometry")?;
                match &resolved {
                    ResolvedSpatial::Geometry(g, _) => Ok(Value::Float64(
                        crate::graph::features::spatial::geometry_area_m2(g)?,
                    )),
                    ResolvedSpatial::Point(_, _) => {
                        Err("area(): arg must be a polygon geometry, not a point".into())
                    }
                }
            }
            "perimeter" => {
                if args.len() != 1 {
                    return Err("perimeter() requires 1 argument".into());
                }
                let resolved = self
                    .resolve_spatial(&args[0], row, true)?
                    .ok_or("perimeter(): arg must resolve to a geometry")?;
                match &resolved {
                    ResolvedSpatial::Geometry(g, _) => Ok(Value::Float64(
                        crate::graph::features::spatial::geometry_perimeter_m(g)?,
                    )),
                    ResolvedSpatial::Point(_, _) => {
                        Err("perimeter(): arg must be a geometry, not a point".into())
                    }
                }
            }
            "latitude" => {
                if args.len() != 1 {
                    return Err("latitude() requires 1 argument".into());
                }
                match self.evaluate_expression(&args[0], row)? {
                    Value::Point { lat, .. } => Ok(Value::Float64(lat)),
                    _ => Err("latitude() requires a Point argument".into()),
                }
            }
            "longitude" => {
                if args.len() != 1 {
                    return Err("longitude() requires 1 argument".into());
                }
                match self.evaluate_expression(&args[0], row)? {
                    Value::Point { lon, .. } => Ok(Value::Float64(lon)),
                    _ => Err("longitude() requires a Point argument".into()),
                }
            }
            // ── Geometry primitives (0.8.20) ──────────────────────────
            "geom_buffer" => {
                if args.len() != 2 {
                    return Err("geom_buffer() requires 2 arguments: (geom, meters)".into());
                }
                let geom = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Value::Null),
                };
                let meters = crate::graph::core::value_operations::value_to_f64(
                    &self.evaluate_expression(&args[1], row)?,
                )
                .ok_or("geom_buffer(): second argument must be numeric (meters)")?;
                let result = crate::graph::features::spatial::geometry_buffer(&geom, meters)?;
                Ok(Value::String(
                    crate::graph::features::spatial::geometry_to_wkt(&result),
                ))
            }
            "geom_convex_hull" => {
                if args.is_empty() {
                    return Err("geom_convex_hull() requires at least 1 argument".into());
                }
                let mut geoms: Vec<geo::Geometry<f64>> = Vec::new();
                // Single list argument: parse list of WKT strings.
                if args.len() == 1 {
                    let val = self.evaluate_expression(&args[0], row)?;
                    if let Value::String(ref s) = val {
                        if s.starts_with('[') && s.ends_with(']') {
                            for item in parse_list_value(&val) {
                                if let Value::String(wkt) = item {
                                    if let Ok(g) = crate::graph::features::spatial::parse_wkt(&wkt)
                                    {
                                        geoms.push(g);
                                    }
                                }
                            }
                        }
                    }
                }
                if geoms.is_empty() {
                    for arg in args {
                        if let Some(g) = self.geom_arg(arg, row)? {
                            geoms.push(g);
                        }
                    }
                }
                if geoms.is_empty() {
                    return Ok(Value::Null);
                }
                let hull = crate::graph::features::spatial::geometries_convex_hull(&geoms)?;
                Ok(Value::String(
                    crate::graph::features::spatial::geometry_to_wkt(&hull),
                ))
            }
            "geom_union" | "geom_intersection" | "geom_difference" => {
                if args.len() != 2 {
                    return Err(format!("{name}() requires 2 arguments: (g1, g2)"));
                }
                let g1 = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Value::Null),
                };
                let g2 = match self.geom_arg(&args[1], row)? {
                    Some(g) => g,
                    None => return Ok(Value::Null),
                };
                let result = match name {
                    "geom_union" => crate::graph::features::spatial::geometry_union(&g1, &g2)?,
                    "geom_intersection" => {
                        crate::graph::features::spatial::geometry_intersection(&g1, &g2)?
                    }
                    "geom_difference" => {
                        crate::graph::features::spatial::geometry_difference(&g1, &g2)?
                    }
                    _ => unreachable!(),
                };
                Ok(Value::String(
                    crate::graph::features::spatial::geometry_to_wkt(&result),
                ))
            }
            "geom_is_valid" => {
                if args.len() != 1 {
                    return Err("geom_is_valid() requires 1 argument".into());
                }
                let geom = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Value::Null),
                };
                Ok(Value::Boolean(
                    crate::graph::features::spatial::geometry_is_valid(&geom),
                ))
            }
            "geom_length" => {
                if args.len() != 1 {
                    return Err("geom_length() requires 1 argument".into());
                }
                let geom = match self.geom_arg(&args[0], row)? {
                    Some(g) => g,
                    None => return Ok(Value::Null),
                };
                Ok(Value::Float64(
                    crate::graph::features::spatial::geometry_length_m(&geom),
                ))
            }
            // vector_score(node, embedding_property, query_vector [, metric])
            // Returns the similarity score (f32→f64) for the node's embedding vs query vector.
            //
            // Performance: The constant arguments (property name, query vector, metric) are
            // parsed once on the first call and cached in self.vs_cache. Subsequent rows
            // skip JSON parsing, String allocation, and metric dispatch entirely.
            "vector_score" => {
                if args.len() < 3 || args.len() > 4 {
                    return Err(
                        "vector_score() requires 3-4 arguments: (node, property, query_vector [, metric])"
                            .into(),
                    );
                }

                // Arg 0: node variable → resolve to NodeIndex (changes per row)
                let node_idx = match &args[0] {
                    Expression::Variable(var) => match row.node_bindings.get(var) {
                        Some(&idx) => idx,
                        None => return Ok(Value::Null),
                    },
                    _ => {
                        return Err("vector_score(): first argument must be a node variable".into())
                    }
                };

                // Get or initialize cache — constant args parsed once, reused for all rows
                let c = match self.vs_cache.get() {
                    Some(c) => c,
                    None => {
                        let prop_name = match self.evaluate_expression(&args[1], row)? {
                            Value::String(s) => s,
                            _ => return Err(
                                "vector_score(): second argument must be a string property name"
                                    .into(),
                            ),
                        };
                        let query_vec = self.extract_float_list(&args[2], row)?;
                        // Resolve metric: explicit arg > stored metric > cosine default
                        let metric_name = if args.len() > 3 {
                            match self.evaluate_expression(&args[3], row)? {
                                Value::String(s) => s,
                                _ => "cosine".to_string(),
                            }
                        } else {
                            // Look up stored metric from the embedding store
                            self.graph
                                .embeddings
                                .iter()
                                .find(|((_, pn), _)| pn == &prop_name)
                                .and_then(|(_, store)| store.metric.clone())
                                .unwrap_or_else(|| "cosine".to_string())
                        };
                        let similarity_fn = match metric_name.as_str() {
                            "cosine" => vs::cosine_similarity as fn(&[f32], &[f32]) -> f32,
                            "dot_product" => vs::dot_product,
                            "euclidean" => vs::neg_euclidean_distance,
                            "poincare" => vs::neg_poincare_distance,
                            other => {
                                return Err(format!(
                                    "vector_score(): unknown metric '{}'. Use 'cosine', 'dot_product', 'euclidean', or 'poincare'.",
                                    other
                                ))
                            }
                        };
                        let _ = self.vs_cache.set(VectorScoreCache {
                            prop_name,
                            query_vec,
                            similarity_fn,
                        });
                        self.vs_cache.get().unwrap()
                    }
                };

                // Per-row: look up node type → embedding store → compute similarity
                let node_type = match self.graph.graph.node_weight(node_idx) {
                    Some(n) => n.node_type_str(&self.graph.interner),
                    None => return Ok(Value::Null),
                };

                let store = match self.graph.embedding_store(node_type, &c.prop_name) {
                    Some(s) => s,
                    None => {
                        return Err(format!(
                            "vector_score(): no embedding '{}' found for node type '{}'",
                            c.prop_name, node_type
                        ))
                    }
                };

                if c.query_vec.len() != store.dimension {
                    return Err(format!(
                        "vector_score(): query vector dimension {} does not match embedding dimension {}",
                        c.query_vec.len(),
                        store.dimension
                    ));
                }

                match store.get_embedding(node_idx.index()) {
                    Some(embedding) => {
                        let score = (c.similarity_fn)(&c.query_vec, embedding);
                        Ok(Value::Float64(score as f64))
                    }
                    None => Ok(Value::Null),
                }
            }
            // ── Timeseries functions ──────────────────────────────────────
            "ts_at" => {
                if args.len() != 2 {
                    return Err("ts_at() requires 2 arguments: (n.channel, '2020-2')".into());
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let date_arg = self.resolve_ts_date_arg(&args[1], row)?;
                match date_arg {
                    Some((date, _prec)) => {
                        match crate::graph::features::timeseries::find_key_index(&ts.keys, date) {
                            Some(idx) => {
                                let v = channel[idx];
                                if v.is_finite() {
                                    Ok(Value::Float64(v))
                                } else {
                                    Ok(Value::Null)
                                }
                            }
                            None => Ok(Value::Null),
                        }
                    }
                    None => Ok(Value::Null), // null date → null
                }
            }
            "ts_sum" | "ts_avg" | "ts_min" | "ts_max" | "ts_count" => {
                if args.is_empty() || args.len() > 3 {
                    return Err(format!(
                        "{}() requires 1-3 arguments: (n.channel [, 'start'] [, 'end'])",
                        name
                    ));
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let (lo, hi) = self.resolve_ts_range(ts, &args[1..], row)?;
                let slice = &channel[lo..hi];
                match name {
                    "ts_sum" => Ok(Value::Float64(crate::graph::features::timeseries::ts_sum(
                        slice,
                    ))),
                    "ts_avg" => {
                        let v = crate::graph::features::timeseries::ts_avg(slice);
                        if v.is_nan() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Float64(v))
                        }
                    }
                    "ts_min" => {
                        let v = crate::graph::features::timeseries::ts_min(slice);
                        if v.is_infinite() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Float64(v))
                        }
                    }
                    "ts_max" => {
                        let v = crate::graph::features::timeseries::ts_max(slice);
                        if v.is_infinite() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Float64(v))
                        }
                    }
                    "ts_count" => Ok(Value::Int64(crate::graph::features::timeseries::ts_count(
                        slice,
                    ) as i64)),
                    _ => unreachable!(),
                }
            }
            "ts_first" => {
                if args.len() != 1 {
                    return Err("ts_first() requires 1 argument: (n.channel)".into());
                }
                let (_, channel, _) = self.resolve_timeseries_channel(&args[0], row)?;
                match channel.iter().find(|v| v.is_finite()) {
                    Some(&v) => Ok(Value::Float64(v)),
                    None => Ok(Value::Null),
                }
            }
            "ts_last" => {
                if args.len() != 1 {
                    return Err("ts_last() requires 1 argument: (n.channel)".into());
                }
                let (_, channel, _) = self.resolve_timeseries_channel(&args[0], row)?;
                match channel.iter().rev().find(|v| v.is_finite()) {
                    Some(&v) => Ok(Value::Float64(v)),
                    None => Ok(Value::Null),
                }
            }
            "ts_delta" => {
                if args.len() != 3 {
                    return Err(
                        "ts_delta() requires 3 arguments: (n.channel, '2019-12', '2021-1')".into(),
                    );
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let a1 = self.resolve_ts_date_arg(&args[1], row)?;
                let a2 = self.resolve_ts_date_arg(&args[2], row)?;
                let v1 = a1.and_then(|(date, prec)| {
                    let end = crate::graph::features::timeseries::expand_end(date, prec);
                    let (lo, hi) = crate::graph::features::timeseries::find_range(
                        &ts.keys,
                        Some(date),
                        Some(end),
                    );
                    if lo < hi { Some(channel[lo]) } else { None }.filter(|v| v.is_finite())
                });
                let v2 = a2.and_then(|(date, prec)| {
                    let end = crate::graph::features::timeseries::expand_end(date, prec);
                    let (lo, hi) = crate::graph::features::timeseries::find_range(
                        &ts.keys,
                        Some(date),
                        Some(end),
                    );
                    if lo < hi { Some(channel[lo]) } else { None }.filter(|v| v.is_finite())
                });
                match (v1, v2) {
                    (Some(a), Some(b)) => Ok(Value::Float64(b - a)),
                    _ => Ok(Value::Null),
                }
            }
            "ts_series" => {
                if args.is_empty() || args.len() > 3 {
                    return Err(
                        "ts_series() requires 1-3 arguments: (n.channel [, 'start'] [, 'end'])"
                            .into(),
                    );
                }
                let (ts, channel, _config) = self.resolve_timeseries_channel(&args[0], row)?;
                let (lo, hi) = self.resolve_ts_range(ts, &args[1..], row)?;
                let mut entries = Vec::with_capacity(hi - lo);
                for (date, &val) in ts.keys[lo..hi].iter().zip(&channel[lo..hi]) {
                    entries.push(format!(
                        "{{\"time\":\"{}\",\"value\":{}}}",
                        date,
                        if val.is_finite() {
                            val.to_string()
                        } else {
                            "null".to_string()
                        }
                    ));
                }
                Ok(Value::String(format!("[{}]", entries.join(","))))
            }
            // ── List functions ────────────────────────────────────
            "range" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(
                        "range() requires 2 or 3 arguments: range(start, end[, step])".into(),
                    );
                }
                let start = as_i64(&self.evaluate_expression(&args[0], row)?)?;
                let end = as_i64(&self.evaluate_expression(&args[1], row)?)?;
                let step = if args.len() == 3 {
                    let s = as_i64(&self.evaluate_expression(&args[2], row)?)?;
                    if s == 0 {
                        return Err("range() step must not be zero".into());
                    }
                    s
                } else {
                    1
                };
                let mut vals = Vec::new();
                let mut cur = start;
                if step > 0 {
                    while cur <= end {
                        vals.push(cur.to_string());
                        cur += step;
                    }
                } else {
                    while cur >= end {
                        vals.push(cur.to_string());
                        cur += step;
                    }
                }
                Ok(Value::String(format!("[{}]", vals.join(","))))
            }

            // ── Numeric math functions ──────────────────────────
            "abs" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Int64(n) => Ok(Value::Int64(n.abs())),
                    Value::Float64(f) => Ok(Value::Float64(f.abs())),
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.abs())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "ceil" | "ceiling" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.ceil())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "floor" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.floor())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "round" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => {
                            if args.len() >= 2 {
                                let prec = self.evaluate_expression(&args[1], row)?;
                                let d = match &prec {
                                    Value::Int64(i) => *i as i32,
                                    Value::Float64(fl) => *fl as i32,
                                    _ => 0,
                                };
                                let factor = 10f64.powi(d);
                                Ok(Value::Float64((f * factor).round() / factor))
                            } else {
                                Ok(Value::Float64(f.round()))
                            }
                        }
                        None => Ok(Value::Null),
                    },
                }
            }
            "sqrt" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f >= 0.0 => Ok(Value::Float64(f.sqrt())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "sign" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f > 0.0 => Ok(Value::Int64(1)),
                        Some(f) if f < 0.0 => Ok(Value::Int64(-1)),
                        Some(_) => Ok(Value::Int64(0)),
                        None => Ok(Value::Null),
                    },
                }
            }
            "log" | "ln" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f > 0.0 => Ok(Value::Float64(f.ln())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "log10" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) if f > 0.0 => Ok(Value::Float64(f.log10())),
                        _ => Ok(Value::Null),
                    },
                }
            }
            "exp" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::Null => Ok(Value::Null),
                    _ => match value_to_f64(&val) {
                        Some(f) => Ok(Value::Float64(f.exp())),
                        None => Ok(Value::Null),
                    },
                }
            }
            "pow" | "power" => {
                if args.len() != 2 {
                    return Err("pow() requires 2 arguments: base, exponent".into());
                }
                let base_val = self.evaluate_expression(&args[0], row)?;
                let exp_val = self.evaluate_expression(&args[1], row)?;
                match (value_to_f64(&base_val), value_to_f64(&exp_val)) {
                    (Some(base), Some(exp)) => Ok(Value::Float64(base.powf(exp))),
                    _ => Ok(Value::Null),
                }
            }
            "pi" => Ok(Value::Float64(std::f64::consts::PI)),
            "rand" | "random" => {
                // Thread-local xorshift64 PRNG. Seeded once per thread from
                // SystemTime mixed with a monotonic per-thread counter;
                // subsequent calls just advance the state. Avoids per-call
                // SystemTime::now() overhead and guarantees distinct values
                // within a tight per-row loop. The counter splat ensures
                // parallel rayon workers don't collide on the same nanosecond.
                use std::cell::Cell;
                use std::sync::atomic::{AtomicU64, Ordering};
                use std::time::SystemTime;
                static THREAD_COUNTER: AtomicU64 = AtomicU64::new(0);
                thread_local! {
                    static XORSHIFT_STATE: Cell<u64> = {
                        let nanos = SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as u64;
                        let counter = THREAD_COUNTER.fetch_add(1, Ordering::Relaxed);
                        // Mix counter via splitmix64-ish avalanche so adjacent
                        // thread IDs produce well-separated seeds.
                        let mut seed = nanos.wrapping_add(counter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        seed ^= seed >> 30;
                        seed = seed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        seed ^= seed >> 27;
                        seed = seed.wrapping_mul(0x94D0_49BB_1331_11EB);
                        seed ^= seed >> 31;
                        Cell::new(seed | 1)
                    };
                }
                let val = XORSHIFT_STATE.with(|state| {
                    let mut x = state.get();
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    state.set(x);
                    // Use top 53 bits → f64 mantissa to avoid precision loss.
                    ((x >> 11) as f64) / ((1u64 << 53) as f64)
                });
                Ok(Value::Float64(val))
            }

            // ── Temporal filtering functions ──────────────────────────────
            "valid_at" => {
                // valid_at(entity, date, 'from_field', 'to_field') → Boolean
                // True when entity.from_field <= date AND entity.to_field >= date.
                // NULL fields = open-ended (always pass).
                if args.len() != 4 {
                    return Err(
                        "valid_at() requires 4 arguments: (entity, date, from_field, to_field)"
                            .into(),
                    );
                }
                let var_name =
                    match &args[0] {
                        Expression::Variable(v) => v,
                        _ => return Err(
                            "valid_at(): first argument must be a node or relationship variable"
                                .into(),
                        ),
                    };
                let date_val = self.evaluate_expression(&args[1], row)?;
                let from_field = match self.evaluate_expression(&args[2], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_at(): from_field (3rd arg) must be a string".into()),
                };
                let to_field = match self.evaluate_expression(&args[3], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_at(): to_field (4th arg) must be a string".into()),
                };
                let from_val = self.resolve_property(var_name, &from_field, row)?;
                let to_val = self.resolve_property(var_name, &to_field, row)?;
                // NULL = open-ended boundary
                let from_ok = match &from_val {
                    Value::Null => true,
                    _ => {
                        evaluate_comparison(&from_val, &ComparisonOp::LessThanEq, &date_val, None)?
                    }
                };
                let to_ok = match &to_val {
                    Value::Null => true,
                    _ => {
                        evaluate_comparison(&to_val, &ComparisonOp::GreaterThanEq, &date_val, None)?
                    }
                };
                Ok(Value::Boolean(from_ok && to_ok))
            }
            "valid_during" => {
                // valid_during(entity, start, end, 'from_field', 'to_field') → Boolean
                // Overlap: entity.from_field <= end AND entity.to_field >= start.
                // NULL fields = open-ended (always pass).
                if args.len() != 5 {
                    return Err(
                        "valid_during() requires 5 arguments: (entity, start, end, from_field, to_field)"
                            .into(),
                    );
                }
                let var_name = match &args[0] {
                    Expression::Variable(v) => v,
                    _ => return Err(
                        "valid_during(): first argument must be a node or relationship variable"
                            .into(),
                    ),
                };
                let start_val = self.evaluate_expression(&args[1], row)?;
                let end_val = self.evaluate_expression(&args[2], row)?;
                let from_field = match self.evaluate_expression(&args[3], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_during(): from_field (4th arg) must be a string".into()),
                };
                let to_field = match self.evaluate_expression(&args[4], row)? {
                    Value::String(s) => s,
                    _ => return Err("valid_during(): to_field (5th arg) must be a string".into()),
                };
                let from_val = self.resolve_property(var_name, &from_field, row)?;
                let to_val = self.resolve_property(var_name, &to_field, row)?;
                // Overlap: entity.from <= query_end AND entity.to >= query_start
                let from_ok = match &from_val {
                    Value::Null => true,
                    _ => evaluate_comparison(&from_val, &ComparisonOp::LessThanEq, &end_val, None)?,
                };
                let to_ok = match &to_val {
                    Value::Null => true,
                    _ => evaluate_comparison(
                        &to_val,
                        &ComparisonOp::GreaterThanEq,
                        &start_val,
                        None,
                    )?,
                };
                Ok(Value::Boolean(from_ok && to_ok))
            }

            // Aggregate functions should not be evaluated per-row
            "count" | "sum" | "avg" | "min" | "max" | "collect" | "mean" | "std" | "stdev" => {
                Err(format!(
                    "Aggregate function '{}' cannot be used outside of RETURN/WITH",
                    name
                ))
            }
            // embedding_norm(node, property) → Float64
            // Returns the L2 norm of the node's embedding vector.
            // Useful for inferring hierarchy depth in Poincaré embeddings
            // (norm close to 0 = root/general, norm close to 1 = leaf/specific).
            "embedding_norm" => {
                if args.len() != 2 {
                    return Err("embedding_norm() requires 2 arguments: (node, property)".into());
                }
                let node_idx = match &args[0] {
                    Expression::Variable(var) => match row.node_bindings.get(var) {
                        Some(&idx) => idx,
                        None => return Ok(Value::Null),
                    },
                    _ => {
                        return Err(
                            "embedding_norm(): first argument must be a node variable".into()
                        )
                    }
                };
                let prop_name = match self.evaluate_expression(&args[1], row)? {
                    Value::String(s) => s,
                    _ => {
                        return Err(
                            "embedding_norm(): second argument must be a string property name"
                                .into(),
                        )
                    }
                };
                let node_type = match self.graph.graph.node_weight(node_idx) {
                    Some(n) => n.node_type_str(&self.graph.interner),
                    None => return Ok(Value::Null),
                };
                let store = match self.graph.embedding_store(node_type, &prop_name) {
                    Some(s) => s,
                    None => {
                        return Err(format!(
                            "embedding_norm(): no embedding '{}' found for node type '{}'",
                            prop_name, node_type
                        ))
                    }
                };
                match store.get_embedding(node_idx.index()) {
                    Some(emb) => {
                        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                        Ok(Value::Float64(norm as f64))
                    }
                    None => Ok(Value::Null),
                }
            }
            "text_score" => Err(
                "text_score() requires set_embedder(). Call g.set_embedder(model) first."
                    .to_string(),
            ),
            _ => Err(format!("Unknown function: {}", name)),
        }
    }

    // ── Timeseries helpers ─────────────────────────────────────────────

    /// Resolve the first argument of a ts_*() function into the node's timeseries
    /// data, the specific channel's values, and the timeseries config.
    /// The argument must be a PropertyAccess (e.g. `f.oil`).
    pub(super) fn resolve_timeseries_channel<'b>(
        &'b self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<
        (
            &'b crate::graph::features::timeseries::NodeTimeseries,
            &'b [f64],
            &'b crate::graph::features::timeseries::TimeseriesConfig,
        ),
        String,
    > {
        let (variable, property) = match expr {
            Expression::PropertyAccess { variable, property } => (variable, property),
            _ => {
                return Err(
                    "ts_*() first argument must be a property access (e.g. n.channel)".into(),
                )
            }
        };
        let node_idx = row
            .node_bindings
            .get(variable)
            .ok_or_else(|| format!("ts_*(): variable '{}' is not bound to a node", variable))?;
        let ts = self
            .graph
            .get_node_timeseries(node_idx.index())
            .ok_or_else(|| format!("ts_*(): node '{}' has no timeseries data", variable))?;
        let channel = ts.channels.get(property.as_str()).ok_or_else(|| {
            let available: Vec<&str> = ts.channels.keys().map(|s| s.as_str()).collect();
            format!(
                "ts_*(): channel '{}' not found on node '{}'. Available: {:?}",
                property, variable, available
            )
        })?;
        // Look up the config for this node type
        let node = self
            .graph
            .graph
            .node_weight(*node_idx)
            .ok_or("ts_*(): node not found in graph")?;
        let node_type_str = node.node_type_str(&self.graph.interner);
        let config = self
            .graph
            .timeseries_configs
            .get(node_type_str)
            .ok_or_else(|| {
                format!(
                    "ts_*(): no timeseries config for node type '{}'",
                    node_type_str
                )
            })?;
        Ok((ts, channel, config))
    }

    /// Parse a date argument from a ts_*() function call.
    /// Accepts string date queries, integer years, DateTime values, and Null.
    pub(super) fn resolve_ts_date_arg(
        &self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<
        Option<(
            chrono::NaiveDate,
            crate::graph::features::timeseries::DatePrecision,
        )>,
        String,
    > {
        let v = self.evaluate_expression(expr, row)?;
        match &v {
            Value::String(s) => crate::graph::features::timeseries::parse_date_query(s).map(Some),
            Value::Int64(year) => {
                let date = chrono::NaiveDate::from_ymd_opt(*year as i32, 1, 1)
                    .ok_or_else(|| format!("ts_*() invalid year: {}", year))?;
                Ok(Some((
                    date,
                    crate::graph::features::timeseries::DatePrecision::Year,
                )))
            }
            Value::DateTime(date) => Ok(Some((
                *date,
                crate::graph::features::timeseries::DatePrecision::Day,
            ))),
            Value::Null => Ok(None),
            _ => Err(format!(
                "ts_*() date argument must be a string, integer, date, or null, got {:?}",
                v
            )),
        }
    }

    /// Resolve 0-2 range arguments into a `(start_idx, end_idx)` slice range.
    pub(super) fn resolve_ts_range(
        &self,
        ts: &crate::graph::features::timeseries::NodeTimeseries,
        range_args: &[Expression],
        row: &ResultRow,
    ) -> Result<(usize, usize), String> {
        if range_args.is_empty() {
            return Ok((0, ts.keys.len()));
        }

        let first = self.resolve_ts_date_arg(&range_args[0], row)?;

        if range_args.len() >= 2 {
            // Two-arg range: [start, end]
            let second = self.resolve_ts_date_arg(&range_args[1], row)?;
            let start = first.map(|(d, _)| d);
            let end =
                second.map(|(d, prec)| crate::graph::features::timeseries::expand_end(d, prec));
            Ok(crate::graph::features::timeseries::find_range(
                &ts.keys, start, end,
            ))
        } else {
            // Single arg: expand to full precision range
            match first {
                Some((date, prec)) => {
                    let end = crate::graph::features::timeseries::expand_end(date, prec);
                    Ok(crate::graph::features::timeseries::find_range(
                        &ts.keys,
                        Some(date),
                        Some(end),
                    ))
                }
                None => Ok((0, ts.keys.len())), // null = no bounds
            }
        }
    }

    /// Extract a Vec<f32> from an expression that is either a ListLiteral or a JSON string.
    pub(super) fn extract_float_list(
        &self,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<Vec<f32>, String> {
        match expr {
            Expression::ListLiteral(items) => {
                let mut result = Vec::with_capacity(items.len());
                for item in items {
                    match self.evaluate_expression(item, row)? {
                        Value::Float64(f) => result.push(f as f32),
                        Value::Int64(i) => result.push(i as f32),
                        other => {
                            return Err(format!(
                                "vector_score(): query vector elements must be numeric, got {:?}",
                                other
                            ))
                        }
                    }
                }
                Ok(result)
            }
            _ => {
                // Evaluate and try to parse from JSON string "[1.0, 2.0, ...]"
                let val = self.evaluate_expression(expr, row)?;
                match val {
                    Value::String(s) => parse_json_float_list(&s),
                    _ => Err("vector_score(): query vector must be a list of numbers".into()),
                }
            }
        }
    }

    // ========================================================================
    // RETURN
    // ========================================================================
}
