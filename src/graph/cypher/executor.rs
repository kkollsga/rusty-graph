// src/graph/cypher/executor.rs
// Pipeline executor for Cypher queries

use super::ast::*;
use super::result::*;
use crate::datatypes::values::Value;
use crate::graph::filtering_methods;
use crate::graph::pattern_matching::{MatchBinding, PatternExecutor, PatternMatch};
use crate::graph::schema::{DirGraph, NodeData};
use crate::graph::value_operations;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Executor
// ============================================================================

pub struct CypherExecutor<'a> {
    graph: &'a DirGraph,
}

impl<'a> CypherExecutor<'a> {
    pub fn new(graph: &'a DirGraph) -> Self {
        CypherExecutor { graph }
    }

    /// Execute a parsed Cypher query
    pub fn execute(&self, query: &CypherQuery) -> Result<CypherResult, String> {
        let mut result_set = ResultSet::new();

        for clause in &query.clauses {
            result_set = match clause {
                Clause::Match(m) => self.execute_match(m, result_set)?,
                Clause::OptionalMatch(m) => self.execute_optional_match(m, result_set)?,
                Clause::Where(w) => self.execute_where(w, result_set)?,
                Clause::Return(r) => self.execute_return(r, result_set)?,
                Clause::With(w) => self.execute_with(w, result_set)?,
                Clause::OrderBy(o) => self.execute_order_by(o, result_set)?,
                Clause::Limit(l) => self.execute_limit(l, result_set)?,
                Clause::Skip(s) => self.execute_skip(s, result_set)?,
                Clause::Unwind(u) => self.execute_unwind(u, result_set)?,
                Clause::Union(u) => self.execute_union(u, result_set)?,
                Clause::Create(_) | Clause::Set(_) | Clause::Delete(_) => {
                    return Err(
                        "Mutation queries (CREATE/SET/DELETE) are not yet supported".to_string()
                    );
                }
            };
        }

        // Convert ResultSet to CypherResult
        self.finalize_result(result_set)
    }

    // ========================================================================
    // MATCH
    // ========================================================================

    fn execute_match(
        &self,
        clause: &MatchClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        if existing.rows.is_empty() {
            // First MATCH: execute patterns to produce initial bindings
            let mut all_rows = Vec::new();

            for pattern in &clause.patterns {
                let executor = PatternExecutor::new(self.graph, None);
                let matches = executor.execute(pattern)?;

                if all_rows.is_empty() {
                    // First pattern - create initial rows
                    for m in matches {
                        all_rows.push(self.pattern_match_to_row(m));
                    }
                } else {
                    // Cross-product with existing rows
                    let mut new_rows = Vec::with_capacity(all_rows.len() * matches.len());
                    for existing_row in &all_rows {
                        for m in &matches {
                            let mut new_row = existing_row.clone();
                            self.merge_match_into_row(&mut new_row, m);
                            new_rows.push(new_row);
                        }
                    }
                    all_rows = new_rows;
                }
            }

            Ok(ResultSet {
                rows: all_rows,
                columns: existing.columns,
            })
        } else {
            // Subsequent MATCH: expand each existing row with new patterns
            let mut new_rows = Vec::new();

            for row in &existing.rows {
                for pattern in &clause.patterns {
                    let executor = PatternExecutor::new(self.graph, None);
                    let matches = executor.execute(pattern)?;

                    for m in matches {
                        // Only keep matches compatible with existing bindings
                        if !self.bindings_compatible(row, &m) {
                            continue;
                        }
                        let mut new_row = row.clone();
                        self.merge_match_into_row(&mut new_row, &m);
                        new_rows.push(new_row);
                    }
                }
            }

            Ok(ResultSet {
                rows: new_rows,
                columns: existing.columns,
            })
        }
    }

    /// Convert a PatternMatch to a lightweight ResultRow
    fn pattern_match_to_row(&self, m: PatternMatch) -> ResultRow {
        let mut row = ResultRow::new();

        for (var, binding) in m.bindings {
            match binding {
                MatchBinding::Node { index, .. } => {
                    row.node_bindings.insert(var, index);
                }
                MatchBinding::Edge {
                    source,
                    target,
                    connection_type,
                    properties,
                } => {
                    row.edge_bindings.insert(
                        var,
                        EdgeBinding {
                            source,
                            target,
                            connection_type,
                            properties,
                        },
                    );
                }
                MatchBinding::VariableLengthPath {
                    source,
                    target,
                    hops,
                    path,
                } => {
                    row.path_bindings.insert(
                        var,
                        PathBinding {
                            source,
                            target,
                            hops,
                            path,
                        },
                    );
                }
            }
        }

        row
    }

    /// Merge a PatternMatch's bindings into an existing ResultRow
    fn merge_match_into_row(&self, row: &mut ResultRow, m: &PatternMatch) {
        for (var, binding) in &m.bindings {
            match binding {
                MatchBinding::Node { index, .. } => {
                    row.node_bindings.insert(var.clone(), *index);
                }
                MatchBinding::Edge {
                    source,
                    target,
                    connection_type,
                    properties,
                } => {
                    row.edge_bindings.insert(
                        var.clone(),
                        EdgeBinding {
                            source: *source,
                            target: *target,
                            connection_type: connection_type.clone(),
                            properties: properties.clone(),
                        },
                    );
                }
                MatchBinding::VariableLengthPath {
                    source,
                    target,
                    hops,
                    path,
                } => {
                    row.path_bindings.insert(
                        var.clone(),
                        PathBinding {
                            source: *source,
                            target: *target,
                            hops: *hops,
                            path: path.clone(),
                        },
                    );
                }
            }
        }
    }

    // ========================================================================
    // OPTIONAL MATCH
    // ========================================================================

    fn execute_optional_match(
        &self,
        clause: &MatchClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        if existing.rows.is_empty() {
            // OPTIONAL MATCH as first clause acts like regular MATCH
            return self.execute_match(clause, existing);
        }

        let mut new_rows = Vec::new();

        for row in &existing.rows {
            let mut found_any = false;

            for pattern in &clause.patterns {
                let executor = PatternExecutor::new(self.graph, None);
                let matches = executor.execute(pattern)?;

                for m in &matches {
                    // Only keep matches compatible with existing bindings
                    if !self.bindings_compatible(row, m) {
                        continue;
                    }
                    let mut new_row = row.clone();
                    self.merge_match_into_row(&mut new_row, m);
                    new_rows.push(new_row);
                    found_any = true;
                }
            }

            if !found_any {
                // Keep the row - OPTIONAL MATCH produces NULLs for unmatched variables
                new_rows.push(row.clone());
            }
        }

        Ok(ResultSet {
            rows: new_rows,
            columns: existing.columns,
        })
    }

    /// Check if a pattern match is compatible with existing bindings in a row.
    /// If a variable is already bound to a node, the match must bind it to the same node.
    fn bindings_compatible(&self, row: &ResultRow, m: &PatternMatch) -> bool {
        for (var, binding) in &m.bindings {
            if let Some(&existing_idx) = row.node_bindings.get(var) {
                // Variable already bound - check it matches
                match binding {
                    MatchBinding::Node { index, .. } => {
                        if *index != existing_idx {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }

    // ========================================================================
    // WHERE
    // ========================================================================

    fn execute_where(
        &self,
        clause: &WhereClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Try index-accelerated filtering for simple equality predicates
        let index_filters = self.extract_indexable_predicates(&clause.predicate);
        for (variable, property, value) in &index_filters {
            if let Some(node_type) = self.infer_node_type(variable, &result_set) {
                if let Some(matching_indices) =
                    self.graph.lookup_by_index(&node_type, property, value)
                {
                    let index_set: HashSet<petgraph::graph::NodeIndex> =
                        matching_indices.into_iter().collect();
                    result_set.rows.retain(|row| {
                        row.node_bindings
                            .get(variable.as_str())
                            .is_some_and(|idx| index_set.contains(idx))
                    });
                }
            }
        }

        // Apply full predicate evaluation for remaining/non-indexable conditions
        result_set.rows.retain(|row| {
            self.evaluate_predicate(&clause.predicate, row)
                .unwrap_or(false)
        });
        Ok(result_set)
    }

    /// Extract simple equality predicates (variable.property = literal) from AND-trees.
    fn extract_indexable_predicates(&self, predicate: &Predicate) -> Vec<(String, String, Value)> {
        let mut results = Vec::new();
        Self::collect_indexable(predicate, &mut results);
        results
    }

    fn collect_indexable(predicate: &Predicate, results: &mut Vec<(String, String, Value)>) {
        match predicate {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => {
                if *operator == ComparisonOp::Equals {
                    if let (
                        Expression::PropertyAccess { variable, property },
                        Expression::Literal(value),
                    ) = (left, right)
                    {
                        results.push((variable.clone(), property.clone(), value.clone()));
                    } else if let (
                        Expression::Literal(value),
                        Expression::PropertyAccess { variable, property },
                    ) = (left, right)
                    {
                        results.push((variable.clone(), property.clone(), value.clone()));
                    }
                }
            }
            Predicate::And(left, right) => {
                Self::collect_indexable(left, results);
                Self::collect_indexable(right, results);
            }
            _ => {}
        }
    }

    /// Infer the node type for a variable by checking the first row's binding.
    fn infer_node_type(&self, variable: &str, result_set: &ResultSet) -> Option<String> {
        result_set.rows.iter().find_map(|row| {
            row.node_bindings
                .get(variable)
                .and_then(|&idx| self.graph.graph.node_weight(idx))
                .map(|node| match node {
                    NodeData::Regular { node_type, .. } | NodeData::Schema { node_type, .. } => {
                        node_type.clone()
                    }
                })
        })
    }

    fn evaluate_predicate(&self, pred: &Predicate, row: &ResultRow) -> Result<bool, String> {
        match pred {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => {
                let left_val = self.evaluate_expression(left, row)?;
                let right_val = self.evaluate_expression(right, row)?;
                Ok(evaluate_comparison(&left_val, operator, &right_val))
            }
            Predicate::And(left, right) => {
                // Short-circuit: if left is false, skip right
                if !self.evaluate_predicate(left, row)? {
                    return Ok(false);
                }
                self.evaluate_predicate(right, row)
            }
            Predicate::Or(left, right) => {
                // Short-circuit: if left is true, skip right
                if self.evaluate_predicate(left, row)? {
                    return Ok(true);
                }
                self.evaluate_predicate(right, row)
            }
            Predicate::Not(inner) => Ok(!self.evaluate_predicate(inner, row)?),
            Predicate::IsNull(expr) => {
                let val = self.evaluate_expression(expr, row)?;
                Ok(matches!(val, Value::Null))
            }
            Predicate::IsNotNull(expr) => {
                let val = self.evaluate_expression(expr, row)?;
                Ok(!matches!(val, Value::Null))
            }
            Predicate::In { expr, list } => {
                let val = self.evaluate_expression(expr, row)?;
                for item in list {
                    let item_val = self.evaluate_expression(item, row)?;
                    if filtering_methods::values_equal(&val, &item_val) {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Predicate::StartsWith { expr, pattern } => {
                let val = self.evaluate_expression(expr, row)?;
                let pat = self.evaluate_expression(pattern, row)?;
                match (&val, &pat) {
                    (Value::String(s), Value::String(p)) => Ok(s.starts_with(p.as_str())),
                    _ => Ok(false),
                }
            }
            Predicate::EndsWith { expr, pattern } => {
                let val = self.evaluate_expression(expr, row)?;
                let pat = self.evaluate_expression(pattern, row)?;
                match (&val, &pat) {
                    (Value::String(s), Value::String(p)) => Ok(s.ends_with(p.as_str())),
                    _ => Ok(false),
                }
            }
            Predicate::Contains { expr, pattern } => {
                let val = self.evaluate_expression(expr, row)?;
                let pat = self.evaluate_expression(pattern, row)?;
                match (&val, &pat) {
                    (Value::String(s), Value::String(p)) => Ok(s.contains(p.as_str())),
                    _ => Ok(false),
                }
            }
        }
    }

    // ========================================================================
    // Expression Evaluation
    // ========================================================================

    /// Evaluate an expression against a row, resolving property access via NodeIndex
    fn evaluate_expression(&self, expr: &Expression, row: &ResultRow) -> Result<Value, String> {
        match expr {
            Expression::PropertyAccess { variable, property } => {
                self.resolve_property(variable, property, row)
            }
            Expression::Variable(name) => {
                // Check projected values first (from WITH)
                if let Some(val) = row.projected.get(name) {
                    return Ok(val.clone());
                }
                // For node variables, return a representative value (the node's title)
                if let Some(&idx) = row.node_bindings.get(name) {
                    if let Some(node) = self.graph.graph.node_weight(idx) {
                        return Ok(node_to_map_value(node));
                    }
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
            Expression::Negate(inner) => {
                let val = self.evaluate_expression(inner, row)?;
                Ok(arithmetic_negate(&val))
            }
            Expression::FunctionCall { name, args, .. } => {
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
                let formatted: Vec<String> = vals.iter().map(format_value_compact).collect();
                Ok(Value::String(format!("[{}]", formatted.join(", "))))
            }
        }
    }

    /// Resolve property access: variable.property
    /// Uses zero-copy get_field_ref when possible
    fn resolve_property(
        &self,
        variable: &str,
        property: &str,
        row: &ResultRow,
    ) -> Result<Value, String> {
        // Check projected values first (from WITH)
        if let Some(val) = row.projected.get(variable) {
            // Projected values are flat - property access not applicable
            return Ok(val.clone());
        }

        // Node variable
        if let Some(&idx) = row.node_bindings.get(variable) {
            if let Some(node) = self.graph.graph.node_weight(idx) {
                return Ok(resolve_node_property(node, property));
            }
            return Ok(Value::Null); // Node was deleted?
        }

        // Edge variable
        if let Some(edge) = row.edge_bindings.get(variable) {
            return Ok(resolve_edge_property(edge, property));
        }

        // Path variable
        if let Some(path) = row.path_bindings.get(variable) {
            return match property {
                "length" | "hops" => Ok(Value::Int64(path.hops as i64)),
                _ => Ok(Value::Null),
            };
        }

        // Variable not found - might be OPTIONAL MATCH null
        Ok(Value::Null)
    }

    /// Evaluate scalar (non-aggregate) functions
    fn evaluate_scalar_function(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        match name.to_lowercase().as_str() {
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
            "size" | "length" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => Ok(Value::Int64(s.len() as i64)),
                    _ => Ok(Value::Null),
                }
            }
            "type" => {
                // type(r) returns the relationship type
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(edge) = row.edge_bindings.get(var) {
                        return Ok(Value::String(edge.connection_type.clone()));
                    }
                }
                Ok(Value::Null)
            }
            "id" => {
                // id(n) returns the node id
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            return Ok(resolve_node_property(node, "id"));
                        }
                    }
                }
                Ok(Value::Null)
            }
            "labels" => {
                // labels(n) returns node type
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            return Ok(resolve_node_property(node, "type"));
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
            // Aggregate functions should not be evaluated per-row
            "count" | "sum" | "avg" | "min" | "max" | "collect" | "mean" | "std" => Err(format!(
                "Aggregate function '{}' cannot be used outside of RETURN/WITH",
                name
            )),
            _ => Err(format!("Unknown function: {}", name)),
        }
    }

    // ========================================================================
    // RETURN
    // ========================================================================

    fn execute_return(
        &self,
        clause: &ReturnClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let has_aggregation = clause
            .items
            .iter()
            .any(|item| is_aggregate_expression(&item.expression));

        if has_aggregation {
            self.execute_return_with_aggregation(clause, result_set)
        } else {
            self.execute_return_projection(clause, result_set)
        }
    }

    /// Simple projection without aggregation
    fn execute_return_projection(
        &self,
        clause: &ReturnClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let columns: Vec<String> = clause.items.iter().map(return_item_column_name).collect();

        let mut rows = Vec::with_capacity(result_set.rows.len());

        for row in &result_set.rows {
            let mut new_row = row.clone();
            for item in &clause.items {
                let key = return_item_column_name(item);
                let val = self.evaluate_expression(&item.expression, row)?;
                new_row.projected.insert(key, val);
            }
            rows.push(new_row);
        }

        // Handle DISTINCT
        if clause.distinct {
            let mut seen = HashSet::new();
            rows.retain(|row| {
                let key: Vec<String> = columns
                    .iter()
                    .map(|col| format_value_compact(row.projected.get(col).unwrap_or(&Value::Null)))
                    .collect();
                seen.insert(key)
            });
        }

        Ok(ResultSet { rows, columns })
    }

    /// RETURN with aggregation (grouping + aggregate functions)
    fn execute_return_with_aggregation(
        &self,
        clause: &ReturnClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Identify grouping keys (non-aggregate expressions) and aggregations
        let group_key_indices: Vec<usize> = clause
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| !is_aggregate_expression(&item.expression))
            .map(|(i, _)| i)
            .collect();

        let columns: Vec<String> = clause.items.iter().map(return_item_column_name).collect();

        // Special case: no grouping keys = aggregate over all rows
        if group_key_indices.is_empty() {
            let mut projected = HashMap::new();
            for item in &clause.items {
                let key = return_item_column_name(item);
                let val = self.evaluate_aggregate(&item.expression, &result_set.rows)?;
                projected.insert(key, val);
            }
            return Ok(ResultSet {
                rows: vec![ResultRow::from_projected(projected)],
                columns,
            });
        }

        // Group rows by grouping key values
        let mut groups: Vec<(Vec<Value>, Vec<usize>)> = Vec::new();
        let mut group_index_map: HashMap<Vec<String>, usize> = HashMap::new();

        for (row_idx, row) in result_set.rows.iter().enumerate() {
            let key_values: Vec<Value> = group_key_indices
                .iter()
                .map(|&i| {
                    self.evaluate_expression(&clause.items[i].expression, row)
                        .unwrap_or(Value::Null)
                })
                .collect();

            let key_strings: Vec<String> = key_values.iter().map(format_value_compact).collect();

            if let Some(&group_idx) = group_index_map.get(&key_strings) {
                groups[group_idx].1.push(row_idx);
            } else {
                let group_idx = groups.len();
                group_index_map.insert(key_strings, group_idx);
                groups.push((key_values, vec![row_idx]));
            }
        }

        // Compute results for each group
        let mut result_rows = Vec::with_capacity(groups.len());

        for (group_key_values, row_indices) in &groups {
            let group_rows: Vec<&ResultRow> =
                row_indices.iter().map(|&i| &result_set.rows[i]).collect();

            let mut projected = HashMap::new();

            // Add group key values
            for (ki, &item_idx) in group_key_indices.iter().enumerate() {
                let key = return_item_column_name(&clause.items[item_idx]);
                projected.insert(key, group_key_values[ki].clone());
            }

            // Compute aggregations
            for (item_idx, item) in clause.items.iter().enumerate() {
                if group_key_indices.contains(&item_idx) {
                    continue; // Already added
                }
                let key = return_item_column_name(item);
                let val = self.evaluate_aggregate_with_rows(&item.expression, &group_rows)?;
                projected.insert(key, val);
            }

            result_rows.push(ResultRow::from_projected(projected));
        }

        // Handle DISTINCT
        if clause.distinct {
            let mut seen = HashSet::new();
            result_rows.retain(|row| {
                let key: Vec<String> = columns
                    .iter()
                    .map(|col| format_value_compact(row.projected.get(col).unwrap_or(&Value::Null)))
                    .collect();
                seen.insert(key)
            });
        }

        Ok(ResultSet {
            rows: result_rows,
            columns,
        })
    }

    /// Evaluate aggregate function over all rows in a ResultSet
    fn evaluate_aggregate(&self, expr: &Expression, rows: &[ResultRow]) -> Result<Value, String> {
        let refs: Vec<&ResultRow> = rows.iter().collect();
        self.evaluate_aggregate_with_rows(expr, &refs)
    }

    /// Evaluate aggregate function over a slice of row references
    fn evaluate_aggregate_with_rows(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
    ) -> Result<Value, String> {
        match expr {
            Expression::FunctionCall {
                name,
                args,
                distinct,
            } => match name.to_lowercase().as_str() {
                "count" => {
                    if args.len() == 1 && matches!(args[0], Expression::Star) {
                        Ok(Value::Int64(rows.len() as i64))
                    } else {
                        let mut count = 0i64;
                        let mut seen = HashSet::new();
                        for row in rows {
                            let val = self.evaluate_expression(&args[0], row)?;
                            if !matches!(val, Value::Null) {
                                if *distinct {
                                    if seen.insert(format_value_compact(&val)) {
                                        count += 1;
                                    }
                                } else {
                                    count += 1;
                                }
                            }
                        }
                        Ok(Value::Int64(count))
                    }
                }
                "sum" => {
                    let values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    if values.is_empty() {
                        Ok(Value::Int64(0))
                    } else {
                        Ok(Value::Float64(values.iter().sum()))
                    }
                }
                "avg" | "mean" | "average" => {
                    let values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    if values.is_empty() {
                        Ok(Value::Null)
                    } else {
                        Ok(Value::Float64(
                            values.iter().sum::<f64>() / values.len() as f64,
                        ))
                    }
                }
                "min" => {
                    let mut min_val: Option<Value> = None;
                    for row in rows {
                        let val = self.evaluate_expression(&args[0], row)?;
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        min_val = Some(match min_val {
                            None => val,
                            Some(current) => {
                                if filtering_methods::compare_values(&val, &current)
                                    == Some(std::cmp::Ordering::Less)
                                {
                                    val
                                } else {
                                    current
                                }
                            }
                        });
                    }
                    Ok(min_val.unwrap_or(Value::Null))
                }
                "max" => {
                    let mut max_val: Option<Value> = None;
                    for row in rows {
                        let val = self.evaluate_expression(&args[0], row)?;
                        if matches!(val, Value::Null) {
                            continue;
                        }
                        max_val = Some(match max_val {
                            None => val,
                            Some(current) => {
                                if filtering_methods::compare_values(&val, &current)
                                    == Some(std::cmp::Ordering::Greater)
                                {
                                    val
                                } else {
                                    current
                                }
                            }
                        });
                    }
                    Ok(max_val.unwrap_or(Value::Null))
                }
                "collect" => {
                    let mut values = Vec::new();
                    let mut seen = HashSet::new();
                    for row in rows {
                        let val = self.evaluate_expression(&args[0], row)?;
                        if !matches!(val, Value::Null) {
                            if *distinct {
                                let key = format_value_compact(&val);
                                if !seen.insert(key) {
                                    continue;
                                }
                            }
                            values.push(format_value_compact(&val));
                        }
                    }
                    Ok(Value::String(format!("[{}]", values.join(", "))))
                }
                "std" => {
                    let values = self.collect_numeric_values(&args[0], rows, *distinct)?;
                    if values.len() < 2 {
                        Ok(Value::Null)
                    } else {
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                            / (values.len() - 1) as f64;
                        Ok(Value::Float64(variance.sqrt()))
                    }
                }
                _ => Err(format!("Unknown aggregate function: {}", name)),
            },
            // Non-aggregate expression in an aggregation context - evaluate with first row
            _ => {
                if let Some(row) = rows.first() {
                    self.evaluate_expression(expr, row)
                } else {
                    Ok(Value::Null)
                }
            }
        }
    }

    /// Collect numeric values from rows for aggregate computation
    fn collect_numeric_values(
        &self,
        expr: &Expression,
        rows: &[&ResultRow],
        distinct: bool,
    ) -> Result<Vec<f64>, String> {
        let mut values = Vec::new();
        let mut seen = HashSet::new();

        for row in rows {
            let val = self.evaluate_expression(expr, row)?;
            if let Some(f) = value_to_f64(&val) {
                if distinct {
                    let bits = f.to_bits();
                    if !seen.insert(bits) {
                        continue;
                    }
                }
                values.push(f);
            }
        }

        Ok(values)
    }

    // ========================================================================
    // WITH
    // ========================================================================

    fn execute_with(
        &self,
        clause: &WithClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // WITH is essentially RETURN that continues the pipeline
        let return_clause = ReturnClause {
            items: clause.items.clone(),
            distinct: clause.distinct,
        };
        let mut projected = self.execute_return(&return_clause, result_set)?;

        // Apply optional WHERE
        if let Some(ref where_clause) = clause.where_clause {
            projected = self.execute_where(where_clause, projected)?;
        }

        Ok(projected)
    }

    // ========================================================================
    // ORDER BY
    // ========================================================================

    fn execute_order_by(
        &self,
        clause: &OrderByClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Pre-compute sort keys for each row to avoid repeated evaluation
        let sort_keys: Vec<Vec<Value>> = result_set
            .rows
            .iter()
            .map(|row| {
                clause
                    .items
                    .iter()
                    .map(|item| {
                        self.evaluate_expression(&item.expression, row)
                            .unwrap_or(Value::Null)
                    })
                    .collect()
            })
            .collect();

        // Create indices and sort them
        let mut indices: Vec<usize> = (0..result_set.rows.len()).collect();
        indices.sort_by(|&a, &b| {
            for (i, item) in clause.items.iter().enumerate() {
                if let Some(ordering) =
                    filtering_methods::compare_values(&sort_keys[a][i], &sort_keys[b][i])
                {
                    let ordering = if item.ascending {
                        ordering
                    } else {
                        ordering.reverse()
                    };
                    if ordering != std::cmp::Ordering::Equal {
                        return ordering;
                    }
                }
            }
            std::cmp::Ordering::Equal
        });

        // Reorder rows
        let mut sorted_rows = Vec::with_capacity(result_set.rows.len());
        let mut old_rows = std::mem::take(&mut result_set.rows);
        // Use index-based reordering
        let mut temp = Vec::with_capacity(old_rows.len());
        std::mem::swap(&mut temp, &mut old_rows);
        let mut indexed: Vec<Option<ResultRow>> = temp.into_iter().map(Some).collect();
        for &idx in &indices {
            if let Some(row) = indexed[idx].take() {
                sorted_rows.push(row);
            }
        }
        // Drop sort_keys
        drop(sort_keys);

        result_set.rows = sorted_rows;
        Ok(result_set)
    }

    // ========================================================================
    // LIMIT / SKIP
    // ========================================================================

    fn execute_limit(
        &self,
        clause: &LimitClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let n = match self.evaluate_expression(&clause.count, &ResultRow::new())? {
            Value::Int64(n) if n >= 0 => n as usize,
            _ => return Err("LIMIT requires a non-negative integer".to_string()),
        };
        result_set.rows.truncate(n);
        Ok(result_set)
    }

    fn execute_skip(
        &self,
        clause: &SkipClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let n = match self.evaluate_expression(&clause.count, &ResultRow::new())? {
            Value::Int64(n) if n >= 0 => n as usize,
            _ => return Err("SKIP requires a non-negative integer".to_string()),
        };
        if n < result_set.rows.len() {
            result_set.rows = result_set.rows.split_off(n);
        } else {
            result_set.rows.clear();
        }
        Ok(result_set)
    }

    // ========================================================================
    // UNWIND
    // ========================================================================

    fn execute_unwind(
        &self,
        clause: &UnwindClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        let mut new_rows = Vec::new();

        let source_rows = if result_set.rows.is_empty() {
            // UNWIND as first clause - create a single empty row
            vec![ResultRow::new()]
        } else {
            result_set.rows
        };

        for row in &source_rows {
            let val = self.evaluate_expression(&clause.expression, row)?;
            // Currently we only support list literals producing strings
            // In the future, this should work with actual list types
            match val {
                Value::String(s) if s.starts_with('[') && s.ends_with(']') => {
                    // Parse the string list representation
                    let inner = &s[1..s.len() - 1];
                    if inner.is_empty() {
                        continue;
                    }
                    for item in inner.split(", ") {
                        let mut new_row = row.clone();
                        let parsed_val = parse_value_string(item.trim());
                        new_row.projected.insert(clause.alias.clone(), parsed_val);
                        new_rows.push(new_row);
                    }
                }
                _ => {
                    // Single value - just add it
                    let mut new_row = row.clone();
                    new_row.projected.insert(clause.alias.clone(), val);
                    new_rows.push(new_row);
                }
            }
        }

        Ok(ResultSet {
            rows: new_rows,
            columns: result_set.columns,
        })
    }

    // ========================================================================
    // UNION
    // ========================================================================

    fn execute_union(
        &self,
        clause: &UnionClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Execute the right side query
        let right_result = self.execute(&clause.query)?;

        // Combine columns (should be compatible)
        let columns = if result_set.columns.is_empty() {
            right_result.columns.clone()
        } else {
            result_set.columns.clone()
        };

        // Convert right result back to ResultSet
        let mut combined_rows = result_set.rows;
        for row_values in right_result.rows {
            let mut projected = HashMap::new();
            for (i, col) in right_result.columns.iter().enumerate() {
                if let Some(val) = row_values.get(i) {
                    projected.insert(col.clone(), val.clone());
                }
            }
            combined_rows.push(ResultRow::from_projected(projected));
        }

        // Remove duplicates for UNION (not UNION ALL)
        if !clause.all {
            let mut seen = HashSet::new();
            combined_rows.retain(|row| {
                let key: Vec<String> = columns
                    .iter()
                    .map(|col| format_value_compact(row.projected.get(col).unwrap_or(&Value::Null)))
                    .collect();
                seen.insert(key)
            });
        }

        Ok(ResultSet {
            rows: combined_rows,
            columns,
        })
    }

    // ========================================================================
    // Finalize
    // ========================================================================

    /// Convert the final ResultSet into a CypherResult for Python consumption
    fn finalize_result(&self, result_set: ResultSet) -> Result<CypherResult, String> {
        if result_set.columns.is_empty() {
            // No RETURN clause - infer columns from available bindings
            if result_set.rows.is_empty() {
                return Ok(CypherResult::empty());
            }

            // Auto-detect columns: collect all variable names from first row
            let first_row = &result_set.rows[0];
            let mut columns = Vec::new();
            for name in first_row.node_bindings.keys() {
                columns.push(name.clone());
            }
            for name in first_row.edge_bindings.keys() {
                columns.push(name.clone());
            }
            for name in first_row.projected.keys() {
                columns.push(name.clone());
            }
            columns.sort(); // Deterministic order

            let rows: Vec<Vec<Value>> = result_set
                .rows
                .iter()
                .map(|row| {
                    columns
                        .iter()
                        .map(|col| {
                            if let Some(val) = row.projected.get(col) {
                                val.clone()
                            } else if let Some(&idx) = row.node_bindings.get(col) {
                                if let Some(node) = self.graph.graph.node_weight(idx) {
                                    node_to_map_value(node)
                                } else {
                                    Value::Null
                                }
                            } else {
                                Value::Null
                            }
                        })
                        .collect()
                })
                .collect();

            return Ok(CypherResult { columns, rows });
        }

        // RETURN was specified - use its columns
        let rows: Vec<Vec<Value>> = result_set
            .rows
            .iter()
            .map(|row| {
                result_set
                    .columns
                    .iter()
                    .map(|col| row.projected.get(col).cloned().unwrap_or(Value::Null))
                    .collect()
            })
            .collect();

        Ok(CypherResult {
            columns: result_set.columns,
            rows,
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if an expression contains an aggregate function call
pub fn is_aggregate_expression(expr: &Expression) -> bool {
    match expr {
        Expression::FunctionCall { name, .. } => {
            matches!(
                name.to_lowercase().as_str(),
                "count" | "sum" | "avg" | "mean" | "average" | "min" | "max" | "collect" | "std"
            )
        }
        Expression::Add(l, r)
        | Expression::Subtract(l, r)
        | Expression::Multiply(l, r)
        | Expression::Divide(l, r) => is_aggregate_expression(l) || is_aggregate_expression(r),
        Expression::Negate(inner) => is_aggregate_expression(inner),
        _ => false,
    }
}

/// Get the column name for a return item
fn return_item_column_name(item: &ReturnItem) -> String {
    if let Some(ref alias) = item.alias {
        alias.clone()
    } else {
        expression_to_string(&item.expression)
    }
}

/// Convert an expression to its string representation (for column naming)
fn expression_to_string(expr: &Expression) -> String {
    match expr {
        Expression::PropertyAccess { variable, property } => format!("{}.{}", variable, property),
        Expression::Variable(name) => name.clone(),
        Expression::Literal(val) => format_value_compact(val),
        Expression::FunctionCall {
            name,
            args,
            distinct,
        } => {
            let args_str: Vec<String> = args.iter().map(expression_to_string).collect();
            if *distinct {
                format!("{}(DISTINCT {})", name, args_str.join(", "))
            } else {
                format!("{}({})", name, args_str.join(", "))
            }
        }
        Expression::Star => "*".to_string(),
        Expression::Add(l, r) => {
            format!("{} + {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Subtract(l, r) => {
            format!("{} - {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Multiply(l, r) => {
            format!("{} * {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Divide(l, r) => {
            format!("{} / {}", expression_to_string(l), expression_to_string(r))
        }
        Expression::Negate(inner) => format!("-{}", expression_to_string(inner)),
        Expression::ListLiteral(items) => {
            let items_str: Vec<String> = items.iter().map(expression_to_string).collect();
            format!("[{}]", items_str.join(", "))
        }
    }
}

/// Evaluate a comparison using existing filtering_methods infrastructure
fn evaluate_comparison(left: &Value, op: &ComparisonOp, right: &Value) -> bool {
    match op {
        ComparisonOp::Equals => filtering_methods::values_equal(left, right),
        ComparisonOp::NotEquals => !filtering_methods::values_equal(left, right),
        ComparisonOp::LessThan => {
            filtering_methods::compare_values(left, right) == Some(std::cmp::Ordering::Less)
        }
        ComparisonOp::LessThanEq => matches!(
            filtering_methods::compare_values(left, right),
            Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
        ),
        ComparisonOp::GreaterThan => {
            filtering_methods::compare_values(left, right) == Some(std::cmp::Ordering::Greater)
        }
        ComparisonOp::GreaterThanEq => matches!(
            filtering_methods::compare_values(left, right),
            Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
        ),
    }
}

/// Resolve a property from a NodeData
fn resolve_node_property(node: &NodeData, property: &str) -> Value {
    match node {
        NodeData::Regular {
            id,
            title,
            node_type,
            properties,
        }
        | NodeData::Schema {
            id,
            title,
            node_type,
            properties,
        } => match property {
            "id" => id.clone(),
            "title" | "name" => title.clone(),
            "type" | "node_type" | "label" => Value::String(node_type.clone()),
            _ => properties.get(property).cloned().unwrap_or(Value::Null),
        },
    }
}

/// Resolve a property from an EdgeBinding
fn resolve_edge_property(edge: &EdgeBinding, property: &str) -> Value {
    match property {
        "type" | "connection_type" => Value::String(edge.connection_type.clone()),
        _ => edge
            .properties
            .get(property)
            .cloned()
            .unwrap_or(Value::Null),
    }
}

/// Convert a NodeData to a representative Value (title string)
fn node_to_map_value(node: &NodeData) -> Value {
    match node {
        NodeData::Regular { title, .. } | NodeData::Schema { title, .. } => title.clone(),
    }
}

// Delegate to shared value_operations module
fn format_value_compact(val: &Value) -> String {
    value_operations::format_value_compact(val)
}
fn value_to_f64(val: &Value) -> Option<f64> {
    value_operations::value_to_f64(val)
}
fn arithmetic_add(a: &Value, b: &Value) -> Value {
    value_operations::arithmetic_add(a, b)
}
fn arithmetic_sub(a: &Value, b: &Value) -> Value {
    value_operations::arithmetic_sub(a, b)
}
fn arithmetic_mul(a: &Value, b: &Value) -> Value {
    value_operations::arithmetic_mul(a, b)
}
fn arithmetic_div(a: &Value, b: &Value) -> Value {
    value_operations::arithmetic_div(a, b)
}
fn arithmetic_negate(a: &Value) -> Value {
    value_operations::arithmetic_negate(a)
}
fn to_integer(val: &Value) -> Value {
    value_operations::to_integer(val)
}
fn to_float(val: &Value) -> Value {
    value_operations::to_float(val)
}
fn parse_value_string(s: &str) -> Value {
    value_operations::parse_value_string(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::values::Value;

    // ========================================================================
    // evaluate_comparison
    // ========================================================================

    #[test]
    fn test_comparison_equals() {
        assert!(evaluate_comparison(
            &Value::Int64(5),
            &ComparisonOp::Equals,
            &Value::Int64(5)
        ));
        assert!(!evaluate_comparison(
            &Value::Int64(5),
            &ComparisonOp::Equals,
            &Value::Int64(6)
        ));
    }

    #[test]
    fn test_comparison_not_equals() {
        assert!(evaluate_comparison(
            &Value::Int64(5),
            &ComparisonOp::NotEquals,
            &Value::Int64(6)
        ));
        assert!(!evaluate_comparison(
            &Value::Int64(5),
            &ComparisonOp::NotEquals,
            &Value::Int64(5)
        ));
    }

    #[test]
    fn test_comparison_less_than() {
        assert!(evaluate_comparison(
            &Value::Int64(3),
            &ComparisonOp::LessThan,
            &Value::Int64(5)
        ));
        assert!(!evaluate_comparison(
            &Value::Int64(5),
            &ComparisonOp::LessThan,
            &Value::Int64(5)
        ));
    }

    #[test]
    fn test_comparison_less_than_eq() {
        assert!(evaluate_comparison(
            &Value::Int64(5),
            &ComparisonOp::LessThanEq,
            &Value::Int64(5)
        ));
        assert!(evaluate_comparison(
            &Value::Int64(3),
            &ComparisonOp::LessThanEq,
            &Value::Int64(5)
        ));
        assert!(!evaluate_comparison(
            &Value::Int64(6),
            &ComparisonOp::LessThanEq,
            &Value::Int64(5)
        ));
    }

    #[test]
    fn test_comparison_greater_than() {
        assert!(evaluate_comparison(
            &Value::Int64(7),
            &ComparisonOp::GreaterThan,
            &Value::Int64(5)
        ));
        assert!(!evaluate_comparison(
            &Value::Int64(5),
            &ComparisonOp::GreaterThan,
            &Value::Int64(5)
        ));
    }

    #[test]
    fn test_comparison_greater_than_eq() {
        assert!(evaluate_comparison(
            &Value::Int64(5),
            &ComparisonOp::GreaterThanEq,
            &Value::Int64(5)
        ));
        assert!(evaluate_comparison(
            &Value::Int64(7),
            &ComparisonOp::GreaterThanEq,
            &Value::Int64(5)
        ));
    }

    #[test]
    fn test_comparison_cross_type() {
        // Int64 vs Float64
        assert!(evaluate_comparison(
            &Value::Int64(5),
            &ComparisonOp::Equals,
            &Value::Float64(5.0)
        ));
        assert!(evaluate_comparison(
            &Value::Int64(3),
            &ComparisonOp::LessThan,
            &Value::Float64(3.5)
        ));
    }

    // ========================================================================
    // arithmetic helpers
    // ========================================================================

    #[test]
    fn test_arithmetic_add_integers() {
        assert_eq!(
            arithmetic_add(&Value::Int64(3), &Value::Int64(4)),
            Value::Int64(7)
        );
    }

    #[test]
    fn test_arithmetic_add_floats() {
        let result = arithmetic_add(&Value::Float64(1.5), &Value::Float64(2.5));
        assert_eq!(result, Value::Float64(4.0));
    }

    #[test]
    fn test_arithmetic_add_string_concatenation() {
        let result = arithmetic_add(
            &Value::String("hello".to_string()),
            &Value::String(" world".to_string()),
        );
        assert_eq!(result, Value::String("hello world".to_string()));
    }

    #[test]
    fn test_arithmetic_add_mixed_numeric() {
        let result = arithmetic_add(&Value::Int64(3), &Value::Float64(1.5));
        assert_eq!(result, Value::Float64(4.5));
    }

    #[test]
    fn test_arithmetic_sub() {
        assert_eq!(
            arithmetic_sub(&Value::Int64(10), &Value::Int64(3)),
            Value::Int64(7)
        );
        assert_eq!(
            arithmetic_sub(&Value::Float64(5.0), &Value::Float64(2.0)),
            Value::Float64(3.0)
        );
    }

    #[test]
    fn test_arithmetic_mul() {
        assert_eq!(
            arithmetic_mul(&Value::Int64(3), &Value::Int64(4)),
            Value::Int64(12)
        );
    }

    #[test]
    fn test_arithmetic_div() {
        assert_eq!(
            arithmetic_div(&Value::Int64(10), &Value::Int64(4)),
            Value::Float64(2.5)
        );
    }

    #[test]
    fn test_arithmetic_div_by_zero() {
        assert_eq!(
            arithmetic_div(&Value::Int64(10), &Value::Int64(0)),
            Value::Null
        );
        assert_eq!(
            arithmetic_div(&Value::Float64(10.0), &Value::Float64(0.0)),
            Value::Null
        );
    }

    #[test]
    fn test_arithmetic_negate() {
        assert_eq!(arithmetic_negate(&Value::Int64(5)), Value::Int64(-5));
        assert_eq!(
            arithmetic_negate(&Value::Float64(3.14)),
            Value::Float64(-3.14)
        );
        assert_eq!(
            arithmetic_negate(&Value::String("x".to_string())),
            Value::Null
        );
    }

    #[test]
    fn test_arithmetic_incompatible_returns_null() {
        assert_eq!(
            arithmetic_add(&Value::Boolean(true), &Value::Boolean(false)),
            Value::Null
        );
        assert_eq!(
            arithmetic_sub(&Value::String("a".to_string()), &Value::Int64(1)),
            Value::Null
        );
    }

    // ========================================================================
    // value_to_f64
    // ========================================================================

    #[test]
    fn test_value_to_f64_conversions() {
        assert_eq!(value_to_f64(&Value::Int64(42)), Some(42.0));
        assert_eq!(value_to_f64(&Value::Float64(3.14)), Some(3.14));
        assert_eq!(value_to_f64(&Value::UniqueId(7)), Some(7.0));
        assert_eq!(value_to_f64(&Value::String("x".to_string())), None);
        assert_eq!(value_to_f64(&Value::Null), None);
        assert_eq!(value_to_f64(&Value::Boolean(true)), None);
    }

    // ========================================================================
    // to_integer / to_float
    // ========================================================================

    #[test]
    fn test_to_integer() {
        assert_eq!(to_integer(&Value::Int64(42)), Value::Int64(42));
        assert_eq!(to_integer(&Value::Float64(3.7)), Value::Int64(3));
        assert_eq!(to_integer(&Value::UniqueId(5)), Value::Int64(5));
        assert_eq!(
            to_integer(&Value::String("123".to_string())),
            Value::Int64(123)
        );
        assert_eq!(to_integer(&Value::String("abc".to_string())), Value::Null);
        assert_eq!(to_integer(&Value::Boolean(true)), Value::Int64(1));
        assert_eq!(to_integer(&Value::Boolean(false)), Value::Int64(0));
        assert_eq!(to_integer(&Value::Null), Value::Null);
    }

    #[test]
    fn test_to_float() {
        assert_eq!(to_float(&Value::Float64(3.14)), Value::Float64(3.14));
        assert_eq!(to_float(&Value::Int64(42)), Value::Float64(42.0));
        assert_eq!(to_float(&Value::UniqueId(5)), Value::Float64(5.0));
        assert_eq!(
            to_float(&Value::String("2.5".to_string())),
            Value::Float64(2.5)
        );
        assert_eq!(to_float(&Value::String("abc".to_string())), Value::Null);
    }

    // ========================================================================
    // format_value_compact
    // ========================================================================

    #[test]
    fn test_format_value_compact() {
        assert_eq!(format_value_compact(&Value::UniqueId(42)), "42");
        assert_eq!(format_value_compact(&Value::Int64(-5)), "-5");
        assert_eq!(format_value_compact(&Value::Float64(3.0)), "3.0");
        assert_eq!(format_value_compact(&Value::Float64(3.14)), "3.14");
        assert_eq!(format_value_compact(&Value::String("hi".to_string())), "hi");
        assert_eq!(format_value_compact(&Value::Boolean(true)), "true");
        assert_eq!(format_value_compact(&Value::Null), "null");
    }

    // ========================================================================
    // parse_value_string
    // ========================================================================

    #[test]
    fn test_parse_value_string() {
        assert_eq!(parse_value_string("null"), Value::Null);
        assert_eq!(parse_value_string("true"), Value::Boolean(true));
        assert_eq!(parse_value_string("false"), Value::Boolean(false));
        assert_eq!(parse_value_string("42"), Value::Int64(42));
        assert_eq!(parse_value_string("3.14"), Value::Float64(3.14));
        assert_eq!(
            parse_value_string("\"hello\""),
            Value::String("hello".to_string())
        );
        assert_eq!(
            parse_value_string("'world'"),
            Value::String("world".to_string())
        );
        assert_eq!(
            parse_value_string("unquoted"),
            Value::String("unquoted".to_string())
        );
    }

    // ========================================================================
    // is_aggregate_expression
    // ========================================================================

    #[test]
    fn test_is_aggregate_expression() {
        let agg = Expression::FunctionCall {
            name: "count".to_string(),
            args: vec![Expression::Star],
            distinct: false,
        };
        assert!(is_aggregate_expression(&agg));

        let non_agg = Expression::FunctionCall {
            name: "toUpper".to_string(),
            args: vec![Expression::Variable("x".to_string())],
            distinct: false,
        };
        assert!(!is_aggregate_expression(&non_agg));
    }

    #[test]
    fn test_is_aggregate_in_arithmetic() {
        let expr = Expression::Add(
            Box::new(Expression::FunctionCall {
                name: "sum".to_string(),
                args: vec![Expression::Variable("x".to_string())],
                distinct: false,
            }),
            Box::new(Expression::Literal(Value::Int64(1))),
        );
        assert!(is_aggregate_expression(&expr));
    }

    #[test]
    fn test_is_aggregate_literal_false() {
        assert!(!is_aggregate_expression(&Expression::Literal(
            Value::Int64(1)
        )));
        assert!(!is_aggregate_expression(&Expression::Variable(
            "x".to_string()
        )));
    }
}
