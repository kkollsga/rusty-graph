// src/graph/cypher/executor.rs
// Pipeline executor for Cypher queries

use super::ast::*;
use super::result::*;
use crate::datatypes::values::Value;
use crate::graph::filtering_methods;
use crate::graph::pattern_matching::{MatchBinding, PatternElement, PatternExecutor, PatternMatch};
use crate::graph::schema::{DirGraph, EdgeData, NodeData};
use crate::graph::value_operations;
use petgraph::graph::NodeIndex;
use petgraph::visit::NodeIndexable;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Executor
// ============================================================================

pub struct CypherExecutor<'a> {
    graph: &'a DirGraph,
    params: &'a HashMap<String, Value>,
}

impl<'a> CypherExecutor<'a> {
    pub fn with_params(graph: &'a DirGraph, params: &'a HashMap<String, Value>) -> Self {
        CypherExecutor { graph, params }
    }

    /// Execute a parsed Cypher query (read-only)
    pub fn execute(&self, query: &CypherQuery) -> Result<CypherResult, String> {
        let mut result_set = ResultSet::new();

        for clause in &query.clauses {
            result_set = self.execute_single_clause(clause, result_set)?;
        }

        // Convert ResultSet to CypherResult
        let mut result = self.finalize_result(result_set)?;
        result.stats = None;
        Ok(result)
    }

    /// Execute a single clause, transforming the result set.
    /// Public so execute_mutable can call it for read clauses.
    pub fn execute_single_clause(
        &self,
        clause: &Clause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        match clause {
            Clause::Match(m) => self.execute_match(m, result_set),
            Clause::OptionalMatch(m) => self.execute_optional_match(m, result_set),
            Clause::Where(w) => self.execute_where(w, result_set),
            Clause::Return(r) => self.execute_return(r, result_set),
            Clause::With(w) => self.execute_with(w, result_set),
            Clause::OrderBy(o) => self.execute_order_by(o, result_set),
            Clause::Limit(l) => self.execute_limit(l, result_set),
            Clause::Skip(s) => self.execute_skip(s, result_set),
            Clause::Unwind(u) => self.execute_unwind(u, result_set),
            Clause::Union(u) => self.execute_union(u, result_set),
            Clause::Create(_)
            | Clause::Set(_)
            | Clause::Delete(_)
            | Clause::Remove(_)
            | Clause::Merge(_) => {
                Err("Mutation clauses cannot be executed in read-only mode".to_string())
            }
        }
    }

    // ========================================================================
    // MATCH
    // ========================================================================

    fn execute_match(
        &self,
        clause: &MatchClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        // Check for shortestPath assignments
        if let Some(pa) = clause.path_assignments.first() {
            if pa.is_shortest_path {
                return self.execute_shortest_path_match(clause, pa, existing);
            }
        }

        if existing.rows.is_empty() {
            // First MATCH: execute patterns to produce initial bindings
            let mut all_rows = Vec::new();

            for pattern in &clause.patterns {
                if all_rows.is_empty() {
                    // First pattern - create initial rows
                    let executor = PatternExecutor::new_lightweight(self.graph, None);
                    let matches = executor.execute(pattern)?;
                    for m in matches {
                        all_rows.push(self.pattern_match_to_row(m));
                    }
                } else {
                    // Subsequent patterns: use shared-variable join
                    // Pass existing node bindings as pre-bindings to constrain the pattern
                    let mut new_rows = Vec::new();
                    for existing_row in &all_rows {
                        let executor = PatternExecutor::with_bindings(
                            self.graph,
                            None,
                            existing_row.node_bindings.clone(),
                        );
                        let matches = executor.execute(pattern)?;
                        for m in matches {
                            if !self.bindings_compatible(existing_row, &m) {
                                continue;
                            }
                            let mut new_row = existing_row.clone();
                            self.merge_match_into_row(&mut new_row, &m);
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
                    let executor =
                        PatternExecutor::with_bindings(self.graph, None, row.node_bindings.clone());
                    let matches = executor.execute(pattern)?;

                    for m in matches {
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

    /// Execute a shortestPath MATCH: find shortest path between anchored endpoints
    fn execute_shortest_path_match(
        &self,
        clause: &MatchClause,
        path_assignment: &PathAssignment,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        use crate::graph::graph_algorithms;

        let pattern = clause
            .patterns
            .get(path_assignment.pattern_index)
            .ok_or("Invalid pattern index for shortestPath")?;

        // Extract source and target node patterns from the pattern
        let elements = &pattern.elements;
        if elements.len() < 3 {
            return Err("shortestPath requires a pattern like (a)-[:REL*..N]->(b)".to_string());
        }

        let source_pattern = match &elements[0] {
            PatternElement::Node(np) => np,
            _ => return Err("shortestPath pattern must start with a node".to_string()),
        };

        let target_pattern = match elements.last() {
            Some(PatternElement::Node(np)) => np,
            _ => return Err("shortestPath pattern must end with a node".to_string()),
        };

        // Find matching source and target nodes
        let executor = PatternExecutor::new_lightweight(self.graph, None);
        let source_nodes = executor.find_matching_nodes_pub(source_pattern)?;
        let target_nodes = executor.find_matching_nodes_pub(target_pattern)?;

        let mut all_rows = Vec::new();

        for &source_idx in &source_nodes {
            for &target_idx in &target_nodes {
                if source_idx == target_idx {
                    continue;
                }

                if let Some(path_result) =
                    graph_algorithms::shortest_path_directed(self.graph, source_idx, target_idx)
                {
                    let mut row = ResultRow::new();

                    // Bind source variable
                    if let Some(ref var) = source_pattern.variable {
                        row.node_bindings.insert(var.clone(), source_idx);
                    }

                    // Bind target variable
                    if let Some(ref var) = target_pattern.variable {
                        row.node_bindings.insert(var.clone(), target_idx);
                    }

                    // Build path with connection types
                    let connections =
                        graph_algorithms::get_path_connections(self.graph, &path_result.path);
                    let path_nodes: Vec<(NodeIndex, String)> = path_result
                        .path
                        .iter()
                        .enumerate()
                        .map(|(i, &idx)| {
                            let conn_type = if i < connections.len() {
                                connections[i].clone().unwrap_or_default()
                            } else {
                                String::new()
                            };
                            (idx, conn_type)
                        })
                        .collect();

                    // Store path binding
                    row.path_bindings.insert(
                        path_assignment.variable.clone(),
                        PathBinding {
                            source: source_idx,
                            target: target_idx,
                            hops: path_result.cost,
                            path: path_nodes,
                        },
                    );

                    all_rows.push(row);
                }
            }
        }

        Ok(ResultSet {
            rows: all_rows,
            columns: existing.columns,
        })
    }

    /// Convert a PatternMatch to a lightweight ResultRow
    fn pattern_match_to_row(&self, m: PatternMatch) -> ResultRow {
        let binding_count = m.bindings.len();
        let mut row = ResultRow::with_capacity(binding_count, binding_count / 2, 0);

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
                let executor =
                    PatternExecutor::with_bindings(self.graph, None, row.node_bindings.clone());
                let matches = executor.execute(pattern)?;

                for m in &matches {
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
        let mut filtered_rows = Vec::new();
        for row in result_set.rows {
            match self.evaluate_predicate(&clause.predicate, &row) {
                Ok(true) => filtered_rows.push(row),
                Ok(false) => {}
                Err(e) => return Err(e),
            }
        }
        result_set.rows = filtered_rows;
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
                .map(|node| node.node_type.clone())
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
            Predicate::Exists { patterns } => {
                // Execute each pattern and check if any match is compatible
                // with the current row's bindings (outer variables must match)
                for pattern in patterns {
                    let executor =
                        PatternExecutor::with_bindings(self.graph, None, row.node_bindings.clone());
                    let matches = executor.execute(pattern)?;
                    let found = matches.iter().any(|m| self.bindings_compatible(row, m));
                    if !found {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    // ========================================================================
    // Expression Evaluation
    // ========================================================================

    /// Evaluate an expression against a row, resolving property access via NodeIndex
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
                // Evaluate the list expression
                let list_val = self.evaluate_expression(list_expr, row)?;
                // Parse list value (currently stored as "[a, b, c]" string)
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

            Expression::IndexAccess { expr, index } => {
                // Fast path: labels(n)[0] — bypass JSON round-trip
                if let Expression::FunctionCall { name, args, .. } = expr.as_ref() {
                    if name.eq_ignore_ascii_case("labels") {
                        if let Some(Expression::Variable(var)) = args.first() {
                            if let Expression::Literal(Value::Int64(lit_idx)) = index.as_ref() {
                                if *lit_idx == 0 {
                                    if let Some(&node_idx) = row.node_bindings.get(var.as_str()) {
                                        if let Some(node) = self.graph.graph.node_weight(node_idx) {
                                            return Ok(Value::String(
                                                node.get_node_type_ref().to_string(),
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
        }
    }

    /// Evaluate a CASE expression
    fn evaluate_case(
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
                    if filtering_methods::values_equal(&operand_val, &cond_val) {
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
            "size" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => Ok(Value::Int64(s.len() as i64)),
                    _ => Ok(Value::Null),
                }
            }
            "length" => {
                // length(p) for paths, length(s) for strings
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        return Ok(Value::Int64(path.hops as i64));
                    }
                }
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => Ok(Value::Int64(s.len() as i64)),
                    _ => Ok(Value::Null),
                }
            }
            "nodes" => {
                // nodes(p) returns list of node dicts in a path (JSON array)
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(path) = row.path_bindings.get(var) {
                        let mut entries = Vec::new();
                        for (node_idx, _) in &path.path {
                            if let Some(node) = self.graph.graph.node_weight(*node_idx) {
                                let mut props = Vec::new();
                                props.push(format!("\"id\": {}", format_value_compact(&node.id)));
                                props.push(format!(
                                    "\"title\": \"{}\"",
                                    format_value_compact(&node.title).replace('"', "\\\"")
                                ));
                                props.push(format!("\"type\": \"{}\"", node.node_type));
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
                        for (i, (_, conn_type)) in path.path.iter().enumerate() {
                            if !conn_type.is_empty() && i < path.path.len() - 1 {
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
                // labels(n) returns list of node labels (as JSON list)
                if let Some(Expression::Variable(var)) = args.first() {
                    if let Some(&idx) = row.node_bindings.get(var) {
                        if let Some(node) = self.graph.graph.node_weight(idx) {
                            let node_type = node.get_node_type_ref();
                            return Ok(Value::String(format!(
                                "[\"{}\"]",
                                node_type.replace('\\', "\\\\").replace('"', "\\\"")
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

        // Group rows by grouping key values (single composite string key to reduce allocations)
        let mut groups: Vec<(Vec<Value>, Vec<usize>)> = Vec::new();
        let mut group_index_map: HashMap<String, usize> = HashMap::new();
        let mut key_buf = String::with_capacity(64);

        for (row_idx, row) in result_set.rows.iter().enumerate() {
            let key_values: Vec<Value> = group_key_indices
                .iter()
                .map(|&i| {
                    self.evaluate_expression(&clause.items[i].expression, row)
                        .unwrap_or(Value::Null)
                })
                .collect();

            key_buf.clear();
            for (i, val) in key_values.iter().enumerate() {
                if i > 0 {
                    key_buf.push('\x1F');
                }
                value_operations::format_value_compact_into(&mut key_buf, val);
            }

            if let Some(&group_idx) = group_index_map.get(&key_buf) {
                groups[group_idx].1.push(row_idx);
            } else {
                let group_idx = groups.len();
                group_index_map.insert(key_buf.clone(), group_idx);
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

            // Preserve node/edge bindings from the first row in the group
            // for variables that appear in the grouping keys.
            // This ensures subsequent MATCH/OPTIONAL MATCH clauses can
            // constrain patterns to the correct nodes.
            let first_row = &result_set.rows[row_indices[0]];
            let mut row = ResultRow::from_projected(projected);
            for &item_idx in &group_key_indices {
                let expr = &clause.items[item_idx].expression;
                if let Expression::Variable(var) = expr {
                    if let Some(&idx) = first_row.node_bindings.get(var) {
                        row.node_bindings.insert(var.clone(), idx);
                    }
                    if let Some(edge) = first_row.edge_bindings.get(var) {
                        row.edge_bindings.insert(var.clone(), edge.clone());
                    }
                }
            }
            result_rows.push(row);
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
                            values.push(format_value_json(&val));
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
    pub fn finalize_result(&self, result_set: ResultSet) -> Result<CypherResult, String> {
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

            return Ok(CypherResult {
                columns,
                rows,
                stats: None,
            });
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
            stats: None,
        })
    }
}

// ============================================================================
// Mutation Execution
// ============================================================================

/// Check if a query contains any mutation clauses
pub fn is_mutation_query(query: &CypherQuery) -> bool {
    query.clauses.iter().any(|c| {
        matches!(
            c,
            Clause::Create(_)
                | Clause::Set(_)
                | Clause::Delete(_)
                | Clause::Remove(_)
                | Clause::Merge(_)
        )
    })
}

/// Execute a mutation query against a mutable graph.
/// Called instead of CypherExecutor::execute() when the query contains CREATE/SET/DELETE.
pub fn execute_mutable(
    graph: &mut DirGraph,
    query: &CypherQuery,
    params: HashMap<String, Value>,
) -> Result<CypherResult, String> {
    let mut result_set = ResultSet::new();
    let mut stats = MutationStats::default();

    for clause in &query.clauses {
        match clause {
            // Write clauses: mutate graph directly
            Clause::Create(create) => {
                result_set = execute_create(graph, create, result_set, &params, &mut stats)?;
            }
            Clause::Set(set) => {
                execute_set(graph, set, &result_set, &params, &mut stats)?;
            }
            Clause::Delete(del) => {
                execute_delete(graph, del, &result_set, &mut stats)?;
            }
            Clause::Remove(rem) => {
                execute_remove(graph, rem, &result_set, &mut stats)?;
            }
            Clause::Merge(merge) => {
                result_set = execute_merge(graph, merge, result_set, &params, &mut stats)?;
            }
            // Read clauses: create temporary immutable executor
            _ => {
                let executor = CypherExecutor::with_params(graph, &params);
                result_set = executor.execute_single_clause(clause, result_set)?;
            }
        }
    }

    // Finalize: if RETURN was in the query, finalize with column projection
    let has_return = query.clauses.iter().any(|c| matches!(c, Clause::Return(_)));

    if has_return || !result_set.columns.is_empty() {
        let executor = CypherExecutor::with_params(graph, &params);
        let mut result = executor.finalize_result(result_set)?;
        result.stats = Some(stats);
        Ok(result)
    } else {
        // No RETURN: return empty result with stats
        Ok(CypherResult {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: Some(stats),
        })
    }
}

/// Execute a CREATE clause, creating nodes and edges in the graph.
fn execute_create(
    graph: &mut DirGraph,
    create: &CreateClause,
    existing: ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<ResultSet, String> {
    let source_rows = if existing.rows.is_empty() {
        // No prior MATCH: execute once with an empty row
        vec![ResultRow::new()]
    } else {
        existing.rows
    };

    let mut new_rows = Vec::with_capacity(source_rows.len());

    for row in &source_rows {
        let mut new_row = row.clone();

        for pattern in &create.patterns {
            // Collect variable -> NodeIndex mappings for this pattern
            let mut pattern_vars: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();

            // Seed with existing bindings from MATCH
            for (var, &idx) in &row.node_bindings {
                pattern_vars.insert(var.clone(), idx);
            }

            // First pass: create all new nodes
            for element in &pattern.elements {
                if let CreateElement::Node(node_pat) = element {
                    // If variable already bound (from MATCH), skip creation
                    if let Some(ref var) = node_pat.variable {
                        if pattern_vars.contains_key(var) {
                            continue;
                        }
                    }

                    let node_idx = create_node(graph, node_pat, &new_row, params, stats)?;

                    if let Some(ref var) = node_pat.variable {
                        pattern_vars.insert(var.clone(), node_idx);
                        new_row.node_bindings.insert(var.clone(), node_idx);
                    }
                }
            }

            // Second pass: create edges
            // Elements are [Node, Edge, Node, Edge, Node, ...]
            let mut i = 1;
            while i < pattern.elements.len() {
                if let CreateElement::Edge(edge_pat) = &pattern.elements[i] {
                    let source_var = get_create_node_variable(&pattern.elements[i - 1]);
                    let target_var = get_create_node_variable(&pattern.elements[i + 1]);

                    let source_idx = resolve_create_node_idx(source_var, &pattern_vars)?;
                    let target_idx = resolve_create_node_idx(target_var, &pattern_vars)?;

                    // Determine actual source/target based on direction
                    let (actual_source, actual_target) = match edge_pat.direction {
                        CreateEdgeDirection::Outgoing => (source_idx, target_idx),
                        CreateEdgeDirection::Incoming => (target_idx, source_idx),
                    };

                    // Evaluate edge properties
                    let mut edge_props = HashMap::new();
                    {
                        let executor = CypherExecutor::with_params(graph, params);
                        for (key, expr) in &edge_pat.properties {
                            let val = executor.evaluate_expression(expr, &new_row)?;
                            edge_props.insert(key.clone(), val);
                        }
                    }

                    graph.register_connection_type(edge_pat.connection_type.clone());
                    stats.relationships_created += 1;

                    // Bind edge variable if named; avoid clone when not needed
                    if let Some(ref var) = edge_pat.variable {
                        let edge_data =
                            EdgeData::new(edge_pat.connection_type.clone(), edge_props.clone());
                        graph
                            .graph
                            .add_edge(actual_source, actual_target, edge_data);
                        new_row.edge_bindings.insert(
                            var.clone(),
                            EdgeBinding {
                                source: actual_source,
                                target: actual_target,
                                connection_type: edge_pat.connection_type.clone(),
                                properties: edge_props,
                            },
                        );
                    } else {
                        let edge_data = EdgeData::new(edge_pat.connection_type.clone(), edge_props);
                        graph
                            .graph
                            .add_edge(actual_source, actual_target, edge_data);
                    }
                }
                i += 2; // Skip to next edge position
            }
        }

        new_rows.push(new_row);
    }

    Ok(ResultSet {
        rows: new_rows,
        columns: existing.columns,
    })
}

/// Create a single node from a CreateNodePattern
fn create_node(
    graph: &mut DirGraph,
    node_pat: &CreateNodePattern,
    row: &ResultRow,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<petgraph::graph::NodeIndex, String> {
    // Evaluate property expressions (borrow graph immutably, then drop)
    let mut properties = HashMap::new();
    {
        let executor = CypherExecutor::with_params(graph, params);
        for (key, expr) in &node_pat.properties {
            let val = executor.evaluate_expression(expr, row)?;
            properties.insert(key.clone(), val);
        }
    }

    // Generate ID
    let id = Value::UniqueId(graph.graph.node_bound() as u32);

    // Determine title: use 'name' or 'title' property if present
    let title = properties
        .get("name")
        .or_else(|| properties.get("title"))
        .cloned()
        .unwrap_or_else(|| {
            let label = node_pat.label.as_deref().unwrap_or("Node");
            Value::String(format!("{}_{}", label, graph.graph.node_bound()))
        });

    let label = node_pat.label.clone().unwrap_or_else(|| "Node".to_string());

    let node_data = NodeData::new(id, title, label.clone(), properties);

    let node_idx = graph.graph.add_node(node_data);

    // Update type_indices
    graph
        .type_indices
        .entry(label.clone())
        .or_default()
        .push(node_idx);

    // Invalidate id_indices for this type (lazy rebuild on next lookup)
    graph.id_indices.remove(&label);

    // Update property and composite indices for the new node
    graph.update_property_indices_for_add(&label, node_idx);

    // Ensure type metadata exists for this type (consistent with Python add_nodes API)
    ensure_type_metadata(graph, &label, node_idx);

    stats.nodes_created += 1;

    Ok(node_idx)
}

/// Ensure type metadata exists for the given node type.
/// Reads property types from the sample node and upserts them into graph metadata.
/// This mirrors the behavior of the Python add_nodes() API in maintain_graph.rs.
fn ensure_type_metadata(
    graph: &mut DirGraph,
    node_type: &str,
    sample_node_idx: petgraph::graph::NodeIndex,
) {
    // Read sample node properties for type inference
    let sample_props: HashMap<String, String> = match graph.graph.node_weight(sample_node_idx) {
        Some(node) => node
            .properties
            .iter()
            .map(|(k, v)| (k.clone(), value_type_name(v)))
            .collect(),
        None => return,
    };

    graph.upsert_node_type_metadata(node_type, sample_props);
}

/// Map a Value variant to its type name string (for SchemaNode property types).
fn value_type_name(v: &Value) -> String {
    match v {
        Value::String(_) => "String",
        Value::Int64(_) => "Int64",
        Value::Float64(_) => "Float64",
        Value::Boolean(_) => "Boolean",
        Value::UniqueId(_) => "UniqueId",
        Value::DateTime(_) => "DateTime",
        Value::Null => "Null",
    }
    .to_string()
}

/// Extract the variable name from a CreateElement::Node
fn get_create_node_variable(element: &CreateElement) -> Option<&str> {
    match element {
        CreateElement::Node(np) => np.variable.as_deref(),
        _ => None,
    }
}

/// Resolve a variable name to a NodeIndex from the pattern vars map
fn resolve_create_node_idx(
    var: Option<&str>,
    pattern_vars: &HashMap<String, petgraph::graph::NodeIndex>,
) -> Result<petgraph::graph::NodeIndex, String> {
    match var {
        Some(name) => pattern_vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("Unbound variable '{}' in CREATE edge", name)),
        None => Err("CREATE edge requires named source and target nodes".to_string()),
    }
}

/// Execute a SET clause, modifying node properties in the graph.
fn execute_set(
    graph: &mut DirGraph,
    set: &SetClause,
    result_set: &ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<(), String> {
    for row in &result_set.rows {
        for item in &set.items {
            match item {
                SetItem::Property {
                    variable,
                    property,
                    expression,
                } => {
                    // Validate: cannot change id or type
                    if property == "id" {
                        return Err("Cannot SET node id — it is immutable".to_string());
                    }
                    if property == "type" || property == "node_type" || property == "label" {
                        return Err("Cannot SET node type via property assignment".to_string());
                    }

                    // Resolve the node
                    let node_idx = row.node_bindings.get(variable).ok_or_else(|| {
                        format!("Variable '{}' not bound to a node in SET", variable)
                    })?;

                    // Evaluate the expression (borrows graph immutably)
                    let value = {
                        let executor = CypherExecutor::with_params(graph, params);
                        executor.evaluate_expression(expression, row)?
                    };

                    // Capture old value + node_type before mutable borrow (for index update)
                    let (old_value, node_type_str) = match graph.get_node(*node_idx) {
                        Some(node) => {
                            let nt = node.get_node_type_ref().to_string();
                            let old = match property.as_str() {
                                "name" => node.get_field_ref("name").cloned(),
                                _ => node.get_field_ref(property).cloned(),
                            };
                            (old, nt)
                        }
                        None => continue,
                    };

                    // Clone value before it may be consumed by the mutation
                    let value_for_index = value.clone();

                    // Apply the mutation (borrows graph mutably)
                    if let Some(node) = graph.get_node_mut(*node_idx) {
                        match property.as_str() {
                            "title" => {
                                node.title = value;
                            }
                            "name" => {
                                // "name" maps to title in Cypher reads;
                                // update both title and properties for consistency
                                node.title = value.clone();
                                node.properties.insert("name".to_string(), value);
                            }
                            _ => {
                                node.properties.insert(property.clone(), value);
                            }
                        }
                        stats.properties_set += 1;
                    }

                    // Update property/composite indices (no active borrows)
                    // "title" only changes the title field, not a HashMap property
                    if property != "title" {
                        graph.update_property_indices_for_set(
                            &node_type_str,
                            *node_idx,
                            property,
                            old_value.as_ref(),
                            &value_for_index,
                        );
                    }
                }
                SetItem::Label { variable, label } => {
                    return Err(format!(
                        "SET label (SET {}:{}) is not yet supported",
                        variable, label
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Execute a DELETE clause, removing nodes and/or edges from the graph.
fn execute_delete(
    graph: &mut DirGraph,
    delete: &DeleteClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
) -> Result<(), String> {
    use petgraph::visit::EdgeRef;
    use std::collections::HashSet;

    let mut nodes_to_delete: HashSet<petgraph::graph::NodeIndex> = HashSet::new();
    // For edge deletion we store (source, target, connection_type) tuples to identify edges
    let mut edge_vars_to_delete: Vec<(
        String,
        petgraph::graph::NodeIndex,
        petgraph::graph::NodeIndex,
        String,
    )> = Vec::new();

    // Phase 1: collect all nodes and edges to delete across all rows
    for row in &result_set.rows {
        for expr in &delete.expressions {
            let var_name = match expr {
                Expression::Variable(name) => name,
                other => return Err(format!("DELETE expects variable names, got {:?}", other)),
            };

            if let Some(&node_idx) = row.node_bindings.get(var_name) {
                nodes_to_delete.insert(node_idx);
            } else if let Some(edge_binding) = row.edge_bindings.get(var_name) {
                edge_vars_to_delete.push((
                    var_name.clone(),
                    edge_binding.source,
                    edge_binding.target,
                    edge_binding.connection_type.clone(),
                ));
            } else {
                return Err(format!(
                    "Variable '{}' not bound to a node or relationship in DELETE",
                    var_name
                ));
            }
        }
    }

    // Phase 2: for plain DELETE (not DETACH), verify no node has edges
    if !delete.detach {
        for &node_idx in &nodes_to_delete {
            let has_edges = graph
                .graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .next()
                .is_some()
                || graph
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                    .next()
                    .is_some();
            if has_edges {
                let name = graph
                    .graph
                    .node_weight(node_idx)
                    .map(|n| {
                        n.get_field_ref("name")
                            .or_else(|| n.get_field_ref("title"))
                            .map(|v| format!("{:?}", v))
                            .unwrap_or_else(|| format!("index {}", node_idx.index()))
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(format!(
                    "Cannot delete node '{}' because it still has relationships. Use DETACH DELETE to delete the node and all its relationships.",
                    name
                ));
            }
        }
    }

    // Phase 3: delete explicitly-requested edges (from edge variable bindings)
    let mut deleted_edges: HashSet<petgraph::graph::EdgeIndex> = HashSet::new();
    for (_var, source, target, conn_type) in &edge_vars_to_delete {
        // Find the edge by source, target, and connection type
        let edge_idx = graph
            .graph
            .edges_directed(*source, petgraph::Direction::Outgoing)
            .find(|e| e.target() == *target && e.weight().connection_type == *conn_type)
            .map(|e| e.id());
        if let Some(idx) = edge_idx {
            if deleted_edges.insert(idx) {
                graph.graph.remove_edge(idx);
                stats.relationships_deleted += 1;
            }
        }
    }

    // Phase 4: for DETACH DELETE, remove all incident edges of nodes being deleted
    if delete.detach {
        for &node_idx in &nodes_to_delete {
            // Collect incident edge indices first (can't mutate while iterating)
            let incident: Vec<petgraph::graph::EdgeIndex> = graph
                .graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .chain(
                    graph
                        .graph
                        .edges_directed(node_idx, petgraph::Direction::Incoming),
                )
                .map(|e| e.id())
                .collect();
            for edge_idx in incident {
                if deleted_edges.insert(edge_idx) {
                    graph.graph.remove_edge(edge_idx);
                    stats.relationships_deleted += 1;
                }
            }
        }
    }

    // Phase 5: collect node types before deletion (for index cleanup)
    let mut affected_types: HashSet<String> = HashSet::new();
    for &node_idx in &nodes_to_delete {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            affected_types.insert(node.get_node_type_ref().to_string());
        }
    }

    // Phase 6: delete nodes
    for &node_idx in &nodes_to_delete {
        graph.graph.remove_node(node_idx);
        stats.nodes_deleted += 1;
    }

    // Phase 7: index cleanup (StableDiGraph keeps remaining indices stable)
    for node_type in &affected_types {
        // type_indices: remove deleted entries
        if let Some(indices) = graph.type_indices.get_mut(node_type) {
            indices.retain(|idx| !nodes_to_delete.contains(idx));
        }
        // id_indices: invalidate for lazy rebuild
        graph.id_indices.remove(node_type);
        // property_indices: remove deleted entries for affected types
        let prop_keys: Vec<_> = graph
            .property_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .cloned()
            .collect();
        for key in prop_keys {
            if let Some(value_map) = graph.property_indices.get_mut(&key) {
                for indices in value_map.values_mut() {
                    indices.retain(|idx| !nodes_to_delete.contains(idx));
                }
            }
        }
        // composite_indices: same treatment
        let comp_keys: Vec<_> = graph
            .composite_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .cloned()
            .collect();
        for key in comp_keys {
            if let Some(value_map) = graph.composite_indices.get_mut(&key) {
                for indices in value_map.values_mut() {
                    indices.retain(|idx| !nodes_to_delete.contains(idx));
                }
            }
        }
    }

    Ok(())
}

/// Execute a REMOVE clause, removing properties from nodes.
fn execute_remove(
    graph: &mut DirGraph,
    remove: &RemoveClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
) -> Result<(), String> {
    for row in &result_set.rows {
        for item in &remove.items {
            match item {
                RemoveItem::Property { variable, property } => {
                    // Protect immutable fields
                    if property == "id" {
                        return Err("Cannot REMOVE node id — it is immutable".to_string());
                    }
                    if property == "type" || property == "node_type" || property == "label" {
                        return Err("Cannot REMOVE node type".to_string());
                    }

                    let node_idx = row.node_bindings.get(variable).ok_or_else(|| {
                        format!("Variable '{}' not bound to a node in REMOVE", variable)
                    })?;

                    // Read node_type before mutable borrow (for index update)
                    let node_type_str = graph
                        .get_node(*node_idx)
                        .map(|n| n.get_node_type_ref().to_string())
                        .unwrap_or_default();

                    // Remove property (mutable borrow, returns old value)
                    let removed_value = if let Some(node) = graph.get_node_mut(*node_idx) {
                        node.properties.remove(property)
                    } else {
                        None
                    };

                    // Update stats + indices (no active borrows)
                    if let Some(old_val) = removed_value {
                        stats.properties_removed += 1;
                        graph.update_property_indices_for_remove(
                            &node_type_str,
                            *node_idx,
                            property,
                            &old_val,
                        );
                    }
                }
                RemoveItem::Label { variable, label } => {
                    return Err(format!(
                        "REMOVE label (REMOVE {}:{}) is not supported — rusty_graph uses single node_type",
                        variable, label
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Execute a MERGE clause: match-or-create a pattern.
fn execute_merge(
    graph: &mut DirGraph,
    merge: &MergeClause,
    existing: ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<ResultSet, String> {
    let source_rows = if existing.rows.is_empty() {
        vec![ResultRow::new()]
    } else {
        existing.rows
    };

    let mut new_rows = Vec::with_capacity(source_rows.len());

    for row in &source_rows {
        let mut new_row = row.clone();

        // Try to match the MERGE pattern
        let matched = try_match_merge_pattern(graph, &merge.pattern, &new_row, params)?;

        if let Some(bound_row) = matched {
            // Pattern matched — merge bindings into row
            for (var, idx) in &bound_row.node_bindings {
                new_row.node_bindings.insert(var.clone(), *idx);
            }
            for (var, binding) in &bound_row.edge_bindings {
                new_row.edge_bindings.insert(var.clone(), binding.clone());
            }

            // Execute ON MATCH SET
            if let Some(ref set_items) = merge.on_match {
                let set_clause = SetClause {
                    items: set_items.clone(),
                };
                let temp_rs = ResultSet {
                    rows: vec![new_row.clone()],
                    columns: Vec::new(),
                };
                execute_set(graph, &set_clause, &temp_rs, params, stats)?;
            }
        } else {
            // No match — CREATE the pattern
            let create_clause = CreateClause {
                patterns: vec![merge.pattern.clone()],
            };
            let temp_rs = ResultSet {
                rows: vec![new_row.clone()],
                columns: existing.columns.clone(),
            };
            let created = execute_create(graph, &create_clause, temp_rs, params, stats)?;

            // Merge newly created bindings into our row
            if let Some(created_row) = created.rows.into_iter().next() {
                for (var, idx) in created_row.node_bindings {
                    new_row.node_bindings.insert(var, idx);
                }
                for (var, binding) in created_row.edge_bindings {
                    new_row.edge_bindings.insert(var, binding);
                }
            }

            // Execute ON CREATE SET
            if let Some(ref set_items) = merge.on_create {
                let set_clause = SetClause {
                    items: set_items.clone(),
                };
                let temp_rs = ResultSet {
                    rows: vec![new_row.clone()],
                    columns: Vec::new(),
                };
                execute_set(graph, &set_clause, &temp_rs, params, stats)?;
            }
        }

        new_rows.push(new_row);
    }

    Ok(ResultSet {
        rows: new_rows,
        columns: existing.columns,
    })
}

/// Try to match a MERGE pattern against the graph.
/// Returns Some(ResultRow) with variable bindings if a match is found, None otherwise.
fn try_match_merge_pattern(
    graph: &DirGraph,
    pattern: &CreatePattern,
    row: &ResultRow,
    params: &HashMap<String, Value>,
) -> Result<Option<ResultRow>, String> {
    use petgraph::visit::EdgeRef;

    let executor = CypherExecutor::with_params(graph, params);

    match pattern.elements.len() {
        1 => {
            // Node-only MERGE: (var:Label {key: val, ...})
            if let CreateElement::Node(node_pat) = &pattern.elements[0] {
                // If variable is already bound from prior MATCH, it's already matched
                if let Some(ref var) = node_pat.variable {
                    if let Some(&existing_idx) = row.node_bindings.get(var) {
                        if graph.graph.node_weight(existing_idx).is_some() {
                            let mut result_row = ResultRow::new();
                            result_row.node_bindings.insert(var.clone(), existing_idx);
                            return Ok(Some(result_row));
                        }
                    }
                }

                let label = node_pat.label.as_deref().unwrap_or("Node");

                // Evaluate expected properties
                let expected_props: Vec<(&str, Value)> = node_pat
                    .properties
                    .iter()
                    .map(|(key, expr)| {
                        executor
                            .evaluate_expression(expr, row)
                            .map(|val| (key.as_str(), val))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // Search type_indices for a matching node
                if let Some(type_indices) = graph.type_indices.get(label) {
                    for &idx in type_indices {
                        if let Some(node) = graph.graph.node_weight(idx) {
                            // Match "name"/"title" to the node's title field,
                            // consistent with pattern_matching::node_matches_properties
                            let all_match = expected_props.iter().all(|(key, expected)| {
                                let value = if *key == "name" || *key == "title" {
                                    node.get_field_ref("title")
                                } else {
                                    node.get_field_ref(key)
                                };
                                value == Some(expected)
                            });
                            if all_match {
                                let mut result_row = ResultRow::new();
                                if let Some(ref var) = node_pat.variable {
                                    result_row.node_bindings.insert(var.clone(), idx);
                                }
                                return Ok(Some(result_row));
                            }
                        }
                    }
                }
                Ok(None)
            } else {
                Err("MERGE pattern must start with a node".to_string())
            }
        }
        3 => {
            // Relationship MERGE: (a)-[r:TYPE]->(b)
            let source_var = get_create_node_variable(&pattern.elements[0]);
            let target_var = get_create_node_variable(&pattern.elements[2]);

            let source_idx = source_var
                .and_then(|v| row.node_bindings.get(v).copied())
                .ok_or("MERGE path: source node must be bound by prior MATCH")?;
            let target_idx = target_var
                .and_then(|v| row.node_bindings.get(v).copied())
                .ok_or("MERGE path: target node must be bound by prior MATCH")?;

            if let CreateElement::Edge(edge_pat) = &pattern.elements[1] {
                let (actual_src, actual_tgt) = match edge_pat.direction {
                    CreateEdgeDirection::Outgoing => (source_idx, target_idx),
                    CreateEdgeDirection::Incoming => (target_idx, source_idx),
                };

                // Search for existing edge matching type
                let matching_edge = graph
                    .graph
                    .edges_directed(actual_src, petgraph::Direction::Outgoing)
                    .find(|e| {
                        e.target() == actual_tgt
                            && e.weight().connection_type == edge_pat.connection_type
                    });

                if let Some(edge_ref) = matching_edge {
                    let mut result_row = ResultRow::new();
                    if let Some(ref var) = edge_pat.variable {
                        result_row.edge_bindings.insert(
                            var.clone(),
                            EdgeBinding {
                                source: actual_src,
                                target: actual_tgt,
                                connection_type: edge_ref.weight().connection_type.clone(),
                                properties: edge_ref.weight().properties.clone(),
                            },
                        );
                    }
                    Ok(Some(result_row))
                } else {
                    Ok(None)
                }
            } else {
                Err("Expected edge in MERGE path pattern".to_string())
            }
        }
        _ => Err("MERGE supports single-node or single-edge patterns only".to_string()),
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
        Expression::Case {
            when_clauses,
            else_expr,
            ..
        } => {
            when_clauses
                .iter()
                .any(|(_, result)| is_aggregate_expression(result))
                || else_expr
                    .as_ref()
                    .is_some_and(|e| is_aggregate_expression(e))
        }
        Expression::ListComprehension {
            list_expr,
            map_expr,
            ..
        } => {
            is_aggregate_expression(list_expr)
                || map_expr
                    .as_ref()
                    .is_some_and(|e| is_aggregate_expression(e))
        }
        Expression::IndexAccess { expr, index } => {
            is_aggregate_expression(expr) || is_aggregate_expression(index)
        }
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
        Expression::Case { .. } => "CASE".to_string(),
        Expression::Parameter(name) => format!("${}", name),
        Expression::ListComprehension {
            variable,
            list_expr,
            filter,
            map_expr,
        } => {
            let mut result = format!("[{} IN {}", variable, expression_to_string(list_expr));
            if filter.is_some() {
                result.push_str(" WHERE ...");
            }
            if let Some(ref expr) = map_expr {
                result.push_str(&format!(" | {}", expression_to_string(expr)));
            }
            result.push(']');
            result
        }
        Expression::IndexAccess { expr, index } => {
            format!(
                "{}[{}]",
                expression_to_string(expr),
                expression_to_string(index)
            )
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
    match property {
        "id" => node.id.clone(),
        "title" | "name" => node.title.clone(),
        "type" | "node_type" | "label" => Value::String(node.node_type.clone()),
        _ => node
            .properties
            .get(property)
            .cloned()
            .unwrap_or(Value::Null),
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
    node.title.clone()
}

// Parse a list value from string format "[a, b, c]"
// This is a simple parser for the string representation of lists
fn parse_list_value(val: &Value) -> Vec<Value> {
    match val {
        Value::String(s) => {
            let trimmed = s.trim();
            if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
                return vec![];
            }
            let inner = &trimmed[1..trimmed.len() - 1];
            if inner.is_empty() {
                return vec![];
            }
            // Simple split by comma - doesn't handle nested lists or quoted strings properly
            // but sufficient for basic list comprehensions
            inner
                .split(',')
                .map(|item| {
                    let trimmed_item = item.trim();
                    // Try to parse as int, float, or keep as string
                    if let Ok(i) = trimmed_item.parse::<i64>() {
                        Value::Int64(i)
                    } else if let Ok(f) = trimmed_item.parse::<f64>() {
                        Value::Float64(f)
                    } else {
                        // Remove quotes if present
                        let unquoted = trimmed_item.trim_matches(|c| c == '"' || c == '\'');
                        Value::String(unquoted.to_string())
                    }
                })
                .collect()
        }
        _ => vec![],
    }
}

// Delegate to shared value_operations module
fn format_value_compact(val: &Value) -> String {
    value_operations::format_value_compact(val)
}
/// JSON-safe value formatting: strings are quoted, others are as-is.
/// Used for list serialization so py_convert can parse via json.loads.
fn format_value_json(val: &Value) -> String {
    match val {
        Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        Value::Null => "null".to_string(),
        Value::Boolean(b) => if *b { "true" } else { "false" }.to_string(),
        _ => format_value_compact(val),
    }
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

    // ========================================================================
    // CASE expression evaluation
    // ========================================================================

    #[test]
    fn test_case_simple_form_evaluation() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params);
        let row = ResultRow::new();

        // CASE 'Oslo' WHEN 'Oslo' THEN 'capital' ELSE 'other' END
        let expr = Expression::Case {
            operand: Some(Box::new(Expression::Literal(Value::String(
                "Oslo".to_string(),
            )))),
            when_clauses: vec![(
                CaseCondition::Expression(Expression::Literal(Value::String("Oslo".to_string()))),
                Expression::Literal(Value::String("capital".to_string())),
            )],
            else_expr: Some(Box::new(Expression::Literal(Value::String(
                "other".to_string(),
            )))),
        };

        let result = executor.evaluate_expression(&expr, &row).unwrap();
        assert_eq!(result, Value::String("capital".to_string()));
    }

    #[test]
    fn test_case_simple_form_else() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params);
        let row = ResultRow::new();

        // CASE 'Bergen' WHEN 'Oslo' THEN 'capital' ELSE 'other' END
        let expr = Expression::Case {
            operand: Some(Box::new(Expression::Literal(Value::String(
                "Bergen".to_string(),
            )))),
            when_clauses: vec![(
                CaseCondition::Expression(Expression::Literal(Value::String("Oslo".to_string()))),
                Expression::Literal(Value::String("capital".to_string())),
            )],
            else_expr: Some(Box::new(Expression::Literal(Value::String(
                "other".to_string(),
            )))),
        };

        let result = executor.evaluate_expression(&expr, &row).unwrap();
        assert_eq!(result, Value::String("other".to_string()));
    }

    #[test]
    fn test_case_no_else_returns_null() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params);
        let row = ResultRow::new();

        // CASE 'Bergen' WHEN 'Oslo' THEN 'capital' END → null
        let expr = Expression::Case {
            operand: Some(Box::new(Expression::Literal(Value::String(
                "Bergen".to_string(),
            )))),
            when_clauses: vec![(
                CaseCondition::Expression(Expression::Literal(Value::String("Oslo".to_string()))),
                Expression::Literal(Value::String("capital".to_string())),
            )],
            else_expr: None,
        };

        let result = executor.evaluate_expression(&expr, &row).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_case_generic_form_evaluation() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params);
        let mut row = ResultRow::new();
        row.projected.insert("val".to_string(), Value::Int64(25));

        // CASE WHEN val > 18 THEN 'adult' ELSE 'minor' END
        let expr = Expression::Case {
            operand: None,
            when_clauses: vec![(
                CaseCondition::Predicate(Predicate::Comparison {
                    left: Expression::Variable("val".to_string()),
                    operator: ComparisonOp::GreaterThan,
                    right: Expression::Literal(Value::Int64(18)),
                }),
                Expression::Literal(Value::String("adult".to_string())),
            )],
            else_expr: Some(Box::new(Expression::Literal(Value::String(
                "minor".to_string(),
            )))),
        };

        let result = executor.evaluate_expression(&expr, &row).unwrap();
        assert_eq!(result, Value::String("adult".to_string()));
    }

    // ========================================================================
    // Parameter evaluation
    // ========================================================================

    #[test]
    fn test_parameter_resolution() {
        let graph = DirGraph::new();
        let params = HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
            ("age".to_string(), Value::Int64(30)),
        ]);
        let executor = CypherExecutor::with_params(&graph, &params);
        let row = ResultRow::new();

        let result = executor
            .evaluate_expression(&Expression::Parameter("name".to_string()), &row)
            .unwrap();
        assert_eq!(result, Value::String("Alice".to_string()));

        let result = executor
            .evaluate_expression(&Expression::Parameter("age".to_string()), &row)
            .unwrap();
        assert_eq!(result, Value::Int64(30));
    }

    #[test]
    fn test_parameter_missing_error() {
        let graph = DirGraph::new();
        let no_params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &no_params);
        let row = ResultRow::new();

        let result =
            executor.evaluate_expression(&Expression::Parameter("missing".to_string()), &row);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing parameter"));
    }

    #[test]
    fn test_expression_to_string_case() {
        let expr = Expression::Case {
            operand: None,
            when_clauses: vec![],
            else_expr: None,
        };
        assert_eq!(expression_to_string(&expr), "CASE");
    }

    #[test]
    fn test_expression_to_string_parameter() {
        let expr = Expression::Parameter("foo".to_string());
        assert_eq!(expression_to_string(&expr), "$foo");
    }

    // ========================================================================
    // CREATE / SET mutation tests
    // ========================================================================

    /// Helper: build a small test graph with 2 Person nodes and 1 KNOWS edge
    fn build_test_graph() -> DirGraph {
        let mut graph = DirGraph::new();
        let alice = NodeData::new(
            Value::UniqueId(1),
            Value::String("Alice".to_string()),
            "Person".to_string(),
            HashMap::from([
                ("name".to_string(), Value::String("Alice".to_string())),
                ("age".to_string(), Value::Int64(30)),
            ]),
        );
        let bob = NodeData::new(
            Value::UniqueId(2),
            Value::String("Bob".to_string()),
            "Person".to_string(),
            HashMap::from([
                ("name".to_string(), Value::String("Bob".to_string())),
                ("age".to_string(), Value::Int64(25)),
            ]),
        );
        let alice_idx = graph.graph.add_node(alice);
        let bob_idx = graph.graph.add_node(bob);
        graph
            .type_indices
            .entry("Person".to_string())
            .or_default()
            .push(alice_idx);
        graph
            .type_indices
            .entry("Person".to_string())
            .or_default()
            .push(bob_idx);

        let edge = EdgeData::new("KNOWS".to_string(), HashMap::new());
        graph.graph.add_edge(alice_idx, bob_idx, edge);
        graph.register_connection_type("KNOWS".to_string());

        graph
    }

    #[test]
    fn test_create_single_node() {
        let mut graph = DirGraph::new();
        let query =
            super::super::parser::parse_cypher("CREATE (n:Person {name: 'Alice', age: 30})")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        assert!(result.stats.is_some());
        let stats = result.stats.unwrap();
        assert_eq!(stats.nodes_created, 1);
        assert_eq!(stats.relationships_created, 0);

        // Verify node was created (no SchemaNodes — metadata stored in HashMap)
        assert_eq!(graph.graph.node_count(), 1);
        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(
            node.get_field_ref("name"),
            Some(&Value::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_create_node_with_properties() {
        let mut graph = DirGraph::new();
        let query =
            super::super::parser::parse_cypher("CREATE (n:Product {name: 'Laptop', price: 999})")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(node.get_field_ref("price"), Some(&Value::Int64(999)));
        assert_eq!(node.get_node_type_ref(), "Product");
    }

    #[test]
    fn test_create_edge_between_matched() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:FRIENDS]->(b)",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        let stats = result.stats.unwrap();
        assert_eq!(stats.nodes_created, 0);
        assert_eq!(stats.relationships_created, 1);

        // Verify edge was created (graph should now have 2 edges: KNOWS + FRIENDS)
        assert_eq!(graph.graph.edge_count(), 2);
    }

    #[test]
    fn test_create_path() {
        let mut graph = DirGraph::new();
        let query = super::super::parser::parse_cypher(
            "CREATE (a:Person {name: 'A'})-[:KNOWS]->(b:Person {name: 'B'})",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        let stats = result.stats.unwrap();
        assert_eq!(stats.nodes_created, 2);
        assert_eq!(stats.relationships_created, 1);
        // 2 Person nodes (no SchemaNodes — metadata stored in HashMap)
        assert_eq!(graph.graph.node_count(), 2);
        assert_eq!(graph.graph.edge_count(), 1);
    }

    #[test]
    fn test_create_with_params() {
        let mut graph = DirGraph::new();
        let query =
            super::super::parser::parse_cypher("CREATE (n:Person {name: $name, age: $age})")
                .unwrap();
        let params = HashMap::from([
            ("name".to_string(), Value::String("Charlie".to_string())),
            ("age".to_string(), Value::Int64(35)),
        ]);
        let result = execute_mutable(&mut graph, &query, params).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(
            node.get_field_ref("name"),
            Some(&Value::String("Charlie".to_string()))
        );
    }

    #[test]
    fn test_create_return() {
        let mut graph = DirGraph::new();
        let query = super::super::parser::parse_cypher(
            "CREATE (n:Person {name: 'Test'}) RETURN n.name AS name",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        assert_eq!(result.columns, vec!["name"]);
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("Test".to_string()));
    }

    #[test]
    fn test_set_property() {
        let mut graph = build_test_graph();
        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) SET n.age = 31")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        let stats = result.stats.unwrap();
        assert_eq!(stats.properties_set, 1);

        // Verify property was updated
        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(node.get_field_ref("age"), Some(&Value::Int64(31)));
    }

    #[test]
    fn test_set_title() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MATCH (n:Person {name: 'Alice'}) SET n.name = 'Alicia'",
        )
        .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        // title is accessed via "name" or "title"
        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(
            node.get_field_ref("name"),
            Some(&Value::String("Alicia".to_string()))
        );
    }

    #[test]
    fn test_set_id_error() {
        let mut graph = build_test_graph();
        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) SET n.id = 999")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new());

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("immutable"));
    }

    #[test]
    fn test_set_expression() {
        let mut graph = build_test_graph();
        // Alice has age 30, add 1
        let query = super::super::parser::parse_cypher(
            "MATCH (n:Person {name: 'Alice'}) SET n.age = n.age + 1",
        )
        .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(node.get_field_ref("age"), Some(&Value::Int64(31)));
    }

    #[test]
    fn test_is_mutation_query() {
        let read_query = super::super::parser::parse_cypher("MATCH (n:Person) RETURN n").unwrap();
        assert!(!is_mutation_query(&read_query));

        let create_query =
            super::super::parser::parse_cypher("CREATE (n:Person {name: 'A'})").unwrap();
        assert!(is_mutation_query(&create_query));

        let set_query =
            super::super::parser::parse_cypher("MATCH (n:Person) SET n.age = 30").unwrap();
        assert!(is_mutation_query(&set_query));

        let delete_query = super::super::parser::parse_cypher("MATCH (n:Person) DELETE n").unwrap();
        assert!(is_mutation_query(&delete_query));

        let merge_query =
            super::super::parser::parse_cypher("MERGE (n:Person {name: 'A'})").unwrap();
        assert!(is_mutation_query(&merge_query));

        let remove_query =
            super::super::parser::parse_cypher("MATCH (n:Person) REMOVE n.age").unwrap();
        assert!(is_mutation_query(&remove_query));
    }

    // ==================================================================
    // DELETE Tests
    // ==================================================================

    #[test]
    fn test_detach_delete_node() {
        let mut graph = build_test_graph();
        assert_eq!(graph.graph.node_count(), 2);
        assert_eq!(graph.graph.edge_count(), 1);

        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        let stats = result.stats.unwrap();
        assert_eq!(stats.nodes_deleted, 1);
        assert_eq!(stats.relationships_deleted, 1);
        assert_eq!(graph.graph.node_count(), 1);
        assert_eq!(graph.graph.edge_count(), 0);
    }

    #[test]
    fn test_delete_node_with_edges_error() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DELETE n")
            .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("DETACH DELETE"));
    }

    #[test]
    fn test_delete_relationship() {
        let mut graph = build_test_graph();
        assert_eq!(graph.graph.edge_count(), 1);

        let query =
            super::super::parser::parse_cypher("MATCH (a:Person)-[r:KNOWS]->(b:Person) DELETE r")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        let stats = result.stats.unwrap();
        assert_eq!(stats.relationships_deleted, 1);
        assert_eq!(graph.graph.edge_count(), 0);
        assert_eq!(graph.graph.node_count(), 2);
    }

    #[test]
    fn test_delete_node_no_edges() {
        let mut graph = DirGraph::new();
        let node = NodeData::new(
            Value::UniqueId(1),
            Value::String("Solo".to_string()),
            "Person".to_string(),
            HashMap::from([("name".to_string(), Value::String("Solo".to_string()))]),
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry("Person".to_string())
            .or_default()
            .push(idx);

        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Solo'}) DELETE n").unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        assert_eq!(result.stats.unwrap().nodes_deleted, 1);
        assert_eq!(graph.graph.node_count(), 0);
    }

    #[test]
    fn test_detach_delete_updates_type_indices() {
        let mut graph = build_test_graph();
        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")
                .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        let person_indices = graph.type_indices.get("Person").unwrap();
        assert_eq!(person_indices.len(), 1);
    }

    // ==================================================================
    // REMOVE Tests
    // ==================================================================

    #[test]
    fn test_remove_property() {
        let mut graph = build_test_graph();
        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n.age")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().properties_removed, 1);

        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(node.get_field_ref("age"), None);
    }

    #[test]
    fn test_remove_nonexistent_property() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MATCH (n:Person {name: 'Alice'}) REMOVE n.nonexistent",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();
        assert_eq!(result.stats.as_ref().unwrap().properties_removed, 0);
    }

    #[test]
    fn test_remove_label_error() {
        let mut graph = build_test_graph();
        let query =
            super::super::parser::parse_cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n:Person")
                .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not supported"));
    }

    // ==================================================================
    // MERGE Tests
    // ==================================================================

    #[test]
    fn test_merge_creates_when_not_found() {
        let mut graph = DirGraph::new();
        let query = super::super::parser::parse_cypher("MERGE (n:Person {name: 'Alice'})").unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
        // 1 Person node (no SchemaNodes — metadata stored in HashMap)
        assert_eq!(graph.graph.node_count(), 1);
    }

    #[test]
    fn test_merge_matches_when_found() {
        let mut graph = build_test_graph();
        let initial_count = graph.graph.node_count();
        let query = super::super::parser::parse_cypher("MERGE (n:Person {name: 'Alice'})").unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 0);
        // No new nodes — MERGE matched existing; schema may or may not exist already
        assert_eq!(graph.graph.node_count(), initial_count);
    }

    #[test]
    fn test_merge_on_create_set() {
        let mut graph = DirGraph::new();
        let query = super::super::parser::parse_cypher(
            "MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.age = 30",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 1);
        assert_eq!(result.stats.as_ref().unwrap().properties_set, 1);
    }

    #[test]
    fn test_merge_on_match_set() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.visits = 1",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().nodes_created, 0);
        assert_eq!(result.stats.as_ref().unwrap().properties_set, 1);

        let node = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(0))
            .unwrap();
        assert_eq!(node.get_field_ref("visits"), Some(&Value::Int64(1)));
    }

    #[test]
    fn test_merge_relationship_matches() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) MERGE (a)-[r:KNOWS]->(b)",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().relationships_created, 0);
        assert_eq!(graph.graph.edge_count(), 1);
    }

    #[test]
    fn test_merge_creates_relationship() {
        let mut graph = build_test_graph();
        let query = super::super::parser::parse_cypher(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) MERGE (a)-[r:FRIENDS]->(b)",
        )
        .unwrap();
        let result = execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        assert_eq!(result.stats.as_ref().unwrap().relationships_created, 1);
        assert_eq!(graph.graph.edge_count(), 2);
    }

    // ========================================================================
    // Index auto-maintenance integration tests
    // ========================================================================

    #[test]
    fn test_create_updates_property_index() {
        let mut graph = build_test_graph();
        graph.create_index("Person", "age");

        // CREATE a new Person — should appear in the age index
        let query =
            super::super::parser::parse_cypher("CREATE (p:Person {name: 'Charlie', age: 40})")
                .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        let found = graph.lookup_by_index("Person", "age", &Value::Int64(40));
        assert!(found.is_some());
        assert_eq!(found.unwrap().len(), 1);
    }

    #[test]
    fn test_set_updates_property_index() {
        let mut graph = build_test_graph();
        graph.create_index("Person", "age");

        // SET Alice.age from 30 to 99
        let query =
            super::super::parser::parse_cypher("MATCH (p:Person {name: 'Alice'}) SET p.age = 99")
                .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        // Old value should be gone
        let old = graph.lookup_by_index("Person", "age", &Value::Int64(30));
        assert!(old.is_none() || old.unwrap().is_empty());

        // New value should be present
        let new = graph.lookup_by_index("Person", "age", &Value::Int64(99));
        assert!(new.is_some());
        assert_eq!(new.unwrap().len(), 1);
    }

    #[test]
    fn test_remove_updates_property_index() {
        let mut graph = build_test_graph();
        graph.create_index("Person", "age");

        // REMOVE Alice.age — should disappear from index
        let query =
            super::super::parser::parse_cypher("MATCH (p:Person {name: 'Alice'}) REMOVE p.age")
                .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        let found = graph.lookup_by_index("Person", "age", &Value::Int64(30));
        assert!(found.is_none() || found.unwrap().is_empty());
    }

    #[test]
    fn test_create_creates_type_metadata() {
        let mut graph = DirGraph::new();
        let query =
            super::super::parser::parse_cypher("CREATE (p:Animal {name: 'Rex', species: 'Dog'})")
                .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        // Type metadata for "Animal" should exist
        let metadata = graph.get_node_type_metadata("Animal");
        assert!(
            metadata.is_some(),
            "Type metadata for Animal should exist after CREATE"
        );
        let props = metadata.unwrap();
        assert!(props.contains_key("name"), "metadata should contain 'name'");
        assert!(
            props.contains_key("species"),
            "metadata should contain 'species'"
        );
    }

    #[test]
    fn test_merge_updates_indices() {
        let mut graph = build_test_graph();
        graph.create_index("Person", "age");

        // MERGE create path — new node should appear in index
        let query = super::super::parser::parse_cypher(
            "MERGE (p:Person {name: 'Dave'}) ON CREATE SET p.age = 50",
        )
        .unwrap();
        execute_mutable(&mut graph, &query, HashMap::new()).unwrap();

        let found = graph.lookup_by_index("Person", "age", &Value::Int64(50));
        assert!(found.is_some());
        assert_eq!(found.unwrap().len(), 1);

        // MERGE match path with SET — index should update
        let query2 = super::super::parser::parse_cypher(
            "MERGE (p:Person {name: 'Alice'}) ON MATCH SET p.age = 31",
        )
        .unwrap();
        execute_mutable(&mut graph, &query2, HashMap::new()).unwrap();

        // Old Alice age gone
        let old = graph.lookup_by_index("Person", "age", &Value::Int64(30));
        assert!(old.is_none() || old.unwrap().is_empty());

        // New Alice age present
        let new = graph.lookup_by_index("Person", "age", &Value::Int64(31));
        assert!(new.is_some());
        assert_eq!(new.unwrap().len(), 1);
    }
}
