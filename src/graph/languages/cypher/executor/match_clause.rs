//! Cypher executor — match_clause methods.

use super::super::ast::*;
use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{
    EdgeDirection, MatchBinding, Pattern, PatternElement, PatternExecutor, PatternMatch,
    PropertyMatcher,
};
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

impl<'a> CypherExecutor<'a> {
    pub(super) fn pattern_match_to_row(&self, m: PatternMatch) -> ResultRow {
        let binding_count = m.bindings.len();
        let mut row = ResultRow::with_capacity(binding_count, binding_count / 2, 0);

        for (var, binding) in m.bindings {
            match binding {
                MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                    row.node_bindings.insert(var, index);
                }
                MatchBinding::Edge {
                    source,
                    target,
                    edge_index,
                    ..
                } => {
                    row.edge_bindings.insert(
                        var,
                        EdgeBinding {
                            source,
                            target,
                            edge_index,
                        },
                    );
                }
                MatchBinding::VariableLengthPath {
                    source,
                    target,
                    hops,
                    path,
                } => {
                    let string_path: Vec<(petgraph::graph::NodeIndex, String)> = path
                        .iter()
                        .map(|(idx, ik)| (*idx, self.graph.interner.resolve(*ik).to_string()))
                        .collect();
                    row.path_bindings.insert(
                        var,
                        PathBinding {
                            source,
                            target,
                            hops,
                            path: string_path,
                        },
                    );
                }
            }
        }

        row
    }

    /// Merge a PatternMatch's bindings into an existing ResultRow
    pub(super) fn merge_match_into_row(&self, row: &mut ResultRow, m: &PatternMatch) {
        for (var, binding) in &m.bindings {
            match binding {
                MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                    row.node_bindings.insert(var.clone(), *index);
                }
                MatchBinding::Edge {
                    source,
                    target,
                    edge_index,
                    ..
                } => {
                    row.edge_bindings.insert(
                        var.clone(),
                        EdgeBinding {
                            source: *source,
                            target: *target,
                            edge_index: *edge_index,
                        },
                    );
                }
                MatchBinding::VariableLengthPath {
                    source,
                    target,
                    hops,
                    path,
                } => {
                    let string_path: Vec<(petgraph::graph::NodeIndex, String)> = path
                        .iter()
                        .map(|(idx, ik)| (*idx, self.graph.interner.resolve(*ik).to_string()))
                        .collect();
                    row.path_bindings.insert(
                        var.clone(),
                        PathBinding {
                            source: *source,
                            target: *target,
                            hops: *hops,
                            path: string_path,
                        },
                    );
                }
            }
        }
    }

    /// Synthesize a PathBinding from a multi-hop pattern.
    /// Iterates ALL pattern elements to capture every hop, not just the first.
    pub(super) fn synthesize_path_from_pattern(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        row: &ResultRow,
    ) -> Option<PathBinding> {
        let mut node_vars: Vec<&str> = Vec::new();
        let mut edge_types: Vec<&str> = Vec::new();
        for elem in &pattern.elements {
            match elem {
                PatternElement::Node(np) => {
                    if let Some(ref v) = np.variable {
                        node_vars.push(v);
                    }
                }
                PatternElement::Edge(ep) => {
                    edge_types.push(ep.connection_type.as_deref().unwrap_or(""));
                }
            }
        }
        if node_vars.len() < 2 || edge_types.is_empty() {
            return None;
        }
        let source_idx = row.node_bindings.get(node_vars[0])?;
        let target_idx = row.node_bindings.get(node_vars[node_vars.len() - 1])?;

        // Build full path: for each edge, record the target node and edge type
        let mut path = Vec::with_capacity(edge_types.len());
        for (i, edge_type) in edge_types.iter().enumerate() {
            let node_idx = row.node_bindings.get(node_vars[i + 1])?;
            path.push((*node_idx, edge_type.to_string()));
        }

        Some(PathBinding {
            source: *source_idx,
            target: *target_idx,
            hops: edge_types.len(),
            path,
        })
    }

    // ========================================================================
    // OPTIONAL MATCH
    // ========================================================================

    pub(super) fn execute_optional_match(
        &self,
        clause: &MatchClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        if existing.rows.is_empty() {
            // OPTIONAL MATCH as first clause: try regular match, but if
            // nothing matches, return one row with all variables set to NULL
            let columns = existing.columns.clone();
            let result = self.execute_match(clause, existing, None)?;
            if !result.rows.is_empty() {
                return Ok(result);
            }
            let mut null_row = ResultRow::new();
            for pattern in &clause.patterns {
                for elem in &pattern.elements {
                    match elem {
                        PatternElement::Node(np) => {
                            if let Some(ref var) = np.variable {
                                null_row.projected.insert(var.clone(), Value::Null);
                            }
                        }
                        PatternElement::Edge(ep) => {
                            if let Some(ref var) = ep.variable {
                                null_row.projected.insert(var.clone(), Value::Null);
                            }
                        }
                    }
                }
            }
            return Ok(ResultSet {
                rows: vec![null_row],
                columns,
            });
        }

        let mut new_rows = Vec::with_capacity(existing.rows.len());

        for row in &existing.rows {
            let mut found_any = false;

            for pattern in &clause.patterns {
                // Resolve EqualsVar references against current row
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

    /// Fast-path count for simple node-edge-node patterns when one end is pre-bound.
    /// Returns Some(count) if the fast-path applies, None to fall back to PatternExecutor.
    ///
    /// For pattern `(a:Type)-[:REL]->(b)` where `b` is already bound in the row:
    /// Instead of scanning all Type nodes and checking edges (O(|Type|)),
    /// traverse edges directly from the bound node (O(degree)).
    /// Fast path for EXISTS / NOT EXISTS: when the subquery is a single
    /// 3-element pattern (node-edge-node) with exactly one node already bound
    /// from the outer row, we can check edge existence directly via
    /// `edges_directed()` instead of creating a full PatternExecutor.
    /// Returns `Some(true/false)` if the fast path applies, `None` otherwise.
    pub(super) fn try_fast_exists_check(
        &self,
        patterns: &[Pattern],
        where_clause: &Option<Box<Predicate>>,
        row: &ResultRow,
    ) -> Option<Result<bool, String>> {
        if patterns.len() != 1 {
            return None;
        }
        let pattern = &patterns[0];
        if pattern.elements.len() != 3 {
            return None;
        }

        let node_a = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return None,
        };
        let edge = match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return None,
        };
        let node_b = match &pattern.elements[2] {
            PatternElement::Node(np) => np,
            _ => return None,
        };

        // Skip variable-length edges and edge property filters
        if edge.var_length.is_some() || edge.properties.is_some() {
            return None;
        }

        // Determine which node is bound from the outer row
        let a_bound = node_a
            .variable
            .as_ref()
            .and_then(|v| row.node_bindings.get(v).copied());
        let b_bound = node_b
            .variable
            .as_ref()
            .and_then(|v| row.node_bindings.get(v).copied());

        let (bound_idx, other_node, other_var, direction) = match (a_bound, b_bound) {
            (Some(idx), None) => {
                let dir = match edge.direction {
                    EdgeDirection::Outgoing => Direction::Outgoing,
                    EdgeDirection::Incoming => Direction::Incoming,
                    EdgeDirection::Both => return None,
                };
                (idx, node_b, &node_b.variable, dir)
            }
            (None, Some(idx)) => {
                let dir = match edge.direction {
                    EdgeDirection::Outgoing => Direction::Incoming,
                    EdgeDirection::Incoming => Direction::Outgoing,
                    EdgeDirection::Both => return None,
                };
                (idx, node_a, &node_a.variable, dir)
            }
            _ => return None, // both bound or neither — fall back
        };

        let interned_conn = edge.connection_type.as_deref().map(InternedKey::from_str);

        // Pre-allocate a mutable row for WHERE evaluation (avoids clone per edge)
        let (has_where, mut eval_row) = if where_clause.is_some() {
            let mut r = row.clone(); // single clone
            if let Some(ref var) = other_var {
                r.node_bindings.insert(var.clone(), NodeIndex::new(0)); // placeholder
            }
            (true, r)
        } else {
            (false, ResultRow::new()) // unused placeholder
        };

        for edge_ref in
            self.graph
                .graph
                .edges_directed_filtered(bound_idx, direction, interned_conn)
        {
            if let Some(ik) = interned_conn {
                if edge_ref.weight().connection_type != ik {
                    continue;
                }
            }

            let other_idx = if direction == Direction::Outgoing {
                edge_ref.target()
            } else {
                edge_ref.source()
            };

            // Check target node type (O(1) mmap read, no materialization)
            if let Some(ref req_type) = other_node.node_type {
                if let Some(nt) = self.graph.graph.node_type_of(other_idx) {
                    if self.graph.interner.resolve(nt) != req_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Check target node inline properties — bail to slow path
            // for non-trivial matchers (EqualsParam, EqualsVar, etc.)
            if let Some(ref props) = other_node.properties {
                if let Some(nd) = self.graph.graph.node_weight(other_idx) {
                    let mut all_match = true;
                    for (key, matcher) in props {
                        let val = nd.get_property(key);
                        let ok = match matcher {
                            PropertyMatcher::Equals(expected) => val.as_deref().is_some_and(|v| {
                                crate::graph::core::filtering_methods::values_equal(v, expected)
                            }),
                            PropertyMatcher::In(values) => val.as_deref().is_some_and(|v| {
                                values.iter().any(|exp| {
                                    crate::graph::core::filtering_methods::values_equal(v, exp)
                                })
                            }),
                            // Complex matchers — fall back to slow path
                            _ => return None,
                        };
                        if !ok {
                            all_match = false;
                            break;
                        }
                    }
                    if !all_match {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Check WHERE clause — reuse pre-allocated row, just update binding
            if has_where {
                if let Some(ref var) = other_var {
                    eval_row.node_bindings.insert(var.clone(), other_idx);
                }
                match self.evaluate_predicate(where_clause.as_ref().unwrap(), &eval_row) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(e) => return Some(Err(e)),
                }
            }

            return Some(Ok(true)); // Found a match
        }
        Some(Ok(false)) // No match found
    }

    pub(super) fn try_count_simple_pattern(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        bindings: &Bindings<NodeIndex>,
    ) -> Result<Option<i64>, String> {
        // Only handle simple 3-element patterns: Node-Edge-Node
        if pattern.elements.len() != 3 {
            return Ok(None);
        }

        let node_a = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return Ok(None),
        };
        let edge = match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return Ok(None),
        };
        let node_b = match &pattern.elements[2] {
            PatternElement::Node(np) => np,
            _ => return Ok(None),
        };

        // Don't use fast-path for variable-length edges or edge property filters
        if edge.var_length.is_some() || edge.properties.is_some() {
            return Ok(None);
        }

        // Don't use fast-path if the bound (group-key) node has property filters
        // — the caller already filtered it. The unbound node's properties are
        // checked inline during counting (supports WHERE push-down on target).
        // Both nodes having properties is rare and we fall back for it.

        // Determine which end is bound
        let a_bound = node_a
            .variable
            .as_ref()
            .and_then(|v| bindings.get(v).copied());
        let b_bound = node_b
            .variable
            .as_ref()
            .and_then(|v| bindings.get(v).copied());

        // We need exactly one end bound for the fast-path to help
        let (bound_idx, other_type, other_props, traverse_dir) = match (a_bound, b_bound) {
            (None, Some(b_idx)) => {
                // b is bound — traverse from b, other node is a
                if node_b.properties.is_some() {
                    return Ok(None); // bound node with props: fall back
                }
                let dir = match edge.direction {
                    EdgeDirection::Outgoing => Direction::Incoming, // (a)->b means b has incoming
                    EdgeDirection::Incoming => Direction::Outgoing, // (a)<-b means b has outgoing
                    EdgeDirection::Both => return Ok(None),
                };
                (b_idx, &node_a.node_type, &node_a.properties, dir)
            }
            (Some(a_idx), None) => {
                // a is bound — traverse from a, other node is b
                if node_a.properties.is_some() {
                    return Ok(None); // bound node with props: fall back
                }
                let dir = match edge.direction {
                    EdgeDirection::Outgoing => Direction::Outgoing,
                    EdgeDirection::Incoming => Direction::Incoming,
                    EdgeDirection::Both => return Ok(None),
                };
                (a_idx, &node_b.node_type, &node_b.properties, dir)
            }
            _ => return Ok(None), // both bound or neither bound — fall back
        };

        let conn_type = edge.connection_type.as_deref();
        let interned_conn = conn_type.map(InternedKey::from_str);
        let interned_other_type = other_type.as_ref().map(|t| InternedKey::from_str(t));

        // Fast path: when no property filters on the unbound node, use
        // count_edges_filtered which avoids EdgeData materialization entirely.
        // On disk with sorted CSR: binary search + sequential count (zero allocations).
        if other_props.is_none() {
            let count = self.graph.graph.count_edges_filtered(
                bound_idx,
                traverse_dir,
                interned_conn,
                interned_other_type,
                self.deadline,
            )?;
            return Ok(Some(count as i64));
        }

        // Slow path: property filters require per-node property access.
        // This loop can iterate millions of edges for hub nodes (Q5 has ~40 M
        // incoming P31 edges), so check the deadline every 1 M iterations.
        let pe = PatternExecutor::new_lightweight_with_params(self.graph, None, self.params);
        let mut count: i64 = 0;
        let mut iter: usize = 0;

        for edge_ref in
            self.graph
                .graph
                .edges_directed_filtered(bound_idx, traverse_dir, interned_conn)
        {
            iter += 1;
            if iter.is_multiple_of(1 << 20) {
                self.check_deadline()?;
            }
            // Connection type already filtered by edges_directed_filtered
            let other_idx = if traverse_dir == Direction::Outgoing {
                edge_ref.target()
            } else {
                edge_ref.source()
            };

            if let Some(required_type) = interned_other_type {
                if let Some(nt) = self.graph.graph.node_type_of(other_idx) {
                    if nt != required_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            if let Some(ref props) = other_props {
                if !pe.node_matches_properties_pub(other_idx, props) {
                    continue;
                }
            }

            count += 1;
        }

        Ok(Some(count))
    }

    /// Count matches for a 5-element pattern (a)-[e1]->(b)<-[e2]-(c)
    /// from a bound first node, without materializing intermediate rows.
    /// Traverses: first_node --e1--> middle_nodes --e2--> count last nodes.
    pub(super) fn count_two_hop_pattern(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        first_idx: NodeIndex,
    ) -> i64 {
        use petgraph::Direction;

        // Extract pattern elements
        let edge1 = match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return 0,
        };
        let mid_node = match &pattern.elements[2] {
            PatternElement::Node(np) => np,
            _ => return 0,
        };
        let edge2 = match &pattern.elements[3] {
            PatternElement::Edge(ep) => ep,
            _ => return 0,
        };
        let last_node = match &pattern.elements[4] {
            PatternElement::Node(np) => np,
            _ => return 0,
        };

        let dir1 = match edge1.direction {
            EdgeDirection::Outgoing => Direction::Outgoing,
            EdgeDirection::Incoming => Direction::Incoming,
            EdgeDirection::Both => return 0, // unsupported in fused path
        };
        let interned_conn1 = edge1.connection_type.as_deref().map(InternedKey::from_str);

        let dir2 = match edge2.direction {
            EdgeDirection::Outgoing => Direction::Outgoing,
            EdgeDirection::Incoming => Direction::Incoming,
            EdgeDirection::Both => return 0,
        };
        let interned_conn2 = edge2.connection_type.as_deref().map(InternedKey::from_str);

        let mut total: i64 = 0;

        // First hop: first_idx --e1--> middle nodes
        for e1_ref in self
            .graph
            .graph
            .edges_directed_filtered(first_idx, dir1, interned_conn1)
        {
            if let Some(ik) = interned_conn1 {
                if e1_ref.weight().connection_type != ik {
                    continue;
                }
            }
            let mid_idx = if dir1 == Direction::Outgoing {
                e1_ref.target()
            } else {
                e1_ref.source()
            };
            // Check middle node type (O(1) mmap read, no materialization)
            if let Some(ref mid_type) = mid_node.node_type {
                if let Some(nt) = self.graph.graph.node_type_of(mid_idx) {
                    if self.graph.interner.resolve(nt) != mid_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Second hop: mid_idx --e2--> last nodes (just count)
            for e2_ref in self
                .graph
                .graph
                .edges_directed_filtered(mid_idx, dir2, interned_conn2)
            {
                if let Some(ik) = interned_conn2 {
                    if e2_ref.weight().connection_type != ik {
                        continue;
                    }
                }
                let last_idx = if dir2 == Direction::Outgoing {
                    e2_ref.target()
                } else {
                    e2_ref.source()
                };
                // Check last node type (O(1) mmap read, no materialization)
                if let Some(ref last_type) = last_node.node_type {
                    if let Some(nt) = self.graph.graph.node_type_of(last_idx) {
                        if self.graph.interner.resolve(nt) != last_type {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                total += 1;
            }
        }

        total
    }

    /// Count matches for a 5-element pattern traversed in reverse:
    /// (a)-[e1]->(b)-[e2]->(c) counted from c (position 4) backward.
    /// Reads elements [3],[2],[1],[0] with flipped edge directions.
    pub(super) fn count_two_hop_pattern_reverse(
        &self,
        pattern: &crate::graph::core::pattern_matching::Pattern,
        last_idx: NodeIndex,
    ) -> i64 {
        use petgraph::Direction;

        // Read pattern elements in reverse
        let edge2 = match &pattern.elements[3] {
            PatternElement::Edge(ep) => ep,
            _ => return 0,
        };
        let mid_node = match &pattern.elements[2] {
            PatternElement::Node(np) => np,
            _ => return 0,
        };
        let edge1 = match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep,
            _ => return 0,
        };
        let first_node = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return 0,
        };

        // Flip edge2 direction (we're traversing from c back toward b)
        let dir2 = match edge2.direction {
            EdgeDirection::Outgoing => Direction::Incoming,
            EdgeDirection::Incoming => Direction::Outgoing,
            EdgeDirection::Both => return 0,
        };
        let interned_conn2 = edge2.connection_type.as_deref().map(InternedKey::from_str);

        // Flip edge1 direction (from b back toward a)
        let dir1 = match edge1.direction {
            EdgeDirection::Outgoing => Direction::Incoming,
            EdgeDirection::Incoming => Direction::Outgoing,
            EdgeDirection::Both => return 0,
        };
        let interned_conn1 = edge1.connection_type.as_deref().map(InternedKey::from_str);

        let mut total: i64 = 0;

        // First hop: last_idx --reverse(e2)--> middle nodes
        for e2_ref in self
            .graph
            .graph
            .edges_directed_filtered(last_idx, dir2, interned_conn2)
        {
            if let Some(ik) = interned_conn2 {
                if e2_ref.weight().connection_type != ik {
                    continue;
                }
            }
            let mid_idx = if dir2 == Direction::Outgoing {
                e2_ref.target()
            } else {
                e2_ref.source()
            };
            // Check middle node type (O(1) mmap read, no materialization)
            if let Some(ref mid_type) = mid_node.node_type {
                if let Some(nt) = self.graph.graph.node_type_of(mid_idx) {
                    if self.graph.interner.resolve(nt) != mid_type {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Second hop: mid_idx --reverse(e1)--> first nodes (just count)
            for e1_ref in self
                .graph
                .graph
                .edges_directed_filtered(mid_idx, dir1, interned_conn1)
            {
                if let Some(ik) = interned_conn1 {
                    if e1_ref.weight().connection_type != ik {
                        continue;
                    }
                }
                let first_idx = if dir1 == Direction::Outgoing {
                    e1_ref.target()
                } else {
                    e1_ref.source()
                };
                // Check first node type (O(1) mmap read, no materialization)
                if let Some(ref first_type) = first_node.node_type {
                    if let Some(nt) = self.graph.graph.node_type_of(first_idx) {
                        if self.graph.interner.resolve(nt) != first_type {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                total += 1;
            }
        }

        total
    }

    /// Fused OPTIONAL MATCH + WITH count() execution.
    /// Instead of expanding each input row into N matched rows then aggregating,
    /// count compatible matches directly per input row — O(N×degree) with zero
    /// intermediate row allocation.
    pub(super) fn execute_fused_optional_match_aggregate(
        &self,
        match_clause: &MatchClause,
        with_clause: &WithClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        if existing.rows.is_empty() {
            return Ok(existing);
        }

        // Identify which WITH items are group keys (variables) vs aggregates (count)
        let mut group_key_indices = Vec::new();
        let mut count_items: Vec<(usize, &ReturnItem)> = Vec::new();

        for (i, item) in with_clause.items.iter().enumerate() {
            if is_aggregate_expression(&item.expression) {
                count_items.push((i, item));
            } else {
                group_key_indices.push(i);
            }
        }

        let mut result_rows = Vec::with_capacity(existing.rows.len());

        for (scan_count, row) in existing.rows.iter().enumerate() {
            if scan_count.is_multiple_of(2048) {
                self.check_deadline()?;
            }
            // Count compatible matches for each pattern without materializing rows
            let mut match_count: i64 = 0;

            for pattern in &match_clause.patterns {
                // Fast-path: direct edge traversal when one end is pre-bound
                if let Some(fast_count) =
                    self.try_count_simple_pattern(pattern, &row.node_bindings)?
                {
                    match_count += fast_count;
                } else {
                    // Fall back to full PatternExecutor
                    let executor = PatternExecutor::with_bindings_and_params(
                        self.graph,
                        None,
                        &row.node_bindings,
                        self.params,
                    )
                    .set_deadline(self.deadline);
                    let matches = executor.execute(pattern)?;

                    for m in &matches {
                        if self.bindings_compatible(row, m) {
                            match_count += 1;
                        }
                    }
                }
            }

            // Build projected values for this row
            let mut projected =
                Bindings::with_capacity(group_key_indices.len() + count_items.len());

            // Group key pass-throughs
            for &idx in &group_key_indices {
                let item = &with_clause.items[idx];
                let key = return_item_column_name(item);
                let val = self.evaluate_expression(&item.expression, row)?;
                projected.insert(key, val);
            }

            // Count aggregates
            for &(_, item) in &count_items {
                let key = return_item_column_name(item);

                // count(*) counts all, count(var) counts non-null matches
                // For OPTIONAL MATCH fusion, match_count already reflects compatible matches
                // count(*) = match_count, count(var) = match_count (matched vars are non-null)
                projected.insert(key, Value::Int64(match_count));
            }

            // Create result row preserving bindings for group-key variables
            let mut new_row = ResultRow::from_projected(projected);
            for &idx in &group_key_indices {
                if let Expression::Variable(var) = &with_clause.items[idx].expression {
                    if let Some(&node_idx) = row.node_bindings.get(var) {
                        new_row.node_bindings.insert(var.clone(), node_idx);
                    }
                    if let Some(edge) = row.edge_bindings.get(var) {
                        new_row.edge_bindings.insert(var.clone(), *edge);
                    }
                    if let Some(path) = row.path_bindings.get(var) {
                        new_row.path_bindings.insert(var.clone(), path.clone());
                    }
                }
            }

            result_rows.push(new_row);
        }

        let mut result = ResultSet {
            rows: result_rows,
            columns: existing.columns,
        };

        // Apply optional WHERE on the aggregated rows (e.g. WHERE cnt > 3)
        if let Some(ref where_clause) = with_clause.where_clause {
            result = self.execute_where(where_clause, result)?;
        }

        Ok(result)
    }

    /// Fused MATCH + RETURN with count() aggregation.
    /// Instead of materializing all (node, edge, node) rows and then grouping,
    /// match only the first-pattern nodes (group keys) and count edges directly.
    pub(super) fn execute_fused_match_return_aggregate(
        &self,
        match_clause: &MatchClause,
        return_clause: &ReturnClause,
        top_k: &Option<(usize, bool, usize)>,
        _existing: ResultSet,
    ) -> Result<ResultSet, String> {
        // The MATCH must have exactly 1 pattern with 3 or 5 elements (validated by planner)
        let pattern = &match_clause.patterns[0];

        // Extract node variables from pattern
        let first_var = match &pattern.elements[0] {
            PatternElement::Node(np) => np.variable.as_ref(),
            _ => return Err("FusedMatchReturnAggregate: expected node pattern".into()),
        };
        let last_elem_idx = pattern.elements.len() - 1;
        let second_var = match &pattern.elements[last_elem_idx] {
            PatternElement::Node(np) => np.variable.as_ref(),
            _ => return Err("FusedMatchReturnAggregate: expected node pattern".into()),
        };

        // Determine which variable is the group key by checking RETURN items.
        // The planner guarantees all non-aggregate items reference the same variable.
        let group_var: &str = {
            let mut gv = None;
            for item in &return_clause.items {
                if !is_aggregate_expression(&item.expression) {
                    gv = match &item.expression {
                        Expression::PropertyAccess { variable, .. } => Some(variable.as_str()),
                        Expression::Variable(v) => Some(v.as_str()),
                        _ => None,
                    };
                    break;
                }
            }
            gv.ok_or("FusedMatchReturnAggregate: no group-by variable found")?
        };

        // Determine which pattern element index is the group key
        let group_elem_idx = if first_var.is_some_and(|v| v == group_var) {
            0
        } else if second_var.is_some_and(|v| v == group_var) {
            last_elem_idx
        } else {
            return Err("FusedMatchReturnAggregate: group variable not in pattern".into());
        };

        // Build a single-node pattern for matching group keys
        let group_only_pattern = crate::graph::core::pattern_matching::Pattern {
            elements: vec![pattern.elements[group_elem_idx].clone()],
        };

        // Match group-key nodes
        let executor = PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
            .set_deadline(self.deadline);
        let group_matches = executor.execute(&group_only_pattern)?;

        // Identify which RETURN items are group keys vs aggregates
        let mut group_key_indices = Vec::new();
        let mut count_indices = Vec::new();
        for (i, item) in return_clause.items.iter().enumerate() {
            if is_aggregate_expression(&item.expression) {
                count_indices.push(i);
            } else {
                group_key_indices.push(i);
            }
        }

        // Helper: extract node index from a match binding
        let extract_node_idx = |m: &crate::graph::core::pattern_matching::PatternMatch| -> Option<petgraph::graph::NodeIndex> {
            m.bindings.iter().find_map(|(name, binding)| {
                if name == group_var {
                    match binding {
                        MatchBinding::Node { index, .. } => Some(*index),
                        MatchBinding::NodeRef(index) => Some(*index),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        };

        // Helper: count edges for a node. Returns Result so the deadline
        // surfaced by try_count_simple_pattern can propagate through the
        // surrounding heap/loop and terminate the query cleanly.
        let count_for_node = |node_idx: petgraph::graph::NodeIndex| -> Result<i64, String> {
            if pattern.elements.len() == 5 {
                if group_elem_idx == 0 {
                    Ok(self.count_two_hop_pattern(pattern, node_idx))
                } else {
                    // group is at position 4 — traverse backward from last node
                    Ok(self.count_two_hop_pattern_reverse(pattern, node_idx))
                }
            } else {
                let mut bindings_for_count = Bindings::with_capacity(1);
                bindings_for_count.insert(group_var.to_string(), node_idx);
                Ok(self
                    .try_count_simple_pattern(pattern, &bindings_for_count)?
                    .unwrap_or(0))
            }
        };

        // Helper: build a result row for a (node_idx, count) pair
        let build_row =
            |node_idx: petgraph::graph::NodeIndex, match_count: i64| -> Result<ResultRow, String> {
                let mut tmp_row = ResultRow::new();
                tmp_row
                    .node_bindings
                    .insert(group_var.to_string(), node_idx);

                let mut projected = Bindings::with_capacity(return_clause.items.len());
                for &idx in &group_key_indices {
                    let item = &return_clause.items[idx];
                    let key = return_item_column_name(item);
                    let val = self.evaluate_expression(&item.expression, &tmp_row)?;
                    projected.insert(key, val);
                }
                for &idx in &count_indices {
                    let item = &return_clause.items[idx];
                    let key = return_item_column_name(item);
                    projected.insert(key, Value::Int64(match_count));
                }
                let mut new_row = ResultRow::from_projected(projected);
                new_row
                    .node_bindings
                    .insert(group_var.to_string(), node_idx);
                Ok(new_row)
            };

        let result_rows = if let Some(&(_, descending, limit)) = top_k.as_ref() {
            use std::cmp::Reverse;
            use std::collections::BinaryHeap;

            // Edge-centric aggregation: for 3-element patterns with a typed connection,
            // scan ALL edges of that type once and accumulate counts by peer. O(E_type)
            // sequential I/O instead of O(all_nodes × per_node_lookup).
            // This is critical for untyped group nodes (e.g., RETURN b.title, count(a))
            // where the node-centric path would iterate 124M nodes.
            let edge_conn_type = match &pattern.elements[1] {
                PatternElement::Edge(ep) => ep.connection_type.as_ref(),
                _ => None,
            };
            let group_node_props = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(np) => &np.properties,
                _ => &None,
            };
            if let (3, Some(ct_str), None) = (
                pattern.elements.len(),
                edge_conn_type,
                group_node_props.as_ref(),
            ) {
                let conn_key = InternedKey::from_str(ct_str);
                // Determine scan direction: if group is the TARGET (elem 2), scan
                // outgoing edges and group by target. If group is SOURCE (elem 0),
                // scan incoming edges and group by source.
                let scan_dir = if group_elem_idx == 0 {
                    // Group = source node (a). We want count of edges FROM a.
                    // Scan outgoing, peer = target → not what we want.
                    // Actually: scan outgoing edges, each edge's source = the node
                    // we iterated from. We want to count by source. But CSR outgoing
                    // is indexed by source — each node's outgoing edges are contiguous.
                    // We can just count the range length per node.
                    // However, the global scan counts by PEER (target), not by source.
                    // For group=source, we need to count outgoing per-source = CSR range size.
                    // That's cheaper: just offsets[i+1] - offsets[i] for the type range.
                    // For simplicity, skip edge-centric for group=source and use node-centric.
                    None
                } else {
                    // Group = target node (b). Scan outgoing, accumulate by target (peer).
                    Some(Direction::Outgoing)
                };

                if let Some(dir) = scan_dir {
                    self.check_deadline()?;
                    // Fast path: persistent per-(conn_type, peer) histogram
                    // answers in O(distinct-peers). Falls back to edge_endpoints
                    // scan for in-memory graphs and older disk graphs that
                    // lack the histogram.
                    let counts = if let Some(cached) = self.graph.graph.lookup_peer_counts(conn_key)
                    {
                        cached
                    } else {
                        self.graph.graph.count_edges_grouped_by_peer(
                            conn_key,
                            dir,
                            self.deadline,
                        )?
                    };
                    // Top-K from the counts HashMap
                    let heap: BinaryHeap<Reverse<(i64, u32)>> = if descending {
                        let mut h = BinaryHeap::with_capacity(limit + 1);
                        for (&peer, &count) in &counts {
                            h.push(Reverse((count, peer)));
                            if h.len() > limit {
                                h.pop();
                            }
                        }
                        h
                    } else {
                        // For ASC we need a max-heap — use negative trick
                        let mut h = BinaryHeap::with_capacity(limit + 1);
                        for (&peer, &count) in &counts {
                            h.push(Reverse((-count, peer)));
                            if h.len() > limit {
                                h.pop();
                            }
                        }
                        h
                    };

                    let top: Vec<_> = heap.into_sorted_vec();
                    let mut rows = Vec::with_capacity(top.len());
                    for Reverse((score, peer)) in &top {
                        let count = if descending { *score } else { -*score };
                        let node_idx = petgraph::graph::NodeIndex::new(*peer as usize);
                        rows.push(build_row(node_idx, count)?);
                    }
                    return Ok(ResultSet {
                        rows,
                        columns: return_clause
                            .items
                            .iter()
                            .map(return_item_column_name)
                            .collect(),
                    });
                }
            }

            // Node-centric top-K path (for typed group nodes or group=source patterns)
            // Get group node candidates directly from type_indices (streaming, no alloc)
            let group_node_type = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(np) => np.node_type.as_deref(),
                _ => None,
            };
            let group_node_props = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(np) => &np.properties,
                _ => &None,
            };
            let group_indices: Vec<petgraph::graph::NodeIndex> = if let Some(nt) = group_node_type {
                self.graph
                    .type_indices
                    .get(nt)
                    .map(|v| v.to_vec())
                    .unwrap_or_default()
            } else {
                {
                    let g = &self.graph.graph;
                    g.node_indices().collect()
                }
            };

            // Property filter executor (if group node has inline properties)
            let prop_executor = group_node_props.as_ref().map(|_| {
                PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
            });

            if descending {
                let mut heap: BinaryHeap<Reverse<(i64, petgraph::graph::NodeIndex)>> =
                    BinaryHeap::with_capacity(limit + 1);
                for (scan_count, &node_idx) in group_indices.iter().enumerate() {
                    if scan_count.is_multiple_of(10000) {
                        self.check_deadline()?;
                    }
                    // Property filter on group node
                    if let Some(ref props) = group_node_props {
                        if !prop_executor
                            .as_ref()
                            .unwrap()
                            .node_matches_properties_pub(node_idx, props)
                        {
                            continue;
                        }
                    }
                    let count = count_for_node(node_idx)?;
                    if count == 0 {
                        continue;
                    }
                    heap.push(Reverse((count, node_idx)));
                    if heap.len() > limit {
                        heap.pop();
                    }
                }
                let top: Vec<_> = heap
                    .into_sorted_vec()
                    .into_iter()
                    .map(|Reverse(x)| x)
                    .collect();
                let mut rows = Vec::with_capacity(top.len());
                for (count, node_idx) in top {
                    rows.push(build_row(node_idx, count)?);
                }
                rows
            } else {
                let mut heap: BinaryHeap<(i64, petgraph::graph::NodeIndex)> =
                    BinaryHeap::with_capacity(limit + 1);
                for (scan_count, &node_idx) in group_indices.iter().enumerate() {
                    if scan_count.is_multiple_of(10000) {
                        self.check_deadline()?;
                    }
                    if let Some(ref props) = group_node_props {
                        if !prop_executor
                            .as_ref()
                            .unwrap()
                            .node_matches_properties_pub(node_idx, props)
                        {
                            continue;
                        }
                    }
                    let count = count_for_node(node_idx)?;
                    if count == 0 {
                        continue;
                    }
                    heap.push((count, node_idx));
                    if heap.len() > limit {
                        heap.pop();
                    }
                }
                let top: Vec<_> = heap.into_sorted_vec();
                let mut rows = Vec::with_capacity(top.len());
                for (count, node_idx) in top {
                    rows.push(build_row(node_idx, count)?);
                }
                rows
            }
        } else {
            // Non-top-k: use edge-centric aggregation when the pattern is a
            // 3-element typed edge and the group key is the target node. This
            // replaces an O(|target-nodes| * avg-degree) per-node scan with a
            // single O(|edges-of-type|) sequential pass — essential when the
            // group variable has no type filter (124 M target candidates on
            // Wikidata would OOM or time out).
            let edge_conn_type = match &pattern.elements[1] {
                PatternElement::Edge(ep) => ep.connection_type.as_ref(),
                _ => None,
            };
            let group_node_props_nontopk = match &pattern.elements[group_elem_idx] {
                PatternElement::Node(np) => &np.properties,
                _ => &None,
            };
            let edge_centric_rows = if let (3, Some(ct_str), None, 2) = (
                pattern.elements.len(),
                edge_conn_type,
                group_node_props_nontopk.as_ref(),
                group_elem_idx,
            ) {
                let conn_key = InternedKey::from_str(ct_str);
                self.check_deadline()?;
                // Fast path: persistent histogram. See matching comment at the
                // top-k branch.
                let counts = if let Some(cached) = self.graph.graph.lookup_peer_counts(conn_key) {
                    cached
                } else {
                    self.graph.graph.count_edges_grouped_by_peer(
                        conn_key,
                        Direction::Outgoing,
                        self.deadline,
                    )?
                };
                let mut rows = Vec::with_capacity(counts.len());
                for (peer, count) in counts {
                    self.check_deadline()?;
                    let node_idx = petgraph::graph::NodeIndex::new(peer as usize);
                    rows.push(build_row(node_idx, count)?);
                }
                Some(rows)
            } else {
                None
            };

            if let Some(rows) = edge_centric_rows {
                rows
            } else {
                // Node-centric fallback: iterate group_matches, count per node.
                let mut rows = Vec::with_capacity(group_matches.len());
                for (scan_count, m) in group_matches.iter().enumerate() {
                    if scan_count.is_multiple_of(2048) {
                        self.check_deadline()?;
                    }
                    let Some(node_idx) = extract_node_idx(m) else {
                        continue;
                    };
                    let match_count = count_for_node(node_idx)?;
                    // MATCH semantics: skip nodes with zero matching edges
                    if match_count == 0 {
                        continue;
                    }
                    rows.push(build_row(node_idx, match_count)?);
                }
                rows
            }
        };

        // Apply HAVING post-aggregation. Cheap: the row set is at most the
        // number of distinct group keys, which is bounded by the type/peer
        // cardinality (thousands to tens of thousands), not the edge count.
        let mut result_rows = result_rows;
        if let Some(ref having) = return_clause.having {
            augment_rows_with_aggregate_keys(&mut result_rows, &return_clause.items);
            result_rows.retain(|row| self.evaluate_predicate(having, row).unwrap_or(false));
        }

        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();

        Ok(ResultSet {
            rows: result_rows,
            columns,
        })
    }

    /// Fused MATCH (n:Type) [WHERE ...] RETURN group_keys, agg_funcs(...)
    /// Single-pass node scan: iterates nodes directly, evaluates group keys
    /// and aggregates without creating intermediate ResultRows.
    pub(super) fn execute_fused_node_scan_aggregate(
        &self,
        match_clause: &MatchClause,
        where_predicate: Option<&Predicate>,
        return_clause: &ReturnClause,
    ) -> Result<ResultSet, String> {
        use crate::graph::core::pattern_matching::PatternElement;

        // Extract node variable and type from the single-element pattern
        let pattern = &match_clause.patterns[0];
        let node_pattern = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return Err("FusedNodeScanAggregate: expected node pattern".into()),
        };
        let node_var = node_pattern.variable.as_deref().unwrap_or("_n");
        let node_type = node_pattern.node_type.as_deref();

        // Get candidate node indices
        let node_indices: Vec<petgraph::graph::NodeIndex> = if let Some(nt) = node_type {
            if let Some(indices) = self.graph.type_indices.get(nt) {
                indices.to_vec()
            } else {
                Vec::new()
            }
        } else {
            {
                let g = &self.graph.graph;
                g.node_indices().collect()
            }
        };

        // Classify RETURN items into group keys and aggregates
        let mut group_key_indices = Vec::new();
        let mut agg_indices = Vec::new();
        for (i, item) in return_clause.items.iter().enumerate() {
            if is_aggregate_expression(&item.expression) {
                agg_indices.push(i);
            } else {
                group_key_indices.push(i);
            }
        }

        // Pre-fold group key and aggregate expressions
        let folded_group_exprs: Vec<Expression> = group_key_indices
            .iter()
            .map(|&i| self.fold_constants_expr(&return_clause.items[i].expression))
            .collect();

        // Pre-fold WHERE predicate once (converts In → InLiteralSet with HashSet, etc.)
        let folded_where = where_predicate.map(|p| self.fold_constants_pred(p));
        let folded_where_ref = folded_where.as_ref();

        // Single-pass: iterate nodes, evaluate group keys, update accumulators
        // Use a single reusable ResultRow to avoid per-node allocation
        let mut eval_row = ResultRow::new();
        eval_row
            .node_bindings
            .insert(node_var.to_string(), petgraph::graph::NodeIndex::new(0));

        // Create PatternExecutor once for property matching (if needed)
        let pattern_executor = if node_pattern.properties.is_some() {
            Some(PatternExecutor::new_lightweight_with_params(
                self.graph,
                None,
                self.params,
            ))
        } else {
            None
        };

        // Inline accumulators for aggregation during scan
        struct InlineAccumulators {
            counts: Vec<i64>,
            sums: Vec<f64>,
            mins: Vec<Option<Value>>,
            maxs: Vec<Option<Value>>,
        }

        // Groups: (group_key_values, first_node_idx_for_binding)
        let mut groups: Vec<(Vec<Value>, petgraph::graph::NodeIndex)> = Vec::new();
        let mut group_accumulators: Vec<InlineAccumulators> = Vec::new();
        let mut group_index_map: HashMap<Vec<Value>, usize> = HashMap::new();

        for (scan_count, &node_idx) in node_indices.iter().enumerate() {
            // Timeout check every 10,000 iterations (matches fused_match_return pattern)
            if scan_count % 10_000 == 0 && scan_count > 0 {
                self.check_deadline()?;
            }
            // Check pattern properties using PatternExecutor's matching logic
            if let Some(ref props) = node_pattern.properties {
                if !pattern_executor
                    .as_ref()
                    .unwrap()
                    .node_matches_properties_pub(node_idx, props)
                {
                    continue;
                }
            }

            // Set the node binding for expression evaluation
            *eval_row.node_bindings.get_mut(node_var).unwrap() = node_idx;

            // Check WHERE predicate (using pre-folded version for optimal evaluation)
            if let Some(pred) = folded_where_ref {
                if !self.evaluate_predicate(pred, &eval_row).unwrap_or(false) {
                    continue;
                }
            }

            // Evaluate group key
            let key_values: Vec<Value> = folded_group_exprs
                .iter()
                .map(|expr| {
                    self.evaluate_expression(expr, &eval_row)
                        .unwrap_or(Value::Null)
                })
                .collect();

            // Evaluate all aggregate expressions for this node
            let agg_vals: Vec<Value> = agg_indices
                .iter()
                .map(|&ai| {
                    let item = &return_clause.items[ai];
                    match &item.expression {
                        Expression::FunctionCall { args, .. } => {
                            if args.is_empty() || matches!(args[0], Expression::Star) {
                                Value::Boolean(true) // count(*) marker — always counted
                            } else {
                                self.evaluate_expression(&args[0], &eval_row)
                                    .unwrap_or(Value::Null)
                            }
                        }
                        _ => self
                            .evaluate_expression(&item.expression, &eval_row)
                            .unwrap_or(Value::Null),
                    }
                })
                .collect();

            if let Some(&group_idx) = group_index_map.get(&key_values) {
                // Update accumulators
                let acc = &mut group_accumulators[group_idx];
                for (ai, _) in agg_indices.iter().enumerate() {
                    let val = &agg_vals[ai];
                    // Only count non-null values (count(*) uses Boolean marker)
                    if !matches!(val, Value::Null) {
                        acc.counts[ai] += 1;
                    }
                    if let Some(f) = value_to_f64(val) {
                        acc.sums[ai] += f;
                    }
                    if !matches!(val, Value::Null) {
                        if acc.mins[ai].is_none()
                            || crate::graph::core::filtering_methods::compare_values(
                                val,
                                acc.mins[ai].as_ref().unwrap(),
                            ) == Some(std::cmp::Ordering::Less)
                        {
                            acc.mins[ai] = Some(val.clone());
                        }
                        if acc.maxs[ai].is_none()
                            || crate::graph::core::filtering_methods::compare_values(
                                val,
                                acc.maxs[ai].as_ref().unwrap(),
                            ) == Some(std::cmp::Ordering::Greater)
                        {
                            acc.maxs[ai] = Some(val.clone());
                        }
                    }
                }
            } else {
                let group_idx = groups.len();
                group_index_map.insert(key_values.clone(), group_idx);
                groups.push((key_values, node_idx));

                // Initialize accumulators
                let na = agg_indices.len();
                let mut acc = InlineAccumulators {
                    counts: vec![0i64; na],
                    sums: vec![0.0f64; na],
                    mins: vec![None; na],
                    maxs: vec![None; na],
                };
                for (ai, _) in agg_indices.iter().enumerate() {
                    let val = &agg_vals[ai];
                    if !matches!(val, Value::Null) {
                        acc.counts[ai] = 1;
                        if let Some(f) = value_to_f64(val) {
                            acc.sums[ai] = f;
                        }
                        acc.mins[ai] = Some(val.clone());
                        acc.maxs[ai] = Some(val.clone());
                    }
                }
                group_accumulators.push(acc);
            }
        }

        // Build result rows from groups
        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();

        // Handle empty-set aggregation: pure aggregation with no group keys
        // and no matching nodes should return one row with defaults (count=0, sum=0, etc.)
        if groups.is_empty() && group_key_indices.is_empty() {
            let empty_rows: Vec<&ResultRow> = Vec::new();
            let mut projected = Bindings::with_capacity(return_clause.items.len());
            for &item_idx in &agg_indices {
                let item = &return_clause.items[item_idx];
                let key = return_item_column_name(item);
                let val = self.evaluate_aggregate_with_rows(&item.expression, &empty_rows)?;
                projected.insert(key, val);
            }
            return Ok(ResultSet {
                rows: vec![ResultRow::from_projected(projected)],
                columns,
            });
        }

        let mut result_rows = Vec::with_capacity(groups.len());

        for (gi, (group_key_values, first_node_idx)) in groups.iter().enumerate() {
            let mut projected = Bindings::with_capacity(return_clause.items.len());

            // Add group key values
            for (ki, &item_idx) in group_key_indices.iter().enumerate() {
                let key = return_item_column_name(&return_clause.items[item_idx]);
                projected.insert(key, group_key_values[ki].clone());
            }

            // Emit aggregate values from accumulators
            let acc = &group_accumulators[gi];
            for (ai, &item_idx) in agg_indices.iter().enumerate() {
                let item = &return_clause.items[item_idx];
                let key = return_item_column_name(item);
                let val = match &item.expression {
                    Expression::FunctionCall {
                        name,
                        args,
                        distinct,
                    } => {
                        if *distinct {
                            // DISTINCT aggregation not supported by inline — shouldn't reach here
                            Value::Null
                        } else {
                            match name.as_str() {
                                "count" => Value::Int64(acc.counts[ai]),
                                "sum" => {
                                    if acc.counts[ai] == 0 {
                                        Value::Int64(0)
                                    } else {
                                        // Check if input is integer-typed
                                        let is_int = acc.mins[ai].as_ref().is_some_and(|v| {
                                            matches!(v, Value::Int64(_) | Value::UniqueId(_))
                                        });
                                        if is_int {
                                            Value::Int64(acc.sums[ai] as i64)
                                        } else {
                                            Value::Float64(acc.sums[ai])
                                        }
                                    }
                                }
                                "avg" | "mean" | "average" => {
                                    if acc.counts[ai] == 0 {
                                        Value::Null
                                    } else {
                                        Value::Float64(acc.sums[ai] / acc.counts[ai] as f64)
                                    }
                                }
                                "min" => acc.mins[ai].clone().unwrap_or(Value::Null),
                                "max" => acc.maxs[ai].clone().unwrap_or(Value::Null),
                                _ => {
                                    // Unsupported aggregate — fall back to evaluate
                                    let mut tmp_row = ResultRow::new();
                                    tmp_row
                                        .node_bindings
                                        .insert(node_var.to_string(), *first_node_idx);
                                    self.evaluate_expression(&args[0], &tmp_row)?
                                }
                            }
                        }
                    }
                    _ => Value::Null,
                };
                projected.insert(key, val);
            }

            let mut row = ResultRow::from_projected(projected);
            row.node_bindings
                .insert(node_var.to_string(), *first_node_idx);
            result_rows.push(row);
        }

        // Handle HAVING
        if let Some(ref having) = return_clause.having {
            augment_rows_with_aggregate_keys(&mut result_rows, &return_clause.items);
            result_rows.retain(|row| self.evaluate_predicate(having, row).unwrap_or(false));
        }

        // Handle DISTINCT
        if return_clause.distinct {
            let mut seen = HashSet::new();
            result_rows.retain(|row| {
                let key: Vec<Value> = columns
                    .iter()
                    .map(|c| row.projected.get(c).cloned().unwrap_or(Value::Null))
                    .collect();
                seen.insert(key)
            });
        }

        Ok(ResultSet {
            rows: result_rows,
            columns,
        })
    }

    /// Fused MATCH (n:Type) [WHERE] RETURN expressions ORDER BY expr LIMIT k.
    /// Single-pass scan: iterates nodes, evaluates sort key per node, maintains
    /// K-element top-K via sorted Vec (insertion sort). RETURN expressions are
    /// only evaluated for the K winners. Avoids materializing all rows.
    pub(super) fn execute_fused_node_scan_top_k(
        &self,
        match_clause: &MatchClause,
        where_predicate: Option<&Predicate>,
        return_clause: &ReturnClause,
        sort_expression: &Expression,
        descending: bool,
        limit: usize,
    ) -> Result<ResultSet, String> {
        use crate::graph::core::pattern_matching::PatternElement;

        let pattern = &match_clause.patterns[0];
        let node_pattern = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return Err("FusedNodeScanTopK: expected node pattern".into()),
        };
        let node_var = node_pattern.variable.as_deref().unwrap_or("_n");
        let node_type = node_pattern.node_type.as_deref();

        // Get candidate node indices
        let node_indices: Vec<petgraph::graph::NodeIndex> = if let Some(nt) = node_type {
            if let Some(indices) = self.graph.type_indices.get(nt) {
                indices.to_vec()
            } else {
                Vec::new()
            }
        } else {
            {
                let g = &self.graph.graph;
                g.node_indices().collect()
            }
        };

        // Pattern property filter
        let pattern_executor = if node_pattern.properties.is_some() {
            Some(PatternExecutor::new_lightweight_with_params(
                self.graph,
                None,
                self.params,
            ))
        } else {
            None
        };

        // Pre-fold expressions
        let folded_sort = self.fold_constants_expr(sort_expression);
        let folded_where = where_predicate.map(|p| self.fold_constants_pred(p));
        let folded_where_ref = folded_where.as_ref();

        // Single reusable eval row
        let mut eval_row = ResultRow::new();
        eval_row
            .node_bindings
            .insert(node_var.to_string(), petgraph::graph::NodeIndex::new(0));

        // Top-K: sorted Vec of (sort_value, node_idx). Insertion sort for small K.
        let mut top_k: Vec<(Value, petgraph::graph::NodeIndex)> = Vec::with_capacity(limit + 1);

        for (scan_count, &node_idx) in node_indices.iter().enumerate() {
            // Periodic deadline check
            if scan_count.is_multiple_of(10000) {
                self.check_deadline()?;
            }

            // Pattern property filter
            if let Some(ref props) = node_pattern.properties {
                if !pattern_executor
                    .as_ref()
                    .unwrap()
                    .node_matches_properties_pub(node_idx, props)
                {
                    continue;
                }
            }

            // Set node binding for expression evaluation
            *eval_row.node_bindings.get_mut(node_var).unwrap() = node_idx;

            // WHERE filter
            if let Some(pred) = folded_where_ref {
                if !self.evaluate_predicate(pred, &eval_row).unwrap_or(false) {
                    continue;
                }
            }

            // Evaluate sort key
            let sort_val = self.evaluate_expression(&folded_sort, &eval_row)?;
            if matches!(sort_val, Value::Null) {
                continue;
            }

            // Insert into top-K sorted Vec
            let pos = if descending {
                top_k.partition_point(|(existing, _)| {
                    crate::graph::core::filtering_methods::compare_values(existing, &sort_val)
                        .is_some_and(|o| o != std::cmp::Ordering::Less)
                })
            } else {
                top_k.partition_point(|(existing, _)| {
                    crate::graph::core::filtering_methods::compare_values(existing, &sort_val)
                        .is_some_and(|o| o != std::cmp::Ordering::Greater)
                })
            };
            if pos < limit {
                top_k.insert(pos, (sort_val, node_idx));
                if top_k.len() > limit {
                    top_k.pop();
                }
            }
        }

        // Build RETURN expressions only for the K winners
        let folded_return_exprs: Vec<Expression> = return_clause
            .items
            .iter()
            .map(|item| self.fold_constants_expr(&item.expression))
            .collect();
        let columns: Vec<String> = return_clause
            .items
            .iter()
            .map(return_item_column_name)
            .collect();

        let mut result_rows = Vec::with_capacity(top_k.len());
        for (_, winner_idx) in &top_k {
            *eval_row.node_bindings.get_mut(node_var).unwrap() = *winner_idx;
            let mut projected = Bindings::with_capacity(columns.len());
            for (j, expr) in folded_return_exprs.iter().enumerate() {
                let val = self.evaluate_expression(expr, &eval_row)?;
                projected.insert(columns[j].clone(), val);
            }
            result_rows.push(ResultRow::from_projected(projected));
        }

        Ok(ResultSet {
            rows: result_rows,
            columns,
        })
    }

    /// Fused MATCH + WITH count() — same as `execute_fused_match_return_aggregate`
    /// but produces ResultSet for pipeline continuation (WITH semantics).
    pub(super) fn execute_fused_match_with_aggregate(
        &self,
        match_clause: &MatchClause,
        with_clause: &WithClause,
        _existing: ResultSet,
    ) -> Result<ResultSet, String> {
        let pattern = &match_clause.patterns[0];

        let first_var = match &pattern.elements[0] {
            PatternElement::Node(np) => np.variable.as_ref(),
            _ => return Err("FusedMatchWithAggregate: expected node pattern".into()),
        };
        let second_var = match &pattern.elements[2] {
            PatternElement::Node(np) => np.variable.as_ref(),
            _ => return Err("FusedMatchWithAggregate: expected node pattern".into()),
        };

        // Determine which variable is the group key
        let group_var: &str = {
            let mut gv = None;
            for item in &with_clause.items {
                if !is_aggregate_expression(&item.expression) {
                    if let Expression::Variable(v) = &item.expression {
                        gv = Some(v.as_str());
                        break;
                    }
                }
            }
            gv.ok_or("FusedMatchWithAggregate: no group-by variable found")?
        };

        let group_elem_idx = if first_var.is_some_and(|v| v == group_var) {
            0
        } else if second_var.is_some_and(|v| v == group_var) {
            2
        } else {
            return Err("FusedMatchWithAggregate: group variable not in pattern".into());
        };

        // Build single-node pattern for matching group keys
        let group_only_pattern = crate::graph::core::pattern_matching::Pattern {
            elements: vec![pattern.elements[group_elem_idx].clone()],
        };

        let executor = PatternExecutor::new_lightweight_with_params(self.graph, None, self.params)
            .set_deadline(self.deadline);
        let group_matches = executor.execute(&group_only_pattern)?;

        // Identify group key and count items
        let mut group_key_indices = Vec::new();
        let mut count_indices = Vec::new();
        for (i, item) in with_clause.items.iter().enumerate() {
            if is_aggregate_expression(&item.expression) {
                count_indices.push(i);
            } else {
                group_key_indices.push(i);
            }
        }

        let columns: Vec<String> = with_clause
            .items
            .iter()
            .map(|item| {
                item.alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}", item.expression))
            })
            .collect();

        let mut result_rows = Vec::with_capacity(group_matches.len());

        for m in &group_matches {
            let node_idx = m.bindings.iter().find_map(|(name, binding)| {
                if name == group_var {
                    match binding {
                        MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                            Some(*index)
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            });
            let Some(node_idx) = node_idx else {
                continue;
            };

            let mut bindings_for_count = Bindings::with_capacity(1);
            bindings_for_count.insert(group_var.to_string(), node_idx);
            let match_count = self
                .try_count_simple_pattern(pattern, &bindings_for_count)?
                .unwrap_or(0);

            // Skip nodes with 0 matches (MATCH semantics — no outer join)
            if match_count == 0 {
                continue;
            }

            // Build a temporary row for evaluating group-key expressions
            let mut tmp_row = ResultRow::new();
            tmp_row
                .node_bindings
                .insert(group_var.to_string(), node_idx);

            let mut projected = Bindings::with_capacity(with_clause.items.len());

            for &idx in &group_key_indices {
                let item = &with_clause.items[idx];
                let key = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}", item.expression));
                let val = self.evaluate_expression(&item.expression, &tmp_row)?;
                projected.insert(key, val);
            }

            for &idx in &count_indices {
                let item = &with_clause.items[idx];
                let key = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}", item.expression));
                projected.insert(key, Value::Int64(match_count));
            }

            let mut new_row = ResultRow::from_projected(projected);
            new_row
                .node_bindings
                .insert(group_var.to_string(), node_idx);
            result_rows.push(new_row);
        }

        // Apply WITH WHERE filter if present
        if let Some(ref where_clause) = with_clause.where_clause {
            let folded = self.fold_constants_pred(&where_clause.predicate);
            result_rows.retain(|row| self.evaluate_predicate(&folded, row).unwrap_or(false));
        }

        Ok(ResultSet {
            rows: result_rows,
            columns,
        })
    }
}
