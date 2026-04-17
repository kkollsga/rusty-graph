//! Cypher executor — call_clause methods.

use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

impl<'a> CypherExecutor<'a> {
    pub(super) fn execute_unwind(
        &self,
        clause: &UnwindClause,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        self.check_deadline()?;
        let mut new_rows = Vec::new();

        // Use into_iter to own rows — enables move-on-last optimization
        for mut row in result_set.rows {
            let val = self.evaluate_expression(&clause.expression, &row)?;
            match val {
                Value::String(s) if s.starts_with('[') && s.ends_with(']') => {
                    let items = split_list_top_level(&s);
                    let total = items.len();
                    for (i, item_str) in items.into_iter().enumerate() {
                        let parsed_val = parse_value_string(item_str.trim());
                        if i + 1 == total {
                            // Last item: move row instead of cloning
                            row.projected.insert(clause.alias.clone(), parsed_val);
                            new_rows.push(row);
                            break;
                        }
                        let mut new_row = row.clone();
                        new_row.projected.insert(clause.alias.clone(), parsed_val);
                        new_rows.push(new_row);
                    }
                }
                Value::Null => {
                    // UNWIND null produces zero rows per Cypher spec
                }
                _ => {
                    // Single value: move directly (no clone needed)
                    row.projected.insert(clause.alias.clone(), val);
                    new_rows.push(row);
                }
            }
        }

        Ok(ResultSet {
            rows: new_rows,
            columns: result_set.columns,
        })
    }

    // ========================================================================
    // CALL (graph algorithm procedures)
    // ========================================================================

    pub(super) fn execute_call(
        &self,
        clause: &CallClause,
        existing: ResultSet,
    ) -> Result<ResultSet, String> {
        self.check_deadline()?;

        let proc_name = clause.procedure_name.to_lowercase();

        // Validate YIELD columns
        let valid_yields: &[&str] = match proc_name.as_str() {
            "pagerank"
            | "betweenness"
            | "betweenness_centrality"
            | "degree"
            | "degree_centrality"
            | "closeness"
            | "closeness_centrality" => &["node", "score"],
            "louvain" | "louvain_communities" | "label_propagation" => &["node", "community"],
            "connected_components" | "weakly_connected_components" => &["node", "component"],
            "cluster" => &["node", "cluster"],
            "list_procedures" => &["name", "description", "yield_columns"],
            _ => {
                return Err(format!(
                    "Unknown procedure '{}'. Available: pagerank, betweenness, degree, \
                     closeness, louvain, label_propagation, connected_components, \
                     cluster, list_procedures",
                    clause.procedure_name
                ));
            }
        };

        for item in &clause.yield_items {
            if !valid_yields.contains(&item.name.as_str()) {
                return Err(format!(
                    "Procedure '{}' does not yield '{}'. Available: {}",
                    clause.procedure_name,
                    item.name,
                    valid_yields.join(", ")
                ));
            }
        }

        // Extract parameters
        let params = self.extract_call_params(&clause.parameters)?;

        // Dispatch to algorithm
        let rows = match proc_name.as_str() {
            "pagerank" => {
                let damping = call_param_f64(&params, "damping_factor", 0.85);
                let max_iter = call_param_usize(&params, "max_iterations", 100);
                let tolerance = call_param_f64(&params, "tolerance", 1e-6);
                let conn = call_param_string_list(&params, "connection_types");
                let results = crate::graph::algorithms::graph_algorithms::pagerank(
                    self.graph,
                    damping,
                    max_iter,
                    tolerance,
                    conn.as_deref(),
                    self.deadline,
                );
                self.centrality_to_rows(&results, &clause.yield_items)
            }
            "betweenness" | "betweenness_centrality" => {
                let normalized = call_param_bool(&params, "normalized", true);
                let sample_size = call_param_opt_usize(&params, "sample_size");
                let conn = call_param_string_list(&params, "connection_types");
                let results = crate::graph::algorithms::graph_algorithms::betweenness_centrality(
                    self.graph,
                    normalized,
                    sample_size,
                    conn.as_deref(),
                    self.deadline,
                );
                self.centrality_to_rows(&results, &clause.yield_items)
            }
            "degree" | "degree_centrality" => {
                let normalized = call_param_bool(&params, "normalized", true);
                let conn = call_param_string_list(&params, "connection_types");
                let results = crate::graph::algorithms::graph_algorithms::degree_centrality(
                    self.graph,
                    normalized,
                    conn.as_deref(),
                    self.deadline,
                );
                self.centrality_to_rows(&results, &clause.yield_items)
            }
            "closeness" | "closeness_centrality" => {
                let normalized = call_param_bool(&params, "normalized", true);
                let sample_size = call_param_opt_usize(&params, "sample_size");
                let conn = call_param_string_list(&params, "connection_types");
                let results = crate::graph::algorithms::graph_algorithms::closeness_centrality(
                    self.graph,
                    normalized,
                    sample_size,
                    conn.as_deref(),
                    self.deadline,
                );
                self.centrality_to_rows(&results, &clause.yield_items)
            }
            "louvain" | "louvain_communities" => {
                let resolution = call_param_f64(&params, "resolution", 1.0);
                let weight_prop = call_param_opt_string(&params, "weight_property");
                let conn = call_param_string_list(&params, "connection_types");
                let result = crate::graph::algorithms::graph_algorithms::louvain_communities(
                    self.graph,
                    weight_prop.as_deref(),
                    resolution,
                    conn.as_deref(),
                    self.deadline,
                );
                self.community_to_rows(&result.assignments, &clause.yield_items)
            }
            "label_propagation" => {
                let max_iter = call_param_usize(&params, "max_iterations", 100);
                let conn = call_param_string_list(&params, "connection_types");
                let result = crate::graph::algorithms::graph_algorithms::label_propagation(
                    self.graph,
                    max_iter,
                    conn.as_deref(),
                    self.deadline,
                );
                self.community_to_rows(&result.assignments, &clause.yield_items)
            }
            "connected_components" | "weakly_connected_components" => {
                let components =
                    crate::graph::algorithms::graph_algorithms::weakly_connected_components(
                        self.graph,
                    );
                let mut rows = Vec::new();
                for (comp_id, nodes) in components.iter().enumerate() {
                    for &node_idx in nodes {
                        let mut row = ResultRow::new();
                        for item in &clause.yield_items {
                            let alias = item.alias.as_deref().unwrap_or(&item.name);
                            match item.name.as_str() {
                                "node" => {
                                    row.node_bindings.insert(alias.to_string(), node_idx);
                                }
                                "component" => {
                                    row.projected
                                        .insert(alias.to_string(), Value::Int64(comp_id as i64));
                                }
                                _ => {}
                            }
                        }
                        rows.push(row);
                    }
                }
                rows
            }
            "cluster" => self.execute_call_cluster(&params, &clause.yield_items, &existing)?,
            "list_procedures" => {
                let procedures = [
                    (
                        "pagerank",
                        "Compute PageRank centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "betweenness",
                        "Compute betweenness centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "degree",
                        "Compute degree centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "closeness",
                        "Compute closeness centrality for all nodes",
                        "node, score",
                    ),
                    (
                        "louvain",
                        "Detect communities using the Louvain algorithm",
                        "node, community",
                    ),
                    (
                        "label_propagation",
                        "Detect communities using label propagation",
                        "node, community",
                    ),
                    (
                        "connected_components",
                        "Find weakly connected components",
                        "node, component",
                    ),
                    (
                        "cluster",
                        "Cluster nodes by spatial location or numeric properties (DBSCAN/K-means). Reads from preceding MATCH.",
                        "node, cluster",
                    ),
                    (
                        "list_procedures",
                        "List all available procedures",
                        "name, description, yield_columns",
                    ),
                ];
                let mut rows = Vec::new();
                for (name, desc, yields) in &procedures {
                    let mut row = ResultRow::new();
                    for item in &clause.yield_items {
                        let alias = item.alias.as_deref().unwrap_or(&item.name);
                        match item.name.as_str() {
                            "name" => {
                                row.projected
                                    .insert(alias.to_string(), Value::String(name.to_string()));
                            }
                            "description" => {
                                row.projected
                                    .insert(alias.to_string(), Value::String(desc.to_string()));
                            }
                            "yield_columns" => {
                                row.projected
                                    .insert(alias.to_string(), Value::String(yields.to_string()));
                            }
                            _ => {}
                        }
                    }
                    rows.push(row);
                }
                rows
            }
            _ => unreachable!(),
        };

        Ok(ResultSet {
            rows,
            columns: Vec::new(),
        })
    }

    /// Extract CALL parameters from {key: expr} pairs into a value map.
    pub(super) fn extract_call_params(
        &self,
        params: &[(String, Expression)],
    ) -> Result<HashMap<String, Value>, String> {
        let empty_row = ResultRow::new();
        let mut map = HashMap::new();
        for (key, expr) in params {
            let val = self.evaluate_expression(expr, &empty_row)?;
            map.insert(key.clone(), val);
        }
        Ok(map)
    }

    /// Execute CALL cluster() — cluster nodes from the preceding MATCH result set.
    pub(super) fn execute_call_cluster(
        &self,
        params: &HashMap<String, Value>,
        yield_items: &[YieldItem],
        existing: &ResultSet,
    ) -> Result<Vec<ResultRow>, String> {
        // Extract parameters
        let method = call_param_opt_string(params, "method")
            .unwrap_or_else(|| "dbscan".to_string())
            .to_lowercase();
        let eps = call_param_f64(params, "eps", 0.5);
        let min_points = call_param_usize(params, "min_points", 3);
        let k = call_param_usize(params, "k", 5);
        let max_iterations = call_param_usize(params, "max_iterations", 100);
        let normalize = call_param_bool(params, "normalize", false);

        // Extract property list (if given)
        let properties: Option<Vec<String>> = params.get("properties").and_then(|v| {
            let items = parse_list_value(v);
            if items.is_empty() {
                return None;
            }
            let strs: Vec<String> = items
                .into_iter()
                .filter_map(|item| match item {
                    Value::String(s) => Some(s),
                    _ => None,
                })
                .collect();
            if strs.is_empty() {
                None
            } else {
                Some(strs)
            }
        });

        // Collect unique node indices from the existing result set
        let mut node_indices: Vec<NodeIndex> = Vec::new();
        let mut seen: HashSet<NodeIndex> = HashSet::new();
        for row in &existing.rows {
            for (_, &idx) in row.node_bindings.iter() {
                if seen.insert(idx) {
                    node_indices.push(idx);
                }
            }
        }

        if node_indices.is_empty() {
            return Err("cluster() requires a preceding MATCH clause that binds nodes".to_string());
        }

        // Validate method
        if method != "dbscan" && method != "kmeans" {
            return Err(format!(
                "Unknown clustering method '{}'. Available: dbscan, kmeans",
                method
            ));
        }

        // Build feature vectors and run clustering
        let assignments = if let Some(ref prop_names) = properties {
            // ── Explicit property mode ──
            // Extract numeric features from named properties
            let mut features: Vec<Vec<f64>> = Vec::new();
            let mut valid_indices: Vec<usize> = Vec::new(); // indices into node_indices

            for (i, &idx) in node_indices.iter().enumerate() {
                if let Some(node) = self.graph.graph.node_weight(idx) {
                    let mut vals = Vec::with_capacity(prop_names.len());
                    let mut all_present = true;
                    for prop in prop_names {
                        if let Some(val) = node.get_property(prop) {
                            if let Some(f) = value_to_f64(&val) {
                                vals.push(f);
                            } else {
                                all_present = false;
                                break;
                            }
                        } else {
                            all_present = false;
                            break;
                        }
                    }
                    if all_present {
                        features.push(vals);
                        valid_indices.push(i);
                    }
                }
            }

            if features.is_empty() {
                return Err(format!(
                    "No nodes have all required numeric properties: {:?}",
                    prop_names
                ));
            }

            if normalize {
                crate::graph::algorithms::clustering::normalize_features(&mut features);
            }

            let cluster_assignments = match method.as_str() {
                "dbscan" => {
                    let dm =
                        crate::graph::algorithms::clustering::euclidean_distance_matrix(&features);
                    crate::graph::algorithms::clustering::dbscan(&dm, eps, min_points)
                }
                "kmeans" => {
                    crate::graph::algorithms::clustering::kmeans(&features, k, max_iterations)
                }
                _ => unreachable!(),
            };

            // Map back to original node_indices
            cluster_assignments
                .into_iter()
                .map(|ca| (node_indices[valid_indices[ca.index]], ca.cluster))
                .collect::<Vec<_>>()
        } else {
            // ── Spatial mode ──
            // Auto-detect lat/lon from spatial config
            let mut points: Vec<(f64, f64)> = Vec::new();
            let mut valid_indices: Vec<usize> = Vec::new();

            for (i, &idx) in node_indices.iter().enumerate() {
                if let Some(node) = self.graph.graph.node_weight(idx) {
                    // Try spatial config for this node type
                    if let Some(config) = self
                        .graph
                        .get_spatial_config(node.node_type_str(&self.graph.interner))
                    {
                        let (lat_f, lon_f) = config
                            .location
                            .as_ref()
                            .map(|(a, b)| (a.as_str(), b.as_str()))
                            .unwrap_or(("latitude", "longitude"));
                        let geom_fallback = config.geometry.as_deref();

                        if let Some((lat, lon)) = crate::graph::features::spatial::node_location(
                            node,
                            lat_f,
                            lon_f,
                            geom_fallback,
                        ) {
                            points.push((lat, lon));
                            valid_indices.push(i);
                        }
                    }
                }
            }

            if points.is_empty() {
                return Err(
                    "No nodes have spatial data. Either configure spatial fields with \
                     set_spatial_config() or provide explicit 'properties' parameter."
                        .to_string(),
                );
            }

            let cluster_assignments = match method.as_str() {
                "dbscan" => {
                    let dm =
                        crate::graph::algorithms::clustering::haversine_distance_matrix(&points);
                    crate::graph::algorithms::clustering::dbscan(&dm, eps, min_points)
                }
                "kmeans" => {
                    // For spatial k-means, convert to feature vectors [lat, lon]
                    let features: Vec<Vec<f64>> =
                        points.iter().map(|(lat, lon)| vec![*lat, *lon]).collect();
                    crate::graph::algorithms::clustering::kmeans(&features, k, max_iterations)
                }
                _ => unreachable!(),
            };

            cluster_assignments
                .into_iter()
                .map(|ca| (node_indices[valid_indices[ca.index]], ca.cluster))
                .collect::<Vec<_>>()
        };

        // Build result rows
        let mut rows = Vec::with_capacity(assignments.len());
        for (node_idx, cluster_id) in &assignments {
            let mut row = ResultRow::new();
            for item in yield_items {
                let alias = item.alias.as_deref().unwrap_or(&item.name);
                match item.name.as_str() {
                    "node" => {
                        row.node_bindings.insert(alias.to_string(), *node_idx);
                    }
                    "cluster" => {
                        row.projected
                            .insert(alias.to_string(), Value::Int64(*cluster_id));
                    }
                    _ => {}
                }
            }
            rows.push(row);
        }

        Ok(rows)
    }

    /// Convert centrality results to ResultRows with node bindings + score.
    pub(super) fn centrality_to_rows(
        &self,
        results: &[crate::graph::algorithms::graph_algorithms::CentralityResult],
        yield_items: &[YieldItem],
    ) -> Vec<ResultRow> {
        results
            .iter()
            .map(|cr| {
                let mut row = ResultRow::new();
                for item in yield_items {
                    let alias = item.alias.as_deref().unwrap_or(&item.name);
                    match item.name.as_str() {
                        "node" => {
                            row.node_bindings.insert(alias.to_string(), cr.node_idx);
                        }
                        "score" => {
                            row.projected
                                .insert(alias.to_string(), Value::Float64(cr.score));
                        }
                        _ => {}
                    }
                }
                row
            })
            .collect()
    }

    /// Convert community assignments to ResultRows with node bindings + community id.
    pub(super) fn community_to_rows(
        &self,
        assignments: &[crate::graph::algorithms::graph_algorithms::CommunityAssignment],
        yield_items: &[YieldItem],
    ) -> Vec<ResultRow> {
        assignments
            .iter()
            .map(|ca| {
                let mut row = ResultRow::new();
                for item in yield_items {
                    let alias = item.alias.as_deref().unwrap_or(&item.name);
                    match item.name.as_str() {
                        "node" => {
                            row.node_bindings.insert(alias.to_string(), ca.node_idx);
                        }
                        "community" => {
                            row.projected
                                .insert(alias.to_string(), Value::Int64(ca.community_id as i64));
                        }
                        _ => {}
                    }
                }
                row
            })
            .collect()
    }

    // ========================================================================
    // UNION
    // ========================================================================

    pub(super) fn execute_union(
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
            let mut projected = Bindings::with_capacity(right_result.columns.len());
            for (i, col) in right_result.columns.iter().enumerate() {
                if let Some(val) = row_values.get(i) {
                    projected.insert(col.clone(), val.clone());
                }
            }
            combined_rows.push(ResultRow::from_projected(projected));
        }

        // Remove duplicates for UNION (not UNION ALL)
        // Use hash-based dedup to avoid cloning Vec<Value> per row
        if !clause.all {
            use std::hash::{Hash, Hasher};
            let mut seen = HashSet::new();
            combined_rows.retain(|row| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                for col in &columns {
                    match row.projected.get(col) {
                        Some(val) => val.hash(&mut hasher),
                        None => Value::Null.hash(&mut hasher),
                    }
                }
                seen.insert(hasher.finish())
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
                profile: None,
            });
        }

        // RETURN was specified - use its columns
        let rows: Vec<Vec<Value>> = if result_set.rows.len() >= RAYON_THRESHOLD {
            let cols = &result_set.columns;
            result_set
                .rows
                .par_iter()
                .map(|row| {
                    cols.iter()
                        .map(|col| row.projected.get(col).cloned().unwrap_or(Value::Null))
                        .collect()
                })
                .collect()
        } else {
            // Move values out of rows (no cloning)
            let cols = &result_set.columns;
            result_set
                .rows
                .into_iter()
                .map(|mut row| {
                    cols.iter()
                        .map(|col| row.projected.remove(col).unwrap_or(Value::Null))
                        .collect()
                })
                .collect()
        };

        Ok(CypherResult {
            columns: result_set.columns,
            rows,
            stats: None,
            profile: None,
        })
    }
}
