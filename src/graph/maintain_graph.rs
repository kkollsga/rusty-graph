// src/graph/maintain_graph.rs
use crate::datatypes::{DataFrame, Value};
use crate::graph::batch_operations::{
    BatchProcessor, ConflictHandling, ConnectionBatchProcessor, NodeAction,
};
use crate::graph::lookups::{CombinedTypeLookup, TypeLookup};
use crate::graph::reporting::{ConnectionOperationReport, NodeOperationReport};
use crate::graph::schema::{CurrentSelection, DirGraph};
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

fn check_data_validity(df_data: &DataFrame, unique_id_field: &str) -> Result<(), String> {
    // Remove strict UniqueId type verification to allow nulls
    if !df_data.verify_column(unique_id_field) {
        let available_cols: Vec<_> = df_data.get_column_names();
        return Err(format!(
            "Column '{}' not found in DataFrame. Available columns: [{}]",
            unique_id_field,
            available_cols.join(", ")
        ));
    }
    Ok(())
}

fn get_column_types(df_data: &DataFrame) -> HashMap<String, String> {
    let mut types = HashMap::new();
    for col_name in df_data.get_column_names() {
        let col_type = df_data.get_column_type(&col_name);
        types.insert(col_name.clone(), col_type.to_string());
    }
    types
}

pub fn add_nodes(
    graph: &mut DirGraph,
    df_data: DataFrame,
    node_type: String,
    unique_id_field: String,
    node_title_field: Option<String>,
    conflict_handling: Option<String>,
) -> Result<NodeOperationReport, String> {
    // Parse conflict handling option
    let conflict_mode = match conflict_handling.as_deref() {
        Some("replace") => ConflictHandling::Replace,
        Some("skip") => ConflictHandling::Skip,
        Some("preserve") => ConflictHandling::Preserve,
        Some("update") | None => ConflictHandling::Update, // Default
        Some(other) => return Err(format!(
            "Unknown conflict handling mode: '{}'. Valid options: 'update' (default), 'replace', 'skip', 'preserve'",
            other
        )),
    };

    let should_update_title = node_title_field.is_some();
    let title_field = node_title_field.unwrap_or_else(|| unique_id_field.clone());
    check_data_validity(&df_data, &unique_id_field)?;

    // Track errors
    let mut errors = Vec::new();

    let df_column_types = get_column_types(&df_data);

    // Check for type mismatches if metadata already exists
    if let Some(existing_meta) = graph.get_node_type_metadata(&node_type) {
        for (col_name, col_type) in &df_column_types {
            if let Some(existing_type) = existing_meta.get(col_name) {
                if existing_type != col_type {
                    errors.push(format!(
                        "Type mismatch for property '{}': existing schema has '{}', but data has '{}'",
                        col_name, existing_type, col_type
                    ));
                }
            }
        }
    }

    // Upsert node type metadata (merges new column types into existing)
    graph.upsert_node_type_metadata(&node_type, df_column_types);

    let type_lookup = TypeLookup::new(&graph.graph, node_type.clone())?;
    let id_idx = df_data
        .get_column_index(&unique_id_field)
        .ok_or_else(|| format!("Column '{}' not found", unique_id_field))?;
    let title_idx = df_data
        .get_column_index(&title_field)
        .ok_or_else(|| format!("Column '{}' not found", title_field))?;

    // OPTIMIZATION: Pre-compute property column info (name + index) to avoid repeated lookups
    // This avoids: 1) string comparisons in the loop, 2) HashMap lookups per property
    let property_columns: Vec<(String, usize)> = df_data
        .get_column_names()
        .into_iter()
        .filter_map(|col_name| {
            if col_name != unique_id_field && col_name != title_field {
                df_data
                    .get_column_index(&col_name)
                    .map(|idx| (col_name, idx))
            } else {
                None
            }
        })
        .collect();

    let property_count = property_columns.len();
    let mut batch = BatchProcessor::new(df_data.row_count());
    let mut skipped_count = 0;

    for row_idx in 0..df_data.row_count() {
        let id = match df_data.get_value_by_index(row_idx, id_idx) {
            Some(Value::Null) => {
                skipped_count += 1;
                continue;
            }
            Some(id) => id,
            None => {
                skipped_count += 1;
                continue;
            }
        };

        let title = df_data
            .get_value_by_index(row_idx, title_idx)
            .unwrap_or(Value::Null);

        // OPTIMIZATION: Use pre-computed indices for direct column access.
        // Skip null values — property access returns Null for missing keys anyway.
        let mut properties = HashMap::with_capacity(property_count);
        for (col_name, col_idx) in &property_columns {
            let value = df_data
                .get_value_by_index(row_idx, *col_idx)
                .unwrap_or(Value::Null);
            if !matches!(value, Value::Null) {
                properties.insert(col_name.clone(), value);
            }
        }

        let action = match type_lookup.check_uid(&id) {
            Some(node_idx) => {
                // Determine if we should update the title
                let title_update = if should_update_title {
                    Some(title)
                } else {
                    None
                };

                NodeAction::Update {
                    node_idx,
                    title: title_update,
                    properties,
                    conflict_mode,
                }
            }
            None => NodeAction::Create {
                node_type: node_type.clone(),
                id,
                title,
                properties,
            },
        };
        batch.add_action(action, graph)?;
    }

    // Execute the batch and get the statistics
    let (stats, metrics) = batch.execute(graph)?;

    // Calculate elapsed time
    let elapsed_ms = metrics.processing_time * 1000.0; // Convert to milliseconds

    // Create and return the operation report with timestamp and errors
    let mut report = NodeOperationReport::new(
        "add_nodes".to_string(),
        stats.creates,
        stats.updates,
        skipped_count,
        elapsed_ms,
    );

    // Add errors if we found any
    if !errors.is_empty() {
        report = report.with_errors(errors);
    }

    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub fn add_connections(
    graph: &mut DirGraph,
    df_data: DataFrame,
    connection_type: String,
    source_type: String,
    source_id_field: String,
    target_type: String,
    target_id_field: String,
    source_title_field: Option<String>,
    target_title_field: Option<String>,
    conflict_handling: Option<String>,
) -> Result<ConnectionOperationReport, String> {
    // Parse conflict handling option
    let conflict_mode = match conflict_handling.as_deref() {
        Some("replace") => ConflictHandling::Replace,
        Some("skip") => ConflictHandling::Skip,
        Some("preserve") => ConflictHandling::Preserve,
        Some("update") | None => ConflictHandling::Update, // Default
        Some(other) => return Err(format!(
            "Unknown conflict handling mode: '{}'. Valid options: 'update' (default), 'replace', 'skip', 'preserve'",
            other
        )),
    };

    // Track errors
    let mut errors = Vec::new();

    let available_cols: Vec<_> = df_data.get_column_names();
    if !df_data.verify_column(&source_id_field) {
        return Err(format!(
            "Source ID column '{}' not found in DataFrame. Available columns: [{}]",
            source_id_field,
            available_cols.join(", ")
        ));
    }
    if !df_data.verify_column(&target_id_field) {
        return Err(format!(
            "Target ID column '{}' not found in DataFrame. Available columns: [{}]",
            target_id_field,
            available_cols.join(", ")
        ));
    }

    // Check if source and target types exist
    if !graph.has_node_type(&source_type) {
        errors.push(format!(
            "Source node type '{}' does not exist in the graph",
            source_type
        ));
    }

    if !graph.has_node_type(&target_type) {
        errors.push(format!(
            "Target node type '{}' does not exist in the graph",
            target_type
        ));
    }

    let source_id_idx = df_data
        .get_column_index(&source_id_field)
        .ok_or_else(|| format!("Source ID column '{}' not found", source_id_field))?;
    let target_id_idx = df_data
        .get_column_index(&target_id_field)
        .ok_or_else(|| format!("Target ID column '{}' not found", target_id_field))?;

    // Use as_ref() to borrow rather than move
    let source_title_idx = source_title_field
        .as_ref()
        .and_then(|field| df_data.get_column_index(field));
    let target_title_idx = target_title_field
        .as_ref()
        .and_then(|field| df_data.get_column_index(field));

    let lookup = CombinedTypeLookup::new(&graph.graph, source_type.clone(), target_type.clone())?;
    let mut batch = ConnectionBatchProcessor::new(df_data.row_count());
    // Set the conflict handling mode
    batch.set_conflict_mode(conflict_mode);

    let mut skipped_count = 0;
    // Instead of tracking ids directly, track counts of missing items
    let mut missing_source_count = 0;
    let mut missing_target_count = 0;

    // Cache column names and pre-compute which columns are property columns (not ID or title fields)
    // This avoids repeated allocations and string comparisons in the loop
    let property_columns: Vec<String> = df_data
        .get_column_names()
        .into_iter()
        .filter(|col_name| {
            let is_id_field = *col_name == source_id_field || *col_name == target_id_field;
            let is_source_title = source_title_field
                .as_ref()
                .is_some_and(|field| *col_name == *field);
            let is_target_title = target_title_field
                .as_ref()
                .is_some_and(|field| *col_name == *field);
            !is_id_field && !is_source_title && !is_target_title
        })
        .collect();

    for row_idx in 0..df_data.row_count() {
        let source_id = match df_data.get_value_by_index(row_idx, source_id_idx) {
            Some(Value::Null) | None => {
                skipped_count += 1;
                continue;
            }
            Some(id) => id,
        };

        let target_id = match df_data.get_value_by_index(row_idx, target_id_idx) {
            Some(Value::Null) | None => {
                skipped_count += 1;
                continue;
            }
            Some(id) => id,
        };

        let (source_idx, target_idx) = match (
            lookup.check_source(&source_id),
            lookup.check_target(&target_id),
        ) {
            (Some(src_idx), Some(tgt_idx)) => (src_idx, tgt_idx),
            (None, Some(_)) => {
                // Track missing source node
                missing_source_count += 1;
                skipped_count += 1;
                continue;
            }
            (Some(_), None) => {
                // Track missing target node
                missing_target_count += 1;
                skipped_count += 1;
                continue;
            }
            (None, None) => {
                // Track both missing
                missing_source_count += 1;
                missing_target_count += 1;
                skipped_count += 1;
                continue;
            }
        };

        update_node_titles(
            graph,
            source_idx,
            target_idx,
            row_idx,
            source_title_idx,
            target_title_idx,
            &df_data,
        )?;

        // Use pre-computed property columns (avoids get_column_names() call per row).
        // Skip null values — property access returns Null for missing keys anyway.
        let mut properties = HashMap::with_capacity(property_columns.len());
        for col_name in &property_columns {
            if let Some(value) = df_data.get_value(row_idx, col_name) {
                if !matches!(value, Value::Null) {
                    properties.insert(col_name.clone(), value);
                }
            }
        }

        // This will respect the conflict handling mode we set earlier
        if let Err(e) =
            batch.add_connection(source_idx, target_idx, properties, graph, &connection_type)
        {
            skipped_count += 1;
            errors.push(format!("Failed to add connection: {}", e));
            continue;
        }
    }

    // Add missing nodes as errors
    if missing_source_count > 0 {
        errors.push(format!(
            "Missing source nodes: {} occurrences",
            missing_source_count
        ));
    }
    if missing_target_count > 0 {
        errors.push(format!(
            "Missing target nodes: {} occurrences",
            missing_target_count
        ));
    }

    update_schema_node(
        graph,
        &connection_type,
        lookup.get_source_type(),
        lookup.get_target_type(),
        batch.get_schema_properties(),
    )?;

    // Execute the batch and get the statistics
    let (stats, metrics) = batch.execute(graph, connection_type)?;

    // Create and return the operation report
    let mut report = ConnectionOperationReport::new(
        "add_connections".to_string(),
        stats.connections_created,
        skipped_count,
        stats.properties_tracked,
        metrics.processing_time * 1000.0, // Convert to milliseconds
    );

    // Add errors if we found any
    if !errors.is_empty() {
        report = report.with_errors(errors);
    }

    Ok(report)
}

fn update_node_titles(
    graph: &mut DirGraph,
    source_idx: NodeIndex,
    target_idx: NodeIndex,
    row_idx: usize,
    source_title_idx: Option<usize>,
    target_title_idx: Option<usize>,
    df_data: &DataFrame,
) -> Result<(), String> {
    if let Some(title_idx) = source_title_idx {
        if let Some(title) = df_data.get_value_by_index(row_idx, title_idx) {
            if let Some(node) = graph.get_node_mut(source_idx) {
                node.title = title;
            }
        }
    }
    if let Some(title_idx) = target_title_idx {
        if let Some(title) = df_data.get_value_by_index(row_idx, title_idx) {
            if let Some(node) = graph.get_node_mut(target_idx) {
                node.title = title;
            }
        }
    }
    Ok(())
}

fn update_schema_node(
    graph: &mut DirGraph,
    connection_type: &str,
    source_type: &str,
    target_type: &str,
    properties: &HashSet<String>,
) -> Result<(), String> {
    if !graph.has_node_type(source_type) {
        return Err(format!(
            "Source type '{}' does not exist in graph",
            source_type
        ));
    }
    if !graph.has_node_type(target_type) {
        return Err(format!(
            "Target type '{}' does not exist in graph",
            target_type
        ));
    }

    // Build property type map — all connection properties default to "Unknown"
    let prop_types: HashMap<String, String> = properties
        .iter()
        .map(|prop| (prop.clone(), "Unknown".to_string()))
        .collect();

    graph.upsert_connection_type_metadata(connection_type, source_type, target_type, prop_types);
    Ok(())
}

pub fn selection_to_new_connections(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    connection_type: String,
    conflict_handling: Option<String>, // Add conflict_handling parameter
) -> Result<ConnectionOperationReport, String> {
    // Parse conflict handling option
    let conflict_mode = match conflict_handling.as_deref() {
        Some("replace") => ConflictHandling::Replace,
        Some("skip") => ConflictHandling::Skip,
        Some("preserve") => ConflictHandling::Preserve,
        Some("update") | None => ConflictHandling::Update, // Default
        Some(other) => return Err(format!(
            "Unknown conflict handling mode: '{}'. Valid options: 'update' (default), 'replace', 'skip', 'preserve'",
            other
        )),
    };

    // Track errors
    let mut errors = Vec::new();

    let current_level = selection.get_level_count().saturating_sub(1);
    let level = match selection.get_level(current_level) {
        Some(level) if !level.is_empty() => level,
        _ => {
            // Return empty report since there's nothing to do
            let report = ConnectionOperationReport::new(
                "selection_to_new_connections".to_string(),
                0,
                0,
                0,
                0.0,
            );
            return Ok(report);
        }
    };

    let mut batch = ConnectionBatchProcessor::new(level.node_count());
    // Set the conflict handling mode
    batch.set_conflict_mode(conflict_mode);

    let mut skipped = 0;
    let mut source_type = None;
    let mut target_type = None;

    for (parent_opt, children) in level.iter_groups() {
        if let Some(parent) = parent_opt {
            if source_type.is_none() {
                if let Some(node) = graph.get_node(*parent) {
                    source_type = Some(node.node_type.clone());
                }
            }

            for &child in children {
                if target_type.is_none() {
                    if let Some(node) = graph.get_node(child) {
                        target_type = Some(node.node_type.clone());
                    }
                }

                if let Err(e) =
                    batch.add_connection(*parent, child, HashMap::new(), graph, &connection_type)
                {
                    skipped += 1;
                    errors.push(format!("Failed to add connection: {}", e));
                    continue;
                }
            }
        }
    }

    if let (Some(source), Some(target)) = (source_type, target_type) {
        update_schema_node(
            graph,
            &connection_type,
            &source,
            &target,
            batch.get_schema_properties(),
        )?;
    }

    // Execute the batch and get the statistics
    let (stats, metrics) = batch.execute(graph, connection_type)?;

    // Create and return the operation report
    let mut report = ConnectionOperationReport::new(
        "selection_to_new_connections".to_string(),
        stats.connections_created,
        skipped,
        stats.properties_tracked,
        metrics.processing_time * 1000.0, // Convert to milliseconds
    );

    // Add errors if we found any
    if !errors.is_empty() {
        report = report.with_errors(errors);
    }

    Ok(report)
}

pub fn update_node_properties(
    graph: &mut DirGraph,
    nodes: &[(Option<NodeIndex>, Value)],
    property: &str,
) -> Result<NodeOperationReport, String> {
    if nodes.is_empty() {
        return Err("No nodes to update".to_string());
    }

    // Track start time for the report
    let start_time = std::time::Instant::now();

    // Create property string once
    let property_string = property.to_string();

    // Track errors
    let mut errors = Vec::new();

    // Step 1: Collect information about node types and check if schema update is needed
    let mut node_types = HashMap::new();
    let mut first_value_type = None;
    let mut skipped_count = 0;

    for (node_idx_opt, value) in nodes {
        if let Some(node_idx) = node_idx_opt {
            if let Some(node) = graph.get_node(*node_idx) {
                // Track node type and count for each node
                *node_types.entry(node.node_type.clone()).or_insert(0) += 1;

                // Capture type of first value for schema
                if first_value_type.is_none() {
                    first_value_type = Some(match value {
                        Value::Int64(_) => "Int64",
                        Value::Float64(_) => "Float64",
                        Value::String(_) => "String",
                        Value::UniqueId(_) => "UniqueId",
                        _ => "Unknown",
                    });
                }
            } else {
                skipped_count += 1;
                errors.push(format!("Node index {:?} not found in graph", node_idx));
            }
        } else {
            skipped_count += 1;
        }
    }

    // Step 2: Update node type metadata for each affected node type
    let type_string = first_value_type
        .map(|t| t.to_string())
        .unwrap_or_else(|| "Calculated".to_string());

    for (node_type, _count) in node_types.iter() {
        // Check for type mismatch with existing metadata
        if let Some(existing_meta) = graph.get_node_type_metadata(node_type) {
            if let Some(existing_type) = existing_meta.get(&property_string) {
                if existing_type != &type_string {
                    errors.push(format!(
                        "Type mismatch for property '{}': existing schema has '{}', but data has '{}'",
                        property_string, existing_type, type_string
                    ));
                }
            }
        }

        let mut new_prop_types = HashMap::new();
        new_prop_types.insert(property_string.clone(), type_string.clone());
        graph.upsert_node_type_metadata(node_type, new_prop_types);
    }

    // Step 3: Prepare batch updates for nodes
    let batch_size = nodes.len();
    let mut batch = BatchProcessor::new(batch_size);

    for (node_idx_opt, value) in nodes {
        if let Some(node_idx) = node_idx_opt {
            // Only add valid nodes to batch
            if graph.graph.node_weight(*node_idx).is_some() {
                let mut properties = HashMap::new();
                properties.insert(property_string.clone(), value.clone());

                // Create update action
                let action = NodeAction::Update {
                    node_idx: *node_idx,
                    title: None, // Don't update title
                    properties,
                    conflict_mode: ConflictHandling::Update,
                };

                if let Err(e) = batch.add_action(action, graph) {
                    errors.push(format!("Failed to update node property: {}", e));
                    skipped_count += 1;
                }
            } else {
                skipped_count += 1;
                errors.push(format!("Node index {:?} is out of bounds", node_idx));
            }
        } else {
            skipped_count += 1;
        }
    }

    // Step 4: Execute batch update
    let (stats, _metrics) = match batch.execute(graph) {
        Ok(result) => result,
        Err(e) => {
            errors.push(format!("Failed to execute batch update: {}", e));
            return Err(format!("Failed to execute batch update: {}", e));
        }
    };

    if stats.updates == 0 && errors.is_empty() {
        errors.push("No nodes were updated".to_string());
    }

    // Calculate elapsed time
    let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    // Create and return the operation report
    let mut report = NodeOperationReport::new(
        "update_node_properties".to_string(),
        0, // We don't create nodes in this function
        stats.updates,
        skipped_count,
        elapsed_ms,
    );

    // Add errors if we found any
    if !errors.is_empty() {
        report = report.with_errors(errors);
    }

    Ok(report)
}
