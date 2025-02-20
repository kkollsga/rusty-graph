// src/graph/maintain_graph.rs
use std::collections::{HashMap, HashSet};
use crate::graph::schema::{DirGraph, NodeData, CurrentSelection};
use crate::graph::lookups::{TypeLookup, CombinedTypeLookup};
use crate::graph::batch_operations::{BatchProcessor, ConnectionBatchProcessor, NodeAction};
use crate::datatypes::{Value, DataFrame};

fn check_data_validity(df_data: &DataFrame, unique_id_field: &str) -> Result<(), String> {
    // Remove strict UniqueId type verification to allow nulls
    if !df_data.verify_column(unique_id_field) {
        return Err(format!("Column '{}' not found", unique_id_field));
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
    _conflict_handling: Option<String>,
) -> Result<(), String> {
    let title_field = node_title_field.unwrap_or_else(|| unique_id_field.clone());
    check_data_validity(&df_data, &unique_id_field)?;

    // Rest of the schema processing remains the same...
    let schema_lookup = TypeLookup::new(&graph.graph, "SchemaNode".to_string())?;
    let schema_title = Value::String(node_type.clone());
    let schema_node_idx = schema_lookup.check_title(&schema_title);

    let df_column_types = get_column_types(&df_data);
    let df_schema_properties: HashMap<String, Value> = df_column_types
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect();

    // Schema node handling remains the same...
    match schema_node_idx {
        Some(idx) => {
            if let Some(NodeData::Schema { properties, .. }) = graph.get_node_mut(idx) {
                for (col_name, col_type) in &df_schema_properties {
                    if !properties.contains_key(col_name) {
                        properties.insert(col_name.clone(), col_type.clone());
                    }
                }
            }
        }
        None => {
            let schema_node_data = NodeData::Schema {
                id: Value::String(node_type.clone()),
                title: Value::String(node_type.clone()),
                node_type: "SchemaNode".to_string(),
                properties: df_schema_properties,
            };
            graph.graph.add_node(schema_node_data);
            println!("New SchemaNode created for node type '{}'.", node_type);
        }
    }

    let type_lookup = TypeLookup::new(&graph.graph, node_type.clone())?;
    let id_idx = df_data.get_column_index(&unique_id_field)
        .ok_or_else(|| format!("Column '{}' not found", unique_id_field))?;
    let title_idx = df_data.get_column_index(&title_field)
        .ok_or_else(|| format!("Column '{}' not found", title_field))?;

    let column_names = df_data.get_column_names();
    let mut batch = BatchProcessor::new(df_data.row_count());
    let mut skipped_count = 0;

    // Process rows
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

        let title = df_data.get_value_by_index(row_idx, title_idx)
            .unwrap_or(Value::Null);

        let mut properties = HashMap::with_capacity(column_names.len());
        for col_name in &column_names {
            if col_name != &unique_id_field && col_name != &title_field {
                if let Some(value) = df_data.get_value(row_idx, col_name) {
                    properties.insert(col_name.clone(), value);
                }
            }
        }

        let action = match type_lookup.check_uid(&id) {
            Some(node_idx) => NodeAction::Update { node_idx, title, properties },
            None => NodeAction::Create { node_type: node_type.clone(), id, title, properties },
        };
        batch.add_action(action, graph)?;
    }

    // Execute remaining actions
    let (stats, metrics) = batch.execute(graph)?;
    println!("Processed {} creates and {} updates in {} batches ({}s), skipped {} rows with null IDs",
             stats.creates, stats.updates, metrics.batch_count, metrics.processing_time, skipped_count);

    Ok(())
}

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
    columns: Option<Vec<String>>,
    _conflict_handling: Option<String>,
) -> Result<(), String> {
    // Remove strict UniqueId type verification
    if !df_data.verify_column(&source_id_field) {
        return Err(format!("Source ID column '{}' not found", source_id_field));
    }
    if !df_data.verify_column(&target_id_field) {
        return Err(format!("Target ID column '{}' not found", target_id_field));
    }

    let source_id_idx = df_data.get_column_index(&source_id_field)
        .ok_or_else(|| format!("Source ID column '{}' not found", source_id_field))?;
    let target_id_idx = df_data.get_column_index(&target_id_field)
        .ok_or_else(|| format!("Target ID column '{}' not found", target_id_field))?;

    // Rest of the connection setup remains the same...
    let source_title_idx = source_title_field
        .and_then(|field| df_data.get_column_index(&field));
    let target_title_idx = target_title_field
        .and_then(|field| df_data.get_column_index(&field));

    let lookup = CombinedTypeLookup::new(&graph.graph, source_type.clone(), target_type.clone())?;
    let mut batch = ConnectionBatchProcessor::new(df_data.row_count());
    let mut skipped_count = 0;

    for row_idx in 0..df_data.row_count() {
        let source_id = match df_data.get_value_by_index(row_idx, source_id_idx) {
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

        let target_id = match df_data.get_value_by_index(row_idx, target_id_idx) {
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

        if let (Some(source_idx), Some(target_idx)) = (
            lookup.check_source(&source_id),
            lookup.check_target(&target_id)
        ) {
            update_node_titles(graph, source_idx, target_idx, row_idx, 
                             source_title_idx, target_title_idx, &df_data)?;

            let mut properties = HashMap::with_capacity(columns.as_ref().map_or(0, |c| c.len()));
            if let Some(cols) = &columns {
                for col_name in cols {
                    if let Some(value) = df_data.get_value(row_idx, col_name) {
                        properties.insert(col_name.clone(), value);
                    }
                }
            }
            batch.add_connection(source_idx, target_idx, properties, graph, &connection_type)?;
        } else {
            skipped_count += 1;
            continue;
        }
    }

    update_schema_node(
        graph,
        &connection_type,
        lookup.get_source_type(),
        lookup.get_target_type(),
        batch.get_schema_properties(),
    )?;

    let (stats, metrics) = batch.execute(graph, connection_type)?;
    println!("Created {} connections with {} property types in {} batches ({}s), skipped {} rows with null IDs", 
             stats.connections_created, stats.properties_tracked, 
             metrics.batch_count, metrics.processing_time, skipped_count);

    Ok(())
}

fn update_node_titles(
    graph: &mut DirGraph,
    source_idx: petgraph::graph::NodeIndex,
    target_idx: petgraph::graph::NodeIndex,
    row_idx: usize,
    source_title_idx: Option<usize>,
    target_title_idx: Option<usize>,
    df_data: &DataFrame,
) -> Result<(), String> {
    if let Some(title_idx) = source_title_idx {
        if let Some(title) = df_data.get_value_by_index(row_idx, title_idx) {
            if let Some(node) = graph.get_node_mut(source_idx) {
                match node {
                    NodeData::Regular { title: t, .. } | NodeData::Schema { title: t, .. } => {
                        *t = title;
                    }
                }
            }
        }
    }
    if let Some(title_idx) = target_title_idx {
        if let Some(title) = df_data.get_value_by_index(row_idx, title_idx) {
            if let Some(node) = graph.get_node_mut(target_idx) {
                match node {
                    NodeData::Regular { title: t, .. } | NodeData::Schema { title: t, .. } => {
                        *t = title;
                    }
                }
            }
        }
    }
    Ok(())
}

// In maintain_graph.rs
fn update_schema_node(
    graph: &mut DirGraph,
    connection_type: &str,
    source_type: &str,
    target_type: &str,
    properties: &HashSet<String>,
) -> Result<(), String> {
    // Validate types first, before any mutable borrows
    if !graph.has_node_type(source_type) {
        return Err(format!("Source type '{}' does not exist in graph", source_type));
    }
    if !graph.has_node_type(target_type) {
        return Err(format!("Target type '{}' does not exist in graph", target_type));
    }

    let schema_title = Value::String(connection_type.to_string());
    let schema_lookup = TypeLookup::new(&graph.graph, "SchemaNode".to_string())?;

    match schema_lookup.check_title(&schema_title) {
        Some(idx) => {
            // Update existing schema node
            if let Some(NodeData::Schema { properties: schema_props, .. }) = graph.get_node_mut(idx) {
                // Add new properties
                let mut properties_added = false;
                for prop in properties {
                    if !schema_props.contains_key(prop) {
                        schema_props.insert(prop.clone(), Value::String("Unknown".to_string()));
                        properties_added = true;
                    }
                }

                // Update type information
                let source_changed = schema_props.get("source_type") != Some(&Value::String(source_type.to_string()));
                let target_changed = schema_props.get("target_type") != Some(&Value::String(target_type.to_string()));
                
                schema_props.insert("source_type".to_string(), Value::String(source_type.to_string()));
                schema_props.insert("target_type".to_string(), Value::String(target_type.to_string()));

                // Log significant changes
                if properties_added {
                    println!("Updated schema for '{}': added new properties", connection_type);
                }
                if source_changed || target_changed {
                    println!("Updated schema for '{}': modified type relationship {}->{}",
                        connection_type, source_type, target_type);
                }
            } else {
                return Err(format!("Invalid schema node found for connection type '{}'", connection_type));
            }
        }
        None => {
            // Create new schema node (existing code)
            let mut schema_properties: HashMap<String, Value> = properties
                .iter()
                .map(|prop| (prop.clone(), Value::String("Unknown".to_string())))
                .collect();

            schema_properties.insert("source_type".to_string(), Value::String(source_type.to_string()));
            schema_properties.insert("target_type".to_string(), Value::String(target_type.to_string()));

            let schema_node_data = NodeData::Schema {
                id: Value::String(connection_type.to_string()),
                title: schema_title,
                node_type: "SchemaNode".to_string(),
                properties: schema_properties,
            };
            graph.graph.add_node(schema_node_data);

            println!("New SchemaConnection created for '{}' ({}->{})",
                connection_type, source_type, target_type);
        }
    }

    Ok(())
}

pub fn selection_to_new_connections(
    graph: &mut DirGraph,
    selection: &CurrentSelection,
    connection_type: String,
) -> Result<(usize, usize), String> {  // Changed return type to include skipped count
    let current_level = selection.get_level_count().saturating_sub(1);
    let level = match selection.get_level(current_level) {
        Some(level) if !level.is_empty() => level,
        _ => return Ok((0, 0)), // Return early if no selections
    };

    let mut batch = ConnectionBatchProcessor::new(level.get_all_nodes().len());
    let mut skipped = 0;
    
    let mut source_type = None;
    let mut target_type = None;

    // Process each parent-child relationship
    for (parent_opt, children) in level.iter_groups() {
        if let Some(parent) = parent_opt {
            if source_type.is_none() {
                if let Some(NodeData::Regular { node_type, .. }) = graph.get_node(*parent) {
                    source_type = Some(node_type.clone());
                }
            }

            for &child in children {
                if target_type.is_none() {
                    if let Some(NodeData::Regular { node_type, .. }) = graph.get_node(child) {
                        target_type = Some(node_type.clone());
                    }
                }

                if let Err(e) = batch.add_connection(
                    *parent,
                    child,
                    HashMap::new(),
                    graph,
                    &connection_type,
                ) {
                    skipped += 1;
                    println!("Warning: Failed to create connection: {}", e);
                    continue;
                }
            }
        }
    }

    // Update schema with the new connection type if we have both types
    if let (Some(source), Some(target)) = (source_type, target_type) {
        update_schema_node(
            graph,
            &connection_type,
            &source,
            &target,
            batch.get_schema_properties(),
        )?;
    }

    // Execute the batch to actually create the connections!
    let (stats, metrics) = batch.execute(graph, connection_type)?;
    println!("Created {} connections in {} batches ({}s), skipped {} invalid connections",
             stats.connections_created, metrics.batch_count, metrics.processing_time, skipped);
    Ok((stats.connections_created, skipped))
}