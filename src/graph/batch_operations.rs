// src/graph/batch_operations.rs
use crate::datatypes::Value;
use crate::graph::schema::{DirGraph, EdgeData, NodeData};
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// Constants for batch size optimization
const SMALL_BATCH_THRESHOLD: usize = 100;
const MEDIUM_BATCH_THRESHOLD: usize = 1000;
const LARGE_BATCH_CHUNK_SIZE: usize = 1000;

#[derive(Debug)]
enum BatchType {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Default)]
pub struct BatchMetrics {
    pub processing_time: f64,
    pub memory_used: usize,
    pub batch_count: usize,
}

// Node Processing
#[derive(Debug)]
pub enum NodeAction {
    Update {
        node_idx: NodeIndex,
        title: Option<Value>, // Changed to Option to indicate if title should be updated
        properties: HashMap<String, Value>,
        conflict_mode: ConflictHandling, // Added conflict mode
    },
    Create {
        node_type: String,
        id: Value,
        title: Value,
        properties: HashMap<String, Value>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ConflictHandling {
    Replace, // Replace all properties and title (current behavior)
    Skip,    // Don't update existing nodes
    #[default]
    Update, // Update properties and title if provided
    Preserve, // Update but prefer existing values
}

#[derive(Debug)]
struct NodeCreation {
    node_type: String,
    id: Value,
    title: Value,
    properties: HashMap<String, Value>,
}

#[derive(Debug)]
struct NodeUpdate {
    node_idx: NodeIndex,
    title: Option<Value>, // Changed to Option
    properties: HashMap<String, Value>,
    conflict_mode: ConflictHandling,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BatchStats {
    pub creates: usize,
    pub updates: usize,
}

impl BatchStats {
    fn combine(&mut self, other: &BatchStats) {
        self.creates += other.creates;
        self.updates += other.updates;
    }
}

#[derive(Debug)]
pub struct BatchProcessor {
    creates: Vec<NodeCreation>,
    updates: Vec<NodeUpdate>,
    capacity: usize,
    batch_type: BatchType,
    metrics: BatchMetrics,
    accumulated_stats: BatchStats, // Track stats across intermediate flushes
}

impl BatchProcessor {
    pub fn new(estimated_size: usize) -> Self {
        let (capacity, batch_type) = match estimated_size {
            n if n < SMALL_BATCH_THRESHOLD => (n, BatchType::Small),
            n if n < MEDIUM_BATCH_THRESHOLD => (n, BatchType::Medium),
            _ => (LARGE_BATCH_CHUNK_SIZE, BatchType::Large),
        };

        BatchProcessor {
            creates: Vec::with_capacity(capacity),
            updates: Vec::with_capacity(capacity),
            capacity,
            batch_type,
            metrics: BatchMetrics::default(),
            accumulated_stats: BatchStats::default(),
        }
    }

    pub fn add_action(&mut self, action: NodeAction, graph: &mut DirGraph) -> Result<(), String> {
        match action {
            NodeAction::Create {
                node_type,
                id,
                title,
                properties,
            } => {
                self.creates.push(NodeCreation {
                    node_type,
                    id,
                    title,
                    properties,
                });
            }
            NodeAction::Update {
                node_idx,
                title,
                properties,
                conflict_mode,
            } => {
                self.updates.push(NodeUpdate {
                    node_idx,
                    title,
                    properties,
                    conflict_mode, // Add this field
                });
            }
        }

        // For large batches, flush if we hit capacity
        if let BatchType::Large = self.batch_type {
            if self.creates.len() >= self.capacity {
                let stats = self.flush_chunk(graph)?;
                self.accumulated_stats.combine(&stats); // Accumulate stats from intermediate flushes
            }
        }

        Ok(())
    }

    fn flush_chunk(&mut self, graph: &mut DirGraph) -> Result<BatchStats, String> {
        let start = Instant::now();
        let mut stats = BatchStats::default();

        // Process creates in current chunk
        for creation in self.creates.drain(..) {
            let id_for_index = creation.id.clone();
            let node_type_for_index = creation.node_type.clone();

            let node_data = NodeData::new(
                creation.id,
                creation.title,
                creation.node_type.clone(),
                creation.properties,
            );
            let node_idx = graph.graph.add_node(node_data);
            // Add to type index
            graph
                .type_indices
                .entry(creation.node_type)
                .or_default()
                .push(node_idx);
            // Add to ID index for O(1) lookups
            graph
                .id_indices
                .entry(node_type_for_index)
                .or_default()
                .insert(id_for_index, node_idx);
            stats.creates += 1;
        }

        // Process updates in current chunk
        for update in self.updates.drain(..) {
            if let Some(node) = graph.get_node_mut(update.node_idx) {
                match node {
                    NodeData::Regular {
                        title, properties, ..
                    }
                    | NodeData::Schema {
                        title, properties, ..
                    } => {
                        match update.conflict_mode {
                            ConflictHandling::Skip => {
                                // Skip this node entirely
                                continue;
                            }
                            ConflictHandling::Replace => {
                                // Current behavior - complete replacement
                                if let Some(new_title) = update.title {
                                    *title = new_title;
                                }
                                *properties = update.properties;
                            }
                            ConflictHandling::Update => {
                                // Update only if provided
                                if let Some(new_title) = update.title {
                                    *title = new_title;
                                }
                                // Merge properties with preference to new values
                                for (k, v) in update.properties {
                                    properties.insert(k, v);
                                }
                            }
                            ConflictHandling::Preserve => {
                                // Update only if provided, but preserve existing values
                                if let Some(new_title) = update.title {
                                    if title == &Value::Null {
                                        *title = new_title;
                                    }
                                }
                                // Merge properties with preference to existing values
                                for (k, v) in update.properties {
                                    properties.entry(k).or_insert(v);
                                }
                            }
                        }
                        stats.updates += 1;
                    }
                }
            }
        }

        // Update metrics
        self.metrics.processing_time += start.elapsed().as_secs_f64();
        self.metrics.batch_count += 1;
        self.metrics.memory_used = self.creates.capacity() + self.updates.capacity();

        Ok(stats)
    }

    pub fn execute(mut self, graph: &mut DirGraph) -> Result<(BatchStats, BatchMetrics), String> {
        // Start with accumulated stats from intermediate flushes (for large batches)
        let mut total_stats = self.accumulated_stats;

        match self.batch_type {
            BatchType::Small | BatchType::Medium => {
                // Process in a single batch
                let stats = self.flush_chunk(graph)?;
                total_stats.combine(&stats);
            }
            BatchType::Large => {
                // Process any remaining items
                if !self.creates.is_empty() || !self.updates.is_empty() {
                    let stats = self.flush_chunk(graph)?;
                    total_stats.combine(&stats);
                }
            }
        }

        Ok((total_stats, self.metrics))
    }
}

// Connection Processing
#[derive(Debug)]
struct ConnectionCreation {
    source_idx: NodeIndex,
    target_idx: NodeIndex,
    properties: HashMap<String, Value>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ConnectionBatchStats {
    pub connections_created: usize,
    pub properties_tracked: usize,
}

impl ConnectionBatchStats {
    fn combine(&mut self, other: &ConnectionBatchStats) {
        self.connections_created += other.connections_created;
        self.properties_tracked = self.properties_tracked.max(other.properties_tracked);
    }
}

#[derive(Debug)]
pub struct ConnectionBatchProcessor {
    connections: Vec<ConnectionCreation>,
    schema_properties: HashSet<String>,
    capacity: usize,
    batch_type: BatchType,
    metrics: BatchMetrics,
    conflict_mode: ConflictHandling,
    accumulated_stats: ConnectionBatchStats, // Track stats across intermediate flushes
}

impl ConnectionBatchProcessor {
    pub fn new(estimated_size: usize) -> Self {
        let (capacity, batch_type) = match estimated_size {
            n if n < SMALL_BATCH_THRESHOLD => (n, BatchType::Small),
            n if n < MEDIUM_BATCH_THRESHOLD => (n, BatchType::Medium),
            _ => (LARGE_BATCH_CHUNK_SIZE, BatchType::Large),
        };

        ConnectionBatchProcessor {
            connections: Vec::with_capacity(capacity),
            schema_properties: HashSet::new(),
            capacity,
            batch_type,
            metrics: BatchMetrics::default(),
            conflict_mode: ConflictHandling::Update,
            accumulated_stats: ConnectionBatchStats::default(),
        }
    }

    // Add setter for conflict mode
    pub fn set_conflict_mode(&mut self, mode: ConflictHandling) {
        self.conflict_mode = mode;
    }

    pub fn add_connection(
        &mut self,
        source_idx: NodeIndex,
        target_idx: NodeIndex,
        properties: HashMap<String, Value>,
        graph: &mut DirGraph,
        connection_type: &str,
    ) -> Result<(), String> {
        // Check if the edge already exists
        let existing_edge = graph.graph.find_edge(source_idx, target_idx);

        // If edge exists and conflict mode is Skip, don't add it
        if existing_edge.is_some() && self.conflict_mode == ConflictHandling::Skip {
            return Ok(());
        }

        // Track property names for schema
        for key in properties.keys() {
            self.schema_properties.insert(key.clone());
        }

        self.connections.push(ConnectionCreation {
            source_idx,
            target_idx,
            properties,
        });

        // For large batches, flush if we hit capacity
        if let BatchType::Large = self.batch_type {
            if self.connections.len() >= self.capacity {
                let stats = self.flush_chunk(graph, connection_type)?;
                self.accumulated_stats.combine(&stats); // Accumulate stats from intermediate flushes
            }
        }

        Ok(())
    }

    fn flush_chunk(
        &mut self,
        graph: &mut DirGraph,
        connection_type: &str,
    ) -> Result<ConnectionBatchStats, String> {
        let start = Instant::now();
        let mut stats = ConnectionBatchStats::default();

        // Create or update edges in current chunk
        for conn in self.connections.drain(..) {
            // Check if the edge already exists
            if let Some(edge_idx) = graph.graph.find_edge(conn.source_idx, conn.target_idx) {
                match self.conflict_mode {
                    ConflictHandling::Skip => {
                        // Skip this edge (should already be filtered in add_connection)
                        continue;
                    }
                    ConflictHandling::Replace => {
                        // Remove the existing edge and create a new one
                        graph.graph.remove_edge(edge_idx);
                        let edge_data = EdgeData::new(connection_type.to_string(), conn.properties);
                        graph
                            .graph
                            .add_edge(conn.source_idx, conn.target_idx, edge_data);
                        stats.connections_created += 1;
                    }
                    ConflictHandling::Update => {
                        // Update existing edge properties
                        if let Some(EdgeData {
                            properties: edge_props,
                            ..
                        }) = graph.graph.edge_weight_mut(edge_idx)
                        {
                            // Merge properties, preferring new values
                            for (k, v) in conn.properties {
                                edge_props.insert(k, v);
                            }
                            stats.connections_created += 1;
                        }
                    }
                    ConflictHandling::Preserve => {
                        // Update but preserve existing values
                        if let Some(EdgeData {
                            properties: edge_props,
                            ..
                        }) = graph.graph.edge_weight_mut(edge_idx)
                        {
                            // Merge properties, preferring existing values
                            for (k, v) in conn.properties {
                                edge_props.entry(k).or_insert(v);
                            }
                            stats.connections_created += 1;
                        }
                    }
                }
            } else {
                // Create new edge
                let edge_data = EdgeData::new(connection_type.to_string(), conn.properties);
                graph
                    .graph
                    .add_edge(conn.source_idx, conn.target_idx, edge_data);
                stats.connections_created += 1;
            }
        }

        // Update metrics
        self.metrics.processing_time += start.elapsed().as_secs_f64();
        self.metrics.batch_count += 1;
        self.metrics.memory_used = self.connections.capacity();

        stats.properties_tracked = self.schema_properties.len();
        Ok(stats)
    }

    pub fn execute(
        mut self,
        graph: &mut DirGraph,
        connection_type: String,
    ) -> Result<(ConnectionBatchStats, BatchMetrics), String> {
        // Register connection type for O(1) lookups
        graph.register_connection_type(connection_type.clone());

        // Start with accumulated stats from intermediate flushes (for large batches)
        let mut total_stats = self.accumulated_stats;

        match self.batch_type {
            BatchType::Small | BatchType::Medium => {
                // Process in a single batch
                let stats = self.flush_chunk(graph, &connection_type)?;
                total_stats.combine(&stats);
            }
            BatchType::Large => {
                // Process any remaining items
                if !self.connections.is_empty() {
                    let stats = self.flush_chunk(graph, &connection_type)?;
                    total_stats.combine(&stats);
                }
            }
        }

        Ok((total_stats, self.metrics))
    }

    pub fn get_schema_properties(&self) -> &HashSet<String> {
        &self.schema_properties
    }
}
