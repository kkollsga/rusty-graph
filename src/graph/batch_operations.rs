// src/graph/batch_operations.rs
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use petgraph::graph::NodeIndex;
use crate::datatypes::Value;
use crate::graph::schema::{DirGraph, NodeData, EdgeData};

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
        title: Option<Value>,  // Changed to Option to indicate if title should be updated
        properties: HashMap<String, Value>,
        conflict_mode: ConflictHandling,  // Added conflict mode
    },
    Create {
        node_type: String,
        id: Value,
        title: Value,
        properties: HashMap<String, Value>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConflictHandling {
    Replace,     // Replace all properties and title (current behavior)
    Skip,        // Don't update existing nodes
    Update,      // Update properties and title if provided
    Preserve,    // Update but prefer existing values
}

impl Default for ConflictHandling {
    fn default() -> Self {
        ConflictHandling::Update  // New default behavior
    }
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
    title: Option<Value>,  // Changed to Option
    properties: HashMap<String, Value>,
    conflict_mode: ConflictHandling,
}


#[derive(Debug, Default)]
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
        }
    }

    pub fn add_action(&mut self, action: NodeAction, graph: &mut DirGraph) -> Result<(), String> {
        match action {
            NodeAction::Create { node_type, id, title, properties } => {
                self.creates.push(NodeCreation {
                    node_type,
                    id,
                    title,
                    properties,
                });
            },
            NodeAction::Update { node_idx, title, properties, conflict_mode } => {
                self.updates.push(NodeUpdate {
                    node_idx,
                    title,
                    properties,
                    conflict_mode,  // Add this field
                });
            }
        }

        // For large batches, flush if we hit capacity
        if let BatchType::Large = self.batch_type {
            if self.creates.len() >= self.capacity {
                self.flush_chunk(graph)?;
            }
        }

        Ok(())
    }

    fn flush_chunk(&mut self, graph: &mut DirGraph) -> Result<BatchStats, String> {
        let start = Instant::now();
        let mut stats = BatchStats::default();

        // Process creates in current chunk
        for creation in self.creates.drain(..) {
            let node_data = NodeData::new(
                creation.id,
                creation.title,
                creation.node_type.clone(),
                creation.properties,
            );
            let node_idx = graph.graph.add_node(node_data);
            // Add to type index
            graph.type_indices.entry(creation.node_type).or_default().push(node_idx);
            stats.creates += 1;
        }

        // Process updates in current chunk
        for update in self.updates.drain(..) {
            if let Some(node) = graph.get_node_mut(update.node_idx) {
                match node {
                    NodeData::Regular { title, properties, .. } |
                    NodeData::Schema { title, properties, .. } => {
                        match update.conflict_mode {
                            ConflictHandling::Skip => {
                                // Skip this node entirely
                                continue;
                            },
                            ConflictHandling::Replace => {
                                // Current behavior - complete replacement
                                if let Some(new_title) = update.title {
                                    *title = new_title;
                                }
                                *properties = update.properties;
                            },
                            ConflictHandling::Update => {
                                // Update only if provided
                                if let Some(new_title) = update.title {
                                    *title = new_title;
                                }
                                // Merge properties with preference to new values
                                for (k, v) in update.properties {
                                    properties.insert(k, v);
                                }
                            },
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
        let mut total_stats = BatchStats::default();

        match self.batch_type {
            BatchType::Small | BatchType::Medium => {
                // Process in a single batch
                let stats = self.flush_chunk(graph)?;
                total_stats.combine(&stats);
            },
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

#[derive(Debug, Default)]
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
        }
    }

    pub fn add_connection(
        &mut self,
        source_idx: NodeIndex,
        target_idx: NodeIndex,
        properties: HashMap<String, Value>,
        graph: &mut DirGraph,
        connection_type: &str,
    ) -> Result<(), String> {
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
                self.flush_chunk(graph, connection_type)?;
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

        // Create edges in current chunk
        for conn in self.connections.drain(..) {
            let edge_data = EdgeData::new(connection_type.to_string(), conn.properties);
            graph.graph.add_edge(conn.source_idx, conn.target_idx, edge_data);
            stats.connections_created += 1;
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
        let mut total_stats = ConnectionBatchStats::default();

        match self.batch_type {
            BatchType::Small | BatchType::Medium => {
                // Process in a single batch
                let stats = self.flush_chunk(graph, &connection_type)?;
                total_stats.combine(&stats);
            },
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