//! Write / maintenance / validation operations.
//!
//! Batch add/update, vacuum/maintenance, schema validation, set ops,
//! and subgraph extraction. These are the mutation side of the graph —
//! everything that changes state, plus the maintenance helpers that
//! keep storage caches coherent.

pub mod batch_operations;
pub mod maintain_graph;
pub mod schema_validation;
pub mod set_operations;
pub mod subgraph;
