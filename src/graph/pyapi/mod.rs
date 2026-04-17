//! `#[pymethods]` blocks — the PyO3 boundary.
//!
//! These files each declare additional `#[pymethods] impl KnowledgeGraph`
//! blocks on top of the core block in `graph::mod.rs`. PyO3 merges them
//! at class-registration time. Keeping the edge code separated makes
//! the Rust internal / Python ergonomic contract visible.

pub mod kg_core;
pub mod kg_fluent;
pub mod kg_introspection;
pub mod kg_mutation;
pub mod pymethods_algorithms;
pub mod pymethods_export;
pub mod pymethods_indexes;
pub mod pymethods_spatial;
pub mod pymethods_timeseries;
pub mod pymethods_vector;
pub mod result_view;
pub mod transaction;
