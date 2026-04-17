//! CSR + mmap disk storage backend (Wikidata-scale).
//!
//! `DiskGraph` (in `disk_graph.rs`) owns a CSR edge format + mmap'd
//! column stores + block pool for out-of-core property data. The
//! per-backend `impl GraphRead` / `impl GraphWrite` for `DiskGraph`
//! lives in `storage::impls`.

pub mod block_column;
pub mod block_pool;
#[allow(dead_code)]
pub mod disk_graph;
