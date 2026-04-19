//! CSR + mmap disk storage backend (Wikidata-scale).
//!
//! `DiskGraph` (in `disk_graph.rs`) owns a CSR edge format + mmap'd
//! column stores + block pool for out-of-core property data.
//!
//! Split (Phase 9):
//! - [`csr`] — CsrEdge / EdgeEndpoints / DiskNodeSlot / MergeSortEntry `#[repr(C)]` types
//! - [`builder`] — CSR construction (merge-sort + partitioned) + histogram rebuild
//! - [`block_column`] / [`block_pool`] — overflow block allocator

pub mod block_column;
pub mod block_pool;
pub mod builder;
pub mod csr;
#[allow(dead_code)]
pub mod graph;
pub mod property_index;
