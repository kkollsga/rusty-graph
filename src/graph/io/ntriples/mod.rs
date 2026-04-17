//! Streaming N-Triples loader with bz2/gz/plain support.
//!
//! Designed for Wikidata truthy dumps but works with any N-Triples file.
//!
//! Single-pass algorithm:
//!   1. Stream-decompress → parse lines → filter by predicate/language
//!   2. Accumulate properties per subject; flush node on subject change
//!   3. Buffer edges (source_id, target_id, predicate)
//!   4. After EOF: create all buffered edges, skip if target missing
//!
//! Split:
//! - [`parser`] — triple AST (Subject/Predicate/Object/EdgeBuffer), line
//!   parser, XSD literal → Value coercion, language filters.
//! - [`writer`] — edge-creation dispatch (compact, strings, qnum map).
//! - [`loader`] — `load_ntriples` entry point + columnar build pipeline +
//!   metadata structs + `flush_entity`.

use std::collections::{HashMap, HashSet};

pub mod loader;
pub mod parser;
pub mod writer;

pub use loader::{load_ntriples, ColumnTypeMeta};

/// Stats returned after loading.
pub struct NTriplesStats {
    pub triples_scanned: u64,
    pub entities_created: u64,
    pub edges_created: u64,
    pub edges_skipped: u64, // target entity not in graph
    pub seconds: f64,
}

/// Configuration for the loader.
pub struct NTriplesConfig {
    /// Only import these Wikidata predicates (P-codes). None = all.
    pub predicates: Option<HashSet<String>>,
    /// Only keep literals in these languages. None = all.
    pub languages: Option<HashSet<String>>,
    /// Map P31 target Q-code → human-readable node type name.
    pub node_types: HashMap<String, String>,
    /// Map P-code → human-readable predicate label.
    pub predicate_labels: HashMap<String, String>,
    /// Stop after this many entities. None = no limit.
    pub max_entities: Option<usize>,
    /// Print progress to stderr.
    pub verbose: bool,
    /// Automatically derive node types from P31 values, resolving Q-codes to
    /// labels when known. Entities without P31 default to "Entity".
    /// When true, node_types mappings still take priority.
    pub auto_type: bool,
}
