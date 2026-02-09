// src/graph/cypher/mod.rs
// Cypher query language implementation for rusty_graph
//
// Architecture:
//   Query String -> Tokenizer -> Parser -> AST -> Planner -> Executor -> Result
//
// The MATCH clause delegates pattern parsing to pattern_matching::parse_pattern()
// WHERE/RETURN/ORDER BY etc. are handled by the Cypher-level parser and executor.

pub mod ast;
pub mod executor;
pub mod parser;
pub mod planner;
pub mod py_convert;
pub mod result;
pub mod tokenizer;

// Re-exports for convenience
pub use executor::{execute_mutable, is_mutation_query, CypherExecutor};
pub use parser::parse_cypher;
pub use planner::optimize;
