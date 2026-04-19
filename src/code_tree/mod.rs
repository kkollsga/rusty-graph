#![allow(
    dead_code,
    clippy::needless_lifetimes,
    clippy::collapsible_match,
    clippy::collapsible_if,
    clippy::manual_pattern_char_comparison,
    clippy::manual_contains,
    clippy::needless_return,
    clippy::if_same_then_else,
    clippy::manual_find,
    clippy::needless_borrow,
    clippy::explicit_auto_deref,
    clippy::useless_conversion
)]
//! Code-tree: parse polyglot codebases into KGLite knowledge graphs.
//!
//! Rewrite of the former `kglite.code_tree` Python module. Tree-sitter grammars
//! are compiled into the native extension — no optional dependencies.
//!
//! Entry points (exposed to Python via `pyapi::register`):
//! - `build(src_dir, ...)` — parse a directory or manifest-rooted project
//! - `read_manifest(project_root)` — extract project metadata
//! - `repo_tree(repo, ...)` — shallow-clone a GitHub repo and build

pub mod builder;
pub mod manifest;
pub mod models;
pub mod parsers;
pub mod pyapi;
pub mod repo;
