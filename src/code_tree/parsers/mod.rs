//! Language parsers: one file per language grammar.
//!
//! `LanguageParser` is the dispatch trait — `parse_file` handles a single
//! source file, `parse_directory` walks the tree in parallel via rayon.

pub mod cpp;
pub mod csharp;
pub mod go;
pub mod java;
pub mod python;
pub mod rust_lang;
pub mod shared;
pub mod typescript;

use crate::code_tree::models::ParseResult;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// File extension → language identifier (mirrors registry.py::EXTENSION_MAP).
pub const EXTENSION_MAP: &[(&str, &str)] = &[
    ("rs", "rust"),
    ("py", "python"),
    ("pyi", "python"),
    ("ts", "typescript"),
    ("tsx", "typescript"),
    ("js", "javascript"),
    ("jsx", "javascript"),
    ("mjs", "javascript"),
    ("go", "go"),
    ("java", "java"),
    ("cs", "csharp"),
    ("c", "c"),
    ("h", "c"),
    ("cpp", "cpp"),
    ("cc", "cpp"),
    ("cxx", "cpp"),
    ("hpp", "cpp"),
    ("hh", "cpp"),
    ("hxx", "cpp"),
];

/// Look up a language identifier for a file path by extension.
pub fn language_for_path(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?;
    EXTENSION_MAP
        .iter()
        .find(|(e, _)| *e == ext)
        .map(|(_, lang)| *lang)
}

/// Abstract language parser. One implementor per grammar.
pub trait LanguageParser: Sync {
    /// Canonical language identifier (e.g. `"python"`, `"rust"`).
    fn language_name(&self) -> &'static str;

    /// File extensions (without leading dot) this parser handles.
    fn file_extensions(&self) -> &'static [&'static str];

    /// Names to exclude from call-edge resolution (common stdlib etc.).
    /// Overridden by parsers that define language-specific noise sets.
    fn noise_names(&self) -> &'static [&'static str] {
        &[]
    }

    /// Parse a single source file at `filepath` (absolute) relative to `src_root`.
    fn parse_file(&self, filepath: &Path, src_root: &Path) -> ParseResult;

    /// Parse every matching file under `src_root`. Default implementation walks
    /// the directory via `walkdir`, collects paths, parses them in parallel with
    /// rayon, and reduces the per-file results into one aggregate.
    fn parse_directory(&self, src_root: &Path, verbose: bool) -> ParseResult {
        let files: Vec<PathBuf> = WalkDir::new(src_root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter_map(|entry| {
                let path = entry.path();
                let ext = path.extension()?.to_str()?;
                if self.file_extensions().contains(&ext) {
                    Some(path.to_path_buf())
                } else {
                    None
                }
            })
            .collect();
        if verbose {
            eprintln!("  Found {} {} files", files.len(), self.language_name());
        }
        files
            .par_iter()
            .map(|fp| self.parse_file(fp, src_root))
            .reduce(ParseResult::new, |mut acc, r| {
                acc.merge(r);
                acc
            })
    }
}

/// Detect every language present under `src_root`. Sorted for deterministic output.
pub fn detect_languages(src_root: &Path) -> Vec<&'static str> {
    let mut langs: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    for entry in WalkDir::new(src_root).into_iter().filter_map(Result::ok) {
        if entry.file_type().is_file() {
            if let Some(lang) = language_for_path(entry.path()) {
                langs.insert(lang);
            }
        }
    }
    langs.into_iter().collect()
}

/// Get a parser instance for the given language identifier.
///
/// Returns `None` if the language is unknown. Individual parsers are
/// slotted in as modules are ported; unimplemented languages return `None`.
pub fn get_parser(language: &str) -> Option<Box<dyn LanguageParser + Send + Sync>> {
    match language {
        "python" => Some(Box::new(python::PythonParser::new())),
        "rust" => Some(Box::new(rust_lang::RustParser::new())),
        "typescript" => Some(Box::new(typescript::JstsParser::typescript())),
        "javascript" => Some(Box::new(typescript::JstsParser::javascript())),
        "go" => Some(Box::new(go::GoParser::new())),
        "java" => Some(Box::new(java::JavaParser::new())),
        "csharp" => Some(Box::new(csharp::CSharpParser::new())),
        "c" => Some(Box::new(cpp::CppParser::c())),
        "cpp" => Some(Box::new(cpp::CppParser::cpp())),
        _ => None,
    }
}

/// Auto-detect languages under `src_root` and return parser instances for each.
pub fn get_parsers_for_directory(src_root: &Path) -> Vec<Box<dyn LanguageParser + Send + Sync>> {
    detect_languages(src_root)
        .into_iter()
        .filter_map(get_parser)
        .collect()
}
