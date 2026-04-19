//! Data models for parsed code entities (ported from parsers/models.py).
//!
//! Field order is preserved from the Python dataclasses so that any
//! downstream JSON (e.g. the `fields` blob embedded on Class/Struct nodes,
//! consumed by the MCP code-review server) remains byte-identical.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

pub type MetadataMap = HashMap<String, serde_json::Value>;

/// An annotation extracted from a comment (TODO/FIXME/HACK/SAFETY/XXX/BUG/NOTE/WARNING).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub kind: String,
    pub text: String,
    pub line: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileInfo {
    /// Path relative to `src_root`.
    pub path: String,
    pub filename: String,
    pub loc: u32,
    /// Language-specific qualified path (e.g. `foo::bar::baz`, `foo.bar.baz`).
    pub module_path: String,
    /// "rust" | "python" | "typescript" | "javascript" | "go" | "java" | "cpp" | "c" | "csharp"
    pub language: String,
    pub submodule_declarations: Vec<String>,
    pub imports: Vec<String>,
    /// Exported names (`__all__`, `pub` items).
    pub exports: Vec<String>,
    /// TODO/FIXME/HACK/SAFETY comments. `None` if none were found.
    pub annotations: Option<Vec<Annotation>>,
    pub is_test: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionInfo {
    pub name: String,
    pub qualified_name: String,
    pub visibility: String,
    pub is_async: bool,
    pub is_method: bool,
    pub signature: String,
    pub file_path: String,
    pub line_number: u32,
    pub docstring: Option<String>,
    pub return_type: Option<String>,
    pub decorators: Vec<String>,
    /// `(callee_name, line_number)` pairs.
    pub calls: Vec<(String, u32)>,
    /// Generic/template parameters, e.g. `"T, U: Display"`.
    pub type_parameters: Option<String>,
    pub end_line: Option<u32>,
    /// Flexible language-specific flags (is_pymethod, is_test, ffi_kind, py_name, ...).
    pub metadata: MetadataMap,
}

/// Represents structs (Rust/C++), classes (Python/TS/Java/C#).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassInfo {
    pub name: String,
    pub qualified_name: String,
    /// "struct" | "class" — determines graph node type.
    pub kind: String,
    pub visibility: String,
    pub file_path: String,
    pub line_number: u32,
    pub docstring: Option<String>,
    pub bases: Vec<String>,
    pub type_parameters: Option<String>,
    pub end_line: Option<u32>,
    pub metadata: MetadataMap,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnumInfo {
    pub name: String,
    pub qualified_name: String,
    pub visibility: String,
    pub file_path: String,
    pub line_number: u32,
    pub docstring: Option<String>,
    pub variants: Vec<String>,
    pub end_line: Option<u32>,
    /// Structured variant info per language (C++ enum values, Rust tuple/struct variants).
    pub variant_details: Option<Vec<serde_json::Value>>,
}

/// Represents traits (Rust), protocols (Python), interfaces (TS/Java/C#).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterfaceInfo {
    pub name: String,
    pub qualified_name: String,
    /// "trait" | "protocol" | "interface"
    pub kind: String,
    pub visibility: String,
    pub file_path: String,
    pub line_number: u32,
    pub docstring: Option<String>,
    pub type_parameters: Option<String>,
    pub end_line: Option<u32>,
}

/// Class/struct field or property.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttributeInfo {
    pub name: String,
    /// Fully-qualified name, e.g. `"chromadb.Collection.name"`.
    pub qualified_name: String,
    /// Owner's qualified name, e.g. `"chromadb.Collection"`.
    pub owner_qualified_name: String,
    pub type_annotation: Option<String>,
    pub visibility: String,
    pub file_path: String,
    pub line_number: u32,
    pub default_value: Option<String>,
}

/// Top-level constant, type alias, or static variable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstantInfo {
    pub name: String,
    pub qualified_name: String,
    /// "constant" | "type_alias" | "static"
    pub kind: String,
    pub type_annotation: Option<String>,
    /// First ~100 chars of the value expression.
    pub value_preview: Option<String>,
    pub visibility: String,
    pub file_path: String,
    pub line_number: u32,
}

/// An impl block (Rust), inheritance (Python/Java/C#), or implements (TS).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeRelationship {
    pub source_type: String,
    /// `None` for inherent impls.
    pub target_type: Option<String>,
    /// "implements" | "extends" | "inherent"
    pub relationship: String,
    pub methods: Vec<FunctionInfo>,
}

/// Unified result of parsing one or more files.
#[derive(Debug, Clone, Default)]
pub struct ParseResult {
    pub files: Vec<FileInfo>,
    pub functions: Vec<FunctionInfo>,
    pub classes: Vec<ClassInfo>,
    pub enums: Vec<EnumInfo>,
    pub interfaces: Vec<InterfaceInfo>,
    pub type_relationships: Vec<TypeRelationship>,
    pub attributes: Vec<AttributeInfo>,
    pub constants: Vec<ConstantInfo>,
}

impl ParseResult {
    pub fn new() -> Self {
        Self::default()
    }

    /// Move the contents of `other` into `self`.
    pub fn merge(&mut self, mut other: ParseResult) {
        self.files.append(&mut other.files);
        self.functions.append(&mut other.functions);
        self.classes.append(&mut other.classes);
        self.enums.append(&mut other.enums);
        self.interfaces.append(&mut other.interfaces);
        self.type_relationships
            .append(&mut other.type_relationships);
        self.attributes.append(&mut other.attributes);
        self.constants.append(&mut other.constants);
    }
}

// ── Manifest / project-level models ────────────────────────────────

/// A directory containing source code to parse.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceRoot {
    /// Absolute path to the directory.
    pub path: PathBuf,
    /// `None` = auto-detect language.
    pub language: Option<String>,
    pub is_test: bool,
    /// e.g. `"python-package"`, `"rust-crate"`.
    pub label: Option<String>,
}

/// A project dependency declared in a manifest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DependencyInfo {
    pub name: String,
    pub version_spec: Option<String>,
    pub is_dev: bool,
    pub is_optional: bool,
    /// Optional dep group name.
    pub group: Option<String>,
}

/// Project metadata extracted from a manifest file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub name: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub languages: Vec<String>,
    pub authors: Vec<String>,
    pub license: Option<String>,
    pub repository_url: Option<String>,
    pub manifest_path: String,
    pub source_roots: Vec<SourceRoot>,
    pub test_roots: Vec<SourceRoot>,
    pub dependencies: Vec<DependencyInfo>,
    /// e.g. `"maturin"`, `"setuptools"`, `"cargo"`.
    pub build_system: Option<String>,
    pub metadata: MetadataMap,
}

// ── Field-embedding JSON (for ClassInfo.fields on graph nodes) ────────
//
// `fields` is serialized as a JSON string attached to Class/Struct nodes.
// Preserve exact key order: name, type, visibility, default.
// (This is verified in tests/test_code_tree_parity.py via a direct JSON
// equality assertion on a known Class node.)

/// One entry in the JSON `fields` blob on a Class/Struct node.
#[derive(Debug, Clone, Serialize)]
pub struct FieldEntry {
    pub name: String,
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    pub visibility: String,
    pub default: Option<String>,
}
