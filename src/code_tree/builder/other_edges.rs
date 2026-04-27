//! USES_TYPE, CONTAINS, IMPORTS, FFI_EXPOSES edges.

use crate::code_tree::models::{
    ClassInfo, ConstantInfo, EnumInfo, FileInfo, FunctionInfo, InterfaceInfo,
};
use aho_corasick::{AhoCorasick, MatchKind};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};

fn get_separator(language: &str) -> &'static str {
    match language {
        "rust" | "cpp" => "::",
        "python" | "java" | "csharp" => ".",
        _ => "/",
    }
}

pub struct ContainsEdge {
    pub parent: String,
    pub child: String,
}

pub struct ImportEdge {
    pub file_path: String,
    pub module: String,
}

pub struct UsesTypeEdge {
    pub function: String,
    pub type_name: String,
    /// Target node type: "Struct" | "Class" | "Enum" | "Trait" | "Protocol" | "Interface"
    pub target_node_type: &'static str,
}

pub struct FfiExposesEdge {
    pub module_fn: String,
    pub target_qname: String,
    pub target_type: &'static str,
    pub py_name: String,
}

/// Module CONTAINS Module edges from file submodule declarations.
pub fn build_contains_edges(files: &[FileInfo]) -> Vec<ContainsEdge> {
    let mut out = Vec::new();
    for f in files {
        let sep = get_separator(&f.language);
        for sub in &f.submodule_declarations {
            out.push(ContainsEdge {
                parent: f.module_path.clone(),
                child: format!("{}{}{}", f.module_path, sep, sub),
            });
        }
    }
    out
}

/// File IMPORTS Module edges — resolve each import string against known modules.
pub fn build_import_edges(files: &[FileInfo], known_modules: &HashSet<String>) -> Vec<ImportEdge> {
    let mut out = Vec::new();
    for f in files {
        let sep = get_separator(&f.language);
        for use_path in &f.imports {
            let parts: Vec<&str> = use_path.split(sep).collect();
            for end in (1..=parts.len()).rev() {
                let candidate = parts[..end].join(sep);
                if known_modules.contains(&candidate) {
                    out.push(ImportEdge {
                        file_path: f.path.clone(),
                        module: candidate,
                    });
                    break;
                }
            }
        }
    }
    out
}

/// USES_TYPE edges: scan function signature/return_type for known type names.
///
/// Returns a map from target node type → list of edges, because add_connections
/// must be called separately for each distinct target type.
pub fn build_uses_type_edges(
    functions: &[FunctionInfo],
    classes: &[ClassInfo],
    enums: &[EnumInfo],
    interfaces: &[InterfaceInfo],
) -> BTreeMap<&'static str, Vec<UsesTypeEdge>> {
    // Collect known type names → (qualified_name, node_type).
    let mut type_lookup: HashMap<String, (String, &'static str)> = HashMap::new();
    for c in classes {
        if c.name.chars().count() > 1 {
            let target = if c.kind == "struct" {
                "Struct"
            } else {
                "Class"
            };
            type_lookup.insert(c.name.clone(), (c.qualified_name.clone(), target));
        }
    }
    for e in enums {
        if e.name.chars().count() > 1 {
            type_lookup.insert(e.name.clone(), (e.qualified_name.clone(), "Enum"));
        }
    }
    for i in interfaces {
        if i.name.chars().count() > 1 {
            let target = match i.kind.as_str() {
                "trait" => "Trait",
                "protocol" => "Protocol",
                _ => "Interface",
            };
            type_lookup.insert(i.name.clone(), (i.qualified_name.clone(), target));
        }
    }

    if type_lookup.is_empty() {
        return BTreeMap::new();
    }

    // Flatten type names into a stable-ordered Vec so pattern IDs from
    // Aho-Corasick map back to the right (qname, node_type) tuple.
    // Longest-match-first so "MyCollection" wins over "Collection".
    let mut names: Vec<String> = type_lookup.keys().cloned().collect();
    names.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    let pattern_meta: Vec<(String, &'static str)> = names
        .iter()
        .map(|n| {
            let (q, t) = type_lookup.get(n).unwrap();
            (q.clone(), *t)
        })
        .collect();

    let ac = match AhoCorasick::builder()
        .match_kind(MatchKind::LeftmostLongest)
        .build(&names)
    {
        Ok(ac) => ac,
        Err(_) => return BTreeMap::new(),
    };

    // Per-function scan in parallel: scan signature+return_type with AC,
    // collect unique pattern matches. Matches are deduped per-function
    // (same type in both signature and return_type counts once).
    let per_fn: Vec<Vec<(u32, &'static str, String)>> = functions
        .par_iter()
        .map(|fn_info| {
            let mut out: Vec<(u32, &'static str, String)> = Vec::new();
            if fn_info.signature.is_empty() && fn_info.return_type.is_none() {
                return out;
            }
            let mut parts: Vec<&str> = Vec::with_capacity(2);
            if !fn_info.signature.is_empty() {
                parts.push(&fn_info.signature);
            }
            if let Some(rt) = &fn_info.return_type {
                parts.push(rt);
            }
            let text = if parts.len() == 1 {
                parts[0].to_string()
            } else {
                parts.join(" ")
            };
            let bytes = text.as_bytes();
            let mut local_seen: HashSet<u32> = HashSet::new();
            for m in ac.find_iter(&text) {
                let start = m.start();
                let end = m.end();
                let before_ok = start == 0
                    || !bytes[start - 1].is_ascii_alphanumeric() && bytes[start - 1] != b'_';
                let after_ok =
                    end == text.len() || !bytes[end].is_ascii_alphanumeric() && bytes[end] != b'_';
                if !before_ok || !after_ok {
                    continue;
                }
                let pat_id = m.pattern().as_usize() as u32;
                if !local_seen.insert(pat_id) {
                    continue;
                }
                let (qname, target) = &pattern_meta[pat_id as usize];
                out.push((pat_id, *target, qname.clone()));
            }
            out
        })
        .collect();

    let mut by_target_type: BTreeMap<&'static str, Vec<UsesTypeEdge>> = BTreeMap::new();
    for (fn_info, matches) in functions.iter().zip(per_fn.into_iter()) {
        for (_pat_id, target, qname) in matches {
            by_target_type
                .entry(target)
                .or_default()
                .push(UsesTypeEdge {
                    function: fn_info.qualified_name.clone(),
                    type_name: qname,
                    target_node_type: target,
                });
        }
    }

    by_target_type
}

pub struct ReferencesEdge {
    pub function: String,
    pub constant: String,
    /// Line number in the function body where the reference appears.
    pub line: u32,
}

/// REFERENCES edges from `Function` to `Constant` — emit one row per
/// `(function, constant)` pair where the constant's terminal name
/// appears in the function body's identifier stream.
///
/// Per-language parsers populate `FunctionInfo.references` with
/// constant-style identifier candidates (the Rust parser uses
/// `SCREAMING_SNAKE_CASE` as the heuristic). This pass resolves each
/// candidate against the project's constant set and dedupes per
/// `(function, constant)` pair so a constant referenced N times in
/// one function still produces a single edge.
pub fn build_references_edges(
    functions: &[FunctionInfo],
    constants: &[ConstantInfo],
) -> Vec<ReferencesEdge> {
    if constants.is_empty() {
        return Vec::new();
    }

    // Name-keyed lookup: constant short-name → qualified_name. When two
    // constants share the same name (cross-module), we keep both —
    // emit edges to all matches. This mirrors how the type-name resolver
    // handles ambiguity (it doesn't disambiguate by import scope yet).
    let mut by_name: HashMap<&str, Vec<&str>> = HashMap::new();
    for c in constants {
        by_name
            .entry(c.name.as_str())
            .or_default()
            .push(c.qualified_name.as_str());
    }

    let mut out: Vec<ReferencesEdge> = Vec::new();
    for f in functions {
        if f.references.is_empty() {
            continue;
        }
        // Dedup per (function, constant_qname) — a function that uses
        // the same constant on three lines emits one edge.
        let mut seen: HashSet<&str> = HashSet::new();
        for (ident, line) in &f.references {
            let Some(matches) = by_name.get(ident.as_str()) else {
                continue;
            };
            for &qname in matches {
                if seen.insert(qname) {
                    out.push(ReferencesEdge {
                        function: f.qualified_name.clone(),
                        constant: qname.to_string(),
                        line: *line,
                    });
                }
            }
        }
    }
    out
}

/// FFI EXPOSES edges — #[pymodule] fn → each #[pyclass]/#[pyfunction] item.
pub fn build_ffi_exposes_edges(
    functions: &[FunctionInfo],
    classes: &[ClassInfo],
) -> Vec<FfiExposesEdge> {
    let pymodule_fns: Vec<&FunctionInfo> = functions
        .iter()
        .filter(|f| {
            f.metadata
                .get("is_pymodule")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })
        .collect();
    if pymodule_fns.is_empty() {
        return Vec::new();
    }

    let pyclass_items: Vec<(&ClassInfo, String)> = classes
        .iter()
        .filter(|c| {
            c.metadata
                .get("is_pyclass")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })
        .map(|c| {
            let py_name = c
                .metadata
                .get("py_name")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| c.name.clone());
            (c, py_name)
        })
        .collect();

    let pyfunc_items: Vec<(&FunctionInfo, String)> = functions
        .iter()
        .filter(|f| {
            !f.is_method
                && !f
                    .metadata
                    .get("is_pymodule")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                && f.metadata.get("ffi_kind").and_then(|v| v.as_str()) == Some("pyo3")
        })
        .map(|f| {
            let py_name = f
                .metadata
                .get("py_name")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| f.name.clone());
            (f, py_name)
        })
        .collect();

    let mut out = Vec::new();
    for mod_fn in &pymodule_fns {
        for (c, py_name) in &pyclass_items {
            out.push(FfiExposesEdge {
                module_fn: mod_fn.qualified_name.clone(),
                target_qname: c.qualified_name.clone(),
                target_type: "Struct",
                py_name: py_name.clone(),
            });
        }
        for (f, py_name) in &pyfunc_items {
            out.push(FfiExposesEdge {
                module_fn: mod_fn.qualified_name.clone(),
                target_qname: f.qualified_name.clone(),
                target_type: "Function",
                py_name: py_name.clone(),
            });
        }
    }
    out
}
