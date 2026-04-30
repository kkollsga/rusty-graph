//! Builder: orchestrates parse → model → load phases.

pub mod call_edges;
pub mod load;
pub mod other_edges;
pub mod type_edges;

use crate::code_tree::models::ParseResult;
use crate::code_tree::parsers::{detect_languages, get_parser, language_for_path};
use crate::graph::KnowledgeGraph;
use pyo3::PyResult;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Full `build()` entry point matching the Python API.
///
/// Accepts either a directory or an explicit manifest file. When given a
/// manifest (or when one is auto-detected in the directory) the parser
/// uses manifest-declared source/test roots; otherwise it falls back to a
/// recursive directory scan.
pub fn run_with_options(
    input: &Path,
    verbose: bool,
    include_tests: bool,
    save_to: Option<&Path>,
) -> PyResult<KnowledgeGraph> {
    let input = input.canonicalize().unwrap_or_else(|_| input.to_path_buf());

    let (project_root, mut project_info) = if input.is_file() {
        let project_root = input
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| input.clone());
        let info = crate::code_tree::manifest::read_manifest_file(&input, &project_root)
            .ok_or_else(|| {
                pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                    "Not a recognised manifest file: {}",
                    input.file_name().and_then(|o| o.to_str()).unwrap_or(""),
                ))
            })?;
        (project_root, Some(info))
    } else if input.is_dir() {
        let info = crate::code_tree::manifest::read_manifest(&input);
        (input.clone(), info)
    } else {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Not a file or directory: {}",
            input.display()
        )));
    };

    let mut combined = ParseResult::new();
    let mut parsed_any = false;

    if let Some(info) = &mut project_info {
        if info.source_roots.is_empty() {
            // Manifest exists but declared no primary source roots (e.g. a
            // tooling-only pyproject.toml in a C/C++ repo). Don't parse just
            // tests — fall through to the whole-repo scan below so the
            // primary codebase isn't silently skipped.
            if verbose {
                eprintln!(
                    "Manifest: {} ({}) — no source roots declared, scanning whole repo",
                    info.manifest_path,
                    info.build_system.as_deref().unwrap_or("")
                );
            }
        } else {
            let mut roots: Vec<_> = info.source_roots.clone();
            if include_tests {
                roots.extend(info.test_roots.iter().cloned());
            }
            if verbose {
                eprintln!(
                    "Manifest: {} ({})",
                    info.manifest_path,
                    info.build_system.as_deref().unwrap_or("")
                );
                let labels: Vec<String> = roots
                    .iter()
                    .map(|r| {
                        r.path
                            .strip_prefix(&project_root)
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|_| r.path.display().to_string())
                    })
                    .collect();
                eprintln!("Source roots: {}", labels.join(", "));
            }
            let t_parse = std::time::Instant::now();
            for root in &roots {
                if !root.path.is_dir() {
                    continue;
                }
                let result = parse_directory(&root.path, verbose);
                combined.merge(result);
                parsed_any = true;
            }
            if verbose && parsed_any {
                eprintln!("[timing] parse: {:.3}s", t_parse.elapsed().as_secs_f64());
            }
        }
    }

    if !parsed_any {
        if !project_root.is_dir() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Not a directory: {}",
                project_root.display()
            )));
        }
        let t_parse = std::time::Instant::now();
        let result = parse_directory(&project_root, verbose);
        combined.merge(result);
        if verbose {
            eprintln!("[timing] parse: {:.3}s", t_parse.elapsed().as_secs_f64());
        }
    }

    finalize_and_load(combined, project_info, verbose, save_to)
}

fn parse_directory(dir: &Path, verbose: bool) -> ParseResult {
    // One walk, partition by language. The previous implementation walked
    // `dir` once for `detect_languages` and again per-language inside each
    // parser's `parse_directory` — N+1 traversals of the same tree. On
    // dotnet/runtime that was 8 walks of 57k entries; consolidating shaves
    // ~1–2s off the parse phase before any per-file work begins.
    let t_walk = std::time::Instant::now();
    let mut by_lang: BTreeMap<&'static str, Vec<PathBuf>> = BTreeMap::new();
    for entry in WalkDir::new(dir).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        if let Some(lang) = language_for_path(entry.path()) {
            by_lang
                .entry(lang)
                .or_default()
                .push(entry.path().to_path_buf());
        }
    }
    if verbose {
        let langs: Vec<&'static str> = by_lang.keys().copied().collect();
        eprintln!("  Detected languages in {}: {:?}", dir.display(), langs);
        for lang in &langs {
            eprintln!("  Found {} {} files", by_lang[lang].len(), lang);
        }
        eprintln!("[timing] walk: {:.3}s", t_walk.elapsed().as_secs_f64());
    }

    let mut combined = ParseResult::new();
    for (lang, files) in by_lang {
        let Some(parser) = get_parser(lang) else {
            continue;
        };
        let t_lang = std::time::Instant::now();
        let result = parser.parse_files(&files, dir);
        if verbose {
            eprintln!(
                "[timing] parse {}: {:.3}s ({} files)",
                lang,
                t_lang.elapsed().as_secs_f64(),
                files.len()
            );
        }
        combined.merge(result);
    }
    combined
}

fn finalize_and_load(
    mut combined: ParseResult,
    project_info: Option<crate::code_tree::models::ProjectInfo>,
    verbose: bool,
    save_to: Option<&Path>,
) -> PyResult<KnowledgeGraph> {
    if verbose {
        eprintln!(
            "Parsed: {} files, {} functions, {} classes, {} enums, {} interfaces, {} attributes, {} constants",
            combined.files.len(),
            combined.functions.len(),
            combined.classes.len(),
            combined.enums.len(),
            combined.interfaces.len(),
            combined.attributes.len(),
            combined.constants.len()
        );
    }

    let t_dedup = std::time::Instant::now();
    dedup_by_key(&mut combined.files, |f| f.path.clone());
    dedup_by_key(&mut combined.functions, |f| f.qualified_name.clone());
    dedup_by_key(&mut combined.classes, |c| c.qualified_name.clone());
    dedup_by_key(&mut combined.enums, |e| e.qualified_name.clone());
    dedup_by_key(&mut combined.interfaces, |i| i.qualified_name.clone());
    dedup_by_key(&mut combined.constants, |c| c.qualified_name.clone());
    if verbose {
        eprintln!("[timing] dedup: {:.3}s", t_dedup.elapsed().as_secs_f64());
    }

    let t_load = std::time::Instant::now();
    let graph = load::load_into_graph(&combined, project_info.as_ref())?;
    if verbose {
        eprintln!("[timing] load: {:.3}s", t_load.elapsed().as_secs_f64());
    }

    if let Some(dest) = save_to {
        // Mirror the prep that `KnowledgeGraph.save()` does — without these
        // steps, property column stores aren't materialised before
        // serialisation and only `id`/`title`/`type` survive the round-trip.
        let mut graph = graph;
        crate::graph::io::file::prepare_save(&mut graph.inner);
        std::sync::Arc::make_mut(&mut graph.inner).enable_columnar();
        let dest_str = dest.to_string_lossy();
        crate::graph::io::file::write_graph_v3(&graph.inner, &dest_str)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        return Ok(graph);
    }
    Ok(graph)
}

/// Legacy entry — directory-only, used by the initial smoke test.
pub fn run(src_dir: &Path, verbose: bool) -> PyResult<KnowledgeGraph> {
    let mut combined = ParseResult::new();
    let languages = detect_languages(src_dir);
    if verbose {
        eprintln!("Detected languages: {:?}", languages);
    }
    for lang in languages {
        let Some(parser) = get_parser(lang) else {
            if verbose {
                eprintln!("  (no Rust parser yet for {lang})");
            }
            continue;
        };
        if verbose {
            eprintln!("Parsing {} files...", lang);
        }
        let result = parser.parse_directory(src_dir, verbose);
        combined.merge(result);
    }

    // Dedup — overlapping source/test roots can parse the same file twice.
    // Last-seen wins so test-root flags take priority (matches builder.py).
    dedup_by_key(&mut combined.files, |f| f.path.clone());
    dedup_by_key(&mut combined.functions, |f| f.qualified_name.clone());
    dedup_by_key(&mut combined.classes, |c| c.qualified_name.clone());
    dedup_by_key(&mut combined.enums, |e| e.qualified_name.clone());
    dedup_by_key(&mut combined.interfaces, |i| i.qualified_name.clone());
    dedup_by_key(&mut combined.constants, |c| c.qualified_name.clone());

    if verbose {
        eprintln!(
            "Parsed: {} files, {} functions, {} classes, {} enums, {} interfaces, {} attributes, {} constants",
            combined.files.len(),
            combined.functions.len(),
            combined.classes.len(),
            combined.enums.len(),
            combined.interfaces.len(),
            combined.attributes.len(),
            combined.constants.len()
        );
    }

    load::load_into_graph(&combined, None)
}

/// Keep the last occurrence of each key, preserving encounter order otherwise.
fn dedup_by_key<T, K, F>(items: &mut Vec<T>, mut key: F)
where
    K: Eq + std::hash::Hash,
    F: FnMut(&T) -> K,
{
    let mut seen: std::collections::HashMap<K, usize> = std::collections::HashMap::new();
    for (idx, item) in items.iter().enumerate() {
        seen.insert(key(item), idx);
    }
    if seen.len() == items.len() {
        return;
    }
    let mut keep: Vec<usize> = seen.into_values().collect();
    keep.sort_unstable();
    let mut out: Vec<T> = Vec::with_capacity(keep.len());
    for (idx, item) in std::mem::take(items).into_iter().enumerate() {
        if keep.binary_search(&idx).is_ok() {
            out.push(item);
        }
    }
    *items = out;
}
