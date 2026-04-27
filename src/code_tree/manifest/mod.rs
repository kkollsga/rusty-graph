//! Project manifest readers — pyproject.toml, Cargo.toml.
//!
//! Port of parsers/manifest.py. The Python module only supports these
//! two formats; package.json and pom.xml are listed as future work but
//! not yet implemented upstream.

use crate::code_tree::models::{DependencyInfo, ProjectInfo, SourceRoot};
use std::path::{Path, PathBuf};
use toml::Value;

pub fn read_manifest(project_root: &Path) -> Option<ProjectInfo> {
    if (project_root.join("pyproject.toml")).is_file() {
        return read_pyproject(&project_root.join("pyproject.toml"), project_root).ok();
    }
    if (project_root.join("Cargo.toml")).is_file() {
        return read_cargo(&project_root.join("Cargo.toml"), project_root).ok();
    }
    None
}

/// Read a specific manifest file by path; delegates to the right reader.
pub fn read_manifest_file(manifest_path: &Path, project_root: &Path) -> Option<ProjectInfo> {
    let name = manifest_path.file_name()?.to_str()?;
    match name {
        "pyproject.toml" => read_pyproject(manifest_path, project_root).ok(),
        "Cargo.toml" => read_cargo(manifest_path, project_root).ok(),
        _ => None,
    }
}

fn load_toml(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    toml::from_str::<Value>(&text).map_err(|e| e.to_string())
}

fn read_pyproject(manifest_path: &Path, project_root: &Path) -> Result<ProjectInfo, String> {
    let data = load_toml(manifest_path)?;
    let project = data
        .get("project")
        .cloned()
        .unwrap_or(Value::Table(toml::map::Map::new()));
    let build_sys = data
        .get("build-system")
        .cloned()
        .unwrap_or(Value::Table(toml::map::Map::new()));
    let tool = data
        .get("tool")
        .cloned()
        .unwrap_or(Value::Table(toml::map::Map::new()));

    let fallback_name = project_root
        .file_name()
        .and_then(|o| o.to_str())
        .unwrap_or("")
        .to_string();
    let name = project
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or(&fallback_name)
        .to_string();
    let build_backend = build_sys
        .get("build-backend")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let mut version = project
        .get("version")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    if version.is_none() {
        if let Some(dyn_arr) = project.get("dynamic").and_then(|v| v.as_array()) {
            if dyn_arr.iter().any(|v| v.as_str() == Some("version")) {
                version = Some("dynamic".to_string());
            }
        }
    }

    let mut info = ProjectInfo {
        name,
        version,
        description: project
            .get("description")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        languages: vec!["python".to_string()],
        authors: extract_authors(project.get("authors")),
        license: extract_license(project.get("license")),
        repository_url: extract_repo_url(project.get("urls")),
        manifest_path: manifest_path
            .strip_prefix(project_root)
            .unwrap_or(manifest_path)
            .to_string_lossy()
            .to_string(),
        source_roots: Vec::new(),
        test_roots: Vec::new(),
        dependencies: Vec::new(),
        build_system: Some(if build_backend.is_empty() {
            "unknown".into()
        } else {
            build_backend.to_string()
        }),
        metadata: Default::default(),
    };

    if let Some(deps) = project.get("dependencies").and_then(|v| v.as_array()) {
        for d in deps {
            if let Some(s) = d.as_str() {
                info.dependencies.push(parse_dep_string(s));
            }
        }
    }
    if let Some(extras) = project
        .get("optional-dependencies")
        .and_then(|v| v.as_table())
    {
        for (group, deps) in extras {
            if let Some(arr) = deps.as_array() {
                for d in arr {
                    if let Some(s) = d.as_str() {
                        let mut dep = parse_dep_string(s);
                        dep.is_optional = true;
                        dep.group = Some(group.clone());
                        info.dependencies.push(dep);
                    }
                }
            }
        }
    }

    if let Some(py_root) = find_python_package(project_root, &info.name) {
        info.source_roots.push(SourceRoot {
            path: py_root,
            language: Some("python".to_string()),
            is_test: false,
            label: Some("python-package".to_string()),
        });
    }

    // Maturin → also read Cargo.toml for Rust roots.
    if build_backend.contains("maturin") {
        let cargo_path = project_root.join("Cargo.toml");
        if cargo_path.is_file() {
            if let Ok(rust_info) = read_cargo(&cargo_path, project_root) {
                info.source_roots.extend(rust_info.source_roots);
                info.languages.push("rust".to_string());
                info.dependencies.extend(rust_info.dependencies);
                if info.version.is_none() {
                    info.version = rust_info.version;
                }
            }
        }
    }

    let existing_test_paths: std::collections::HashSet<PathBuf> =
        info.test_roots.iter().map(|r| r.path.clone()).collect();
    if let Some(testpaths) = tool
        .get("pytest")
        .and_then(|v| v.get("ini_options"))
        .and_then(|v| v.get("testpaths"))
        .and_then(|v| v.as_array())
    {
        for tp in testpaths {
            if let Some(s) = tp.as_str() {
                let test_dir = project_root.join(s);
                if test_dir.is_dir() && !existing_test_paths.contains(&test_dir) {
                    info.test_roots.push(SourceRoot {
                        path: test_dir,
                        language: None,
                        is_test: true,
                        label: Some("pytest".to_string()),
                    });
                }
            }
        }
    }
    if info.test_roots.is_empty() {
        for candidate in ["tests", "test"] {
            let test_dir = project_root.join(candidate);
            if test_dir.is_dir() && !existing_test_paths.contains(&test_dir) {
                info.test_roots.push(SourceRoot {
                    path: test_dir,
                    language: None,
                    is_test: true,
                    label: Some("test-dir".to_string()),
                });
                break;
            }
        }
    }

    Ok(info)
}

fn read_cargo(manifest_path: &Path, project_root: &Path) -> Result<ProjectInfo, String> {
    let data = load_toml(manifest_path)?;
    let package = data
        .get("package")
        .cloned()
        .unwrap_or(Value::Table(toml::map::Map::new()));

    let fallback_name = project_root
        .file_name()
        .and_then(|o| o.to_str())
        .unwrap_or("")
        .to_string();
    let name = package
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or(&fallback_name)
        .to_string();

    let authors: Vec<String> = package
        .get("authors")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let mut info = ProjectInfo {
        name,
        version: package
            .get("version")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        description: package
            .get("description")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        languages: vec!["rust".to_string()],
        authors,
        license: package
            .get("license")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        repository_url: package
            .get("repository")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        manifest_path: manifest_path
            .strip_prefix(project_root)
            .unwrap_or(manifest_path)
            .to_string_lossy()
            .to_string(),
        source_roots: Vec::new(),
        test_roots: Vec::new(),
        dependencies: Vec::new(),
        build_system: Some("cargo".into()),
        metadata: Default::default(),
    };

    let src_dir = project_root.join("src");
    if src_dir.is_dir() {
        info.source_roots.push(SourceRoot {
            path: src_dir,
            language: Some("rust".to_string()),
            is_test: false,
            label: Some("rust-crate".to_string()),
        });
    }

    // Capture `[lib] crate-type` so downstream queries can tell apart a
    // regular `lib` crate (where `pub fn` is a real Rust API export) from a
    // `cdylib` PyO3 crate (where only `#[pyfunction]` / `#[pymethods]`
    // exposure matters and `pub fn` is meaningless to the outside world).
    if let Some(crate_types) = data
        .get("lib")
        .and_then(|v| v.as_table())
        .and_then(|lib| lib.get("crate-type"))
        .and_then(|v| v.as_array())
    {
        let types: Vec<serde_json::Value> = crate_types
            .iter()
            .filter_map(|v| v.as_str().map(|s| serde_json::Value::String(s.to_string())))
            .collect();
        if !types.is_empty() {
            info.metadata
                .insert("crate_type".to_string(), serde_json::Value::Array(types));
        }
    }

    // Workspace members — glob manually.
    if let Some(members) = data
        .get("workspace")
        .and_then(|v| v.get("members"))
        .and_then(|v| v.as_array())
    {
        for member_val in members {
            let Some(member_glob) = member_val.as_str() else {
                continue;
            };
            // Resolve simple globs relative to project_root. Full glob would
            // require the `glob` crate; Python used Path.glob here. Keep it
            // simple: support either a literal path or "dir/*".
            let matches = match member_glob.strip_suffix("/*") {
                Some(parent) => std::fs::read_dir(project_root.join(parent))
                    .ok()
                    .map(|iter| {
                        iter.filter_map(Result::ok)
                            .map(|e| e.path())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
                None => vec![project_root.join(member_glob)],
            };
            for member_dir in matches {
                let member_src = member_dir.join("src");
                if member_src.is_dir() {
                    let label = format!(
                        "workspace:{}",
                        member_dir
                            .file_name()
                            .and_then(|o| o.to_str())
                            .unwrap_or(""),
                    );
                    info.source_roots.push(SourceRoot {
                        path: member_src,
                        language: Some("rust".to_string()),
                        is_test: false,
                        label: Some(label),
                    });
                }
            }
        }
    }

    let tests_dir = project_root.join("tests");
    if tests_dir.is_dir() {
        info.test_roots.push(SourceRoot {
            path: tests_dir,
            language: Some("rust".to_string()),
            is_test: true,
            label: Some("rust-integration-tests".to_string()),
        });
    }

    if let Some(deps) = data.get("dependencies").and_then(|v| v.as_table()) {
        for (name, spec) in deps {
            info.dependencies.push(parse_cargo_dep(name, spec, false));
        }
    }
    if let Some(deps) = data.get("dev-dependencies").and_then(|v| v.as_table()) {
        for (name, spec) in deps {
            info.dependencies.push(parse_cargo_dep(name, spec, true));
        }
    }

    Ok(info)
}

// ── Helpers ───────────────────────────────────────────────────────

fn find_python_package(project_root: &Path, project_name: &str) -> Option<PathBuf> {
    let pkg_name = project_name.replace('-', "_");
    let flat = project_root.join(&pkg_name);
    if flat.is_dir() && flat.join("__init__.py").is_file() {
        return Some(flat);
    }
    let src_layout = project_root.join("src").join(&pkg_name);
    if src_layout.is_dir() && src_layout.join("__init__.py").is_file() {
        return Some(src_layout);
    }
    None
}

fn extract_authors(value: Option<&Value>) -> Vec<String> {
    let Some(arr) = value.and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|v| match v {
            Value::String(s) => Some(s.clone()),
            Value::Table(t) => {
                let name = t.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let email = t.get("email").and_then(|v| v.as_str()).unwrap_or("");
                if email.is_empty() {
                    Some(name.to_string())
                } else {
                    Some(format!("{} <{}>", name, email))
                }
            }
            _ => None,
        })
        .collect()
}

fn extract_license(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) => Some(s.clone()),
        Value::Table(t) => t
            .get("text")
            .and_then(|v| v.as_str())
            .or_else(|| t.get("file").and_then(|v| v.as_str()))
            .map(str::to_string),
        _ => None,
    }
}

fn extract_repo_url(urls: Option<&Value>) -> Option<String> {
    let urls = urls?.as_table()?;
    for key in ["Repository", "repository", "Source", "source", "Homepage"] {
        if let Some(v) = urls.get(key).and_then(|v| v.as_str()) {
            return Some(v.to_string());
        }
    }
    None
}

fn parse_dep_string(dep_str: &str) -> DependencyInfo {
    for (i, ch) in dep_str.char_indices() {
        if matches!(ch, '>' | '=' | '<' | '!' | '~') {
            return DependencyInfo {
                name: dep_str[..i].trim().to_string(),
                version_spec: Some(dep_str[i..].trim().to_string()),
                is_dev: false,
                is_optional: false,
                group: None,
            };
        }
    }
    DependencyInfo {
        name: dep_str.trim().to_string(),
        version_spec: None,
        is_dev: false,
        is_optional: false,
        group: None,
    }
}

fn parse_cargo_dep(name: &str, spec: &Value, is_dev: bool) -> DependencyInfo {
    let version = match spec {
        Value::String(s) => Some(s.clone()),
        Value::Table(t) => t
            .get("version")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        _ => None,
    };
    DependencyInfo {
        name: name.to_string(),
        version_spec: version,
        is_dev,
        is_optional: false,
        group: None,
    }
}
