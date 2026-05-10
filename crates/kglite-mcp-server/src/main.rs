//! `kglite-mcp-server` — single-binary MCP server for KGLite knowledge graphs.
//!
//! Layers three kglite-specific tools on top of the generic
//! `mcp-server` framework: `cypher_query`, `graph_overview`, and
//! `save_graph`. All three close over a [`GraphState`] holding the
//! active `KnowledgeGraph` Python object.
//!
//! Modes:
//! - `--graph X.kgl` — load a pre-built graph file at boot.
//! - `--workspace DIR` — multi-repo. Post-activate hook runs
//!   `kglite.code_tree.build()` on each cloned repo.
//! - `--watch DIR` — file-watcher mode. Change handler rebuilds the
//!   code-tree graph and atomic-swaps the active slot.
//! - `--source-root DIR` — generic file-tree mode (no graph).
//! - bare — framework + manifest tools only.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use mcp_server::manifest::{find_sibling_manifest, find_workspace_manifest, ManifestError};
use mcp_server::server::{McpServer, ServerOptions};
use mcp_server::{
    apply_python_extensions, init_tracing, maybe_watch, resolve_source_roots, watch, workspace,
    Manifest, PythonExtensions, WorkspaceKind,
};
use rmcp::transport::stdio;
use rmcp::ServiceExt;

mod cypher_tools;
mod tools;
use crate::tools::GraphState;

#[derive(Parser, Debug)]
#[command(
    name = "kglite-mcp-server",
    about = "MCP server for KGLite knowledge graphs (Rust-native)"
)]
struct Cli {
    /// Path to a .kgl knowledge graph file. Loaded at boot.
    #[arg(long, conflicts_with_all = ["workspace", "watch", "source_root"])]
    graph: Option<PathBuf>,

    /// Source-root mode (no graph).
    #[arg(long = "source-root", conflicts_with_all = ["graph", "workspace", "watch"])]
    source_root: Option<PathBuf>,

    /// Workspace mode: clone GitHub repos and build code-tree graphs.
    #[arg(long, conflicts_with_all = ["graph", "source_root", "watch"])]
    workspace: Option<PathBuf>,

    /// Watch mode: rebuild the code-tree graph on file changes.
    #[arg(long, conflicts_with_all = ["graph", "source_root", "workspace"])]
    watch: Option<PathBuf>,

    #[arg(long = "mcp-config")]
    mcp_config: Option<PathBuf>,
    #[arg(long)]
    name: Option<String>,
    #[arg(long = "trust-tools")]
    trust_tools: bool,
    #[arg(long = "stale-after-days", default_value_t = 7)]
    stale_after_days: u32,
}

#[derive(Debug, Clone)]
enum Mode {
    Graph {
        path: PathBuf,
    },
    SourceRoot {
        dir: PathBuf,
    },
    Workspace {
        dir: PathBuf,
    },
    /// `manifest.workspace.kind: local`. Equivalent to `--workspace`
    /// but bound to a fixed local directory (no clone) and with
    /// `set_root_dir` registered for runtime root swap. Manifest
    /// declaration wins over the `--workspace` CLI flag.
    LocalWorkspace {
        root: PathBuf,
        watch: bool,
    },
    Watch {
        dir: PathBuf,
    },
    Bare,
}

fn pick_mode(cli: &Cli) -> Mode {
    if let Some(p) = &cli.graph {
        Mode::Graph { path: p.clone() }
    } else if let Some(d) = &cli.source_root {
        Mode::SourceRoot { dir: d.clone() }
    } else if let Some(d) = &cli.workspace {
        Mode::Workspace { dir: d.clone() }
    } else if let Some(d) = &cli.watch {
        Mode::Watch { dir: d.clone() }
    } else {
        Mode::Bare
    }
}

fn fallback_name(mode: &Mode) -> &'static str {
    match mode {
        Mode::Graph { .. } => "KGLite (single-graph)",
        Mode::SourceRoot { .. } => "KGLite (source-root)",
        Mode::Workspace { .. } => "KGLite (workspace)",
        Mode::LocalWorkspace { .. } => "KGLite (local-workspace)",
        Mode::Watch { .. } => "KGLite (watch)",
        Mode::Bare => "KGLite",
    }
}

fn default_manifest_path(mode: &Mode) -> Option<PathBuf> {
    match mode {
        Mode::Graph { path } => find_sibling_manifest(path),
        Mode::Workspace { dir } | Mode::Watch { dir } => find_workspace_manifest(dir),
        Mode::LocalWorkspace { root, .. } => find_workspace_manifest(root),
        Mode::SourceRoot { .. } | Mode::Bare => None,
    }
}

fn load_manifest(cli: &Cli, mode: &Mode) -> Result<Option<Manifest>, ManifestError> {
    let path = match &cli.mcp_config {
        Some(p) if !p.is_file() => {
            return Err(ManifestError::bare(format!(
                "--mcp-config path does not exist: {}",
                p.display()
            )))
        }
        Some(p) => Some(p.clone()),
        None => default_manifest_path(mode),
    };
    match path {
        Some(p) => Ok(Some(mcp_server::load_manifest(&p)?)),
        None => Ok(None),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let mut mode = pick_mode(&cli);

    if let Mode::Graph { path } = &mode {
        if !path.is_file() {
            anyhow::bail!("--graph path does not exist: {}", path.display());
        }
    }
    if let Mode::SourceRoot { dir } | Mode::Watch { dir } = &mode {
        if !dir.is_dir() {
            anyhow::bail!(
                "path does not exist or is not a directory: {}",
                dir.display()
            );
        }
    }

    let manifest = load_manifest(&cli, &mode).context("manifest load failed")?;

    // Manifest `workspace.kind: local` wins over CLI flags — promote
    // before mode-specific binding runs so the rest of the boot path
    // sees `Mode::LocalWorkspace`. Mirrors the framework's own
    // `mcp-server` binary (`crates/mcp-server/src/main.rs` in 0.3.23+).
    if let Some(m) = manifest.as_ref() {
        if let Some(wcfg) = m.workspace.as_ref() {
            if wcfg.kind == WorkspaceKind::Local {
                let raw_root = wcfg.root.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("manifest.workspace.kind=local is missing required `root`")
                })?;
                let base = m
                    .yaml_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."));
                let resolved = base.join(raw_root).canonicalize().with_context(|| {
                    format!("workspace.root {raw_root:?} resolves to a path that does not exist")
                })?;
                mode = Mode::LocalWorkspace {
                    root: resolved,
                    watch: wcfg.watch,
                };
            }
        }
    }

    let mut options = ServerOptions::from_manifest(manifest.as_ref(), fallback_name(&mode));
    if cli.name.is_some() {
        options.name = cli.name.clone();
    }

    let graph_state = GraphState::new();

    // Mode-specific bindings: source roots, workspace handle, initial graph build.
    match &mode {
        Mode::Graph { path } => {
            let canon = path.canonicalize()?;
            graph_state.load_kgl(&canon).context("kglite.load failed")?;
            if let Some(parent) = canon.parent() {
                options =
                    options.with_static_source_roots(vec![parent.to_string_lossy().into_owned()]);
            }
        }
        Mode::SourceRoot { dir } | Mode::Watch { dir } => {
            let canon = dir.canonicalize()?;
            options = options.with_static_source_roots(vec![canon.to_string_lossy().into_owned()]);
            if matches!(mode, Mode::Watch { .. }) {
                graph_state
                    .build_code_tree(&canon)
                    .context("initial code_tree build failed")?;
            }
        }
        Mode::Workspace { dir } => {
            let canon = dir.canonicalize().unwrap_or_else(|_| dir.clone());
            let gs = graph_state.clone();
            let hook: workspace::PostActivateHook = Arc::new(move |path, name| {
                tracing::info!(repo = name, "code_tree::build on activate");
                gs.build_code_tree(path)
            });
            let ws = workspace::Workspace::open(canon, cli.stale_after_days, Some(hook))
                .context("workspace init failed")?;
            options = options.with_workspace(ws);
        }
        Mode::LocalWorkspace { root, .. } => {
            let gs = graph_state.clone();
            let hook: workspace::PostActivateHook = Arc::new(move |path, name| {
                tracing::info!(repo = name, "code_tree::build on local-workspace activate");
                gs.build_code_tree(path)
            });
            let ws = workspace::Workspace::open_local(root.clone(), Some(hook))
                .context("local-workspace init failed")?;
            options = options.with_workspace(ws);
        }
        Mode::Bare => {
            if let Some(m) = manifest.as_ref() {
                if !m.source_roots.is_empty() {
                    let resolved =
                        resolve_source_roots(m).context("source root resolution failed")?;
                    options = options.with_static_source_roots(resolved);
                }
            }
        }
    }

    let mut server = McpServer::new(options);
    tools::register(&mut server, graph_state.clone());

    // Manifest python: tools + custom embedder. As of mcp-methods
    // 0.3.22 the framework returns an `Arc<EmbedderHandle>` (load /
    // unload / embed + idle-watch tracking) instead of a raw
    // `Py<PyAny>`. We pull the underlying Python instance out of
    // the handle and bind that to the active graph — kglite's
    // per-batch `try_load_embedder` / `try_unload_embedder` in
    // `vector.rs` then drives the same instance the framework's
    // idle-watch task is observing. Both lifecycle layers operate
    // on the same Python object; per-batch is the primary driver,
    // idle-watch is a safety net (per the 0.3.22 ack).
    let py_ext = match manifest.as_ref() {
        Some(m) => apply_python_extensions(&mut server, m, cli.trust_tools)?,
        None => PythonExtensions::default(),
    };
    if let Some(handle) = py_ext.embedder {
        let instance = pyo3::Python::attach(|py| handle.instance().clone_ref(py));
        graph_state
            .bind_embedder(instance)
            .context("graph.set_embedder failed")?;
    }

    // YAML-declared `tools[].cypher` entries. The mcp-methods framework
    // parses them into `manifest.tools` but stays domain-agnostic and
    // doesn't know how to run Cypher — so the kglite shim owns the
    // registration loop, using the framework's now-public
    // `build_tool_attr` plus an in-shim runner that dispatches into the
    // active graph's `cypher(template, params=args)` method.
    if let Some(m) = manifest.as_ref() {
        let runner = cypher_tools::make_runner(graph_state.clone());
        let registered = cypher_tools::register_cypher_tools(&mut server, m, runner)
            .context("YAML cypher tool registration failed")?;
        if registered > 0 {
            tracing::info!(count = registered, "manifest cypher tools registered");
        }
    }

    // Watch handler: rebuild on every debounced change batch. Both
    // explicit `--watch DIR` and `manifest.workspace.kind: local` with
    // `watch: true` wire the same change-handler shape.
    let watch_handle = match &mode {
        Mode::Watch { dir } => {
            let canon = dir.canonicalize()?;
            let gs = graph_state.clone();
            let cb: watch::ChangeHandler = Arc::new(move |_paths| {
                if let Err(e) = gs.build_code_tree(&canon) {
                    tracing::warn!(error = %e, "code_tree rebuild failed");
                }
            });
            maybe_watch(Some(dir), Some(cb))?
        }
        Mode::LocalWorkspace { root, watch: true } => {
            let canon = root.clone();
            let gs = graph_state.clone();
            let cb: watch::ChangeHandler = Arc::new(move |_paths| {
                if let Err(e) = gs.build_code_tree(&canon) {
                    tracing::warn!(error = %e, "code_tree rebuild failed (local workspace)");
                }
            });
            maybe_watch(Some(root), Some(cb))?
        }
        _ => None,
    };
    let _watch_handle = watch_handle;

    print_boot_summary(&mode, manifest.as_ref(), &graph_state);

    let service = server
        .serve(stdio())
        .await
        .context("failed to start MCP service over stdio")?;
    service.waiting().await?;
    Ok(())
}

fn print_boot_summary(mode: &Mode, manifest: Option<&Manifest>, graph_state: &GraphState) {
    let label = match mode {
        Mode::Graph { path } => format!("graph [{}]", path.display()),
        Mode::SourceRoot { dir } => format!("source-root [{}]", dir.display()),
        Mode::Workspace { dir } => format!("workspace [{}]", dir.display()),
        Mode::LocalWorkspace { root, watch } => format!(
            "local-workspace [{}{}]",
            root.display(),
            if *watch { " +watch" } else { "" }
        ),
        Mode::Watch { dir } => format!("watch [{}]", dir.display()),
        Mode::Bare => "bare".to_string(),
    };
    let mut parts = vec![format!("mode: {label}")];
    if let Some(m) = manifest {
        parts.push(format!("manifest: {}", m.yaml_path.display()));
    }
    if let Some((nodes, edges)) = graph_state.schema() {
        parts.push(format!("graph: {nodes} nodes, {edges} edges"));
    }
    eprintln!("kglite-mcp-server: {}", parts.join("; "));
}
