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
    Manifest, PythonExtensions,
};
use rmcp::transport::stdio;
use rmcp::ServiceExt;

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
    Graph { path: PathBuf },
    SourceRoot { dir: PathBuf },
    Workspace { dir: PathBuf },
    Watch { dir: PathBuf },
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
        Mode::Watch { .. } => "KGLite (watch)",
        Mode::Bare => "KGLite",
    }
}

fn default_manifest_path(mode: &Mode) -> Option<PathBuf> {
    match mode {
        Mode::Graph { path } => find_sibling_manifest(path),
        Mode::Workspace { dir } | Mode::Watch { dir } => find_workspace_manifest(dir),
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
    let mode = pick_mode(&cli);

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
    tools::register(server.tool_router_mut(), graph_state.clone());

    // Manifest python: tools + custom embedder. The framework returns
    // the embedder PyObject; we bind it to the active graph here so
    // `text_score()` queries work post-load.
    let py_ext = match manifest.as_ref() {
        Some(m) => apply_python_extensions(&mut server, m, cli.trust_tools)?,
        None => PythonExtensions::default(),
    };
    if let Some(emb) = py_ext.embedder {
        graph_state
            .bind_embedder(emb)
            .context("graph.set_embedder failed")?;
    }

    // Watch handler: rebuild on every debounced change batch.
    let watch_handle = if let Mode::Watch { dir } = &mode {
        let canon = dir.canonicalize()?;
        let gs = graph_state.clone();
        let cb: watch::ChangeHandler = Arc::new(move |_paths| {
            if let Err(e) = gs.build_code_tree(&canon) {
                tracing::warn!(error = %e, "code_tree rebuild failed");
            }
        });
        maybe_watch(Some(dir), Some(cb))?
    } else {
        None
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
