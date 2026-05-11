//! CSV-over-HTTP server — operator pinch-point P3.
//!
//! When the manifest declares
//!
//! ```yaml
//! extensions:
//!   csv_http_server:
//!     port: 8765
//!     dir: temp/
//! ```
//!
//! the binary spawns a tiny tokio HTTP listener bound to
//! `127.0.0.1:<port>` that serves CSV files out of the configured
//! directory. The `cypher_query` tool, when it sees `FORMAT CSV`
//! in the query, writes the result to `<dir>/<uuid>.csv` and
//! returns the URL instead of the inline CSV blob — agents fetch
//! the URL when they're ready to consume the table, which keeps
//! the MCP response budget small even for million-row exports.
//!
//! Only GETs of files inside `<dir>` are served. There is no
//! directory listing, no file upload, no write surface. The
//! server is bound to loopback and not exposed to the network.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use bytes::Bytes;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use tokio::net::TcpListener;

/// Resolved configuration for the CSV-over-HTTP server.
#[derive(Clone, Debug)]
pub struct CsvHttpConfig {
    /// TCP port to bind on 127.0.0.1. Defaults to 8765.
    pub port: u16,
    /// Directory containing the CSV files. Resolved relative to
    /// the manifest directory at config-load time.
    pub dir: PathBuf,
    /// Optional `Access-Control-Allow-Origin` value. When unset,
    /// the server emits `Access-Control-Allow-Origin: *` so any
    /// agent UI can fetch the CSV without preflight friction.
    pub cors_origin: Option<String>,
}

impl CsvHttpConfig {
    /// Parse the `extensions.csv_http_server` mapping from the
    /// manifest. Accepted shapes:
    ///
    /// ```yaml
    /// csv_http_server: true                 # defaults: port 8765, dir temp/
    /// csv_http_server: { port: 9000 }       # custom port, default dir
    /// csv_http_server: { dir: out/ }        # custom dir, default port
    /// csv_http_server: { port: 9000, dir: out/, cors_origin: "https://my.app" }
    /// ```
    ///
    /// `dir` is resolved relative to `base_dir` (typically the
    /// manifest's parent directory) — operators write
    /// project-relative paths and the runtime translates them.
    pub fn from_manifest_value(value: &serde_json::Value, base_dir: &Path) -> Result<Option<Self>> {
        let obj = match value {
            serde_json::Value::Bool(false) | serde_json::Value::Null => return Ok(None),
            serde_json::Value::Bool(true) => {
                return Ok(Some(Self::resolved("temp", base_dir, None, None)));
            }
            serde_json::Value::Object(o) => o,
            _ => anyhow::bail!(
                "extensions.csv_http_server must be a mapping or boolean (got: {value:?})"
            ),
        };

        let port = match obj.get("port") {
            Some(serde_json::Value::Number(n)) => n
                .as_u64()
                .and_then(|n| u16::try_from(n).ok())
                .context("csv_http_server.port must fit in u16")?,
            Some(other) => anyhow::bail!("csv_http_server.port must be a number (got: {other:?})"),
            None => 8765,
        };

        let dir = obj.get("dir").and_then(|v| v.as_str()).unwrap_or("temp");

        let cors_origin = obj
            .get("cors_origin")
            .and_then(|v| v.as_str())
            .map(String::from);

        Ok(Some(Self::resolved(dir, base_dir, Some(port), cors_origin)))
    }

    fn resolved(
        dir: &str,
        base_dir: &Path,
        port: Option<u16>,
        cors_origin: Option<String>,
    ) -> Self {
        Self {
            port: port.unwrap_or(8765),
            dir: base_dir.join(dir),
            cors_origin,
        }
    }

    /// Construct the public URL for a generated CSV file. The
    /// filename is sanitised against directory traversal at write
    /// time, so `name` is trusted by the server.
    pub fn url_for(&self, name: &str) -> String {
        format!("http://127.0.0.1:{}/{}", self.port, name)
    }
}

/// Spawn the CSV-over-HTTP listener as a tokio task. Returns
/// immediately after the listener has bound; the task runs for
/// the lifetime of the process. Bind errors (port-in-use, etc.)
/// propagate to the caller so the boot path can fail fast.
pub async fn spawn(config: CsvHttpConfig) -> Result<()> {
    std::fs::create_dir_all(&config.dir).with_context(|| {
        format!(
            "csv_http_server: failed to create directory {}",
            config.dir.display()
        )
    })?;

    let addr = SocketAddr::from(([127, 0, 0, 1], config.port));
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("csv_http_server: bind {addr} failed"))?;
    tracing::info!(
        port = config.port,
        dir = %config.dir.display(),
        "csv_http_server listening"
    );

    let state = Arc::new(config);
    tokio::spawn(async move {
        loop {
            let (stream, _peer) = match listener.accept().await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(error = %e, "csv_http_server: accept failed");
                    continue;
                }
            };
            let state = state.clone();
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let svc = service_fn(move |req: Request<Incoming>| handle(req, state.clone()));
                let builder = Builder::new(TokioExecutor::new());
                if let Err(e) = builder.serve_connection(io, svc).await {
                    tracing::debug!(error = %e, "csv_http_server: connection ended");
                }
            });
        }
    });
    Ok(())
}

async fn handle(
    req: Request<Incoming>,
    state: Arc<CsvHttpConfig>,
) -> Result<Response<Full<Bytes>>, std::convert::Infallible> {
    let cors = state.cors_origin.clone().unwrap_or_else(|| "*".to_string());

    // OPTIONS preflight — return CORS headers, no body.
    if req.method() == Method::OPTIONS {
        return Ok(cors_response(StatusCode::NO_CONTENT, &cors, Vec::new()));
    }
    if req.method() != Method::GET {
        return Ok(cors_response(
            StatusCode::METHOD_NOT_ALLOWED,
            &cors,
            b"only GET is supported".to_vec(),
        ));
    }

    // Strip leading `/` and reject any path component containing
    // a slash, backslash, or `..` — only flat filenames in the
    // configured directory are servable. This is enforced again
    // after canonicalisation below as defence-in-depth.
    let raw = req.uri().path().trim_start_matches('/');
    if raw.is_empty()
        || raw.contains('/')
        || raw.contains('\\')
        || raw.split('/').any(|c| c == "..")
    {
        return Ok(cors_response(
            StatusCode::BAD_REQUEST,
            &cors,
            b"invalid path".to_vec(),
        ));
    }
    let path = state.dir.join(raw);

    // Canonicalise both paths and verify the resolved file lives
    // under the configured directory.
    let dir_canon = match state.dir.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(error = %e, dir = %state.dir.display(), "canonicalize failed");
            return Ok(cors_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &cors,
                b"server misconfigured".to_vec(),
            ));
        }
    };
    let file_canon = match path.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            return Ok(cors_response(
                StatusCode::NOT_FOUND,
                &cors,
                b"not found".to_vec(),
            ));
        }
    };
    if !file_canon.starts_with(&dir_canon) {
        return Ok(cors_response(
            StatusCode::FORBIDDEN,
            &cors,
            b"forbidden".to_vec(),
        ));
    }

    let body = match tokio::fs::read(&file_canon).await {
        Ok(b) => b,
        Err(_) => {
            return Ok(cors_response(
                StatusCode::NOT_FOUND,
                &cors,
                b"not found".to_vec(),
            ));
        }
    };
    let content_type = if raw.ends_with(".csv") {
        "text/csv; charset=utf-8"
    } else {
        "application/octet-stream"
    };
    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", content_type)
        .header("Access-Control-Allow-Origin", &cors)
        .header("Cache-Control", "no-store")
        .body(Full::new(Bytes::from(body)))
        .unwrap();
    Ok(response)
}

fn cors_response(status: StatusCode, cors: &str, body: Vec<u8>) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("Access-Control-Allow-Origin", cors)
        .header("Access-Control-Allow-Methods", "GET, OPTIONS")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

/// Write a CSV body to a fresh file inside the configured
/// directory and return the file name (basename only — caller
/// joins with `CsvHttpConfig::url_for` to make the public URL).
/// The filename is a short random suffix so concurrent queries
/// don't collide.
pub fn write_csv(config: &CsvHttpConfig, csv: &str) -> Result<String> {
    std::fs::create_dir_all(&config.dir).with_context(|| {
        format!(
            "csv_http_server: failed to create directory {}",
            config.dir.display()
        )
    })?;
    // Use nanoseconds + a counter-based suffix derived from the
    // CSV content's hash so two queries fired in the same nanosec
    // don't collide. Avoids pulling in `uuid` for a one-call site.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(csv, &mut hasher);
    let h = std::hash::Hasher::finish(&hasher);
    let name = format!("kglite-{stamp:x}-{h:x}.csv");
    let path = config.dir.join(&name);
    std::fs::write(&path, csv)
        .with_context(|| format!("csv_http_server: failed to write {}", path.display()))?;
    Ok(name)
}
