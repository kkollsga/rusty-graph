//! Embedder trait — pluggable text-embedding backend behind the
//! `KnowledgeGraph`'s `embedder` field.
//!
//! kglite supports semantic search via `text_score()` in Cypher
//! queries. To embed query strings at lookup time the graph holds an
//! optional embedder. Originally this was a `Py<PyAny>` referencing
//! a user-provided Python class — that worked for the Python wheel
//! but forced libpython into every downstream Rust binary that held
//! a `KnowledgeGraph`.
//!
//! 0.9.18 introduces this trait so the embedder can be backed by:
//!
//! - [`py_adapter::PyEmbedderAdapter`] — wraps a user-provided Python
//!   class. Used by the Python API path (`g.set_embedder(my_python_obj)`).
//!   Lives behind PyO3 but only ever instantiated under `#[pymethods]`.
//!
//! - [`fastembed::FastEmbedAdapter`] (gated on the `fastembed` Cargo
//!   feature) — Rust-native ONNX inference via `fastembed-rs`.
//!   Operators get this via `extensions.embedder: { backend: fastembed,
//!   model: BAAI/bge-m3 }` in their YAML manifest. Lets kglite-mcp-server
//!   support `text_score()` without embedding Python.
//!
//! Both implement [`Embedder`]; downstream consumers (the Cypher
//! engine's text-score rewrite, the `embed_texts` / `search_text` pymethods)
//! call through the trait without caring which backend they got.

pub mod py_adapter;

#[cfg(feature = "fastembed")]
pub mod fastembed;

/// Pluggable text-embedding backend. Implementations must be
/// `Send + Sync` because the `KnowledgeGraph` is freely cloned across
/// threads (its `embedder` field is an `Arc<dyn Embedder>`).
pub trait Embedder: Send + Sync {
    /// Embedding vector dimensionality (e.g. 1024 for BAAI/bge-m3,
    /// 384 for all-MiniLM-L6-v2). Used at `set_embeddings()` time to
    /// validate that user-supplied vectors match what the embedder
    /// produces.
    fn dimension(&self) -> usize;

    /// Embed a batch of texts into vectors. The returned outer Vec
    /// has the same length as the input slice (one vector per text);
    /// each inner Vec has length [`Self::dimension`].
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;

    /// Optional lifecycle hook. Called by `embed_texts` / `search_text`
    /// before each embedding pass so the implementation can lazily
    /// materialise heavy resources (model weights, ONNX session, etc.)
    /// Default: no-op.
    fn load(&self) -> Result<(), String> {
        Ok(())
    }

    /// Optional lifecycle hook. Called after each embedding pass —
    /// implementations typically use this to schedule a cooldown
    /// timer that frees resources after some idle period. Default:
    /// no-op. Errors are silently ignored by callers since this is
    /// cleanup.
    fn unload(&self) {}
}
