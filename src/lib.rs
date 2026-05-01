// src/lib.rs

// mimalloc as the global allocator. samply profile of the N-Triples
// build showed libsystem_malloc accounting for ~32% of loader-thread
// CPU time. mimalloc is consistently faster than macOS's default
// allocator on small-object-heavy workloads (Strings, HashMaps, Vecs
// in the parser hot loop). Pure Rust dependency — no system dep, just
// a slightly larger build artifact.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::prelude::*;
mod code_tree;
mod datatypes;
mod graph;
use graph::io::file::load_file;
use graph::pyapi::blueprint::from_blueprint_rust;
use graph::pyapi::result_view::{ResultIter, ResultView};
use graph::{KnowledgeGraph, Transaction};

#[pyfunction]
fn load(py: Python<'_>, path: String) -> PyResult<KnowledgeGraph> {
    py.detach(|| load_file(&path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

/// Names of every Cypher optimizer pass, in execution order. Useful for
/// the `disabled_passes=` kwarg of `KnowledgeGraph.cypher()` and for
/// bisection scripts. The list is the source of truth — names that
/// aren't here will be rejected by `cypher(..., disabled_passes=[...])`.
#[pyfunction]
fn cypher_pass_names() -> Vec<String> {
    graph::languages::cypher::planner::all_pass_names()
}

#[pymodule]
fn kglite(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(from_blueprint_rust, m)?)?;
    m.add_function(wrap_pyfunction!(cypher_pass_names, m)?)?;
    m.add_class::<KnowledgeGraph>()?;
    m.add_class::<Transaction>()?;
    m.add_class::<ResultView>()?;
    m.add_class::<ResultIter>()?;
    code_tree::pyapi::register(py, m)?;
    Ok(())
}
