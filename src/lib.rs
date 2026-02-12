// src/lib.rs
use pyo3::prelude::*;
mod datatypes;
mod graph;
use graph::io_operations::load_file;
use graph::{KnowledgeGraph, Transaction};

#[pyfunction]
fn load(path: String) -> PyResult<KnowledgeGraph> {
    load_file(&path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

#[pymodule]
fn kglite(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_class::<KnowledgeGraph>()?;
    m.add_class::<Transaction>()?;
    Ok(())
}
