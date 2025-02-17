// src/lib.rs
use pyo3::prelude::*;
mod graph;
mod datatypes;
use graph::KnowledgeGraph;
use graph::io_operations::load_file;

#[pyfunction]
fn load(path: String) -> PyResult<KnowledgeGraph> {
    load_file(&path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

#[pymodule]
fn rusty_graph(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_class::<KnowledgeGraph>()?;
    Ok(())
}