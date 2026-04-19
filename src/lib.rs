// src/lib.rs
use pyo3::prelude::*;
mod code_tree;
mod datatypes;
mod graph;
use graph::io::file::load_file;
use graph::pyapi::result_view::{ResultIter, ResultView};
use graph::{KnowledgeGraph, Transaction};

#[pyfunction]
fn load(py: Python<'_>, path: String) -> PyResult<KnowledgeGraph> {
    py.detach(|| load_file(&path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

#[pymodule]
fn kglite(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_class::<KnowledgeGraph>()?;
    m.add_class::<Transaction>()?;
    m.add_class::<ResultView>()?;
    m.add_class::<ResultIter>()?;
    code_tree::pyapi::register(py, m)?;
    Ok(())
}
