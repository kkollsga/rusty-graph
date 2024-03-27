use pyo3::prelude::*;

mod schema;
mod graph;
mod data_types;

use graph::KnowledgeGraph;

#[pymodule]
fn rusty_graph(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<KnowledgeGraph>()?;
    Ok(())
}