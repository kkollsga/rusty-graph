mod node;
mod relation;
mod graph;

use pyo3::prelude::*;
//use node::Node;
//use relation::Relation;
use graph::KnowledgeGraph;

#[pymodule]
fn rusty_graph(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<KnowledgeGraph>()?;
    // If you want to expose Entity and Relation to Python, add them here as well
    Ok(())
}
