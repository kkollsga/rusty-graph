use pyo3::prelude::*;
use pyo3::types::PyDict;
use petgraph::graph::DiGraph;
use crate::schema::{Node, Relation};
use std::collections::HashMap;

pub fn query_nodes(
    graph: &mut DiGraph<Node, Relation>,
    indices: Vec<usize>,
    filter_dict: Option<&PyDict>,
) -> PyResult<Vec<usize>> {
    if let Some(dict) = filter_dict {
        let mut filters = Vec::new();
        let mut filter_map = HashMap::new();

        for (key, value) in dict.iter() {
            filter_map.insert(
                key.extract::<String>()?,
                value.extract::<String>()?,
            );
        }
        filters.push(filter_map);

        Ok(crate::graph::navigate_graph::get_nodes(
            graph,
            None,
            Some(filters),
        ))
    } else {
        Ok(indices)
    }
}

pub fn traverse_nodes_in(
    graph: &DiGraph<Node, Relation>,
    indices: Vec<usize>,
    relationship_type: String,
) -> Vec<usize> {
    crate::graph::navigate_graph::traverse_nodes(
        graph,
        indices,
        relationship_type,
        true,
        None,
        None,
        None,
    )
}

pub fn traverse_nodes_out(
    graph: &DiGraph<Node, Relation>,
    indices: Vec<usize>,
    relationship_type: String,
) -> Vec<usize> {
    crate::graph::navigate_graph::traverse_nodes(
        graph,
        indices,
        relationship_type,
        false,
        None,
        None,
        None,
    )
}