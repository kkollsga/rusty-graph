use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use pyo3::PyResult;
use petgraph::graph::DiGraph;
use std::collections::HashMap;

use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue; 

mod add_nodes;
mod add_relationships;
mod get_attributes;
mod get_schema;
mod navigate_graph;

#[pyclass]
pub struct KnowledgeGraph {
    pub graph: DiGraph<Node, Relation>,
}

#[pymethods]
impl KnowledgeGraph {
    #[new]
    pub fn new() -> Self {
        KnowledgeGraph {
            graph: DiGraph::new(),
        }
    }

    // Method to add a single node
    pub fn add_node(
        &mut self, node_type: String, unique_id: String,  attributes: Option<HashMap<String, AttributeValue>>, node_title: Option<String>
    ) -> usize {
        let node = Node::new(&node_type, &unique_id, attributes, node_title.as_deref());
        let index = self.graph.add_node(node);
        index.index() // Convert NodeIndex to usize before returning
    }

    // Add nodes to graph
    pub fn add_nodes(
        &mut self, data: &PyList, columns: Vec<String>, node_type: String, unique_id_field: String, node_title_field: Option<String>, 
        conflict_handling: Option<String>, column_types: Option<&PyDict>,
    ) -> PyResult<Vec<usize>> {
        add_nodes::add_nodes(
            &mut self.graph, 
            data,
            columns,
            node_type,
            unique_id_field,
            node_title_field,
            conflict_handling,
            column_types,
        ) // Call the standalone function
    }

    // Add relationships to graph
    pub fn add_relationships(
        &mut self, data: &PyList, columns: Vec<String>, relationship_type: String, source_type: String, source_id_field: String, 
        target_type: String, target_id_field: String, source_title_field: Option<String>, target_title_field: Option<String>, conflict_handling: Option<String>,
    ) -> PyResult<Vec<(usize, usize)>> {
        add_relationships::add_relationships(
            &mut self.graph,
            data,
            columns,
            relationship_type,
            source_type,
            source_id_field,
            target_type,            
            target_id_field,
            source_title_field,
            target_title_field,
            conflict_handling,
        )
    }
    // Get attributes from nodes
    pub fn get_node_attributes(
        &mut self, py: Python, indices: Vec<usize>, specified_attributes: Option<Vec<String>>, max_relations: Option<usize>,
    ) -> PyResult<PyObject> {
        get_attributes::get_node_attributes(
            &mut self.graph, 
            py,
            indices,
            specified_attributes,
            max_relations,
        )
    }

    // Navigate the graph
    pub fn get_nodes(
        &mut self, attribute_key: &str, attribute_value: &str, filter_node_type: Option<&str>,
    ) -> Vec<usize> {
        navigate_graph::get_nodes(
            &mut self.graph, 
            attribute_key,
            attribute_value,
            filter_node_type
        )
    }
    pub fn get_relationships(
        &mut self, py: Python, indices: Vec<usize>,
    ) -> PyResult<PyObject> {
        navigate_graph::get_relationships(
            &mut self.graph, 
            py,
            indices
        )
    }
    pub fn traverse_incoming(&self, indices: Vec<usize>, relationship_type: String, sort_attribute: Option<&str>, ascending: Option<bool>, max_relations: Option<usize>) -> Vec<usize> {
        navigate_graph::traverse_nodes(&self.graph, indices, relationship_type, true, sort_attribute, ascending, max_relations)
    }
    pub fn traverse_outgoing(&self, indices: Vec<usize>, relationship_type: String, sort_attribute: Option<&str>, ascending: Option<bool>, max_relations: Option<usize>) -> Vec<usize> {
        navigate_graph::traverse_nodes(&self.graph, indices, relationship_type, false, sort_attribute, ascending, max_relations)
    }
    

    

    // Additional methods as needed...
}
