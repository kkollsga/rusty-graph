use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use pyo3::PyResult;
use pyo3::exceptions::{PyIOError, PyValueError};
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, BufReader};
use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue;

mod add_nodes;
mod add_relationships;
mod get_attributes;
mod get_schema;
mod navigate_graph;
mod query;

#[pyclass]
pub struct GraphQuery {
    indices: Vec<usize>,
    graph: *mut DiGraph<Node, Relation>,
}

unsafe impl Send for GraphQuery {}

#[pymethods]
impl GraphQuery {
    fn select(&self, indices: Option<Vec<usize>>) -> PyResult<Self> {
        Ok(Self {
            indices: indices.unwrap_or_else(|| self.indices.clone()),
            graph: self.graph,
        })
    }

    fn filter(&self, filter_dict: &PyDict) -> PyResult<Self> {
        let filtered_indices = unsafe {
            query::query_nodes(&mut *self.graph, self.indices.clone(), Some(filter_dict))?
        };
        
        Ok(Self {
            indices: filtered_indices,
            graph: self.graph,
        })
    }

    fn traverse_in(&self, relationship_type: String) -> PyResult<Self> {
        let traversed_indices = unsafe {
            query::traverse_nodes_in(
                &*self.graph,
                self.indices.clone(),
                relationship_type,
            )
        };

        Ok(Self {
            indices: traversed_indices,
            graph: self.graph,
        })
    }

    fn traverse_out(&self, relationship_type: String) -> PyResult<Self> {
        let traversed_indices = unsafe {
            query::traverse_nodes_out(
                &*self.graph,
                self.indices.clone(),
                relationship_type,
            )
        };

        Ok(Self {
            indices: traversed_indices,
            graph: self.graph,
        })
    }

    fn get_attributes(&self, py: Python) -> PyResult<PyObject> {
        unsafe {
            get_attributes::get_node_attributes(
                &mut *self.graph,
                py,
                self.indices.clone(),
                None,
                None,
            )
        }
    }

    fn get_title(&self, py: Python) -> PyResult<PyObject> {
        unsafe {
            get_attributes::get_node_attributes(
                &mut *self.graph,
                py,
                self.indices.clone(),
                Some(vec!["title".to_string()]),
                None,
            )
        }
    }

    fn get_id(&self, py: Python) -> PyResult<PyObject> {
        unsafe {
            get_attributes::get_node_attributes(
                &mut *self.graph,
                py,
                self.indices.clone(),
                Some(vec!["unique_id".to_string()]),
                None,
            )
        }
    }
}

#[pyclass]
pub struct KnowledgeGraph {
    pub graph: DiGraph<Node, Relation>,
}

impl KnowledgeGraph {
    fn new_impl() -> Self {
        KnowledgeGraph {
            graph: DiGraph::new(),
        }
    }
}

fn parse_id_field<'a>(value: &'a PyAny) -> PyResult<Option<i32>> {
    if let Ok(s) = value.extract::<String>() {
        if let Ok(num) = s.parse::<i32>() {
            return Ok(Some(num));
        }
        if let Ok(num) = s.parse::<f64>() {
            return Ok(Some(num as i32));
        }
        return Ok(None);
    }
    
    if let Ok(num) = value.extract::<i32>() {
        return Ok(Some(num));
    }
    if let Ok(num) = value.extract::<f64>() {
        return Ok(Some(num as i32)); 
    }
    
    Ok(None)
}

#[pymethods]
impl KnowledgeGraph {
    #[new]
    pub fn new() -> Self {
        Self::new_impl()
    }

    pub fn query(&mut self) -> PyResult<GraphQuery> {
        Ok(GraphQuery {
            indices: Vec::new(),
            graph: &mut self.graph,
        })
    }

    pub fn add_node(
        &mut self,
        node_type: String,
        unique_id: &PyAny,
        attributes: Option<HashMap<String, AttributeValue>>,
        node_title: Option<String>
    ) -> PyResult<Option<usize>> {
        match parse_id_field(unique_id)? {
            Some(id) => Ok(Some(self.add_node_impl(node_type, id, attributes, node_title))),
            None => Ok(None)
        }
    }

    fn add_node_impl(
        &mut self,
        node_type: String,
        unique_id: i32,
        attributes: Option<HashMap<String, AttributeValue>>,
        node_title: Option<String>
    ) -> usize {
        let node = Node::new(&node_type, unique_id, attributes, node_title.as_deref());
        let index = self.graph.add_node(node);
        index.index()
    }

    pub fn add_nodes(
        &mut self,
        data: &PyList,
        columns: Vec<String>,
        node_type: String,
        unique_id_field: &PyAny,
        node_title_field: Option<String>,
        conflict_handling: Option<String>,
        column_types: Option<&PyDict>,
    ) -> PyResult<Vec<usize>> {
        let unique_id_field = unique_id_field.extract::<String>()?;
        
        add_nodes::add_nodes(
            &mut self.graph,
            data,
            columns,
            node_type,
            unique_id_field,
            node_title_field,
            conflict_handling,
            column_types,
        )
    }

    pub fn add_relationships(
        &mut self,
        data: &PyList,
        columns: Vec<String>,
        relationship_type: String,
        source_type: String,
        source_id_field: &PyAny,
        target_type: String,
        target_id_field: &PyAny,
        source_title_field: Option<String>,
        target_title_field: Option<String>,
    ) -> PyResult<Vec<(usize, usize)>> {
        let source_id_field = source_id_field.extract::<String>()?;
        let target_id_field = target_id_field.extract::<String>()?;
        
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
        )
    }

    pub fn get_node_attributes(
        &mut self,
        py: Python,
        indices: Vec<usize>,
        specified_attributes: Option<Vec<String>>,
        max_relations: Option<usize>,
    ) -> PyResult<PyObject> {
        get_attributes::get_node_attributes(
            &mut self.graph,
            py,
            indices,
            specified_attributes,
            max_relations,
        )
    }

    pub fn get_nodes(
        &mut self,
        node_type: Option<&str>,
        filters: Option<Vec<HashMap<String, String>>>,
    ) -> Vec<usize> {
        navigate_graph::get_nodes(
            &mut self.graph,
            node_type,
            filters
        )
    }

    pub fn get_relationships(
        &mut self,
        py: Python,
        indices: Vec<usize>,
    ) -> PyResult<PyObject> {
        navigate_graph::get_relationships(
            &mut self.graph,
            py,
            indices
        )
    }

    pub fn traverse_incoming(
        &self,
        indices: Vec<usize>,
        relationship_type: String,
        sort_attribute: Option<&str>,
        ascending: Option<bool>,
        max_relations: Option<usize>
    ) -> Vec<usize> {
        navigate_graph::traverse_nodes(
            &self.graph,
            indices,
            relationship_type,
            true,
            sort_attribute,
            ascending,
            max_relations
        )
    }

    pub fn traverse_outgoing(
        &self,
        indices: Vec<usize>,
        relationship_type: String,
        sort_attribute: Option<&str>,
        ascending: Option<bool>,
        max_relations: Option<usize>
    ) -> Vec<usize> {
        navigate_graph::traverse_nodes(
            &self.graph,
            indices,
            relationship_type,
            false,
            sort_attribute,
            ascending,
            max_relations
        )
    }
    
    pub fn save_to_file(&self, file_path: &str) -> PyResult<()> {
        let file = File::create(file_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &self.graph)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    pub fn load_from_file(&mut self, file_path: &str) -> PyResult<()> {
        let file = File::open(file_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let reader = BufReader::new(file);
        match bincode::deserialize_from(reader) {
            Ok(graph) => {
                self.graph = graph;
                Ok(())
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))
        }
    }

    pub fn get_schema(&self, py: Python) -> PyResult<PyObject> {
        get_schema::get_schema(py, &self.graph)
    }
}