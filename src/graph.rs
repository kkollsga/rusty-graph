use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use pyo3::PyResult;
use pyo3::exceptions::{PyIOError, PyValueError};
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, BufReader};
use std::sync::{Arc, RwLock};

use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue;
use traversal_functions::{TraversalContext, traverse_relationships, process_traversal_levels, process_attributes_levels, count_traversal_levels, store_traversal_values};

mod types;
mod add_nodes;
mod add_relationships;
mod get_schema;
mod query_functions;
mod traversal_functions;

use types::DataInput;
use query_functions::extract_dataframe_content;

#[derive(Debug)]
enum SortSetting {
   Attribute(String),
   AttributeWithOrder(String, bool)
}

#[pyclass]
#[derive(Clone)]
pub struct KnowledgeGraph {
    graph: Arc<RwLock<DiGraph<Node, Relation>>>,
    traversal_context: Arc<RwLock<TraversalContext>>,
}

impl KnowledgeGraph {
    fn new_impl() -> Self {
        KnowledgeGraph {
            graph: Arc::new(RwLock::new(DiGraph::new())),
            traversal_context: Arc::new(RwLock::new(TraversalContext::new_base())),
        }
    }

    // Helper to safely get graph
    fn get_graph(&self) -> PyResult<std::sync::RwLockReadGuard<DiGraph<Node, Relation>>> {
        self.graph.read()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire read lock on graph"))
    }

    // Helper to safely get mutable graph
    fn get_graph_mut(&self) -> PyResult<std::sync::RwLockWriteGuard<DiGraph<Node, Relation>>> {
        self.graph.write()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire write lock on graph"))
    }

    // Helper to safely get context
    fn get_context(&self) -> PyResult<TraversalContext> {
        self.traversal_context.read()
            .map(|guard| (*guard).clone())
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire read lock"))
    }

    // Helper to safely update context
    fn update_context(&self, new_context: TraversalContext) -> PyResult<()> {
        let mut guard = self.traversal_context.write()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire write lock"))?;
        *guard = new_context;
        Ok(())
    }

    // Helper to get nodes with empty check
    fn get_context_nodes(&self) -> PyResult<Vec<usize>> {
        let context = self.get_context()?;
        let graph = self.get_graph()?;
        Ok(if context.is_empty() {
            graph.node_indices().map(|n| n.index()).collect()
        } else {
            context.current_nodes()
        })
    }

    fn process_input_data(input: &PyAny) -> PyResult<DataInput> {
        if let Ok(true) = input.getattr("__class__")?
            .getattr("__name__")?
            .extract::<String>()
            .map(|x| x == "DataFrame") 
        {
            return extract_dataframe_content(input);
        }
    
        if let Ok(true) = input.getattr("__class__")?
            .getattr("__name__")?
            .extract::<String>()
            .map(|x| x == "ndarray")
        {
            let values = input.call_method0("tolist")?;
            let columns = input.getattr("dtype")?.getattr("names")?.extract()?;
            
            return Ok(DataInput {
                data: values.downcast::<PyList>()?.into(),
                columns
            });
        }
        
        let data = input.downcast::<PyList>()?;
        let columns: Vec<String> = if let Ok(cols) = data.get_item(0)?.call_method0("keys") {
            cols.extract()?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Data must be a pandas DataFrame, NumPy array, or list of dicts"
            ));
        };
        
        Ok(DataInput {
            data: data.into(),
            columns
        })
    }
}

#[pymethods]
impl KnowledgeGraph {
    #[new]
    pub fn new() -> Self {
        Self::new_impl()
    }

    pub fn reset(&self) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        self.update_context(TraversalContext::new_base())?;
        Py::new(py, self.clone())
    }

    pub fn filter(&self, filter_dict: &PyDict) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let graph = self.get_graph()?;
        let filtered_nodes = query_functions::filter_nodes(&graph, None, filter_dict)?;
        let mut context = TraversalContext::new_base();
        context.add_level(filtered_nodes, None, None);
        self.update_context(context)?;
        Py::new(py, self.clone())
    }
    
    pub fn type_filter(&self, node_type: String) -> PyResult<Py<Self>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let filter_dict = PyDict::new(py);
        let type_condition = PyDict::new(py);
        type_condition.set_item("==", node_type)?;
        filter_dict.set_item("node_type", type_condition)?;
        let graph = self.get_graph()?;
        let filtered_nodes = query_functions::filter_nodes(&graph, None, &filter_dict)?;
        let mut context = TraversalContext::new_base();
        context.add_level(filtered_nodes, None, None);
        self.update_context(context)?;
        Py::new(py, self.clone())
    }

    pub fn select_nodes(&self, node_indices: Vec<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let mut context = TraversalContext::new_base();
        context.add_level(node_indices, None, None);
        self.update_context(context)?;
        Py::new(py, self.clone())
    }

    pub fn traverse(&self, rel_type: String, filter: Option<&PyDict>, direction: Option<String>) -> PyResult<Py<Self>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let current_context = self.get_context()?;
        let graph = self.get_graph()?;
        
        let start_nodes = current_context.current_nodes();
        
        if start_nodes.is_empty() {
            return Py::new(py, self.clone());
        }
    
        let relationships = traverse_relationships(&graph, &start_nodes, rel_type.clone(), direction, filter)?;
        
        let mut new_context = current_context;
        let mut all_target_nodes = Vec::new();
        for targets in relationships.values() {
            for &target in targets {
                if !all_target_nodes.contains(&target) {
                    all_target_nodes.push(target);
                }
            }
        }
        new_context.add_level(all_target_nodes, None, Some(rel_type));
        new_context.add_relationships(relationships);
        
        self.update_context(new_context)?;
        Py::new(py, self.clone())
    }

    pub fn sort(&self, sort_attribute: &str, ascending: Option<bool>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let mut context = self.get_context()?;
        let graph = self.get_graph()?;
        
        if let Some(last_level) = context.levels.last_mut() {
            let sorted_nodes = query_functions::sort_nodes(&graph, last_level.nodes.clone(), |&a, &b| {
                let compare_values = |idx: usize| {
                    let node = graph.node_weight(petgraph::graph::NodeIndex::new(idx));
                    
                    match node {
                        Some(Node::StandardNode { attributes, .. }) => {
                            match sort_attribute {
                                "title" => node.and_then(|n| match n {
                                    Node::StandardNode { title, .. } => title.clone().map(AttributeValue::String),
                                    _ => None
                                }),
                                _ => attributes.get(sort_attribute).cloned()
                            }
                        },
                        _ => None
                    }
                };
        
                let a_val = compare_values(a);
                let b_val = compare_values(b);
        
                let ordering = match (a_val, b_val) {
                    (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                };
                
                if !ascending.unwrap_or(true) {
                    ordering.reverse()
                } else {
                    ordering
                }
            });
    
            last_level.nodes = sorted_nodes;
        }
        
        self.update_context(context)?;
        Py::new(py, self.clone())
    }

    pub fn sort_by(&self, sort_settings: Vec<PyObject>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let settings = sort_settings.iter().map(|setting| {
            if let Ok(attr) = setting.extract::<String>(py) {
                Ok(SortSetting::Attribute(attr))
            } else if let Ok(tuple) = setting.extract::<Vec<PyObject>>(py) {
                if tuple.len() == 2 {
                    let attr = tuple[0].extract::<String>(py)?;
                    let ascending = tuple[1].extract::<bool>(py)?;
                    Ok(SortSetting::AttributeWithOrder(attr, ascending))
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Sort tuple must contain [attribute, ascending]"
                    ))
                }
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Sort settings must be either string or [attribute, ascending] tuple"
                ))
            }
        }).collect::<PyResult<Vec<_>>>()?;

        let nodes = self.get_context_nodes()?;
        let graph = self.get_graph()?;
        
        let sorted_nodes = query_functions::sort_nodes(&graph, nodes, |&a, &b| {
            for setting in &settings {
                let (sort_attribute, ascending) = match setting {
                    SortSetting::Attribute(attr) => (attr, true),
                    SortSetting::AttributeWithOrder(attr, asc) => (attr, *asc),
                };

                let compare_values = |idx: usize| {
                    let node = graph.node_weight(petgraph::graph::NodeIndex::new(idx));
                    
                    match node {
                        Some(Node::StandardNode { attributes, .. }) => {
                            match sort_attribute.as_str() {
                                "title" => node.and_then(|n| match n {
                                    Node::StandardNode { title, .. } => title.clone().map(AttributeValue::String),
                                    _ => None
                                }),
                                _ => attributes.get(sort_attribute).cloned()
                            }
                        },
                        _ => None
                    }
                };

                let a_val = compare_values(a);
                let b_val = compare_values(b);

                let ordering = match (a_val, b_val) {
                    (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                };

                if ordering != std::cmp::Ordering::Equal {
                    return if ascending { ordering } else { ordering.reverse() };
                }
            }
            std::cmp::Ordering::Equal
        });
        
        self.update_context(TraversalContext::new_with_nodes(sorted_nodes))?;
        Py::new(py, self.clone())
    }

    pub fn get_attributes(&self, _py: Python, max_results: Option<usize>, attributes: Option<Vec<String>>) -> PyResult<PyObject> {
        let graph = self.get_graph()?;
        process_attributes_levels(&graph, &self.get_context()?, attributes, max_results)
    }

    pub fn get_id(&self, max_results: Option<usize>) -> PyResult<PyObject> {
        let graph = self.get_graph()?;
        process_traversal_levels(&graph, &self.get_context()?, "unique_id", max_results)
    }

    pub fn get_title(&self, max_results: Option<usize>) -> PyResult<PyObject> {
        let graph = self.get_graph()?;
        process_traversal_levels(&graph, &self.get_context()?, "title", max_results)
    }

    pub fn get_index(&self, max_results: Option<usize>) -> PyResult<PyObject> {
        let graph = self.get_graph()?;
        process_traversal_levels(&graph, &self.get_context()?, "graph_index", max_results)
    }

    pub fn count(&self) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let context = self.get_context()?;
        let counts = count_traversal_levels(&context, None)?;
        
        Python::with_gil(|py| -> PyResult<()> {
            let mut guard = self.traversal_context.write()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire write lock"))?;
            guard.results = Some(counts.into_py(py));
            Ok(())
        })?;
        
        Py::new(py, self.clone())
    }

    pub fn get_results(&self) -> PyResult<PyObject> {
        let context = self.get_context()?;
        match &context.results {
            Some(results) => Ok(results.clone()),
            None => Python::with_gil(|py| Ok(py.None()))
        }
    }

    pub fn store(&self, attribute_name: String) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let context = self.get_context()?;
        let mut graph = self.get_graph_mut()?;
        
        match &context.results {
            Some(results) => {
                store_traversal_values(&mut graph, &context, &attribute_name, results)?;
                Py::new(py, self.clone())
            },
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No results available to store. Call count() first."
            ))
        }
    }

    pub fn get_relationships(&self, py: Python, max_results: Option<usize>) -> PyResult<PyObject> {
        let mut indices = self.get_context_nodes()?;
        let graph = self.get_graph()?;
    
        if let Some(limit) = max_results {
            if limit == 0 {
                return Err(PyValueError::new_err("max_results must be positive"));
            }
            indices.truncate(limit);
        }
    
        let results = query_functions::get_simplified_relationships(&graph, indices)?;
        Ok(results.into_py(py))
    }

    pub fn add_node(
        &self,
        node_type: String,
        unique_id: &PyAny,
        attributes: Option<HashMap<String, AttributeValue>>,
        node_title: Option<String>
    ) -> PyResult<Option<usize>> {
        if let Ok(id) = unique_id.extract::<i32>() {
            Ok(Some(self.add_node_impl(node_type, id, attributes, node_title)))
        } else if let Ok(id) = unique_id.extract::<i64>() {
            Ok(Some(self.add_node_impl(node_type, id as i32, attributes, node_title)))
        } else if let Ok(id) = unique_id.extract::<f64>() {
            Ok(Some(self.add_node_impl(node_type, id as i32, attributes, node_title)))
        } else {
            Ok(None)
        }
    }

    fn add_node_impl(
        &self,
        node_type: String,
        unique_id: i32,
        attributes: Option<HashMap<String, AttributeValue>>,
        node_title: Option<String>
    ) -> usize {
        let node = Node::new(&node_type, unique_id, attributes, node_title.as_deref());
        let mut graph = self.get_graph_mut().expect("Failed to acquire write lock");
        let index = graph.add_node(node);
        index.index()
    }

    pub fn add_nodes(
        &self,
        data: &PyAny,
        node_type: String,
        unique_id_field: &PyAny,
        node_title_field: Option<String>,
        conflict_handling: Option<String>,
        column_types: Option<&PyDict>,
    ) -> PyResult<Vec<usize>> {
        let input = Self::process_input_data(data)?;
        let mut graph = self.get_graph_mut()?;
        
        add_nodes::add_nodes(
            &mut graph,
            input.data.as_ref(data.py()),
            input.columns,
            node_type,
            unique_id_field.extract()?,
            node_title_field,
            conflict_handling,
            column_types,
        )
    }

    pub fn add_relationships(
        &self,
        data: &PyAny,
        relationship_type: String,
        source_type: String,
        source_id_field: &PyAny,
        target_type: String,
        target_id_field: &PyAny,
        source_title_field: Option<String>,
        target_title_field: Option<String>,
        attribute_columns: Option<Vec<String>>,
        conflict_handling: Option<String>,
    ) -> PyResult<Vec<(usize, usize)>> {
        let source_id_field = source_id_field.extract::<String>()?;
        let target_id_field = target_id_field.extract::<String>()?;
        let input = Self::process_input_data(data)?;
        let mut graph = self.get_graph_mut()?;
        
        add_relationships::add_relationships(
            &mut graph,
            data,
            input.columns,
            relationship_type,
            source_type,
            source_id_field,
            target_type,
            target_id_field,
            source_title_field,
            target_title_field,
            attribute_columns,
            conflict_handling,
        )
    }

    pub fn save_to_file(&self, file_path: &str) -> PyResult<()> {
        let file = File::create(file_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let writer = BufWriter::new(file);
        let graph = self.get_graph()?;
        bincode::serialize_into(writer, &*graph)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    pub fn load_from_file(&self, file_path: &str) -> PyResult<()> {
        let file = File::open(file_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let reader = BufReader::new(file);
        match bincode::deserialize_from(reader) {
            Ok(graph) => {
                let mut graph_guard = self.get_graph_mut()?;
                *graph_guard = graph;
                self.update_context(TraversalContext::new_base())?;
                Ok(())
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))
        }
    }

    pub fn get_schema(&self, py: Python) -> PyResult<PyObject> {
        let graph = self.get_graph()?;
        get_schema::get_schema(py, &graph)
    }
}