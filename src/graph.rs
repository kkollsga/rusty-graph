use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use pyo3::PyResult;
use pyo3::exceptions::{PyIOError, PyValueError};
use petgraph::graph::{DiGraph, NodeIndex}; 
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, BufReader};
use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue;

mod types;
mod add_nodes;
mod add_relationships;
mod get_schema;
mod query_functions;

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
    pub graph: DiGraph<Node, Relation>,
    selected_nodes: Vec<usize>,
}

impl KnowledgeGraph {
    fn new_impl() -> Self {
        KnowledgeGraph {
            graph: DiGraph::new(),
            selected_nodes: Vec::new(),
        }
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

    pub fn filter(&mut self, filter_dict: &PyDict) -> PyResult<Self> {
        self.selected_nodes = query_functions::filter_nodes(&self.graph, None, filter_dict)?;
        Ok(self.clone())
    }

    pub fn type_filter(&mut self, node_type: String) -> PyResult<Self> {
        let py = unsafe { Python::assume_gil_acquired() };
        let filter_dict = PyDict::new(py);
        filter_dict.set_item("node_type", node_type)?;
        self.selected_nodes = query_functions::filter_nodes(&self.graph, None, filter_dict)?;
        Ok(self.clone())
    }

    pub fn select_nodes(&mut self, node_indices: Vec<usize>) -> Self {
        self.selected_nodes = node_indices;
        self.clone()
    }

    pub fn traverse_in(
        &mut self,
        relationship_type: String,
        sort_attribute: Option<String>,
        ascending: Option<bool>,
        max_relations: Option<usize>,
    ) -> Self {
        let indices = if self.selected_nodes.is_empty() {
            self.graph.node_indices().map(|n| n.index()).collect()
        } else {
            self.selected_nodes.clone()
        };

        self.selected_nodes = query_functions::traverse_relationships(
            &self.graph,
            indices,
            &relationship_type,
            true,
            sort_attribute.as_deref(),
            ascending,
            max_relations,
        );
        self.clone()
    }

    pub fn traverse_out(
        &mut self,
        relationship_type: String,
        sort_attribute: Option<String>,
        ascending: Option<bool>,
        max_relations: Option<usize>,
    ) -> Self {
        let indices = if self.selected_nodes.is_empty() {
            self.graph.node_indices().map(|n| n.index()).collect()
        } else {
            self.selected_nodes.clone()
        };

        self.selected_nodes = query_functions::traverse_relationships(
            &self.graph,
            indices,
            &relationship_type,
            false,
            sort_attribute.as_deref(),
            ascending,
            max_relations,
        );
        self.clone()
    }

    pub fn sort(&mut self, sort_attribute: &str, ascending: Option<bool>) -> PyResult<Self> {
        self.selected_nodes = query_functions::sort_nodes(&self.graph, self.selected_nodes.clone(), |&a, &b| {
            let compare_values = |idx: usize| {
                let node = self.graph.node_weight(petgraph::graph::NodeIndex::new(idx));
                
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
        Ok(self.clone())
    }

    pub fn sort_by(&mut self, sort_settings: Vec<PyObject>) -> PyResult<Self> {
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

        self.selected_nodes = query_functions::sort_nodes(&self.graph, self.selected_nodes.clone(), |&a, &b| {
            for setting in &settings {
                let (sort_attribute, ascending) = match setting {
                    SortSetting::Attribute(attr) => (attr, true),
                    SortSetting::AttributeWithOrder(attr, asc) => (attr, *asc),
                };

                let compare_values = |idx: usize| {
                    let node = self.graph.node_weight(petgraph::graph::NodeIndex::new(idx));
                    
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
        
        Ok(self.clone())
    }

    pub fn get_attributes(
        &self,
        py: Python,
        attributes: Option<Vec<String>>,
        max_results: Option<usize>
    ) -> PyResult<PyObject> {
        let mut indices = if self.selected_nodes.is_empty() {
            self.graph.node_indices().map(|n| n.index()).collect()
        } else {
            self.selected_nodes.clone()
        };
        if let Some(limit) = max_results {
            if limit == 0 {
                return Err(PyValueError::new_err("max_results must be positive"));
            }
            indices.truncate(limit);
        }
        let data = query_functions::get_node_data(&self.graph, indices, attributes)?;
        Ok(data.into_py(py))
    }

    pub fn get_title(&self, py: Python, max_results: Option<usize>) -> PyResult<PyObject> {
        let mut indices = if self.selected_nodes.is_empty() {
            self.graph.node_indices().map(|n| n.index()).collect()
        } else {
            self.selected_nodes.clone()
        };
        if let Some(limit) = max_results {
            if limit == 0 {
                return Err(PyValueError::new_err("max_results must be positive"));
            }
            indices.truncate(limit);
        }
        let data = query_functions::get_simple_node_data(&self.graph, indices, "title")?;
        Ok(data.into_py(py))
    }
    
    pub fn get_id(&self, py: Python, max_results: Option<usize>) -> PyResult<PyObject> {
        let mut indices = if self.selected_nodes.is_empty() {
            self.graph.node_indices().map(|n| n.index()).collect()
        } else {
            self.selected_nodes.clone()
        };
        if let Some(limit) = max_results {
            if limit == 0 {
                return Err(PyValueError::new_err("max_results must be positive"));
            }
            indices.truncate(limit);
        }
        let data = query_functions::get_simple_node_data(&self.graph, indices, "unique_id")?;
        Ok(data.into_py(py))
    }

    pub fn get_index(&self, py: Python, max_results: Option<usize>) -> PyResult<PyObject> {
        let mut indices = if self.selected_nodes.is_empty() {
            self.graph.node_indices().map(|n| n.index()).collect()
        } else {
            self.selected_nodes.clone()
        };
        if let Some(limit) = max_results {
            if limit == 0 {
                return Err(PyValueError::new_err("max_results must be positive"));
            }
            indices.truncate(limit);
        }
        Ok(indices.into_py(py))
    }

    pub fn get_relationships(&self, py: Python, max_results: Option<usize>) -> PyResult<PyObject> {
        let mut indices = if self.selected_nodes.is_empty() {
            self.graph.node_indices().map(|n| n.index()).collect()
        } else {
            self.selected_nodes.clone()
        };
    
        if let Some(limit) = max_results {
            if limit == 0 {
                return Err(PyValueError::new_err("max_results must be positive"));
            }
            indices.truncate(limit);
        }
    
        let results = query_functions::get_simplified_relationships(&self.graph, indices)?;
        Ok(results.into_py(py))
    }

    pub fn add_node(
        &mut self,
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
        data: &PyAny,
        node_type: String,
        unique_id_field: &PyAny,
        node_title_field: Option<String>,
        conflict_handling: Option<String>,
        column_types: Option<&PyDict>,
    ) -> PyResult<Vec<usize>> {
        let input = Self::process_input_data(data)?;
        
        add_nodes::add_nodes(
            &mut self.graph,
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
        &mut self,
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
        
        add_relationships::add_relationships(
            &mut self.graph,
            data,
            input.columns,  // Pass the columns from the processed input
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