use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use pyo3::PyResult;
use serde_json::Value as JsonValue;
use pyo3::exceptions::{PyIOError, PyValueError};
use petgraph::graph::DiGraph;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, BufReader};
use std::sync::{Arc, RwLock};

use crate::schema::{Node, Relation};
use crate::data_types::AttributeValue;
use crate::graph::traversal_functions::{TraversalContext, traverse_relationships, process_traversal_levels, process_node_levels, count_traversal_levels};
use crate::graph::calculation_functions::{calculate_aggregate, store_calculation_values, process_calculation_levels};

mod types;
mod add_nodes;
mod add_relationships;
mod get_schema;
mod query_functions;
mod traversal_functions;
mod calculation_functions;  // Add the new module

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

    fn get_context_value(&self) -> PyResult<TraversalContext> {
        self.get_context().map(|guard| (*guard).clone())
    }

    fn get_graph(&self) -> PyResult<std::sync::RwLockReadGuard<DiGraph<Node, Relation>>> {
        self.graph.read()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire read lock on graph"))
    }

    fn get_graph_mut(&self) -> PyResult<std::sync::RwLockWriteGuard<DiGraph<Node, Relation>>> {
        self.graph.write()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire write lock on graph"))
    }

    fn get_context(&self) -> PyResult<std::sync::RwLockReadGuard<TraversalContext>> {
        self.traversal_context.read()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire read lock on traversal context"))
    }

    fn get_context_mut(&self) -> PyResult<std::sync::RwLockWriteGuard<TraversalContext>> {
        self.traversal_context.write()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire write lock on traversal context"))
    }

    fn get_context_nodes(&self) -> PyResult<Vec<usize>> {
        let context = self.get_context()?;
        let graph = self.get_graph()?;
        Ok(if context.is_empty() {
            graph.node_indices().map(|n| n.index()).collect()
        } else {
            context.current_nodes()
        })
    }

    fn update_context(&self, new_context: TraversalContext) -> PyResult<()> {
        let mut context = self.traversal_context.write().map_err(|_| {
            PyValueError::new_err("Failed to acquire write lock on traversal context")
        })?;
        *context = new_context;
        Ok(())
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

    pub fn add_node(
        &self,
        node_type: String,
        unique_id: &PyAny,
        attributes: Option<HashMap<String, AttributeValue>>,
        node_title: Option<String>,
        attribute_columns: Option<Vec<String>>,
    ) -> PyResult<()> {  // Changed return type
        let filtered_attributes = match (attributes, attribute_columns) {
            (Some(attrs), Some(cols)) => {
                let filtered: HashMap<String, AttributeValue> = attrs.into_iter()
                    .filter(|(key, _)| cols.iter().any(|col| col.eq_ignore_ascii_case(key)))
                    .collect();
                Some(filtered)
            },
            (attrs, None) => attrs,
            (None, _) => None,
        };

        if let Ok(id) = unique_id.extract::<i32>() {
            self.add_node_impl(node_type, id, filtered_attributes, node_title);
            Ok(())
        } else if let Ok(id) = unique_id.extract::<i64>() {
            self.add_node_impl(node_type, id as i32, filtered_attributes, node_title);
            Ok(())
        } else if let Ok(id) = unique_id.extract::<f64>() {
            self.add_node_impl(node_type, id as i32, filtered_attributes, node_title);
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid unique_id type"))
        }
    }
    
    fn add_node_impl(
        &self,
        node_type: String,
        unique_id: i32,
        attributes: Option<HashMap<String, AttributeValue>>,
        node_title: Option<String>
    ) {  // Changed return type to ()
        let node = Node::new(&node_type, unique_id, attributes, node_title.as_deref());
        let mut graph = self.get_graph_mut().expect("Failed to acquire write lock");
        graph.add_node(node);
    }
    
    pub fn add_nodes(
        &self,
        data: &PyAny,
        node_type: String,
        unique_id_field: &PyAny,
        node_title_field: Option<String>,
        conflict_handling: Option<String>,
        column_types: Option<&PyDict>,
        attribute_columns: Option<Vec<String>>,
    ) -> PyResult<()> {  // Changed return type
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
            attribute_columns,
        )?;  // Added ? to propagate error
        
        Ok(())
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
    ) -> PyResult<()> {  // Changed return type
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
        )?;  // Added ? to propagate error
        
        Ok(())
    }

    pub fn reset(&self) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let mut context = self.get_context_mut()?;
        
        let mut params = HashMap::new();
        params.insert("function".to_string(), JsonValue::String("reset".to_string()));
        
        context.levels.clear();  // Fix: Access levels directly
        context.add_level(Vec::new(), Some(vec![params]), true);
        
        Py::new(py, self.clone())
    }

    pub fn print_context(&self, _py: Python) -> PyResult<String> {
        let context = self.get_context()?;
        let mut output = String::new();
        
        if context.levels.is_empty() {
            output.push_str("Empty traversal context\n");
            return Ok(output);
        }
    
        for (level_idx, level) in context.levels.iter().enumerate() {
            output.push_str(&format!("Level {}: {} nodes\n", level_idx, level.nodes.len()));
            
            if let Some(params) = &level.selection_params {
                output.push_str("  Selection parameters: \n");
                let param_strings: Vec<String> = params.iter()
                    .map(|param_map| {
                        format!("    {}", serde_json::to_string(param_map).unwrap_or_default())
                    })
                    .collect();
                output.push_str(&param_strings.join(",\n"));
                output.push_str("\n");
            }
            
            output.push_str("  Nodes: [");
            output.push_str(&level.nodes.iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", "));
            output.push_str("]\n");
            
            if !level.node_relationships.is_empty() {
                output.push_str("  Relationships:\n");
                for (source, targets) in &level.node_relationships {
                    if !targets.is_empty() {
                        output.push_str(&format!("    {} -> [", source));
                        output.push_str(&targets.iter()
                            .map(|t| t.to_string())
                            .collect::<Vec<_>>()
                            .join(", "));
                        output.push_str("]\n");
                    }
                }
            }
            
            output.push_str("\n");
        }
    
        if let Some(results) = &context.results {
            output.push_str("Results:\n  "); // Combined "Results:" with newline and indent
            Python::with_gil(|py| {
                output.push_str(&match results.as_ref(py) {
                    x if x.str().is_ok() => x.str()?.extract::<String>()?,
                    x if x.downcast::<PyDict>().is_ok() => format!("{:?}", x.downcast::<PyDict>()?),
                    x if x.downcast::<PyList>().is_ok() => format!("{:?}", x.downcast::<PyList>()?),
                    _ => "[Complex result type]".to_string()
                });
                Ok::<_, PyErr>(())
            })?;
            output.push_str("\n");
        }
    
        Ok(output)
    }

    pub fn filter(&self, filter_dict: &PyDict, max_nodes: Option<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let graph = self.get_graph()?;
        
        // Validate max_nodes if provided
        if let Some(limit) = max_nodes {
            if limit == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "max_nodes must be positive"
                ));
            }
        }
        
        let mut filtered_nodes = query_functions::filter_nodes(&graph, None, filter_dict)?;
        
        // Apply max_nodes limit if specified
        if let Some(limit) = max_nodes {
            filtered_nodes.truncate(limit);
        }
        
        let mut params = HashMap::new();
        params.insert("function".to_string(), JsonValue::String("filter".to_string()));
        params.insert("filter".to_string(), pydict_to_json(filter_dict)?);
        if let Some(limit) = max_nodes {
            params.insert("max_nodes".to_string(), JsonValue::Number(limit.into()));
        }
        
        // Create new context as an owned value
        let mut new_context = TraversalContext::new_base();
        new_context.add_level(filtered_nodes, Some(vec![params]), true);
        
        // Update using the owned context
        self.update_context(new_context)?;
        
        Py::new(py, self.clone())
    }
    
    pub fn type_filter(&self, node_types: &PyAny, max_nodes: Option<usize>) -> PyResult<Py<Self>> {
        let py = unsafe { Python::assume_gil_acquired() };
        
        // Validate max_nodes if provided
        if let Some(limit) = max_nodes {
            if limit == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "max_nodes must be positive"
                ));
            }
        }
        
        let filter_dict = PyDict::new(py);
        let type_condition = PyDict::new(py);
        
        if let Ok(single_type) = node_types.extract::<String>() {
            type_condition.set_item("==", single_type)?;
        } else if let Ok(type_list) = node_types.extract::<Vec<String>>() {
            type_condition.set_item("==", type_list)?;
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "node_types must be either a string or a list of strings"
            ));
        }
        
        filter_dict.set_item("node_type", type_condition)?;
        let graph = self.get_graph()?;
        let mut filtered_nodes = query_functions::filter_nodes(&graph, None, &filter_dict)?;
        
        // Apply max_nodes limit if specified
        if let Some(limit) = max_nodes {
            filtered_nodes.truncate(limit);
        }
        
        let mut params = HashMap::new();
        params.insert("function".to_string(), JsonValue::String("type_filter".to_string()));
        params.insert("node_types".to_string(), pydict_to_json(&type_condition)?);
        if let Some(limit) = max_nodes {
            params.insert("max_nodes".to_string(), JsonValue::Number(limit.into()));
        }
        
        let mut context = TraversalContext::new_base();
        context.add_level(filtered_nodes, Some(vec![params]), true);
        self.update_context(context)?;
        Py::new(py, self.clone())
    }

    pub fn select_nodes(&self, node_indices: Vec<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        
        let mut params = HashMap::new();
        params.insert("function".to_string(), JsonValue::String("select_nodes".to_string()));
        params.insert("node_indices".to_string(), JsonValue::Array(
            node_indices.iter().map(|&idx| JsonValue::Number(idx.into())).collect()
        ));
        
        let mut context = TraversalContext::new_base();
        context.add_level(node_indices, Some(vec![params]), true);
        self.update_context(context)?;
        Py::new(py, self.clone())
    }
    

    pub fn traverse(
        &self,
        rel_type: String,
        filter: Option<&PyDict>,
        direction: Option<String>,
        sort: Option<&PyDict>,
        max_traversals: Option<usize>,
        skip_level: Option<bool>,
     ) -> PyResult<Py<Self>> {
        let py = unsafe { Python::assume_gil_acquired() };
        
        // Get initial context and graph safely
        let current_context = {
            let ctx = self.get_context()?;
            (*ctx).clone()
        };
        let graph = self.get_graph()?;
        
        let start_nodes = current_context.current_nodes();
        
        if start_nodes.is_empty() {
            return Py::new(py, self.clone());
        }
     
        // Get the relationships for the traversal
        let relationships = traverse_relationships(&graph, &start_nodes, rel_type.clone(), direction.clone(), filter, sort, max_traversals)?;
     
        let all_target_nodes: Vec<usize> = relationships.values()
            .flat_map(|targets| targets.iter().copied())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
     
        // Build the parameters for the traversal
        let mut params = HashMap::new();
        params.insert("function".to_string(), JsonValue::String("traverse".to_string()));
        params.insert("rel_type".to_string(), JsonValue::String(rel_type));
        
        if let Some(ref d) = direction { 
            params.insert("direction".to_string(), JsonValue::String(d.clone())); 
        }
        if let Some(f) = filter { 
            params.insert("filter".to_string(), pydict_to_json(f)?);
        }
        if let Some(s) = sort { 
            params.insert("sort".to_string(), pydict_to_json(s)?);
        }
        if let Some(m) = max_traversals { 
            params.insert("max_traversals".to_string(), JsonValue::Number(m.into())); 
        }
        
        let new_context = if skip_level.unwrap_or(false) {
            // Safety check for skip-level
            if current_context.levels.len() < 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Cannot skip level with less than 2 levels of traversal"
                ));
            }
    
            // Preserve the initial context level
            let mut new_context = TraversalContext::new_base();
            new_context.levels.push(current_context.levels[0].clone());
            
            let level_0 = &current_context.levels[0].nodes;
            let level_1_rels = &current_context.levels[1].node_relationships;
            
            // Create the translated relationships
            let mut translated_relationships = HashMap::new();
            for &class_id in level_0 {
                let mut evaluated_students = Vec::new();
                if let Some(students) = level_1_rels.get(&class_id) {
                    for &student_id in students {
                        if let Some(student_subjects) = relationships.get(&student_id) {
                            evaluated_students.extend(student_subjects);
                        }
                    }
                }
                evaluated_students.sort_unstable();
                evaluated_students.dedup();
                translated_relationships.insert(class_id, evaluated_students);
            }
     
            params.insert("skip_level".to_string(), JsonValue::Bool(true));
    
            // Combine previous and new parameters
            let mut combined_params = current_context.levels.get(1)
                .and_then(|level| level.selection_params.clone())
                .unwrap_or_default();
            combined_params.push(params);
    
            // Add the final level with preserved history and combined parameters
            new_context.add_level(all_target_nodes, Some(combined_params), false);
            new_context.add_relationships(translated_relationships);
            new_context
        } else {
            let mut next_context = current_context;
            next_context.add_level(all_target_nodes, Some(vec![params]), false);
            next_context.add_relationships(relationships);
            next_context
        };
        
        self.update_context(new_context)?;
        
        Py::new(py, self.clone())
    }

    pub fn sort(&self, sort_attribute: &str, ascending: Option<bool>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let graph = self.get_graph()?;
        
        // Create new context from existing one
        let mut new_context = {
            let context = self.get_context()?;
            (*context).clone()
        };
    
        
        if let Some(last_level) = new_context.levels.last_mut() {
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
        
        self.update_context(new_context)?;
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

    pub fn get_nodes(
        &self,
        _py: Python,
        max_results: Option<usize>,
        attributes: Option<bool>,
        calculations: Option<bool>,
        connections: Option<bool>,
        only_return: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let graph = self.get_graph()?;
        let context = self.get_context_value()?;
        process_node_levels(&graph, &context, attributes.unwrap_or(true), calculations.unwrap_or(true), connections.unwrap_or(true), only_return.as_deref(), max_results)
    }

    pub fn get_calculations(&self, _py: Python, calculation_names: Option<Vec<String>>, max_results: Option<usize>) -> PyResult<PyObject> {
        let graph = self.get_graph()?;
        let context = self.get_context_value()?;
        process_calculation_levels(&graph, &context, calculation_names, max_results)
    }

    pub fn get_ids(&self, max_results: Option<usize>) -> PyResult<PyObject> {
        let graph = self.get_graph()?;
        let context = self.get_context_value()?;
        process_traversal_levels(&graph, &context, "unique_id", max_results)
    }

    pub fn get_titles(&self, max_results: Option<usize>) -> PyResult<PyObject> {
        let graph = self.get_graph()?;
        let context = self.get_context_value()?;
        process_traversal_levels(&graph, &context, "title", max_results)
    }

    pub fn get_index(&self, max_results: Option<usize>) -> PyResult<PyObject> {
        let graph = self.get_graph()?;
        let context = self.get_context_value()?;
        process_traversal_levels(&graph, &context, "graph_index", max_results)
    }

    pub fn count(&self) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        
        // Scope the read lock to ensure it's released
        let counts = {
            let context = self.get_context()?;
            count_traversal_levels(&context, None)?
        };
        // Update results with a separate write lock
        {
            let mut guard = self.traversal_context.write()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire write lock"))?;
            guard.results = Some(counts.into_py(py));
        }  // Write lock is explicitly dropped here
        let result = Py::new(py, self.clone());
        result
    }
    
    pub fn median(&self, attribute: String, max_results: Option<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let context = self.get_context()?;
        let graph = self.get_graph()?;
        let result = calculate_aggregate(&graph, &context, &attribute, "median", None, max_results)?;
        
        self.update_results(result)?;
        Py::new(py, self.clone())
    }

    pub fn mode(&self, attribute: String, max_results: Option<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let context = self.get_context()?;
        let graph = self.get_graph()?;
        let result = calculate_aggregate(&graph, &context, &attribute, "mode", None, max_results)?;
        
        self.update_results(result)?;
        Py::new(py, self.clone())
    }

    pub fn std(&self, attribute: String, max_results: Option<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        
        // Scope the locks to ensure they're released
        let result = {
            let context = self.get_context()?;
            let graph = self.get_graph()?;
            calculate_aggregate(&graph, &context, &attribute, "std", None, max_results)?
        };  // context and graph locks are released here
        
        // Explicitly update results in its own scope
        {
            let mut guard = self.traversal_context.write()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire write lock"))?;
            guard.results = Some(result.into_py(py));
        }  // write lock is released here
        
        Py::new(py, self.clone())
    }

    pub fn var(&self, attribute: String, max_results: Option<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let context = self.get_context()?;
        let graph = self.get_graph()?;
        let result = calculate_aggregate(&graph, &context, &attribute, "var", None, max_results)?;
        
        self.update_results(result)?;
        Py::new(py, self.clone())
    }

    pub fn sum(&self, attribute: String, max_results: Option<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let context = self.get_context()?;
        let graph = self.get_graph()?;
        let result = calculate_aggregate(&graph, &context, &attribute, "sum", None, max_results)?;
        
        self.update_results(result)?;
        Py::new(py, self.clone())
    }

    pub fn avg(&self, attribute: String, max_results: Option<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let context = self.get_context()?;
        let graph = self.get_graph()?;
        let result = calculate_aggregate(&graph, &context, &attribute, "avg", None, max_results)?;
        
        self.update_results(result)?;
        Py::new(py, self.clone())
    }

    pub fn min(&self, attribute: String, max_results: Option<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let context = self.get_context()?;
        let graph = self.get_graph()?;
        let result = calculate_aggregate(&graph, &context, &attribute, "min", None, max_results)?;
        
        self.update_results(result)?;
        Py::new(py, self.clone())
    }

    pub fn max(&self, attribute: String, max_results: Option<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let context = self.get_context()?;
        let graph = self.get_graph()?;
        let result = calculate_aggregate(&graph, &context, &attribute, "max", None, max_results)?;
        
        self.update_results(result)?;
        Py::new(py, self.clone())
    }

    pub fn quantile(&self, attribute: String, q: f64, max_results: Option<usize>) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let context = self.get_context()?;
        let graph = self.get_graph()?;
        let result = calculate_aggregate(&graph, &context, &attribute, "quantile", Some(q), max_results)?;
        
        self.update_results(result)?;
        Py::new(py, self.clone())
    }

    pub fn store(&self, calculation_name: String) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let context = self.get_context()?;
        let mut graph = self.get_graph_mut()?;
        
        match &context.results {
            Some(results) => {
                store_calculation_values(&mut graph, &context, &calculation_name, results)?;
                Py::new(py, self.clone())
            },
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No results available to store. Call an aggregation function first."
            ))
        }
    }

    pub fn has_calculation(&self, calculation_name: &str) -> PyResult<bool> {
        let graph = self.get_graph()?;
        let context = self.get_context()?;
        
        for node_idx in context.current_nodes() {
            if let Some(Node::StandardNode { calculations, .. }) = graph.node_weight(petgraph::graph::NodeIndex::new(node_idx)) {
                if let Some(calcs) = calculations {
                    if calcs.contains_key(calculation_name) {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    pub fn clear_calculations(&self) -> PyResult<Py<KnowledgeGraph>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let mut graph = self.get_graph_mut()?;
        let context = self.get_context()?;
        
        for node_idx in context.current_nodes() {
            if let Some(Node::StandardNode { calculations, .. }) = graph.node_weight_mut(petgraph::graph::NodeIndex::new(node_idx)) {
                *calculations = None;
            }
        }
        
        Py::new(py, self.clone())
    }

    pub fn get_results(&self) -> PyResult<PyObject> {
        let context = self.get_context()?;
        match &context.results {
            Some(results) => Ok(results.clone()),
            None => Python::with_gil(|py| Ok(py.None()))
        }
    }

    fn update_results(&self, result: PyObject) -> PyResult<()> {
        Python::with_gil(|py| -> PyResult<()> {
            let mut guard = self.traversal_context.write()
                .map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to acquire write lock")
                })?;
            guard.results = Some(result.into_py(py));
            Ok(())
        })
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

fn pydict_to_json(dict: &PyDict) -> PyResult<JsonValue> {
    let mut map = serde_json::Map::new();
    for (key, value) in dict.iter() {
        let key_str = key.extract::<String>()?;
        let json_value = match value.get_type().name()? {
            "str" => JsonValue::String(value.extract()?),
            "int" => JsonValue::Number(value.extract::<i64>()?.into()),
            "float" => {
                let f: f64 = value.extract()?;
                serde_json::Number::from_f64(f)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null)
            },
            "bool" => JsonValue::Bool(value.extract()?),
            "list" => {
                let py_list: Vec<String> = value.extract()?;
                JsonValue::Array(py_list.into_iter().map(JsonValue::String).collect())
            },
            "dict" => {
                let inner_dict = value.downcast::<PyDict>()?;
                pydict_to_json(inner_dict)?
            },
            _ => JsonValue::Null
        };
        map.insert(key_str, json_value);
    }
    Ok(JsonValue::Object(map))
}