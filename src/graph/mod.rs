// src/graph/mod.rs
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use std::collections::HashMap;
use std::sync::Arc;
use std::mem;
use crate::datatypes::{py_in, py_out};
use crate::datatypes::values::{Value, FilterCondition};
use crate::graph::io_operations::save_to_file;
use crate::graph::calculations::StatResult;
use crate::graph::reporting::{OperationReports, OperationReport};


pub mod maintain_graph;
pub mod filtering_methods;
pub mod traversal_methods;
pub mod statistics_methods;
pub mod io_operations;
pub mod lookups;
pub mod debugging;
pub mod calculations;
pub mod equation_parser;
pub mod batch_operations;
pub mod schema;
pub mod data_retrieval;
pub mod reporting;

use schema::{DirGraph, CurrentSelection};

#[pyclass]
pub struct KnowledgeGraph {
    inner: Arc<DirGraph>,
    selection: CurrentSelection,
    reports: OperationReports,  // Add reports field
}


impl Clone for KnowledgeGraph {
    fn clone(&self) -> Self {
        KnowledgeGraph {
            inner: Arc::clone(&self.inner),
            selection: self.selection.clone(),
            reports: self.reports.clone(),  // Clone reports as well
        }
    }
}

impl KnowledgeGraph {
    fn add_report(&mut self, report: OperationReport) -> usize {
        self.reports.add_report(report)
    }
}

/// Helper function to extract graph from Arc if possible or clone it
fn extract_or_clone_graph(arc: &mut Arc<DirGraph>) -> DirGraph {
    if Arc::strong_count(arc) == 1 {
        // We're the only owner, we can modify in place by extracting
        match Arc::try_unwrap(mem::replace(arc, Arc::new(DirGraph::new()))) {
            Ok(graph) => graph,
            Err(original_arc) => {
                // This shouldn't happen, but recover if it does
                *arc = original_arc;
                (**arc).clone()
            }
        }
    } else {
        // Multiple references exist, need to clone
        (**arc).clone()
    }
}

#[pymethods]
impl KnowledgeGraph {
    #[new]
    fn new() -> Self {
        KnowledgeGraph {
            inner: Arc::new(DirGraph::new()),
            selection: CurrentSelection::new(),
            reports: OperationReports::new(),  // Initialize reports
        }
    }

    fn add_nodes(
        &mut self,
        data: &Bound<'_, PyAny>,
        node_type: String,
        unique_id_field: String,
        node_title_field: Option<String>,
        columns: Option<&Bound<'_, PyList>>,
        conflict_handling: Option<String>,
        skip_columns: Option<&Bound<'_, PyList>>,
        column_types: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        // Get all columns from the dataframe
        let df_cols = data.getattr("columns")?;
        let all_columns: Vec<String> = df_cols.extract()?;
        
        // Create default columns array
        let mut default_cols = vec![unique_id_field.as_str()];
        if let Some(ref title_field) = node_title_field {
            default_cols.push(title_field);
        }
        
        // Use enforce_columns=false for add_nodes
        let enforce_columns = Some(false);
        
        // Get the filtered columns
        let column_list = py_in::ensure_columns(
            &all_columns,
            &default_cols,
            columns,
            skip_columns,
            enforce_columns,
        )?;
    
        let df_result = py_in::pandas_to_dataframe(
            data, 
            &[unique_id_field.clone()], 
            &column_list,
            column_types,
        )?;
    
        // Extract graph or clone it if needed
        let mut graph = extract_or_clone_graph(&mut self.inner);
        
        // Call the maintain_graph function
        let result = maintain_graph::add_nodes(
            &mut graph,
            df_result,
            node_type,
            unique_id_field,
            node_title_field,
            conflict_handling,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        // Replace the Arc with the new graph
        self.inner = Arc::new(graph);
        self.selection.clear();
        
        // Store the report
        self.add_report(OperationReport::NodeOperation(result.clone()));
        
        // Convert the report to a Python dictionary
        Python::with_gil(|py| {
            let report_dict = PyDict::new_bound(py);
            report_dict.set_item("operation", &result.operation_type)?;
            report_dict.set_item("timestamp", result.timestamp.to_rfc3339())?;
            report_dict.set_item("nodes_created", result.nodes_created)?;
            report_dict.set_item("nodes_updated", result.nodes_updated)?;
            report_dict.set_item("nodes_skipped", result.nodes_skipped)?;
            report_dict.set_item("processing_time_ms", result.processing_time_ms)?;
            
            // Add errors array if there are any
            if !result.errors.is_empty() {
                report_dict.set_item("errors", &result.errors)?;
                report_dict.set_item("has_errors", true)?;
            } else {
                report_dict.set_item("has_errors", false)?;
            }
            
            Ok(report_dict.into())
        })
    }

    fn add_connections(
        &mut self,
        data: &Bound<'_, PyAny>,
        connection_type: String,
        source_type: String,
        source_id_field: String,
        target_type: String,
        target_id_field: String,
        source_title_field: Option<String>,
        target_title_field: Option<String>,
        columns: Option<&Bound<'_, PyList>>,
        skip_columns: Option<&Bound<'_, PyList>>,
        conflict_handling: Option<String>,
        column_types: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        // Get all columns from the dataframe
        let df_cols = data.getattr("columns")?;
        let all_columns: Vec<String> = df_cols.extract()?;
        
        // Create default columns array
        let mut default_cols = vec![source_id_field.as_str(), target_id_field.as_str()];
        if let Some(ref src_title) = source_title_field {
            default_cols.push(src_title);
        }
        if let Some(ref tgt_title) = target_title_field {
            default_cols.push(tgt_title);
        }
        
        // Use enforce_columns=true for add_connections
        let enforce_columns = Some(true);
        
        // Get the filtered columns
        let column_list = py_in::ensure_columns(
            &all_columns,
            &default_cols,
            columns,
            skip_columns,
            enforce_columns,
        )?;
        
        let df_result = py_in::pandas_to_dataframe(
            data,
            &[source_id_field.clone(), target_id_field.clone()],
            &column_list,
            column_types,
        )?;
    
        // Extract graph or clone it if needed
        let mut graph = extract_or_clone_graph(&mut self.inner);
        
        let result = maintain_graph::add_connections(
            &mut graph,
            df_result,
            connection_type,
            source_type,
            source_id_field,
            target_type,
            target_id_field,
            source_title_field,
            target_title_field,
            conflict_handling,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
        // Replace the Arc with the new graph
        self.inner = Arc::new(graph);
        self.selection.clear();
        
        // Store the report
        self.add_report(OperationReport::ConnectionOperation(result.clone()));
        
        // Convert the report to a Python dictionary
        Python::with_gil(|py| {
            let report_dict = PyDict::new_bound(py);
            report_dict.set_item("operation", &result.operation_type)?;
            report_dict.set_item("timestamp", result.timestamp.to_rfc3339())?;
            report_dict.set_item("connections_created", result.connections_created)?;
            report_dict.set_item("connections_skipped", result.connections_skipped)?;
            report_dict.set_item("property_fields_tracked", result.property_fields_tracked)?;
            report_dict.set_item("processing_time_ms", result.processing_time_ms)?;
            
            // Add errors array if there are any
            if !result.errors.is_empty() {
                report_dict.set_item("errors", &result.errors)?;
                report_dict.set_item("has_errors", true)?;
            } else {
                report_dict.set_item("has_errors", false)?;
            }
            
            Ok(report_dict.into())
        })
    }

    fn type_filter(
        &mut self, 
        node_type: String,
        sort: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>
    ) -> PyResult<Self> {
        let mut new_kg = self.clone();
        let mut conditions = HashMap::new();
        conditions.insert("type".to_string(), FilterCondition::Equals(Value::String(node_type)));
        
        let sort_fields = if let Some(spec) = sort {
            match spec.extract::<String>() {
                Ok(field) => Some(vec![(field, true)]),
                Err(_) => Some(py_in::parse_sort_fields(spec, None)?)
            }
        } else {
            None
        };
        
        filtering_methods::filter_nodes(
            &self.inner, 
            &mut new_kg.selection, 
            conditions, 
            sort_fields, 
            max_nodes
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(new_kg)
    }

    fn filter(&mut self, conditions: &Bound<'_, PyDict>, sort: Option<&Bound<'_, PyAny>>, max_nodes: Option<usize>) -> PyResult<Self> {
        let mut new_kg = self.clone();
        let filter_conditions = py_in::pydict_to_filter_conditions(conditions)?;
        let sort_fields = match sort {
            Some(spec) => Some(py_in::parse_sort_fields(spec, None)?),
            None => None,
        };
        
        filtering_methods::filter_nodes(&self.inner, &mut new_kg.selection, filter_conditions, sort_fields, max_nodes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(new_kg)
    }

    fn filter_orphans(
        &mut self,
        include_orphans: Option<bool>,
        sort: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>
    ) -> PyResult<Self> {
        let mut new_kg = self.clone();
        let include = include_orphans.unwrap_or(true);
        
        let sort_fields = if let Some(spec) = sort {
            Some(py_in::parse_sort_fields(spec, None)?)
        } else {
            None
        };
        
        filtering_methods::filter_orphan_nodes(
            &self.inner,
            &mut new_kg.selection,
            include,
            sort_fields.as_ref(),
            max_nodes
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(new_kg)
    }

    fn sort(&mut self, sort: &Bound<'_, PyAny>, ascending: Option<bool>) -> PyResult<Self> {
        let mut new_kg = self.clone();
        let sort_fields = py_in::parse_sort_fields(sort, ascending)?;
        
        filtering_methods::sort_nodes(&self.inner, &mut new_kg.selection, sort_fields)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(new_kg)
    }

    fn max_nodes(&mut self, max_per_group: usize) -> PyResult<Self> {
        let mut new_kg = self.clone();
        filtering_methods::limit_nodes_per_group(
            &self.inner,
            &mut new_kg.selection,
            max_per_group
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(new_kg)
    }

    #[pyo3(signature = (max_nodes=None, indices=None, parent_type=None, parent_info=None, 
                         flatten_single_parent=true))]
    fn get_nodes(
        &self,
        max_nodes: Option<usize>,
        indices: Option<Vec<usize>>, 
        parent_type: Option<&str>,
        parent_info: Option<bool>,
        flatten_single_parent: Option<bool>
    ) -> PyResult<PyObject> {
        let nodes = data_retrieval::get_nodes(
            &self.inner, 
            &self.selection, 
            None,
            indices.as_deref(),
            max_nodes
        );
        Python::with_gil(|py| py_out::level_nodes_to_pydict(
            py, 
            &nodes, 
            parent_type, 
            parent_info, 
            flatten_single_parent
        ))
    }
    
    #[pyo3(signature = (indices=None, parent_info=None, include_node_properties=None, 
                        flatten_single_parent=true))]
    fn get_connections(
        &self,
        indices: Option<Vec<usize>>,
        parent_info: Option<bool>,
        include_node_properties: Option<bool>,
        flatten_single_parent: Option<bool>,
    ) -> PyResult<PyObject> {
        let connections = data_retrieval::get_connections(
            &self.inner,
            &self.selection,
            None,
            indices.as_deref(),
            include_node_properties.unwrap_or(true),
        );
        Python::with_gil(|py| py_out::level_connections_to_pydict(
            py, 
            &connections, 
            parent_info,
            flatten_single_parent
        ))
    }
    
    fn get_titles(&self, max_nodes: Option<usize>, indices: Option<Vec<usize>>) -> PyResult<PyObject> {
        let values = data_retrieval::get_property_values(
            &self.inner,
            &self.selection,
            None,
            &["title"],
            indices.as_deref(),
            max_nodes
        );
        Python::with_gil(|py| py_out::level_single_values_to_pydict(py, &values))
    }
    
    fn get_properties(&self, properties: Vec<String>, max_nodes: Option<usize>, indices: Option<Vec<usize>>) -> PyResult<PyObject> {
        let property_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
        let values = data_retrieval::get_property_values(
            &self.inner,
            &self.selection,
            None,
            &property_refs,
            indices.as_deref(),
            max_nodes
        );
        Python::with_gil(|py| py_out::level_values_to_pydict(py, &values))
    }

    fn unique_values(
        &mut self,
        property: String,
        group_by_parent: Option<bool>,
        level_index: Option<usize>,
        indices: Option<Vec<usize>>,
        store_as: Option<&str>,
        max_length: Option<usize>,
        keep_selection: Option<bool>,
    ) -> PyResult<PyObject> {
        let values = data_retrieval::get_unique_values(
            &self.inner,
            &self.selection,
            &property,
            level_index,
            group_by_parent.unwrap_or(true),
            indices.as_deref()
        );
        
        if let Some(target_property) = store_as {
            let nodes = data_retrieval::format_unique_values_for_storage(&values, max_length);

            // Extract graph or clone it if needed
            let mut graph = extract_or_clone_graph(&mut self.inner);
            
            maintain_graph::update_node_properties(&mut graph, &nodes, target_property)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

            // Replace the Arc with the updated graph
            self.inner = Arc::new(graph);
            
            if !keep_selection.unwrap_or(false) {
                self.selection.clear();
            }

            Python::with_gil(|py| Ok(self.clone().into_py(py)))
        } else {
            Python::with_gil(|py| py_out::level_unique_values_to_pydict(py, &values))
        }
    }
    
    fn traverse(
        &mut self,
        connection_type: String,
        level_index: Option<usize>,
        direction: Option<String>,
        filter_target: Option<&Bound<'_, PyDict>>,
        sort_target: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>,
        new_level: Option<bool>,
    ) -> PyResult<Self> {
        let mut new_kg = self.clone();
        
        let conditions = if let Some(cond) = filter_target {
            Some(py_in::pydict_to_filter_conditions(cond)?)
        } else {
            None
        };
        
        let sort_fields = if let Some(spec) = sort_target {
            Some(py_in::parse_sort_fields(spec, None)?)
        } else {
            None
        };
    
        traversal_methods::make_traversal(
            &self.inner,
            &mut new_kg.selection,
            connection_type,
            level_index,
            direction,
            conditions.as_ref(),
            sort_fields.as_ref(),
            max_nodes,
            new_level,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(new_kg)
    }

    fn selection_to_new_connections(
        &mut self,
        connection_type: String,
        keep_selection: Option<bool>,
        conflict_handling: Option<String>,
    ) -> PyResult<Self> {
        // Extract graph or clone it if needed
        let mut graph = extract_or_clone_graph(&mut self.inner);
        
        let result = maintain_graph::selection_to_new_connections(
            &mut graph, 
            &self.selection, 
            connection_type,
            conflict_handling,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        // Create new graph with the updated data
        let mut new_kg = KnowledgeGraph {
            inner: Arc::new(graph),
            selection: if keep_selection.unwrap_or(false) {
                self.selection.clone()
            } else {
                CurrentSelection::new()
            },
            reports: self.reports.clone(), // Copy over existing reports
        };
        
        // Store the report in the new graph
        new_kg.add_report(OperationReport::ConnectionOperation(result));
        
        // Just return the new KnowledgeGraph
        Ok(new_kg)
    }
    
    #[pyo3(signature = (property=None, filter=None, sort=None, max_nodes=None, store_as=None, max_length=None, keep_selection=None))]
    fn children_properties_to_list(
        &mut self,
        property: Option<&str>,
        filter: Option<&Bound<'_, PyDict>>,
        sort: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>,
        store_as: Option<&str>,
        max_length: Option<usize>,
        keep_selection: Option<bool>,
    ) -> PyResult<PyObject> {
        let property_name = property.unwrap_or("title");
        
        // Apply filtering and sorting if needed
        let mut filtered_kg = self.clone();
        
        if let Some(filter_dict) = filter {
            let conditions = py_in::pydict_to_filter_conditions(filter_dict)?;
            let sort_fields = match sort {
                Some(spec) => Some(py_in::parse_sort_fields(spec, None)?),
                None => None,
            };
            
            filtering_methods::filter_nodes(
                &self.inner, 
                &mut filtered_kg.selection, 
                conditions, 
                sort_fields, 
                max_nodes
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        } else if let Some(spec) = sort {
            let sort_fields = py_in::parse_sort_fields(spec, None)?;
            
            filtering_methods::sort_nodes(
                &self.inner, 
                &mut filtered_kg.selection, 
                sort_fields
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
            
            if let Some(max) = max_nodes {
                filtering_methods::limit_nodes_per_group(
                    &self.inner, 
                    &mut filtered_kg.selection, 
                    max
                ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
            }
        } else if let Some(max) = max_nodes {
            filtering_methods::limit_nodes_per_group(
                &self.inner, 
                &mut filtered_kg.selection, 
                max
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        }
        
        // Generate the property lists with titles already included
        let property_groups = traversal_methods::get_children_properties(
            &filtered_kg.inner,
            &filtered_kg.selection,
            property_name
        );
        
        // If store_as is not provided, return the properties as a dictionary
        if store_as.is_none() {
            // Format for dictionary display
            let dict_pairs = traversal_methods::format_for_dictionary(
                &property_groups,
                max_length
            );
            
            return Python::with_gil(|py| {
                py_out::string_pairs_to_pydict(py, &dict_pairs)
            });
        }
        
        // Format for storage
        let nodes = traversal_methods::format_for_storage(
            &property_groups,
            max_length
        );
        
        // Extract graph or clone it if needed
        let mut graph = extract_or_clone_graph(&mut self.inner);
        
        // Update parent properties
        let result = maintain_graph::update_node_properties(&mut graph, &nodes, store_as.unwrap())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        // Create a new graph with the updated data
        let mut new_kg = KnowledgeGraph {
            inner: Arc::new(graph),
            selection: if keep_selection.unwrap_or(false) {
                self.selection.clone()
            } else {
                CurrentSelection::new()
            },
            reports: self.reports.clone(),
        };
        
        // Store the report
        new_kg.add_report(OperationReport::NodeOperation(result));
        
        // Return the updated graph (no report in return value)
        Python::with_gil(|py| Ok(new_kg.into_py(py)))
    }

    fn statistics(
        &self,
        property: &str,
        level_index: Option<usize>,
    ) -> PyResult<PyObject> {
        let pairs = statistics_methods::get_parent_child_pairs(&self.selection, level_index);
        let stats = statistics_methods::calculate_property_stats(&self.inner, &pairs, property);
        py_out::convert_stats_for_python(stats)
    }

    fn calculate(
        &mut self,
        expression: &str,
        level_index: Option<usize>,
        store_as: Option<&str>,
        keep_selection: Option<bool>,
    ) -> PyResult<PyObject> {
        // If we're storing results, we'll need a mutable graph
        if let Some(target_property) = store_as {
            // Extract graph or clone it if needed
            let mut graph = extract_or_clone_graph(&mut self.inner);
            
            // Process the expression
            let process_result = calculations::process_equation(
                &mut graph,
                &self.selection,
                expression,
                level_index,
                Some(target_property),
            );
            
            // Handle errors
            match process_result {
                Ok(calculations::EvaluationResult::Stored(report)) => {
                    // Create a new graph with the updated data
                    let mut new_kg = KnowledgeGraph {
                        inner: Arc::new(graph),
                        selection: if keep_selection.unwrap_or(false) {
                            self.selection.clone()
                        } else {
                            CurrentSelection::new()
                        },
                        reports: self.reports.clone(), // Copy existing reports
                    };
                    
                    // Store the calculation report
                    new_kg.add_report(OperationReport::CalculationOperation(report));
                    
                    Python::with_gil(|py| Ok(new_kg.into_py(py)))
                },
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unexpected result type when storing calculation result"
                )),
                Err(e) => {
                    let error_msg = format!("Error evaluating expression '{}': {}", expression, e);
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(error_msg))
                }
            }
        } else {
            // Just computing without storing - no need to modify graph
            let process_result = calculations::process_equation(
                &mut (*self.inner).clone(), // Create a temporary clone just for calculation
                &self.selection,
                expression,
                level_index,
                None,
            );
            
            // Handle regular errors with descriptive messages
            match process_result {
                Ok(calculations::EvaluationResult::Computed(results)) => {
                    // Check for errors
                    let error_count = results.iter().filter(|r| r.error_msg.is_some()).count();
                    if error_count == results.len() && !results.is_empty() {
                        if let Some(first_error) = results.iter().find(|r| r.error_msg.is_some()) {
                            if let Some(error_text) = &first_error.error_msg {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    format!("Error in calculation '{}': {}", expression, error_text)
                                ));
                            }
                        }
                    }
                    
                    // Filter out results with errors
                    let valid_results: Vec<StatResult> = results.into_iter()
                        .filter(|r| r.error_msg.is_none())
                        .collect();
                    
                    if valid_results.is_empty() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("No valid results found for expression '{}'", expression)
                        ));
                    }
                    
                    py_out::convert_computation_results_for_python(valid_results)
                },
                Ok(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unexpected result type when computing"
                )),
                Err(e) => {
                    let error_msg = format!("Error evaluating expression '{}': {}", expression, e);
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(error_msg))
                }
            }
        }
    }

    fn count(
        &mut self,
        level_index: Option<usize>,
        group_by_parent: Option<bool>,
        store_as: Option<&str>,
        keep_selection: Option<bool>,
    ) -> PyResult<PyObject> {
        // Default to grouping by parent if we have a nested structure
        let has_multiple_levels = self.selection.get_level_count() > 1;
        // Use the provided group_by_parent if given, otherwise default based on structure
        let use_grouping = group_by_parent.unwrap_or(has_multiple_levels);
        
        if let Some(target_property) = store_as {
            // Extract graph or clone it if needed
            let mut graph = extract_or_clone_graph(&mut self.inner);
            
            // Store count results as node properties and get report
            let result = match calculations::store_count_results(
                &mut graph,
                &self.selection,
                level_index,
                use_grouping,
                target_property
            ) {
                Ok(report) => report,
                Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
            };
            
            // Replace the Arc with updated graph
            self.inner = Arc::new(graph);
            
            // Create a new graph with the updated data
            let mut new_kg = KnowledgeGraph {
                inner: Arc::clone(&self.inner),
                selection: if keep_selection.unwrap_or(false) {
                    self.selection.clone()
                } else {
                    CurrentSelection::new()
                },
                reports: self.reports.clone(), // Copy existing reports
            };
            
            // Add the report
            new_kg.add_report(OperationReport::CalculationOperation(result));
            
            Python::with_gil(|py| Ok(new_kg.into_py(py)))
        } else if use_grouping {
            // Return counts grouped by parent
            let counts = calculations::count_nodes_by_parent(&self.inner, &self.selection, level_index);
            py_out::convert_computation_results_for_python(counts)
        } else {
            // Simple flat count
            let count = calculations::count_nodes_in_level(&self.selection, level_index);
            Python::with_gil(|py| Ok(count.into_py(py)))
        }
    }

    fn get_schema(&self) -> PyResult<String> {
        let schema_string = debugging::get_schema_string(&self.inner);
        Ok(schema_string)
    }

    fn get_selection(&self) -> PyResult<String> {
        Ok(debugging::get_selection_string(&self.inner, &self.selection))
    }

    fn clear(&mut self) -> PyResult<()> {
        self.selection.clear();
        Ok(())
    }

    fn save(&self, path: &str) -> PyResult<()> {
        save_to_file(&self.inner, path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))
    }

    /// Get the most recent operation report as a Python dictionary
    fn get_last_report(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            if let Some(report) = self.reports.get_last_report() {
                match report {
                    OperationReport::NodeOperation(node_report) => {
                        let report_dict = PyDict::new_bound(py);
                        report_dict.set_item("operation", &node_report.operation_type)?;
                        report_dict.set_item("timestamp", node_report.timestamp.to_rfc3339())?;
                        report_dict.set_item("nodes_created", node_report.nodes_created)?;
                        report_dict.set_item("nodes_updated", node_report.nodes_updated)?;
                        report_dict.set_item("nodes_skipped", node_report.nodes_skipped)?;
                        report_dict.set_item("processing_time_ms", node_report.processing_time_ms)?;
                        
                        // Add errors array if there are any
                        if !node_report.errors.is_empty() {
                            report_dict.set_item("errors", &node_report.errors)?;
                            report_dict.set_item("has_errors", true)?;
                        } else {
                            report_dict.set_item("has_errors", false)?;
                        }
                        
                        Ok(report_dict.into())
                    },
                    OperationReport::ConnectionOperation(conn_report) => {
                        let report_dict = PyDict::new_bound(py);
                        report_dict.set_item("operation", &conn_report.operation_type)?;
                        report_dict.set_item("timestamp", conn_report.timestamp.to_rfc3339())?;
                        report_dict.set_item("connections_created", conn_report.connections_created)?;
                        report_dict.set_item("connections_skipped", conn_report.connections_skipped)?;
                        report_dict.set_item("property_fields_tracked", conn_report.property_fields_tracked)?;
                        report_dict.set_item("processing_time_ms", conn_report.processing_time_ms)?;
                        
                        // Add errors array if there are any
                        if !conn_report.errors.is_empty() {
                            report_dict.set_item("errors", &conn_report.errors)?;
                            report_dict.set_item("has_errors", true)?;
                        } else {
                            report_dict.set_item("has_errors", false)?;
                        }
                        
                        Ok(report_dict.into())
                    },
                    OperationReport::CalculationOperation(calc_report) => {
                        let report_dict = PyDict::new_bound(py);
                        report_dict.set_item("operation", &calc_report.operation_type)?;
                        report_dict.set_item("timestamp", calc_report.timestamp.to_rfc3339())?;
                        report_dict.set_item("expression", &calc_report.expression)?;
                        report_dict.set_item("nodes_processed", calc_report.nodes_processed)?;
                        report_dict.set_item("nodes_updated", calc_report.nodes_updated)?;
                        report_dict.set_item("nodes_with_errors", calc_report.nodes_with_errors)?;
                        report_dict.set_item("processing_time_ms", calc_report.processing_time_ms)?;
                        report_dict.set_item("is_aggregation", calc_report.is_aggregation)?;
                        
                        // Add errors array if there are any
                        if !calc_report.errors.is_empty() {
                            report_dict.set_item("errors", &calc_report.errors)?;
                            report_dict.set_item("has_errors", true)?;
                        } else {
                            report_dict.set_item("has_errors", false)?;
                        }
                        
                        Ok(report_dict.into())
                    }
                }
            } else {
                let empty_dict = PyDict::new_bound(py);
                Ok(empty_dict.into())
            }
        })
    }

    /// Get the last operation index (a sequential ID of operations performed)
    fn get_operation_index(&self) -> usize {
        self.reports.get_last_operation_index()
    }

    /// Get all report history as a list of dictionaries
    fn get_report_history(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            // Create an empty list with PyList::empty_bound
            let report_list = PyList::empty_bound(py);
            
            for report in self.reports.get_all_reports() {
                let report_dict = match report {
                    OperationReport::NodeOperation(node_report) => {
                        let dict = PyDict::new_bound(py);
                        dict.set_item("operation", &node_report.operation_type)?;
                        dict.set_item("timestamp", node_report.timestamp.to_rfc3339())?;
                        dict.set_item("nodes_created", node_report.nodes_created)?;
                        dict.set_item("nodes_updated", node_report.nodes_updated)?;
                        dict.set_item("nodes_skipped", node_report.nodes_skipped)?;
                        dict.set_item("processing_time_ms", node_report.processing_time_ms)?;
                        
                        // Add errors array if there are any
                        if !node_report.errors.is_empty() {
                            dict.set_item("errors", &node_report.errors)?;
                            dict.set_item("has_errors", true)?;
                        } else {
                            dict.set_item("has_errors", false)?;
                        }
                        
                        dict
                    },
                    OperationReport::ConnectionOperation(conn_report) => {
                        let dict = PyDict::new_bound(py);
                        dict.set_item("operation", &conn_report.operation_type)?;
                        dict.set_item("timestamp", conn_report.timestamp.to_rfc3339())?;
                        dict.set_item("connections_created", conn_report.connections_created)?;
                        dict.set_item("connections_skipped", conn_report.connections_skipped)?;
                        dict.set_item("property_fields_tracked", conn_report.property_fields_tracked)?;
                        dict.set_item("processing_time_ms", conn_report.processing_time_ms)?;
                        
                        // Add errors array if there are any
                        if !conn_report.errors.is_empty() {
                            dict.set_item("errors", &conn_report.errors)?;
                            dict.set_item("has_errors", true)?;
                        } else {
                            dict.set_item("has_errors", false)?;
                        }
                        
                        dict
                    },
                    OperationReport::CalculationOperation(calc_report) => {
                        let dict = PyDict::new_bound(py);
                        dict.set_item("operation", &calc_report.operation_type)?;
                        dict.set_item("timestamp", calc_report.timestamp.to_rfc3339())?;
                        dict.set_item("expression", &calc_report.expression)?;
                        dict.set_item("nodes_processed", calc_report.nodes_processed)?;
                        dict.set_item("nodes_updated", calc_report.nodes_updated)?;
                        dict.set_item("nodes_with_errors", calc_report.nodes_with_errors)?;
                        dict.set_item("processing_time_ms", calc_report.processing_time_ms)?;
                        dict.set_item("is_aggregation", calc_report.is_aggregation)?;
                        
                        // Add errors array if there are any
                        if !calc_report.errors.is_empty() {
                            dict.set_item("errors", &calc_report.errors)?;
                            dict.set_item("has_errors", true)?;
                        } else {
                            dict.set_item("has_errors", false)?;
                        }
                        
                        dict
                    }
                };
                report_list.append(report_dict)?;
            }
            Ok(report_list.into())
        })
    }
}