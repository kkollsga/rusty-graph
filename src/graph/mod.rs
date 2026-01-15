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
pub mod set_operations;
pub mod schema_validation;
pub mod graph_algorithms;
pub mod subgraph;
pub mod export;

use schema::{DirGraph, CurrentSelection, PlanStep, SchemaDefinition, NodeSchemaDefinition, ConnectionSchemaDefinition};

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

        // Record plan step: estimate based on type index
        let estimated = self.inner.type_indices.get(&node_type).map(|v| v.len()).unwrap_or(0);
        new_kg.selection.clear_execution_plan(); // Start fresh plan

        let mut conditions = HashMap::new();
        conditions.insert("type".to_string(), FilterCondition::Equals(Value::String(node_type.clone())));

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

        // Record actual result
        let actual = new_kg.selection.get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.get_all_nodes().len()).unwrap_or(0);
        new_kg.selection.add_plan_step(
            PlanStep::new("TYPE_FILTER", Some(&node_type), estimated).with_actual_rows(actual)
        );

        Ok(new_kg)
    }

    fn filter(&mut self, conditions: &Bound<'_, PyDict>, sort: Option<&Bound<'_, PyAny>>, max_nodes: Option<usize>) -> PyResult<Self> {
        let mut new_kg = self.clone();

        // Estimate based on current selection
        let estimated = new_kg.selection.get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.get_all_nodes().len()).unwrap_or(0);

        let filter_conditions = py_in::pydict_to_filter_conditions(conditions)?;
        let sort_fields = match sort {
            Some(spec) => Some(py_in::parse_sort_fields(spec, None)?),
            None => None,
        };

        filtering_methods::filter_nodes(&self.inner, &mut new_kg.selection, filter_conditions, sort_fields, max_nodes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        // Record actual result
        let actual = new_kg.selection.get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.get_all_nodes().len()).unwrap_or(0);
        new_kg.selection.add_plan_step(
            PlanStep::new("FILTER", None, estimated).with_actual_rows(actual)
        );

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

    /// Filter nodes that are valid at a specific date
    ///
    /// This is a convenience method for temporal queries. It filters nodes where:
    /// - date_from_field <= date <= date_to_field
    ///
    /// Default field names are 'date_from' and 'date_to' if not specified.
    #[pyo3(signature = (date, date_from_field=None, date_to_field=None))]
    fn valid_at(
        &mut self,
        date: &str,
        date_from_field: Option<&str>,
        date_to_field: Option<&str>,
    ) -> PyResult<Self> {
        let from_field = date_from_field.unwrap_or("date_from");
        let to_field = date_to_field.unwrap_or("date_to");

        let mut new_kg = self.clone();

        // Estimate based on current selection
        let estimated = new_kg.selection.get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.get_all_nodes().len()).unwrap_or(0);

        // Build compound filter: date_from <= date AND date_to >= date
        let mut conditions = HashMap::new();
        conditions.insert(
            from_field.to_string(),
            FilterCondition::LessThanEquals(Value::String(date.to_string()))
        );
        conditions.insert(
            to_field.to_string(),
            FilterCondition::GreaterThanEquals(Value::String(date.to_string()))
        );

        filtering_methods::filter_nodes(&self.inner, &mut new_kg.selection, conditions, None, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        // Record actual result
        let actual = new_kg.selection.get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.get_all_nodes().len()).unwrap_or(0);
        new_kg.selection.add_plan_step(
            PlanStep::new("VALID_AT", None, estimated).with_actual_rows(actual)
        );

        Ok(new_kg)
    }

    /// Filter nodes that are valid during a date range
    ///
    /// This filters nodes where their validity period overlaps with the given range:
    /// - date_from_field <= end_date AND date_to_field >= start_date
    ///
    /// Default field names are 'date_from' and 'date_to' if not specified.
    #[pyo3(signature = (start_date, end_date, date_from_field=None, date_to_field=None))]
    fn valid_during(
        &mut self,
        start_date: &str,
        end_date: &str,
        date_from_field: Option<&str>,
        date_to_field: Option<&str>,
    ) -> PyResult<Self> {
        let from_field = date_from_field.unwrap_or("date_from");
        let to_field = date_to_field.unwrap_or("date_to");

        let mut new_kg = self.clone();

        // Estimate based on current selection
        let estimated = new_kg.selection.get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.get_all_nodes().len()).unwrap_or(0);

        // Build compound filter for overlapping ranges:
        // node.date_from <= end_date AND node.date_to >= start_date
        let mut conditions = HashMap::new();
        conditions.insert(
            from_field.to_string(),
            FilterCondition::LessThanEquals(Value::String(end_date.to_string()))
        );
        conditions.insert(
            to_field.to_string(),
            FilterCondition::GreaterThanEquals(Value::String(start_date.to_string()))
        );

        filtering_methods::filter_nodes(&self.inner, &mut new_kg.selection, conditions, None, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        // Record actual result
        let actual = new_kg.selection.get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.get_all_nodes().len()).unwrap_or(0);
        new_kg.selection.add_plan_step(
            PlanStep::new("VALID_DURING", None, estimated).with_actual_rows(actual)
        );

        Ok(new_kg)
    }

    /// Update properties on all currently selected nodes
    ///
    /// This allows batch updating of properties on nodes matching the current selection.
    /// Returns a new KnowledgeGraph with the updated nodes.
    #[pyo3(signature = (properties, keep_selection=None))]
    fn update(
        &mut self,
        properties: &Bound<'_, PyDict>,
        keep_selection: Option<bool>,
    ) -> PyResult<PyObject> {
        // Get the current level's nodes
        let current_index = self.selection.get_level_count().saturating_sub(1);
        let level = self.selection.get_level(current_index)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No active selection level"
            ))?;

        let nodes = level.get_all_nodes();
        if nodes.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No nodes selected for update"
            ));
        }

        // Extract graph for modification
        let mut graph = extract_or_clone_graph(&mut self.inner);

        // Track total updates
        let mut total_updated = 0;
        let mut errors = Vec::new();

        // Update each property
        for (key, value) in properties.iter() {
            let property_name: String = key.extract()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Property names must be strings"
                ))?;

            let property_value = py_in::py_value_to_value(&value)?;

            // Build node-value pairs for this property
            let node_values: Vec<(Option<petgraph::graph::NodeIndex>, Value)> = nodes.iter()
                .map(|&idx| (Some(idx), property_value.clone()))
                .collect();

            // Update this property on all nodes
            match maintain_graph::update_node_properties(&mut graph, &node_values, &property_name) {
                Ok(report) => {
                    total_updated += report.nodes_updated;
                    errors.extend(report.errors);
                }
                Err(e) => {
                    errors.push(format!("Error updating property '{}': {}", property_name, e));
                }
            }
        }

        // Create the result KnowledgeGraph
        let mut new_kg = KnowledgeGraph {
            inner: Arc::new(graph),
            selection: if keep_selection.unwrap_or(false) {
                self.selection.clone()
            } else {
                CurrentSelection::new()
            },
            reports: self.reports.clone(),
        };

        // Create and add a report
        let report = reporting::NodeOperationReport {
            operation_type: "update".to_string(),
            timestamp: chrono::Utc::now(),
            nodes_created: 0,
            nodes_updated: total_updated,
            nodes_skipped: 0,
            processing_time_ms: 0.0, // Could track this if needed
            errors,
        };

        let report_index = new_kg.add_report(OperationReport::NodeOperation(report));

        // Return the new KnowledgeGraph and the report
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("graph", new_kg.into_py(py))?;
            dict.set_item("nodes_updated", total_updated)?;
            dict.set_item("report_index", report_index)?;
            Ok(dict.into())
        })
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

    /// Returns a string representation of the query execution plan.
    ///
    /// Shows each operation in the query chain with estimated and actual row counts.
    /// Example output: "TYPE_FILTER Prospect (6775 nodes) -> TRAVERSE HAS_ESTIMATE (10954 nodes)"
    fn explain(&self) -> PyResult<String> {
        let plan = self.selection.get_execution_plan();
        if plan.is_empty() {
            return Ok("No query operations recorded".to_string());
        }

        let steps: Vec<String> = plan.iter().map(|step| {
            let type_info = step.node_type.as_ref()
                .map(|t| format!(" {}", t))
                .unwrap_or_default();
            let rows = step.actual_rows.unwrap_or(step.estimated_rows);
            format!("{}{} ({} nodes)", step.operation, type_info, rows)
        }).collect();

        Ok(steps.join(" -> "))
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
        filter_connection: Option<&Bound<'_, PyDict>>,
        sort_target: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>,
        new_level: Option<bool>,
    ) -> PyResult<Self> {
        let mut new_kg = self.clone();

        // Estimate based on current selection (source nodes)
        let estimated = new_kg.selection.get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.get_all_nodes().len()).unwrap_or(0);

        let conditions = if let Some(cond) = filter_target {
            Some(py_in::pydict_to_filter_conditions(cond)?)
        } else {
            None
        };

        let conn_conditions = if let Some(cond) = filter_connection {
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
            connection_type.clone(),
            level_index,
            direction,
            conditions.as_ref(),
            conn_conditions.as_ref(),
            sort_fields.as_ref(),
            max_nodes,
            new_level,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        // Record actual result
        let actual = new_kg.selection.get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.get_all_nodes().len()).unwrap_or(0);
        new_kg.selection.add_plan_step(
            PlanStep::new("TRAVERSE", Some(&connection_type), estimated).with_actual_rows(actual)
        );

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

    #[pyo3(signature = (expression, level_index=None, store_as=None, keep_selection=None, aggregate_connections=None))]
    fn calculate(
        &mut self,
        expression: &str,
        level_index: Option<usize>,
        store_as: Option<&str>,
        keep_selection: Option<bool>,
        aggregate_connections: Option<bool>,
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
                aggregate_connections,
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
                aggregate_connections,
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

    /// Perform union of two selections - combines all nodes from both selections
    /// Returns a new KnowledgeGraph with the union of both selections
    fn union(&self, other: &Self) -> PyResult<Self> {
        let mut new_kg = self.clone();
        set_operations::union_selections(&mut new_kg.selection, &other.selection)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(new_kg)
    }

    /// Perform intersection of two selections - keeps only nodes present in both
    /// Returns a new KnowledgeGraph with only nodes that exist in both selections
    fn intersection(&self, other: &Self) -> PyResult<Self> {
        let mut new_kg = self.clone();
        set_operations::intersection_selections(&mut new_kg.selection, &other.selection)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(new_kg)
    }

    /// Perform difference of two selections - keeps nodes in self but not in other
    /// Returns a new KnowledgeGraph with nodes from self that are not in other
    fn difference(&self, other: &Self) -> PyResult<Self> {
        let mut new_kg = self.clone();
        set_operations::difference_selections(&mut new_kg.selection, &other.selection)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(new_kg)
    }

    /// Perform symmetric difference of two selections - keeps nodes in either but not both
    /// Returns a new KnowledgeGraph with nodes that are in exactly one of the selections
    fn symmetric_difference(&self, other: &Self) -> PyResult<Self> {
        let mut new_kg = self.clone();
        set_operations::symmetric_difference_selections(&mut new_kg.selection, &other.selection)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(new_kg)
    }

    // ========================================================================
    // Schema Definition & Validation Methods
    // ========================================================================

    /// Define the expected schema for the graph
    ///
    /// Args:
    ///     schema_dict: A dictionary defining the schema with the following structure:
    ///         {
    ///             'nodes': {
    ///                 'NodeType': {
    ///                     'required': ['field1', 'field2'],  # Required fields
    ///                     'optional': ['field3'],            # Optional fields (for documentation)
    ///                     'types': {'field1': 'string', 'field2': 'integer'}  # Field types
    ///                 }
    ///             },
    ///             'connections': {
    ///                 'CONNECTION_TYPE': {
    ///                     'source': 'SourceNodeType',
    ///                     'target': 'TargetNodeType',
    ///                     'cardinality': 'one-to-many',  # Optional
    ///                     'required_properties': ['prop1'],  # Optional
    ///                     'property_types': {'prop1': 'float'}  # Optional
    ///                 }
    ///             }
    ///         }
    ///
    /// Returns:
    ///     Self with schema defined
    fn define_schema(&mut self, schema_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut schema = SchemaDefinition::new();

        // Parse node schemas
        if let Some(nodes_dict) = schema_dict.get_item("nodes")? {
            if let Ok(nodes) = nodes_dict.downcast::<PyDict>() {
                for (node_type_key, node_schema_val) in nodes.iter() {
                    let node_type: String = node_type_key.extract()?;
                    let node_schema_dict = node_schema_val.downcast::<PyDict>()
                        .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            format!("Schema for node type '{}' must be a dictionary", node_type)
                        ))?;

                    let mut node_schema = NodeSchemaDefinition::default();

                    // Parse required fields
                    if let Some(required) = node_schema_dict.get_item("required")? {
                        node_schema.required_fields = required.extract::<Vec<String>>()?;
                    }

                    // Parse optional fields
                    if let Some(optional) = node_schema_dict.get_item("optional")? {
                        node_schema.optional_fields = optional.extract::<Vec<String>>()?;
                    }

                    // Parse field types
                    if let Some(types) = node_schema_dict.get_item("types")? {
                        let types_dict = types.downcast::<PyDict>()
                            .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "types must be a dictionary"
                            ))?;
                        for (field, type_val) in types_dict.iter() {
                            node_schema.field_types.insert(
                                field.extract::<String>()?,
                                type_val.extract::<String>()?
                            );
                        }
                    }

                    schema.add_node_schema(node_type, node_schema);
                }
            }
        }

        // Parse connection schemas
        if let Some(connections_dict) = schema_dict.get_item("connections")? {
            if let Ok(connections) = connections_dict.downcast::<PyDict>() {
                for (conn_type_key, conn_schema_val) in connections.iter() {
                    let conn_type: String = conn_type_key.extract()?;
                    let conn_schema_dict = conn_schema_val.downcast::<PyDict>()
                        .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            format!("Schema for connection type '{}' must be a dictionary", conn_type)
                        ))?;

                    let source_type: String = conn_schema_dict
                        .get_item("source")?
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                            format!("Connection '{}' missing required 'source' field", conn_type)
                        ))?
                        .extract()?;

                    let target_type: String = conn_schema_dict
                        .get_item("target")?
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                            format!("Connection '{}' missing required 'target' field", conn_type)
                        ))?
                        .extract()?;

                    let mut conn_schema = ConnectionSchemaDefinition {
                        source_type,
                        target_type,
                        cardinality: None,
                        required_properties: Vec::new(),
                        property_types: HashMap::new(),
                    };

                    // Parse optional cardinality
                    if let Some(cardinality) = conn_schema_dict.get_item("cardinality")? {
                        conn_schema.cardinality = Some(cardinality.extract::<String>()?);
                    }

                    // Parse required_properties
                    if let Some(required_props) = conn_schema_dict.get_item("required_properties")? {
                        conn_schema.required_properties = required_props.extract::<Vec<String>>()?;
                    }

                    // Parse property_types
                    if let Some(prop_types) = conn_schema_dict.get_item("property_types")? {
                        let types_dict = prop_types.downcast::<PyDict>()
                            .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "property_types must be a dictionary"
                            ))?;
                        for (field, type_val) in types_dict.iter() {
                            conn_schema.property_types.insert(
                                field.extract::<String>()?,
                                type_val.extract::<String>()?
                            );
                        }
                    }

                    schema.add_connection_schema(conn_type, conn_schema);
                }
            }
        }

        // Store the schema in the graph
        let mut graph = extract_or_clone_graph(&mut self.inner);
        graph.set_schema(schema);
        self.inner = Arc::new(graph);

        Ok(self.clone())
    }

    /// Validate the graph against the defined schema
    ///
    /// Args:
    ///     strict: If True, reports node/connection types that exist in the graph
    ///             but are not defined in the schema. Default is False.
    ///
    /// Returns:
    ///     A list of validation error dictionaries. Empty list means validation passed.
    ///     Each error dict contains:
    ///         - 'error_type': Type of error (e.g., 'missing_required_field', 'type_mismatch')
    ///         - 'message': Human-readable error message
    ///         - Additional fields depending on error type
    fn validate_schema(&self, py: Python<'_>, strict: Option<bool>) -> PyResult<PyObject> {
        let schema = self.inner.get_schema()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No schema defined. Call define_schema() first."
            ))?;

        let errors = schema_validation::validate_graph(
            &self.inner,
            schema,
            strict.unwrap_or(false)
        );

        // Convert errors to Python list of dicts
        let result = PyList::empty_bound(py);
        for error in errors {
            let error_dict = PyDict::new_bound(py);

            match &error {
                schema::ValidationError::MissingRequiredField { node_type, node_title, field } => {
                    error_dict.set_item("error_type", "missing_required_field")?;
                    error_dict.set_item("node_type", node_type)?;
                    error_dict.set_item("node_title", node_title)?;
                    error_dict.set_item("field", field)?;
                }
                schema::ValidationError::TypeMismatch { node_type, node_title, field, expected_type, actual_type } => {
                    error_dict.set_item("error_type", "type_mismatch")?;
                    error_dict.set_item("node_type", node_type)?;
                    error_dict.set_item("node_title", node_title)?;
                    error_dict.set_item("field", field)?;
                    error_dict.set_item("expected_type", expected_type)?;
                    error_dict.set_item("actual_type", actual_type)?;
                }
                schema::ValidationError::InvalidConnectionEndpoint { connection_type, expected_source, expected_target, actual_source, actual_target } => {
                    error_dict.set_item("error_type", "invalid_connection_endpoint")?;
                    error_dict.set_item("connection_type", connection_type)?;
                    error_dict.set_item("expected_source", expected_source)?;
                    error_dict.set_item("expected_target", expected_target)?;
                    error_dict.set_item("actual_source", actual_source)?;
                    error_dict.set_item("actual_target", actual_target)?;
                }
                schema::ValidationError::MissingConnectionProperty { connection_type, source_title, target_title, property } => {
                    error_dict.set_item("error_type", "missing_connection_property")?;
                    error_dict.set_item("connection_type", connection_type)?;
                    error_dict.set_item("source_title", source_title)?;
                    error_dict.set_item("target_title", target_title)?;
                    error_dict.set_item("property", property)?;
                }
                schema::ValidationError::UndefinedNodeType { node_type, count } => {
                    error_dict.set_item("error_type", "undefined_node_type")?;
                    error_dict.set_item("node_type", node_type)?;
                    error_dict.set_item("count", count)?;
                }
                schema::ValidationError::UndefinedConnectionType { connection_type, count } => {
                    error_dict.set_item("error_type", "undefined_connection_type")?;
                    error_dict.set_item("connection_type", connection_type)?;
                    error_dict.set_item("count", count)?;
                }
            }

            error_dict.set_item("message", error.to_string())?;
            result.append(error_dict)?;
        }

        Ok(result.into())
    }

    /// Check if a schema has been defined for this graph
    fn has_schema(&self) -> bool {
        self.inner.get_schema().is_some()
    }

    /// Clear the schema definition from the graph
    fn clear_schema(&mut self) -> PyResult<Self> {
        let mut graph = extract_or_clone_graph(&mut self.inner);
        graph.clear_schema();
        self.inner = Arc::new(graph);
        Ok(self.clone())
    }

    /// Get the current schema definition as a dictionary
    fn get_schema_definition(&self, py: Python<'_>) -> PyResult<PyObject> {
        let schema = match self.inner.get_schema() {
            Some(s) => s,
            None => return Ok(py.None()),
        };

        let result = PyDict::new_bound(py);

        // Convert node schemas
        let nodes_dict = PyDict::new_bound(py);
        for (node_type, node_schema) in &schema.node_schemas {
            let schema_dict = PyDict::new_bound(py);
            schema_dict.set_item("required", &node_schema.required_fields)?;
            schema_dict.set_item("optional", &node_schema.optional_fields)?;

            let types_dict = PyDict::new_bound(py);
            for (field, field_type) in &node_schema.field_types {
                types_dict.set_item(field, field_type)?;
            }
            schema_dict.set_item("types", types_dict)?;

            nodes_dict.set_item(node_type, schema_dict)?;
        }
        result.set_item("nodes", nodes_dict)?;

        // Convert connection schemas
        let connections_dict = PyDict::new_bound(py);
        for (conn_type, conn_schema) in &schema.connection_schemas {
            let schema_dict = PyDict::new_bound(py);
            schema_dict.set_item("source", &conn_schema.source_type)?;
            schema_dict.set_item("target", &conn_schema.target_type)?;

            if let Some(cardinality) = &conn_schema.cardinality {
                schema_dict.set_item("cardinality", cardinality)?;
            }

            if !conn_schema.required_properties.is_empty() {
                schema_dict.set_item("required_properties", &conn_schema.required_properties)?;
            }

            if !conn_schema.property_types.is_empty() {
                let types_dict = PyDict::new_bound(py);
                for (prop, prop_type) in &conn_schema.property_types {
                    types_dict.set_item(prop, prop_type)?;
                }
                schema_dict.set_item("property_types", types_dict)?;
            }

            connections_dict.set_item(conn_type, schema_dict)?;
        }
        result.set_item("connections", connections_dict)?;

        Ok(result.into())
    }

    // ========================================================================
    // Graph Algorithms: Path Finding & Connectivity
    // ========================================================================

    /// Find the shortest path between two nodes.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     A dictionary with:
    ///         - 'path': List of node info dicts along the path
    ///         - 'connections': List of connection types between nodes
    ///         - 'length': Number of hops in the path
    ///     Returns None if no path exists.
    fn shortest_path(
        &self,
        py: Python<'_>,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        // Look up source node
        let source_lookup = lookups::TypeLookup::new(&self.inner.graph, source_type.to_string())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = source_lookup.check_uid(&source_value)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Source node with id {:?} not found in type '{}'", source_value, source_type)
            ))?;

        // Look up target node
        let target_lookup = if target_type == source_type {
            source_lookup
        } else {
            lookups::TypeLookup::new(&self.inner.graph, target_type.to_string())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?
        };

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = target_lookup.check_uid(&target_value)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Target node with id {:?} not found in type '{}'", target_value, target_type)
            ))?;

        // Find shortest path
        let result = graph_algorithms::shortest_path(&self.inner, source_idx, target_idx);

        match result {
            Some(path_result) => {
                let result_dict = PyDict::new_bound(py);

                // Build path info list
                let path_list = PyList::empty_bound(py);
                for &node_idx in &path_result.path {
                    if let Some(info) = graph_algorithms::get_node_info(&self.inner, node_idx) {
                        let node_dict = PyDict::new_bound(py);
                        node_dict.set_item("type", &info.node_type)?;
                        node_dict.set_item("title", &info.title)?;
                        node_dict.set_item("id", py_out::value_to_py(py, &info.id)?)?;
                        path_list.append(node_dict)?;
                    }
                }
                result_dict.set_item("path", path_list)?;

                // Build connections list
                let connections = graph_algorithms::get_path_connections(&self.inner, &path_result.path);
                let conn_list = PyList::empty_bound(py);
                for conn in connections {
                    match conn {
                        Some(c) => conn_list.append(&c)?,
                        None => conn_list.append(py.None())?,
                    }
                }
                result_dict.set_item("connections", conn_list)?;
                result_dict.set_item("length", path_result.cost)?;

                Ok(result_dict.into())
            }
            None => Ok(py.None()),
        }
    }

    /// Find all paths between two nodes up to a maximum number of hops.
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///     max_hops: Maximum path length to search (default: 5)
    ///
    /// Returns:
    ///     A list of path dictionaries, each with 'path', 'connections', and 'length'
    ///
    /// Warning: This can be expensive for graphs with many paths!
    fn all_paths(
        &self,
        py: Python<'_>,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
        max_hops: Option<usize>,
    ) -> PyResult<PyObject> {
        let max_hops = max_hops.unwrap_or(5);

        // Look up source node
        let source_lookup = lookups::TypeLookup::new(&self.inner.graph, source_type.to_string())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = source_lookup.check_uid(&source_value)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Source node with id {:?} not found in type '{}'", source_value, source_type)
            ))?;

        // Look up target node
        let target_lookup = if target_type == source_type {
            source_lookup
        } else {
            lookups::TypeLookup::new(&self.inner.graph, target_type.to_string())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?
        };

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = target_lookup.check_uid(&target_value)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Target node with id {:?} not found in type '{}'", target_value, target_type)
            ))?;

        // Find all paths
        let paths = graph_algorithms::all_paths(&self.inner, source_idx, target_idx, max_hops);

        // Convert to Python output
        let result_list = PyList::empty_bound(py);
        for path in paths {
            let path_dict = PyDict::new_bound(py);

            // Build path info list
            let path_list = PyList::empty_bound(py);
            for &node_idx in &path {
                if let Some(info) = graph_algorithms::get_node_info(&self.inner, node_idx) {
                    let node_dict = PyDict::new_bound(py);
                    node_dict.set_item("type", &info.node_type)?;
                    node_dict.set_item("title", &info.title)?;
                    node_dict.set_item("id", py_out::value_to_py(py, &info.id)?)?;
                    path_list.append(node_dict)?;
                }
            }
            path_dict.set_item("path", path_list)?;

            // Build connections list
            let connections = graph_algorithms::get_path_connections(&self.inner, &path);
            let conn_list = PyList::empty_bound(py);
            for conn in connections {
                match conn {
                    Some(c) => conn_list.append(&c)?,
                    None => conn_list.append(py.None())?,
                }
            }
            path_dict.set_item("connections", conn_list)?;
            path_dict.set_item("length", path.len().saturating_sub(1))?;

            result_list.append(path_dict)?;
        }

        Ok(result_list.into())
    }

    /// Find all connected components in the graph.
    ///
    /// Args:
    ///     weak: If True (default), find weakly connected components (treating graph as undirected).
    ///           If False, find strongly connected components (respecting edge direction).
    ///
    /// Returns:
    ///     A list of components, each component is a list of node info dicts.
    ///     Components are sorted by size (largest first).
    fn connected_components(&self, py: Python<'_>, weak: Option<bool>) -> PyResult<PyObject> {
        let weak = weak.unwrap_or(true);

        let components = if weak {
            graph_algorithms::weakly_connected_components(&self.inner)
        } else {
            graph_algorithms::connected_components(&self.inner)
        };

        // Convert to Python output
        let result_list = PyList::empty_bound(py);
        for component in components {
            let comp_list = PyList::empty_bound(py);
            for &node_idx in &component {
                if let Some(info) = graph_algorithms::get_node_info(&self.inner, node_idx) {
                    let node_dict = PyDict::new_bound(py);
                    node_dict.set_item("type", &info.node_type)?;
                    node_dict.set_item("title", &info.title)?;
                    node_dict.set_item("id", py_out::value_to_py(py, &info.id)?)?;
                    comp_list.append(node_dict)?;
                }
            }
            result_list.append(comp_list)?;
        }

        Ok(result_list.into())
    }

    /// Check if two nodes are connected (directly or indirectly).
    ///
    /// Args:
    ///     source_type: The node type of the source node
    ///     source_id: The unique ID of the source node
    ///     target_type: The node type of the target node
    ///     target_id: The unique ID of the target node
    ///
    /// Returns:
    ///     True if the nodes are connected, False otherwise
    fn are_connected(
        &self,
        source_type: &str,
        source_id: &Bound<'_, PyAny>,
        target_type: &str,
        target_id: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        // Look up source node
        let source_lookup = lookups::TypeLookup::new(&self.inner.graph, source_type.to_string())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        let source_value = py_in::py_value_to_value(source_id)?;
        let source_idx = source_lookup.check_uid(&source_value)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Source node with id {:?} not found in type '{}'", source_value, source_type)
            ))?;

        // Look up target node
        let target_lookup = if target_type == source_type {
            source_lookup
        } else {
            lookups::TypeLookup::new(&self.inner.graph, target_type.to_string())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?
        };

        let target_value = py_in::py_value_to_value(target_id)?;
        let target_idx = target_lookup.check_uid(&target_value)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Target node with id {:?} not found in type '{}'", target_value, target_type)
            ))?;

        Ok(graph_algorithms::are_connected(&self.inner, source_idx, target_idx))
    }

    /// Get the degree (number of connections) for nodes in the current selection.
    ///
    /// Returns:
    ///     A dictionary mapping node titles to their degree counts
    fn get_degrees(&self, py: Python<'_>) -> PyResult<PyObject> {
        let result_dict = PyDict::new_bound(py);

        let level_count = self.selection.get_level_count();
        if level_count == 0 {
            return Ok(result_dict.into());
        }

        let level = self.selection.get_level(level_count - 1)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No selection level"))?;

        for node_idx in level.get_all_nodes() {
            if let Some(info) = graph_algorithms::get_node_info(&self.inner, node_idx) {
                let degree = graph_algorithms::node_degree(&self.inner, node_idx);
                result_dict.set_item(&info.title, degree)?;
            }
        }

        Ok(result_dict.into())
    }

    // ========================================================================
    // Subgraph Extraction Methods
    // ========================================================================

    /// Expand the current selection by N hops.
    ///
    /// This performs a breadth-first expansion from all currently selected nodes,
    /// including all nodes within the specified number of hops. The expansion
    /// considers edges in both directions (undirected).
    ///
    /// Args:
    ///     hops: Number of hops to expand (default: 1)
    ///
    /// Returns:
    ///     A new KnowledgeGraph with the expanded selection
    ///
    /// Example:
    ///     ```python
    ///     # Start with a single field and expand to include connected nodes
    ///     expanded = graph.type_filter('Field').filter({'name': 'EKOFISK'}).expand(hops=2)
    ///     ```
    fn expand(&mut self, hops: Option<usize>) -> PyResult<Self> {
        let hops = hops.unwrap_or(1);
        let mut new_kg = self.clone();

        // Record plan step
        let estimated = new_kg.selection.get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.get_all_nodes().len()).unwrap_or(0);

        subgraph::expand_selection(&self.inner, &mut new_kg.selection, hops)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        // Record actual result
        let actual = new_kg.selection.get_level(new_kg.selection.get_level_count().saturating_sub(1))
            .map(|l| l.get_all_nodes().len()).unwrap_or(0);
        new_kg.selection.add_plan_step(
            PlanStep::new("EXPAND", None, estimated).with_actual_rows(actual)
        );

        Ok(new_kg)
    }

    /// Extract the currently selected nodes into a new independent subgraph.
    ///
    /// Creates a new KnowledgeGraph containing only the nodes in the current
    /// selection and all edges that connect those nodes. The new graph is
    /// completely independent from the original.
    ///
    /// Returns:
    ///     A new KnowledgeGraph containing only the selected nodes and their
    ///     connecting edges
    ///
    /// Example:
    ///     ```python
    ///     # Extract a subgraph of a specific region
    ///     subgraph = (
    ///         graph.type_filter('Field')
    ///         .filter({'region': 'North Sea'})
    ///         .expand(hops=2)
    ///         .to_subgraph()
    ///     )
    ///     # Save the subgraph for later use
    ///     subgraph.save('north_sea_region.bin')
    ///     ```
    fn to_subgraph(&self) -> PyResult<Self> {
        let extracted = subgraph::extract_subgraph(&self.inner, &self.selection)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        Ok(KnowledgeGraph {
            inner: Arc::new(extracted),
            selection: CurrentSelection::new(),
            reports: OperationReports::new(), // Fresh reports for new graph
        })
    }

    /// Get statistics about the subgraph that would be extracted.
    ///
    /// Returns information about what would be included in a subgraph extraction
    /// without actually creating the subgraph. Useful for understanding the
    /// scope of an extraction before committing to it.
    ///
    /// Returns:
    ///     A dictionary with:
    ///         - 'node_count': Total number of nodes
    ///         - 'edge_count': Total number of edges
    ///         - 'node_types': Dict of node type -> count
    ///         - 'connection_types': Dict of connection type -> count
    fn subgraph_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = subgraph::get_subgraph_stats(&self.inner, &self.selection)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        let result_dict = PyDict::new_bound(py);
        result_dict.set_item("node_count", stats.node_count)?;
        result_dict.set_item("edge_count", stats.edge_count)?;

        let node_types_dict = PyDict::new_bound(py);
        for (node_type, count) in &stats.node_types {
            node_types_dict.set_item(node_type, count)?;
        }
        result_dict.set_item("node_types", node_types_dict)?;

        let conn_types_dict = PyDict::new_bound(py);
        for (conn_type, count) in &stats.connection_types {
            conn_types_dict.set_item(conn_type, count)?;
        }
        result_dict.set_item("connection_types", conn_types_dict)?;

        Ok(result_dict.into())
    }

    // ========================================================================
    // Export Methods
    // ========================================================================

    /// Export the graph or current selection to a file in the specified format.
    ///
    /// Supported formats:
    /// - "graphml" - GraphML XML format (Gephi, yEd, Cytoscape)
    /// - "gexf" - GEXF XML format (Gephi native)
    /// - "d3" or "json" - D3.js compatible JSON format
    /// - "csv" - CSV format (creates two files: path_nodes.csv and path_edges.csv)
    ///
    /// Args:
    ///     path: Output file path
    ///     format: Export format (default: inferred from file extension)
    ///     selection_only: If True, export only selected nodes (default: True if selection exists)
    ///
    /// Example:
    ///     ```python
    ///     # Export entire graph to GraphML
    ///     graph.export('output.graphml')
    ///
    ///     # Export selection to D3 format
    ///     graph.type_filter('Field').expand(hops=2).export('fields.json', format='d3')
    ///
    ///     # Export to GEXF for Gephi
    ///     graph.export('network.gexf', format='gexf')
    ///     ```
    #[pyo3(signature = (path, format=None, selection_only=None))]
    fn export(
        &self,
        path: &str,
        format: Option<&str>,
        selection_only: Option<bool>,
    ) -> PyResult<()> {
        // Infer format from extension if not specified
        let fmt = format.unwrap_or_else(|| {
            if path.ends_with(".graphml") {
                "graphml"
            } else if path.ends_with(".gexf") {
                "gexf"
            } else if path.ends_with(".json") {
                "d3"
            } else if path.ends_with(".csv") {
                "csv"
            } else {
                "graphml" // Default
            }
        });

        // Determine if we should use selection
        let use_selection = selection_only.unwrap_or(self.selection.get_level_count() > 0);
        let selection = if use_selection {
            Some(&self.selection)
        } else {
            None
        };

        match fmt {
            "graphml" => {
                let content = export::to_graphml(&self.inner, selection)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
                std::fs::write(path, content)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            "gexf" => {
                let content = export::to_gexf(&self.inner, selection)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
                std::fs::write(path, content)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            "d3" | "json" => {
                let content = export::to_d3_json(&self.inner, selection)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
                std::fs::write(path, content)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            "csv" => {
                let (nodes_csv, edges_csv) = export::to_csv(&self.inner, selection)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

                // Write nodes file
                let nodes_path = path.replace(".csv", "_nodes.csv");
                std::fs::write(&nodes_path, nodes_csv)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

                // Write edges file
                let edges_path = path.replace(".csv", "_edges.csv");
                std::fs::write(&edges_path, edges_csv)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unknown export format: '{}'. Supported: graphml, gexf, d3, json, csv", fmt)
                ));
            }
        }

        Ok(())
    }

    /// Export to a string instead of a file.
    ///
    /// Useful for web APIs or further processing.
    ///
    /// Args:
    ///     format: Export format (graphml, gexf, d3, json)
    ///     selection_only: If True, export only selected nodes
    ///
    /// Returns:
    ///     The exported data as a string
    #[pyo3(signature = (format, selection_only=None))]
    fn export_string(
        &self,
        format: &str,
        selection_only: Option<bool>,
    ) -> PyResult<String> {
        let use_selection = selection_only.unwrap_or(self.selection.get_level_count() > 0);
        let selection = if use_selection {
            Some(&self.selection)
        } else {
            None
        };

        match format {
            "graphml" => {
                export::to_graphml(&self.inner, selection)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
            }
            "gexf" => {
                export::to_gexf(&self.inner, selection)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
            }
            "d3" | "json" => {
                export::to_d3_json(&self.inner, selection)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
            }
            _ => {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unknown export format: '{}'. Supported: graphml, gexf, d3, json", format)
                ))
            }
        }
    }

    // ========================================================================
    // Index Management Methods
    // ========================================================================

    /// Create an index on a property for a specific node type.
    ///
    /// Indexes dramatically speed up equality filters on the indexed property.
    /// Once created, the index is automatically used by filter() operations.
    ///
    /// Args:
    ///     node_type: The type of nodes to index
    ///     property: The property name to index
    ///
    /// Returns:
    ///     Dictionary with 'unique_values' count and success status
    ///
    /// Example:
    ///     ```python
    ///     # Create an index for faster lookups
    ///     graph.create_index('Prospect', 'geoprovince')
    ///
    ///     # Now this filter will use the index (O(1) instead of O(n))
    ///     graph.type_filter('Prospect').filter({'geoprovince': 'North Sea'})
    ///     ```
    fn create_index(&mut self, py: Python<'_>, node_type: &str, property: &str) -> PyResult<PyObject> {
        let mut graph = extract_or_clone_graph(&mut self.inner);
        let unique_values = graph.create_index(node_type, property);
        self.inner = Arc::new(graph);

        let result_dict = PyDict::new_bound(py);
        result_dict.set_item("node_type", node_type)?;
        result_dict.set_item("property", property)?;
        result_dict.set_item("unique_values", unique_values)?;
        result_dict.set_item("created", true)?;

        Ok(result_dict.into())
    }

    /// Drop (remove) an index.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     property: The property name
    ///
    /// Returns:
    ///     True if index existed and was removed, False otherwise
    fn drop_index(&mut self, node_type: &str, property: &str) -> PyResult<bool> {
        let mut graph = extract_or_clone_graph(&mut self.inner);
        let removed = graph.drop_index(node_type, property);
        self.inner = Arc::new(graph);
        Ok(removed)
    }

    /// List all existing indexes.
    ///
    /// Returns:
    ///     List of dictionaries with 'node_type' and 'property' keys
    ///
    /// Example:
    ///     ```python
    ///     indexes = graph.list_indexes()
    ///     for idx in indexes:
    ///         print(f"{idx['node_type']}.{idx['property']}")
    ///     ```
    fn list_indexes(&self, py: Python<'_>) -> PyResult<PyObject> {
        let indexes = self.inner.list_indexes();

        let result_list = pyo3::types::PyList::empty_bound(py);
        for (node_type, property) in indexes {
            let idx_dict = PyDict::new_bound(py);
            idx_dict.set_item("node_type", node_type)?;
            idx_dict.set_item("property", property)?;
            result_list.append(idx_dict)?;
        }

        Ok(result_list.into())
    }

    /// Check if an index exists.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     property: The property name
    ///
    /// Returns:
    ///     True if index exists, False otherwise
    fn has_index(&self, node_type: &str, property: &str) -> bool {
        self.inner.has_index(node_type, property)
    }

    /// Get statistics about an index.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     property: The property name
    ///
    /// Returns:
    ///     Dictionary with index statistics, or None if index doesn't exist
    ///
    /// Example:
    ///     ```python
    ///     stats = graph.index_stats('Prospect', 'geoprovince')
    ///     print(f"Unique values: {stats['unique_values']}")
    ///     print(f"Total entries: {stats['total_entries']}")
    ///     ```
    fn index_stats(&self, py: Python<'_>, node_type: &str, property: &str) -> PyResult<PyObject> {
        match self.inner.get_index_stats(node_type, property) {
            Some(stats) => {
                let result_dict = PyDict::new_bound(py);
                result_dict.set_item("node_type", node_type)?;
                result_dict.set_item("property", property)?;
                result_dict.set_item("unique_values", stats.unique_values)?;
                result_dict.set_item("total_entries", stats.total_entries)?;
                result_dict.set_item("avg_entries_per_value", stats.avg_entries_per_value)?;
                Ok(result_dict.into())
            }
            None => Ok(py.None())
        }
    }

    /// Rebuild all indexes.
    ///
    /// Call this after batch updates to ensure indexes are current.
    ///
    /// Returns:
    ///     Number of indexes rebuilt
    fn rebuild_indexes(&mut self) -> PyResult<usize> {
        let mut graph = extract_or_clone_graph(&mut self.inner);

        // Get list of current indexes
        let index_keys: Vec<_> = graph.property_indices.keys().cloned().collect();

        // Rebuild each index
        for (node_type, property) in &index_keys {
            graph.create_index(node_type, property);
        }

        self.inner = Arc::new(graph);
        Ok(index_keys.len())
    }
}