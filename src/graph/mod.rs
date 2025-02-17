// src/graph/mod.rs
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use std::collections::HashMap;
use crate::datatypes::python_conversions::{
    convert_stats_for_python, 
    pandas_to_dataframe, 
    ensure_columns, 
    parse_sort_fields, 
    pydict_to_filter_conditions,
    level_nodes_to_pydict,
    level_values_to_pydict,
    level_single_values_to_pydict
};
use crate::datatypes::values::{Value, FilterCondition};
use crate::graph::io_operations::save_to_file;

pub mod maintain_graph;
pub mod filtering_methods;
pub mod traversal_methods;
pub mod statistics_methods;
pub mod io_operations;
pub mod lookups;
pub mod debugging;
pub mod batch_operations;
pub mod schema;
pub mod data_retrieval;

use schema::{DirGraph, CurrentSelection};

#[pyclass]
#[derive(Clone)]
pub struct KnowledgeGraph {
    inner: DirGraph,
    selection: CurrentSelection,
}

#[pymethods]
impl KnowledgeGraph {
    #[new]
    fn new() -> Self {
        KnowledgeGraph {
            inner: DirGraph::new(),
            selection: CurrentSelection::new(),
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
        _column_types: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let modified_columns: Option<Vec<String>> = ensure_columns(
            columns, &unique_id_field, &node_title_field
        )?;
        let df_result = pandas_to_dataframe(
            data, &[unique_id_field.clone()], modified_columns.as_deref(),
        )?;
        maintain_graph::add_nodes(
            &mut self.inner,
            df_result,
            node_type,
            unique_id_field,
            node_title_field,
            conflict_handling,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        self.selection.clear();
        Ok(())
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
        conflict_handling: Option<String>,
    ) -> PyResult<()> {
        let cols: Option<Vec<String>> = match columns {
            Some(py_list) => Some(py_list.iter().map(|item| item.extract::<String>().unwrap()).collect()),
            None => None,
        };

        let df_result = pandas_to_dataframe(
            data,
            &[source_id_field.clone(), target_id_field.clone()],
            cols.as_deref(),
        )?;

        maintain_graph::add_connections(
            &mut self.inner,
            df_result,
            connection_type,
            source_type,
            source_id_field,
            target_type,
            target_id_field,
            source_title_field,
            target_title_field,
            cols,
            conflict_handling,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        self.selection.clear();
        Ok(())
    }

    fn type_filter(
        &mut self, 
        node_type: String,
        sort_spec: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>
    ) -> PyResult<Self> {
        let mut new_graph = self.clone();
        let mut conditions = HashMap::new();
        conditions.insert("type".to_string(), FilterCondition::Equals(Value::String(node_type)));
        
        let sort_fields = if let Some(spec) = sort_spec {
            // Handle both string and PyAny cases
            match spec.extract::<String>() {
                Ok(field) => Some(vec![(field, true)]),  // Default to ascending
                Err(_) => Some(parse_sort_fields(spec, None)?)
            }
        } else {
            None
        };
        filtering_methods::filter_nodes(
            &new_graph.inner, 
            &mut new_graph.selection, 
            conditions, 
            sort_fields, 
            max_nodes
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(new_graph)
    }

    fn filter(&mut self, conditions: &Bound<'_, PyDict>, sort_spec: Option<&Bound<'_, PyAny>>, max_nodes: Option<usize>) -> PyResult<Self> {
        let mut new_graph = self.clone();
        let filter_conditions = pydict_to_filter_conditions(conditions)?;
        let sort_fields = match sort_spec {
            Some(spec) => Some(parse_sort_fields(spec, None)?),
            None => None,
        };
        
        filtering_methods::filter_nodes(&new_graph.inner, &mut new_graph.selection, filter_conditions, sort_fields, max_nodes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(new_graph)
    }

    fn sort(&mut self, sort_spec: &Bound<'_, PyAny>, ascending: Option<bool>) -> PyResult<Self> {
        let mut new_graph = self.clone();
        let sort_fields = parse_sort_fields(sort_spec, ascending)?;
        
        filtering_methods::sort_nodes(&new_graph.inner, &mut new_graph.selection, sort_fields)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(new_graph)
    }

    fn max_nodes(&mut self, max_per_group: usize) -> PyResult<Self> {
        let mut new_graph = self.clone();
        filtering_methods::limit_nodes_per_group(
            &new_graph.inner,
            &mut new_graph.selection,
            max_per_group
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(new_graph)
    }

    fn get_nodes(
        &self, 
        indices: Option<Vec<usize>>, 
        parent_key: Option<&str>,
        parent_info: Option<bool>,
        level_index: Option<usize>
    ) -> PyResult<PyObject> {
        let nodes = data_retrieval::get_nodes(
            &self.inner, 
            &self.selection, 
            None,
            indices.as_deref()
        );
        Python::with_gil(|py| level_nodes_to_pydict(py, &nodes, parent_key, parent_info))
    }
    
    fn get_titles(&self, indices: Option<Vec<usize>>) -> PyResult<PyObject> {
        let values = data_retrieval::get_property_values(
            &self.inner,
            &self.selection,
            None,
            &["title"],
            indices.as_deref()
        );
        Python::with_gil(|py| level_single_values_to_pydict(py, &values))
    }
    
    fn get_properties(&self, properties: Vec<String>, indices: Option<Vec<usize>>) -> PyResult<PyObject> {
        let property_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
        let values = data_retrieval::get_property_values(
            &self.inner,
            &self.selection,
            None,
            &property_refs,
            indices.as_deref()
        );
        Python::with_gil(|py| level_values_to_pydict(py, &values))
    }

    fn get_unique_values(
        &self,
        property: String,
        group_by_parent: Option<bool>,
        level_index: Option<usize>,
        indices: Option<Vec<usize>>
    ) -> PyResult<PyObject> {
        let values = data_retrieval::get_unique_values(
            &self.inner,
            &self.selection,
            &property,
            level_index,
            group_by_parent.unwrap_or(true),
            indices.as_deref()
        );
        
        Python::with_gil(|py| {
            let result = PyDict::new(py);
            for unique_values in values {
                let py_values: Vec<PyObject> = unique_values.values.into_iter()
                    .map(|v| v.to_object(py))
                    .collect();
                result.set_item(unique_values.parent_title, PyList::new(py, &py_values))?;
            }
            Ok(result.to_object(py))
        })
    }
    
    fn traverse(
        &mut self,
        connection_type: String,
        level_index: Option<usize>,
        direction: Option<String>,
        filter_conditions: Option<&Bound<'_, PyDict>>,
        sort_spec: Option<&Bound<'_, PyAny>>,
        max_nodes: Option<usize>,
        new_level: Option<bool>,
    ) -> PyResult<Self> {
        let mut new_graph = self.clone();
        
        let conditions = if let Some(cond) = filter_conditions {
            Some(pydict_to_filter_conditions(cond)?)
        } else {
            None
        };
        
        let sort_fields = if let Some(spec) = sort_spec {
            Some(parse_sort_fields(spec, None)?)
        } else {
            None
        };
    
        traversal_methods::make_traversal(
            &new_graph.inner,
            &mut new_graph.selection,
            connection_type,
            level_index,
            direction,
            conditions,
            sort_fields,
            max_nodes,
            new_level,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(new_graph)
    }

    fn calculate_stats(
        &self,
        property: &str,
        level_index: Option<usize>,
    ) -> PyResult<PyObject> {
        let pairs = statistics_methods::get_parent_child_pairs(&self.selection, level_index);
        let stats = statistics_methods::calculate_property_stats(&self.inner, &pairs, property);
        convert_stats_for_python(stats)
    }

    fn get_schema(&self) -> PyResult<String> {
        let schema_string = debugging::get_schema_string(&self.inner);
        Ok(schema_string)
    }

    fn get_selection(&self) -> PyResult<String> {
        Ok(debugging::get_selection_string(&self.inner, &self.selection))
    }

    fn clear_selection(&mut self) -> PyResult<()> {
        self.selection.clear();
        Ok(())
    }

    fn save(&self, path: &str) -> PyResult<()> {
        save_to_file(&self.inner, path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))
    }
}