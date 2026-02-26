// Index Management #[pymethods] — extracted from mod.rs

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{get_graph_mut, KnowledgeGraph};

#[pymethods]
impl KnowledgeGraph {
    // ========================================================================
    // Index Management Methods
    // ========================================================================

    /// Create an index on a property for a specific node type.
    ///
    /// Indexes dramatically speed up equality filters on the indexed property.
    /// Once created, the index is automatically used by where() operations.
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
    ///     graph.select('Prospect').where({'geoprovince': 'North Sea'})
    ///     ```
    fn create_index(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        property: &str,
    ) -> PyResult<Py<PyAny>> {
        let graph = get_graph_mut(&mut self.inner);
        let unique_values = graph.create_index(node_type, property);

        let result_dict = PyDict::new(py);
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
        let removed = get_graph_mut(&mut self.inner).drop_index(node_type, property);
        Ok(removed)
    }

    /// List all existing indexes.
    ///
    /// Returns:
    ///     List of dictionaries with 'type' and 'property' keys
    ///
    /// Example:
    ///     ```python
    ///     indexes = graph.list_indexes()
    ///     for idx in indexes:
    ///         print(f"{idx['type']}.{idx['property']}")
    ///     ```
    fn list_indexes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let indexes = self.inner.list_indexes();

        let result_list = pyo3::types::PyList::empty(py);
        for (node_type, property) in indexes {
            let idx_dict = PyDict::new(py);
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
    fn index_stats(&self, py: Python<'_>, node_type: &str, property: &str) -> PyResult<Py<PyAny>> {
        match self.inner.get_index_stats(node_type, property) {
            Some(stats) => {
                let result_dict = PyDict::new(py);
                result_dict.set_item("node_type", node_type)?;
                result_dict.set_item("property", property)?;
                result_dict.set_item("unique_values", stats.unique_values)?;
                result_dict.set_item("total_entries", stats.total_entries)?;
                result_dict.set_item("avg_entries_per_value", stats.avg_entries_per_value)?;
                Ok(result_dict.into())
            }
            None => Ok(py.None()),
        }
    }

    /// Create a range index (B-Tree) on a property for a specific node type.
    ///
    /// Range indexes enable efficient range queries (>, >=, <, <=, BETWEEN)
    /// using ``where()`` with comparison conditions.
    ///
    /// Args:
    ///     node_type: The type of nodes to index.
    ///     property: The property name to index.
    ///
    /// Returns:
    ///     dict with keys: ``type``, ``property``, ``unique_values``, ``created``
    ///
    /// Example:
    ///     ```python
    ///     graph.create_range_index('Person', 'age')
    ///     # Now range queries on age use the B-Tree index:
    ///     result = graph.select('Person').where({'age': {'>': 25}}).collect()
    ///     ```
    fn create_range_index(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        property: &str,
    ) -> PyResult<Py<PyAny>> {
        let graph = get_graph_mut(&mut self.inner);
        let unique_values = graph.create_range_index(node_type, property);

        let result_dict = PyDict::new(py);
        result_dict.set_item("node_type", node_type)?;
        result_dict.set_item("property", property)?;
        result_dict.set_item("unique_values", unique_values)?;
        result_dict.set_item("created", true)?;

        Ok(result_dict.into())
    }

    /// Drop a range index.
    ///
    /// Args:
    ///     node_type: The type of nodes.
    ///     property: The property name.
    ///
    /// Returns:
    ///     True if index existed and was removed, False otherwise.
    fn drop_range_index(&mut self, node_type: &str, property: &str) -> PyResult<bool> {
        let removed = get_graph_mut(&mut self.inner).drop_range_index(node_type, property);
        Ok(removed)
    }

    /// Rebuild all indexes.
    ///
    /// Call this after batch updates to ensure indexes are current.
    ///
    /// Returns:
    ///     Number of indexes rebuilt
    fn rebuild_indexes(&mut self) -> PyResult<usize> {
        let graph = get_graph_mut(&mut self.inner);

        let index_keys: Vec<_> = graph.property_indices.keys().cloned().collect();

        for (node_type, property) in &index_keys {
            graph.create_index(node_type, property);
        }

        Ok(index_keys.len())
    }

    // ========================================================================
    // Composite Index Methods
    // ========================================================================

    /// Create a composite index on multiple properties for efficient multi-field queries.
    ///
    /// Composite indexes are useful when you frequently filter on the same combination
    /// of fields together. They provide O(1) lookup for exact matches on all indexed fields.
    ///
    /// Args:
    ///     node_type: The type of nodes to index
    ///     properties: A list of property names to include in the composite index
    ///
    /// Returns:
    ///     Number of unique value combinations indexed
    ///
    /// Example:
    ///     ```python
    ///     # Create an index for queries filtering on both 'geoprovince' and 'status'
    ///     graph.create_composite_index('Prospect', ['geoprovince', 'status'])
    ///
    ///     # Now this filter is very fast:
    ///     graph.select('Prospect').where({
    ///         'geoprovince': 'N3',
    ///         'status': 'Active'
    ///     })
    ///     ```
    fn create_composite_index(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        properties: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        let graph = get_graph_mut(&mut self.inner);

        let props_refs: Vec<&str> = properties.iter().map(|s| s.as_str()).collect();
        let unique_values = graph.create_composite_index(node_type, &props_refs);
        let result_dict = PyDict::new(py);
        result_dict.set_item("node_type", node_type)?;
        result_dict.set_item("properties", properties)?;
        result_dict.set_item("unique_combinations", unique_values)?;

        Ok(result_dict.into())
    }

    /// Drop a composite index.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     properties: The list of property names in the composite index
    ///
    /// Returns:
    ///     True if index existed and was dropped, False otherwise
    fn drop_composite_index(&mut self, node_type: &str, properties: Vec<String>) -> PyResult<bool> {
        let removed = get_graph_mut(&mut self.inner).drop_composite_index(node_type, &properties);
        Ok(removed)
    }

    /// List all composite indexes in the graph.
    ///
    /// Returns:
    ///     A list of dicts with 'type' and 'properties' keys
    fn list_composite_indexes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let indexes = self.inner.list_composite_indexes();

        let result_list = pyo3::types::PyList::empty(py);
        for (node_type, properties) in indexes {
            let idx_dict = PyDict::new(py);
            idx_dict.set_item("node_type", node_type)?;
            idx_dict.set_item("properties", properties)?;
            result_list.append(idx_dict)?;
        }

        Ok(result_list.into())
    }

    /// Check if a composite index exists.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     properties: The list of property names in the composite index
    ///
    /// Returns:
    ///     True if index exists, False otherwise
    fn has_composite_index(&self, node_type: &str, properties: Vec<String>) -> bool {
        self.inner.has_composite_index(node_type, &properties)
    }

    /// Get statistics about a composite index.
    ///
    /// Args:
    ///     node_type: The type of nodes
    ///     properties: The list of property names in the composite index
    ///
    /// Returns:
    ///     Dictionary with index statistics, or None if index doesn't exist
    fn composite_index_stats(
        &self,
        py: Python<'_>,
        node_type: &str,
        properties: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        match self.inner.get_composite_index_stats(node_type, &properties) {
            Some(stats) => {
                let result_dict = PyDict::new(py);
                result_dict.set_item("node_type", node_type)?;
                result_dict.set_item("properties", properties)?;
                result_dict.set_item("unique_combinations", stats.unique_values)?;
                result_dict.set_item("total_entries", stats.total_entries)?;
                result_dict.set_item("avg_entries_per_combination", stats.avg_entries_per_value)?;
                Ok(result_dict.into())
            }
            None => Ok(py.None()),
        }
    }
}
