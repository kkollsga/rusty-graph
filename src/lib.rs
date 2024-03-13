mod node;
mod relation;
mod graph;
mod utils;

use pyo3::prelude::*;
use utils::DataType;
use graph::KnowledgeGraph;

#[pymodule]
fn rusty_graph(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<KnowledgeGraph>()?;
    
    // Expose DataType enum directly to Python under the same module
    m.add_class::<DataTypeWrapper>()?;

    Ok(())
}

/// Wrapper for DataType to expose it to Python
#[pyclass(name = "DataType")]
pub enum DataTypeWrapper {
    Int,
    Float,
    DateTime,
    Date,
}

#[pymethods]
impl DataTypeWrapper {
    #[staticmethod]
    fn int() -> Self {
        DataTypeWrapper::Int
    }

    #[staticmethod]
    fn float() -> Self {
        DataTypeWrapper::Float
    }

    #[staticmethod]
    fn datetime() -> Self {
        DataTypeWrapper::DateTime
    }

    #[staticmethod]
    fn date() -> Self {
        DataTypeWrapper::Date
    }
    // Add more data types as needed
}

// Implement conversion from DataTypeWrapper to DataType
impl From<DataTypeWrapper> for DataType {
    fn from(wrapper: DataTypeWrapper) -> Self {
        match wrapper {
            DataTypeWrapper::Int => DataType::Int,
            DataTypeWrapper::Float => DataType::Float,
            DataTypeWrapper::DateTime => DataType::DateTime,
            DataTypeWrapper::Date => DataType::Date,
            // Handle other data types as needed
        }
    }
}
