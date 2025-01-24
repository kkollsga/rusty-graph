use pyo3::prelude::*;
use pyo3::types::PyList;

pub struct DataInput {
   pub data: Py<PyList>, 
   pub columns: Vec<String>
}