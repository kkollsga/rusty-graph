use chrono::{NaiveDateTime, NaiveDate};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[derive(Debug, Clone)]
pub enum DataType {
    Int,
    Float,
    DateTime, // Assuming DateTime is in "%Y-%m-%d %H:%M:%S" format
    Date,
    // Add more data types as needed
}

pub fn convert_column(column: Vec<String>, data_type: DataType) -> PyResult<Vec<String>> {
    column.into_iter().map(|item_str| {
        match data_type {
            DataType::Int => {
                item_str.parse::<i32>()
                    .map_err(|_| PyTypeError::new_err(format!("Could not convert '{}' to int", item_str)))
                    .map(|_| item_str) // Return the original string if parsing is successful
            },
            DataType::Float => {
                item_str.parse::<f64>()
                    .map_err(|_| PyTypeError::new_err(format!("Could not convert '{}' to float", item_str)))
                    .map(|_| item_str) // Return the original string if parsing is successful
            },
            DataType::DateTime => {
                NaiveDateTime::parse_from_str(&item_str, "%Y-%m-%d %H:%M:%S")
                    .map_err(|_| PyTypeError::new_err(format!("Could not parse '{}' as datetime", item_str)))
                    .map(|_| item_str) // Return the original string if parsing is successful
            },
            DataType::Date => {
                NaiveDate::parse_from_str(&item_str, "%d.%m.%Y")
                    .map_err(|_| PyTypeError::new_err(format!("Could not parse '{}' as date", item_str)))
                    .map(|_| item_str) // Return the original string if parsing is successful
            },
            // Handle other data types as needed
        }
    }).collect() // Collect all results into a Vec<String> or return the first error encountered
}
