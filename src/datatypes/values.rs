// src/datatypes/values.rs
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub enum FilterCondition {
    Equals(Value),
    NotEquals(Value),
    GreaterThan(Value),
    GreaterThanEquals(Value),
    LessThan(Value),
    LessThanEquals(Value),
    In(Vec<Value>),
    Between(Value, Value), // Inclusive range [min, max]
    IsNull,
    IsNotNull,
    Contains(Value),
    StartsWith(Value),
    EndsWith(Value),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum Value {
    UniqueId(u32),
    Int64(i64),
    Float64(f64),
    String(String),
    Boolean(bool),
    DateTime(NaiveDate),
    Null,
}

// Implement Eq for Value
impl Eq for Value {
    // We need this empty impl because we already have PartialEq
    // and all variants can be exactly equal except Float64,
    // which we handle specially in PartialEq
}

// Implement Hash for Value
impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // First hash discriminant to differentiate variants
        std::mem::discriminant(self).hash(state);

        // Then hash the contained value
        match self {
            Value::UniqueId(v) => v.hash(state),
            Value::Int64(v) => v.hash(state),
            Value::Float64(v) => {
                // Special handling for NaN and -0.0
                if v.is_nan() {
                    // Hash all NaN values the same
                    f64::NAN.to_bits().hash(state)
                } else {
                    // Handle -0.0 == 0.0
                    if *v == 0.0 {
                        0.0f64.to_bits().hash(state)
                    } else {
                        v.to_bits().hash(state)
                    }
                }
            }
            Value::String(v) => v.hash(state),
            Value::Boolean(v) => v.hash(state),
            Value::DateTime(v) => v.hash(state),
            Value::Null => 0.hash(state),
        }
    }
}

impl Value {
    pub fn as_string(&self) -> Option<String> {
        match self {
            Value::String(s) => Some(s.clone()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType {
    UniqueId,
    Int64,
    Float64,
    String,
    Boolean,
    DateTime,
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_str = match self {
            ColumnType::UniqueId => "UniqueId",
            ColumnType::Int64 => "Int64",
            ColumnType::Float64 => "Float64",
            ColumnType::String => "String",
            ColumnType::Boolean => "Boolean",
            ColumnType::DateTime => "DateTime",
        };
        write!(f, "{}", type_str)
    }
}

#[derive(Debug)]
pub struct Column {
    pub(crate) name: String,
    pub(crate) col_type: ColumnType,
    pub(crate) data: ColumnData,
}

#[derive(Debug)]
pub enum ColumnData {
    UniqueId(Vec<Option<u32>>),
    Int64(Vec<Option<i64>>),
    Float64(Vec<Option<f64>>),
    String(Vec<Option<String>>),
    Boolean(Vec<Option<bool>>),
    DateTime(Vec<Option<NaiveDate>>),
}

#[derive(Debug)]
pub struct DataFrame {
    columns: Vec<Column>,
    column_indices: HashMap<String, usize>,
}

impl Column {
    fn get_value(&self, row_idx: usize) -> Option<Value> {
        match &self.data {
            ColumnData::UniqueId(vec) => vec.get(row_idx)?.map(Value::UniqueId),
            ColumnData::Int64(vec) => vec.get(row_idx)?.map(Value::Int64),
            ColumnData::Float64(vec) => vec.get(row_idx)?.map(Value::Float64),
            ColumnData::String(vec) => vec.get(row_idx)?.as_ref().map(|s| Value::String(s.clone())),
            ColumnData::Boolean(vec) => vec.get(row_idx)?.map(Value::Boolean),
            ColumnData::DateTime(vec) => vec.get(row_idx)?.map(Value::DateTime),
        }
    }

    fn len(&self) -> usize {
        match &self.data {
            ColumnData::UniqueId(vec) => vec.len(),
            ColumnData::Int64(vec) => vec.len(),
            ColumnData::Float64(vec) => vec.len(),
            ColumnData::String(vec) => vec.len(),
            ColumnData::Boolean(vec) => vec.len(),
            ColumnData::DateTime(vec) => vec.len(),
        }
    }
}

impl DataFrame {
    pub fn new(columns: Vec<(String, ColumnType)>) -> Self {
        let mut column_indices = HashMap::with_capacity(columns.len());
        let columns: Vec<Column> = columns
            .into_iter()
            .enumerate()
            .map(|(idx, (name, col_type))| {
                let data = match col_type {
                    ColumnType::UniqueId => ColumnData::UniqueId(Vec::new()),
                    ColumnType::Int64 => ColumnData::Int64(Vec::new()),
                    ColumnType::Float64 => ColumnData::Float64(Vec::new()),
                    ColumnType::String => ColumnData::String(Vec::new()),
                    ColumnType::Boolean => ColumnData::Boolean(Vec::new()),
                    ColumnType::DateTime => ColumnData::DateTime(Vec::new()),
                };
                column_indices.insert(name.clone(), idx);
                Column {
                    name,
                    col_type,
                    data,
                }
            })
            .collect();

        DataFrame {
            columns,
            column_indices,
        }
    }

    pub fn get_value(&self, row: usize, column: &str) -> Option<Value> {
        self.column_indices
            .get(column)
            .and_then(|&idx| self.columns.get(idx))
            .and_then(|col| col.get_value(row))
    }

    pub fn get_value_by_index(&self, row_idx: usize, col_idx: usize) -> Option<Value> {
        self.columns
            .get(col_idx)
            .and_then(|col| col.get_value(row_idx))
    }

    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        self.column_indices.get(name).copied()
    }

    pub fn verify_column(&self, name: &str) -> bool {
        self.column_indices.contains_key(name)
    }

    pub fn row_count(&self) -> usize {
        self.columns.first().map_or(0, |col| col.len())
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn get_column_names(&self) -> Vec<String> {
        self.columns.iter().map(|col| col.name.clone()).collect()
    }

    pub fn get_column_type(&self, col_name: &str) -> ColumnType {
        self.column_indices
            .get(col_name)
            .and_then(|&idx| self.columns.get(idx))
            .map(|col| col.col_type.clone())
            .unwrap_or_else(|| panic!("Column {} not found", col_name))
    }

    pub fn add_column(
        &mut self,
        name: String,
        col_type: ColumnType,
        data: ColumnData,
    ) -> Result<(), String> {
        if self.column_indices.contains_key(&name) {
            return Err(format!("Column {} already exists", name));
        }

        // Validate that the provided data matches the column type
        match (&col_type, &data) {
            (ColumnType::UniqueId, ColumnData::UniqueId(_))
            | (ColumnType::Int64, ColumnData::Int64(_))
            | (ColumnType::Float64, ColumnData::Float64(_))
            | (ColumnType::String, ColumnData::String(_))
            | (ColumnType::Boolean, ColumnData::Boolean(_))
            | (ColumnType::DateTime, ColumnData::DateTime(_)) => (),
            _ => return Err(format!("Data type mismatch for column {}", name)),
        }

        let idx = self.columns.len();
        self.column_indices.insert(name.clone(), idx);
        self.columns.push(Column {
            name,
            col_type,
            data,
        });

        Ok(())
    }
}

impl std::fmt::Display for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let row_limit = 10.min(self.row_count());
        let columns = self.get_column_names();

        // Determine max width for each column
        let mut col_widths: Vec<usize> = columns.iter().map(|col| col.len()).collect();

        // Adjust widths based on values and column types
        for (col_idx, col) in self.columns.iter().enumerate() {
            // Include column type width
            let type_width = format_col_type(&col.col_type).len();
            col_widths[col_idx] = col_widths[col_idx].max(type_width);

            // Include value widths
            for row_idx in 0..row_limit {
                if let Some(value) = col.get_value(row_idx) {
                    col_widths[col_idx] = col_widths[col_idx].max(format_value(&value).len());
                }
            }
        }

        // Format helper
        let format_row = |values: Vec<String>| -> String {
            values
                .into_iter()
                .enumerate()
                .map(|(i, val)| format!(" {:^width$} ", val, width = col_widths[i]))
                .collect::<Vec<_>>()
                .join("|")
        };

        // Print headers
        writeln!(f, "\n| #  |{}|", format_row(columns))?;

        // Print column types
        let type_row: Vec<String> = self
            .columns
            .iter()
            .map(|col| format_col_type(&col.col_type))
            .collect();
        writeln!(f, "|    |{}|", format_row(type_row))?;

        // Print separator
        let separator = col_widths
            .iter()
            .map(|w| format!("{:-^width$}", "-", width = w + 2))
            .collect::<Vec<_>>()
            .join("|");
        writeln!(f, "|----|{}|", separator)?;

        // Print data rows
        for row_idx in 0..row_limit {
            let row_data: Vec<String> = (0..self.column_count())
                .map(|col_idx| {
                    format_value(
                        &self
                            .get_value_by_index(row_idx, col_idx)
                            .unwrap_or(Value::Null),
                    )
                })
                .collect();
            writeln!(f, "| {:^2} |{}|", row_idx, format_row(row_data))?;
        }

        // Show if there are more rows
        if self.row_count() > row_limit {
            let more_row = format_row(col_widths.iter().map(|_| "...".to_string()).collect());
            writeln!(f, "| .. |{}|", more_row)?;
        }

        Ok(())
    }
}

pub fn format_value(value: &Value) -> String {
    match value {
        Value::UniqueId(v) => format!("{}", v),
        Value::Int64(v) => format!("{}", v),
        Value::Float64(v) => {
            if v.is_nan() {
                "NULL".to_string()
            } else {
                format!("{:.2}", v)
            }
        }
        Value::String(v) => format!("\"{}\"", v),
        Value::Boolean(v) => format!("{}", v),
        Value::DateTime(v) => format!("\"{}\"", v.format("%Y-%m-%d")),
        Value::Null => "NULL".to_string(),
    }
}

fn format_col_type(col_type: &ColumnType) -> String {
    match col_type {
        ColumnType::UniqueId => "uID",
        ColumnType::Int64 => "i64",
        ColumnType::Float64 => "f64",
        ColumnType::String => "str",
        ColumnType::Boolean => "bool",
        ColumnType::DateTime => "datetime",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Value::as_string
    // ========================================================================

    #[test]
    fn test_as_string_with_string_value() {
        let v = Value::String("hello".to_string());
        assert_eq!(v.as_string(), Some("hello".to_string()));
    }

    #[test]
    fn test_as_string_with_non_string_values() {
        assert_eq!(Value::Int64(42).as_string(), None);
        assert_eq!(Value::Float64(3.14).as_string(), None);
        assert_eq!(Value::Boolean(true).as_string(), None);
        assert_eq!(Value::Null.as_string(), None);
        assert_eq!(Value::UniqueId(1).as_string(), None);
    }

    // ========================================================================
    // Value equality and hash
    // ========================================================================

    #[test]
    fn test_value_equality_same_types() {
        assert_eq!(Value::Int64(42), Value::Int64(42));
        assert_eq!(Value::Float64(3.14), Value::Float64(3.14));
        assert_eq!(
            Value::String("a".to_string()),
            Value::String("a".to_string())
        );
        assert_eq!(Value::Boolean(true), Value::Boolean(true));
        assert_eq!(Value::Null, Value::Null);
        assert_eq!(Value::UniqueId(5), Value::UniqueId(5));
    }

    #[test]
    fn test_value_inequality() {
        assert_ne!(Value::Int64(1), Value::Int64(2));
        assert_ne!(
            Value::String("a".to_string()),
            Value::String("b".to_string())
        );
        assert_ne!(Value::Boolean(true), Value::Boolean(false));
    }

    #[test]
    fn test_value_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Value::Int64(42));
        set.insert(Value::Int64(42)); // duplicate
        assert_eq!(set.len(), 1);

        set.insert(Value::String("test".to_string()));
        assert_eq!(set.len(), 2);

        set.insert(Value::Null);
        set.insert(Value::Null); // duplicate
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_float_hash_negative_zero() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Value::Float64(0.0));
        set.insert(Value::Float64(-0.0));
        // 0.0 and -0.0 should hash the same
        assert_eq!(set.len(), 1);
    }

    // ========================================================================
    // format_value
    // ========================================================================

    #[test]
    fn test_format_value_types() {
        assert_eq!(format_value(&Value::UniqueId(42)), "42");
        assert_eq!(format_value(&Value::Int64(-5)), "-5");
        assert_eq!(format_value(&Value::Float64(3.14)), "3.14");
        assert_eq!(format_value(&Value::String("hi".to_string())), "\"hi\"");
        assert_eq!(format_value(&Value::Boolean(true)), "true");
        assert_eq!(format_value(&Value::Null), "NULL");
    }

    #[test]
    fn test_format_value_nan_is_null() {
        assert_eq!(format_value(&Value::Float64(f64::NAN)), "NULL");
    }

    // ========================================================================
    // ColumnType Display
    // ========================================================================

    #[test]
    fn test_column_type_display() {
        assert_eq!(format!("{}", ColumnType::UniqueId), "UniqueId");
        assert_eq!(format!("{}", ColumnType::Int64), "Int64");
        assert_eq!(format!("{}", ColumnType::Float64), "Float64");
        assert_eq!(format!("{}", ColumnType::String), "String");
        assert_eq!(format!("{}", ColumnType::Boolean), "Boolean");
        assert_eq!(format!("{}", ColumnType::DateTime), "DateTime");
    }

    // ========================================================================
    // DataFrame
    // ========================================================================

    #[test]
    fn test_dataframe_new_empty() {
        let df = DataFrame::new(vec![
            ("id".to_string(), ColumnType::Int64),
            ("name".to_string(), ColumnType::String),
        ]);
        assert_eq!(df.row_count(), 0);
        assert_eq!(df.column_count(), 2);
        assert!(df.verify_column("id"));
        assert!(df.verify_column("name"));
        assert!(!df.verify_column("missing"));
    }

    #[test]
    fn test_dataframe_column_names() {
        let df = DataFrame::new(vec![
            ("a".to_string(), ColumnType::Int64),
            ("b".to_string(), ColumnType::String),
        ]);
        let names = df.get_column_names();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_dataframe_column_type() {
        let df = DataFrame::new(vec![
            ("id".to_string(), ColumnType::Int64),
            ("name".to_string(), ColumnType::String),
        ]);
        assert_eq!(df.get_column_type("id"), ColumnType::Int64);
        assert_eq!(df.get_column_type("name"), ColumnType::String);
    }

    #[test]
    fn test_dataframe_add_column() {
        let mut df = DataFrame::new(vec![("id".to_string(), ColumnType::Int64)]);
        let result = df.add_column(
            "name".to_string(),
            ColumnType::String,
            ColumnData::String(vec![]),
        );
        assert!(result.is_ok());
        assert_eq!(df.column_count(), 2);
    }

    #[test]
    fn test_dataframe_add_duplicate_column() {
        let mut df = DataFrame::new(vec![("id".to_string(), ColumnType::Int64)]);
        let result = df.add_column(
            "id".to_string(),
            ColumnType::Int64,
            ColumnData::Int64(vec![]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dataframe_add_column_type_mismatch() {
        let mut df = DataFrame::new(vec![]);
        let result = df.add_column(
            "x".to_string(),
            ColumnType::Int64,
            ColumnData::String(vec![]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dataframe_get_column_index() {
        let df = DataFrame::new(vec![
            ("a".to_string(), ColumnType::Int64),
            ("b".to_string(), ColumnType::String),
        ]);
        assert_eq!(df.get_column_index("a"), Some(0));
        assert_eq!(df.get_column_index("b"), Some(1));
        assert_eq!(df.get_column_index("c"), None);
    }
}
