// src/graph/column_store.rs
//
// Per-type columnar property storage. Each node type gets a ColumnStore
// containing one TypedColumn per property key. Rows map 1:1 to nodes
// via a u32 row_id stored in PropertyStorage::Columnar.
//
// TypedColumn uses MmapOrVec<T> for fixed-size types (i64, f64, u32, bool, i32)
// and MmapBytes for string data. Mixed columns stay heap-only (Vec<Value>).

use crate::datatypes::values::Value;
use crate::graph::mmap_vec::{MmapBytes, MmapOrVec};
use crate::graph::schema::{InternedKey, StringInterner, TypeSchema};
use chrono::NaiveDate;
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

// ─── TypedColumn ─────────────────────────────────────────────────────────────

/// A single column of homogeneously-typed property values.
/// Column type is determined from `node_type_metadata` at construction time.
/// Falls back to `Mixed` for heterogeneous or unknown types.
///
/// Fixed-size columns use `MmapOrVec<T>` which can be heap- or file-backed.
/// String columns use `MmapOrVec<u64>` for offsets and `MmapBytes` for UTF-8 data.
/// Mixed columns use plain `Vec<Value>` (not mmap-eligible).
#[derive(Debug, Clone)]
pub enum TypedColumn {
    Int64 {
        data: MmapOrVec<i64>,
        nulls: MmapOrVec<u8>, // 0 = non-null, 1 = null
    },
    Float64 {
        data: MmapOrVec<f64>,
        nulls: MmapOrVec<u8>,
    },
    UniqueId {
        data: MmapOrVec<u32>,
        nulls: MmapOrVec<u8>,
    },
    Bool {
        data: MmapOrVec<u8>, // 0 = false, 1 = true
        nulls: MmapOrVec<u8>,
    },
    /// Days since Unix epoch (1970-01-01)
    Date {
        data: MmapOrVec<i32>,
        nulls: MmapOrVec<u8>,
    },
    /// Offset-based string storage: `offsets[i]..offsets[i+1]` is the byte range in `data`.
    Str {
        offsets: MmapOrVec<u64>,
        data: MmapBytes,
        nulls: MmapOrVec<u8>,
    },
    /// Fallback for heterogeneous columns — stores boxed Values directly.
    /// Cannot be mmap'd, but preserves correctness.
    Mixed { data: Vec<Value> },
}

/// Number of days from the Unix epoch to chrono's internal epoch.
const UNIX_EPOCH_DATE: NaiveDate = match NaiveDate::from_ymd_opt(1970, 1, 1) {
    Some(d) => d,
    None => unreachable!(),
};

impl TypedColumn {
    /// Create an empty column of the given type based on metadata type string.
    /// Matching is case-insensitive (metadata stores "Int64", "String", etc.).
    pub fn from_type_str(type_str: &str) -> Self {
        match type_str.to_ascii_lowercase().as_str() {
            "int64" => TypedColumn::Int64 {
                data: MmapOrVec::new(),
                nulls: MmapOrVec::new(),
            },
            "float64" => TypedColumn::Float64 {
                data: MmapOrVec::new(),
                nulls: MmapOrVec::new(),
            },
            "uniqueid" => TypedColumn::UniqueId {
                data: MmapOrVec::new(),
                nulls: MmapOrVec::new(),
            },
            "bool" | "boolean" => TypedColumn::Bool {
                data: MmapOrVec::new(),
                nulls: MmapOrVec::new(),
            },
            "date" | "datetime" => TypedColumn::Date {
                data: MmapOrVec::new(),
                nulls: MmapOrVec::new(),
            },
            "string" => TypedColumn::Str {
                offsets: {
                    let mut o = MmapOrVec::new();
                    o.push(0u64);
                    o
                },
                data: MmapBytes::new(),
                nulls: MmapOrVec::new(),
            },
            _ => TypedColumn::Mixed { data: Vec::new() },
        }
    }

    /// Number of rows in this column.
    pub fn len(&self) -> usize {
        match self {
            TypedColumn::Int64 { nulls, .. }
            | TypedColumn::Float64 { nulls, .. }
            | TypedColumn::UniqueId { nulls, .. }
            | TypedColumn::Bool { nulls, .. }
            | TypedColumn::Date { nulls, .. }
            | TypedColumn::Str { nulls, .. } => nulls.len(),
            TypedColumn::Mixed { data } => data.len(),
        }
    }

    /// Push a value onto this column. Returns Ok(()) on success,
    /// Err(value) if the value type doesn't match (caller should demote to Mixed).
    pub fn push(&mut self, value: &Value) -> Result<(), ()> {
        match (self, value) {
            (TypedColumn::Int64 { data, nulls }, Value::Int64(v)) => {
                data.push(*v);
                nulls.push(0);
            }
            (TypedColumn::Int64 { data, nulls }, Value::Null) => {
                data.push(0);
                nulls.push(1);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Float64(v)) => {
                data.push(*v);
                nulls.push(0);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Int64(v)) => {
                // Allow int→float promotion (common from pandas)
                data.push(*v as f64);
                nulls.push(0);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Null) => {
                data.push(0.0);
                nulls.push(1);
            }
            (TypedColumn::UniqueId { data, nulls }, Value::UniqueId(v)) => {
                data.push(*v);
                nulls.push(0);
            }
            (TypedColumn::UniqueId { data, nulls }, Value::Null) => {
                data.push(0);
                nulls.push(1);
            }
            (TypedColumn::Bool { data, nulls }, Value::Boolean(v)) => {
                data.push(*v as u8);
                nulls.push(0);
            }
            (TypedColumn::Bool { data, nulls }, Value::Null) => {
                data.push(0);
                nulls.push(1);
            }
            (TypedColumn::Date { data, nulls }, Value::DateTime(d)) => {
                let days = (*d - UNIX_EPOCH_DATE).num_days() as i32;
                data.push(days);
                nulls.push(0);
            }
            (TypedColumn::Date { data, nulls }, Value::Null) => {
                data.push(0);
                nulls.push(1);
            }
            (
                TypedColumn::Str {
                    offsets,
                    data,
                    nulls,
                },
                Value::String(s),
            ) => {
                data.extend(s.as_bytes());
                offsets.push(data.len() as u64);
                nulls.push(0);
            }
            (TypedColumn::Str { offsets, nulls, .. }, Value::Null) => {
                // Null string: push same offset (zero-length range)
                let last = if !offsets.is_empty() {
                    offsets.get(offsets.len() - 1)
                } else {
                    0
                };
                offsets.push(last);
                nulls.push(1);
            }
            (TypedColumn::Mixed { data }, value) => {
                data.push(value.clone());
            }
            _ => return Err(()),
        }
        Ok(())
    }

    /// Read the value at the given row index.
    pub fn get(&self, row: u32) -> Option<Value> {
        let idx = row as usize;
        match self {
            TypedColumn::Int64 { data, nulls } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                Some(Value::Int64(data.get(idx)))
            }
            TypedColumn::Float64 { data, nulls } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                Some(Value::Float64(data.get(idx)))
            }
            TypedColumn::UniqueId { data, nulls } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                Some(Value::UniqueId(data.get(idx)))
            }
            TypedColumn::Bool { data, nulls } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                Some(Value::Boolean(data.get(idx) != 0))
            }
            TypedColumn::Date { data, nulls } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                let date = UNIX_EPOCH_DATE + chrono::Duration::days(data.get(idx) as i64);
                Some(Value::DateTime(date))
            }
            TypedColumn::Str {
                offsets,
                data,
                nulls,
            } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                let start = offsets.get(idx) as usize;
                let end = offsets.get(idx + 1) as usize;
                let bytes = data.slice(start, end);
                // SAFETY: We only write valid UTF-8 via Value::String, so skip validation.
                let s = unsafe { String::from_utf8_unchecked(bytes.to_vec()) };
                Some(Value::String(s))
            }
            TypedColumn::Mixed { data } => {
                let val = data.get(idx)?;
                if matches!(val, Value::Null) {
                    return None;
                }
                Some(val.clone())
            }
        }
    }

    /// Update the value at the given row index.
    /// Returns Ok(()) on success, Err(()) on type mismatch.
    pub fn set(&mut self, row: u32, value: &Value) -> Result<(), ()> {
        let idx = row as usize;
        match (self, value) {
            (TypedColumn::Int64 { data, nulls }, Value::Int64(v)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, *v);
                nulls.set(idx, 0);
            }
            (TypedColumn::Int64 { data, nulls }, Value::Null) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, 0);
                nulls.set(idx, 1);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Float64(v)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, *v);
                nulls.set(idx, 0);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Int64(v)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, *v as f64);
                nulls.set(idx, 0);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Null) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, 0.0);
                nulls.set(idx, 1);
            }
            (TypedColumn::UniqueId { data, nulls }, Value::UniqueId(v)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, *v);
                nulls.set(idx, 0);
            }
            (TypedColumn::UniqueId { data, nulls }, Value::Null) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, 0);
                nulls.set(idx, 1);
            }
            (TypedColumn::Bool { data, nulls }, Value::Boolean(v)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, *v as u8);
                nulls.set(idx, 0);
            }
            (TypedColumn::Bool { data, nulls }, Value::Null) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, 0);
                nulls.set(idx, 1);
            }
            (TypedColumn::Date { data, nulls }, Value::DateTime(d)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, (*d - UNIX_EPOCH_DATE).num_days() as i32);
                nulls.set(idx, 0);
            }
            (TypedColumn::Date { data, nulls }, Value::Null) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, 0);
                nulls.set(idx, 1);
            }
            (
                TypedColumn::Str {
                    offsets,
                    data,
                    nulls,
                },
                Value::String(s),
            ) => {
                if idx >= nulls.len() {
                    return Err(());
                }
                // Append new string data (old data becomes a hole — compacted on save)
                let new_start = data.len() as u64;
                data.extend(s.as_bytes());
                offsets.set(idx, new_start);
                // Shift idx+1 offset to new end
                offsets.set(idx + 1, data.len() as u64);
                nulls.set(idx, 0);
            }
            (TypedColumn::Str { offsets, nulls, .. }, Value::Null) => {
                if idx >= nulls.len() {
                    return Err(());
                }
                let last = if !offsets.is_empty() {
                    offsets.get(offsets.len() - 1)
                } else {
                    0
                };
                offsets.set(idx, last);
                offsets.set(idx + 1, last);
                nulls.set(idx, 1);
            }
            (TypedColumn::Mixed { data }, value) => {
                if idx >= data.len() {
                    return Err(());
                }
                data[idx] = value.clone();
            }
            _ => return Err(()),
        }
        Ok(())
    }

    /// Push a null value for this column type.
    pub fn push_null(&mut self) {
        let _ = self.push(&Value::Null);
    }

    /// Whether this column can be mmap'd (all non-Mixed types can).
    #[allow(dead_code)]
    pub fn can_mmap(&self) -> bool {
        !matches!(self, TypedColumn::Mixed { .. })
    }

    /// Whether this column's data is currently file-backed.
    #[allow(dead_code)]
    pub fn is_mapped(&self) -> bool {
        match self {
            TypedColumn::Int64 { data, .. } => data.is_mapped(),
            TypedColumn::Float64 { data, .. } => data.is_mapped(),
            TypedColumn::UniqueId { data, .. } => data.is_mapped(),
            TypedColumn::Bool { data, .. } => data.is_mapped(),
            TypedColumn::Date { data, .. } => data.is_mapped(),
            TypedColumn::Str { data, .. } => data.is_mapped(),
            TypedColumn::Mixed { .. } => false,
        }
    }

    /// Heap-resident bytes across all sub-buffers (0 if fully mmap'd).
    pub fn heap_bytes(&self) -> usize {
        match self {
            TypedColumn::Int64 { data, nulls } => data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::Float64 { data, nulls } => data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::UniqueId { data, nulls } => data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::Bool { data, nulls } => data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::Date { data, nulls } => data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::Str {
                offsets,
                data,
                nulls,
            } => offsets.heap_bytes() + data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::Mixed { data } => data.len() * std::mem::size_of::<Value>(),
        }
    }

    /// Materialize this column's data to file-backed mmap.
    /// `base_path` is the directory; files are named `{col_name}.{ext}`.
    #[allow(dead_code)]
    pub fn materialize_to_file(&mut self, base_dir: &Path, col_name: &str) -> io::Result<()> {
        match self {
            TypedColumn::Int64 { data, nulls } => {
                data.materialize_to_file(&base_dir.join(format!("{col_name}.i64")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::Float64 { data, nulls } => {
                data.materialize_to_file(&base_dir.join(format!("{col_name}.f64")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::UniqueId { data, nulls } => {
                data.materialize_to_file(&base_dir.join(format!("{col_name}.u32")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::Bool { data, nulls } => {
                data.materialize_to_file(&base_dir.join(format!("{col_name}.bool")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::Date { data, nulls } => {
                data.materialize_to_file(&base_dir.join(format!("{col_name}.i32")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::Str {
                offsets,
                data,
                nulls,
            } => {
                offsets.materialize_to_file(&base_dir.join(format!("{col_name}.off")))?;
                data.materialize_to_file(&base_dir.join(format!("{col_name}.str")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::Mixed { .. } => {
                // Mixed columns cannot be mmap'd — no-op
            }
        }
        Ok(())
    }

    /// Convert this column back to heap-backed storage.
    #[allow(dead_code)]
    pub fn materialize_to_heap(&mut self) {
        match self {
            TypedColumn::Int64 { data, nulls } => {
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::Float64 { data, nulls } => {
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::UniqueId { data, nulls } => {
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::Bool { data, nulls } => {
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::Date { data, nulls } => {
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::Str {
                offsets,
                data,
                nulls,
            } => {
                offsets.materialize_to_heap();
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::Mixed { .. } => {} // already heap
        }
    }

    /// Write column data to a writer (for v3 packed format).
    /// Writes data bytes, then null bytes. For strings: offsets + data + nulls.
    /// For mixed: bincode-serialized Vec<Value>.
    pub fn write_to(&self, writer: &mut impl io::Write) -> io::Result<()> {
        match self {
            TypedColumn::Int64 { data, nulls } => {
                data.write_to(writer)?;
                nulls.write_to(writer)?;
            }
            TypedColumn::Float64 { data, nulls } => {
                data.write_to(writer)?;
                nulls.write_to(writer)?;
            }
            TypedColumn::UniqueId { data, nulls } => {
                data.write_to(writer)?;
                nulls.write_to(writer)?;
            }
            TypedColumn::Bool { data, nulls } => {
                data.write_to(writer)?;
                nulls.write_to(writer)?;
            }
            TypedColumn::Date { data, nulls } => {
                data.write_to(writer)?;
                nulls.write_to(writer)?;
            }
            TypedColumn::Str {
                offsets,
                data,
                nulls,
            } => {
                offsets.write_to(writer)?;
                data.write_to(writer)?;
                nulls.write_to(writer)?;
            }
            TypedColumn::Mixed { data } => {
                let encoded = bincode::serialize(data)
                    .map_err(|e| io::Error::other(format!("bincode error: {}", e)))?;
                writer.write_all(&encoded)?;
            }
        }
        Ok(())
    }

    /// Return the type tag string for serialization.
    #[allow(dead_code)]
    pub fn type_tag(&self) -> &'static str {
        match self {
            TypedColumn::Int64 { .. } => "int64",
            TypedColumn::Float64 { .. } => "float64",
            TypedColumn::UniqueId { .. } => "uniqueid",
            TypedColumn::Bool { .. } => "bool",
            TypedColumn::Date { .. } => "date",
            TypedColumn::Str { .. } => "string",
            TypedColumn::Mixed { .. } => "mixed",
        }
    }
}

// ─── ColumnStore ─────────────────────────────────────────────────────────────

/// Per-node-type columnar store. Holds one TypedColumn per property key.
/// All columns have the same number of rows.
#[derive(Debug, Clone)]
pub struct ColumnStore {
    /// Schema mapping property keys to slot indices (shared with Compact storage)
    schema: Arc<TypeSchema>,
    /// One column per property key, indexed by slot index from schema
    columns: Vec<TypedColumn>,
    /// Number of rows (nodes of this type)
    row_count: u32,
    /// Tombstone bitmap: true = row deleted
    tombstones: Vec<bool>,
}

#[allow(dead_code)]
impl ColumnStore {
    /// Create a new ColumnStore from a TypeSchema and type metadata.
    /// `type_meta` maps property name → type string (e.g., "int64", "string").
    pub fn new(
        schema: Arc<TypeSchema>,
        type_meta: &HashMap<String, String>,
        interner: &StringInterner,
    ) -> Self {
        let mut columns = Vec::with_capacity(schema.len());
        for (_slot, ik) in schema.iter() {
            let prop_name = interner.resolve(ik);
            let type_str = type_meta
                .get(prop_name)
                .map(|s| s.as_str())
                .unwrap_or("mixed");
            columns.push(TypedColumn::from_type_str(type_str));
        }
        ColumnStore {
            schema,
            columns,
            row_count: 0,
            tombstones: Vec::new(),
        }
    }

    /// Create a ColumnStore from an existing schema with all Mixed columns (for unknown types).
    pub fn new_mixed(schema: Arc<TypeSchema>) -> Self {
        let columns = (0..schema.len())
            .map(|_| TypedColumn::Mixed { data: Vec::new() })
            .collect();
        ColumnStore {
            schema,
            columns,
            row_count: 0,
            tombstones: Vec::new(),
        }
    }

    /// Number of rows (including tombstoned).
    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    /// Number of live (non-tombstoned) rows.
    pub fn live_count(&self) -> u32 {
        self.row_count - self.tombstones.iter().filter(|&&t| t).count() as u32
    }

    /// Reference to the shared schema.
    pub fn schema(&self) -> &Arc<TypeSchema> {
        &self.schema
    }

    /// Append a row of property values. Returns the row_id for this row.
    /// `values` is a list of (InternedKey, Value) pairs.
    pub fn push_row(&mut self, values: &[(InternedKey, Value)]) -> u32 {
        let row_id = self.row_count;

        // Build slot→value lookup to push values directly (avoids null-then-overwrite).
        let mut slot_values: Vec<Option<&Value>> = vec![None; self.columns.len()];
        for (key, value) in values {
            if let Some(slot) = self.schema.slot(*key) {
                slot_values[slot as usize] = Some(value);
            }
        }

        for (slot, slot_val) in slot_values.iter().enumerate() {
            let col = &mut self.columns[slot];
            if let Some(value) = slot_val {
                if col.push(value).is_err() {
                    // Type mismatch — demote column to Mixed and retry
                    self.demote_to_mixed(slot);
                    let _ = self.columns[slot].push(value);
                }
            } else {
                col.push_null();
            }
        }

        self.row_count += 1;
        self.tombstones.push(false);
        row_id
    }

    /// Get a property value by (row_id, interned key).
    pub fn get(&self, row_id: u32, key: InternedKey) -> Option<Value> {
        if row_id >= self.row_count {
            return None;
        }
        if self
            .tombstones
            .get(row_id as usize)
            .copied()
            .unwrap_or(false)
        {
            return None;
        }
        let slot = self.schema.slot(key)?;
        self.columns.get(slot as usize)?.get(row_id)
    }

    /// Set a property value for a given row.
    /// Extends the schema if the key is new.
    pub fn set(
        &mut self,
        row_id: u32,
        key: InternedKey,
        value: &Value,
        type_meta: Option<&str>,
    ) -> bool {
        if row_id >= self.row_count {
            return false;
        }
        let slot = match self.schema.slot(key) {
            Some(s) => s,
            None => {
                // New property — extend schema and add a column
                let s = Arc::make_mut(&mut self.schema).add_key(key);
                let type_str = type_meta.unwrap_or("mixed");
                let mut col = TypedColumn::from_type_str(type_str);
                // Backfill nulls for existing rows
                for _ in 0..self.row_count {
                    col.push_null();
                }
                self.columns.push(col);
                s
            }
        };
        let col = &mut self.columns[slot as usize];
        if col.set(row_id, value).is_err() {
            self.demote_to_mixed(slot as usize);
            let _ = self.columns[slot as usize].set(row_id, value);
        }
        true
    }

    /// Mark a row as deleted (tombstoned).
    pub fn tombstone(&mut self, row_id: u32) {
        if let Some(t) = self.tombstones.get_mut(row_id as usize) {
            *t = true;
        }
    }

    /// Check if a row has a property (non-null, non-tombstoned).
    pub fn contains(&self, row_id: u32, key: InternedKey) -> bool {
        self.get(row_id, key).is_some()
    }

    /// Iterate over all non-null properties for a row.
    /// Returns (InternedKey, Value) pairs.
    pub fn row_properties(&self, row_id: u32) -> Vec<(InternedKey, Value)> {
        if row_id >= self.row_count
            || self
                .tombstones
                .get(row_id as usize)
                .copied()
                .unwrap_or(false)
        {
            return Vec::new();
        }
        let mut result = Vec::new();
        for (slot, ik) in self.schema.iter() {
            if let Some(val) = self.columns.get(slot as usize).and_then(|c| c.get(row_id)) {
                result.push((ik, val));
            }
        }
        result
    }

    /// Reconstruct all properties for a row as a HashMap<String, Value>.
    pub fn row_properties_map(
        &self,
        row_id: u32,
        interner: &StringInterner,
    ) -> HashMap<String, Value> {
        self.row_properties(row_id)
            .into_iter()
            .map(|(ik, v)| (interner.resolve(ik).to_string(), v))
            .collect()
    }

    /// Demote a column from typed to Mixed, preserving all existing data.
    fn demote_to_mixed(&mut self, slot: usize) {
        let old_col = &self.columns[slot];
        let mut mixed_data = Vec::with_capacity(old_col.len());
        for i in 0..old_col.len() {
            mixed_data.push(old_col.get(i as u32).unwrap_or(Value::Null));
        }
        self.columns[slot] = TypedColumn::Mixed { data: mixed_data };
    }

    /// Materialize all columns to file-backed mmap in the given directory.
    pub fn materialize_to_files(
        &mut self,
        dir: &Path,
        interner: &StringInterner,
    ) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;
        for (slot, ik) in self.schema.iter() {
            let col_name = interner.resolve(ik);
            if let Some(col) = self.columns.get_mut(slot as usize) {
                col.materialize_to_file(dir, col_name)?;
            }
        }
        Ok(())
    }

    /// Convert all columns back to heap-backed storage.
    pub fn materialize_to_heap(&mut self) {
        for col in &mut self.columns {
            col.materialize_to_heap();
        }
    }

    /// Whether any column is file-backed.
    pub fn is_mapped(&self) -> bool {
        self.columns.iter().any(|c| c.is_mapped())
    }

    /// Heap-resident bytes across all columns (0 if fully mmap'd).
    pub fn heap_bytes(&self) -> usize {
        let col_bytes: usize = self.columns.iter().map(|c| c.heap_bytes()).sum();
        col_bytes + self.tombstones.len()
    }

    /// Access columns for introspection (e.g., getting type tags).
    pub fn columns_ref(&self) -> &[TypedColumn] {
        &self.columns
    }

    /// Serialize all columns to a packed byte buffer for the v3 file format.
    ///
    /// Format per column:
    ///   [2B] col_name_len  [NB] col_name_utf8
    ///   [2B] type_tag_len  [NB] type_tag
    ///   [8B] data_len      [NB] data_bytes (+ null_bytes for typed columns)
    ///   For "string": data_bytes = offsets + str_data + null_bitmap
    ///   For "mixed": data_bytes = bincode(Vec<Value>)
    pub fn write_packed(&self, interner: &StringInterner) -> io::Result<Vec<u8>> {
        let mut buf: Vec<u8> = Vec::new();

        // Number of columns
        let num_cols = self.columns.len() as u32;
        buf.extend_from_slice(&num_cols.to_le_bytes());

        for (slot, ik) in self.schema.iter() {
            let col_name = interner.resolve(ik);
            let col = &self.columns[slot as usize];
            let type_tag = col.type_tag();

            // Column name
            let name_bytes = col_name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(name_bytes);

            // Type tag
            let tag_bytes = type_tag.as_bytes();
            buf.extend_from_slice(&(tag_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(tag_bytes);

            // Column data — write length placeholder, then data directly, then patch length
            let len_offset = buf.len();
            buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder
            col.write_to(&mut buf)?;
            let data_len = (buf.len() - len_offset - 8) as u64;
            buf[len_offset..len_offset + 8].copy_from_slice(&data_len.to_le_bytes());
        }

        Ok(buf)
    }

    /// Load columns from a packed byte buffer (v3 format).
    ///
    /// If `temp_dir` is `Some`, writes column data to temp files and mmaps them
    /// (for larger-than-RAM support). If `None`, loads into heap.
    pub fn load_packed(
        schema: Arc<TypeSchema>,
        type_meta: &HashMap<String, String>,
        interner: &StringInterner,
        packed: &[u8],
        row_count: u32,
        temp_dir: Option<&Path>,
    ) -> io::Result<Self> {
        use std::io::Read;

        let mut store = ColumnStore::new(Arc::clone(&schema), type_meta, interner);
        store.row_count = row_count;
        store.tombstones = vec![false; row_count as usize];

        let mut cursor = std::io::Cursor::new(packed);

        // Read number of columns
        let mut u32_buf = [0u8; 4];
        cursor.read_exact(&mut u32_buf)?;
        let num_cols = u32::from_le_bytes(u32_buf);

        for _ in 0..num_cols {
            // Column name
            let mut u16_buf = [0u8; 2];
            cursor.read_exact(&mut u16_buf)?;
            let name_len = u16::from_le_bytes(u16_buf) as usize;
            let mut name_bytes = vec![0u8; name_len];
            cursor.read_exact(&mut name_bytes)?;
            let col_name = String::from_utf8(name_bytes)
                .map_err(|e| io::Error::other(format!("invalid column name: {}", e)))?;

            // Type tag
            cursor.read_exact(&mut u16_buf)?;
            let tag_len = u16::from_le_bytes(u16_buf) as usize;
            let mut tag_bytes = vec![0u8; tag_len];
            cursor.read_exact(&mut tag_bytes)?;
            let type_tag = String::from_utf8(tag_bytes)
                .map_err(|e| io::Error::other(format!("invalid type tag: {}", e)))?;

            // Data blob
            let mut u64_buf = [0u8; 8];
            cursor.read_exact(&mut u64_buf)?;
            let data_len = u64::from_le_bytes(u64_buf) as usize;
            let mut data_blob = vec![0u8; data_len];
            cursor.read_exact(&mut data_blob)?;

            // Find the slot for this column
            let ik = InternedKey::from_str(&col_name);
            let slot = match schema.slot(ik) {
                Some(s) => s as usize,
                None => continue, // schema doesn't have this column, skip
            };

            // Build the TypedColumn from the data blob
            let col = Self::unpack_column(&type_tag, &data_blob, row_count, temp_dir, &col_name)?;

            if slot < store.columns.len() {
                store.columns[slot] = col;
            }
        }

        Ok(store)
    }

    /// Unpack a single column from its raw data blob.
    fn unpack_column(
        type_tag: &str,
        data_blob: &[u8],
        row_count: u32,
        temp_dir: Option<&Path>,
        col_name: &str,
    ) -> io::Result<TypedColumn> {
        let rc = row_count as usize;
        match type_tag {
            "int64" => {
                let data_size = rc * std::mem::size_of::<i64>();
                let null_size = rc;
                Self::check_blob_size(data_blob, data_size + null_size, type_tag, col_name)?;
                let data = Self::load_typed_vec::<i64>(
                    &data_blob[..data_size],
                    rc,
                    temp_dir,
                    col_name,
                    "i64",
                )?;
                let nulls = Self::load_typed_vec::<u8>(
                    &data_blob[data_size..],
                    rc,
                    temp_dir,
                    col_name,
                    "null",
                )?;
                Ok(TypedColumn::Int64 { data, nulls })
            }
            "float64" => {
                let data_size = rc * std::mem::size_of::<f64>();
                let null_size = rc;
                Self::check_blob_size(data_blob, data_size + null_size, type_tag, col_name)?;
                let data = Self::load_typed_vec::<f64>(
                    &data_blob[..data_size],
                    rc,
                    temp_dir,
                    col_name,
                    "f64",
                )?;
                let nulls = Self::load_typed_vec::<u8>(
                    &data_blob[data_size..],
                    rc,
                    temp_dir,
                    col_name,
                    "null",
                )?;
                Ok(TypedColumn::Float64 { data, nulls })
            }
            "uniqueid" => {
                let data_size = rc * std::mem::size_of::<u32>();
                let null_size = rc;
                Self::check_blob_size(data_blob, data_size + null_size, type_tag, col_name)?;
                let data = Self::load_typed_vec::<u32>(
                    &data_blob[..data_size],
                    rc,
                    temp_dir,
                    col_name,
                    "u32",
                )?;
                let nulls = Self::load_typed_vec::<u8>(
                    &data_blob[data_size..],
                    rc,
                    temp_dir,
                    col_name,
                    "null",
                )?;
                Ok(TypedColumn::UniqueId { data, nulls })
            }
            "bool" | "boolean" => {
                let data_size = rc; // u8 per row
                let null_size = rc;
                Self::check_blob_size(data_blob, data_size + null_size, type_tag, col_name)?;
                let data = Self::load_typed_vec::<u8>(
                    &data_blob[..data_size],
                    rc,
                    temp_dir,
                    col_name,
                    "bool",
                )?;
                let nulls = Self::load_typed_vec::<u8>(
                    &data_blob[data_size..],
                    rc,
                    temp_dir,
                    col_name,
                    "null",
                )?;
                Ok(TypedColumn::Bool { data, nulls })
            }
            "date" | "datetime" => {
                let data_size = rc * std::mem::size_of::<i32>();
                let null_size = rc;
                Self::check_blob_size(data_blob, data_size + null_size, type_tag, col_name)?;
                let data = Self::load_typed_vec::<i32>(
                    &data_blob[..data_size],
                    rc,
                    temp_dir,
                    col_name,
                    "i32",
                )?;
                let nulls = Self::load_typed_vec::<u8>(
                    &data_blob[data_size..],
                    rc,
                    temp_dir,
                    col_name,
                    "null",
                )?;
                Ok(TypedColumn::Date { data, nulls })
            }
            "string" => {
                // offsets: (rc+1) * u64, then str_data, then nulls: rc * u8
                let offsets_size = (rc + 1) * std::mem::size_of::<u64>();
                if data_blob.len() < offsets_size {
                    return Err(io::Error::other(format!(
                        "column '{}' (string): blob too small for offsets ({} < {})",
                        col_name,
                        data_blob.len(),
                        offsets_size
                    )));
                }
                let offsets_bytes = &data_blob[..offsets_size];
                let rest = &data_blob[offsets_size..];

                // Determine string data length from last offset
                let last_offset = u64::from_le_bytes(
                    offsets_bytes[offsets_size - 8..offsets_size]
                        .try_into()
                        .unwrap(),
                ) as usize;
                let null_size = rc;

                if rest.len() < last_offset + null_size {
                    return Err(io::Error::other(format!(
                        "column '{}' (string): blob too small for data+nulls",
                        col_name
                    )));
                }
                let str_bytes = &rest[..last_offset];
                let null_bytes = &rest[last_offset..last_offset + null_size];

                let offsets =
                    Self::load_typed_vec::<u64>(offsets_bytes, rc + 1, temp_dir, col_name, "off")?;
                let data = Self::load_bytes(str_bytes, temp_dir, col_name, "str")?;
                let nulls = Self::load_typed_vec::<u8>(null_bytes, rc, temp_dir, col_name, "null")?;
                Ok(TypedColumn::Str {
                    offsets,
                    data,
                    nulls,
                })
            }
            _ => {
                // Mixed — bincode deserialize
                let data: Vec<Value> = bincode::deserialize(data_blob).map_err(|e| {
                    io::Error::other(format!("bincode error for '{}': {}", col_name, e))
                })?;
                Ok(TypedColumn::Mixed { data })
            }
        }
    }

    /// Load raw bytes into a MmapOrVec<T>, optionally via temp file + mmap.
    fn load_typed_vec<T: Copy + Default + 'static>(
        bytes: &[u8],
        len: usize,
        temp_dir: Option<&Path>,
        col_name: &str,
        ext: &str,
    ) -> io::Result<MmapOrVec<T>> {
        if let Some(dir) = temp_dir {
            let path = dir.join(format!("{col_name}.{ext}"));
            std::fs::write(&path, bytes)?;
            MmapOrVec::load_mapped(&path, len)
        } else {
            // Load into heap
            let elem_size = std::mem::size_of::<T>();
            if elem_size == 0 {
                return Ok(MmapOrVec::new());
            }
            let mut data = Vec::with_capacity(len);
            for i in 0..len {
                let offset = i * elem_size;
                let val = unsafe { std::ptr::read(bytes.as_ptr().add(offset) as *const T) };
                data.push(val);
            }
            Ok(MmapOrVec::Heap { data })
        }
    }

    /// Load raw bytes into a MmapBytes, optionally via temp file + mmap.
    fn load_bytes(
        bytes: &[u8],
        temp_dir: Option<&Path>,
        col_name: &str,
        ext: &str,
    ) -> io::Result<MmapBytes> {
        if let Some(dir) = temp_dir {
            let path = dir.join(format!("{col_name}.{ext}"));
            std::fs::write(&path, bytes)?;
            MmapBytes::load_mapped(&path, bytes.len())
        } else {
            Ok(MmapBytes::Heap {
                data: bytes.to_vec(),
            })
        }
    }

    fn check_blob_size(
        blob: &[u8],
        expected: usize,
        type_tag: &str,
        col_name: &str,
    ) -> io::Result<()> {
        if blob.len() < expected {
            Err(io::Error::other(format!(
                "column '{}' ({}): blob too small ({} < {})",
                col_name,
                type_tag,
                blob.len(),
                expected
            )))
        } else {
            Ok(())
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_schema_and_meta() -> (Arc<TypeSchema>, HashMap<String, String>, StringInterner) {
        let mut interner = StringInterner::new();
        let keys = vec!["name", "age", "salary", "active", "joined"];
        let interned: Vec<InternedKey> = keys.iter().map(|k| interner.get_or_intern(k)).collect();

        let schema = Arc::new(TypeSchema::from_keys(interned));

        let mut meta = HashMap::new();
        meta.insert("name".to_string(), "string".to_string());
        meta.insert("age".to_string(), "int64".to_string());
        meta.insert("salary".to_string(), "float64".to_string());
        meta.insert("active".to_string(), "bool".to_string());
        meta.insert("joined".to_string(), "date".to_string());

        (schema, meta, interner)
    }

    #[test]
    fn test_typed_column_int64_roundtrip() {
        let mut col = TypedColumn::from_type_str("int64");
        assert!(col.push(&Value::Int64(42)).is_ok());
        assert!(col.push(&Value::Int64(-7)).is_ok());
        assert!(col.push(&Value::Null).is_ok());

        assert_eq!(col.get(0), Some(Value::Int64(42)));
        assert_eq!(col.get(1), Some(Value::Int64(-7)));
        assert_eq!(col.get(2), None); // null
        assert_eq!(col.get(3), None); // out of bounds
        assert_eq!(col.len(), 3);
    }

    #[test]
    fn test_typed_column_float64_with_int_promotion() {
        let mut col = TypedColumn::from_type_str("float64");
        assert!(col.push(&Value::Float64(3.14)).is_ok());
        assert!(col.push(&Value::Int64(42)).is_ok()); // int→float promotion
        assert_eq!(col.get(0), Some(Value::Float64(3.14)));
        assert_eq!(col.get(1), Some(Value::Float64(42.0)));
    }

    #[test]
    fn test_typed_column_string_roundtrip() {
        let mut col = TypedColumn::from_type_str("string");
        assert!(col.push(&Value::String("hello".into())).is_ok());
        assert!(col.push(&Value::String("world".into())).is_ok());
        assert!(col.push(&Value::Null).is_ok());
        assert!(col.push(&Value::String("".into())).is_ok());

        assert_eq!(col.get(0), Some(Value::String("hello".into())));
        assert_eq!(col.get(1), Some(Value::String("world".into())));
        assert_eq!(col.get(2), None);
        assert_eq!(col.get(3), Some(Value::String("".into())));
        assert_eq!(col.len(), 4);
    }

    #[test]
    fn test_typed_column_bool_roundtrip() {
        let mut col = TypedColumn::from_type_str("bool");
        assert!(col.push(&Value::Boolean(true)).is_ok());
        assert!(col.push(&Value::Boolean(false)).is_ok());
        assert_eq!(col.get(0), Some(Value::Boolean(true)));
        assert_eq!(col.get(1), Some(Value::Boolean(false)));
    }

    #[test]
    fn test_typed_column_date_roundtrip() {
        let mut col = TypedColumn::from_type_str("date");
        let d = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
        assert!(col.push(&Value::DateTime(d)).is_ok());
        assert!(col.push(&Value::Null).is_ok());
        assert_eq!(col.get(0), Some(Value::DateTime(d)));
        assert_eq!(col.get(1), None);
    }

    #[test]
    fn test_typed_column_uniqueid_roundtrip() {
        let mut col = TypedColumn::from_type_str("uniqueid");
        assert!(col.push(&Value::UniqueId(100)).is_ok());
        assert_eq!(col.get(0), Some(Value::UniqueId(100)));
    }

    #[test]
    fn test_typed_column_mixed_fallback() {
        let mut col = TypedColumn::from_type_str("mixed");
        assert!(col.push(&Value::Int64(1)).is_ok());
        assert!(col.push(&Value::String("hello".into())).is_ok());
        assert!(col.push(&Value::Boolean(true)).is_ok());
        assert_eq!(col.get(0), Some(Value::Int64(1)));
        assert_eq!(col.get(1), Some(Value::String("hello".into())));
        assert_eq!(col.get(2), Some(Value::Boolean(true)));
    }

    #[test]
    fn test_typed_column_type_mismatch_rejected() {
        let mut col = TypedColumn::from_type_str("int64");
        assert!(col.push(&Value::String("oops".into())).is_err());
    }

    #[test]
    fn test_column_store_basic_roundtrip() {
        let (schema, meta, interner) = make_schema_and_meta();
        let mut store = ColumnStore::new(schema, &meta, &interner);

        let name_key = InternedKey::from_str("name");
        let age_key = InternedKey::from_str("age");
        let salary_key = InternedKey::from_str("salary");

        let row0 = store.push_row(&[
            (name_key, Value::String("Alice".into())),
            (age_key, Value::Int64(30)),
            (salary_key, Value::Float64(75000.0)),
        ]);
        assert_eq!(row0, 0);

        let row1 = store.push_row(&[
            (name_key, Value::String("Bob".into())),
            (age_key, Value::Int64(25)),
        ]);
        assert_eq!(row1, 1);

        assert_eq!(store.get(0, name_key), Some(Value::String("Alice".into())));
        assert_eq!(store.get(0, age_key), Some(Value::Int64(30)));
        assert_eq!(store.get(0, salary_key), Some(Value::Float64(75000.0)));

        assert_eq!(store.get(1, name_key), Some(Value::String("Bob".into())));
        assert_eq!(store.get(1, age_key), Some(Value::Int64(25)));
        assert_eq!(store.get(1, salary_key), None); // null

        assert_eq!(store.row_count(), 2);
        assert_eq!(store.live_count(), 2);
    }

    #[test]
    fn test_column_store_property_update() {
        let (schema, meta, interner) = make_schema_and_meta();
        let mut store = ColumnStore::new(schema, &meta, &interner);

        let name_key = InternedKey::from_str("name");
        let age_key = InternedKey::from_str("age");

        store.push_row(&[
            (name_key, Value::String("Alice".into())),
            (age_key, Value::Int64(30)),
        ]);

        // Update age
        assert!(store.set(0, age_key, &Value::Int64(31), None));
        assert_eq!(store.get(0, age_key), Some(Value::Int64(31)));

        // Update name
        assert!(store.set(0, name_key, &Value::String("Alicia".into()), None));
        assert_eq!(store.get(0, name_key), Some(Value::String("Alicia".into())));
    }

    #[test]
    fn test_column_store_schema_extension() {
        let (schema, meta, interner) = make_schema_and_meta();
        let mut store = ColumnStore::new(schema, &meta, &interner);

        let name_key = InternedKey::from_str("name");
        let new_key = InternedKey::from_str("email");

        store.push_row(&[(name_key, Value::String("Alice".into()))]);

        // Set a property that doesn't exist in the schema yet
        assert!(store.set(
            0,
            new_key,
            &Value::String("alice@example.com".into()),
            Some("string")
        ));
        assert_eq!(
            store.get(0, new_key),
            Some(Value::String("alice@example.com".into()))
        );
    }

    #[test]
    fn test_column_store_tombstone() {
        let (schema, meta, interner) = make_schema_and_meta();
        let mut store = ColumnStore::new(schema, &meta, &interner);

        let name_key = InternedKey::from_str("name");
        store.push_row(&[(name_key, Value::String("Alice".into()))]);
        store.push_row(&[(name_key, Value::String("Bob".into()))]);

        store.tombstone(0);
        assert_eq!(store.get(0, name_key), None);
        assert_eq!(store.get(1, name_key), Some(Value::String("Bob".into())));
        assert_eq!(store.live_count(), 1);
    }

    #[test]
    fn test_column_store_row_properties() {
        let (schema, meta, interner) = make_schema_and_meta();
        let mut store = ColumnStore::new(schema, &meta, &interner);

        let name_key = InternedKey::from_str("name");
        let age_key = InternedKey::from_str("age");

        store.push_row(&[
            (name_key, Value::String("Alice".into())),
            (age_key, Value::Int64(30)),
        ]);

        let props = store.row_properties(0);
        assert_eq!(props.len(), 2);

        let map = store.row_properties_map(0, &interner);
        assert_eq!(map.get("name"), Some(&Value::String("Alice".into())));
        assert_eq!(map.get("age"), Some(&Value::Int64(30)));
    }

    #[test]
    fn test_column_store_demote_to_mixed() {
        let (schema, meta, interner) = make_schema_and_meta();
        let mut store = ColumnStore::new(schema, &meta, &interner);

        let age_key = InternedKey::from_str("age");

        // Push an int64 row
        store.push_row(&[(age_key, Value::Int64(30))]);

        // Now try to set a string into an int64 column — should demote to Mixed
        assert!(store.set(0, age_key, &Value::String("thirty".into()), None));
        assert_eq!(store.get(0, age_key), Some(Value::String("thirty".into())));
    }

    #[test]
    fn test_column_store_new_mixed() {
        let mut interner = StringInterner::new();
        let keys = vec![interner.get_or_intern("a"), interner.get_or_intern("b")];
        let schema = Arc::new(TypeSchema::from_keys(keys));
        let mut store = ColumnStore::new_mixed(schema);

        let a_key = InternedKey::from_str("a");
        let b_key = InternedKey::from_str("b");

        store.push_row(&[
            (a_key, Value::Int64(1)),
            (b_key, Value::String("hello".into())),
        ]);

        assert_eq!(store.get(0, a_key), Some(Value::Int64(1)));
        assert_eq!(store.get(0, b_key), Some(Value::String("hello".into())));
    }

    #[test]
    fn test_column_store_materialize_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let (schema, meta, interner) = make_schema_and_meta();
        let mut store = ColumnStore::new(schema, &meta, &interner);

        let name_key = InternedKey::from_str("name");
        let age_key = InternedKey::from_str("age");
        let salary_key = InternedKey::from_str("salary");
        let active_key = InternedKey::from_str("active");

        store.push_row(&[
            (name_key, Value::String("Alice".into())),
            (age_key, Value::Int64(30)),
            (salary_key, Value::Float64(75000.0)),
            (active_key, Value::Boolean(true)),
        ]);
        store.push_row(&[
            (name_key, Value::String("Bob".into())),
            (age_key, Value::Int64(25)),
            (salary_key, Value::Float64(50000.0)),
            (active_key, Value::Boolean(false)),
        ]);

        // Materialize to files
        store.materialize_to_files(dir.path(), &interner).unwrap();
        assert!(store.is_mapped());

        // Verify data still accessible
        assert_eq!(store.get(0, name_key), Some(Value::String("Alice".into())));
        assert_eq!(store.get(0, age_key), Some(Value::Int64(30)));
        assert_eq!(store.get(1, salary_key), Some(Value::Float64(50000.0)));
        assert_eq!(store.get(1, active_key), Some(Value::Boolean(false)));

        // Convert back to heap
        store.materialize_to_heap();
        assert!(!store.is_mapped());
        assert_eq!(store.get(0, name_key), Some(Value::String("Alice".into())));
        assert_eq!(store.get(1, age_key), Some(Value::Int64(25)));
    }
}
