// src/graph/build_column_store.rs
//
// Per-type column builder for the DiskGraph build phase.
// Uses BlockPool-backed BlockColumns for memory-managed append-only storage.
// Supports late-binding schema: new property keys create new columns on first
// encounter, with null backfill for all prior rows.

use crate::datatypes::values::Value;
use crate::graph::block_column::{BlockColumn, ColumnType};
use crate::graph::block_pool::BlockPool;
use crate::graph::column_store::{ColumnStore, TypedColumn};
use crate::graph::mmap_vec::{MmapBytes, MmapOrVec};
use crate::graph::schema::{InternedKey, StringInterner, TypeSchema};
use std::collections::HashMap;
use std::io;
use std::sync::Arc;

// ─── BuildColumnStore ───────────────────────────────────────────────────────

/// Per-type column builder for the build phase.
///
/// During ingest, properties are appended directly to BlockPool-backed columns.
/// New columns are created on first encounter with null backfill.
/// After ingest, call `into_column_store()` to produce a final ColumnStore.
pub struct BuildColumnStore {
    /// property_key → (column_index, column_type)
    key_map: HashMap<InternedKey, usize>,
    /// The actual columns, indexed by key_map values.
    columns: Vec<(InternedKey, BlockColumn)>,
    /// Dedicated id column (always UniqueId or Str).
    id_column: BlockColumn,
    /// Dedicated title column (always Str).
    title_column: BlockColumn,
    /// Number of rows pushed so far.
    row_count: u32,
}

#[allow(dead_code)]
impl BuildColumnStore {
    /// Create a new builder.
    pub fn new(pool: &mut BlockPool) -> io::Result<Self> {
        let id_column = BlockColumn::new(ColumnType::UniqueId, pool)?;
        let title_column = BlockColumn::new(ColumnType::Str, pool)?;

        Ok(BuildColumnStore {
            key_map: HashMap::new(),
            columns: Vec::new(),
            id_column,
            title_column,
            row_count: 0,
        })
    }

    /// Push a complete row of properties.
    ///
    /// - `id`: node ID (typically UniqueId or String)
    /// - `title`: node title
    /// - `properties`: interned key-value pairs
    ///
    /// Returns the assigned row_id.
    pub fn push_row(
        &mut self,
        id: &Value,
        title: &Value,
        properties: &[(InternedKey, Value)],
        pool: &mut BlockPool,
    ) -> io::Result<u32> {
        let row_id = self.row_count;

        // Push id
        self.push_id(id, pool)?;

        // Push title
        self.push_title(title, pool)?;

        // Track which columns got a value this row
        let mut pushed = vec![false; self.columns.len()];

        for (key, value) in properties {
            if matches!(value, Value::Null) {
                continue;
            }
            let col_idx = match self.key_map.get(key) {
                Some(&idx) => idx,
                None => {
                    // New column — create it and backfill nulls for prior rows
                    let col_type = ColumnType::from_value(value).unwrap_or(ColumnType::Str);
                    let mut col = BlockColumn::new(col_type, pool)?;
                    if self.row_count > 0 {
                        col.backfill_nulls(self.row_count, pool)?;
                    }
                    let idx = self.columns.len();
                    self.key_map.insert(*key, idx);
                    self.columns.push((*key, col));
                    pushed.push(false);
                    idx
                }
            };

            let (_, col) = &mut self.columns[col_idx];
            match col.push(value, pool)? {
                Ok(()) => {
                    pushed[col_idx] = true;
                }
                Err(()) => {
                    // Type mismatch — push null for this column, value is lost
                    // (could promote to Str in the future)
                    col.push_null(pool)?;
                    pushed[col_idx] = true;
                }
            }
        }

        // Push null for columns that didn't get a value this row
        for (i, was_pushed) in pushed.iter().enumerate() {
            if !was_pushed {
                self.columns[i].1.push_null(pool)?;
            }
        }

        self.row_count += 1;
        Ok(row_id)
    }

    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    /// Mark all completed blocks as cold (eligible for eviction).
    /// Call periodically (e.g., every 100K entities) during ingest.
    pub fn mark_cold(&mut self, pool: &mut BlockPool) {
        self.id_column.mark_completed_blocks_cold(pool);
        self.title_column.mark_completed_blocks_cold(pool);
        for (_, col) in &mut self.columns {
            col.mark_completed_blocks_cold(pool);
        }
    }

    /// Convert this builder into a final ColumnStore.
    ///
    /// Reads all blocks from the pool (reloading evicted ones) and constructs
    /// the TypedColumn-based ColumnStore used for queries and persistence.
    pub fn into_column_store(
        mut self,
        interner: &StringInterner,
        pool: &mut BlockPool,
    ) -> io::Result<(ColumnStore, Arc<TypeSchema>)> {
        // Build TypeSchema from the columns we discovered
        let mut schema = TypeSchema::new();
        for (key, _) in &self.columns {
            schema.add_key(*key);
        }
        let schema = Arc::new(schema);

        // Build metadata for type hints
        let meta: HashMap<String, String> = self
            .columns
            .iter()
            .map(|(key, col)| {
                let name = interner.resolve(*key).to_string();
                let type_str = col.col_type().type_str().to_string();
                (name, type_str)
            })
            .collect();

        let mut store = ColumnStore::new(Arc::clone(&schema), &meta, interner);

        // Drain id column
        let id_typed = Self::drain_block_column_to_typed(&self.id_column, pool)?;
        store.set_id_column(id_typed);

        // Drain title column
        let title_typed = Self::drain_block_column_to_typed(&self.title_column, pool)?;
        store.set_title_column(title_typed);

        // Drain property columns in schema slot order
        let slot_keys: Vec<(u16, InternedKey)> = schema.iter().collect();
        for (slot, key) in slot_keys {
            if let Some(&col_idx) = self.key_map.get(&key) {
                let col = &self.columns[col_idx].1;
                let typed = Self::drain_block_column_to_typed(col, pool)?;
                store.set_column(slot as usize, typed);
            }
        }

        store.set_row_count(self.row_count);

        // Free all blocks
        self.id_column.free_all(pool);
        self.title_column.free_all(pool);
        for (_, col) in &mut self.columns {
            col.free_all(pool);
        }

        Ok((store, schema))
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn push_id(&mut self, value: &Value, pool: &mut BlockPool) -> io::Result<()> {
        match self.id_column.push(value, pool)? {
            Ok(()) => Ok(()),
            Err(()) => {
                // UniqueId column can't hold this type — push null
                self.id_column.push_null(pool)
            }
        }
    }

    fn push_title(&mut self, value: &Value, pool: &mut BlockPool) -> io::Result<()> {
        match self.title_column.push(value, pool)? {
            Ok(()) => Ok(()),
            Err(()) => self.title_column.push_null(pool),
        }
    }

    fn drain_block_column_to_typed(
        col: &BlockColumn,
        pool: &mut BlockPool,
    ) -> io::Result<TypedColumn> {
        let (data, nulls, offsets) = col.drain(pool)?;
        let null_vec = MmapOrVec::from_vec(nulls);

        let typed = match col.col_type() {
            ColumnType::Int64 => {
                let values: Vec<i64> = data
                    .chunks_exact(8)
                    .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                    .collect();
                TypedColumn::Int64 {
                    data: MmapOrVec::from_vec(values),
                    nulls: null_vec,
                }
            }
            ColumnType::Float64 => {
                let values: Vec<f64> = data
                    .chunks_exact(8)
                    .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
                    .collect();
                TypedColumn::Float64 {
                    data: MmapOrVec::from_vec(values),
                    nulls: null_vec,
                }
            }
            ColumnType::UniqueId => {
                let values: Vec<u32> = data
                    .chunks_exact(4)
                    .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
                    .collect();
                TypedColumn::UniqueId {
                    data: MmapOrVec::from_vec(values),
                    nulls: null_vec,
                }
            }
            ColumnType::Bool => TypedColumn::Bool {
                data: MmapOrVec::from_vec(data),
                nulls: null_vec,
            },
            ColumnType::Date => {
                let values: Vec<i32> = data
                    .chunks_exact(4)
                    .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
                    .collect();
                TypedColumn::Date {
                    data: MmapOrVec::from_vec(values),
                    nulls: null_vec,
                }
            }
            ColumnType::Str => {
                let offsets_data = offsets.unwrap_or_default();
                let offset_values: Vec<u64> = std::iter::once(0u64)
                    .chain(
                        offsets_data
                            .chunks_exact(8)
                            .map(|c| u64::from_ne_bytes(c.try_into().unwrap())),
                    )
                    .collect();
                TypedColumn::Str {
                    offsets: MmapOrVec::from_vec(offset_values),
                    data: MmapBytes::from_vec(data),
                    nulls: null_vec,
                }
            }
        };
        Ok(typed)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pool() -> BlockPool {
        let dir = tempfile::tempdir().unwrap();
        BlockPool::new(10 << 20, dir.into_path()).unwrap()
    }

    fn test_interner() -> StringInterner {
        StringInterner::new()
    }

    #[test]
    fn push_single_row() {
        let mut pool = test_pool();
        let mut interner = test_interner();
        let mut builder = BuildColumnStore::new(&mut pool).unwrap();

        let name_key = interner.get_or_intern("name");
        let age_key = interner.get_or_intern("age");

        let row_id = builder
            .push_row(
                &Value::UniqueId(1),
                &Value::String("Alice".into()),
                &[
                    (name_key, Value::String("Alice".into())),
                    (age_key, Value::Int64(30)),
                ],
                &mut pool,
            )
            .unwrap();

        assert_eq!(row_id, 0);
        assert_eq!(builder.row_count(), 1);
    }

    #[test]
    fn late_binding_schema() {
        let mut pool = test_pool();
        let mut interner = test_interner();
        let mut builder = BuildColumnStore::new(&mut pool).unwrap();

        let name_key = interner.get_or_intern("name");
        let age_key = interner.get_or_intern("age");
        let email_key = interner.get_or_intern("email");

        // Row 0: only name
        builder
            .push_row(
                &Value::UniqueId(1),
                &Value::String("Alice".into()),
                &[(name_key, Value::String("Alice".into()))],
                &mut pool,
            )
            .unwrap();

        // Row 1: name + age (age column created, backfilled with null for row 0)
        builder
            .push_row(
                &Value::UniqueId(2),
                &Value::String("Bob".into()),
                &[
                    (name_key, Value::String("Bob".into())),
                    (age_key, Value::Int64(25)),
                ],
                &mut pool,
            )
            .unwrap();

        // Row 2: name + email (email column created, backfilled with nulls for rows 0-1)
        builder
            .push_row(
                &Value::UniqueId(3),
                &Value::String("Carol".into()),
                &[
                    (name_key, Value::String("Carol".into())),
                    (email_key, Value::String("carol@example.com".into())),
                ],
                &mut pool,
            )
            .unwrap();

        assert_eq!(builder.row_count(), 3);
        assert_eq!(builder.columns.len(), 3); // name, age, email
    }

    #[test]
    fn into_column_store_preserves_data() {
        let mut pool = test_pool();
        let mut interner = test_interner();
        let mut builder = BuildColumnStore::new(&mut pool).unwrap();

        let name_key = interner.get_or_intern("name");
        let age_key = interner.get_or_intern("age");

        builder
            .push_row(
                &Value::UniqueId(1),
                &Value::String("Alice".into()),
                &[
                    (name_key, Value::String("Alice".into())),
                    (age_key, Value::Int64(30)),
                ],
                &mut pool,
            )
            .unwrap();

        builder
            .push_row(
                &Value::UniqueId(2),
                &Value::String("Bob".into()),
                &[(name_key, Value::String("Bob".into()))],
                &mut pool,
            )
            .unwrap();

        let (store, _schema) = builder.into_column_store(&interner, &mut pool).unwrap();

        assert_eq!(store.row_count(), 2);

        // Verify ID column
        let id0 = store.get_id(0);
        assert_eq!(id0, Some(Value::UniqueId(1)));
        let id1 = store.get_id(1);
        assert_eq!(id1, Some(Value::UniqueId(2)));

        // Verify title column
        let t0 = store.get_title(0);
        assert_eq!(t0, Some(Value::String("Alice".into())));

        // Verify property columns via row_properties
        let props0 = store.row_properties(0);
        assert!(props0
            .iter()
            .any(|(_, v)| *v == Value::String("Alice".into())));
        assert!(props0.iter().any(|(_, v)| *v == Value::Int64(30)));

        let props1 = store.row_properties(1);
        assert!(props1
            .iter()
            .any(|(_, v)| *v == Value::String("Bob".into())));
    }
}
