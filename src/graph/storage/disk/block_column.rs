// src/graph/block_column.rs
//
// Append-only block-chain column storage for the DiskGraph build phase.
// Each column stores homogeneously-typed values across a chain of blocks
// managed by a BlockPool. Supports late-binding schema: columns can be
// created after rows already exist (null backfill via null bitmap).

use crate::datatypes::values::Value;
use crate::graph::storage::disk::block_pool::{BlockId, BlockPool};
use chrono::NaiveDate;
use std::io;

/// Block size for column data: 1 MB.
const BLOCK_SIZE: usize = 1 << 20;

const UNIX_EPOCH_DATE: NaiveDate = match NaiveDate::from_ymd_opt(1970, 1, 1) {
    Some(d) => d,
    None => unreachable!(),
};

// ─── ColumnType ─────────────────────────────────────────────────────────────

/// The physical type of a column, determining byte layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Int64,    // 8 bytes/value
    Float64,  // 8 bytes/value
    UniqueId, // 4 bytes/value
    Bool,     // 1 byte/value
    Date,     // 4 bytes/value (days since epoch as i32)
    Str,      // Variable-length: offset (8B) + data bytes
}

#[allow(dead_code)]
impl ColumnType {
    /// Bytes per value for fixed-width types. None for Str.
    pub fn value_size(&self) -> Option<usize> {
        match self {
            ColumnType::Int64 | ColumnType::Float64 => Some(8),
            ColumnType::UniqueId | ColumnType::Date => Some(4),
            ColumnType::Bool => Some(1),
            ColumnType::Str => None,
        }
    }

    /// Infer column type from a Value variant.
    pub fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Int64(_) => Some(ColumnType::Int64),
            Value::Float64(_) => Some(ColumnType::Float64),
            Value::UniqueId(_) => Some(ColumnType::UniqueId),
            Value::Boolean(_) => Some(ColumnType::Bool),
            Value::DateTime(_) => Some(ColumnType::Date),
            Value::String(_) => Some(ColumnType::Str),
            Value::Null => None,
            _ => Some(ColumnType::Str), // Fallback: coerce to string
        }
    }

    /// Type name string matching TypedColumn's from_type_str convention.
    pub fn type_str(&self) -> &'static str {
        match self {
            ColumnType::Int64 => "int64",
            ColumnType::Float64 => "float64",
            ColumnType::UniqueId => "uniqueid",
            ColumnType::Bool => "bool",
            ColumnType::Date => "date",
            ColumnType::Str => "string",
        }
    }
}

// ─── BlockColumn ────────────────────────────────────────────────────────────

/// A single typed column stored as a chain of BlockPool blocks.
///
/// Fixed-width types: data blocks contain packed values (e.g., [i64; N]).
/// Str type: separate block chains for offsets (u64) and string data (bytes).
/// All types have a null bitmap chain (1 byte per row: 0=non-null, 1=null).
pub struct BlockColumn {
    col_type: ColumnType,
    /// Data blocks (fixed-width values or string bytes).
    data_blocks: Vec<BlockId>,
    /// Null bitmap blocks (1 byte per row: 0=non-null, 1=null).
    null_blocks: Vec<BlockId>,
    /// Str-only: offset blocks (u64 per row).
    offset_blocks: Vec<BlockId>,

    row_count: u32,
    /// Write cursor: byte position in the last data block.
    data_pos: usize,
    /// Write cursor: byte position in the last null block.
    null_pos: usize,
    /// Str-only: write cursor in the last offset block.
    offset_pos: usize,
    /// Str-only: running byte offset into string data.
    str_data_len: u64,
}

#[allow(dead_code)]
impl BlockColumn {
    /// Create a new empty column of the given type.
    pub fn new(col_type: ColumnType, pool: &mut BlockPool) -> io::Result<Self> {
        let data_block = pool.allocate(BLOCK_SIZE)?;
        let null_block = pool.allocate(BLOCK_SIZE)?;

        let mut col = BlockColumn {
            col_type,
            data_blocks: vec![data_block],
            null_blocks: vec![null_block],
            offset_blocks: Vec::new(),
            row_count: 0,
            data_pos: 0,
            null_pos: 0,
            offset_pos: 0,
            str_data_len: 0,
        };

        if col_type == ColumnType::Str {
            let offset_block = pool.allocate(BLOCK_SIZE)?;
            col.offset_blocks.push(offset_block);
        }

        Ok(col)
    }

    pub fn col_type(&self) -> ColumnType {
        self.col_type
    }

    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    /// Push a typed value. Returns Err if the value type doesn't match.
    pub fn push(&mut self, value: &Value, pool: &mut BlockPool) -> io::Result<Result<(), ()>> {
        match (&self.col_type, value) {
            (ColumnType::Int64, Value::Int64(v)) => {
                self.push_fixed_bytes(&v.to_ne_bytes(), pool)?;
                self.push_null_byte(0, pool)?;
            }
            (ColumnType::Float64, Value::Float64(v)) => {
                self.push_fixed_bytes(&v.to_ne_bytes(), pool)?;
                self.push_null_byte(0, pool)?;
            }
            (ColumnType::Float64, Value::Int64(v)) => {
                // int→float promotion
                self.push_fixed_bytes(&(*v as f64).to_ne_bytes(), pool)?;
                self.push_null_byte(0, pool)?;
            }
            (ColumnType::UniqueId, Value::UniqueId(v)) => {
                self.push_fixed_bytes(&v.to_ne_bytes(), pool)?;
                self.push_null_byte(0, pool)?;
            }
            (ColumnType::Bool, Value::Boolean(v)) => {
                self.push_fixed_bytes(&[*v as u8], pool)?;
                self.push_null_byte(0, pool)?;
            }
            (ColumnType::Date, Value::DateTime(d)) => {
                let days = (*d - UNIX_EPOCH_DATE).num_days() as i32;
                self.push_fixed_bytes(&days.to_ne_bytes(), pool)?;
                self.push_null_byte(0, pool)?;
            }
            (ColumnType::Str, Value::String(s)) => {
                self.push_str_bytes(s.as_bytes(), pool)?;
                self.push_null_byte(0, pool)?;
            }
            (_, Value::Null) => {
                self.push_null_value(pool)?;
            }
            _ => return Ok(Err(())),
        }
        self.row_count += 1;
        Ok(Ok(()))
    }

    /// Push a null value (works for any column type).
    pub fn push_null(&mut self, pool: &mut BlockPool) -> io::Result<()> {
        self.push_null_value(pool)?;
        self.row_count += 1;
        Ok(())
    }

    /// Backfill N nulls (for late-binding schema: new column discovered mid-stream).
    pub fn backfill_nulls(&mut self, count: u32, pool: &mut BlockPool) -> io::Result<()> {
        for _ in 0..count {
            self.push_null_value(pool)?;
            self.row_count += 1;
        }
        Ok(())
    }

    /// Mark all completed (non-active) blocks as cold for LRU eviction.
    pub fn mark_completed_blocks_cold(&mut self, pool: &mut BlockPool) {
        // Mark all blocks except the last (active) one
        for blocks in [&self.data_blocks, &self.null_blocks, &self.offset_blocks] {
            if blocks.len() > 1 {
                for &id in &blocks[..blocks.len() - 1] {
                    pool.mark_cold(id);
                }
            }
        }
    }

    /// Total number of data blocks (across data + null + offset chains).
    pub fn total_blocks(&self) -> usize {
        self.data_blocks.len() + self.null_blocks.len() + self.offset_blocks.len()
    }

    /// Read all data out as a flat byte vector (for final ColumnStore conversion).
    /// Returns (data_bytes, null_bytes, offset_bytes_if_str).
    #[allow(clippy::type_complexity)]
    pub fn drain(&self, pool: &mut BlockPool) -> io::Result<(Vec<u8>, Vec<u8>, Option<Vec<u8>>)> {
        let data = self.read_chain(&self.data_blocks, self.data_pos, pool)?;
        let nulls = self.read_chain(&self.null_blocks, self.null_pos, pool)?;
        let offsets = if self.col_type == ColumnType::Str {
            Some(self.read_chain(&self.offset_blocks, self.offset_pos, pool)?)
        } else {
            None
        };
        Ok((data, nulls, offsets))
    }

    /// Free all blocks in this column.
    pub fn free_all(&mut self, pool: &mut BlockPool) {
        for &id in self
            .data_blocks
            .iter()
            .chain(&self.null_blocks)
            .chain(&self.offset_blocks)
        {
            let _ = pool.free(id);
        }
        self.data_blocks.clear();
        self.null_blocks.clear();
        self.offset_blocks.clear();
    }

    // ── Internal push helpers ───────────────────────────────────────────

    /// Push bytes to the data block chain. Allocates new blocks as needed.
    fn push_fixed_bytes(&mut self, bytes: &[u8], pool: &mut BlockPool) -> io::Result<()> {
        let need = bytes.len();
        if self.data_pos + need > BLOCK_SIZE {
            // Current block is full — mark it cold and allocate a new one
            if let Some(&last) = self.data_blocks.last() {
                pool.mark_cold(last);
            }
            let new_block = pool.allocate(BLOCK_SIZE)?;
            self.data_blocks.push(new_block);
            self.data_pos = 0;
        }
        let block_id = *self.data_blocks.last().unwrap();
        let block = pool.get_mut(block_id)?;
        block[self.data_pos..self.data_pos + need].copy_from_slice(bytes);
        self.data_pos += need;
        Ok(())
    }

    /// Push string bytes to the data block chain + offset to offset chain.
    fn push_str_bytes(&mut self, bytes: &[u8], pool: &mut BlockPool) -> io::Result<()> {
        // Append string data — may span multiple blocks
        let mut remaining = bytes;
        while !remaining.is_empty() {
            let space = BLOCK_SIZE - self.data_pos;
            if space == 0 {
                if let Some(&last) = self.data_blocks.last() {
                    pool.mark_cold(last);
                }
                let new_block = pool.allocate(BLOCK_SIZE)?;
                self.data_blocks.push(new_block);
                self.data_pos = 0;
                continue;
            }
            let chunk = remaining.len().min(space);
            let block_id = *self.data_blocks.last().unwrap();
            let block = pool.get_mut(block_id)?;
            block[self.data_pos..self.data_pos + chunk].copy_from_slice(&remaining[..chunk]);
            self.data_pos += chunk;
            remaining = &remaining[chunk..];
        }
        self.str_data_len += bytes.len() as u64;

        // Append offset (end position of this string in the data stream)
        self.push_offset(self.str_data_len, pool)?;
        Ok(())
    }

    fn push_offset(&mut self, offset: u64, pool: &mut BlockPool) -> io::Result<()> {
        let need = 8; // u64
        if self.offset_pos + need > BLOCK_SIZE {
            if let Some(&last) = self.offset_blocks.last() {
                pool.mark_cold(last);
            }
            let new_block = pool.allocate(BLOCK_SIZE)?;
            self.offset_blocks.push(new_block);
            self.offset_pos = 0;
        }
        let block_id = *self.offset_blocks.last().unwrap();
        let block = pool.get_mut(block_id)?;
        block[self.offset_pos..self.offset_pos + need].copy_from_slice(&offset.to_ne_bytes());
        self.offset_pos += need;
        Ok(())
    }

    fn push_null_byte(&mut self, val: u8, pool: &mut BlockPool) -> io::Result<()> {
        if self.null_pos >= BLOCK_SIZE {
            if let Some(&last) = self.null_blocks.last() {
                pool.mark_cold(last);
            }
            let new_block = pool.allocate(BLOCK_SIZE)?;
            self.null_blocks.push(new_block);
            self.null_pos = 0;
        }
        let block_id = *self.null_blocks.last().unwrap();
        let block = pool.get_mut(block_id)?;
        block[self.null_pos] = val;
        self.null_pos += 1;
        Ok(())
    }

    /// Push a null for the column's type (zero data + null flag).
    fn push_null_value(&mut self, pool: &mut BlockPool) -> io::Result<()> {
        match self.col_type {
            ColumnType::Int64 | ColumnType::Float64 => {
                self.push_fixed_bytes(&[0u8; 8], pool)?;
            }
            ColumnType::UniqueId | ColumnType::Date => {
                self.push_fixed_bytes(&[0u8; 4], pool)?;
            }
            ColumnType::Bool => {
                self.push_fixed_bytes(&[0u8; 1], pool)?;
            }
            ColumnType::Str => {
                // Null string: push same offset (zero-length range)
                self.push_offset(self.str_data_len, pool)?;
            }
        }
        self.push_null_byte(1, pool)?;
        Ok(())
    }

    /// Read all blocks in a chain, concatenating their used portions.
    fn read_chain(
        &self,
        blocks: &[BlockId],
        last_pos: usize,
        pool: &mut BlockPool,
    ) -> io::Result<Vec<u8>> {
        if blocks.is_empty() {
            return Ok(Vec::new());
        }
        let full_blocks = blocks.len() - 1;
        let total = full_blocks * BLOCK_SIZE + last_pos;
        let mut result = Vec::with_capacity(total);

        for (i, &id) in blocks.iter().enumerate() {
            let data = pool.get(id)?;
            let len = if i < full_blocks {
                BLOCK_SIZE
            } else {
                last_pos
            };
            result.extend_from_slice(&data[..len]);
        }
        Ok(result)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pool() -> BlockPool {
        let dir = tempfile::tempdir().unwrap();
        // 10 MB budget — plenty for tests
        BlockPool::new(10 << 20, dir.keep()).unwrap()
    }

    #[test]
    fn push_int64_values() {
        let mut pool = test_pool();
        let mut col = BlockColumn::new(ColumnType::Int64, &mut pool).unwrap();

        for i in 0..100i64 {
            col.push(&Value::Int64(i), &mut pool).unwrap().unwrap();
        }
        assert_eq!(col.row_count(), 100);

        let (data, nulls, offsets) = col.drain(&mut pool).unwrap();
        assert!(offsets.is_none());
        assert_eq!(data.len(), 100 * 8);
        assert_eq!(nulls.len(), 100);

        // Verify first and last values
        let first = i64::from_ne_bytes(data[0..8].try_into().unwrap());
        let last = i64::from_ne_bytes(data[792..800].try_into().unwrap());
        assert_eq!(first, 0);
        assert_eq!(last, 99);

        // All non-null
        assert!(nulls.iter().all(|&b| b == 0));
    }

    #[test]
    fn push_with_nulls() {
        let mut pool = test_pool();
        let mut col = BlockColumn::new(ColumnType::Int64, &mut pool).unwrap();

        col.push(&Value::Int64(42), &mut pool).unwrap().unwrap();
        col.push(&Value::Null, &mut pool).unwrap().unwrap();
        col.push(&Value::Int64(99), &mut pool).unwrap().unwrap();

        let (data, nulls, _) = col.drain(&mut pool).unwrap();
        assert_eq!(data.len(), 3 * 8);
        assert_eq!(nulls, vec![0, 1, 0]);
    }

    #[test]
    fn push_string_values() {
        let mut pool = test_pool();
        let mut col = BlockColumn::new(ColumnType::Str, &mut pool).unwrap();

        col.push(&Value::String("hello".into()), &mut pool)
            .unwrap()
            .unwrap();
        col.push(&Value::String("world".into()), &mut pool)
            .unwrap()
            .unwrap();
        col.push(&Value::Null, &mut pool).unwrap().unwrap();

        assert_eq!(col.row_count(), 3);

        let (data, nulls, offsets) = col.drain(&mut pool).unwrap();
        let offsets = offsets.unwrap();

        // String data: "helloworld"
        assert_eq!(&data, b"helloworld");
        assert_eq!(nulls, vec![0, 0, 1]);

        // Offsets: [5, 10, 10] (null string gets same offset)
        let off0 = u64::from_ne_bytes(offsets[0..8].try_into().unwrap());
        let off1 = u64::from_ne_bytes(offsets[8..16].try_into().unwrap());
        let off2 = u64::from_ne_bytes(offsets[16..24].try_into().unwrap());
        assert_eq!(off0, 5);
        assert_eq!(off1, 10);
        assert_eq!(off2, 10); // null → same offset as previous
    }

    #[test]
    fn backfill_nulls() {
        let mut pool = test_pool();
        let mut col = BlockColumn::new(ColumnType::Int64, &mut pool).unwrap();

        col.backfill_nulls(5, &mut pool).unwrap();
        col.push(&Value::Int64(42), &mut pool).unwrap().unwrap();

        assert_eq!(col.row_count(), 6);
        let (data, nulls, _) = col.drain(&mut pool).unwrap();
        assert_eq!(data.len(), 6 * 8);
        assert_eq!(nulls, vec![1, 1, 1, 1, 1, 0]);

        let val = i64::from_ne_bytes(data[40..48].try_into().unwrap());
        assert_eq!(val, 42);
    }

    #[test]
    fn type_mismatch_returns_err() {
        let mut pool = test_pool();
        let mut col = BlockColumn::new(ColumnType::Int64, &mut pool).unwrap();
        let result = col.push(&Value::String("oops".into()), &mut pool).unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn float_int_promotion() {
        let mut pool = test_pool();
        let mut col = BlockColumn::new(ColumnType::Float64, &mut pool).unwrap();
        col.push(&Value::Int64(42), &mut pool).unwrap().unwrap();

        let (data, nulls, _) = col.drain(&mut pool).unwrap();
        let val = f64::from_ne_bytes(data[0..8].try_into().unwrap());
        assert_eq!(val, 42.0);
        assert_eq!(nulls, vec![0]);
    }

    #[test]
    fn bool_column() {
        let mut pool = test_pool();
        let mut col = BlockColumn::new(ColumnType::Bool, &mut pool).unwrap();
        col.push(&Value::Boolean(true), &mut pool).unwrap().unwrap();
        col.push(&Value::Boolean(false), &mut pool)
            .unwrap()
            .unwrap();

        let (data, nulls, _) = col.drain(&mut pool).unwrap();
        assert_eq!(data, vec![1, 0]);
        assert_eq!(nulls, vec![0, 0]);
    }

    #[test]
    fn many_values_span_multiple_blocks() {
        let mut pool = test_pool();
        let mut col = BlockColumn::new(ColumnType::Int64, &mut pool).unwrap();

        // 1 MB / 8 bytes = 131072 values per block. Push 200K to span 2 blocks.
        let n = 200_000u32;
        for i in 0..n {
            col.push(&Value::Int64(i as i64), &mut pool)
                .unwrap()
                .unwrap();
        }
        assert_eq!(col.row_count(), n);
        assert!(col.data_blocks.len() >= 2);

        let (data, nulls, _) = col.drain(&mut pool).unwrap();
        assert_eq!(data.len(), n as usize * 8);
        assert_eq!(nulls.len(), n as usize);

        // Verify boundary values
        let first = i64::from_ne_bytes(data[0..8].try_into().unwrap());
        let last_offset = (n as usize - 1) * 8;
        let last = i64::from_ne_bytes(data[last_offset..last_offset + 8].try_into().unwrap());
        assert_eq!(first, 0);
        assert_eq!(last, n as i64 - 1);
    }

    #[test]
    fn large_strings_span_blocks() {
        let mut pool = test_pool();
        let mut col = BlockColumn::new(ColumnType::Str, &mut pool).unwrap();

        // Push a string larger than one block
        let big = "x".repeat(BLOCK_SIZE + 100);
        col.push(&Value::String(big.clone()), &mut pool)
            .unwrap()
            .unwrap();
        col.push(&Value::String("small".into()), &mut pool)
            .unwrap()
            .unwrap();

        let (data, nulls, offsets) = col.drain(&mut pool).unwrap();
        let offsets = offsets.unwrap();

        assert_eq!(data.len(), big.len() + 5);
        assert_eq!(&data[..big.len()], big.as_bytes());
        assert_eq!(&data[big.len()..], b"small");

        let off0 = u64::from_ne_bytes(offsets[0..8].try_into().unwrap());
        let off1 = u64::from_ne_bytes(offsets[8..16].try_into().unwrap());
        assert_eq!(off0, big.len() as u64);
        assert_eq!(off1, big.len() as u64 + 5);
        assert_eq!(nulls, vec![0, 0]);
    }

    #[test]
    fn eviction_during_build() {
        // Tiny budget: 3 MB — building many values will force eviction
        let dir = tempfile::tempdir().unwrap();
        let mut pool = BlockPool::new(3 << 20, dir.keep()).unwrap();
        let mut col = BlockColumn::new(ColumnType::Int64, &mut pool).unwrap();

        let n = 500_000u32; // ~4 MB of data → must evict some blocks
        for i in 0..n {
            col.push(&Value::Int64(i as i64), &mut pool)
                .unwrap()
                .unwrap();
        }

        assert!(pool.eviction_count() > 0, "expected evictions to occur");

        // Verify all data survives round-trip through eviction
        let (data, _, _) = col.drain(&mut pool).unwrap();
        assert_eq!(data.len(), n as usize * 8);

        let first = i64::from_ne_bytes(data[0..8].try_into().unwrap());
        let last_off = (n as usize - 1) * 8;
        let last = i64::from_ne_bytes(data[last_off..last_off + 8].try_into().unwrap());
        assert_eq!(first, 0);
        assert_eq!(last, n as i64 - 1);
    }

    #[test]
    fn free_all_releases_blocks() {
        let mut pool = test_pool();
        let mut col = BlockColumn::new(ColumnType::Int64, &mut pool).unwrap();
        for i in 0..1000i64 {
            col.push(&Value::Int64(i), &mut pool).unwrap().unwrap();
        }
        let before = pool.memory_used();
        assert!(before > 0);

        col.free_all(&mut pool);
        assert!(pool.memory_used() < before);
    }
}
