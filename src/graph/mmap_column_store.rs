// src/graph/mmap_column_store.rs
//
// Disk-backed column store that reads directly from a shared mmap file.
// No heap allocation for column data — values are read from mmap at query time.
// The OS page cache manages which pages are resident.
//
// String offset convention: offsets[row] = cumulative end byte for that row.
// Row 0 starts at byte 0; row i>0 starts at offsets[i-1].
// (No leading-zero prefix — same as ntriples build_columns_direct.)
//
// Null bitmap convention: 0 = non-null, non-zero = null (same as ColumnStore).

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use crate::graph::type_build_meta::ColType;
use chrono::NaiveDate;
use memmap2::MmapMut;
use std::collections::HashMap;
use std::sync::Arc;

const UNIX_EPOCH_DATE: NaiveDate = match NaiveDate::from_ymd_opt(1970, 1, 1) {
    Some(d) => d,
    None => unreachable!(),
};

// ─── Region ──────────────────────────────────────────────────────────────────

/// Byte region in the shared mmap file.
/// A region with `len == 0` means "not present".
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub offset: usize,
    pub len: usize, // in bytes
}

impl Region {
    /// A zero-length sentinel meaning "not present".
    pub const EMPTY: Region = Region { offset: 0, len: 0 };
}

// ─── Column metadata ─────────────────────────────────────────────────────────

/// Metadata for a fixed-width column in the mmap.
#[derive(Clone, Debug)]
pub struct FixedColumnMeta {
    pub col_type: ColType,
    pub data: Region,  // raw typed data
    pub nulls: Region, // null bitmap (1 byte per row)
}

/// Metadata for a string column in the mmap.
#[derive(Clone, Debug)]
pub struct StrColumnMeta {
    pub data: Region,    // string bytes
    pub offsets: Region, // u64 offset array (one per row, cumulative end)
    pub nulls: Region,   // null bitmap
}

/// Reference to a column in the MmapColumnStore.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ColRef {
    Fixed(usize), // index into fixed_cols
    Str(usize),   // index into str_cols
}

// ─── MmapColumnStore ─────────────────────────────────────────────────────────

/// Per-type column store backed by a shared mmap.
///
/// Clone is cheap: it clones the `Arc` and small metadata vecs, not the mmap data.
#[derive(Clone, Debug)]
pub struct MmapColumnStore {
    /// Shared reference to the mmap file containing all column data.
    pub(crate) mmap: Arc<MmapMut>,
    /// Number of rows in this type.
    pub(crate) row_count: u32,
    /// Whether the id column is stored as string (true) or UniqueId (false).
    pub(crate) id_is_string: bool,
    /// Id column if stored as UniqueId (fixed-width).
    pub(crate) id_fixed: Option<FixedColumnMeta>,
    /// Id column if stored as string.
    pub(crate) id_str: Option<StrColumnMeta>,
    /// Title column (always string).
    pub(crate) title: StrColumnMeta,
    /// Property key → column reference (fixed or string).
    pub(crate) col_map: HashMap<InternedKey, ColRef>,
    /// Dense fixed-width columns.
    pub(crate) fixed_cols: Vec<FixedColumnMeta>,
    /// Dense string columns.
    pub(crate) str_cols: Vec<StrColumnMeta>,
    /// Overflow bag offset array region: u64 array, row_count+1 entries.
    pub(crate) overflow_offsets: Region,
    /// Overflow bag serialized data region.
    pub(crate) overflow_data: Region,
    /// Whether this type has overflow data.
    pub(crate) has_overflow: bool,
}

// ─── Constructor ─────────────────────────────────────────────────────────────

impl MmapColumnStore {
    /// Create an empty store with no columns. Caller should set fields directly
    /// or use a builder before calling read methods.
    pub fn new(mmap: Arc<MmapMut>, row_count: u32) -> Self {
        MmapColumnStore {
            mmap,
            row_count,
            id_is_string: false,
            id_fixed: None,
            id_str: None,
            title: StrColumnMeta {
                data: Region::EMPTY,
                offsets: Region::EMPTY,
                nulls: Region::EMPTY,
            },
            col_map: HashMap::new(),
            fixed_cols: Vec::new(),
            str_cols: Vec::new(),
            overflow_offsets: Region::EMPTY,
            overflow_data: Region::EMPTY,
            has_overflow: false,
        }
    }

    /// Number of rows in this type.
    #[inline]
    pub fn row_count(&self) -> u32 {
        self.row_count
    }
}

// ─── Low-level mmap readers ──────────────────────────────────────────────────

impl MmapColumnStore {
    /// Read the null bitmap at `row`. Returns `true` if the value IS null.
    #[inline]
    fn read_null(&self, region: &Region, row: usize) -> bool {
        if region.len == 0 {
            return true;
        }
        self.mmap[region.offset + row] != 0
    }

    #[inline]
    fn read_i64(&self, region: &Region, row: usize) -> i64 {
        let off = region.offset + row * 8;
        i64::from_ne_bytes(self.mmap[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn read_f64(&self, region: &Region, row: usize) -> f64 {
        let off = region.offset + row * 8;
        f64::from_ne_bytes(self.mmap[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn read_u64(&self, region: &Region, row: usize) -> u64 {
        let off = region.offset + row * 8;
        u64::from_ne_bytes(self.mmap[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn read_u32(&self, region: &Region, row: usize) -> u32 {
        let off = region.offset + row * 4;
        u32::from_ne_bytes(self.mmap[off..off + 4].try_into().unwrap())
    }

    #[inline]
    fn read_i32(&self, region: &Region, row: usize) -> i32 {
        let off = region.offset + row * 4;
        i32::from_ne_bytes(self.mmap[off..off + 4].try_into().unwrap())
    }

    #[inline]
    fn read_u8(&self, region: &Region, row: usize) -> u8 {
        self.mmap[region.offset + row]
    }

    /// Read a string from the mmap for the given row.
    ///
    /// Offset convention: `offsets[row]` is the cumulative end byte.
    /// Row 0 starts at byte 0; row i>0 starts at `offsets[i-1]`.
    #[inline]
    fn read_str(&self, data_region: &Region, offsets_region: &Region, row: usize) -> &str {
        let end = self.read_u64(offsets_region, row) as usize;
        let start = if row > 0 {
            self.read_u64(offsets_region, row - 1) as usize
        } else {
            0
        };
        let bytes = &self.mmap[data_region.offset + start..data_region.offset + end];
        std::str::from_utf8(bytes).unwrap_or("")
    }
}

// ─── Public accessors ────────────────────────────────────────────────────────

#[allow(dead_code)]
impl MmapColumnStore {
    /// Read the node ID for a given row.
    pub fn get_id(&self, row_id: u32) -> Option<Value> {
        if row_id >= self.row_count {
            return None;
        }
        let row = row_id as usize;
        if self.id_is_string {
            let sc = self.id_str.as_ref()?;
            if self.read_null(&sc.nulls, row) {
                return None;
            }
            Some(Value::String(
                self.read_str(&sc.data, &sc.offsets, row).to_string(),
            ))
        } else {
            let fc = self.id_fixed.as_ref()?;
            if self.read_null(&fc.nulls, row) {
                return None;
            }
            Some(Value::UniqueId(self.read_u32(&fc.data, row)))
        }
    }

    /// Read the node title for a given row.
    pub fn get_title(&self, row_id: u32) -> Option<Value> {
        if row_id >= self.row_count {
            return None;
        }
        let row = row_id as usize;
        if self.title.nulls.len == 0 && self.title.data.len == 0 {
            return None;
        }
        if self.read_null(&self.title.nulls, row) {
            return None;
        }
        Some(Value::String(
            self.read_str(&self.title.data, &self.title.offsets, row)
                .to_string(),
        ))
    }

    /// Read a property value by (row_id, interned key).
    /// Checks dense columns first, falls back to the overflow bag.
    pub fn get(&self, row_id: u32, key: InternedKey) -> Option<Value> {
        if row_id >= self.row_count {
            return None;
        }
        let row = row_id as usize;

        if let Some(col_ref) = self.col_map.get(&key) {
            match col_ref {
                ColRef::Fixed(idx) => {
                    let fc = &self.fixed_cols[*idx];
                    if !self.read_null(&fc.nulls, row) {
                        return Some(self.read_fixed_value(fc, row));
                    }
                }
                ColRef::Str(idx) => {
                    let sc = &self.str_cols[*idx];
                    if !self.read_null(&sc.nulls, row) {
                        let s = self.read_str(&sc.data, &sc.offsets, row);
                        return Some(Value::String(s.to_string()));
                    }
                }
            }
        }

        // Fall back to overflow bag
        self.get_overflow_property(row_id, key)
    }

    /// Read a fixed-width value from the mmap, dispatching on ColType.
    #[inline]
    fn read_fixed_value(&self, fc: &FixedColumnMeta, row: usize) -> Value {
        match fc.col_type {
            ColType::Int64 => Value::Int64(self.read_i64(&fc.data, row)),
            ColType::Float64 => Value::Float64(self.read_f64(&fc.data, row)),
            ColType::UniqueId => Value::UniqueId(self.read_u32(&fc.data, row)),
            ColType::Bool => Value::Boolean(self.read_u8(&fc.data, row) != 0),
            ColType::Date => {
                let days = self.read_i32(&fc.data, row);
                let date = UNIX_EPOCH_DATE + chrono::Duration::days(days as i64);
                Value::DateTime(date)
            }
            ColType::Str => unreachable!("string columns use ColRef::Str"),
        }
    }

    /// Iterate over all non-null properties for a row.
    /// Returns (InternedKey, Value) pairs from both dense columns and overflow bag.
    pub fn row_properties(&self, row_id: u32) -> Vec<(InternedKey, Value)> {
        if row_id >= self.row_count {
            return Vec::new();
        }
        let row = row_id as usize;
        let mut result = Vec::new();

        for (&key, col_ref) in &self.col_map {
            match col_ref {
                ColRef::Fixed(idx) => {
                    let fc = &self.fixed_cols[*idx];
                    if !self.read_null(&fc.nulls, row) {
                        result.push((key, self.read_fixed_value(fc, row)));
                    }
                }
                ColRef::Str(idx) => {
                    let sc = &self.str_cols[*idx];
                    if !self.read_null(&sc.nulls, row) {
                        let s = self.read_str(&sc.data, &sc.offsets, row);
                        result.push((key, Value::String(s.to_string())));
                    }
                }
            }
        }

        // Append overflow bag properties
        result.extend(self.overflow_row_properties(row_id));
        result
    }
}

// ─── Overflow bag ────────────────────────────────────────────────────────────
//
// Format per entity blob: [num_entries: u16]
// then repeated: [key_u64: 8B][type_tag: 1B][value_bytes: variable]
//
// Type tags: 0=Null, 1=Int64(8B), 2=Float64(8B), 3=UniqueId(4B),
//            4=Bool(1B), 5=Date(4B i32), 6=String(u32 len + bytes)

#[allow(dead_code)]
impl MmapColumnStore {
    /// Look up a single property in the overflow bag for a given row.
    pub fn get_overflow_property(&self, row_id: u32, key: InternedKey) -> Option<Value> {
        let blob = self.overflow_blob(row_id)?;
        Self::scan_overflow_blob(blob, key)
    }

    /// Decode all properties from the overflow bag for a given row.
    fn overflow_row_properties(&self, row_id: u32) -> Vec<(InternedKey, Value)> {
        match self.overflow_blob(row_id) {
            Some(blob) => Self::decode_overflow_blob(blob),
            None => Vec::new(),
        }
    }

    /// Extract the overflow blob slice for a given row, or None if not present.
    fn overflow_blob(&self, row_id: u32) -> Option<&[u8]> {
        if !self.has_overflow {
            return None;
        }
        let idx = row_id as usize;
        // overflow_offsets has row_count + 1 entries (u64 each)
        let expected_len = (self.row_count as usize + 1) * 8;
        if self.overflow_offsets.len < expected_len {
            return None;
        }
        let start = self.read_u64(&self.overflow_offsets, idx) as usize;
        let end = self.read_u64(&self.overflow_offsets, idx + 1) as usize;
        if start >= end || end > self.overflow_data.len {
            return None;
        }
        Some(&self.mmap[self.overflow_data.offset + start..self.overflow_data.offset + end])
    }

    /// Scan an overflow blob for a specific key. Returns the value if found.
    fn scan_overflow_blob(blob: &[u8], key: InternedKey) -> Option<Value> {
        let target = key.as_u64();
        if blob.len() < 2 {
            return None;
        }
        let num_entries = u16::from_le_bytes([blob[0], blob[1]]) as usize;
        let mut pos = 2;
        for _ in 0..num_entries {
            if pos + 9 > blob.len() {
                break;
            }
            let entry_key = u64::from_le_bytes(blob[pos..pos + 8].try_into().ok()?);
            let type_tag = blob[pos + 8];
            pos += 9;
            if entry_key == target {
                return Self::read_overflow_value(blob, &mut pos, type_tag);
            }
            // Skip value
            Self::skip_overflow_value(blob, &mut pos, type_tag);
        }
        None
    }

    /// Decode all entries from an overflow blob.
    fn decode_overflow_blob(blob: &[u8]) -> Vec<(InternedKey, Value)> {
        if blob.len() < 2 {
            return Vec::new();
        }
        let num_entries = u16::from_le_bytes([blob[0], blob[1]]) as usize;
        let mut result = Vec::with_capacity(num_entries);
        let mut pos = 2;
        for _ in 0..num_entries {
            if pos + 9 > blob.len() {
                break;
            }
            let entry_key = u64::from_le_bytes(blob[pos..pos + 8].try_into().unwrap_or([0; 8]));
            let type_tag = blob[pos + 8];
            pos += 9;
            let key = InternedKey::from_u64(entry_key);
            if let Some(val) = Self::read_overflow_value(blob, &mut pos, type_tag) {
                result.push((key, val));
            }
        }
        result
    }

    /// Read a single value from the overflow blob at the given position.
    fn read_overflow_value(blob: &[u8], pos: &mut usize, type_tag: u8) -> Option<Value> {
        match type_tag {
            0 => Some(Value::Null),
            1 => {
                // Int64: 8 bytes
                if *pos + 8 > blob.len() {
                    return None;
                }
                let v = i64::from_le_bytes(blob[*pos..*pos + 8].try_into().ok()?);
                *pos += 8;
                Some(Value::Int64(v))
            }
            2 => {
                // Float64: 8 bytes
                if *pos + 8 > blob.len() {
                    return None;
                }
                let v = f64::from_le_bytes(blob[*pos..*pos + 8].try_into().ok()?);
                *pos += 8;
                Some(Value::Float64(v))
            }
            3 => {
                // UniqueId: 4 bytes
                if *pos + 4 > blob.len() {
                    return None;
                }
                let v = u32::from_le_bytes(blob[*pos..*pos + 4].try_into().ok()?);
                *pos += 4;
                Some(Value::UniqueId(v))
            }
            4 => {
                // Bool: 1 byte
                if *pos + 1 > blob.len() {
                    return None;
                }
                let v = blob[*pos] != 0;
                *pos += 1;
                Some(Value::Boolean(v))
            }
            5 => {
                // Date: 4 bytes as i32 days since epoch
                if *pos + 4 > blob.len() {
                    return None;
                }
                let days = i32::from_le_bytes(blob[*pos..*pos + 4].try_into().ok()?);
                *pos += 4;
                let date = UNIX_EPOCH_DATE + chrono::Duration::days(days as i64);
                Some(Value::DateTime(date))
            }
            6 => {
                // String: u32 length prefix + bytes
                if *pos + 4 > blob.len() {
                    return None;
                }
                let slen = u32::from_le_bytes(blob[*pos..*pos + 4].try_into().ok()?) as usize;
                *pos += 4;
                if *pos + slen > blob.len() {
                    return None;
                }
                let s = String::from_utf8_lossy(&blob[*pos..*pos + slen]).into_owned();
                *pos += slen;
                Some(Value::String(s))
            }
            _ => None,
        }
    }

    /// Skip over a value in the overflow blob without decoding it.
    fn skip_overflow_value(blob: &[u8], pos: &mut usize, type_tag: u8) {
        match type_tag {
            0 => {}             // Null: no data
            1 | 2 => *pos += 8, // Int64 or Float64
            3 | 5 => *pos += 4, // UniqueId or Date
            4 => *pos += 1,     // Bool
            6 if *pos + 4 <= blob.len() => {
                // String: u32 length prefix + bytes
                let slen =
                    u32::from_le_bytes(blob[*pos..*pos + 4].try_into().unwrap_or([0; 4])) as usize;
                *pos += 4 + slen;
            }
            _ => {}
        }
    }
}
