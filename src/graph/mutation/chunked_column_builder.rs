//! Chunked column-store materializer for the streaming subgraph filter.
//!
//! Mirrors `csr_build::merge_sort_build`'s memory shape: fill a heap-
//! bounded chunk, write it to mmap'd files, **drop the chunk** to
//! release file handles, repeat. After all rows pushed, merge per-chunk
//! files into the destination's canonical column-store layout.
//!
//! Why "drop the chunk" matters: macOS's Unified Buffer Cache holds dirty
//! file-backed mmap pages resident regardless of `madvise(MADV_DONTNEED)`.
//! Closing the file handle (which happens when the `MmapOrVec::Mapped`
//! struct drops) is what triggers eviction. Keeping a single growing
//! ColumnStore alive for the entire push loop — as the v1 streaming
//! pipeline did — kept all dirty pages resident and drove peak RSS to
//! 8.2 GB on Wikidata Articles+P50+Authors. v2 chunks each push run into
//! short-lived ColumnStores so the kernel can free their pages between
//! chunks; peak heap stays bounded by `chunk_size_rows × per_row_bytes`.
//!
//! Per-chunk on-disk layout (matches `ColumnStore::materialize_to_files`):
//!
//!   chunks/chunk_000000/<col>.i64   + <col>.null    (Int64)
//!   chunks/chunk_000000/<col>.f64   + <col>.null    (Float64)
//!   chunks/chunk_000000/<col>.u32   + <col>.null    (UniqueId)
//!   chunks/chunk_000000/<col>.bool  + <col>.null    (Bool)
//!   chunks/chunk_000000/<col>.i32   + <col>.null    (Date)
//!   chunks/chunk_000000/<col>.off   + <col>.str + <col>.null   (Str)
//!   chunks/chunk_000000/<col>.mixed.zst                (Mixed; sidecar
//!                                                       since
//!                                                       materialize_to_files
//!                                                       skips Mixed)
//!
//! Per-typed-column merge (in `finalize`):
//!
//!   Fixed-size: byte-concat data + nulls files; truncate to logical len.
//!   Str:        byte-concat data blob; concat offsets with running-
//!               cumulative rebase (skip duplicate leading 0 on chunks
//!               1..N).
//!   Mixed:      deserialize each chunk's bincode-zstd, concat
//!               `Vec<Value>`, hand back as `TypedColumn::Mixed`.

use crate::datatypes::values::Value;
use crate::graph::schema::{InternedKey, StringInterner, TypeSchema};
use crate::graph::storage::column_store::{ColumnStore, TypedColumn};
use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// On-disk record of one spilled chunk.
struct ChunkInfo {
    dir: PathBuf,
    /// Row count actually pushed into this chunk (not the chunk-size cap).
    rows: u32,
}

/// Push rows in bounded-heap chunks; merge into a final ColumnStore.
pub(super) struct ChunkedColumnBuilder {
    schema: Arc<TypeSchema>,
    /// Property-name → type-string. Cloned per chunk's ColumnStore::new.
    meta: HashMap<String, String>,
    /// Where chunks live; each chunk gets a numbered subdirectory.
    chunk_root: PathBuf,
    chunk_size_rows: u32,

    /// True → seed each chunk's id_column with an empty
    /// `TypedColumn::Str` before the first `push_id`, so push_id
    /// appends to mmap-backed Str instead of lazy heap-Mixed. Caller
    /// checks the source's id column variant; falls back to Mixed if
    /// ambiguous.
    prefer_str_id: bool,

    current: Option<ColumnStore>,
    /// 0 if no rows pushed into `current` yet.
    current_rows: u32,
    chunks: Vec<ChunkInfo>,
    total_rows: u32,
}

impl ChunkedColumnBuilder {
    /// Initialize a new builder. `chunk_root` does NOT need to exist
    /// yet; per-chunk subdirs are created on spill.
    pub(super) fn new(
        schema: Arc<TypeSchema>,
        meta: HashMap<String, String>,
        chunk_root: PathBuf,
        chunk_size_rows: u32,
        prefer_str_id: bool,
    ) -> Self {
        Self {
            schema,
            meta,
            chunk_root,
            chunk_size_rows: chunk_size_rows.max(1),
            prefer_str_id,
            current: None,
            current_rows: 0,
            chunks: Vec::new(),
            total_rows: 0,
        }
    }

    /// Push one row of (id, title, properties). Returns the assigned
    /// destination row_id (global, across all chunks).
    pub(super) fn push_row(
        &mut self,
        id: &Value,
        title: &Value,
        props: &[(InternedKey, Value)],
        interner: &StringInterner,
    ) -> io::Result<u32> {
        if self.current.is_none() {
            let mut store = ColumnStore::new(Arc::clone(&self.schema), &self.meta, interner);
            if self.prefer_str_id {
                // Pre-seed id_column with empty Str so push_id appends
                // there instead of lazy-creating Mixed. push_id's
                // type-mismatch path demotes-to-Mixed if a non-string
                // ever arrives, so this is safe even when the caller's
                // detection was a bit optimistic.
                store.replace_id_column(TypedColumn::from_type_str("string"));
            }
            self.current = Some(store);
        }
        let store = self
            .current
            .as_mut()
            .expect("ChunkedColumnBuilder: store must be initialized after lazy-create");
        store.push_id(id);
        store.push_title(title);
        let _row_id_within_chunk = store.push_row(props);

        let global_row_id = self.total_rows;
        self.total_rows = self
            .total_rows
            .checked_add(1)
            .expect("ChunkedColumnBuilder: row count overflow (>u32::MAX rows)");
        self.current_rows += 1;

        if self.current_rows >= self.chunk_size_rows {
            self.spill_current_chunk(interner)?;
        }
        Ok(global_row_id)
    }

    /// True when no chunks have been spilled and `current` is None or empty.
    #[allow(dead_code)]
    pub(super) fn is_empty(&self) -> bool {
        self.total_rows == 0
    }

    fn spill_current_chunk(&mut self, interner: &StringInterner) -> io::Result<()> {
        let mut store = match self.current.take() {
            Some(s) => s,
            None => return Ok(()),
        };
        if self.current_rows == 0 {
            return Ok(());
        }
        let chunk_idx = self.chunks.len();
        let chunk_dir = self.chunk_root.join(format!("chunk_{:06}", chunk_idx));
        std::fs::create_dir_all(&chunk_dir)?;

        // Typed columns + nulls + str offsets/data → mmap files.
        store.materialize_to_files(&chunk_dir, interner)?;
        // Mixed columns: write bincode-zstd sidecars so the chunk dir
        // is fully self-contained (materialize_to_files skips Mixed).
        write_mixed_sidecars(&store, &chunk_dir, &self.schema, interner)?;

        // Dropping `store` closes file handles → kernel evicts the
        // chunk's mmap pages. THIS is the point of chunking.
        drop(store);

        self.chunks.push(ChunkInfo {
            dir: chunk_dir,
            rows: self.current_rows,
        });
        self.current_rows = 0;
        Ok(())
    }

    /// Spill the partial last chunk (if any), then merge all chunks
    /// into the destination's canonical column-store layout under
    /// `final_dir`. Returns an `Arc<ColumnStore>` whose typed columns
    /// are mmap-backed at `final_dir`, ready to install in
    /// `dir_graph.column_stores`.
    ///
    /// Cleans up per-chunk subdirectories on success; on error, leaves
    /// them in place for forensics.
    pub(super) fn finalize(
        mut self,
        final_dir: &Path,
        interner: &StringInterner,
    ) -> io::Result<Arc<ColumnStore>> {
        if self.current_rows > 0 {
            self.spill_current_chunk(interner)?;
        }
        std::fs::create_dir_all(final_dir)?;
        let total_rows = self.total_rows;

        // Merge per-property-slot columns.
        let mut merged_columns: Vec<TypedColumn> = Vec::with_capacity(self.schema.len());
        for (_slot, ik) in self.schema.iter() {
            let col_name = interner.resolve(ik);
            let type_str = self
                .meta
                .get(col_name)
                .map(|s| s.as_str())
                .unwrap_or("mixed");
            let merged =
                merge_typed_column(type_str, col_name, &self.chunks, final_dir, total_rows)?;
            merged_columns.push(merged);
        }

        // id and title — written by ColumnStore::materialize_to_files
        // under fixed names "__id__" and "__title__". The variant the
        // chunks used depends on lazy-create / prefer_str_id; probe
        // the first chunk to find which file extensions exist.
        let id_column = merge_id_or_title("__id__", &self.chunks, final_dir, total_rows)?;
        let title_column = merge_id_or_title("__title__", &self.chunks, final_dir, total_rows)?;

        // Build the final ColumnStore shell and inject merged columns.
        let mut store = ColumnStore::new(Arc::clone(&self.schema), &self.meta, interner);
        store.replace_columns(merged_columns);
        if let Some(c) = id_column {
            store.replace_id_column(c);
        }
        if let Some(c) = title_column {
            store.replace_title_column(c);
        }
        store.set_row_count(total_rows);

        // Cleanup chunk subdirs.
        for chunk in &self.chunks {
            let _ = std::fs::remove_dir_all(&chunk.dir);
        }

        Ok(Arc::new(store))
    }
}

// ─── Per-chunk Mixed sidecar I/O ────────────────────────────────────

fn write_mixed_sidecars(
    store: &ColumnStore,
    chunk_dir: &Path,
    schema: &TypeSchema,
    interner: &StringInterner,
) -> io::Result<()> {
    for (slot, ik) in schema.iter() {
        // Only write the sidecar if this slot's variant is Mixed.
        if store.column_type_str(slot as usize) != Some("mixed") {
            continue;
        }
        let Some(values) = store.column_values_mixed(slot as usize) else {
            continue;
        };
        let col_name = interner.resolve(ik);
        let bytes = bincode::serialize(values).map_err(io::Error::other)?;
        let compressed = zstd::encode_all(bytes.as_slice(), 3)?;
        std::fs::write(chunk_dir.join(format!("{col_name}.mixed.zst")), compressed)?;
    }
    Ok(())
}

fn read_mixed_chunk(chunk_dir: &Path, col_name: &str) -> io::Result<Vec<Value>> {
    let path = chunk_dir.join(format!("{col_name}.mixed.zst"));
    if !path.exists() {
        return Ok(Vec::new());
    }
    let compressed = std::fs::read(path)?;
    let raw = zstd::decode_all(compressed.as_slice())?;
    bincode::deserialize::<Vec<Value>>(&raw).map_err(io::Error::other)
}

// ─── Per-typed-column merge dispatch ────────────────────────────────

fn merge_typed_column(
    type_str: &str,
    col_name: &str,
    chunks: &[ChunkInfo],
    final_dir: &Path,
    total_rows: u32,
) -> io::Result<TypedColumn> {
    match type_str.to_ascii_lowercase().as_str() {
        "int64" => merge_fixed::<i64>(
            col_name,
            chunks,
            final_dir,
            "i64",
            std::mem::size_of::<i64>(),
            total_rows,
            |data, nulls| TypedColumn::Int64 { data, nulls },
        ),
        "float64" => merge_fixed::<f64>(
            col_name,
            chunks,
            final_dir,
            "f64",
            std::mem::size_of::<f64>(),
            total_rows,
            |data, nulls| TypedColumn::Float64 { data, nulls },
        ),
        "uniqueid" => merge_fixed::<u32>(
            col_name,
            chunks,
            final_dir,
            "u32",
            std::mem::size_of::<u32>(),
            total_rows,
            |data, nulls| TypedColumn::UniqueId { data, nulls },
        ),
        "bool" | "boolean" => merge_fixed::<u8>(
            col_name,
            chunks,
            final_dir,
            "bool",
            std::mem::size_of::<u8>(),
            total_rows,
            |data, nulls| TypedColumn::Bool { data, nulls },
        ),
        "date" | "datetime" => merge_fixed::<i32>(
            col_name,
            chunks,
            final_dir,
            "i32",
            std::mem::size_of::<i32>(),
            total_rows,
            |data, nulls| TypedColumn::Date { data, nulls },
        ),
        "string" => merge_str(col_name, chunks, final_dir, total_rows as usize),
        _ => merge_mixed_column(col_name, chunks),
    }
}

fn merge_fixed<T>(
    col_name: &str,
    chunks: &[ChunkInfo],
    final_dir: &Path,
    data_ext: &str,
    elem_size: usize,
    total_rows: u32,
    make_col: impl FnOnce(MmapOrVec<T>, MmapOrVec<u8>) -> TypedColumn,
) -> io::Result<TypedColumn>
where
    T: Copy + Default + 'static,
{
    let final_data = final_dir.join(format!("{col_name}.{data_ext}"));
    let final_nulls = final_dir.join(format!("{col_name}.null"));

    // Concat data bytes + nulls bytes. Each chunk's file may have
    // mmap-padding past its logical length; use the per-chunk row count
    // to bound the read.
    write_concat_fixed(
        chunks.iter().map(|c| {
            (
                c.dir.join(format!("{col_name}.{data_ext}")),
                (c.rows as usize) * elem_size,
            )
        }),
        &final_data,
    )?;
    write_concat_fixed(
        chunks
            .iter()
            .map(|c| (c.dir.join(format!("{col_name}.null")), c.rows as usize)),
        &final_nulls,
    )?;
    truncate_to(&final_data, (total_rows as usize) * elem_size)?;
    truncate_to(&final_nulls, total_rows as usize)?;

    let data = MmapOrVec::<T>::load_mapped(&final_data, total_rows as usize)?;
    let nulls = MmapOrVec::<u8>::load_mapped(&final_nulls, total_rows as usize)?;
    Ok(make_col(data, nulls))
}

fn merge_str(
    col_name: &str,
    chunks: &[ChunkInfo],
    final_dir: &Path,
    total_rows: usize,
) -> io::Result<TypedColumn> {
    let final_offsets = final_dir.join(format!("{col_name}.off"));
    let final_data = final_dir.join(format!("{col_name}.str"));
    let final_nulls = final_dir.join(format!("{col_name}.null"));

    let mut offsets_w = open_truncate_for_write(&final_offsets)?;
    let mut data_w = open_truncate_for_write(&final_data)?;
    let mut nulls_w = open_truncate_for_write(&final_nulls)?;

    // Each chunk's offsets are relative to that chunk's data blob;
    // rebase by the running cumulative data length. Skip the duplicate
    // leading 0 on chunks 1..N so the merged offsets file has exactly
    // total_rows + 1 entries.
    let mut running_data_bytes: u64 = 0;
    let mut wrote_leading_zero = false;
    for (idx, chunk) in chunks.iter().enumerate() {
        let off_path = chunk.dir.join(format!("{col_name}.off"));
        let str_path = chunk.dir.join(format!("{col_name}.str"));
        let null_path = chunk.dir.join(format!("{col_name}.null"));

        // Offsets file: chunk has chunk.rows + 1 entries, each u64 LE.
        if off_path.exists() {
            let bytes = std::fs::read(&off_path)?;
            let logical_bytes = (chunk.rows as usize + 1) * 8;
            let logical_bytes = logical_bytes.min(bytes.len());
            let mut iter = bytes[..logical_bytes]
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()));
            // Chunk 0: keep all entries (including leading 0).
            // Chunks 1..: skip the leading 0.
            if idx > 0 {
                let _ = iter.next();
            } else {
                wrote_leading_zero = true;
            }
            for off in iter {
                let rebased = off + running_data_bytes;
                offsets_w.write_all(&rebased.to_le_bytes())?;
            }
        }

        // Data blob: raw concat. Honor logical length from the chunk's
        // offsets[rows] entry (mmap-padded data files would otherwise
        // corrupt subsequent chunks' offset rebases).
        if str_path.exists() {
            let logical_data_len = chunk_str_data_len(&chunk.dir, col_name, chunk.rows)?;
            let mut f = std::fs::File::open(&str_path)?;
            let mut taken = (&mut f).take(logical_data_len);
            let n = io::copy(&mut taken, &mut data_w)?;
            running_data_bytes += n;
        }

        // Nulls: 1 byte per row, simple concat with logical-length
        // bound (mmap-padded null files would over-read).
        if null_path.exists() {
            let mut f = std::fs::File::open(&null_path)?;
            let mut taken = (&mut f).take(chunk.rows as u64);
            io::copy(&mut taken, &mut nulls_w)?;
        }
    }
    // Edge case: no chunks at all (total_rows == 0) — write the implicit
    // leading 0 so the offsets file has the expected structure.
    if !wrote_leading_zero {
        offsets_w.write_all(&0u64.to_le_bytes())?;
    }

    drop(offsets_w);
    drop(data_w);
    drop(nulls_w);

    truncate_to(&final_offsets, (total_rows + 1) * 8)?;
    truncate_to(&final_data, running_data_bytes as usize)?;
    truncate_to(&final_nulls, total_rows)?;

    let offsets = MmapOrVec::<u64>::load_mapped(&final_offsets, total_rows + 1)?;
    let data = MmapBytes::load_mapped(&final_data, running_data_bytes as usize)?;
    let nulls = MmapOrVec::<u8>::load_mapped(&final_nulls, total_rows)?;
    Ok(TypedColumn::Str {
        offsets,
        data,
        nulls,
    })
}

fn chunk_str_data_len(chunk_dir: &Path, col_name: &str, chunk_rows: u32) -> io::Result<u64> {
    // The (rows)-th entry of the offsets file is the chunk's cumulative
    // data length — equal to the logical length of `<col>.str`. The
    // offsets file has `rows + 1` valid entries (offsets[0..=rows]), then
    // mmap-padding to the chunk's MmapOrVec capacity. Read at the
    // logical byte position rather than the trailing u64 of the file.
    let off_path = chunk_dir.join(format!("{col_name}.off"));
    if !off_path.exists() || chunk_rows == 0 {
        return Ok(0);
    }
    let mut f = std::fs::File::open(&off_path)?;
    let pos = (chunk_rows as u64) * 8;
    if f.metadata()?.len() < pos + 8 {
        return Ok(0);
    }
    f.seek(SeekFrom::Start(pos))?;
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn merge_mixed_column(col_name: &str, chunks: &[ChunkInfo]) -> io::Result<TypedColumn> {
    let mut all = Vec::new();
    for chunk in chunks {
        all.extend(read_mixed_chunk(&chunk.dir, col_name)?);
    }
    Ok(TypedColumn::Mixed { data: all })
}

/// Probe the first non-empty chunk for which file extensions exist; use
/// that to dispatch to the right merge kernel. Used for `__id__` and
/// `__title__` since their variant depends on lazy-create / prefer_str_id.
fn merge_id_or_title(
    col_name: &str,
    chunks: &[ChunkInfo],
    final_dir: &Path,
    total_rows: u32,
) -> io::Result<Option<TypedColumn>> {
    let probe = chunks.iter().find(|c| c.rows > 0).map(|c| &c.dir);
    let Some(probe) = probe else {
        return Ok(None);
    };
    let variant = if probe.join(format!("{col_name}.i64")).exists() {
        "int64"
    } else if probe.join(format!("{col_name}.f64")).exists() {
        "float64"
    } else if probe.join(format!("{col_name}.u32")).exists() {
        "uniqueid"
    } else if probe.join(format!("{col_name}.bool")).exists() {
        "bool"
    } else if probe.join(format!("{col_name}.i32")).exists() {
        "date"
    } else if probe.join(format!("{col_name}.off")).exists() {
        "string"
    } else if probe.join(format!("{col_name}.mixed.zst")).exists() {
        "mixed"
    } else {
        // No id/title was pushed for this type — common when the
        // ColumnStore::push_id/push_title path was never exercised.
        return Ok(None);
    };
    let merged = merge_typed_column(variant, col_name, chunks, final_dir, total_rows)?;
    Ok(Some(merged))
}

// ─── File helpers ───────────────────────────────────────────────────

fn open_truncate_for_write(path: &Path) -> io::Result<std::fs::File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
}

/// Concat each `(src, max_bytes)` pair into `dst`. Stops reading each
/// source at `max_bytes` so mmap padding past the logical row count
/// doesn't corrupt the merge.
fn write_concat_fixed<I>(srcs: I, dst: &Path) -> io::Result<()>
where
    I: Iterator<Item = (PathBuf, usize)>,
{
    let mut w = open_truncate_for_write(dst)?;
    for (src, max_bytes) in srcs {
        if !src.exists() {
            continue;
        }
        let mut f = std::fs::File::open(&src)?;
        let mut taken = (&mut f).take(max_bytes as u64);
        io::copy(&mut taken, &mut w)?;
    }
    Ok(())
}

fn truncate_to(path: &Path, logical_bytes: usize) -> io::Result<()> {
    let f = OpenOptions::new().write(true).open(path)?;
    let cur = f.metadata()?.len() as usize;
    if cur > logical_bytes {
        f.set_len(logical_bytes as u64)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::values::Value;
    use crate::graph::schema::{InternedKey, StringInterner, TypeSchema};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn make_schema(interner: &mut StringInterner, props: &[&str]) -> Arc<TypeSchema> {
        let keys: Vec<InternedKey> = props.iter().map(|p| interner.get_or_intern(p)).collect();
        Arc::new(TypeSchema::from_keys(keys))
    }

    /// Build a dest ColumnStore via the chunked builder with chunk_size
    /// forced small, then verify reads match the rows pushed.
    #[test]
    fn chunk_merge_int64_string_mixed_round_trip() {
        let mut interner = StringInterner::new();
        let schema = make_schema(&mut interner, &["age", "city", "tags"]);
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        meta.insert("city".to_string(), "string".to_string());
        meta.insert("tags".to_string(), "mixed".to_string());

        let scratch = TempDir::new().unwrap();
        let chunk_root = scratch.path().join("chunks");
        let final_dir = scratch.path().join("final");

        // Force 2 rows per chunk → 5 chunks for 10 rows. Stresses the
        // merge across multiple chunks, including string-offset rebase.
        let mut builder = ChunkedColumnBuilder::new(
            Arc::clone(&schema),
            meta.clone(),
            chunk_root.clone(),
            2,
            true, // prefer_str_id
        );

        // Strings of varying length to stress offset rebasing.
        let cities = [
            "X",
            "London",
            "Aix-en-Provence",
            "ABC",
            "Z",
            "Mid-Length",
            "Tokyo",
            "S",
            "VeryLongCityNameHereForTesting",
            "Q",
        ];
        let key_age = interner.get_or_intern("age");
        let key_city = interner.get_or_intern("city");
        let key_tags = interner.get_or_intern("tags");

        for i in 0..10u32 {
            let id = Value::String(format!("id-{i:03}"));
            let title = Value::String(format!("Title #{i}"));
            let props = vec![
                (key_age, Value::Int64(i as i64 * 7 + 13)),
                (key_city, Value::String(cities[i as usize].to_string())),
                (
                    key_tags,
                    if i % 2 == 0 {
                        Value::Int64(i as i64)
                    } else {
                        Value::String(format!("tag-{i}"))
                    },
                ),
            ];
            let row = builder.push_row(&id, &title, &props, &interner).unwrap();
            assert_eq!(row, i);
        }

        let store = builder.finalize(&final_dir, &interner).unwrap();
        assert_eq!(store.row_count(), 10);

        // Read back every cell and compare.
        for i in 0..10u32 {
            let id = store.get_id(i).unwrap();
            assert_eq!(id, Value::String(format!("id-{i:03}")));
            let title = store.get_title(i).unwrap();
            assert_eq!(title, Value::String(format!("Title #{i}")));

            let age = store.get(i, key_age).unwrap();
            assert_eq!(age, Value::Int64(i as i64 * 7 + 13));
            let city = store.get(i, key_city).unwrap();
            assert_eq!(city, Value::String(cities[i as usize].to_string()));
            let tags = store.get(i, key_tags).unwrap();
            let expected_tag = if i % 2 == 0 {
                Value::Int64(i as i64)
            } else {
                Value::String(format!("tag-{i}"))
            };
            assert_eq!(tags, expected_tag);
        }
    }

    /// Empty builder → finalize yields a row_count=0 ColumnStore. No
    /// panics on the offsets-file edge case.
    #[test]
    fn empty_builder_finalize() {
        let mut interner = StringInterner::new();
        let schema = make_schema(&mut interner, &["name"]);
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("name".to_string(), "string".to_string());

        let scratch = TempDir::new().unwrap();
        let builder =
            ChunkedColumnBuilder::new(schema, meta, scratch.path().join("chunks"), 100, false);
        assert!(builder.is_empty());
        let store = builder
            .finalize(&scratch.path().join("final"), &interner)
            .unwrap();
        assert_eq!(store.row_count(), 0);
    }

    /// Single chunk (rows < chunk_size) → finalize still works.
    #[test]
    fn single_chunk_finalize() {
        let mut interner = StringInterner::new();
        let schema = make_schema(&mut interner, &["x"]);
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("x".to_string(), "int64".to_string());
        let key_x = interner.get_or_intern("x");

        let scratch = TempDir::new().unwrap();
        let mut builder = ChunkedColumnBuilder::new(
            Arc::clone(&schema),
            meta,
            scratch.path().join("chunks"),
            // chunk_size_rows >> row count → single tail chunk.
            1000,
            false,
        );
        for i in 0..5u32 {
            builder
                .push_row(
                    &Value::Int64(i as i64),
                    &Value::Null,
                    &[(key_x, Value::Int64(i as i64 * 100))],
                    &interner,
                )
                .unwrap();
        }
        let store = builder
            .finalize(&scratch.path().join("final"), &interner)
            .unwrap();
        assert_eq!(store.row_count(), 5);
        for i in 0..5u32 {
            assert_eq!(store.get(i, key_x).unwrap(), Value::Int64(i as i64 * 100));
        }
    }

    /// Variable-length strings interleaved across chunks — stresses the
    /// offset-rebasing logic. If rebasing is off-by-one, reading by row
    /// returns garbage strings.
    #[test]
    fn str_offset_rebase_across_many_chunks() {
        let mut interner = StringInterner::new();
        let schema = make_schema(&mut interner, &["s"]);
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("s".to_string(), "string".to_string());
        let key_s = interner.get_or_intern("s");

        let scratch = TempDir::new().unwrap();
        let mut builder = ChunkedColumnBuilder::new(
            Arc::clone(&schema),
            meta,
            scratch.path().join("chunks"),
            // chunk_size = 3 → with 17 rows, 5 full + 1 partial chunks.
            3,
            false,
        );

        // Lengths chosen to be varied: 1, 50, 500, 1, 50, …
        let strings: Vec<String> = (0..17u32)
            .map(|i| match i % 4 {
                0 => "X".to_string(),
                1 => "M".repeat(50),
                2 => "L".repeat(500),
                _ => format!("row{i}"),
            })
            .collect();

        for (i, s) in strings.iter().enumerate() {
            builder
                .push_row(
                    &Value::String(format!("id{i}")),
                    &Value::Null,
                    &[(key_s, Value::String(s.clone()))],
                    &interner,
                )
                .unwrap();
        }

        let store = builder
            .finalize(&scratch.path().join("final"), &interner)
            .unwrap();
        assert_eq!(store.row_count(), 17);
        for (i, expected) in strings.iter().enumerate() {
            let got = store.get(i as u32, key_s).unwrap();
            assert_eq!(got, Value::String(expected.clone()), "row {i}");
        }
    }
}
