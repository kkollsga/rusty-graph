//! Persistent per-type string property index for `DiskGraph`.
//!
//! On-disk layout (mirrors the connection-type inverted index pattern):
//!
//! ```text
//! property_index_{type}_{property}_meta.bin      // [u64 count, u64 keys_len]
//! property_index_{type}_{property}_keys.bin      // u8 concatenated sorted UTF-8 bytes
//! property_index_{type}_{property}_offsets.bin   // u64 cumulative byte offsets (count + 1 entries)
//! property_index_{type}_{property}_ids.bin       // u32 NodeIndex parallel to keys (count entries)
//! ```
//!
//! `meta.bin` is explicit because `MmapOrVec::mapped(...)` pads the
//! file to a minimum of 64 elements; the file size alone cannot recover
//! the logical count.
//!
//! Keys are sorted lexicographically; duplicates are adjacent and keyed
//! to their own `NodeIndex`. A single binary search yields the left and
//! right bounds of a run, so equality lookup is O(log N + k) where k is
//! the number of matches. The same layout supports prefix lookup by
//! range-scanning between `lower_bound(prefix)` and
//! `lower_bound(next_prefix)`.
//!
//! Restricted to `TypedColumn::Str` columns — numeric equality is a
//! follow-up.

use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec};
use petgraph::graph::NodeIndex;
use std::fs;
use std::path::{Path, PathBuf};

/// Filename prefix for all on-disk property index files.
const FILE_PREFIX: &str = "property_index_";

/// A single `(node_type, property)` string index, backed by three
/// mmap'd files.
pub struct PropertyIndex {
    keys: MmapBytes,
    offsets: MmapOrVec<u64>,
    ids: MmapOrVec<u32>,
    count: usize,
}

/// Sanitise a node-type or property identifier for inclusion in a
/// filename. Strips anything outside `[A-Za-z0-9_-]` to `_` so users
/// with exotic type names don't confuse the path layer.
fn sanitise(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Four file paths for `(node_type, property)` under `data_dir`:
/// returns `(meta, keys, offsets, ids)`.
pub fn file_paths(
    data_dir: &Path,
    node_type: &str,
    property: &str,
) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let stem = format!(
        "{}{}_{}",
        FILE_PREFIX,
        sanitise(node_type),
        sanitise(property)
    );
    (
        data_dir.join(format!("{}_meta.bin", stem)),
        data_dir.join(format!("{}_keys.bin", stem)),
        data_dir.join(format!("{}_offsets.bin", stem)),
        data_dir.join(format!("{}_ids.bin", stem)),
    )
}

/// Scan `data_dir` for all `property_index_*_meta.bin` files and
/// return the `(node_type, property)` pairs they cover. Used at
/// `DiskGraph::open()` time to populate the in-memory index registry.
#[allow(dead_code)]
pub fn scan_data_dir(data_dir: &Path) -> Vec<(String, String)> {
    let Ok(entries) = fs::read_dir(data_dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if let Some(stem) = name.strip_prefix(FILE_PREFIX) {
            if let Some(stem) = stem.strip_suffix("_meta.bin") {
                if let Some(sep) = stem.rfind('_') {
                    let (ty, prop) = stem.split_at(sep);
                    let prop = &prop[1..];
                    if !ty.is_empty() && !prop.is_empty() {
                        out.push((ty.to_string(), prop.to_string()));
                    }
                }
            }
        }
    }
    out
}

/// Write the meta file for a `(node_type, property)` index.
fn write_meta(path: &Path, count: usize, keys_len: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = fs::File::create(path)?;
    f.write_all(&(count as u64).to_le_bytes())?;
    f.write_all(&(keys_len as u64).to_le_bytes())?;
    Ok(())
}

fn read_meta(path: &Path) -> std::io::Result<(usize, usize)> {
    let bytes = fs::read(path)?;
    if bytes.len() < 16 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "meta file too small",
        ));
    }
    let count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let keys_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    Ok((count, keys_len))
}

impl PropertyIndex {
    /// Number of indexed `(key, node_id)` entries.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Borrow the i-th key as a byte slice.
    fn key_at(&self, i: usize) -> &[u8] {
        let start = self.offsets.get(i) as usize;
        let end = self.offsets.get(i + 1) as usize;
        self.keys.slice(start, end)
    }

    /// Return the left-most index `i` such that `key_at(i) >= target`.
    fn lower_bound(&self, target: &[u8]) -> usize {
        let (mut lo, mut hi) = (0usize, self.count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.key_at(mid) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Exact-match lookup: return all `NodeIndex` values whose key equals
    /// `value`. Duplicates are returned in the order they were emitted
    /// by the build pass (ascending `NodeIndex` within each key —
    /// produced by a stable sort).
    pub fn lookup_eq_str(&self, value: &str) -> Vec<NodeIndex> {
        let target = value.as_bytes();
        let start = self.lower_bound(target);
        if start >= self.count || self.key_at(start) != target {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut i = start;
        while i < self.count && self.key_at(i) == target {
            out.push(NodeIndex::new(self.ids.get(i) as usize));
            i += 1;
        }
        out
    }

    /// Prefix lookup: return `NodeIndex` values whose key starts with
    /// `prefix`, capped at `limit` results. Traverses a contiguous run
    /// of the sorted index, so work is bounded by the number of matches.
    pub fn lookup_prefix_str(&self, prefix: &str, limit: usize) -> Vec<NodeIndex> {
        if limit == 0 {
            return Vec::new();
        }
        let target = prefix.as_bytes();
        let start = self.lower_bound(target);
        let mut out = Vec::with_capacity(limit.min(16));
        let mut i = start;
        while i < self.count && out.len() < limit {
            let key = self.key_at(i);
            if !key.starts_with(target) {
                break;
            }
            out.push(NodeIndex::new(self.ids.get(i) as usize));
            i += 1;
        }
        out
    }

    /// Build a fresh index from a collection of `(key, node_index)`
    /// pairs, serialise to `data_dir`, and return an opened handle.
    ///
    /// `entries` may be in any order; this routine sorts them by
    /// `(key, node_index)` for a deterministic layout.
    pub fn build(
        data_dir: &Path,
        node_type: &str,
        property: &str,
        mut entries: Vec<(String, u32)>,
    ) -> std::io::Result<Self> {
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let total_bytes: usize = entries.iter().map(|(k, _)| k.len()).sum();
        let count = entries.len();

        let (meta_path, keys_path, offsets_path, ids_path) =
            file_paths(data_dir, node_type, property);

        let mut keys_buf = MmapBytes::mapped(&keys_path, total_bytes.max(4096))
            .unwrap_or_else(|_| MmapBytes::new());
        let mut offsets_buf = MmapOrVec::<u64>::mapped(&offsets_path, count + 1)
            .unwrap_or_else(|_| MmapOrVec::with_capacity(count + 1));
        let mut ids_buf = MmapOrVec::<u32>::mapped(&ids_path, count.max(1))
            .unwrap_or_else(|_| MmapOrVec::with_capacity(count));

        let mut offset: u64 = 0;
        for (key, id) in &entries {
            offsets_buf.push(offset);
            keys_buf.extend(key.as_bytes());
            ids_buf.push(*id);
            offset += key.len() as u64;
        }
        offsets_buf.push(offset);

        write_meta(&meta_path, count, total_bytes)?;

        Ok(PropertyIndex {
            keys: keys_buf,
            offsets: offsets_buf,
            ids: ids_buf,
            count,
        })
    }

    /// Re-open a previously built index. Returns `Ok(None)` if any
    /// file is missing (the index was never built or was partially
    /// cleaned up).
    pub fn open(data_dir: &Path, node_type: &str, property: &str) -> std::io::Result<Option<Self>> {
        let (meta_path, keys_path, offsets_path, ids_path) =
            file_paths(data_dir, node_type, property);
        if !meta_path.exists()
            || !keys_path.exists()
            || !offsets_path.exists()
            || !ids_path.exists()
        {
            return Ok(None);
        }
        let (count, keys_len) = read_meta(&meta_path)?;
        let keys = MmapBytes::load_mapped(&keys_path, keys_len)?;
        let offsets = MmapOrVec::<u64>::load_mapped(&offsets_path, count + 1)?;
        let ids = MmapOrVec::<u32>::load_mapped(&ids_path, count)?;
        Ok(Some(PropertyIndex {
            keys,
            offsets,
            ids,
            count,
        }))
    }

    /// Delete the on-disk files for this index. Used when a
    /// `drop_index` is routed to disk.
    pub fn remove_files(data_dir: &Path, node_type: &str, property: &str) -> std::io::Result<()> {
        let (meta_path, keys_path, offsets_path, ids_path) =
            file_paths(data_dir, node_type, property);
        for p in [&meta_path, &keys_path, &offsets_path, &ids_path] {
            if p.exists() {
                fs::remove_file(p)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "kglite_prop_idx_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn equality_lookup_finds_single_match() {
        let dir = tmp_dir();
        let idx = PropertyIndex::build(
            &dir,
            "Human",
            "label",
            vec![
                ("Alice".into(), 1),
                ("Bob".into(), 2),
                ("Charlie".into(), 3),
            ],
        )
        .unwrap();
        assert_eq!(idx.lookup_eq_str("Bob"), vec![NodeIndex::new(2)]);
        assert_eq!(idx.lookup_eq_str("Alice"), vec![NodeIndex::new(1)]);
        assert!(idx.lookup_eq_str("Missing").is_empty());
    }

    #[test]
    fn equality_lookup_returns_all_duplicates() {
        let dir = tmp_dir();
        let idx = PropertyIndex::build(
            &dir,
            "Human",
            "label",
            vec![
                ("Alice".into(), 1),
                ("Alice".into(), 7),
                ("Alice".into(), 4),
                ("Bob".into(), 2),
            ],
        )
        .unwrap();
        let hits = idx.lookup_eq_str("Alice");
        assert_eq!(
            hits,
            vec![NodeIndex::new(1), NodeIndex::new(4), NodeIndex::new(7)]
        );
    }

    #[test]
    fn prefix_lookup_respects_limit_and_sort_order() {
        let dir = tmp_dir();
        let idx = PropertyIndex::build(
            &dir,
            "Human",
            "label",
            vec![
                ("Oslo".into(), 10),
                ("Ottawa".into(), 11),
                ("Oxford".into(), 12),
                ("Paris".into(), 13),
            ],
        )
        .unwrap();
        let hits = idx.lookup_prefix_str("O", 10);
        assert_eq!(
            hits,
            vec![NodeIndex::new(10), NodeIndex::new(11), NodeIndex::new(12)]
        );
        // Limit 2 returns the first two only.
        let hits = idx.lookup_prefix_str("O", 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0], NodeIndex::new(10));
    }

    #[test]
    fn persistence_roundtrip_via_open() {
        let dir = tmp_dir();
        {
            let _ = PropertyIndex::build(
                &dir,
                "Human",
                "label",
                vec![("Alice".into(), 1), ("Bob".into(), 2)],
            )
            .unwrap();
        } // drop: flush to disk
        let reopened = PropertyIndex::open(&dir, "Human", "label")
            .unwrap()
            .expect("index files should be present");
        assert_eq!(reopened.lookup_eq_str("Bob"), vec![NodeIndex::new(2)]);
        assert_eq!(reopened.len(), 2);
    }

    #[test]
    fn scan_data_dir_discovers_built_indexes() {
        let dir = tmp_dir();
        let _ = PropertyIndex::build(&dir, "Human", "label", vec![("A".into(), 1)]).unwrap();
        let _ = PropertyIndex::build(&dir, "Paper", "title", vec![("Z".into(), 2)]).unwrap();
        let mut pairs = scan_data_dir(&dir);
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("Human".to_string(), "label".to_string()),
                ("Paper".to_string(), "title".to_string()),
            ]
        );
    }

    #[test]
    fn remove_files_cleans_up() {
        let dir = tmp_dir();
        let _ = PropertyIndex::build(&dir, "Human", "label", vec![("A".into(), 1)]).unwrap();
        PropertyIndex::remove_files(&dir, "Human", "label").unwrap();
        assert!(PropertyIndex::open(&dir, "Human", "label")
            .unwrap()
            .is_none());
    }

    #[test]
    fn empty_index_lookup_returns_empty() {
        let dir = tmp_dir();
        let idx = PropertyIndex::build(&dir, "Human", "label", Vec::new()).unwrap();
        assert!(idx.lookup_eq_str("anything").is_empty());
        assert!(idx.lookup_prefix_str("x", 10).is_empty());
    }
}
