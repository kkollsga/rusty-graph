// src/graph/mmap_vec.rs
//
// MmapOrVec<T>: a contiguous buffer of Copy+Pod values backed by either
// a heap Vec<T> or a memory-mapped file. Provides transparent read/write
// access regardless of backing. When mmap-backed, grow operations
// require dropping and recreating the mapping (memmap2 limitation).

use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

// ─── MmapOrVec ──────────────────────────────────────────────────────────────

/// A resizable buffer of `T: Copy + Default` values, optionally file-backed.
///
/// - `Heap` variant: plain `Vec<T>` — default, no file I/O.
/// - `Mapped` variant: memory-mapped file — data lives on disk, OS pages in/out.
///
/// Switching from Heap to Mapped writes current data to a file and mmaps it.
/// Growing a Mapped buffer requires unmap → ftruncate → remap (invalidates
/// old pointers, which is safe because `PropertyStorage::Columnar` returns
/// owned `Cow::Owned` values, never references into the mapping).
#[derive(Debug)]
pub enum MmapOrVec<T: Copy + Default + 'static> {
    Heap {
        data: Vec<T>,
    },
    Mapped {
        mmap: MmapMut,
        len: usize,
        capacity: usize, // in elements, not bytes
        file: File,
        path: PathBuf,
        _phantom: std::marker::PhantomData<T>,
    },
}

#[allow(dead_code)]
impl<T: Copy + Default + 'static> MmapOrVec<T> {
    /// Create a new heap-backed buffer.
    pub fn new() -> Self {
        MmapOrVec::Heap { data: Vec::new() }
    }

    /// Create a new heap-backed buffer with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        MmapOrVec::Heap {
            data: Vec::with_capacity(cap),
        }
    }

    /// Create a heap-backed buffer from an existing Vec.
    pub fn from_vec(data: Vec<T>) -> Self {
        MmapOrVec::Heap { data }
    }

    /// Create a file-backed buffer pre-sized to `count` elements.
    /// The file is created at full size but NO data is written — the OS zero-fills
    /// mmap pages lazily. Use `set(index, value)` to write individual positions.
    /// This avoids the O(N) push loop needed to pre-fill with defaults.
    pub fn mapped_zeroed(path: &Path, count: usize) -> io::Result<Self> {
        let cap = count.max(64);
        let byte_len = cap * std::mem::size_of::<T>();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(byte_len as u64)?;
        let mmap = unsafe { MmapOptions::new().len(byte_len).map_mut(&file)? };
        Ok(MmapOrVec::Mapped {
            mmap,
            len: cap, // all positions addressable via set()
            capacity: cap,
            file,
            path: path.to_path_buf(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create a file-backed buffer at the given path.
    /// The file is created/truncated with initial capacity for `initial_cap` elements.
    /// `len` starts at 0 — use `push()` to add elements.
    pub fn mapped(path: &Path, initial_cap: usize) -> io::Result<Self> {
        let cap = initial_cap.max(64); // minimum 64 elements
        let byte_len = cap * std::mem::size_of::<T>();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(byte_len as u64)?;

        let mmap = unsafe { MmapOptions::new().len(byte_len).map_mut(&file)? };

        Ok(MmapOrVec::Mapped {
            mmap,
            len: 0,
            capacity: cap,
            file,
            path: path.to_path_buf(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create a file-backed buffer pre-sized to `count` elements.
    /// Elements are zero-initialized by the OS (lazy page-fault zero-fill).
    /// Allows immediate `set(index, value)` without prior `push()`.
    /// No pre-fill I/O — pages are only allocated when first written.
    pub fn mapped_prefilled(path: &Path, count: usize) -> io::Result<Self> {
        let cap = count.max(64);
        let byte_len = cap * std::mem::size_of::<T>();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(byte_len as u64)?;

        let mmap = unsafe { MmapOptions::new().len(byte_len).map_mut(&file)? };

        Ok(MmapOrVec::Mapped {
            mmap,
            len: count, // pre-sized — set() works immediately
            capacity: cap,
            file,
            path: path.to_path_buf(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Load an existing file-backed buffer (e.g. from save_mmap).
    /// `len` is the number of valid elements in the file.
    pub fn load_mapped(path: &Path, len: usize) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let elem_size = std::mem::size_of::<T>();
        let capacity = if elem_size > 0 {
            file_len / elem_size
        } else {
            len
        };

        if capacity < len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "File too small: {} bytes for {} elements of size {}",
                    file_len, len, elem_size
                ),
            ));
        }

        let mmap = unsafe { MmapOptions::new().len(file_len).map_mut(&file)? };

        Ok(MmapOrVec::Mapped {
            mmap,
            len,
            capacity,
            file,
            path: path.to_path_buf(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Load a region of an existing file as a mapped buffer.
    /// `offset` is the byte offset into the file, `len` is the number of elements.
    /// The mmap covers only the region [offset..offset + len * sizeof(T)].
    pub fn load_mapped_region(path: &Path, offset: usize, len: usize) -> io::Result<Self> {
        if len == 0 {
            return Ok(MmapOrVec::new());
        }
        let elem_size = std::mem::size_of::<T>();
        let byte_len = len * elem_size;
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mmap = unsafe {
            MmapOptions::new()
                .offset(offset as u64)
                .len(byte_len)
                .map_mut(&file)?
        };
        Ok(MmapOrVec::Mapped {
            mmap,
            len,
            capacity: len,
            file,
            path: path.to_path_buf(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            MmapOrVec::Heap { data } => data.len(),
            MmapOrVec::Mapped { len, .. } => *len,
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read element at index. Panics if out of bounds.
    pub fn get(&self, index: usize) -> T {
        match self {
            MmapOrVec::Heap { data } => data[index],
            MmapOrVec::Mapped { mmap, len, .. } => {
                assert!(index < *len, "MmapOrVec index out of bounds");
                let offset = index * std::mem::size_of::<T>();
                // SAFETY: mmap regions are page-aligned, elements stored contiguously
                // from the start, so all accesses are naturally aligned.
                unsafe { std::ptr::read(mmap.as_ptr().add(offset) as *const T) }
            }
        }
    }

    /// Set element at index. Panics if out of bounds.
    pub fn set(&mut self, index: usize, value: T) {
        match self {
            MmapOrVec::Heap { data } => data[index] = value,
            MmapOrVec::Mapped { mmap, len, .. } => {
                assert!(index < *len, "MmapOrVec index out of bounds");
                let offset = index * std::mem::size_of::<T>();
                // SAFETY: mmap regions are page-aligned, elements stored contiguously.
                unsafe {
                    std::ptr::write(mmap.as_mut_ptr().add(offset) as *mut T, value);
                }
            }
        }
    }

    /// Append an element. Grows the buffer if needed.
    pub fn push(&mut self, value: T) {
        match self {
            MmapOrVec::Heap { data } => data.push(value),
            MmapOrVec::Mapped {
                mmap,
                len,
                capacity,
                file,
                path,
                ..
            } => {
                if *len >= *capacity {
                    // Grow: double capacity
                    let new_cap = (*capacity * 2).max(64);
                    let new_byte_len = new_cap * std::mem::size_of::<T>();
                    file.set_len(new_byte_len as u64).expect("ftruncate failed");
                    *mmap = unsafe {
                        MmapOptions::new()
                            .len(new_byte_len)
                            .map_mut(&*file)
                            .unwrap_or_else(|e| {
                                panic!("mmap remap failed for {}: {}", path.display(), e)
                            })
                    };
                    *capacity = new_cap;
                }
                let offset = *len * std::mem::size_of::<T>();
                unsafe {
                    std::ptr::write(mmap.as_mut_ptr().add(offset) as *mut T, value);
                }
                *len += 1;
            }
        }
    }

    /// Get a slice of the data (heap only). Returns None for mapped.
    pub fn as_slice(&self) -> Option<&[T]> {
        match self {
            MmapOrVec::Heap { data } => Some(data.as_slice()),
            MmapOrVec::Mapped { .. } => None,
        }
    }

    /// Get a mutable byte-level view of the backing storage.
    /// Works for both Heap and Mapped variants.
    /// Useful for bulk byte operations (memset, copy_from_slice) during build.
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        match self {
            MmapOrVec::Heap { data } => unsafe {
                std::slice::from_raw_parts_mut(
                    data.as_mut_ptr() as *mut u8,
                    data.len() * std::mem::size_of::<T>(),
                )
            },
            MmapOrVec::Mapped { mmap, len, .. } => &mut mmap[..*len * std::mem::size_of::<T>()],
        }
    }

    /// Convert from Heap to Mapped (file-backed). No-op if already mapped.
    pub fn materialize_to_file(&mut self, path: &Path) -> io::Result<()> {
        if matches!(self, MmapOrVec::Mapped { .. }) {
            return Ok(()); // already mapped
        }
        let MmapOrVec::Heap { data } = self else {
            unreachable!()
        };

        let len = data.len();
        let cap = len.max(64);
        let byte_len = cap * std::mem::size_of::<T>();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(byte_len as u64)?;

        let mut mmap = unsafe { MmapOptions::new().len(byte_len).map_mut(&file)? };

        // Copy data into mmap
        let src_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, len * std::mem::size_of::<T>())
        };
        mmap[..src_bytes.len()].copy_from_slice(src_bytes);
        mmap.flush_async()?;

        *self = MmapOrVec::Mapped {
            mmap,
            len,
            capacity: cap,
            file,
            path: path.to_path_buf(),
            _phantom: std::marker::PhantomData,
        };

        Ok(())
    }

    /// Convert from Mapped back to Heap. No-op if already heap.
    pub fn materialize_to_heap(&mut self) {
        if matches!(self, MmapOrVec::Heap { .. }) {
            return;
        }
        let data = match self {
            MmapOrVec::Mapped { mmap, len, .. } => {
                unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const T, *len) }.to_vec()
            }
            _ => unreachable!(),
        };
        *self = MmapOrVec::Heap { data };
    }

    /// Whether this buffer is file-backed.
    pub fn is_mapped(&self) -> bool {
        matches!(self, MmapOrVec::Mapped { .. })
    }

    /// Heap-resident bytes (0 if file-backed).
    pub fn heap_bytes(&self) -> usize {
        match self {
            MmapOrVec::Heap { data } => data.len() * std::mem::size_of::<T>(),
            MmapOrVec::Mapped { .. } => 0,
        }
    }

    /// Flush mmap to disk (no-op for heap).
    pub fn flush(&self) -> io::Result<()> {
        match self {
            MmapOrVec::Heap { .. } => Ok(()),
            MmapOrVec::Mapped { mmap, .. } => mmap.flush(),
        }
    }

    /// Return the raw bytes of the data (without copying for heap).
    pub fn as_raw_bytes(&self) -> &[u8] {
        match self {
            MmapOrVec::Heap { data } => unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<T>(),
                )
            },
            MmapOrVec::Mapped { mmap, len, .. } => &mmap[..*len * std::mem::size_of::<T>()],
        }
    }

    /// Write raw bytes to a writer (for v3 packed column format).
    pub fn write_to(&self, writer: &mut impl Write) -> io::Result<()> {
        writer.write_all(self.as_raw_bytes())
    }

    /// Write the data to a file (for save_mmap). For heap, writes Vec contents.
    /// For mapped, flushes then copies the file.
    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        match self {
            MmapOrVec::Heap { data } => {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<T>(),
                    )
                };
                std::fs::write(path, bytes)
            }
            MmapOrVec::Mapped {
                mmap, len, file, ..
            } => {
                // Flush first
                mmap.flush()?;
                let byte_len = *len * std::mem::size_of::<T>();
                // If it's the same file, just flush; otherwise copy
                let src_path = self.file_path();
                if let Some(sp) = src_path {
                    if sp == path {
                        // Just truncate to exact size
                        file.set_len(byte_len as u64)?;
                        return Ok(());
                    }
                }
                std::fs::write(path, &mmap[..byte_len])
            }
        }
    }

    /// The file path for a mapped buffer.
    pub fn file_path(&self) -> Option<&Path> {
        match self {
            MmapOrVec::Heap { .. } => None,
            MmapOrVec::Mapped { path, .. } => Some(path.as_path()),
        }
    }

    /// Iterate over elements. Returns a Vec for simplicity (avoids lifetime issues
    /// with mmap slices).
    pub fn to_vec(&self) -> Vec<T> {
        match self {
            MmapOrVec::Heap { data } => data.clone(),
            MmapOrVec::Mapped { mmap, len, .. } => {
                unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const T, *len) }.to_vec()
            }
        }
    }
}

impl<T: Copy + Default + 'static> Clone for MmapOrVec<T> {
    /// Clone always produces a Heap variant (cloning a mapped file doesn't make sense).
    fn clone(&self) -> Self {
        MmapOrVec::Heap {
            data: self.to_vec(),
        }
    }
}

impl<T: Copy + Default + 'static> Default for MmapOrVec<T> {
    fn default() -> Self {
        MmapOrVec::new()
    }
}

// ─── MmapBytes ──────────────────────────────────────────────────────────────

/// Variable-length byte buffer (for strings) with mmap support.
/// Similar to MmapOrVec but for raw bytes with append-only semantics.
#[derive(Debug)]
pub enum MmapBytes {
    Heap {
        data: Vec<u8>,
    },
    Mapped {
        mmap: MmapMut,
        len: usize,
        capacity: usize,
        file: File,
        path: PathBuf,
    },
}

#[allow(dead_code)]
impl MmapBytes {
    pub fn new() -> Self {
        MmapBytes::Heap { data: Vec::new() }
    }

    /// Create a heap-backed byte buffer from an existing Vec.
    pub fn from_vec(data: Vec<u8>) -> Self {
        MmapBytes::Heap { data }
    }

    pub fn mapped(path: &Path, initial_cap: usize) -> io::Result<Self> {
        let cap = initial_cap.max(4096); // minimum 4KB
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(cap as u64)?;
        let mmap = unsafe { MmapOptions::new().len(cap).map_mut(&file)? };
        Ok(MmapBytes::Mapped {
            mmap,
            len: 0,
            capacity: cap,
            file,
            path: path.to_path_buf(),
        })
    }

    pub fn load_mapped(path: &Path, len: usize) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let capacity = file.metadata()?.len() as usize;
        if capacity < len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File too small for byte buffer",
            ));
        }
        let mmap = unsafe { MmapOptions::new().len(capacity).map_mut(&file)? };
        Ok(MmapBytes::Mapped {
            mmap,
            len,
            capacity,
            file,
            path: path.to_path_buf(),
        })
    }

    pub fn len(&self) -> usize {
        match self {
            MmapBytes::Heap { data } => data.len(),
            MmapBytes::Mapped { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append bytes, return the start offset.
    pub fn extend(&mut self, bytes: &[u8]) -> usize {
        let start = self.len();
        match self {
            MmapBytes::Heap { data } => data.extend_from_slice(bytes),
            MmapBytes::Mapped {
                mmap,
                len,
                capacity,
                file,
                path,
            } => {
                let needed = *len + bytes.len();
                if needed > *capacity {
                    let new_cap = (needed * 2).max(*capacity * 2);
                    file.set_len(new_cap as u64).expect("ftruncate failed");
                    *mmap = unsafe {
                        MmapOptions::new()
                            .len(new_cap)
                            .map_mut(&*file)
                            .unwrap_or_else(|e| {
                                panic!("mmap remap failed for {}: {}", path.display(), e)
                            })
                    };
                    *capacity = new_cap;
                }
                mmap[*len..*len + bytes.len()].copy_from_slice(bytes);
                *len += bytes.len();
            }
        }
        start
    }

    /// Read a byte range.
    pub fn slice(&self, start: usize, end: usize) -> &[u8] {
        match self {
            MmapBytes::Heap { data } => &data[start..end],
            MmapBytes::Mapped { mmap, .. } => &mmap[start..end],
        }
    }

    pub fn materialize_to_file(&mut self, path: &Path) -> io::Result<()> {
        if matches!(self, MmapBytes::Mapped { .. }) {
            return Ok(());
        }
        let MmapBytes::Heap { data } = self else {
            unreachable!()
        };
        let len = data.len();
        let cap = len.max(4096);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(cap as u64)?;
        let mut mmap = unsafe { MmapOptions::new().len(cap).map_mut(&file)? };
        mmap[..len].copy_from_slice(data);
        mmap.flush_async()?;
        *self = MmapBytes::Mapped {
            mmap,
            len,
            capacity: cap,
            file,
            path: path.to_path_buf(),
        };
        Ok(())
    }

    pub fn materialize_to_heap(&mut self) {
        if matches!(self, MmapBytes::Heap { .. }) {
            return;
        }
        let len = self.len();
        let data = match self {
            MmapBytes::Mapped { mmap, .. } => mmap[..len].to_vec(),
            _ => unreachable!(),
        };
        *self = MmapBytes::Heap { data };
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self, MmapBytes::Mapped { .. })
    }

    /// Heap-resident bytes (0 if file-backed).
    pub fn heap_bytes(&self) -> usize {
        match self {
            MmapBytes::Heap { data } => data.len(),
            MmapBytes::Mapped { .. } => 0,
        }
    }

    pub fn flush(&self) -> io::Result<()> {
        match self {
            MmapBytes::Heap { .. } => Ok(()),
            MmapBytes::Mapped { mmap, .. } => mmap.flush(),
        }
    }

    /// Return the raw bytes.
    pub fn as_raw_bytes(&self) -> &[u8] {
        match self {
            MmapBytes::Heap { data } => data,
            MmapBytes::Mapped { mmap, len, .. } => &mmap[..*len],
        }
    }

    /// Write raw bytes to a writer (for v3 packed column format).
    pub fn write_to(&self, writer: &mut impl Write) -> io::Result<()> {
        writer.write_all(self.as_raw_bytes())
    }

    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        match self {
            MmapBytes::Heap { data } => std::fs::write(path, data),
            MmapBytes::Mapped { mmap, len, .. } => std::fs::write(path, &mmap[..*len]),
        }
    }

    pub fn to_vec(&self) -> Vec<u8> {
        match self {
            MmapBytes::Heap { data } => data.clone(),
            MmapBytes::Mapped { mmap, len, .. } => mmap[..*len].to_vec(),
        }
    }
}

impl Clone for MmapBytes {
    fn clone(&self) -> Self {
        MmapBytes::Heap {
            data: self.to_vec(),
        }
    }
}

impl Default for MmapBytes {
    fn default() -> Self {
        MmapBytes::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn tmp_path(dir: &TempDir, name: &str) -> PathBuf {
        dir.path().join(name)
    }

    #[test]
    fn test_heap_basic() {
        let mut buf: MmapOrVec<i64> = MmapOrVec::new();
        buf.push(10);
        buf.push(20);
        buf.push(30);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), 10);
        assert_eq!(buf.get(1), 20);
        assert_eq!(buf.get(2), 30);
        buf.set(1, 99);
        assert_eq!(buf.get(1), 99);
        assert!(!buf.is_mapped());
    }

    #[test]
    fn test_mapped_basic() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "col.bin");
        let mut buf: MmapOrVec<i64> = MmapOrVec::mapped(&path, 4).unwrap();
        assert!(buf.is_mapped());
        assert_eq!(buf.len(), 0);

        buf.push(100);
        buf.push(200);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.get(0), 100);
        assert_eq!(buf.get(1), 200);

        buf.set(0, 999);
        assert_eq!(buf.get(0), 999);
    }

    #[test]
    fn test_mapped_grow() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "grow.bin");
        let mut buf: MmapOrVec<u32> = MmapOrVec::mapped(&path, 2).unwrap();

        // Push beyond initial capacity (min 64 due to .max(64))
        for i in 0..200 {
            buf.push(i);
        }
        assert_eq!(buf.len(), 200);
        for i in 0..200u32 {
            assert_eq!(buf.get(i as usize), i);
        }
    }

    #[test]
    fn test_heap_to_mapped() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "convert.bin");

        let mut buf: MmapOrVec<f64> = MmapOrVec::new();
        buf.push(1.5);
        buf.push(2.7);
        buf.push(3.9);
        assert!(!buf.is_mapped());

        buf.materialize_to_file(&path).unwrap();
        assert!(buf.is_mapped());
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), 1.5);
        assert_eq!(buf.get(1), 2.7);
        assert_eq!(buf.get(2), 3.9);
    }

    #[test]
    fn test_mapped_to_heap() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "to_heap.bin");

        let mut buf: MmapOrVec<i32> = MmapOrVec::mapped(&path, 4).unwrap();
        buf.push(10);
        buf.push(20);
        buf.materialize_to_heap();

        assert!(!buf.is_mapped());
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.get(0), 10);
        assert_eq!(buf.get(1), 20);
    }

    #[test]
    fn test_clone_always_heap() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "clone.bin");

        let mut buf: MmapOrVec<i64> = MmapOrVec::mapped(&path, 4).unwrap();
        buf.push(42);
        let cloned = buf.clone();
        assert!(!cloned.is_mapped());
        assert_eq!(cloned.get(0), 42);
    }

    #[test]
    fn test_save_load_mapped() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "save.bin");

        let mut buf: MmapOrVec<i64> = MmapOrVec::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.save_to_file(&path).unwrap();

        let loaded: MmapOrVec<i64> = MmapOrVec::load_mapped(&path, 3).unwrap();
        assert!(loaded.is_mapped());
        assert_eq!(loaded.get(0), 1);
        assert_eq!(loaded.get(1), 2);
        assert_eq!(loaded.get(2), 3);
    }

    // ─── MmapBytes tests ────────────────────────────────────────────────

    #[test]
    fn test_bytes_heap_basic() {
        let mut buf = MmapBytes::new();
        let off0 = buf.extend(b"hello");
        let off1 = buf.extend(b"world");
        assert_eq!(off0, 0);
        assert_eq!(off1, 5);
        assert_eq!(buf.slice(0, 5), b"hello");
        assert_eq!(buf.slice(5, 10), b"world");
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn test_bytes_mapped() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "bytes.bin");

        let mut buf = MmapBytes::mapped(&path, 16).unwrap();
        assert!(buf.is_mapped());

        let off0 = buf.extend(b"hello");
        let off1 = buf.extend(b"world");
        assert_eq!(off0, 0);
        assert_eq!(off1, 5);
        assert_eq!(buf.slice(0, 5), b"hello");
        assert_eq!(buf.slice(5, 10), b"world");
    }

    #[test]
    fn test_bytes_mapped_grow() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "bytes_grow.bin");

        let mut buf = MmapBytes::mapped(&path, 16).unwrap();
        // Exceed initial capacity (min 4096)
        let big = vec![b'x'; 5000];
        buf.extend(&big);
        assert_eq!(buf.len(), 5000);
        assert_eq!(buf.slice(0, 3), b"xxx");
    }

    #[test]
    fn test_bytes_save_load() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "bytes_save.bin");

        let mut buf = MmapBytes::new();
        buf.extend(b"test data here");
        buf.save_to_file(&path).unwrap();

        let loaded = MmapBytes::load_mapped(&path, 14).unwrap();
        assert_eq!(loaded.slice(0, 14), b"test data here");
    }

    #[test]
    fn test_bytes_clone_always_heap() {
        let dir = TempDir::new().unwrap();
        let path = tmp_path(&dir, "bytes_clone.bin");

        let mut buf = MmapBytes::mapped(&path, 16).unwrap();
        buf.extend(b"data");
        let cloned = buf.clone();
        assert!(!cloned.is_mapped());
        assert_eq!(cloned.slice(0, 4), b"data");
    }
}
