// src/graph/block_pool.rs
//
// Memory-managed block allocator with LRU eviction and zstd compression.
// Used during DiskGraph build phase to manage column data within a memory budget.
//
// Design inspired by DuckDB's BufferPool but simplified for single-threaded
// offline graph building (no Pin/Unpin, no concurrent access).

use std::fs;
use std::io;
use std::path::PathBuf;

// ─── BlockId ────────────────────────────────────────────────────────────────

/// Opaque handle to a block in the pool.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct BlockId(u32);

// ─── BlockState ─────────────────────────────────────────────────────────────

enum BlockState {
    /// Block data is in memory.
    Resident(Vec<u8>),
    /// Block has been zstd-compressed and written to a temp file.
    Evicted {
        path: PathBuf,
        #[allow(dead_code)]
        compressed_size: u64,
    },
    /// Slot is available for reuse.
    Free,
}

// ─── BlockEntry ─────────────────────────────────────────────────────────────

struct BlockEntry {
    state: BlockState,
    last_access: u64,
    uncompressed_size: usize,
    /// If true, this block cannot be evicted (currently being written to).
    pinned: bool,
}

// ─── BlockPool ──────────────────────────────────────────────────────────────

/// A block allocator that enforces a global memory budget.
///
/// When resident memory exceeds the budget, the least-recently-used unpinned
/// blocks are evicted: zstd-compressed and written to temp files. Accessing an
/// evicted block transparently reloads it (and may trigger further eviction).
pub struct BlockPool {
    blocks: Vec<BlockEntry>,
    memory_used: usize,
    memory_budget: usize,
    temp_dir: PathBuf,
    clock: u64,
    free_list: Vec<u32>,
    /// Blocks explicitly marked cold — evicted first in O(1) before falling
    /// back to an O(n) LRU scan. Critical for large builds (10M+ blocks)
    /// where the linear scan becomes the bottleneck.
    cold_queue: Vec<u32>,
    next_file_id: u64,
    eviction_count: u64,
    reload_count: u64,
}

#[allow(dead_code)]
impl BlockPool {
    /// Create a new block pool.
    ///
    /// `memory_budget`: maximum bytes of resident (uncompressed) block data.
    /// `temp_dir`: directory for evicted block files (created if needed).
    pub fn new(memory_budget: usize, temp_dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&temp_dir)?;
        Ok(Self {
            blocks: Vec::new(),
            memory_used: 0,
            memory_budget,
            temp_dir,
            clock: 0,
            free_list: Vec::new(),
            cold_queue: Vec::new(),
            next_file_id: 0,
            eviction_count: 0,
            reload_count: 0,
        })
    }

    /// Current resident memory usage in bytes.
    pub fn memory_used(&self) -> usize {
        self.memory_used
    }

    /// Memory budget in bytes.
    pub fn memory_budget(&self) -> usize {
        self.memory_budget
    }

    /// Number of allocated (non-free) blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len() - self.free_list.len()
    }

    /// Number of times a block was evicted to disk.
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    /// Number of times an evicted block was reloaded from disk.
    pub fn reload_count(&self) -> u64 {
        self.reload_count
    }

    /// Allocate a new block of `size` bytes (zeroed). May trigger eviction.
    pub fn allocate(&mut self, size: usize) -> io::Result<BlockId> {
        let ts = self.tick();
        let id = if let Some(slot) = self.free_list.pop() {
            let entry = &mut self.blocks[slot as usize];
            entry.state = BlockState::Resident(vec![0u8; size]);
            entry.last_access = ts;
            entry.uncompressed_size = size;
            entry.pinned = false;
            BlockId(slot)
        } else {
            let idx = self.blocks.len() as u32;
            self.blocks.push(BlockEntry {
                state: BlockState::Resident(vec![0u8; size]),
                last_access: ts,
                uncompressed_size: size,
                pinned: false,
            });
            BlockId(idx)
        };
        self.memory_used += size;
        self.evict_until_budget()?;
        Ok(id)
    }

    /// Allocate a block pre-filled with `data`. May trigger eviction.
    pub fn allocate_with(&mut self, data: Vec<u8>) -> io::Result<BlockId> {
        let size = data.len();
        let ts = self.tick();
        let id = if let Some(slot) = self.free_list.pop() {
            let entry = &mut self.blocks[slot as usize];
            entry.state = BlockState::Resident(data);
            entry.last_access = ts;
            entry.uncompressed_size = size;
            entry.pinned = false;
            BlockId(slot)
        } else {
            let idx = self.blocks.len() as u32;
            self.blocks.push(BlockEntry {
                state: BlockState::Resident(data),
                last_access: ts,
                uncompressed_size: size,
                pinned: false,
            });
            BlockId(idx)
        };
        self.memory_used += size;
        self.evict_until_budget()?;
        Ok(id)
    }

    /// Get mutable access to a block's data. Reloads from disk if evicted.
    pub fn get_mut(&mut self, id: BlockId) -> io::Result<&mut [u8]> {
        self.ensure_resident(id)?;
        let ts = self.tick();
        let entry = &mut self.blocks[id.0 as usize];
        entry.last_access = ts;
        match &mut entry.state {
            BlockState::Resident(data) => Ok(data.as_mut_slice()),
            _ => unreachable!("ensure_resident guarantees Resident state"),
        }
    }

    /// Get read-only access to a block's data. Reloads from disk if evicted.
    pub fn get(&mut self, id: BlockId) -> io::Result<&[u8]> {
        self.ensure_resident(id)?;
        let ts = self.tick();
        let entry = &mut self.blocks[id.0 as usize];
        entry.last_access = ts;
        match &entry.state {
            BlockState::Resident(data) => Ok(data.as_slice()),
            _ => unreachable!("ensure_resident guarantees Resident state"),
        }
    }

    /// Pin a block so it cannot be evicted. Reloads from disk if evicted.
    pub fn pin(&mut self, id: BlockId) -> io::Result<()> {
        self.ensure_resident(id)?;
        self.blocks[id.0 as usize].pinned = true;
        Ok(())
    }

    /// Unpin a block, making it eligible for eviction again.
    pub fn unpin(&mut self, id: BlockId) {
        self.blocks[id.0 as usize].pinned = false;
    }

    /// Hint that this block won't be accessed soon (lower LRU priority).
    pub fn mark_cold(&mut self, id: BlockId) {
        let entry = &mut self.blocks[id.0 as usize];
        entry.last_access = 0;
        // Only queue if still resident and not pinned
        if !entry.pinned {
            if let BlockState::Resident(_) = &entry.state {
                self.cold_queue.push(id.0);
            }
        }
    }

    /// Release a block entirely, freeing its memory and any temp file.
    pub fn free(&mut self, id: BlockId) -> io::Result<()> {
        let entry = &mut self.blocks[id.0 as usize];
        match &entry.state {
            BlockState::Resident(_) => {
                self.memory_used = self.memory_used.saturating_sub(entry.uncompressed_size);
            }
            BlockState::Evicted { path, .. } => {
                let _ = fs::remove_file(path);
            }
            BlockState::Free => return Ok(()),
        }
        entry.state = BlockState::Free;
        entry.pinned = false;
        self.free_list.push(id.0);
        Ok(())
    }

    /// Returns true if the block is currently resident in memory.
    pub fn is_resident(&self, id: BlockId) -> bool {
        matches!(self.blocks[id.0 as usize].state, BlockState::Resident(_))
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Ensure block is in Resident state. If evicted, reload and decompress.
    fn ensure_resident(&mut self, id: BlockId) -> io::Result<()> {
        let entry = &self.blocks[id.0 as usize];
        match &entry.state {
            BlockState::Resident(_) => Ok(()),
            BlockState::Evicted { path, .. } => {
                let compressed = fs::read(path)?;
                let data = zstd::decode_all(compressed.as_slice()).map_err(io::Error::other)?;
                let _ = fs::remove_file(path);
                self.reload_count += 1;

                let size = data.len();
                // Pin temporarily to prevent immediate re-eviction
                let ts = self.tick();
                let entry = &mut self.blocks[id.0 as usize];
                entry.state = BlockState::Resident(data);
                entry.uncompressed_size = size;
                entry.last_access = ts;
                entry.pinned = true;
                self.memory_used += size;

                // Reloading may push us over budget — evict others (not this one)
                self.evict_until_budget()?;
                self.blocks[id.0 as usize].pinned = false;
                Ok(())
            }
            BlockState::Free => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "access to freed block",
            )),
        }
    }

    /// Evict blocks until memory is within budget.
    /// Drains the cold_queue first (O(1) per eviction), then falls back to
    /// an O(n) LRU scan only if no cold blocks remain.
    fn evict_until_budget(&mut self) -> io::Result<()> {
        while self.memory_used > self.memory_budget {
            // Fast path: evict from cold_queue
            if let Some(idx) = self.pop_cold_candidate() {
                self.evict_block(idx)?;
                continue;
            }
            // Slow path: linear scan for LRU candidate
            match self.find_eviction_candidate() {
                Some(idx) => self.evict_block(idx)?,
                None => break,
            }
        }
        Ok(())
    }

    /// Pop a valid eviction candidate from the cold_queue.
    /// Skips entries that are no longer resident, are pinned, or were re-accessed.
    fn pop_cold_candidate(&mut self) -> Option<u32> {
        while let Some(idx) = self.cold_queue.pop() {
            let entry = &self.blocks[idx as usize];
            if entry.pinned {
                continue;
            }
            if let BlockState::Resident(_) = &entry.state {
                return Some(idx);
            }
            // Already evicted or freed — skip
        }
        None
    }

    /// Find the unpinned, resident block with the lowest last_access time.
    fn find_eviction_candidate(&self) -> Option<u32> {
        let mut best: Option<(u32, u64)> = None;
        for (i, entry) in self.blocks.iter().enumerate() {
            if entry.pinned {
                continue;
            }
            if let BlockState::Resident(_) = &entry.state {
                match best {
                    None => best = Some((i as u32, entry.last_access)),
                    Some((_, best_time)) if entry.last_access < best_time => {
                        best = Some((i as u32, entry.last_access));
                    }
                    _ => {}
                }
            }
        }
        best.map(|(idx, _)| idx)
    }

    /// Evict a single block: compress and write to temp file.
    fn evict_block(&mut self, idx: u32) -> io::Result<()> {
        let entry = &mut self.blocks[idx as usize];

        // Take the data out, replacing with a temporary Free state
        let data = match std::mem::replace(&mut entry.state, BlockState::Free) {
            BlockState::Resident(data) => data,
            other => {
                entry.state = other;
                return Ok(());
            }
        };

        let file_id = self.next_file_id;
        self.next_file_id += 1;
        let path = self.temp_dir.join(format!("blk_{file_id}.zst"));

        let compressed = zstd::encode_all(data.as_slice(), 1).map_err(io::Error::other)?;
        let compressed_size = compressed.len() as u64;
        fs::write(&path, &compressed)?;

        self.memory_used = self.memory_used.saturating_sub(data.len());
        self.eviction_count += 1;

        let entry = &mut self.blocks[idx as usize];
        entry.state = BlockState::Evicted {
            path,
            compressed_size,
        };

        Ok(())
    }
}

impl Drop for BlockPool {
    fn drop(&mut self) {
        // Clean up all temp files
        for entry in &self.blocks {
            if let BlockState::Evicted { path, .. } = &entry.state {
                let _ = fs::remove_file(path);
            }
        }
        // Try to remove the temp directory (succeeds only if empty)
        let _ = fs::remove_dir(&self.temp_dir);
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pool(budget: usize) -> BlockPool {
        let dir = tempfile::tempdir().unwrap();
        BlockPool::new(budget, dir.into_path()).unwrap()
    }

    #[test]
    fn allocate_and_read_write() {
        let mut pool = test_pool(1 << 20); // 1 MB budget
        let id = pool.allocate(1024).unwrap();
        {
            let data = pool.get_mut(id).unwrap();
            assert_eq!(data.len(), 1024);
            data[0] = 42;
            data[1023] = 99;
        }
        {
            let data = pool.get(id).unwrap();
            assert_eq!(data[0], 42);
            assert_eq!(data[1023], 99);
        }
    }

    #[test]
    fn allocate_with_data() {
        let mut pool = test_pool(1 << 20);
        let src = vec![1u8, 2, 3, 4, 5];
        let id = pool.allocate_with(src.clone()).unwrap();
        assert_eq!(pool.get(id).unwrap(), &src);
    }

    #[test]
    fn eviction_and_reload() {
        // Budget of 2 KB — allocate 3 × 1 KB blocks → forces eviction
        let mut pool = test_pool(2048);

        let a = pool.allocate(1024).unwrap();
        pool.get_mut(a).unwrap().fill(0xAA);

        let b = pool.allocate(1024).unwrap();
        pool.get_mut(b).unwrap().fill(0xBB);

        // At this point, 2 KB resident, at budget
        assert_eq!(pool.memory_used(), 2048);

        // Allocating a third should evict the LRU (a, since b was accessed more recently)
        let c = pool.allocate(1024).unwrap();
        pool.get_mut(c).unwrap().fill(0xCC);

        assert!(pool.eviction_count() >= 1);

        // Reading a should reload it from disk
        let a_data = pool.get(a).unwrap();
        assert!(a_data.iter().all(|&b| b == 0xAA));
        assert!(pool.reload_count() >= 1);

        // b and c should still be accessible
        assert!(pool.get(b).unwrap().iter().all(|&b| b == 0xBB));
        assert!(pool.get(c).unwrap().iter().all(|&b| b == 0xCC));
    }

    #[test]
    fn pinned_blocks_not_evicted() {
        let mut pool = test_pool(2048);

        let a = pool.allocate(1024).unwrap();
        pool.get_mut(a).unwrap().fill(0xAA);
        pool.pin(a).unwrap();

        let b = pool.allocate(1024).unwrap();
        pool.get_mut(b).unwrap().fill(0xBB);

        // a is pinned, so b (unpinned, more recent) should be evicted instead
        let c = pool.allocate(1024).unwrap();
        pool.get_mut(c).unwrap().fill(0xCC);

        // a should still be resident (pinned)
        assert!(pool.is_resident(a));
        // b should have been evicted
        assert!(!pool.is_resident(b));

        pool.unpin(a);
    }

    #[test]
    fn mark_cold_evicts_first() {
        let mut pool = test_pool(2048);

        let a = pool.allocate(1024).unwrap();
        pool.get_mut(a).unwrap().fill(0xAA);

        let b = pool.allocate(1024).unwrap();
        pool.get_mut(b).unwrap().fill(0xBB);

        // Mark b as cold — it should be evicted before a despite being newer
        pool.mark_cold(b);

        let _c = pool.allocate(1024).unwrap();
        assert!(pool.is_resident(a));
        assert!(!pool.is_resident(b));
    }

    #[test]
    fn free_block() {
        let mut pool = test_pool(1 << 20);
        let id = pool.allocate(4096).unwrap();
        assert_eq!(pool.memory_used(), 4096);
        assert_eq!(pool.block_count(), 1);

        pool.free(id).unwrap();
        assert_eq!(pool.memory_used(), 0);
        assert_eq!(pool.block_count(), 0);
    }

    #[test]
    fn free_slot_reused() {
        let mut pool = test_pool(1 << 20);
        let a = pool.allocate(1024).unwrap();
        pool.free(a).unwrap();

        let b = pool.allocate(512).unwrap();
        // Should reuse slot 0
        assert_eq!(b, BlockId(0));
        assert_eq!(pool.get(b).unwrap().len(), 512);
    }

    #[test]
    fn many_blocks_round_trip() {
        // Allocate 100 blocks with a tiny budget to force heavy eviction
        let mut pool = test_pool(4096); // 4 KB budget
        let mut ids = Vec::new();

        for i in 0u8..100 {
            let id = pool.allocate(256).unwrap();
            pool.get_mut(id).unwrap().fill(i);
            ids.push(id);
        }

        // Verify all blocks survive round-trip
        for (i, &id) in ids.iter().enumerate() {
            let data = pool.get(id).unwrap();
            assert!(data.iter().all(|&b| b == i as u8), "block {i} corrupted");
        }
    }

    #[test]
    fn drop_cleans_temp_files() {
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();
        let pool_dir = dir_path.join("pool_test");

        {
            let mut pool = BlockPool::new(1024, pool_dir.clone()).unwrap();
            let a = pool.allocate(1024).unwrap();
            pool.get_mut(a).unwrap().fill(0xFF);
            // Force eviction
            let _b = pool.allocate(1024).unwrap();
            assert!(pool.eviction_count() >= 1);
            // Drop pool — should clean up
        }

        // Temp files should be cleaned up
        if pool_dir.exists() {
            let entries: Vec<_> = fs::read_dir(&pool_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .collect();
            assert!(entries.is_empty(), "temp files not cleaned up: {entries:?}");
        }
    }
}
