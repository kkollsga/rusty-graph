# Disk Mode: Iterative Updates & Live Graph Modification

## Current State (v0.7.10)

### What works
- **N-Triples bulk build**: single-pass parse → accumulate pending_edges → one CSR build at end. Fully optimized, tested at 124M nodes / 863M edges.
- **Small DataFrame builds**: add_nodes → add_connections → save → reload → query. Verified on 3-node graph.
- **Post-load queries**: all Cypher features work after loading a properly built disk graph.

### What doesn't work
- **Large DataFrame builds** (legal 157K nodes, 1M edges): CSR build at save time panics with "MmapOrVec index out of bounds". Root cause: mmap file-handle conflict when CSR build writes to files that are already mmap'd from graph construction.
- **Iterative modifications after load**: no support for add_nodes/add_connections on a loaded disk graph. The CSR is immutable after build.

## Root Cause: Build-in-Same-Directory Mmap Conflict

When `KnowledgeGraph(storage="disk", path="/some/dir")` is created:
1. `DiskGraph::new_at_path()` creates mmap'd files: `node_slots.bin`, `out_offsets.bin`, `in_offsets.bin`, `_pending_edges.bin`
2. `add_nodes()` writes to `node_slots.bin` via mmap — works fine
3. `add_connections()` appends to `_pending_edges.bin` via mmap — works fine
4. `save_disk()` → `ensure_disk_edges_built()` → `build_csr_from_pending()`:
   - Reads `_pending_edges.bin` (the source)
   - Tries to write NEW `out_edges.bin`, `edge_endpoints.bin`, etc. to the SAME directory
   - Also overwrites `out_offsets.bin` and `in_offsets.bin` which are ALREADY mmap'd from step 1
   - The old mmap handles still point to the pre-build file contents
   - After CSR build, the old mmap handles are stale → any access through them panics

The N-Triples path avoids this because `ntriples.rs` calls `build_csr_from_pending()` explicitly, and the CSR arrays are assigned to `self.out_offsets`, `self.out_edges`, etc. DIRECTLY from the build output — replacing the old mmap handles. But in the `save_disk()` path, the build happens inside `ensure_disk_edges_built()` which updates the DiskGraph fields, and then `save_to_dir()` tries to copy/flush them to the target directory.

## What Needs Further Investigation

### 1. Mmap lifecycle during CSR rebuild
- When `build_csr_partitioned()` creates new `out_offsets.bin`, does it invalidate the existing mmap'd `self.out_offsets`?
- The build methods (`build_csr_merge_sort`, `build_csr_partitioned`) assign new `MmapOrVec` values to `self.out_offsets`, `self.out_edges`, etc. at the end. But `save_to_dir()` is called AFTER this — it reads from the new values and writes to target_dir. If target_dir == data_dir, it's writing to files that the new mmaps already point to. Is this a double-write issue?
- Need to trace the exact file handle lifecycle with `RUST_BACKTRACE=1` to find the exact OOB access.

### 2. The `save_to_dir` same-dir optimization
- `save_to_dir()` at `disk_graph.rs:2262` has: `if target_dir != self.data_dir { ... copy files ... }`. When saving to the same dir, it skips the copy because the files are already there (mmap-backed). But after CSR rebuild, the new CSR files may have different sizes than the pre-allocated ones from `new_at_path()`.
- The initial `out_offsets` has capacity 1025 (from `new_at_path`), but after CSR build with 157K nodes, it needs 157K+ entries. The CSR build creates a NEW file, but the save skips copying because it's the same dir.

### 3. Edge properties on disk mode
- `add_edge()` stores edge properties in `self.edge_properties: HashMap<u32, Vec<(InternedKey, Value)>>` (in-heap). These are saved to `edge_properties.bin.zst` during `save_to_dir()`. Need to verify this works for large property sets.

## The Real Goal: Live Graph Modification

The current architecture assumes a build-then-query lifecycle. For a live graph that supports modifications without performance regression, we need:

### Requirements
1. **Add nodes without full rebuild**: append to node_slots, update type_indices, update column stores
2. **Add edges without full CSR rebuild**: append to overflow lists, periodically compact
3. **Remove nodes/edges**: tombstone mechanism (already exists via `TOMBSTONE_EDGE`)
4. **Property updates**: update column store values in place
5. **No query degradation**: modifications should not cause O(N) work on the next query

### Current Overflow Mechanism
DiskGraph already has `overflow_out: HashMap<u32, Vec<CsrEdge>>` and `overflow_in: HashMap<u32, Vec<CsrEdge>>`. These are used for edges added AFTER CSR build. The edge iterators check both CSR arrays AND overflow lists. This is the foundation for live updates.

The problem: overflow lists are in-heap (not persisted), and they grow without bound. A compaction step is needed to merge overflows back into the CSR periodically.

## How Other Libraries Handle This

### DuckDB (Append-Optimized)
- Uses a row-group based storage model
- New data goes into a write-ahead buffer
- Periodically flushed to immutable row groups with zone maps
- Deletes use a validity bitmap (per-row "alive" flag)
- No in-place modification — append + mark old as deleted
- Compaction merges small row groups into larger ones

### RocksDB / LevelDB (LSM-Tree)
- Write-ahead log (WAL) for durability
- Memtable (in-memory sorted buffer) for recent writes
- Immutable SST files on disk (sorted, with bloom filters)
- Compaction merges SST files to maintain read performance
- Reads check memtable first, then each SST level
- Tombstones for deletes, merged during compaction

### Neo4j (Native Graph)
- Record-based storage: fixed-size records for nodes and edges
- Free lists for reuse of deleted record slots
- Edges stored as doubly-linked lists per node
- Updates are in-place mutations of records
- No compaction needed — fixed-size records allow direct overwrite
- Transaction log for durability

### SQLite (WAL Mode)
- WAL (Write-Ahead Log) for concurrent reads during writes
- Checkpoint merges WAL into main database file
- B-tree pages can be updated in place (with copy-on-write for transactions)

## Proposed Architecture for Live Disk Graph Updates

### Approach: CSR + Overflow + Periodic Compaction

Based on DuckDB/RocksDB patterns, adapted for graph structure:

**Write Path (hot)**:
1. New nodes: append to `node_slots.bin` (O(1) mmap append)
2. New edges: append to `overflow_out/overflow_in` (in-heap HashMap)
3. Property updates: update column store values via `set()` (O(1) mmap write)
4. Deletes: tombstone in node_slots or edge_endpoints (O(1) flag flip)

**Read Path (no degradation)**:
1. Node access: `node_slots[idx]` — O(1), same as now
2. Edge traversal: check CSR range THEN overflow list — O(degree + overflow_size)
3. Property access: column store `get(row_id, key)` — O(1), same as now

**Compaction (background, periodic)**:
1. Triggered when overflow lists exceed threshold (e.g., >10% of total edges)
2. Merges overflow edges into CSR arrays
3. Rebuilds CSR from scratch (or incrementally if sorted insert is feasible)
4. Clears overflow lists
5. Updates metadata + connection-type inverted index
6. Can run in background while queries continue reading old CSR + overflow

**Persistence**:
1. Overflow edges saved to `overflow_edges.bin.zst` on save (already implemented)
2. On load, overflow edges restored to in-heap HashMap (already implemented)
3. Metadata tracks whether compaction is needed

### Key Design Decisions Needed

1. **When to compact**: threshold-based (overflow > X% of edges)? Manual (`graph.compact()`)? On save?
2. **Concurrent reads during compaction**: need copy-on-write for CSR arrays, or stop-the-world?
3. **Column store growth**: when a new type appears or a type gets more rows than the column store can hold, need to grow the mmap region. Current `MmapOrVec` supports grow via unmap/truncate/remap.
4. **Id index updates**: `id_indices` is in-heap HashMap. New nodes need to be added. Currently rebuilt from column stores on load.
5. **Type index updates**: `type_indices` is in-heap Vec<NodeIndex>. New nodes need to be appended.

### What's Already Built (Foundation)
- `overflow_out/overflow_in`: edge overflow mechanism ✓
- `DiskNodeSlot.is_alive()`: tombstone for nodes ✓
- `TOMBSTONE_EDGE`: tombstone for edges ✓
- `free_node_slots/free_edge_slots`: slot reuse after deletes ✓
- `save_to_dir()`: persists overflow edges ✓
- `load_from_dir()`: restores overflow edges ✓
- `metadata_dirty` flag: tracks unsaved changes ✓

### What Needs Building
- Fix mmap lifecycle in `save_disk()` for same-dir writes
- Persist overflow edges incrementally (not just on save)
- Compaction: merge overflow into CSR
- Column store append for new nodes of existing types
- Column store creation for new types at runtime
- Type index / id index incremental updates
- Connection-type inverted index incremental updates (or rebuild on compaction)
