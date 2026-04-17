// src/graph/io_operations.rs
//
// Versioned binary format for KnowledgeGraph persistence.
//
// File format v3 layout:
//   [0..4]    Magic: b"RGF\x03" (Rusty Graph Format, version 3)
//   [4..8]    core_data_version: u32 LE (tracks NodeData/EdgeData/Value changes)
//   [8..12]   metadata_length: u32 LE
//   [12..12+N]  JSON metadata (column schemas, section sizes, all config)
//   [section]  topology.zst — graph structure WITHOUT node properties
//   [section]  columns_<Type>.zst — one per node type, packed column data
//   [section]  embeddings.zst (optional)
//   [section]  timeseries.zst (optional)

use crate::graph::storage::memory::column_store::ColumnStore;
use crate::graph::introspection::reporting::OperationReports;
use crate::graph::schema::{
    CompositeIndexKey, ConnectionTypeInfo, ConnectivityTriple, CowSelection, DirGraph,
    EmbeddingStore, IndexKey, PropertyStorage, SaveMetadata, SchemaDefinition,
    SerdeDeserializeGuard, SerdeSerializeGuard, SpatialConfig, StringInterner,
    StripPropertiesGuard, TemporalConfig,
};
use crate::graph::storage::{GraphRead, GraphWrite};
use crate::graph::timeseries::{NodeTimeseries, TimeseriesConfig};
use crate::graph::{KnowledgeGraph, TemporalContext};
use bincode::Options;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::sync::Arc;

/// Return a pinned bincode configuration that is identical to the legacy
/// `bincode::serialize` / `bincode::deserialize` encoding:
///   - Fixed-size integer encoding (not varint)
///   - Little-endian byte order
///   - No trailing bytes rejected
///   - 2 GiB size limit (generous, prevents OOM on corrupt files)
///
/// Using explicit options guarantees wire-format stability regardless of
/// bincode crate default changes or future upgrades.
fn bincode_options() -> impl bincode::Options {
    bincode::options()
        .with_fixint_encoding()
        .with_little_endian()
        .allow_trailing_bytes()
        .with_limit(2 * 1024 * 1024 * 1024) // 2 GiB
}

/// Magic bytes for the v3 columnar format: "RGF\x03"
const V3_MAGIC: [u8; 4] = [0x52, 0x47, 0x46, 0x03];

/// Current core data version. Bump ONLY when NodeData, EdgeData, or Value enum changes.
/// This is independent of metadata — metadata uses JSON and handles changes via serde defaults.
const CURRENT_CORE_DATA_VERSION: u32 = 1;

/// File format version exposed for tests and diagnostics.
#[allow(dead_code)]
pub const CURRENT_FORMAT_VERSION: u32 = 3;

/// Column section metadata for v3 format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct V3ColumnSection {
    type_name: String,
    compressed_size: u64,
    row_count: u32,
    columns: HashMap<String, String>, // prop_name → type_tag
}

/// Metadata serialized as JSON in v3 files. All fields use `#[serde(default)]`
/// so that adding/removing fields never breaks existing files.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct FileMetadata {
    /// Core data version at save time — must match or be migratable.
    #[serde(default)]
    core_data_version: u32,
    /// Library version string at save time (e.g. "0.6.5").
    #[serde(default)]
    library_version: String,
    /// Optional schema definition.
    #[serde(default)]
    schema_definition: Option<SchemaDefinition>,
    /// Property index keys to rebuild after load.
    #[serde(default)]
    property_index_keys: Vec<IndexKey>,
    /// Composite index keys to rebuild after load.
    #[serde(default)]
    composite_index_keys: Vec<CompositeIndexKey>,
    /// Range index keys to rebuild after load.
    #[serde(default)]
    range_index_keys: Vec<IndexKey>,
    /// Node type metadata: node_type → { property_name → type_string }
    #[serde(default)]
    node_type_metadata: HashMap<String, HashMap<String, String>>,
    /// Connection type metadata: connection_type → ConnectionTypeInfo
    #[serde(default)]
    connection_type_metadata: HashMap<String, ConnectionTypeInfo>,
    /// Original ID field name per node type (for alias resolution)
    #[serde(default)]
    id_field_aliases: HashMap<String, String>,
    /// Original title field name per node type (for alias resolution)
    #[serde(default)]
    title_field_aliases: HashMap<String, String>,
    /// Auto-vacuum threshold (None = disabled, default Some(0.3))
    #[serde(default = "default_auto_vacuum_threshold")]
    auto_vacuum_threshold: Option<f64>,
    /// Parent types: child_type → parent_type. Determines which types are
    /// "core" vs "supporting" in describe() output.
    #[serde(default)]
    parent_types: HashMap<String, String>,
    /// Spatial configuration per node type.
    #[serde(default)]
    spatial_configs: HashMap<String, SpatialConfig>,
    /// Timeseries configuration per node type.
    #[serde(default)]
    timeseries_configs: HashMap<String, TimeseriesConfig>,
    /// Temporal configuration per node type (valid_from/valid_to on nodes).
    #[serde(default)]
    temporal_node_configs: HashMap<String, TemporalConfig>,
    /// Temporal configuration per connection type (valid_from/valid_to on edges).
    #[serde(default)]
    temporal_edge_configs: HashMap<String, Vec<TemporalConfig>>,
    /// Timeseries data version: 1 = Vec<Vec<i64>> keys (legacy), 2 = NaiveDate keys.
    #[serde(default = "default_ts_data_version")]
    timeseries_data_version: u32,
    /// v3: compressed size of topology section.
    #[serde(default)]
    topology_compressed_size: u64,
    /// v3: column sections metadata (one per node type).
    #[serde(default)]
    column_sections: Vec<V3ColumnSection>,
    /// v3: compressed size of embedding section (0 if none).
    #[serde(default)]
    embeddings_compressed_size: u64,
    /// v3: compressed size of timeseries section (0 if none).
    #[serde(default)]
    timeseries_compressed_size: u64,
    /// Cached edge type counts (connection_type → count).
    /// Persisted from warm cache on save, restored to cache on load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    edge_type_counts: Option<HashMap<String, usize>>,
    /// Type connectivity triples: (src_type, conn_type, tgt_type, count).
    /// Pre-computed type-level graph for instant describe() at any scale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    type_connectivity: Option<Vec<ConnectivityTriple>>,
}

fn default_auto_vacuum_threshold() -> Option<f64> {
    Some(0.3)
}

fn default_ts_data_version() -> u32 {
    2
}

// ─── Metadata transfer helpers ───────────────────────────────────────────────

impl FileMetadata {
    /// Build metadata from a DirGraph, leaving v3 section sizes at zero
    /// (caller fills them in after compression).
    pub(crate) fn from_graph(graph: &DirGraph) -> Self {
        FileMetadata {
            core_data_version: CURRENT_CORE_DATA_VERSION,
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            schema_definition: graph.schema_definition.clone(),
            property_index_keys: graph.property_index_keys.clone(),
            composite_index_keys: graph.composite_index_keys.clone(),
            range_index_keys: graph.range_index_keys.clone(),
            node_type_metadata: graph.node_type_metadata.clone(),
            connection_type_metadata: graph.connection_type_metadata.clone(),
            id_field_aliases: graph.id_field_aliases.clone(),
            title_field_aliases: graph.title_field_aliases.clone(),
            auto_vacuum_threshold: graph.auto_vacuum_threshold,
            parent_types: graph.parent_types.clone(),
            spatial_configs: graph.spatial_configs.clone(),
            timeseries_configs: graph.timeseries_configs.clone(),
            temporal_node_configs: graph.temporal_node_configs.clone(),
            temporal_edge_configs: graph.temporal_edge_configs.clone(),
            timeseries_data_version: 2,
            // Section sizes filled in by caller:
            topology_compressed_size: 0,
            column_sections: Vec::new(),
            embeddings_compressed_size: 0,
            timeseries_compressed_size: 0,
            // Persist edge type counts if cache is warm (no O(E) scan if cold)
            edge_type_counts: if graph.has_edge_type_counts_cache() {
                Some(graph.get_edge_type_counts())
            } else {
                None
            },
            // Persist type connectivity if computed
            type_connectivity: graph.get_type_connectivity(),
        }
    }

    /// Apply metadata fields to a DirGraph during load.
    pub(crate) fn apply_to(self, graph: &mut DirGraph) {
        graph.schema_definition = self.schema_definition;
        graph.property_index_keys = self.property_index_keys;
        graph.composite_index_keys = self.composite_index_keys;
        graph.range_index_keys = self.range_index_keys;
        graph.node_type_metadata = self.node_type_metadata;
        graph.connection_type_metadata = self.connection_type_metadata;
        graph.id_field_aliases = self.id_field_aliases;
        graph.title_field_aliases = self.title_field_aliases;
        graph.auto_vacuum_threshold = self.auto_vacuum_threshold;
        graph.parent_types = self.parent_types;
        graph.spatial_configs = self.spatial_configs;
        graph.timeseries_configs = self.timeseries_configs;
        graph.temporal_node_configs = self.temporal_node_configs;
        graph.temporal_edge_configs = self.temporal_edge_configs;
        graph.save_metadata = SaveMetadata {
            format_version: 3,
            library_version: self.library_version,
        };
        // Restore edge type counts cache if persisted
        if let Some(counts) = self.edge_type_counts {
            *graph.edge_type_counts_cache.write().unwrap() = Some(counts);
        }
        // Restore type connectivity cache if persisted
        if let Some(triples) = self.type_connectivity {
            *graph.type_connectivity_cache.write().unwrap() = Some(triples);
        } else if !graph.connection_type_metadata.is_empty() {
            // Derive type connectivity from connection_type_metadata (instant, no I/O).
            // This covers older graphs that don't have persisted type_connectivity.
            let edge_counts = graph.edge_type_counts_cache.read().unwrap();
            let mut triples = Vec::new();
            for (conn_type, info) in &graph.connection_type_metadata {
                let count = edge_counts
                    .as_ref()
                    .and_then(|c| c.get(conn_type).copied())
                    .unwrap_or(0);
                for src in &info.source_types {
                    for tgt in &info.target_types {
                        triples.push(crate::graph::schema::ConnectivityTriple {
                            src: src.clone(),
                            conn: conn_type.clone(),
                            tgt: tgt.clone(),
                            count,
                        });
                    }
                }
            }
            if !triples.is_empty() {
                *graph.type_connectivity_cache.write().unwrap() = Some(triples);
            }
        }
    }
}

/// Build metadata for disk-mode save (reuses the same FileMetadata structure).
pub(crate) fn build_disk_metadata(graph: &DirGraph) -> FileMetadata {
    FileMetadata::from_graph(graph)
}

// ─── Save ────────────────────────────────────────────────────────────────────

/// Stamp save metadata and snapshot index keys. Quick, runs with GIL held.
pub fn prepare_save(graph: &mut Arc<DirGraph>) {
    let g = Arc::make_mut(graph);
    g.save_metadata = SaveMetadata::current();
    g.populate_index_keys();
}

/// Compress data using zstd (level 1 — fastest with good ratio).
fn zstd_compress(data: &[u8]) -> io::Result<Vec<u8>> {
    zstd::encode_all(std::io::Cursor::new(data), 1)
}

/// Decompress zstd-compressed data.
fn zstd_decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    zstd::decode_all(std::io::Cursor::new(data))
}

/// Serialize a value using the project's pinned bincode options.
fn bincode_ser<T: Serialize>(val: &T) -> io::Result<Vec<u8>> {
    bincode_options().serialize(val).map_err(io::Error::other)
}

/// Deserialize a value using the project's pinned bincode options.
fn bincode_deser<'a, T: Deserialize<'a>>(buf: &'a [u8]) -> io::Result<T> {
    bincode_options()
        .deserialize(buf)
        .map_err(|e| io::Error::other(format!("bincode deserialization failed: {}", e)))
}

/// Serialize, compress, and write graph data to v3 file. Heavy I/O, safe to run without GIL.
///
/// The graph MUST have columnar storage enabled before calling this function.
/// The caller (Python `save()`) handles auto-enable/disable.
pub fn write_graph_v3(graph: &DirGraph, path: &str) -> io::Result<()> {
    // 1. Serialize topology with properties stripped (v3: node props are in column sections)
    let topology_raw = {
        let _strip = StripPropertiesGuard::new();
        let _guard = SerdeSerializeGuard::new(&graph.interner);
        bincode_ser(&graph.graph)?
    };
    let topology_compressed = zstd_compress(&topology_raw)?;
    drop(topology_raw); // free before compressing columns

    // 2. Serialize column sections (one per node type).
    //
    // Iterate column_stores in sorted order by type_name. `graph.column_stores`
    // is a HashMap whose per-instance RandomState would otherwise cause the
    // section order to vary across processes — breaking byte-level reproducibility
    // that the Phase 4 golden-hash test relies on. Sorting here is free
    // (type_name count is small) and doesn't affect the format: each section
    // is self-describing and the decoder iterates column_sections_meta in order.
    let mut column_sections_meta: Vec<V3ColumnSection> = Vec::new();
    let mut column_sections_data: Vec<Vec<u8>> = Vec::new();

    let mut column_stores_sorted: Vec<(&String, &Arc<ColumnStore>)> =
        graph.column_stores.iter().collect();
    column_stores_sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (type_name, store) in column_stores_sorted {
        let packed = store.write_packed(&graph.interner)?;
        let compressed = zstd_compress(&packed)?;
        drop(packed); // free uncompressed before next type

        // Build column schema
        let mut cols = HashMap::new();
        for (slot, ik) in store.schema().iter() {
            let prop_name = graph.interner.resolve(ik);
            if let Some(col) = store.columns_ref().get(slot as usize) {
                cols.insert(prop_name.to_string(), col.type_tag().to_string());
            }
        }

        column_sections_meta.push(V3ColumnSection {
            type_name: type_name.clone(),
            compressed_size: compressed.len() as u64,
            row_count: store.row_count(),
            columns: cols,
        });
        column_sections_data.push(compressed);
    }

    // 3. Compress embeddings if any
    let embedding_compressed = if !graph.embeddings.is_empty() {
        let raw = bincode_ser(&graph.embeddings)?;
        Some(zstd_compress(&raw)?)
    } else {
        None
    };

    // 4. Compress timeseries if any
    let timeseries_compressed = if !graph.timeseries_store.is_empty() {
        let raw = bincode_ser(&graph.timeseries_store)?;
        Some(zstd_compress(&raw)?)
    } else {
        None
    };

    // 5. Build metadata (common fields from graph, then fill in section sizes)
    let mut metadata = FileMetadata::from_graph(graph);
    metadata.topology_compressed_size = topology_compressed.len() as u64;
    metadata.column_sections = column_sections_meta;
    metadata.embeddings_compressed_size = embedding_compressed
        .as_ref()
        .map(|b| b.len() as u64)
        .unwrap_or(0);
    metadata.timeseries_compressed_size = timeseries_compressed
        .as_ref()
        .map(|b| b.len() as u64)
        .unwrap_or(0);

    // Canonical JSON: round-trip through serde_json::Value so that all
    // HashMap<String, T> fields (nested at any depth) emit with sorted keys.
    // serde_json::Value::Object is backed by BTreeMap<String, Value> (default
    // feature set), so to_value sorts object keys and to_vec walks the tree
    // in sorted order. Prevents per-process HashMap-randomization from
    // producing different save bytes for the same graph — the byte-level
    // tripwire in `tests/test_phase4_parity.py` depends on this.
    let metadata_value = serde_json::to_value(&metadata).map_err(io::Error::other)?;
    let metadata_json = serde_json::to_vec(&metadata_value).map_err(io::Error::other)?;

    // 6. Write file
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Header: magic (4B) + core_data_version (4B) + metadata_length (4B)
    writer.write_all(&V3_MAGIC)?;
    writer.write_all(&CURRENT_CORE_DATA_VERSION.to_le_bytes())?;
    writer.write_all(&(metadata_json.len() as u32).to_le_bytes())?;
    writer.write_all(&metadata_json)?;

    // Topology section
    writer.write_all(&topology_compressed)?;

    // Column sections (one per node type, in metadata order)
    for section_data in &column_sections_data {
        writer.write_all(section_data)?;
    }

    // Embeddings section
    if let Some(emb_data) = &embedding_compressed {
        writer.write_all(emb_data)?;
    }

    // Timeseries section
    if let Some(ts_data) = &timeseries_compressed {
        writer.write_all(ts_data)?;
    }

    writer.flush()?;
    Ok(())
}

// ─── Load ────────────────────────────────────────────────────────────────────

/// Minimum file size to use mmap for the initial file read.
/// Below this threshold, `std::fs::read()` is faster (avoids mmap syscall overhead).
const FILE_MMAP_THRESHOLD: u64 = 65_536; // 64 KB

pub fn load_file(path: &str) -> io::Result<KnowledgeGraph> {
    // If path is a directory, load as disk graph
    let p = std::path::Path::new(path);
    if p.is_dir() {
        return load_disk_dir(p);
    }

    let file = File::open(path)?;
    let file_len = file.metadata()?.len();

    // For large files, mmap avoids the full copy into a Vec<u8>
    if file_len >= FILE_MMAP_THRESHOLD {
        // SAFETY: `Mmap::map` is unsafe because a concurrent writer could
        // race with the reader. The caller of `load_kgl` is the KGLite
        // Python binding, which holds the GIL; no other process is
        // expected to mutate the file during load.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 4 {
            return Err(io::Error::other(
                "File is too small to be a valid kglite file.",
            ));
        }
        if mmap[..4] == V3_MAGIC {
            return load_v3(&mmap);
        }
        return Err(io::Error::other(
            "Unrecognized file format. This file was saved with an older version of kglite. \
             Please rebuild the graph with the current version and save again.",
        ));
    }

    // Small files: direct read is faster
    let buf = std::fs::read(path)?;
    if buf.len() < 4 {
        return Err(io::Error::other(
            "File is too small to be a valid kglite file.",
        ));
    }
    if buf[..4] == V3_MAGIC {
        load_v3(&buf)
    } else {
        Err(io::Error::other(
            "Unrecognized file format. This file was saved with an older version of kglite. \
             Please rebuild the graph with the current version and save again.",
        ))
    }
}

/// Load a disk-mode graph from a directory.
fn load_disk_dir(dir: &std::path::Path) -> io::Result<KnowledgeGraph> {
    use crate::graph::schema::GraphBackend;

    // Verify this is a disk graph directory
    if !dir.join("disk_graph_meta.json").exists() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Directory does not contain a valid disk graph (missing disk_graph_meta.json)",
        ));
    }

    let mut graph = DirGraph::new();

    // Load DirGraph metadata
    if dir.join("metadata.json").exists() {
        let meta_str = std::fs::read_to_string(dir.join("metadata.json"))?;
        let meta: FileMetadata = serde_json::from_str(&meta_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        meta.apply_to(&mut graph);
    }

    // Load interner from JSON map { hash_string: original_string }
    if dir.join("interner.json").exists() {
        let interner_str = std::fs::read_to_string(dir.join("interner.json"))?;
        let interner_map: std::collections::HashMap<String, String> =
            serde_json::from_str(&interner_str)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        for original in interner_map.values() {
            graph.interner.get_or_intern(original);
        }
    }

    // Load DiskGraph — compressed files decompressed to temp dir, then mmap'd
    let (disk_graph, temp_dir) = crate::graph::storage::disk::disk_graph::DiskGraph::load_from_dir(dir)?;
    // Prefetch hot mmap regions (offset arrays + node_slots) into page cache.
    // Non-blocking — kernel reads asynchronously while we continue loading metadata.
    disk_graph.prefetch_hot_regions();
    // Phase 5: this is the `.kgl` → `KnowledgeGraph` construction boundary;
    // assembling the backend variant here is analogous to the PyO3 boundary
    // the storage refactor exempts. Stays as an enum literal.
    graph.graph = GraphBackend::Disk(Box::new(disk_graph));

    // Register temp dir for cleanup on drop
    if let Ok(mut dirs) = graph.temp_dirs.lock() {
        dirs.push(temp_dir);
    }

    // Load type_indices from disk, or rebuild from node_slots if file missing.
    //
    // Phase 5: the fallback below reads `dg.node_slots` (a disk-internal mmap
    // column) directly — no trait surface exposes it. The per-backend
    // `impl GraphRead for DiskGraph` that Phase 5 adds will let this
    // scan move into disk-backend code, collapsing the enum pattern. Until
    // then, `GraphBackend::Disk(ref dg)` destructuring is required.
    if let GraphBackend::Disk(ref dg) = graph.graph {
        let ti_path = dir.join("type_indices.bin.zst");
        let mut loaded = false;
        if ti_path.exists() {
            if let Ok(compressed) = std::fs::read(&ti_path) {
                if let Ok(bytes) = zstd::decode_all(compressed.as_slice()) {
                    if let Ok(indices) = bincode::deserialize(&bytes) {
                        graph.type_indices = indices;
                        loaded = true;
                    }
                }
            }
        }
        if !loaded {
            // Fallback: rebuild from node_slots scan
            let mut new_type_indices: std::collections::HashMap<
                String,
                Vec<petgraph::graph::NodeIndex>,
            > = std::collections::HashMap::new();
            for i in 0..dg.node_slots.len() {
                let slot = dg.node_slots.get(i);
                if slot.is_alive() {
                    let key = crate::graph::schema::InternedKey::from_u64(slot.node_type);
                    if let Some(type_name) = graph.interner.try_resolve(key) {
                        new_type_indices
                            .entry(type_name.to_string())
                            .or_default()
                            .push(petgraph::graph::NodeIndex::new(i));
                    }
                }
            }
            graph.type_indices = new_type_indices;
        }
    }

    // Build type_schemas from node_type_metadata (needed for column loading)
    for (node_type, props) in &graph.node_type_metadata {
        let mut schema = crate::graph::schema::TypeSchema::new();
        for prop_name in props.keys() {
            let key = graph.interner.get_or_intern(prop_name);
            schema.add_key(key);
        }
        graph
            .type_schemas
            .insert(node_type.clone(), std::sync::Arc::new(schema));
    }

    // Load column stores — prefer mmap-backed (columns.bin + columns_meta)
    let mmap_path = dir.join("columns.bin");
    let meta_bin_path = dir.join("columns_meta.bin.zst");
    let meta_json_path = dir.join("columns_meta.json");
    let has_mmap = mmap_path.exists() && (meta_bin_path.exists() || meta_json_path.exists());
    if has_mmap {
        use crate::graph::io::ntriples::ColumnTypeMeta;
        use memmap2::MmapMut;

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&mmap_path)?;
        // SAFETY: columns.bin exists in the disk-graph directory and is
        // opened read-write by this loader. KGLite holds the Python GIL
        // during load; no other process writes to the file concurrently.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        let mmap_arc = std::sync::Arc::new(mmap);

        // Prefer bincode (fast) over JSON (slow for 295 MB)
        let type_metas: Vec<ColumnTypeMeta> = if meta_bin_path.exists() {
            let compressed = std::fs::read(&meta_bin_path)?;
            let bytes = zstd::decode_all(compressed.as_slice()).map_err(io::Error::other)?;
            bincode::deserialize(&bytes).map_err(io::Error::other)?
        } else {
            let meta_json = std::fs::read_to_string(&meta_json_path)?;
            serde_json::from_str(&meta_json).map_err(io::Error::other)?
        };

        for tm in type_metas {
            let store = tm.to_mmap_store(std::sync::Arc::clone(&mmap_arc));
            let cs = crate::graph::storage::memory::column_store::ColumnStore::from_mmap_store(std::sync::Arc::new(
                store,
            ));
            graph.column_stores.insert(tm.type_name, Arc::new(cs));
        }
    } else {
        // Legacy path: load from columns/<type>/columns.zst files
        let columns_dir = dir.join("columns");
        if columns_dir.exists() {
            for entry in std::fs::read_dir(&columns_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    let type_name = entry.file_name().to_string_lossy().to_string();
                    let col_file = entry.path().join("columns.zst");
                    if col_file.exists() {
                        let compressed = std::fs::read(&col_file)?;
                        let packed =
                            zstd::decode_all(compressed.as_slice()).map_err(io::Error::other)?;
                        let schema =
                            graph
                                .type_schemas
                                .get(&type_name)
                                .cloned()
                                .unwrap_or_else(|| {
                                    std::sync::Arc::new(crate::graph::schema::TypeSchema::new())
                                });
                        let type_meta = graph
                            .node_type_metadata
                            .get(&type_name)
                            .cloned()
                            .unwrap_or_default();
                        let row_count = graph
                            .type_indices
                            .get(&type_name)
                            .map(|v| v.len() as u32)
                            .unwrap_or(0);
                        let store = crate::graph::storage::memory::column_store::ColumnStore::load_packed(
                            schema,
                            &type_meta,
                            &graph.interner,
                            &packed,
                            row_count,
                            None,
                        )?;
                        graph.column_stores.insert(type_name, Arc::new(store));
                    }
                }
            }
        }
    }

    // Sync column stores to DiskGraph
    graph.sync_disk_column_stores();

    // Load id_indices from disk (saved during build as bincode + zstd).
    if crate::graph::storage::GraphRead::is_disk(&graph.graph) {
        let id_indices_path = dir.join("id_indices.bin.zst");
        if id_indices_path.exists() {
            if let Ok(compressed) = std::fs::read(&id_indices_path) {
                if let Ok(bytes) = zstd::decode_all(compressed.as_slice()) {
                    if let Ok(indices) = bincode::deserialize(&bytes) {
                        graph.id_indices = indices;
                    }
                }
            }
        }
    }

    // Load embeddings if present
    let emb_path = dir.join("embeddings.bin.zst");
    if emb_path.exists() {
        if let Ok(compressed) = std::fs::read(&emb_path) {
            if let Ok(bytes) = zstd::decode_all(compressed.as_slice()) {
                if let Ok(embeddings) =
                    bincode::deserialize::<HashMap<(String, String), EmbeddingStore>>(&bytes)
                {
                    graph.embeddings = embeddings;
                }
            }
        }
    }

    // Load timeseries if present
    let ts_path = dir.join("timeseries.bin.zst");
    if ts_path.exists() {
        if let Ok(compressed) = std::fs::read(&ts_path) {
            if let Ok(bytes) = zstd::decode_all(compressed.as_slice()) {
                if let Ok(ts_store) = bincode::deserialize::<HashMap<usize, NodeTimeseries>>(&bytes)
                {
                    graph.timeseries_store = ts_store;
                }
            }
        }
    }

    Ok(KnowledgeGraph {
        inner: Arc::new(graph),
        selection: CowSelection::new(),
        reports: OperationReports::new(),
        last_mutation_stats: None,
        embedder: None,
        temporal_context: TemporalContext::default(),
        default_timeout_ms: None,
        default_max_rows: None,
    })
}

/// Load v3 columnar format.
fn load_v3(buf: &[u8]) -> io::Result<KnowledgeGraph> {
    if buf.len() < 12 {
        return Err(io::Error::other(
            "v3 file is truncated — header incomplete.",
        ));
    }

    // Parse header
    let core_version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let metadata_len = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;

    if core_version > CURRENT_CORE_DATA_VERSION {
        return Err(io::Error::other(format!(
            "File uses core data version {} but this library only supports up to version {}. \
             Please upgrade kglite.",
            core_version, CURRENT_CORE_DATA_VERSION,
        )));
    }

    let metadata_end = 12 + metadata_len;
    if buf.len() < metadata_end {
        return Err(io::Error::other(
            "v3 file is truncated — metadata incomplete.",
        ));
    }

    // Parse JSON metadata
    let metadata: FileMetadata = serde_json::from_slice(&buf[12..metadata_end])
        .map_err(|e| io::Error::other(format!("Failed to parse v3 metadata: {}", e)))?;

    // Section offsets
    let topology_start = metadata_end;
    let topology_end = topology_start + metadata.topology_compressed_size as usize;

    // Decompress + deserialize topology (properties are empty maps)
    let topology_compressed = &buf[topology_start..topology_end];
    let topology_raw = zstd_decompress(topology_compressed)?;

    let mut interner = StringInterner::new();
    let graph: crate::graph::schema::Graph = {
        let _guard = SerdeDeserializeGuard::new(&mut interner);
        bincode_deser(&topology_raw)?
    };
    drop(topology_raw);

    // Extract v3 section metadata before apply_to consumes the rest
    let column_sections = metadata.column_sections.clone();
    let embeddings_compressed_size = metadata.embeddings_compressed_size;
    let timeseries_compressed_size = metadata.timeseries_compressed_size;

    // Reassemble DirGraph
    let mut dir_graph = DirGraph::from_graph(graph);
    dir_graph.interner = interner;
    metadata.apply_to(&mut dir_graph);

    // Rebuild type indices and schemas (needed for ColumnStore construction).
    // Note: rebuild_indices_from_keys is deferred until after column loading
    // because properties are empty at this point (stripped during save).
    dir_graph.rebuild_type_indices_and_compact();
    dir_graph.build_connection_types_cache();

    // Load column sections one type at a time
    let mut section_offset = topology_end;

    // Create temp directory for mmap column files (unique per load to avoid collisions)
    let temp_dir = std::env::temp_dir().join(format!(
        "kglite_v3_{}_{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    // Register for cleanup on DirGraph drop
    if let Ok(mut dirs) = dir_graph.temp_dirs.lock() {
        dirs.push(temp_dir.clone());
    }

    for section_meta in &column_sections {
        let section_end = section_offset + section_meta.compressed_size as usize;
        if buf.len() < section_end {
            return Err(io::Error::other(format!(
                "v3 file truncated — column section '{}' incomplete.",
                section_meta.type_name
            )));
        }

        let compressed = &buf[section_offset..section_end];
        let packed = zstd_decompress(compressed)?;

        // Build schema from the column section metadata (exact match for saved
        // columns). Using type_schemas here would include id/title columns that
        // are NOT in the column data, creating empty placeholder columns that
        // corrupt the file on re-save.
        {
            let col_keys: Vec<crate::graph::schema::InternedKey> = section_meta
                .columns
                .keys()
                .map(|name| {
                    dir_graph.interner.get_or_intern(name);
                    crate::graph::schema::InternedKey::from_str(name)
                })
                .collect();
            let column_schema = Arc::new(crate::graph::schema::TypeSchema::from_keys(col_keys));

            let type_meta = dir_graph
                .node_type_metadata
                .get(&section_meta.type_name)
                .cloned()
                .unwrap_or_default();

            // Create temp dir for this type's column files
            let type_temp_dir = temp_dir.join(&section_meta.type_name);
            std::fs::create_dir_all(&type_temp_dir)?;

            let store = ColumnStore::load_packed(
                column_schema,
                &type_meta,
                &dir_graph.interner,
                &packed,
                section_meta.row_count,
                Some(&type_temp_dir),
            )?;
            drop(packed); // free before next type

            dir_graph
                .column_stores
                .insert(section_meta.type_name.clone(), Arc::new(store));
        }

        section_offset = section_end;
    }

    // Re-point nodes to columnar storage
    for (type_name, store) in &dir_graph.column_stores {
        let has_id_title = store.has_id_title_columns();
        if let Some(indices) = dir_graph.type_indices.get(type_name) {
            for (row_id, &node_idx) in indices.iter().enumerate() {
                if let Some(node) = dir_graph.graph.node_weight_mut(node_idx) {
                    node.properties = PropertyStorage::Columnar {
                        store: Arc::clone(store),
                        row_id: row_id as u32,
                    };
                    // Set sentinel values if store has id/title columns (mapped mode)
                    if has_id_title {
                        node.id = Value::Null;
                        node.title = Value::Null;
                    }
                }
            }
        }
    }

    // Now that nodes have columnar properties, rebuild property/range/composite indices
    dir_graph.rebuild_indices_from_keys();

    // Load embeddings if present
    if embeddings_compressed_size > 0 {
        let emb_end = section_offset + embeddings_compressed_size as usize;
        if buf.len() >= emb_end {
            let emb_compressed = &buf[section_offset..emb_end];
            let emb_raw = zstd_decompress(emb_compressed)?;
            let embeddings: HashMap<(String, String), EmbeddingStore> = bincode_deser(&emb_raw)?;
            dir_graph.embeddings = embeddings;
            section_offset = emb_end;
        }
    }

    // Load timeseries if present
    if timeseries_compressed_size > 0 {
        let ts_end = section_offset + timeseries_compressed_size as usize;
        if buf.len() >= ts_end {
            let ts_compressed = &buf[section_offset..ts_end];
            let ts_raw = zstd_decompress(ts_compressed)?;
            let ts_store: HashMap<usize, NodeTimeseries> = bincode_deser(&ts_raw)?;
            dir_graph.timeseries_store = ts_store;
        }
    }

    Ok(KnowledgeGraph {
        inner: Arc::new(dir_graph),
        selection: CowSelection::new(),
        reports: OperationReports::new(),
        last_mutation_stats: None,
        embedder: None,
        temporal_context: TemporalContext::default(),
        default_timeout_ms: None,
        default_max_rows: None,
    })
}

// ─── Embedding Export / Import ────────────────────────────────────────────

use crate::datatypes::values::Value;

/// Magic bytes for the embedding export format.
const KGLE_MAGIC: [u8; 4] = *b"KGLE";
const KGLE_VERSION: u32 = 1;

/// A single embedding store serialized with node IDs (not internal indices).
#[derive(Serialize, Deserialize)]
struct ExportedEmbeddingStore {
    node_type: String,
    text_column: String, // e.g. "summary" (without _emb suffix)
    dimension: usize,
    entries: Vec<(Value, Vec<f32>)>, // (node_id, embedding) pairs
}

/// Filter for selective embedding export.
pub enum EmbeddingExportFilter {
    /// Export all embedding stores for these node types.
    Types(Vec<String>),
    /// Export specific (node_type → [text_columns]) pairs.
    /// An empty vec means all properties for that type.
    TypeProperties(HashMap<String, Vec<String>>),
}

pub struct ExportStats {
    pub stores: usize,
    pub embeddings: usize,
}

pub struct ImportStats {
    pub stores: usize,
    pub imported: usize,
    pub skipped: usize,
}

/// Export embeddings to a standalone .kgle file, keyed by node ID.
pub fn export_embeddings_to_file(
    graph: &DirGraph,
    path: &str,
    filter: Option<&EmbeddingExportFilter>,
) -> io::Result<ExportStats> {
    let mut exported_stores: Vec<ExportedEmbeddingStore> = Vec::new();
    let mut total_embeddings = 0usize;

    for ((node_type, store_name), store) in &graph.embeddings {
        let text_column = store_name
            .strip_suffix("_emb")
            .unwrap_or(store_name.as_str());

        // Apply filter
        if let Some(f) = filter {
            match f {
                EmbeddingExportFilter::Types(types) => {
                    if !types.iter().any(|t| t == node_type) {
                        continue;
                    }
                }
                EmbeddingExportFilter::TypeProperties(map) => {
                    match map.get(node_type) {
                        None => continue, // type not in filter
                        Some(props) if !props.is_empty() => {
                            if !props.iter().any(|p| p == text_column) {
                                continue;
                            }
                        }
                        Some(_) => {} // empty list = all properties for this type
                    }
                }
            }
        }

        // Resolve node indices → node IDs
        let mut entries: Vec<(Value, Vec<f32>)> = Vec::with_capacity(store.len());
        for &node_index in &store.slot_to_node {
            if let Some(node) = graph
                .graph
                .node_weight(petgraph::graph::NodeIndex::new(node_index))
            {
                if let Some(embedding) = store.get_embedding(node_index) {
                    entries.push((node.id().into_owned(), embedding.to_vec()));
                }
            }
        }

        total_embeddings += entries.len();
        exported_stores.push(ExportedEmbeddingStore {
            node_type: node_type.clone(),
            text_column: text_column.to_string(),
            dimension: store.dimension,
            entries,
        });
    }

    // Write: magic + version + gzip(bincode(stores))
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&KGLE_MAGIC)?;
    writer.write_all(&KGLE_VERSION.to_le_bytes())?;

    let gz = GzEncoder::new(&mut writer, Compression::new(3));
    bincode_options()
        .serialize_into(gz, &exported_stores)
        .map_err(|e| io::Error::other(format!("Failed to serialize embeddings: {}", e)))?;

    writer.flush()?;

    Ok(ExportStats {
        stores: exported_stores.len(),
        embeddings: total_embeddings,
    })
}

/// Import embeddings from a .kgle file, resolving node IDs to current graph indices.
pub fn import_embeddings_from_file(graph: &mut DirGraph, path: &str) -> io::Result<ImportStats> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;

    if buf.len() < 8 {
        return Err(io::Error::other(
            "File is too small to be a valid .kgle file.",
        ));
    }

    // Validate magic and version
    if buf[..4] != KGLE_MAGIC {
        return Err(io::Error::other(
            "Not a valid .kgle file (bad magic bytes).",
        ));
    }
    let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    if version > KGLE_VERSION {
        return Err(io::Error::other(format!(
            "Embedding file version {} is newer than supported version {}. Please upgrade kglite.",
            version, KGLE_VERSION,
        )));
    }

    // Decompress and deserialize
    let gz = GzDecoder::new(&buf[8..]);
    let exported_stores: Vec<ExportedEmbeddingStore> = bincode_options()
        .deserialize_from(gz)
        .map_err(|e| io::Error::other(format!("Failed to deserialize embedding data: {}", e)))?;

    let mut total_imported = 0usize;
    let mut total_skipped = 0usize;
    let mut stores_count = 0usize;

    for exported in exported_stores {
        // Build ID index for this node type so lookup_by_id works
        graph.build_id_index(&exported.node_type);

        let mut store = crate::graph::schema::EmbeddingStore::new(exported.dimension);
        store
            .data
            .reserve(exported.entries.len() * exported.dimension);

        let mut imported = 0usize;
        let mut skipped = 0usize;

        for (id, vec) in &exported.entries {
            match graph.lookup_by_id(&exported.node_type, id) {
                Some(node_idx) => {
                    store.set_embedding(node_idx.index(), vec);
                    imported += 1;
                }
                None => {
                    skipped += 1;
                }
            }
        }

        if imported > 0 {
            let key = (exported.node_type, format!("{}_emb", exported.text_column));
            graph.embeddings.insert(key, store);
            stores_count += 1;
        }

        total_imported += imported;
        total_skipped += skipped;
    }

    Ok(ImportStats {
        stores: stores_count,
        imported: total_imported,
        skipped: total_skipped,
    })
}
