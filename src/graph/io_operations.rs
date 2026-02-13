// src/graph/io_operations.rs
//
// Versioned binary format for KnowledgeGraph persistence.
//
// File format v2 layout:
//   [0..4]    Magic: b"RGF\x02" (Rusty Graph Format, version 2)
//   [4..8]    core_data_version: u32 LE (tracks NodeData/EdgeData/Value changes)
//   [8..12]   metadata_length: u32 LE
//   [12..12+N]  JSON metadata (UTF-8, uncompressed)
//   [12+N..12+N+G]  Gzip-compressed bincode of StableDiGraph<NodeData, EdgeData>
//   [12+N+G..]  (optional) Gzip-compressed bincode of embedding stores
//               Present only when metadata.graph_compressed_size is set.
//
// Format detection (first bytes):
//   b"RGF\x02"  → v2 (sectioned format)
//   0x1f 0x8b   → v1 (gzip-compressed full DirGraph)
//   other       → v0 (raw bincode full DirGraph)

use crate::graph::reporting::OperationReports;
use crate::graph::schema::{
    CompositeIndexKey, ConnectionTypeInfo, CowSelection, DirGraph, EmbeddingStore, IndexKey,
    SaveMetadata, SchemaDefinition,
};
use crate::graph::KnowledgeGraph;
use bincode;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::sync::Arc;

/// Magic bytes for the v2 sectioned format: "RGF\x02"
const V2_MAGIC: [u8; 4] = [0x52, 0x47, 0x46, 0x02];

/// Current core data version. Bump ONLY when NodeData, EdgeData, or Value enum changes.
/// This is independent of metadata — metadata uses JSON and handles changes via serde defaults.
const CURRENT_CORE_DATA_VERSION: u32 = 1;

/// File format version exposed for tests and diagnostics.
#[allow(dead_code)]
pub const CURRENT_FORMAT_VERSION: u32 = 2;

/// Metadata serialized as JSON in v2 files. All fields use `#[serde(default)]`
/// so that adding/removing fields never breaks existing files.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct FileMetadata {
    /// Core data version at save time — must match or be migratable.
    #[serde(default)]
    core_data_version: u32,
    /// Library version string at save time (e.g. "0.4.22").
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
    /// Compressed size of the graph section in bytes.
    /// When present, bytes after graph_compressed_size contain the embedding section.
    #[serde(default)]
    graph_compressed_size: Option<u64>,
}

fn default_auto_vacuum_threshold() -> Option<f64> {
    Some(0.3)
}

// ─── Save ────────────────────────────────────────────────────────────────────

pub fn save_to_file(graph: &mut Arc<DirGraph>, path: &str) -> io::Result<()> {
    let g = Arc::make_mut(graph);
    // Stamp save metadata
    g.save_metadata = SaveMetadata::current();
    // Snapshot index keys (indices themselves are #[serde(skip)])
    g.populate_index_keys();

    // Compress graph to buffer first so we know its size (needed for embedding section)
    let mut graph_compressed = Vec::new();
    {
        let gz = GzEncoder::new(&mut graph_compressed, Compression::new(3));
        bincode::serialize_into(gz, &g.graph).map_err(io::Error::other)?;
    }

    // Compress embeddings if any exist
    let has_embeddings = !g.embeddings.is_empty();
    let embedding_compressed = if has_embeddings {
        let mut buf = Vec::new();
        let gz = GzEncoder::new(&mut buf, Compression::new(3));
        bincode::serialize_into(gz, &g.embeddings).map_err(io::Error::other)?;
        Some(buf)
    } else {
        None
    };

    // Build JSON metadata from DirGraph fields
    let metadata = FileMetadata {
        core_data_version: CURRENT_CORE_DATA_VERSION,
        library_version: env!("CARGO_PKG_VERSION").to_string(),
        schema_definition: g.schema_definition.clone(),
        property_index_keys: g.property_index_keys.clone(),
        composite_index_keys: g.composite_index_keys.clone(),
        range_index_keys: g.range_index_keys.clone(),
        node_type_metadata: g.node_type_metadata.clone(),
        connection_type_metadata: g.connection_type_metadata.clone(),
        id_field_aliases: g.id_field_aliases.clone(),
        title_field_aliases: g.title_field_aliases.clone(),
        auto_vacuum_threshold: g.auto_vacuum_threshold,
        graph_compressed_size: if has_embeddings {
            Some(graph_compressed.len() as u64)
        } else {
            None
        },
    };

    let metadata_json = serde_json::to_vec(&metadata).map_err(io::Error::other)?;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header: magic (4B) + core_data_version (4B) + metadata_length (4B)
    writer.write_all(&V2_MAGIC)?;
    writer.write_all(&CURRENT_CORE_DATA_VERSION.to_le_bytes())?;
    writer.write_all(&(metadata_json.len() as u32).to_le_bytes())?;

    // Write JSON metadata (uncompressed — small and self-describing)
    writer.write_all(&metadata_json)?;

    // Write pre-compressed graph data
    writer.write_all(&graph_compressed)?;

    // Write embedding section if present
    if let Some(emb_data) = embedding_compressed {
        writer.write_all(&emb_data)?;
    }

    writer.flush()?;
    Ok(())
}

// ─── Load ────────────────────────────────────────────────────────────────────

pub fn load_file(path: &str) -> io::Result<KnowledgeGraph> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;

    if buf.len() < 4 {
        return Err(io::Error::other(
            "File is too small to be a valid kglite file.",
        ));
    }

    // Detect format by first bytes
    let dir_graph = if buf[..4] == V2_MAGIC {
        load_v2(&buf)?
    } else if buf.len() >= 2 && buf[0] == 0x1f && buf[1] == 0x8b {
        load_v1(&buf)?
    } else {
        return Err(io::Error::other(
            "Unrecognized file format. This file was likely saved with a very old version of \
             kglite (pre-v0.4.18) that used raw bincode without compression. \
             Please rebuild the graph with the current version.",
        ));
    };

    Ok(finalize_load(dir_graph))
}

/// Load v2 sectioned format: header + JSON metadata + gzip bincode graph + optional embeddings.
fn load_v2(buf: &[u8]) -> io::Result<DirGraph> {
    if buf.len() < 12 {
        return Err(io::Error::other(
            "v2 file is truncated — header incomplete.",
        ));
    }

    // Parse header
    let core_version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let metadata_len = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;

    let metadata_end = 12 + metadata_len;
    if buf.len() < metadata_end {
        return Err(io::Error::other(format!(
            "v2 file is truncated — expected {} bytes of metadata but file has only {} bytes after header.",
            metadata_len,
            buf.len() - 12
        )));
    }

    // Parse JSON metadata (self-describing — missing/extra fields handled by serde defaults)
    let metadata: FileMetadata = serde_json::from_slice(&buf[12..metadata_end])
        .map_err(|e| io::Error::other(format!("Failed to parse v2 metadata JSON: {}", e)))?;

    // Determine graph section boundaries
    let graph_start = metadata_end;
    let (graph_bytes, embedding_bytes) = match metadata.graph_compressed_size {
        Some(size) => {
            let graph_end = graph_start + size as usize;
            if buf.len() < graph_end {
                return Err(io::Error::other(
                    "v2 file is truncated — graph section incomplete.",
                ));
            }
            (&buf[graph_start..graph_end], Some(&buf[graph_end..]))
        }
        None => (&buf[graph_start..], None),
    };

    // Decompress and deserialize graph
    let graph = load_core_data(graph_bytes, core_version)?;

    // Reassemble DirGraph
    let mut dir_graph = DirGraph::from_graph(graph);
    dir_graph.schema_definition = metadata.schema_definition;
    dir_graph.property_index_keys = metadata.property_index_keys;
    dir_graph.composite_index_keys = metadata.composite_index_keys;
    dir_graph.range_index_keys = metadata.range_index_keys;
    dir_graph.node_type_metadata = metadata.node_type_metadata;
    dir_graph.connection_type_metadata = metadata.connection_type_metadata;
    dir_graph.id_field_aliases = metadata.id_field_aliases;
    dir_graph.title_field_aliases = metadata.title_field_aliases;
    dir_graph.auto_vacuum_threshold = metadata.auto_vacuum_threshold;
    dir_graph.save_metadata = SaveMetadata {
        format_version: 2,
        library_version: metadata.library_version,
    };

    // Load embeddings if present
    if let Some(emb_bytes) = embedding_bytes {
        if !emb_bytes.is_empty() {
            let gz = GzDecoder::new(emb_bytes);
            let embeddings: HashMap<(String, String), EmbeddingStore> =
                bincode::deserialize_from(gz).map_err(|e| {
                    io::Error::other(format!("Failed to deserialize embedding data: {}", e))
                })?;
            dir_graph.embeddings = embeddings;
        }
    }

    Ok(dir_graph)
}

/// Load v1 format: gzip-compressed bincode of full DirGraph.
fn load_v1(buf: &[u8]) -> io::Result<DirGraph> {
    let gz = GzDecoder::new(buf);
    let dir_graph: DirGraph = bincode::deserialize_from(gz).map_err(|e| {
        io::Error::other(format!(
            "Failed to load v1 (gzip+bincode) file. This file may have been saved with an \
                 incompatible version of kglite. Error: {}",
            e
        ))
    })?;

    // v1 files stored format_version in save_metadata
    let saved_version = dir_graph.save_metadata.format_version;
    if saved_version > 1 {
        return Err(io::Error::other(format!(
            "v1 file claims format_version {} (library {}), but v1 loader only supports version 0-1. \
             This should not happen — file may be corrupt.",
            saved_version, dir_graph.save_metadata.library_version,
        )));
    }

    Ok(dir_graph)
}

/// Decompress and deserialize the core graph data, running migrations if needed.
fn load_core_data(
    graph_bytes: &[u8],
    file_core_version: u32,
) -> io::Result<crate::graph::schema::Graph> {
    if file_core_version > CURRENT_CORE_DATA_VERSION {
        return Err(io::Error::other(format!(
            "File uses core data version {} but this library only supports up to version {}. \
             Please upgrade kglite to load this file.",
            file_core_version, CURRENT_CORE_DATA_VERSION,
        )));
    }

    // Decompress
    let gz = GzDecoder::new(graph_bytes);
    let graph: crate::graph::schema::Graph = bincode::deserialize_from(gz).map_err(|e| {
        io::Error::other(format!(
            "Failed to deserialize graph data (core version {}). Error: {}",
            file_core_version, e
        ))
    })?;

    // Run migration chain: version N → N+1 → ... → CURRENT_CORE_DATA_VERSION
    if file_core_version < CURRENT_CORE_DATA_VERSION {
        migrate_core_data(graph, file_core_version)
    } else {
        Ok(graph)
    }
}

/// Migrate core graph data from an older version to the current version.
/// Called sequentially: v1→v2, v2→v3, etc.
///
/// Currently a no-op since CURRENT_CORE_DATA_VERSION is 1.
/// When NodeData/EdgeData/Value changes in the future, add migration steps here:
///
/// ```ignore
/// fn migrate_core_data(graph: Graph, from: u32) -> io::Result<Graph> {
///     let mut g = graph;
///     if from < 2 { g = migrate_v1_to_v2(g)?; }
///     if from < 3 { g = migrate_v2_to_v3(g)?; }
///     Ok(g)
/// }
/// ```
fn migrate_core_data(
    graph: crate::graph::schema::Graph,
    from_version: u32,
) -> io::Result<crate::graph::schema::Graph> {
    // No migrations needed yet (only version 1 exists).
    // This function exists as infrastructure for future core data changes.
    let _ = from_version;
    Ok(graph)
}

/// Rebuild caches and wrap in KnowledgeGraph.
fn finalize_load(mut dir_graph: DirGraph) -> KnowledgeGraph {
    dir_graph.rebuild_type_indices();
    dir_graph.build_connection_types_cache();
    dir_graph.rebuild_indices_from_keys();

    KnowledgeGraph {
        inner: Arc::new(dir_graph),
        selection: CowSelection::new(),
        reports: OperationReports::new(),
        last_mutation_stats: None,
    }
}
