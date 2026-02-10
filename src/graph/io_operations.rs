// src/graph/io_operations.rs
use crate::graph::reporting::OperationReports;
use crate::graph::schema::{CowSelection, DirGraph, SaveMetadata};
use crate::graph::KnowledgeGraph;
use bincode;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read};
use std::sync::Arc;

/// Current format version. Bump when DirGraph layout changes in a breaking way.
pub const CURRENT_FORMAT_VERSION: u32 = 1;

pub fn save_to_file(graph: &mut Arc<DirGraph>, path: &str) -> io::Result<()> {
    let g = Arc::make_mut(graph);
    // Stamp save metadata before serializing
    g.save_metadata = SaveMetadata::current();
    // Snapshot index keys so they survive (indices themselves are #[serde(skip)])
    g.populate_index_keys();

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let gz = GzEncoder::new(writer, Compression::new(3));
    bincode::serialize_into(gz, &**graph).map_err(io::Error::other)?;

    Ok(())
}

pub fn load_file(path: &str) -> io::Result<KnowledgeGraph> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Try gzip first; if invalid header, fall back to raw bincode (old files)
    let mut dir_graph = {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;

        // Gzip magic bytes: 0x1f 0x8b
        if buf.len() >= 2 && buf[0] == 0x1f && buf[1] == 0x8b {
            let gz = GzDecoder::new(&buf[..]);
            bincode::deserialize_from::<_, DirGraph>(gz).map_err(io::Error::other)?
        } else {
            bincode::deserialize_from::<_, DirGraph>(&buf[..]).map_err(io::Error::other)?
        }
    };

    // Check format version
    let saved_version = dir_graph.save_metadata.format_version;
    if saved_version > CURRENT_FORMAT_VERSION {
        return Err(io::Error::other(format!(
            "File was saved with format version {} (library {}), but this library only supports up to version {}. \
             Upgrade rusty-graph to load this file.",
            saved_version,
            dir_graph.save_metadata.library_version,
            CURRENT_FORMAT_VERSION,
        )));
    }

    // Rebuild skipped caches after deserialization
    dir_graph.rebuild_type_indices();
    dir_graph.build_connection_types_cache();
    // Rebuild property/composite indices from persisted keys
    dir_graph.rebuild_indices_from_keys();

    // Create a new KnowledgeGraph with a fresh reports object
    Ok(KnowledgeGraph {
        inner: Arc::new(dir_graph),
        selection: CowSelection::new(),
        reports: OperationReports::new(),
        last_mutation_stats: None,
    })
}
