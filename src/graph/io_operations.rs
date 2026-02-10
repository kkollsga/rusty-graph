// src/graph/io_operations.rs
use crate::graph::reporting::OperationReports;
use crate::graph::schema::{CowSelection, DirGraph, SaveMetadata};
use crate::graph::KnowledgeGraph;
use bincode;
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::sync::Arc;

/// Current format version. Bump when DirGraph layout changes in a breaking way.
pub const CURRENT_FORMAT_VERSION: u32 = 1;

pub fn save_to_file(graph: &mut Arc<DirGraph>, path: &str) -> io::Result<()> {
    // Stamp save metadata before serializing
    Arc::make_mut(graph).save_metadata = SaveMetadata::current();

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &**graph).map_err(io::Error::other)?;

    Ok(())
}

pub fn load_file(path: &str) -> io::Result<KnowledgeGraph> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut dir_graph =
        bincode::deserialize_from::<_, DirGraph>(reader).map_err(io::Error::other)?;

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
    dir_graph.build_connection_types_cache();

    // Create a new KnowledgeGraph with a fresh reports object
    Ok(KnowledgeGraph {
        inner: Arc::new(dir_graph),
        selection: CowSelection::new(),
        reports: OperationReports::new(),
        last_mutation_stats: None,
    })
}
