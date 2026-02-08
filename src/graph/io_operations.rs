// src/graph/io_operations.rs
use crate::graph::reporting::OperationReports;
use crate::graph::schema::{CowSelection, DirGraph};
use crate::graph::KnowledgeGraph;
use bincode;
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::sync::Arc;

pub fn save_to_file(graph: &Arc<DirGraph>, path: &str) -> io::Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    // Dereference the Arc to get the DirGraph for serialization
    bincode::serialize_into(writer, &**graph).map_err(io::Error::other)?;

    Ok(())
}

pub fn load_file(path: &str) -> io::Result<KnowledgeGraph> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut dir_graph =
        bincode::deserialize_from::<_, DirGraph>(reader).map_err(io::Error::other)?;

    // Rebuild skipped caches after deserialization
    dir_graph.build_connection_types_cache();

    // Create a new KnowledgeGraph with a fresh reports object
    Ok(KnowledgeGraph {
        inner: Arc::new(dir_graph),
        selection: CowSelection::new(),
        reports: OperationReports::new(),
    })
}
