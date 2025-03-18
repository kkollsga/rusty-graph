// src/graph/io_operations.rs
use std::fs::File;
use std::io::{self, BufWriter, BufReader};
use std::sync::Arc;
use bincode;
use crate::graph::schema::{DirGraph, CurrentSelection};
use crate::graph::KnowledgeGraph;
use crate::graph::reporting::OperationReports;

pub fn save_to_file(graph: &Arc<DirGraph>, path: &str) -> io::Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    
    // Dereference the Arc to get the DirGraph for serialization
    bincode::serialize_into(writer, &**graph)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    
    Ok(())
}

pub fn load_file(path: &str) -> io::Result<KnowledgeGraph> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let dir_graph = bincode::deserialize_from::<_, DirGraph>(reader)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    
    // Create a new KnowledgeGraph with a fresh reports object
    Ok(KnowledgeGraph {
        inner: Arc::new(dir_graph),
        selection: CurrentSelection::new(),
        reports: OperationReports::new(), // Start with empty reports
    })
}