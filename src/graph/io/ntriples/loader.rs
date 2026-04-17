//! `load_ntriples` entry point + columnar build pipeline.
//!
//! Streaming single-pass load: parse → accumulate → flush → build columns.
//! Supports mem/mapped/disk storage modes; disk mode uses an overflow
//! edge buffer and mmap-backed column builders.

use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, InternedKey, NodeData, PropertyStorage, TypeSchema};
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use crate::graph::storage::type_build_meta::{ColType, TypeBuildMeta};
use crate::graph::storage::{GraphRead, GraphWrite};
use bzip2::read::BzDecoder;
use flate2::read::GzDecoder;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use super::parser::{
    extract_lang_text, language_matches, parse_line, parse_qcode_number, typed_literal_to_value,
    EdgeBuffer, EntityAccumulator, Object, Predicate, Subject,
};
use super::writer::{create_edges_from_buffer, create_edges_with_qnum_map};
use super::{NTriplesConfig, NTriplesStats};

pub fn load_ntriples(
    graph: &mut DirGraph,
    path: &str,
    config: &NTriplesConfig,
) -> Result<NTriplesStats, String> {
    let start = Instant::now();
    let path_obj = Path::new(path);

    // Open decompression reader
    let file = File::open(path_obj).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let reader: Box<dyn Read + Send> = if path.ends_with(".bz2") {
        Box::new(BzDecoder::new(BufReader::new(file)))
    } else if path.ends_with(".gz") {
        Box::new(GzDecoder::new(BufReader::new(file)))
    } else if path.ends_with(".zst") || path.ends_with(".zstd") {
        Box::new(
            zstd::Decoder::new(BufReader::new(file))
                .map_err(|e| format!("zstd decoder error: {}", e))?,
        )
    } else {
        Box::new(file)
    };
    // Reader thread: decompresses + reads lines via channel (hides I/O latency)
    const BATCH_SIZE: usize = 50_000;
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<String>>(8);
    let reader_handle = std::thread::spawn(move || {
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, reader);
        let mut raw = Vec::with_capacity(512);
        let mut batch = Vec::with_capacity(BATCH_SIZE);
        let prefix: &[u8] = b"<http://www.wikidata.org/entity/Q";

        loop {
            raw.clear();
            if reader.read_until(b'\n', &mut raw).unwrap_or(0) == 0 {
                if !batch.is_empty() {
                    let _ = tx.send(batch);
                }
                break;
            }

            // Fast reject at byte level — skip non-entity lines without String allocation
            if !raw.starts_with(prefix) {
                continue;
            }

            // Only allocate String for entity lines
            let line = String::from_utf8_lossy(&raw).into_owned();
            batch.push(line);
            if batch.len() >= BATCH_SIZE {
                if tx.send(batch).is_err() {
                    break;
                }
                batch = Vec::with_capacity(BATCH_SIZE);
            }
        }
    });

    let mut stats = NTriplesStats {
        triples_scanned: 0,
        entities_created: 0,
        edges_created: 0,
        edges_skipped: 0,
        seconds: 0.0,
    };

    // Phase 1: Parse and ingest.
    // For Disk mode: serialize properties to a compressed log file (fast, ~100 ns/entity).
    // Phase 1b replays the log to build ColumnStores in bulk.
    // For other modes: use fast non-mapped insertion (HashMap properties, then Phase 1b).
    // final_mode was graph.storage_mode; now read from graph.graph variant
    let mapped = false;
    let mut current: Option<EntityAccumulator> = None;
    let use_compact = graph.graph.is_disk();

    // Property log for Disk mode: serialize properties during Phase 1, replay in Phase 1b
    let mut prop_log: Option<crate::graph::storage::memory::property_log::PropertyLogWriter> =
        if graph.graph.is_disk() {
            let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
                std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
            });
            // Clean up stale spill dirs from previous killed builds
            if let Some(parent) = spill_dir.parent() {
                if let Ok(entries) = std::fs::read_dir(parent) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        let name = name.to_string_lossy();
                        if name.starts_with("kglite_build_") && entry.path() != spill_dir {
                            let _ = std::fs::remove_dir_all(entry.path());
                        }
                    }
                }
            }
            // Clean up stale pending_edges from previous killed builds
            if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
                let stale = dg.data_dir.join("_pending_edges.bin");
                if stale.exists() {
                    let _ = std::fs::remove_file(&stale);
                }
            }
            let log_path = spill_dir.join("properties.log.zst");
            if config.verbose {
                eprintln!("  Property log: {}", log_path.display());
            }
            Some(
                crate::graph::storage::memory::property_log::PropertyLogWriter::new(&log_path, 1)
                    .map_err(|e| format!("Failed to create property log: {}", e))?,
            )
        } else {
            None
        };
    let mut edge_buffer = if use_compact {
        if graph.graph.is_disk() {
            // File-backed edge buffer: avoids holding ~14 GB in RAM during Phase 1b
            let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
                std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
            });
            std::fs::create_dir_all(&spill_dir)
                .map_err(|e| format!("Failed to create spill dir: {}", e))?;
            let edge_path = spill_dir.join("edges.bin");
            EdgeBuffer::Compact(
                MmapOrVec::mapped(&edge_path, 1 << 20)
                    .map_err(|e| format!("Failed to create edge buffer file: {}", e))?,
            )
        } else {
            EdgeBuffer::Compact(MmapOrVec::new())
        }
    } else {
        EdgeBuffer::Strings(Vec::new())
    };

    // Label cache for auto-typing: qnum → label, built during Phase 1.
    let mut label_cache: HashMap<u32, String> = HashMap::new();

    // Per-type build metadata for Phase 1b pre-allocation.
    let mut type_meta: HashMap<String, TypeBuildMeta> = HashMap::new();

    // For disk mode: build qnum_to_idx during Phase 1 (instead of from id_indices in Phase 2).
    // This lets us skip id_indices entirely, saving ~11 GB at full Wikidata scale.
    // Pre-allocate for 150M Q-numbers (~600 MB mmap, lazily paged).
    let mut qnum_to_idx: Option<MmapOrVec<u32>> = if graph.graph.is_disk() {
        let spill_dir = graph.spill_dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
        });
        let _ = std::fs::create_dir_all(&spill_dir);
        // 150M covers all Wikidata Q-numbers. OS allocates pages lazily.
        Some(
            MmapOrVec::mapped_prefilled(&spill_dir.join("qnum_to_idx.bin"), 150_000_000)
                .unwrap_or_else(|_| MmapOrVec::from_vec(vec![0u32; 150_000_000])),
        )
    } else {
        None
    };

    let mut entity_limit_reached = false;
    let mut progress_countdown = 5_000_000u64;
    let include_labels = true;
    const ENTITY_PREFIX: &[u8] = b"<http://www.wikidata.org/entity/Q";

    'outer: for batch in &rx {
        for line in &batch {
            if entity_limit_reached && current.is_none() {
                break 'outer;
            }

            stats.triples_scanned += 1;

            if config.verbose {
                progress_countdown -= 1;
                if progress_countdown == 0 {
                    progress_countdown = 5_000_000;
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = stats.triples_scanned as f64 / elapsed;
                    let buf_len = edge_buffer.len() as u64;
                    eprintln!(
                    "  {:>12} triples | {:>8} entities | {:>8} edges buffered | {:.0} triples/s",
                    format_count(stats.triples_scanned),
                    format_count(stats.entities_created),
                    format_count(buf_len),
                    rate,
                );
                }
            }

            // Fast reject: skip non-entity lines before parsing (~60-80% of all lines)
            if !line.as_bytes().starts_with(ENTITY_PREFIX) {
                continue;
            }

            let (subject, predicate, object) = match parse_line(line) {
                Some(parsed) => parsed,
                None => continue,
            };

            let subj_id = match subject {
                Subject::Entity(id) => id,
                Subject::Other => continue,
            };

            // Subject changed → flush previous entity
            if current.as_ref().is_some_and(|c| c.id != subj_id) {
                if let Some(acc) = current.take() {
                    flush_entity(
                        graph,
                        acc,
                        config,
                        &mut edge_buffer,
                        &mut stats,
                        mapped,
                        &mut prop_log,
                        &mut label_cache,
                        &mut type_meta,
                        &mut qnum_to_idx,
                    );
                }
                if entity_limit_reached {
                    break;
                }
            }

            if current.is_none() {
                if let Some(max) = config.max_entities {
                    if stats.entities_created >= max as u64 {
                        entity_limit_reached = true;
                        continue;
                    }
                }
                current = Some(EntityAccumulator::new(subj_id.to_string()));
            }

            let acc = current.as_mut().unwrap();

            // Process triple based on predicate type
            match predicate {
                Predicate::Label => {
                    if include_labels {
                        if let Some(text) = extract_lang_text(&object, &config.languages) {
                            acc.label = Some(text);
                        }
                    }
                }
                Predicate::Description => {
                    if let Some(text) = extract_lang_text(&object, &config.languages) {
                        acc.description = Some(text);
                    }
                }
                Predicate::AltLabel => {}
                Predicate::Type => {}
                Predicate::WikidataDirect(pcode) => {
                    if let Some(ref allowed) = config.predicates {
                        if !allowed.contains(pcode) {
                            continue;
                        }
                    }

                    let pred_label = config
                        .predicate_labels
                        .get(pcode)
                        .cloned()
                        .unwrap_or_else(|| pcode.to_string());

                    match &object {
                        Object::Entity(target_qcode) => {
                            if pcode == "P31" && acc.type_qcode.is_none() {
                                acc.type_qcode = Some(target_qcode.to_string());
                            }
                            acc.outgoing_edges
                                .push((pred_label, target_qcode.to_string()));
                        }
                        Object::Literal(text) => {
                            acc.properties
                                .insert(pred_label, Value::String(text.clone()));
                        }
                        Object::LangLiteral(text, lang) => {
                            if language_matches(lang, &config.languages) {
                                acc.properties
                                    .insert(pred_label, Value::String(text.clone()));
                            }
                        }
                        Object::TypedLiteral(text, type_uri) => {
                            acc.properties
                                .insert(pred_label, typed_literal_to_value(text, type_uri));
                        }
                        Object::Other => {}
                    }
                }
                Predicate::Other => {}
            }
        } // end for line in batch
    } // end for batch in rx
    let _ = reader_handle.join();

    // Flush last entity
    if let Some(acc) = current.take() {
        flush_entity(
            graph,
            acc,
            config,
            &mut edge_buffer,
            &mut stats,
            mapped,
            &mut prop_log,
            &mut label_cache,
            &mut type_meta,
            &mut qnum_to_idx,
        );
    }

    // Post-Phase-1: resolve Q-code type names using the now-complete label cache.
    // During streaming, types are assigned as entities arrive — if the P31 target
    // entity hasn't been seen yet, the raw Q-code is used. Now that all labels are
    // cached, we can fix these. This renames type_indices keys, node_type_metadata,
    // type_schemas, and updates node_type InternedKeys in the graph.
    // Mapping from old Q-code type names to new label names (populated by merge below).
    // Passed to Phase 1b so property log entries with old names find the right column writer.
    let mut type_rename_map: HashMap<String, String> = HashMap::new();

    if config.auto_type {
        let mut renames: Vec<(String, String)> = Vec::new();
        for type_name in graph.type_indices.keys() {
            if let Some(qnum) = parse_qcode_number(type_name) {
                if let Some(label) = label_cache.get(&qnum) {
                    if label != type_name {
                        renames.push((type_name.clone(), label.clone()));
                    }
                }
            }
        }
        if !renames.is_empty() {
            let rename_start = std::time::Instant::now();
            if config.verbose {
                eprintln!(
                    "  Resolving {} Q-code type names to labels...",
                    renames.len()
                );
            }
            let old_key_to_new_key: Vec<(InternedKey, InternedKey)> = renames
                .iter()
                .map(|(old, new)| {
                    let old_key = graph.interner.get_or_intern(old);
                    let new_key = graph.interner.get_or_intern(new);
                    (old_key, new_key)
                })
                .collect();
            for (old_name, new_name) in &renames {
                // Merge type_indices: if target name already exists, append indices
                if let Some(indices) = graph.type_indices.remove(old_name) {
                    graph
                        .type_indices
                        .entry(new_name.clone())
                        .or_default()
                        .extend(indices);
                }
                // Merge node_type_metadata: keep the richer entry (more property keys)
                if let Some(old_meta) = graph.node_type_metadata.remove(old_name) {
                    let entry = graph
                        .node_type_metadata
                        .entry(new_name.clone())
                        .or_default();
                    for (k, v) in old_meta {
                        entry.entry(k).or_insert(v);
                    }
                }
                // Merge type_schemas: union property keys
                if let Some(old_schema) = graph.type_schemas.remove(old_name) {
                    if let Some(existing) = graph.type_schemas.get(new_name) {
                        let merged = existing.merge(&old_schema);
                        graph
                            .type_schemas
                            .insert(new_name.clone(), Arc::new(merged));
                    } else {
                        graph.type_schemas.insert(new_name.clone(), old_schema);
                    }
                }
                // Merge type_build_meta: combine row counts and column info
                if let Some(old_build) = type_meta.remove(old_name) {
                    let entry = type_meta
                        .entry(new_name.clone())
                        .or_insert_with(TypeBuildMeta::new);
                    entry.merge_from(&old_build);
                }
            }
            // Build type_rename_map for Phase 1b (property log entries use old names)
            for (old_name, new_name) in &renames {
                type_rename_map.insert(old_name.clone(), new_name.clone());
            }

            // Update node_type InternedKey on affected nodes.
            // Build lookup map first, then ONE sequential pass over all nodes.
            // This is O(nodes) not O(renames × nodes).
            let rename_map: HashMap<u64, u64> = old_key_to_new_key
                .iter()
                .map(|(old, new)| (old.as_u64(), new.as_u64()))
                .collect();
            match &mut graph.graph {
                crate::graph::schema::GraphBackend::Disk(ref mut dg) => {
                    let n = dg.node_slots.len();
                    for i in 0..n {
                        let slot = dg.node_slots.get(i);
                        if slot.is_alive() {
                            if let Some(&new_type) = rename_map.get(&slot.node_type) {
                                let mut new_slot = slot;
                                new_slot.node_type = new_type;
                                dg.node_slots.set(i, new_slot);
                            }
                        }
                    }
                }
                crate::graph::schema::GraphBackend::Memory(ref mut g) => {
                    for i in 0..g.node_bound() {
                        let idx = petgraph::graph::NodeIndex::new(i);
                        if let Some(node) = g.node_weight_mut(idx) {
                            if let Some(&new_key) = rename_map.get(&node.node_type.as_u64()) {
                                node.node_type = InternedKey::from_u64(new_key);
                            }
                        }
                    }
                }
                crate::graph::schema::GraphBackend::Mapped(ref mut g) => {
                    for i in 0..g.node_bound() {
                        let idx = petgraph::graph::NodeIndex::new(i);
                        if let Some(node) = g.node_weight_mut(idx) {
                            if let Some(&new_key) = rename_map.get(&node.node_type.as_u64()) {
                                node.node_type = InternedKey::from_u64(new_key);
                            }
                        }
                    }
                }
                // RecordingGraph is a Phase 6 validation wrapper only
                // constructed in Rust tests; the ntriples loader never
                // sees it in practice.
                crate::graph::schema::GraphBackend::Recording(_) => {
                    unreachable!("ntriples loader does not run on a Recording-wrapped graph");
                }
            }
            if config.verbose {
                eprintln!(
                    "  [T+{:.0}s] Resolved {} Q-code types ({:.1}s)",
                    start.elapsed().as_secs_f64(),
                    renames.len(),
                    rename_start.elapsed().as_secs_f64()
                );
            }
        }
    }

    let label_cache_size = label_cache.len();
    drop(label_cache);

    if config.verbose {
        let t = start.elapsed().as_secs_f64();
        let buf_len = edge_buffer.len() as u64;
        let num_types = type_meta.len();
        let total_cols: usize = type_meta.values().map(|m| m.columns.len()).sum();
        eprintln!(
            "  [T+{:.0}s] Phase 1 done: {} entities, {} edges buffered, {} types, {} columns",
            t,
            format_count(stats.entities_created),
            format_count(buf_len),
            format_count(num_types as u64),
            format_count(total_cols as u64),
        );
        eprintln!(
            "  [T+{:.0}s] Label cache: {} entries (freed)",
            t,
            format_count(label_cache_size as u64),
        );
    }

    // Phase 1b: Convert to columnar storage.
    // For Disk mode: pre-allocate columns from metadata, then direct-write from log.
    // For Mapped mode: bulk convert from HashMap properties.
    if let Some(log_writer) = prop_log.take() {
        if config.verbose {
            eprintln!(
                "  [T+{:.0}s] Phase 1b: building columns with direct-write ({} entities, {} types)...",
                start.elapsed().as_secs_f64(),
                format_count(log_writer.count()),
                type_meta.len(),
            );
        }
        let conv_start = Instant::now();
        let log_path = log_writer
            .finish()
            .map_err(|e| format!("Failed to finish property log: {}", e))?;
        build_columns_direct(
            graph,
            &log_path,
            &type_meta,
            &type_rename_map,
            config.verbose,
        )
        .map_err(|e| format!("Failed to build columns: {}", e))?;
        let _ = std::fs::remove_file(&log_path);
        if let Some(parent) = log_path.parent() {
            let _ = std::fs::remove_dir(parent);
        }
        if config.verbose {
            eprintln!(
                "  Phase 1b done ({:.1}s)",
                conv_start.elapsed().as_secs_f64()
            );
        }
    } else if graph.graph.is_mapped() {
        if config.verbose {
            eprintln!("  Phase 1b: converting to columnar storage...");
        }
        let conv_start = Instant::now();
        graph.enable_columnar();
        if config.verbose {
            eprintln!(
                "  Phase 1b done ({:.1}s)",
                conv_start.elapsed().as_secs_f64()
            );
        }
    }

    // Free everything not needed for Phase 2+3 to maximize page cache.
    // Phase 2 only needs qnum_to_idx + edge_buffer. Phase 3 only needs pending_edges.
    if graph.graph.is_disk() {
        let dropped_stores = graph.column_stores.len();
        graph.column_stores.clear();
        graph.sync_disk_column_stores();
        drop(type_meta);
        // type_indices: 1 GB — not needed for Phase 2/3. Rebuild from node_slots after.
        let type_indices_count = graph.type_indices.len();
        graph.type_indices.clear();
        if config.verbose {
            eprintln!(
                "  [T+{:.0}s] Freed {} column stores + {} type indices before Phase 2",
                start.elapsed().as_secs_f64(),
                dropped_stores,
                type_indices_count,
            );
        }
    }

    if config.verbose {
        eprintln!(
            "  [T+{:.0}s] Phase 2: creating edges...",
            start.elapsed().as_secs_f64()
        );
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }

    // Phase 2: Create edges from buffer
    let edge_start = Instant::now();
    if let Some(ref qt) = qnum_to_idx {
        // Fast path: use pre-built qnum_to_idx from Phase 1 (disk mode)
        create_edges_with_qnum_map(graph, &edge_buffer, &mut stats, qt);
    } else {
        create_edges_from_buffer(graph, &edge_buffer, &mut stats);
    }

    if config.verbose {
        eprintln!(
            "  [T+{:.0}s] Phase 2 done: {} edges created, {} skipped ({:.1}s)",
            start.elapsed().as_secs_f64(),
            format_count(stats.edges_created),
            format_count(stats.edges_skipped),
            edge_start.elapsed().as_secs_f64(),
        );
    }

    // Free qnum_to_idx after Phase 2
    if let Some(qt) = qnum_to_idx.take() {
        let qt_path = qt.file_path().map(|p| p.to_path_buf());
        drop(qt);
        if let Some(path) = qt_path {
            let _ = std::fs::remove_file(path);
        }
    }

    // Free edge_buffer before Phase 3.
    let edge_file_path = match &edge_buffer {
        EdgeBuffer::Compact(buf) => buf.file_path().map(|p| p.to_path_buf()),
        _ => None,
    };
    drop(edge_buffer);
    if let Some(path) = edge_file_path {
        let _ = std::fs::remove_file(&path);
    }

    // Phase 3: Build CSR from pending edges (disk mode)
    if let crate::graph::schema::GraphBackend::Disk(ref mut dg) = graph.graph {
        if config.verbose {
            eprintln!(
                "  [T+{:.0}s] Phase 3: building CSR from pending edges...",
                start.elapsed().as_secs_f64()
            );
        }
        let csr_start = Instant::now();
        dg.build_csr_from_pending();
        if config.verbose {
            eprintln!(
                "  [T+{:.0}s] Phase 3 done ({:.1}s)",
                start.elapsed().as_secs_f64(),
                csr_start.elapsed().as_secs_f64()
            );
        }
    }

    // Warm edge_type_counts_cache from CSR build data (avoids 14 GB rescan on first query)
    if graph.graph.is_disk() {
        if let crate::graph::schema::GraphBackend::Disk(ref mut dg) = graph.graph {
            if let Some(raw_counts) = dg.edge_type_counts_raw.take() {
                let string_counts: HashMap<String, usize> = raw_counts
                    .into_iter()
                    .map(|(key_u64, count)| {
                        let key = InternedKey::from_u64(key_u64);
                        let name = graph.interner.resolve(key).to_string();
                        (name, count)
                    })
                    .collect();
                if config.verbose {
                    eprintln!(
                        "  [T+{:.0}s] Cached {} edge type counts from CSR build",
                        start.elapsed().as_secs_f64(),
                        string_counts.len()
                    );
                }
                *graph.edge_type_counts_cache.write().unwrap() = Some(string_counts);
            }
        }
    }

    // Rebuild type_indices from DiskNodeSlots (dropped before Phase 2 to save 1 GB)
    if graph.graph.is_disk() {
        if config.verbose {
            eprintln!(
                "  [T+{:.0}s] Rebuilding type indices from node slots...",
                start.elapsed().as_secs_f64()
            );
        }
        let rebuild_start = Instant::now();
        if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
            for i in 0..dg.node_slots.len() {
                let slot = dg.node_slots.get(i);
                if slot.is_alive() {
                    let type_key = InternedKey::from_u64(slot.node_type);
                    let type_name = graph.interner.resolve(type_key).to_string();
                    graph
                        .type_indices
                        .entry(type_name)
                        .or_default()
                        .push(petgraph::graph::NodeIndex::new(i));
                }
            }
        }
        if config.verbose {
            eprintln!(
                "  [T+{:.0}s] Rebuilt {} type indices ({:.1}s)",
                start.elapsed().as_secs_f64(),
                graph.type_indices.len(),
                rebuild_start.elapsed().as_secs_f64(),
            );
        }
    }

    // Reload column stores by re-opening the mmap file + reading saved metadata.
    if graph.graph.is_disk() {
        let data_dir = if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
            dg.data_dir.clone()
        } else {
            std::path::PathBuf::new()
        };
        let mmap_path = data_dir.join("columns.bin");
        let meta_path = data_dir.join("columns_meta.json");
        if mmap_path.exists() && meta_path.exists() {
            if config.verbose {
                eprintln!(
                    "  [T+{:.0}s] Reloading column stores from mmap...",
                    start.elapsed().as_secs_f64()
                );
            }
            let reload_start = Instant::now();
            let reload_result: Result<(), String> = (|| {
                let meta_json = std::fs::read_to_string(&meta_path)
                    .map_err(|e| format!("read columns_meta.json: {}", e))?;
                let columns_meta: Vec<ColumnTypeMeta> = serde_json::from_str(&meta_json)
                    .map_err(|e| format!("parse columns_meta.json: {}", e))?;

                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&mmap_path)
                    .map_err(|e| format!("open columns.bin: {}", e))?;
                // SAFETY: columns.bin was just written by a prior save pass;
                // we re-open + mmap the existing file read-write. No other
                // handle is writing to it concurrently.
                let mmap = unsafe {
                    memmap2::MmapMut::map_mut(&file)
                        .map_err(|e| format!("mmap columns.bin: {}", e))?
                };
                let mmap_arc = Arc::new(mmap);

                for type_meta in &columns_meta {
                    let mmap_store = type_meta.to_mmap_store(Arc::clone(&mmap_arc));
                    let store =
                        crate::graph::storage::memory::column_store::ColumnStore::from_mmap_store(
                            Arc::new(mmap_store),
                        );
                    graph
                        .column_stores
                        .insert(type_meta.type_name.clone(), Arc::new(store));
                }
                Ok(())
            })();

            if let Err(e) = reload_result {
                if config.verbose {
                    eprintln!("  Warning: failed to reload column stores: {}", e);
                }
            }

            if config.verbose {
                eprintln!(
                    "  Reloaded {} column stores from mmap ({:.1}s)",
                    graph.column_stores.len(),
                    reload_start.elapsed().as_secs_f64(),
                );
            }
        }
        graph.sync_disk_column_stores();
    }

    // Build id_indices for all types so WHERE id(n) = X is O(1).
    // Uses column stores directly — no node materialization, no arena growth.
    if graph.graph.is_disk() {
        if config.verbose {
            eprintln!(
                "  [T+{:.0}s] Building id indices from column stores...",
                start.elapsed().as_secs_f64()
            );
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }
        let id_start = Instant::now();
        let type_names: Vec<String> = graph.type_indices.keys().cloned().collect();
        for type_name in &type_names {
            graph.build_id_index_from_columns(type_name);
        }
        if config.verbose {
            eprintln!(
                "  Built {} id indices ({:.1}s)",
                type_names.len(),
                id_start.elapsed().as_secs_f64(),
            );
        }
    }

    // Save interner + metadata to disk so load() works.
    if graph.graph.is_disk() {
        if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
            let data_dir = dg.data_dir.clone();

            // Build type_connectivity_cache from connection_type_metadata + edge_type_counts.
            // This makes describe(types=['human']) instant instead of scanning 10K nodes.
            {
                let mut triples = Vec::new();
                for (conn_type, info) in &graph.connection_type_metadata {
                    let edge_count = graph
                        .edge_type_counts_cache
                        .read()
                        .unwrap()
                        .as_ref()
                        .and_then(|counts| counts.get(conn_type).copied())
                        .unwrap_or(0);
                    for src in &info.source_types {
                        for tgt in &info.target_types {
                            triples.push(crate::graph::schema::ConnectivityTriple {
                                src: src.clone(),
                                conn: conn_type.clone(),
                                tgt: tgt.clone(),
                                count: edge_count,
                            });
                        }
                    }
                }
                if !triples.is_empty() {
                    *graph.type_connectivity_cache.write().unwrap() = Some(triples);
                    if config.verbose {
                        eprintln!(
                            "  [T+{:.0}s] Built type connectivity cache ({} triples)",
                            start.elapsed().as_secs_f64(),
                            graph
                                .type_connectivity_cache
                                .read()
                                .unwrap()
                                .as_ref()
                                .map(|t| t.len())
                                .unwrap_or(0),
                        );
                    }
                }
            }

            if config.verbose {
                eprintln!(
                    "  [T+{:.0}s] Saving interner + metadata...",
                    start.elapsed().as_secs_f64()
                );
            }

            // Save interner
            let save_step = Instant::now();
            let interner_map: HashMap<String, String> = graph
                .interner
                .iter()
                .map(|(k, v)| (k.as_u64().to_string(), v.to_string()))
                .collect();
            if let Ok(json) = serde_json::to_string(&interner_map) {
                let _ = std::fs::write(data_dir.join("interner.json"), json);
            }
            if config.verbose {
                eprintln!(
                    "    interner.json ({} entries): {:.1}s",
                    interner_map.len(),
                    save_step.elapsed().as_secs_f64()
                );
            }

            // Save DirGraph metadata
            let save_step = Instant::now();
            let meta = crate::graph::io::io_operations::build_disk_metadata(graph);
            if let Ok(json) = serde_json::to_string_pretty(&meta) {
                let _ = std::fs::write(data_dir.join("metadata.json"), json);
            }
            if config.verbose {
                eprintln!(
                    "    metadata.json: {:.1}s",
                    save_step.elapsed().as_secs_f64()
                );
            }

            // Save id_indices (bincode + zstd) so load() doesn't rebuild them
            let save_step = Instant::now();
            if !graph.id_indices.is_empty() {
                if let Ok(bytes) = bincode::serialize(&graph.id_indices) {
                    if let Ok(compressed) = zstd::encode_all(bytes.as_slice(), 3) {
                        let _ = std::fs::write(data_dir.join("id_indices.bin.zst"), compressed);
                    }
                }
            }
            if config.verbose {
                eprintln!(
                    "    id_indices.bin.zst ({} types): {:.1}s",
                    graph.id_indices.len(),
                    save_step.elapsed().as_secs_f64()
                );
            }

            // Save type_indices (bincode + zstd) so load() doesn't rebuild from node_slots
            let save_step = Instant::now();
            if !graph.type_indices.is_empty() {
                if let Ok(bytes) = bincode::serialize(&graph.type_indices) {
                    if let Ok(compressed) = zstd::encode_all(bytes.as_slice(), 3) {
                        let _ = std::fs::write(data_dir.join("type_indices.bin.zst"), compressed);
                    }
                }
            }
            if config.verbose {
                eprintln!(
                    "    type_indices.bin.zst ({} types): {:.1}s",
                    graph.type_indices.len(),
                    save_step.elapsed().as_secs_f64()
                );
            }

            if config.verbose {
                eprintln!("  [T+{:.0}s] Save complete", start.elapsed().as_secs_f64());
            }
        }
    }

    stats.seconds = start.elapsed().as_secs_f64();
    Ok(stats)
}

// ─── Column metadata for mmap reload ────────────────────────────────────────

/// Serializable metadata for a single Region in the mmap file.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy)]
pub struct RegionMeta {
    pub offset: usize,
    pub len: usize,
}

/// Serializable metadata for a fixed-width column.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct FixedColMeta {
    pub col_type_str: String,
    pub data: RegionMeta,
    pub nulls: RegionMeta,
}

/// Serializable metadata for a string column.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct StrColMeta {
    pub data: RegionMeta,
    pub offsets: RegionMeta,
    pub nulls: RegionMeta,
}

/// Serializable metadata for a column mapping entry.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct ColMapEntry {
    pub key_u64: u64,
    pub col_type_str: String,
    pub idx: usize,
}

/// Parse a ColType from its tag string.
fn col_type_from_str(s: &str) -> ColType {
    match s {
        "int64" => ColType::Int64,
        "float64" => ColType::Float64,
        "uniqueid" => ColType::UniqueId,
        "bool" => ColType::Bool,
        "date" => ColType::Date,
        "string" => ColType::Str,
        _ => ColType::Str,
    }
}

/// Per-type metadata saved to columns_meta.json for post-Phase-3 mmap reload.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct ColumnTypeMeta {
    pub type_name: String,
    pub row_count: u32,
    pub id_is_string: bool,
    pub id_data: RegionMeta,
    pub id_nulls: RegionMeta,
    pub id_str_data: RegionMeta,
    pub id_str_offsets: RegionMeta,
    pub title_data: RegionMeta,
    pub title_offsets: RegionMeta,
    pub title_nulls: RegionMeta,
    pub col_map: Vec<ColMapEntry>,
    pub fixed_cols: Vec<FixedColMeta>,
    pub str_cols: Vec<StrColMeta>,
    pub overflow_offsets: RegionMeta,
    pub overflow_data: RegionMeta,
    pub has_overflow: bool,
}

impl RegionMeta {
    fn from_region(r: &crate::graph::storage::mapped::mmap_column_store::Region) -> Self {
        RegionMeta {
            offset: r.offset,
            len: r.len,
        }
    }

    fn to_region(self) -> crate::graph::storage::mapped::mmap_column_store::Region {
        crate::graph::storage::mapped::mmap_column_store::Region {
            offset: self.offset,
            len: self.len,
        }
    }
}

impl ColumnTypeMeta {
    /// Rebuild an MmapColumnStore from saved metadata + a shared mmap.
    pub fn to_mmap_store(
        &self,
        mmap: Arc<memmap2::MmapMut>,
    ) -> crate::graph::storage::mapped::mmap_column_store::MmapColumnStore {
        use crate::graph::storage::mapped::mmap_column_store::{
            ColRef, FixedColumnMeta, MmapColumnStore, StrColumnMeta,
        };

        MmapColumnStore {
            mmap,
            row_count: self.row_count,
            id_is_string: self.id_is_string,
            id_fixed: if !self.id_is_string {
                Some(FixedColumnMeta {
                    col_type: ColType::UniqueId,
                    data: self.id_data.to_region(),
                    nulls: self.id_nulls.to_region(),
                })
            } else {
                None
            },
            id_str: if self.id_is_string {
                Some(StrColumnMeta {
                    data: self.id_str_data.to_region(),
                    offsets: self.id_str_offsets.to_region(),
                    nulls: self.id_nulls.to_region(),
                })
            } else {
                None
            },
            title: StrColumnMeta {
                data: self.title_data.to_region(),
                offsets: self.title_offsets.to_region(),
                nulls: self.title_nulls.to_region(),
            },
            col_map: self
                .col_map
                .iter()
                .map(|e| {
                    let key = InternedKey::from_u64(e.key_u64);
                    let ct = col_type_from_str(&e.col_type_str);
                    let cr = if matches!(ct, ColType::Str) {
                        ColRef::Str(e.idx)
                    } else {
                        ColRef::Fixed(e.idx)
                    };
                    (key, cr)
                })
                .collect(),
            fixed_cols: self
                .fixed_cols
                .iter()
                .map(|fc| FixedColumnMeta {
                    col_type: col_type_from_str(&fc.col_type_str),
                    data: fc.data.to_region(),
                    nulls: fc.nulls.to_region(),
                })
                .collect(),
            str_cols: self
                .str_cols
                .iter()
                .map(|sc| StrColumnMeta {
                    data: sc.data.to_region(),
                    offsets: sc.offsets.to_region(),
                    nulls: sc.nulls.to_region(),
                })
                .collect(),
            overflow_offsets: self.overflow_offsets.to_region(),
            overflow_data: self.overflow_data.to_region(),
            has_overflow: self.has_overflow,
        }
    }
}

/// Build ColumnStores by replaying the property log with pre-allocated direct-write.
///
/// Uses TypeBuildMeta (collected during Phase 1) to pre-allocate exact-sized arrays,
/// then writes values directly to their final positions in a single pass.
/// No BlockPool, no eviction, no intermediate compression.
fn build_columns_direct(
    graph: &mut DirGraph,
    log_path: &std::path::Path,
    type_meta: &HashMap<String, TypeBuildMeta>,
    type_rename_map: &HashMap<String, String>,
    verbose: bool,
) -> std::io::Result<()> {
    use crate::graph::storage::mapped::mmap_column_store::{
        ColRef, FixedColumnMeta, MmapColumnStore, Region, StrColumnMeta,
    };
    use crate::graph::storage::memory::column_store::ColumnStore;
    use crate::graph::storage::memory::property_log::PropertyLogReader;
    use memmap2::MmapMut;

    let alloc_start = Instant::now();

    // Get data_dir for placing the final columns.bin
    let data_dir = if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
        dg.data_dir.clone()
    } else {
        std::path::PathBuf::from("/tmp/kglite_columns")
    };
    let _ = std::fs::create_dir_all(&data_dir);

    // ── Step 1: Layout computation — single mmap file ────────────────────
    //
    // ONE file (`columns.bin`) in data_dir holds ALL column data for ALL types.
    //
    // Each region is 8-byte aligned so typed reads are safe.

    struct TypeWriter {
        row_cursor: u32,
        id_is_string: bool,
        id_data: Region,
        id_nulls: Region,
        id_str_data: Region,
        id_str_offsets: Region,
        title_data: Region,
        title_offsets: Region,
        title_nulls: Region,
        title_cursor: u64,
        col_map: HashMap<InternedKey, (ColType, usize)>,
        fixed_cols: Vec<FixedColLayout>,
        str_cols: Vec<StrColLayout>,
        /// Keys with fill rate < threshold go to the overflow bag instead of dense columns.
        overflow_keys: HashSet<InternedKey>,
        /// Grows during write pass — serialized overflow entries for all rows.
        overflow_bag_data: Vec<u8>,
        /// One offset per row (+ final sentinel), recording the start of each row's blob.
        overflow_offsets: Vec<u64>,
        /// Region in the mmap for overflow offsets (set after appending to file).
        overflow_offsets_region: Region,
        /// Region in the mmap for overflow bag data (set after appending to file).
        overflow_data_region: Region,
    }

    struct FixedColLayout {
        col_type: ColType,
        data: Region,
        nulls: Region,
    }

    struct StrColLayout {
        data: Region,
        offsets: Region,
        nulls: Region,
        cursor: u64,
    }

    // Helpers to write typed values into the mmap
    #[inline]
    fn write_i64(mmap: &mut MmapMut, region: &Region, row: usize, val: i64) {
        let off = region.offset + row * 8;
        mmap[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    }
    #[inline]
    fn write_f64(mmap: &mut MmapMut, region: &Region, row: usize, val: f64) {
        let off = region.offset + row * 8;
        mmap[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    }
    #[inline]
    fn write_u32(mmap: &mut MmapMut, region: &Region, row: usize, val: u32) {
        let off = region.offset + row * 4;
        mmap[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    }
    #[inline]
    fn write_i32(mmap: &mut MmapMut, region: &Region, row: usize, val: i32) {
        let off = region.offset + row * 4;
        mmap[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    }
    #[inline]
    fn write_u8(mmap: &mut MmapMut, region: &Region, row: usize, val: u8) {
        mmap[region.offset + row] = val;
    }
    #[inline]
    fn write_u64(mmap: &mut MmapMut, region: &Region, row: usize, val: u64) {
        let off = region.offset + row * 8;
        mmap[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    }
    #[inline]
    fn read_u64(mmap: &MmapMut, region: &Region, row: usize) -> u64 {
        let off = region.offset + row * 8;
        u64::from_ne_bytes(mmap[off..off + 8].try_into().unwrap())
    }

    /// Align a byte offset up to 8-byte boundary.
    #[inline]
    fn align8(v: usize) -> usize {
        (v + 7) & !7
    }

    let col_dir = data_dir.clone();

    let mut writers: HashMap<String, TypeWriter> = HashMap::new();

    // Accumulate regions that need null-fill (0xFF) so we can batch-fill after mmap creation.
    let mut null_regions: Vec<Region> = Vec::new();
    let mut cursor: usize = 0;

    for (type_name, meta) in type_meta {
        let rc = meta.row_count as usize;
        let id_is_string = meta.id_is_string;

        // Id column regions
        let id_data = if !id_is_string {
            cursor = align8(cursor);
            let r = Region {
                offset: cursor,
                len: rc * 4,
            };
            cursor += r.len;
            r
        } else {
            Region { offset: 0, len: 0 }
        };

        cursor = align8(cursor);
        let id_nulls = Region {
            offset: cursor,
            len: rc,
        };
        null_regions.push(id_nulls);
        cursor += id_nulls.len;

        let id_str_data = if id_is_string {
            cursor = align8(cursor);
            let r = Region {
                offset: cursor,
                len: meta.id_string_bytes as usize,
            };
            cursor += r.len;
            r
        } else {
            Region { offset: 0, len: 0 }
        };

        let id_str_offsets = if id_is_string {
            cursor = align8(cursor);
            let r = Region {
                offset: cursor,
                len: rc * 8,
            };
            cursor += r.len;
            r
        } else {
            Region { offset: 0, len: 0 }
        };

        // Title column regions
        cursor = align8(cursor);
        let title_data = Region {
            offset: cursor,
            len: meta.title_string_bytes as usize,
        };
        cursor += title_data.len;

        cursor = align8(cursor);
        let title_offsets = Region {
            offset: cursor,
            len: rc * 8,
        };
        cursor += title_offsets.len;

        cursor = align8(cursor);
        let title_nulls = Region {
            offset: cursor,
            len: rc,
        };
        null_regions.push(title_nulls);
        cursor += title_nulls.len;

        // Property columns — split into dense (>= 5% fill) and overflow (< 5% fill)
        const FILL_RATE_THRESHOLD: f64 = 0.05;
        let mut col_map: HashMap<InternedKey, (ColType, usize)> = HashMap::new();
        let mut fixed_cols: Vec<FixedColLayout> = Vec::new();
        let mut str_cols: Vec<StrColLayout> = Vec::new();
        let mut overflow_keys: HashSet<InternedKey> = HashSet::new();
        let mut dense_count = 0u32;
        let mut overflow_count = 0u32;

        for (key, col_meta) in &meta.columns {
            let fill_rate = meta.fill_rate(col_meta);
            if fill_rate < FILL_RATE_THRESHOLD {
                overflow_keys.insert(*key);
                overflow_count += 1;
                continue;
            }

            dense_count += 1;
            if let Some(val_size) = col_meta.col_type.value_size() {
                let idx = fixed_cols.len();
                col_map.insert(*key, (col_meta.col_type, idx));

                cursor = align8(cursor);
                let data = Region {
                    offset: cursor,
                    len: rc * val_size,
                };
                cursor += data.len;

                cursor = align8(cursor);
                let nulls = Region {
                    offset: cursor,
                    len: rc,
                };
                null_regions.push(nulls);
                cursor += nulls.len;

                fixed_cols.push(FixedColLayout {
                    col_type: col_meta.col_type,
                    data,
                    nulls,
                });
            } else {
                // String column
                let idx = str_cols.len();
                col_map.insert(*key, (col_meta.col_type, idx));

                cursor = align8(cursor);
                let data = Region {
                    offset: cursor,
                    len: col_meta.string_bytes as usize,
                };
                cursor += data.len;

                cursor = align8(cursor);
                let offsets = Region {
                    offset: cursor,
                    len: rc * 8,
                };
                cursor += offsets.len;

                cursor = align8(cursor);
                let nulls = Region {
                    offset: cursor,
                    len: rc,
                };
                null_regions.push(nulls);
                cursor += nulls.len;

                str_cols.push(StrColLayout {
                    data,
                    offsets,
                    nulls,
                    cursor: 0,
                });
            }
        }

        if verbose && overflow_count > 0 {
            eprintln!(
                "    {}: {} dense cols, {} overflow cols (< {:.0}% fill)",
                if type_name.len() > 40 {
                    &type_name[..40]
                } else {
                    type_name
                },
                dense_count,
                overflow_count,
                FILL_RATE_THRESHOLD * 100.0,
            );
        }

        writers.insert(
            type_name.clone(),
            TypeWriter {
                row_cursor: 0,
                id_is_string,
                id_data,
                id_nulls,
                id_str_data,
                id_str_offsets,
                title_data,
                title_offsets,
                title_nulls,
                title_cursor: 0,
                col_map,
                fixed_cols,
                str_cols,
                overflow_keys,
                overflow_bag_data: Vec::new(),
                overflow_offsets: Vec::with_capacity(rc + 1),
                overflow_offsets_region: Region::EMPTY,
                overflow_data_region: Region::EMPTY,
            },
        );
    }

    let total_bytes = align8(cursor).max(8); // at least 8 bytes for valid mmap

    // ── Step 1b: Create single file and mmap ──────────────────────────────

    if verbose {
        eprintln!(
            "  Phase 1b: layout computed — {:.1} GB for {} types ({:.1}s)",
            total_bytes as f64 / (1u64 << 30) as f64,
            writers.len(),
            alloc_start.elapsed().as_secs_f64(),
        );
        // Show top 10 types by space consumption
        let mut type_sizes: Vec<(&str, u64, u32)> = Vec::new();
        for (tn, meta) in type_meta {
            let mut sz = meta.title_string_bytes + (meta.row_count as u64) * 9; // title offsets+nulls + id
            if meta.id_is_string {
                sz += meta.id_string_bytes + (meta.row_count as u64) * 8;
            } else {
                sz += (meta.row_count as u64) * 4;
            }
            for cm in meta.columns.values() {
                sz += if let Some(vs) = cm.col_type.value_size() {
                    (meta.row_count as u64) * (vs as u64 + 1)
                } else {
                    cm.string_bytes + (meta.row_count as u64) * 9
                };
            }
            type_sizes.push((tn.as_str(), sz, meta.row_count));
        }
        type_sizes.sort_by_key(|t| std::cmp::Reverse(t.1));
        for (tn, sz, rc) in type_sizes.iter().take(10) {
            eprintln!(
                "    {:>8.1} GB  {:>10} rows  {}",
                *sz as f64 / (1u64 << 30) as f64,
                format_count(*rc as u64),
                if tn.len() > 50 { &tn[..50] } else { tn },
            );
        }
    }

    if total_bytes == 0 && verbose {
        eprintln!(
            "  Phase 1b: no types to pre-allocate ({:.1}s)",
            alloc_start.elapsed().as_secs_f64(),
        );
    }

    // Create the single mmap file (only if there are types to store)
    let mmap_path = col_dir.join("columns.bin");
    let mmap_opt: Option<MmapMut> = if total_bytes > 0 {
        if verbose {
            eprintln!(
                "  Phase 1b: creating {:.1} GB mmap file...",
                total_bytes as f64 / (1u64 << 30) as f64,
            );
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&mmap_path)?;
        file.set_len(total_bytes as u64)?;
        // SAFETY: file was just created+truncated to total_bytes; mmap
        // region matches the on-disk length.
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        if verbose {
            eprintln!("  Phase 1b: filling null bitmaps...");
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }
        // Fill all null bitmap regions with 0xFF (all-null)
        for region in &null_regions {
            mmap[region.offset..region.offset + region.len].fill(0xFF);
        }
        Some(mmap)
    } else {
        None
    };
    drop(null_regions);

    if verbose {
        eprintln!(
            "  Phase 1b: mmap ready — {:.1} GB for {} types ({:.1}s)",
            total_bytes as f64 / (1u64 << 30) as f64,
            writers.len(),
            alloc_start.elapsed().as_secs_f64(),
        );
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }

    // Unwrap mmap for the write/read phases (guaranteed to exist if writers is non-empty)
    let mut mmap = mmap_opt.unwrap_or_else(|| {
        // Create a dummy 1-byte mmap if no types
        let p = col_dir.join("columns.bin");
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&p)
            .unwrap();
        f.set_len(1).unwrap();
        // SAFETY: placeholder file just created+truncated to 1 byte; the
        // caller replaces this with a real-sized mmap once sizes are known.
        unsafe { MmapMut::map_mut(&f).unwrap() }
    });

    // ── Step 2: Buffered type-batched write pass ──────────────────────────
    //
    // Instead of writing each entity to the mmap immediately (random I/O across
    // 35 GB when entities are interleaved by type), we buffer entries grouped by
    // type and flush the biggest type when the buffer exceeds a threshold.
    // This makes writes sequential within each type's mmap region.

    struct BufferedEntry {
        node_idx: petgraph::graph::NodeIndex,
        id: Value,
        title: Value,
        properties: Vec<(InternedKey, Value)>,
    }

    /// Estimate the in-memory size of a log entry for buffer accounting.
    fn estimate_entry_bytes(
        id: &Value,
        title: &Value,
        properties: &[(InternedKey, Value)],
    ) -> usize {
        let mut sz = 100; // base overhead (Vec headers, NodeIndex, etc.)
        if let Value::String(s) = id {
            sz += s.len();
        }
        if let Value::String(s) = title {
            sz += s.len();
        }
        for (_, v) in properties {
            sz += match v {
                Value::String(s) => 16 + s.len(),
                _ => 16,
            };
        }
        sz
    }

    /// Flush all buffered entries for a single type to the mmap.
    /// Returns the approximate byte count that was flushed.
    #[allow(clippy::too_many_arguments)]
    fn flush_type_entries(
        type_name: &str,
        entries: &[BufferedEntry],
        writers: &mut HashMap<String, TypeWriter>,
        mmap: &mut MmapMut,
        row_ids: &mut Vec<(petgraph::graph::NodeIndex, u32)>,
    ) -> usize {
        let writer = match writers.get_mut(type_name) {
            Some(w) => w,
            None => return 0,
        };

        let mut flushed_bytes = 0;

        for entry in entries {
            let row = writer.row_cursor as usize;
            writer.row_cursor += 1;
            row_ids.push((entry.node_idx, row as u32));

            // Write id
            if writer.id_is_string {
                if let Value::String(s) = &entry.id {
                    let bytes = s.as_bytes();
                    let c = if row > 0 {
                        read_u64(mmap, &writer.id_str_offsets, row - 1) as usize
                    } else {
                        0
                    };
                    let end = c + bytes.len();
                    if c + bytes.len() <= writer.id_str_data.len {
                        let off = writer.id_str_data.offset + c;
                        mmap[off..off + bytes.len()].copy_from_slice(bytes);
                    }
                    write_u64(mmap, &writer.id_str_offsets, row, end as u64);
                    write_u8(mmap, &writer.id_nulls, row, 0);
                }
            } else if let Value::UniqueId(n) = &entry.id {
                write_u32(mmap, &writer.id_data, row, *n);
                write_u8(mmap, &writer.id_nulls, row, 0);
            }

            // Write title
            if let Value::String(s) = &entry.title {
                let bytes = s.as_bytes();
                let c = writer.title_cursor as usize;
                if c + bytes.len() <= writer.title_data.len {
                    let off = writer.title_data.offset + c;
                    mmap[off..off + bytes.len()].copy_from_slice(bytes);
                }
                writer.title_cursor += bytes.len() as u64;
                write_u64(mmap, &writer.title_offsets, row, writer.title_cursor);
                write_u8(mmap, &writer.title_nulls, row, 0);
            }

            // Write properties — dense columns and overflow bag
            let has_overflow = !writer.overflow_keys.is_empty();
            let overflow_start = writer.overflow_bag_data.len() as u64;
            if has_overflow {
                writer.overflow_offsets.push(overflow_start);
            }

            let mut overflow_entry_buf: Vec<u8> = Vec::new();
            let mut overflow_entry_count: u16 = 0;
            if has_overflow {
                overflow_entry_buf.extend_from_slice(&0u16.to_le_bytes());
            }

            for (key, value) in &entry.properties {
                if matches!(value, Value::Null) {
                    continue;
                }

                if writer.overflow_keys.contains(key) {
                    ColumnStore::serialize_overflow_value(&mut overflow_entry_buf, *key, value);
                    overflow_entry_count += 1;
                    continue;
                }

                let (col_type, idx) = match writer.col_map.get(key) {
                    Some(v) => *v,
                    None => continue,
                };

                match col_type {
                    ColType::Str => {
                        if let Value::String(s) = value {
                            let sc = &mut writer.str_cols[idx];
                            let bytes = s.as_bytes();
                            let c = sc.cursor as usize;
                            if c + bytes.len() <= sc.data.len {
                                let off = sc.data.offset + c;
                                mmap[off..off + bytes.len()].copy_from_slice(bytes);
                            }
                            sc.cursor += bytes.len() as u64;
                            write_u64(mmap, &sc.offsets, row, sc.cursor);
                            write_u8(mmap, &sc.nulls, row, 0);
                        }
                    }
                    _ => {
                        let fc = &writer.fixed_cols[idx];
                        let written = match (fc.col_type, value) {
                            (ColType::Int64, Value::Int64(v)) => {
                                write_i64(mmap, &fc.data, row, *v);
                                true
                            }
                            (ColType::Float64, Value::Float64(v)) => {
                                write_f64(mmap, &fc.data, row, *v);
                                true
                            }
                            (ColType::Float64, Value::Int64(v)) => {
                                write_f64(mmap, &fc.data, row, *v as f64);
                                true
                            }
                            (ColType::UniqueId, Value::UniqueId(v)) => {
                                write_u32(mmap, &fc.data, row, *v);
                                true
                            }
                            (ColType::Bool, Value::Boolean(v)) => {
                                write_u8(mmap, &fc.data, row, *v as u8);
                                true
                            }
                            (ColType::Date, Value::DateTime(dt)) => {
                                let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                                write_i32(mmap, &fc.data, row, (*dt - epoch).num_days() as i32);
                                true
                            }
                            _ => false, // type mismatch — leave as null
                        };
                        if written {
                            write_u8(mmap, &fc.nulls, row, 0);
                        }
                    }
                }
            }

            // Finalize this row's overflow blob
            if has_overflow {
                if overflow_entry_count > 0 {
                    overflow_entry_buf[0..2].copy_from_slice(&overflow_entry_count.to_le_bytes());
                    writer
                        .overflow_bag_data
                        .extend_from_slice(&overflow_entry_buf);
                } else {
                    writer
                        .overflow_bag_data
                        .extend_from_slice(&0u16.to_le_bytes());
                }
            }

            flushed_bytes += estimate_entry_bytes(&entry.id, &entry.title, &entry.properties);
        }

        flushed_bytes
    }

    let write_start = Instant::now();
    let reader = PropertyLogReader::open(log_path)?;
    let mut row_ids: Vec<(petgraph::graph::NodeIndex, u32)> = Vec::new();
    let mut entity_count = 0u64;
    let total_entities = type_meta.values().map(|m| m.row_count as u64).sum::<u64>();

    // Buffer: InternedKey -> list of entries (avoids per-entity string allocation)
    let mut buffers: HashMap<InternedKey, Vec<BufferedEntry>> = HashMap::new();
    let mut total_buffer_bytes: usize = 0;

    // Build InternedKey lookup for writers (resolve strings once, not per entity)
    let writer_keys: HashSet<InternedKey> = writers
        .keys()
        .filter_map(|name| graph.interner.try_resolve_to_key(name))
        .collect();

    // Build InternedKey rename map: old Q-code keys → new label keys.
    // Property log entries carry old InternedKeys from Phase 1; the type merge
    // renamed types after Phase 1 but before the log is replayed here.
    let key_rename: HashMap<InternedKey, InternedKey> = type_rename_map
        .iter()
        .filter_map(|(old, new)| {
            let old_key = graph.interner.try_resolve_to_key(old)?;
            let new_key = graph.interner.try_resolve_to_key(new)?;
            Some((old_key, new_key))
        })
        .collect();

    // Flush threshold: configurable via KGLITE_FLUSH_MB env var, default 2048 MB (2 GB)
    let flush_threshold_bytes: usize = std::env::var("KGLITE_FLUSH_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048)
        * (1 << 20);

    for entry_result in reader {
        let entry = entry_result?;
        // Remap old Q-code type keys to merged label keys
        let type_key = key_rename
            .get(&entry.node_type)
            .copied()
            .unwrap_or(entry.node_type);

        // Skip types not in writers (O(1) InternedKey lookup, no string allocation)
        if !writer_keys.contains(&type_key) {
            continue;
        }

        let entry_bytes = estimate_entry_bytes(&entry.id, &entry.title, &entry.properties);

        buffers.entry(type_key).or_default().push(BufferedEntry {
            node_idx: entry.node_idx,
            id: entry.id,
            title: entry.title,
            properties: entry.properties,
        });
        total_buffer_bytes += entry_bytes;
        entity_count += 1;

        if verbose && entity_count.is_multiple_of(10_000_000) {
            eprintln!(
                "  Phase 1b: {}/{} entities read ({:.1}s)",
                format_count(entity_count),
                format_count(total_entities),
                write_start.elapsed().as_secs_f64(),
            );
        }

        // Flush the biggest type when buffer exceeds threshold
        if total_buffer_bytes > flush_threshold_bytes {
            let biggest_key = buffers
                .iter()
                .max_by_key(|(_, entries)| entries.len())
                .map(|(key, _)| *key);
            if let Some(flush_key) = biggest_key {
                let entries = buffers.remove(&flush_key).unwrap();
                let n_entries = entries.len();
                // Resolve InternedKey → String only at flush time (once per flush)
                let flush_name = graph.interner.resolve(flush_key).to_string();
                let flushed = flush_type_entries(
                    &flush_name,
                    &entries,
                    &mut writers,
                    &mut mmap,
                    &mut row_ids,
                );
                total_buffer_bytes = total_buffer_bytes.saturating_sub(flushed);
                if verbose {
                    eprintln!(
                        "  Phase 1b: flushed {} ({} entries, {:.1} MB)",
                        flush_name,
                        format_count(n_entries as u64),
                        flushed as f64 / (1 << 20) as f64,
                    );
                }
            }
        }
    }

    // Flush all remaining buffers
    let remaining_keys: Vec<InternedKey> = buffers.keys().copied().collect();
    for flush_key in remaining_keys {
        let entries = buffers.remove(&flush_key).unwrap();
        if entries.is_empty() {
            continue;
        }
        let n_entries = entries.len();
        let flush_type = graph.interner.resolve(flush_key).to_string();
        flush_type_entries(&flush_type, &entries, &mut writers, &mut mmap, &mut row_ids);
        if verbose {
            eprintln!(
                "  Phase 1b: flushed {} ({} entries, final)",
                flush_type,
                format_count(n_entries as u64),
            );
        }
    }

    // Finalize overflow offsets: push sentinel end offset for each type writer
    for writer in writers.values_mut() {
        if !writer.overflow_keys.is_empty() {
            writer
                .overflow_offsets
                .push(writer.overflow_bag_data.len() as u64);
        }
    }

    if verbose {
        eprintln!(
            "  Phase 1b: write pass done — {} entities ({:.1}s)",
            format_count(entity_count),
            write_start.elapsed().as_secs_f64(),
        );
    }

    // ── Step 3: Forward-fill string offsets for null rows ───────────────────

    for writer in writers.values() {
        // Title offsets
        let r = &writer.title_offsets;
        let rc = r.len / 8;
        for i in 1..rc {
            if read_u64(&mmap, r, i) == 0 {
                let prev = read_u64(&mmap, r, i - 1);
                write_u64(&mut mmap, r, i, prev);
            }
        }
        // Id string offsets (if string type)
        if writer.id_is_string {
            let r = &writer.id_str_offsets;
            let rc = r.len / 8;
            for i in 1..rc {
                if read_u64(&mmap, r, i) == 0 {
                    let prev = read_u64(&mmap, r, i - 1);
                    write_u64(&mut mmap, r, i, prev);
                }
            }
        }
        // Property string offsets
        for sc in &writer.str_cols {
            let r = &sc.offsets;
            let rc = r.len / 8;
            for i in 1..rc {
                if read_u64(&mmap, r, i) == 0 {
                    let prev = read_u64(&mmap, r, i - 1);
                    write_u64(&mut mmap, r, i, prev);
                }
            }
        }
    }

    // ── Step 4: Append overflow data to mmap file and create MmapColumnStores ──

    let assemble_start = Instant::now();

    // First pass: compute total overflow bytes to append
    let mut total_overflow_bytes: usize = 0;
    for writer in writers.values() {
        if !writer.overflow_keys.is_empty() && !writer.overflow_bag_data.is_empty() {
            total_overflow_bytes = align8(total_overflow_bytes);
            // overflow offsets: (row_count+1) * 8 bytes (but stored as raw Vec<u64>)
            total_overflow_bytes += writer.overflow_offsets.len() * 8;
            total_overflow_bytes = align8(total_overflow_bytes);
            total_overflow_bytes += writer.overflow_bag_data.len();
        }
    }

    // If there's overflow data, extend the mmap file and append it
    if total_overflow_bytes > 0 {
        // Flush and drop the current mmap so we can extend the file
        mmap.flush()?;
        let current_len = mmap.len();
        drop(mmap);

        // Extend the file
        let new_len = align8(current_len) + total_overflow_bytes;
        {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&mmap_path)?;
            file.set_len(new_len as u64)?;
        }

        // Re-mmap the extended file
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&mmap_path)?;
        // SAFETY: file was just ftruncated to include the overflow region;
        // re-mmapping the full length is safe.
        mmap = unsafe { MmapMut::map_mut(&file)? };

        // Write overflow data and record regions
        let mut cursor = align8(current_len);
        for writer in writers.values_mut() {
            if writer.overflow_keys.is_empty() || writer.overflow_bag_data.is_empty() {
                continue;
            }

            if verbose {
                eprintln!(
                    "    {}: overflow bag {:.1} MB for {} sparse cols",
                    if writer.overflow_keys.len() > 40 {
                        "..."
                    } else {
                        "type"
                    },
                    writer.overflow_bag_data.len() as f64 / (1024.0 * 1024.0),
                    writer.overflow_keys.len(),
                );
            }

            // Write overflow offsets
            cursor = align8(cursor);
            let offsets_region = Region {
                offset: cursor,
                len: writer.overflow_offsets.len() * 8,
            };
            for (i, &off) in writer.overflow_offsets.iter().enumerate() {
                write_u64(&mut mmap, &offsets_region, i, off);
            }
            cursor += offsets_region.len;

            // Write overflow bag data
            cursor = align8(cursor);
            let data_region = Region {
                offset: cursor,
                len: writer.overflow_bag_data.len(),
            };
            mmap[data_region.offset..data_region.offset + data_region.len]
                .copy_from_slice(&writer.overflow_bag_data);
            cursor += data_region.len;

            // Store regions back into writer for MmapColumnStore creation
            writer.overflow_offsets_region = offsets_region;
            writer.overflow_data_region = data_region;
        }
    }

    // Create shared mmap Arc for all MmapColumnStore instances
    let mmap_arc = Arc::new(mmap);

    // Metadata to serialize for post-Phase-3 reload
    let mut columns_meta: Vec<ColumnTypeMeta> = Vec::new();

    for (type_name, writer) in &writers {
        let meta = match type_meta.get(type_name) {
            Some(m) => m,
            None => continue,
        };

        let has_overflow = !writer.overflow_keys.is_empty() && !writer.overflow_bag_data.is_empty();

        let mmap_store = MmapColumnStore {
            mmap: Arc::clone(&mmap_arc),
            row_count: meta.row_count,
            id_is_string: writer.id_is_string,
            id_fixed: if !writer.id_is_string {
                Some(FixedColumnMeta {
                    col_type: ColType::UniqueId,
                    data: writer.id_data,
                    nulls: writer.id_nulls,
                })
            } else {
                None
            },
            id_str: if writer.id_is_string {
                Some(StrColumnMeta {
                    data: writer.id_str_data,
                    offsets: writer.id_str_offsets,
                    nulls: writer.id_nulls,
                })
            } else {
                None
            },
            title: StrColumnMeta {
                data: writer.title_data,
                offsets: writer.title_offsets,
                nulls: writer.title_nulls,
            },
            col_map: writer
                .col_map
                .iter()
                .map(|(k, (ct, idx))| {
                    let cr = if matches!(ct, ColType::Str) {
                        ColRef::Str(*idx)
                    } else {
                        ColRef::Fixed(*idx)
                    };
                    (*k, cr)
                })
                .collect(),
            fixed_cols: writer
                .fixed_cols
                .iter()
                .map(|fc| FixedColumnMeta {
                    col_type: fc.col_type,
                    data: fc.data,
                    nulls: fc.nulls,
                })
                .collect(),
            str_cols: writer
                .str_cols
                .iter()
                .map(|sc| StrColumnMeta {
                    data: sc.data,
                    offsets: sc.offsets,
                    nulls: sc.nulls,
                })
                .collect(),
            overflow_offsets: if has_overflow {
                writer.overflow_offsets_region
            } else {
                Region::EMPTY
            },
            overflow_data: if has_overflow {
                writer.overflow_data_region
            } else {
                Region::EMPTY
            },
            has_overflow,
        };

        // Collect metadata for serialization
        columns_meta.push(ColumnTypeMeta {
            type_name: type_name.clone(),
            row_count: meta.row_count,
            id_is_string: writer.id_is_string,
            id_data: RegionMeta::from_region(&writer.id_data),
            id_nulls: RegionMeta::from_region(&writer.id_nulls),
            id_str_data: RegionMeta::from_region(&writer.id_str_data),
            id_str_offsets: RegionMeta::from_region(&writer.id_str_offsets),
            title_data: RegionMeta::from_region(&writer.title_data),
            title_offsets: RegionMeta::from_region(&writer.title_offsets),
            title_nulls: RegionMeta::from_region(&writer.title_nulls),
            col_map: writer
                .col_map
                .iter()
                .map(|(k, (ct, idx))| ColMapEntry {
                    key_u64: k.as_u64(),
                    col_type_str: ct.type_tag().to_string(),
                    idx: *idx,
                })
                .collect(),
            fixed_cols: writer
                .fixed_cols
                .iter()
                .map(|fc| FixedColMeta {
                    col_type_str: fc.col_type.type_tag().to_string(),
                    data: RegionMeta::from_region(&fc.data),
                    nulls: RegionMeta::from_region(&fc.nulls),
                })
                .collect(),
            str_cols: writer
                .str_cols
                .iter()
                .map(|sc| StrColMeta {
                    data: RegionMeta::from_region(&sc.data),
                    offsets: RegionMeta::from_region(&sc.offsets),
                    nulls: RegionMeta::from_region(&sc.nulls),
                })
                .collect(),
            overflow_offsets: RegionMeta::from_region(&writer.overflow_offsets_region),
            overflow_data: RegionMeta::from_region(&writer.overflow_data_region),
            has_overflow: !writer.overflow_keys.is_empty() && !writer.overflow_bag_data.is_empty(),
        });

        let store = ColumnStore::from_mmap_store(Arc::new(mmap_store));

        // Register schema from dense columns
        if !graph.type_schemas.contains_key(type_name) {
            let schema = TypeSchema::from_keys(writer.col_map.keys().copied());
            graph
                .type_schemas
                .insert(type_name.clone(), Arc::new(schema));
        }
        graph
            .column_stores
            .insert(type_name.clone(), Arc::new(store));
    }

    if verbose {
        eprintln!(
            "  Phase 1b: assembled {} column stores ({:.1}s)",
            graph.column_stores.len(),
            assemble_start.elapsed().as_secs_f64(),
        );
    }

    // ── Step 5: Fix up DiskNodeSlot row_ids ────────────────────────────────
    //
    // `update_row_id` is on `GraphWrite` (disk override, default no-op on
    // memory/mapped). Phase 2 introduced this exactly to replace the
    // enum-match here.
    if GraphRead::is_disk(&graph.graph) {
        for &(node_idx, row_id) in &row_ids {
            GraphWrite::update_row_id(&mut graph.graph, node_idx, row_id);
        }
        if verbose {
            eprintln!(
                "  Phase 1b: fixed up {} row_ids",
                format_count(row_ids.len() as u64),
            );
        }
    }

    // Save columns metadata for post-Phase-3 reload.
    // The mmap file stays on disk at data_dir/columns.bin — don't delete it.
    let meta_path = data_dir.join("columns_meta.json");
    if !columns_meta.is_empty() {
        if let Ok(json) = serde_json::to_string(&columns_meta) {
            let _ = std::fs::write(&meta_path, json);
        }
        // Also save as bincode+zstd for fast loading (~10x faster than JSON parse)
        if let Ok(bytes) = bincode::serialize(&columns_meta) {
            if let Ok(compressed) = zstd::encode_all(bytes.as_slice(), 3) {
                let _ = std::fs::write(data_dir.join("columns_meta.bin.zst"), compressed);
            }
        }
    }

    if verbose {
        eprintln!(
            "  Phase 1b: saved columns metadata ({} types) to {}",
            columns_meta.len(),
            meta_path.display(),
        );
    }

    Ok(())
}

/// Flush an accumulated entity into the graph as a node.
#[allow(clippy::too_many_arguments)]
fn flush_entity(
    graph: &mut DirGraph,
    acc: EntityAccumulator,
    config: &NTriplesConfig,
    edge_buffer: &mut EdgeBuffer,
    stats: &mut NTriplesStats,
    mapped: bool,
    prop_log: &mut Option<crate::graph::storage::memory::property_log::PropertyLogWriter>,
    label_cache: &mut HashMap<u32, String>,
    type_meta: &mut HashMap<String, TypeBuildMeta>,
    qnum_to_idx: &mut Option<MmapOrVec<u32>>,
) {
    let title = acc.label.unwrap_or_else(|| acc.id.clone());

    // Cache this entity's label for auto-type resolution.
    // Later entities may reference this one via P31.
    if config.auto_type {
        if let Some(qnum) = parse_qcode_number(&acc.id) {
            label_cache.insert(qnum, title.clone());
        }
    }

    // Determine node type from P31 value.
    // Priority: explicit node_types mapping > auto-type (label cache) > Q-code > "Entity"
    let node_type = if let Some(ref tq) = acc.type_qcode {
        if let Some(mapped_name) = config.node_types.get(tq) {
            mapped_name.clone()
        } else if config.auto_type {
            // Try to resolve Q-code to label via cache
            if let Some(qnum) = parse_qcode_number(tq) {
                label_cache
                    .get(&qnum)
                    .cloned()
                    .unwrap_or_else(|| tq.clone())
            } else {
                tq.clone()
            }
        } else {
            "Entity".to_string()
        }
    } else {
        "Entity".to_string()
    };

    let mut properties = acc.properties;
    // Store nid as a queryable string property (e.g., "Q42") so Cypher {nid: 'Q42'} works
    properties.insert("nid".to_string(), Value::String(acc.id.clone()));
    if let Some(desc) = acc.description {
        properties.insert("description".to_string(), Value::String(desc));
    }
    // Store P31 Q-code as a property (preserves type info when defaulting to "Entity")
    if let Some(ref tq) = acc.type_qcode {
        properties.insert("P31".to_string(), Value::String(tq.clone()));
    }

    // Choose ID representation: compact u32 for disk only, String for default/mapped
    let use_compact_ids = graph.graph.is_disk();
    let id_value = if mapped || use_compact_ids {
        parse_qcode_number(&acc.id)
            .map(Value::UniqueId)
            .unwrap_or_else(|| Value::String(acc.id.clone()))
    } else {
        Value::String(acc.id.clone())
    };
    let title_value = Value::String(title);

    let mut node_data = NodeData::new(
        id_value.clone(),
        title_value,
        node_type.clone(),
        properties,
        &mut graph.interner,
    );

    // Mapped mode: push properties into ColumnStore (existing path).
    if mapped {
        let interned_props = node_data
            .properties
            .drain_to_interned_pairs(&graph.interner);
        let keys: Vec<_> = interned_props.iter().map(|(k, _)| *k).collect();
        graph.ensure_type_schema_keys(&node_type, &keys);
        let store = graph.ensure_column_store_for_push(&node_type);
        store.push_id(&node_data.id);
        store.push_title(&node_data.title);
        store.push_row(&interned_props);
        node_data.properties = PropertyStorage::Map(HashMap::new());
        node_data.id = Value::Null;
        node_data.title = Value::Null;
    }

    // For disk mode: serialize properties to the log file BEFORE clearing them.
    // We need node_data properties to still be present for the log write,
    // then clear them before add_node (DiskGraph discards them anyway).
    let saved_id = node_data.id.clone();
    let saved_title = node_data.title.clone();
    let saved_props: Vec<(InternedKey, Value)> = if prop_log.is_some() {
        let pairs = node_data
            .properties
            .drain_to_interned_pairs(&graph.interner);
        node_data.properties = PropertyStorage::Map(HashMap::new());
        node_data.id = Value::Null;
        node_data.title = Value::Null;
        pairs
    } else {
        Vec::new()
    };

    let node_idx = GraphWrite::add_node(&mut graph.graph, node_data);

    // Write to property log after add_node so we have the node_idx
    if let Some(ref mut log) = prop_log {
        let node_type_key = graph.interner.get_or_intern(&node_type);
        log.write_entity(
            node_type_key,
            node_idx,
            &saved_id,
            &saved_title,
            &saved_props,
        )
        .expect("Property log write failed");

        // Collect per-type metadata for Phase 1b pre-allocation
        type_meta
            .entry(node_type.clone())
            .or_insert_with(TypeBuildMeta::new)
            .record_entity(&saved_id, &saved_title, &saved_props);
    }

    // Update type_indices
    graph
        .type_indices
        .entry(node_type.clone())
        .or_default()
        .push(node_idx);

    // For disk mode: write directly to qnum_to_idx mmap (skip id_indices to save ~11 GB RAM).
    // For other modes: use id_indices HashMap as before.
    if let Some(ref mut qt) = qnum_to_idx {
        if let Some(qnum) = parse_qcode_number(&acc.id) {
            if (qnum as usize) < qt.len() {
                qt.set(qnum as usize, node_idx.index() as u32 + 1); // +1: 0 = not present
            }
        }
    } else {
        graph
            .id_indices
            .entry(node_type)
            .or_default()
            .insert(id_value, node_idx);
    }

    stats.entities_created += 1;

    // Periodic spill: every 100K entities, check if columns should be spilled to disk
    if mapped && stats.entities_created.is_multiple_of(100_000) {
        graph.maybe_spill_columns();
    }

    // Buffer outgoing edges
    match edge_buffer {
        EdgeBuffer::Compact(buf) => {
            if let Some(src_num) = parse_qcode_number(&acc.id) {
                for (pred_label, target_qcode) in acc.outgoing_edges {
                    if let Some(tgt_num) = parse_qcode_number(&target_qcode) {
                        let pred_key = graph.interner.get_or_intern(&pred_label);
                        buf.push((src_num, tgt_num, pred_key));
                    }
                }
            }
        }
        EdgeBuffer::Strings(buf) => {
            for (pred_label, target_qcode) in acc.outgoing_edges {
                buf.push((acc.id.clone(), target_qcode, pred_label));
            }
        }
    }
}

/// Create edges from the buffer. Looks up source/target by Q-code across all types.
/// Fast edge creation using pre-built qnum_to_idx from Phase 1 (disk mode).
/// Avoids rebuilding the lookup table from id_indices (saves ~11 GB RAM at full scale).
fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

#[cfg(test)]
mod tests {
    use super::super::parser::{XSD_BOOLEAN, XSD_DECIMAL, XSD_DOUBLE};
    use super::*;

    #[test]
    fn test_parse_entity_triple() {
        let line = r#"<http://www.wikidata.org/entity/Q42> <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> ."#;
        let (subj, pred, obj) = parse_line(line).unwrap();
        assert!(matches!(subj, Subject::Entity("Q42")));
        assert!(matches!(pred, Predicate::WikidataDirect("P31")));
        assert!(matches!(obj, Object::Entity("Q5")));
    }

    #[test]
    fn test_parse_literal_triple() {
        let line = r#"<http://www.wikidata.org/entity/Q42> <http://www.w3.org/2000/01/rdf-schema#label> "Douglas Adams"@en ."#;
        let (subj, pred, obj) = parse_line(line).unwrap();
        assert!(matches!(subj, Subject::Entity("Q42")));
        assert!(matches!(pred, Predicate::Label));
        assert!(matches!(obj, Object::LangLiteral(ref t, "en") if t == "Douglas Adams"));
    }

    #[test]
    fn test_parse_typed_literal() {
        let line = r#"<http://www.wikidata.org/entity/Q31> <http://www.wikidata.org/prop/direct/P1082> "+11825551"^^<http://www.w3.org/2001/XMLSchema#decimal> ."#;
        let (_, pred, obj) = parse_line(line).unwrap();
        assert!(matches!(pred, Predicate::WikidataDirect("P1082")));
        assert!(matches!(obj, Object::TypedLiteral(ref t, _) if t == "+11825551"));
    }

    #[test]
    fn test_parse_escaped_string() {
        let line = r#"<http://www.wikidata.org/entity/Q31> <http://www.wikidata.org/prop/direct/P1448> "K\u00F6nigreich Belgien"@de ."#;
        let (_, _, obj) = parse_line(line).unwrap();
        assert!(matches!(obj, Object::LangLiteral(ref t, "de") if t == "Königreich Belgien"));
    }

    #[test]
    fn test_typed_literal_to_value() {
        assert_eq!(
            typed_literal_to_value("+11825551", XSD_DECIMAL),
            Value::Int64(11825551)
        );
        assert_eq!(
            typed_literal_to_value("3.14", XSD_DOUBLE),
            Value::Float64(3.14)
        );
        assert_eq!(
            typed_literal_to_value("true", XSD_BOOLEAN),
            Value::Boolean(true)
        );
    }

    #[test]
    fn test_language_filter() {
        let filter = Some(HashSet::from(["en".to_string()]));
        assert!(language_matches("en", &filter));
        assert!(!language_matches("de", &filter));
        assert!(language_matches("de", &None));
    }

    #[test]
    fn test_parse_qcode_number() {
        assert_eq!(parse_qcode_number("Q42"), Some(42));
        assert_eq!(parse_qcode_number("Q0"), Some(0));
        assert_eq!(parse_qcode_number("Q130000000"), Some(130_000_000));
        assert_eq!(parse_qcode_number("P31"), None); // not a Q-code
        assert_eq!(parse_qcode_number("Q"), None); // no number
        assert_eq!(parse_qcode_number(""), None); // empty
        assert_eq!(parse_qcode_number("Q-1"), None); // negative
    }

    #[test]
    fn test_edge_buffer_compact_size() {
        // Verify compact edge buffer entry is much smaller than string-based
        assert_eq!(
            std::mem::size_of::<(u32, u32, InternedKey)>(),
            16 // 4 + 4 + 8
        );
        // String tuple is at least 72 bytes on stack (3 × 24 for String)
        assert!(std::mem::size_of::<(String, String, String)>() >= 72);
    }
}
