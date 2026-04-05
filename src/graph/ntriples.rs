// src/graph/ntriples.rs
//
// Streaming N-Triples loader with bz2/gz/plain support.
// Designed for Wikidata truthy dumps but works with any N-Triples file.
//
// Single-pass algorithm:
//   1. Stream-decompress → parse lines → filter by predicate/language
//   2. Accumulate properties per subject; flush node on subject change
//   3. Buffer edges (source_id, target_id, predicate)
//   4. After EOF: create all buffered edges, skip if target missing

use crate::datatypes::values::Value;
use crate::graph::schema::{
    DirGraph, EdgeData, InternedKey, NodeData, PropertyStorage, StorageMode,
};
use bzip2::read::BzDecoder;
use flate2::read::GzDecoder;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::Instant;

// ─── Public types ───────────────────────────────────────────────────────────

/// Stats returned after loading.
pub struct NTriplesStats {
    pub triples_scanned: u64,
    pub entities_created: u64,
    pub edges_created: u64,
    pub edges_skipped: u64, // target entity not in graph
    pub seconds: f64,
}

/// Configuration for the loader.
pub struct NTriplesConfig {
    /// Only import these Wikidata predicates (P-codes). None = all.
    pub predicates: Option<HashSet<String>>,
    /// Only keep literals in these languages. None = all.
    pub languages: Option<HashSet<String>>,
    /// Map P31 target Q-code → human-readable node type name.
    pub node_types: HashMap<String, String>,
    /// Map P-code → human-readable predicate label.
    pub predicate_labels: HashMap<String, String>,
    /// Stop after this many entities. None = no limit.
    pub max_entities: Option<usize>,
    /// Print progress to stderr.
    pub verbose: bool,
}

// ─── Parsed triple ──────────────────────────────────────────────────────────

/// The subject of a triple (only Wikidata Q-entities are interesting).
/// Borrows from the input line for zero-copy parsing.
enum Subject<'a> {
    Entity(&'a str), // Q-code slice, e.g. "Q42"
    Other,
}

/// The predicate of a triple. Borrows P-code from input line.
enum Predicate<'a> {
    WikidataDirect(&'a str), // P-code slice, e.g. "P31"
    Label,
    Description,
    AltLabel,
    Type,
    Other,
}

/// The object of a triple.
/// Entity and typed/lang metadata borrow from input line.
/// Literal values are owned (may contain escape sequences).
enum Object<'a> {
    Entity(&'a str),               // Q-code slice
    Literal(String),               // owned (escape-processed)
    LangLiteral(String, &'a str),  // (owned value, borrowed lang)
    TypedLiteral(String, &'a str), // (owned value, borrowed type_uri)
    Other,
}

// ─── Edge buffer ────────────────────────────────────────────────────────────

/// Edge buffer: String-based (default mode) or compact u32-based (mapped mode).
enum EdgeBuffer {
    /// Default mode: (source_qcode, target_qcode, predicate_label) — ~80 bytes each
    Strings(Vec<(String, String, String)>),
    /// Mapped mode: (source_qnum, target_qnum, interned_predicate) — 16 bytes each
    Compact(Vec<(u32, u32, InternedKey)>),
}

impl EdgeBuffer {
    fn len(&self) -> usize {
        match self {
            Self::Strings(v) => v.len(),
            Self::Compact(v) => v.len(),
        }
    }
}

/// Parse "Q42" → 42. Returns None for non-Q-code or overflow.
fn parse_qcode_number(qcode: &str) -> Option<u32> {
    qcode.strip_prefix('Q')?.parse::<u32>().ok()
}

// ─── Accumulated entity ─────────────────────────────────────────────────────

/// Collects all triples for one subject entity before flushing to graph.
struct EntityAccumulator {
    id: String, // Q-code
    label: Option<String>,
    description: Option<String>,
    type_qcode: Option<String>, // P31 target, e.g. "Q5"
    properties: HashMap<String, Value>,
    outgoing_edges: Vec<(String, String)>, // (predicate, target Q-code)
}

impl EntityAccumulator {
    fn new(id: String) -> Self {
        Self {
            id,
            label: None,
            description: None,
            type_qcode: None,
            properties: HashMap::new(),
            outgoing_edges: Vec::new(),
        }
    }
}

// ─── Line parser ────────────────────────────────────────────────────────────

const WD_ENTITY: &str = "http://www.wikidata.org/entity/";
const WD_PROP_DIRECT: &str = "http://www.wikidata.org/prop/direct/";
const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
const SCHEMA_DESC: &str = "http://schema.org/description";
const SKOS_ALT: &str = "http://www.w3.org/2004/02/skos/core#altLabel";
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

/// Parse a single N-Triples line into (subject, predicate, object).
/// Returns borrowed slices into the input line — zero heap allocations
/// for subject/predicate/entity references.
fn parse_line(line: &str) -> Option<(Subject<'_>, Predicate<'_>, Object<'_>)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    if !line.starts_with('<') {
        return None;
    }
    let subj_end = line.find("> ")?;
    let subj_uri = &line[1..subj_end];

    let subject = if let Some(qcode) = subj_uri.strip_prefix(WD_ENTITY) {
        if qcode.starts_with('Q') {
            Subject::Entity(qcode) // borrow, no allocation
        } else {
            Subject::Other
        }
    } else {
        Subject::Other
    };

    let rest = &line[subj_end + 2..];
    if !rest.starts_with('<') {
        return None;
    }
    let pred_end = rest.find("> ")?;
    let pred_uri = &rest[1..pred_end];

    let predicate = if let Some(pcode) = pred_uri.strip_prefix(WD_PROP_DIRECT) {
        if pcode.starts_with('P') {
            Predicate::WikidataDirect(pcode) // borrow, no allocation
        } else {
            Predicate::Other
        }
    } else if pred_uri == RDFS_LABEL {
        Predicate::Label
    } else if pred_uri == SCHEMA_DESC {
        Predicate::Description
    } else if pred_uri == SKOS_ALT {
        Predicate::AltLabel
    } else if pred_uri == RDF_TYPE {
        Predicate::Type
    } else {
        Predicate::Other
    };

    let obj_str = rest[pred_end + 2..].trim_end();
    let obj_str = obj_str.strip_suffix('.')?.trim_end();

    let object = parse_object(obj_str);

    Some((subject, predicate, object))
}

/// Parse the object portion of an N-Triples line.
/// Entity Q-codes and lang/type tags borrow from the input. Literal values are owned.
fn parse_object<'a>(s: &'a str) -> Object<'a> {
    if s.starts_with('<') {
        let uri = s.trim_start_matches('<').trim_end_matches('>');
        if let Some(qcode) = uri.strip_prefix(WD_ENTITY) {
            if qcode.starts_with('Q') {
                return Object::Entity(qcode); // borrow, no allocation
            }
        }
        Object::Other
    } else if s.starts_with('"') {
        if let Some((value, suffix)) = extract_quoted_string(s) {
            if suffix.is_empty() {
                Object::Literal(value)
            } else if let Some(lang) = suffix.strip_prefix('@') {
                Object::LangLiteral(value, lang) // lang borrows from input
            } else if let Some(type_part) = suffix.strip_prefix("^^<") {
                let type_uri = type_part.trim_end_matches('>');
                Object::TypedLiteral(value, type_uri) // type_uri borrows
            } else {
                Object::Literal(value)
            }
        } else {
            Object::Other
        }
    } else {
        Object::Other
    }
}

/// Extract the string content from a quoted N-Triples literal,
/// returning (unescaped_value, suffix_after_closing_quote).
fn extract_quoted_string(s: &str) -> Option<(String, &str)> {
    let s = s.strip_prefix('"')?;
    let mut value = String::new();
    let mut chars = s.char_indices();
    let mut end_idx = 0;

    while let Some((idx, ch)) = chars.next() {
        if ch == '\\' {
            // Escape sequence
            if let Some((_, next_ch)) = chars.next() {
                match next_ch {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '"' => value.push('"'),
                    '\\' => value.push('\\'),
                    'u' => {
                        // \uXXXX
                        let hex: String = chars.by_ref().take(4).map(|(_, c)| c).collect();
                        if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(cp) {
                                value.push(c);
                            }
                        }
                    }
                    'U' => {
                        // \UXXXXXXXX
                        let hex: String = chars.by_ref().take(8).map(|(_, c)| c).collect();
                        if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(cp) {
                                value.push(c);
                            }
                        }
                    }
                    _ => {
                        value.push('\\');
                        value.push(next_ch);
                    }
                }
            }
        } else if ch == '"' {
            end_idx = idx + 1; // position after the closing quote in `s`
            break;
        } else {
            value.push(ch);
        }
    }

    Some((value, &s[end_idx..]))
}

// ─── Typed literal conversion ───────────────────────────────────────────────

const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";
const XSD_DECIMAL: &str = "http://www.w3.org/2001/XMLSchema#decimal";
const XSD_DOUBLE: &str = "http://www.w3.org/2001/XMLSchema#double";
const XSD_FLOAT: &str = "http://www.w3.org/2001/XMLSchema#float";
const XSD_DATE: &str = "http://www.w3.org/2001/XMLSchema#dateTime";
const XSD_BOOLEAN: &str = "http://www.w3.org/2001/XMLSchema#boolean";

fn typed_literal_to_value(text: &str, type_uri: &str) -> Value {
    match type_uri {
        XSD_INTEGER => text
            .parse::<i64>()
            .map(Value::Int64)
            .unwrap_or(Value::String(text.to_string())),
        XSD_DECIMAL => {
            // Wikidata decimals often have leading "+"
            let cleaned = text.trim_start_matches('+');
            if let Ok(i) = cleaned.parse::<i64>() {
                Value::Int64(i)
            } else if let Ok(f) = cleaned.parse::<f64>() {
                Value::Float64(f)
            } else {
                Value::String(text.to_string())
            }
        }
        XSD_DOUBLE | XSD_FLOAT => text
            .parse::<f64>()
            .map(Value::Float64)
            .unwrap_or(Value::String(text.to_string())),
        XSD_BOOLEAN => match text {
            "true" | "1" => Value::Boolean(true),
            "false" | "0" => Value::Boolean(false),
            _ => Value::String(text.to_string()),
        },
        XSD_DATE => {
            // Try to parse ISO date, keep as string if it fails
            Value::String(text.to_string())
        }
        _ => {
            // GeoSPARQL WKT literals, etc. — keep as string
            Value::String(text.to_string())
        }
    }
}

// ─── Main loader ────────────────────────────────────────────────────────────

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

    // Phase 1: Always use fast non-mapped insertion (HashMap properties).
    // Columnar conversion in Phase 1b is faster in bulk than per-entity ColumnStore push.
    let final_mode = graph.storage_mode;
    let mapped = false;
    let mut current: Option<EntityAccumulator> = None;
    let use_compact = final_mode == StorageMode::Mapped || final_mode == StorageMode::Disk;
    let mut edge_buffer = if use_compact {
        EdgeBuffer::Compact(Vec::new())
    } else {
        EdgeBuffer::Strings(Vec::new())
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
                    flush_entity(graph, acc, config, &mut edge_buffer, &mut stats, mapped);
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
        flush_entity(graph, acc, config, &mut edge_buffer, &mut stats, mapped);
    }

    if config.verbose {
        let buf_len = edge_buffer.len() as u64;
        eprintln!(
            "  Phase 1 done: {} entities, {} edges buffered ({:.1}s)",
            format_count(stats.entities_created),
            format_count(buf_len),
            start.elapsed().as_secs_f64(),
        );
    }

    // Phase 1b: Convert to columnar storage in bulk.
    if final_mode == StorageMode::Mapped || final_mode == StorageMode::Disk {
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

    if config.verbose {
        eprintln!("  Phase 2: creating edges...");
    }

    // Phase 2: Create edges from buffer
    let edge_start = Instant::now();
    create_edges_from_buffer(graph, &edge_buffer, &mut stats);

    if config.verbose {
        eprintln!(
            "  Phase 2 done: {} edges created, {} skipped ({:.1}s)",
            format_count(stats.edges_created),
            format_count(stats.edges_skipped),
            edge_start.elapsed().as_secs_f64(),
        );
    }

    // Free edge_buffer before Phase 3 — at Wikidata scale this is ~13.8 GB.
    drop(edge_buffer);

    // Phase 3: Build CSR from pending edges (disk mode)
    if let crate::graph::schema::GraphBackend::Disk(ref mut dg) = graph.graph {
        if config.verbose {
            eprintln!("  Phase 3: building CSR from pending edges...");
        }
        let csr_start = Instant::now();
        dg.build_csr_from_pending();
        if config.verbose {
            eprintln!("  Phase 3 done ({:.1}s)", csr_start.elapsed().as_secs_f64());
        }
    }

    // Sync column stores to DiskGraph
    if final_mode == StorageMode::Disk {
        graph.sync_disk_column_stores();
    }

    stats.seconds = start.elapsed().as_secs_f64();
    Ok(stats)
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Check if a language tag matches the filter.
fn language_matches(lang: &str, filter: &Option<HashSet<String>>) -> bool {
    match filter {
        None => true,
        Some(langs) => langs.contains(lang),
    }
}

/// Extract text from a literal object, respecting language filter.
fn extract_lang_text(object: &Object<'_>, languages: &Option<HashSet<String>>) -> Option<String> {
    match object {
        Object::LangLiteral(text, lang) => {
            if language_matches(lang, languages) {
                Some(text.clone())
            } else {
                None
            }
        }
        Object::Literal(text) => Some(text.clone()),
        _ => None,
    }
}

/// Flush an accumulated entity into the graph as a node.
fn flush_entity(
    graph: &mut DirGraph,
    acc: EntityAccumulator,
    config: &NTriplesConfig,
    edge_buffer: &mut EdgeBuffer,
    stats: &mut NTriplesStats,
    mapped: bool,
) {
    // Determine node type from P31 value.
    // Only use mapped type names from config.node_types. If no mapping exists,
    // default to "Entity" (avoids creating thousands of types from raw Q-codes).
    let node_type = acc
        .type_qcode
        .as_ref()
        .and_then(|q| config.node_types.get(q).cloned())
        .unwrap_or_else(|| "Entity".to_string());

    let title = acc.label.unwrap_or_else(|| acc.id.clone());

    let mut properties = acc.properties;
    if let Some(desc) = acc.description {
        properties.insert("description".to_string(), Value::String(desc));
    }
    // Store P31 Q-code as a property (preserves type info when defaulting to "Entity")
    if let Some(ref tq) = acc.type_qcode {
        properties.insert("P31".to_string(), Value::String(tq.clone()));
    }

    // Choose ID representation: compact u32 for mapped/disk, String for default
    let use_compact_ids = !mapped
        && (graph.storage_mode == StorageMode::Mapped || graph.storage_mode == StorageMode::Disk);
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

    // Mapped/Disk mode: push properties into ColumnStore.
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

    let node_idx = graph.graph.add_node(node_data);

    // Update type_indices
    graph
        .type_indices
        .entry(node_type.clone())
        .or_default()
        .push(node_idx);

    // Update id_indices for O(1) lookup
    graph
        .id_indices
        .entry(node_type)
        .or_default()
        .insert(id_value, node_idx);

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
fn create_edges_from_buffer(
    graph: &mut DirGraph,
    edge_buffer: &EdgeBuffer,
    stats: &mut NTriplesStats,
) {
    match edge_buffer {
        EdgeBuffer::Compact(buf) => create_edges_compact(graph, buf, stats),
        EdgeBuffer::Strings(buf) => create_edges_strings(graph, buf, stats),
    }
}

/// Compact path: edges stored as (u32, u32, InternedKey).
/// Uses dense Vec lookup (not HashMap) for cache-friendly O(1) access.
/// Streams edges directly — no intermediate allocation.
fn create_edges_compact(
    graph: &mut DirGraph,
    buf: &[(u32, u32, InternedKey)],
    stats: &mut NTriplesStats,
) {
    // Build dense Vec lookup: Q-number → NodeIndex.
    // Much faster than HashMap for Wikidata's dense Q-number space.
    let mut max_qnum: u32 = 0;
    for id_map in graph.id_indices.values() {
        for (id_val, _) in id_map.iter() {
            let n = match id_val {
                Value::UniqueId(n) => n,
                Value::String(s) => {
                    if let Some(n) = parse_qcode_number(s.as_str()) {
                        n
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            if n > max_qnum {
                max_qnum = n;
            }
        }
    }

    // NodeIndex is u32-sized, use 0xFFFFFFFF as "not present"
    const EMPTY: u32 = u32::MAX;
    let mut qnum_to_idx: Vec<u32> = vec![EMPTY; max_qnum as usize + 1];
    for id_map in graph.id_indices.values() {
        for (id_val, node_idx) in id_map.iter() {
            let n = match id_val {
                Value::UniqueId(n) => n,
                Value::String(s) => {
                    if let Some(n) = parse_qcode_number(s.as_str()) {
                        n
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            qnum_to_idx[n as usize] = node_idx.index() as u32;
        }
    }

    // Track unique connection types (for metadata, computed once — not per edge)
    let mut conn_types_seen: HashSet<InternedKey> = HashSet::new();

    // Stream edges — use direct pending_edges push for disk mode (bypass add_edge overhead)
    let qnum_len = qnum_to_idx.len();
    let is_disk = matches!(graph.graph, crate::graph::schema::GraphBackend::Disk(_));

    if is_disk {
        // Fast path for disk mode: push directly to pending_edges Vec
        if let crate::graph::schema::GraphBackend::Disk(ref mut dg) = graph.graph {
            dg.pending_edges.get_mut().reserve(buf.len());
            for &(src_num, tgt_num, pred_key) in buf {
                let src_ok =
                    (src_num as usize) < qnum_len && qnum_to_idx[src_num as usize] != EMPTY;
                let tgt_ok =
                    (tgt_num as usize) < qnum_len && qnum_to_idx[tgt_num as usize] != EMPTY;
                if src_ok && tgt_ok {
                    let src_idx = qnum_to_idx[src_num as usize];
                    let tgt_idx = qnum_to_idx[tgt_num as usize];
                    dg.pending_edges
                        .get_mut()
                        .push((src_idx, tgt_idx, pred_key.as_u64()));
                    dg.edge_count += 1;
                    dg.next_edge_idx += 1;
                    conn_types_seen.insert(pred_key);
                    stats.edges_created += 1;
                } else {
                    stats.edges_skipped += 1;
                }
            }
        }
    } else {
        // Standard path for petgraph: per-edge add_edge
        for &(src_num, tgt_num, pred_key) in buf {
            let src_ok = (src_num as usize) < qnum_len && qnum_to_idx[src_num as usize] != EMPTY;
            let tgt_ok = (tgt_num as usize) < qnum_len && qnum_to_idx[tgt_num as usize] != EMPTY;
            if src_ok && tgt_ok {
                let src = petgraph::graph::NodeIndex::new(qnum_to_idx[src_num as usize] as usize);
                let tgt = petgraph::graph::NodeIndex::new(qnum_to_idx[tgt_num as usize] as usize);
                let edge_data = EdgeData {
                    connection_type: pred_key,
                    properties: Vec::new(),
                };
                graph.graph.add_edge(src, tgt, edge_data);
                conn_types_seen.insert(pred_key);
                stats.edges_created += 1;
            } else {
                stats.edges_skipped += 1;
            }
        }
    }

    // Register connection type metadata once (not per edge)
    let node_types: Vec<String> = graph.type_indices.keys().cloned().collect();
    for conn_key in &conn_types_seen {
        let conn_name = graph.interner.resolve(*conn_key).to_string();
        for src_type in &node_types {
            for tgt_type in &node_types {
                graph.upsert_connection_type_metadata(
                    &conn_name,
                    src_type,
                    tgt_type,
                    HashMap::new(),
                );
            }
        }
    }

    graph.invalidate_edge_type_counts_cache();
}

/// String path: edges stored as (String, String, String).
fn create_edges_strings(
    graph: &mut DirGraph,
    buf: &[(String, String, String)],
    stats: &mut NTriplesStats,
) {
    // Build Q-code string → NodeIndex lookup
    let mut qcode_to_idx: HashMap<String, petgraph::graph::NodeIndex> = HashMap::new();
    for id_map in graph.id_indices.values() {
        for (id_val, node_idx) in id_map.iter() {
            if let Value::String(ref s) = id_val {
                if s.starts_with('Q') {
                    qcode_to_idx.insert(s.clone(), node_idx);
                }
            }
        }
    }

    let mut conn_type_pairs: HashMap<String, (HashSet<String>, HashSet<String>)> = HashMap::new();

    for (source_qcode, target_qcode, pred_label) in buf {
        let source_idx = qcode_to_idx.get(source_qcode.as_str());
        let target_idx = qcode_to_idx.get(target_qcode.as_str());

        match (source_idx, target_idx) {
            (Some(&src), Some(&tgt)) => {
                let edge_data =
                    EdgeData::new(pred_label.clone(), HashMap::new(), &mut graph.interner);

                let src_type = graph
                    .graph
                    .node_weight(src)
                    .unwrap()
                    .node_type_str(&graph.interner)
                    .to_string();
                let tgt_type = graph
                    .graph
                    .node_weight(tgt)
                    .unwrap()
                    .node_type_str(&graph.interner)
                    .to_string();
                let entry = conn_type_pairs
                    .entry(pred_label.clone())
                    .or_insert_with(|| (HashSet::new(), HashSet::new()));
                entry.0.insert(src_type);
                entry.1.insert(tgt_type);

                graph.graph.add_edge(src, tgt, edge_data);
                stats.edges_created += 1;
            }
            _ => {
                stats.edges_skipped += 1;
            }
        }
    }

    for (conn_type, (source_types, target_types)) in conn_type_pairs {
        for src_type in &source_types {
            for tgt_type in &target_types {
                graph.upsert_connection_type_metadata(
                    &conn_type,
                    src_type,
                    tgt_type,
                    HashMap::new(),
                );
            }
        }
    }

    graph.invalidate_edge_type_counts_cache();
}

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
