//! N-Triples parser — line-level parsing and literal type inference.
//!
//! Triple AST (`Subject` / `Predicate` / `Object` / `EdgeBuffer`),
//! XSD literal → `Value` coercion, Wikidata Q-code helpers, and
//! language-filter utilities.

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use std::collections::{HashMap, HashSet};

// ─── Parsed triple ──────────────────────────────────────────────────────────

/// The subject of a triple (only Wikidata Q-entities are interesting).
/// Borrows from the input line for zero-copy parsing.
pub(super) enum Subject<'a> {
    Entity(&'a str), // Q-code slice, e.g. "Q42"
    Other,
}

/// The predicate of a triple. Borrows P-code from input line.
pub(super) enum Predicate<'a> {
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
pub(super) enum Object<'a> {
    Entity(&'a str),               // Q-code slice
    Literal(String),               // owned (escape-processed)
    LangLiteral(String, &'a str),  // (owned value, borrowed lang)
    TypedLiteral(String, &'a str), // (owned value, borrowed type_uri)
    Other,
}

// ─── Edge buffer ────────────────────────────────────────────────────────────

/// Edge buffer: String-based (default mode) or compact u32-based (mapped mode).
pub(super) enum EdgeBuffer {
    /// Default mode: (source_qcode, target_qcode, predicate_label) — ~80 bytes each
    Strings(Vec<(String, String, String)>),
    /// Mapped mode: (source_qnum, target_qnum, interned_predicate) — 16 bytes each.
    /// File-backed (MmapOrVec) in disk mode to avoid holding ~14 GB in RAM.
    Compact(MmapOrVec<(u32, u32, InternedKey)>),
}

impl EdgeBuffer {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Strings(v) => v.len(),
            Self::Compact(v) => v.len(),
        }
    }
}

/// Parse "Q42" → 42. Returns None for non-Q-code or overflow.
pub(super) fn parse_qcode_number(qcode: &str) -> Option<u32> {
    qcode.strip_prefix('Q')?.parse::<u32>().ok()
}

// ─── Accumulated entity ─────────────────────────────────────────────────────

/// Collects all triples for one subject entity before flushing to graph.
pub(super) struct EntityAccumulator {
    pub(super) id: String, // Q-code
    pub(super) label: Option<String>,
    pub(super) description: Option<String>,
    pub(super) type_qcode: Option<String>, // P31 target, e.g. "Q5"
    pub(super) properties: HashMap<String, Value>,
    pub(super) outgoing_edges: Vec<(String, String)>, // (predicate, target Q-code)
}

impl EntityAccumulator {
    pub(super) fn new(id: String) -> Self {
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
pub(super) fn parse_line(line: &str) -> Option<(Subject<'_>, Predicate<'_>, Object<'_>)> {
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
pub(super) fn parse_object<'a>(s: &'a str) -> Object<'a> {
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
pub(super) fn extract_quoted_string(s: &str) -> Option<(String, &str)> {
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

pub(super) const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";
pub(super) const XSD_DECIMAL: &str = "http://www.w3.org/2001/XMLSchema#decimal";
pub(super) const XSD_DOUBLE: &str = "http://www.w3.org/2001/XMLSchema#double";
pub(super) const XSD_FLOAT: &str = "http://www.w3.org/2001/XMLSchema#float";
pub(super) const XSD_DATE: &str = "http://www.w3.org/2001/XMLSchema#dateTime";
pub(super) const XSD_BOOLEAN: &str = "http://www.w3.org/2001/XMLSchema#boolean";

pub(super) fn typed_literal_to_value(text: &str, type_uri: &str) -> Value {
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

pub(super) fn language_matches(lang: &str, filter: &Option<HashSet<String>>) -> bool {
    match filter {
        None => true,
        Some(langs) => langs.contains(lang),
    }
}

/// Extract text from a literal object, respecting language filter.
pub(super) fn extract_lang_text(
    object: &Object<'_>,
    languages: &Option<HashSet<String>>,
) -> Option<String> {
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
