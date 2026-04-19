//! Shared parser helpers (ported from parsers/base.py).

use crate::code_tree::models::Annotation;
use regex::Regex;
use std::sync::OnceLock;
use tree_sitter::Node;

/// Extract the UTF-8 text of a tree-sitter node.
#[inline]
pub fn node_text<'a>(node: Node<'a>, source: &'a [u8]) -> &'a str {
    // UTF-8 boundaries are enforced by tree-sitter; safe unwrap on well-formed input.
    std::str::from_utf8(&source[node.byte_range()]).unwrap_or("")
}

/// Count lines of code in `source`. Matches the Python `count_lines` semantics:
/// number of newlines, plus one if the file is non-empty and doesn't end with `\n`.
pub fn count_lines(source: &[u8]) -> u32 {
    let newlines = bytecount_newlines(source);
    if !source.is_empty() && !source.ends_with(b"\n") {
        newlines + 1
    } else {
        newlines
    }
}

#[inline]
fn bytecount_newlines(src: &[u8]) -> u32 {
    src.iter().filter(|&&b| b == b'\n').count() as u32
}

/// Extract generic/template parameters from a declaration node.
///
/// Looks for a child whose type matches `node_type` (default: `"type_parameters"`)
/// and returns the inner text with surrounding `<>` or `[]` stripped.
pub fn get_type_parameters(node: Node, source: &[u8], node_type: &str) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == node_type {
            let text = node_text(child, source).trim();
            let stripped = if let Some(inner) =
                text.strip_prefix('<').and_then(|s| s.strip_suffix('>'))
            {
                inner.trim()
            } else if let Some(inner) = text.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                inner.trim()
            } else {
                text
            };
            if stripped.is_empty() {
                return None;
            }
            return Some(stripped.to_string());
        }
    }
    None
}

/// Names that count as `self`/`cls` parameters and must be stripped from signatures.
const SELF_PARAMS: &[&str] = &["self", "&self", "&mut self", "cls"];

/// Extract the parameter list from a function signature string.
///
/// Finds the first balanced `(...)` group, drops `self`/`cls` equivalents,
/// and returns the cleaned comma-separated list or `None` if empty.
pub fn extract_parameters_from_signature(signature: &str) -> Option<String> {
    let start = signature.find('(')?;
    let mut depth = 0_i32;
    let mut end = start;
    for (i, ch) in signature[start..].char_indices() {
        let idx = start + i;
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = idx;
                    break;
                }
            }
            _ => {}
        }
    }
    if end <= start {
        return None;
    }
    let params_text = signature[start + 1..end].trim();
    if params_text.is_empty() {
        return None;
    }
    let kept: Vec<&str> = params_text
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty() && !SELF_PARAMS.contains(p))
        .collect();
    if kept.is_empty() {
        None
    } else {
        Some(kept.join(", "))
    }
}

fn annotation_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(TODO|FIXME|HACK|SAFETY|XXX|BUG|NOTE|WARNING)\b[:\s]*(.*)")
            .expect("annotation regex compiles")
    })
}

/// Default comment node kinds scanned by `extract_comment_annotations`.
pub const DEFAULT_COMMENT_TYPES: &[&str] = &["line_comment", "block_comment", "comment"];

/// Scan every comment node under `root` for TODO/FIXME/etc annotations.
///
/// Iterative stack traversal — avoids blowing Rust's call stack on deeply
/// nested ASTs (same reason the Python version uses an explicit stack).
pub fn extract_comment_annotations(
    root: Node,
    source: &[u8],
    comment_types: &[&str],
) -> Option<Vec<Annotation>> {
    let re = annotation_regex();
    let mut out: Vec<Annotation> = Vec::new();
    let mut stack: Vec<Node> = vec![root];

    while let Some(node) = stack.pop() {
        let kind = node.kind();
        if comment_types.iter().any(|t| *t == kind) {
            let text = node_text(node, source);
            for caps in re.captures_iter(text) {
                let kind = caps
                    .get(1)
                    .map(|m| m.as_str().to_ascii_uppercase())
                    .unwrap_or_default();
                let body = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
                let truncated = if body.len() > 200 { &body[..200] } else { body };
                out.push(Annotation {
                    kind,
                    text: truncated.to_string(),
                    line: node.start_position().row as u32 + 1,
                });
            }
        }
        // Push children in reverse so we pop them in source order.
        let mut cursor = node.walk();
        let children: Vec<Node> = node.children(&mut cursor).collect();
        for child in children.into_iter().rev() {
            stack.push(child);
        }
    }

    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}
