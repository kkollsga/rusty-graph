//! Rust language parser (ported from parsers/rust.py).

use regex::Regex;
use serde_json::json;
use std::path::Path;
use std::sync::OnceLock;
use tree_sitter::{Node, Parser, Tree};

use super::shared::{
    count_lines, extract_comment_annotations, get_type_parameters, node_text, DEFAULT_COMMENT_TYPES,
};
use super::LanguageParser;
use crate::code_tree::models::{
    AttributeInfo, ClassInfo, ConstantInfo, EnumInfo, FileInfo, FunctionInfo, InterfaceInfo,
    ParseResult, TypeRelationship,
};

pub const RUST_NOISE_NAMES: &[&str] = &[
    // iterator / collection methods
    "len",
    "is_empty",
    "contains",
    "get",
    "insert",
    "remove",
    "push",
    "pop",
    "clear",
    "extend",
    "iter",
    "next",
    "collect",
    "map",
    "filter",
    "with_capacity",
    "reserve",
    // clone/conversion traits
    "clone",
    "to_string",
    "to_owned",
    "from",
    "into",
    "as_ref",
    "as_mut",
    // common trait methods
    "new",
    "default",
    "fmt",
    "eq",
    "ne",
    "cmp",
    "partial_cmp",
    "hash",
    "deref",
    "drop",
    // Option/Result
    "unwrap",
    "expect",
    "ok",
    "err",
    "map_err",
    "unwrap_or",
    "unwrap_or_else",
    "unwrap_or_default",
    // display / debug
    "write",
    "writeln",
    // set/get
    "set",
];

const NESTED_SCOPES: &[&str] = &["function_item", "closure_expression"];

fn py_name_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"name\s*=\s*"([^"]+)""#).expect("py_name regex compiles"))
}

pub struct RustParser;

thread_local! {
    static RS_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("loading tree-sitter-rust grammar");
        std::cell::RefCell::new(p)
    };
}

impl RustParser {
    pub fn new() -> Self {
        RustParser
    }

    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        RS_PARSER.with(|p| p.borrow_mut().parse(source, None))
    }

    fn get_visibility(node: Node, source: &[u8]) -> &'static str {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "visibility_modifier" {
                let text = node_text(child, source);
                if text.contains("crate") {
                    return "pub(crate)";
                }
                return "pub";
            }
        }
        "private"
    }

    fn get_doc_comment(node: Node, source: &[u8]) -> Option<String> {
        let mut doc_lines: Vec<String> = Vec::new();
        let mut sibling = node.prev_named_sibling();
        while let Some(s) = sibling {
            match s.kind() {
                "line_comment" => {
                    let text = node_text(s, source).trim();
                    if let Some(rest) = text.strip_prefix("///") {
                        let content = rest.strip_prefix(' ').unwrap_or(rest);
                        doc_lines.insert(0, content.to_string());
                        sibling = s.prev_named_sibling();
                        continue;
                    }
                    break;
                }
                "block_comment" => {
                    let text = node_text(s, source).trim();
                    if let Some(rest) = text.strip_prefix("/**") {
                        let rest = rest.strip_suffix("*/").unwrap_or(rest);
                        let mut lines = Vec::new();
                        for line in rest.split('\n') {
                            let line = line.trim();
                            let cleaned = if let Some(r) = line.strip_prefix("* ") {
                                r
                            } else if let Some(r) = line.strip_prefix('*') {
                                r
                            } else {
                                line
                            };
                            lines.push(cleaned);
                        }
                        let content = lines.join("\n").trim().to_string();
                        if !content.is_empty() {
                            doc_lines.insert(0, content);
                        }
                        break;
                    }
                    break;
                }
                "attribute_item" => {
                    sibling = s.prev_named_sibling();
                    continue;
                }
                _ => break,
            }
        }
        if doc_lines.is_empty() {
            None
        } else {
            Some(doc_lines.join("\n"))
        }
    }

    fn get_attributes(node: Node, source: &[u8]) -> Vec<String> {
        let mut attrs: Vec<String> = Vec::new();
        let mut sibling = node.prev_named_sibling();
        while let Some(s) = sibling {
            match s.kind() {
                "attribute_item" => {
                    attrs.insert(0, node_text(s, source).to_string());
                    sibling = s.prev_named_sibling();
                }
                "line_comment" => {
                    sibling = s.prev_named_sibling();
                }
                _ => break,
            }
        }
        attrs
    }

    fn has_pyclass(attrs: &[String]) -> bool {
        attrs.iter().any(|a| a.contains("#[pyclass"))
    }

    fn is_pymethods_block(attrs: &[String]) -> bool {
        attrs.iter().any(|a| a.contains("#[pymethods]"))
    }

    fn is_pymethod_fn(fn_attrs: &[String], impl_is_pymethods: bool) -> bool {
        if impl_is_pymethods {
            return true;
        }
        fn_attrs
            .iter()
            .any(|a| ["#[pyfunction]", "#[new]"].iter().any(|m| a.contains(m)))
    }

    fn extract_py_name(attrs: &[String], keyword: &str) -> Option<String> {
        for a in attrs {
            if a.contains(keyword) {
                if let Some(m) = py_name_re().captures(a) {
                    return m.get(1).map(|g| g.as_str().to_string());
                }
            }
        }
        None
    }

    fn get_return_type(node: Node, source: &[u8]) -> Option<String> {
        let mut saw_arrow = false;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if !child.is_named() && node_text(child, source) == "->" {
                saw_arrow = true;
            } else if saw_arrow && child.kind() != "block" {
                return Some(node_text(child, source).to_string());
            }
        }
        None
    }

    fn get_signature(node: Node, source: &[u8]) -> String {
        let mut parts: Vec<&str> = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "block" {
                break;
            }
            parts.push(node_text(child, source));
        }
        parts.join(" ")
    }

    fn is_async_fn(node: Node, source: &[u8]) -> bool {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if !child.is_named() && node_text(child, source) == "async" {
                return true;
            }
            if child.kind() == "identifier" || node_text(child, source) == "fn" {
                break;
            }
        }
        false
    }

    fn get_name<'a>(node: Node<'a>, source: &'a [u8], name_type: &str) -> Option<&'a str> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == name_type {
                return Some(node_text(child, source));
            }
        }
        None
    }

    fn extract_type_name_from_node<'a>(node: Node<'a>, source: &'a [u8]) -> Option<&'a str> {
        match node.kind() {
            "type_identifier" => Some(node_text(node, source)),
            "generic_type" => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    match child.kind() {
                        "type_identifier" => return Some(node_text(child, source)),
                        "scoped_type_identifier" => {
                            return Self::extract_type_name_from_node(child, source);
                        }
                        _ => {}
                    }
                }
                None
            }
            "scoped_type_identifier" => {
                // Walk children in reverse; take last type_identifier.
                let mut cursor = node.walk();
                let children: Vec<Node> = node.children(&mut cursor).collect();
                for child in children.into_iter().rev() {
                    if child.kind() == "type_identifier" {
                        return Some(node_text(child, source));
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn extract_calls(body: Node, source: &[u8]) -> Vec<(String, u32)> {
        let mut calls: Vec<(String, u32)> = Vec::new();
        fn walk(node: Node, source: &[u8], out: &mut Vec<(String, u32)>) {
            if node.kind() == "call_expression" {
                let line = node.start_position().row as u32 + 1;
                let func = node
                    .child_by_field_name("function")
                    .or_else(|| node.child(0));
                if let Some(func) = func {
                    match func.kind() {
                        "identifier" => {
                            out.push((node_text(func, source).to_string(), line));
                        }
                        "field_expression" => {
                            // field_expression has value and field children.
                            let field = func.child_by_field_name("field").or_else(|| {
                                let mut cursor = func.walk();
                                let mut found: Option<Node> = None;
                                for c in func.children(&mut cursor) {
                                    if c.kind() == "field_identifier" {
                                        found = Some(c);
                                        break;
                                    }
                                }
                                found
                            });
                            let value = func.child_by_field_name("value");
                            if let Some(field) = field {
                                let field_name = node_text(field, source);
                                if let Some(value) = value {
                                    let val_text = node_text(value, source);
                                    // Receiver hint: last segment after "." or "::".
                                    let hint = val_text
                                        .rsplit('.')
                                        .next()
                                        .and_then(|p| p.rsplit("::").next())
                                        .unwrap_or(val_text);
                                    if hint == "self" || hint == "&self" || hint == "Self" {
                                        out.push((field_name.to_string(), line));
                                    } else {
                                        out.push((format!("{}.{}", hint, field_name), line));
                                    }
                                } else {
                                    out.push((field_name.to_string(), line));
                                }
                            }
                        }
                        "scoped_identifier" => {
                            let text = node_text(func, source);
                            let parts: Vec<&str> = text.split("::").collect();
                            if parts.len() >= 2 {
                                out.push((
                                    format!(
                                        "{}.{}",
                                        parts[parts.len() - 2],
                                        parts[parts.len() - 1]
                                    ),
                                    line,
                                ));
                            } else if let Some(last) = parts.last() {
                                out.push(((*last).to_string(), line));
                            }
                        }
                        _ => {}
                    }
                }
            }
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if !NESTED_SCOPES.contains(&child.kind()) {
                    walk(child, source, out);
                }
            }
        }
        walk(body, source, &mut calls);
        calls
    }

    fn file_to_module_path(filepath: &Path, src_root: &Path) -> String {
        let rel = filepath.strip_prefix(src_root).unwrap_or(filepath);
        let mut parts: Vec<String> = rel
            .components()
            .filter_map(|c| c.as_os_str().to_str().map(str::to_string))
            .collect();
        if let Some(last) = parts.last_mut() {
            if let Some(stem) = last.strip_suffix(".rs") {
                *last = stem.to_string();
            }
            if last == "mod" || last == "lib" {
                parts.pop();
            }
        }
        if parts.is_empty() {
            "crate".to_string()
        } else {
            format!("crate::{}", parts.join("::"))
        }
    }

    fn extract_struct_fields(
        node: Node,
        source: &[u8],
        owner_qname: &str,
        rel_path: &str,
    ) -> Vec<AttributeInfo> {
        let mut attrs = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "field_declaration_list" {
                continue;
            }
            let mut fc = child.walk();
            for field in child.children(&mut fc) {
                if field.kind() != "field_declaration" {
                    continue;
                }
                let mut name: Option<String> = None;
                let mut type_ann: Option<String> = None;
                let mut vis = "private".to_string();
                let mut saw_colon = false;
                let mut inner = field.walk();
                for fc2 in field.children(&mut inner) {
                    match fc2.kind() {
                        "visibility_modifier" => {
                            let text = node_text(fc2, source);
                            vis = if text.contains("crate") {
                                "pub(crate)".into()
                            } else {
                                "pub".into()
                            };
                        }
                        "field_identifier" | "identifier" if !saw_colon => {
                            name = Some(node_text(fc2, source).to_string());
                        }
                        _ => {
                            if !fc2.is_named() && node_text(fc2, source) == ":" {
                                saw_colon = true;
                            } else if saw_colon && type_ann.is_none() && fc2.is_named() {
                                type_ann = Some(node_text(fc2, source).to_string());
                            }
                        }
                    }
                }
                if let Some(name) = name {
                    attrs.push(AttributeInfo {
                        qualified_name: format!("{}::{}", owner_qname, name),
                        owner_qualified_name: owner_qname.to_string(),
                        type_annotation: type_ann,
                        visibility: vis,
                        name,
                        file_path: rel_path.to_string(),
                        line_number: field.start_position().row as u32 + 1,
                        default_value: None,
                    });
                }
            }
        }
        attrs
    }

    fn get_enum_variants(node: Node, source: &[u8]) -> (Vec<String>, Vec<serde_json::Value>) {
        let mut names = Vec::new();
        let mut details: Vec<serde_json::Value> = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "enum_variant_list" {
                continue;
            }
            let mut lc = child.walk();
            for variant in child.children(&mut lc) {
                if variant.kind() != "enum_variant" {
                    continue;
                }
                let Some(name) = Self::get_name(variant, source, "identifier") else {
                    continue;
                };
                names.push(name.to_string());
                let mut detail = serde_json::Map::new();
                detail.insert("name".into(), json!(name));
                detail.insert("kind".into(), json!("unit"));
                let mut vc = variant.walk();
                for sub in variant.children(&mut vc) {
                    match sub.kind() {
                        "field_declaration_list" => {
                            detail.insert("kind".into(), json!("struct"));
                            detail.insert(
                                "fields".into(),
                                json!(Self::extract_variant_struct_fields(sub, source)),
                            );
                        }
                        "ordered_field_declaration_list" => {
                            detail.insert("kind".into(), json!("tuple"));
                            detail.insert(
                                "fields".into(),
                                json!(Self::extract_variant_tuple_fields(sub, source)),
                            );
                        }
                        _ => {}
                    }
                }
                details.push(serde_json::Value::Object(detail));
            }
        }
        (names, details)
    }

    fn extract_variant_struct_fields(field_list: Node, source: &[u8]) -> Vec<serde_json::Value> {
        let mut fields = Vec::new();
        let mut cursor = field_list.walk();
        for child in field_list.children(&mut cursor) {
            if child.kind() != "field_declaration" {
                continue;
            }
            let mut name: Option<String> = None;
            let mut type_ann: Option<String> = None;
            let mut saw_colon = false;
            let mut fc = child.walk();
            for sub in child.children(&mut fc) {
                match sub.kind() {
                    "field_identifier" | "identifier" if !saw_colon => {
                        name = Some(node_text(sub, source).to_string());
                    }
                    _ => {
                        if !sub.is_named() && node_text(sub, source) == ":" {
                            saw_colon = true;
                        } else if saw_colon && type_ann.is_none() && sub.is_named() {
                            type_ann = Some(node_text(sub, source).to_string());
                        }
                    }
                }
            }
            if let Some(name) = name {
                let mut entry = serde_json::Map::new();
                entry.insert("name".into(), json!(name));
                if let Some(t) = type_ann {
                    entry.insert("type".into(), json!(t));
                }
                fields.push(serde_json::Value::Object(entry));
            }
        }
        fields
    }

    fn extract_variant_tuple_fields(field_list: Node, source: &[u8]) -> Vec<serde_json::Value> {
        let mut fields = Vec::new();
        let mut cursor = field_list.walk();
        for child in field_list.children(&mut cursor) {
            if child.is_named() && child.kind() != "visibility_modifier" {
                let mut entry = serde_json::Map::new();
                entry.insert("type".into(), json!(node_text(child, source)));
                fields.push(serde_json::Value::Object(entry));
            }
        }
        fields
    }

    // ── Parsing ────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn parse_function(
        node: Node,
        source: &[u8],
        module_path: &str,
        file_path: &str,
        is_method: bool,
        owner: Option<&str>,
        impl_is_pymethods: bool,
    ) -> FunctionInfo {
        let name = Self::get_name(node, source, "identifier")
            .unwrap_or("unknown")
            .to_string();
        let prefix = match owner {
            Some(o) => format!("{}::{}", module_path, o),
            None => module_path.to_string(),
        };
        let qualified_name = format!("{}::{}", prefix, name);
        let attrs = Self::get_attributes(node, source);

        let mut body: Option<Node> = None;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "block" {
                body = Some(child);
                break;
            }
        }

        let is_pymethod = Self::is_pymethod_fn(&attrs, impl_is_pymethods);
        let is_ffi = attrs.iter().any(|a| a.contains("#[no_mangle]"));
        let is_test = attrs.iter().any(|a| {
            a == "#[test]"
                || a == "#[bench]"
                || a.contains("#[tokio::test")
                || a.contains("#[rstest")
        });
        let ffi_kind = if is_pymethod {
            Some("pyo3")
        } else if is_ffi {
            Some("extern_c")
        } else {
            None
        };
        let mut visibility = Self::get_visibility(node, source).to_string();
        if is_pymethod && visibility == "private" {
            visibility = "pub(py)".to_string();
        }
        let is_pymodule = attrs.iter().any(|a| a.contains("#[pymodule]"));

        let mut metadata = crate::code_tree::models::MetadataMap::new();
        metadata.insert("is_pymethod".into(), json!(is_pymethod));
        if is_test {
            metadata.insert("is_test".into(), json!(true));
        }
        if is_ffi || is_pymethod {
            metadata.insert("is_ffi".into(), json!(true));
            if let Some(k) = ffi_kind {
                metadata.insert("ffi_kind".into(), json!(k));
            }
            if let Some(py_name) = Self::extract_py_name(&attrs, "#[pyfunction") {
                metadata.insert("py_name".into(), json!(py_name));
            }
        }
        if is_pymodule {
            metadata.insert("is_pymodule".into(), json!(true));
            metadata.insert("is_ffi".into(), json!(true));
            metadata.insert("ffi_kind".into(), json!("pyo3"));
        }

        let calls = body
            .map(|b| Self::extract_calls(b, source))
            .unwrap_or_default();

        FunctionInfo {
            visibility,
            qualified_name,
            is_async: Self::is_async_fn(node, source),
            is_method,
            signature: Self::get_signature(node, source),
            file_path: file_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            end_line: Some(node.end_position().row as u32 + 1),
            docstring: Self::get_doc_comment(node, source),
            return_type: Self::get_return_type(node, source),
            calls,
            type_parameters: get_type_parameters(node, source, "type_parameters"),
            decorators: Vec::new(),
            metadata,
            name,
        }
    }

    fn parse_items(
        node: Node,
        source: &[u8],
        module_path: &str,
        rel_path: &str,
        file_info: &mut FileInfo,
        result: &mut ParseResult,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "function_item" => {
                    result.functions.push(Self::parse_function(
                        child,
                        source,
                        module_path,
                        rel_path,
                        false,
                        None,
                        false,
                    ));
                }
                "struct_item" => {
                    let name = Self::get_name(child, source, "type_identifier")
                        .unwrap_or("unknown")
                        .to_string();
                    let attrs = Self::get_attributes(child, source);
                    let qname = format!("{}::{}", module_path, name);
                    let is_pyclass = Self::has_pyclass(&attrs);
                    let mut visibility = Self::get_visibility(child, source).to_string();
                    if is_pyclass && visibility == "private" {
                        visibility = "pub(py)".to_string();
                    }
                    let mut metadata = crate::code_tree::models::MetadataMap::new();
                    metadata.insert("is_pyclass".into(), json!(is_pyclass));
                    if is_pyclass {
                        if let Some(py_name) = Self::extract_py_name(&attrs, "#[pyclass") {
                            metadata.insert("py_name".into(), json!(py_name));
                        }
                    }
                    result.classes.push(ClassInfo {
                        qualified_name: qname.clone(),
                        kind: "struct".into(),
                        visibility,
                        file_path: rel_path.to_string(),
                        line_number: child.start_position().row as u32 + 1,
                        end_line: Some(child.end_position().row as u32 + 1),
                        docstring: Self::get_doc_comment(child, source),
                        bases: Vec::new(),
                        type_parameters: get_type_parameters(child, source, "type_parameters"),
                        metadata,
                        name: name.clone(),
                    });
                    result
                        .attributes
                        .extend(Self::extract_struct_fields(child, source, &qname, rel_path));
                }
                "enum_item" => {
                    let name = Self::get_name(child, source, "type_identifier")
                        .unwrap_or("unknown")
                        .to_string();
                    let (variant_names, variant_details) = Self::get_enum_variants(child, source);
                    result.enums.push(EnumInfo {
                        qualified_name: format!("{}::{}", module_path, name),
                        visibility: Self::get_visibility(child, source).to_string(),
                        file_path: rel_path.to_string(),
                        line_number: child.start_position().row as u32 + 1,
                        end_line: Some(child.end_position().row as u32 + 1),
                        docstring: Self::get_doc_comment(child, source),
                        variants: variant_names,
                        variant_details: if variant_details.is_empty() {
                            None
                        } else {
                            Some(variant_details)
                        },
                        name,
                    });
                }
                "trait_item" => {
                    let name = Self::get_name(child, source, "type_identifier")
                        .unwrap_or("unknown")
                        .to_string();
                    let qname = format!("{}::{}", module_path, name);
                    result.interfaces.push(InterfaceInfo {
                        qualified_name: qname.clone(),
                        kind: "trait".into(),
                        visibility: Self::get_visibility(child, source).to_string(),
                        file_path: rel_path.to_string(),
                        line_number: child.start_position().row as u32 + 1,
                        end_line: Some(child.end_position().row as u32 + 1),
                        docstring: Self::get_doc_comment(child, source),
                        type_parameters: get_type_parameters(child, source, "type_parameters"),
                        name: name.clone(),
                    });
                    let mut trait_rel = TypeRelationship {
                        source_type: qname.clone(),
                        target_type: None,
                        relationship: "inherent".into(),
                        methods: Vec::new(),
                    };
                    let mut tc = child.walk();
                    for inner in child.children(&mut tc) {
                        if inner.kind() == "declaration_list" {
                            let mut ic = inner.walk();
                            for item in inner.children(&mut ic) {
                                if matches!(
                                    item.kind(),
                                    "function_item" | "function_signature_item"
                                ) {
                                    let fn_info = Self::parse_function(
                                        item,
                                        source,
                                        module_path,
                                        rel_path,
                                        true,
                                        Some(&name),
                                        false,
                                    );
                                    trait_rel.methods.push(fn_info.clone());
                                    result.functions.push(fn_info);
                                }
                            }
                        }
                    }
                    if !trait_rel.methods.is_empty() {
                        result.type_relationships.push(trait_rel);
                    }
                }
                "impl_item" => {
                    let attrs = Self::get_attributes(child, source);
                    let pymethods = Self::is_pymethods_block(&attrs);
                    let mut seen_for = false;
                    let mut types_before: Vec<Node> = Vec::new();
                    let mut types_after: Vec<Node> = Vec::new();
                    let mut cc = child.walk();
                    for c in child.children(&mut cc) {
                        if !c.is_named() && node_text(c, source) == "for" {
                            seen_for = true;
                        } else if matches!(
                            c.kind(),
                            "type_identifier" | "generic_type" | "scoped_type_identifier"
                        ) {
                            if seen_for {
                                types_after.push(c);
                            } else {
                                types_before.push(c);
                            }
                        }
                    }
                    let (trait_name, self_type): (Option<&str>, Option<&str>) =
                        if seen_for && !types_before.is_empty() && !types_after.is_empty() {
                            (
                                Self::extract_type_name_from_node(types_before[0], source),
                                Self::extract_type_name_from_node(types_after[0], source),
                            )
                        } else if !types_before.is_empty() {
                            (
                                None,
                                Self::extract_type_name_from_node(types_before[0], source),
                            )
                        } else {
                            (None, None)
                        };
                    let Some(self_type) = self_type else { continue };
                    let relationship = if trait_name.is_some() {
                        "implements"
                    } else {
                        "inherent"
                    };
                    let mut type_rel = TypeRelationship {
                        source_type: self_type.to_string(),
                        target_type: trait_name.map(|s| s.to_string()),
                        relationship: relationship.into(),
                        methods: Vec::new(),
                    };
                    let mut cc2 = child.walk();
                    for inner in child.children(&mut cc2) {
                        if inner.kind() == "declaration_list" {
                            let mut ic = inner.walk();
                            for item in inner.children(&mut ic) {
                                if item.kind() == "function_item" {
                                    let fn_info = Self::parse_function(
                                        item,
                                        source,
                                        module_path,
                                        rel_path,
                                        true,
                                        Some(self_type),
                                        pymethods,
                                    );
                                    type_rel.methods.push(fn_info.clone());
                                    result.functions.push(fn_info);
                                }
                            }
                        }
                    }
                    result.type_relationships.push(type_rel);
                }
                "use_declaration" => {
                    let mut uc = child.walk();
                    let mut path_text: Option<String> = None;
                    for sub in child.children(&mut uc) {
                        if matches!(
                            sub.kind(),
                            "scoped_identifier" | "use_wildcard" | "scoped_use_list" | "identifier"
                        ) {
                            path_text = Some(node_text(sub, source).to_string());
                        }
                    }
                    if let Some(p) = path_text {
                        file_info.imports.push(p);
                    }
                }
                "mod_item" => {
                    let Some(mod_name) = Self::get_name(child, source, "identifier") else {
                        continue;
                    };
                    let mod_name = mod_name.to_string();
                    let mut decl_list: Option<Node> = None;
                    let mut mc = child.walk();
                    for sub in child.children(&mut mc) {
                        if sub.kind() == "declaration_list" {
                            decl_list = Some(sub);
                            break;
                        }
                    }
                    if let Some(decl_list) = decl_list {
                        let inner_path = format!("{}::{}", module_path, mod_name);
                        Self::parse_items(
                            decl_list,
                            source,
                            &inner_path,
                            rel_path,
                            file_info,
                            result,
                        );
                    } else {
                        file_info.submodule_declarations.push(mod_name);
                    }
                }
                "type_item" => {
                    if let Some(name) = Self::get_name(child, source, "type_identifier") {
                        let name = name.to_string();
                        let mut saw_eq = false;
                        let mut val_text: Option<String> = None;
                        let mut tc = child.walk();
                        for sub in child.children(&mut tc) {
                            if !sub.is_named() && node_text(sub, source) == "=" {
                                saw_eq = true;
                            } else if saw_eq && sub.is_named() {
                                let text = node_text(sub, source);
                                let take = text
                                    .char_indices()
                                    .nth(100)
                                    .map(|(i, _)| i)
                                    .unwrap_or(text.len());
                                val_text = Some(text[..take].to_string());
                                break;
                            }
                        }
                        result.constants.push(ConstantInfo {
                            qualified_name: format!("{}::{}", module_path, name),
                            kind: "type_alias".into(),
                            type_annotation: val_text,
                            value_preview: None,
                            visibility: Self::get_visibility(child, source).to_string(),
                            file_path: rel_path.to_string(),
                            line_number: child.start_position().row as u32 + 1,
                            name,
                        });
                    }
                }
                "const_item" | "static_item" => {
                    if let Some(name) = Self::get_name(child, source, "identifier") {
                        let name = name.to_string();
                        let kind = if child.kind() == "const_item" {
                            "constant"
                        } else {
                            "static"
                        };
                        let mut type_ann: Option<String> = None;
                        let mut val_text: Option<String> = None;
                        let mut saw_colon = false;
                        let mut saw_eq = false;
                        let mut tc = child.walk();
                        for sub in child.children(&mut tc) {
                            if !sub.is_named() && node_text(sub, source) == ":" {
                                saw_colon = true;
                            } else if saw_colon && !saw_eq && sub.is_named() {
                                type_ann = Some(node_text(sub, source).to_string());
                                saw_colon = false;
                            } else if !sub.is_named() && node_text(sub, source) == "=" {
                                saw_eq = true;
                            } else if saw_eq && sub.is_named() {
                                let text = node_text(sub, source);
                                let take = text
                                    .char_indices()
                                    .nth(100)
                                    .map(|(i, _)| i)
                                    .unwrap_or(text.len());
                                val_text = Some(text[..take].to_string());
                                break;
                            }
                        }
                        result.constants.push(ConstantInfo {
                            qualified_name: format!("{}::{}", module_path, name),
                            kind: kind.into(),
                            type_annotation: type_ann,
                            value_preview: val_text,
                            visibility: Self::get_visibility(child, source).to_string(),
                            file_path: rel_path.to_string(),
                            line_number: child.start_position().row as u32 + 1,
                            name,
                        });
                    }
                }
                _ => {}
            }
        }
    }
}

impl Default for RustParser {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageParser for RustParser {
    fn language_name(&self) -> &'static str {
        "rust"
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        &["rs"]
    }

    fn noise_names(&self) -> &'static [&'static str] {
        RUST_NOISE_NAMES
    }

    fn parse_file(&self, filepath: &Path, src_root: &Path) -> ParseResult {
        let Ok(source) = std::fs::read(filepath) else {
            return ParseResult::new();
        };
        let Some(tree) = self.parse_tree(&source) else {
            return ParseResult::new();
        };
        let root = tree.root_node();

        let rel_path = filepath
            .strip_prefix(src_root)
            .unwrap_or(filepath)
            .to_string_lossy()
            .replace('\\', "/");
        let module_path = Self::file_to_module_path(filepath, src_root);
        let loc = count_lines(&source);

        let filename = filepath
            .file_name()
            .and_then(|o| o.to_str())
            .unwrap_or("")
            .to_string();
        let stem = filepath
            .file_stem()
            .and_then(|o| o.to_str())
            .unwrap_or("")
            .to_string();

        let is_test = stem.ends_with("_test")
            || stem.ends_with("_tests")
            || stem.starts_with("test_")
            || rel_path.contains("/tests/")
            || rel_path.starts_with("tests/")
            || rel_path.contains("/benches/")
            || rel_path.starts_with("benches/");

        let mut file_info = FileInfo {
            path: rel_path.clone(),
            filename,
            loc,
            module_path: module_path.clone(),
            language: "rust".to_string(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test,
        };

        let mut result = ParseResult::new();
        Self::parse_items(
            root,
            &source,
            &module_path,
            &rel_path,
            &mut file_info,
            &mut result,
        );
        file_info.annotations = extract_comment_annotations(root, &source, DEFAULT_COMMENT_TYPES);
        result.files.push(file_info);
        result
    }
}
