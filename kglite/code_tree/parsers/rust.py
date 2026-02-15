"""Rust language parser using tree-sitter-rust."""

from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_rust as ts_rust

from .base import LanguageParser, node_text, count_lines
from .models import (
    ParseResult, FileInfo, FunctionInfo, ClassInfo,
    EnumInfo, InterfaceInfo, TypeRelationship,
    AttributeInfo, ConstantInfo,
)

RUST_LANGUAGE = Language(ts_rust.language())

RUST_NOISE_NAMES: frozenset[str] = frozenset({
    # Iterator / collection methods
    "len", "is_empty", "contains", "get", "insert", "remove", "push", "pop",
    "clear", "extend", "iter", "next", "collect", "map", "filter",
    "with_capacity", "reserve",
    # Clone / conversion traits
    "clone", "to_string", "to_owned", "from", "into", "as_ref", "as_mut",
    # Common trait methods
    "new", "default", "fmt", "eq", "ne", "cmp", "partial_cmp", "hash",
    "deref", "drop",
    # Option/Result methods
    "unwrap", "expect", "ok", "err", "map_err", "unwrap_or",
    "unwrap_or_else", "unwrap_or_default",
    # Display / Debug
    "write", "writeln",
    # Set/Get patterns
    "set",
})


class RustParser(LanguageParser):

    @property
    def language_name(self) -> str:
        return "rust"

    @property
    def file_extensions(self) -> list[str]:
        return [".rs"]

    @property
    def noise_names(self) -> frozenset[str]:
        return RUST_NOISE_NAMES

    def __init__(self):
        self._parser = Parser(RUST_LANGUAGE)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_visibility(self, node) -> str:
        for child in node.children:
            if child.type == "visibility_modifier":
                text = child.text.decode("utf8")
                if "crate" in text:
                    return "pub(crate)"
                return "pub"
        return "private"

    def _get_doc_comment(self, node, source: bytes) -> str | None:
        """Walk backward through siblings to collect /// doc comments."""
        doc_lines = []
        sibling = node.prev_named_sibling
        while sibling is not None:
            if sibling.type == "line_comment":
                text = node_text(sibling, source).strip()
                if text.startswith("///"):
                    content = text[3:]
                    if content.startswith(" "):
                        content = content[1:]
                    doc_lines.insert(0, content)
                    sibling = sibling.prev_named_sibling
                    continue
            elif sibling.type == "attribute_item":
                sibling = sibling.prev_named_sibling
                continue
            break
        return "\n".join(doc_lines) if doc_lines else None

    def _get_attributes(self, node, source: bytes) -> list[str]:
        """Walk backward through siblings to collect #[...] attributes."""
        attrs = []
        sibling = node.prev_named_sibling
        while sibling is not None:
            if sibling.type == "attribute_item":
                attrs.insert(0, node_text(sibling, source))
                sibling = sibling.prev_named_sibling
                continue
            elif sibling.type == "line_comment":
                sibling = sibling.prev_named_sibling
                continue
            break
        return attrs

    def _has_pyclass(self, attrs: list[str]) -> bool:
        return any("#[pyclass" in a for a in attrs)

    def _is_pymethods_block(self, attrs: list[str]) -> bool:
        return any("#[pymethods]" in a for a in attrs)

    def _is_pymethod_fn(self, fn_attrs: list[str], impl_is_pymethods: bool) -> bool:
        if impl_is_pymethods:
            return True
        return any(m in a for a in fn_attrs for m in ("#[pyfunction]", "#[new]"))

    def _get_return_type(self, node, source: bytes) -> str | None:
        saw_arrow = False
        for child in node.children:
            if not child.is_named and node_text(child, source) == "->":
                saw_arrow = True
            elif saw_arrow and child.type != "block":
                return node_text(child, source)
        return None

    def _get_signature(self, node, source: bytes) -> str:
        parts = []
        for child in node.children:
            if child.type == "block":
                break
            parts.append(node_text(child, source))
        return " ".join(parts)

    def _is_async_fn(self, node, source: bytes) -> bool:
        for child in node.children:
            if not child.is_named and node_text(child, source) == "async":
                return True
            if child.type == "identifier" or node_text(child, source) == "fn":
                break
        return False

    def _get_name(self, node, source: bytes, name_type: str = "identifier") -> str | None:
        for child in node.children:
            if child.type == name_type:
                return node_text(child, source)
        return None

    def _extract_calls(self, body_node, source: bytes) -> list[str]:
        """Recursively extract function/method names called within a block."""
        calls = []

        def walk(node):
            if node.type == "call_expression":
                func = node.child_by_field_name("function")
                if func is None and node.children:
                    func = node.children[0]
                if func:
                    if func.type == "identifier":
                        calls.append(node_text(func, source))
                    elif func.type == "field_expression":
                        field = func.child_by_field_name("field")
                        if field is None:
                            for c in func.children:
                                if c.type == "field_identifier":
                                    field = c
                                    break
                        if field:
                            calls.append(node_text(field, source))
                    elif func.type == "scoped_identifier":
                        name = func.child_by_field_name("name")
                        if name:
                            calls.append(node_text(name, source))
                        else:
                            parts = node_text(func, source).split("::")
                            if parts:
                                calls.append(parts[-1])
            for child in node.children:
                walk(child)

        walk(body_node)
        return calls

    def _file_to_module_path(self, filepath: Path, src_root: Path) -> str:
        rel = filepath.relative_to(src_root)
        parts = list(rel.parts)
        parts[-1] = parts[-1].replace(".rs", "")
        if parts[-1] in ("mod", "lib"):
            parts = parts[:-1]
        if not parts:
            return "crate"
        return "crate::" + "::".join(parts)

    def _extract_struct_fields(self, node, source: bytes,
                                owner_qname: str, rel_path: str) -> list:
        """Extract fields from a struct's field_declaration_list."""
        attrs = []
        for child in node.children:
            if child.type == "field_declaration_list":
                for field in child.children:
                    if field.type == "field_declaration":
                        name = None
                        type_ann = None
                        vis = "private"
                        saw_colon = False
                        for fc in field.children:
                            if fc.type == "visibility_modifier":
                                text = node_text(fc, source)
                                vis = "pub(crate)" if "crate" in text else "pub"
                            elif fc.type in ("field_identifier", "identifier") and not saw_colon:
                                name = node_text(fc, source)
                            elif not fc.is_named and node_text(fc, source) == ":":
                                saw_colon = True
                            elif saw_colon and type_ann is None and fc.is_named:
                                type_ann = node_text(fc, source)
                        if name:
                            attrs.append(AttributeInfo(
                                name=name,
                                qualified_name=f"{owner_qname}::{name}",
                                owner_qualified_name=owner_qname,
                                type_annotation=type_ann,
                                visibility=vis,
                                file_path=rel_path,
                                line_number=field.start_point[0] + 1,
                            ))
        return attrs

    def _get_enum_variants(self, node, source: bytes) -> list[str]:
        """Extract variant names from an enum_variant_list."""
        variants = []
        for child in node.children:
            if child.type == "enum_variant_list":
                for variant in child.children:
                    if variant.type == "enum_variant":
                        name = self._get_name(variant, source, "identifier")
                        if name:
                            variants.append(name)
        return variants

    # ── Parsing ─────────────────────────────────────────────────────────

    def _parse_function(self, node, source: bytes, module_path: str,
                        file_path: str, is_method: bool = False,
                        owner: str | None = None,
                        impl_is_pymethods: bool = False) -> FunctionInfo:
        name = self._get_name(node, source, "identifier") or "unknown"
        prefix = f"{module_path}::{owner}" if owner else module_path
        qualified_name = f"{prefix}::{name}"
        attrs = self._get_attributes(node, source)

        body = None
        for child in node.children:
            if child.type == "block":
                body = child
                break

        is_pymethod = self._is_pymethod_fn(attrs, impl_is_pymethods)
        visibility = self._get_visibility(node)
        if is_pymethod and visibility == "private":
            visibility = "pub(py)"

        return FunctionInfo(
            name=name,
            qualified_name=qualified_name,
            visibility=visibility,
            is_async=self._is_async_fn(node, source),
            is_method=is_method,
            signature=self._get_signature(node, source),
            file_path=file_path,
            line_number=node.start_point[0] + 1,
            docstring=self._get_doc_comment(node, source),
            return_type=self._get_return_type(node, source),
            calls=self._extract_calls(body, source) if body else [],
            metadata={"is_pymethod": is_pymethod},
        )

    def parse_file(self, filepath: Path, src_root: Path) -> ParseResult:
        source = filepath.read_bytes()
        tree = self._parser.parse(source)
        root = tree.root_node

        rel_path = str(filepath.relative_to(src_root))
        module_path = self._file_to_module_path(filepath, src_root)
        loc = count_lines(source)

        file_info = FileInfo(
            path=rel_path,
            filename=filepath.name,
            loc=loc,
            module_path=module_path,
            language="rust",
        )

        result = ParseResult()
        result.files.append(file_info)

        for child in root.children:
            if child.type == "function_item":
                result.functions.append(self._parse_function(
                    child, source, module_path, rel_path,
                    is_method=False, owner=None, impl_is_pymethods=False,
                ))

            elif child.type == "struct_item":
                name = self._get_name(child, source, "type_identifier") or "unknown"
                attrs = self._get_attributes(child, source)
                qname = f"{module_path}::{name}"
                is_pyclass = self._has_pyclass(attrs)
                visibility = self._get_visibility(child)
                if is_pyclass and visibility == "private":
                    visibility = "pub(py)"
                result.classes.append(ClassInfo(
                    name=name,
                    qualified_name=qname,
                    kind="struct",
                    visibility=visibility,
                    file_path=rel_path,
                    line_number=child.start_point[0] + 1,
                    docstring=self._get_doc_comment(child, source),
                    metadata={"is_pyclass": is_pyclass},
                ))
                # Extract struct fields as attributes
                result.attributes.extend(
                    self._extract_struct_fields(child, source, qname, rel_path)
                )

            elif child.type == "enum_item":
                name = self._get_name(child, source, "type_identifier") or "unknown"
                result.enums.append(EnumInfo(
                    name=name,
                    qualified_name=f"{module_path}::{name}",
                    visibility=self._get_visibility(child),
                    file_path=rel_path,
                    line_number=child.start_point[0] + 1,
                    docstring=self._get_doc_comment(child, source),
                    variants=self._get_enum_variants(child, source),
                ))

            elif child.type == "trait_item":
                name = self._get_name(child, source, "type_identifier") or "unknown"
                result.interfaces.append(InterfaceInfo(
                    name=name,
                    qualified_name=f"{module_path}::{name}",
                    kind="trait",
                    visibility=self._get_visibility(child),
                    file_path=rel_path,
                    line_number=child.start_point[0] + 1,
                    docstring=self._get_doc_comment(child, source),
                ))
                # Extract trait method signatures as functions
                for tc in child.children:
                    if tc.type == "declaration_list":
                        for item in tc.children:
                            if item.type in ("function_item", "function_signature_item"):
                                result.functions.append(self._parse_function(
                                    item, source, module_path, rel_path,
                                    is_method=True, owner=name,
                                    impl_is_pymethods=False,
                                ))

            elif child.type == "impl_item":
                attrs = self._get_attributes(child, source)
                pymethods = self._is_pymethods_block(attrs)

                type_ids = [c for c in child.children
                            if c.type == "type_identifier"]
                has_for = any(
                    not c.is_named and node_text(c, source) == "for"
                    for c in child.children
                )

                if has_for and len(type_ids) >= 2:
                    trait_name = node_text(type_ids[0], source)
                    self_type = node_text(type_ids[1], source)
                elif type_ids:
                    trait_name = None
                    self_type = node_text(type_ids[0], source)
                else:
                    continue

                if trait_name:
                    relationship = "implements"
                else:
                    relationship = "inherent"

                type_rel = TypeRelationship(
                    source_type=self_type,
                    target_type=trait_name,
                    relationship=relationship,
                )

                for ic in child.children:
                    if ic.type == "declaration_list":
                        for item in ic.children:
                            if item.type == "function_item":
                                fn_info = self._parse_function(
                                    item, source, module_path, rel_path,
                                    is_method=True, owner=self_type,
                                    impl_is_pymethods=pymethods,
                                )
                                type_rel.methods.append(fn_info)
                                result.functions.append(fn_info)

                result.type_relationships.append(type_rel)

            elif child.type == "use_declaration":
                path_text = None
                for uc in child.children:
                    if uc.type in ("scoped_identifier", "use_wildcard",
                                   "scoped_use_list", "identifier"):
                        path_text = node_text(uc, source)
                if path_text and path_text.startswith("crate::"):
                    file_info.imports.append(path_text)

            elif child.type == "mod_item":
                mod_name = self._get_name(child, source, "identifier")
                if mod_name:
                    file_info.submodule_declarations.append(mod_name)

            elif child.type == "type_item":
                name = self._get_name(child, source, "type_identifier")
                if name:
                    saw_eq = False
                    val_text = None
                    for tc in child.children:
                        if not tc.is_named and node_text(tc, source) == "=":
                            saw_eq = True
                        elif saw_eq and tc.is_named:
                            val_text = node_text(tc, source)[:100]
                            break
                    result.constants.append(ConstantInfo(
                        name=name,
                        qualified_name=f"{module_path}::{name}",
                        kind="type_alias",
                        type_annotation=val_text,
                        value_preview=None,
                        visibility=self._get_visibility(child),
                        file_path=rel_path,
                        line_number=child.start_point[0] + 1,
                    ))

            elif child.type in ("const_item", "static_item"):
                name = self._get_name(child, source, "identifier")
                if name:
                    kind = "constant" if child.type == "const_item" else "static"
                    type_ann = None
                    val_text = None
                    saw_colon = False
                    saw_eq = False
                    for tc in child.children:
                        if not tc.is_named and node_text(tc, source) == ":":
                            saw_colon = True
                        elif saw_colon and not saw_eq and tc.is_named:
                            type_ann = node_text(tc, source)
                            saw_colon = False
                        elif not tc.is_named and node_text(tc, source) == "=":
                            saw_eq = True
                        elif saw_eq and tc.is_named:
                            val = node_text(tc, source)
                            val_text = val[:100]
                            break
                    result.constants.append(ConstantInfo(
                        name=name,
                        qualified_name=f"{module_path}::{name}",
                        kind=kind,
                        type_annotation=type_ann,
                        value_preview=val_text,
                        visibility=self._get_visibility(child),
                        file_path=rel_path,
                        line_number=child.start_point[0] + 1,
                    ))

        return result
