"""Rust language parser using tree-sitter-rust."""

import re
from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_rust as ts_rust

from .base import (
    LanguageParser, node_text, count_lines, get_type_parameters,
    extract_comment_annotations,
)
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
        """Walk backward through siblings to collect /// or /** */ doc comments."""
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
            elif sibling.type == "block_comment":
                text = node_text(sibling, source).strip()
                if text.startswith("/**"):
                    text = text[3:]
                    if text.endswith("*/"):
                        text = text[:-2]
                    lines = []
                    for line in text.split("\n"):
                        line = line.strip()
                        if line.startswith("* "):
                            line = line[2:]
                        elif line.startswith("*"):
                            line = line[1:]
                        lines.append(line)
                    content = "\n".join(lines).strip()
                    if content:
                        doc_lines.insert(0, content)
                    break
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

    _PY_NAME_RE = re.compile(r'name\s*=\s*"([^"]+)"')

    def _has_pyclass(self, attrs: list[str]) -> bool:
        return any("#[pyclass" in a for a in attrs)

    def _extract_py_name(self, attrs: list[str], keyword: str = "#[pyclass") -> str | None:
        """Extract name="X" from #[pyclass(name = "X")] or similar."""
        for a in attrs:
            if keyword in a:
                m = self._PY_NAME_RE.search(a)
                if m:
                    return m.group(1)
        return None

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

    def _extract_type_name_from_node(self, node, source: bytes) -> str | None:
        """Extract the base type name from a type node.

        Handles: type_identifier, generic_type (e.g. Foo<T>),
        and scoped_type_identifier (e.g. crate::module::Foo).
        """
        if node.type == "type_identifier":
            return node_text(node, source)
        elif node.type == "generic_type":
            for child in node.children:
                if child.type == "type_identifier":
                    return node_text(child, source)
                elif child.type == "scoped_type_identifier":
                    return self._extract_type_name_from_node(child, source)
        elif node.type == "scoped_type_identifier":
            for child in reversed(node.children):
                if child.type == "type_identifier":
                    return node_text(child, source)
        return None

    def _extract_calls(self, body_node, source: bytes) -> list[tuple[str, int]]:
        """Recursively extract function/method names called within a block.

        Emits qualified calls where possible: "Receiver.method" for
        field expressions and "Type.method" for scoped identifiers.
        Returns list of (call_name, line_number) tuples.

        Receiver hints use the syntactic field name, not the resolved type.
        For example, ``self.inner.has_index()`` emits ``("inner.has_index", line)``
        because the parser sees the field name ``inner``, not its type
        ``DirGraph``.  Downstream call-edge resolution therefore cannot match
        this to ``DirGraph::has_index`` — resolving field names to types would
        require full type inference, which is out of scope for a tree-sitter
        based parser.
        """
        calls: list[tuple[str, int]] = []

        def walk(node):
            if node.type == "call_expression":
                line = node.start_point[0] + 1
                func = node.child_by_field_name("function")
                if func is None and node.children:
                    func = node.children[0]
                if func:
                    if func.type == "identifier":
                        calls.append((node_text(func, source), line))
                    elif func.type == "field_expression":
                        field = func.child_by_field_name("field")
                        if field is None:
                            for c in func.children:
                                if c.type == "field_identifier":
                                    field = c
                                    break
                        if field:
                            field_name = node_text(field, source)
                            value = func.child_by_field_name("value")
                            if value:
                                val_text = node_text(value, source)
                                hint = val_text.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
                                if hint in ("self", "&self", "Self"):
                                    calls.append((field_name, line))
                                else:
                                    calls.append((f"{hint}.{field_name}", line))
                            else:
                                calls.append((field_name, line))
                    elif func.type == "scoped_identifier":
                        parts = node_text(func, source).split("::")
                        if len(parts) >= 2:
                            calls.append((f"{parts[-2]}.{parts[-1]}", line))
                        elif parts:
                            calls.append((parts[-1], line))
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

    def _get_enum_variants(self, node, source: bytes) -> tuple[list[str], list[dict]]:
        """Extract variant names and structured details from an enum_variant_list."""
        names = []
        details = []
        for child in node.children:
            if child.type == "enum_variant_list":
                for variant in child.children:
                    if variant.type == "enum_variant":
                        name = self._get_name(variant, source, "identifier")
                        if not name:
                            continue
                        names.append(name)
                        detail: dict = {"name": name, "kind": "unit"}
                        for vc in variant.children:
                            if vc.type == "field_declaration_list":
                                # Struct variant: Variant { field: Type }
                                detail["kind"] = "struct"
                                detail["fields"] = self._extract_variant_struct_fields(vc, source)
                            elif vc.type == "ordered_field_declaration_list":
                                # Tuple variant: Variant(Type1, Type2)
                                detail["kind"] = "tuple"
                                detail["fields"] = self._extract_variant_tuple_fields(vc, source)
                        details.append(detail)
        return names, details

    def _extract_variant_struct_fields(self, field_list, source: bytes) -> list[dict]:
        """Extract named fields from a struct enum variant."""
        fields = []
        for child in field_list.children:
            if child.type == "field_declaration":
                name = None
                type_ann = None
                saw_colon = False
                for fc in child.children:
                    if fc.type in ("field_identifier", "identifier") and not saw_colon:
                        name = node_text(fc, source)
                    elif not fc.is_named and node_text(fc, source) == ":":
                        saw_colon = True
                    elif saw_colon and type_ann is None and fc.is_named:
                        type_ann = node_text(fc, source)
                if name:
                    entry: dict = {"name": name}
                    if type_ann:
                        entry["type"] = type_ann
                    fields.append(entry)
        return fields

    def _extract_variant_tuple_fields(self, field_list, source: bytes) -> list[dict]:
        """Extract positional types from a tuple enum variant."""
        fields = []
        for child in field_list.children:
            if child.is_named and child.type != "visibility_modifier":
                fields.append({"type": node_text(child, source)})
        return fields

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
        is_ffi = any("#[no_mangle]" in a for a in attrs)
        is_test = any(a in ("#[test]", "#[bench]") or
                       "#[tokio::test" in a or "#[rstest" in a
                       for a in attrs)
        ffi_kind = None
        if is_pymethod:
            ffi_kind = "pyo3"
        elif is_ffi:
            ffi_kind = "extern_c"
        visibility = self._get_visibility(node)
        if is_pymethod and visibility == "private":
            visibility = "pub(py)"

        is_pymodule = any("#[pymodule]" in a for a in attrs)

        metadata: dict = {"is_pymethod": is_pymethod}
        if is_test:
            metadata["is_test"] = True
        if is_ffi or is_pymethod:
            metadata["is_ffi"] = True
            metadata["ffi_kind"] = ffi_kind
            py_name = self._extract_py_name(attrs, "#[pyfunction")
            if py_name:
                metadata["py_name"] = py_name
        if is_pymodule:
            metadata["is_pymodule"] = True
            metadata["is_ffi"] = True
            metadata["ffi_kind"] = "pyo3"

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
            type_parameters=get_type_parameters(node, source),
            end_line=node.end_point[0] + 1,
            metadata=metadata,
        )

    def _parse_items(self, node, source: bytes, module_path: str,
                     rel_path: str, file_info: FileInfo,
                     result: ParseResult) -> None:
        """Parse all item-level children of a node (top-level or inside mod block)."""
        for child in node.children:
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
                metadata: dict = {"is_pyclass": is_pyclass}
                if is_pyclass:
                    py_name = self._extract_py_name(attrs, "#[pyclass")
                    if py_name:
                        metadata["py_name"] = py_name
                result.classes.append(ClassInfo(
                    name=name,
                    qualified_name=qname,
                    kind="struct",
                    visibility=visibility,
                    file_path=rel_path,
                    line_number=child.start_point[0] + 1,
                    docstring=self._get_doc_comment(child, source),
                    type_parameters=get_type_parameters(child, source),
                    end_line=child.end_point[0] + 1,
                    metadata=metadata,
                ))
                # Extract struct fields as attributes
                result.attributes.extend(
                    self._extract_struct_fields(child, source, qname, rel_path)
                )

            elif child.type == "enum_item":
                name = self._get_name(child, source, "type_identifier") or "unknown"
                variant_names, variant_details = self._get_enum_variants(child, source)
                result.enums.append(EnumInfo(
                    name=name,
                    qualified_name=f"{module_path}::{name}",
                    visibility=self._get_visibility(child),
                    file_path=rel_path,
                    line_number=child.start_point[0] + 1,
                    docstring=self._get_doc_comment(child, source),
                    variants=variant_names,
                    end_line=child.end_point[0] + 1,
                    variant_details=variant_details if variant_details else None,
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
                    type_parameters=get_type_parameters(child, source),
                    end_line=child.end_point[0] + 1,
                ))
                # Extract trait method signatures as functions
                trait_rel = TypeRelationship(
                    source_type=f"{module_path}::{name}",
                    target_type=None,
                    relationship="inherent",
                )
                for tc in child.children:
                    if tc.type == "declaration_list":
                        for item in tc.children:
                            if item.type in ("function_item", "function_signature_item"):
                                fn = self._parse_function(
                                    item, source, module_path, rel_path,
                                    is_method=True, owner=name,
                                    impl_is_pymethods=False,
                                )
                                result.functions.append(fn)
                                trait_rel.methods.append(fn)
                if trait_rel.methods:
                    result.type_relationships.append(trait_rel)

            elif child.type == "impl_item":
                attrs = self._get_attributes(child, source)
                pymethods = self._is_pymethods_block(attrs)

                # Collect type nodes before/after "for" keyword.
                # Handles bare type_identifier, generic_type (e.g. Foo<'a>),
                # and scoped_type_identifier (e.g. crate::mod::Foo).
                seen_for = False
                types_before: list = []
                types_after: list = []
                for c in child.children:
                    if not c.is_named and node_text(c, source) == "for":
                        seen_for = True
                    elif c.type in ("type_identifier", "generic_type",
                                    "scoped_type_identifier"):
                        (types_after if seen_for else types_before).append(c)

                if seen_for and types_before and types_after:
                    trait_name = self._extract_type_name_from_node(
                        types_before[0], source)
                    self_type = self._extract_type_name_from_node(
                        types_after[0], source)
                elif types_before:
                    trait_name = None
                    self_type = self._extract_type_name_from_node(
                        types_before[0], source)
                else:
                    continue

                if self_type is None:
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
                if path_text:
                    file_info.imports.append(path_text)

            elif child.type == "mod_item":
                mod_name = self._get_name(child, source, "identifier")
                if mod_name:
                    # Check for inline module body (mod name { ... })
                    decl_list = None
                    for mc in child.children:
                        if mc.type == "declaration_list":
                            decl_list = mc
                            break
                    if decl_list is not None:
                        # Inline module — recurse with updated module path
                        inner_path = f"{module_path}::{mod_name}"
                        self._parse_items(decl_list, source, inner_path,
                                          rel_path, file_info, result)
                    else:
                        # External module declaration (mod name;)
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
        stem = filepath.stem
        if (stem.endswith("_test") or stem.endswith("_tests")
                or stem.startswith("test_")
                or "/tests/" in rel_path or rel_path.startswith("tests/")
                or "/benches/" in rel_path or rel_path.startswith("benches/")):
            file_info.is_test = True

        result = ParseResult()
        result.files.append(file_info)
        self._parse_items(root, source, module_path, rel_path, file_info, result)
        file_info.annotations = extract_comment_annotations(root, source)
        return result
