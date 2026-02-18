"""C# language parser using tree-sitter-c-sharp."""

from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_c_sharp as ts_csharp

from .base import LanguageParser, node_text, count_lines, get_type_parameters, extract_comment_annotations
from .models import (
    ParseResult, FileInfo, FunctionInfo, ClassInfo,
    EnumInfo, InterfaceInfo, TypeRelationship,
    AttributeInfo, ConstantInfo,
)

CSHARP_LANGUAGE = Language(ts_csharp.language())

CSHARP_NOISE_NAMES: frozenset[str] = frozenset({
    # Object methods
    "ToString", "Equals", "GetHashCode", "CompareTo", "GetType",
    "ReferenceEquals", "MemberwiseClone",
    # Collection methods
    "Count", "Add", "Remove", "Contains", "Clear", "Insert",
    "ContainsKey", "TryGetValue", "Keys", "Values",
    "IndexOf", "CopyTo",
    # LINQ
    "Any", "All", "Select", "Where", "FirstOrDefault", "First",
    "LastOrDefault", "Last", "Single", "SingleOrDefault",
    "ToList", "ToArray", "ToDictionary",
    "OrderBy", "OrderByDescending", "GroupBy",
    "Sum", "Max", "Min", "Average", "Aggregate",
    # I/O
    "Write", "WriteLine", "ReadLine", "Read", "Format",
    "Close", "Dispose", "Flush",
})


class CSharpParser(LanguageParser):

    @property
    def language_name(self) -> str:
        return "csharp"

    @property
    def file_extensions(self) -> list[str]:
        return [".cs"]

    @property
    def noise_names(self) -> frozenset[str]:
        return CSHARP_NOISE_NAMES

    def __init__(self):
        self._parser = Parser(CSHARP_LANGUAGE)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_visibility(self, node, source: bytes,
                        default: str = "private") -> str:
        """Extract visibility from modifier keywords."""
        for child in node.children:
            if child.type == "modifier":
                text = node_text(child, source)
                if text in ("public", "private", "protected", "internal"):
                    return text
        # Check for combined modifiers
        mods = set()
        for child in node.children:
            if child.type == "modifier":
                mods.add(node_text(child, source))
        if "protected" in mods and "internal" in mods:
            return "protected internal"
        if "private" in mods and "protected" in mods:
            return "private protected"
        return default

    def _has_modifier(self, node, source: bytes, modifier: str) -> bool:
        for child in node.children:
            if child.type == "modifier" and node_text(child, source) == modifier:
                return True
        return False

    def _get_attributes(self, node, source: bytes) -> list[str]:
        """Extract C# attributes [Foo] from attribute_list nodes."""
        attrs: list[str] = []
        sibling = node.prev_named_sibling
        while sibling is not None:
            if sibling.type == "attribute_list":
                for sub in sibling.children:
                    if sub.type == "attribute":
                        name = self._get_name(sub, source)
                        if name:
                            attrs.insert(0, name)
                sibling = sibling.prev_named_sibling
                continue
            elif sibling.type == "comment":
                sibling = sibling.prev_named_sibling
                continue
            break
        return attrs

    def _get_doc_comment(self, node, source: bytes) -> str | None:
        """Extract XML doc comment (/// ...) before a node."""
        doc_lines: list[str] = []
        sibling = node.prev_named_sibling
        while sibling is not None:
            if sibling.type == "comment":
                text = node_text(sibling, source).strip()
                if text.startswith("///"):
                    content = text[3:]
                    if content.startswith(" "):
                        content = content[1:]
                    doc_lines.insert(0, content)
                    sibling = sibling.prev_named_sibling
                    continue
            elif sibling.type == "attribute_list":
                sibling = sibling.prev_named_sibling
                continue
            break
        return "\n".join(doc_lines) if doc_lines else None

    def _get_name(self, node, source: bytes,
                  name_type: str = "identifier") -> str | None:
        for child in node.children:
            if child.type == name_type:
                return node_text(child, source)
        return None

    def _get_signature(self, node, source: bytes) -> str:
        parts = []
        for child in node.children:
            if child.type in ("block", "arrow_expression_clause"):
                break
            parts.append(node_text(child, source))
        return " ".join(parts)

    def _get_return_type(self, node, source: bytes) -> str | None:
        """Return type appears before the method name in C#."""
        for child in node.children:
            if child.type == "identifier":
                break
            if child.type in ("predefined_type", "type_identifier",
                              "generic_name", "nullable_type",
                              "array_type", "void_keyword",
                              "qualified_name"):
                text = node_text(child, source)
                if text not in ("public", "private", "protected",
                                "internal", "static", "virtual",
                                "override", "abstract", "async",
                                "sealed", "partial"):
                    return text
        return None

    def _extract_calls(self, body_node, source: bytes) -> list[tuple[str, int]]:
        """Emits qualified calls where possible: "receiver.Method" for
        member access calls, bare names for this/base calls.
        Returns list of (call_name, line_number) tuples."""
        calls: list[tuple[str, int]] = []

        def walk(node):
            if node.type == "invocation_expression":
                line = node.start_point[0] + 1
                func = node.children[0] if node.children else None
                if func:
                    if func.type == "identifier":
                        calls.append((node_text(func, source), line))
                    elif func.type == "member_access_expression":
                        name = func.child_by_field_name("name")
                        expr = func.child_by_field_name("expression")
                        if name and expr:
                            method_name = node_text(name, source)
                            expr_text = node_text(expr, source)
                            hint = expr_text.rsplit(".", 1)[-1]
                            if hint in ("this", "base"):
                                calls.append((method_name, line))
                            else:
                                calls.append((f"{hint}.{method_name}", line))
                        elif name:
                            calls.append((node_text(name, source), line))
                        else:
                            for c in reversed(func.children):
                                if c.type == "identifier":
                                    calls.append((node_text(c, source), line))
                                    break
            elif node.type == "object_creation_expression":
                line = node.start_point[0] + 1
                for child in node.children:
                    if child.type in ("identifier", "type_identifier",
                                      "generic_name"):
                        calls.append((node_text(child, source), line))
                        break
            for child in node.children:
                walk(child)

        walk(body_node)
        return calls

    def _get_namespace(self, root, source: bytes) -> str:
        """Extract namespace from declaration."""
        for child in root.children:
            if child.type == "namespace_declaration":
                for sub in child.children:
                    if sub.type in ("qualified_name", "identifier"):
                        return node_text(sub, source)
            elif child.type == "file_scoped_namespace_declaration":
                for sub in child.children:
                    if sub.type in ("qualified_name", "identifier"):
                        return node_text(sub, source)
        return ""

    def _file_to_module_path(self, filepath: Path, src_root: Path,
                              namespace: str) -> str:
        if namespace:
            return namespace
        rel = filepath.relative_to(src_root)
        parts = list(rel.parent.parts) if rel.parent != Path(".") else []
        return ".".join(parts) if parts else filepath.stem

    def _get_base_types(self, node, source: bytes) -> list[str]:
        """Extract base types from base_list."""
        bases: list[str] = []
        for child in node.children:
            if child.type == "base_list":
                for sub in child.children:
                    if sub.type in ("identifier", "type_identifier",
                                    "generic_name", "qualified_name"):
                        bases.append(node_text(sub, source))
        return bases

    def _get_enum_members(self, node, source: bytes) -> list[str]:
        members: list[str] = []
        for child in node.children:
            if child.type == "enum_member_declaration_list":
                for sub in child.children:
                    if sub.type == "enum_member_declaration":
                        name = self._get_name(sub, source)
                        if name:
                            members.append(name)
        return members

    # ── Parsing ─────────────────────────────────────────────────────────

    def _parse_method(self, node, source: bytes, module_path: str,
                      rel_path: str, owner: str | None = None) -> FunctionInfo:
        name = self._get_name(node, source) or "unknown"
        if owner:
            prefix = f"{module_path}.{owner}"
        else:
            prefix = module_path
        qualified_name = f"{prefix}.{name}"

        body = None
        for child in node.children:
            if child.type in ("block", "arrow_expression_clause"):
                body = child
                break

        metadata: dict = {}
        if self._has_modifier(node, source, "static"):
            metadata["is_static"] = True
        if self._has_modifier(node, source, "abstract"):
            metadata["is_abstract"] = True
        if self._has_modifier(node, source, "virtual"):
            metadata["is_virtual"] = True
        if self._has_modifier(node, source, "override"):
            metadata["is_override"] = True
        if self._has_modifier(node, source, "extern"):
            metadata["is_ffi"] = True
            attrs = self._get_attributes(node, source)
            if any("DllImport" in a for a in attrs):
                metadata["ffi_kind"] = "pinvoke"
            else:
                metadata["ffi_kind"] = "extern"

        return FunctionInfo(
            name=name,
            qualified_name=qualified_name,
            visibility=self._get_visibility(node, source),
            is_async=self._has_modifier(node, source, "async"),
            is_method=owner is not None,
            signature=self._get_signature(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            docstring=self._get_doc_comment(node, source),
            return_type=self._get_return_type(node, source),
            decorators=self._get_attributes(node, source),
            calls=self._extract_calls(body, source) if body else [],
            type_parameters=get_type_parameters(node, source, "type_parameter_list"),
            metadata=metadata,
            end_line=node.end_point[0] + 1,
        )

    def _parse_type_declaration(self, node, source: bytes, module_path: str,
                                 rel_path: str, result: ParseResult,
                                 outer_name: str | None = None):
        """Parse class, struct, record, or interface declaration."""
        name = self._get_name(node, source) or "unknown"
        if outer_name:
            qualified_name = f"{module_path}.{outer_name}.{name}"
        else:
            qualified_name = f"{module_path}.{name}"

        base_types = self._get_base_types(node, source)
        attributes = self._get_attributes(node, source)
        docstring = self._get_doc_comment(node, source)

        metadata: dict = {}
        if attributes:
            metadata["decorators"] = attributes
        if self._has_modifier(node, source, "abstract"):
            metadata["is_abstract"] = True
        if self._has_modifier(node, source, "sealed"):
            metadata["is_sealed"] = True
        if self._has_modifier(node, source, "partial"):
            metadata["is_partial"] = True

        if node.type == "interface_declaration":
            result.interfaces.append(InterfaceInfo(
                name=name,
                qualified_name=qualified_name,
                kind="interface",
                visibility=self._get_visibility(node, source,
                                                default="internal"),
                file_path=rel_path,
                line_number=node.start_point[0] + 1,
                docstring=docstring,
                type_parameters=get_type_parameters(node, source, "type_parameter_list"),
                end_line=node.end_point[0] + 1,
            ))
            # Interface extends
            for base in base_types:
                result.type_relationships.append(TypeRelationship(
                    source_type=name,
                    target_type=base,
                    relationship="extends",
                ))
        else:
            # class, struct, record
            if node.type == "struct_declaration":
                kind = "struct"
            else:
                kind = "class"

            result.classes.append(ClassInfo(
                name=name,
                qualified_name=qualified_name,
                kind=kind,
                visibility=self._get_visibility(node, source,
                                                default="internal"),
                file_path=rel_path,
                line_number=node.start_point[0] + 1,
                docstring=docstring,
                bases=base_types[:1],  # First base is the class
                type_parameters=get_type_parameters(node, source, "type_parameter_list"),
                metadata=metadata,
                end_line=node.end_point[0] + 1,
            ))

            # Relationships — first base could be class (extends),
            # rest are interfaces (implements)
            if base_types:
                result.type_relationships.append(TypeRelationship(
                    source_type=name,
                    target_type=base_types[0],
                    relationship="extends",
                ))
                for iface in base_types[1:]:
                    result.type_relationships.append(TypeRelationship(
                        source_type=name,
                        target_type=iface,
                        relationship="implements",
                    ))

        # Parse body
        self._parse_type_body(node, source, module_path, rel_path,
                              name, qualified_name, result)

    def _parse_type_body(self, node, source: bytes, module_path: str,
                         rel_path: str, type_name: str,
                         type_qname: str, result: ParseResult):
        """Parse methods, properties, fields from a type body."""
        method_type_rel = TypeRelationship(
            source_type=type_qname,
            target_type=None,
            relationship="inherent",
        )

        for child in node.children:
            if child.type == "declaration_list":
                for item in child.children:
                    if item.type in ("method_declaration",
                                     "constructor_declaration"):
                        fn = self._parse_method(
                            item, source, module_path, rel_path,
                            owner=type_name,
                        )
                        result.functions.append(fn)
                        method_type_rel.methods.append(fn)

                    elif item.type == "field_declaration":
                        self._parse_field(item, source, rel_path,
                                          type_name, type_qname, result)

                    elif item.type == "property_declaration":
                        self._parse_property(item, source, rel_path,
                                             type_qname, result)

                    elif item.type in ("class_declaration",
                                       "struct_declaration",
                                       "record_declaration",
                                       "interface_declaration"):
                        self._parse_type_declaration(
                            item, source, module_path, rel_path,
                            result, outer_name=type_name)

                    elif item.type == "enum_declaration":
                        self._parse_enum(item, source, module_path,
                                         rel_path, result)

        if method_type_rel.methods:
            result.type_relationships.append(method_type_rel)

    def _parse_field(self, node, source: bytes, rel_path: str,
                     type_name: str, type_qname: str,
                     result: ParseResult):
        """Parse a field_declaration as AttributeInfo or ConstantInfo."""
        is_static = self._has_modifier(node, source, "static")
        is_const = self._has_modifier(node, source, "const")
        is_readonly = self._has_modifier(node, source, "readonly")
        visibility = self._get_visibility(node, source)

        # Extract type
        type_ann = None
        for child in node.children:
            if child.type == "variable_declaration":
                for sub in child.children:
                    if sub.type in ("predefined_type", "type_identifier",
                                    "generic_name", "nullable_type",
                                    "array_type", "qualified_name"):
                        type_ann = node_text(sub, source)
                        break
                # Extract variable declarators
                for sub in child.children:
                    if sub.type == "variable_declarator":
                        name = self._get_name(sub, source)
                        if not name:
                            continue
                        val_text = None
                        for inner in sub.children:
                            if inner.type == "equals_value_clause":
                                for v in inner.children:
                                    if v.is_named:
                                        val_text = node_text(v, source)[:100]
                                        break
                                break

                        if is_const or (is_static and is_readonly):
                            result.constants.append(ConstantInfo(
                                name=name,
                                qualified_name=f"{type_qname}.{name}",
                                kind="constant",
                                type_annotation=type_ann,
                                value_preview=val_text,
                                visibility=visibility,
                                file_path=rel_path,
                                line_number=node.start_point[0] + 1,
                            ))
                        else:
                            result.attributes.append(AttributeInfo(
                                name=name,
                                qualified_name=f"{type_qname}.{name}",
                                owner_qualified_name=type_qname,
                                type_annotation=type_ann,
                                visibility=visibility,
                                file_path=rel_path,
                                line_number=node.start_point[0] + 1,
                                default_value=val_text,
                            ))

    def _parse_property(self, node, source: bytes, rel_path: str,
                        type_qname: str, result: ParseResult):
        """Parse a property_declaration as AttributeInfo."""
        name = self._get_name(node, source)
        if not name:
            return
        type_ann = None
        for child in node.children:
            if child.type in ("predefined_type", "type_identifier",
                              "generic_name", "nullable_type",
                              "array_type", "qualified_name"):
                type_ann = node_text(child, source)
                break
        result.attributes.append(AttributeInfo(
            name=name,
            qualified_name=f"{type_qname}.{name}",
            owner_qualified_name=type_qname,
            type_annotation=type_ann,
            visibility=self._get_visibility(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
        ))

    def _parse_enum(self, node, source: bytes, module_path: str,
                    rel_path: str, result: ParseResult):
        name = self._get_name(node, source) or "unknown"
        result.enums.append(EnumInfo(
            name=name,
            qualified_name=f"{module_path}.{name}",
            visibility=self._get_visibility(node, source, default="internal"),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            docstring=self._get_doc_comment(node, source),
            variants=self._get_enum_members(node, source),
            end_line=node.end_point[0] + 1,
        ))

    def _parse_top_level(self, node, source: bytes, module_path: str,
                         rel_path: str, result: ParseResult,
                         file_info: FileInfo):
        """Parse top-level declarations, handling namespace nesting."""
        if node.type in ("class_declaration", "struct_declaration",
                         "record_declaration", "interface_declaration"):
            self._parse_type_declaration(node, source, module_path,
                                         rel_path, result)
        elif node.type == "enum_declaration":
            self._parse_enum(node, source, module_path, rel_path, result)
        elif node.type in ("namespace_declaration",
                           "file_scoped_namespace_declaration"):
            # Extract namespace and recurse into its body
            ns_name = None
            for child in node.children:
                if child.type in ("qualified_name", "identifier"):
                    ns_name = node_text(child, source)
                elif child.type == "declaration_list":
                    ns_path = ns_name if ns_name else module_path
                    for item in child.children:
                        self._parse_top_level(item, source, ns_path,
                                              rel_path, result, file_info)
                else:
                    # file_scoped_namespace: children are siblings
                    if ns_name and child.is_named:
                        self._parse_top_level(child, source, ns_name,
                                              rel_path, result, file_info)
        elif node.type == "using_directive":
            for child in node.children:
                if child.type in ("qualified_name", "identifier"):
                    file_info.imports.append(node_text(child, source))
                    break
        elif node.type == "global_statement":
            # Top-level statements in C# 9+
            for child in node.children:
                if child.is_named:
                    self._parse_top_level(child, source, module_path,
                                          rel_path, result, file_info)

    def parse_file(self, filepath: Path, src_root: Path) -> ParseResult:
        source = filepath.read_bytes()
        tree = self._parser.parse(source)
        root = tree.root_node

        rel_path = str(filepath.relative_to(src_root))
        namespace = self._get_namespace(root, source)
        module_path = self._file_to_module_path(filepath, src_root, namespace)
        loc = count_lines(source)

        file_info = FileInfo(
            path=rel_path,
            filename=filepath.name,
            loc=loc,
            module_path=module_path,
            language="csharp",
        )
        stem = filepath.stem
        if (stem.endswith("Test") or stem.endswith("Tests")
                or "/Tests/" in rel_path or "/.Tests/" in rel_path
                or rel_path.startswith("Tests/")
                or rel_path.startswith("tests/")):
            file_info.is_test = True

        result = ParseResult()
        result.files.append(file_info)

        for child in root.children:
            self._parse_top_level(child, source, module_path, rel_path,
                                  result, file_info)

        file_info.annotations = extract_comment_annotations(root, source)

        return result
