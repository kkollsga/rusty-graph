"""Java language parser using tree-sitter-java."""

from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_java as ts_java

from .base import LanguageParser, node_text, count_lines, get_type_parameters, extract_comment_annotations
from .models import (
    ParseResult, FileInfo, FunctionInfo, ClassInfo,
    EnumInfo, InterfaceInfo, TypeRelationship,
    AttributeInfo, ConstantInfo,
)

JAVA_LANGUAGE = Language(ts_java.language())

JAVA_NOISE_NAMES: frozenset[str] = frozenset({
    # Object methods
    "toString", "equals", "hashCode", "compareTo", "getClass",
    "notify", "notifyAll", "wait", "clone", "finalize",
    # Collection methods
    "length", "size", "get", "set", "add", "remove", "contains",
    "isEmpty", "put", "keySet", "values", "entrySet",
    "iterator", "next", "hasNext", "clear",
    # Stream API
    "stream", "map", "filter", "collect", "forEach", "reduce",
    "flatMap", "sorted", "distinct", "limit",
    # I/O
    "println", "printf", "print", "format",
    "read", "write", "close", "flush",
})


class JavaParser(LanguageParser):

    @property
    def language_name(self) -> str:
        return "java"

    @property
    def file_extensions(self) -> list[str]:
        return [".java"]

    @property
    def noise_names(self) -> frozenset[str]:
        return JAVA_NOISE_NAMES

    def __init__(self):
        self._parser = Parser(JAVA_LANGUAGE)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_visibility(self, node, source: bytes) -> str:
        """Extract visibility from modifiers node."""
        for child in node.children:
            if child.type == "modifiers":
                text = node_text(child, source)
                if "public" in text:
                    return "public"
                elif "protected" in text:
                    return "protected"
                elif "private" in text:
                    return "private"
                return "package-private"
        return "package-private"

    def _has_modifier(self, node, source: bytes, modifier: str) -> bool:
        """Check if a node has a specific modifier (static, abstract, final, etc.)."""
        for child in node.children:
            if child.type == "modifiers":
                for sub in child.children:
                    if node_text(sub, source) == modifier:
                        return True
        return False

    def _get_annotations(self, node, source: bytes) -> list[str]:
        """Extract annotation names from the modifiers node."""
        annotations: list[str] = []
        for child in node.children:
            if child.type == "modifiers":
                for sub in child.children:
                    if sub.type in ("annotation", "marker_annotation"):
                        # Strip the @
                        text = node_text(sub, source)
                        if text.startswith("@"):
                            text = text[1:]
                        annotations.append(text)
        return annotations

    def _get_doc_comment(self, node, source: bytes) -> str | None:
        """Extract Javadoc comment (/** ... */) before a node."""
        sibling = node.prev_named_sibling
        if sibling and sibling.type in ("comment", "block_comment"):
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
                return "\n".join(lines).strip()
        return None

    def _get_name(self, node, source: bytes,
                  name_type: str = "identifier") -> str | None:
        for child in node.children:
            if child.type == name_type:
                return node_text(child, source)
        return None

    def _get_signature(self, node, source: bytes) -> str:
        """Extract method signature (everything before the body)."""
        parts = []
        for child in node.children:
            if child.type in ("block", "constructor_body"):
                break
            parts.append(node_text(child, source))
        return " ".join(parts)

    def _get_return_type(self, node, source: bytes) -> str | None:
        """Extract return type from a method declaration.

        In Java, the return type comes before the method name.
        """
        # Look for type nodes that appear before the identifier
        for child in node.children:
            if child.type == "identifier":
                break
            if child.type in ("void_type", "integral_type", "boolean_type",
                              "floating_point_type", "type_identifier",
                              "generic_type", "array_type",
                              "scoped_type_identifier"):
                return node_text(child, source)
        return None

    def _extract_calls(self, body_node, source: bytes) -> list[tuple[str, int]]:
        """Recursively extract function/method names called within a block.

        Emits qualified calls where possible: "receiver.method" for
        method invocations on objects, bare names for this/super calls.
        Returns list of (call_name, line_number) tuples.
        """
        calls: list[tuple[str, int]] = []

        def walk(node):
            if node.type == "method_invocation":
                line = node.start_point[0] + 1
                name = node.child_by_field_name("name")
                obj = node.child_by_field_name("object")
                if name and obj:
                    method_name = node_text(name, source)
                    obj_text = node_text(obj, source)
                    hint = obj_text.rsplit(".", 1)[-1]
                    if hint in ("this", "super"):
                        calls.append((method_name, line))
                    else:
                        calls.append((f"{hint}.{method_name}", line))
                elif name:
                    calls.append((node_text(name, source), line))
            elif node.type == "object_creation_expression":
                line = node.start_point[0] + 1
                type_node = node.child_by_field_name("type")
                if type_node:
                    calls.append((node_text(type_node, source), line))
            for child in node.children:
                walk(child)

        walk(body_node)
        return calls

    def _get_package(self, root, source: bytes) -> str:
        """Extract package name from package_declaration."""
        for child in root.children:
            if child.type == "package_declaration":
                for sub in child.children:
                    if sub.type == "scoped_identifier":
                        return node_text(sub, source)
                    elif sub.type == "identifier":
                        return node_text(sub, source)
        return ""

    def _file_to_module_path(self, filepath: Path, src_root: Path,
                              package: str) -> str:
        if package:
            return package
        # Fallback: use directory structure
        rel = filepath.relative_to(src_root)
        parts = list(rel.parent.parts) if rel.parent != Path(".") else []
        return ".".join(parts) if parts else filepath.stem

    def _get_superclass(self, node, source: bytes) -> str | None:
        """Extract extends clause from class."""
        for child in node.children:
            if child.type == "superclass":
                for sub in child.children:
                    if sub.type == "type_identifier":
                        return node_text(sub, source)
                    elif sub.type == "generic_type":
                        for inner in sub.children:
                            if inner.type == "type_identifier":
                                return node_text(inner, source)
                                break
        return None

    def _get_interfaces(self, node, source: bytes) -> list[str]:
        """Extract implements/extends interfaces from class/interface."""
        interfaces: list[str] = []
        for child in node.children:
            if child.type in ("super_interfaces", "extends_interfaces"):
                for sub in child.children:
                    if sub.type == "type_list":
                        for t in sub.children:
                            if t.type == "type_identifier":
                                interfaces.append(node_text(t, source))
                            elif t.type == "generic_type":
                                for inner in t.children:
                                    if inner.type == "type_identifier":
                                        interfaces.append(
                                            node_text(inner, source))
                                        break
        return interfaces

    def _get_enum_constants(self, node, source: bytes) -> list[str]:
        """Extract enum constant names."""
        constants: list[str] = []
        for child in node.children:
            if child.type == "enum_body":
                for sub in child.children:
                    if sub.type == "enum_constant":
                        name = self._get_name(sub, source)
                        if name:
                            constants.append(name)
        return constants

    # ── Parsing ─────────────────────────────────────────────────────────

    def _parse_method(self, node, source: bytes, module_path: str,
                      rel_path: str, owner: str | None = None) -> FunctionInfo:
        if node.type == "constructor_declaration":
            name = self._get_name(node, source) or "unknown"
        else:
            name = self._get_name(node, source) or "unknown"

        if owner:
            prefix = f"{module_path}.{owner}"
        else:
            prefix = module_path
        qualified_name = f"{prefix}.{name}"

        body = None
        for child in node.children:
            if child.type in ("block", "constructor_body"):
                body = child
                break

        metadata: dict = {}
        if self._has_modifier(node, source, "static"):
            metadata["is_static"] = True
        if self._has_modifier(node, source, "abstract"):
            metadata["is_abstract"] = True
        if self._has_modifier(node, source, "native"):
            metadata["is_ffi"] = True
            metadata["ffi_kind"] = "jni"

        return FunctionInfo(
            name=name,
            qualified_name=qualified_name,
            visibility=self._get_visibility(node, source),
            is_async=False,
            is_method=owner is not None,
            signature=self._get_signature(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=self._get_doc_comment(node, source),
            return_type=self._get_return_type(node, source),
            decorators=self._get_annotations(node, source),
            calls=self._extract_calls(body, source) if body else [],
            type_parameters=get_type_parameters(node, source),
            metadata=metadata,
        )

    def _parse_class(self, node, source: bytes, module_path: str,
                     rel_path: str, result: ParseResult,
                     outer_name: str | None = None):
        """Parse a class_declaration node (handles nested classes)."""
        name = self._get_name(node, source) or "unknown"
        if outer_name:
            qualified_name = f"{module_path}.{outer_name}.{name}"
        else:
            qualified_name = f"{module_path}.{name}"

        superclass = self._get_superclass(node, source)
        interfaces = self._get_interfaces(node, source)
        annotations = self._get_annotations(node, source)
        docstring = self._get_doc_comment(node, source)

        bases = [superclass] if superclass else []
        metadata: dict = {}
        if annotations:
            metadata["decorators"] = annotations
        if self._has_modifier(node, source, "abstract"):
            metadata["is_abstract"] = True

        result.classes.append(ClassInfo(
            name=name,
            qualified_name=qualified_name,
            kind="class",
            visibility=self._get_visibility(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=docstring,
            bases=bases,
            type_parameters=get_type_parameters(node, source),
            metadata=metadata,
        ))

        # EXTENDS edge
        if superclass:
            result.type_relationships.append(TypeRelationship(
                source_type=name,
                target_type=superclass,
                relationship="extends",
            ))

        # IMPLEMENTS edges
        for iface in interfaces:
            result.type_relationships.append(TypeRelationship(
                source_type=name,
                target_type=iface,
                relationship="implements",
            ))

        # Parse class body
        self._parse_class_body(node, source, module_path, rel_path,
                               name, qualified_name, result)

    def _parse_class_body(self, node, source: bytes, module_path: str,
                          rel_path: str, class_name: str,
                          class_qname: str, result: ParseResult):
        """Parse methods, fields, and nested classes from a class body."""
        method_type_rel = TypeRelationship(
            source_type=class_qname,
            target_type=None,
            relationship="inherent",
        )

        for child in node.children:
            if child.type == "class_body":
                for item in child.children:
                    if item.type in ("method_declaration",
                                     "constructor_declaration"):
                        fn = self._parse_method(
                            item, source, module_path, rel_path,
                            owner=class_name,
                        )
                        result.functions.append(fn)
                        method_type_rel.methods.append(fn)

                    elif item.type == "field_declaration":
                        self._parse_field(item, source, module_path,
                                          rel_path, class_name,
                                          class_qname, result)

                    elif item.type == "class_declaration":
                        # Nested class
                        self._parse_class(item, source, module_path,
                                          rel_path, result,
                                          outer_name=class_name)

                    elif item.type == "interface_declaration":
                        self._parse_interface(item, source, module_path,
                                              rel_path, result)

                    elif item.type == "enum_declaration":
                        self._parse_enum(item, source, module_path,
                                         rel_path, result)

        if method_type_rel.methods:
            result.type_relationships.append(method_type_rel)

    def _parse_field(self, node, source: bytes, module_path: str,
                     rel_path: str, class_name: str,
                     class_qname: str, result: ParseResult):
        """Parse a field_declaration as AttributeInfo or ConstantInfo."""
        is_static = self._has_modifier(node, source, "static")
        is_final = self._has_modifier(node, source, "final")
        visibility = self._get_visibility(node, source)

        # Extract type
        type_ann = None
        for child in node.children:
            if child.type in ("type_identifier", "integral_type",
                              "boolean_type", "floating_point_type",
                              "void_type", "generic_type", "array_type",
                              "scoped_type_identifier"):
                type_ann = node_text(child, source)
                break

        # Extract variable declarators
        for child in node.children:
            if child.type == "variable_declarator":
                name = self._get_name(child, source)
                if not name:
                    continue

                # Get initializer value
                val_text = None
                for sub in child.children:
                    if sub.type not in ("identifier", "dimensions"):
                        if sub.is_named:
                            val_text = node_text(sub, source)[:100]
                            break

                if is_static and is_final:
                    # static final -> constant
                    result.constants.append(ConstantInfo(
                        name=name,
                        qualified_name=f"{class_qname}.{name}",
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
                        qualified_name=f"{class_qname}.{name}",
                        owner_qualified_name=class_qname,
                        type_annotation=type_ann,
                        visibility=visibility,
                        file_path=rel_path,
                        line_number=node.start_point[0] + 1,
                        default_value=val_text,
                    ))

    def _parse_interface(self, node, source: bytes, module_path: str,
                         rel_path: str, result: ParseResult):
        """Parse an interface_declaration node."""
        name = self._get_name(node, source) or "unknown"
        qualified_name = f"{module_path}.{name}"
        extends = self._get_interfaces(node, source)
        docstring = self._get_doc_comment(node, source)

        result.interfaces.append(InterfaceInfo(
            name=name,
            qualified_name=qualified_name,
            kind="interface",
            visibility=self._get_visibility(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=docstring,
            type_parameters=get_type_parameters(node, source),
        ))

        # Interface extends edges
        for base in extends:
            result.type_relationships.append(TypeRelationship(
                source_type=name,
                target_type=base,
                relationship="extends",
            ))

        # Parse interface method signatures
        iface_rel = TypeRelationship(
            source_type=qualified_name,
            target_type=None,
            relationship="inherent",
        )
        for child in node.children:
            if child.type == "interface_body":
                for item in child.children:
                    if item.type == "method_declaration":
                        fn = self._parse_method(
                            item, source, module_path, rel_path,
                            owner=name,
                        )
                        result.functions.append(fn)
                        iface_rel.methods.append(fn)
        if iface_rel.methods:
            result.type_relationships.append(iface_rel)

    def _parse_enum(self, node, source: bytes, module_path: str,
                    rel_path: str, result: ParseResult):
        """Parse an enum_declaration node."""
        name = self._get_name(node, source) or "unknown"
        qualified_name = f"{module_path}.{name}"

        result.enums.append(EnumInfo(
            name=name,
            qualified_name=qualified_name,
            visibility=self._get_visibility(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=self._get_doc_comment(node, source),
            variants=self._get_enum_constants(node, source),
        ))

    def parse_file(self, filepath: Path, src_root: Path) -> ParseResult:
        source = filepath.read_bytes()
        tree = self._parser.parse(source)
        root = tree.root_node

        rel_path = str(filepath.relative_to(src_root))
        package = self._get_package(root, source)
        module_path = self._file_to_module_path(filepath, src_root, package)
        loc = count_lines(source)

        file_info = FileInfo(
            path=rel_path,
            filename=filepath.name,
            loc=loc,
            module_path=module_path,
            language="java",
        )
        stem = filepath.stem
        if (stem.endswith("Test") or stem.endswith("Tests")
                or "/src/test/" in rel_path
                or rel_path.startswith("src/test/")):
            file_info.is_test = True

        result = ParseResult()
        result.files.append(file_info)

        for child in root.children:
            if child.type == "class_declaration":
                self._parse_class(child, source, module_path, rel_path,
                                  result)
            elif child.type == "interface_declaration":
                self._parse_interface(child, source, module_path, rel_path,
                                      result)
            elif child.type == "enum_declaration":
                self._parse_enum(child, source, module_path, rel_path,
                                 result)
            elif child.type == "import_declaration":
                for sub in child.children:
                    if sub.type == "scoped_identifier":
                        file_info.imports.append(node_text(sub, source))
                        break

        file_info.annotations = extract_comment_annotations(root, source)

        return result
