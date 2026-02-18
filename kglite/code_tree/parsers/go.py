"""Go language parser using tree-sitter-go."""

from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_go as ts_go

from .base import LanguageParser, node_text, count_lines, extract_comment_annotations, get_type_parameters
from .models import (
    ParseResult, FileInfo, FunctionInfo, ClassInfo,
    EnumInfo, InterfaceInfo, TypeRelationship,
    AttributeInfo, ConstantInfo,
)

GO_LANGUAGE = Language(ts_go.language())

GO_NOISE_NAMES: frozenset[str] = frozenset({
    # Builtins
    "len", "cap", "append", "copy", "delete", "make", "new", "close",
    "panic", "recover", "print", "println",
    # Common interface methods
    "Error", "String",
    # fmt package
    "Format", "Sprintf", "Fprintf", "Errorf", "Printf", "Println",
    "Print", "Fprintln", "Fprint",
    # log package
    "Fatal", "Fatalf", "Fatalln",
})


class GoParser(LanguageParser):

    @property
    def language_name(self) -> str:
        return "go"

    @property
    def file_extensions(self) -> list[str]:
        return [".go"]

    @property
    def noise_names(self) -> frozenset[str]:
        return GO_NOISE_NAMES

    def __init__(self):
        self._parser = Parser(GO_LANGUAGE)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_visibility(self, name: str) -> str:
        """Go uses capitalization: uppercase = exported, lowercase = unexported."""
        if name and name[0].isupper():
            return "exported"
        return "unexported"

    def _get_doc_comment(self, node, source: bytes) -> str | None:
        """Walk backward through siblings to collect // doc comments."""
        doc_lines = []
        sibling = node.prev_named_sibling
        while sibling is not None:
            if sibling.type == "comment":
                text = node_text(sibling, source).strip()
                if text.startswith("//"):
                    content = text[2:]
                    if content.startswith(" "):
                        content = content[1:]
                    doc_lines.insert(0, content)
                    sibling = sibling.prev_named_sibling
                    continue
            break
        return "\n".join(doc_lines) if doc_lines else None

    def _get_name(self, node, source: bytes,
                  name_type: str = "identifier") -> str | None:
        for child in node.children:
            if child.type in (name_type, "type_identifier",
                              "field_identifier"):
                return node_text(child, source)
        return None

    def _get_signature(self, node, source: bytes) -> str:
        """Extract function signature (everything before the body block)."""
        parts = []
        for child in node.children:
            if child.type == "block":
                break
            parts.append(node_text(child, source))
        return " ".join(parts)

    def _get_return_type(self, node, source: bytes) -> str | None:
        """Extract return type from function — the node after parameter_list."""
        saw_params = False
        for child in node.children:
            if child.type == "parameter_list":
                saw_params = True
            elif saw_params and child.type == "block":
                break
            elif saw_params and child.is_named:
                return node_text(child, source)
        return None

    def _is_async(self, node, source: bytes) -> bool:
        """Go doesn't have async keyword — always False."""
        return False

    def _extract_calls(self, body_node, source: bytes) -> list[tuple[str, int]]:
        """Recursively extract function/method names called within a block.

        Emits qualified calls where possible: "receiver.Method" for
        selector expressions, bare names for plain function calls.
        Returns list of (call_name, line_number) tuples.
        """
        calls: list[tuple[str, int]] = []

        def walk(node):
            if node.type == "call_expression":
                line = node.start_point[0] + 1
                func = node.children[0] if node.children else None
                if func:
                    if func.type == "identifier":
                        calls.append((node_text(func, source), line))
                    elif func.type == "selector_expression":
                        field = func.child_by_field_name("field")
                        operand = func.child_by_field_name("operand")
                        if field and operand:
                            field_name = node_text(field, source)
                            op_text = node_text(operand, source)
                            hint = op_text.rsplit(".", 1)[-1]
                            calls.append((f"{hint}.{field_name}", line))
                        elif field:
                            calls.append((node_text(field, source), line))
                        else:
                            for c in func.children:
                                if c.type == "field_identifier":
                                    calls.append((node_text(c, source), line))
                                    break
            for child in node.children:
                walk(child)

        walk(body_node)
        return calls

    def _get_package_name(self, root, source: bytes) -> str:
        """Extract package name from the package clause."""
        for child in root.children:
            if child.type == "package_clause":
                for sub in child.children:
                    if sub.type == "package_identifier":
                        return node_text(sub, source)
        return "main"

    def _file_to_module_path(self, filepath: Path, src_root: Path,
                              package_name: str) -> str:
        rel = filepath.relative_to(src_root)
        parts = list(rel.parent.parts) if rel.parent != Path(".") else []
        if parts:
            return f"{package_name}/{'/'.join(parts)}"
        return package_name

    def _extract_struct_fields(self, node, source: bytes,
                                owner_qname: str,
                                rel_path: str) -> list[AttributeInfo]:
        """Extract fields from a struct type's field_declaration_list."""
        attrs: list[AttributeInfo] = []
        for child in node.children:
            if child.type == "field_declaration_list":
                for field in child.children:
                    if field.type == "field_declaration":
                        names: list[str] = []
                        type_ann = None
                        for fc in field.children:
                            if fc.type == "field_identifier":
                                names.append(node_text(fc, source))
                            elif fc.type == "type_identifier" or (
                                fc.is_named and fc.type not in (
                                    "field_identifier", "tag", "comment")):
                                if type_ann is None and not names:
                                    # Embedded field: type name is also field name
                                    text = node_text(fc, source)
                                    # Strip pointer
                                    if text.startswith("*"):
                                        text = text[1:]
                                    names.append(text)
                                    type_ann = node_text(fc, source)
                                elif type_ann is None:
                                    type_ann = node_text(fc, source)
                        for name in names:
                            attrs.append(AttributeInfo(
                                name=name,
                                qualified_name=f"{owner_qname}.{name}",
                                owner_qualified_name=owner_qname,
                                type_annotation=type_ann,
                                visibility=self._get_visibility(name),
                                file_path=rel_path,
                                line_number=field.start_point[0] + 1,
                            ))
        return attrs

    def _get_receiver_type(self, node, source: bytes) -> str | None:
        """Extract receiver type from a method_declaration."""
        for child in node.children:
            if child.type == "parameter_list":
                # First parameter_list is the receiver
                for param in child.children:
                    if param.type == "parameter_declaration":
                        # Look for the type (last type-like child)
                        for sub in param.children:
                            if sub.type == "type_identifier":
                                return node_text(sub, source)
                            elif sub.type == "pointer_type":
                                for inner in sub.children:
                                    if inner.type == "type_identifier":
                                        return node_text(inner, source)
                break  # Only check first parameter_list
        return None

    def _get_interface_methods(self, node, source: bytes,
                                module_path: str,
                                rel_path: str) -> list[str]:
        """Extract method signature names from an interface type."""
        methods: list[str] = []
        for child in node.children:
            if child.type == "method_spec":
                name = self._get_name(child, source, "field_identifier")
                if name:
                    methods.append(name)
        return methods

    # ── Parsing ─────────────────────────────────────────────────────────

    def _parse_function(self, node, source: bytes, module_path: str,
                        rel_path: str, is_method: bool = False,
                        owner: str | None = None) -> FunctionInfo:
        name = self._get_name(node, source, "identifier") or "unknown"
        if owner:
            prefix = f"{module_path}.{owner}"
        else:
            prefix = module_path
        qualified_name = f"{prefix}.{name}"

        body = None
        for child in node.children:
            if child.type == "block":
                body = child
                break

        return FunctionInfo(
            name=name,
            qualified_name=qualified_name,
            visibility=self._get_visibility(name),
            is_async=False,
            is_method=is_method,
            signature=self._get_signature(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=self._get_doc_comment(node, source),
            return_type=self._get_return_type(node, source),
            calls=self._extract_calls(body, source) if body else [],
            type_parameters=get_type_parameters(node, source, "type_parameter_list"),
        )

    def parse_file(self, filepath: Path, src_root: Path) -> ParseResult:
        source = filepath.read_bytes()
        tree = self._parser.parse(source)
        root = tree.root_node

        rel_path = str(filepath.relative_to(src_root))
        package_name = self._get_package_name(root, source)
        module_path = self._file_to_module_path(filepath, src_root,
                                                 package_name)
        loc = count_lines(source)

        file_info = FileInfo(
            path=rel_path,
            filename=filepath.name,
            loc=loc,
            module_path=module_path,
            language="go",
        )
        if filepath.name.endswith("_test.go"):
            file_info.is_test = True

        result = ParseResult()
        result.files.append(file_info)

        for child in root.children:
            if child.type == "function_declaration":
                result.functions.append(self._parse_function(
                    child, source, module_path, rel_path,
                ))

            elif child.type == "method_declaration":
                receiver_type = self._get_receiver_type(child, source)
                fn = self._parse_function(
                    child, source, module_path, rel_path,
                    is_method=True, owner=receiver_type,
                )
                result.functions.append(fn)
                # Create inherent relationship for method ownership
                if receiver_type:
                    result.type_relationships.append(TypeRelationship(
                        source_type=receiver_type,
                        target_type=None,
                        relationship="inherent",
                        methods=[fn],
                    ))

            elif child.type == "type_declaration":
                for spec in child.children:
                    if spec.type == "type_spec":
                        name = self._get_name(spec, source,
                                              "type_identifier")
                        if not name:
                            continue
                        qname = f"{module_path}.{name}"
                        docstring = self._get_doc_comment(child, source)

                        # Check what kind of type it is
                        # Skip the first type_identifier (that's the name)
                        type_node = None
                        saw_name = False
                        for sub in spec.children:
                            if sub.type == "type_identifier" and not saw_name:
                                saw_name = True
                                continue
                            if sub.type in ("struct_type", "interface_type",
                                            "type_identifier",
                                            "qualified_type",
                                            "pointer_type", "slice_type",
                                            "map_type", "channel_type",
                                            "function_type", "array_type"):
                                type_node = sub
                                break

                        if type_node is None:
                            continue

                        if type_node.type == "struct_type":
                            result.classes.append(ClassInfo(
                                name=name,
                                qualified_name=qname,
                                kind="struct",
                                visibility=self._get_visibility(name),
                                file_path=rel_path,
                                line_number=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                docstring=docstring,
                                type_parameters=get_type_parameters(spec, source, "type_parameter_list"),
                            ))
                            result.attributes.extend(
                                self._extract_struct_fields(
                                    type_node, source, qname, rel_path))

                        elif type_node.type == "interface_type":
                            result.interfaces.append(InterfaceInfo(
                                name=name,
                                qualified_name=qname,
                                kind="interface",
                                visibility=self._get_visibility(name),
                                file_path=rel_path,
                                line_number=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                docstring=docstring,
                                type_parameters=get_type_parameters(spec, source, "type_parameter_list"),
                            ))
                            # Parse interface method specs as functions
                            iface_rel = TypeRelationship(
                                source_type=qname,
                                target_type=None,
                                relationship="inherent",
                            )
                            for ms in type_node.children:
                                if ms.type == "method_spec":
                                    fn_name = self._get_name(
                                        ms, source, "field_identifier")
                                    if fn_name:
                                        fn = FunctionInfo(
                                            name=fn_name,
                                            qualified_name=f"{qname}.{fn_name}",
                                            visibility=self._get_visibility(fn_name),
                                            is_async=False,
                                            is_method=True,
                                            signature=node_text(ms, source),
                                            file_path=rel_path,
                                            line_number=ms.start_point[0] + 1,
                                            end_line=ms.end_point[0] + 1,
                                            docstring=None,
                                            return_type=None,
                                            calls=[],
                                        )
                                        result.functions.append(fn)
                                        iface_rel.methods.append(fn)
                            if iface_rel.methods:
                                result.type_relationships.append(iface_rel)

                        else:
                            # Type alias: type Foo = Bar or type Foo Bar
                            result.constants.append(ConstantInfo(
                                name=name,
                                qualified_name=qname,
                                kind="type_alias",
                                type_annotation=node_text(type_node, source),
                                value_preview=None,
                                visibility=self._get_visibility(name),
                                file_path=rel_path,
                                line_number=child.start_point[0] + 1,
                            ))

            elif child.type == "const_declaration":
                for spec in child.children:
                    if spec.type == "const_spec":
                        name = self._get_name(spec, source)
                        if not name:
                            continue
                        type_ann = None
                        val_text = None
                        for sub in spec.children:
                            if sub.type == "type_identifier":
                                type_ann = node_text(sub, source)
                            elif sub.type == "expression_list":
                                val_text = node_text(sub, source)[:100]
                        result.constants.append(ConstantInfo(
                            name=name,
                            qualified_name=f"{module_path}.{name}",
                            kind="constant",
                            type_annotation=type_ann,
                            value_preview=val_text,
                            visibility=self._get_visibility(name),
                            file_path=rel_path,
                            line_number=spec.start_point[0] + 1,
                        ))

            elif child.type == "var_declaration":
                for spec in child.children:
                    if spec.type == "var_spec":
                        name = self._get_name(spec, source)
                        if not name:
                            continue
                        type_ann = None
                        val_text = None
                        for sub in spec.children:
                            if sub.type == "type_identifier":
                                type_ann = node_text(sub, source)
                            elif sub.type == "expression_list":
                                val_text = node_text(sub, source)[:100]
                        result.constants.append(ConstantInfo(
                            name=name,
                            qualified_name=f"{module_path}.{name}",
                            kind="static",
                            type_annotation=type_ann,
                            value_preview=val_text,
                            visibility=self._get_visibility(name),
                            file_path=rel_path,
                            line_number=spec.start_point[0] + 1,
                        ))

            elif child.type == "import_declaration":
                for spec in child.children:
                    if spec.type == "import_spec":
                        for sub in spec.children:
                            if sub.type == "interpreted_string_literal":
                                path = node_text(sub, source).strip('"')
                                file_info.imports.append(path)
                    elif spec.type == "import_spec_list":
                        for item in spec.children:
                            if item.type == "import_spec":
                                for sub in item.children:
                                    if sub.type == "interpreted_string_literal":
                                        path = node_text(sub, source).strip('"')
                                        file_info.imports.append(path)

        file_info.annotations = extract_comment_annotations(root, source)

        return result
