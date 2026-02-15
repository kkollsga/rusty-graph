"""TypeScript and JavaScript parsers using tree-sitter."""

from pathlib import Path
from tree_sitter import Language, Parser

from .base import LanguageParser, node_text, count_lines
from .models import (
    ParseResult, FileInfo, FunctionInfo, ClassInfo,
    EnumInfo, InterfaceInfo, TypeRelationship,
)


def _get_ts_language():
    import tree_sitter_typescript as ts_typescript
    return Language(ts_typescript.language_typescript())


def _get_tsx_language():
    import tree_sitter_typescript as ts_typescript
    return Language(ts_typescript.language_tsx())


def _get_js_language():
    import tree_sitter_javascript as ts_javascript
    return Language(ts_javascript.language())


JSTS_NOISE_NAMES: frozenset[str] = frozenset({
    # Array methods
    "push", "pop", "shift", "unshift", "map", "filter", "reduce",
    "forEach", "find", "findIndex", "some", "every", "includes",
    "indexOf", "slice", "splice", "concat", "join", "flat", "flatMap",
    "sort", "reverse",
    # Object methods
    "keys", "values", "entries", "assign", "freeze",
    "hasOwnProperty", "toString", "valueOf",
    # String methods
    "trim", "split", "replace", "match", "test", "search",
    "startsWith", "endsWith", "substring", "toLowerCase", "toUpperCase",
    # Promise methods
    "then", "catch", "finally", "resolve", "reject",
    # Console methods
    "log", "warn", "error", "info", "debug",
    # DOM / common
    "addEventListener", "removeEventListener", "querySelector",
    "getElementById", "createElement",
})


class _BaseJSTSParser(LanguageParser):
    """Shared logic for TypeScript and JavaScript parsing."""

    @property
    def noise_names(self) -> frozenset[str]:
        return JSTS_NOISE_NAMES

    def __init__(self, language):
        self._parser = Parser(language)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_name(self, node, source: bytes,
                  name_type: str = "identifier") -> str:
        """Get the first identifier-like child as the item name."""
        for child in node.children:
            if child.type in (name_type, "type_identifier",
                              "property_identifier"):
                return node_text(child, source)
        return "unknown"

    def _get_visibility(self, node, source: bytes) -> str:
        """Determine visibility: 'export' if exported, else 'private'."""
        # Check if this node or its parent is an export_statement
        parent = node.parent
        if parent and parent.type == "export_statement":
            return "export"
        # Check for accessibility modifiers inside the node
        for child in node.children:
            if child.type == "accessibility_modifier":
                text = node_text(child, source)
                if text == "private":
                    return "private"
                elif text == "protected":
                    return "protected"
                return "public"
        return "private"

    def _get_block(self, node):
        """Get the body/block of a definition."""
        for child in node.children:
            if child.type in ("statement_block", "class_body"):
                return child
        return None

    def _get_docstring(self, node, source: bytes) -> str | None:
        """Extract JSDoc comment before a node."""
        sibling = node.prev_named_sibling
        if sibling and sibling.type == "comment":
            text = node_text(sibling, source).strip()
            if text.startswith("/**"):
                # Strip /** ... */ delimiters and leading * on each line
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

    def _get_heritage(self, node, source: bytes) -> tuple[list[str], list[str]]:
        """Extract extends and implements from a class/interface declaration.

        Returns (extends_list, implements_list)
        """
        extends = []
        implements = []
        current_list = None

        for child in node.children:
            if child.type == "extends_clause":
                current_list = extends
                for sub in child.children:
                    if sub.type in ("identifier", "type_identifier"):
                        current_list.append(node_text(sub, source))
                    elif sub.type == "member_expression":
                        current_list.append(node_text(sub, source))
            elif child.type == "implements_clause":
                current_list = implements
                for sub in child.children:
                    if sub.type in ("identifier", "type_identifier"):
                        current_list.append(node_text(sub, source))
                    elif sub.type == "member_expression":
                        current_list.append(node_text(sub, source))
                    elif sub.type == "generic_type":
                        # e.g. Iterable<T> -> extract "Iterable"
                        for inner in sub.children:
                            if inner.type in ("identifier", "type_identifier"):
                                current_list.append(node_text(inner, source))
                                break

        return extends, implements

    def _get_signature(self, node, source: bytes) -> str:
        """Extract function/method signature (everything before the body)."""
        parts = []
        for child in node.children:
            if child.type in ("statement_block", "class_body"):
                break
            parts.append(node_text(child, source))
        return " ".join(parts)

    def _get_return_type(self, node, source: bytes) -> str | None:
        """Extract return type annotation from a function."""
        for child in node.children:
            if child.type == "type_annotation":
                # Skip the ":"  prefix
                text = node_text(child, source)
                if text.startswith(":"):
                    text = text[1:].strip()
                return text
        return None

    def _is_async(self, node, source: bytes) -> bool:
        for child in node.children:
            if not child.is_named and node_text(child, source) == "async":
                return True
            if not child.is_named and node_text(child, source) in ("function", "("):
                break
        return False

    def _extract_calls(self, body_node, source: bytes) -> list[str]:
        """Recursively extract function/method names called within a block."""
        calls = []

        def walk(node):
            if node.type == "call_expression":
                func = node.children[0] if node.children else None
                if func:
                    if func.type == "identifier":
                        calls.append(node_text(func, source))
                    elif func.type == "member_expression":
                        # obj.method() -> extract "method"
                        prop = func.child_by_field_name("property")
                        if prop:
                            calls.append(node_text(prop, source))
                        else:
                            for child in reversed(func.children):
                                if child.type in ("property_identifier",
                                                   "identifier"):
                                    calls.append(node_text(child, source))
                                    break
            elif node.type == "new_expression":
                # new Foo() -> extract "Foo"
                for child in node.children:
                    if child.type == "identifier":
                        calls.append(node_text(child, source))
                        break
            for child in node.children:
                walk(child)

        walk(body_node)
        return calls

    def _file_to_module_path(self, filepath: Path, src_root: Path) -> str:
        rel = filepath.relative_to(src_root)
        parts = list(rel.parts)
        # Strip extension from last part
        for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs"):
            if parts[-1].endswith(ext):
                parts[-1] = parts[-1][:-len(ext)]
                break
        # index files represent the directory
        if parts[-1] == "index":
            parts = parts[:-1]
        if not parts:
            return src_root.name
        return "/".join(parts)

    def _get_enum_members(self, node, source: bytes) -> list[str]:
        """Extract member names from an enum body."""
        members = []
        body = self._get_block(node)
        if body is None:
            # Try enum_body specifically
            for child in node.children:
                if child.type == "enum_body":
                    body = child
                    break
        if body:
            for child in body.children:
                if child.type in ("enum_assignment", "property_identifier"):
                    members.append(node_text(child, source).split("=")[0].strip())
                elif child.type == "identifier":
                    members.append(node_text(child, source))
        return members

    # ── Parsing ─────────────────────────────────────────────────────────

    def _parse_function(self, node, source: bytes, module_path: str,
                        rel_path: str, is_method: bool = False,
                        owner: str | None = None) -> FunctionInfo:
        name = self._get_name(node, source)
        sep = "/"
        if owner:
            prefix = f"{module_path}.{owner}"
        else:
            prefix = module_path
        qualified_name = f"{prefix}.{name}"

        block = self._get_block(node)

        return FunctionInfo(
            name=name,
            qualified_name=qualified_name,
            visibility=self._get_visibility(node, source),
            is_async=self._is_async(node, source),
            is_method=is_method,
            signature=self._get_signature(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            docstring=self._get_docstring(node, source),
            return_type=self._get_return_type(node, source),
            calls=self._extract_calls(block, source) if block else [],
        )

    def _parse_class(self, node, source: bytes, module_path: str,
                     rel_path: str, result: ParseResult):
        """Parse a class_declaration node."""
        name = self._get_name(node, source)
        qualified_name = f"{module_path}.{name}"
        extends, implements = self._get_heritage(node, source)
        docstring = self._get_docstring(node, source)

        result.classes.append(ClassInfo(
            name=name,
            qualified_name=qualified_name,
            kind="class",
            visibility=self._get_visibility(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            docstring=docstring,
            bases=extends,
        ))

        # EXTENDS edges
        for base in extends:
            result.type_relationships.append(TypeRelationship(
                source_type=name,
                target_type=base,
                relationship="extends",
            ))

        # IMPLEMENTS edges
        for iface in implements:
            result.type_relationships.append(TypeRelationship(
                source_type=name,
                target_type=iface,
                relationship="implements",
            ))

        # Parse methods
        method_type_rel = TypeRelationship(
            source_type=qualified_name,
            target_type=None,
            relationship="inherent",
        )

        body = self._get_block(node)
        if body:
            for child in body.children:
                fn_node = None
                if child.type in ("method_definition", "public_field_definition"):
                    if child.type == "method_definition":
                        fn_node = child
                elif child.type == "function_declaration":
                    fn_node = child

                if fn_node:
                    fn = self._parse_function(
                        fn_node, source, module_path, rel_path,
                        is_method=True, owner=name,
                    )
                    result.functions.append(fn)
                    method_type_rel.methods.append(fn)

        if method_type_rel.methods:
            result.type_relationships.append(method_type_rel)

    def _parse_interface(self, node, source: bytes, module_path: str,
                         rel_path: str, result: ParseResult):
        """Parse an interface_declaration node (TypeScript only)."""
        name = self._get_name(node, source)
        qualified_name = f"{module_path}.{name}"
        extends, _ = self._get_heritage(node, source)
        docstring = self._get_docstring(node, source)

        result.interfaces.append(InterfaceInfo(
            name=name,
            qualified_name=qualified_name,
            kind="interface",
            visibility=self._get_visibility(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            docstring=docstring,
        ))

        # Interface extends edges (interface extending another interface)
        for base in extends:
            result.type_relationships.append(TypeRelationship(
                source_type=name,
                target_type=base,
                relationship="implements",
            ))

    def _parse_enum(self, node, source: bytes, module_path: str,
                    rel_path: str, result: ParseResult):
        """Parse an enum_declaration node."""
        name = self._get_name(node, source)
        qualified_name = f"{module_path}.{name}"

        result.enums.append(EnumInfo(
            name=name,
            qualified_name=qualified_name,
            visibility=self._get_visibility(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            docstring=self._get_docstring(node, source),
            variants=self._get_enum_members(node, source),
        ))

    def _parse_top_level(self, node, source: bytes, module_path: str,
                         rel_path: str, result: ParseResult,
                         file_info: FileInfo):
        """Parse a top-level AST node."""
        if node.type in ("function_declaration", "function"):
            result.functions.append(self._parse_function(
                node, source, module_path, rel_path,
            ))
        elif node.type == "class_declaration":
            self._parse_class(node, source, module_path, rel_path, result)
        elif node.type == "interface_declaration":
            self._parse_interface(node, source, module_path, rel_path, result)
        elif node.type == "enum_declaration":
            self._parse_enum(node, source, module_path, rel_path, result)
        elif node.type == "export_statement":
            # Unwrap export and parse inner declaration
            for child in node.children:
                if child.type in ("function_declaration", "class_declaration",
                                   "interface_declaration", "enum_declaration"):
                    self._parse_top_level(
                        child, source, module_path, rel_path, result, file_info,
                    )
        elif node.type == "import_statement":
            # Extract the module path from the import source
            for child in node.children:
                if child.type == "string":
                    path = node_text(child, source).strip("'\"")
                    if not path.startswith("."):
                        file_info.imports.append(path)
                    break
        elif node.type == "lexical_declaration":
            # const foo = () => {} or const foo = function() {}
            for child in node.children:
                if child.type == "variable_declarator":
                    value = child.child_by_field_name("value")
                    if value and value.type in ("arrow_function", "function"):
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            fn = self._parse_function(
                                value, source, module_path, rel_path,
                            )
                            fn.name = node_text(name_node, source)
                            fn.qualified_name = f"{module_path}.{fn.name}"
                            result.functions.append(fn)

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
            language=self.language_name,
        )

        result = ParseResult()
        result.files.append(file_info)

        for child in root.children:
            self._parse_top_level(
                child, source, module_path, rel_path, result, file_info,
            )

        return result


class TypeScriptParser(_BaseJSTSParser):

    @property
    def language_name(self) -> str:
        return "typescript"

    @property
    def file_extensions(self) -> list[str]:
        return [".ts", ".tsx"]

    def __init__(self):
        # Use TypeScript language for .ts files
        # Note: .tsx files might need the TSX language, but the TS parser
        # handles most cases
        super().__init__(_get_ts_language())


class JavaScriptParser(_BaseJSTSParser):

    @property
    def language_name(self) -> str:
        return "javascript"

    @property
    def file_extensions(self) -> list[str]:
        return [".js", ".jsx", ".mjs"]

    def __init__(self):
        super().__init__(_get_js_language())
