"""C and C++ parsers using tree-sitter-c and tree-sitter-cpp."""

from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_c as ts_c
import tree_sitter_cpp as ts_cpp

from .base import LanguageParser, node_text, count_lines, get_type_parameters
from .models import (
    ParseResult, FileInfo, FunctionInfo, ClassInfo,
    EnumInfo, InterfaceInfo, TypeRelationship,
    AttributeInfo, ConstantInfo,
)

C_LANGUAGE = Language(ts_c.language())
CPP_LANGUAGE = Language(ts_cpp.language())

C_NOISE_NAMES: frozenset[str] = frozenset({
    # stdio
    "printf", "fprintf", "sprintf", "snprintf", "scanf", "sscanf",
    "puts", "fputs", "fgets", "getchar", "putchar",
    # stdlib
    "malloc", "calloc", "realloc", "free",
    "exit", "abort", "atexit", "atoi", "atof",
    # string
    "memcpy", "memset", "memmove", "memcmp",
    "strlen", "strcmp", "strncmp", "strcpy", "strncpy", "strcat",
    "strstr", "strchr",
    # assert
    "assert",
    # file I/O
    "fopen", "fclose", "fread", "fwrite", "fseek", "ftell",
})

CPP_NOISE_NAMES: frozenset[str] = C_NOISE_NAMES | frozenset({
    # STL containers
    "size", "empty", "begin", "end", "cbegin", "cend",
    "rbegin", "rend",
    "push_back", "pop_back", "emplace_back",
    "push_front", "pop_front", "emplace_front",
    "insert", "erase", "find", "count", "at",
    "front", "back", "data",
    "resize", "reserve", "clear", "swap", "shrink_to_fit",
    # Smart pointers / utility
    "move", "forward", "make_shared", "make_unique", "get", "reset",
    # iostream
    "cout", "cerr", "endl",
    # Casts
    "static_cast", "dynamic_cast", "reinterpret_cast", "const_cast",
})


class _BaseCCppParser(LanguageParser):
    """Shared logic for C and C++ parsing."""

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_name(self, node, source: bytes,
                  name_type: str = "identifier") -> str | None:
        for child in node.children:
            if child.type in (name_type, "type_identifier",
                              "field_identifier"):
                return node_text(child, source)
        return None

    def _get_doc_comment(self, node, source: bytes) -> str | None:
        """Walk backward to collect /** */ or /// doc comments."""
        sibling = node.prev_named_sibling
        if sibling and sibling.type == "comment":
            text = node_text(sibling, source).strip()
            # Block comment: /** ... */
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
            # Line comment: ///
            if text.startswith("///"):
                doc_lines = []
                while sibling is not None and sibling.type == "comment":
                    text = node_text(sibling, source).strip()
                    if text.startswith("///"):
                        content = text[3:]
                        if content.startswith(" "):
                            content = content[1:]
                        doc_lines.insert(0, content)
                        sibling = sibling.prev_named_sibling
                    else:
                        break
                return "\n".join(doc_lines) if doc_lines else None
        return None

    def _get_signature(self, node, source: bytes) -> str:
        parts = []
        for child in node.children:
            if child.type in ("compound_statement", "field_declaration_list"):
                break
            parts.append(node_text(child, source))
        return " ".join(parts)

    def _get_return_type(self, node, source: bytes) -> str | None:
        """Return type is the first type-like child before the declarator."""
        for child in node.children:
            if child.type in ("function_declarator", "identifier",
                              "pointer_declarator"):
                break
            if child.type in ("primitive_type", "type_identifier",
                              "sized_type_specifier", "struct_specifier",
                              "enum_specifier", "union_specifier",
                              "type_qualifier"):
                return node_text(child, source)
        return None

    def _has_storage_class(self, node, source: bytes,
                            specifier: str) -> bool:
        for child in node.children:
            if child.type == "storage_class_specifier":
                if node_text(child, source) == specifier:
                    return True
        return False

    def _extract_calls(self, body_node, source: bytes) -> list[str]:
        """Emits qualified calls where possible: "Receiver.method" for
        field/member access and "Type.method" for scoped identifiers."""
        calls: list[str] = []

        def walk(node):
            if node.type == "call_expression":
                func = node.children[0] if node.children else None
                if func:
                    if func.type == "identifier":
                        calls.append(node_text(func, source))
                    elif func.type == "field_expression":
                        field = func.child_by_field_name("field")
                        argument = func.child_by_field_name("argument")
                        if field and argument:
                            field_name = node_text(field, source)
                            arg_text = node_text(argument, source)
                            hint = arg_text.rsplit(".", 1)[-1].rsplit("->", 1)[-1]
                            calls.append(f"{hint}.{field_name}")
                        elif field:
                            calls.append(node_text(field, source))
                        else:
                            for c in func.children:
                                if c.type == "field_identifier":
                                    calls.append(node_text(c, source))
                                    break
                    elif func.type == "scoped_identifier":
                        parts = node_text(func, source).split("::")
                        if len(parts) >= 2:
                            calls.append(f"{parts[-2]}.{parts[-1]}")
                        elif parts:
                            calls.append(parts[-1])
            for child in node.children:
                walk(child)

        walk(body_node)
        return calls

    def _file_to_module_path(self, filepath: Path, src_root: Path) -> str:
        rel = filepath.relative_to(src_root)
        parts = list(rel.parts)
        # Strip extension from last part
        name = parts[-1]
        for ext in (".c", ".h", ".cpp", ".cc", ".cxx",
                    ".hpp", ".hh", ".hxx"):
            if name.endswith(ext):
                parts[-1] = name[:-len(ext)]
                break
        return "/".join(parts)

    def _extract_struct_fields(self, node, source: bytes,
                                owner_qname: str,
                                rel_path: str) -> list[AttributeInfo]:
        """Extract fields from a struct/union field_declaration_list."""
        attrs: list[AttributeInfo] = []
        for child in node.children:
            if child.type == "field_declaration_list":
                for field in child.children:
                    if field.type == "field_declaration":
                        type_ann = None
                        names: list[str] = []
                        for fc in field.children:
                            if fc.type in ("primitive_type",
                                           "type_identifier",
                                           "sized_type_specifier",
                                           "struct_specifier",
                                           "enum_specifier",
                                           "union_specifier"):
                                type_ann = node_text(fc, source)
                            elif fc.type == "field_identifier":
                                names.append(node_text(fc, source))
                            elif fc.type in ("pointer_declarator",
                                             "array_declarator"):
                                for inner in fc.children:
                                    if inner.type == "field_identifier":
                                        names.append(
                                            node_text(inner, source))
                                        break
                        for name in names:
                            sep = "::" if "::" in owner_qname else "."
                            attrs.append(AttributeInfo(
                                name=name,
                                qualified_name=f"{owner_qname}{sep}{name}",
                                owner_qualified_name=owner_qname,
                                type_annotation=type_ann,
                                visibility="public",
                                file_path=rel_path,
                                line_number=field.start_point[0] + 1,
                            ))
        return attrs

    def _get_enum_variants(self, node, source: bytes) -> list[str]:
        variants: list[str] = []
        for child in node.children:
            if child.type == "enumerator_list":
                for sub in child.children:
                    if sub.type == "enumerator":
                        name = self._get_name(sub, source)
                        if name:
                            variants.append(name)
        return variants

    # ── Common parsing methods ──────────────────────────────────────────

    def _parse_function(self, node, source: bytes, module_path: str,
                        rel_path: str, is_method: bool = False,
                        owner: str | None = None) -> FunctionInfo:
        """Parse a function_definition or declaration node."""
        # Extract name from the declarator
        name = "unknown"
        declarator = None
        for child in node.children:
            if child.type in ("function_declarator", "pointer_declarator"):
                declarator = child
                break

        if declarator:
            # Unwrap pointer_declarator
            while declarator and declarator.type == "pointer_declarator":
                for c in declarator.children:
                    if c.type == "function_declarator":
                        declarator = c
                        break
                else:
                    break
            if declarator:
                fn_name = self._get_name(declarator, source)
                if fn_name:
                    name = fn_name

        sep = "::" if self.language_name == "cpp" else "/"
        if owner:
            prefix = f"{module_path}{sep}{owner}"
        else:
            prefix = module_path
        qualified_name = f"{prefix}{sep}{name}"

        body = None
        for child in node.children:
            if child.type == "compound_statement":
                body = child
                break

        is_static = self._has_storage_class(node, source, "static")
        visibility = "private" if is_static else "public"

        return FunctionInfo(
            name=name,
            qualified_name=qualified_name,
            visibility=visibility,
            is_async=False,
            is_method=is_method,
            signature=self._get_signature(node, source),
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=self._get_doc_comment(node, source),
            return_type=self._get_return_type(node, source),
            calls=self._extract_calls(body, source) if body else [],
        )

    def _parse_struct(self, node, source: bytes, module_path: str,
                      rel_path: str, result: ParseResult):
        """Parse a struct_specifier (standalone or in type_definition)."""
        name = self._get_name(node, source, "type_identifier")
        if not name:
            return
        sep = "::" if self.language_name == "cpp" else "/"
        qname = f"{module_path}{sep}{name}"

        result.classes.append(ClassInfo(
            name=name,
            qualified_name=qname,
            kind="struct",
            visibility="public",
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=self._get_doc_comment(node, source),
        ))
        result.attributes.extend(
            self._extract_struct_fields(node, source, qname, rel_path))

    def _parse_enum(self, node, source: bytes, module_path: str,
                    rel_path: str, result: ParseResult):
        """Parse an enum_specifier."""
        name = self._get_name(node, source, "type_identifier")
        if not name:
            return
        sep = "::" if self.language_name == "cpp" else "/"
        result.enums.append(EnumInfo(
            name=name,
            qualified_name=f"{module_path}{sep}{name}",
            visibility="public",
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=self._get_doc_comment(node, source),
            variants=self._get_enum_variants(node, source),
        ))

    def _parse_typedef(self, node, source: bytes, module_path: str,
                       rel_path: str, result: ParseResult):
        """Parse a type_definition: typedef ... Name;"""
        sep = "::" if self.language_name == "cpp" else "/"

        # Get the typedef alias name (the type_identifier that's NOT inside
        # the struct/enum specifier — it's a direct child of the typedef)
        typedef_name = None
        for child in node.children:
            if child.type == "type_identifier":
                typedef_name = node_text(child, source)

        # Check if it wraps a struct/enum
        for child in node.children:
            if child.type == "struct_specifier":
                # Struct may be anonymous — use typedef name as fallback
                struct_name = self._get_name(child, source, "type_identifier")
                name = struct_name or typedef_name
                if not name:
                    continue
                qname = f"{module_path}{sep}{name}"
                result.classes.append(ClassInfo(
                    name=name,
                    qualified_name=qname,
                    kind="struct",
                    visibility="public",
                    file_path=rel_path,
                    line_number=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    docstring=self._get_doc_comment(node, source),
                ))
                result.attributes.extend(
                    self._extract_struct_fields(child, source, qname,
                                                rel_path))
                return
            elif child.type == "enum_specifier":
                enum_name = self._get_name(child, source, "type_identifier")
                name = enum_name or typedef_name
                if not name:
                    continue
                result.enums.append(EnumInfo(
                    name=name,
                    qualified_name=f"{module_path}{sep}{name}",
                    visibility="public",
                    file_path=rel_path,
                    line_number=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    docstring=self._get_doc_comment(node, source),
                    variants=self._get_enum_variants(child, source),
                ))
                return

        # Otherwise it's a type alias
        name = typedef_name
        if name is None:
            for child in node.children:
                if child.type in ("pointer_declarator", "array_declarator",
                                  "function_declarator"):
                    for sub in child.children:
                        if sub.type in ("type_identifier", "identifier"):
                            name = node_text(sub, source)
                            break
        if name:
            result.constants.append(ConstantInfo(
                name=name,
                qualified_name=f"{module_path}{sep}{name}",
                kind="type_alias",
                type_annotation=None,
                value_preview=self._get_signature(node, source)[:100],
                visibility="public",
                file_path=rel_path,
                line_number=node.start_point[0] + 1,
            ))

    def _parse_preproc_include(self, node, source: bytes,
                                file_info: FileInfo):
        """Parse #include directive."""
        for child in node.children:
            if child.type in ("string_literal", "system_lib_string"):
                path = node_text(child, source).strip('"<>')
                file_info.imports.append(path)

    def _parse_preproc_def(self, node, source: bytes, module_path: str,
                            rel_path: str, result: ParseResult):
        """Parse #define NAME value."""
        name = self._get_name(node, source)
        if not name:
            return
        val_text = None
        for child in node.children:
            if child.type == "preproc_arg":
                val_text = node_text(child, source).strip()[:100]
        sep = "::" if self.language_name == "cpp" else "/"
        result.constants.append(ConstantInfo(
            name=name,
            qualified_name=f"{module_path}{sep}{name}",
            kind="constant",
            type_annotation=None,
            value_preview=val_text,
            visibility="public",
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
        ))


# ── C Parser ────────────────────────────────────────────────────────────


class CParser(_BaseCCppParser):

    @property
    def language_name(self) -> str:
        return "c"

    @property
    def file_extensions(self) -> list[str]:
        return [".c", ".h"]

    @property
    def noise_names(self) -> frozenset[str]:
        return C_NOISE_NAMES

    def __init__(self):
        self._parser = Parser(C_LANGUAGE)

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
            language="c",
        )

        result = ParseResult()
        result.files.append(file_info)

        for child in root.children:
            if child.type == "function_definition":
                result.functions.append(self._parse_function(
                    child, source, module_path, rel_path))

            elif child.type == "declaration":
                # Could be a function prototype, global var, or struct/enum
                has_struct = any(c.type == "struct_specifier"
                                for c in child.children)
                has_enum = any(c.type == "enum_specifier"
                               for c in child.children)
                if has_struct:
                    for sub in child.children:
                        if sub.type == "struct_specifier":
                            self._parse_struct(sub, source, module_path,
                                               rel_path, result)
                elif has_enum:
                    for sub in child.children:
                        if sub.type == "enum_specifier":
                            self._parse_enum(sub, source, module_path,
                                             rel_path, result)
                else:
                    # Global variable or function declaration
                    pass

            elif child.type == "type_definition":
                self._parse_typedef(child, source, module_path,
                                    rel_path, result)

            elif child.type == "preproc_include":
                self._parse_preproc_include(child, source, file_info)

            elif child.type == "preproc_def":
                self._parse_preproc_def(child, source, module_path,
                                        rel_path, result)

        return result


# ── C++ Parser ──────────────────────────────────────────────────────────


class CppParser(_BaseCCppParser):

    @property
    def language_name(self) -> str:
        return "cpp"

    @property
    def file_extensions(self) -> list[str]:
        return [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"]

    @property
    def noise_names(self) -> frozenset[str]:
        return CPP_NOISE_NAMES

    def __init__(self):
        self._parser = Parser(CPP_LANGUAGE)

    # ── C++-specific helpers ────────────────────────────────────────────

    def _get_access_specifier(self, node, source: bytes) -> str | None:
        """Extract access specifier text (public, private, protected)."""
        for child in node.children:
            if child.type in ("public", "private", "protected"):
                return node_text(child, source)
            # Sometimes it's just text content
            text = node_text(node, source).rstrip(":")
            if text in ("public", "private", "protected"):
                return text
        return None

    def _get_base_classes(self, node, source: bytes) -> list[str]:
        """Extract base classes from base_class_clause."""
        bases: list[str] = []
        for child in node.children:
            if child.type == "base_class_clause":
                for sub in child.children:
                    if sub.type == "type_identifier":
                        bases.append(node_text(sub, source))
                    elif sub.type == "qualified_identifier":
                        bases.append(node_text(sub, source))
                    elif sub.type == "template_type":
                        for inner in sub.children:
                            if inner.type == "type_identifier":
                                bases.append(node_text(inner, source))
                                break
        return bases

    def _parse_class_specifier(self, node, source: bytes, module_path: str,
                                rel_path: str, result: ParseResult,
                                is_struct: bool = False):
        """Parse a class_specifier or struct_specifier with C++ features."""
        name = self._get_name(node, source, "type_identifier")
        if not name:
            return
        qname = f"{module_path}::{name}"
        bases = self._get_base_classes(node, source)
        docstring = self._get_doc_comment(node, source)

        kind = "struct" if is_struct else "class"
        result.classes.append(ClassInfo(
            name=name,
            qualified_name=qname,
            kind=kind,
            visibility="public",
            file_path=rel_path,
            line_number=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=docstring,
            bases=bases,
        ))

        # Extends edges for all bases (C++ has no interface distinction)
        for base in bases:
            result.type_relationships.append(TypeRelationship(
                source_type=name,
                target_type=base,
                relationship="extends",
            ))

        # Parse class body with access specifier tracking
        default_vis = "public" if is_struct else "private"
        self._parse_class_body(node, source, module_path, rel_path,
                                name, qname, result,
                                default_vis=default_vis)

    def _parse_class_body(self, node, source: bytes, module_path: str,
                           rel_path: str, class_name: str,
                           class_qname: str, result: ParseResult,
                           default_vis: str = "private"):
        """Parse class body with access specifier state machine."""
        method_type_rel = TypeRelationship(
            source_type=class_qname,
            target_type=None,
            relationship="inherent",
        )
        current_vis = default_vis

        for child in node.children:
            if child.type == "field_declaration_list":
                for item in child.children:
                    if item.type == "access_specifier":
                        spec = self._get_access_specifier(item, source)
                        if spec:
                            current_vis = spec

                    elif item.type == "function_definition":
                        fn = self._parse_function(
                            item, source, module_path, rel_path,
                            is_method=True, owner=class_name,
                        )
                        fn.visibility = current_vis
                        result.functions.append(fn)
                        method_type_rel.methods.append(fn)

                    elif item.type == "declaration":
                        # Could be a method declaration or field
                        has_func_decl = any(
                            c.type == "function_declarator"
                            for c in item.children)
                        if has_func_decl:
                            # Method declaration (no body)
                            fn = self._parse_function(
                                item, source, module_path, rel_path,
                                is_method=True, owner=class_name,
                            )
                            fn.visibility = current_vis
                            result.functions.append(fn)
                            method_type_rel.methods.append(fn)
                        else:
                            # Field declaration
                            self._parse_cpp_field(
                                item, source, rel_path,
                                class_qname, current_vis, result)

                    elif item.type == "field_declaration":
                        self._parse_cpp_field(
                            item, source, rel_path,
                            class_qname, current_vis, result)

                    elif item.type == "template_declaration":
                        # Unwrap template
                        tparams = get_type_parameters(item, source, "template_parameter_list")
                        for sub in item.children:
                            if sub.type == "function_definition":
                                fn = self._parse_function(
                                    sub, source, module_path, rel_path,
                                    is_method=True, owner=class_name,
                                )
                                fn.visibility = current_vis
                                fn.metadata["is_template"] = True
                                fn.type_parameters = tparams
                                result.functions.append(fn)
                                method_type_rel.methods.append(fn)

                    elif item.type in ("class_specifier",
                                       "struct_specifier"):
                        # Nested class/struct
                        is_struct = item.type == "struct_specifier"
                        self._parse_class_specifier(
                            item, source, f"{module_path}::{class_name}",
                            rel_path, result, is_struct=is_struct)

        if method_type_rel.methods:
            result.type_relationships.append(method_type_rel)

    def _parse_cpp_field(self, node, source: bytes, rel_path: str,
                          class_qname: str, visibility: str,
                          result: ParseResult):
        """Parse a field declaration in a C++ class."""
        type_ann = None
        names: list[str] = []
        for child in node.children:
            if child.type in ("primitive_type", "type_identifier",
                              "sized_type_specifier", "template_type",
                              "qualified_identifier"):
                if type_ann is None:
                    type_ann = node_text(child, source)
            elif child.type == "field_identifier":
                names.append(node_text(child, source))
            elif child.type in ("init_declarator", "pointer_declarator",
                                "reference_declarator"):
                for sub in child.children:
                    if sub.type in ("field_identifier", "identifier"):
                        names.append(node_text(sub, source))
                        break
        for name in names:
            result.attributes.append(AttributeInfo(
                name=name,
                qualified_name=f"{class_qname}::{name}",
                owner_qualified_name=class_qname,
                type_annotation=type_ann,
                visibility=visibility,
                file_path=rel_path,
                line_number=node.start_point[0] + 1,
            ))

    def _parse_namespace(self, node, source: bytes, parent_path: str,
                          rel_path: str, result: ParseResult,
                          file_info: FileInfo):
        """Parse a namespace_definition, recursing into its body."""
        ns_name = self._get_name(node, source)
        if ns_name:
            ns_path = f"{parent_path}::{ns_name}"
            file_info.submodule_declarations.append(ns_name)
        else:
            ns_path = parent_path

        for child in node.children:
            if child.type == "declaration_list":
                for item in child.children:
                    self._parse_cpp_top_level(item, source, ns_path,
                                              rel_path, result, file_info)

    def _parse_cpp_top_level(self, node, source: bytes, module_path: str,
                              rel_path: str, result: ParseResult,
                              file_info: FileInfo):
        """Parse a top-level C++ node."""
        if node.type == "function_definition":
            result.functions.append(self._parse_function(
                node, source, module_path, rel_path))

        elif node.type == "class_specifier":
            self._parse_class_specifier(node, source, module_path,
                                         rel_path, result,
                                         is_struct=False)

        elif node.type == "struct_specifier":
            self._parse_class_specifier(node, source, module_path,
                                         rel_path, result,
                                         is_struct=True)

        elif node.type == "declaration":
            # Check for class/struct/enum inside declarations
            for sub in node.children:
                if sub.type == "class_specifier":
                    self._parse_class_specifier(sub, source, module_path,
                                                 rel_path, result,
                                                 is_struct=False)
                    return
                elif sub.type == "struct_specifier":
                    self._parse_class_specifier(sub, source, module_path,
                                                 rel_path, result,
                                                 is_struct=True)
                    return
                elif sub.type == "enum_specifier":
                    self._parse_enum(sub, source, module_path,
                                     rel_path, result)
                    return
            # Check for function declaration
            has_func = any(c.type == "function_declarator"
                           for c in node.children)
            if has_func:
                result.functions.append(self._parse_function(
                    node, source, module_path, rel_path))

        elif node.type == "enum_specifier":
            self._parse_enum(node, source, module_path, rel_path, result)

        elif node.type == "namespace_definition":
            self._parse_namespace(node, source, module_path, rel_path,
                                   result, file_info)

        elif node.type == "template_declaration":
            # Unwrap template and parse inner, tracking what gets added
            tparams = get_type_parameters(node, source, "template_parameter_list")
            n_funcs = len(result.functions)
            n_classes = len(result.classes)
            for sub in node.children:
                if sub.type in ("function_definition", "class_specifier",
                                "struct_specifier", "declaration"):
                    self._parse_cpp_top_level(sub, source, module_path,
                                              rel_path, result, file_info)
            # Set type_parameters on newly added entities
            if tparams:
                for fn in result.functions[n_funcs:]:
                    fn.metadata["is_template"] = True
                    fn.type_parameters = tparams
                for cls in result.classes[n_classes:]:
                    cls.type_parameters = tparams

        elif node.type == "type_definition":
            self._parse_typedef(node, source, module_path, rel_path, result)

        elif node.type == "preproc_include":
            self._parse_preproc_include(node, source, file_info)

        elif node.type == "preproc_def":
            self._parse_preproc_def(node, source, module_path,
                                    rel_path, result)

        elif node.type == "linkage_specification":
            # extern "C" { ... } or extern "C" fn()
            n_funcs = len(result.functions)
            for sub in node.children:
                if sub.type == "declaration_list":
                    for item in sub.children:
                        self._parse_cpp_top_level(
                            item, source, module_path,
                            rel_path, result, file_info)
                elif sub.type in ("function_definition", "declaration"):
                    self._parse_cpp_top_level(
                        sub, source, module_path,
                        rel_path, result, file_info)
            for fn in result.functions[n_funcs:]:
                fn.metadata["is_ffi"] = True
                fn.metadata["ffi_kind"] = "extern_c"

        elif node.type == "using_declaration":
            for sub in node.children:
                if sub.type in ("scoped_identifier", "qualified_identifier"):
                    file_info.imports.append(node_text(sub, source))
                    break

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
            language="cpp",
        )

        result = ParseResult()
        result.files.append(file_info)

        for child in root.children:
            self._parse_cpp_top_level(child, source, module_path, rel_path,
                                      result, file_info)

        return result
