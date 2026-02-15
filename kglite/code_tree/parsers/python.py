"""Python language parser using tree-sitter-python."""

from pathlib import Path
import re
from tree_sitter import Language, Parser
import tree_sitter_python as ts_python

from .base import LanguageParser, node_text, count_lines
from .models import (
    ParseResult, FileInfo, FunctionInfo, ClassInfo,
    EnumInfo, InterfaceInfo, TypeRelationship,
    AttributeInfo, ConstantInfo,
)

PY_LANGUAGE = Language(ts_python.language())

# Base classes that indicate an enum
_ENUM_BASES = frozenset({
    "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "auto",
})

# Base classes that indicate a protocol/interface
_PROTOCOL_BASES = frozenset({"Protocol",})

_CONSTANT_RE = re.compile(r'^[A-Z][A-Z_0-9]+$')

PYTHON_NOISE_NAMES: frozenset[str] = frozenset({
    # Builtins
    "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "print", "isinstance", "issubclass", "type", "range", "enumerate",
    "zip", "map", "filter", "sorted", "reversed", "any", "all",
    "min", "max", "sum", "abs", "round", "hash", "id", "repr",
    "super", "getattr", "setattr", "hasattr", "delattr",
    "callable", "iter", "next", "open", "format",
    # Common method names
    "append", "extend", "update", "pop", "get", "keys", "values", "items",
    "join", "split", "strip", "replace", "startswith", "endswith",
})


class PythonParser(LanguageParser):

    @property
    def language_name(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> list[str]:
        return [".py"]

    @property
    def noise_names(self) -> frozenset[str]:
        return PYTHON_NOISE_NAMES

    def __init__(self):
        self._parser = Parser(PY_LANGUAGE)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_visibility(self, name: str) -> str:
        if name.startswith("__") and not name.endswith("__"):
            return "private"
        if name.startswith("_"):
            return "private"
        return "public"

    def _get_name(self, node, source: bytes) -> str:
        """Get the identifier name from a definition node."""
        for child in node.children:
            if child.type == "identifier":
                return node_text(child, source)
        return "unknown"

    def _get_block(self, node):
        """Get the body block of a class/function definition."""
        for child in node.children:
            if child.type == "block":
                return child
        return None

    def _get_docstring(self, node, source: bytes) -> str | None:
        """Extract docstring from the first expression_statement in a block."""
        block = self._get_block(node)
        if block is None:
            return None
        for child in block.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        raw = node_text(sub, source)
                        # Strip triple-quote delimiters
                        for delim in ('"""', "'''", '"', "'"):
                            if raw.startswith(delim) and raw.endswith(delim):
                                return raw[len(delim):-len(delim)].strip()
                        return raw
                break  # only first statement can be a docstring
            elif child.type != "comment":
                break
        return None

    def _get_bases(self, node, source: bytes) -> list[str]:
        """Extract base class names from a class_definition's argument_list."""
        bases = []
        for child in node.children:
            if child.type == "argument_list":
                for arg in child.children:
                    if arg.type == "identifier":
                        bases.append(node_text(arg, source))
                    elif arg.type == "attribute":
                        bases.append(node_text(arg, source))
                    elif arg.type == "subscript":
                        # e.g. Generic[T], Protocol[T] - take the base name
                        for sub in arg.children:
                            if sub.type == "identifier":
                                bases.append(node_text(sub, source))
                                break
                            elif sub.type == "attribute":
                                bases.append(node_text(sub, source))
                                break
                    elif arg.type == "keyword_argument":
                        # e.g. metaclass=ABCMeta - skip
                        pass
        return bases

    def _get_decorators(self, decorated_node, source: bytes) -> list[str]:
        """Extract decorator names from a decorated_definition."""
        decorators = []
        for child in decorated_node.children:
            if child.type == "decorator":
                # Get the text after @
                text = node_text(child, source).strip()
                if text.startswith("@"):
                    text = text[1:]
                decorators.append(text)
        return decorators

    def _get_decorated_inner(self, node):
        """Get the inner definition (function/class) from a decorated_definition."""
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return child
        return None

    def _get_signature(self, node, source: bytes) -> str:
        """Extract function signature (everything before the body)."""
        parts = []
        for child in node.children:
            if child.type == "block":
                break
            if child.type == "comment":
                continue
            parts.append(node_text(child, source))
        return " ".join(parts).rstrip(" :")

    def _get_return_type(self, node, source: bytes) -> str | None:
        """Extract return type annotation."""
        saw_arrow = False
        for child in node.children:
            if not child.is_named and node_text(child, source) == "->":
                saw_arrow = True
            elif saw_arrow:
                if child.type == ":":
                    break
                if child.type == "block":
                    break
                return node_text(child, source)
        return None

    def _is_async(self, node, source: bytes) -> bool:
        """Check if function is async (look for 'async' keyword before 'def')."""
        for child in node.children:
            if not child.is_named and node_text(child, source) == "async":
                return True
            if not child.is_named and node_text(child, source) == "def":
                break
        return False

    def _extract_calls(self, body_node, source: bytes) -> list[str]:
        """Recursively extract function/method names called within a block."""
        calls = []

        def walk(node):
            if node.type == "call":
                func = node.children[0] if node.children else None
                if func:
                    if func.type == "identifier":
                        calls.append(node_text(func, source))
                    elif func.type == "attribute":
                        # obj.method() -> extract "method"
                        for child in reversed(func.children):
                            if child.type == "identifier":
                                calls.append(node_text(child, source))
                                break
            for child in node.children:
                walk(child)

        walk(body_node)
        return calls

    def _file_to_module_path(self, filepath: Path, src_root: Path) -> str:
        """Convert a Python file path to a dotted module path.

        Uses src_root's name as the package prefix.
        e.g. src_root=chromadb/, file=chromadb/api/client.py -> chromadb.api.client
        """
        rel = filepath.relative_to(src_root)
        parts = list(rel.parts)
        # Strip file extension from last part
        parts[-1] = parts[-1].replace(".py", "")
        # __init__ means the directory itself is the module
        if parts[-1] == "__init__":
            parts = parts[:-1]
        # Prepend the src_root directory name as the package
        pkg_name = src_root.name
        if parts:
            return pkg_name + "." + ".".join(parts)
        return pkg_name

    def _get_enum_variants(self, node, source: bytes) -> list[str]:
        """Extract enum variant names from a class body."""
        variants = []
        block = self._get_block(node)
        if block is None:
            return variants
        for child in block.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "assignment":
                        for target in sub.children:
                            if target.type == "identifier":
                                variants.append(node_text(target, source))
                                break
        return variants

    def _parse_import(self, node, source: bytes) -> str | None:
        """Parse an import statement and return the module path."""
        if node.type == "import_statement":
            # import foo.bar
            for child in node.children:
                if child.type == "dotted_name":
                    return node_text(child, source)
            return None
        elif node.type == "import_from_statement":
            # from foo.bar import baz
            for child in node.children:
                if child.type == "dotted_name":
                    return node_text(child, source)
                elif child.type == "relative_import":
                    return None  # skip relative imports for now
            return None
        return None

    def _classify_decorators(self, decorators: list[str]) -> dict:
        """Extract semantic flags from decorator names."""
        flags = {}
        for dec in decorators:
            base = dec.split("(")[0].split(".")[-1]
            if base == "abstractmethod":
                flags["is_abstract"] = True
            elif base == "property":
                flags["is_property"] = True
            elif base == "staticmethod":
                flags["is_static"] = True
            elif base == "classmethod":
                flags["is_classmethod"] = True
            elif base == "overload":
                flags["is_overload"] = True
        return flags

    def _extract_class_attributes(self, class_node, source: bytes,
                                   owner_qname: str, rel_path: str,
                                   result: ParseResult):
        """Extract attributes from class body and __init__ self assignments."""
        block = self._get_block(class_node)
        if block is None:
            return

        # 1. Class-body assignments: x = value, x: type = value
        for child in block.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "assignment":
                        attr_name = None
                        type_ann = None
                        default_val = None

                        for sc in sub.children:
                            if sc.type == "identifier" and attr_name is None:
                                attr_name = node_text(sc, source)
                            elif sc.type == "type":
                                type_ann = node_text(sc, source)

                        if attr_name and not _CONSTANT_RE.match(attr_name):
                            saw_eq = False
                            for sc in sub.children:
                                if not sc.is_named and node_text(sc, source) == "=":
                                    saw_eq = True
                                elif saw_eq and sc.is_named:
                                    val = node_text(sc, source)
                                    default_val = val[:100]
                                    break

                            result.attributes.append(AttributeInfo(
                                name=attr_name,
                                qualified_name=f"{owner_qname}.{attr_name}",
                                owner_qualified_name=owner_qname,
                                type_annotation=type_ann,
                                visibility=self._get_visibility(attr_name),
                                file_path=rel_path,
                                line_number=child.start_point[0] + 1,
                                default_value=default_val,
                            ))

        # 2. self.x assignments in __init__
        seen_names = {a.name for a in result.attributes
                      if a.owner_qualified_name == owner_qname}

        for child in block.children:
            fn_node = None
            if child.type == "function_definition":
                fn_node = child
            elif child.type == "decorated_definition":
                inner = self._get_decorated_inner(child)
                if inner and inner.type == "function_definition":
                    fn_node = inner

            if fn_node and self._get_name(fn_node, source) == "__init__":
                init_block = self._get_block(fn_node)
                if init_block:
                    self._walk_self_attrs(
                        init_block, source, owner_qname, rel_path,
                        result, seen_names,
                    )
                break

    def _walk_self_attrs(self, node, source: bytes, owner_qname: str,
                         rel_path: str, result: ParseResult,
                         seen_names: set[str]):
        """Recursively find self.x = ... assignments."""
        if node.type == "assignment":
            left = node.children[0] if node.children else None
            if left and left.type == "attribute":
                text = node_text(left, source)
                if text.startswith("self."):
                    attr_name = text[5:]
                    if "." not in attr_name and attr_name not in seen_names:
                        seen_names.add(attr_name)
                        default_val = None
                        saw_eq = False
                        for sc in node.children:
                            if not sc.is_named and node_text(sc, source) == "=":
                                saw_eq = True
                            elif saw_eq and sc.is_named:
                                val = node_text(sc, source)
                                default_val = val[:100]
                                break

                        result.attributes.append(AttributeInfo(
                            name=attr_name,
                            qualified_name=f"{owner_qname}.{attr_name}",
                            owner_qualified_name=owner_qname,
                            type_annotation=None,
                            visibility=self._get_visibility(attr_name),
                            file_path=rel_path,
                            line_number=node.start_point[0] + 1,
                            default_value=default_val,
                        ))
        for c in node.children:
            self._walk_self_attrs(c, source, owner_qname, rel_path,
                                  result, seen_names)

    # ── Parsing ─────────────────────────────────────────────────────────

    def _parse_function(self, node, source: bytes, module_path: str,
                        rel_path: str, is_method: bool = False,
                        owner: str | None = None) -> FunctionInfo:
        name = self._get_name(node, source)
        sep = "."
        prefix = f"{module_path}{sep}{owner}" if owner else module_path
        qualified_name = f"{prefix}{sep}{name}"

        block = self._get_block(node)

        return FunctionInfo(
            name=name,
            qualified_name=qualified_name,
            visibility=self._get_visibility(name),
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
                     rel_path: str, result: ParseResult,
                     decorators: list[str] | None = None):
        """Parse a class_definition node and populate the result."""
        name = self._get_name(node, source)
        qualified_name = f"{module_path}.{name}"
        bases = self._get_bases(node, source)
        docstring = self._get_docstring(node, source)

        # Determine what kind of class this is
        is_enum = bool(_ENUM_BASES & set(bases))
        is_protocol = "Protocol" in bases

        if is_enum:
            result.enums.append(EnumInfo(
                name=name,
                qualified_name=qualified_name,
                visibility=self._get_visibility(name),
                file_path=rel_path,
                line_number=node.start_point[0] + 1,
                docstring=docstring,
                variants=self._get_enum_variants(node, source),
            ))
            return

        if is_protocol:
            result.interfaces.append(InterfaceInfo(
                name=name,
                qualified_name=qualified_name,
                kind="protocol",
                visibility=self._get_visibility(name),
                file_path=rel_path,
                line_number=node.start_point[0] + 1,
                docstring=docstring,
            ))
        else:
            result.classes.append(ClassInfo(
                name=name,
                qualified_name=qualified_name,
                kind="class",
                visibility=self._get_visibility(name),
                file_path=rel_path,
                line_number=node.start_point[0] + 1,
                docstring=docstring,
                bases=bases,
                metadata={"decorators": decorators or []},
            ))

        # Inheritance / extends edges
        non_protocol_bases = [b for b in bases if b != "Protocol"]
        for base in non_protocol_bases:
            result.type_relationships.append(TypeRelationship(
                source_type=name,
                target_type=base,
                relationship="extends",
            ))

        # Parse methods inside the class body
        method_type_rel = TypeRelationship(
            source_type=qualified_name,
            target_type=None,
            relationship="inherent",
        )

        block = self._get_block(node)
        if block:
            for child in block.children:
                fn_node = None
                fn_decorators: list[str] = []
                if child.type == "function_definition":
                    fn_node = child
                elif child.type == "decorated_definition":
                    inner = self._get_decorated_inner(child)
                    if inner and inner.type == "function_definition":
                        fn_node = inner
                        fn_decorators = self._get_decorators(child, source)
                    elif inner and inner.type == "class_definition":
                        # Nested class
                        self._parse_class(
                            inner, source, qualified_name, rel_path, result,
                            decorators=self._get_decorators(child, source),
                        )
                elif child.type == "class_definition":
                    # Nested class without decorators
                    self._parse_class(
                        child, source, qualified_name, rel_path, result,
                    )

                if fn_node:
                    fn = self._parse_function(
                        fn_node, source, module_path, rel_path,
                        is_method=True, owner=name,
                    )
                    fn.decorators = fn_decorators
                    fn.metadata.update(self._classify_decorators(fn_decorators))
                    result.functions.append(fn)
                    method_type_rel.methods.append(fn)

        if method_type_rel.methods:
            result.type_relationships.append(method_type_rel)

        # Extract class attributes
        self._extract_class_attributes(node, source, qualified_name, rel_path, result)

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
            language="python",
        )

        result = ParseResult()
        result.files.append(file_info)

        for child in root.children:
            if child.type == "function_definition":
                result.functions.append(self._parse_function(
                    child, source, module_path, rel_path,
                    is_method=False, owner=None,
                ))

            elif child.type == "decorated_definition":
                inner = self._get_decorated_inner(child)
                if inner and inner.type == "function_definition":
                    fn = self._parse_function(
                        inner, source, module_path, rel_path,
                        is_method=False, owner=None,
                    )
                    fn.decorators = self._get_decorators(child, source)
                    fn.metadata.update(self._classify_decorators(fn.decorators))
                    result.functions.append(fn)
                elif inner and inner.type == "class_definition":
                    self._parse_class(
                        inner, source, module_path, rel_path, result,
                        decorators=self._get_decorators(child, source),
                    )

            elif child.type == "class_definition":
                self._parse_class(
                    child, source, module_path, rel_path, result,
                )

            elif child.type in ("import_statement", "import_from_statement"):
                imp = self._parse_import(child, source)
                if imp:
                    file_info.imports.append(imp)

            elif child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "assignment":
                        first_id = None
                        for sc in sub.children:
                            if sc.type == "identifier":
                                first_id = node_text(sc, source)
                                break
                        if first_id == "__all__":
                            # Extract __all__ = [...] exports
                            for sc in sub.children:
                                if sc.type == "list":
                                    for item in sc.children:
                                        if item.type == "string":
                                            text = node_text(item, source).strip("'\"")
                                            file_info.exports.append(text)
                        elif first_id and _CONSTANT_RE.match(first_id):
                            # Module-level constant (ALL_CAPS)
                            type_ann = None
                            default_val = None
                            for sc in sub.children:
                                if sc.type == "type":
                                    type_ann = node_text(sc, source)
                            saw_eq = False
                            for sc in sub.children:
                                if not sc.is_named and node_text(sc, source) == "=":
                                    saw_eq = True
                                elif saw_eq and sc.is_named:
                                    val = node_text(sc, source)
                                    default_val = val[:100]
                                    break
                            result.constants.append(ConstantInfo(
                                name=first_id,
                                qualified_name=f"{module_path}.{first_id}",
                                kind="constant",
                                type_annotation=type_ann,
                                value_preview=default_val,
                                visibility=self._get_visibility(first_id),
                                file_path=rel_path,
                                line_number=child.start_point[0] + 1,
                            ))

            elif child.type == "type_alias_statement":
                # Python 3.12: type X = int
                alias_name = None
                for sc in child.children:
                    if sc.type == "identifier":
                        alias_name = node_text(sc, source)
                        break
                if alias_name:
                    val_text = None
                    saw_eq = False
                    for sc in child.children:
                        if not sc.is_named and node_text(sc, source) == "=":
                            saw_eq = True
                        elif saw_eq and sc.is_named:
                            val_text = node_text(sc, source)[:100]
                            break
                    result.constants.append(ConstantInfo(
                        name=alias_name,
                        qualified_name=f"{module_path}.{alias_name}",
                        kind="type_alias",
                        type_annotation=val_text,
                        value_preview=None,
                        visibility=self._get_visibility(alias_name),
                        file_path=rel_path,
                        line_number=child.start_point[0] + 1,
                    ))

        # Detect submodule declarations from __init__.py
        if filepath.name == "__init__.py":
            parent_dir = filepath.parent
            for item in sorted(parent_dir.iterdir()):
                if item.is_dir() and (item / "__init__.py").exists():
                    file_info.submodule_declarations.append(item.name)
                elif (item.is_file() and item.suffix == ".py"
                      and item.name != "__init__.py"):
                    file_info.submodule_declarations.append(
                        item.stem  # filename without .py
                    )

        return result
