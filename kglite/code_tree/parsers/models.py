"""Language-agnostic data models for code graph entities."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileInfo:
    path: str                  # relative to src_root
    filename: str
    loc: int
    module_path: str           # language-specific qualified path
    language: str              # "rust", "python", "typescript", "javascript"
    submodule_declarations: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)  # __all__, pub items
    annotations: list[dict] | None = None  # TODO/FIXME/HACK/SAFETY comments
    is_test: bool = False


@dataclass
class FunctionInfo:
    name: str
    qualified_name: str
    visibility: str
    is_async: bool
    is_method: bool
    signature: str
    file_path: str
    line_number: int
    docstring: str | None
    return_type: str | None
    decorators: list[str] = field(default_factory=list)
    calls: list[tuple[str, int]] = field(default_factory=list)  # (name, line_number)
    type_parameters: str | None = None  # e.g. "T, U: Display"
    end_line: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ClassInfo:
    """Represents structs (Rust), classes (Python/TS)."""
    name: str
    qualified_name: str
    kind: str                  # "struct" | "class" -> determines graph node type
    visibility: str
    file_path: str
    line_number: int
    docstring: str | None
    bases: list[str] = field(default_factory=list)
    type_parameters: str | None = None  # e.g. "T, U"
    end_line: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EnumInfo:
    name: str
    qualified_name: str
    visibility: str
    file_path: str
    line_number: int
    docstring: str | None
    variants: list[str] = field(default_factory=list)
    end_line: int | None = None
    variant_details: list[dict] | None = None  # structured variant info per language


@dataclass
class InterfaceInfo:
    """Represents traits (Rust), protocols (Python), interfaces (TS)."""
    name: str
    qualified_name: str
    kind: str                  # "trait" | "protocol" | "interface"
    visibility: str
    file_path: str
    line_number: int
    docstring: str | None
    type_parameters: str | None = None  # e.g. "T"
    end_line: int | None = None


@dataclass
class AttributeInfo:
    """Class/struct field or property."""
    name: str
    qualified_name: str        # e.g. "chromadb.Collection.name"
    owner_qualified_name: str  # e.g. "chromadb.Collection"
    type_annotation: str | None
    visibility: str
    file_path: str
    line_number: int
    default_value: str | None = None


@dataclass
class ConstantInfo:
    """Top-level constant, type alias, or static variable."""
    name: str
    qualified_name: str
    kind: str                  # "constant" | "type_alias" | "static"
    type_annotation: str | None
    value_preview: str | None  # first ~100 chars of the value
    visibility: str
    file_path: str
    line_number: int


@dataclass
class TypeRelationship:
    """Represents an impl block (Rust), inheritance (Python), or implements (TS)."""
    source_type: str
    target_type: str | None    # None for inherent impl
    relationship: str          # "implements" | "extends" | "inherent"
    methods: list[FunctionInfo] = field(default_factory=list)


@dataclass
class ParseResult:
    """Unified result from any language parser."""
    files: list[FileInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    enums: list[EnumInfo] = field(default_factory=list)
    interfaces: list[InterfaceInfo] = field(default_factory=list)
    type_relationships: list[TypeRelationship] = field(default_factory=list)
    attributes: list[AttributeInfo] = field(default_factory=list)
    constants: list[ConstantInfo] = field(default_factory=list)

    def merge(self, other: "ParseResult") -> "ParseResult":
        """Merge another ParseResult into this one (mutates self)."""
        self.files.extend(other.files)
        self.functions.extend(other.functions)
        self.classes.extend(other.classes)
        self.enums.extend(other.enums)
        self.interfaces.extend(other.interfaces)
        self.type_relationships.extend(other.type_relationships)
        self.attributes.extend(other.attributes)
        self.constants.extend(other.constants)
        return self


# ── Manifest / project-level models ──────────────────────────────────


@dataclass
class SourceRoot:
    """A directory containing source code to parse."""
    path: Path                          # absolute path to directory
    language: str | None = None         # if known; None = auto-detect
    is_test: bool = False
    label: str | None = None            # e.g. "python-package", "rust-crate"


@dataclass
class DependencyInfo:
    """A project dependency declared in a manifest."""
    name: str
    version_spec: str | None = None
    is_dev: bool = False
    is_optional: bool = False
    group: str | None = None            # optional dep group name


@dataclass
class ProjectInfo:
    """Project metadata extracted from a manifest file."""
    name: str
    version: str | None = None
    description: str | None = None
    languages: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    license: str | None = None
    repository_url: str | None = None
    manifest_path: str = ""
    source_roots: list[SourceRoot] = field(default_factory=list)
    test_roots: list[SourceRoot] = field(default_factory=list)
    dependencies: list[DependencyInfo] = field(default_factory=list)
    build_system: str | None = None     # "maturin", "setuptools", "cargo", etc.
    metadata: dict = field(default_factory=dict)
