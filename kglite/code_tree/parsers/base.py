"""Abstract base class for language parsers and shared helpers."""

from abc import ABC, abstractmethod
from pathlib import Path

from .models import ParseResult


class LanguageParser(ABC):
    """Base class that all language parsers must extend."""

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return the language identifier (e.g. 'rust', 'python')."""
        ...

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """Return list of file extensions this parser handles (e.g. ['.rs'])."""
        ...

    @property
    def noise_names(self) -> frozenset[str]:
        """Names to exclude from call-edge resolution (common stdlib methods).

        Override in subclasses to provide language-specific noise sets.
        """
        return frozenset()

    @abstractmethod
    def parse_file(self, filepath: Path, src_root: Path) -> ParseResult:
        """Parse a single source file and return extracted entities."""
        ...

    def parse_directory(self, src_root: Path) -> ParseResult:
        """Parse all matching files under src_root."""
        combined = ParseResult()
        source_files = []
        for ext in self.file_extensions:
            source_files.extend(sorted(src_root.rglob(f"*{ext}")))
        print(f"  Found {len(source_files)} {self.language_name} files")
        for filepath in source_files:
            result = self.parse_file(filepath, src_root)
            combined.merge(result)
        return combined


# ── Shared helpers ─────────────────────────────────────────────────────


def node_text(node, source: bytes) -> str:
    """Extract the text of a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf8")


def count_lines(source: bytes) -> int:
    """Count lines of code in source bytes."""
    return source.count(b"\n") + (1 if source and not source.endswith(b"\n") else 0)


def get_type_parameters(node, source: bytes,
                        node_type: str = "type_parameters") -> str | None:
    """Extract generic/template type parameters from a declaration node.

    Looks for a child of the given node_type (e.g. "type_parameters",
    "type_parameter_list") and returns the inner text with angle brackets
    stripped.  Returns None if no type parameters are found.
    """
    for child in node.children:
        if child.type == node_type:
            text = source[child.start_byte:child.end_byte].decode("utf8")
            # Strip surrounding < > if present
            if text.startswith("<") and text.endswith(">"):
                text = text[1:-1].strip()
            return text if text else None
    return None


_SELF_PARAMS = frozenset({"self", "&self", "&mut self", "cls"})


def extract_parameters_from_signature(signature: str) -> str | None:
    """Extract the parameter list from a function signature string.

    Finds the first balanced (...) group, filters out self/cls, and
    returns the cleaned parameter text or None if empty.
    """
    start = signature.find("(")
    if start == -1:
        return None
    depth = 0
    end = start
    for i in range(start, len(signature)):
        if signature[i] == "(":
            depth += 1
        elif signature[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    params_text = signature[start + 1:end].strip()
    if not params_text:
        return None
    parts = [p.strip() for p in params_text.split(",")]
    filtered = [p for p in parts if p and p.strip() not in _SELF_PARAMS]
    return ", ".join(filtered) if filtered else None
