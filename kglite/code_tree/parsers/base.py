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
