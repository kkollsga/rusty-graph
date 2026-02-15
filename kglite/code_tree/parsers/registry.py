"""Language detection and parser registry."""

from pathlib import Path
from .base import LanguageParser

EXTENSION_MAP: dict[str, str] = {
    ".rs": "rust",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
}


def get_parser(language: str) -> LanguageParser:
    """Get a parser instance for the given language name."""
    if language == "rust":
        from .rust import RustParser
        return RustParser()
    elif language == "python":
        from .python import PythonParser
        return PythonParser()
    elif language == "typescript":
        from .typescript import TypeScriptParser
        return TypeScriptParser()
    elif language == "javascript":
        from .typescript import JavaScriptParser
        return JavaScriptParser()
    else:
        raise ValueError(f"Unsupported language: {language}")


def detect_languages(src_root: Path) -> set[str]:
    """Scan a directory and return the set of languages found."""
    languages = set()
    for path in src_root.rglob("*"):
        if path.is_file() and path.suffix in EXTENSION_MAP:
            languages.add(EXTENSION_MAP[path.suffix])
    return languages


def get_parsers_for_directory(src_root: Path) -> list[LanguageParser]:
    """Auto-detect languages in a directory and return parser instances."""
    languages = detect_languages(src_root)
    parsers = []
    for lang in sorted(languages):
        try:
            parsers.append(get_parser(lang))
        except ImportError as e:
            print(f"  Warning: Skipping {lang} (grammar not installed): {e}")
    return parsers
