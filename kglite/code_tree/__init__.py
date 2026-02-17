"""Parse codebases into KGLite knowledge graphs using tree-sitter.

Requires optional dependencies: ``pip install kglite[code-tree]``

Usage::

    from kglite.code_tree import build

    graph = build("/path/to/project/src")
"""

try:
    import tree_sitter  # noqa: F401
except ImportError:
    raise ImportError(
        "The kglite.code_tree module requires tree-sitter. "
        "Install with: pip install kglite[code-tree]"
    ) from None

from .builder import build
from .parsers.manifest import read_manifest

__all__ = ["build", "read_manifest"]
