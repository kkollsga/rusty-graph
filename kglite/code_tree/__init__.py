"""Parse codebases into KGLite knowledge graphs.

All functionality is implemented in Rust (with bundled tree-sitter
grammars) and exposed through the native `_kglite_code_tree` submodule.
No optional dependencies are required.

Usage::

    from kglite.code_tree import build

    graph = build("/path/to/project")
"""

from kglite._kglite_code_tree import build, read_manifest, repo_tree

__all__ = ["build", "read_manifest", "repo_tree"]
