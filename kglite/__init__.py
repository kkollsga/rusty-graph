"""KGLite - A high-performance graph database library with Python bindings written in Rust."""

from .kglite import *  # noqa: F401, F403
from .kglite import (  # explicit re-exports for type checkers
    __version__,
    KnowledgeGraph,
    Transaction,
    ResultView,
    ResultIter,
    load,
)
from .blueprint import from_blueprint

__all__ = [
    "__version__",
    "KnowledgeGraph",
    "Transaction",
    "ResultView",
    "ResultIter",
    "load",
    "from_blueprint",
]
