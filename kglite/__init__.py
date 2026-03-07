"""KGLite - A high-performance graph database library with Python bindings written in Rust."""

from .blueprint import from_blueprint
from .kglite import *  # noqa: F401, F403
from .kglite import (  # explicit re-exports for type checkers
    KnowledgeGraph,
    ResultIter,
    ResultView,
    Transaction,
    __version__,
    load,
)

__all__ = [
    "__version__",
    "KnowledgeGraph",
    "Transaction",
    "ResultView",
    "ResultIter",
    "load",
    "from_blueprint",
]
