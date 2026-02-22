"""Build a KGLite knowledge graph from a JSON blueprint and CSV files.

Usage::

    from kglite.blueprint import from_blueprint

    graph = from_blueprint("blueprint.json")
"""

from .loader import from_blueprint

__all__ = ["from_blueprint"]
