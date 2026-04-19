"""Build a KGLite knowledge graph from a JSON blueprint and CSV files.

Usage::

    from kglite.blueprint import from_blueprint

    graph = from_blueprint("blueprint.json")

Implemented entirely in Rust; see ``src/graph/blueprint/`` and the
``from_blueprint_rust`` ``#[pyfunction]`` in ``src/graph/pyapi/blueprint.rs``.
This module is a ~20-line shim that handles optional save + schema lock
on top of the Rust build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from kglite.kglite import KnowledgeGraph
from kglite.kglite import from_blueprint_rust as _from_blueprint_rs


def from_blueprint(
    blueprint_path: Union[str, Path],
    *,
    verbose: bool = False,
    save: bool = True,
    lock_schema: bool = False,
    storage: str = "default",
    path: Optional[str] = None,
) -> KnowledgeGraph:
    """Build a KnowledgeGraph from a JSON blueprint + CSV files.

    Args:
        blueprint_path: Path to the blueprint JSON file.
        verbose: Print a summary line after the build.
        save: If True and the blueprint has an ``output_file`` key, save
            the built graph to that path.
        lock_schema: If True, lock the schema so subsequent Cypher
            mutations are validated against the blueprint types.
        storage: ``"default"`` (in-memory), ``"mapped"`` (mmap columns),
            or ``"disk"`` (CSR + mmap). Disk requires ``path``.
        path: Directory for disk storage (only used with ``storage="disk"``).
    """
    if verbose:
        print(f"Loading blueprint from {blueprint_path}...")
    graph, output_path = _from_blueprint_rs(
        str(blueprint_path),
        verbose=verbose,
        storage=storage if storage else "default",
        path=path,
    )
    if verbose:
        counts = graph.node_type_counts()
        for node_type, n in sorted(counts.items()):
            print(f"  {node_type}: {n} nodes")
    if save and output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        graph.save(str(out))
    if lock_schema:
        graph.lock_schema()
    return graph


__all__ = ["from_blueprint"]
