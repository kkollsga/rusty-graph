"""Push KGLite graph data to a Neo4j database."""

from __future__ import annotations

from collections import defaultdict
import json
import time
import typing


def to_neo4j(
    graph,
    uri: str,
    *,
    auth: tuple[str, str] | None = None,
    database: str = "neo4j",
    batch_size: int = 5000,
    clear: bool = False,
    merge: bool = False,
    selection_only: bool | None = None,
    verbose: bool = False,
) -> dict[str, typing.Any]:
    """Push graph data to a Neo4j database.

    Extracts all nodes and edges (or the current selection) and writes
    them to Neo4j using batched ``UNWIND`` operations for performance.

    Requires the ``neo4j`` package: ``pip install neo4j``.

    Args:
        graph: The :class:`KnowledgeGraph` to export.
        uri: Neo4j connection URI (e.g. ``"bolt://localhost:7687"``).
        auth: Tuple of ``(username, password)``. ``None`` for no auth.
        database: Neo4j database name (default ``"neo4j"``).
        batch_size: Nodes/relationships per ``UNWIND`` batch (default 5000).
        clear: If ``True``, delete all existing data before import.
        merge: If ``True``, use ``MERGE`` instead of ``CREATE`` (upsert).
        selection_only: If ``True``, export only selected nodes.
            Default: auto-detect from active selection.
        verbose: Print progress information.

    Returns:
        Summary dict with ``nodes_created``, ``relationships_created``,
        ``constraints_created``, ``elapsed``, ``database``.

    Example::

        import kglite

        g = kglite.load("graph.kgl")
        kglite.to_neo4j(g, "bolt://localhost:7687", auth=("neo4j", "password"))

        # Push only a subgraph
        kglite.to_neo4j(
            g.select("Person").traverse("KNOWS"),
            "bolt://localhost:7687",
            auth=("neo4j", "password"),
        )
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError("The 'neo4j' package is required for to_neo4j(). Install with: pip install neo4j") from None

    t0 = time.perf_counter()

    # -- Extract data via D3 JSON export --
    raw = graph.export_string("d3", selection_only=selection_only)
    data = json.loads(raw)
    nodes = data.get("nodes", [])
    links = data.get("links", [])

    if verbose:
        print(f"to_neo4j: Extracted {len(nodes)} nodes, {len(links)} relationships")

    # -- Connect --
    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        summary = _push(
            driver,
            nodes,
            links,
            database=database,
            batch_size=batch_size,
            clear=clear,
            merge=merge,
            verbose=verbose,
        )
    finally:
        driver.close()

    summary["elapsed"] = round(time.perf_counter() - t0, 2)
    summary["database"] = database

    if verbose:
        print(
            f"to_neo4j: Done. {summary['nodes_created']} nodes, "
            f"{summary['relationships_created']} relationships "
            f"in {summary['elapsed']}s"
        )

    return summary


def _push(
    driver,
    nodes: list[dict],
    links: list[dict],
    *,
    database: str,
    batch_size: int,
    clear: bool,
    merge: bool,
    verbose: bool,
) -> dict[str, typing.Any]:
    """Push extracted data to Neo4j."""
    nodes_created = 0
    rels_created = 0
    constraints_created = 0

    # -- Clear --
    if clear:
        with driver.session(database=database) as session:
            while True:
                result = session.run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n")
                deleted = result.consume().counters.nodes_deleted
                if deleted == 0:
                    break
        if verbose:
            print("to_neo4j: Cleared existing data")

    # -- Group nodes by type --
    nodes_by_type: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        label = node.get("type", "Node")
        props = _sanitize_props(node, exclude={"type"}, id_key="id")
        nodes_by_type[label].append(props)

    # -- Create constraints --
    with driver.session(database=database) as session:
        for label in nodes_by_type:
            try:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n._kglite_id IS UNIQUE")
                constraints_created += 1
            except Exception:
                pass  # constraint may already exist or not be supported
    if verbose:
        print(f"to_neo4j: Created {constraints_created} constraints")

    # -- Push nodes --
    for label, typed_nodes in nodes_by_type.items():
        for i in range(0, len(typed_nodes), batch_size):
            batch = typed_nodes[i : i + batch_size]
            with driver.session(database=database) as session:
                if merge:
                    result = session.run(
                        f"UNWIND $batch AS item "
                        f"MERGE (n:`{label}` {{_kglite_id: item._kglite_id}}) "
                        f"SET n = item "
                        f"RETURN count(n) AS cnt",
                        batch=batch,
                    )
                else:
                    result = session.run(
                        f"UNWIND $batch AS item CREATE (n:`{label}`) SET n = item RETURN count(n) AS cnt",
                        batch=batch,
                    )
                record = result.single()
                nodes_created += record["cnt"] if record else len(batch)
        if verbose:
            n_batches = (len(typed_nodes) + batch_size - 1) // batch_size
            print(f"to_neo4j: {label}: {len(typed_nodes)} nodes ({n_batches} batch{'es' if n_batches != 1 else ''})")

    # -- Prepare relationship batches --
    rels_by_type: dict[str, list[dict]] = defaultdict(list)
    for link in links:
        rel_type = link.get("type", "CONNECTED")
        source_idx = link.get("source", 0)
        target_idx = link.get("target", 0)

        # Resolve positional indices to actual node IDs
        if source_idx < len(nodes) and target_idx < len(nodes):
            source_id = nodes[source_idx].get("id")
            target_id = nodes[target_idx].get("id")
        else:
            continue

        props = _sanitize_props(link, exclude={"type", "source", "target"})
        rels_by_type[rel_type].append({"_start": source_id, "_end": target_id, "props": props})

    # -- Push relationships --
    for rel_type, typed_rels in rels_by_type.items():
        for i in range(0, len(typed_rels), batch_size):
            batch = typed_rels[i : i + batch_size]
            with driver.session(database=database) as session:
                if merge:
                    result = session.run(
                        f"UNWIND $batch AS item "
                        f"MATCH (a {{_kglite_id: item._start}}) "
                        f"MATCH (b {{_kglite_id: item._end}}) "
                        f"MERGE (a)-[r:`{rel_type}`]->(b) "
                        f"SET r = item.props "
                        f"RETURN count(r) AS cnt",
                        batch=batch,
                    )
                else:
                    result = session.run(
                        f"UNWIND $batch AS item "
                        f"MATCH (a {{_kglite_id: item._start}}) "
                        f"MATCH (b {{_kglite_id: item._end}}) "
                        f"CREATE (a)-[r:`{rel_type}`]->(b) "
                        f"SET r = item.props "
                        f"RETURN count(r) AS cnt",
                        batch=batch,
                    )
                record = result.single()
                rels_created += record["cnt"] if record else len(batch)
        if verbose:
            n_batches = (len(typed_rels) + batch_size - 1) // batch_size
            suffix = "es" if n_batches != 1 else ""
            print(f"to_neo4j: {rel_type}: {len(typed_rels)} relationships ({n_batches} batch{suffix})")

    return {
        "nodes_created": nodes_created,
        "relationships_created": rels_created,
        "constraints_created": constraints_created,
    }


def _sanitize_props(raw: dict, exclude: set[str] | None = None, id_key: str | None = None) -> dict:
    """Prepare a property dict for Neo4j.

    - Renames ``id`` → ``_kglite_id`` (if *id_key* is set)
    - Removes keys in *exclude*
    - Converts dict values to JSON strings (Neo4j can't store dicts)
    - Drops None values
    """
    out: dict[str, typing.Any] = {}
    for k, v in raw.items():
        if exclude and k in exclude:
            continue
        if v is None:
            continue

        # Rename id to _kglite_id
        key = "_kglite_id" if (id_key and k == id_key) else k

        # Neo4j cannot store dicts — serialize to JSON string
        if isinstance(v, dict):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out
