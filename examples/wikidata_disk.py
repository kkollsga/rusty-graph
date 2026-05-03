"""Build (or load) a disk graph from the Wikidata truthy dump.

First run downloads the dump (~80 GB) into WORKDIR and builds the
graph under WORKDIR/graph/. Subsequent runs cache-hit (or refetch if
the upstream dump has changed and the cooldown has elapsed — see
`kglite.datasets.wikidata.open` for the full lifecycle rules).

Usage:  python examples/wikidata_disk.py
"""

import time

from kglite.datasets import wikidata

WORKDIR = "/Volumes/EksternalHome/Data/Wikidata"


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"  [{(time.perf_counter() - t0) * 1000:>7.1f} ms]  {label}")
    return out


def einstein_facts(g):
    """Hello-world: identity lookup + awards for Albert Einstein (Q937)."""
    e = timed(
        "lookup Q937",
        lambda: list(g.cypher("MATCH (e {nid: 'Q937'}) RETURN e.title AS name, e.description AS desc"))[0],
    )
    awards = timed(
        "awards (P166)",
        lambda: list(g.cypher("MATCH ({nid: 'Q937'})-[:P166]->(a) RETURN a.title AS t LIMIT 5")),
    )
    print(f"\n  {e['name']} — {e['desc']}")
    print("  Awards received:")
    for a in awards:
        print(f"    • {a['t']}")


g = timed("load graph", lambda: wikidata.open(WORKDIR))
info = g.graph_info()
print(f"  Wikidata: {info['node_count']:,} nodes, {info['edge_count']:,} edges")
einstein_facts(g)
