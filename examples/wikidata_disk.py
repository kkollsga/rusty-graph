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
    """Find Einstein by name+type (no Q-code memorised), grab his awards."""
    query = """
        MATCH (e:human {title: 'Albert Einstein'})
        OPTIONAL MATCH (e)-[:P166]->(a)
        WITH e, collect(a.title)[0..5] AS awards
        RETURN e.nid AS qid, e.title AS name, e.description AS desc, awards
        LIMIT 1
    """
    e = timed("find 'Albert Einstein' + awards", lambda: list(g.cypher(query))[0])
    print(f"\n  {e['qid']}  {e['name']} — {e['desc']}")
    print("  Awards received:")
    for a in e["awards"]:
        print(f"    • {a}")


g = timed("load graph", lambda: wikidata.open(WORKDIR))
info = g.graph_info()
print(f"  Wikidata: {info['node_count']:,} nodes, {info['edge_count']:,} edges")
einstein_facts(g)
