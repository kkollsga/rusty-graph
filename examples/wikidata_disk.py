"""Build (or load) a disk graph from the Wikidata truthy dump.

First run downloads the dump (~80 GB) into WORKDIR and builds the
graph under WORKDIR/graph/. Subsequent runs cache-hit (or refetch if
the upstream dump has changed and the cooldown has elapsed — see
`kglite.datasets.wikidata.open` for the full lifecycle rules).

Usage:
    python examples/wikidata_disk.py
"""

from kglite.datasets import wikidata

WORKDIR = "/Volumes/EksternalHome/Data/Wikidata"


g = wikidata.open(WORKDIR)

info = g.graph_info()
print(f"\nLoaded Wikidata: {info['node_count']:,} nodes, {info['edge_count']:,} edges.\n")

# Hello-world: Albert Einstein (Q937), born in Ulm, theoretical physicist.
einstein = list(g.cypher("MATCH (e {nid: 'Q937'}) RETURN e.title AS name, e.description AS desc"))[0]
print(f"{einstein['name']} — {einstein['desc']}")

works = list(g.cypher("MATCH (e {nid: 'Q937'})-[:P800]->(w) RETURN w.title AS t LIMIT 5"))
print("Notable works:")
for w in works:
    print(f"  • {w['t']}")
