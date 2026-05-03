"""Build (or load) a disk graph from the Wikidata truthy dump.

First run downloads the dump (~80 GB) into WORKDIR and builds the
graph under WORKDIR/graph/. Subsequent runs cache-hit (or refetch if
the upstream dump has changed and the cooldown has elapsed — see
`kglite.datasets.wikidata.open` for the full lifecycle rules).

Usage:
    python examples/wikidata_disk.py
"""

import random

from kglite.datasets import wikidata

WORKDIR = "/Volumes/EksternalHome/Data/Wikidata"


g = wikidata.open(WORKDIR)

info = g.graph_info()
print(f"\nLoaded Wikidata: {info['node_count']:,} nodes, {info['edge_count']:,} edges.\n")

# Hello-world: a "did you know?" about Albert Einstein (Q937). Each
# run picks a different facet so the output stays interesting on
# repeated invocations.
einstein = list(g.cypher("MATCH (e {nid: 'Q937'}) RETURN e.title AS name, e.description AS desc"))[0]
print(f"{einstein['name']} — {einstein['desc']}")

facets = [
    ("Notable works", "P800"),
    ("Was influenced by", "P737"),
    ("Awards received", "P166"),
    ("Educated at", "P69"),
    ("Employed by", "P108"),
    ("Member of", "P463"),
    ("Worked in", "P937"),
    ("Doctoral advisor", "P184"),
    ("Spouses", "P26"),
    ("Children", "P40"),
]
label, prop = random.choice(facets)
items = [r["t"] for r in g.cypher(f"MATCH ({{nid: 'Q937'}})-[:{prop}]->(x) RETURN x.title AS t LIMIT 5")]
print(f"\nDid you know? {label}:")
for it in items:
    print(f"  • {it}")
