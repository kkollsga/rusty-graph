"""Build a Wikidata disk graph from the N-Triples truthy dump.

Usage:
    python examples/wikidata_disk.py build   # build from source
    python examples/wikidata_disk.py         # load existing graph

Adjust SOURCE / GRAPH below to match your paths.
"""

import sys

import kglite

SOURCE = "/Volumes/EksternalHome/Data/Wikidata/latest-truthy.nt.zst"
GRAPH = "/Volumes/EksternalHome/Data/Wikidata/wikidata_disk_graph"


if len(sys.argv) > 1 and sys.argv[1] == "build":
    g = kglite.KnowledgeGraph(storage="disk", path=GRAPH)
    g.load_ntriples(SOURCE, languages=["en"], verbose=True)
else:
    g = kglite.load(GRAPH)

info = g.graph_info()
print(f"Nodes: {info['node_count']:,}")
print(f"Edges: {info['edge_count']:,}")
