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
print(f"Nodes: {info['node_count']:,}")
print(f"Edges: {info['edge_count']:,}")
