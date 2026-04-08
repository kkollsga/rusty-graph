"""
Wikidata disk graph: build from N-Triples, load, and run queries.

Build:  python examples/wikidata_disk.py build
Load:   python examples/wikidata_disk.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kglite import KnowledgeGraph, load

WIKIDATA_ZST = "/Volumes/EksternalHome/Data/Wikidata/latest-truthy.nt.zst"
GRAPH_DIR = "/Volumes/EksternalHome/Data/Wikidata/wikidata_disk_graph"

# Enable sub-step timing for CSR build
os.environ["KGLITE_CSR_VERBOSE"] = "1"
# Use merge sort CSR algorithm — all sequential I/O, no random scatter
os.environ["KGLITE_CSR_ALGO"] = "merge_sort"


def build():
    print("=" * 70)
    print("BUILD WIKIDATA DISK GRAPH")
    print("=" * 70)
    print(f"  Source: {WIKIDATA_ZST}")
    print(f"  Target: {GRAPH_DIR}")
    print()

    g = KnowledgeGraph(storage="disk", path=GRAPH_DIR)

    t0 = time.perf_counter()
    g.load_ntriples(WIKIDATA_ZST, languages=["en"], verbose=True)
    t_total = time.perf_counter() - t0

    info = g.graph_info()
    print("\n  Build complete:")
    print(f"    Nodes: {info.get('node_count', 0):,}")
    print(f"    Edges: {info.get('edge_count', 0):,}")
    print(f"    Time:  {t_total:.1f}s ({t_total / 60:.1f} min)")

    total_bytes = sum(os.path.getsize(os.path.join(r, f)) for r, _, files in os.walk(GRAPH_DIR) for f in files)
    print(f"    Disk:  {total_bytes / (1024**3):.2f} GB")
    return g


def load_graph():
    print("=" * 70)
    print("LOAD WIKIDATA DISK GRAPH")
    print("=" * 70)

    t0 = time.perf_counter()
    g = load(GRAPH_DIR)
    t_load = time.perf_counter() - t0

    info = g.graph_info()
    print(f"  Loaded in {t_load:.1f}s")
    print(f"  Nodes: {info.get('node_count', 0):,}")
    print(f"  Edges: {info.get('edge_count', 0):,}")
    return g


def run_queries(g):
    print("\n" + "=" * 70)
    print("CYPHER QUERIES")
    print("=" * 70)

    queries = [
        ("Count nodes", "MATCH (n) RETURN count(n) AS c"),
        ("Q42 Douglas Adams", "MATCH (n {nid: 'Q42'}) RETURN n.label, n.description"),
        ("Q64 Berlin", "MATCH (n {nid: 'Q64'}) RETURN n.label, n.description"),
        ("Q76 Barack Obama", "MATCH (n {nid: 'Q76'}) RETURN n.label, n.description"),
        ("Q42 outgoing", "MATCH ({nid:'Q42'})-[r]->(m) RETURN type(r), m.label LIMIT 20"),
        ("Q42 incoming", "MATCH (m)-[r]->({nid:'Q42'}) RETURN type(r), m.label LIMIT 20"),
        ("Q76 outgoing", "MATCH ({nid:'Q76'})-[r]->(m) RETURN type(r), m.label LIMIT 20"),
        ("Instances of Human", "MATCH (n)-[:P31]->({nid:'Q5'}) RETURN n.label LIMIT 50"),
        ("Instances of City", "MATCH (n)-[:P31]->({nid:'Q515'}) RETURN n.label LIMIT 50"),
        ("Instances of Country", "MATCH (n)-[:P31]->({nid:'Q6256'}) RETURN n.label LIMIT 50"),
        ("Q42 2-hop", "MATCH ({nid:'Q42'})-[]->(b)-[]->(c) RETURN b.label, c.label LIMIT 20"),
        ("Born in Berlin", "MATCH (p)-[:P19]->({nid:'Q64'}) RETURN p.label LIMIT 20"),
        ("Contains Einstein", "MATCH (n) WHERE n.label CONTAINS 'Einstein' RETURN n.nid, n.label LIMIT 20"),
        ("Contains Norway", "MATCH (n) WHERE n.label CONTAINS 'Norway' RETURN n.nid, n.label LIMIT 20"),
        ("Q42->Q5 direct?", "MATCH ({nid:'Q42'})-[r]->({nid:'Q5'}) RETURN type(r)"),
        ("Q42 optional", "MATCH (n {nid:'Q42'}) OPTIONAL MATCH (n)-[r]->(m) RETURN type(r), m.label LIMIT 30"),
    ]

    print(f"\n  {'Query':40s} {'Time':>8s}  {'Rows':>8s}")
    print("  " + "-" * 60)

    total = 0.0
    for label, query in queries:
        t0 = time.perf_counter()
        try:
            df = g.cypher(query).to_df()
            rows = len(df)
        except Exception as e:
            rows = f"ERR: {e}"
        elapsed = time.perf_counter() - t0
        total += elapsed
        r = f"{rows:,}" if isinstance(rows, int) else str(rows)[:30]
        print(f"  {label:40s} {elapsed:>7.3f}s  {r:>8s}")

    print(f"\n  Total: {total:.2f}s across {len(queries)} queries")

    print("\n" + "=" * 70)
    print("FLUENT API QUERIES")
    print("=" * 70)

    s = g.select("Entity")
    fluent_ops = [
        ("select(Entity).len()", lambda: s.len()),
        ("traverse P31 out limit 100", lambda: s.traverse("P31", direction="outgoing", limit=100).len()),
        ("traverse P31 in limit 50", lambda: s.traverse("P31", direction="incoming", limit=50).len()),
        ("where label contains Berlin", lambda: s.where_({"label": ("contains", "Berlin")}, limit=20).len()),
        ("where P31 = Q5 limit 50", lambda: s.where_({"P31": "Q5"}, limit=50).len()),
        ("where_connected P31", lambda: s.where_connected("P31").len()),
        ("2-hop P31 then P17", lambda: s.traverse("P31", limit=10).traverse("P17", limit=10).len()),
        ("describe() length", lambda: len(g.describe())),
    ]

    print(f"\n  {'Operation':40s} {'Time':>8s}  {'Result':>12s}")
    print("  " + "-" * 64)

    total = 0.0
    for label, fn in fluent_ops:
        t0 = time.perf_counter()
        try:
            result = fn()
        except Exception as e:
            result = f"ERR: {e}"
        elapsed = time.perf_counter() - t0
        total += elapsed
        r = f"{result:,}" if isinstance(result, int) else str(result)[:30]
        print(f"  {label:40s} {elapsed:>7.3f}s  {r:>12s}")

    print(f"\n  Total: {total:.2f}s across {len(fluent_ops)} fluent queries")


def main():
    do_build = len(sys.argv) > 1 and sys.argv[1] == "build"

    if do_build or not os.path.exists(os.path.join(GRAPH_DIR, "disk_graph_meta.json")):
        g = build()
    else:
        g = load_graph()

    run_queries(g)


if __name__ == "__main__":
    main()
