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


def _run_bench_table(title, cases):
    """Run (label, callable) pairs with uniform timing/result rendering."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"\n  {'Query':50s} {'Time':>9s}  {'Result':>14s}")
    print("  " + "-" * 78)

    total = 0.0
    for label, fn in cases:
        t0 = time.perf_counter()
        try:
            result = fn()
            rendered = f"{result:,}" if isinstance(result, int) else str(result)[:30]
        except Exception as e:
            rendered = f"ERR: {e}"[:30]
        elapsed = time.perf_counter() - t0
        total += elapsed
        print(f"  {label:50s} {elapsed:>8.3f}s  {rendered:>14s}")

    print(f"\n  Total: {total:.2f}s across {len(cases)} queries")


def run_cypher(g):
    """Cypher coverage: identity lookups, 1-/2-hop patterns, aggregations,
    full-text scans, and parameterized queries. Each returns row count."""

    def q(cql, params=None):
        return len(g.cypher(cql, params=params).to_df())

    cases = [
        # ── Identity / point lookups ─────────────────────────────────────────
        ("count all nodes", lambda: g.cypher("MATCH (n) RETURN count(n) AS c").to_df().iloc[0, 0]),
        ("Q42 Douglas Adams", lambda: q("MATCH (n {nid:'Q42'}) RETURN n.label, n.description")),
        ("Q64 Berlin", lambda: q("MATCH (n {nid:'Q64'}) RETURN n.label, n.description")),
        ("Q76 Barack Obama", lambda: q("MATCH (n {nid:'Q76'}) RETURN n.label, n.description")),
        # ── 1-hop, bound-source, any relation ────────────────────────────────
        ("Q42 outgoing any rel (LIMIT 20)", lambda: q("MATCH ({nid:'Q42'})-[r]->(m) RETURN type(r), m.label LIMIT 20")),
        ("Q42 incoming any rel (LIMIT 20)", lambda: q("MATCH (m)-[r]->({nid:'Q42'}) RETURN type(r), m.label LIMIT 20")),
        # ── 1-hop, bound-target, typed relation — CSR index test ─────────────
        ("instances of human Q5 (LIMIT 50)", lambda: q("MATCH (n)-[:P31]->({nid:'Q5'}) RETURN n.label LIMIT 50")),
        ("instances of city Q515 (LIMIT 50)", lambda: q("MATCH (n)-[:P31]->({nid:'Q515'}) RETURN n.label LIMIT 50")),
        (
            "instances of country Q6256 (LIMIT 50)",
            lambda: q("MATCH (n)-[:P31]->({nid:'Q6256'}) RETURN n.label LIMIT 50"),
        ),
        ("born in Berlin (P19, LIMIT 20)", lambda: q("MATCH (p)-[:P19]->({nid:'Q64'}) RETURN p.label LIMIT 20")),
        ("citizens of Germany (P27, LIMIT 20)", lambda: q("MATCH (p)-[:P27]->({nid:'Q183'}) RETURN p.label LIMIT 20")),
        # ── 2-hop typed chains ───────────────────────────────────────────────
        ("Q42 direct P31? (→Q5)", lambda: q("MATCH ({nid:'Q42'})-[r:P31]->({nid:'Q5'}) RETURN type(r)")),
        ("Q42 2-hop any (LIMIT 20)", lambda: q("MATCH ({nid:'Q42'})-[]->(b)-[]->(c) RETURN b.label, c.label LIMIT 20")),
        (
            "humans born in Germany (LIMIT 20)",
            lambda: q("""MATCH (p)-[:P31]->({nid:'Q5'})
                      MATCH (p)-[:P27]->({nid:'Q183'})
                      RETURN p.label LIMIT 20"""),
        ),
        # ── Aggregations over bounded scans ──────────────────────────────────
        (
            "count cities (P31=Q515)",
            lambda: g.cypher("MATCH (n)-[:P31]->({nid:'Q515'}) RETURN count(n) AS c").to_df().iloc[0, 0],
        ),
        (
            "count countries (P31=Q6256)",
            lambda: g.cypher("MATCH (n)-[:P31]->({nid:'Q6256'}) RETURN count(n) AS c").to_df().iloc[0, 0],
        ),
        (
            "count humans (P31=Q5)",
            lambda: g.cypher("MATCH (n)-[:P31]->({nid:'Q5'}) RETURN count(n) AS c").to_df().iloc[0, 0],
        ),
        # ── Full-text scans (no index — intentionally slow reference) ────────
        (
            "label CONTAINS Einstein (LIMIT 20)",
            lambda: q("MATCH (n) WHERE n.label CONTAINS 'Einstein' RETURN n.nid, n.label LIMIT 20"),
        ),
        (
            "label CONTAINS Norway (LIMIT 20)",
            lambda: q("MATCH (n) WHERE n.label CONTAINS 'Norway' RETURN n.nid, n.label LIMIT 20"),
        ),
        (
            "label STARTS WITH 'The ' (LIMIT 20)",
            lambda: q("MATCH (n) WHERE n.label STARTS WITH 'The ' RETURN n.label LIMIT 20"),
        ),
        # ── Parameter binding + OPTIONAL MATCH ───────────────────────────────
        ("params: lookup by nid", lambda: q("MATCH (n {nid:$nid}) RETURN n.label", params={"nid": "Q76"})),
        (
            "Q42 OPTIONAL MATCH (LIMIT 30)",
            lambda: q("MATCH (n {nid:'Q42'}) OPTIONAL MATCH (n)-[r]->(m) RETURN type(r), m.label LIMIT 30"),
        ),
        # ── ORDER BY on bounded result ───────────────────────────────────────
        (
            "5 longest human labels",
            lambda: q("""MATCH (n)-[:P31]->({nid:'Q5'})
                      WITH n.label AS lbl WHERE lbl IS NOT NULL
                      RETURN lbl ORDER BY size(lbl) DESC LIMIT 5"""),
        ),
    ]

    _run_bench_table("CYPHER QUERIES", cases)


def run_fluent(g):
    """Fluent (select/traverse/where) coverage — parity with Cypher above.

    Uses the human-readable type names emitted by load_ntriples; if a given
    type isn't present in this graph the op is skipped with a note."""

    # Discover which types are available — Wikidata emits human-readable
    # names from the P31 target's label, plus a fallback "Entity" bucket
    # for entities without a resolvable P31 target.
    types_map = g.node_type_counts() or {}
    have = lambda t: t in types_map  # noqa: E731

    def safe(op):
        """Wrap an op so missing types are flagged rather than exceptions."""

        def run():
            try:
                return op()
            except Exception as e:
                return f"skipped: {e}"[:30]

        return run

    cases = [
        # ── Selects ──────────────────────────────────────────────────────────
        ("select('Entity').len()", lambda: g.select("Entity").len() if have("Entity") else "no Entity type"),
        ("select('human').len()", lambda: g.select("human").len() if have("human") else "no human type"),
        ("select('city').len()", lambda: g.select("city").len() if have("city") else "no city type"),
        ("select('country').len()", lambda: g.select("country").len() if have("country") else "no country type"),
        # ── where property equality / range (dict syntax) ────────────────────
        (
            "select('human').where(gender=male, limit=100)",
            safe(lambda: g.select("human").where({"gender": "male"}, limit=100).len()),
        ),
        (
            "select('country').where(population>10M, limit=50)",
            safe(lambda: g.select("country").where({"population": {">": 10_000_000}}, limit=50).len()),
        ),
        # ── traverse: typed out/in with limit ────────────────────────────────
        (
            "select('human').traverse(P19 out, limit=20)",
            safe(lambda: g.select("human").traverse("P19", direction="outgoing", limit=20).len()),
        ),
        (
            "select('human').traverse(P27 out, limit=20)",
            safe(lambda: g.select("human").traverse("P27", direction="outgoing", limit=20).len()),
        ),
        (
            "select('city').traverse(P17 out, limit=50)",
            safe(lambda: g.select("city").traverse("P17", direction="outgoing", limit=50).len()),
        ),
        # ── where_connected (all nodes touching a typed edge) ────────────────
        ("select('human').where_connected(P19)", safe(lambda: g.select("human").where_connected("P19").len())),
        # ── Chained traversal (2-hop) ────────────────────────────────────────
        (
            "select('human').traverse(P19).traverse(P17)",
            safe(lambda: g.select("human").traverse("P19", limit=50).traverse("P17", limit=50).len()),
        ),
        # ── String filters via where ─────────────────────────────────────────
        (
            "select('Entity').where(label contains 'Berlin')",
            safe(lambda: g.select("Entity").where({"label": {"contains": "Berlin"}}, limit=20).len()),
        ),
        (
            "select('human').where(label starts_with 'Albert Ein')",
            safe(lambda: g.select("human").where({"label": {"starts_with": "Albert Ein"}}, limit=20).len()),
        ),
        # ── Schema + introspection ───────────────────────────────────────────
        ("len(graph.describe())", lambda: len(g.describe())),
        ("len(graph_info)", lambda: len(g.graph_info())),
    ]

    _run_bench_table("FLUENT API QUERIES", cases)


def run_queries(g):
    """Backwards-compatible entry point: run Cypher then fluent suites."""
    run_cypher(g)
    run_fluent(g)


def main():
    do_build = len(sys.argv) > 1 and sys.argv[1] == "build"

    if do_build or not os.path.exists(os.path.join(GRAPH_DIR, "disk_graph_meta.json")):
        g = build()
    else:
        g = load_graph()

    run_queries(g)


if __name__ == "__main__":
    main()
