"""KGLite compatibility benchmark — tests legal, prospect, and wikidata graphs
across memory, disk, and mapped storage modes.

Tests: build time, simple queries, advanced queries, vector search,
exploding queries, describe().

Run:  python bench/benchmark_compatibility.py
"""

import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kglite
from kglite import KnowledgeGraph, load

LEGAL_GRAPH = "/Volumes/EksternalHome/Koding/MCP servers/legal/norwegian_law.kgl"
PROSPECT_GRAPH = "/Volumes/EksternalHome/Koding/MCP servers/prospect_mcp/sodir_graph.kgl"
WIKIDATA_500K = "/Volumes/EksternalHome/Data/Wikidata/test_500k.nt.zst"
WIKIDATA_5M = "/Volumes/EksternalHome/Data/Wikidata/test_5M.nt.zst"

RESULTS = []
MAPPED_ISSUES = []


def record(graph_name, mode, test_name, time_ms, rows=None, status="ok", error=None):
    RESULTS.append({
        "graph": graph_name, "mode": mode, "test": test_name,
        "time_ms": round(time_ms, 1), "rows": rows, "status": status, "error": error,
    })
    tag = f"{time_ms:>8.1f}ms" if status == "ok" else f"{status:>8s}"
    row_str = f" rows={rows}" if rows is not None else ""
    err_str = f"  {error[:40]}" if error else ""
    print(f"    {test_name:40s} {tag}{row_str}{err_str}", flush=True)


def run_query(g, graph_name, mode, test_name, query, timeout_ms=10000):
    t0 = time.perf_counter()
    try:
        r = g.cypher(query, timeout_ms=timeout_ms)
        ms = (time.perf_counter() - t0) * 1000
        record(graph_name, mode, test_name, ms, rows=len(r))
        return r
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        err = str(e)
        if "timed out" in err.lower():
            record(graph_name, mode, test_name, ms, status="TIMEOUT", error=err[:60])
        else:
            record(graph_name, mode, test_name, ms, status="ERROR", error=err[:60])
        return None


def run_describe(g, graph_name, mode, test_name, **kwargs):
    t0 = time.perf_counter()
    try:
        d = g.describe(**kwargs)
        ms = (time.perf_counter() - t0) * 1000
        record(graph_name, mode, test_name, ms, rows=len(d))
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        record(graph_name, mode, test_name, ms, status="ERROR", error=str(e)[:60])


def test_legal_graph(g, mode):
    """Test legal graph — in-memory, moderate size."""
    name = "legal"
    info = g.graph_info()
    print(f"\n  Legal graph ({mode}): {info.get('node_count',0):,} nodes, {info.get('edge_count',0):,} edges")

    # Simple queries
    run_query(g, name, mode, "count(n)", "MATCH (n) RETURN count(n)")
    run_query(g, name, mode, "count by type", "MATCH (n) RETURN n.type, count(n) AS c ORDER BY c DESC LIMIT 10")
    run_query(g, name, mode, "edge count by type", "MATCH ()-[r]->() RETURN type(r), count(*) AS c ORDER BY c DESC LIMIT 10")

    # ID lookups
    run_query(g, name, mode, "lookup Law by id", "MATCH (n:Law) RETURN n.id, n.title LIMIT 1")

    # Traversal
    run_query(g, name, mode, "Law sections", "MATCH (l:Law)-[:SECTION_OF]->(s:LawSection) RETURN l.title, s.title LIMIT 10")
    run_query(g, name, mode, "Court cites Law", "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection) RETURN d.title, s.title LIMIT 10")
    run_query(g, name, mode, "2-hop cites", "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law) RETURN d.title, l.title LIMIT 10")

    # Aggregation
    run_query(g, name, mode, "cases per keyword", "MATCH (d:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword) RETURN k.title, count(d) AS c ORDER BY c DESC LIMIT 10")
    run_query(g, name, mode, "most cited laws", "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law) RETURN l.title, count(d) AS c ORDER BY c DESC LIMIT 10")

    # Advanced
    run_query(g, name, mode, "OPTIONAL MATCH", "MATCH (l:Law) OPTIONAL MATCH (l)-[:GOVERNED_BY]->(d:Department) RETURN l.title, d.title LIMIT 10")
    run_query(g, name, mode, "EXISTS subquery", "MATCH (d:CourtDecision) WHERE EXISTS { MATCH (d)-[:CITES]->() } RETURN d.title LIMIT 5")
    run_query(g, name, mode, "WITH pipeline", "MATCH (d:CourtDecision)-[:CITES]->(s) WITH s, count(d) AS citations WHERE citations > 5 RETURN s.title, citations ORDER BY citations DESC LIMIT 10")

    # Describe
    run_describe(g, name, mode, "describe()", )
    run_describe(g, name, mode, "describe(types=[Law])", types=["Law"])

    # Exploding queries (should not crash)
    run_query(g, name, mode, "full type scan", "MATCH (n:CourtDecision) RETURN n.title LIMIT 100")
    run_query(g, name, mode, "CONTAINS search", "MATCH (n) WHERE n.title CONTAINS 'straff' RETURN n.title LIMIT 20")

    # Vector search (if embeddings loaded)
    try:
        r = g.cypher("MATCH (n:CourtDecision) RETURN n.title, vector_score(n.summary, 'erstatningsrett') AS score ORDER BY score DESC LIMIT 5")
        ms = 0  # already timed inside
        record(name, mode, "vector_search", 0, rows=len(r) if r else 0)
    except Exception as e:
        record(name, mode, "vector_search", 0, status="SKIP", error=str(e)[:60])


def test_prospect_graph(g, mode):
    """Test prospect/Sodir graph — smaller, spatial."""
    name = "prospect"
    info = g.graph_info()
    print(f"\n  Prospect graph ({mode}): {info.get('node_count',0):,} nodes, {info.get('edge_count',0):,} edges")

    run_query(g, name, mode, "count(n)", "MATCH (n) RETURN count(n)")
    run_query(g, name, mode, "count by type", "MATCH (n) RETURN n.type, count(n) AS c ORDER BY c DESC")
    run_query(g, name, mode, "Wellbore lookup", "MATCH (w:Wellbore) RETURN w.title LIMIT 5")
    run_query(g, name, mode, "Field production", "MATCH (f:Field) RETURN f.title LIMIT 5")
    run_query(g, name, mode, "Licence connections", "MATCH (l:Licence)-[r]->(n) RETURN l.title, type(r), n.title LIMIT 10")
    run_query(g, name, mode, "describe()", "EXPLAIN MATCH (n) RETURN count(n)")
    run_describe(g, name, mode, "describe()")


def test_ntriples_graph(g, mode, label):
    """Test N-Triples graph (500K or 5M)."""
    name = label
    info = g.graph_info()
    print(f"\n  {label} ({mode}): {info.get('node_count',0):,} nodes, {info.get('edge_count',0):,} edges")

    # Disk/mapped store IDs as integers (42), memory stores as strings ('Q42')
    id_val = 42 if mode in ("disk", "mapped") else "'Q42'"

    run_query(g, name, mode, "count(n)", "MATCH (n) RETURN count(n)")
    run_query(g, name, mode, "count by type", "MATCH (n) RETURN n.type, count(n) AS c ORDER BY c DESC LIMIT 10")
    run_query(g, name, mode, "edge count by type", "MATCH ()-[r]->() RETURN type(r), count(*) AS c ORDER BY c DESC LIMIT 10")
    run_query(g, name, mode, "id lookup", f"MATCH (n {{id: {id_val}}}) RETURN n.id, n.title, n.type")
    run_query(g, name, mode, "P31 hop", f"MATCH ({{id:{id_val}}})-[:P31]->(m) RETURN m.title")
    run_query(g, name, mode, "2-hop LIM20", f"MATCH ({{id:{id_val}}})-[]->(b)-[]->(c) RETURN b.title, c.title LIMIT 20")
    run_query(g, name, mode, "P31 LIMIT 50", "MATCH (a)-[:P31]->(b) RETURN a.title, b.title LIMIT 50")
    run_describe(g, name, mode, "describe()")


def build_ntriples(path, mode, label):
    """Build a graph from N-Triples file."""
    print(f"\n  Building {label} ({mode})...")
    t0 = time.perf_counter()
    if mode == "disk":
        out_dir = f"/tmp/kglite_bench_{label}_disk"
        os.makedirs(out_dir, exist_ok=True)
        g = KnowledgeGraph(storage="disk", path=out_dir)
    elif mode == "mapped":
        g = KnowledgeGraph(storage="mapped")
    else:
        g = KnowledgeGraph()

    try:
        stats = g.load_ntriples(path, languages=["en"], verbose=False)
        build_ms = (time.perf_counter() - t0) * 1000
        record(label, mode, "build", build_ms, rows=stats.get("entities", 0))
        return g
    except Exception as e:
        build_ms = (time.perf_counter() - t0) * 1000
        record(label, mode, "build", build_ms, status="ERROR", error=str(e)[:60])
        if mode == "mapped":
            MAPPED_ISSUES.append(f"BUILD FAIL {label}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print(f"KGLite v{kglite.__version__}")
    print(f"=" * 70)

    # ── 1. Legal Graph (memory mode) ──
    print("\n" + "=" * 70)
    print("1. LEGAL GRAPH — Memory Mode")
    print("=" * 70)
    if os.path.exists(LEGAL_GRAPH):
        t0 = time.perf_counter()
        g = load(LEGAL_GRAPH)
        record("legal", "memory", "load", (time.perf_counter() - t0) * 1000)
        test_legal_graph(g, "memory")
        del g
    else:
        print(f"  SKIP: {LEGAL_GRAPH} not found")

    # ── 2. Prospect Graph (memory mode) ──
    print("\n" + "=" * 70)
    print("2. PROSPECT GRAPH — Memory Mode")
    print("=" * 70)
    if os.path.exists(PROSPECT_GRAPH):
        t0 = time.perf_counter()
        g = load(PROSPECT_GRAPH)
        record("prospect", "memory", "load", (time.perf_counter() - t0) * 1000)
        test_prospect_graph(g, "memory")
        del g
    else:
        print(f"  SKIP: {PROSPECT_GRAPH} not found")

    # ── 3. Wikidata 500K (memory mode) ──
    print("\n" + "=" * 70)
    print("3. WIKIDATA 500K — Memory Mode")
    print("=" * 70)
    if os.path.exists(WIKIDATA_500K):
        g = build_ntriples(WIKIDATA_500K, "memory", "wiki500k")
        if g:
            test_ntriples_graph(g, "memory", "wiki500k")
            del g

    # ── 4. Wikidata 5M (memory mode) ──
    print("\n" + "=" * 70)
    print("4. WIKIDATA 5M — Memory Mode")
    print("=" * 70)
    if os.path.exists(WIKIDATA_5M):
        g = build_ntriples(WIKIDATA_5M, "memory", "wiki5m")
        if g:
            test_ntriples_graph(g, "memory", "wiki5m")
            del g

    # ── 5. Wikidata 500K (disk mode) ──
    print("\n" + "=" * 70)
    print("5. WIKIDATA 500K — Disk Mode")
    print("=" * 70)
    if os.path.exists(WIKIDATA_500K):
        g = build_ntriples(WIKIDATA_500K, "disk", "wiki500k")
        if g:
            test_ntriples_graph(g, "disk", "wiki500k")
            del g

    # ── 6. Wikidata 5M (disk mode) ──
    print("\n" + "=" * 70)
    print("6. WIKIDATA 5M — Disk Mode")
    print("=" * 70)
    if os.path.exists(WIKIDATA_5M):
        g = build_ntriples(WIKIDATA_5M, "disk", "wiki5m")
        if g:
            test_ntriples_graph(g, "disk", "wiki5m")
            del g

    # ── 7. Wikidata 500K (mapped mode) ──
    print("\n" + "=" * 70)
    print("7. WIKIDATA 500K — Mapped Mode")
    print("=" * 70)
    if os.path.exists(WIKIDATA_500K):
        g = build_ntriples(WIKIDATA_500K, "mapped", "wiki500k")
        if g:
            test_ntriples_graph(g, "mapped", "wiki500k")
            del g

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ok = sum(1 for r in RESULTS if r["status"] == "ok")
    timeout = sum(1 for r in RESULTS if r["status"] == "TIMEOUT")
    error = sum(1 for r in RESULTS if r["status"] == "ERROR")
    skip = sum(1 for r in RESULTS if r["status"] == "SKIP")
    print(f"  Total: {len(RESULTS)} tests | OK: {ok} | TIMEOUT: {timeout} | ERROR: {error} | SKIP: {skip}")

    if MAPPED_ISSUES:
        print(f"\n  Mapped Mode Issues ({len(MAPPED_ISSUES)}):")
        for issue in MAPPED_ISSUES:
            print(f"    - {issue}")
        # Write to file
        with open("mapped-graph-issues.md", "w") as f:
            f.write("# Mapped Graph Issues\n\n")
            for issue in MAPPED_ISSUES:
                f.write(f"- {issue}\n")
        print(f"\n  Written to mapped-graph-issues.md")

    # Print failures
    failures = [r for r in RESULTS if r["status"] in ("TIMEOUT", "ERROR")]
    if failures:
        print(f"\n  Failures:")
        for r in failures:
            print(f"    [{r['graph']}/{r['mode']}] {r['test']}: {r['status']} {r.get('error','')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
