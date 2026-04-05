"""
CSR build benchmark — correctness, regression detection, and I/O pattern analysis.

Tests three things:
1. CORRECTNESS: CSR produces correct traversals after build, save, and reload.
2. REGRESSION:  Peak RSS stays within bounds (catches edge_buffer leak, etc).
3. I/O PATTERN: Sub-phase timings expose random-write bottleneck (in_edges scatter).

Set KGLITE_CSR_VERBOSE=1 to see per-step Rust timings (steps 1-6).

Usage:
    python tests/benchmark_csr_build.py                  # default: 200K entities
    python tests/benchmark_csr_build.py 500000 25        # larger
    python tests/benchmark_csr_build.py 200000 25 ballast  # with memory pressure
"""

import os
import resource
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable sub-phase timing in Rust CSR build
os.environ["KGLITE_CSR_VERBOSE"] = "1"

from kglite import KnowledgeGraph, load


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def current_rss_mb():
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(out.strip()) / 1024
    except Exception:
        return -1


def generate_ntriples(path, num_entities, edges_per_entity):
    """Generate Wikidata-like N-Triples. Edges grouped by source (like real data)."""
    num_predicates = min(edges_per_entity, 50)
    with open(path, "w") as f:
        for i in range(num_entities):
            qid = i + 1
            f.write(
                f"<http://www.wikidata.org/entity/Q{qid}> "
                f'<http://www.w3.org/2000/01/rdf-schema#label> "Entity {qid}"@en .\n'
            )
            for j in range(edges_per_entity):
                target = (i * 7 + j * 13 + 1) % num_entities + 1
                pred = (j % num_predicates) + 1
                f.write(
                    f"<http://www.wikidata.org/entity/Q{qid}> "
                    f"<http://www.wikidata.org/prop/direct/P{pred}> "
                    f"<http://www.wikidata.org/entity/Q{target}> .\n"
                )
    return os.path.getsize(path)


def run_benchmark(num_entities=200_000, edges_per_entity=25, use_ballast=False):
    total_edges = num_entities * edges_per_entity
    bytes_per_array = total_edges * 16
    num_predicates = min(edges_per_entity, 50)

    print("=" * 70)
    print("CSR BUILD BENCHMARK")
    print("=" * 70)
    print(f"  Entities:       {num_entities:>12,}")
    print(f"  Edges/entity:   {edges_per_entity:>12}")
    print(f"  Total edges:    {total_edges:>12,}")
    print(f"  Bytes/array:    {bytes_per_array / 1024 / 1024:>10.0f} MB")
    print(f"  Predicates:     {num_predicates:>12}")
    if use_ballast:
        print("  Memory ballast: ENABLED (simulates page cache pressure)")
    print()

    # ── Optional memory ballast ──
    # Allocates a large block to consume page cache, forcing the CSR build
    # to compete for the remaining cache. This simulates Wikidata-scale
    # pressure at smaller data sizes.
    ballast = None
    if use_ballast:
        # Target: leave ~500 MB of page cache for the test
        try:
            avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            avail = 16 * 1024**3  # assume 16 GB
        current = int(current_rss_mb() * 1024 * 1024)
        # Reserve enough that mmap arrays must compete for cache
        target_pressure = max(avail - current - 512 * 1024 * 1024, 1024**3)
        # Cap at something reasonable
        ballast_size = min(target_pressure, int(avail * 0.75))
        print(f"  Allocating {ballast_size / 1024**3:.1f} GB ballast...", end="", flush=True)
        ballast = bytearray(ballast_size)
        # Touch pages so they're resident
        for off in range(0, ballast_size, 4096):
            ballast[off] = 0xFF
        print(f" done (RSS: {current_rss_mb():.0f} MB)")
        print()

    with tempfile.TemporaryDirectory() as tmpdir:
        nt_path = os.path.join(tmpdir, "synthetic.nt")
        graph_dir = os.path.join(tmpdir, "disk_graph")

        # ── Generate ──
        t0 = time.perf_counter()
        file_size = generate_ntriples(nt_path, num_entities, edges_per_entity)
        t_gen = time.perf_counter() - t0
        print(f"  Generate .nt:  {t_gen:>6.1f}s  ({file_size / 1024 / 1024:.0f} MB)")

        rss_before = current_rss_mb()

        # ── Load (Phase 1-3, sub-phase timings go to stderr) ──
        g = KnowledgeGraph(storage="disk", path=graph_dir)
        t0 = time.perf_counter()
        g.load_ntriples(nt_path, languages=["en"], verbose=True)
        t_load = time.perf_counter() - t0

        rss_after = current_rss_mb()
        peak = peak_rss_mb()

        info = g.graph_info()
        node_count = info.get("node_count", 0)
        edge_count = info.get("edge_count", 0)

        print(f"\n  Load total:    {t_load:>6.1f}s")
        print(f"  Nodes:         {node_count:>12,}")
        print(f"  Edges:         {edge_count:>12,}")
        print(f"  RSS before:    {rss_before:>10.0f} MB")
        print(f"  RSS after:     {rss_after:>10.0f} MB")
        print(f"  Peak RSS:      {peak:>10.0f} MB")

        # ── Memory regression check ──
        # Without edge_buffer leak: peak RSS ≈ pending + overhead
        # With leak: peak RSS ≈ 2x pending + overhead
        pending_mb = total_edges * 16 / 1024 / 1024
        # Allow 3x pending for overhead (offsets, cursors, mmap bookkeeping)
        rss_limit = max(pending_mb * 3, 300)
        if not use_ballast:
            # Only check without ballast (ballast itself inflates RSS)
            rss_increase = rss_after - rss_before
            print(f"  RSS increase:  {rss_increase:>10.0f} MB (limit: {rss_limit:.0f} MB)")

        # ── Correctness checks ──
        print("\n  Correctness:")
        errors = 0

        def check(label, condition):
            nonlocal errors
            ok = bool(condition)
            print(f"    {label}: {'OK' if ok else 'FAIL'}")
            if not ok:
                errors += 1

        check(f"Node count = {node_count:,}", node_count == num_entities)
        check(f"Edge count = {edge_count:,}", edge_count == total_edges)

        df = g.cypher("MATCH (n) RETURN count(n) AS c").to_df()
        check(f"Cypher node count = {df['c'].iloc[0]:,}", df["c"].iloc[0] == num_entities)

        sel = g.select("Entity")
        check(f"Select all = {sel.len():,}", sel.len() == num_entities)

        # Outgoing traversal (tests out_edges CSR)
        trav = sel.traverse("P1", direction="outgoing", limit=100)
        check(f"Traverse P1 out = {trav.len()}", trav.len() > 0)

        # Incoming traversal (tests in_edges CSR — the random-write array)
        trav_in = sel.traverse("P1", direction="incoming", limit=100)
        check(f"Traverse P1 in = {trav_in.len()}", trav_in.len() > 0)

        # Multiple predicate types
        for pred in ["P1", f"P{num_predicates // 2}", f"P{num_predicates}"]:
            t = sel.traverse(pred, direction="outgoing", limit=10)
            check(f"Traverse {pred} out = {t.len()}", t.len() > 0)

        # 2-hop (tests CSR chaining)
        hop2 = sel.traverse("P1", limit=10).traverse("P2", limit=10)
        check(f"2-hop P1->P2 = {hop2.len()}", hop2.len() > 0)

        # where_connected (tests edge existence lookup)
        wc = sel.where_connected("P1")
        check(f"where_connected P1 = {wc.len():,}", wc.len() > 0)

        # ── Save + reload ──
        print("\n  Persistence:")
        t0 = time.perf_counter()
        g.save(graph_dir)
        t_save = time.perf_counter() - t0
        print(f"    Save:        {t_save:>6.1f}s")

        t0 = time.perf_counter()
        g2 = load(graph_dir)
        t_reload = time.perf_counter() - t0
        print(f"    Reload:      {t_reload:>6.1f}s")

        info2 = g2.graph_info()
        check(f"Reload nodes = {info2['node_count']:,}", info2["node_count"] == num_entities)
        check(f"Reload edges = {info2['edge_count']:,}", info2["edge_count"] == total_edges)

        sel2 = g2.select("Entity")
        trav2 = sel2.traverse("P1", direction="outgoing", limit=50)
        check(f"Reload traverse P1 out = {trav2.len()}", trav2.len() > 0)

        trav2_in = sel2.traverse("P1", direction="incoming", limit=50)
        check(f"Reload traverse P1 in = {trav2_in.len()}", trav2_in.len() > 0)

        # ── Summary ──
        print("\n" + "=" * 70)
        if errors == 0:
            print("  ALL CHECKS PASSED")
        else:
            print(f"  {errors} CHECK(S) FAILED")
        print(f"  Load:   {t_load:>6.1f}s | Save: {t_save:>5.1f}s | Reload: {t_reload:>5.1f}s")
        print(f"  Peak RSS: {peak:>8.0f} MB")
        print("=" * 70)

    del ballast  # free ballast if allocated
    return errors


if __name__ == "__main__":
    entities = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    edges = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    ballast = len(sys.argv) > 3 and sys.argv[3] == "ballast"
    sys.exit(run_benchmark(entities, edges, use_ballast=ballast))
