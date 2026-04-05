"""
Phase 3 ranking benchmark — compare CSR build algorithm variants.

Bypasses N-Triples entirely: directly populates pending_edges with synthetic
edges, then runs build_csr_from_pending(). Tests at multiple scales to
reveal I/O patterns vs CPU-bound behavior.

After changing Rust code: `maturin develop --release && python tests/benchmark_phase3_ranking.py`

Usage:
    python tests/benchmark_phase3_ranking.py              # default scales
    python tests/benchmark_phase3_ranking.py 50000000     # specific edge count
"""

import os
import resource
import subprocess
import sys
import tempfile

os.environ["KGLITE_CSR_VERBOSE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kglite import KnowledgeGraph


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def current_rss_mb():
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return int(out.strip()) / 1024
    except Exception:
        return -1


def run_single(edge_count, node_count, label=""):
    """Run one Phase 3 benchmark. Returns timing dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_dir = os.path.join(tmpdir, "bench_graph")
        g = KnowledgeGraph(storage="disk", path=graph_dir)

        current_rss_mb()
        result = g._bench_phase3(edge_count, node_count, graph_dir)
        rss_after = peak_rss_mb()

        ok = result["out_edges_ok"] and result["in_edges_ok"] and result["endpoints_ok"]
        elapsed = result["elapsed"]

        edges_per_sec = edge_count / elapsed if elapsed > 0 else float("inf")
        array_mb = edge_count * 16 / 1024 / 1024

        status = "OK" if ok else "FAIL"
        print(
            f"  {label:20s}  {edge_count:>12,} edges  "
            f"{elapsed:>6.1f}s  {edges_per_sec / 1e6:>5.1f}M/s  "
            f"{array_mb:>7.0f} MB/arr  {rss_after:>6.0f} MB peak  [{status}]"
        )
        return {
            "edges": edge_count,
            "elapsed": elapsed,
            "edges_per_sec": edges_per_sec,
            "array_mb": array_mb,
            "peak_rss": rss_after,
            "ok": ok,
        }


def main():
    print("=" * 90)
    print("PHASE 3 CSR BUILD — RANKING BENCHMARK")
    print("=" * 90)
    print()
    print("Synthetic edges, source-grouped (mimics Wikidata N-Triples ordering).")
    print("Set KGLITE_CSR_VERBOSE=1 for sub-step timings on stderr.")
    print()

    # Determine scales to test
    if len(sys.argv) > 1:
        scales = [(int(sys.argv[1]), max(int(sys.argv[1]) // 25, 1000))]
    else:
        scales = [
            (1_000_000, 40_000),  # 1M edges: baseline, everything in L3 cache
            (10_000_000, 400_000),  # 10M: fits in RAM, shows sort scaling
            (50_000_000, 2_000_000),  # 50M: ~800 MB per array, moderate scale
            (100_000_000, 4_000_000),  # 100M: ~1.6 GB per array, starts to stress
            (200_000_000, 8_000_000),  # 200M: ~3.2 GB per array, significant pressure
        ]

    hdr = f"  {'Label':20s}  {'Edges':>12s}  {'Time':>6s}  {'Rate':>7s}  {'Array':>10s}  {'Peak':>9s}  Status"
    print(hdr)
    print("  " + "-" * 85)

    results = []
    for edge_count, node_count in scales:
        label = f"{edge_count / 1e6:.0f}M edges"
        r = run_single(edge_count, node_count, label)
        results.append(r)
        print()  # blank line between stderr output and next run

    # Summary
    print()
    print("=" * 90)
    print("SCALING SUMMARY")
    print("=" * 90)
    if len(results) >= 2:
        base = results[0]
        print(f"  {'Scale':>12s}  {'Time':>6s}  {'Rate':>7s}  {'Scaling':>10s}")
        print("  " + "-" * 42)
        for r in results:
            ratio = r["edges"] / base["edges"]
            time_ratio = r["elapsed"] / base["elapsed"] if base["elapsed"] > 0 else 0
            # O(n log n) expected ratio
            ratio * (1 + 0.1 * (ratio - 1))  # rough n log n scaling
            print(
                f"  {r['edges'] / 1e6:>10.0f}M  {r['elapsed']:>5.1f}s  "
                f"{r['edges_per_sec'] / 1e6:>5.1f}M/s  "
                f"{time_ratio:>5.1f}x actual"
            )

    all_ok = all(r["ok"] for r in results)
    print()
    print(f"  All correct: {'YES' if all_ok else 'NO — REGRESSION DETECTED'}")
    print("=" * 90)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
