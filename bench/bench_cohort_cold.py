"""Cold-cache cohort query benchmark.

Measures how cold-cache I/O cost compares across cohort top-K query
shapes. Designed to settle "is plan X faster than plan Y?" questions
that warm-cache timings can't answer because the OS page cache leaks
state between back-to-back queries.

Usage::

    # Compare three pre-defined shapes of the Norwegians cohort query
    # at country id=20 against /Volumes/EksternalHome/Data/Wikidata/graph
    sudo python bench/bench_cohort_cold.py

    # Custom queries: pass file paths, label them with NAME=path
    sudo python bench/bench_cohort_cold.py q1=/tmp/q1.cypher q2=/tmp/q2.cypher

    # Custom graph + iterations
    sudo python bench/bench_cohort_cold.py --graph /path/to/graph --iters 3

Each (query, iteration) pair runs in its own subprocess. Between runs
the OS page cache is dropped via ``sudo purge`` so the kglite mmap
files have to be re-faulted from disk. Sudo is required because
``purge`` needs root.

The benchmark reports a per-query table::

    shape         iter  cold-cache time
    Q1_WHERE      1     21.43s
    Q1_WHERE      2     20.85s
    Q1_WHERE      3     21.91s
    Q_NARROW      1     20.97s
    Q_NARROW      2     21.18s
    Q_NARROW      3     20.62s

…and a summary row per shape (median across iterations).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

DEFAULT_GRAPH = "/Volumes/EksternalHome/Data/Wikidata/graph"

# Built-in queries — the three shapes that drove the WHERE-on-rel-var
# investigation. All three return the same top-10 cohort by `connections`.
DEFAULT_QUERIES: dict[str, str] = {
    "Q1_WHERE": """
MATCH (p)-[:P27]->({id: 20})
WITH p
MATCH (p)-[r]-(other)
WHERE NOT (type(r) = 'P50' AND startNode(r) = other)
RETURN p.title AS name, p.description AS desc, count(r) AS connections
ORDER BY connections DESC LIMIT 10
""",
    "Q2_OPTIONAL": """
MATCH (p)-[:P27]->({id: 20})
WITH p
MATCH (p)-[r]-()
WITH p, count(r) AS total
OPTIONAL MATCH ()-[rp:P50]->(p)
RETURN p.title AS name, p.description AS desc, total - count(rp) AS connections
ORDER BY connections DESC LIMIT 10
""",
    "Q_NARROW": """
MATCH (p)-[:P27]->({id: 20})
WITH p
MATCH (p)-[r]-(other)
WHERE NOT (type(r) = 'P50' AND startNode(r) = other)
WITH p, count(r) AS connections
ORDER BY connections DESC LIMIT 10
RETURN p.title AS name, p.description AS desc, connections
""",
}

# Subprocess driver. Loads the graph, runs the query once with a long
# timeout, prints `WALL <seconds>` followed by the EXPLAIN plan and the
# top-1 row so the parent process can verify correctness.
_DRIVER = """
import sys, time, kglite

graph_path = sys.argv[1]
query = sys.stdin.read()

g = kglite.load(graph_path)
g.set_default_timeout(900_000)  # 15 min — we want completion, not a timeout

plan = [r["operation"] for r in g.cypher("EXPLAIN " + query)]
print(f"PLAN {plan}", flush=True)

t0 = time.perf_counter()
rows = list(g.cypher(query))
t = time.perf_counter() - t0
print(f"WALL {t:.4f}", flush=True)
if rows:
    print(f"TOP1 {rows[0]}", flush=True)
"""


def _purge_page_cache() -> None:
    """Drop the OS page cache. Requires sudo on macOS."""
    if sys.platform == "darwin":
        subprocess.run(["sudo", "purge"], check=True)
    elif sys.platform.startswith("linux"):
        # `echo 3 > /proc/sys/vm/drop_caches` needs root.
        subprocess.run(
            ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            check=True,
        )
    else:
        raise RuntimeError(f"don't know how to drop page cache on {sys.platform}; run on macOS or Linux")


def _run_one(graph_path: str, query: str) -> tuple[float, list[str], str]:
    """Spawn a subprocess that loads the graph and runs the query once.

    Returns (wall_seconds, plan, top1_row_repr).
    """
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, graph_path],
        input=query,
        capture_output=True,
        text=True,
        check=True,
    )
    plan: list[str] = []
    wall: float | None = None
    top1 = ""
    for line in proc.stdout.splitlines():
        if line.startswith("PLAN "):
            plan = eval(line[len("PLAN ") :])  # noqa: S307 — we wrote it
        elif line.startswith("WALL "):
            wall = float(line[len("WALL ") :])
        elif line.startswith("TOP1 "):
            top1 = line[len("TOP1 ") :]
    if wall is None:
        raise RuntimeError("driver did not emit WALL line; stdout:\n" + proc.stdout + "\nstderr:\n" + proc.stderr)
    return wall, plan, top1


def _parse_named_queries(args: list[str]) -> dict[str, str]:
    """Parse `name=path/to/file.cypher` arguments into a query dict."""
    out: dict[str, str] = {}
    for arg in args:
        if "=" not in arg:
            raise SystemExit(f"expected NAME=path, got {arg!r}")
        name, path = arg.split("=", 1)
        out[name] = Path(path).read_text()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--graph",
        default=DEFAULT_GRAPH,
        help=f"disk-graph directory (default: {DEFAULT_GRAPH})",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=3,
        help="iterations per query (default: 3)",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="skip sudo purge between runs (warm-cache benchmark)",
    )
    parser.add_argument(
        "queries",
        nargs="*",
        help="optional NAME=path.cypher pairs to override the built-in shapes",
    )
    args = parser.parse_args()

    if not Path(args.graph).exists():
        raise SystemExit(f"graph directory not found: {args.graph}")

    if args.queries:
        queries = _parse_named_queries(args.queries)
    else:
        queries = DEFAULT_QUERIES

    if not args.no_purge and sys.platform == "darwin" and shutil.which("purge") is None:
        raise SystemExit("`purge` not on PATH (need macOS or pass --no-purge)")
    if not args.no_purge:
        # Probe sudo once up-front so the user gets a single password
        # prompt instead of one per iteration.
        subprocess.run(["sudo", "-v"], check=True)

    print(f"graph     : {args.graph}")
    print(f"iterations: {args.iters} per shape")
    print(f"purge     : {'on' if not args.no_purge else 'OFF (warm-cache mode)'}")
    print()

    rows: list[tuple[str, int, float, list[str], str]] = []
    for name, query in queries.items():
        for it in range(1, args.iters + 1):
            if not args.no_purge:
                _purge_page_cache()
                # Give the OS a beat to settle after purge.
                time.sleep(0.5)
            wall, plan, top1 = _run_one(args.graph, query)
            rows.append((name, it, wall, plan, top1))
            print(f"  {name:<14}  iter {it}: {wall:7.3f}s  plan={plan}")
            if it == 1:
                print(f"      top1 = {top1}")

    print()
    print("Summary:")
    print(f"  {'shape':<14}  {'min':>8}  {'median':>8}  {'max':>8}")
    for name in queries:
        times = [w for n, _, w, _, _ in rows if n == name]
        print(f"  {name:<14}  {min(times):>7.3f}s  {statistics.median(times):>7.3f}s  {max(times):>7.3f}s")

    # Write CSV alongside the script so multiple runs accumulate.
    csv_path = Path(__file__).resolve().parent / "bench_cohort_cold.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a") as f:
        if new_file:
            f.write("timestamp,shape,iter,wall_seconds,plan\n")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        for name, it, wall, plan, _top1 in rows:
            f.write(f"{ts},{name},{it},{wall:.4f},{'|'.join(plan)}\n")
    print(f"\nAppended results to {csv_path}")


if __name__ == "__main__":
    main()
