"""Baseline benchmark for Tier B perf items.

Covers the four hot paths touched by the Tier B changes:
- property iteration (keys/map projection / PropertyKeyIter)
- count(DISTINCT) over nodes
- substring() across rows
- function-heavy scan (exercises to_lowercase on function names)

Run: python tests/benchmarks/bench_tier_b.py
Optionally set KGLITE_BENCH_ITERS to change iteration count.
"""

import os
import time

import pandas as pd

from kglite import KnowledgeGraph

ITERS = int(os.environ.get("KGLITE_BENCH_ITERS", "5"))
NODES = int(os.environ.get("KGLITE_BENCH_NODES", "50000"))


def build_graph(n: int) -> KnowledgeGraph:
    g = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "pid": range(n),
            "title": [f"person_{i:05d}" for i in range(n)],
            "email": [f"user{i}@example.com" for i in range(n)],
            "age": [20 + (i % 60) for i in range(n)],
            "dept": [["eng", "ops", "sales", "hr"][i % 4] for i in range(n)],
        }
    )
    g.add_nodes(df, "Person", "pid", "title")
    return g


def timeit(label: str, fn) -> float:
    # warmup
    fn()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    best = min(samples) * 1000
    median = sorted(samples)[len(samples) // 2] * 1000
    print(f"  {label:<40s} best={best:7.2f} ms  median={median:7.2f} ms")
    return best


def main():
    print(f"Building graph with {NODES:,} Person nodes...")
    g = build_graph(NODES)
    print(f"ITERS={ITERS}\n")

    print("== property iteration (keys() / map projection) ==")
    timeit("keys(n) over all nodes", lambda: g.cypher("MATCH (n:Person) RETURN keys(n) AS k"))
    timeit(
        "map projection n {.*}",
        lambda: g.cypher("MATCH (n:Person) RETURN n {.*}"),
    )
    timeit(
        "explicit props RETURN n.email, n.age, n.dept",
        lambda: g.cypher("MATCH (n:Person) RETURN n.email, n.age, n.dept"),
    )

    print("\n== count(DISTINCT) over nodes ==")
    timeit(
        "count(DISTINCT n) across all",
        lambda: g.cypher("MATCH (n:Person) RETURN count(DISTINCT n) AS c"),
    )
    timeit(
        "count(DISTINCT n) grouped by dept",
        lambda: g.cypher("MATCH (n:Person) RETURN n.dept, count(DISTINCT n) AS c"),
    )

    print("\n== substring() in RETURN ==")
    timeit(
        "substring(title, 0, 6)",
        lambda: g.cypher("MATCH (n:Person) RETURN substring(n.title, 0, 6) AS s"),
    )
    timeit(
        "substring(email, 5)",
        lambda: g.cypher("MATCH (n:Person) RETURN substring(n.email, 5) AS s"),
    )

    print("\n== function-heavy scan (toLower/toUpper) ==")
    timeit(
        "toLower(email) + toUpper(title)",
        lambda: g.cypher("MATCH (n:Person) RETURN toLower(n.email) AS e, toUpper(n.title) AS t"),
    )
    timeit(
        "mixed: toLower, substring, size",
        lambda: g.cypher("MATCH (n:Person) RETURN toLower(substring(n.email, 0, 10)) AS s, size(n.title) AS sz"),
    )


if __name__ == "__main__":
    main()
