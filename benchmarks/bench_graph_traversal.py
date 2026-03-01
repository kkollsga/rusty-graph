"""
Benchmark: SQLite recursive CTEs vs KGLite (graph DB) — traversal performance.

Tests BosonCollider's claim that B-tree joins outperform index-free adjacency
for graph traversals. Both engines are embedded/in-process (no network overhead).

Dataset: ~30k nodes, ~220k edges — heterogeneous academic knowledge graph
  Nodes: Person (10k), Paper (20k), Topic (500), Institution (200)
  Edges: AUTHORED, CITES, COVERS, AFFILIATED, COLLABORATES

Queries: hop depths 1–8, variable-depth, shortest path.

Usage:
    pip install kglite
    python bench_graph_traversal.py
"""

import random
import sqlite3
import statistics
import time

import kglite

# ── Config ──────────────────────────────────────────────────────────────────

SEED = 42
N_PERSONS = 10_000
N_PAPERS = 20_000
N_TOPICS = 500
N_INSTITUTIONS = 200
RUNS = 10  # median of N runs per query

# ── Data generation ─────────────────────────────────────────────────────────

def generate_data():
    """Generate deterministic graph data. Returns dict of edge lists."""
    rng = random.Random(SEED)

    # Edge lists
    authored = []  # (person_id, paper_id)
    cites = []  # (paper_id, cited_paper_id)
    covers = []  # (paper_id, topic_id)
    affiliated = []  # (person_id, institution_id)
    collaborates = []  # (person_id, person_id)

    # Each person authors 2-10 papers
    for pid in range(N_PERSONS):
        n_papers = rng.randint(2, 10)
        for _ in range(n_papers):
            authored.append((pid, rng.randint(0, N_PAPERS - 1)))

    # Each paper cites 1-8 other papers
    for paper_id in range(N_PAPERS):
        n_cited = rng.randint(1, 8)
        for _ in range(n_cited):
            cited = rng.randint(0, N_PAPERS - 1)
            if cited != paper_id:
                cites.append((paper_id, cited))

    # Each paper covers 1-4 topics
    for paper_id in range(N_PAPERS):
        n_topics = rng.randint(1, 4)
        for _ in range(n_topics):
            covers.append((paper_id, rng.randint(0, N_TOPICS - 1)))

    # Each person affiliated with 1 institution
    for pid in range(N_PERSONS):
        affiliated.append((pid, rng.randint(0, N_INSTITUTIONS - 1)))

    # ~3 collaborators per person (undirected, stored both ways)
    for pid in range(N_PERSONS):
        n_collabs = rng.randint(1, 5)
        for _ in range(n_collabs):
            other = rng.randint(0, N_PERSONS - 1)
            if other != pid:
                collaborates.append((pid, other))

    # ── Guaranteed graph structure ──────────────────────────────────────────
    # Random sparse graphs have almost no triangles or shared citations.
    # Add deterministic structure so pattern-matching benchmarks find results.

    # Citation triangles: paper i → i+1 → i+2 → i (100 triangles)
    for i in range(0, 300, 3):
        cites.append((i, i + 1))
        cites.append((i + 1, i + 2))
        cites.append((i + 2, i))

    # Collaboration triangles: person i — i+1 — i+2 — i (100 triangles)
    for i in range(0, 300, 3):
        collaborates.append((i, i + 1))
        collaborates.append((i + 1, i + 2))
        collaborates.append((i + 2, i))

    # Hub citations: papers 0–99 all cite papers 5000, 5001 (shared targets)
    for pid in range(100):
        cites.append((pid, 5000))
        cites.append((pid, 5001))

    return {
        "authored": authored,
        "cites": cites,
        "covers": covers,
        "affiliated": affiliated,
        "collaborates": collaborates,
    }


# ── SQLite setup ────────────────────────────────────────────────────────────

def setup_sqlite(data):
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-64000")  # 64MB cache

    db.executescript("""
        CREATE TABLE person (id INTEGER PRIMARY KEY);
        CREATE TABLE paper (id INTEGER PRIMARY KEY);
        CREATE TABLE topic (id INTEGER PRIMARY KEY);
        CREATE TABLE institution (id INTEGER PRIMARY KEY);

        CREATE TABLE authored (person_id INT, paper_id INT);
        CREATE TABLE cites (paper_id INT, cited_id INT);
        CREATE TABLE covers (paper_id INT, topic_id INT);
        CREATE TABLE affiliated (person_id INT, inst_id INT);
        CREATE TABLE collaborates (pid1 INT, pid2 INT);
    """)

    db.executemany("INSERT INTO person VALUES (?)", [(i,) for i in range(N_PERSONS)])
    db.executemany("INSERT INTO paper VALUES (?)", [(i,) for i in range(N_PAPERS)])
    db.executemany("INSERT INTO topic VALUES (?)", [(i,) for i in range(N_TOPICS)])
    db.executemany("INSERT INTO institution VALUES (?)", [(i,) for i in range(N_INSTITUTIONS)])
    db.executemany("INSERT INTO authored VALUES (?,?)", data["authored"])
    db.executemany("INSERT INTO cites VALUES (?,?)", data["cites"])
    db.executemany("INSERT INTO covers VALUES (?,?)", data["covers"])
    db.executemany("INSERT INTO affiliated VALUES (?,?)", data["affiliated"])
    db.executemany("INSERT INTO collaborates VALUES (?,?)", data["collaborates"])

    db.executescript("""
        CREATE INDEX idx_auth_p ON authored(person_id);
        CREATE INDEX idx_auth_pp ON authored(paper_id);
        CREATE INDEX idx_cite_p ON cites(paper_id);
        CREATE INDEX idx_cite_c ON cites(cited_id);
        CREATE INDEX idx_cov_p ON covers(paper_id);
        CREATE INDEX idx_cov_t ON covers(topic_id);
        CREATE INDEX idx_aff_p ON affiliated(person_id);
        CREATE INDEX idx_aff_i ON affiliated(inst_id);
        CREATE INDEX idx_col_1 ON collaborates(pid1);
        CREATE INDEX idx_col_2 ON collaborates(pid2);
    """)
    db.commit()
    return db


# ── KGLite setup ────────────────────────────────────────────────────────────

def setup_kglite(data):
    import pandas as pd

    g = kglite.KnowledgeGraph()

    g.add_nodes(pd.DataFrame({"id": range(N_PERSONS)}), "Person", "id", "id")
    g.add_nodes(pd.DataFrame({"id": range(N_PAPERS)}), "Paper", "id", "id")
    g.add_nodes(pd.DataFrame({"id": range(N_TOPICS)}), "Topic", "id", "id")
    g.add_nodes(pd.DataFrame({"id": range(N_INSTITUTIONS)}), "Institution", "id", "id")

    g.add_connections(
        pd.DataFrame(data["authored"], columns=["person_id", "paper_id"]),
        "AUTHORED", "Person", "person_id", "Paper", "paper_id",
    )
    g.add_connections(
        pd.DataFrame(data["cites"], columns=["paper_id", "cited_id"]),
        "CITES", "Paper", "paper_id", "Paper", "cited_id",
    )
    g.add_connections(
        pd.DataFrame(data["covers"], columns=["paper_id", "topic_id"]),
        "COVERS", "Paper", "paper_id", "Topic", "topic_id",
    )
    g.add_connections(
        pd.DataFrame(data["affiliated"], columns=["person_id", "inst_id"]),
        "AFFILIATED", "Person", "person_id", "Institution", "inst_id",
    )
    g.add_connections(
        pd.DataFrame(data["collaborates"], columns=["pid1", "pid2"]),
        "COLLABORATES", "Person", "pid1", "Person", "pid2",
    )
    return g


# ── Benchmark harness ──────────────────────────────────────────────────────

def bench(fn, runs=RUNS):
    """Return median execution time in ms over `runs` iterations."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    if result is None:
        count = 0
    elif isinstance(result, (int, float)):
        count = int(result)
    elif isinstance(result, tuple):
        count = result[0] if result[0] is not None else 0
    else:
        count = len(list(result))
    return statistics.median(times), count


# ── Queries ─────────────────────────────────────────────────────────────────
#
# For each hop depth we follow citation chains (Paper→Paper via CITES).
# This is the purest traversal benchmark: same entity type, same edge type,
# varying only the depth — isolating the join-vs-pointer cost.

START_PAPER = 42  # deterministic start node
START_PERSON = 7


def sqlite_citation_hops(db, depth):
    """N-hop citation traversal using recursive CTE."""
    return db.execute(f"""
        WITH RECURSIVE chain(paper_id, d) AS (
            SELECT cited_id, 1 FROM cites WHERE paper_id = {START_PAPER}
            UNION
            SELECT c.cited_id, chain.d + 1
            FROM cites c JOIN chain ON c.paper_id = chain.paper_id
            WHERE chain.d < {depth}
        )
        SELECT DISTINCT paper_id FROM chain
    """).fetchall()


def kglite_citation_hops(g, depth):
    """N-hop citation traversal — full materialization."""
    return g.cypher(f"""
        MATCH (start:Paper {{id: {START_PAPER}}})-[:CITES*1..{depth}]->(p:Paper)
        RETURN DISTINCT p.id
    """).to_list()


def kglite_citation_hops_raw(g, depth):
    """N-hop citation traversal — raw Cypher (no Python materialization)."""
    result = g.cypher(f"""
        MATCH (start:Paper {{id: {START_PAPER}}})-[:CITES*1..{depth}]->(p:Paper)
        RETURN DISTINCT p.id
    """)
    return len(result)


def sqlite_collab_hops(db, depth):
    """N-hop collaboration traversal using recursive CTE."""
    return db.execute(f"""
        WITH RECURSIVE chain(person_id, d) AS (
            SELECT pid2, 1 FROM collaborates WHERE pid1 = {START_PERSON}
            UNION
            SELECT c.pid2, chain.d + 1
            FROM collaborates c JOIN chain ON c.pid1 = chain.person_id
            WHERE chain.d < {depth}
        )
        SELECT DISTINCT person_id FROM chain
    """).fetchall()


def kglite_collab_hops(g, depth):
    """N-hop collaboration traversal — full materialization."""
    return g.cypher(f"""
        MATCH (start:Person {{id: {START_PERSON}}})-[:COLLABORATES*1..{depth}]->(p:Person)
        RETURN DISTINCT p.id
    """).to_list()


def kglite_collab_hops_raw(g, depth):
    """N-hop collaboration traversal — raw Cypher (no Python materialization)."""
    result = g.cypher(f"""
        MATCH (start:Person {{id: {START_PERSON}}})-[:COLLABORATES*1..{depth}]->(p:Person)
        RETURN DISTINCT p.id
    """)
    return len(result)


def sqlite_hetero_traversal(db):
    """4-hop heterogeneous: Person→Paper→Topic→Paper→Person (via authored/covers)."""
    return db.execute(f"""
        SELECT DISTINCT p2.person_id
        FROM authored a1
        JOIN covers c1 ON a1.paper_id = c1.paper_id
        JOIN covers c2 ON c1.topic_id = c2.topic_id AND c2.paper_id != c1.paper_id
        JOIN authored p2 ON c2.paper_id = p2.paper_id
        WHERE a1.person_id = {START_PERSON}
    """).fetchall()


def kglite_hetero_traversal(g):
    """4-hop heterogeneous: Person→Paper→Topic→Paper→Person — full materialization."""
    return g.cypher(f"""
        MATCH (start:Person {{id: {START_PERSON}}})-[:AUTHORED]->(paper:Paper)
              -[:COVERS]->(t:Topic)<-[:COVERS]-(other_paper:Paper)
              <-[:AUTHORED]-(other:Person)
        WHERE other.id <> {START_PERSON}
        RETURN DISTINCT other.id
    """).to_list()


def kglite_hetero_traversal_raw(g):
    """4-hop heterogeneous — raw Cypher (no Python materialization)."""
    result = g.cypher(f"""
        MATCH (start:Person {{id: {START_PERSON}}})-[:AUTHORED]->(paper:Paper)
              -[:COVERS]->(t:Topic)<-[:COVERS]-(other_paper:Paper)
              <-[:AUTHORED]-(other:Person)
        WHERE other.id <> {START_PERSON}
        RETURN DISTINCT other.id
    """)
    return len(result)


def sqlite_shortest_path(db, source, target):
    """BFS shortest path between two persons via collaborates."""
    return db.execute(f"""
        WITH RECURSIVE bfs(person_id, depth) AS (
            SELECT {source}, 0
            UNION
            SELECT c.pid2, bfs.depth + 1
            FROM collaborates c JOIN bfs ON c.pid1 = bfs.person_id
            WHERE bfs.depth < 8
        )
        SELECT MIN(depth) FROM bfs WHERE person_id = {target}
    """).fetchone()


def kglite_shortest_path(g, source, target):
    """Shortest path between two persons via collaborates."""
    result = g.cypher(f"""
        MATCH p = shortestPath(
            (a:Person {{id: {source}}})-[:COLLABORATES*..8]-(b:Person {{id: {target}}})
        )
        RETURN length(p) AS dist
    """).to_list()
    return result[0]["dist"] if result else None


# ── Triangle detection ─────────────────────────────────────────────────────
# Find citation triangles: A cites B, B cites C, C cites A.
# Classic graph problem — requires 3-way self-join in SQL.

TRIANGLE_PAPERS = list(range(300))  # papers 0-299 have guaranteed triangles


def sqlite_triangles(db):
    """Count citation triangles among sampled papers."""
    placeholders = ",".join(str(p) for p in TRIANGLE_PAPERS)
    return db.execute(f"""
        SELECT count(*) FROM (
            SELECT DISTINCT c1.paper_id AS a, c2.paper_id AS b, c3.paper_id AS c
            FROM cites c1
            JOIN cites c2 ON c1.cited_id = c2.paper_id
            JOIN cites c3 ON c2.cited_id = c3.paper_id AND c3.cited_id = c1.paper_id
            WHERE c1.paper_id IN ({placeholders})
        )
    """).fetchone()


def kglite_triangles_raw(g):
    """Count citation triangles among sampled papers."""
    ids = TRIANGLE_PAPERS
    result = g.cypher(f"""
        MATCH (a:Paper)-[:CITES]->(b:Paper)-[:CITES]->(c:Paper)-[:CITES]->(a)
        WHERE a.id IN {ids} AND a.id < b.id AND b.id < c.id
        RETURN count(*) AS triangles
    """).to_list()
    return result[0]["triangles"] if result else 0


# ── Reachability check (early termination) ────────────────────────────────
# "Can person X reach person Y within N hops?" — graph DB stops on first hit.

def sqlite_reachable(db, source, target, max_hops):
    """Reachability check — SQLite must materialize entire BFS frontier."""
    return db.execute(f"""
        WITH RECURSIVE bfs(person_id, d) AS (
            SELECT {source}, 0
            UNION
            SELECT c.pid2, bfs.d + 1
            FROM collaborates c JOIN bfs ON c.pid1 = bfs.person_id
            WHERE bfs.d < {max_hops}
        )
        SELECT 1 FROM bfs WHERE person_id = {target} LIMIT 1
    """).fetchone()


def kglite_reachable_raw(g, source, target, max_hops):
    """Reachability check — Cypher with LIMIT 1 for early termination."""
    result = g.cypher(f"""
        MATCH (a:Person {{id: {source}}})-[:COLLABORATES*1..{max_hops}]-(b:Person {{id: {target}}})
        RETURN 1 AS found
        LIMIT 1
    """)
    return len(result)


# ── Neighborhood aggregation ──────────────────────────────────────────────
# Count distinct topics covered by a person's N-hop collaboration network.
# Multi-type: Person → COLLABORATES → Person → AUTHORED → Paper → COVERS → Topic

def sqlite_neighbor_topics(db, max_hops):
    """Topics covered by papers authored by N-hop collaborators."""
    return db.execute(f"""
        WITH RECURSIVE collab_net(person_id, d) AS (
            SELECT pid2, 1 FROM collaborates WHERE pid1 = {START_PERSON}
            UNION
            SELECT c.pid2, collab_net.d + 1
            FROM collaborates c JOIN collab_net ON c.pid1 = collab_net.person_id
            WHERE collab_net.d < {max_hops}
        )
        SELECT count(DISTINCT cv.topic_id)
        FROM collab_net cn
        JOIN authored a ON cn.person_id = a.person_id
        JOIN covers cv ON a.paper_id = cv.paper_id
    """).fetchone()


def kglite_neighbor_topics_raw(g, max_hops):
    """Topics covered by papers authored by N-hop collaborators."""
    result = g.cypher(f"""
        MATCH (start:Person {{id: {START_PERSON}}})-[:COLLABORATES*1..{max_hops}]->(collab:Person)
              -[:AUTHORED]->(paper:Paper)-[:COVERS]->(t:Topic)
        RETURN count(DISTINCT t.id) AS topic_count
    """).to_list()
    return result[0]["topic_count"] if result else 0


# ── Fan-out / influence spread ─────────────────────────────────────────────
# How many unique people can each of N start nodes reach in K hops?
# Batch query that exercises the BFS engine repeatedly.

def sqlite_fan_out(db, persons, max_hops):
    """Count reachable persons within K hops for N start nodes."""
    total = 0
    for pid in persons:
        row = db.execute(f"""
            WITH RECURSIVE chain(person_id, d) AS (
                SELECT pid2, 1 FROM collaborates WHERE pid1 = {pid}
                UNION
                SELECT c.pid2, chain.d + 1
                FROM collaborates c JOIN chain ON c.pid1 = chain.person_id
                WHERE chain.d < {max_hops}
            )
            SELECT count(DISTINCT person_id) FROM chain
        """).fetchone()
        total += row[0]
    return total


def kglite_fan_out_raw(g, persons, max_hops):
    """Count reachable persons within K hops for N start nodes."""
    result = g.cypher(f"""
        UNWIND {persons} AS pid
        MATCH (start:Person {{id: pid}})-[:COLLABORATES*1..{max_hops}]->(p:Person)
        RETURN pid, count(DISTINCT p.id) AS reach
    """).to_list()
    return sum(row["reach"] for row in result)


# ── Multi-start deep traversal ────────────────────────────────────────────
# Multiple start nodes each doing deep BFS — stresses both engines.

def sqlite_multi_start_deep(db, starts, depth):
    """Deep citation traversal from multiple start nodes."""
    total = 0
    for sid in starts:
        rows = db.execute(f"""
            WITH RECURSIVE chain(paper_id, d) AS (
                SELECT cited_id, 1 FROM cites WHERE paper_id = {sid}
                UNION
                SELECT c.cited_id, chain.d + 1
                FROM cites c JOIN chain ON c.paper_id = chain.paper_id
                WHERE chain.d < {depth}
            )
            SELECT count(DISTINCT paper_id) FROM chain
        """).fetchone()
        total += rows[0]
    return total


def kglite_multi_start_deep_raw(g, starts, depth):
    """Deep citation traversal from multiple start nodes."""
    result = g.cypher(f"""
        UNWIND {starts} AS sid
        MATCH (start:Paper {{id: sid}})-[:CITES*1..{depth}]->(p:Paper)
        RETURN sid, count(DISTINCT p.id) AS reach
    """).to_list()
    return sum(row["reach"] for row in result)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import pandas as pd

    print("Generating dataset...")
    data = generate_data()
    edge_count = sum(len(v) for v in data.values())
    node_count = N_PERSONS + N_PAPERS + N_TOPICS + N_INSTITUTIONS
    print(f"  {node_count:,} nodes, {edge_count:,} edges\n")

    print("Loading into SQLite...", end=" ", flush=True)
    t0 = time.perf_counter()
    db = setup_sqlite(data)
    print(f"{(time.perf_counter() - t0)*1000:.0f} ms")

    print("Loading into KGLite...", end=" ", flush=True)
    t0 = time.perf_counter()
    g = setup_kglite(data)
    print(f"{(time.perf_counter() - t0)*1000:.0f} ms\n")

    rows = []

    def run_pair(label, sqlite_fn, kglite_fn, kglite_raw_fn=None, category=""):
        sql_ms, sql_n = bench(sqlite_fn)
        kg_ms, kg_n = bench(kglite_fn)
        if kglite_raw_fn:
            kg_raw_ms, _ = bench(kglite_raw_fn)
        else:
            kg_raw_ms = kg_ms
        ratio = sql_ms / kg_raw_ms if kg_raw_ms > 0 else float("inf")
        winner = "KGLite" if ratio > 1 else "SQLite"
        rows.append({
            "Category": category,
            "Query": label,
            "SQLite (ms)": round(sql_ms, 4),
            "KGLite (ms)": round(kg_raw_ms, 4),
            "Winner": winner,
            "Ratio": f"{ratio:.1f}x",
            "Rows": max(sql_n, kg_n),
        })

    # ── Citation chain (shallow → crossover → deep) ──
    for depth in [5, 8, 15, 20]:
        run_pair(
            f"{depth}-hop citations",
            lambda d=depth: sqlite_citation_hops(db, d),
            lambda d=depth: kglite_citation_hops(g, d),
            lambda d=depth: kglite_citation_hops_raw(g, d),
            category="Citation chain",
        )

    # ── Collaboration chain ──
    for depth in [6, 10]:
        run_pair(
            f"{depth}-hop collaborators",
            lambda d=depth: sqlite_collab_hops(db, d),
            lambda d=depth: kglite_collab_hops(g, d),
            lambda d=depth: kglite_collab_hops_raw(g, d),
            category="Collab chain",
        )

    # ── Heterogeneous traversal ──
    run_pair(
        "Person->Paper->Topic->Paper->Person",
        lambda: sqlite_hetero_traversal(db),
        lambda: kglite_hetero_traversal(g),
        lambda: kglite_hetero_traversal_raw(g),
        category="Heterogeneous",
    )

    # ── Shortest path ──
    rng = random.Random(SEED)
    for _ in range(2):
        src = rng.randint(0, N_PERSONS - 1)
        tgt = rng.randint(0, N_PERSONS - 1)
        run_pair(
            f"Person {src} -> {tgt}",
            lambda s=src, t=tgt: sqlite_shortest_path(db, s, t),
            lambda s=src, t=tgt: kglite_shortest_path(g, s, t),
            category="Shortest path",
        )

    # ── Triangle detection ──
    run_pair(
        f"A->B->C->A in {len(TRIANGLE_PAPERS)} papers",
        lambda: sqlite_triangles(db),
        lambda: kglite_triangles_raw(g),
        category="Triangles",
    )

    # ── Reachability (early termination) ──
    rng2 = random.Random(SEED + 1)
    for max_hops in [8, 12]:
        src = rng2.randint(0, N_PERSONS - 1)
        tgt = rng2.randint(0, N_PERSONS - 1)
        run_pair(
            f"Person {src} <-> {tgt} ({max_hops} hops)",
            lambda s=src, t=tgt, h=max_hops: sqlite_reachable(db, s, t, h),
            lambda s=src, t=tgt, h=max_hops: kglite_reachable_raw(g, s, t, h),
            category="Reachability",
        )

    # ── Neighborhood aggregation (mixed types) ──
    run_pair(
        "Topics via 2-hop collabs",
        lambda: sqlite_neighbor_topics(db, 2),
        lambda: kglite_neighbor_topics_raw(g, 2),
        category="Neighborhood agg.",
    )

    # ── Fan-out (batch BFS) ──
    fan_out_50 = list(range(50))
    run_pair(
        "50 x 3-hop reach",
        lambda: sqlite_fan_out(db, fan_out_50, 3),
        lambda: kglite_fan_out_raw(g, fan_out_50, 3),
        category="Fan-out",
    )

    # ── Multi-start deep traversal ──
    deep_starts = list(range(0, 100, 10))  # 10 start nodes
    run_pair(
        "10 x 12-hop citations",
        lambda: sqlite_multi_start_deep(db, deep_starts, 12),
        lambda: kglite_multi_start_deep_raw(g, deep_starts, 12),
        category="Multi-start deep",
    )

    # Build DataFrame, sort by SQLite time, print as markdown
    df = pd.DataFrame(rows)
    df = df.sort_values("SQLite (ms)").reset_index(drop=True)
    print(df.to_markdown(index=False, floatfmt=".4f"))

    # Summary
    kg_wins = (df["Winner"] == "KGLite").sum()
    sql_wins = (df["Winner"] == "SQLite").sum()
    print(f"\nKGLite wins: {kg_wins}/{len(df)}  |  SQLite wins: {sql_wins}/{len(df)}")
    print(f"Each query: median of {RUNS} runs. Dataset: {node_count:,} nodes, {edge_count:,} edges.")


if __name__ == "__main__":
    main()
