"""NetworkX comparison benchmarks: accuracy and performance.

Validates kglite algorithms against NetworkX ground truth and measures
performance across graph scales (100 to 50K nodes).

Run: pytest tests/benchmarks/test_nx_comparison.py -v -s
Run accuracy only: pytest tests/benchmarks/test_nx_comparison.py -v -s -k "Accuracy"
Run performance only: pytest tests/benchmarks/test_nx_comparison.py -v -s -k "Performance"
"""

from collections import defaultdict
import math
import pickle
import random
import tempfile
import time

import pandas as pd
import pytest

nx = pytest.importorskip("networkx", reason="networkx required for comparison benchmarks")

from kglite import KnowledgeGraph

pytestmark = pytest.mark.benchmark

# Optional deps for memory / disk footprint measurements.
try:
    import psutil as _psutil
except ImportError:
    _psutil = None


def _rss_bytes():
    """Current RSS of the test process in bytes (None if psutil missing)."""
    return _psutil.Process().memory_info().rss if _psutil is not None else None


def _dir_size(path):
    """Total bytes occupied by a directory tree. Returns 0 if missing."""
    import os

    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _fmt_bytes(n):
    if n is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


# ============================================================================
# Helper Functions
# ============================================================================


def _build_paired_graphs(n_nodes, edge_factor=3, seed=42):
    """Build identical graphs in kglite and NetworkX.

    Returns (kg, nx_graph, node_names).
    Nodes have properties: name (str), value (float), group (int 1-5).
    """
    rng = random.Random(seed)

    # Generate node data
    node_names = [f"N{i}" for i in range(n_nodes)]
    node_values = [rng.uniform(0, 100) for _ in range(n_nodes)]
    node_groups = [rng.randint(1, 5) for _ in range(n_nodes)]

    # Generate edges (random, avoiding self-loops and duplicates)
    n_edges = edge_factor * n_nodes
    edge_set = set()
    while len(edge_set) < n_edges:
        s = rng.randint(0, n_nodes - 1)
        t = rng.randint(0, n_nodes - 1)
        if s != t:
            edge_set.add((s, t))
    edges = list(edge_set)

    # Build NetworkX graph
    nx_g = nx.DiGraph()
    for i in range(n_nodes):
        nx_g.add_node(node_names[i], value=node_values[i], group=node_groups[i])
    for s, t in edges:
        nx_g.add_edge(node_names[s], node_names[t])

    # Build kglite
    kg = KnowledgeGraph()
    df_nodes = pd.DataFrame(
        {
            "id": list(range(n_nodes)),
            "name": node_names,
            "value": node_values,
            "group": node_groups,
        }
    )
    kg.add_nodes(df_nodes, "Node", "id", "name")

    df_edges = pd.DataFrame(
        {
            "from_id": [s for s, t in edges],
            "to_id": [t for s, t in edges],
        }
    )
    kg.add_connections(df_edges, "EDGE", "Node", "from_id", "Node", "to_id")

    return kg, nx_g, node_names


def _build_scale_free(n, m=3, seed=42):
    """Build Barabasi-Albert scale-free graph in both systems."""
    nx_g = nx.barabasi_albert_graph(n, m, seed=seed)
    nx_dg = nx.DiGraph()
    node_names = [f"N{i}" for i in range(n)]
    for i in range(n):
        nx_dg.add_node(node_names[i], value=float(i), group=(i % 5) + 1)
    for u, v in nx_g.edges():
        nx_dg.add_edge(node_names[u], node_names[v])

    kg = KnowledgeGraph()
    df_nodes = pd.DataFrame(
        {
            "id": list(range(n)),
            "name": node_names,
            "value": [float(i) for i in range(n)],
            "group": [(i % 5) + 1 for i in range(n)],
        }
    )
    kg.add_nodes(df_nodes, "Node", "id", "name")

    edges = list(nx_dg.edges())
    if edges:
        name_to_id = {name: i for i, name in enumerate(node_names)}
        df_edges = pd.DataFrame(
            {
                "from_id": [name_to_id[u] for u, v in edges],
                "to_id": [name_to_id[v] for u, v in edges],
            }
        )
        kg.add_connections(df_edges, "EDGE", "Node", "from_id", "Node", "to_id")

    return kg, nx_dg, node_names


def _build_chain(n, seed=42):
    """Build linear chain: N0 -> N1 -> N2 -> ... -> N(n-1)."""
    node_names = [f"N{i}" for i in range(n)]

    nx_g = nx.DiGraph()
    for i in range(n):
        nx_g.add_node(node_names[i], value=float(i), group=(i % 5) + 1)
    for i in range(n - 1):
        nx_g.add_edge(node_names[i], node_names[i + 1])

    kg = KnowledgeGraph()
    df_nodes = pd.DataFrame(
        {
            "id": list(range(n)),
            "name": node_names,
            "value": [float(i) for i in range(n)],
            "group": [(i % 5) + 1 for i in range(n)],
        }
    )
    kg.add_nodes(df_nodes, "Node", "id", "name")

    if n > 1:
        df_edges = pd.DataFrame(
            {
                "from_id": list(range(n - 1)),
                "to_id": list(range(1, n)),
            }
        )
        kg.add_connections(df_edges, "EDGE", "Node", "from_id", "Node", "to_id")

    return kg, nx_g, node_names


def _build_star(n, seed=42):
    """Build star: N0 (hub) -> N1, N2, ..., N(n-1)."""
    node_names = [f"N{i}" for i in range(n)]

    nx_g = nx.DiGraph()
    for i in range(n):
        nx_g.add_node(node_names[i], value=float(i), group=(i % 5) + 1)
    for i in range(1, n):
        nx_g.add_edge(node_names[0], node_names[i])

    kg = KnowledgeGraph()
    df_nodes = pd.DataFrame(
        {
            "id": list(range(n)),
            "name": node_names,
            "value": [float(i) for i in range(n)],
            "group": [(i % 5) + 1 for i in range(n)],
        }
    )
    kg.add_nodes(df_nodes, "Node", "id", "name")

    if n > 1:
        df_edges = pd.DataFrame(
            {
                "from_id": [0] * (n - 1),
                "to_id": list(range(1, n)),
            }
        )
        kg.add_connections(df_edges, "EDGE", "Node", "from_id", "Node", "to_id")

    return kg, nx_g, node_names


def _build_clusters(n_clusters, cluster_size, inter_edges=5, seed=42):
    """Build disconnected clusters with sparse inter-cluster bridges.

    Each cluster is a dense random graph. A few inter-cluster edges connect them.
    """
    rng = random.Random(seed)
    n = n_clusters * cluster_size
    node_names = [f"N{i}" for i in range(n)]

    nx_g = nx.DiGraph()
    for i in range(n):
        nx_g.add_node(node_names[i], value=float(i), group=(i // cluster_size))

    # Dense intra-cluster edges
    edges = []
    for c in range(n_clusters):
        base = c * cluster_size
        for i in range(cluster_size):
            for j in range(i + 1, cluster_size):
                if rng.random() < 0.4:
                    edges.append((base + i, base + j))
                if rng.random() < 0.4:
                    edges.append((base + j, base + i))

    # Sparse inter-cluster edges
    for _ in range(inter_edges):
        c1, c2 = rng.sample(range(n_clusters), 2)
        s = c1 * cluster_size + rng.randint(0, cluster_size - 1)
        t = c2 * cluster_size + rng.randint(0, cluster_size - 1)
        edges.append((s, t))

    for s, t in edges:
        nx_g.add_edge(node_names[s], node_names[t])

    kg = KnowledgeGraph()
    df_nodes = pd.DataFrame(
        {
            "id": list(range(n)),
            "name": node_names,
            "value": [float(i) for i in range(n)],
            "group": [i // cluster_size for i in range(n)],
        }
    )
    kg.add_nodes(df_nodes, "Node", "id", "name")

    if edges:
        df_edges = pd.DataFrame(
            {
                "from_id": [s for s, t in edges],
                "to_id": [t for s, t in edges],
            }
        )
        kg.add_connections(df_edges, "EDGE", "Node", "from_id", "Node", "to_id")

    return kg, nx_g, node_names


def _pearson_correlation(xs, ys):
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 1.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    num = sum(a * b for a, b in zip(dx, dy))
    den_x = math.sqrt(sum(a * a for a in dx))
    den_y = math.sqrt(sum(b * b for b in dy))
    if den_x < 1e-15 or den_y < 1e-15:
        return 1.0 if den_x < 1e-15 and den_y < 1e-15 else 0.0
    return num / (den_x * den_y)


def _extract_rg_scores(kg, algo_name):
    """Run algorithm on kglite and return {name: score} dict.

    algo_name: 'pagerank', 'betweenness_centrality', 'closeness_centrality',
               'degree_centrality'
    """
    method = getattr(kg, algo_name)
    results = method()
    return {r["title"]: r["score"] for r in results}


def _compare_scores(rg_scores, nx_scores, node_names, algo_name, min_corr=0.99, max_abs_err=0.01):
    """Compare scores between kglite and NetworkX.

    Returns (correlation, max_error, passed).
    """
    rg_vals = []
    nx_vals = []
    for name in node_names:
        rg_val = rg_scores.get(name, 0.0)
        nx_val = nx_scores.get(name, 0.0)
        rg_vals.append(rg_val)
        nx_vals.append(nx_val)

    corr = _pearson_correlation(rg_vals, nx_vals)
    max_err = max(abs(r - n) for r, n in zip(rg_vals, nx_vals))

    passed = corr >= min_corr and max_err <= max_abs_err
    print(
        f"  {algo_name}: corr={corr:.6f} (min={min_corr}), "
        f"max_err={max_err:.6f} (max={max_abs_err}), "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return corr, max_err, passed


# ============================================================================
# Accuracy Tests
# ============================================================================


class TestPageRankAccuracy:
    """Validate PageRank scores against NetworkX."""

    def test_random_100(self):
        kg, nx_g, names = _build_paired_graphs(100, edge_factor=3)
        rg = _extract_rg_scores(kg, "pagerank")
        nx_scores = nx.pagerank(nx_g, alpha=0.85)
        corr, max_err, _ = _compare_scores(rg, nx_scores, names, "pagerank_random_100")
        assert corr > 0.99, f"Correlation too low: {corr}"
        assert max_err < 0.01, f"Max error too high: {max_err}"

    def test_random_1k(self):
        kg, nx_g, names = _build_paired_graphs(1000, edge_factor=3)
        rg = _extract_rg_scores(kg, "pagerank")
        nx_scores = nx.pagerank(nx_g, alpha=0.85)
        corr, max_err, _ = _compare_scores(rg, nx_scores, names, "pagerank_random_1k")
        assert corr > 0.99, f"Correlation too low: {corr}"
        assert max_err < 0.01, f"Max error too high: {max_err}"

    def test_scale_free_100(self):
        kg, nx_g, names = _build_scale_free(100)
        rg = _extract_rg_scores(kg, "pagerank")
        nx_scores = nx.pagerank(nx_g, alpha=0.85)
        corr, max_err, _ = _compare_scores(rg, nx_scores, names, "pagerank_scalefree_100")
        assert corr > 0.99, f"Correlation too low: {corr}"
        assert max_err < 0.01, f"Max error too high: {max_err}"

    def test_scale_free_1k(self):
        kg, nx_g, names = _build_scale_free(1000)
        rg = _extract_rg_scores(kg, "pagerank")
        nx_scores = nx.pagerank(nx_g, alpha=0.85)
        corr, max_err, _ = _compare_scores(rg, nx_scores, names, "pagerank_scalefree_1k")
        assert corr > 0.99, f"Correlation too low: {corr}"
        assert max_err < 0.01, f"Max error too high: {max_err}"

    def test_star_50(self):
        kg, nx_g, names = _build_star(50)
        rg = _extract_rg_scores(kg, "pagerank")
        nx_scores = nx.pagerank(nx_g, alpha=0.85)
        corr, max_err, _ = _compare_scores(rg, nx_scores, names, "pagerank_star_50")
        assert corr > 0.99, f"Correlation too low: {corr}"
        assert max_err < 0.01, f"Max error too high: {max_err}"

    def test_chain_100(self):
        kg, nx_g, names = _build_chain(100)
        rg = _extract_rg_scores(kg, "pagerank")
        nx_scores = nx.pagerank(nx_g, alpha=0.85)
        corr, max_err, _ = _compare_scores(rg, nx_scores, names, "pagerank_chain_100")
        assert corr > 0.99, f"Correlation too low: {corr}"
        assert max_err < 0.01, f"Max error too high: {max_err}"


class TestShortestPathAccuracy:
    """Validate shortest path results against NetworkX."""

    def _compare_paths(self, kg, nx_g, node_names, n_pairs=20, label=""):
        """Compare shortest path lengths for random pairs."""
        rng = random.Random(42)
        matches = 0
        tested = 0

        for _ in range(n_pairs):
            src_name = rng.choice(node_names)
            tgt_name = rng.choice(node_names)
            if src_name == tgt_name:
                continue
            tested += 1

            # NetworkX
            try:
                nx_len = nx.shortest_path_length(nx_g, src_name, tgt_name)
            except nx.NetworkXNoPath:
                nx_len = None

            # kglite via Cypher
            try:
                result = kg.cypher(
                    f"MATCH p = shortestPath((a:Node {{name: '{src_name}'}})"
                    f"-[*..200]->"
                    f"(b:Node {{name: '{tgt_name}'}})) "
                    f"RETURN length(p) AS len"
                )
                rg_len = result[0]["len"] if result else None
            except Exception:
                rg_len = None

            if nx_len == rg_len:
                matches += 1
            elif nx_len is not None and rg_len is not None:
                print(f"    MISMATCH {src_name}->{tgt_name}: rg={rg_len}, nx={nx_len}")

        print(f"  shortest_path {label}: {matches}/{tested} exact matches")
        return matches, tested

    def test_random_100(self):
        kg, nx_g, names = _build_paired_graphs(100, edge_factor=3)
        matches, tested = self._compare_paths(kg, nx_g, names, 20, "random_100")
        assert matches == tested, f"Only {matches}/{tested} paths matched"

    def test_random_1k(self):
        kg, nx_g, names = _build_paired_graphs(1000, edge_factor=3)
        matches, tested = self._compare_paths(kg, nx_g, names, 20, "random_1k")
        assert matches == tested, f"Only {matches}/{tested} paths matched"

    def test_chain_100(self):
        kg, nx_g, names = _build_chain(100)
        # Test specific known distances
        for dist in [1, 5, 10, 50, 99]:
            src, tgt = "N0", f"N{dist}"
            nx_len = nx.shortest_path_length(nx_g, src, tgt)
            result = kg.cypher(
                f"MATCH p = shortestPath((a:Node {{name: '{src}'}})"
                f"-[*..200]->"
                f"(b:Node {{name: '{tgt}'}})) "
                f"RETURN length(p) AS len"
            )
            rg_len = result[0]["len"]
            assert rg_len == nx_len == dist, f"Chain dist {dist}: rg={rg_len}, nx={nx_len}"
        print("  shortest_path chain_100: all distances correct")

    def test_disconnected_no_path(self):
        """Ensure both systems agree when no path exists."""
        kg, nx_g, names = _build_clusters(3, 20, inter_edges=0)

        # Pick nodes from different clusters — no path should exist
        src, tgt = "N0", f"N{20}"  # cluster 0 -> cluster 1

        try:
            nx.shortest_path_length(nx_g, src, tgt)
            nx_has_path = True
        except nx.NetworkXNoPath:
            nx_has_path = False

        result = kg.cypher(
            f"MATCH p = shortestPath((a:Node {{name: '{src}'}})"
            f"-[*..200]->"
            f"(b:Node {{name: '{tgt}'}})) "
            f"RETURN length(p) AS len"
        )
        rg_has_path = len(result) > 0

        assert rg_has_path == nx_has_path, f"Path existence disagrees: rg={rg_has_path}, nx={nx_has_path}"
        print(f"  shortest_path disconnected: both agree path={'exists' if nx_has_path else 'absent'}")


class TestConnectedComponentsAccuracy:
    """Validate connected components against NetworkX."""

    def _compare_components(self, kg, nx_g, label=""):
        """Compare weakly connected components."""
        # NetworkX: weakly connected components on DiGraph
        nx_components = list(nx.weakly_connected_components(nx_g))
        nx_sizes = sorted([len(c) for c in nx_components], reverse=True)

        # kglite
        rg_components = kg.connected_components()
        rg_sizes = sorted([len(c) for c in rg_components], reverse=True)

        print(f"  connected_components {label}: rg={len(rg_sizes)} components, nx={len(nx_sizes)} components")
        print(f"    rg sizes (top 5): {rg_sizes[:5]}")
        print(f"    nx sizes (top 5): {nx_sizes[:5]}")

        return rg_sizes, nx_sizes

    def test_random_100(self):
        kg, nx_g, _ = _build_paired_graphs(100, edge_factor=3)
        rg_sizes, nx_sizes = self._compare_components(kg, nx_g, "random_100")
        assert rg_sizes == nx_sizes, f"Component sizes differ: rg={rg_sizes}, nx={nx_sizes}"

    def test_random_1k(self):
        kg, nx_g, _ = _build_paired_graphs(1000, edge_factor=3)
        rg_sizes, nx_sizes = self._compare_components(kg, nx_g, "random_1k")
        assert rg_sizes == nx_sizes, "Component sizes differ"

    def test_clusters(self):
        kg, nx_g, _ = _build_clusters(5, 40, inter_edges=0)
        rg_sizes, nx_sizes = self._compare_components(kg, nx_g, "clusters_5x40")
        assert len(rg_sizes) == len(nx_sizes) == 5, f"Expected 5 components, got rg={len(rg_sizes)}, nx={len(nx_sizes)}"
        assert rg_sizes == nx_sizes, "Component sizes differ"


class TestCentralityAccuracy:
    """Validate centrality measures against NetworkX."""

    def test_betweenness_random_100(self):
        kg, nx_g, names = _build_paired_graphs(100, edge_factor=3)
        rg = _extract_rg_scores(kg, "betweenness_centrality")
        nx_scores = nx.betweenness_centrality(nx_g, normalized=True)
        corr, max_err, _ = _compare_scores(rg, nx_scores, names, "betweenness_100", min_corr=0.95, max_abs_err=0.05)
        assert corr > 0.95, f"Correlation too low: {corr}"

    def test_betweenness_random_500(self):
        kg, nx_g, names = _build_paired_graphs(500, edge_factor=3)
        rg = _extract_rg_scores(kg, "betweenness_centrality")
        nx_scores = nx.betweenness_centrality(nx_g, normalized=True)
        corr, max_err, _ = _compare_scores(rg, nx_scores, names, "betweenness_500", min_corr=0.95, max_abs_err=0.05)
        assert corr > 0.95, f"Correlation too low: {corr}"

    def test_betweenness_scale_free_100(self):
        kg, nx_g, names = _build_scale_free(100)
        rg = _extract_rg_scores(kg, "betweenness_centrality")
        nx_scores = nx.betweenness_centrality(nx_g, normalized=True)
        corr, max_err, _ = _compare_scores(rg, nx_scores, names, "betweenness_sf_100", min_corr=0.95, max_abs_err=0.05)
        assert corr > 0.95, f"Correlation too low: {corr}"

    def test_closeness_random_100(self):
        kg, nx_g, names = _build_paired_graphs(100, edge_factor=3)
        rg = _extract_rg_scores(kg, "closeness_centrality")
        nx_scores = nx.closeness_centrality(nx_g)
        corr, max_err, _ = _compare_scores(rg, nx_scores, names, "closeness_100", min_corr=0.95, max_abs_err=0.1)
        assert corr > 0.95, f"Correlation too low: {corr}"

    def test_closeness_random_500(self):
        kg, nx_g, names = _build_paired_graphs(500, edge_factor=3)
        rg = _extract_rg_scores(kg, "closeness_centrality")
        nx_scores = nx.closeness_centrality(nx_g)
        corr, max_err, _ = _compare_scores(rg, nx_scores, names, "closeness_500", min_corr=0.95, max_abs_err=0.1)
        assert corr > 0.95, f"Correlation too low: {corr}"

    def test_degree_random_100(self):
        kg, nx_g, names = _build_paired_graphs(100, edge_factor=3)
        rg = _extract_rg_scores(kg, "degree_centrality")
        nx_scores = nx.degree_centrality(nx_g)
        # Degree centrality should be exact
        for name in names:
            rg_val = rg.get(name, 0.0)
            nx_val = nx_scores.get(name, 0.0)
            assert abs(rg_val - nx_val) < 1e-10, f"Degree mismatch for {name}: rg={rg_val}, nx={nx_val}"
        print("  degree_centrality_100: exact match")


class TestCommunityDetectionAccuracy:
    """Validate community detection against NetworkX."""

    def test_louvain_planted_communities(self):
        """Louvain should recover planted community structure."""
        kg, nx_g, names = _build_clusters(4, 50, inter_edges=3)

        # kglite
        rg_result = kg.louvain_communities()
        rg_modularity = rg_result["modularity"]
        rg_n_comm = rg_result["num_communities"]

        # NetworkX (convert to undirected for community detection)
        nx_ug = nx_g.to_undirected()
        nx_communities = nx.community.louvain_communities(nx_ug, seed=42)
        nx_modularity = nx.community.modularity(nx_ug, nx_communities)
        nx_n_comm = len(nx_communities)

        print(f"  louvain: rg_communities={rg_n_comm}, nx_communities={nx_n_comm}")
        print(f"  louvain: rg_modularity={rg_modularity:.4f}, nx_modularity={nx_modularity:.4f}")

        # Both should find approximately 4 communities
        assert rg_n_comm >= 3, f"Too few communities: {rg_n_comm}"
        assert rg_n_comm <= 8, f"Too many communities: {rg_n_comm}"

        # Modularity should be positive and reasonably close
        assert rg_modularity > 0.1, f"Modularity too low: {rg_modularity}"
        # Allow 50% relative difference since algorithms are non-deterministic
        if nx_modularity > 0:
            ratio = rg_modularity / nx_modularity
            assert ratio > 0.5, f"Modularity ratio too low: {ratio:.2f}"

    def test_label_propagation_planted_communities(self):
        """Label propagation should recover planted communities."""
        kg, nx_g, names = _build_clusters(4, 50, inter_edges=3)

        # kglite
        rg_result = kg.label_propagation()
        rg_n_comm = rg_result["num_communities"]

        # NetworkX
        nx_ug = nx_g.to_undirected()
        nx_communities = list(nx.community.label_propagation_communities(nx_ug))
        nx_n_comm = len(nx_communities)

        print(f"  label_prop: rg_communities={rg_n_comm}, nx_communities={nx_n_comm}")

        # Both should find approximately 4 communities (label prop is non-deterministic)
        assert rg_n_comm >= 2, f"Too few communities: {rg_n_comm}"
        assert rg_n_comm <= 10, f"Too many communities: {rg_n_comm}"


class TestAggregationAccuracy:
    """Validate Cypher aggregations against Python/NetworkX equivalents."""

    def test_aggregations_100(self):
        kg, nx_g, names = _build_paired_graphs(100, edge_factor=3)

        result = kg.cypher("""
            MATCH (n:Node)
            RETURN count(n) AS cnt,
                   sum(n.value) AS total,
                   avg(n.value) AS mean,
                   min(n.value) AS lo,
                   max(n.value) AS hi
        """)
        row = result[0]

        # Compute from NetworkX
        values = [nx_g.nodes[n]["value"] for n in nx_g.nodes()]
        expected_cnt = len(values)
        expected_sum = sum(values)
        expected_avg = expected_sum / expected_cnt
        expected_min = min(values)
        expected_max = max(values)

        assert row["cnt"] == expected_cnt
        assert abs(row["total"] - expected_sum) < 1e-6, f"sum: {row['total']} vs {expected_sum}"
        assert abs(row["mean"] - expected_avg) < 1e-6, f"avg: {row['mean']} vs {expected_avg}"
        assert abs(row["lo"] - expected_min) < 1e-6, f"min: {row['lo']} vs {expected_min}"
        assert abs(row["hi"] - expected_max) < 1e-6, f"max: {row['hi']} vs {expected_max}"
        print(f"  aggregations_100: all match (cnt={expected_cnt}, sum={expected_sum:.2f})")

    def test_aggregations_1k(self):
        kg, nx_g, names = _build_paired_graphs(1000, edge_factor=3)

        result = kg.cypher("""
            MATCH (n:Node)
            RETURN count(n) AS cnt, sum(n.value) AS total, avg(n.value) AS mean
        """)
        row = result[0]

        values = [nx_g.nodes[n]["value"] for n in nx_g.nodes()]
        assert row["cnt"] == len(values)
        assert abs(row["total"] - sum(values)) < 1e-4
        assert abs(row["mean"] - sum(values) / len(values)) < 1e-4
        print("  aggregations_1k: all match")

    def test_group_by_aggregation(self):
        kg, nx_g, names = _build_paired_graphs(100, edge_factor=3)

        result = kg.cypher("""
            MATCH (n:Node)
            RETURN n.group AS grp, count(n) AS cnt, avg(n.value) AS mean
            ORDER BY grp
        """)

        # Compute from NetworkX
        groups = defaultdict(list)
        for n in nx_g.nodes():
            groups[nx_g.nodes[n]["group"]].append(nx_g.nodes[n]["value"])

        for row in result:
            grp = row["grp"]
            expected_cnt = len(groups[grp])
            expected_mean = sum(groups[grp]) / len(groups[grp])
            assert row["cnt"] == expected_cnt, f"Group {grp}: cnt {row['cnt']} vs {expected_cnt}"
            assert abs(row["mean"] - expected_mean) < 1e-6, f"Group {grp}: mean {row['mean']} vs {expected_mean}"
        print(f"  group_by_aggregation: {len(result)} groups all match")


# ============================================================================
# Performance Tests
# ============================================================================


class TestConstructionPerformance:
    """Measure graph construction times."""

    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_construction(self, n):
        rng = random.Random(42)
        node_names = [f"N{i}" for i in range(n)]
        values = [rng.uniform(0, 100) for _ in range(n)]
        groups = [rng.randint(1, 5) for _ in range(n)]

        edges = set()
        while len(edges) < 3 * n:
            s = rng.randint(0, n - 1)
            t = rng.randint(0, n - 1)
            if s != t:
                edges.add((s, t))
        edge_list = list(edges)

        # NetworkX
        t0 = time.perf_counter()
        nx_g = nx.DiGraph()
        for i in range(n):
            nx_g.add_node(node_names[i], value=values[i], group=groups[i])
        for s, t in edge_list:
            nx_g.add_edge(node_names[s], node_names[t])
        nx_time = time.perf_counter() - t0

        # kglite (bulk)
        t0 = time.perf_counter()
        kg = KnowledgeGraph()
        df_nodes = pd.DataFrame(
            {
                "id": list(range(n)),
                "name": node_names,
                "value": values,
                "group": groups,
            }
        )
        kg.add_nodes(df_nodes, "Node", "id", "name")
        df_edges = pd.DataFrame(
            {
                "from_id": [s for s, t in edge_list],
                "to_id": [t for s, t in edge_list],
            }
        )
        kg.add_connections(df_edges, "EDGE", "Node", "from_id", "Node", "to_id")
        rg_time = time.perf_counter() - t0

        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  construction_{n}: rg={rg_time:.3f}s, nx={nx_time:.3f}s, ratio={ratio:.1f}x")


class TestAlgorithmPerformance:
    """Compare algorithm execution times."""

    def _time_algo(self, kg, nx_g, rg_func, nx_func, algo_name, n):
        """Time both implementations and print comparison."""
        t0 = time.perf_counter()
        rg_func(kg)
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        nx_func(nx_g)
        nx_time = time.perf_counter() - t0

        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  {algo_name}_{n}: rg={rg_time:.4f}s, nx={nx_time:.4f}s, speedup={ratio:.1f}x")
        return rg_time, nx_time, ratio

    # --- PageRank ---
    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_pagerank(self, n):
        kg, nx_g, _ = _build_paired_graphs(n, edge_factor=3)
        self._time_algo(
            kg,
            nx_g,
            lambda g: g.pagerank(),
            lambda g: nx.pagerank(g, alpha=0.85),
            "pagerank",
            n,
        )

    # --- Shortest Path via Cypher (20 random pairs) ---
    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_shortest_path_cypher(self, n):
        """Shortest path via Cypher (includes tokenize/parse/plan/execute overhead)."""
        kg, nx_g, names = _build_paired_graphs(n, edge_factor=3)
        rng = random.Random(42)
        pairs = [(rng.choice(names), rng.choice(names)) for _ in range(20)]

        t0 = time.perf_counter()
        for src, tgt in pairs:
            if src != tgt:
                try:
                    kg.cypher(
                        f"MATCH p = shortestPath((a:Node {{name: '{src}'}})"
                        f"-[*..500]->"
                        f"(b:Node {{name: '{tgt}'}})) "
                        f"RETURN length(p)"
                    )
                except Exception:
                    pass
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for src, tgt in pairs:
            if src != tgt:
                try:
                    nx.shortest_path_length(nx_g, src, tgt)
                except nx.NetworkXNoPath:
                    pass
        nx_time = time.perf_counter() - t0

        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(
            f"  shortest_path_cypher_20pairs_{n}: rg={rg_time:.4f}s, nx={nx_time:.4f}s, "
            f"speedup={ratio:.1f}x (includes Cypher pipeline overhead)"
        )

    # --- Shortest Path via fluent API (20 random pairs) ---
    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_shortest_path_fluent(self, n):
        """Shortest path via direct API (no Cypher pipeline overhead)."""
        kg, nx_g, names = _build_paired_graphs(n, edge_factor=3)
        rng = random.Random(42)
        pairs = [(rng.choice(names), rng.choice(names)) for _ in range(20)]

        t0 = time.perf_counter()
        for src, tgt in pairs:
            if src != tgt:
                try:
                    src_id = int(src[1:])  # "N123" -> 123
                    tgt_id = int(tgt[1:])
                    kg.shortest_path_length("Node", src_id, "Node", tgt_id)
                except Exception:
                    pass
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for src, tgt in pairs:
            if src != tgt:
                try:
                    nx.shortest_path_length(nx_g, src, tgt)
                except nx.NetworkXNoPath:
                    pass
        nx_time = time.perf_counter() - t0

        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  shortest_path_fluent_20pairs_{n}: rg={rg_time:.4f}s, nx={nx_time:.4f}s, speedup={ratio:.1f}x")

    # --- Connected Components ---
    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_connected_components(self, n):
        kg, nx_g, _ = _build_paired_graphs(n, edge_factor=3)
        self._time_algo(
            kg,
            nx_g,
            lambda g: g.connected_components(),
            lambda g: list(nx.weakly_connected_components(g)),
            "connected_components",
            n,
        )

    # --- Betweenness Centrality (skip 50K — O(VE)) ---
    @pytest.mark.parametrize("n", [100, 1000, 10000])
    def test_betweenness_centrality(self, n):
        kg, nx_g, _ = _build_paired_graphs(n, edge_factor=3)
        self._time_algo(
            kg,
            nx_g,
            lambda g: g.betweenness_centrality(),
            lambda g: nx.betweenness_centrality(g),
            "betweenness",
            n,
        )

    # --- Closeness Centrality (skip 50K — O(VE)) ---
    @pytest.mark.parametrize("n", [100, 1000, 10000])
    def test_closeness_centrality(self, n):
        kg, nx_g, _ = _build_paired_graphs(n, edge_factor=3)
        self._time_algo(
            kg,
            nx_g,
            lambda g: g.closeness_centrality(),
            lambda g: nx.closeness_centrality(g),
            "closeness",
            n,
        )

    # --- Degree Centrality ---
    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_degree_centrality(self, n):
        kg, nx_g, _ = _build_paired_graphs(n, edge_factor=3)
        self._time_algo(
            kg,
            nx_g,
            lambda g: g.degree_centrality(),
            lambda g: nx.degree_centrality(g),
            "degree",
            n,
        )

    # --- Optimized API variants (as_dict / titles_only / batch) ---

    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_pagerank_as_dict(self, n):
        """PageRank returning {title: score} dict — same format as NetworkX."""
        kg, nx_g, _ = _build_paired_graphs(n, edge_factor=3)
        self._time_algo(
            kg,
            nx_g,
            lambda g: g.pagerank(as_dict=True),
            lambda g: nx.pagerank(g, alpha=0.85),
            "pagerank_dict",
            n,
        )

    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_degree_centrality_as_dict(self, n):
        """Degree centrality returning {title: score} dict."""
        kg, nx_g, _ = _build_paired_graphs(n, edge_factor=3)
        self._time_algo(
            kg,
            nx_g,
            lambda g: g.degree_centrality(as_dict=True),
            lambda g: nx.degree_centrality(g),
            "degree_dict",
            n,
        )

    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_connected_components_titles(self, n):
        """Connected components returning title lists only."""
        kg, nx_g, _ = _build_paired_graphs(n, edge_factor=3)
        self._time_algo(
            kg,
            nx_g,
            lambda g: g.connected_components(titles_only=True),
            lambda g: list(nx.weakly_connected_components(g)),
            "connected_titles",
            n,
        )

    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_shortest_path_batch(self, n):
        """Shortest path via batch API — amortizes adj list + allocation."""
        kg, nx_g, names = _build_paired_graphs(n, edge_factor=3)
        rng = random.Random(42)
        pairs = [(rng.choice(names), rng.choice(names)) for _ in range(20)]
        # Filter out self-pairs
        pairs = [(s, t) for s, t in pairs if s != t]
        id_pairs = [(int(s[1:]), int(t[1:])) for s, t in pairs]

        t0 = time.perf_counter()
        kg.shortest_path_lengths_batch("Node", id_pairs)
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for src, tgt in pairs:
            try:
                nx.shortest_path_length(nx_g, src, tgt)
            except nx.NetworkXNoPath:
                pass
        nx_time = time.perf_counter() - t0

        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  shortest_path_batch_20pairs_{n}: rg={rg_time:.4f}s, nx={nx_time:.4f}s, speedup={ratio:.1f}x")

    # --- Louvain Communities ---
    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_louvain(self, n):
        kg, nx_g, _ = _build_paired_graphs(n, edge_factor=3)
        self._time_algo(
            kg,
            nx_g,
            lambda g: g.louvain_communities(),
            lambda g: nx.community.louvain_communities(g.to_undirected(), seed=42),
            "louvain",
            n,
        )

    # --- Label Propagation ---
    @pytest.mark.parametrize("n", [100, 1000, 10000, 50000])
    def test_label_propagation(self, n):
        kg, nx_g, _ = _build_paired_graphs(n, edge_factor=3)
        self._time_algo(
            kg,
            nx_g,
            lambda g: g.label_propagation(),
            lambda g: list(nx.community.label_propagation_communities(g.to_undirected())),
            "label_prop",
            n,
        )


class TestCypherQueryPerformance:
    """Compare complex Cypher queries against equivalent Python/NetworkX code."""

    @pytest.fixture(autouse=True)
    def setup_graph(self):
        """Build a 10K-node graph for Cypher benchmarks."""
        self.kg, self.nx_g, self.names = _build_paired_graphs(10000, edge_factor=3)

    def test_simple_filter(self):
        """MATCH (n:Node) WHERE n.group = 1 RETURN n.name"""
        t0 = time.perf_counter()
        rg_result = self.kg.cypher("MATCH (n:Node) WHERE n.group = 1 RETURN n.name")
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        nx_result = [n for n, d in self.nx_g.nodes(data=True) if d["group"] == 1]
        nx_time = time.perf_counter() - t0

        assert len(rg_result) == len(nx_result)
        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(
            f"  simple_filter_10k: rg={rg_time:.4f}s, nx={nx_time:.4f}s, speedup={ratio:.1f}x, {len(rg_result)} results"
        )

    def test_pattern_match(self):
        """MATCH (a)-[:EDGE]->(b) WHERE a.group = b.group"""
        t0 = time.perf_counter()
        rg_result = self.kg.cypher("""
            MATCH (a:Node)-[:EDGE]->(b:Node)
            WHERE a.group = b.group
            RETURN count(*) AS cnt
        """)
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        nx_count = sum(1 for u, v in self.nx_g.edges() if self.nx_g.nodes[u]["group"] == self.nx_g.nodes[v]["group"])
        nx_time = time.perf_counter() - t0

        assert rg_result[0]["cnt"] == nx_count
        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  pattern_match_10k: rg={rg_time:.4f}s, nx={nx_time:.4f}s, speedup={ratio:.1f}x, count={nx_count}")

    def test_aggregation(self):
        """GROUP BY + aggregation"""
        t0 = time.perf_counter()
        rg_result = self.kg.cypher("""
            MATCH (n:Node)
            RETURN n.group AS grp, count(n) AS cnt, avg(n.value) AS mean
            ORDER BY grp
        """)
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        groups = defaultdict(list)
        for n, d in self.nx_g.nodes(data=True):
            groups[d["group"]].append(d["value"])
        nx_result = sorted(
            [{"grp": g, "cnt": len(v), "mean": sum(v) / len(v)} for g, v in groups.items()], key=lambda x: x["grp"]
        )
        nx_time = time.perf_counter() - t0

        assert len(rg_result) == len(nx_result)
        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  aggregation_10k: rg={rg_time:.4f}s, nx={nx_time:.4f}s, speedup={ratio:.1f}x, {len(rg_result)} groups")

    def test_optional_match_count(self):
        """OPTIONAL MATCH + count (tests fusion optimization)"""
        # Use smaller graph for this since OPTIONAL MATCH can be expensive
        kg, nx_g, names = _build_paired_graphs(1000, edge_factor=3)

        t0 = time.perf_counter()
        rg_result = kg.cypher("""
            MATCH (n:Node)
            OPTIONAL MATCH (n)-[:EDGE]->(m:Node)
            WITH n, count(m) AS deg
            RETURN n.name, deg
        """)
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        nx_result = {n: nx_g.out_degree(n) for n in nx_g.nodes()}
        nx_time = time.perf_counter() - t0

        assert len(rg_result) == len(nx_result)
        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  optional_match_count_1k: rg={rg_time:.4f}s, nx={nx_time:.4f}s, speedup={ratio:.1f}x")

    def test_order_by_limit(self):
        """ORDER BY + LIMIT"""
        t0 = time.perf_counter()
        rg_result = self.kg.cypher("""
            MATCH (n:Node)
            RETURN n.name, n.value
            ORDER BY n.value DESC
            LIMIT 10
        """)
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = sorted([(n, d["value"]) for n, d in self.nx_g.nodes(data=True)], key=lambda x: -x[1])[:10]
        nx_time = time.perf_counter() - t0

        assert len(rg_result) == 10
        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  order_limit_10k: rg={rg_time:.4f}s, nx={nx_time:.4f}s, speedup={ratio:.1f}x")

    def test_exists_filter(self):
        """WHERE EXISTS subquery"""
        t0 = time.perf_counter()
        rg_result = self.kg.cypher("""
            MATCH (n:Node)
            WHERE EXISTS { (n)-[:EDGE]->() }
            RETURN count(n) AS cnt
        """)
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        nx_count = sum(1 for n in self.nx_g.nodes() if self.nx_g.out_degree(n) > 0)
        nx_time = time.perf_counter() - t0

        assert rg_result[0]["cnt"] == nx_count
        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  exists_filter_10k: rg={rg_time:.4f}s, nx={nx_time:.4f}s, speedup={ratio:.1f}x, count={nx_count}")

    def test_case_expression(self):
        """CASE expression aggregation"""
        t0 = time.perf_counter()
        rg_result = self.kg.cypher("""
            MATCH (n:Node)
            RETURN CASE WHEN n.value > 50 THEN 'high' ELSE 'low' END AS cat,
                   count(n) AS cnt
        """)
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        cats = defaultdict(int)
        for n, d in self.nx_g.nodes(data=True):
            cat = "high" if d["value"] > 50 else "low"
            cats[cat] += 1
        nx_time = time.perf_counter() - t0

        rg_cats = {r["cat"]: r["cnt"] for r in rg_result}
        assert rg_cats == dict(cats)
        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  case_expr_10k: rg={rg_time:.4f}s, nx={nx_time:.4f}s, speedup={ratio:.1f}x")


class TestSerializationPerformance:
    """Compare save times and file sizes (no load API exposed to Python yet)."""

    @pytest.mark.parametrize("n", [1000, 10000, 50000])
    def test_serialization(self, n):
        kg, nx_g, _ = _build_paired_graphs(n, edge_factor=3)

        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=True) as f:
            rg_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=True) as f:
            nx_path = f.name

        try:
            # kglite save
            t0 = time.perf_counter()
            kg.save(rg_path)
            rg_save_time = time.perf_counter() - t0

            # NetworkX pickle save
            t0 = time.perf_counter()
            with open(nx_path, "wb") as f:
                pickle.dump(nx_g, f, protocol=pickle.HIGHEST_PROTOCOL)
            nx_save_time = time.perf_counter() - t0

            import os

            rg_size = os.path.getsize(rg_path)
            nx_size = os.path.getsize(nx_path)
            size_ratio = rg_size / nx_size if nx_size > 0 else float("inf")

            save_ratio = nx_save_time / rg_save_time if rg_save_time > 0 else float("inf")

            print(f"  serialization_{n}:")
            print(f"    save: rg={rg_save_time:.3f}s, nx={nx_save_time:.3f}s, ratio={save_ratio:.1f}x")
            print(f"    size: rg={rg_size / 1024:.0f}KB, nx={nx_size / 1024:.0f}KB, ratio={size_ratio:.1f}x")
        finally:
            import os

            for p in [rg_path, nx_path]:
                if os.path.exists(p):
                    os.unlink(p)


# ============================================================================
# Storage-mode helpers + benchmarks
# ============================================================================


def _build_kg_mode(n, mode, path=None, edge_factor=3, seed=42):
    """Build a kglite graph in the given storage mode with the same shape as
    ``_build_paired_graphs``. ``mode`` is one of 'memory', 'mapped', 'disk'.

    ``path`` is required for ``mode='disk'`` (directory that becomes the graph).
    Returns (kg, node_names).
    """
    rng = random.Random(seed)
    node_names = [f"N{i}" for i in range(n)]
    node_values = [rng.uniform(0, 100) for _ in range(n)]
    node_groups = [rng.randint(1, 5) for _ in range(n)]

    n_edges = edge_factor * n
    edge_set = set()
    while len(edge_set) < n_edges:
        s = rng.randint(0, n - 1)
        t = rng.randint(0, n - 1)
        if s != t:
            edge_set.add((s, t))
    edges = list(edge_set)

    if mode == "memory":
        kg = KnowledgeGraph()
    elif mode == "mapped":
        kg = KnowledgeGraph(storage="mapped")
    elif mode == "disk":
        if path is None:
            raise ValueError("mode='disk' requires path")
        kg = KnowledgeGraph(storage="disk", path=str(path))
    else:
        raise ValueError(f"unknown storage mode: {mode}")

    df_nodes = pd.DataFrame(
        {
            "id": list(range(n)),
            "name": node_names,
            "value": node_values,
            "group": node_groups,
        }
    )
    kg.add_nodes(df_nodes, "Node", "id", "name")
    df_edges = pd.DataFrame(
        {
            "from_id": [s for s, t in edges],
            "to_id": [t for s, t in edges],
        }
    )
    kg.add_connections(df_edges, "EDGE", "Node", "from_id", "Node", "to_id")
    return kg, node_names


STORAGE_MODES = ("memory", "mapped", "disk")


# ============================================================================
# Schema introspection benchmarks
# ============================================================================


class TestSchemaIntrospectionPerformance:
    """Locks in the cost of the agent-facing describe()/schema()/find() path.

    These are the default entry points for MCP servers — a regression here
    adds latency to every tool session. NX has no direct analog, so we only
    record kglite timings but include find-by-name vs NX linear scan.
    """

    @pytest.mark.parametrize("n", [100, 1000, 10000])
    def test_describe(self, n):
        kg, _, _ = _build_paired_graphs(n, edge_factor=3)
        # Warm the caches (describe() touches capability + connectivity metadata)
        kg.describe()
        t0 = time.perf_counter()
        for _ in range(5):
            kg.describe()
        rg_time = (time.perf_counter() - t0) / 5
        print(f"  describe_{n}: rg={rg_time * 1000:.2f}ms/call")

    @pytest.mark.parametrize("n", [100, 1000, 10000])
    def test_schema(self, n):
        kg, _, _ = _build_paired_graphs(n, edge_factor=3)
        kg.schema()
        t0 = time.perf_counter()
        for _ in range(5):
            kg.schema()
        rg_time = (time.perf_counter() - t0) / 5
        print(f"  schema_{n}: rg={rg_time * 1000:.2f}ms/call")

    @pytest.mark.parametrize("n", [100, 1000, 10000])
    def test_find_by_name(self, n):
        """Indexed node lookup vs NX linear scan (20 random targets)."""
        kg, nx_g, names = _build_paired_graphs(n, edge_factor=3)
        rng = random.Random(7)
        targets = [rng.choice(names) for _ in range(20)]

        t0 = time.perf_counter()
        for name in targets:
            kg.find(name, node_type="Node")
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for name in targets:
            _ = [n for n, d in nx_g.nodes(data=True) if n == name]
        nx_time = time.perf_counter() - t0

        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  find_by_name_20x_{n}: rg={rg_time * 1000:.3f}ms, nx={nx_time * 1000:.3f}ms, speedup={ratio:.1f}x")


# ============================================================================
# Property-heavy query benchmarks
# ============================================================================


class TestPropertyQueryPerformance:
    """Filter-style queries that dominate real KG workloads.

    kglite pushes predicates through the Cypher planner; NX is a Python scan.
    These should show a consistent multi-× win for kglite.
    """

    @pytest.fixture(autouse=True)
    def setup_graph(self):
        self.kg, self.nx_g, _ = _build_paired_graphs(10000, edge_factor=3)

    def test_multi_predicate_filter(self):
        t0 = time.perf_counter()
        rg_result = self.kg.cypher("MATCH (n:Node) WHERE n.group = 1 AND n.value > 50 RETURN n.name")
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        nx_result = [n for n, d in self.nx_g.nodes(data=True) if d["group"] == 1 and d["value"] > 50]
        nx_time = time.perf_counter() - t0

        assert len(rg_result) == len(nx_result)
        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(
            f"  multi_predicate_10k: rg={rg_time * 1000:.3f}ms, "
            f"nx={nx_time * 1000:.3f}ms, speedup={ratio:.1f}x, {len(rg_result)} results"
        )

    def test_in_list_filter(self):
        t0 = time.perf_counter()
        rg_result = self.kg.cypher("MATCH (n:Node) WHERE n.group IN [1, 3, 5] RETURN n.name")
        rg_time = time.perf_counter() - t0

        in_set = {1, 3, 5}
        t0 = time.perf_counter()
        nx_result = [n for n, d in self.nx_g.nodes(data=True) if d["group"] in in_set]
        nx_time = time.perf_counter() - t0

        assert len(rg_result) == len(nx_result)
        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(
            f"  in_list_10k: rg={rg_time * 1000:.3f}ms, "
            f"nx={nx_time * 1000:.3f}ms, speedup={ratio:.1f}x, {len(rg_result)} results"
        )

    def test_string_contains(self):
        t0 = time.perf_counter()
        rg_result = self.kg.cypher("MATCH (n:Node) WHERE n.name CONTAINS '42' RETURN n.name")
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        nx_result = [n for n, _ in self.nx_g.nodes(data=True) if "42" in n]
        nx_time = time.perf_counter() - t0

        assert len(rg_result) == len(nx_result)
        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(
            f"  contains_10k: rg={rg_time * 1000:.3f}ms, "
            f"nx={nx_time * 1000:.3f}ms, speedup={ratio:.1f}x, {len(rg_result)} results"
        )

    def test_range_filter(self):
        t0 = time.perf_counter()
        rg_result = self.kg.cypher("MATCH (n:Node) WHERE n.value >= 25 AND n.value < 75 RETURN n.name")
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        nx_result = [n for n, d in self.nx_g.nodes(data=True) if 25 <= d["value"] < 75]
        nx_time = time.perf_counter() - t0

        assert len(rg_result) == len(nx_result)
        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(
            f"  range_10k: rg={rg_time * 1000:.3f}ms, "
            f"nx={nx_time * 1000:.3f}ms, speedup={ratio:.1f}x, {len(rg_result)} results"
        )


# ============================================================================
# Multi-hop traversal benchmarks
# ============================================================================


class TestTraversalPerformance:
    """Multi-hop neighborhood queries — kglite Cypher vs NX bfs_tree/successors."""

    @pytest.mark.parametrize("n", [1000, 10000])
    def test_two_hop_neighbors(self, n):
        """Count 2-hop out-neighbors of a sample of nodes."""
        kg, nx_g, names = _build_paired_graphs(n, edge_factor=3)
        rng = random.Random(3)
        seeds = [rng.choice(names) for _ in range(20)]

        t0 = time.perf_counter()
        for name in seeds:
            kg.cypher(f"MATCH (a:Node {{name: '{name}'}})-[:EDGE*2]->(b:Node) RETURN count(DISTINCT b) AS cnt")
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for name in seeds:
            # 2-hop via NX: expand out-neighbors twice
            lvl1 = set(nx_g.successors(name)) if name in nx_g else set()
            lvl2 = set()
            for m in lvl1:
                lvl2.update(nx_g.successors(m))
            _ = len(lvl2)
        nx_time = time.perf_counter() - t0

        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  two_hop_20x_{n}: rg={rg_time * 1000:.2f}ms, nx={nx_time * 1000:.2f}ms, speedup={ratio:.1f}x")

    @pytest.mark.parametrize("n", [1000, 10000])
    def test_three_hop_aggregation(self, n):
        """3-hop traversal with per-group aggregation."""
        kg, nx_g, names = _build_paired_graphs(n, edge_factor=3)
        rng = random.Random(9)
        seeds = [rng.choice(names) for _ in range(10)]

        t0 = time.perf_counter()
        for name in seeds:
            kg.cypher(
                f"MATCH (a:Node {{name: '{name}'}})-[:EDGE*3]->(b:Node) RETURN b.group AS grp, count(DISTINCT b) AS cnt"
            )
        rg_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for name in seeds:
            lvl = {name} if name in nx_g else set()
            for _ in range(3):
                nxt = set()
                for m in lvl:
                    nxt.update(nx_g.successors(m))
                lvl = nxt
            groups = defaultdict(int)
            for m in lvl:
                groups[nx_g.nodes[m]["group"]] += 1
        nx_time = time.perf_counter() - t0

        ratio = nx_time / rg_time if rg_time > 0 else float("inf")
        print(f"  three_hop_agg_10x_{n}: rg={rg_time * 1000:.2f}ms, nx={nx_time * 1000:.2f}ms, speedup={ratio:.1f}x")


# ============================================================================
# Storage-mode matrix: memory vs mapped vs disk (+ NX as baseline)
# ============================================================================


class TestStorageModeMatrix:
    """Cross-cut kglite's three storage modes at 10K nodes.

    **API parity contract**: the same Python/Cypher calls must work
    identically in ``memory``, ``mapped``, and ``disk`` modes. This class
    runs each critical operation in all three modes — any per-mode failure
    surfaces as its own pytest failure, exposing API mismatches rather than
    hiding them. Result values are asserted equal across modes where
    applicable, so subtle silent divergences also fail.
    """

    N = 10_000

    def _build(self, mode, tmp_path):
        if mode == "disk":
            dpath = tmp_path / f"disk_{self.N}"
            dpath.mkdir(exist_ok=True)
            return _build_kg_mode(self.N, mode, path=dpath)
        return _build_kg_mode(self.N, mode)

    @pytest.mark.parametrize("mode", STORAGE_MODES)
    def test_construction(self, mode, tmp_path):
        """Wall-clock build time, mode by mode."""
        t0 = time.perf_counter()
        kg, _ = self._build(mode, tmp_path)
        rg_time = time.perf_counter() - t0
        del kg
        print(f"  construction_{self.N}_{mode}: {rg_time:.3f}s")

    @pytest.mark.parametrize("mode", STORAGE_MODES)
    def test_describe(self, mode, tmp_path):
        """Agent-facing introspection must work in every mode."""
        kg, _ = self._build(mode, tmp_path)
        kg.describe()
        t0 = time.perf_counter()
        for _ in range(5):
            out = kg.describe()
        rg_time = (time.perf_counter() - t0) / 5
        assert "<graph" in out, f"describe() on {mode} returned malformed output"
        print(f"  describe_{self.N}_{mode}: {rg_time * 1000:.2f}ms/call")

    @pytest.mark.parametrize("mode", STORAGE_MODES)
    def test_schema(self, mode, tmp_path):
        kg, _ = self._build(mode, tmp_path)
        t0 = time.perf_counter()
        for _ in range(5):
            s = kg.schema()
        rg_time = (time.perf_counter() - t0) / 5
        assert "Node" in s["node_types"], f"schema() on {mode} missing Node type"
        print(f"  schema_{self.N}_{mode}: {rg_time * 1000:.2f}ms/call")

    @pytest.mark.parametrize("mode", STORAGE_MODES)
    def test_find(self, mode, tmp_path):
        kg, names = self._build(mode, tmp_path)
        rng = random.Random(5)
        targets = [rng.choice(names) for _ in range(20)]
        t0 = time.perf_counter()
        for name in targets:
            hit = kg.find(name, node_type="Node")
            assert hit, f"find('{name}') on {mode} returned no result"
        rg_time = time.perf_counter() - t0
        print(f"  find_20x_{self.N}_{mode}: {rg_time * 1000:.2f}ms")

    @pytest.mark.parametrize("mode", STORAGE_MODES)
    def test_multi_predicate(self, mode, tmp_path):
        kg, _ = self._build(mode, tmp_path)
        # Warm
        kg.cypher("MATCH (n:Node) WHERE n.group = 1 AND n.value > 50 RETURN count(n) AS c")
        t0 = time.perf_counter()
        for _ in range(3):
            r = kg.cypher("MATCH (n:Node) WHERE n.group = 1 AND n.value > 50 RETURN count(n) AS c")
        rg_time = (time.perf_counter() - t0) / 3
        assert r[0]["c"] > 0, f"multi-predicate on {mode} returned zero — parity bug?"
        print(f"  multi_predicate_{self.N}_{mode}: {rg_time * 1000:.2f}ms/call")

    @pytest.mark.parametrize("mode", STORAGE_MODES)
    def test_pagerank(self, mode, tmp_path):
        kg, _ = self._build(mode, tmp_path)
        t0 = time.perf_counter()
        scores = kg.pagerank()
        rg_time = time.perf_counter() - t0
        assert len(scores) == self.N, f"pagerank on {mode} returned {len(scores)} != {self.N}"
        print(f"  pagerank_{self.N}_{mode}: {rg_time * 1000:.2f}ms")

    @pytest.mark.parametrize("mode", STORAGE_MODES)
    def test_two_hop(self, mode, tmp_path):
        kg, names = self._build(mode, tmp_path)
        rng = random.Random(3)
        seeds = [rng.choice(names) for _ in range(10)]
        t0 = time.perf_counter()
        for name in seeds:
            kg.cypher(f"MATCH (a:Node {{name: '{name}'}})-[:EDGE*2]->(b:Node) RETURN count(DISTINCT b) AS cnt")
        rg_time = time.perf_counter() - t0
        print(f"  two_hop_10x_{self.N}_{mode}: {rg_time * 1000:.2f}ms")

    def test_cross_mode_result_parity(self, tmp_path):
        """Same query must return the same rows in every storage mode.

        A silent divergence here (e.g. disk returns a subset) is an API
        parity bug — much harder to spot than a crash.
        """
        query = "MATCH (n:Node) WHERE n.group = 1 AND n.value > 50 RETURN count(n) AS c"
        results = {}
        for mode in STORAGE_MODES:
            kg, _ = self._build(mode, tmp_path)
            results[mode] = kg.cypher(query)[0]["c"]
            del kg
        expected = results["memory"]
        for mode, got in results.items():
            assert got == expected, f"storage mode '{mode}' returned {got}, expected {expected} (all modes: {results})"
        print(f"  parity_multi_predicate: all modes agree on {expected} rows")


# ============================================================================
# Footprint: RAM and on-disk size across storage modes + NX
# ============================================================================


class TestFootprint:
    """Report resident memory + serialized graph size at representative scales.

    Memory is approximated by the RSS delta around graph construction — noisy
    but useful for relative comparison. ``psutil`` is required to emit numbers;
    without it we just report wall time and disk size.

    For kglite memory mode we save to a ``.kgl`` file to measure disk size.
    For kglite mapped mode we ``save()`` to a ``.kgl`` (mmap state is backed
    by the save file). For kglite disk mode the graph *is* a directory, so we
    report the directory size as-is. For NX we pickle.
    """

    @pytest.mark.parametrize("n", [1000, 10000, 50000])
    def test_footprint(self, n, tmp_path):
        import gc
        import pickle

        # ---- NetworkX ---------------------------------------------------
        gc.collect()
        rss_before = _rss_bytes()
        nx_g = nx.DiGraph()
        rng = random.Random(42)
        node_names = [f"N{i}" for i in range(n)]
        for i, name in enumerate(node_names):
            nx_g.add_node(name, value=rng.uniform(0, 100), group=rng.randint(1, 5))
        edge_set = set()
        while len(edge_set) < 3 * n:
            s = rng.randint(0, n - 1)
            t = rng.randint(0, n - 1)
            if s != t:
                edge_set.add((s, t))
        for s, t in edge_set:
            nx_g.add_edge(node_names[s], node_names[t])
        rss_after = _rss_bytes()
        nx_rss = (rss_after - rss_before) if rss_before is not None else None
        nx_path = tmp_path / f"nx_{n}.pkl"
        with open(nx_path, "wb") as f:
            pickle.dump(nx_g, f, protocol=pickle.HIGHEST_PROTOCOL)
        nx_disk = nx_path.stat().st_size
        del nx_g
        gc.collect()

        results = []
        results.append(("networkx", nx_rss, nx_disk))

        # ---- kglite storage modes --------------------------------------
        for mode in STORAGE_MODES:
            gc.collect()
            rss_before = _rss_bytes()
            if mode == "disk":
                dpath = tmp_path / f"kg_{mode}_{n}"
                dpath.mkdir(exist_ok=True)
                kg, _ = _build_kg_mode(n, mode, path=dpath)
            else:
                kg, _ = _build_kg_mode(n, mode)
            rss_after = _rss_bytes()
            rss_delta = (rss_after - rss_before) if rss_before is not None else None

            if mode == "disk":
                # The directory IS the graph; no separate save needed.
                disk_size = _dir_size(str(dpath))
            else:
                save_path = tmp_path / f"kg_{mode}_{n}.kgl"
                kg.save(str(save_path))
                disk_size = save_path.stat().st_size

            results.append((f"kglite-{mode}", rss_delta, disk_size))
            del kg
            gc.collect()

        print(f"  footprint_{n}:")
        for label, rss, disk in results:
            print(f"    {label:15s} rss={_fmt_bytes(rss):>8s}  disk={_fmt_bytes(disk):>8s}")


# ============================================================================
# Summary Report (runs last via z-prefix)
# ============================================================================


class TestZZZBenchmarkSummary:
    """Print summary table (class name starts with ZZZ to run last)."""

    def test_print_summary(self):
        """Placeholder to remind users to check output above."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print("Review the output above for detailed timing comparisons.")
        print("Look for 'speedup=NNx' values in the output.")
        print("Target: 10x+ for core algorithms, 5x+ for community detection.")
        print("=" * 70)
