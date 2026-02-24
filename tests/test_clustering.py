"""Tests for CALL cluster() — general-purpose clustering, plus round(x,d) and || operator."""

import pytest
from kglite import KnowledgeGraph


@pytest.fixture
def spatial_graph():
    """Graph with spatial nodes in two clear geographic clusters."""
    g = KnowledgeGraph()

    # Configure spatial
    g.set_spatial("City", location=("lat", "lon"))

    # Cluster 1: Nordic cities (lat ~59-60, lon ~5-11)
    g.cypher("CREATE (:City {name: 'Oslo', lat: 59.91, lon: 10.75})")
    g.cypher("CREATE (:City {name: 'Bergen', lat: 60.39, lon: 5.32})")
    g.cypher("CREATE (:City {name: 'Stavanger', lat: 58.97, lon: 5.73})")

    # Cluster 2: Mediterranean cities (lat ~37-41, lon ~12-24)
    g.cypher("CREATE (:City {name: 'Rome', lat: 41.90, lon: 12.50})")
    g.cypher("CREATE (:City {name: 'Athens', lat: 37.98, lon: 23.73})")
    g.cypher("CREATE (:City {name: 'Naples', lat: 40.85, lon: 14.27})")

    # Outlier
    g.cypher("CREATE (:City {name: 'Reykjavik', lat: 64.13, lon: -21.90})")

    return g


@pytest.fixture
def property_graph():
    """Graph with numeric properties for property-based clustering."""
    g = KnowledgeGraph()

    # Cluster 1: deep, hot wells
    g.cypher("CREATE (:Well {name: 'W1', depth: 4000.0, temperature: 150.0})")
    g.cypher("CREATE (:Well {name: 'W2', depth: 4200.0, temperature: 155.0})")
    g.cypher("CREATE (:Well {name: 'W3', depth: 3800.0, temperature: 145.0})")

    # Cluster 2: shallow, cool wells
    g.cypher("CREATE (:Well {name: 'W4', depth: 500.0, temperature: 30.0})")
    g.cypher("CREATE (:Well {name: 'W5', depth: 600.0, temperature: 35.0})")
    g.cypher("CREATE (:Well {name: 'W6', depth: 450.0, temperature: 28.0})")

    return g


class TestClusterSpatialDbscan:
    """Test CALL cluster() with spatial auto-detection (DBSCAN)."""

    def test_two_clusters_plus_noise(self, spatial_graph):
        # eps=1200km groups Med cities (Rome-Athens ~1050km), min_points=1 for pairs
        result = spatial_graph.cypher("""
            MATCH (c:City)
            CALL cluster({method: 'dbscan', eps: 1200000, min_points: 1})
            YIELD node, cluster
            RETURN node.name AS name, cluster
            ORDER BY cluster, name
        """)
        assert len(result) == 7

        # Group by cluster
        clusters = {}
        for r in result:
            cid = r["cluster"]
            clusters.setdefault(cid, []).append(r["name"])

        # Should have 2 real clusters and possibly noise (-1)
        real_clusters = {k: v for k, v in clusters.items() if k >= 0}
        assert len(real_clusters) == 2, f"Expected 2 clusters, got {real_clusters}"

        # Nordic cities should be in one cluster, Mediterranean in another
        nordic = {"Oslo", "Bergen", "Stavanger"}
        med = {"Rome", "Athens", "Naples"}
        cluster_sets = [set(v) for v in real_clusters.values()]
        assert nordic in cluster_sets, f"Nordic cities not grouped: {cluster_sets}"
        assert med in cluster_sets, f"Mediterranean cities not grouped: {cluster_sets}"

        # Reykjavik should be noise (>1200km from both clusters)
        assert -1 in clusters, "Reykjavik should be noise"
        assert "Reykjavik" in clusters[-1]

    def test_all_one_cluster_with_large_eps(self, spatial_graph):
        result = spatial_graph.cypher("""
            MATCH (c:City)
            CALL cluster({method: 'dbscan', eps: 5000000, min_points: 1})
            YIELD node, cluster
            RETURN cluster, count(*) AS n
        """)
        # With a 5000km eps, everything should be in one cluster
        assert len(result) == 1
        assert result[0]["n"] == 7
        assert result[0]["cluster"] >= 0


class TestClusterSpatialKmeans:
    """Test CALL cluster() with spatial K-means."""

    def test_kmeans_two_groups(self, spatial_graph):
        result = spatial_graph.cypher("""
            MATCH (c:City)
            CALL cluster({method: 'kmeans', k: 2})
            YIELD node, cluster
            RETURN node.name AS name, cluster
        """)
        assert len(result) == 7

        # Group by cluster
        clusters = {}
        for r in result:
            clusters.setdefault(r["cluster"], []).append(r["name"])

        # Nordic cities (+ Reykjavik) should land in one group,
        # Mediterranean in another — k-means has no noise concept
        assert len(clusters) == 2


class TestClusterPropertyDbscan:
    """Test CALL cluster() with explicit numeric properties."""

    def test_two_property_clusters(self, property_graph):
        result = property_graph.cypher("""
            MATCH (w:Well)
            CALL cluster({
                properties: ['depth', 'temperature'],
                method: 'dbscan',
                eps: 0.3,
                min_points: 2,
                normalize: true
            })
            YIELD node, cluster
            RETURN node.name AS name, cluster
            ORDER BY cluster, name
        """)
        assert len(result) == 6

        clusters = {}
        for r in result:
            clusters.setdefault(r["cluster"], []).append(r["name"])

        real = {k: v for k, v in clusters.items() if k >= 0}
        assert len(real) == 2

        deep = {"W1", "W2", "W3"}
        shallow = {"W4", "W5", "W6"}
        cluster_sets = [set(v) for v in real.values()]
        assert deep in cluster_sets
        assert shallow in cluster_sets


class TestClusterPropertyKmeans:
    """Test CALL cluster() with K-means on explicit properties."""

    def test_kmeans_properties(self, property_graph):
        result = property_graph.cypher("""
            MATCH (w:Well)
            CALL cluster({
                properties: ['depth', 'temperature'],
                method: 'kmeans',
                k: 2,
                normalize: true
            })
            YIELD node, cluster
            RETURN node.name AS name, cluster
        """)
        assert len(result) == 6

        clusters = {}
        for r in result:
            clusters.setdefault(r["cluster"], []).append(r["name"])

        assert len(clusters) == 2

        deep = {"W1", "W2", "W3"}
        shallow = {"W4", "W5", "W6"}
        cluster_sets = [set(v) for v in clusters.values()]
        assert deep in cluster_sets
        assert shallow in cluster_sets


class TestClusterWithGeometryCentroid:
    """Test cluster() spatial mode with WKT geometry centroid fallback."""

    def test_geometry_centroid_clustering(self):
        g = KnowledgeGraph()
        g.set_spatial("Area", geometry="wkt")

        # Two areas near each other (Nordic) and one far away (Mediterranean)
        g.cypher("""CREATE (:Area {name: 'NorthSea', wkt: 'POLYGON((4 58, 8 58, 8 62, 4 62, 4 58))'})""")
        g.cypher("""CREATE (:Area {name: 'NorwegianSea', wkt: 'POLYGON((3 62, 7 62, 7 66, 3 66, 3 62))'})""")
        g.cypher("""CREATE (:Area {name: 'Mediterranean', wkt: 'POLYGON((10 35, 25 35, 25 42, 10 42, 10 35))'})""")

        result = g.cypher("""
            MATCH (a:Area)
            CALL cluster({method: 'dbscan', eps: 1000000, min_points: 1})
            YIELD node, cluster
            RETURN node.name AS name, cluster
            ORDER BY cluster
        """)
        assert len(result) == 3

        clusters = {}
        for r in result:
            clusters.setdefault(r["cluster"], []).append(r["name"])

        # NorthSea and NorwegianSea should cluster together
        real = {k: v for k, v in clusters.items() if k >= 0}
        assert len(real) >= 1
        north_cluster = None
        for cid, names in real.items():
            if "NorthSea" in names:
                north_cluster = cid
                break
        assert north_cluster is not None
        assert "NorwegianSea" in real[north_cluster]


class TestClusterErrors:
    """Test error handling for cluster()."""

    def test_no_match_gives_error(self):
        g = KnowledgeGraph()
        with pytest.raises(Exception, match="requires a preceding MATCH"):
            g.cypher("""
                CALL cluster({method: 'dbscan'}) YIELD node, cluster
                RETURN node, cluster
            """)

    def test_unknown_method(self, spatial_graph):
        with pytest.raises(Exception, match="Unknown clustering method"):
            spatial_graph.cypher("""
                MATCH (c:City)
                CALL cluster({method: 'spectral'})
                YIELD node, cluster
                RETURN node, cluster
            """)

    def test_no_spatial_config_gives_error(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:Thing {name: 'A', x: 1.0})")
        g.cypher("CREATE (:Thing {name: 'B', x: 2.0})")
        with pytest.raises(Exception, match="spatial"):
            g.cypher("""
                MATCH (t:Thing)
                CALL cluster({method: 'dbscan'})
                YIELD node, cluster
                RETURN node, cluster
            """)


class TestClusterAggregation:
    """Test cluster() combined with aggregation (the dream query pattern)."""

    def test_cluster_with_count_and_collect(self, spatial_graph):
        result = spatial_graph.cypher("""
            MATCH (c:City)
            CALL cluster({method: 'dbscan', eps: 1200000, min_points: 1})
            YIELD node, cluster
            RETURN cluster, count(*) AS n, collect(node.name) AS cities
            ORDER BY n DESC
        """)
        assert len(result) >= 2  # at least 2 groups (real clusters + noise)

        # The two real clusters should each have 3 cities
        real = [r for r in result if r["cluster"] >= 0]
        assert len(real) == 2
        for r in real:
            assert r["n"] == 3


class TestRoundPrecision:
    """Test round(x, decimals) precision argument."""

    def test_round_two_decimals(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {val: 3.14159})")
        result = g.cypher("MATCH (n:N) RETURN round(n.val, 2) AS r")
        assert result[0]["r"] == 3.14

    def test_round_zero_decimals(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {val: 3.14159})")
        result = g.cypher("MATCH (n:N) RETURN round(n.val, 0) AS r")
        assert result[0]["r"] == 3.0

    def test_round_backward_compat(self):
        """round() with no precision argument should still work."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {val: 3.7})")
        result = g.cypher("MATCH (n:N) RETURN round(n.val) AS r")
        assert result[0]["r"] == 4.0

    def test_round_null_propagation(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {val: 3.14})")
        result = g.cypher("MATCH (n:N) RETURN round(n.missing, 2) AS r")
        assert result[0]["r"] is None


class TestStringConcat:
    """Test || string concatenation operator."""

    def test_basic_concat(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {a: 'hello', b: 'world'})")
        result = g.cypher("MATCH (n:N) RETURN n.a || ' ' || n.b AS msg")
        assert result[0]["msg"] == "hello world"

    def test_concat_with_number(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {name: 'block', num: 35})")
        result = g.cypher("MATCH (n:N) RETURN n.name || '-' || n.num AS label")
        assert result[0]["label"] == "block-35"

    def test_concat_null_propagation(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {name: 'test'})")
        result = g.cypher("MATCH (n:N) RETURN n.name || n.missing AS r")
        assert result[0]["r"] is None

    def test_concat_in_return(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:N {first: 'A', last: 'B'})")
        result = g.cypher("MATCH (n:N) RETURN n.first || n.last AS code")
        assert result[0]["code"] == "AB"


class TestDescribeCypher:
    """Test describe(cypher=True) returns Cypher language reference."""

    def test_describe_cypher_false_no_cypher_block(self):
        g = KnowledgeGraph()
        desc = g.describe()
        # Should have the hint but not the full <cypher> block
        assert "cypher" in desc.lower()

    def test_describe_cypher_true_has_reference(self):
        g = KnowledgeGraph()
        desc = g.describe(cypher=True)
        assert "<cypher>" in desc
        assert "<clauses>" in desc
        assert "<operators>" in desc
        assert "<functions>" in desc
        assert "coalesce" in desc
        assert "CONTAINS" in desc
        assert "||" in desc

    def test_describe_cluster_in_algorithms(self):
        g = KnowledgeGraph()
        desc = g.describe()
        assert "cluster" in desc.lower()
