"""Tests for Cypher spatial functions: point(), distance(), contains(), etc."""
import pytest
import pandas as pd
import kglite


@pytest.fixture
def geo_graph():
    """Graph with cities that have lat/lon coordinates and areas with WKT polygons."""
    g = kglite.KnowledgeGraph()

    cities = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "title": ["Oslo", "Bergen", "Stavanger", "Trondheim", "Drammen"],
            "latitude": [59.91, 60.39, 58.97, 63.43, 59.74],
            "longitude": [10.75, 5.32, 5.73, 10.40, 10.20],
        }
    )
    g.add_nodes(cities, "City", "id", "title")

    # Add connections between nearby cities
    edges = pd.DataFrame(
        {
            "source": [1, 1, 2, 4],
            "target": [5, 2, 3, 1],
            "type": ["NEAR", "ROUTE", "ROUTE", "ROUTE"],
        }
    )
    g.add_connections(edges, "NEAR", "City", "source", "City", "target")

    # Add areas with WKT polygons
    areas = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "title": ["SouthNorway", "WestNorway", "SmallBox"],
            "geometry": [
                "POLYGON((5 58, 12 58, 12 62, 5 62, 5 58))",
                "POLYGON((4 58, 7 58, 7 62, 4 62, 4 58))",
                "POLYGON((10 59, 11 59, 11 60, 10 60, 10 59))",
            ],
        }
    )
    g.add_nodes(areas, "Area", "id", "title")

    return g


# ============================================================================
# point() constructor
# ============================================================================


class TestPoint:
    def test_point_constructor(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' "
            "RETURN point(59.91, 10.75) AS p"
        ).to_list()
        assert len(rows) == 1
        p = rows[0]["p"]
        assert isinstance(p, dict)
        assert p["latitude"] == 59.91
        assert p["longitude"] == 10.75

    def test_point_from_property(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' "
            "RETURN point(n.latitude, n.longitude) AS p"
        ).to_list()
        assert len(rows) == 1
        p = rows[0]["p"]
        assert abs(p["latitude"] - 59.91) < 0.01
        assert abs(p["longitude"] - 10.75) < 0.01

    def test_point_equality(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' "
            "WITH point(n.latitude, n.longitude) AS p "
            "WHERE p = point(59.91, 10.75) "
            "RETURN p"
        ).to_list()
        assert len(rows) == 1

    def test_point_wrong_args(self, geo_graph):
        with pytest.raises(Exception, match="point.*requires 2"):
            geo_graph.cypher("MATCH (n:City) RETURN point(1.0)")

    def test_point_non_numeric(self, geo_graph):
        with pytest.raises(Exception, match="numeric"):
            geo_graph.cypher("MATCH (n:City) RETURN point('hello', 10.0)")


# ============================================================================
# distance()
# ============================================================================


class TestDistance:
    def test_distance_two_points(self, geo_graph):
        """Oslo to Bergen should be ~305 km (~305,000 m)."""
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' "
            "RETURN distance(point(59.91, 10.75), point(60.39, 5.32)) AS d"
        ).to_list()
        d = rows[0]["d"]
        assert 300_000 < d < 310_000

    def test_distance_four_args(self, geo_graph):
        """4-arg shorthand should give same result."""
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' "
            "RETURN distance(59.91, 10.75, 60.39, 5.32) AS d"
        ).to_list()
        d = rows[0]["d"]
        assert 300_000 < d < 310_000

    def test_distance_same_point(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' "
            "RETURN distance(point(60.0, 10.0), point(60.0, 10.0)) AS d"
        ).to_list()
        assert rows[0]["d"] == 0.0

    def test_distance_in_where(self, geo_graph):
        """Filter cities within 50km of Oslo."""
        rows = geo_graph.cypher(
            "MATCH (n:City) "
            "WHERE distance(point(n.latitude, n.longitude), point(59.91, 10.75)) < 50000.0 "
            "RETURN n.title AS name ORDER BY name"
        ).to_list()
        names = [r["name"] for r in rows]
        assert "Oslo" in names
        assert "Drammen" in names  # ~40km from Oslo
        assert "Bergen" not in names  # ~305km away

    def test_distance_in_return(self, geo_graph):
        """Return distance values for each city."""
        rows = geo_graph.cypher(
            "MATCH (n:City) "
            "RETURN n.title AS name, "
            "distance(point(n.latitude, n.longitude), point(59.91, 10.75)) AS dist "
            "ORDER BY dist"
        ).to_list()
        assert rows[0]["name"] == "Oslo"
        assert rows[0]["dist"] < 1000.0

    def test_distance_with_traversal(self, geo_graph):
        """Combine graph traversal with spatial filtering."""
        rows = geo_graph.cypher(
            "MATCH (a:City)-[]->(b:City) "
            "RETURN a.title AS a_name, b.title AS b_name, "
            "distance(point(a.latitude, a.longitude), "
            "point(b.latitude, b.longitude)) AS dist "
            "ORDER BY dist"
        ).to_list()
        assert len(rows) >= 1
        # Closest pair should be Oslo-Drammen (~40km)
        assert rows[0]["dist"] < 50_000.0

    def test_distance_wrong_arg_count(self, geo_graph):
        with pytest.raises(Exception, match="distance.*requires"):
            geo_graph.cypher(
                "MATCH (n:City) RETURN distance(point(1,2), point(3,4), point(5,6))"
            )


# ============================================================================
# contains()
# ============================================================================


class TestContains:
    def test_contains_with_point(self, geo_graph):
        """Oslo (59.91, 10.75) is inside SouthNorway polygon."""
        rows = geo_graph.cypher(
            "MATCH (a:Area) "
            "WHERE a.title = 'SouthNorway' "
            "AND contains(a.geometry, point(59.91, 10.75)) "
            "RETURN a.title AS title"
        ).to_list()
        assert len(rows) == 1
        assert rows[0]["title"] == "SouthNorway"

    def test_contains_with_wkt_and_point(self, geo_graph):
        """contains() with WKT string property and point literal."""
        rows = geo_graph.cypher(
            "MATCH (a:Area) "
            "WHERE contains(a.geometry, point(59.91, 10.75)) "
            "RETURN a.title AS title ORDER BY title"
        ).to_list()
        titles = [r["title"] for r in rows]
        assert "SouthNorway" in titles
        assert "SmallBox" in titles

    def test_contains_false(self, geo_graph):
        """Trondheim (63.43, 10.40) is outside SouthNorway (max lat 62)."""
        rows = geo_graph.cypher(
            "MATCH (a:Area) "
            "WHERE a.title = 'SouthNorway' "
            "AND contains(a.geometry, point(63.43, 10.40)) "
            "RETURN a.title"
        ).to_list()
        assert len(rows) == 0

    def test_contains_with_node_properties(self, geo_graph):
        """Find which areas contain each city."""
        rows = geo_graph.cypher(
            "MATCH (c:City), (a:Area) "
            "WHERE contains(a.geometry, point(c.latitude, c.longitude)) "
            "RETURN c.title AS city, a.title AS area ORDER BY city, area"
        ).to_list()
        assert len(rows) >= 1
        oslo_areas = [r["area"] for r in rows if r["city"] == "Oslo"]
        assert "SouthNorway" in oslo_areas


# ============================================================================
# intersects()
# ============================================================================


class TestIntersects:
    def test_intersects_overlapping(self, geo_graph):
        """SouthNorway and WestNorway polygons overlap."""
        rows = geo_graph.cypher(
            "MATCH (a:Area), (b:Area) "
            "WHERE a.title = 'SouthNorway' AND b.title = 'WestNorway' "
            "AND intersects(a.geometry, b.geometry) "
            "RETURN a.title"
        ).to_list()
        assert len(rows) == 1

    def test_intersects_non_overlapping(self, geo_graph):
        """SmallBox (lon 10-11) and WestNorway (lon 4-7) don't overlap."""
        rows = geo_graph.cypher(
            "MATCH (a:Area), (b:Area) "
            "WHERE a.title = 'SmallBox' AND b.title = 'WestNorway' "
            "AND intersects(a.geometry, b.geometry) "
            "RETURN a.title"
        ).to_list()
        assert len(rows) == 0


# ============================================================================
# centroid()
# ============================================================================


class TestCentroid:
    def test_centroid_polygon(self, geo_graph):
        """Centroid of SmallBox should be (59.5, 10.5)."""
        rows = geo_graph.cypher(
            "MATCH (a:Area) WHERE a.title = 'SmallBox' "
            "RETURN centroid(a.geometry) AS c"
        ).to_list()
        c = rows[0]["c"]
        assert isinstance(c, dict)
        assert abs(c["latitude"] - 59.5) < 0.01
        assert abs(c["longitude"] - 10.5) < 0.01

    def test_centroid_in_distance(self, geo_graph):
        """Use centroid in distance calculation."""
        rows = geo_graph.cypher(
            "MATCH (a:Area) "
            "RETURN a.title AS title, "
            "distance(centroid(a.geometry), point(59.91, 10.75)) AS dist "
            "ORDER BY dist"
        ).to_list()
        assert len(rows) == 3


# ============================================================================
# latitude() / longitude()
# ============================================================================


class TestAccessors:
    def test_latitude(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' "
            "RETURN latitude(point(59.91, 10.75)) AS lat"
        ).to_list()
        assert abs(rows[0]["lat"] - 59.91) < 0.001

    def test_longitude(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) WHERE n.title = 'Oslo' "
            "RETURN longitude(point(59.91, 10.75)) AS lon"
        ).to_list()
        assert abs(rows[0]["lon"] - 10.75) < 0.001

    def test_accessors_with_centroid(self, geo_graph):
        """Extract lat/lon from centroid."""
        rows = geo_graph.cypher(
            "MATCH (a:Area) WHERE a.title = 'SmallBox' "
            "RETURN latitude(centroid(a.geometry)) AS lat, "
            "longitude(centroid(a.geometry)) AS lon"
        ).to_list()
        assert abs(rows[0]["lat"] - 59.5) < 0.01
        assert abs(rows[0]["lon"] - 10.5) < 0.01


# ============================================================================
# Spatial aggregation
# ============================================================================


class TestSpatialAggregation:
    def test_avg_distance(self, geo_graph):
        """Average distance from Oslo to all cities."""
        rows = geo_graph.cypher(
            "MATCH (n:City) "
            "RETURN avg(distance(point(n.latitude, n.longitude), point(59.91, 10.75))) AS avg_dist"
        ).to_list()
        assert rows[0]["avg_dist"] > 0

    def test_min_max_distance(self, geo_graph):
        rows = geo_graph.cypher(
            "MATCH (n:City) "
            "RETURN min(distance(point(n.latitude, n.longitude), point(59.91, 10.75))) AS min_d, "
            "max(distance(point(n.latitude, n.longitude), point(59.91, 10.75))) AS max_d"
        ).to_list()
        assert rows[0]["min_d"] < 1000.0  # Oslo to itself ~ 0
        assert rows[0]["max_d"] > 300_000  # Farthest city > 300km
