"""Tests for shapely/geopandas integration with spatial operations."""

import pytest
import pandas as pd

shapely = pytest.importorskip("shapely")
gpd = pytest.importorskip("geopandas")

from shapely.geometry import Point, Polygon, box

from kglite import KnowledgeGraph


@pytest.fixture
def geo_graph():
    """Graph with lat/lon coordinate data."""
    graph = KnowledgeGraph()
    df = pd.DataFrame({
        'id': list(range(1, 11)),
        'name': [f'Location_{i}' for i in range(1, 11)],
        'latitude': [58.0 + i * 0.5 for i in range(10)],
        'longitude': [2.0 + i * 0.3 for i in range(10)],
    })
    graph.add_nodes(df, 'Location', 'id', 'name')
    return graph


@pytest.fixture
def wkt_graph():
    """Graph with WKT geometry data."""
    graph = KnowledgeGraph()
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['FieldA', 'FieldB', 'FieldC'],
        'geometry': [
            'POLYGON((1 58, 5 58, 5 62, 1 62, 1 58))',
            'POLYGON((3 60, 7 60, 7 64, 3 64, 3 60))',
            'POLYGON((10 50, 15 50, 15 55, 10 55, 10 50))',
        ],
    })
    graph.add_nodes(df, 'Field', 'id', 'name')
    return graph


# ── Shapely Input ─────────────────────────────────────────────────────


class TestShapelyInput:
    def test_intersects_geometry_with_shapely_polygon(self, wkt_graph):
        poly = Polygon([(2, 59), (6, 59), (6, 63), (2, 63), (2, 59)])
        result = wkt_graph.type_filter('Field').intersects_geometry(poly)
        assert result.node_count() >= 1

    def test_intersects_geometry_wkt_string_still_works(self, wkt_graph):
        """Backwards compatibility: plain WKT strings still work."""
        result = wkt_graph.type_filter('Field').intersects_geometry(
            'POLYGON((2 59, 6 59, 6 63, 2 63, 2 59))',
        )
        assert result.node_count() >= 1

    def test_intersects_geometry_with_shapely_box(self, wkt_graph):
        bbox = box(2, 59, 6, 63)
        result = wkt_graph.type_filter('Field').intersects_geometry(bbox)
        assert result.node_count() >= 1

    def test_wkt_centroid_with_shapely_polygon(self, wkt_graph):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        centroid = wkt_graph.wkt_centroid(poly)
        assert abs(centroid['latitude'] - 5.0) < 0.01
        assert abs(centroid['longitude'] - 5.0) < 0.01

    def test_wkt_centroid_with_shapely_point(self, wkt_graph):
        pt = Point(5.0, 10.0)  # (x=lon, y=lat)
        centroid = wkt_graph.wkt_centroid(pt)
        assert abs(centroid['latitude'] - 10.0) < 0.01
        assert abs(centroid['longitude'] - 5.0) < 0.01

    def test_intersects_bad_type_raises(self, wkt_graph):
        with pytest.raises(TypeError, match="WKT string or a geometry object"):
            wkt_graph.type_filter('Field').intersects_geometry(12345)

    def test_wkt_centroid_bad_type_raises(self, wkt_graph):
        with pytest.raises(TypeError, match="WKT string or a geometry object"):
            wkt_graph.wkt_centroid(42)


# ── Shapely Output ────────────────────────────────────────────────────


class TestShapelyOutput:
    def test_wkt_centroid_as_shapely(self, wkt_graph):
        result = wkt_graph.wkt_centroid(
            'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))', as_shapely=True,
        )
        assert isinstance(result, Point)
        assert abs(result.x - 5.0) < 0.01  # lon = x
        assert abs(result.y - 5.0) < 0.01  # lat = y

    def test_wkt_centroid_shapely_input_and_output(self, wkt_graph):
        """Shapely in, shapely out."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        result = wkt_graph.wkt_centroid(poly, as_shapely=True)
        assert isinstance(result, Point)
        assert abs(result.x - 5.0) < 0.01
        assert abs(result.y - 5.0) < 0.01

    def test_get_centroid_as_shapely(self, geo_graph):
        result = geo_graph.type_filter('Location').get_centroid(
            lat_field='latitude', lon_field='longitude', as_shapely=True,
        )
        assert isinstance(result, Point)

    def test_get_centroid_as_shapely_empty(self):
        graph = KnowledgeGraph()
        result = graph.get_centroid(as_shapely=True)
        assert result is None

    def test_get_bounds_as_shapely(self, geo_graph):
        result = geo_graph.type_filter('Location').get_bounds(
            lat_field='latitude', lon_field='longitude', as_shapely=True,
        )
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_get_bounds_as_shapely_empty(self):
        graph = KnowledgeGraph()
        result = graph.get_bounds(as_shapely=True)
        assert result is None

    def test_get_bounds_default_is_dict(self, geo_graph):
        """Default behavior unchanged — returns dict."""
        result = geo_graph.type_filter('Location').get_bounds(
            lat_field='latitude', lon_field='longitude',
        )
        assert isinstance(result, dict)
        assert 'min_lat' in result


# ── ResultView.to_gdf() ──────────────────────────────────────────────


class TestToGdf:
    def test_to_gdf_from_get_nodes(self, wkt_graph):
        rv = wkt_graph.type_filter('Field').get_nodes()
        gdf = rv.to_gdf(geometry_column='geometry')
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3
        assert gdf.geometry.name == 'geometry'
        # Each geometry should be a shapely Polygon
        for geom in gdf.geometry:
            assert isinstance(geom, Polygon)

    def test_to_gdf_with_crs(self, wkt_graph):
        rv = wkt_graph.type_filter('Field').get_nodes()
        gdf = rv.to_gdf(geometry_column='geometry', crs='EPSG:4326')
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_to_gdf_from_cypher(self, wkt_graph):
        rv = wkt_graph.cypher(
            "MATCH (n:Field) RETURN n.name AS name, n.geometry AS geometry"
        )
        gdf = rv.to_gdf(geometry_column='geometry')
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3

    def test_to_gdf_missing_column_raises(self, wkt_graph):
        rv = wkt_graph.type_filter('Field').get_nodes()
        with pytest.raises(KeyError):
            rv.to_gdf(geometry_column='nonexistent')
