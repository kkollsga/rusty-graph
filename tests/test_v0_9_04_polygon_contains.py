"""0.9.0 gate item §4 — polygon-vs-polygon contains() returns 0
incorrectly, and MULTIPOLYGON contains() emits duplicate pairs.

Documented workaround: route every spatial join through `centroid(p)`
PIP. The Sodir prospect-graph enhance script is built around this
workaround; the agent flagged it as the gating item for "polygon
analytics without a thousand `centroid()` calls".

These tests are xfail-strict — see test_v0_9_05_integer_division.py
header for the workflow.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

NOT_IMPLEMENTED_POLY = "0.9.0 §4 — polygon-vs-polygon contains() returns 0; flip when fixed."
NOT_IMPLEMENTED_MULTI = "0.9.0 §4 — MULTIPOLYGON contains() emits one row per matched component; flip when deduped."


# ---------------------------------------------------------------------------
# Polygon-inside-polygon
# ---------------------------------------------------------------------------


@pytest.fixture
def nested_polygons_graph():
    """outer = 10x10 square; inner = 2x2 square fully inside outer."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "outer",
                "wkt_geometry": "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
            },
            {
                "id": 2,
                "title": "inner",
                "wkt_geometry": "POLYGON((3 3, 5 3, 5 5, 3 5, 3 3))",
            },
        ]
    )
    g.add_nodes(df, "Region", "id", "title", column_types={"wkt_geometry": "geometry"})
    return g


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED_POLY)
def test_polygon_contains_polygon_inside(nested_polygons_graph):
    rows = list(
        nested_polygons_graph.cypher(
            """
            MATCH (a:Region), (b:Region)
            WHERE a.title = 'outer' AND b.title = 'inner' AND contains(a, b)
            RETURN count(*) AS n
            """
        )
    )
    assert rows[0]["n"] == 1


def test_polygon_contains_polygon_outside_negative(nested_polygons_graph):
    """Already passes — inner polygon does not contain outer polygon
    (correctly returns 0). Locks the negative-direction behavior so
    the §4 fix doesn't accidentally flip false positives.
    """
    rows = list(
        nested_polygons_graph.cypher(
            """
            MATCH (a:Region), (b:Region)
            WHERE a.title = 'inner' AND b.title = 'outer' AND contains(a, b)
            RETURN count(*) AS n
            """
        )
    )
    assert rows[0]["n"] == 0


def test_polygon_contains_partial_overlap_returns_zero():
    """Already passes — partial overlap correctly returns 0. Locks
    the boundary case so the positive-direction fix doesn't
    over-fire."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "left",
                "wkt_geometry": "POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))",
            },
            {
                "id": 2,
                "title": "right",
                "wkt_geometry": "POLYGON((3 0, 8 0, 8 5, 3 5, 3 0))",
            },
        ]
    )
    g.add_nodes(df, "Region", "id", "title", column_types={"wkt_geometry": "geometry"})
    rows = list(
        g.cypher(
            """
            MATCH (a:Region), (b:Region)
            WHERE a.id <> b.id AND contains(a, b)
            RETURN count(*) AS n
            """
        )
    )
    assert rows[0]["n"] == 0


# ---------------------------------------------------------------------------
# MULTIPOLYGON dedupe
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=NOT_IMPLEMENTED_MULTI)
def test_multipolygon_contains_point_no_duplicates():
    """A MULTIPOLYGON with two components, only one of which contains
    the point. Must emit exactly one (a, b) row, not multiple.
    """
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "multi",
                "wkt_geometry": ("MULTIPOLYGON(((0 0, 10 0, 10 10, 0 10, 0 0)),((20 0, 30 0, 30 10, 20 10, 20 0)))"),
            },
        ]
    )
    g.add_nodes(df, "Region", "id", "title", column_types={"wkt_geometry": "geometry"})

    df_pt = pd.DataFrame([{"id": 100, "title": "point", "wkt_geometry": "POINT(5 5)"}])
    g.add_nodes(df_pt, "Site", "id", "title", column_types={"wkt_geometry": "geometry"})

    rows = list(
        g.cypher(
            """
            MATCH (a:Region), (b:Site)
            WHERE contains(a, b)
            RETURN count(*) AS n
            """
        )
    )
    # Expected 1, not 2 (each MULTIPOLYGON component-match shouldn't fire a separate row).
    assert rows[0]["n"] == 1
