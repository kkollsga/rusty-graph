"""Tests for Cypher ts_*() timeseries functions with date-string syntax."""

import pytest
import pandas as pd
import kglite


@pytest.fixture
def ts_graph():
    """Graph with two fields, each with monthly timeseries data."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({"id": [1, 2], "title": ["TROLL", "EKOFISK"]})
    g.add_nodes(df, "Field", "id", "title")

    g.set_timeseries(
        "Field", resolution="month", channels=["oil", "gas"],
        units={"oil": "MSm3", "gas": "BSm3"}, bin_type="total",
    )

    # TROLL: 2019-12 through 2021-01
    g.set_time_index(
        1,
        [[2019, 12], [2020, 1], [2020, 2], [2020, 3], [2021, 1]],
    )
    g.add_ts_channel(1, "oil", [0.9, 1.0, 1.5, 2.0, 3.0])
    g.add_ts_channel(1, "gas", [0.1, 0.2, 0.3, 0.4, 0.5])

    # EKOFISK: 2020-01 through 2020-03
    g.set_time_index(2, [[2020, 1], [2020, 2], [2020, 3]])
    g.add_ts_channel(2, "oil", [0.5, 0.6, 0.7])

    return g


# ── ts_at ─────────────────────────────────────────────────────────────────


def test_ts_at_exact(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_at(f.oil, '2020-2') AS val"
    )
    assert result[0]["val"] == 1.5


def test_ts_at_missing_key(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_at(f.oil, '2022-1') AS val"
    )
    assert result[0]["val"] is None


def test_ts_at_wrong_depth(ts_graph):
    """ts_at on month data with year-only key should error."""
    with pytest.raises(Exception, match="Exact lookup requires"):
        ts_graph.cypher(
            "MATCH (f:Field {title: 'TROLL'}) RETURN ts_at(f.oil, '2020') AS val"
        )


# ── ts_sum ────────────────────────────────────────────────────────────────


def test_ts_sum_single_year(ts_graph):
    """ts_sum(f.oil, '2020') = sum of all months in 2020."""
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_sum(f.oil, '2020') AS val"
    )
    assert abs(result[0]["val"] - 4.5) < 1e-10  # 1.0 + 1.5 + 2.0


def test_ts_sum_range(ts_graph):
    """ts_sum(f.oil, '2020', '2021') = sum from 2020 through 2021."""
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_sum(f.oil, '2020', '2021') AS val"
    )
    assert abs(result[0]["val"] - 7.5) < 1e-10  # 1.0 + 1.5 + 2.0 + 3.0


def test_ts_sum_month_range(ts_graph):
    """ts_sum(f.oil, '2020-1', '2020-2') = sum of Jan and Feb."""
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_sum(f.oil, '2020-1', '2020-2') AS val"
    )
    assert abs(result[0]["val"] - 2.5) < 1e-10  # 1.0 + 1.5


def test_ts_sum_all(ts_graph):
    """ts_sum(f.oil) = sum of entire series."""
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_sum(f.oil) AS val"
    )
    assert abs(result[0]["val"] - 8.4) < 1e-10  # 0.9 + 1.0 + 1.5 + 2.0 + 3.0


# ── ts_avg ────────────────────────────────────────────────────────────────


def test_ts_avg_single_year(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_avg(f.oil, '2020') AS val"
    )
    assert abs(result[0]["val"] - 1.5) < 1e-10  # (1.0 + 1.5 + 2.0) / 3


# ── ts_min / ts_max ──────────────────────────────────────────────────────


def test_ts_min(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_min(f.oil, '2020') AS val"
    )
    assert result[0]["val"] == 1.0


def test_ts_max(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_max(f.oil, '2020') AS val"
    )
    assert result[0]["val"] == 2.0


# ── ts_count ──────────────────────────────────────────────────────────────


def test_ts_count(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_count(f.oil) AS val"
    )
    assert result[0]["val"] == 5


def test_ts_count_with_nan():
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({"id": [1], "title": ["S"]})
    g.add_nodes(df, "Sensor", "id", "title")
    g.set_timeseries("Sensor", resolution="year", channels=["temp"])
    g.set_time_index(1, [[1], [2], [3]])
    g.add_ts_channel(1, "temp", [1.0, float("nan"), 3.0])
    result = g.cypher("MATCH (s:Sensor) RETURN ts_count(s.temp) AS val")
    assert result[0]["val"] == 2


# ── ts_first / ts_last ───────────────────────────────────────────────────


def test_ts_first(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_first(f.oil) AS val"
    )
    assert result[0]["val"] == 0.9


def test_ts_last(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_last(f.oil) AS val"
    )
    assert result[0]["val"] == 3.0


# ── ts_delta ──────────────────────────────────────────────────────────────


def test_ts_delta(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_delta(f.oil, '2019', '2021') AS val"
    )
    # value at first entry in [2021,...] - first entry in [2019,...] = 3.0 - 0.9 = 2.1
    assert abs(result[0]["val"] - 2.1) < 1e-10


def test_ts_delta_exact_months(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_delta(f.oil, '2020-1', '2020-3') AS val"
    )
    # value at [2020,3] - value at [2020,1] = 2.0 - 1.0 = 1.0
    assert abs(result[0]["val"] - 1.0) < 1e-10


# ── ts_series ─────────────────────────────────────────────────────────────


def test_ts_series(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'EKOFISK'}) RETURN ts_series(f.oil) AS val"
    )
    series = result[0]["val"]  # auto-parsed from JSON to native Python list
    assert len(series) == 3
    assert series[0]["time"] == [2020, 1]
    assert series[0]["value"] == 0.5
    assert series[2]["value"] == 0.7


def test_ts_series_with_range(ts_graph):
    result = ts_graph.cypher(
        "MATCH (f:Field {title: 'TROLL'}) RETURN ts_series(f.oil, '2020', '2020') AS val"
    )
    series = result[0]["val"]  # auto-parsed from JSON to native Python list
    assert len(series) == 3  # 3 months in 2020


# ── Precision validation ─────────────────────────────────────────────────


def test_day_query_on_month_data_errors(ts_graph):
    """Querying with day precision on month-resolution data should error."""
    with pytest.raises(Exception, match="exceeds data resolution"):
        ts_graph.cypher(
            "MATCH (f:Field {title: 'TROLL'}) RETURN ts_sum(f.oil, '2020-2-15') AS val"
        )


def test_day_range_on_month_data_errors(ts_graph):
    """Day-precision range on month data should error."""
    with pytest.raises(Exception, match="exceeds data resolution"):
        ts_graph.cypher(
            "MATCH (f:Field) RETURN ts_sum(f.oil, '2020-1-1', '2020-3-15') AS val"
        )


# ── WHERE clause filtering ───────────────────────────────────────────────


def test_where_with_ts_function(ts_graph):
    result = ts_graph.cypher("""
        MATCH (f:Field)
        WHERE ts_sum(f.oil, '2020') > 2.0
        RETURN f.title AS title
    """)
    titles = [r["title"] for r in result]
    assert "TROLL" in titles  # ts_sum = 4.5
    assert "EKOFISK" not in titles or len(titles) == 2  # ts_sum = 1.8


# ── ORDER BY with ts_* ───────────────────────────────────────────────────


def test_order_by_ts_function(ts_graph):
    result = ts_graph.cypher("""
        MATCH (f:Field)
        RETURN f.title AS title, ts_sum(f.oil, '2020') AS prod
        ORDER BY prod DESC
    """)
    assert result[0]["title"] == "TROLL"  # 4.5 > 1.8
    assert result[1]["title"] == "EKOFISK"


# ── Error cases ───────────────────────────────────────────────────────────


def test_ts_missing_channel(ts_graph):
    with pytest.raises(Exception, match="water"):
        ts_graph.cypher(
            "MATCH (f:Field {title: 'TROLL'}) RETURN ts_sum(f.water)"
        )


def test_ts_no_timeseries_data():
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({"id": [1], "title": ["X"]})
    g.add_nodes(df, "Node", "id", "title")
    with pytest.raises(Exception, match="no timeseries"):
        g.cypher("MATCH (n:Node) RETURN ts_sum(n.value)")


# ── Multi-channel ─────────────────────────────────────────────────────────


def test_multi_channel_query(ts_graph):
    result = ts_graph.cypher("""
        MATCH (f:Field {title: 'TROLL'})
        RETURN ts_sum(f.oil, '2020') AS oil, ts_sum(f.gas, '2020') AS gas
    """)
    assert abs(result[0]["oil"] - 4.5) < 1e-10
    assert abs(result[0]["gas"] - 0.9) < 1e-10  # 0.2 + 0.3 + 0.4


# ── NaN handling ──────────────────────────────────────────────────────────


def test_nan_skipped_in_sum():
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({"id": [1], "title": ["S"]})
    g.add_nodes(df, "Sensor", "id", "title")
    g.set_timeseries("Sensor", resolution="year", channels=["temp"])
    g.set_time_index(1, [[1], [2], [3]])
    g.add_ts_channel(1, "temp", [1.0, float("nan"), 3.0])
    result = g.cypher("MATCH (s:Sensor) RETURN ts_sum(s.temp) AS val")
    assert abs(result[0]["val"] - 4.0) < 1e-10


# ── Real-world example queries ──────────────────────────────────────────
# Tests for the example use-cases discussed in design review.


@pytest.fixture
def production_graph():
    """Graph with TROLL (24 months: 2019-01..2020-12) and EKOFISK (12 months: 2020-01..2020-12)."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({"id": [1, 2], "title": ["TROLL", "EKOFISK"]})
    g.add_nodes(df, "Field", "id", "title")
    g.set_timeseries(
        "Field", resolution="month", channels=["oil", "gas"],
        units={"oil": "MSm3", "gas": "BSm3"}, bin_type="total",
    )

    # TROLL: 2019-01 through 2020-12 (24 months)
    troll_keys = [[y, m] for y in [2019, 2020] for m in range(1, 13)]
    # Oil: ramp from 1.0 to 2.4 over 24 months
    troll_oil = [1.0 + 0.06 * i for i in range(24)]
    g.set_time_index(1, troll_keys)
    g.add_ts_channel(1, "oil", troll_oil)
    g.add_ts_channel(1, "gas", [v * 0.1 for v in troll_oil])

    # EKOFISK: 2020-01 through 2020-12 (12 months)
    eko_keys = [[2020, m] for m in range(1, 13)]
    eko_oil = [0.5 + 0.02 * i for i in range(12)]
    g.set_time_index(2, eko_keys)
    g.add_ts_channel(2, "oil", eko_oil)

    return g


def test_all_fields_in_specific_month(production_graph):
    """Production of all fields in Feb 2020."""
    result = production_graph.cypher("""
        MATCH (f:Field)
        RETURN f.title AS name, ts_at(f.oil, '2020-2') AS oil_feb
        ORDER BY oil_feb DESC
    """)
    assert len(result) == 2
    # Both fields should have a Feb 2020 value
    assert result[0]["oil_feb"] is not None
    assert result[1]["oil_feb"] is not None
    # TROLL > EKOFISK
    assert result[0]["name"] == "TROLL"


def test_field_production_between_months(production_graph):
    """Production of TROLL between Feb 2020 and Dec 2020."""
    result = production_graph.cypher("""
        MATCH (f:Field {title: 'TROLL'})
        RETURN ts_sum(f.oil, '2020-2', '2020-12') AS total_oil
    """)
    # Months 2020-02 through 2020-12 = indices 13..23 (11 months)
    expected = sum(1.0 + 0.06 * i for i in range(13, 24))
    assert abs(result[0]["total_oil"] - expected) < 1e-6


def test_production_difference_pct_with_arithmetic(production_graph):
    """Difference in production between summer months and rest using Cypher arithmetic."""
    result = production_graph.cypher("""
        MATCH (f:Field {title: 'TROLL'})
        WITH ts_sum(f.oil, '2020-6', '2020-8') AS summer,
             ts_sum(f.oil, '2020') AS total
        RETURN summer, total - summer AS rest,
               (summer / (total - summer)) * 100.0 AS summer_pct_of_rest
    """)
    assert result[0]["summer"] > 0
    assert result[0]["rest"] > 0
    assert result[0]["summer_pct_of_rest"] > 0


def test_latest_recorded_month_via_series(production_graph):
    """What is the latest recorded month of TROLL production."""
    result = production_graph.cypher("""
        MATCH (f:Field {title: 'TROLL'})
        RETURN ts_series(f.oil) AS data
    """)
    series = result[0]["data"]
    last = series[-1]
    assert last["time"] == [2020, 12]


def test_first_production_record_via_series(production_graph):
    """When did TROLL start production (first record)."""
    result = production_graph.cypher("""
        MATCH (f:Field {title: 'TROLL'})
        RETURN ts_series(f.oil) AS data
    """)
    series = result[0]["data"]
    first = series[0]
    assert first["time"] == [2019, 1]


def test_top_producers_order_by(production_graph):
    """Top fields by production, ordered."""
    result = production_graph.cypher("""
        MATCH (f:Field)
        RETURN f.title AS name, ts_sum(f.oil, '2020') AS prod
        ORDER BY prod DESC LIMIT 10
    """)
    assert result[0]["name"] == "TROLL"
    assert result[1]["name"] == "EKOFISK"
    assert result[0]["prod"] > result[1]["prod"]


def test_unwind_dynamic_ts_args(production_graph):
    """UNWIND + toString() for dynamic date-string args to ts_sum."""
    result = production_graph.cypher("""
        MATCH (f:Field {title: 'TROLL'})
        UNWIND range(2019, 2020) AS year
        WITH f, year, ts_sum(f.oil, toString(year) + '-6', toString(year) + '-8') AS summer_yr
        RETURN sum(summer_yr) AS summer_total
    """)
    # Summer months (Jun-Aug) for 2019 and 2020
    # 2019: indices 5,6,7 → oil = 1.30 + 1.36 + 1.42
    # 2020: indices 17,18,19 → oil = 2.02 + 2.08 + 2.14
    expected = sum(1.0 + 0.06 * i for i in [5, 6, 7, 17, 18, 19])
    assert abs(result[0]["summer_total"] - expected) < 1e-6


def test_unwind_summer_vs_rest_pct(production_graph):
    """Full summer-vs-rest percentage using UNWIND loop."""
    result = production_graph.cypher("""
        MATCH (f:Field {title: 'TROLL'})
        UNWIND range(2019, 2020) AS year
        WITH f, sum(ts_sum(f.oil, toString(year) + '-6', toString(year) + '-8')) AS summer
        RETURN summer,
               ts_sum(f.oil, '2019', '2020') AS total,
               summer / ts_sum(f.oil, '2019', '2020') * 100.0 AS summer_pct
    """)
    assert result[0]["summer"] > 0
    assert result[0]["total"] > 0
    # Summer is 6 out of 24 months = ~25%
    assert 20 < result[0]["summer_pct"] < 30


def test_ts_at_with_dynamic_string(production_graph):
    """ts_at with dynamically constructed date string."""
    result = production_graph.cypher("""
        MATCH (f:Field {title: 'TROLL'})
        UNWIND [3, 6, 9, 12] AS month
        RETURN month, ts_at(f.oil, '2020-' + toString(month)) AS val
        ORDER BY month
    """)
    assert len(result) == 4
    assert result[0]["month"] == 3
    assert result[0]["val"] is not None
    # Values should be increasing (ramp data)
    for i in range(len(result) - 1):
        assert result[i]["val"] < result[i + 1]["val"]
