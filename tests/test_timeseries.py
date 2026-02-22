"""Tests for timeseries Python API."""

import pytest
import pandas as pd
import kglite


@pytest.fixture
def graph_with_fields():
    """Create a graph with two Field nodes and attach timeseries data."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["TROLL", "EKOFISK"],
        }
    )
    g.add_nodes(df, "Field", "id", "title")
    return g


# ── Config ────────────────────────────────────────────────────────────────


def test_set_get_timeseries_config(graph_with_fields):
    g = graph_with_fields
    g.set_timeseries(
        "Field", resolution="month", channels=["oil", "gas"],
        units={"oil": "MSm3", "gas": "BSm3"}, bin_type="total",
    )
    config = g.get_timeseries_config("Field")
    assert config is not None
    assert config["resolution"] == "month"
    assert config["channels"] == ["oil", "gas"]
    assert config["units"] == {"oil": "MSm3", "gas": "BSm3"}
    assert config["bin_type"] == "total"


def test_get_timeseries_config_none(graph_with_fields):
    g = graph_with_fields
    assert g.get_timeseries_config("Field") is None
    assert g.get_timeseries_config() is None


def test_get_timeseries_config_all(graph_with_fields):
    g = graph_with_fields
    g.set_timeseries("Field", resolution="year")
    result = g.get_timeseries_config()
    assert isinstance(result, dict)
    assert "Field" in result


def test_config_no_units_or_bin_type(graph_with_fields):
    g = graph_with_fields
    g.set_timeseries("Field", resolution="month", channels=["oil"])
    config = g.get_timeseries_config("Field")
    assert config["resolution"] == "month"
    assert config.get("units", {}) == {}
    assert config.get("bin_type") is None


# ── Per-node: set_time_index + add_ts_channel ─────────────────────────────


def test_set_time_index_with_date_strings(graph_with_fields):
    g = graph_with_fields
    g.set_time_index(1, ["2020-01", "2020-02", "2020-03"])
    g.add_ts_channel(1, "oil", [1.0, 2.0, 3.0])
    ts = g.get_timeseries(1)
    assert ts is not None
    assert ts["keys"] == ["2020-01-01", "2020-02-01", "2020-03-01"]
    assert ts["channels"]["oil"] == [1.0, 2.0, 3.0]


def test_set_time_index_with_int_lists_compat(graph_with_fields):
    """Backwards-compatible: list of int lists still works."""
    g = graph_with_fields
    g.set_time_index(1, [[2020, 1], [2020, 2], [2020, 3]])
    g.add_ts_channel(1, "oil", [1.0, 2.0, 3.0])
    ts = g.get_timeseries(1)
    assert ts is not None
    assert ts["keys"] == ["2020-01-01", "2020-02-01", "2020-03-01"]
    assert ts["channels"]["oil"] == [1.0, 2.0, 3.0]


def test_add_ts_channel_length_mismatch(graph_with_fields):
    g = graph_with_fields
    g.set_time_index(1, ["2020-01", "2020-02"])
    with pytest.raises(ValueError, match="2 keys"):
        g.add_ts_channel(1, "oil", [1.0, 2.0, 3.0])


def test_add_ts_channel_no_time_index(graph_with_fields):
    g = graph_with_fields
    with pytest.raises(ValueError, match="time index"):
        g.add_ts_channel(1, "oil", [1.0])


def test_set_time_index_unsorted(graph_with_fields):
    g = graph_with_fields
    with pytest.raises(ValueError, match="sorted"):
        g.set_time_index(1, ["2020-02", "2020-01"])


def test_get_time_index(graph_with_fields):
    g = graph_with_fields
    g.set_time_index(1, ["2020-01", "2020-02"])
    idx = g.get_time_index(1)
    assert idx == ["2020-01-01", "2020-02-01"]


def test_get_time_index_none(graph_with_fields):
    g = graph_with_fields
    assert g.get_time_index(1) is None


def test_get_timeseries_none(graph_with_fields):
    g = graph_with_fields
    assert g.get_timeseries(1) is None


def test_get_timeseries_single_channel(graph_with_fields):
    g = graph_with_fields
    g.set_time_index(1, ["2020-01", "2020-02"])
    g.add_ts_channel(1, "oil", [1.0, 2.0])
    g.add_ts_channel(1, "gas", [0.5, 0.6])
    ts = g.get_timeseries(1, channel="oil")
    assert "values" in ts
    assert ts["values"] == [1.0, 2.0]


def test_get_timeseries_missing_channel(graph_with_fields):
    g = graph_with_fields
    g.set_time_index(1, ["2020-01"])
    g.add_ts_channel(1, "oil", [1.0])
    with pytest.raises(KeyError, match="water"):
        g.get_timeseries(1, channel="water")


def test_get_timeseries_with_date_string_range(graph_with_fields):
    g = graph_with_fields
    g.set_timeseries("Field", resolution="month")
    g.set_time_index(
        1,
        ["2019-12", "2020-01", "2020-02", "2021-01"],
    )
    g.add_ts_channel(1, "oil", [0.9, 1.0, 2.0, 3.0])
    ts = g.get_timeseries(1, start="2020", end="2020")
    assert ts["keys"] == ["2020-01-01", "2020-02-01"]
    assert ts["channels"]["oil"] == [1.0, 2.0]


def test_get_timeseries_with_month_range(graph_with_fields):
    g = graph_with_fields
    g.set_timeseries("Field", resolution="month")
    g.set_time_index(
        1,
        ["2020-01", "2020-02", "2020-03", "2020-04"],
    )
    g.add_ts_channel(1, "oil", [1.0, 2.0, 3.0, 4.0])
    ts = g.get_timeseries(1, start="2020-2", end="2020-3")
    assert ts["keys"] == ["2020-02-01", "2020-03-01"]
    assert ts["channels"]["oil"] == [2.0, 3.0]


def test_multiple_channels(graph_with_fields):
    g = graph_with_fields
    g.set_time_index(1, ["2020-01", "2020-02"])
    g.add_ts_channel(1, "oil", [1.0, 2.0])
    g.add_ts_channel(1, "gas", [0.5, 0.6])
    ts = g.get_timeseries(1)
    assert "oil" in ts["channels"]
    assert "gas" in ts["channels"]


def test_set_time_index_replaces(graph_with_fields):
    """Setting a new time index clears existing channels."""
    g = graph_with_fields
    g.set_time_index(1, ["2020-01", "2020-02"])
    g.add_ts_channel(1, "oil", [1.0, 2.0])
    g.set_time_index(1, ["2021-01"])
    ts = g.get_timeseries(1)
    assert ts["keys"] == ["2021-01-01"]
    assert ts["channels"] == {}


# ── Bulk: add_timeseries from DataFrame ───────────────────────────────────


def test_add_timeseries_from_dataframe(graph_with_fields):
    g = graph_with_fields
    ts_df = pd.DataFrame(
        {
            "field_id": [1, 1, 1, 2, 2, 2],
            "year": [2020, 2020, 2020, 2020, 2020, 2020],
            "month": [1, 2, 3, 1, 2, 3],
            "oil": [1.0, 1.5, 2.0, 0.5, 0.6, 0.7],
            "gas": [0.1, 0.2, 0.3, 0.05, 0.06, 0.07],
        }
    )
    result = g.add_timeseries(
        "Field",
        data=ts_df,
        fk="field_id",
        time_key=["year", "month"],
        channels={"oil": "oil", "gas": "gas"},
        resolution="month",
    )
    assert result["nodes_loaded"] == 2
    assert result["total_records"] == 6

    ts1 = g.get_timeseries(1)
    assert ts1["keys"] == ["2020-01-01", "2020-02-01", "2020-03-01"]
    assert ts1["channels"]["oil"] == [1.0, 1.5, 2.0]
    assert ts1["channels"]["gas"] == [0.1, 0.2, 0.3]

    ts2 = g.get_timeseries(2)
    assert ts2["keys"] == ["2020-01-01", "2020-02-01", "2020-03-01"]
    assert ts2["channels"]["oil"] == [0.5, 0.6, 0.7]


def test_add_timeseries_channels_as_list(graph_with_fields):
    g = graph_with_fields
    ts_df = pd.DataFrame(
        {
            "field_id": [1, 1],
            "year": [2020, 2021],
            "oil": [1.0, 2.0],
        }
    )
    result = g.add_timeseries(
        "Field",
        data=ts_df,
        fk="field_id",
        time_key=["year"],
        channels=["oil"],
        resolution="year",
    )
    assert result["nodes_loaded"] == 1
    ts = g.get_timeseries(1)
    assert ts["channels"]["oil"] == [1.0, 2.0]


def test_add_timeseries_auto_config(graph_with_fields):
    g = graph_with_fields
    ts_df = pd.DataFrame(
        {
            "field_id": [1],
            "year": [2020],
            "oil": [1.0],
        }
    )
    g.add_timeseries(
        "Field",
        data=ts_df,
        fk="field_id",
        time_key=["year"],
        channels=["oil"],
        resolution="year",
    )
    config = g.get_timeseries_config("Field")
    assert config is not None
    assert config["resolution"] == "year"
    assert "oil" in config["channels"]


def test_add_timeseries_with_units(graph_with_fields):
    g = graph_with_fields
    ts_df = pd.DataFrame(
        {
            "field_id": [1],
            "year": [2020],
            "month": [1],
            "oil": [1.0],
        }
    )
    g.add_timeseries(
        "Field",
        data=ts_df,
        fk="field_id",
        time_key=["year", "month"],
        channels=["oil"],
        resolution="month",
        units={"oil": "MSm3"},
    )
    config = g.get_timeseries_config("Field")
    assert config["units"] == {"oil": "MSm3"}


def test_add_timeseries_uses_existing_config(graph_with_fields):
    """add_timeseries uses resolution from existing config if not passed."""
    g = graph_with_fields
    g.set_timeseries("Field", resolution="month", channels=["oil"])
    ts_df = pd.DataFrame(
        {
            "field_id": [1],
            "year": [2020],
            "month": [1],
            "oil": [1.0],
        }
    )
    g.add_timeseries(
        "Field",
        data=ts_df,
        fk="field_id",
        time_key=["year", "month"],
        channels=["oil"],
    )
    ts = g.get_timeseries(1)
    assert ts["channels"]["oil"] == [1.0]


def test_add_timeseries_unsorted_data(graph_with_fields):
    """Data should be sorted by time_key during loading."""
    g = graph_with_fields
    ts_df = pd.DataFrame(
        {
            "field_id": [1, 1, 1],
            "year": [2020, 2020, 2020],
            "month": [3, 1, 2],
            "oil": [3.0, 1.0, 2.0],
        }
    )
    g.add_timeseries(
        "Field",
        data=ts_df,
        fk="field_id",
        time_key=["year", "month"],
        channels=["oil"],
        resolution="month",
    )
    ts = g.get_timeseries(1)
    assert ts["keys"] == ["2020-01-01", "2020-02-01", "2020-03-01"]
    assert ts["channels"]["oil"] == [1.0, 2.0, 3.0]


# ── Persistence ──────────────────────────────────────────────────────────


def test_save_load_roundtrip(graph_with_fields, tmp_path):
    g = graph_with_fields
    g.set_timeseries(
        "Field", resolution="month", channels=["oil"],
        units={"oil": "MSm3"}, bin_type="total",
    )
    g.set_time_index(1, ["2020-01", "2020-02"])
    g.add_ts_channel(1, "oil", [1.0, 2.0])

    path = str(tmp_path / "test.kgl")
    g.save(path)
    g2 = kglite.load(path)

    ts = g2.get_timeseries(1)
    assert ts is not None
    assert ts["keys"] == ["2020-01-01", "2020-02-01"]
    assert ts["channels"]["oil"] == [1.0, 2.0]

    config = g2.get_timeseries_config("Field")
    assert config is not None
    assert config["resolution"] == "month"
    assert config["units"] == {"oil": "MSm3"}
    assert config["bin_type"] == "total"


def test_save_load_no_timeseries(tmp_path):
    """Files without timeseries should load fine."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({"id": [1], "title": ["A"]})
    g.add_nodes(df, "Node", "id", "title")
    path = str(tmp_path / "no_ts.kgl")
    g.save(path)
    g2 = kglite.load(path)
    assert g2.get_timeseries(1) is None
