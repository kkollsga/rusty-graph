"""Sodir REST fetcher: paginated GeoJSON → CSV.

The Sodir / FactMaps endpoint is a public ArcGIS FeatureServer. Each
dataset is fetched by paginating `/{layer_id}/query?f=geojson` 1000
records at a time, flattening properties + geometry into a flat
DataFrame, and writing CSV.

We rate-limit to 0.2s between requests (matches `factpages-py` defaults)
and retry on 429/5xx via `urllib.request` retries handled by a thin
manual loop. No third-party HTTP client is required — the standard
library covers everything.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
import urllib.parse
from urllib.request import Request, urlopen

import pandas as pd

from .catalog import is_known, kind_of, resolve

USER_AGENT = "kglite-datasets-sodir/1"
PAGE_SIZE = 1000
RATE_LIMIT_SECS = 0.2
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECS = 0.5
RETRY_STATUSES = {429, 500, 502, 503, 504}
TIMEOUT_SECS = 60


def _http_get_json(url: str) -> dict:
    """GET a URL with retry on 429/5xx + connection errors."""
    last_err: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=TIMEOUT_SECS) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            last_err = e
            if e.code not in RETRY_STATUSES:
                raise
        except URLError as e:
            last_err = e
        time.sleep(RETRY_BACKOFF_SECS * (2**attempt))
    raise RuntimeError(f"GET {url} failed after {RETRY_ATTEMPTS} attempts: {last_err}")


def count(stem: str) -> int:
    """Cheap row-count probe via `returnCountOnly=true`. ~50-byte
    response, used to detect remote changes without re-downloading."""
    base_url, layer_id = resolve(stem)
    params = urllib.parse.urlencode({"where": "1=1", "returnCountOnly": "true", "f": "json"})
    url = f"{base_url}/{layer_id}/query?{params}"
    time.sleep(RATE_LIMIT_SECS)
    data = _http_get_json(url)
    return int(data.get("count", 0))


def fetch_to_csv(stem: str, csv_path: Path) -> int:
    """Paginate the dataset's `/query?f=geojson` endpoint and write to
    `csv_path`. Returns the number of rows written. Geometry is
    flattened into a `wkt_geometry` column when present.

    Direct write — if killed mid-pandas-write, the partial CSV will
    be re-fetched on the next run because the index isn't updated
    until after `fetch_to_csv` returns successfully."""
    base_url, layer_id = resolve(stem)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    offset = 0
    while True:
        params = urllib.parse.urlencode(
            {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "f": "geojson",
            }
        )
        url = f"{base_url}/{layer_id}/query?{params}"
        time.sleep(RATE_LIMIT_SECS)
        data = _http_get_json(url)
        features = data.get("features", []) or []
        if not features:
            break
        for f in features:
            row = dict(f.get("properties", {}) or {})
            geom = f.get("geometry")
            if geom is not None:
                row["wkt_geometry"] = _geometry_to_wkt(geom)
            rows.append(row)
        if len(features) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if rows:
        df = pd.DataFrame(rows)
    else:
        # Layer has zero rows. Write a header-only CSV using the
        # layer's declared schema so downstream blueprint builds can
        # still resolve column names instead of crashing on an empty
        # file. Costs one extra HTTP call only for empty datasets.
        df = pd.DataFrame(columns=_layer_field_names(base_url, layer_id) + ["wkt_geometry"])
    _convert_timestamp_columns(df)
    df.to_csv(csv_path, index=False)
    return len(df)


def _layer_field_names(base_url: str, layer_id: int) -> list[str]:
    """Fetch the layer's field metadata and return field names in
    declaration order. Used as a fallback when a dataset is empty so
    we can still emit a header row."""
    url = f"{base_url}/{layer_id}?f=json"
    time.sleep(RATE_LIMIT_SECS)
    try:
        data = _http_get_json(url)
    except RuntimeError:
        return []
    return [f.get("name") for f in (data.get("fields") or []) if f.get("name")]


def _geometry_to_wkt(geom: dict) -> str | None:
    """Best-effort GeoJSON → WKT conversion for the geometries Sodir
    publishes (Point, MultiPoint, LineString, Polygon, MultiPolygon).

    Returns None when coordinates are missing or malformed — Sodir
    occasionally emits an empty ``coordinates: []`` for entities with
    unknown location (some pre-1970 wellbores, in-progress surveys),
    and we'd rather drop the geometry than crash the whole fetch."""
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return None
    try:
        if gtype == "Point":
            if len(coords) < 2:
                return None
            return f"POINT({coords[0]} {coords[1]})"
        if gtype == "MultiPoint":
            pts = [c for c in coords if len(c) >= 2]
            if not pts:
                return None
            return "MULTIPOINT(" + ", ".join(f"{c[0]} {c[1]}" for c in pts) + ")"
        if gtype == "LineString":
            pts = [c for c in coords if len(c) >= 2]
            if len(pts) < 2:
                return None
            return "LINESTRING(" + ", ".join(f"{c[0]} {c[1]}" for c in pts) + ")"
        if gtype == "Polygon":
            rings = []
            for r in coords:
                pts = [c for c in r if len(c) >= 2]
                if len(pts) < 3:
                    continue
                rings.append("(" + ", ".join(f"{c[0]} {c[1]}" for c in pts) + ")")
            if not rings:
                return None
            return "POLYGON(" + ", ".join(rings) + ")"
        if gtype == "MultiPolygon":
            polys = []
            for poly in coords:
                rings = []
                for r in poly:
                    pts = [c for c in r if len(c) >= 2]
                    if len(pts) < 3:
                        continue
                    rings.append("(" + ", ".join(f"{c[0]} {c[1]}" for c in pts) + ")")
                if rings:
                    polys.append("(" + ", ".join(rings) + ")")
            if not polys:
                return None
            return "MULTIPOLYGON(" + ", ".join(polys) + ")"
    except (TypeError, IndexError):
        return None
    return None


def _convert_timestamp_columns(df: pd.DataFrame) -> None:
    """ArcGIS REST returns dates as Unix-ms ints. Heuristic: any column
    whose name ends with `Date`/`From`/`To`/`Updated` and whose values
    look like ms-since-epoch becomes an ISO-8601 string in place. Same
    logic factpages-py uses (`client.py:77-111`)."""
    for col in df.columns:
        lower = col.lower()
        if not (lower.endswith("date") or lower.endswith("from") or lower.endswith("to") or lower.endswith("updated")):
            continue
        if df[col].dtype.kind not in ("i", "f"):
            continue
        # Likely epoch-ms if values are big enough to look like dates
        # past year 2000 (>= 946684800000).
        sample = df[col].dropna()
        if sample.empty or sample.iloc[0] < 946_684_800_000:
            continue
        df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")


def is_fetchable(stem: str) -> bool:
    """Return True if the dataset stem is in our REST catalog."""
    return is_known(stem)


def describe(stem: str) -> dict:
    """Return identifying info for the dataset (kind, layer_id,
    base_url) without making any HTTP calls."""
    base_url, layer_id = resolve(stem)
    return {
        "stem": stem,
        "kind": kind_of(stem),
        "layer_id": layer_id,
        "base_url": base_url,
        "described_at_iso": datetime.now(timezone.utc).isoformat(),
    }
