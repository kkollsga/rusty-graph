//! GeoJSON → WKT + centroid conversion for blueprint node CSVs.
//!
//! Two modes, both matching the old Python loader at `loader.py:660`:
//! - If the CSV already has a WKT column (e.g. Sodir's `wkt_geometry`),
//!   we do nothing — the value passes through as a string property.
//! - If the CSV has a `_geometry` column containing GeoJSON strings, we
//!   parse with the `geojson` crate, convert to `geo::Geometry`, then
//!   write WKT + centroid lat/lon back into the declared columns.

use super::csv_loader::RawCsv;
use geo::Centroid;
use indexmap::IndexMap;
use wkt::ToWkt;

/// Is any property type a spatial virtual type?
pub fn has_spatial_properties(properties: &IndexMap<String, String>) -> bool {
    properties
        .values()
        .any(|v| matches!(v.as_str(), "geometry" | "location.lat" | "location.lon"))
}

/// Find which columns hold the WKT / lat / lon derived values.
pub fn spatial_targets(properties: &IndexMap<String, String>) -> SpatialTargets {
    let mut t = SpatialTargets::default();
    for (col, ty) in properties {
        match ty.as_str() {
            "geometry" => t.wkt = Some(col.clone()),
            "location.lat" => t.lat = Some(col.clone()),
            "location.lon" => t.lon = Some(col.clone()),
            _ => {}
        }
    }
    t
}

#[derive(Debug, Default)]
pub struct SpatialTargets {
    pub wkt: Option<String>,
    pub lat: Option<String>,
    pub lon: Option<String>,
}

/// If the CSV has a `_geometry` column with GeoJSON, parse each row and
/// populate the WKT / lat / lon target columns. If the target column
/// already exists in the CSV it is overwritten; otherwise it is appended.
pub fn convert_geojson(raw: &mut RawCsv, targets: &SpatialTargets) -> Result<(), String> {
    let Some(geo_idx) = raw.col_index("_geometry") else {
        return Ok(());
    };

    let n = raw.row_count();
    let mut wkts = vec![None; n];
    let mut lats = vec![None; n];
    let mut lons = vec![None; n];

    for (r, row) in raw.rows.iter().enumerate() {
        if raw.nulls[r][geo_idx] {
            continue;
        }
        let s = &row[geo_idx];
        if s.is_empty() {
            continue;
        }
        let Ok(gj): Result<geojson::GeoJson, _> = s.parse() else {
            continue;
        };
        let geom_opt: Option<geo::Geometry<f64>> = match gj {
            geojson::GeoJson::Geometry(g) => (&g).try_into().ok(),
            geojson::GeoJson::Feature(f) => f.geometry.as_ref().and_then(|g| g.try_into().ok()),
            geojson::GeoJson::FeatureCollection(_) => None,
        };
        let Some(g) = geom_opt else { continue };

        if targets.wkt.is_some() {
            wkts[r] = Some(g.wkt_string());
        }
        if targets.lat.is_some() || targets.lon.is_some() {
            if let Some(c) = g.centroid() {
                lats[r] = Some(c.y());
                lons[r] = Some(c.x());
            }
        }
    }

    write_column(raw, &targets.wkt, |r| wkts[r].clone());
    write_column(raw, &targets.lat, |r| lats[r].map(|v| v.to_string()));
    write_column(raw, &targets.lon, |r| lons[r].map(|v| v.to_string()));

    Ok(())
}

fn write_column<F>(raw: &mut RawCsv, target: &Option<String>, mut value_for: F)
where
    F: FnMut(usize) -> Option<String>,
{
    let Some(name) = target else { return };
    let idx = match raw.col_index(name) {
        Some(i) => i,
        None => {
            raw.headers.push(name.clone());
            for row in &mut raw.rows {
                row.push(String::new());
            }
            for nrow in &mut raw.nulls {
                nrow.push(true);
            }
            raw.headers.len() - 1
        }
    };
    let n = raw.row_count();
    for r in 0..n {
        match value_for(r) {
            Some(v) => {
                raw.rows[r][idx] = v;
                raw.nulls[r][idx] = false;
            }
            None => {
                raw.rows[r][idx].clear();
                raw.nulls[r][idx] = true;
            }
        }
    }
}
