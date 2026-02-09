// Spatial/Geometry Operations Module
// Provides spatial filtering and geometry operations on graph nodes

use geo::geometry::{Geometry, LineString, Point, Polygon};
use geo::{Centroid, Contains, Intersects, Rect};
use petgraph::graph::NodeIndex;
use wkt::TryFromWkt;

use crate::datatypes::values::Value;
use crate::graph::schema::{CurrentSelection, DirGraph};

/// Filter nodes that fall within a geographic bounding box
///
/// # Arguments
/// * `graph` - The graph to filter
/// * `selection` - Current selection to filter from (if None, uses all nodes of specified type)
/// * `node_type` - Optional node type to filter
/// * `lat_field` - Name of the latitude field
/// * `lon_field` - Name of the longitude field
/// * `min_lat` - Minimum latitude (south bound)
/// * `max_lat` - Maximum latitude (north bound)
/// * `min_lon` - Minimum longitude (west bound)
/// * `max_lon` - Maximum longitude (east bound)
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn within_bounds(
    graph: &DirGraph,
    selection: &CurrentSelection,
    lat_field: &str,
    lon_field: &str,
    min_lat: f64,
    max_lat: f64,
    min_lon: f64,
    max_lon: f64,
) -> Result<Vec<NodeIndex>, String> {
    // Get nodes from current selection
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        // If no selection, return empty (must have a selection to filter)
        return Ok(Vec::new());
    };

    let mut matching_nodes = Vec::new();

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let lat = node.properties.get(lat_field).and_then(value_to_f64);
            let lon = node.properties.get(lon_field).and_then(value_to_f64);

            if let (Some(lat), Some(lon)) = (lat, lon) {
                // Use simple numeric comparison
                if lat >= min_lat && lat <= max_lat && lon >= min_lon && lon <= max_lon {
                    matching_nodes.push(node_idx);
                }
            }
        }
    }

    Ok(matching_nodes)
}

/// Filter nodes within a certain distance of a point (in degrees)
///
/// Note: This uses Euclidean distance in degrees, which is approximate.
/// For more accurate distance calculations on the Earth's surface,
/// you would need to use Haversine or Vincenty formulas.
///
/// # Arguments
/// * `graph` - The graph to filter
/// * `selection` - Current selection to filter from
/// * `lat_field` - Name of the latitude field
/// * `lon_field` - Name of the longitude field
/// * `center_lat` - Center point latitude
/// * `center_lon` - Center point longitude
/// * `max_distance` - Maximum distance in degrees
pub fn near_point(
    graph: &DirGraph,
    selection: &CurrentSelection,
    lat_field: &str,
    lon_field: &str,
    center_lat: f64,
    center_lon: f64,
    max_distance: f64,
) -> Result<Vec<NodeIndex>, String> {
    // Get nodes from current selection
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return Ok(Vec::new());
    };

    let mut matching_nodes = Vec::new();

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let lat = node.properties.get(lat_field).and_then(value_to_f64);
            let lon = node.properties.get(lon_field).and_then(value_to_f64);

            if let (Some(lat), Some(lon)) = (lat, lon) {
                // Simple Euclidean distance in degrees
                let d_lat = lat - center_lat;
                let d_lon = lon - center_lon;
                let distance = (d_lat * d_lat + d_lon * d_lon).sqrt();
                if distance <= max_distance {
                    matching_nodes.push(node_idx);
                }
            }
        }
    }

    Ok(matching_nodes)
}

/// Earth's mean radius in kilometers
const EARTH_RADIUS_KM: f64 = 6371.0;

/// Calculate the Haversine distance between two points in kilometers
///
/// This gives the great-circle distance between two points on a sphere,
/// which is more accurate than Euclidean distance for geographic coordinates.
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_KM * c
}

/// Filter nodes within a certain distance of a point (in kilometers)
///
/// Uses the Haversine formula to calculate accurate great-circle distances
/// on the Earth's surface.
///
/// # Arguments
/// * `graph` - The graph to filter
/// * `selection` - Current selection to filter from
/// * `lat_field` - Name of the latitude field
/// * `lon_field` - Name of the longitude field
/// * `center_lat` - Center point latitude
/// * `center_lon` - Center point longitude
/// * `max_distance_km` - Maximum distance in kilometers
pub fn near_point_km(
    graph: &DirGraph,
    selection: &CurrentSelection,
    lat_field: &str,
    lon_field: &str,
    center_lat: f64,
    center_lon: f64,
    max_distance_km: f64,
) -> Result<Vec<NodeIndex>, String> {
    // Get nodes from current selection
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return Ok(Vec::new());
    };

    let mut matching_nodes = Vec::new();

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let lat = node.properties.get(lat_field).and_then(value_to_f64);
            let lon = node.properties.get(lon_field).and_then(value_to_f64);

            if let (Some(lat), Some(lon)) = (lat, lon) {
                let distance = haversine_distance(center_lat, center_lon, lat, lon);
                if distance <= max_distance_km {
                    matching_nodes.push(node_idx);
                }
            }
        }
    }

    Ok(matching_nodes)
}

/// Parse a WKT string into a geo Geometry
pub fn parse_wkt(wkt_string: &str) -> Result<Geometry<f64>, String> {
    Geometry::try_from_wkt_str(wkt_string).map_err(|e| format!("Invalid WKT: {:?}", e))
}

/// Extract centroid coordinates from a WKT geometry string
///
/// Returns (lat, lon) tuple where lat=y and lon=x (geographic convention)
pub fn wkt_centroid(wkt_string: &str) -> Result<(f64, f64), String> {
    let geometry = parse_wkt(wkt_string)?;

    let centroid: Option<Point<f64>> = match &geometry {
        Geometry::Point(p) => Some(*p),
        Geometry::Polygon(p) => p.centroid(),
        Geometry::MultiPolygon(mp) => mp.centroid(),
        Geometry::LineString(ls) => ls.centroid(),
        Geometry::MultiLineString(mls) => mls.centroid(),
        Geometry::MultiPoint(mp) => mp.centroid(),
        Geometry::Rect(r) => Some(r.centroid()),
        _ => None,
    };

    match centroid {
        Some(point) => Ok((point.y(), point.x())), // lat=y, lon=x
        None => Err("Could not calculate centroid for geometry".to_string()),
    }
}

/// Filter nodes within distance of a point using WKT geometry centroids
///
/// Uses the Haversine formula to calculate accurate great-circle distances
/// from geometry centroids to a query point.
///
/// # Arguments
/// * `graph` - The graph to filter
/// * `selection` - Current selection to filter from
/// * `geometry_field` - Name of the field containing WKT geometry
/// * `center_lat` - Center point latitude
/// * `center_lon` - Center point longitude
/// * `max_distance_km` - Maximum distance in kilometers
pub fn near_point_km_from_geometry(
    graph: &DirGraph,
    selection: &CurrentSelection,
    geometry_field: &str,
    center_lat: f64,
    center_lon: f64,
    max_distance_km: f64,
) -> Result<Vec<NodeIndex>, String> {
    // Get nodes from current selection
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return Ok(Vec::new());
    };

    let mut matching_nodes = Vec::new();

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let wkt_value = node.properties.get(geometry_field);

            if let Some(Value::String(wkt_str)) = wkt_value {
                if let Ok((lat, lon)) = wkt_centroid(wkt_str) {
                    let distance = haversine_distance(center_lat, center_lon, lat, lon);
                    if distance <= max_distance_km {
                        matching_nodes.push(node_idx);
                    }
                }
            }
        }
    }

    Ok(matching_nodes)
}

/// Filter nodes whose WKT polygon contains a query point
///
/// # Arguments
/// * `graph` - The graph to filter
/// * `selection` - Current selection to filter from
/// * `geometry_field` - Name of the field containing WKT geometry
/// * `lat` - Query point latitude
/// * `lon` - Query point longitude
pub fn contains_point(
    graph: &DirGraph,
    selection: &CurrentSelection,
    geometry_field: &str,
    lat: f64,
    lon: f64,
) -> Result<Vec<NodeIndex>, String> {
    let query_point = Point::new(lon, lat); // geo uses (x, y) = (lon, lat)

    // Get nodes from current selection
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return Ok(Vec::new());
    };

    let mut matching_nodes = Vec::new();

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let wkt_value = node.properties.get(geometry_field);

            if let Some(Value::String(wkt_str)) = wkt_value {
                if let Ok(geometry) = parse_wkt(wkt_str) {
                    let contains = match &geometry {
                        Geometry::Polygon(p) => p.contains(&query_point),
                        Geometry::MultiPolygon(mp) => mp.contains(&query_point),
                        Geometry::Rect(r) => r.contains(&query_point),
                        _ => false,
                    };
                    if contains {
                        matching_nodes.push(node_idx);
                    }
                }
            }
        }
    }

    Ok(matching_nodes)
}

/// Filter nodes whose geometry intersects with a query geometry
///
/// # Arguments
/// * `graph` - The graph to filter
/// * `selection` - Current selection to filter from
/// * `geometry_field` - Name of the field containing WKT geometry
/// * `query_wkt` - WKT string of the query geometry
pub fn intersects_geometry(
    graph: &DirGraph,
    selection: &CurrentSelection,
    geometry_field: &str,
    query_wkt: &str,
) -> Result<Vec<NodeIndex>, String> {
    let query_geometry = parse_wkt(query_wkt)?;

    // Get nodes from current selection
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return Ok(Vec::new());
    };

    let mut matching_nodes = Vec::new();

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let wkt_value = node.properties.get(geometry_field);

            if let Some(Value::String(wkt_str)) = wkt_value {
                if let Ok(node_geometry) = parse_wkt(wkt_str) {
                    if geometries_intersect(&node_geometry, &query_geometry) {
                        matching_nodes.push(node_idx);
                    }
                }
            }
        }
    }

    Ok(matching_nodes)
}

/// Check if two geometries intersect
fn geometries_intersect(a: &Geometry<f64>, b: &Geometry<f64>) -> bool {
    match (a, b) {
        (Geometry::Point(p1), Geometry::Point(p2)) => p1 == p2,
        (Geometry::Point(p), Geometry::Polygon(poly))
        | (Geometry::Polygon(poly), Geometry::Point(p)) => poly.contains(p),
        (Geometry::Point(p), Geometry::Rect(r)) | (Geometry::Rect(r), Geometry::Point(p)) => {
            r.contains(p)
        }
        (Geometry::Polygon(p1), Geometry::Polygon(p2)) => p1.intersects(p2),
        (Geometry::Rect(r1), Geometry::Rect(r2)) => r1.intersects(r2),
        (Geometry::Polygon(p), Geometry::Rect(r)) | (Geometry::Rect(r), Geometry::Polygon(p)) => {
            // Convert rect to polygon for intersection check
            let rect_poly = rect_to_polygon(r);
            p.intersects(&rect_poly)
        }
        (Geometry::LineString(l1), Geometry::LineString(l2)) => l1.intersects(l2),
        (Geometry::LineString(l), Geometry::Polygon(p))
        | (Geometry::Polygon(p), Geometry::LineString(l)) => l.intersects(p),
        // For other combinations, try a general approach
        _ => false,
    }
}

/// Convert a Rect to a Polygon
fn rect_to_polygon(rect: &Rect<f64>) -> Polygon<f64> {
    let min = rect.min();
    let max = rect.max();
    Polygon::new(
        LineString::from(vec![
            (min.x, min.y),
            (max.x, min.y),
            (max.x, max.y),
            (min.x, max.y),
            (min.x, min.y),
        ]),
        vec![],
    )
}

/// Convert a Value to f64 if possible
fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Float64(f) => Some(*f),
        Value::Int64(i) => Some(*i as f64),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Calculate the centroid of nodes in a selection
pub fn calculate_centroid(
    graph: &DirGraph,
    selection: &CurrentSelection,
    lat_field: &str,
    lon_field: &str,
) -> Option<(f64, f64)> {
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return None;
    };

    let mut sum_lat = 0.0;
    let mut sum_lon = 0.0;
    let mut count = 0;

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let lat = node.properties.get(lat_field).and_then(value_to_f64);
            let lon = node.properties.get(lon_field).and_then(value_to_f64);

            if let (Some(lat), Some(lon)) = (lat, lon) {
                sum_lat += lat;
                sum_lon += lon;
                count += 1;
            }
        }
    }

    if count > 0 {
        Some((sum_lat / count as f64, sum_lon / count as f64))
    } else {
        None
    }
}

/// Get the bounding box of all nodes in a selection
pub fn get_bounds(
    graph: &DirGraph,
    selection: &CurrentSelection,
    lat_field: &str,
    lon_field: &str,
) -> Option<(f64, f64, f64, f64)> {
    let level_count = selection.get_level_count();
    let nodes: Vec<NodeIndex> = if level_count > 0 {
        selection
            .get_level(level_count - 1)
            .map(|l| l.get_all_nodes())
            .unwrap_or_default()
    } else {
        return None;
    };

    let mut min_lat = f64::MAX;
    let mut max_lat = f64::MIN;
    let mut min_lon = f64::MAX;
    let mut max_lon = f64::MIN;
    let mut found_any = false;

    for node_idx in nodes {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let lat = node.properties.get(lat_field).and_then(value_to_f64);
            let lon = node.properties.get(lon_field).and_then(value_to_f64);

            if let (Some(lat), Some(lon)) = (lat, lon) {
                min_lat = min_lat.min(lat);
                max_lat = max_lat.max(lat);
                min_lon = min_lon.min(lon);
                max_lon = max_lon.max(lon);
                found_any = true;
            }
        }
    }

    if found_any {
        Some((min_lat, max_lat, min_lon, max_lon))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_wkt_point() {
        let result = parse_wkt("POINT(10.5 59.9)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_wkt_polygon() {
        let result = parse_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_wkt_invalid() {
        let result = parse_wkt("INVALID(abc)");
        assert!(result.is_err());
    }

    #[test]
    fn test_value_to_f64() {
        assert_eq!(value_to_f64(&Value::Float64(1.5)), Some(1.5));
        assert_eq!(value_to_f64(&Value::Int64(42)), Some(42.0));
        assert_eq!(value_to_f64(&Value::String("3.14".to_string())), Some(3.14));
        assert_eq!(value_to_f64(&Value::Null), None);
    }
}
