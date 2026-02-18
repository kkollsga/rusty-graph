// Spatial/Geometry #[pymethods] — extracted from mod.rs

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::schema::{PlanStep, SelectionOperation};
use super::{spatial, KnowledgeGraph};

#[pymethods]
impl KnowledgeGraph {
    // ========================================================================
    // Spatial/Geometry Methods
    // ========================================================================

    /// Filter nodes within a geographic bounding box.
    ///
    /// Filters nodes from the current selection that have latitude/longitude
    /// coordinates falling within the specified bounding box.
    ///
    /// Args:
    ///     min_lat: Minimum latitude (south bound)
    ///     max_lat: Maximum latitude (north bound)
    ///     min_lon: Minimum longitude (west bound)
    ///     max_lon: Maximum longitude (east bound)
    ///     lat_field: Name of the latitude property (default: 'latitude')
    ///     lon_field: Name of the longitude property (default: 'longitude')
    ///
    /// Returns:
    ///     A new KnowledgeGraph with only nodes within the bounding box
    ///
    /// Example:
    ///     ```python
    ///     # Filter discoveries in the North Sea area
    ///     north_sea = graph.type_filter('Discovery').within_bounds(
    ///         min_lat=56.0, max_lat=62.0,
    ///         min_lon=0.0, max_lon=8.0
    ///     )
    ///     ```
    #[pyo3(signature = (min_lat, max_lat, min_lon, max_lon, lat_field=None, lon_field=None))]
    fn within_bounds(
        &mut self,
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
        lat_field: Option<&str>,
        lon_field: Option<&str>,
    ) -> PyResult<Self> {
        let lat_field = lat_field.unwrap_or("latitude");
        let lon_field = lon_field.unwrap_or("longitude");

        let matching_nodes = spatial::within_bounds(
            &self.inner,
            &self.selection,
            lat_field,
            lon_field,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Create new selection with matching nodes
        let mut new_kg = self.clone();
        new_kg.selection.clear(); // clear() already adds a fresh level
        if let Some(level) = new_kg.selection.get_level_mut(0) {
            if !matching_nodes.is_empty() {
                level.add_selection(None, matching_nodes.clone());
            }
            level
                .operations
                .push(SelectionOperation::Custom("within_bounds".to_string()));
        }

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("WITHIN_BOUNDS", None, matching_nodes.len())
                .with_actual_rows(matching_nodes.len()),
        );

        Ok(new_kg)
    }

    /// Filter nodes within a certain distance of a point.
    ///
    /// Filters nodes from the current selection that are within the specified
    /// distance (in degrees) from the center point.
    ///
    /// Note: Distance is calculated using Euclidean distance in degrees,
    /// which is an approximation. For rough estimates:
    /// - 1 degree latitude ≈ 111 km
    /// - 1 degree longitude ≈ 111 km * cos(latitude)
    ///
    /// Args:
    ///     center_lat: Center point latitude
    ///     center_lon: Center point longitude
    ///     max_distance: Maximum distance in degrees
    ///     lat_field: Name of the latitude property (default: 'latitude')
    ///     lon_field: Name of the longitude property (default: 'longitude')
    ///
    /// Returns:
    ///     A new KnowledgeGraph with only nodes within the distance
    ///
    /// Example:
    ///     ```python
    ///     # Find discoveries near a point (within ~50km)
    ///     nearby = graph.type_filter('Discovery').near_point(
    ///         center_lat=60.0, center_lon=4.0,
    ///         max_distance=0.5  # ~50km
    ///     )
    ///     ```
    #[pyo3(signature = (center_lat, center_lon, max_distance, lat_field=None, lon_field=None))]
    fn near_point(
        &mut self,
        center_lat: f64,
        center_lon: f64,
        max_distance: f64,
        lat_field: Option<&str>,
        lon_field: Option<&str>,
    ) -> PyResult<Self> {
        let lat_field = lat_field.unwrap_or("latitude");
        let lon_field = lon_field.unwrap_or("longitude");

        let matching_nodes = spatial::near_point(
            &self.inner,
            &self.selection,
            lat_field,
            lon_field,
            center_lat,
            center_lon,
            max_distance,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Create new selection with matching nodes
        let mut new_kg = self.clone();
        new_kg.selection.clear(); // clear() already adds a fresh level
        if let Some(level) = new_kg.selection.get_level_mut(0) {
            if !matching_nodes.is_empty() {
                level.add_selection(None, matching_nodes.clone());
            }
            level
                .operations
                .push(SelectionOperation::Custom("near_point".to_string()));
        }

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("NEAR_POINT", None, matching_nodes.len())
                .with_actual_rows(matching_nodes.len()),
        );

        Ok(new_kg)
    }

    /// Filter nodes within a certain distance of a point (in kilometers).
    ///
    /// Uses the Haversine formula to calculate accurate great-circle distances
    /// on the Earth's surface. This is more accurate than `near_point()` which
    /// uses Euclidean distance in degrees.
    ///
    /// Args:
    ///     center_lat: Center point latitude
    ///     center_lon: Center point longitude
    ///     max_distance_km: Maximum distance in kilometers
    ///     lat_field: Name of the latitude property (default: 'latitude')
    ///     lon_field: Name of the longitude property (default: 'longitude')
    ///
    /// Returns:
    ///     A new KnowledgeGraph with only nodes within the distance
    ///
    /// Example:
    ///     ```python
    ///     # Find discoveries within 50km of a point
    ///     nearby = graph.type_filter('Discovery').near_point_km(
    ///         center_lat=60.0, center_lon=5.0,
    ///         max_distance_km=50.0
    ///     )
    ///     ```
    #[pyo3(signature = (center_lat, center_lon, max_distance_km, lat_field=None, lon_field=None))]
    fn near_point_km(
        &mut self,
        center_lat: f64,
        center_lon: f64,
        max_distance_km: f64,
        lat_field: Option<&str>,
        lon_field: Option<&str>,
    ) -> PyResult<Self> {
        let lat_field = lat_field.unwrap_or("latitude");
        let lon_field = lon_field.unwrap_or("longitude");

        let matching_nodes = spatial::near_point_km(
            &self.inner,
            &self.selection,
            lat_field,
            lon_field,
            center_lat,
            center_lon,
            max_distance_km,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Create new selection with matching nodes
        let mut new_kg = self.clone();
        new_kg.selection.clear(); // clear() already adds a fresh level
        if let Some(level) = new_kg.selection.get_level_mut(0) {
            if !matching_nodes.is_empty() {
                level.add_selection(None, matching_nodes.clone());
            }
            level
                .operations
                .push(SelectionOperation::Custom("near_point_km".to_string()));
        }

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("NEAR_POINT_KM", None, matching_nodes.len())
                .with_actual_rows(matching_nodes.len()),
        );

        Ok(new_kg)
    }

    /// Filter nodes within distance of a point using WKT geometry centroids.
    ///
    /// Uses the centroid of WKT geometries (polygons, etc.) to calculate distance.
    /// This eliminates the need for external libraries like shapely when working
    /// with polygon geometries.
    ///
    /// Args:
    ///     center_lat: Center point latitude
    ///     center_lon: Center point longitude
    ///     max_distance_km: Maximum distance in kilometers
    ///     geometry_field: Name of the WKT geometry property (default: 'geometry')
    ///
    /// Returns:
    ///     A new KnowledgeGraph with only nodes whose geometry centroid is within distance
    ///
    /// Example:
    ///     ```python
    ///     # Find prospects (with WKT polygons) within 50km of a point
    ///     nearby = graph.type_filter('Prospect').near_point_km_from_wkt(
    ///         center_lat=61.4, center_lon=4.0,
    ///         max_distance_km=50.0,
    ///         geometry_field='shape'
    ///     )
    ///     ```
    #[pyo3(signature = (center_lat, center_lon, max_distance_km, geometry_field=None))]
    fn near_point_km_from_wkt(
        &mut self,
        center_lat: f64,
        center_lon: f64,
        max_distance_km: f64,
        geometry_field: Option<&str>,
    ) -> PyResult<Self> {
        let geometry_field = geometry_field.unwrap_or("geometry");

        let matching_nodes = spatial::near_point_km_from_geometry(
            &self.inner,
            &self.selection,
            geometry_field,
            center_lat,
            center_lon,
            max_distance_km,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Create new selection with matching nodes
        let mut new_kg = self.clone();
        new_kg.selection.clear();
        if let Some(level) = new_kg.selection.get_level_mut(0) {
            if !matching_nodes.is_empty() {
                level.add_selection(None, matching_nodes.clone());
            }
            level
                .operations
                .push(SelectionOperation::Custom("near_point_km_wkt".to_string()));
        }

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("NEAR_POINT_KM_WKT", None, matching_nodes.len())
                .with_actual_rows(matching_nodes.len()),
        );

        Ok(new_kg)
    }

    /// Filter nodes whose WKT polygon contains a query point.
    ///
    /// Useful for finding which geographic regions (stored as WKT polygons)
    /// contain a specific point location.
    ///
    /// Args:
    ///     lat: Query point latitude
    ///     lon: Query point longitude
    ///     geometry_field: Name of the WKT geometry property (default: 'geometry')
    ///
    /// Returns:
    ///     A new KnowledgeGraph with only nodes whose geometry contains the point
    ///
    /// Example:
    ///     ```python
    ///     # Find which blocks contain a specific location
    ///     containing = graph.type_filter('Block').contains_point(
    ///         lat=61.4, lon=4.0,
    ///         geometry_field='boundary'
    ///     )
    ///     ```
    #[pyo3(signature = (lat, lon, geometry_field=None))]
    fn contains_point(
        &mut self,
        lat: f64,
        lon: f64,
        geometry_field: Option<&str>,
    ) -> PyResult<Self> {
        let geometry_field = geometry_field.unwrap_or("geometry");

        let matching_nodes =
            spatial::contains_point(&self.inner, &self.selection, geometry_field, lat, lon)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Create new selection with matching nodes
        let mut new_kg = self.clone();
        new_kg.selection.clear();
        if let Some(level) = new_kg.selection.get_level_mut(0) {
            if !matching_nodes.is_empty() {
                level.add_selection(None, matching_nodes.clone());
            }
            level
                .operations
                .push(SelectionOperation::Custom("contains_point".to_string()));
        }

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("CONTAINS_POINT", None, matching_nodes.len())
                .with_actual_rows(matching_nodes.len()),
        );

        Ok(new_kg)
    }

    /// Filter nodes whose geometry intersects with a WKT geometry.
    ///
    /// Filters nodes that have a geometry property (stored as WKT string)
    /// that intersects with the provided query geometry.
    ///
    /// Args:
    ///     query_wkt: WKT string of the query geometry
    ///     geometry_field: Name of the geometry property (default: 'geometry')
    ///
    /// Returns:
    ///     A new KnowledgeGraph with only intersecting nodes
    ///
    /// Example:
    ///     ```python
    ///     # Find blocks intersecting with a polygon
    ///     intersecting = graph.type_filter('Block').intersects_geometry(
    ///         'POLYGON((2 58, 4 58, 4 60, 2 60, 2 58))'
    ///     )
    ///     ```
    #[pyo3(signature = (query_wkt, geometry_field=None))]
    fn intersects_geometry(
        &mut self,
        query_wkt: &str,
        geometry_field: Option<&str>,
    ) -> PyResult<Self> {
        let geometry_field = geometry_field.unwrap_or("geometry");

        let matching_nodes =
            spatial::intersects_geometry(&self.inner, &self.selection, geometry_field, query_wkt)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Create new selection with matching nodes
        let mut new_kg = self.clone();
        new_kg.selection.clear(); // clear() already adds a fresh level
        if let Some(level) = new_kg.selection.get_level_mut(0) {
            if !matching_nodes.is_empty() {
                level.add_selection(None, matching_nodes.clone());
            }
            level.operations.push(SelectionOperation::Custom(
                "intersects_geometry".to_string(),
            ));
        }

        // Record plan step
        new_kg.selection.add_plan_step(
            PlanStep::new("INTERSECTS_GEOMETRY", None, matching_nodes.len())
                .with_actual_rows(matching_nodes.len()),
        );

        Ok(new_kg)
    }

    /// Get the geographic bounds of nodes in the current selection.
    ///
    /// Returns the minimum and maximum latitude/longitude of all nodes
    /// in the current selection.
    ///
    /// Args:
    ///     lat_field: Name of the latitude property (default: 'latitude')
    ///     lon_field: Name of the longitude property (default: 'longitude')
    ///
    /// Returns:
    ///     Dictionary with 'min_lat', 'max_lat', 'min_lon', 'max_lon',
    ///     or None if no valid coordinates found
    ///
    /// Example:
    ///     ```python
    ///     bounds = graph.type_filter('Discovery').get_bounds()
    ///     print(f"Latitude: {bounds['min_lat']} to {bounds['max_lat']}")
    ///     ```
    #[pyo3(signature = (lat_field=None, lon_field=None))]
    fn get_bounds(
        &self,
        py: Python<'_>,
        lat_field: Option<&str>,
        lon_field: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let lat_field = lat_field.unwrap_or("latitude");
        let lon_field = lon_field.unwrap_or("longitude");

        match spatial::get_bounds(&self.inner, &self.selection, lat_field, lon_field) {
            Some((min_lat, max_lat, min_lon, max_lon)) => {
                let result = PyDict::new(py);
                result.set_item("min_lat", min_lat)?;
                result.set_item("max_lat", max_lat)?;
                result.set_item("min_lon", min_lon)?;
                result.set_item("max_lon", max_lon)?;
                Ok(result.into())
            }
            None => Ok(py.None()),
        }
    }

    /// Get the centroid (center point) of nodes in the current selection.
    ///
    /// Calculates the average latitude and longitude of all nodes
    /// in the current selection.
    ///
    /// Args:
    ///     lat_field: Name of the latitude property (default: 'latitude')
    ///     lon_field: Name of the longitude property (default: 'longitude')
    ///
    /// Returns:
    ///     Dictionary with 'latitude' and 'longitude',
    ///     or None if no valid coordinates found
    ///
    /// Example:
    ///     ```python
    ///     center = graph.type_filter('Discovery').get_centroid()
    ///     print(f"Center: {center['latitude']}, {center['longitude']}")
    ///     ```
    #[pyo3(signature = (lat_field=None, lon_field=None))]
    fn get_centroid(
        &self,
        py: Python<'_>,
        lat_field: Option<&str>,
        lon_field: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let lat_field = lat_field.unwrap_or("latitude");
        let lon_field = lon_field.unwrap_or("longitude");

        match spatial::calculate_centroid(&self.inner, &self.selection, lat_field, lon_field) {
            Some((lat, lon)) => {
                let result = PyDict::new(py);
                result.set_item("latitude", lat)?;
                result.set_item("longitude", lon)?;
                Ok(result.into())
            }
            None => Ok(py.None()),
        }
    }

    /// Calculate the centroid (center point) of a WKT geometry string.
    ///
    /// Parses a WKT geometry (POLYGON, POINT, LINESTRING, etc.) and returns
    /// its centroid coordinates.
    ///
    /// Args:
    ///     wkt_string: A WKT geometry string (e.g., 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))')
    ///
    /// Returns:
    ///     Dictionary with 'latitude' (y) and 'longitude' (x) of the centroid,
    ///     or None if the geometry could not be parsed
    ///
    /// Example:
    ///     ```python
    ///     centroid = graph.wkt_centroid('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))')
    ///     print(f"Center: {centroid['latitude']}, {centroid['longitude']}")
    ///     # Output: Center: 0.5, 0.5
    ///     ```
    fn wkt_centroid(&self, py: Python<'_>, wkt_string: &str) -> PyResult<Py<PyAny>> {
        match spatial::wkt_centroid(wkt_string) {
            Ok((lat, lon)) => {
                let result = PyDict::new(py);
                result.set_item("latitude", lat)?;
                result.set_item("longitude", lon)?;
                Ok(result.into())
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to calculate centroid: {}",
                e
            ))),
        }
    }
}
