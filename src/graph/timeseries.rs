// src/graph/timeseries.rs
//
// Per-node timeseries storage: a sorted time index with multiple aligned float channels.
// Follows the embeddings pattern — data lives in DirGraph::timeseries_store,
// separate from NodeData.properties.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-node-type timeseries configuration. Declares the resolution (time granularity),
/// known channels with optional units, and bin semantics.
/// Persisted in FileMetadata JSON (like spatial_configs).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TimeseriesConfig {
    /// Time resolution: "year", "month", or "day".
    /// Determines key depth (year=1, month=2, day=3) and date-string validation.
    #[serde(default)]
    pub resolution: String,
    /// Known channel names (informational, for introspection).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<String>,
    /// Channel name → unit string (e.g. "MSm3", "°C", "bar").
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub units: HashMap<String, String>,
    /// What values represent: "total" (sum over bin), "mean" (average over bin),
    /// or "sample" (point-in-time reading). None = unspecified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_type: Option<String>,
}

/// A single node's timeseries data: a sorted time index with multiple value channels.
/// Stored in `DirGraph::timeseries_store`, keyed by `NodeIndex.index()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTimeseries {
    /// Sorted composite time keys. Key depth matches the resolution:
    /// year → `[2020]`, month → `[2020, 2]`, day → `[2020, 2, 15]`.
    pub keys: Vec<Vec<i64>>,
    /// Channel name → values array (must have same length as `keys`). NaN = missing.
    pub channels: HashMap<String, Vec<f64>>,
}

// ─── Resolution helpers ─────────────────────────────────────────────────────

/// Map resolution string to key depth. Returns Err for unknown resolutions.
pub fn resolution_depth(resolution: &str) -> Result<usize, String> {
    match resolution {
        "year" => Ok(1),
        "month" => Ok(2),
        "day" => Ok(3),
        "hour" => Ok(4),
        "minute" => Ok(5),
        _ => Err(format!(
            "Unknown timeseries resolution '{}'. Expected: year, month, day, hour, minute",
            resolution
        )),
    }
}

/// Map key depth back to a resolution string. Used for auto-detection.
pub fn depth_to_resolution(depth: usize) -> Result<&'static str, String> {
    match depth {
        1 => Ok("year"),
        2 => Ok("month"),
        3 => Ok("day"),
        4 => Ok("hour"),
        5 => Ok("minute"),
        _ => Err(format!(
            "Invalid key depth {}. Expected 1-5 (year through minute)",
            depth
        )),
    }
}

/// Validate a resolution string.
pub fn validate_resolution(resolution: &str) -> Result<(), String> {
    resolution_depth(resolution).map(|_| ())
}

// ─── Date-string parsing ────────────────────────────────────────────────────

/// Parse a date string into a composite key.
///
/// Supported formats:
/// - `"2020"` → `[2020]`
/// - `"2020-2"` → `[2020, 2]`
/// - `"2020-2-15"` → `[2020, 2, 15]`
/// - `"2020-2-15 10"` → `[2020, 2, 15, 10]`
/// - `"2020-2-15 10:30"` → `[2020, 2, 15, 10, 30]`
pub fn parse_date_string(s: &str) -> Result<Vec<i64>, String> {
    let trimmed = s.trim();

    // Split into date part and optional time part (separated by space or 'T')
    let (date_part, time_part) = if let Some(idx) = trimmed.find([' ', 'T']) {
        (&trimmed[..idx], Some(trimmed[idx + 1..].trim()))
    } else {
        (trimmed, None)
    };

    // Parse date components: YYYY-M-D
    let date_parts: Vec<&str> = date_part.split('-').collect();
    if date_parts.is_empty() || date_parts.len() > 3 {
        return Err(format!(
            "Invalid date string '{}'. Expected: 'YYYY', 'YYYY-M', or 'YYYY-M-D'",
            s
        ));
    }

    let labels = ["year", "month", "day", "hour", "minute"];
    let mut key = Vec::with_capacity(5);

    for (i, part) in date_parts.iter().enumerate() {
        let val = part
            .parse::<i64>()
            .map_err(|_| format!("Invalid {} '{}' in date string '{}'", labels[i], part, s))?;
        key.push(val);
    }

    // Parse time components: HH:MM (only valid if we have a full date YYYY-M-D)
    if let Some(tp) = time_part {
        if date_parts.len() < 3 {
            return Err(format!(
                "Time component requires a full date (YYYY-M-D) in '{}'",
                s
            ));
        }
        let time_parts: Vec<&str> = tp.split(':').collect();
        if time_parts.is_empty() || time_parts.len() > 2 {
            return Err(format!(
                "Invalid time component in '{}'. Expected: 'HH' or 'HH:MM'",
                s
            ));
        }
        for (j, part) in time_parts.iter().enumerate() {
            let idx = 3 + j; // hour=3, minute=4
            let val = part.parse::<i64>().map_err(|_| {
                format!("Invalid {} '{}' in date string '{}'", labels[idx], part, s)
            })?;
            key.push(val);
        }
    }

    if key.len() > 5 {
        return Err(format!(
            "Too many components in date string '{}' (max 5: year, month, day, hour, minute)",
            s
        ));
    }

    Ok(key)
}

/// Validate that query depth is compatible with data resolution.
/// For exact lookups (ts_at, ts_delta): query depth must equal data depth.
/// For range queries (ts_sum, etc.): query depth must be ≤ data depth.
pub fn validate_query_depth(
    query_depth: usize,
    data_depth: usize,
    resolution: &str,
    exact: bool,
) -> Result<(), String> {
    if exact {
        if query_depth != data_depth {
            return Err(format!(
                "Exact lookup requires {} date components for '{}' resolution, got {}",
                data_depth, resolution, query_depth
            ));
        }
    } else if query_depth > data_depth {
        let depth_name = match query_depth {
            1 => "year",
            2 => "month",
            3 => "day",
            4 => "hour",
            5 => "minute",
            _ => "unknown",
        };
        return Err(format!(
            "Query precision '{}' (depth {}) exceeds data resolution '{}' (depth {})",
            depth_name, query_depth, resolution, data_depth
        ));
    }
    Ok(())
}

// ─── Lookup helpers ──────────────────────────────────────────────────────────

/// Binary search for an exact composite key. Returns the index if found.
pub fn find_key_index(keys: &[Vec<i64>], target: &[i64]) -> Option<usize> {
    keys.binary_search_by(|k| k.as_slice().cmp(target)).ok()
}

/// Find the slice range `[start_idx, end_idx)` for keys in the inclusive range
/// `[start, end]`. Supports prefix matching: if start/end have fewer components
/// than the keys, pads start with `i64::MIN` and end with `i64::MAX`.
pub fn find_range(
    keys: &[Vec<i64>],
    start: Option<&[i64]>,
    end: Option<&[i64]>,
    key_depth: usize,
) -> (usize, usize) {
    let lo = match start {
        Some(s) => {
            let mut padded = s.to_vec();
            while padded.len() < key_depth {
                padded.push(i64::MIN);
            }
            keys.partition_point(|k| k.as_slice() < padded.as_slice())
        }
        None => 0,
    };
    let hi = match end {
        Some(e) => {
            let mut padded = e.to_vec();
            while padded.len() < key_depth {
                padded.push(i64::MAX);
            }
            keys.partition_point(|k| k.as_slice() <= padded.as_slice())
        }
        None => keys.len(),
    };
    (lo, hi)
}

// ─── Aggregation helpers (operate on slices, skip NaN) ───────────────────────

/// Sum of non-NaN values in the slice.
pub fn ts_sum(values: &[f64]) -> f64 {
    values.iter().filter(|v| v.is_finite()).sum()
}

/// Average of non-NaN values in the slice. Returns NaN if no finite values.
pub fn ts_avg(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in values {
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Minimum of non-NaN values. Returns NaN if no finite values.
pub fn ts_min(values: &[f64]) -> f64 {
    values
        .iter()
        .filter(|v| v.is_finite())
        .copied()
        .fold(f64::INFINITY, f64::min)
}

/// Maximum of non-NaN values. Returns NaN if no finite values.
pub fn ts_max(values: &[f64]) -> f64 {
    values
        .iter()
        .filter(|v| v.is_finite())
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Count of non-NaN values.
pub fn ts_count(values: &[f64]) -> usize {
    values.iter().filter(|v| v.is_finite()).count()
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate that a channel's length matches the time index length.
pub fn validate_channel_length(
    keys_len: usize,
    channel_len: usize,
    channel_name: &str,
) -> Result<(), String> {
    if channel_len != keys_len {
        Err(format!(
            "Channel '{}' has {} values but time index has {} keys",
            channel_name, channel_len, keys_len
        ))
    } else {
        Ok(())
    }
}

/// Validate that keys are sorted in ascending order.
pub fn validate_keys_sorted(keys: &[Vec<i64>]) -> Result<(), String> {
    for w in keys.windows(2) {
        if w[0] >= w[1] {
            return Err(format!(
                "Time index keys are not strictly sorted: {:?} >= {:?}",
                w[0], w[1]
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_depth() {
        assert_eq!(resolution_depth("year").unwrap(), 1);
        assert_eq!(resolution_depth("month").unwrap(), 2);
        assert_eq!(resolution_depth("day").unwrap(), 3);
        assert_eq!(resolution_depth("hour").unwrap(), 4);
        assert_eq!(resolution_depth("minute").unwrap(), 5);
        assert!(resolution_depth("invalid").is_err());
    }

    #[test]
    fn test_depth_to_resolution() {
        assert_eq!(depth_to_resolution(1).unwrap(), "year");
        assert_eq!(depth_to_resolution(2).unwrap(), "month");
        assert_eq!(depth_to_resolution(3).unwrap(), "day");
        assert_eq!(depth_to_resolution(4).unwrap(), "hour");
        assert_eq!(depth_to_resolution(5).unwrap(), "minute");
        assert!(depth_to_resolution(0).is_err());
        assert!(depth_to_resolution(6).is_err());
    }

    #[test]
    fn test_parse_date_string() {
        // Date-only formats
        assert_eq!(parse_date_string("2020").unwrap(), vec![2020]);
        assert_eq!(parse_date_string("2020-2").unwrap(), vec![2020, 2]);
        assert_eq!(parse_date_string("2020-02").unwrap(), vec![2020, 2]);
        assert_eq!(parse_date_string("2020-2-15").unwrap(), vec![2020, 2, 15]);
        // Date+time formats
        assert_eq!(
            parse_date_string("2020-1-15 10").unwrap(),
            vec![2020, 1, 15, 10]
        );
        assert_eq!(
            parse_date_string("2020-01-15 10:30").unwrap(),
            vec![2020, 1, 15, 10, 30]
        );
        // ISO format with T separator
        assert_eq!(
            parse_date_string("2020-01-15T10:30").unwrap(),
            vec![2020, 1, 15, 10, 30]
        );
        // Whitespace tolerance
        assert_eq!(
            parse_date_string("  2020-01-15 10:30  ").unwrap(),
            vec![2020, 1, 15, 10, 30]
        );
        // Error cases
        assert!(parse_date_string("abc").is_err());
        assert!(parse_date_string("2020-abc").is_err());
        // Time without full date
        assert!(parse_date_string("2020-01 10:30").is_err());
    }

    #[test]
    fn test_validate_query_depth() {
        // Range queries: depth ≤ data depth
        assert!(validate_query_depth(1, 2, "month", false).is_ok()); // year on month data
        assert!(validate_query_depth(2, 2, "month", false).is_ok()); // month on month data
        assert!(validate_query_depth(3, 2, "month", false).is_err()); // day on month data

        // Exact queries: depth must equal
        assert!(validate_query_depth(2, 2, "month", true).is_ok());
        assert!(validate_query_depth(1, 2, "month", true).is_err());
    }

    #[test]
    fn test_find_key_index() {
        let keys = vec![vec![2020, 1], vec![2020, 2], vec![2020, 3], vec![2021, 1]];
        assert_eq!(find_key_index(&keys, &[2020, 2]), Some(1));
        assert_eq!(find_key_index(&keys, &[2020, 4]), None);
        assert_eq!(find_key_index(&keys, &[2021, 1]), Some(3));
    }

    #[test]
    fn test_find_range_full_depth() {
        let keys = vec![
            vec![2019, 12],
            vec![2020, 1],
            vec![2020, 2],
            vec![2020, 3],
            vec![2021, 1],
        ];
        // Exact month range: Feb through Apr 2020
        assert_eq!(
            find_range(&keys, Some(&[2020, 2]), Some(&[2020, 3]), 2),
            (2, 4)
        );
        // Single month
        assert_eq!(
            find_range(&keys, Some(&[2020, 2]), Some(&[2020, 2]), 2),
            (2, 3)
        );
    }

    #[test]
    fn test_find_range_prefix() {
        let keys = vec![
            vec![2019, 12],
            vec![2020, 1],
            vec![2020, 2],
            vec![2020, 3],
            vec![2021, 1],
        ];
        // Year prefix on month-depth data: "2020" → all months in 2020
        assert_eq!(find_range(&keys, Some(&[2020]), Some(&[2020]), 2), (1, 4));
        // Year range: 2020-2021
        assert_eq!(find_range(&keys, Some(&[2020]), Some(&[2021]), 2), (1, 5));
        // Everything
        assert_eq!(find_range(&keys, None, None, 2), (0, 5));
        // From 2020 onward
        assert_eq!(find_range(&keys, Some(&[2020]), None, 2), (1, 5));
    }

    #[test]
    fn test_ts_aggregations() {
        let values = vec![1.0, 2.0, 3.0, f64::NAN, 5.0];
        assert_eq!(ts_sum(&values), 11.0);
        assert!((ts_avg(&values) - 2.75).abs() < 1e-10);
        assert_eq!(ts_min(&values), 1.0);
        assert_eq!(ts_max(&values), 5.0);
        assert_eq!(ts_count(&values), 4);
    }

    #[test]
    fn test_ts_empty() {
        let empty: Vec<f64> = vec![];
        assert_eq!(ts_sum(&empty), 0.0);
        assert!(ts_avg(&empty).is_nan());
        assert_eq!(ts_count(&empty), 0);
    }

    #[test]
    fn test_validate_channel_length() {
        assert!(validate_channel_length(5, 5, "oil").is_ok());
        assert!(validate_channel_length(5, 3, "oil").is_err());
    }

    #[test]
    fn test_validate_keys_sorted() {
        assert!(validate_keys_sorted(&[vec![1], vec![2], vec![3]]).is_ok());
        assert!(validate_keys_sorted(&[vec![2020, 1], vec![2020, 2]]).is_ok());
        assert!(validate_keys_sorted(&[vec![2], vec![1]]).is_err());
        assert!(validate_keys_sorted(&[vec![1], vec![1]]).is_err()); // duplicates
    }
}
