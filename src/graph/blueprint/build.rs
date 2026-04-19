//! Blueprint build orchestrator: JSON + CSVs → populated `DirGraph`.
//!
//! Phase order mirrors the Python loader:
//!   1. Manual nodes — types without a CSV, synthesised from FK values
//!      referring to that type.
//!   2. Core nodes — top-level node types with CSVs.
//!   3. Sub-nodes — types declared inside a parent spec's `sub_nodes`.
//!   4. FK edges — single-column foreign keys on node CSVs (plus
//!      implicit `parent` → `OF_{PARENT}` edges).
//!   5. Junction edges — many-to-many CSVs with two FK columns + optional
//!      property columns.

use super::csv_loader::{map_blueprint_type, read_csv_raw, typed_dataframe, RawCsv};
use super::filter::apply_filter;
use super::geometry::{convert_geojson, has_spatial_properties, spatial_targets};
use super::schema::{Blueprint, NodeSpec};
use super::timeseries as ts;
use crate::datatypes::values::DataFrame;
use crate::graph::mutation::maintain;
use crate::graph::schema::{DirGraph, SpatialConfig};
use indexmap::IndexMap;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

pub struct BuildOptions {
    #[allow(dead_code)]
    pub verbose: bool,
}

pub struct BuildReport {
    pub nodes_by_type: BTreeMap<String, usize>,
    pub edges_by_type: BTreeMap<String, usize>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

pub fn build(
    graph: &mut DirGraph,
    blueprint: Blueprint,
    blueprint_dir: &Path,
    opts: &BuildOptions,
) -> Result<BuildReport, String> {
    let root = blueprint
        .settings
        .input_root
        .as_deref()
        .map(|r| {
            if Path::new(r).is_absolute() {
                PathBuf::from(r)
            } else {
                blueprint_dir.join(r)
            }
        })
        .unwrap_or_else(|| blueprint_dir.to_path_buf());

    let mut report = BuildReport {
        nodes_by_type: BTreeMap::new(),
        edges_by_type: BTreeMap::new(),
        warnings: Vec::new(),
        errors: Vec::new(),
    };

    let (core_specs, sub_specs) = collect_specs(&blueprint.nodes);

    // Phase 0: pre-parse all distinct CSV paths in parallel so later serial
    // phases can hit the cache without blocking on disk I/O. We walk the
    // core + sub specs for their `csv` field and the junction edges for their
    // edge CSVs.
    let csv_cache: CsvCache = CsvCache::default();
    let mut all_csv_paths: Vec<String> = Vec::new();
    for s in core_specs.iter().chain(sub_specs.iter()) {
        if let Some(p) = s.spec.csv.as_deref() {
            all_csv_paths.push(p.to_string());
        }
        for (_, j) in &s.spec.connections.junction_edges {
            all_csv_paths.push(j.csv.clone());
        }
    }
    all_csv_paths.sort();
    all_csv_paths.dedup();
    parse_in_parallel(&all_csv_paths, &root, &csv_cache);

    // Phase 1: manual nodes.
    load_manual_nodes(graph, &core_specs, &sub_specs, &root, &mut report)?;

    load_node_specs(
        graph,
        &core_specs,
        &root,
        &csv_cache,
        &mut report,
        opts,
        "core nodes",
    )?;
    load_node_specs(
        graph,
        &sub_specs,
        &root,
        &csv_cache,
        &mut report,
        opts,
        "sub-nodes",
    )?;

    // Register parent types for sub-nodes
    for sub in &sub_specs {
        if let Some(parent) = &sub.parent {
            if graph.type_indices.contains_key(&sub.node_type)
                && graph.type_indices.contains_key(parent)
            {
                graph
                    .parent_types
                    .insert(sub.node_type.clone(), parent.clone());
            }
        }
    }

    // Phase 4: FK edges
    let all_specs: Vec<&FlatSpec> = core_specs.iter().chain(sub_specs.iter()).collect();
    load_fk_edges(graph, &all_specs, &root, &csv_cache, &mut report, opts)?;

    // Phase 5: junction edges
    load_junction_edges(graph, &all_specs, &root, &csv_cache, &mut report, opts)?;

    Ok(report)
}

// ─── Spec flattening ──────────────────────────────────────────────────────

/// Flattened view of one node spec with parent info carried along.
pub struct FlatSpec {
    pub node_type: String,
    pub spec: NodeSpec,
    pub parent: Option<String>,
    pub is_manual: bool,
}

fn collect_specs(nodes: &IndexMap<String, NodeSpec>) -> (Vec<FlatSpec>, Vec<FlatSpec>) {
    let mut core = Vec::new();
    let mut subs = Vec::new();
    for (name, spec) in nodes {
        let is_manual = spec.csv.is_none();
        core.push(FlatSpec {
            node_type: name.clone(),
            spec: clone_without_subs(spec),
            parent: None,
            is_manual,
        });
        for (sub_name, sub_spec) in &spec.sub_nodes {
            // Sub-nodes keep their raw `parent` field untouched — the
            // enclosing type name is recorded on `FlatSpec.parent` so we
            // can call `set_parent_type` without also generating an
            // implicit OF_PARENT edge (that is reserved for top-level
            // specs that explicitly declare `parent` + `parent_fk`).
            let sub_clone = clone_without_subs(sub_spec);
            subs.push(FlatSpec {
                node_type: sub_name.clone(),
                spec: sub_clone,
                parent: Some(name.clone()),
                is_manual: false,
            });
        }
    }
    (core, subs)
}

fn clone_without_subs(spec: &NodeSpec) -> NodeSpec {
    NodeSpec {
        csv: spec.csv.clone(),
        pk: spec.pk.clone(),
        title: spec.title.clone(),
        parent: spec.parent.clone(),
        parent_fk: spec.parent_fk.clone(),
        properties: spec.properties.clone(),
        skipped: spec.skipped.clone(),
        filter: spec.filter.clone(),
        connections: super::schema::Connections {
            fk_edges: spec
                .connections
                .fk_edges
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        super::schema::FkEdge {
                            target: v.target.clone(),
                            fk: v.fk.clone(),
                        },
                    )
                })
                .collect(),
            junction_edges: spec
                .connections
                .junction_edges
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        super::schema::JunctionEdge {
                            csv: v.csv.clone(),
                            source_fk: v.source_fk.clone(),
                            target: v.target.clone(),
                            target_fk: v.target_fk.clone(),
                            properties: v.properties.clone(),
                            property_types: v.property_types.clone(),
                        },
                    )
                })
                .collect(),
        },
        sub_nodes: IndexMap::new(),
        timeseries: spec
            .timeseries
            .as_ref()
            .map(|t| super::schema::TimeseriesSpec {
                time_key: match &t.time_key {
                    super::schema::TimeKey::Single(s) => super::schema::TimeKey::Single(s.clone()),
                    super::schema::TimeKey::Composite(m) => {
                        super::schema::TimeKey::Composite(m.clone())
                    }
                },
                channels: t.channels.clone(),
                resolution: t.resolution.clone(),
                units: t.units.clone(),
            }),
    }
}

// ─── CSV cache ────────────────────────────────────────────────────────────

/// Cache of raw CSVs keyed by relative path. Populated in parallel at the
/// start of the build (see `parse_in_parallel`) so serial phases that read
/// the same CSV (e.g. node load + FK edges + junction edges) never block
/// on disk.
#[derive(Default)]
struct CsvCache {
    inner: std::sync::Mutex<HashMap<String, std::sync::Arc<RawCsv>>>,
}

impl CsvCache {
    fn get(&self, root: &Path, rel: &str) -> Result<std::sync::Arc<RawCsv>, String> {
        {
            let guard = self.inner.lock().unwrap();
            if let Some(hit) = guard.get(rel) {
                return Ok(hit.clone());
            }
        }
        let full = root.join(rel);
        let raw = read_csv_raw(&full)?;
        let arc = std::sync::Arc::new(raw);
        self.inner
            .lock()
            .unwrap()
            .insert(rel.to_string(), arc.clone());
        Ok(arc)
    }

    fn insert(&self, rel: &str, raw: RawCsv) {
        self.inner
            .lock()
            .unwrap()
            .insert(rel.to_string(), std::sync::Arc::new(raw));
    }
}

/// Parse all given CSV paths in parallel, populating the cache. Failures
/// are silently skipped — the caller will see the `Err` again when it tries
/// to look up that path serially (and can emit a targeted error then).
fn parse_in_parallel(paths: &[String], root: &Path, cache: &CsvCache) {
    use rayon::prelude::*;
    paths.par_iter().for_each(|rel| {
        let full = root.join(rel);
        if let Ok(raw) = read_csv_raw(&full) {
            cache.insert(rel, raw);
        }
    });
}

// ─── Phase 1: manual nodes ────────────────────────────────────────────────

fn load_manual_nodes(
    graph: &mut DirGraph,
    core: &[FlatSpec],
    subs: &[FlatSpec],
    root: &Path,
    report: &mut BuildReport,
) -> Result<(), String> {
    let manual: Vec<&FlatSpec> = core.iter().filter(|s| s.is_manual).collect();
    if manual.is_empty() {
        return Ok(());
    }

    for ms in &manual {
        let mut distinct: HashSet<String> = HashSet::new();
        // Scan every spec's fk_edges for targets pointing at this manual type.
        for spec in core.iter().chain(subs.iter()) {
            let Some(csv) = spec.spec.csv.as_deref() else {
                continue;
            };
            for (_, edge) in &spec.spec.connections.fk_edges {
                if edge.target != ms.node_type {
                    continue;
                }
                let full = root.join(csv);
                let raw = match read_csv_raw(&full) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                if let Some(fk_idx) = raw.col_index(&edge.fk) {
                    for (r, row) in raw.rows.iter().enumerate() {
                        if raw.nulls[r][fk_idx] {
                            continue;
                        }
                        let trimmed = row[fk_idx].trim();
                        if !trimmed.is_empty() {
                            distinct.insert(trimmed.to_string());
                        }
                    }
                }
            }
        }

        if distinct.is_empty() {
            continue;
        }

        let pk = ms.spec.pk.clone().unwrap_or_else(|| "name".to_string());
        let title = ms.spec.title.clone().unwrap_or_else(|| pk.clone());

        // Build a tiny single-column (or two-column) DataFrame by hand.
        let mut df = DataFrame::new(Vec::new());
        let values: Vec<String> = distinct.into_iter().collect();
        let col_type_strings = vec![Some(String::from("string")); values.len()]
            .iter()
            .all(|_| true);
        let _ = col_type_strings; // silence unused
        let data = crate::datatypes::values::ColumnData::String(
            values.iter().cloned().map(Some).collect(),
        );
        df.add_column(
            pk.clone(),
            crate::datatypes::values::ColumnType::String,
            data,
        )
        .map_err(|e| format!("manual nodes: {}", e))?;
        if title != pk {
            let data2 = crate::datatypes::values::ColumnData::String(
                values.into_iter().map(Some).collect(),
            );
            df.add_column(
                title.clone(),
                crate::datatypes::values::ColumnType::String,
                data2,
            )
            .map_err(|e| format!("manual nodes: {}", e))?;
        }

        let title_field = if title != pk {
            Some(title.clone())
        } else {
            None
        };
        let result = maintain::add_nodes(graph, df, ms.node_type.clone(), pk, title_field, None)
            .map_err(|e| format!("manual nodes '{}': {}", ms.node_type, e))?;

        let count = result.nodes_created + result.nodes_updated;
        report.nodes_by_type.insert(ms.node_type.clone(), count);
    }

    Ok(())
}

// ─── Phase 2 + 3: node loading ────────────────────────────────────────────

fn load_node_specs(
    graph: &mut DirGraph,
    specs: &[FlatSpec],
    root: &Path,
    cache: &CsvCache,
    report: &mut BuildReport,
    _opts: &BuildOptions,
    _phase_name: &str,
) -> Result<(), String> {
    for spec in specs {
        if spec.is_manual {
            continue;
        }
        let Some(csv_rel) = spec.spec.csv.as_deref() else {
            continue;
        };
        let raw_rc = match cache.get(root, csv_rel) {
            Ok(r) => r,
            Err(e) => {
                report.errors.push(format!("[{}] {}", spec.node_type, e));
                continue;
            }
        };
        let mut raw: RawCsv = (*raw_rc).clone_raw();

        // Apply filter
        if !spec.spec.filter.is_empty() {
            apply_filter(&mut raw, &spec.spec.filter);
        }

        // Apply timeseries zero-row filter
        if let Some(tspec) = &spec.spec.timeseries {
            ts::drop_zero_time_components(&mut raw, tspec);
        }

        // Handle pk: "auto"
        let pk = spec.spec.pk.clone().unwrap_or_else(|| "id".to_string());
        let (pk, synth_pk_values) = if pk == "auto" {
            let synth = format!("_{}_id", spec.node_type);
            let n = raw.row_count();
            let values: Vec<String> = (1..=n).map(|i| i.to_string()).collect();
            (synth, Some(values))
        } else {
            (pk, None)
        };
        if let Some(vals) = &synth_pk_values {
            // Append synthesised column to raw.
            raw.headers.push(pk.clone());
            for (r, row) in raw.rows.iter_mut().enumerate() {
                row.push(vals[r].clone());
                raw.nulls[r].push(false);
            }
        }

        let title_field = spec.spec.title.clone().unwrap_or_else(|| pk.clone());

        // Geometry conversion (GeoJSON → WKT + centroid, in-place on raw)
        let has_geo = has_spatial_properties(&spec.spec.properties);
        let targets = if has_geo {
            let t = spatial_targets(&spec.spec.properties);
            convert_geojson(&mut raw, &t)?;
            Some(t)
        } else {
            None
        };

        // Resolve timeseries (rename handled in-place when we read the channel
        // columns; we do NOT rename raw.headers because that would break
        // downstream column lookups.)
        let ts_resolved = if let Some(tspec) = &spec.spec.timeseries {
            let resolved = ts::resolve(tspec, &raw)?;
            Some(resolved)
        } else {
            None
        };

        // If this spec is a timeseries type, node rows must be deduped by pk
        // (one node per unique carrier; the timeseries data carries the per-row
        // values). Build a deduplicated view used only for the node DataFrame.
        let raw_for_nodes = if ts_resolved.is_some() {
            dedupe_by_pk(&raw, &pk)
        } else {
            raw.clone_raw()
        };

        // Decide which columns to type and keep
        let skip_set: HashSet<&String> = spec.spec.skipped.iter().collect();
        let ts_excluded: HashSet<String> = ts_resolved
            .as_ref()
            .map(|r| r.excluded_columns.iter().cloned().collect())
            .unwrap_or_default();
        let geometry_passthrough: HashSet<String> = HashSet::from_iter(["_geometry".to_string()]);

        // Also skip the parent_fk column if it isn't a real property. Python
        // loader does this whenever `parent_fk` is set, regardless of whether
        // `parent` is also set — mirror that.
        let parent_fk_skip: HashSet<String> = match &spec.spec.parent_fk {
            Some(pfk) if !spec.spec.properties.contains_key(pfk) => {
                HashSet::from_iter([pfk.clone()])
            }
            _ => HashSet::new(),
        };

        // Build declared types map (only columns present in the CSV headers).
        let mut declared: HashMap<String, String> = HashMap::new();
        for (col, ty) in &spec.spec.properties {
            // Non-spatial, non-virtual types only
            if map_blueprint_type(ty).is_some() {
                declared.insert(col.clone(), ty.clone());
            }
        }
        // Force known ID columns to Int64 when possible (mirrors pandas Int64 coercion)
        // — if the pk looks like an integer, declare it so.
        // (Safe: declared has higher precedence than inference, but we don't
        // override user-supplied types.)
        let keep: Vec<String> = raw
            .headers
            .iter()
            .filter(|h| {
                !skip_set.contains(h)
                    && !ts_excluded.contains(h.as_str())
                    && !geometry_passthrough.contains(h.as_str())
                    && !parent_fk_skip.contains(h.as_str())
                    // Never drop pk or title
                    || *h == &pk
                    || *h == &title_field
            })
            .cloned()
            .collect();

        // Dedupe keep list while preserving order
        let mut seen = HashSet::new();
        let keep: Vec<String> = keep
            .into_iter()
            .filter(|h| seen.insert(h.clone()))
            .collect();

        let df = typed_dataframe(&raw_for_nodes, &keep, &declared)?;

        let title_arg = if title_field != pk {
            Some(title_field.clone())
        } else {
            None
        };
        let result = maintain::add_nodes(
            graph,
            df,
            spec.node_type.clone(),
            pk.clone(),
            title_arg,
            None,
        )
        .map_err(|e| format!("add_nodes '{}': {}", spec.node_type, e))?;

        let count = result.nodes_created + result.nodes_updated;
        *report
            .nodes_by_type
            .entry(spec.node_type.clone())
            .or_insert(0) += count;

        // Spatial config
        if has_geo {
            let tgt = targets.unwrap_or_default();
            let mut cfg = SpatialConfig {
                geometry: tgt.wkt,
                ..Default::default()
            };
            if let (Some(lat), Some(lon)) = (tgt.lat, tgt.lon) {
                cfg.location = Some((lat, lon));
            }
            graph.spatial_configs.insert(spec.node_type.clone(), cfg);
        }

        // Timeseries
        if let Some(resolved) = ts_resolved {
            apply_timeseries(graph, &spec.node_type, &pk, &raw, &resolved)?;
        }
    }

    Ok(())
}

fn apply_timeseries(
    graph: &mut DirGraph,
    node_type: &str,
    pk_col: &str,
    raw: &RawCsv,
    resolved: &ts::ResolvedTimeseries,
) -> Result<(), String> {
    let per_node = ts::build_node_timeseries(raw, pk_col, resolved)?;

    graph.build_id_index(node_type);
    for (key_str, node_ts) in per_node {
        let str_val = crate::datatypes::values::Value::String(key_str.clone());
        let node_idx = graph
            .lookup_by_id_normalized(node_type, &str_val)
            .or_else(|| {
                key_str.parse::<i64>().ok().and_then(|i| {
                    graph.lookup_by_id_normalized(
                        node_type,
                        &crate::datatypes::values::Value::Int64(i),
                    )
                })
            });
        let Some(idx) = node_idx else { continue };
        graph.timeseries_store.insert(idx.index(), node_ts);
    }

    let merged = ts::merge_config(graph.timeseries_configs.get(node_type), resolved);
    graph
        .timeseries_configs
        .insert(node_type.to_string(), merged);
    Ok(())
}

// ─── Phase 4: FK edges ────────────────────────────────────────────────────

fn load_fk_edges(
    graph: &mut DirGraph,
    specs: &[&FlatSpec],
    root: &Path,
    cache: &CsvCache,
    report: &mut BuildReport,
    _opts: &BuildOptions,
) -> Result<(), String> {
    for spec in specs {
        let Some(csv_rel) = spec.spec.csv.as_deref() else {
            continue;
        };

        // Build the effective fk_edges map: blueprint's + implicit parent edge.
        let mut fk_edges: IndexMap<String, super::schema::FkEdge> = spec
            .spec
            .connections
            .fk_edges
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    super::schema::FkEdge {
                        target: v.target.clone(),
                        fk: v.fk.clone(),
                    },
                )
            })
            .collect();
        if let (Some(parent_type), Some(parent_fk)) = (&spec.spec.parent, &spec.spec.parent_fk) {
            let edge_type = format!("OF_{}", parent_type.to_uppercase());
            fk_edges.entry(edge_type).or_insert(super::schema::FkEdge {
                target: parent_type.clone(),
                fk: parent_fk.clone(),
            });
        }
        if fk_edges.is_empty() {
            continue;
        }

        let raw_rc = match cache.get(root, csv_rel) {
            Ok(r) => r,
            Err(_) => continue, // already reported in node phase
        };
        let mut raw: RawCsv = (*raw_rc).clone_raw();

        // Apply same filter as node loading
        if !spec.spec.filter.is_empty() {
            apply_filter(&mut raw, &spec.spec.filter);
        }
        if let Some(tspec) = &spec.spec.timeseries {
            ts::drop_zero_time_components(&mut raw, tspec);
        }

        // Handle pk: auto (regenerate same synthesised IDs)
        let raw_pk = spec.spec.pk.clone().unwrap_or_else(|| "id".to_string());
        let pk = if raw_pk == "auto" {
            let synth = format!("_{}_id", spec.node_type);
            let n = raw.row_count();
            let values: Vec<String> = (1..=n).map(|i| i.to_string()).collect();
            raw.headers.push(synth.clone());
            for (r, row) in raw.rows.iter_mut().enumerate() {
                row.push(values[r].clone());
                raw.nulls[r].push(false);
            }
            synth
        } else {
            raw_pk
        };

        for (edge_type, edge) in &fk_edges {
            if raw.col_index(&edge.fk).is_none() {
                report.errors.push(format!(
                    "[{}] FK column '{}' not found for edge {}",
                    spec.node_type, edge.fk, edge_type
                ));
                continue;
            }

            // Build a 2-column DataFrame (pk, fk). When pk == fk (self-reference
            // from the sub-node pattern), synthesise a _target_{col}.
            let pk_idx = raw
                .col_index(&pk)
                .ok_or_else(|| format!("pk column '{}' not found for edge {}", pk, edge_type))?;
            let fk_idx = raw
                .col_index(&edge.fk)
                .ok_or_else(|| format!("fk column '{}' not found", edge.fk))?;

            let (target_col, pk_column, fk_column) = if pk == edge.fk {
                let pk_col = raw
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(r, row)| {
                        if raw.nulls[r][pk_idx] {
                            None
                        } else {
                            Some(row[pk_idx].clone())
                        }
                    })
                    .collect::<Vec<_>>();
                let fk_col: Vec<Option<String>> = pk_col.clone();
                (format!("_target_{}", edge.fk), pk_col, fk_col)
            } else {
                let pk_col = raw
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(r, row)| {
                        if raw.nulls[r][pk_idx] {
                            None
                        } else {
                            Some(row[pk_idx].clone())
                        }
                    })
                    .collect::<Vec<_>>();
                let fk_col: Vec<Option<String>> = raw
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(r, row)| {
                        if raw.nulls[r][fk_idx] {
                            None
                        } else {
                            Some(row[fk_idx].clone())
                        }
                    })
                    .collect::<Vec<_>>();
                (edge.fk.clone(), pk_col, fk_col)
            };

            // Drop rows where the target id is null (source null is handled by
            // the batch processor as a skipped row — matches Python loader at
            // loader.py:468 where only fk_col is dropna'd).
            let mut src_vals: Vec<Option<String>> = Vec::with_capacity(pk_column.len());
            let mut tgt_vals: Vec<Option<String>> = Vec::with_capacity(pk_column.len());
            for (s, t) in pk_column.iter().zip(fk_column.iter()) {
                if t.is_none() {
                    continue;
                }
                src_vals.push(s.clone());
                tgt_vals.push(t.clone());
            }
            if src_vals.is_empty() {
                continue;
            }

            let df = build_edge_df(&pk, &target_col, src_vals, tgt_vals)?;
            let count = connect(
                graph,
                df,
                edge_type,
                &spec.node_type,
                &pk,
                &edge.target,
                &target_col,
                report,
            )?;
            *report.edges_by_type.entry(edge_type.clone()).or_insert(0) += count;
        }
    }
    Ok(())
}

fn build_edge_df(
    src_name: &str,
    tgt_name: &str,
    src: Vec<Option<String>>,
    tgt: Vec<Option<String>>,
) -> Result<DataFrame, String> {
    // Decide column types: try i64, fall back to string.
    let src_type = infer_id_type(&src);
    let tgt_type = infer_id_type(&tgt);
    let mut df = DataFrame::new(Vec::new());
    add_id_column(&mut df, src_name, src, src_type)?;
    add_id_column(&mut df, tgt_name, tgt, tgt_type)?;
    Ok(df)
}

fn infer_id_type(vals: &[Option<String>]) -> crate::datatypes::values::ColumnType {
    let mut all_int = true;
    for v in vals {
        let Some(s) = v else { continue };
        let t = s.trim();
        if t.is_empty() {
            continue;
        }
        if t.parse::<i64>().is_ok() {
            continue;
        }
        if let Ok(f) = t.parse::<f64>() {
            if f.is_finite() && f.fract() == 0.0 {
                continue;
            }
        }
        all_int = false;
        break;
    }
    if all_int {
        crate::datatypes::values::ColumnType::Int64
    } else {
        crate::datatypes::values::ColumnType::String
    }
}

fn add_id_column(
    df: &mut DataFrame,
    name: &str,
    vals: Vec<Option<String>>,
    col_type: crate::datatypes::values::ColumnType,
) -> Result<(), String> {
    use crate::datatypes::values::{ColumnData, ColumnType};
    let data = match col_type {
        ColumnType::Int64 => {
            let ints: Vec<Option<i64>> = vals
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|s| {
                        let t = s.trim();
                        if t.is_empty() {
                            None
                        } else if let Ok(i) = t.parse::<i64>() {
                            Some(i)
                        } else if let Ok(f) = t.parse::<f64>() {
                            if f.is_finite()
                                && f.fract() == 0.0
                                && f >= i64::MIN as f64
                                && f <= i64::MAX as f64
                            {
                                Some(f as i64)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                })
                .collect();
            ColumnData::Int64(ints)
        }
        _ => ColumnData::String(
            vals.into_iter()
                .map(|v| v.and_then(|s| if s.is_empty() { None } else { Some(s) }))
                .collect(),
        ),
    };
    df.add_column(name.to_string(), col_type, data)
}

#[allow(clippy::too_many_arguments)]
fn connect(
    graph: &mut DirGraph,
    df: DataFrame,
    connection_type: &str,
    source_type: &str,
    source_id_field: &str,
    target_type: &str,
    target_id_field: &str,
    report: &mut BuildReport,
) -> Result<usize, String> {
    match maintain::add_connections(
        graph,
        df,
        connection_type.to_string(),
        source_type.to_string(),
        source_id_field.to_string(),
        target_type.to_string(),
        target_id_field.to_string(),
        None,
        None,
        None,
    ) {
        Ok(r) => {
            if r.connections_skipped > 0 {
                let detail = r.errors.join("; ");
                report.warnings.push(format!(
                    "[{}] -[{}]-> {}: {} skipped ({})",
                    source_type, connection_type, target_type, r.connections_skipped, detail
                ));
            }
            Ok(r.connections_created)
        }
        Err(e) => {
            report
                .errors
                .push(format!("[{}] edge {}: {}", source_type, connection_type, e));
            Ok(0)
        }
    }
}

// ─── Phase 5: junction edges ──────────────────────────────────────────────

fn load_junction_edges(
    graph: &mut DirGraph,
    specs: &[&FlatSpec],
    root: &Path,
    _cache: &CsvCache,
    report: &mut BuildReport,
    _opts: &BuildOptions,
) -> Result<(), String> {
    for spec in specs {
        for (edge_type, junc) in &spec.spec.connections.junction_edges {
            let full = root.join(&junc.csv);
            let raw = match read_csv_raw(&full) {
                Ok(r) => r,
                Err(e) => {
                    report.errors.push(format!(
                        "[{}] junction {}: {}",
                        spec.node_type, edge_type, e
                    ));
                    continue;
                }
            };

            // Determine which columns to keep: source FK, target FK, declared properties
            let mut keep: Vec<String> = vec![junc.source_fk.clone(), junc.target_fk.clone()];
            for p in &junc.properties {
                if raw.col_index(p).is_some() && !keep.contains(p) {
                    keep.push(p.clone());
                }
            }

            let mut declared: HashMap<String, String> = HashMap::new();
            for (col, ty) in &junc.property_types {
                if map_blueprint_type(ty).is_some() {
                    declared.insert(col.clone(), ty.clone());
                }
            }

            let df = match typed_dataframe(&raw, &keep, &declared) {
                Ok(d) => d,
                Err(e) => {
                    report.errors.push(format!(
                        "[{}] junction {}: {}",
                        spec.node_type, edge_type, e
                    ));
                    continue;
                }
            };

            let count = connect(
                graph,
                df,
                edge_type,
                &spec.node_type,
                &junc.source_fk,
                &junc.target,
                &junc.target_fk,
                report,
            )?;
            *report.edges_by_type.entry(edge_type.clone()).or_insert(0) += count;
        }
    }
    Ok(())
}

// ─── Helpers ──────────────────────────────────────────────────────────────

impl RawCsv {
    fn clone_raw(&self) -> RawCsv {
        RawCsv {
            headers: self.headers.clone(),
            rows: self.rows.clone(),
            nulls: self.nulls.clone(),
        }
    }
}

/// Keep only the first row per unique pk value. Used for timeseries specs:
/// one node per carrier, time samples stored separately.
fn dedupe_by_pk(raw: &RawCsv, pk_col: &str) -> RawCsv {
    let Some(idx) = raw.col_index(pk_col) else {
        return raw.clone_raw();
    };
    let mut seen: HashSet<String> = HashSet::new();
    let mut new_rows = Vec::new();
    let mut new_nulls = Vec::new();
    for r in 0..raw.row_count() {
        if raw.nulls[r][idx] {
            new_rows.push(raw.rows[r].clone());
            new_nulls.push(raw.nulls[r].clone());
            continue;
        }
        let key = raw.rows[r][idx].clone();
        if seen.insert(key) {
            new_rows.push(raw.rows[r].clone());
            new_nulls.push(raw.nulls[r].clone());
        }
    }
    RawCsv {
        headers: raw.headers.clone(),
        rows: new_rows,
        nulls: new_nulls,
    }
}
