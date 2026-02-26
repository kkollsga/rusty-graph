"""Core loading logic for building a KnowledgeGraph from a JSON blueprint."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import kglite


# Blueprint property type -> KGLite column_types value
_TYPE_MAP = {
    "string": "string",
    "int": "integer",
    "float": "float",
    "date": "datetime",
    # "bool" is auto-detected by pandas, no explicit mapping needed
    # spatial types handled separately
}

# Spatial virtual types that need geometry conversion
_SPATIAL_TYPES = {"geometry", "location.lat", "location.lon"}


def from_blueprint(
    blueprint_path: Union[str, Path],
    *,
    verbose: bool = False,
    save: bool = True,
) -> kglite.KnowledgeGraph:
    """Parse a JSON blueprint and build a KnowledgeGraph from CSV files.

    Args:
        blueprint_path: Path to the blueprint JSON file.
        verbose: If True, print detailed progress for every node/edge type.
            When False (default), only warnings and a final summary are printed.
        save: If True and the blueprint specifies an output path, save the graph.

    Returns:
        A populated KnowledgeGraph.

    Raises:
        FileNotFoundError: If the blueprint file is missing.
        ValueError: If the blueprint JSON is malformed.
    """
    blueprint_path = Path(blueprint_path)
    if not blueprint_path.exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")

    with open(blueprint_path) as f:
        raw = json.load(f)

    loader = BlueprintLoader(raw, verbose=verbose)

    # Suppress UserWarning from add_connections — we track skips in the loader
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        graph = loader.build()

    # Always print warnings (regardless of verbose)
    if loader.warnings:
        print(f"\n  {len(loader.warnings)} warning(s):")
        for w in loader.warnings:
            print(f"    [{w['context']}] {w['message']}")

    if loader.errors:
        print(f"\n  {len(loader.errors)} error(s):")
        for err in loader.errors:
            print(f"    [{err['context']}] {err['message']}")

    if save and loader.output_path:
        loader.output_path.parent.mkdir(parents=True, exist_ok=True)
        graph.save(str(loader.output_path))
        if verbose:
            print(f"  Saved to {loader.output_path}")

    return graph


class BlueprintLoader:
    """Stateful loader that processes a blueprint dict into a KnowledgeGraph."""

    def __init__(self, raw: dict[str, Any], verbose: bool = False):
        self.raw = raw
        self.verbose = verbose
        self.graph = kglite.KnowledgeGraph()
        self.errors: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []

        # Parse settings — support both old ("root") and new ("input_root") keys
        settings = raw.get("settings", {})
        self.root = Path(settings.get("input_root", settings.get("root", ".")))

        # Output path: optional output_path + output_file, or legacy "output"
        output_path = settings.get("output_path")
        output_file = settings.get("output_file", settings.get("output"))
        if output_file:
            if output_path:
                self.output_path = (Path(output_path) / output_file).resolve()
            else:
                # Relative to input_root (supports ../ notation)
                self.output_path = (self.root / output_file).resolve()
        else:
            self.output_path = None

        # Node specs keyed by type name
        self.nodes: dict[str, dict[str, Any]] = raw.get("nodes", {})

        # CSV cache: relative_path -> DataFrame
        self._csv_cache: dict[str, pd.DataFrame] = {}

        # Track which node types have been loaded (for edge creation)
        self._loaded_types: set[str] = set()

        # Stats
        self.stats: dict[str, Any] = {
            "nodes_by_type": {},
            "edges_by_type": {},
        }

    def build(self) -> kglite.KnowledgeGraph:
        """Execute the full loading sequence."""
        t0 = time.time()
        if self.verbose:
            print(f"Loading blueprint...")
            print(f"  Root: {self.root}")

        # Collect all node specs (core + sub_nodes) with their parent info
        core_specs, sub_specs = self._collect_specs()

        # Phase 1: Manual nodes (no csv)
        self._load_manual_nodes(core_specs, sub_specs)

        # Phase 2: Core nodes (with csv)
        self._load_nodes(core_specs, phase_name="core nodes")

        # Phase 3: Sub-nodes
        self._load_nodes(sub_specs, phase_name="sub-nodes")

        # Phase 3b: Register parent types for supporting node tiers
        for sub in sub_specs:
            sub_type = sub["_node_type"]
            parent = sub["_parent_type"]
            if sub_type in self._loaded_types:
                self.graph.set_parent_type(sub_type, parent)

        # Phase 4: FK edges
        self._load_fk_edges(core_specs + sub_specs)

        # Phase 5: Junction edges
        self._load_junction_edges(core_specs + sub_specs)

        elapsed = time.time() - t0
        total_nodes = sum(self.stats["nodes_by_type"].values())
        total_edges = sum(self.stats["edges_by_type"].values())
        n_types = len(self.stats["nodes_by_type"])
        e_types = len(self.stats["edges_by_type"])

        # Always print the summary line
        print(
            f"\nDone in {elapsed:.1f}s: "
            f"{total_nodes:,} nodes ({n_types} types), "
            f"{total_edges:,} edges ({e_types} types)"
        )

        return self.graph

    # ── Spec collection ─────────────────────────────────────────────

    def _collect_specs(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Separate core nodes from sub_nodes, annotating each with node_type."""
        core = []
        subs = []
        for node_type, spec in self.nodes.items():
            spec = dict(spec)  # shallow copy
            spec["_node_type"] = node_type
            if spec.get("csv") is not None:
                core.append(spec)
            elif not spec.get("_is_manual"):
                # Nodes without csv are manual — handled separately
                spec["_is_manual"] = True
                core.append(spec)

            # Collect sub_nodes
            for sub_type, sub_spec in spec.get("sub_nodes", {}).items():
                sub = dict(sub_spec)
                sub["_node_type"] = sub_type
                sub["_parent_type"] = node_type
                sub["_parent_pk"] = spec.get("pk", "id")
                subs.append(sub)

        return core, subs

    # ── Phase 1: Manual nodes ───────────────────────────────────────

    def _load_manual_nodes(
        self,
        core_specs: list[dict[str, Any]],
        sub_specs: list[dict[str, Any]],
    ) -> None:
        """Create nodes for types without a CSV, from distinct FK values."""
        all_specs = core_specs + sub_specs
        manual_types = {
            s["_node_type"]: s for s in core_specs if s.get("_is_manual")
        }
        if not manual_types:
            return

        if self.verbose:
            print("  Loading manual nodes...")

        for manual_type, manual_spec in manual_types.items():
            distinct_values: set[Any] = set()

            # Scan all FK edges across all specs to find references to this type
            for spec in all_specs:
                for _edge_type, edge_def in (
                    spec.get("connections", {}).get("fk_edges", {}).items()
                ):
                    if edge_def.get("target") != manual_type:
                        continue
                    csv_path = spec.get("csv")
                    if csv_path is None:
                        continue
                    try:
                        df = self._read_csv(csv_path)
                    except FileNotFoundError:
                        continue
                    fk_col = edge_def["fk"]
                    if fk_col in df.columns:
                        vals = df[fk_col].dropna().unique()
                        distinct_values.update(vals)

            if not distinct_values:
                continue

            pk = manual_spec.get("pk", "name")
            title = manual_spec.get("title", pk)

            # Clean values (strip whitespace, title-case for string values)
            cleaned = []
            for v in distinct_values:
                if isinstance(v, str):
                    cleaned.append(v.strip())
                else:
                    cleaned.append(v)

            manual_df = pd.DataFrame({pk: cleaned})
            if title != pk:
                manual_df[title] = manual_df[pk]

            report = self.graph.add_nodes(
                data=manual_df,
                node_type=manual_type,
                unique_id_field=pk,
                node_title_field=title,
            )
            count = report.get("nodes_created", 0)
            self.stats["nodes_by_type"][manual_type] = count
            self._loaded_types.add(manual_type)
            if self.verbose:
                print(f"    {manual_type}: {count} nodes (from FK values)")

    # ── Phase 2 & 3: Load nodes ─────────────────────────────────────

    def _load_nodes(
        self, specs: list[dict[str, Any]], phase_name: str
    ) -> None:
        """Load nodes from CSV files."""
        loadable = [s for s in specs if not s.get("_is_manual") and s.get("csv")]
        if not loadable:
            return

        if self.verbose:
            print(f"  Loading {phase_name}...")

        for spec in loadable:
            node_type = spec["_node_type"]
            csv_path = spec["csv"]

            try:
                df = self._read_csv(csv_path)
            except FileNotFoundError:
                self._report_error(node_type, f"CSV not found: {csv_path}")
                continue

            # Apply filter
            filt = spec.get("filter")
            if filt:
                df = self._apply_filter(df, filt)

            # Handle pk: "auto"
            pk = spec.get("pk", "id")
            if pk == "auto":
                pk = f"_{node_type}_id"
                df = df.copy()
                df[pk] = range(1, len(df) + 1)

            title = spec.get("title", pk)
            properties = spec.get("properties", {})

            # Handle geometry virtual types
            has_geo = bool(_SPATIAL_TYPES & set(properties.values()))
            if has_geo:
                df = self._convert_geometry(df, properties, node_type)

            # Build column_types for non-spatial, non-bool types
            column_types = self._build_column_types(properties)

            # Build skip_columns: explicit skipped + FK columns + _geometry
            skip = list(spec.get("skipped", []))
            if "_geometry" in df.columns:
                skip.append("_geometry")
            # Also skip the parent_fk column if present (it's used for edges, not properties)
            parent_fk = spec.get("parent_fk")
            if parent_fk and parent_fk not in properties:
                skip.append(parent_fk)

            # Handle timeseries
            ts_param = None
            ts_config = spec.get("timeseries")
            if ts_config:
                ts_param, df = self._prepare_timeseries(
                    df, ts_config, skip, node_type
                )

            # Never skip the pk or title columns — add_nodes needs them
            skip = [
                c for c in skip
                if c != pk and c != title
            ]

            try:
                report = self.graph.add_nodes(
                    data=df,
                    node_type=node_type,
                    unique_id_field=pk,
                    node_title_field=title if title != pk else None,
                    column_types=column_types if column_types else None,
                    skip_columns=skip if skip else None,
                    timeseries=ts_param,
                )
            except Exception as e:
                self._report_error(node_type, f"add_nodes failed: {e}")
                continue

            count = report.get("nodes_created", 0) + report.get(
                "nodes_updated", 0
            )
            self.stats["nodes_by_type"][node_type] = count
            self._loaded_types.add(node_type)

            # Set spatial config if geometry/location types present
            if has_geo:
                self._apply_spatial_config(node_type, properties)

            label = ""
            if ts_config:
                channels = ts_config.get("channels", {})
                label = f" (timeseries: {ts_config.get('resolution', '?')}, {len(channels)} channels)"
            if has_geo:
                label += " (with geometry)"

            if self.verbose:
                print(
                    f"    {node_type}: {count} nodes from {Path(csv_path).name}{label}"
                )

    # ── Phase 4: FK edges ───────────────────────────────────────────

    def _load_fk_edges(self, all_specs: list[dict[str, Any]]) -> None:
        """Create edges from FK columns in source CSVs.

        Also handles core nodes with a ``parent`` key — these get an
        implicit FK edge to their parent type.
        """
        has_edges = any(
            spec.get("connections", {}).get("fk_edges")
            or spec.get("parent")
            for spec in all_specs
        )
        if not has_edges:
            return

        if self.verbose:
            print("  Loading FK edges...")

        for spec in all_specs:
            node_type = spec["_node_type"]
            csv_path = spec.get("csv")
            fk_edges = spec.get("connections", {}).get("fk_edges", {})

            # Core nodes with "parent" key get an implicit FK edge
            parent_type = spec.get("parent")
            parent_fk = spec.get("parent_fk")
            if parent_type and parent_fk:
                edge_type = f"OF_{parent_type.upper()}"
                fk_edges = dict(fk_edges)  # copy to avoid mutating spec
                fk_edges.setdefault(edge_type, {
                    "target": parent_type,
                    "fk": parent_fk,
                })

            if not fk_edges or csv_path is None:
                continue

            try:
                df = self._read_csv(csv_path)
            except FileNotFoundError:
                continue

            # Apply same filter as node loading
            filt = spec.get("filter")
            if filt:
                df = self._apply_filter(df, filt)

            # Apply same timeseries time-component filtering as node loading.
            # Without this, aggregate rows (e.g. month=0) produce edge rows
            # that reference source nodes that were never created.
            ts_config = spec.get("timeseries")
            if ts_config:
                df = self._apply_timeseries_filter(df, ts_config)

            # Handle pk: "auto" — regenerate the same IDs
            pk = spec.get("pk", "id")
            if pk == "auto":
                pk = f"_{node_type}_id"
                df = df.copy()
                df[pk] = range(1, len(df) + 1)

            for edge_type, edge_def in fk_edges.items():
                target_type = edge_def["target"]
                fk_col = edge_def["fk"]

                if fk_col not in df.columns:
                    self._report_error(
                        node_type,
                        f"FK column '{fk_col}' not found for edge {edge_type}",
                    )
                    continue

                # Build edge DataFrame with just the two ID columns
                # Handle case where pk == fk_col (e.g. sub-node FK to parent
                # uses the same column as the sub-node's primary key)
                if pk == fk_col:
                    edge_df = df[[pk]].dropna(subset=[pk]).copy()
                    target_col = f"_target_{fk_col}"
                    edge_df[target_col] = edge_df[pk]
                else:
                    edge_df = df[[pk, fk_col]].dropna(subset=[fk_col]).copy()
                    target_col = fk_col
                if edge_df.empty:
                    continue

                # Coerce float ID columns to int (pandas reads nullable-int as float)
                _coerce_id_columns(edge_df, [pk, target_col])

                try:
                    report = self.graph.add_connections(
                        data=edge_df,
                        connection_type=edge_type,
                        source_type=node_type,
                        source_id_field=pk,
                        target_type=target_type,
                        target_id_field=target_col,
                    )
                except Exception as e:
                    self._report_error(
                        node_type, f"FK edge {edge_type} failed: {e}"
                    )
                    continue

                count = report.get("connections_created", 0)
                skipped = report.get("connections_skipped", 0)
                self.stats["edges_by_type"][edge_type] = (
                    self.stats["edges_by_type"].get(edge_type, 0) + count
                )

                if self.verbose:
                    print(
                        f"    {node_type} -[{edge_type}]-> {target_type}: {count} edges"
                    )

                if skipped:
                    errors = report.get("errors", [])
                    detail = "; ".join(errors) if errors else f"{skipped} skipped"
                    self._report_warning(
                        f"{node_type} -[{edge_type}]-> {target_type}",
                        detail,
                    )

    # ── Phase 5: Junction edges ─────────────────────────────────────

    def _load_junction_edges(self, all_specs: list[dict[str, Any]]) -> None:
        """Create edges from junction (many-to-many) CSVs."""
        has_junctions = any(
            spec.get("connections", {}).get("junction_edges")
            for spec in all_specs
        )
        if not has_junctions:
            return

        if self.verbose:
            print("  Loading junction edges...")

        for spec in all_specs:
            node_type = spec["_node_type"]
            junction_edges = (
                spec.get("connections", {}).get("junction_edges", {})
            )

            for edge_type, junc_def in junction_edges.items():
                csv_path = junc_def.get("csv")
                if not csv_path:
                    self._report_error(
                        node_type,
                        f"Junction edge {edge_type} missing 'csv' key",
                    )
                    continue

                try:
                    df = self._read_csv(csv_path)
                except FileNotFoundError:
                    self._report_error(
                        node_type,
                        f"Junction CSV not found: {csv_path}",
                    )
                    continue

                source_fk = junc_def["source_fk"]
                target_type = junc_def["target"]
                target_fk = junc_def["target_fk"]
                prop_cols = junc_def.get("properties", [])

                # Apply property_types conversions (e.g. epoch millis → datetime)
                prop_types = junc_def.get("property_types", {})
                for col, typ in prop_types.items():
                    if col not in df.columns:
                        continue
                    if typ == "date":
                        df[col] = pd.to_datetime(
                            df[col], unit="ms", errors="coerce"
                        )

                # Coerce float ID columns to int
                _coerce_id_columns(df, [source_fk, target_fk])

                try:
                    report = self.graph.add_connections(
                        data=df,
                        connection_type=edge_type,
                        source_type=node_type,
                        source_id_field=source_fk,
                        target_type=target_type,
                        target_id_field=target_fk,
                        columns=prop_cols if prop_cols else None,
                    )
                except Exception as e:
                    self._report_error(
                        node_type,
                        f"Junction edge {edge_type} failed: {e}",
                    )
                    continue

                count = report.get("connections_created", 0)
                skipped = report.get("connections_skipped", 0)
                self.stats["edges_by_type"][edge_type] = (
                    self.stats["edges_by_type"].get(edge_type, 0) + count
                )

                if self.verbose:
                    print(
                        f"    {node_type} -[{edge_type}]-> {target_type}: {count} edges"
                    )

                if skipped:
                    errors = report.get("errors", [])
                    detail = "; ".join(errors) if errors else f"{skipped} skipped"
                    self._report_warning(
                        f"{node_type} -[{edge_type}]-> {target_type}",
                        detail,
                    )

    # ── Helpers ──────────────────────────────────────────────────────

    def _read_csv(self, relative_path: str) -> pd.DataFrame:
        """Read a CSV, caching by path. Returns a copy to avoid mutation."""
        if relative_path in self._csv_cache:
            return self._csv_cache[relative_path].copy()

        full_path = self.root / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"CSV file not found: {full_path}")

        df = pd.read_csv(full_path, low_memory=False)
        self._csv_cache[relative_path] = df
        return df.copy()

    def _apply_filter(
        self, df: pd.DataFrame, filt: dict[str, Any]
    ) -> pd.DataFrame:
        """Apply filters to a DataFrame.

        Supports two forms:
            - Simple equality: ``{"column": value}``
            - Operator dict:   ``{"column": {"!=": value}}``

        Supported operators: ``=``, ``!=``, ``>``, ``<``, ``>=``, ``<=``.
        """
        for col, val in filt.items():
            if col not in df.columns:
                continue
            if isinstance(val, dict):
                for op, operand in val.items():
                    if op == "!=":
                        df = df[df[col] != operand]
                    elif op == ">":
                        df = df[df[col] > operand]
                    elif op == "<":
                        df = df[df[col] < operand]
                    elif op == ">=":
                        df = df[df[col] >= operand]
                    elif op == "<=":
                        df = df[df[col] <= operand]
                    elif op == "=":
                        df = df[df[col] == operand]
            else:
                df = df[df[col] == val]
        return df

    def _apply_timeseries_filter(
        self, df: pd.DataFrame, ts_config: dict[str, Any]
    ) -> pd.DataFrame:
        """Drop aggregate rows (e.g. month=0) from timeseries data.

        Mirrors the filtering in ``_prepare_timeseries`` so FK edge building
        operates on the same row set as node creation.
        """
        time_key = ts_config.get("time_key")
        if not isinstance(time_key, dict):
            return df
        for label, col in time_key.items():
            if label != "year" and col in df.columns:
                df = df[df[col] != 0]
        return df

    def _build_column_types(
        self, properties: dict[str, str]
    ) -> dict[str, str]:
        """Convert blueprint property types to KGLite column_types dict."""
        result = {}
        for col, typ in properties.items():
            if typ in _SPATIAL_TYPES:
                # Spatial types are handled separately via set_spatial
                continue
            mapped = _TYPE_MAP.get(typ)
            if mapped:
                result[col] = mapped
        return result

    def _convert_geometry(
        self,
        df: pd.DataFrame,
        properties: dict[str, str],
        node_type: str,
    ) -> pd.DataFrame:
        """Convert _geometry GeoJSON column to WKT + centroid lat/lon."""
        try:
            from shapely.geometry import shape as shapely_shape
        except ImportError:
            raise ImportError(
                "Blueprint uses geometry/location types which require shapely. "
                "Install with: pip install shapely"
            ) from None

        if "_geometry" not in df.columns:
            self._report_warning(
                node_type,
                "Blueprint uses geometry types but CSV has no '_geometry' column",
            )
            return df

        df = df.copy()

        # Find which columns need which geometry-derived values
        wkt_col = None
        lat_col = None
        lon_col = None
        for col, typ in properties.items():
            if typ == "geometry":
                wkt_col = col
            elif typ == "location.lat":
                lat_col = col
            elif typ == "location.lon":
                lon_col = col

        def _parse_geom(geom_str):
            if pd.isna(geom_str) or not geom_str:
                return None, None, None
            try:
                geojson = json.loads(geom_str) if isinstance(geom_str, str) else geom_str
                geom = shapely_shape(geojson)
                centroid = geom.centroid
                return geom.wkt, centroid.y, centroid.x
            except Exception:
                return None, None, None

        parsed = df["_geometry"].apply(_parse_geom)
        if wkt_col:
            df[wkt_col] = parsed.apply(lambda t: t[0])
        if lat_col:
            df[lat_col] = parsed.apply(lambda t: t[1])
        if lon_col:
            df[lon_col] = parsed.apply(lambda t: t[2])

        return df

    def _apply_spatial_config(
        self, node_type: str, properties: dict[str, str]
    ) -> None:
        """Call set_spatial() based on blueprint property types."""
        lat_col = None
        lon_col = None
        geom_col = None

        for col, typ in properties.items():
            if typ == "location.lat":
                lat_col = col
            elif typ == "location.lon":
                lon_col = col
            elif typ == "geometry":
                geom_col = col

        location = (lat_col, lon_col) if lat_col and lon_col else None
        self.graph.set_spatial(
            node_type,
            location=location,
            geometry=geom_col,
        )

    def _prepare_timeseries(
        self,
        df: pd.DataFrame,
        ts_config: dict[str, Any],
        skip: list[str],
        node_type: str,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Prepare timeseries parameter for add_nodes() inline loading.

        Renames DataFrame columns from CSV names to channel names.
        Returns the timeseries dict and the (possibly modified) DataFrame.
        """
        df = df.copy()

        time_key = ts_config["time_key"]
        channels_map = ts_config.get("channels", {})
        resolution = ts_config.get("resolution")
        units = ts_config.get("units", {})

        # time_key can be a string (single column) or dict (composite)
        if isinstance(time_key, str):
            ts_time = time_key
        else:
            # Dict: {"year": "col_y", "month": "col_m"} -> use as-is for add_nodes
            ts_time = time_key

        # Auto-skip rows where any time component is 0 (e.g. month=0 annual
        # totals).  A zero month/day/hour is never a valid time index.
        if isinstance(time_key, dict):
            for _label, col in time_key.items():
                if _label != "year" and col in df.columns:
                    before = len(df)
                    df = df[df[col] != 0]
                    dropped = before - len(df)
                    if dropped and self.verbose:
                        print(
                            f"      Dropped {dropped} rows with {col}=0 "
                            f"(aggregate totals)"
                        )

        # Rename channel columns: csv_col -> channel_name
        # channels_map is {channel_name: csv_column_name}
        rename_map = {}
        channel_names = []
        for ch_name, csv_col in channels_map.items():
            if csv_col in df.columns and ch_name != csv_col:
                rename_map[csv_col] = ch_name
            channel_names.append(ch_name)

        if rename_map:
            df = df.rename(columns=rename_map)

        # Skip timeseries columns from being stored as node properties
        if isinstance(time_key, dict):
            skip.extend(time_key.values())
        elif isinstance(time_key, str):
            skip.append(time_key)
        # Skip original CSV column names (before rename)
        skip.extend(channels_map.values())

        ts_param: dict[str, Any] = {
            "time": ts_time,
            "channels": channel_names,
        }
        if resolution:
            ts_param["resolution"] = resolution
        if units:
            ts_param["units"] = units

        return ts_param, df

    def _report_error(self, context: str, message: str) -> None:
        """Accumulate a fatal error."""
        self.errors.append({"context": context, "message": message})

    def _report_warning(self, context: str, message: str) -> None:
        """Accumulate a non-fatal warning."""
        self.warnings.append({"context": context, "message": message})


# ── Module-level helpers ──────────────────────────────────────────────


def _coerce_id_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """In-place coerce float ID columns to int.

    Pandas reads integer columns with NaN as float64 (e.g. 260.0 instead
    of 260).  This converts whole-number floats back to int so that ID
    matching works correctly.
    """
    for col in columns:
        if col not in df.columns:
            continue
        if df[col].dtype.kind == "f":
            # Only convert if all non-null values are whole numbers
            non_null = df[col].dropna()
            if len(non_null) > 0 and (non_null == non_null.astype(np.int64)).all():
                df[col] = df[col].astype("Int64")  # nullable int
