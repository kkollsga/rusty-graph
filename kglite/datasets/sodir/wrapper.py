"""Sodir dataset wrapper — full lifecycle: fetch + pre-process + build.

Mirrors `kglite.datasets.wikidata.open()` ergonomics with a two-tier
cooldown model unique to Sodir's cadence:

- ``index_cooldown_days`` (default 14) — cheap row-count probe per
  dataset. Re-fetches only when the count changed.
- ``dataset_cooldown_days`` (default 30) — hard refresh per dataset
  even when the count is unchanged (catches silent edits).

Both are tracked per-dataset in ``workdir/sodir_index.json``, which
also serves as a reverse lookup ("which file holds dataset X?
when was it fetched?").
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from ... import KnowledgeGraph, from_blueprint, load
from . import catalog, fetcher, preprocess

INDEX_FILE = "sodir_index.json"
GRAPH_SUBDIR = "graph"
SOURCE_META_FILENAME = "sodir_source.json"
COMPLEMENT_FILENAME = "blueprint_complement.json"
PACKAGED_BLUEPRINT = Path(__file__).with_name("blueprint.json")
DEFAULT_WORKERS = 10


def open(  # noqa: A001
    workdir: str | Path,
    *,
    storage: str = "memory",
    index_cooldown_days: int = 14,
    dataset_cooldown_days: int = 30,
    blueprint_path: str | Path | None = None,
    complement_blueprint: str | Path | None = None,
    use_complement: bool = True,
    complement_overrides: bool = False,
    workers: int = DEFAULT_WORKERS,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> KnowledgeGraph:
    """Return a KGLite graph backed by Sodir factmaps data, fetching
    and building only what's missing or stale. Defaults to in-memory
    storage — the full Sodir graph is small enough (a few minutes to
    build) that disk caching adds little on top of CSV caching.

    Decision tree:

    1. Disk-mode + graph already built + within ``dataset_cooldown_days``
       since last build → load + return. No network.
    2. Index refresh due (either no index yet, or
       ``index_cooldown_days`` since last sweep): cheap
       ``returnCountOnly=true`` probe per needed dataset; re-fetch
       only those whose row-count drifted.
    3. Hard cooldown: any dataset whose ``fetched_at_iso`` is older
       than ``dataset_cooldown_days`` is re-fetched even if count
       unchanged (catches silent edits).
    4. Apply pre-processing (FK joins). Build via
       ``kglite.from_blueprint``. For ``storage="disk"``, save and
       stamp source metadata.

    Blueprint resolution:
    - ``blueprint_path`` *replaces* the packaged base blueprint
      entirely. Use for a fully custom node/edge vocabulary.
    - ``complement_blueprint`` *adds to* the base. On the first call
      the file is copied to ``workdir/blueprint_complement.json`` and
      registered permanently — subsequent ``open()`` calls auto-load
      it. Pass a new path to replace; delete the file (or pass
      ``use_complement=False`` for a single call) to skip it.
    - Merge semantics: deep merge. By default **base wins** on key
      collisions — the packaged Sodir baseline tracks the live REST
      catalog and stays authoritative; complements add new node types
      and new edges without disturbing built-in mappings. Pass
      ``complement_overrides=True`` to flip the precedence when you
      genuinely want the complement to replace base values.

    :param workdir: directory holding the cached CSVs + index +
        (for disk mode) the built graph. Created if missing.
    :param storage: ``"memory"`` *(default — Sodir is small)* or
        ``"disk"`` (persistent + cached graph for cross-process reuse).
    :param index_cooldown_days: cheap-probe cadence (default 14).
    :param dataset_cooldown_days: hard-refresh cadence per dataset
        (default 30).
    :param blueprint_path: replace the packaged base blueprint.
    :param complement_blueprint: add (and persist) a complementary
        blueprint that's merged onto the base. The file is copied into
        the workdir on first call; later calls re-use the saved copy
        without needing this argument.
    :param use_complement: if False, skip the saved complement for
        this call only. The saved file is left untouched.
    :param complement_overrides: when True, complement keys override
        base on collision (default False — base wins, since base
        tracks the canonical Sodir REST catalog).
    :param workers: thread-pool size for concurrent CSV fetches
        (default 10). Each worker self-rate-limits at 0.2s/request.
    :param force_rebuild: skip the disk-mode cache short-circuit and
        rebuild the graph even if a recent build exists. CSVs are
        still reused from cache (subject to cooldown) — only the
        graph build itself is forced. Memory mode always rebuilds,
        so this flag is a no-op there.
    :param verbose: forwarded to fetch + build. Progress is shown
        via tqdm rather than per-dataset prints.
    """
    if storage not in ("disk", "memory"):
        raise ValueError(f"storage must be 'disk' or 'memory', got {storage!r}")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    csv_dir = workdir / "csv"
    csv_dir.mkdir(exist_ok=True)
    bp_path = Path(blueprint_path) if blueprint_path else PACKAGED_BLUEPRINT

    # Persist a freshly-supplied complement, then load whatever's saved.
    complement_state = _resolve_complement(workdir, complement_blueprint, use_complement, verbose)

    blueprint = json.loads(bp_path.read_text())
    if complement_state is not None:
        blueprint = _merge_blueprints(blueprint, complement_state, complement_overrides=complement_overrides)
    needed = _datasets_used_by_blueprint(blueprint)

    # Disk mode short-circuit: existing graph within hard cooldown.
    graph_dir = workdir / GRAPH_SUBDIR
    if storage == "disk" and not force_rebuild:
        graph_meta = graph_dir / "disk_graph_meta.json"
        if graph_meta.exists():
            graph_age = _age_days(_file_mtime_utc(graph_meta))
            if graph_age < dataset_cooldown_days:
                if verbose:
                    print(
                        f"  Sodir graph at {graph_dir} is {graph_age:.1f}d old "
                        f"(< {dataset_cooldown_days}d cooldown). Loading."
                    )
                return load(str(graph_dir))
    elif storage == "disk" and force_rebuild and verbose:
        print("  force_rebuild=True — skipping cache, rebuilding graph from CSVs.")

    # Refresh CSVs as needed (per-dataset).
    index = _load_index(workdir)
    fetched_now = _refresh_csvs(
        workdir,
        needed,
        index,
        index_cooldown_days=index_cooldown_days,
        dataset_cooldown_days=dataset_cooldown_days,
        workers=workers,
        verbose=verbose,
    )
    _save_index(workdir, index)

    # Pre-process (FK joins). Idempotent — re-runs are no-ops on
    # CSVs already populated.
    if verbose:
        print("  Applying pre-processing FK joins...")
    preprocess.apply(csv_dir, verbose=verbose)

    # Build via from_blueprint.
    if verbose:
        print(f"  Building graph from blueprint ({len(needed)} datasets, storage={storage})")

    if storage == "memory":
        return _build_memory_graph(workdir, blueprint, verbose)

    # Disk: clean previous graph (if any) and rebuild.
    if graph_dir.exists():
        import shutil

        shutil.rmtree(graph_dir)
    graph_dir.mkdir(parents=True)
    g = _build_disk_graph(workdir, graph_dir, blueprint, verbose)
    _write_source_meta(graph_dir / SOURCE_META_FILENAME, fetched_now, index)
    return g


def fetch_all(
    workdir: str | Path,
    *,
    index_cooldown_days: int = 14,
    dataset_cooldown_days: int = 30,
    blueprint_path: str | Path | None = None,
    complement_blueprint: str | Path | None = None,
    use_complement: bool = True,
    complement_overrides: bool = False,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = True,
) -> dict[str, dict]:
    """Refresh CSVs and return the index entry for each needed dataset.
    Useful when callers want raw CSVs without building a graph."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "csv").mkdir(exist_ok=True)
    bp_path = Path(blueprint_path) if blueprint_path else PACKAGED_BLUEPRINT
    complement_state = _resolve_complement(workdir, complement_blueprint, use_complement, verbose)
    blueprint = json.loads(bp_path.read_text())
    if complement_state is not None:
        blueprint = _merge_blueprints(blueprint, complement_state, complement_overrides=complement_overrides)
    needed = _datasets_used_by_blueprint(blueprint)

    index = _load_index(workdir)
    _refresh_csvs(
        workdir,
        needed,
        index,
        index_cooldown_days=index_cooldown_days,
        dataset_cooldown_days=dataset_cooldown_days,
        workers=workers,
        verbose=verbose,
    )
    _save_index(workdir, index)
    return {stem: index["datasets"][stem] for stem in needed if stem in index["datasets"]}


# ── Complement blueprint ───────────────────────────────────────────────


def _resolve_complement(
    workdir: Path,
    incoming: str | Path | None,
    use_complement: bool,
    verbose: bool,
) -> dict | None:
    """Persist a freshly-supplied complement, then load whatever's
    saved. Returns the parsed complement dict or ``None``."""
    saved_path = workdir / COMPLEMENT_FILENAME

    if incoming is not None:
        incoming = Path(incoming)
        if not incoming.exists():
            raise FileNotFoundError(f"complement_blueprint not found: {incoming}")
        # Round-trip through json to validate before persisting.
        payload = json.loads(incoming.read_text())
        saved_path.write_text(json.dumps(payload, indent=2))
        if verbose:
            print(f"  Registered complement blueprint at {saved_path}")

    if not use_complement:
        if verbose and saved_path.exists():
            print("  use_complement=False — skipping saved complement for this call.")
        return None

    if saved_path.exists():
        if verbose:
            print(f"  Applying saved complement blueprint ({saved_path.name}).")
        return json.loads(saved_path.read_text())
    return None


def _merge_blueprints(base: dict, complement: dict, *, complement_overrides: bool = False) -> dict:
    """Deep merge ``complement`` onto ``base``. Defaults to
    base-wins on leaf collisions (Sodir baseline is authoritative),
    flippable via ``complement_overrides=True``. Nested dicts always
    merge recursively. Lists are replaced wholesale (no element-level
    merging — keeps semantics predictable for blueprint authors)."""
    merged = json.loads(json.dumps(base))
    return _deep_merge(merged, complement, b_overrides=complement_overrides)


def _deep_merge(a: dict, b: dict, *, b_overrides: bool) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v, b_overrides=b_overrides)
        elif k in out and not b_overrides:
            # Collision; preserve `a`'s leaf value.
            continue
        else:
            out[k] = v
    return out


def remove_complement(workdir: str | Path) -> bool:
    """Delete any saved complement blueprint. Returns True if a file
    was removed, False if there was nothing to remove."""
    path = Path(workdir) / COMPLEMENT_FILENAME
    if path.exists():
        path.unlink()
        return True
    return False


# ── Blueprint walker ───────────────────────────────────────────────────


def _datasets_used_by_blueprint(blueprint: dict) -> set[str]:
    """Walk node defs, junction edges, and sub_nodes; return dataset
    stems (filename without `.csv`)."""
    paths: set[str] = set()
    for node_def in blueprint.get("nodes", {}).values():
        if "csv" in node_def:
            paths.add(node_def["csv"])
        for edge in node_def.get("connections", {}).get("junction_edges", {}).values():
            if "csv" in edge:
                paths.add(edge["csv"])
        for sub in node_def.get("sub_nodes", {}).values():
            if "csv" in sub:
                paths.add(sub["csv"])
    return {Path(p).stem for p in paths}


# ── Index management ───────────────────────────────────────────────────


def _load_index(workdir: Path) -> dict:
    p = workdir / INDEX_FILE
    if not p.exists():
        return {
            "schema_version": 1,
            "endpoint": "https://factmaps.sodir.no/api/rest/services/DataService",
            "last_full_check_iso": None,
            "datasets": {},
        }
    return json.loads(p.read_text())


def _save_index(workdir: Path, index: dict) -> None:
    """Index write — small JSON, ~85 entries. Written incrementally
    after each fetch completes so a Ctrl-C never loses progress."""
    (workdir / INDEX_FILE).write_text(json.dumps(index, indent=2, sort_keys=True))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _refresh_csvs(
    workdir: Path,
    needed: set[str],
    index: dict,
    *,
    index_cooldown_days: int,
    dataset_cooldown_days: int,
    workers: int,
    verbose: bool,
) -> set[str]:
    """Per-dataset refresh logic with a worker pool. Mutates `index`
    in place. Returns the set of stems that were actually re-fetched.

    Pass 1 (sequential, in-process): classify each needed dataset as
    skip / probe / fetch / user-supplied / unfetchable. No network.

    Pass 2 (parallel, ``workers`` threads): execute probes + fetches
    concurrently. tqdm progress bar shows throughput; per-dataset
    chatter is suppressed in favour of the aggregate."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    csv_dir = workdir / "csv"
    fetched: set[str] = set()
    probed_unchanged: set[str] = set()
    user_supplied: set[str] = set()
    skipped_unknown: set[str] = set()

    last_full_check = index.get("last_full_check_iso")
    sweep_due = last_full_check is None or _age_days_iso(last_full_check) >= index_cooldown_days

    # ── Pass 1: classify ──
    work: list[tuple[str, str]] = []  # [(stem, action), ...]
    for stem in sorted(needed):
        csv_path = csv_dir / f"{stem}.csv"
        entry = index["datasets"].get(stem)

        if not catalog.is_known(stem):
            if csv_path.exists():
                user_supplied.add(stem)
                if entry is None:
                    index["datasets"][stem] = {
                        "kind": "user_supplied",
                        "csv_path": f"csv/{stem}.csv",
                        "row_count": _quick_row_count(csv_path),
                        "fetched_at_iso": _now_iso(),
                        "count_checked_at_iso": _now_iso(),
                    }
                continue
            skipped_unknown.add(stem)
            continue

        action = _decide_action(
            entry,
            csv_path,
            sweep_due=sweep_due,
            dataset_cooldown_days=dataset_cooldown_days,
        )
        if action != "skip":
            work.append((stem, action))

    # Submit large datasets first so the long pole (wellbore, seismic_*)
    # doesn't end up tail-of-queue — meaningful win on subsequent runs
    # where the index has accurate `row_count` from prior fetches.
    # Fresh runs have row_count=0 for every entry → stable alpha order.
    work.sort(key=lambda x: _size_hint(index, x[0]), reverse=True)

    # ── Pass 2: execute in parallel ──
    errors: list[tuple[str, BaseException]] = []
    if work:
        bar = _make_refresh_bar(len(work), workers, verbose)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(_execute_one, stem, action, csv_dir, index["datasets"].get(stem)): (stem, action)
                for stem, action in work
            }
            for fut in as_completed(future_map):
                stem, _action = future_map[fut]
                try:
                    result = fut.result()
                except BaseException as e:  # noqa: BLE001  (record per-task)
                    errors.append((stem, e))
                    if bar is not None:
                        bar.update(1)
                    continue

                kind = result[0]
                if kind == "fetched":
                    _, _, rows, elapsed = result
                    base_url, layer_id = catalog.resolve(stem)
                    index["datasets"][stem] = {
                        "kind": catalog.kind_of(stem),
                        "layer_id": layer_id,
                        "base_url": base_url,
                        "csv_path": f"csv/{stem}.csv",
                        "row_count": rows,
                        "fetched_at_iso": _now_iso(),
                        "count_checked_at_iso": _now_iso(),
                        "fetch_duration_secs": round(elapsed, 2),
                    }
                    fetched.add(stem)
                elif kind == "unchanged":
                    if stem in index["datasets"]:
                        index["datasets"][stem]["count_checked_at_iso"] = _now_iso()
                    probed_unchanged.add(stem)

                # Flush index after every completion so a Ctrl-C never
                # loses progress — the next run picks up from where
                # we stopped.
                _save_index(workdir, index)

                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(fetched=len(fetched), unchanged=len(probed_unchanged))
        if bar is not None:
            bar.close()

    if sweep_due:
        index["last_full_check_iso"] = _now_iso()

    if verbose:
        skipped = len(needed) - len(fetched) - len(probed_unchanged) - len(user_supplied) - len(skipped_unknown)
        print(
            f"  Refresh: fetched {len(fetched)}, unchanged {len(probed_unchanged)}, "
            f"user-supplied {len(user_supplied)}, cached {skipped}, "
            f"unfetchable {len(skipped_unknown)}"
        )
        if errors:
            print(f"  ERRORS ({len(errors)}):")
            for stem, err in errors[:5]:
                print(f"    {stem}: {err}")
            if len(errors) > 5:
                print(f"    … and {len(errors) - 5} more")
        if skipped_unknown:
            print(
                f"  WARNING: {len(skipped_unknown)} blueprint datasets not in REST catalog "
                f"and not pre-supplied at csv/<stem>.csv: {sorted(skipped_unknown)[:5]}"
                f"{' …' if len(skipped_unknown) > 5 else ''}"
            )

    return fetched


def _execute_one(
    stem: str,
    action: str,
    csv_dir: Path,
    entry_snapshot: dict | None,
) -> tuple:
    """Worker body: run the probe or fetch. Returns one of:

    - ``("fetched", stem, rows, elapsed_secs)``
    - ``("unchanged", stem, remote_count, 0.0)``

    Pure with respect to the index dict — the main thread merges
    results in via ``as_completed``. ``entry_snapshot`` is the index
    entry seen at submission time, used for the count-equality check
    inside a probe."""
    csv_path = csv_dir / f"{stem}.csv"
    if action == "probe":
        remote_count = fetcher.count(stem)
        if entry_snapshot is not None and remote_count == entry_snapshot.get("row_count"):
            return ("unchanged", stem, remote_count, 0.0)
        # Count drifted → upgrade to fetch.
        action = "fetch"
    if action == "fetch":
        t0 = time.time()
        rows = fetcher.fetch_to_csv(stem, csv_path)
        return ("fetched", stem, rows, time.time() - t0)
    raise ValueError(f"unexpected action {action!r}")


def _size_hint(index: dict, stem: str) -> int:
    """Best-known size for scheduling. Returns the prior fetch's
    ``row_count`` if cached, else 0 (unknown — sorts last)."""
    entry = index["datasets"].get(stem)
    return entry.get("row_count", 0) if entry else 0


def _make_refresh_bar(total: int, workers: int, verbose: bool):
    if not verbose:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(
        total=total,
        desc=f"Sodir refresh ({workers} workers)",
        unit="dataset",
        miniters=1,
    )


def _decide_action(
    entry: dict | None,
    csv_path: Path,
    *,
    sweep_due: bool,
    dataset_cooldown_days: int,
) -> str:
    """Return one of: 'fetch' (download fresh), 'probe' (cheap count
    check), 'skip' (keep existing CSV)."""
    if entry is None or not csv_path.exists():
        return "fetch"
    # Corrupt-tiny CSV — anything under 5 bytes can't even hold a
    # single column header. This catches the empty-dataset bug from
    # earlier versions where pd.DataFrame([]).to_csv emitted just a
    # newline; re-fetch to get a header-only CSV.
    if csv_path.stat().st_size < 5:
        return "fetch"
    fetched_at = entry.get("fetched_at_iso")
    if fetched_at and _age_days_iso(fetched_at) >= dataset_cooldown_days:
        return "fetch"
    if sweep_due:
        return "probe"
    return "skip"


def _quick_row_count(csv_path: Path) -> int:
    """Cheap row count for user-supplied CSVs — line count minus header."""
    try:
        with csv_path.open("rb") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


# ── Build helpers ──────────────────────────────────────────────────────


def _build_disk_graph(workdir: Path, graph_dir: Path, blueprint: dict, verbose: bool) -> KnowledgeGraph:
    """Build a disk-backed graph from the workdir and save. The
    ``from_blueprint`` call is intentionally silent — it would
    otherwise emit ~150 lines (per-edge-type tallies, per-node-type
    listings, every skipped-FK warning). The wrapper-level summary
    after this returns gives the headline numbers."""
    bp_with_root = _blueprint_with_input_root(blueprint, workdir)
    bp_path = workdir / "_compiled_blueprint.json"
    bp_path.write_text(json.dumps(bp_with_root))
    g = from_blueprint(str(bp_path), verbose=False, save=False, storage="disk", path=str(graph_dir))
    g.save(str(graph_dir))
    bp_path.unlink(missing_ok=True)
    if verbose:
        info = g.graph_info()
        print(f"  Built graph: {info.get('node_count', 0):,} nodes, {info.get('edge_count', 0):,} edges")
    return g


def _build_memory_graph(workdir: Path, blueprint: dict, verbose: bool) -> KnowledgeGraph:
    bp_with_root = _blueprint_with_input_root(blueprint, workdir)
    bp_path = workdir / "_compiled_blueprint.json"
    bp_path.write_text(json.dumps(bp_with_root))
    g = from_blueprint(str(bp_path), verbose=False, save=False)
    bp_path.unlink(missing_ok=True)
    if verbose:
        info = g.graph_info()
        print(f"  Built graph: {info.get('node_count', 0):,} nodes, {info.get('edge_count', 0):,} edges")
    return g


def _blueprint_with_input_root(blueprint: dict, workdir: Path) -> dict:
    """Inject the workdir as `settings.input_root` so blueprint paths
    resolve correctly."""
    out = json.loads(json.dumps(blueprint))  # deep copy
    settings = out.setdefault("settings", {})
    settings["input_root"] = str(workdir)
    return out


def _write_source_meta(path: Path, fetched: set[str], index: dict) -> None:
    payload: dict[str, Any] = {
        "built_at_iso": _now_iso(),
        "fetched_during_build": sorted(fetched),
        "datasets": {stem: dict(entry) for stem, entry in index["datasets"].items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ── Time helpers ───────────────────────────────────────────────────────


def _file_mtime_utc(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _age_days(when: datetime | None) -> float:
    if when is None:
        return float("inf")
    return (datetime.now(timezone.utc) - when).total_seconds() / 86400


def _age_days_iso(iso: str | None) -> float:
    if iso is None:
        return float("inf")
    try:
        return _age_days(datetime.fromisoformat(iso))
    except ValueError:
        return float("inf")
