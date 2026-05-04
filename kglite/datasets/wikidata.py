"""Helpers for fetching and building KGLite disk graphs from the
Wikimedia Foundation's official `latest-truthy` RDF dumps published at
https://dumps.wikimedia.org/wikidatawiki/entities/.

KGLite is an independent project — not affiliated with the Wikimedia
Foundation or Wikidata. The dump format and licensing are defined by
upstream; this module only handles the cache + build lifecycle on the
client side.

Public API:
    open(workdir, ...)    -> KnowledgeGraph   # full lifecycle
    fetch_truthy(workdir) -> Path             # dump-only path

Layout managed under `workdir`:

    workdir/
        latest-truthy.nt.bz2          # cached dump
        latest-truthy.nt.bz2.part     # in-progress download (resumable)
        graph/                        # disk graph dir built from the dump
            wikidata_source.json      # build-time dump metadata
            disk_graph_meta.json
            ...
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from .. import KnowledgeGraph, load

WIKIDATA_FILE = "latest-truthy.nt.bz2"
WIKIDATA_URL = f"https://dumps.wikimedia.org/wikidatawiki/entities/{WIKIDATA_FILE}"
SOURCE_META_FILENAME = "wikidata_source.json"
GRAPH_SUBDIR = "graph"

# Process-local cache of loaded disk graphs — keyed by (canonical
# workdir path, entity_limit_millions). Re-running `open(workdir)` in
# the same process (typical Jupyter "rerun-cell" workflow) returns the
# already-loaded instance instead of allocating a fresh ~400 MB
# in-memory state for every call. Invalidated when disk_graph_meta.json
# mtime advances (rebuild happened) or `force_rebuild=True` is passed.
# Memory-mode opens skip this cache — they're meant to be reproducible
# rebuilds.
_PROCESS_CACHE: dict[tuple[str, int | None], tuple[KnowledgeGraph, float]] = {}


def open(  # noqa: A001  (intentional `open` shadow — module-scoped, follows stdlib `tarfile.open`/`gzip.open` precedent)
    workdir: str | Path,
    *,
    storage: str = "disk",
    cooldown_days: int = 31,
    languages: tuple[str, ...] = ("en",),
    entity_limit_millions: int | None = None,
    verbose: bool = True,
    progress: object | None = None,
    force_rebuild: bool = False,
) -> KnowledgeGraph:
    """Return a KGLite graph backed by the Wikidata `latest-truthy`
    dump, fetching and building if needed.

    Two storage backends:

    - ``storage="disk"`` *(default)* — persists the build to
      ``workdir/graph[_<N>m]/`` so subsequent calls cache-hit.
      Decision tree:

      1. Graph exists and within ``cooldown_days`` → load + return.
         No network.
      2. Graph exists, cooldown elapsed, remote `Last-Modified` ≤
         embedded build timestamp → load + return.
      3. Graph exists, cooldown elapsed, remote newer → refetch
         dump (resumable) + rebuild.
      4. No graph → fetch + build.

    - ``storage="memory"`` — pure in-memory build, ideal for fast
      perf benchmarks. Reuses the cached dump at
      ``workdir/latest-truthy.nt.bz2`` (still subject to
      ``cooldown_days`` for refetch decisions) but never writes
      graph artifacts. Each call rebuilds from the dump.

    :param workdir: directory holding the cached dump (and, for
        ``storage="disk"``, the built graph). Created if missing.
    :param storage: ``"disk"`` (default, persistent + cached) or
        ``"memory"`` (in-memory, always rebuilds).
    :param cooldown_days: minimum age before re-checking the remote
        dump.
    :param languages: language filter passed to ``load_ntriples``.
    :param entity_limit_millions: build a sized slice (e.g. ``100`` →
        first 100M entities) for fast performance checks. For disk
        mode, slices are stored under ``workdir/graph_{N}m/`` and
        coexist with the full ``workdir/graph/``. The shared dump at
        ``workdir/latest-truthy.nt.bz2`` is reused across all sizes
        and across both storage modes.
    :param verbose: forwarded to ``curl`` and ``load_ntriples``.
    :param progress: optional callable receiving structured progress
        events from ``load_ntriples`` (see ``kglite.progress`` for the
        event schema and a tqdm-backed reporter). Ignored on cache hits.
    :param force_rebuild: if True, rebuild from the dump even when a
        cached graph exists at ``workdir/graph[_<N>m]/``. The dump
        itself is still served from cache when fresh — pass
        ``cooldown_days=0`` if you also want the dump re-checked.
    """
    if storage not in ("disk", "memory"):
        raise ValueError(f"storage must be 'disk' or 'memory', got {storage!r}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if storage == "memory":
        # No persistence, no cache check — always rebuild from the
        # cached dump (which is itself cooldown-managed).
        dump_path, _ = _ensure_dump(workdir, cooldown_days, verbose)
        return _build_memory_graph(dump_path, languages, entity_limit_millions, verbose, progress)

    graph_dir = workdir / _graph_subdir(entity_limit_millions)
    graph_meta = graph_dir / "disk_graph_meta.json"
    source_meta = graph_dir / SOURCE_META_FILENAME

    # Process-cache hit: return the same KnowledgeGraph instance we
    # handed back on a prior call this process, as long as the
    # underlying disk artifacts haven't been rebuilt (mtime check)
    # and the caller didn't pass `force_rebuild`. Saves the
    # ~400 MB in-memory reload on Jupyter cell re-runs.
    cache_key = (str(graph_dir.resolve()), entity_limit_millions)
    if not force_rebuild and graph_meta.exists():
        meta_mtime = graph_meta.stat().st_mtime
        cached = _PROCESS_CACHE.get(cache_key)
        if cached is not None and cached[1] == meta_mtime:
            if verbose:
                print(f"  Wikidata graph at {graph_dir} already loaded in this process. Reusing.")
            return cached[0]

    if force_rebuild and graph_dir.exists():
        import shutil

        if verbose:
            print(f"  force_rebuild=True — deleting cached graph at {graph_dir}.")
        shutil.rmtree(graph_dir)
        _PROCESS_CACHE.pop(cache_key, None)
    elif graph_meta.exists():
        graph_age = _age_days(_file_mtime_utc(graph_meta))
        if graph_age < cooldown_days:
            if verbose:
                print(
                    f"  Wikidata graph at {graph_dir} is {graph_age:.1f}d old (< {cooldown_days}d cooldown). Loading."
                )
            g = load(str(graph_dir))
            _PROCESS_CACHE[cache_key] = (g, graph_meta.stat().st_mtime)
            return g

        # Cooldown elapsed — check whether the remote dump is newer than
        # the one this graph was built from.
        embedded_mtime = _read_source_mtime(source_meta)
        remote_mtime, _ = _remote_metadata(WIKIDATA_URL, verbose=verbose)
        if remote_mtime is None:
            if verbose:
                print(f"  Remote unreachable. Loading existing graph (built from {embedded_mtime}).")
            g = load(str(graph_dir))
            _PROCESS_CACHE[cache_key] = (g, graph_meta.stat().st_mtime)
            return g
        if embedded_mtime is None or remote_mtime > embedded_mtime:
            if verbose:
                print(f"  Remote dump is newer than the cached graph (built from {embedded_mtime}). Rebuilding.")
            # Fall through to rebuild path
        else:
            if verbose:
                print("  Remote dump unchanged since last build. Loading existing graph.")
            g = load(str(graph_dir))
            _PROCESS_CACHE[cache_key] = (g, graph_meta.stat().st_mtime)
            return g

    dump_path, dump_mtime = _ensure_dump(workdir, cooldown_days, verbose)
    g = _build_graph(workdir, dump_path, dump_mtime, languages, entity_limit_millions, verbose, progress)
    if graph_meta.exists():
        _PROCESS_CACHE[cache_key] = (g, graph_meta.stat().st_mtime)
    return g


def cache_clear() -> int:
    """Drop every graph held by the process-local `wikidata.open` cache.

    Use when you want a genuinely fresh load — typically in tests
    that need isolation, or after manually editing on-disk artifacts
    in ways `disk_graph_meta.json` mtime doesn't reflect.

    Returns the number of cached graphs that were released. Note
    Python may keep the underlying `KnowledgeGraph` alive until its
    other references (notebook variables, closures) drop too.
    """
    n = len(_PROCESS_CACHE)
    _PROCESS_CACHE.clear()
    return n


def fetch_truthy(
    workdir: str | Path,
    *,
    cooldown_days: int = 31,
    verbose: bool = True,
) -> Path:
    """Ensure ``workdir/latest-truthy.nt.bz2`` exists and return its
    path. Downloads (or resumes) when missing or stale per
    ``cooldown_days``. Useful when you want the dump only — for
    example, running ``load_ntriples`` with custom filters."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    dump_path, _ = _ensure_dump(workdir, cooldown_days, verbose)
    return dump_path


# ── Internals ──────────────────────────────────────────────────────────


def _ensure_dump(workdir: Path, cooldown_days: int, verbose: bool) -> tuple[Path, datetime | None]:
    """Resolve the local dump file, downloading or resuming as needed.
    Returns (path, remote_last_modified_at_fetch_time)."""
    local_path = workdir / WIKIDATA_FILE
    part_path = local_path.with_suffix(local_path.suffix + ".part")

    remote_mtime, remote_size = _remote_metadata(WIKIDATA_URL, verbose=verbose)

    local_mtime = _file_mtime_utc(local_path)
    if local_mtime is not None:
        age = _age_days(local_mtime)
        if verbose:
            print(f"  Local dump: {_fmt_size(local_path.stat().st_size)}, age {age:.1f}d")
        if remote_mtime is None:
            if verbose:
                print("  Remote unreachable — using local copy.")
            return local_path, None
        if remote_mtime <= local_mtime:
            if verbose:
                print("  Local dump matches latest remote.")
            return local_path, remote_mtime
        if age < cooldown_days:
            if verbose:
                print(f"  Newer dump available, but local is within cooldown ({age:.1f}d < {cooldown_days}d).")
            return local_path, local_mtime
        if verbose:
            print("  Newer dump available + cooldown elapsed. Refreshing.")
        local_path.unlink()
        if part_path.exists():
            part_path.unlink()

    part_mtime = _file_mtime_utc(part_path)
    if part_mtime is not None:
        age = _age_days(part_mtime)
        if age >= cooldown_days:
            if verbose:
                print(f"  Stale partial download ({age:.1f}d ≥ {cooldown_days}d). Discarding.")
            part_path.unlink()
        else:
            if verbose:
                print(f"  Partial download present ({age:.1f}d). Resuming.")
            try:
                _curl_download(WIKIDATA_URL, part_path, remote_size, resume=True, verbose=verbose)
            except RuntimeError as e:
                if verbose:
                    print(f"  Resume failed: {e}. Restarting from scratch.")
                part_path.unlink()
                _curl_download(WIKIDATA_URL, part_path, remote_size, resume=False, verbose=verbose)
            os.rename(part_path, local_path)
            return local_path, remote_mtime

    _curl_download(WIKIDATA_URL, part_path, remote_size, resume=False, verbose=verbose)
    os.rename(part_path, local_path)
    return local_path, remote_mtime


def _graph_subdir(entity_limit_millions: int | None) -> str:
    """Directory name under workdir for a given size slice. ``None`` →
    ``graph`` (full); ``100`` → ``graph_100m`` so different slices
    coexist for fast perf comparisons."""
    if entity_limit_millions is None:
        return GRAPH_SUBDIR
    if entity_limit_millions <= 0:
        raise ValueError(f"entity_limit_millions must be positive, got {entity_limit_millions}")
    return f"{GRAPH_SUBDIR}_{entity_limit_millions}m"


def _build_graph(
    workdir: Path,
    dump_path: Path,
    dump_mtime: datetime | None,
    languages: tuple[str, ...],
    entity_limit_millions: int | None,
    verbose: bool,
    progress: object | None = None,
) -> KnowledgeGraph:
    """Disk-mode build: persists to ``workdir/graph[_<N>m]/``."""
    graph_dir = workdir / _graph_subdir(entity_limit_millions)
    if graph_dir.exists():
        import shutil

        shutil.rmtree(graph_dir)
    graph_dir.mkdir(parents=True)

    g = KnowledgeGraph(storage="disk", path=str(graph_dir))
    # When the caller wired a progress sink (tqdm), suppress the loader's
    # `[Phase X]` stderr lines — they fight tqdm for the same terminal.
    # The wrapper's own `if verbose: print(...)` status messages are
    # unaffected.
    loader_verbose = verbose and progress is None
    load_kwargs: dict = {"languages": list(languages), "verbose": loader_verbose}
    if entity_limit_millions is not None:
        load_kwargs["max_entities"] = entity_limit_millions * 1_000_000
    if progress is not None:
        load_kwargs["progress"] = progress
    g.load_ntriples(str(dump_path), **load_kwargs)
    # `load_ntriples` writes per-segment artifacts under `seg_000/`;
    # `save()` consolidates those into top-level `disk_graph_meta.json`
    # so `kglite.load(graph_dir)` works and our cache-hit check fires
    # on subsequent calls. Disk mode only — memory mode skips this.
    g.save(str(graph_dir))

    _write_source_meta(graph_dir / SOURCE_META_FILENAME, dump_path, dump_mtime, entity_limit_millions)
    return g


def _build_memory_graph(
    dump_path: Path,
    languages: tuple[str, ...],
    entity_limit_millions: int | None,
    verbose: bool,
    progress: object | None = None,
) -> KnowledgeGraph:
    """Memory-mode build: in-memory `KnowledgeGraph`, no persistence."""
    g = KnowledgeGraph()  # default = in-memory backend
    # See `_build_graph` for the verbose↔progress interaction.
    loader_verbose = verbose and progress is None
    load_kwargs: dict = {"languages": list(languages), "verbose": loader_verbose}
    if entity_limit_millions is not None:
        load_kwargs["max_entities"] = entity_limit_millions * 1_000_000
    if progress is not None:
        load_kwargs["progress"] = progress
    g.load_ntriples(str(dump_path), **load_kwargs)
    return g


def _write_source_meta(
    path: Path,
    dump_path: Path,
    remote_mtime: datetime | None,
    entity_limit_millions: int | None,
) -> None:
    source_mtime = _file_mtime_utc(dump_path)
    payload = {
        "source_file": dump_path.name,
        "source_mtime_iso": source_mtime.isoformat() if source_mtime else None,
        "remote_last_modified_iso": remote_mtime.isoformat() if remote_mtime else None,
        "entity_limit_millions": entity_limit_millions,
        "built_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))


def _read_source_mtime(path: Path) -> datetime | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        iso = data.get("remote_last_modified_iso") or data.get("source_mtime_iso")
        if iso is None:
            return None
        return datetime.fromisoformat(iso)
    except (json.JSONDecodeError, ValueError, OSError):
        return None


def _remote_metadata(url: str, *, verbose: bool) -> tuple[datetime | None, int | None]:
    try:
        with urlopen(Request(url, method="HEAD"), timeout=15) as resp:
            lm = resp.headers.get("Last-Modified")
            cl = resp.headers.get("Content-Length")
            return (
                parsedate_to_datetime(lm) if lm else None,
                int(cl) if cl else None,
            )
    except (URLError, OSError) as e:
        if verbose:
            print(f"  Could not reach remote: {e}")
        return None, None


def _file_mtime_utc(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _age_days(when: datetime | None) -> float:
    if when is None:
        return float("inf")
    return (datetime.now(timezone.utc) - when).total_seconds() / 86400


def _fmt_size(b: int | None) -> str:
    if b is None:
        return "—"
    if b < 1024**2:
        return f"{b / 1024:.1f} KB"
    if b < 1024**3:
        return f"{b / (1024**2):.1f} MB"
    if b < 1024**4:
        return f"{b / (1024**3):.2f} GB"
    return f"{b / (1024**4):.2f} TB"


def _fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _curl_download(
    url: str,
    dest: Path,
    total_bytes: int | None,
    *,
    resume: bool,
    verbose: bool,
) -> None:
    """Run curl with optional resume. Renders a tqdm progress bar when
    available, otherwise falls back to one-line-per-minute prints.
    Raises RuntimeError on failure."""
    cmd = ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3", "--retry-delay", "5"]
    start_bytes = dest.stat().st_size if (resume and dest.exists()) else 0
    if start_bytes > 0:
        cmd += ["-C", "-"]
    cmd += ["-o", str(dest), url]

    try:
        proc = subprocess.Popen(cmd)
    except FileNotFoundError as e:
        raise RuntimeError(f"`curl` not available — install curl or set up PATH ({e})") from e

    started = time.time()
    bar = _make_progress_bar(total_bytes, start_bytes, dest.name) if verbose else None

    if verbose and bar is None:
        # tqdm not installed — emit a single header so the user knows
        # what's happening before the per-minute lines start.
        if start_bytes > 0:
            remaining = (total_bytes - start_bytes) if total_bytes else None
            print(f"  Resuming download from {_fmt_size(start_bytes)} ({_fmt_size(remaining)} remaining).")
        else:
            print(f"  Downloading {_fmt_size(total_bytes)}.")

    last_t = started
    last_b = start_bytes
    interval = 1.0 if bar is not None else 60.0
    try:
        while True:
            try:
                rc = proc.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                if not verbose:
                    continue
                cur_b = dest.stat().st_size if dest.exists() else start_bytes
                if bar is not None:
                    bar.update(cur_b - last_b)
                    last_b = cur_b
                else:
                    now = time.time()
                    rate = (cur_b - last_b) / max(0.001, now - last_t)
                    elapsed = now - started
                    pct = (cur_b * 100 / total_bytes) if total_bytes else 0
                    eta = ((total_bytes - cur_b) / rate) if (rate > 0 and total_bytes) else None
                    print(
                        f"  [download {_fmt_dur(elapsed)}] "
                        f"{_fmt_size(cur_b)}/{_fmt_size(total_bytes)} "
                        f"({pct:.1f}%)  {_fmt_size(int(rate))}/s  "
                        f"ETA {_fmt_dur(eta) if eta is not None else '—'}",
                        flush=True,
                    )
                    last_t = now
                    last_b = cur_b
        if bar is not None:
            # Snap the bar to the final byte count (covers the gap
            # between the last poll and process exit).
            final_b = dest.stat().st_size if dest.exists() else last_b
            bar.update(max(0, final_b - last_b))
    finally:
        if bar is not None:
            bar.close()

    if rc != 0:
        raise RuntimeError(f"curl exited {rc}")
    if verbose and bar is None:
        # Without tqdm, print a final summary line. tqdm prints its
        # own on close.
        elapsed = time.time() - started
        final_b = dest.stat().st_size
        avg_rate = (final_b - start_bytes) / max(0.001, elapsed)
        print(f"  Download complete: {_fmt_size(final_b)} in {_fmt_dur(elapsed)} (avg {_fmt_size(int(avg_rate))}/s).")


def _make_progress_bar(total_bytes: int | None, start_bytes: int, filename: str):
    """Return a `tqdm` progress bar when the package is installed,
    else `None`. Optional dep — we don't take a hard requirement."""
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(
        total=total_bytes,
        initial=start_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=filename,
        miniters=1,
        smoothing=0.3,
    )
