"""File watcher — thin wrapper around the mcp-methods Rust debouncer.

0.9.24: replaced the pure-Python `watchdog` + threading debouncer with
a single call into `kglite._mcp_internal.start_watch`. Rust drives the
`notify-debouncer-mini` event loop on a background thread; the
callback runs with the GIL re-acquired automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from kglite import _mcp_internal

log = logging.getLogger("kglite.mcp_server.watch")


def start(dir_path: Path, on_change: Callable[[], None], debounce_seconds: float = 0.5) -> Any:
    """Watch `dir_path` recursively; call `on_change()` after each
    debounce window. Returns a `WatchHandle` whose `.stop()` tears
    the watcher down — caller must keep a reference (the handle's
    Drop in Rust is what unregisters the watcher)."""

    def _dispatch(_paths: list[str]) -> None:
        try:
            on_change()
        except Exception as e:
            log.warning("watch on_change handler raised: %s", e)

    handle = _mcp_internal.start_watch(
        str(dir_path),
        _dispatch,
        debounce_ms=int(debounce_seconds * 1000),
    )
    log.info("watching %s (debounce=%.2fs)", dir_path, debounce_seconds)
    return handle
