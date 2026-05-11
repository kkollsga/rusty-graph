"""File watcher for `--watch DIR` mode (and workspace `watch: true`).

Mirrors `crates/kglite-mcp-server/src/main.rs`'s watch hook. Uses
`watchdog` for cross-platform inotify/FSEvents/ReadDirectoryChangesW
abstraction. Debounces bursts of changes — e.g. an editor saving 12
related files in 50 ms — into a single rebuild.
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
from typing import Callable

log = logging.getLogger("kglite.mcp_server.watch")


def start(dir_path: Path, on_change: Callable[[], None], debounce_seconds: float = 0.5) -> object:
    """Start watching `dir_path` for changes; call `on_change()` after
    each quiet period of length `debounce_seconds`. Returns the
    watchdog observer (caller can `.stop()` it; we keep a reference so
    it isn't GC'd)."""
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    last_event = [0.0]
    pending = [False]
    lock = threading.Lock()

    def trigger() -> None:
        while True:
            time.sleep(debounce_seconds)
            with lock:
                if not pending[0]:
                    continue
                idle = time.monotonic() - last_event[0]
                if idle < debounce_seconds:
                    continue
                pending[0] = False
            try:
                on_change()
            except Exception as e:  # noqa: BLE001
                log.warning("watch on_change handler raised: %s", e)

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event: FileSystemEvent) -> None:  # type: ignore[override]
            if event.is_directory:
                return
            with lock:
                last_event[0] = time.monotonic()
                pending[0] = True

    observer = Observer()
    observer.schedule(Handler(), str(dir_path), recursive=True)
    observer.start()

    debounce_thread = threading.Thread(target=trigger, name="kglite-watch-debounce", daemon=True)
    debounce_thread.start()

    log.info("watching %s (debounce=%.2fs)", dir_path, debounce_seconds)
    return observer
