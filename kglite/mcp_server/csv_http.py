"""CSV-over-HTTP server — operator pinch-point P3.

Mirrors `crates/kglite-mcp-server/src/csv_http.rs` for the Python
implementation. When the manifest declares

    extensions:
      csv_http_server:
        port: 8765
        dir: temp/
        cors_origin: "*"

the server spawns an aiohttp listener bound to 127.0.0.1:<port> that
serves CSV files out of the configured directory. The `cypher_query`
tool, when it sees `FORMAT CSV`, writes the result to
`<dir>/<hash>.csv` and returns the URL instead of inlining.

Only GETs of flat filenames inside `<dir>` are served. No directory
listing, no upload, no write surface. Loopback-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import time
from typing import Any

log = logging.getLogger("kglite.mcp_server.csv_http")


@dataclass
class CsvHttpConfig:
    """Resolved configuration for the CSV-over-HTTP server."""

    port: int
    dir: Path
    cors_origin: str = "*"

    def url_for(self, name: str) -> str:
        """Construct the public URL for a generated CSV file."""
        return f"http://127.0.0.1:{self.port}/{name}"


def from_manifest_value(value: Any, base_dir: Path) -> CsvHttpConfig | None:
    """Parse `extensions.csv_http_server` from the manifest. Accepted
    shapes (same as the Rust version):

        csv_http_server: true                     # defaults
        csv_http_server: { port: 9000 }
        csv_http_server: { dir: out/ }
        csv_http_server: { port: 9000, dir: out/, cors_origin: "https://my.app" }

    Returns None when the block is absent or explicitly false.
    """
    if value is None or value is False:
        return None
    if value is True:
        return CsvHttpConfig(port=8765, dir=(base_dir / "temp").resolve())
    if not isinstance(value, dict):
        raise ValueError(f"extensions.csv_http_server must be a mapping or boolean (got {value!r})")
    port = value.get("port", 8765)
    if not isinstance(port, int) or port < 0 or port > 65535:
        raise ValueError(f"csv_http_server.port must be a u16 (got {port!r})")
    dir_raw = value.get("dir", "temp")
    if not isinstance(dir_raw, str):
        raise ValueError(f"csv_http_server.dir must be a string (got {dir_raw!r})")
    cors = value.get("cors_origin", "*")
    if cors is not None and not isinstance(cors, str):
        raise ValueError(f"csv_http_server.cors_origin must be a string (got {cors!r})")
    return CsvHttpConfig(
        port=port,
        dir=(base_dir / dir_raw).resolve(),
        cors_origin=cors or "*",
    )


def write_csv(config: CsvHttpConfig, csv: str) -> str:
    """Write a CSV body to a fresh file inside the configured
    directory; return the basename (caller joins via url_for). The
    filename combines a nanosecond timestamp + content hash so
    concurrent queries don't collide."""
    config.dir.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    h = hashlib.blake2b(csv.encode("utf-8"), digest_size=8).hexdigest()
    name = f"kglite-{stamp:x}-{h}.csv"
    (config.dir / name).write_text(csv, encoding="utf-8")
    return name


async def spawn(config: CsvHttpConfig) -> None:
    """Start the aiohttp listener as a background task. Awaits the
    bind but not the serve loop. Bind failures (port-in-use, etc.)
    propagate to the caller so boot fails fast."""
    from aiohttp import web

    config.dir.mkdir(parents=True, exist_ok=True)

    async def handle(request: web.Request) -> web.Response:
        cors = config.cors_origin
        if request.method == "OPTIONS":
            return web.Response(
                status=204,
                headers={
                    "Access-Control-Allow-Origin": cors,
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                },
            )
        if request.method != "GET":
            return web.Response(
                status=405,
                text="only GET is supported",
                headers={"Access-Control-Allow-Origin": cors},
            )
        raw = request.match_info["name"]
        # Defence-in-depth: reject any path component with a separator
        # or `..` before touching the filesystem.
        if not raw or "/" in raw or "\\" in raw or raw == ".." or raw.startswith("."):
            return web.Response(
                status=400,
                text="invalid path",
                headers={"Access-Control-Allow-Origin": cors},
            )
        target = (config.dir / raw).resolve()
        try:
            target.relative_to(config.dir.resolve())
        except ValueError:
            return web.Response(
                status=403,
                text="forbidden",
                headers={"Access-Control-Allow-Origin": cors},
            )
        if not target.is_file():
            return web.Response(
                status=404,
                text="not found",
                headers={"Access-Control-Allow-Origin": cors},
            )
        body = target.read_bytes()
        content_type = "text/csv; charset=utf-8" if raw.endswith(".csv") else "application/octet-stream"
        return web.Response(
            body=body,
            content_type=content_type,
            headers={
                "Access-Control-Allow-Origin": cors,
                "Cache-Control": "no-store",
            },
        )

    app = web.Application()
    app.router.add_route("*", "/{name}", handle)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", config.port)
    await site.start()
    log.info("csv_http_server listening on 127.0.0.1:%d (dir=%s)", config.port, config.dir)
