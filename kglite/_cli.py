"""Console-script launcher for the bundled `kglite-mcp-server` binary.

`pip install kglite` installs `kglite-mcp-server` on PATH as a Python
console-script entry point that resolves to `main()` below. We then
exec the native Rust binary shipped under `kglite/_bin/`, so the
operator gets the same UX as a `cargo install`-built binary without
having to install Rust.

The native binary is built by maturin during the wheel build (via
the `[[bin]]` target in `crates/kglite-mcp-server`); the build
workflow copies it into this `_bin/` directory before maturin
packages the Python tree.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

_BINARY_NAME = "kglite-mcp-server.exe" if sys.platform == "win32" else "kglite-mcp-server"


def _binary_path() -> Path:
    return Path(__file__).resolve().parent / "_bin" / _BINARY_NAME


def main() -> None:
    binary = _binary_path()
    if not binary.exists():
        sys.stderr.write(
            f"kglite-mcp-server: bundled binary not found at {binary}.\n"
            f"The wheel may have been built without the bundled binary, "
            f"or you installed from source without the build hook. "
            f"Build manually with `cargo install --path crates/kglite-mcp-server` "
            f"from a kglite source clone if you have a Rust toolchain.\n"
        )
        sys.exit(1)
    if sys.platform == "win32":
        # `os.execvp` doesn't replace the current process on Windows
        # the same way it does on POSIX. Spawn + wait + propagate exit.
        import subprocess

        result = subprocess.run([str(binary), *sys.argv[1:]])
        sys.exit(result.returncode)
    os.execvp(str(binary), [str(binary), *sys.argv[1:]])


if __name__ == "__main__":  # pragma: no cover — invoked via entry point
    main()
