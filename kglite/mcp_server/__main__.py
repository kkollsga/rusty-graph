"""Entry point for ``python -m kglite.mcp_server``."""

from __future__ import annotations

import sys

from kglite.mcp_server.cli import main

if __name__ == "__main__":
    sys.exit(main())
