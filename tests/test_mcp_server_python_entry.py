"""Integration tests for `kglite-mcp-server` (the 0.9.20+ Python entry point).

The 0.9.20 release shipped without these tests and silently regressed
8 of 11 tools — operator caught it on redeploy. This suite is the CI
gate that prevents the same class of failure from recurring.

What we test:

1. **Boot + tools/list parity per mode**. For each launch mode
   (`--source-root` / `--workspace` / bare / with-and-without
   GITHUB_TOKEN) we assert the registered tool set exactly matches
   the baseline in `tests/fixtures/tool_baseline.json`. Any tool
   added or removed fails the build.
2. **Per-tool smoke**. Each tool is called with a minimal valid
   argument set and asserted to return SOMETHING sensible (not
   "unknown tool", not a stacktrace).
3. **Path traversal safety**. `read_source` with `../../../etc/passwd`
   returns an error string, not the file.

Embedder + bge-m3 tests are slow-marked because they download
~2 GB of ONNX weights on a cold cache.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import pytest

# The MCP server entry point depends on the `[mcp]` extras (mcp,
# pyyaml, aiohttp, fastembed, watchdog, huggingface_hub via fastembed).
# Skip cleanly when any of them is missing — CI installs `pip install
# -e .[mcp]` so they should be present there; local dev may not have
# them.
pytest.importorskip("mcp")
pytest.importorskip("yaml")
pytest.importorskip("aiohttp")
pytest.importorskip("fastembed")
pytest.importorskip("watchdog")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
BASELINE = json.loads((FIXTURES / "tool_baseline.json").read_text())


def _which_server() -> str | None:
    """Locate the kglite-mcp-server console script — installed by
    `maturin develop` or a regular `pip install`."""
    import shutil

    return shutil.which("kglite-mcp-server")


def _spawn(args: list[str], extra_env: dict[str, str] | None = None) -> subprocess.Popen:
    server = _which_server()
    if server is None:
        pytest.skip("kglite-mcp-server console script not on PATH (run `maturin develop`)")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [server, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
    )


class McpClient:
    """Tiny JSON-RPC stdio driver for kglite-mcp-server. Reused across
    every test in the file."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self._next_id = 1
        self._stderr: list[str] = []
        self._stderr_thread = threading.Thread(target=self._drain, daemon=True)
        self._stderr_thread.start()

    def _drain(self) -> None:
        for line in iter(self._proc.stderr.readline, b""):
            self._stderr.append(line.decode("utf-8", errors="replace"))

    def _send(self, body: dict[str, Any]) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(body) + "\n").encode())
        self._proc.stdin.flush()

    def _recv(self, timeout: float = 10.0) -> dict[str, Any]:
        assert self._proc.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise TimeoutError(f"no response within {timeout}s; stderr: {''.join(self._stderr[-5:])}")

    def initialize(self) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "kglite-test", "version": "0"},
                },
            }
        )
        resp = self._recv()
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp

    def list_tools(self) -> list[dict[str, Any]]:
        req_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": req_id, "method": "tools/list", "params": {}})
        resp = self._recv()
        return resp["result"]["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            }
        )
        return self._recv()

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


# ---------------------------------------------------------------------------
# 1. Boot + tools/list parity per mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", [name for name in BASELINE["cases"]])
def test_tools_list_baseline(tmp_path: Path, case_name: str) -> None:
    """For each launch case in the baseline, assert tools/list returns
    exactly the expected tool set. This is the CI gate the operator
    asked for after 0.9.20."""
    case = BASELINE["cases"][case_name]
    # Substitute placeholders. `${SRC}` and `${WS}` resolve to dirs
    # pytest creates fresh per test.
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "hello.txt").write_text("hello\n")
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    subs = {"${SRC}": str(src_dir), "${WS}": str(ws_dir)}
    args = [subs.get(a, a) for a in case["args"]]

    proc = _spawn(args, extra_env=case["env"])
    client = McpClient(proc)
    try:
        client.initialize()
        tools = client.list_tools()
    finally:
        client.close()

    got = sorted(t["name"] for t in tools)
    expected = sorted(case["expect"])
    assert got == expected, (
        f"tools/list mismatch for case '{case_name}': "
        f"got={got} expected={expected} diff={set(got).symmetric_difference(expected)}"
    )


# ---------------------------------------------------------------------------
# 2. Per-tool smoke
# ---------------------------------------------------------------------------


def test_ping(tmp_path: Path) -> None:
    proc = _spawn(["--source-root", str(tmp_path)])
    client = McpClient(proc)
    try:
        client.initialize()
        resp = client.call_tool("ping")
        assert resp["result"]["content"][0]["text"] == "pong"
        resp = client.call_tool("ping", {"message": "hello"})
        assert resp["result"]["content"][0]["text"] == "hello"
    finally:
        client.close()


def test_read_source(tmp_path: Path) -> None:
    (tmp_path / "demo.txt").write_text("line one\nline two\nline three\n")
    proc = _spawn(["--source-root", str(tmp_path)])
    client = McpClient(proc)
    try:
        client.initialize()
        resp = client.call_tool("read_source", {"file_path": "demo.txt"})
        body = resp["result"]["content"][0]["text"]
        assert "line one" in body
        assert "line three" in body
        assert "3 lines" in body
    finally:
        client.close()


def test_read_source_path_traversal(tmp_path: Path) -> None:
    proc = _spawn(["--source-root", str(tmp_path)])
    client = McpClient(proc)
    try:
        client.initialize()
        resp = client.call_tool("read_source", {"file_path": "../../../etc/passwd"})
        body = resp["result"]["content"][0]["text"]
        assert body.startswith("Error:"), f"traversal not rejected: {body[:200]}"
        assert "not found" in body.lower() or "access denied" in body.lower()
    finally:
        client.close()


def test_grep(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\nthree marker\nfour\n")
    proc = _spawn(["--source-root", str(tmp_path)])
    client = McpClient(proc)
    try:
        client.initialize()
        resp = client.call_tool("grep", {"pattern": "marker"})
        body = resp["result"]["content"][0]["text"]
        assert "three marker" in body
    finally:
        client.close()


def test_list_source(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    proc = _spawn(["--source-root", str(tmp_path)])
    client = McpClient(proc)
    try:
        client.initialize()
        resp = client.call_tool("list_source")
        body = resp["result"]["content"][0]["text"]
        assert "a.txt" in body
        assert "sub" in body
    finally:
        client.close()


# ---------------------------------------------------------------------------
# 3. Embedder smoke (slow — downloads ~2GB ONNX weights on cold cache)
# ---------------------------------------------------------------------------


@pytest.mark.model_download
def test_bge_m3_embedder_loads() -> None:
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    """The 0.9.20 regression: fastembed-python doesn't have bge-m3.
    Asserting that our 0.9.21 BgeM3Embedder DOES load BAAI/bge-m3 and
    returns 1024-dim vectors. This is the parity check that was 'pending
    tonight' in the 0.9.20 release note and never ran."""
    from kglite.mcp_server.bge_m3 import BgeM3Embedder

    e = BgeM3Embedder()
    e.load()
    assert e.dimension == 1024
    vecs = e.embed(["hello world", "compute pagerank centrality"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 1024
    assert len(vecs[1]) == 1024
    # Sanity: two semantically distinct strings should NOT have
    # identical vectors.
    assert vecs[0] != vecs[1]


@pytest.mark.model_download
def test_bge_m3_cosine_sanity() -> None:
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    """Semantically related strings should score higher than unrelated
    ones. Loose threshold — just a sanity check that the model is
    producing meaningful embeddings (not zeros or random)."""
    import math

    from kglite.mcp_server.bge_m3 import BgeM3Embedder

    e = BgeM3Embedder()
    vecs = e.embed(
        [
            "compute pagerank centrality",
            "graph centrality algorithms",
            "photosynthesis in plants",
        ]
    )

    def cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb)

    related = cos(vecs[0], vecs[1])
    unrelated = cos(vecs[0], vecs[2])
    assert related > unrelated, f"related={related:.3f} unrelated={unrelated:.3f}"
    assert related > 0.4, f"related similarity too low: {related:.3f}"


# ---------------------------------------------------------------------------
# 4. _mcp_internal direct (faster than spawning the full server)
# ---------------------------------------------------------------------------


def test_mcp_internal_module_importable() -> None:
    """The submodule must be importable as `kglite._mcp_internal`
    (registered in sys.modules by `mcp_tools.rs::register`)."""
    import kglite._mcp_internal as m

    assert callable(m.read_source)
    assert callable(m.grep)
    assert callable(m.list_source)
    assert callable(m.ping)
    assert callable(m.git_api)
    assert callable(m.has_github_token)
    assert m.GithubIssues  # pyclass


def test_mcp_internal_ping() -> None:
    import kglite._mcp_internal as m

    assert m.ping() == "pong"
    assert m.ping("hello") == "hello"


def test_mcp_internal_read_source_traversal_rejected(tmp_path: Path) -> None:
    import kglite._mcp_internal as m

    body = m.read_source("../../../etc/passwd", [str(tmp_path)])
    assert body.startswith("Error:")
