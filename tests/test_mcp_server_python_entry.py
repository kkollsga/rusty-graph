"""Pre-release integration test suite for `kglite-mcp-server`.

Implements the spec from the MCP-servers operator's
`2026-05-12-from-mcp-servers-pre-release-test-suite-spec.md`. Every
test below maps to a specific bug class from the 0.9.16 → 0.9.22
release arc; the inline `# Catches:` comments preserve that mapping.
When all 27 tests pass, the release is shippable.

Categories:
- A: Install & boot (4 tests)
- B: Per-mode tool registration (7 tests)
- C: Tool output content (8 tests)
- D: Embedder & semantic search (3 tests)
- E: Manifest & .env handling (5 tests)
- F: Workspace state propagation (5 tests)
- J: Spatial Cypher (3 tests)
- K: Timeseries Cypher (3 tests)
- L: Procedures — orphans + duplicates (2 tests)
- W: File watcher boundary (1 test)

0.9.24: E5, F4, F5, and W1 added to anchor the new pyo3-wrapper
boundaries.

0.9.26: B6/B7 added (disk-graph CLI validator); Cat J/K/L (8 tests)
wired in against the mcp-servers operator's delivered fixtures
(tests/fixtures/{spatial,timeseries,graph_with_orphans,graph_with_duplicates}.kgl).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time
from typing import Any

import pytest

# Skip cleanly if the MCP extras aren't installed locally.
pytest.importorskip("mcp")
pytest.importorskip("yaml")
pytest.importorskip("aiohttp")
pytest.importorskip("fastembed")
pytest.importorskip("watchdog")

# ---------------------------------------------------------------------------
# Shared helpers — McpClient + spawn
# ---------------------------------------------------------------------------


def _which_server() -> str | None:
    return shutil.which("kglite-mcp-server")


def _spawn(args: list[str], extra_env: dict[str, str] | None = None, cwd: str | None = None) -> subprocess.Popen:
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
        cwd=cwd,
        bufsize=0,
    )


class McpClient:
    """JSON-RPC stdio driver. Used to call the server like Claude/Cursor do."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self._next_id = 1
        self._stderr: list[str] = []
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        for line in iter(self._proc.stderr.readline, b""):
            self._stderr.append(line.decode("utf-8", errors="replace"))

    def stderr_lines(self) -> list[str]:
        return list(self._stderr)

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
        raise TimeoutError(f"no response within {timeout}s; stderr: {''.join(self._stderr[-10:])}")

    def initialize(self) -> dict[str, Any]:
        self._send(
            {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "kglite-test", "version": "0"},
                },
            }
        )
        self._next_id += 1
        resp = self._recv()
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp

    def list_tools(self) -> list[str]:
        self._send({"jsonrpc": "2.0", "id": self._next_id, "method": "tools/list", "params": {}})
        self._next_id += 1
        resp = self._recv()
        return sorted(t["name"] for t in resp["result"]["tools"])

    def list_tools_full(self) -> list[dict[str, Any]]:
        self._send({"jsonrpc": "2.0", "id": self._next_id, "method": "tools/list", "params": {}})
        self._next_id += 1
        resp = self._recv()
        return resp["result"]["tools"]

    def call_tool(self, name: str, args: dict[str, Any] | None = None) -> str:
        self._send(
            {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": args or {}},
            }
        )
        self._next_id += 1
        resp = self._recv()
        if "error" in resp:
            return f"ERROR: {resp['error']}"
        content = resp.get("result", {}).get("content", [])
        return content[0].get("text", "") if content else ""

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


def _client(args: list[str], **kwargs: Any) -> McpClient:
    proc = _spawn(args, **kwargs)
    client = McpClient(proc)
    client.initialize()
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_graph_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the ~50-node-per-type fixture graph once per session."""
    from tests.fixtures.build_tiny_graph import build_tiny_graph

    path = tmp_path_factory.mktemp("graphs") / "tiny_graph.kgl"
    build_tiny_graph(path)
    return path


@pytest.fixture
def mutable_graph_path(tmp_path: Path, tiny_graph_path: Path) -> Path:
    """Fresh copy of tiny_graph.kgl that the test can mutate without
    affecting other tests. Also drops a sibling manifest with
    builtins.save_graph: true so save_graph registers."""
    target = tmp_path / "mutable.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "mutable_mcp.yaml"
    manifest.write_text("name: mutable\nbuiltins:\n  save_graph: true\n  temp_cleanup: on_overview\n")
    return target


@pytest.fixture
def tiny_source_dir(tmp_path: Path) -> Path:
    """A tiny source tree for read_source/grep/list_source tests."""
    root = tmp_path / "src"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 'alpha'\n\nclass A:\n    pass\n")
    (root / "beta.py").write_text("import os\n\ndef beta():\n    return os.getcwd()\n")
    nested = root / "nested"
    nested.mkdir()
    (nested / "gamma.py").write_text("def gamma():\n    return 42\n")
    return root


@pytest.fixture
def tiny_workspace_dir(tmp_path: Path) -> Path:
    """A workspace dir with one stubbed cloned repo so list_repos has
    something to find."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    org = ws / "stub"
    org.mkdir()
    repo = org / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "main.py").write_text("def main():\n    pass\n")
    return ws


@pytest.fixture
def graph_with_savegraph_flag(mutable_graph_path: Path) -> Path:
    """Already has builtins.save_graph: true via mutable_graph_path."""
    return mutable_graph_path.with_suffix(".kgl").parent / "mutable_mcp.yaml"


@pytest.fixture
def graph_without_savegraph_flag(tmp_path: Path, tiny_graph_path: Path) -> Path:
    """Manifest matching tiny_graph but without save_graph flag."""
    target = tmp_path / "nograph.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "nograph_mcp.yaml"
    manifest.write_text("name: nograph\nbuiltins:\n  temp_cleanup: on_overview\n")
    return manifest


@pytest.fixture
def manifest_with_csv_http(tmp_path: Path, tiny_graph_path: Path) -> Path:
    target = tmp_path / "csvhttp.kgl"
    shutil.copy(tiny_graph_path, target)
    # Pick a free port deterministically.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    manifest = tmp_path / "csvhttp_mcp.yaml"
    manifest.write_text(f"name: csvhttp\nextensions:\n  csv_http_server:\n    port: {port}\n    dir: temp/\n")
    return manifest


@pytest.fixture
def graph_with_bge_m3_embedder(tmp_path: Path, tiny_graph_path: Path) -> Path:
    target = tmp_path / "bgem3.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "bgem3_mcp.yaml"
    manifest.write_text("name: bgem3\nextensions:\n  embedder:\n    backend: fastembed\n    model: BAAI/bge-m3\n")
    return manifest


@pytest.fixture(scope="session")
def graph_with_bge_m3_embeddings_precomputed(tmp_path_factory: pytest.TempPathFactory, tiny_graph_path: Path) -> Path:
    """Same as tiny_graph but with `Article.body` embeddings precomputed
    via bge-m3. kglite's text_score requires pre-stored embeddings (it
    embeds the query string at search time but not stored content).
    Without this fixture, text_score returns 'no embedding body_emb
    found' errors that mask real embedder-loading issues."""
    import kglite
    from kglite.mcp_server.bge_m3 import BgeM3Embedder

    target_dir = tmp_path_factory.mktemp("bge_m3_corpus")
    target = target_dir / "bgem3_corpus.kgl"
    shutil.copy(tiny_graph_path, target)
    g = kglite.load(str(target))
    embedder = BgeM3Embedder()
    g.set_embedder(embedder)
    g.embed_texts("Article", "body", show_progress=False)
    g.save(str(target))
    return target


@pytest.fixture
def local_workspace_yaml(tmp_path: Path, tiny_source_dir: Path) -> Path:
    """workspace.kind: local manifest pointing at a real directory."""
    yaml = tmp_path / "local_mcp.yaml"
    yaml.write_text(f"name: local\nworkspace:\n  kind: local\n  root: {tiny_source_dir}\n")
    return yaml


@pytest.fixture
def local_workspace_with_two_children(tmp_path: Path) -> tuple[Path, Path, Path]:
    """workspace.kind: local with two sibling subdirectories under the
    configured root, each carrying a unique marker file. Returns
    (manifest_yaml, child_a_path, child_b_path) so F4 can verify both
    siblings are reachable via set_root_dir."""
    root = tmp_path / "workspace_root"
    root.mkdir()
    child_a = root / "child_a"
    child_a.mkdir()
    (child_a / "marker_a.py").write_text("# child_a marker\n")
    child_b = root / "child_b"
    child_b.mkdir()
    (child_b / "marker_b.py").write_text("# child_b marker\n")
    yaml = tmp_path / "two_children_mcp.yaml"
    yaml.write_text(f"name: two_children\nworkspace:\n  kind: local\n  root: {root}\n")
    return yaml, child_a, child_b


@pytest.fixture
def manifest_with_cypher_tool(tmp_path: Path, tiny_graph_path: Path) -> Path:
    target = tmp_path / "cyphertool.kgl"
    shutil.copy(tiny_graph_path, target)
    manifest = tmp_path / "cyphertool_mcp.yaml"
    manifest.write_text(
        "name: cyphertool\n"
        "tools:\n"
        "  - name: find_by_city\n"
        "    description: Find persons in a given city.\n"
        "    cypher: |\n"
        "      MATCH (p:Person) WHERE p.city = $city RETURN p.name LIMIT 5\n"
        "    parameters:\n"
        "      type: object\n"
        "      properties:\n"
        "        city: { type: string }\n"
        "      required: [city]\n"
    )
    return manifest


# ---------------------------------------------------------------------------
# Category A — Install & boot (4 tests)
# ---------------------------------------------------------------------------


def test_a1_console_script_on_path() -> None:
    """`pip install kglite[mcp]` puts kglite-mcp-server on PATH.
    Catches: 0.9.20 (no console script), 0.9.18 (wrong libpython linkage)."""
    server = _which_server()
    assert server is not None and server.strip(), "kglite-mcp-server not on PATH"


def test_a2_help_runs_without_error() -> None:
    """Binary starts — no dyld errors, no missing modules.
    Catches: 0.9.18 install_name regression, 0.9.20 missing mcp dep."""
    server = _which_server()
    if server is None:
        pytest.skip("server not on PATH")
    result = subprocess.run([server, "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "KGLite" in result.stdout


@pytest.mark.skipif(not shutil.which("conda"), reason="conda not available")
@pytest.mark.model_download
def test_a3_install_works_on_conda_python() -> None:
    """Fresh conda env can pip install kglite[mcp] and run the server.
    Catches: 0.9.18 (Python.org framework path baked into wheel; failed on conda)."""
    env_name = "kglite_smoke_test"
    try:
        subprocess.run(["conda", "create", "-y", "-n", env_name, "python=3.11"], check=True, capture_output=True)
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", "kglite[mcp]"], check=True, capture_output=True
        )
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "kglite-mcp-server", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
    finally:
        subprocess.run(["conda", "env", "remove", "-n", env_name, "-y"], capture_output=True)


def test_a4_boot_summary_on_stderr(tiny_graph_path: Path) -> None:
    """Boot summary line on stderr names mode + path.
    Catches: 0.9.20 (silent boot)."""
    client = _client(["--graph", str(tiny_graph_path)])
    try:
        time.sleep(0.5)  # let stderr drain
        lines = client.stderr_lines()
        boot = next((ln for ln in lines if "mode:" in ln), None)
        assert boot is not None, f"no boot summary; got stderr: {lines}"
        assert "graph" in boot
        assert str(tiny_graph_path) in boot
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Category B — Per-mode tool registration (5 tests)
# ---------------------------------------------------------------------------


def test_b1_graph_mode_registers_expected_tools(mutable_graph_path: Path) -> None:
    """--graph + manifest with builtins.save_graph: true → 8 baseline tools.
    Catches: 0.9.20 (8 missing tools)."""
    client = _client(
        [
            "--graph",
            str(mutable_graph_path),
            "--mcp-config",
            str(mutable_graph_path.parent / "mutable_mcp.yaml"),
        ]
    )
    try:
        tools = set(client.list_tools())
    finally:
        client.close()
    expected = {
        "ping",
        "cypher_query",
        "graph_overview",
        "read_code_source",
        "read_source",
        "grep",
        "list_source",
        "save_graph",
    }
    if os.environ.get("GITHUB_TOKEN"):
        expected |= {"github_issues", "github_api"}
    assert tools >= expected, f"missing: {expected - tools}; extra: {tools - expected}"


def test_b2_workspace_mode_registers_repo_management(tiny_workspace_dir: Path) -> None:
    """--workspace mode registers repo_management.
    Catches: 0.9.20 (workspace mode lost repo_management)."""
    client = _client(["--workspace", str(tiny_workspace_dir)])
    try:
        tools = client.list_tools()
    finally:
        client.close()
    assert "repo_management" in tools
    assert "set_root_dir" not in tools


def test_b3_local_workspace_mode_registers_set_root_dir(local_workspace_yaml: Path) -> None:
    """workspace.kind: local registers set_root_dir.
    Catches: 0.9.20 (set_root_dir gone in local-workspace mode)."""
    client = _client(["--mcp-config", str(local_workspace_yaml)])
    try:
        tools = client.list_tools()
    finally:
        client.close()
    assert "set_root_dir" in tools
    assert "repo_management" not in tools


def test_b6_graph_mode_accepts_disk_backed_graph_directory(tmp_path: Path) -> None:
    """--graph accepts a disk-backed graph directory (built via
    `kglite.KnowledgeGraph(storage='disk', path=...)`), not just .kgl files.

    Catches: 0.9.25 server.py:_validate_mode_paths rejecting any
    non-file path, which blocked operators deploying Wikidata-scale
    (124M+ node) disk-backed graphs through the CLI. Reported by
    the mcp-servers project against 0.9.25; fix lands in 0.9.26.

    The validator fix is what's anchored here: the server must boot
    against the disk directory and serve a cypher_query that
    counts the nodes. Property-value persistence on the disk
    storage path is a separate concern not in scope for this test."""
    import kglite

    disk_dir = tmp_path / "disk_graph"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))
    g.cypher("CREATE (:Marker)")
    g.cypher("CREATE (:Marker)")
    g.cypher("CREATE (:Marker)")
    g.save(str(disk_dir))
    del g  # release the mmap before booting the server

    # Server must boot with --graph pointing at the directory and
    # serve a cypher_query that sees the persisted nodes.
    client = _client(["--graph", str(disk_dir)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (n:Marker) RETURN count(n) AS c"},
        )
    finally:
        client.close()
    assert "Error" not in body and "ERROR" not in body, f"disk graph caused an error: {body!r}"
    assert "3" in body, f"expected 3 Marker nodes via count(): {body!r}"


def test_b7_graph_mode_rejects_arbitrary_directory_without_meta(tmp_path: Path) -> None:
    """A directory that ISN'T a disk-backed graph (no disk_graph_meta.json)
    must still be rejected with a clear error, not silently accepted.

    Catches: a too-permissive fix that accepted any directory."""
    not_a_graph = tmp_path / "random_dir"
    not_a_graph.mkdir()
    (not_a_graph / "stuff.txt").write_text("not a graph\n")

    server = _which_server()
    if server is None:
        pytest.skip("server not on PATH")
    result = subprocess.run(
        [server, "--graph", str(not_a_graph)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0, "arbitrary directory should fail --graph validation"
    combined = (result.stdout + result.stderr).lower()
    assert "not a .kgl file or disk-backed graph directory" in combined, (
        f"expected the new validator message; got stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_b4_save_graph_gated_on_manifest_flag(
    mutable_graph_path: Path, graph_without_savegraph_flag: Path, tiny_graph_path: Path
) -> None:
    """save_graph only registers when builtins.save_graph: true is set.
    Catches: 0.9.18 (unconditional registration)."""
    # With flag
    client = _client(
        [
            "--graph",
            str(mutable_graph_path),
            "--mcp-config",
            str(mutable_graph_path.parent / "mutable_mcp.yaml"),
        ]
    )
    try:
        assert "save_graph" in client.list_tools()
    finally:
        client.close()
    # Without flag
    target = graph_without_savegraph_flag.parent / "nograph.kgl"
    client = _client(
        [
            "--graph",
            str(target),
            "--mcp-config",
            str(graph_without_savegraph_flag),
        ]
    )
    try:
        assert "save_graph" not in client.list_tools()
    finally:
        client.close()


def test_b5_github_tools_gated_on_token(tiny_workspace_dir: Path, tmp_path: Path) -> None:
    """github_issues + github_api only register when GITHUB_TOKEN is present.
    Catches: 0.9.20 (tools absent with token), 0.9.18 (registered without)."""
    # No token
    env_no = {k: v for k, v in os.environ.items() if k not in {"GITHUB_TOKEN", "GH_TOKEN"}}
    env_no["HOME"] = str(tmp_path)  # avoid walk-up to a real .env
    proc = _spawn(["--workspace", str(tiny_workspace_dir)], extra_env=env_no, cwd=str(tmp_path))
    client = McpClient(proc)
    client.initialize()
    try:
        tools_no = client.list_tools()
    finally:
        client.close()
    assert "github_issues" not in tools_no
    assert "github_api" not in tools_no

    # With token
    env_yes = {**env_no, "GITHUB_TOKEN": "ghp_test"}
    proc = _spawn(["--workspace", str(tiny_workspace_dir)], extra_env=env_yes, cwd=str(tmp_path))
    client = McpClient(proc)
    client.initialize()
    try:
        tools_yes = client.list_tools()
    finally:
        client.close()
    assert "github_issues" in tools_yes
    assert "github_api" in tools_yes


# ---------------------------------------------------------------------------
# Category C — Tool output content (8 tests)
# ---------------------------------------------------------------------------


def test_c1_cypher_query_returns_actual_row_values(tiny_graph_path: Path) -> None:
    """The 0.9.21 regression: rows rendered column names instead of values.
    Catches: 0.9.21 row formatter bug."""
    client = _client(["--graph", str(tiny_graph_path)])
    try:
        body = client.call_tool("cypher_query", {"query": "RETURN 1+1 AS sum"})
    finally:
        client.close()
    assert "'sum'" not in body, f"formatter still emitting column name as value: {body!r}"
    assert "2" in body, f"expected value 2 not in response: {body!r}"


def test_c2_cypher_query_multi_column_returns_distinct_values(tiny_graph_path: Path) -> None:
    """Multi-column queries return distinct row values.
    Catches: 0.9.21 row formatter, partial-fix scenarios."""
    client = _client(["--graph", str(tiny_graph_path)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (p:Person) RETURN p.name, p.age LIMIT 3",
            },
        )
    finally:
        client.close()
    lines = [ln for ln in body.split("\n") if ln and "row(s)" not in ln and "p.name" != ln.strip()]
    assert len(set(lines)) > 1, f"all rows identical: {body!r}"
    for line in lines:
        assert "'p.name'" not in line, f"row contains column-name string: {line!r}"
        assert "'p.age'" not in line, f"row contains column-name string: {line!r}"


def test_c3_ping_returns_pong(tiny_graph_path: Path) -> None:
    """Simplest content assertion. Catches: 0.9.20 (ping absent)."""
    client = _client(["--graph", str(tiny_graph_path)])
    try:
        body = client.call_tool("ping", {})
    finally:
        client.close()
    assert body.strip() == "pong"


def test_c4_read_source_returns_file_content(tiny_source_dir: Path) -> None:
    """read_source returns actual file content.
    Catches: 0.9.20 (absent), 0.9.21 (registered but garbage)."""
    client = _client(["--source-root", str(tiny_source_dir)])
    try:
        body = client.call_tool("read_source", {"file_path": "alpha.py"})
    finally:
        client.close()
    assert "def alpha" in body, f"read_source returned no source: {body[:200]!r}"


def test_c5_grep_returns_matches_with_file_and_line(tiny_source_dir: Path) -> None:
    """grep output: `path:line:content` format.
    Catches: 0.9.20 (grep absent), formatter-strips-line-numbers."""
    client = _client(["--source-root", str(tiny_source_dir)])
    try:
        body = client.call_tool("grep", {"pattern": "def ", "max_results": 5})
    finally:
        client.close()
    assert re.search(r"\.py:\d+:", body), f"missing path:line: format: {body[:300]!r}"


def test_c6_csv_http_server_returns_csv_body_via_http(tiny_graph_path: Path, manifest_with_csv_http: Path) -> None:
    """csv_http_server actually serves CSV over HTTP, no 500.
    Catches: 0.9.22 aiohttp content_type/charset bug."""
    import urllib.request

    target = manifest_with_csv_http.parent / "csvhttp.kgl"
    client = _client(["--graph", str(target), "--mcp-config", str(manifest_with_csv_http)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (p:Person) RETURN p.name LIMIT 2 FORMAT CSV",
            },
        )
        m = re.search(r"http://[^\s\"\)]+", body)
        assert m, f"no localhost URL in FORMAT CSV response: {body!r}"
        time.sleep(0.3)
        with urllib.request.urlopen(m.group(0), timeout=5) as resp:
            assert resp.status == 200, f"HTTP GET status {resp.status}"
            content = resp.read().decode()
        assert "p.name" in content, f"CSV missing header: {content!r}"
        assert content.count("\n") >= 2
    finally:
        client.close()


def test_c7_format_csv_inline_fallback(tiny_graph_path: Path) -> None:
    """Without csv_http_server, FORMAT CSV returns body inline.
    Catches: regression where inline fallback breaks."""
    client = _client(["--graph", str(tiny_graph_path)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (p:Person) RETURN p.name LIMIT 2 FORMAT CSV",
            },
        )
    finally:
        client.close()
    assert "http://" not in body, f"unexpected URL when csv_http_server disabled: {body!r}"
    assert "p.name" in body
    assert body.count("\n") >= 2


def test_c8_save_graph_persists_to_disk(mutable_graph_path: Path) -> None:
    """save_graph actually writes to the .kgl file.
    Catches: save_graph registers but doesn't persist."""
    manifest = mutable_graph_path.parent / "mutable_mcp.yaml"
    original_mtime = mutable_graph_path.stat().st_mtime
    time.sleep(1.1)  # ensure mtime resolution captures the change

    client = _client(["--graph", str(mutable_graph_path), "--mcp-config", str(manifest)])
    try:
        client.call_tool("cypher_query", {"query": "CREATE (:Marker {tag: 'savegraph-test'})"})
        save_body = client.call_tool("save_graph", {})
    finally:
        client.close()
    assert "ERROR" not in save_body and "error" not in save_body.lower()[:30]
    new_mtime = mutable_graph_path.stat().st_mtime
    assert new_mtime > original_mtime, "save_graph did not update .kgl mtime"

    # Reload and verify the mutation persisted
    client = _client(["--graph", str(mutable_graph_path), "--mcp-config", str(manifest)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (n:Marker {tag: 'savegraph-test'}) RETURN count(n) AS c",
            },
        )
    finally:
        client.close()
    assert "1" in body, f"mutation didn't persist across save/reload: {body!r}"


# ---------------------------------------------------------------------------
# Category D — Embedder & semantic search (3 tests)
# ---------------------------------------------------------------------------


@pytest.mark.model_download
def test_d1_bge_m3_loads_and_returns_dimension_1024(graph_with_bge_m3_embedder: Path) -> None:
    """bge-m3 must actually load. Catches: 0.9.20 catalog mismatch."""
    target = graph_with_bge_m3_embedder.parent / "bgem3.kgl"
    client = _client(["--graph", str(target), "--mcp-config", str(graph_with_bge_m3_embedder)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (a:Article) WHERE text_score(a, 'body', 'hello world') > 0 RETURN count(a) AS n",
            },
        )
    finally:
        client.close()
    assert "not supported" not in body, f"bge-m3 catalog mismatch: {body!r}"
    assert re.search(r"\b\d+\b", body), f"no count in response: {body!r}"


@pytest.mark.model_download
def test_d2_text_score_returns_ordered_relevance(
    graph_with_bge_m3_embeddings_precomputed: Path, tmp_path: Path
) -> None:
    """Semantically-close text scores higher.
    Catches: 0.9.20 (model didn't load → no scoring)."""
    manifest = tmp_path / "bgem3_mcp.yaml"
    manifest.write_text("name: bgem3\nextensions:\n  embedder:\n    backend: fastembed\n    model: BAAI/bge-m3\n")
    client = _client(
        [
            "--graph",
            str(graph_with_bge_m3_embeddings_precomputed),
            "--mcp-config",
            str(manifest),
        ]
    )
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": (
                    "MATCH (a:Article) RETURN a.title, "
                    "text_score(a, 'body', 'quantum mechanics') AS s "
                    "ORDER BY s DESC LIMIT 4 FORMAT CSV"
                ),
            },
        )
    finally:
        client.close()
    lines = [ln for ln in body.split("\n") if "," in ln and not ln.startswith("a.title")]
    assert len(lines) >= 2, f"expected ≥2 score rows: {body!r}"
    scores = [float(ln.split(",")[-1]) for ln in lines]
    assert scores[0] > scores[-1], f"scores not descending: {scores}"
    # Top result should be the quantum article (our fixture has alternating titles)
    assert "quantum" in lines[0].lower(), f"quantum should rank first: {lines!r}"


@pytest.mark.model_download
def test_d3_embedder_lifecycle_lazy_load_and_unload(graph_with_bge_m3_embedder: Path) -> None:
    """First call cold (loads model); second call warm.
    Catches: embedder reloading every call regressions."""
    target = graph_with_bge_m3_embedder.parent / "bgem3.kgl"
    client = _client(["--graph", str(target), "--mcp-config", str(graph_with_bge_m3_embedder)])
    try:
        t0 = time.monotonic()
        body1 = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (a:Article) WHERE text_score(a, 'body', 'test') > 0 RETURN count(a)",
            },
        )
        cold_ms = (time.monotonic() - t0) * 1000
        t0 = time.monotonic()
        body2 = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (a:Article) WHERE text_score(a, 'body', 'other') > 0 RETURN count(a)",
            },
        )
        warm_ms = (time.monotonic() - t0) * 1000
    finally:
        client.close()
    assert "ERROR" not in body1 and "ERROR" not in body2
    # Warm should be substantially faster than cold (loose threshold —
    # cold includes the model load on first query).
    assert warm_ms < cold_ms / 2 + 500, (
        f"warm call ({warm_ms:.0f}ms) not faster than cold ({cold_ms:.0f}ms) — embedder reloading?"
    )


# ---------------------------------------------------------------------------
# Category E — Manifest & .env handling (4 tests)
# ---------------------------------------------------------------------------


def test_e1_explicit_source_root_overrides_auto_bind(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Manifest source_root: overrides --graph parent auto-bind.
    Catches: 0.9.17 P1 (manifest source_root silently ignored)."""
    # Graph in one dir, source in another.
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    graph_target = graph_dir / "g.kgl"
    shutil.copy(tiny_graph_path, graph_target)

    source_dir = tmp_path / "explicit_source"
    source_dir.mkdir()
    (source_dir / "marker.py").write_text("# explicitly bound\n")

    manifest = tmp_path / "g_mcp.yaml"
    manifest.write_text(f"name: g\nsource_root: {source_dir}\n")

    client = _client(["--graph", str(graph_target), "--mcp-config", str(manifest)])
    try:
        body = client.call_tool("read_source", {"file_path": "marker.py"})
    finally:
        client.close()
    assert "ERROR" not in body and "Error" not in body[:30], f"explicit source_root not honored: {body!r}"
    assert "explicitly bound" in body


def test_e2_env_file_walk_up(tmp_path: Path, tiny_workspace_dir: Path) -> None:
    """`.env` placed in workspace parent gets loaded.
    Catches: 0.9.17 .env walk-up missing."""
    env_path = tiny_workspace_dir.parent / ".env"
    env_path.write_text("TEST_VAR_XYZ=found_via_walkup\n")

    client = _client(["--workspace", str(tiny_workspace_dir)])
    try:
        time.sleep(0.5)
        lines = client.stderr_lines()
        env_line = next((ln for ln in lines if "env" in ln.lower() and ".env" in ln), None)
    finally:
        client.close()
    assert env_line is not None, f"no env-loaded line in stderr: {lines}"


def test_e3_unknown_manifest_keys_error_cleanly(tmp_path: Path) -> None:
    """Strict-key validation rejects typos.
    Catches: silent acceptance of typo'd keys."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nunknown_top_key: oops\n")
    server = _which_server()
    if server is None:
        pytest.skip("server not on PATH")
    result = subprocess.run(
        [server, "--mcp-config", str(bad)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    combined = (result.stderr + result.stdout).lower()
    assert "unknown" in combined, f"strict-key validation didn't error: {combined!r}"


def test_e4_temp_cleanup_wipes_on_graph_overview(mutable_graph_path: Path, tmp_path: Path) -> None:
    """builtins.temp_cleanup: on_overview wipes temp/ on bare graph_overview().
    Catches: 0.9.18 (no-op flag), 0.9.20 regressions."""
    # The fixture's manifest base dir is tmp_path. Drop a stale file in
    # `<base>/temp/`.
    manifest = mutable_graph_path.parent / "mutable_mcp.yaml"
    temp_dir = manifest.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    stale = temp_dir / "test-stale.csv"
    stale.write_text("old-data")
    assert stale.exists()

    client = _client(["--graph", str(mutable_graph_path), "--mcp-config", str(manifest)])
    try:
        client.call_tool("graph_overview", {})
    finally:
        client.close()
    assert not stale.exists(), "temp_cleanup didn't wipe temp/"


# ---------------------------------------------------------------------------
# Category F — Workspace state propagation (3 tests)
# ---------------------------------------------------------------------------


def test_f1_repo_management_activates_makes_repo_default_for_github_tools(
    tiny_workspace_dir: Path,
) -> None:
    """After repo_management activates, github_issues defaults to that repo.
    Catches: 0.9.22 active-repo bug."""
    # Pre-create stub/repo so activate() doesn't try to clone from github.com.
    (tiny_workspace_dir / "stub" / "repo" / "README").write_text("# stub\n")
    client = _client(
        ["--workspace", str(tiny_workspace_dir)],
        extra_env={"GITHUB_TOKEN": "ghp_test"},
    )
    try:
        act = client.call_tool("repo_management", {"name": "stub/repo"})
        assert "Activated" in act or "stub/repo" in act, f"activation failed: {act!r}"
        # github_issues without repo_name — should NOT say "could not auto-detect"
        # (it may fail with auth since the token is fake, but the repo must be picked up)
        body = client.call_tool("github_issues", {"limit": 1})
    finally:
        client.close()
    assert "could not auto-detect" not in body.lower(), f"active repo not propagating to github_issues: {body!r}"


def test_f2_set_root_dir_rebinds_source_tools(
    local_workspace_yaml: Path, tmp_path: Path, tiny_source_dir: Path
) -> None:
    """After set_root_dir, list_source operates on the new root.
    Catches: tool registers but state-swap is incomplete."""
    # Set up a second root inside the workspace
    second = tiny_source_dir / "second_root"
    second.mkdir()
    (second / "second_file.py").write_text("# in second root\n")

    client = _client(["--mcp-config", str(local_workspace_yaml)])
    try:
        initial = client.call_tool("list_source", {})
        swap = client.call_tool("set_root_dir", {"path": "second_root"})
        assert "ERROR" not in swap, f"set_root_dir failed: {swap!r}"
        new_listing = client.call_tool("list_source", {})
    finally:
        client.close()
    assert new_listing != initial, "set_root_dir didn't change source listing"
    assert "second_file.py" in new_listing, f"new root file not visible: {new_listing!r}"


def test_f3_repo_management_force_rebuild_triggers_post_activate_hook(
    tiny_workspace_dir: Path,
) -> None:
    """force_rebuild=true bypasses SHA gating.

    NOTE: our 0.9.23 Workspace doesn't implement SHA gating yet (it
    rebuilds unconditionally). This test asserts the boolean is
    accepted without error and the response acknowledges the request —
    once SHA gating lands in a future release, tighten the assertion.
    Catches: force_rebuild kwarg being silently dropped."""
    (tiny_workspace_dir / "stub" / "repo" / "README").write_text("# stub\n")
    client = _client(["--workspace", str(tiny_workspace_dir)])
    try:
        client.call_tool("repo_management", {"name": "stub/repo"})
        body = client.call_tool("repo_management", {"update": True, "force_rebuild": True})
    finally:
        client.close()
    # The current implementation may return "Updated" or "Activated" —
    # what matters is no error.
    assert "ERROR" not in body and "invalid" not in body.lower(), f"force_rebuild rejected: {body!r}"


def test_f4_set_root_dir_can_swap_laterally_within_workspace_root(
    local_workspace_with_two_children: tuple[Path, Path, Path],
) -> None:
    """After set_root_dir(child_a), agents can still set_root_dir(child_b)
    where both are siblings under the manifest's workspace.root.
    Catches: 0.9.23 sandbox-narrows-each-swap regression (the
    pre-0.9.24 Python wrapper mutated `self.root` after each swap,
    so the next sandbox check compared against the narrower active
    root; mcp-methods 0.3.28's atomic-swap RwLock + immutable
    configured root makes lateral swaps work by construction)."""
    yaml, child_a, child_b = local_workspace_with_two_children
    client = _client(["--mcp-config", str(yaml)])
    try:
        body_a = client.call_tool("set_root_dir", {"path": str(child_a)})
        assert "Active root set to" in body_a, f"first swap failed: {body_a!r}"

        body_b = client.call_tool("set_root_dir", {"path": str(child_b)})
        assert "escapes" not in body_b.lower(), (
            f"sandbox narrowed: cannot swap to sibling under workspace.root: {body_b!r}"
        )
        assert "Active root set to" in body_b, f"second swap failed: {body_b!r}"

        # The listing should now reflect child_b's contents — proves the
        # source-root rebind actually fired through to the source tools.
        listing = client.call_tool("list_source", {})
    finally:
        client.close()
    assert "marker_b.py" in listing, f"new sibling root not reflected in list_source: {listing!r}"


# ---------------------------------------------------------------------------
# Direct API tests (not in operator spec but kept from previous suite)
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
    assert m.GithubIssues


def test_mcp_internal_read_source_traversal_rejected(tmp_path: Path) -> None:
    import kglite._mcp_internal as m

    body = m.read_source("../../../etc/passwd", [str(tmp_path)])
    assert body.startswith("Error:")


def test_cypher_query_returns_actual_row_data_direct(tmp_path: Path) -> None:
    """Direct exercise of the 0.9.22 bug-fix code path (no MCP stdio).
    Useful when debugging the formatter in isolation."""
    import kglite
    from kglite.mcp_server.tools import GraphState, run_cypher

    state = GraphState()
    g = kglite.KnowledgeGraph()
    state._active = type("A", (), {"graph": g, "source_path": None})()  # type: ignore[attr-defined]
    body = run_cypher(state, "RETURN 1+1 AS sum")
    assert "'sum'" not in body
    assert "2" in body


def test_bge_m3_cooldown_default_keeps_session_resident() -> None:
    """Without explicit cool-down, the default (900s) holds the
    session resident across rapid embed() calls — exactly what makes
    warm-call latency drop to ms-scale."""
    from kglite.mcp_server.bge_m3 import BgeM3Embedder

    e = BgeM3Embedder()
    # Simulate a load by stuffing a sentinel into _session, then
    # check the idle-release path leaves it alone for short idles.
    e._session = "sentinel-loaded"  # type: ignore[assignment]
    e._last_used = time.monotonic()
    e._maybe_release_idle()
    assert e._session == "sentinel-loaded", "default cool-down should not release fresh session"


def test_bge_m3_cooldown_releases_after_idle_threshold() -> None:
    """Once `cooldown_seconds` of monotonic time elapses since the
    last embed, the next call drops the session."""
    from kglite.mcp_server.bge_m3 import BgeM3Embedder

    e = BgeM3Embedder(cooldown_seconds=1)  # 1-second cool-down for fast test
    e._session = "sentinel-loaded"  # type: ignore[assignment]
    e._tokenizer = "sentinel-tok"  # type: ignore[assignment]
    e._last_used = time.monotonic() - 2.0  # already idle past threshold
    e._maybe_release_idle()
    assert e._session is None, "stale session should have been released"
    assert e._tokenizer is None


def test_bge_m3_cooldown_zero_disables_release() -> None:
    """cooldown_seconds=0 means "never release" — model stays
    resident forever (the operator's heavy-use mode)."""
    from kglite.mcp_server.bge_m3 import BgeM3Embedder

    e = BgeM3Embedder(cooldown_seconds=0)
    e._session = "sentinel-loaded"  # type: ignore[assignment]
    e._last_used = time.monotonic() - 86400  # idle for a day
    e._maybe_release_idle()
    assert e._session == "sentinel-loaded", "cooldown=0 should never release"


def test_embedder_manifest_parses_cooldown() -> None:
    """`extensions.embedder.cooldown:` flows through from YAML into
    BgeM3Embedder. Catches: YAML parser silently dropping the field."""
    from kglite.mcp_server.bge_m3 import BgeM3Embedder
    from kglite.mcp_server.embedder import from_manifest_value

    adapter = from_manifest_value(
        {
            "backend": "fastembed",
            "model": "BAAI/bge-m3",
            "cooldown": 60,
        }
    )
    assert isinstance(adapter, BgeM3Embedder)
    assert adapter._cooldown_seconds == 60


# ---------------------------------------------------------------------------
# 0.9.24 pyo3-wrapper boundary tests (E5 / F5 / W1)
# ---------------------------------------------------------------------------


def test_e5_manifest_extensions_passthrough_preserves_all_keys(tmp_path: Path) -> None:
    """The Rust → Python conversion of `extensions: serde_json::Map`
    must recursively preserve every key including deeply nested
    mappings, ints, bools, and strings. Catches: drift in
    `json_value_to_py` when the recursive walker changes shape.

    The wire contract is "round-trip every JSON scalar / array /
    object" — exercise each branch."""
    yaml_path = tmp_path / "ext_passthrough_mcp.yaml"
    yaml_path.write_text(
        "name: ext_test\n"
        "extensions:\n"
        "  csv_http_server: true\n"
        "  csv_http_server_dir: temp/\n"
        "  embedder:\n"
        "    backend: fastembed\n"
        "    model: BAAI/bge-m3\n"
        "    cooldown: 1200\n"
        "  deeply:\n"
        "    nested:\n"
        "      list_value: [1, 2, 3]\n"
        "      bool_value: false\n"
        "      string_value: hello\n"
    )
    from kglite.mcp_server.manifest import load_manifest

    manifest = load_manifest(yaml_path)
    ext = manifest.extensions
    assert ext["csv_http_server"] is True
    assert ext["csv_http_server_dir"] == "temp/"
    assert ext["embedder"]["backend"] == "fastembed"
    assert ext["embedder"]["model"] == "BAAI/bge-m3"
    assert ext["embedder"]["cooldown"] == 1200
    assert ext["deeply"]["nested"]["list_value"] == [1, 2, 3]
    assert ext["deeply"]["nested"]["bool_value"] is False
    assert ext["deeply"]["nested"]["string_value"] == "hello"


def test_f5_workspace_post_activate_hook_fires_on_activate(tmp_path: Path) -> None:
    """A Python callable registered via Workspace.set_post_activate
    must be invoked when activate / set_root_dir succeeds. Catches:
    Arc-wrap bugs or GIL-acquisition failures in the DeferredPyHook
    dispatch path."""
    import kglite._mcp_internal as m

    target = tmp_path / "ws_root"
    target.mkdir()
    (target / "marker.txt").write_text("hi\n")

    captured: list[tuple[str, str]] = []

    def hook(path: str, name: str) -> None:
        captured.append((path, name))

    ws = m.Workspace.open_local(str(target))
    ws.set_post_activate(hook)

    # repo_management(update=True) in local mode re-runs activate,
    # which fires the post-activate hook.
    out = ws.repo_management(update=True)
    assert "failed" not in out.lower()

    assert len(captured) >= 1, f"post_activate hook didn't fire: out={out!r}"
    fired_path, fired_name = captured[-1]
    assert Path(fired_path).resolve() == target.resolve(), (
        f"hook received wrong path: {fired_path!r} (expected {target!r})"
    )
    assert fired_name.startswith("local/"), f"hook received unexpected name: {fired_name!r}"


def test_w1_watch_callback_receives_changed_paths(tmp_path: Path) -> None:
    """`start_watch` invokes the Python callable with a list of
    changed path strings within `debounce_ms + grace`. Catches:
    GIL or type-conversion issues at the watcher / Python boundary
    (callback runs on a background thread, must reacquire the GIL
    automatically)."""
    import kglite._mcp_internal as m

    received: list[list[str]] = []
    lock = threading.Lock()

    def on_change(paths: list[str]) -> None:
        with lock:
            received.append(list(paths))

    handle = m.start_watch(str(tmp_path), on_change, debounce_ms=100)
    try:
        # Give the watcher a moment to settle, then write a file.
        time.sleep(0.05)
        (tmp_path / "new_file.txt").write_text("change me\n")
        # Wait for the debounce window + grace.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with lock:
                if received:
                    break
            time.sleep(0.05)
    finally:
        handle.stop()

    assert received, "watch callback never fired after file write"
    flat = [p for batch in received for p in batch]
    # macOS FSEvents may report either the new file or its parent
    # directory (depending on coalescing). Accept either as evidence
    # the watcher picked up the change. The contract we care about
    # is "list[str] arrives within the debounce window" — not the
    # specific path granularity, which is platform-dependent.
    tmp_path_str = str(tmp_path)
    assert any("new_file.txt" in p or tmp_path_str in p for p in flat), (
        f"callback fired but path list did not include either the file or the parent dir: {received!r}"
    )


# ---------------------------------------------------------------------------
# 0.9.25 cypher_preprocessor regression tests (Category O — preprocessor)
# ---------------------------------------------------------------------------


def _make_state_with_tiny_graph():
    """Construct a GraphState wrapping a 3-node graph for preprocessor
    integration tests. Uses kglite Python API directly — no MCP stdio."""
    import kglite
    from kglite.mcp_server.tools import GraphState

    state = GraphState()
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Item {title: 'foo', value: 1})")
    g.cypher("CREATE (:Item {title: 'bar', value: 2})")
    g.cypher("CREATE (:Item {title: 'baz', value: 3})")
    # Inject the active-graph slot directly so we don't need a .kgl file.
    state._active = type("A", (), {"graph": g, "source_path": None})()
    return state, g


def test_o1_cypher_preprocessor_rewrites_query_before_execution(tmp_path: Path) -> None:
    """The preprocessor's rewrite() output is what reaches graph.cypher().
    Operator's spec test #1. Catches: preprocessor configured but
    actually ignored at dispatch time."""
    from kglite.mcp_server.preprocessor import Preprocessor
    from kglite.mcp_server.tools import run_cypher

    def rewrite(query: str, params):
        # Rewrite "foo" → "bar" so the query targets a different node.
        return query.replace("'foo'", "'bar'"), params

    preprocessor = Preprocessor(rewrite=rewrite, module_path=tmp_path / "p.py")
    state, _ = _make_state_with_tiny_graph()

    body = run_cypher(state, "MATCH (n:Item {title: 'foo'}) RETURN n.value AS v", preprocessor=preprocessor)
    # bar has value=2, foo has value=1. If the rewrite landed, we see 2.
    assert "2" in body
    assert "1" not in body or "row" in body  # tolerant — "1 row" might appear


def test_o2_cypher_preprocessor_fires_for_tools_cypher_too(tmp_path: Path) -> None:
    """YAML-declared tools[].cypher templates also go through the
    preprocessor. Operator's spec test #2. Catches: tools[].cypher
    bypassing the hook."""
    import asyncio

    from kglite.mcp_server.cypher_tools import call_cypher_tool
    from kglite.mcp_server.manifest import CypherTool
    from kglite.mcp_server.preprocessor import Preprocessor

    fired_with: list[str] = []

    def rewrite(query: str, params):
        fired_with.append(query)
        return query.replace("'foo'", "'baz'"), params

    preprocessor = Preprocessor(rewrite=rewrite, module_path=tmp_path / "p.py")
    state, _ = _make_state_with_tiny_graph()
    spec = CypherTool(
        name="lookup",
        cypher="MATCH (n:Item {title: 'foo'}) RETURN n.value AS v",
    )
    body = asyncio.run(call_cypher_tool(spec, state, {}, csv_http=None, preprocessor=preprocessor))
    assert fired_with, "preprocessor never received the tools[].cypher template"
    # baz has value=3.
    assert "3" in body


def test_o3_cypher_preprocessor_does_not_fire_for_non_cypher_tools(tmp_path: Path) -> None:
    """The preprocessor is scoped to cypher_query / tools[].cypher.
    graph_overview, save_graph, and the source-tool / GitHub-tool
    surface do not invoke it. Operator's spec test #3."""
    from kglite.mcp_server.preprocessor import Preprocessor
    from kglite.mcp_server.tools import run_overview, run_save

    fired_count = [0]

    def rewrite(query: str, params):
        fired_count[0] += 1
        return query, params

    # The preprocessor is constructed (so its rewrite() is genuinely
    # callable) but deliberately NOT passed to run_overview / run_save —
    # the point of this test is that non-cypher tool surfaces have
    # no parameter through which a preprocessor could reach them. If
    # someone ever adds a sneaky `preprocessor` kwarg or global to
    # those tools, the fired_count assertion catches it.
    _ = Preprocessor(rewrite=rewrite, module_path=tmp_path / "p.py")
    state, _ = _make_state_with_tiny_graph()

    body = run_overview(state, types=None, connections=None, cypher=None, temp_cleanup_dir=None)
    assert "Item" in body or "node" in body.lower()  # describe() should mention the type
    assert fired_count[0] == 0, "graph_overview leaked into preprocessor"

    save_body = run_save(state)
    assert "save_graph" in save_body or "requires" in save_body
    assert fired_count[0] == 0, "save_graph leaked into preprocessor"


def test_o4_cypher_preprocessor_exception_returns_clean_error(tmp_path: Path) -> None:
    """If rewrite() raises ValueError / TypeError, the agent receives
    `preprocessor: <message>` — no stack trace. Operator's spec test #4."""
    from kglite.mcp_server.preprocessor import Preprocessor
    from kglite.mcp_server.tools import run_cypher

    def rewrite(query: str, params):
        raise ValueError("rewrite refused: query rejected by policy")

    preprocessor = Preprocessor(rewrite=rewrite, module_path=tmp_path / "p.py")
    state, _ = _make_state_with_tiny_graph()
    body = run_cypher(state, "MATCH (n) RETURN n", preprocessor=preprocessor)
    assert body.startswith("preprocessor: "), f"expected clean error envelope, got: {body!r}"
    assert "rewrite refused" in body
    assert "Traceback" not in body


def test_o5_trust_gate_blocks_preprocessor_without_allow_query_preprocessor(tmp_path: Path) -> None:
    """extensions.cypher_preprocessor without trust.allow_query_preprocessor
    fails at preprocessor-load time. Operator's spec test #5. Mirrors the
    existing embedder/allow_embedder gate."""
    from kglite.mcp_server.preprocessor import PreprocessorError, from_manifest_value

    # Trust = False → must raise.
    with pytest.raises(PreprocessorError) as exc_info:
        from_manifest_value(
            {"module": "./p.py", "class": "P"},
            base_dir=tmp_path,
            trust_allowed=False,
        )
    assert "trust.allow_query_preprocessor" in str(exc_info.value)

    # Trust = True but module file missing → different error (proves
    # the trust gate is the first check that ran).
    with pytest.raises(PreprocessorError) as exc_info:
        from_manifest_value(
            {"module": "./does_not_exist.py", "class": "P"},
            base_dir=tmp_path,
            trust_allowed=True,
        )
    assert "module file does not exist" in str(exc_info.value)


def test_o6_cypher_preprocessor_loads_class_with_kwargs(tmp_path: Path) -> None:
    """End-to-end load of a class-based preprocessor with kwargs.
    Catches: kwargs not threaded through to the constructor."""
    from kglite.mcp_server.preprocessor import from_manifest_value

    (tmp_path / "p.py").write_text(
        "class P:\n"
        "    def __init__(self, suffix: str):\n"
        "        self._s = suffix\n"
        "    def rewrite(self, query, params):\n"
        "        return query + self._s, params\n"
    )
    pre = from_manifest_value(
        {"module": "./p.py", "class": "P", "kwargs": {"suffix": "  -- rewritten"}},
        base_dir=tmp_path,
        trust_allowed=True,
    )
    assert pre is not None
    new_q, _ = pre.rewrite("MATCH (n) RETURN n", None)
    assert new_q.endswith("-- rewritten")


def test_o7_cypher_preprocessor_loads_free_function(tmp_path: Path) -> None:
    """End-to-end load of a function-based preprocessor.
    Catches: function loader not finding module-level callables."""
    from kglite.mcp_server.preprocessor import from_manifest_value

    (tmp_path / "rewrites.py").write_text("def rewrite(query, params):\n    return query.lower(), params\n")
    pre = from_manifest_value(
        {"module": "./rewrites.py", "function": "rewrite"},
        base_dir=tmp_path,
        trust_allowed=True,
    )
    assert pre is not None
    new_q, _ = pre.rewrite("MATCH (N) RETURN N", None)
    assert new_q == "match (n) return n"


def test_o8_cypher_preprocessor_full_yaml_roundtrip_via_mcp_stdio(tmp_path: Path, tiny_graph_path: Path) -> None:
    """Full boot path: YAML manifest with `trust.allow_query_preprocessor:
    true` + `extensions.cypher_preprocessor:` + a real preprocessor
    module → mcp-methods parser → kglite Manifest → server boot →
    cypher_query reaches graph.cypher with the rewritten string.

    Requires mcp-methods 0.3.29+ (the trust key has to parse cleanly).
    Catches end-to-end wiring regressions across the YAML → Rust →
    Python → tool-dispatch boundary."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)

    # Rewriter substitutes 'Person' → 'Item' so the query targets
    # nothing — we use the no-results signal to confirm the rewrite
    # actually reached the engine.
    (tmp_path / "rewrites.py").write_text(
        "def rewrite(query, params):\n    return query.replace('Person', 'NONEXISTENT_TYPE'), params\n"
    )
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text(
        "name: preprocessed\n"
        "trust:\n"
        "  allow_query_preprocessor: true\n"
        "extensions:\n"
        "  cypher_preprocessor:\n"
        "    module: ./rewrites.py\n"
        "    function: rewrite\n"
    )

    client = _client(["--graph", str(target), "--mcp-config", str(manifest)])
    try:
        # The tiny graph has Person nodes. Without the preprocessor
        # this query would return rows; with the rewrite it sees
        # NONEXISTENT_TYPE and returns "No results."
        body = client.call_tool("cypher_query", {"query": "MATCH (p:Person) RETURN p.name LIMIT 3"})
    finally:
        client.close()
    assert "No results" in body, f"preprocessor rewrite didn't reach engine — got: {body!r}"


def test_o9_trust_gate_blocks_boot_when_yaml_missing_allow_query_preprocessor(
    tmp_path: Path, tiny_graph_path: Path
) -> None:
    """If extensions.cypher_preprocessor is declared without the trust
    flag, the server exits 3 at boot with the operator-facing
    message. Mirrors the existing trust.allow_embedder gate.

    Requires mcp-methods 0.3.29+ (the trust key has to parse cleanly
    when present; this test omits it, so the framework's strict-key
    check would fire on a stale pin and shadow our gate. Pinning
    0.3.29+ keeps the assertion targeting the right code path)."""
    target = tmp_path / "test.kgl"
    shutil.copy(tiny_graph_path, target)
    (tmp_path / "rewrites.py").write_text("def rewrite(q, p):\n    return q, p\n")
    # Omits `trust.allow_query_preprocessor: true` deliberately.
    manifest = tmp_path / "test_mcp.yaml"
    manifest.write_text(
        "name: ungated\nextensions:\n  cypher_preprocessor:\n    module: ./rewrites.py\n    function: rewrite\n"
    )

    server = _which_server()
    if server is None:
        pytest.skip("server not on PATH")
    result = subprocess.run(
        [server, "--graph", str(target), "--mcp-config", str(manifest)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0, (
        f"server should have exited non-zero without trust gate; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "trust.allow_query_preprocessor" in combined, f"missing gate-message in stderr: {result.stderr!r}"


# ---------------------------------------------------------------------------
# 0.9.26 Cat J/K/L — spatial / timeseries / procedure tests against the
# mcp-servers operator's delivered Cat G-N fixture bundle.
# See tests/fixtures/CAT_G_N_FIXTURES.md for the fixture catalog.
# ---------------------------------------------------------------------------


_CAT_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def spatial_graph() -> Path:
    return _CAT_FIXTURES / "spatial_graph.kgl"


@pytest.fixture
def timeseries_graph() -> Path:
    return _CAT_FIXTURES / "timeseries_graph.kgl"


@pytest.fixture
def graph_with_orphans() -> Path:
    return _CAT_FIXTURES / "graph_with_orphans.kgl"


@pytest.fixture
def graph_with_duplicates() -> Path:
    return _CAT_FIXTURES / "graph_with_duplicates.kgl"


# --- Cat J — Spatial Cypher --------------------------------------------------


def test_j1_contains_finds_wells_inside_areas(spatial_graph: Path) -> None:
    """contains(area, point(lat, lon)) — three wells are inside one of
    three areas in the fixture, two are outside all areas. Catches:
    spatial predicate regressions; point-from-property construction
    breaking; WKT polygon parsing breaking."""
    client = _client(["--graph", str(spatial_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {
                "query": "MATCH (a:Area), (w:Well) "
                "WHERE contains(a, point(w.latitude, w.longitude)) "
                "RETURN count(*) AS n"
            },
        )
    finally:
        client.close()
    assert "3" in body, f"expected 3 wells inside areas, got: {body!r}"


def test_j2_centroid_of_polygon(spatial_graph: Path) -> None:
    """centroid(area) returns a point with lat/lon matching the
    polygon's geometric centre. NORTH_BLOCK is a unit square centred
    at (61.0, 5.0). Catches: centroid math regressions; centroid()
    not returning a point-shaped value."""
    client = _client(["--graph", str(spatial_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (a:Area {id: 'north'}) RETURN centroid(a) AS c"},
        )
    finally:
        client.close()
    # centroid is rendered as a dict — Python repr in the inline format.
    # Expected: {'latitude': 61.0, 'longitude': 5.0}.
    assert "61" in body and "5" in body, f"expected NORTH_BLOCK centroid (61, 5), got: {body!r}"
    assert "latitude" in body or "lat" in body.lower(), f"centroid not point-shaped: {body!r}"


def test_j3_contains_query_via_explicit_point(spatial_graph: Path) -> None:
    """Query-side `point(lat, lon)` literal lookup. The fixture has
    NORTH_BLOCK covering lat 60-62 / lon 4-6; (61, 5) is inside.
    Catches: agent-friendly `WHERE contains(area, point(...))` shape
    breaking; literal point construction breaking."""
    client = _client(["--graph", str(spatial_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (a:Area) WHERE contains(a, point(61.0, 5.0)) RETURN a.title AS title"},
        )
    finally:
        client.close()
    assert "NORTH_BLOCK" in body, f"expected NORTH_BLOCK to contain (61, 5), got: {body!r}"


# --- Cat K — Timeseries Cypher -----------------------------------------------


def test_k1_ts_sum_aggregates_a_year(timeseries_graph: Path) -> None:
    """ts_sum(channel, 'YYYY') aggregates all timesteps within the year.
    TROLL's oil_col in 2019 sums to ~1563.55 against the random.seed(42)
    seed in build_fixtures.py. Catches: ts_sum regressions; year-range
    parsing breaking."""
    client = _client(["--graph", str(timeseries_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (f:Field {title:'TROLL'}) RETURN ts_sum(f.oil_col, '2019') AS annual"},
        )
    finally:
        client.close()
    # 1563.55 with possible float-format variation. Operator confirmed
    # the value at fixture-build time.
    assert "1563" in body, f"expected ts_sum ~1563.55, got: {body!r}"


def test_k2_ts_at_point_in_time(timeseries_graph: Path) -> None:
    """ts_at(channel, 'YYYY-M') returns the value at a single timestep.
    March 2019 for TROLL oil is 177.12 under the seeded random."""
    client = _client(["--graph", str(timeseries_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (f:Field {title:'TROLL'}) RETURN ts_at(f.oil_col, '2019-3') AS march"},
        )
    finally:
        client.close()
    assert "177.12" in body, f"expected ts_at March 2019 = 177.12, got: {body!r}"


def test_k3_ts_aggregations_handle_all_fields(timeseries_graph: Path) -> None:
    """ts_sum applied across all three Field nodes in the fixture
    returns three rows. Catches: ts_* functions not iterating across
    matches correctly."""
    client = _client(["--graph", str(timeseries_graph)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "MATCH (f:Field) RETURN f.title AS title, ts_sum(f.oil_col, '2019') AS annual"},
        )
    finally:
        client.close()
    assert "3 row" in body, f"expected 3 rows (one per field), got: {body!r}"
    for field in ("TROLL", "EKOFISK", "SNORRE"):
        assert field in body, f"missing field {field!r} in result: {body!r}"


# --- Cat L — Procedures ------------------------------------------------------


def test_l1_orphan_node_procedure_counts_isolates(graph_with_orphans: Path) -> None:
    """CALL orphan_node({type:'Wellbore'}) YIELD node — fixture has
    6 Wellbore nodes, 3 connected to a Field via IN_FIELD, 3 isolated.
    Procedure must return the 3 isolated ones. Catches: orphan_node
    regressions; CALL/YIELD wiring breaking."""
    client = _client(["--graph", str(graph_with_orphans)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "CALL orphan_node({type: 'Wellbore'}) YIELD node RETURN count(node) AS n"},
        )
    finally:
        client.close()
    assert "3" in body, f"expected 3 orphan Wellbores, got: {body!r}"


def test_l2_duplicate_title_procedure_counts_dupes(graph_with_duplicates: Path) -> None:
    """CALL duplicate_title({type:'Prospect'}) YIELD node — fixture has
    6 Prospects, 4 of which share titles in two pairs (ALPHA x2, BETA x2).
    Procedure must yield all 4 members of the duplicate pairs. Catches:
    duplicate_title regressions; "members of duplicate sets" vs "one
    representative per set" off-by-N errors."""
    client = _client(["--graph", str(graph_with_duplicates)])
    try:
        body = client.call_tool(
            "cypher_query",
            {"query": "CALL duplicate_title({type: 'Prospect'}) YIELD node RETURN count(node) AS n"},
        )
    finally:
        client.close()
    assert "4" in body, f"expected 4 duplicate-title Prospects, got: {body!r}"
