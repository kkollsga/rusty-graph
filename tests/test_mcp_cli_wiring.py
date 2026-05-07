"""Tests for cli.py manifest wiring — path resolution, _apply_manifest,
and the --mcp-config / sibling-detection branch."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("mcp_methods")

from kglite.mcp_server.cli import (  # noqa: E402
    _apply_manifest,
    _load_manifest_from_args,
    _resolve_source_roots,
)
from kglite.mcp_server.manifest import (  # noqa: E402
    Manifest,
    ManifestError,
    TrustConfig,
    load_manifest,
)


class _CaptureMcp:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *_args, **_kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def _args(**overrides) -> argparse.Namespace:
    base = argparse.Namespace(graph="graph.kgl", embedder=None, name="x", mcp_config=None)
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class TestResolveSourceRoots:
    def test_relative_path_under_yaml_parent(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "demo_mcp.yaml"
        yaml_path.write_text("name: demo\n")
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        resolved = _resolve_source_roots(["./data"], yaml_path)
        assert resolved == [str(data_dir.resolve())]

    def test_dotdot_traversal(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        sibling = tmp_path / "sibling"
        sibling.mkdir()
        yaml_path = sub / "demo_mcp.yaml"
        yaml_path.write_text("\n")
        resolved = _resolve_source_roots(["../sibling"], yaml_path)
        assert resolved == [str(sibling.resolve())]

    def test_absolute_path(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "demo_mcp.yaml"
        yaml_path.write_text("\n")
        target = tmp_path / "elsewhere"
        target.mkdir()
        resolved = _resolve_source_roots([str(target)], yaml_path)
        assert resolved == [str(target.resolve())]

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "demo_mcp.yaml"
        yaml_path.write_text("\n")
        with pytest.raises(ManifestError, match="not an existing directory"):
            _resolve_source_roots(["./nope"], yaml_path)

    def test_multiple_roots(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "demo_mcp.yaml"
        yaml_path.write_text("\n")
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        resolved = _resolve_source_roots(["./a", "./b"], yaml_path)
        assert resolved == [str(a.resolve()), str(b.resolve())]


class TestApplyManifest:
    def test_no_source_roots_registers_nothing(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "demo_mcp.yaml"
        yaml_path.write_text("\n")
        manifest = Manifest(yaml_path=yaml_path)
        mcp = _CaptureMcp()
        summary = _apply_manifest(mcp, manifest)
        assert mcp.tools == {}
        assert summary["source_roots"] == []

    def test_source_roots_registers_three_tools(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "demo_mcp.yaml"
        yaml_path.write_text("\n")
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        manifest = Manifest(
            yaml_path=yaml_path,
            source_roots=["./data"],
            trust=TrustConfig(),
        )
        mcp = _CaptureMcp()
        summary = _apply_manifest(mcp, manifest)
        assert set(mcp.tools) == {"read_source", "grep", "list_source"}
        assert summary["source_roots"] == [str(data_dir.resolve())]

    def test_missing_source_root_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "demo_mcp.yaml"
        yaml_path.write_text("\n")
        manifest = Manifest(yaml_path=yaml_path, source_roots=["./missing"])
        mcp = _CaptureMcp()
        with pytest.raises(ManifestError, match="not an existing directory"):
            _apply_manifest(mcp, manifest)


class TestLoadManifestFromArgs:
    def test_no_config_no_sibling_returns_none(self, tmp_path: Path) -> None:
        graph = tmp_path / "demo.kgl"
        graph.write_bytes(b"\x00")
        result = _load_manifest_from_args(_args(), graph)
        assert result is None

    def test_sibling_auto_detected(self, tmp_path: Path) -> None:
        graph = tmp_path / "demo.kgl"
        graph.write_bytes(b"\x00")
        sibling = tmp_path / "demo_mcp.yaml"
        sibling.write_text("name: From Sibling\n")
        result = _load_manifest_from_args(_args(), graph)
        assert result is not None
        assert result.name == "From Sibling"

    def test_explicit_config_overrides(self, tmp_path: Path) -> None:
        graph = tmp_path / "demo.kgl"
        graph.write_bytes(b"\x00")
        sibling = tmp_path / "demo_mcp.yaml"
        sibling.write_text("name: Sibling\n")
        explicit = tmp_path / "explicit.yaml"
        explicit.write_text("name: Explicit\n")
        result = _load_manifest_from_args(_args(mcp_config=str(explicit)), graph)
        assert result is not None
        assert result.name == "Explicit"

    def test_explicit_missing_raises(self, tmp_path: Path) -> None:
        graph = tmp_path / "demo.kgl"
        graph.write_bytes(b"\x00")
        with pytest.raises(ManifestError, match="--mcp-config path does not exist"):
            _load_manifest_from_args(_args(mcp_config=str(tmp_path / "nope.yaml")), graph)

    def test_validation_error_propagates(self, tmp_path: Path) -> None:
        graph = tmp_path / "demo.kgl"
        graph.write_bytes(b"\x00")
        sibling = tmp_path / "demo_mcp.yaml"
        sibling.write_text("bogus: 1\n")
        with pytest.raises(ManifestError, match="unknown top-level keys"):
            _load_manifest_from_args(_args(), graph)


class TestEndToEndManifest:
    """Build a real yaml file, load it via load_manifest, apply via _apply_manifest."""

    def test_minimal_source_root_roundtrip(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "hello.txt").write_text("hi\n")
        yaml_path = tmp_path / "demo_mcp.yaml"
        yaml_path.write_text("source_root: ./data\n")
        manifest = load_manifest(yaml_path)
        mcp = _CaptureMcp()
        _apply_manifest(mcp, manifest)
        # Verify the registered read_source actually reads files in the resolved root.
        out = mcp.tools["read_source"](file_path="hello.txt")
        assert "hi" in out

    def test_dotdot_root_works_end_to_end(self, tmp_path: Path) -> None:
        # /tmp/X/scrape/data.json + /tmp/X/graphs/foo_mcp.yaml referencing ../scrape
        scrape = tmp_path / "scrape"
        scrape.mkdir()
        (scrape / "snapshot.json").write_text('{"ok": true}\n')
        graphs = tmp_path / "graphs"
        graphs.mkdir()
        yaml_path = graphs / "foo_mcp.yaml"
        yaml_path.write_text("source_root: ../scrape\n")
        manifest = load_manifest(yaml_path)
        mcp = _CaptureMcp()
        summary = _apply_manifest(mcp, manifest)
        assert summary["source_roots"] == [str(scrape.resolve())]
        out = mcp.tools["read_source"](file_path="snapshot.json")
        assert '"ok": true' in out
