"""Tests for kglite.mcp_server.embedder — manifest-driven embedder factory loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite.mcp_server.embedder import load_embedder
from kglite.mcp_server.manifest import EmbedderConfig, ManifestError


def _write_module(tmp_path: Path, content: str, name: str = "user_emb.py") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


SIMPLE_EMBEDDER = """
class FakeEmbedder:
    def __init__(self, dim=4, name="fake"):
        self.dim = dim
        self.name = name

    def embed(self, texts):
        return [[0.0] * self.dim for _ in texts]
"""


class TestTrustGates:
    def test_both_signals_required_yaml_off(self, tmp_path: Path) -> None:
        _write_module(tmp_path, SIMPLE_EMBEDDER)
        cfg = EmbedderConfig(module="user_emb.py", klass="FakeEmbedder")
        with pytest.raises(ManifestError, match="allow_embedder"):
            load_embedder(cfg, manifest_dir=tmp_path, trust_flag=True, allow_embedder=False)

    def test_both_signals_required_cli_off(self, tmp_path: Path) -> None:
        _write_module(tmp_path, SIMPLE_EMBEDDER)
        cfg = EmbedderConfig(module="user_emb.py", klass="FakeEmbedder")
        with pytest.raises(ManifestError, match="--trust-tools"):
            load_embedder(cfg, manifest_dir=tmp_path, trust_flag=False, allow_embedder=True)


class TestLoading:
    def test_loads_and_instantiates(self, tmp_path: Path) -> None:
        _write_module(tmp_path, SIMPLE_EMBEDDER)
        cfg = EmbedderConfig(module="user_emb.py", klass="FakeEmbedder", kwargs={"dim": 8})
        emb = load_embedder(cfg, manifest_dir=tmp_path, trust_flag=True, allow_embedder=True)
        assert emb.dim == 8
        assert emb.name == "fake"
        assert emb.embed(["a", "b"]) == [[0.0] * 8, [0.0] * 8]

    def test_kwargs_overrides(self, tmp_path: Path) -> None:
        _write_module(tmp_path, SIMPLE_EMBEDDER)
        cfg = EmbedderConfig(module="user_emb.py", klass="FakeEmbedder", kwargs={"dim": 2, "name": "x"})
        emb = load_embedder(cfg, manifest_dir=tmp_path, trust_flag=True, allow_embedder=True)
        assert emb.dim == 2 and emb.name == "x"

    def test_resolves_relative_to_manifest_dir(self, tmp_path: Path) -> None:
        sub = tmp_path / "nested"
        sub.mkdir()
        _write_module(sub, SIMPLE_EMBEDDER, name="emb.py")
        cfg = EmbedderConfig(module="emb.py", klass="FakeEmbedder")
        emb = load_embedder(cfg, manifest_dir=sub, trust_flag=True, allow_embedder=True)
        assert emb.dim == 4


class TestErrors:
    def test_module_missing(self, tmp_path: Path) -> None:
        cfg = EmbedderConfig(module="nope.py", klass="X")
        with pytest.raises(ManifestError, match="does not exist"):
            load_embedder(cfg, manifest_dir=tmp_path, trust_flag=True, allow_embedder=True)

    def test_class_missing(self, tmp_path: Path) -> None:
        _write_module(tmp_path, "x = 1\n")
        cfg = EmbedderConfig(module="user_emb.py", klass="Missing")
        with pytest.raises(ManifestError, match="not found"):
            load_embedder(cfg, manifest_dir=tmp_path, trust_flag=True, allow_embedder=True)

    def test_class_not_callable(self, tmp_path: Path) -> None:
        _write_module(tmp_path, "FakeEmbedder = 42\n")
        cfg = EmbedderConfig(module="user_emb.py", klass="FakeEmbedder")
        with pytest.raises(ManifestError, match="not callable"):
            load_embedder(cfg, manifest_dir=tmp_path, trust_flag=True, allow_embedder=True)

    def test_constructor_explodes(self, tmp_path: Path) -> None:
        _write_module(
            tmp_path,
            "class Bad:\n    def __init__(self):\n        raise RuntimeError('boom')\n",
        )
        cfg = EmbedderConfig(module="user_emb.py", klass="Bad")
        with pytest.raises(ManifestError, match="failed to instantiate"):
            load_embedder(cfg, manifest_dir=tmp_path, trust_flag=True, allow_embedder=True)

    def test_module_with_syntax_error(self, tmp_path: Path) -> None:
        _write_module(tmp_path, "def bad(:\n    pass\n")
        cfg = EmbedderConfig(module="user_emb.py", klass="X")
        with pytest.raises(ManifestError, match="error executing"):
            load_embedder(cfg, manifest_dir=tmp_path, trust_flag=True, allow_embedder=True)
