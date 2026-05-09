"""Tests for new manifest fields: overview_prefix, embedder, builtins, allow_embedder."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from kglite.mcp_server.manifest import (  # noqa: E402
    BuiltinsConfig,
    EmbedderConfig,
    ManifestError,
    find_workspace_manifest,
    load_manifest,
)


def _write(tmp_path: Path, content: str, name: str = "manifest.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


class TestOverviewPrefix:
    def test_simple_string(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "overview_prefix: 'Quick context'\n"))
        assert m.overview_prefix == "Quick context"

    def test_multiline_block(self, tmp_path: Path) -> None:
        yaml_text = "overview_prefix: |\n  line one\n  line two\n"
        m = load_manifest(_write(tmp_path, yaml_text))
        assert m.overview_prefix is not None
        assert "line one" in m.overview_prefix and "line two" in m.overview_prefix

    def test_unset_is_none(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "name: x\n"))
        assert m.overview_prefix is None

    def test_non_string_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="overview_prefix must be a string"):
            load_manifest(_write(tmp_path, "overview_prefix: 42\n"))


class TestBuiltins:
    def test_default_builtins(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "name: x\n"))
        assert isinstance(m.builtins, BuiltinsConfig)
        assert m.builtins.save_graph is False
        assert m.builtins.temp_cleanup == "never"

    def test_save_graph_true(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "builtins:\n  save_graph: true\n"))
        assert m.builtins.save_graph is True

    def test_temp_cleanup_on_overview(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "builtins:\n  temp_cleanup: on_overview\n"))
        assert m.builtins.temp_cleanup == "on_overview"

    def test_temp_cleanup_invalid_value(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="temp_cleanup must be one of"):
            load_manifest(_write(tmp_path, "builtins:\n  temp_cleanup: nuke\n"))

    def test_save_graph_non_bool(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="save_graph must be a bool"):
            load_manifest(_write(tmp_path, "builtins:\n  save_graph: yes_please\n"))

    def test_unknown_builtin_key(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="unknown builtins keys"):
            load_manifest(_write(tmp_path, "builtins:\n  fancy_thing: true\n"))


class TestEmbedderField:
    def test_unset_is_none(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "name: x\n"))
        assert m.embedder is None

    def test_minimal_module_class(self, tmp_path: Path) -> None:
        yaml_text = "embedder:\n  module: ./embedder.py\n  class: MyEmbedder\n"
        m = load_manifest(_write(tmp_path, yaml_text))
        assert isinstance(m.embedder, EmbedderConfig)
        assert m.embedder.module == "./embedder.py"
        assert m.embedder.klass == "MyEmbedder"
        assert m.embedder.kwargs == {}

    def test_with_kwargs(self, tmp_path: Path) -> None:
        yaml_text = (
            "embedder:\n  module: ./embedder.py\n  class: BgeM3\n  kwargs:\n    cooldown: 900\n    model: BAAI/bge-m3\n"
        )
        m = load_manifest(_write(tmp_path, yaml_text))
        assert m.embedder.kwargs == {"cooldown": 900, "model": "BAAI/bge-m3"}

    def test_missing_module_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="embedder.module"):
            load_manifest(_write(tmp_path, "embedder:\n  class: MyEmbedder\n"))

    def test_invalid_class_rejected(self, tmp_path: Path) -> None:
        yaml_text = "embedder:\n  module: ./embedder.py\n  class: '0bad-name'\n"
        with pytest.raises(ManifestError, match="embedder.class"):
            load_manifest(_write(tmp_path, yaml_text))

    def test_unknown_embedder_key(self, tmp_path: Path) -> None:
        yaml_text = "embedder:\n  module: ./e.py\n  class: E\n  extra: ignored\n"
        with pytest.raises(ManifestError, match="unknown embedder keys"):
            load_manifest(_write(tmp_path, yaml_text))


class TestAllowEmbedderTrust:
    def test_default_false(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "name: x\n"))
        assert m.trust.allow_embedder is False

    def test_set_true(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "trust:\n  allow_embedder: true\n"))
        assert m.trust.allow_embedder is True

    def test_non_bool_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="allow_embedder must be a bool"):
            load_manifest(_write(tmp_path, "trust:\n  allow_embedder: maybe\n"))


class TestWorkspaceManifestDetection:
    def test_finds_workspace_yaml(self, tmp_path: Path) -> None:
        manifest = tmp_path / "workspace_mcp.yaml"
        manifest.write_text("name: ws\n")
        assert find_workspace_manifest(tmp_path) == manifest

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert find_workspace_manifest(tmp_path) is None
