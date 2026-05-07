"""Tests for kglite.mcp_server.manifest — yaml manifest loader + validator."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from kglite.mcp_server.manifest import (  # noqa: E402
    CypherTool,
    Manifest,
    ManifestError,
    PythonTool,
    find_sibling_manifest,
    load_manifest,
)


def _write(tmp_path: Path, content: str, name: str = "manifest.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


class TestSiblingDetection:
    def test_finds_sibling_when_present(self, tmp_path: Path) -> None:
        graph = tmp_path / "demo.kgl"
        graph.write_bytes(b"\x00")
        sibling = tmp_path / "demo_mcp.yaml"
        sibling.write_text("name: demo\n")
        assert find_sibling_manifest(graph) == sibling

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        graph = tmp_path / "demo.kgl"
        graph.write_bytes(b"\x00")
        assert find_sibling_manifest(graph) is None

    def test_uses_basename_not_full_filename(self, tmp_path: Path) -> None:
        # demo.kgl -> demo_mcp.yaml (NOT demo.kgl_mcp.yaml)
        graph = tmp_path / "demo.kgl"
        graph.write_bytes(b"\x00")
        wrong = tmp_path / "demo.kgl_mcp.yaml"
        wrong.write_text("name: wrong\n")
        assert find_sibling_manifest(graph) is None


class TestEmptyAndMinimal:
    def test_empty_yaml_returns_empty_manifest(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, ""))
        assert isinstance(m, Manifest)
        assert m.tools == []
        assert m.source_roots == []
        assert m.trust.allow_python_tools is False

    def test_just_name(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "name: My Graph\n"))
        assert m.name == "My Graph"
        assert m.tools == []


class TestSourceRoots:
    def test_single_source_root(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "source_root: ./data\n"))
        assert m.source_roots == ["./data"]

    def test_source_roots_list(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "source_roots: [./data, ../other]\n"))
        assert m.source_roots == ["./data", "../other"]

    def test_both_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "source_root: ./data\nsource_roots: [./other]\n")
        with pytest.raises(ManifestError, match="either source_root .* or source_roots"):
            load_manifest(path)

    def test_empty_string_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "source_root: ''\n")
        with pytest.raises(ManifestError, match="non-empty string"):
            load_manifest(path)

    def test_non_string_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "source_root: 42\n")
        with pytest.raises(ManifestError, match="non-empty string"):
            load_manifest(path)

    def test_source_roots_with_non_string_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "source_roots: [./data, 42]\n")
        with pytest.raises(ManifestError, match="list of non-empty strings"):
            load_manifest(path)


class TestTrust:
    def test_default(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, ""))
        assert m.trust.allow_python_tools is False

    def test_explicit_true(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "trust:\n  allow_python_tools: true\n"))
        assert m.trust.allow_python_tools is True

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "trust:\n  bogus: true\n")
        with pytest.raises(ManifestError, match="unknown trust keys.*bogus"):
            load_manifest(path)

    def test_non_bool_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "trust:\n  allow_python_tools: yes_please\n")
        with pytest.raises(ManifestError, match="must be a bool"):
            load_manifest(path)


class TestCypherTool:
    def test_minimal(self, tmp_path: Path) -> None:
        m = load_manifest(
            _write(
                tmp_path,
                "tools:\n  - name: list_people\n    cypher: MATCH (p:Person) RETURN p.name\n",
            )
        )
        assert len(m.tools) == 1
        tool = m.tools[0]
        assert isinstance(tool, CypherTool)
        assert tool.name == "list_people"
        assert "MATCH (p:Person)" in tool.cypher

    def test_full(self, tmp_path: Path) -> None:
        m = load_manifest(
            _write(
                tmp_path,
                "tools:\n"
                "  - name: similar_sessions\n"
                "    description: top-k similar sessions\n"
                "    parameters:\n"
                "      type: object\n"
                "      properties:\n"
                "        session_id: {type: string}\n"
                "      required: [session_id]\n"
                "    cypher: |\n"
                "      MATCH (s:Session {id: $session_id})-[r:SIMILAR_TO]->(t)\n"
                "      RETURN t.title AS title, r.score AS score\n",
            )
        )
        tool = m.tools[0]
        assert isinstance(tool, CypherTool)
        assert tool.description == "top-k similar sessions"
        assert tool.parameters == {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        }
        assert "$session_id" in tool.cypher

    def test_empty_cypher_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "tools:\n  - {name: t, cypher: ''}\n")
        with pytest.raises(ManifestError, match="non-empty string"):
            load_manifest(path)


class TestPythonTool:
    def test_minimal(self, tmp_path: Path) -> None:
        m = load_manifest(
            _write(
                tmp_path,
                "tools:\n  - name: session_detail\n    python: ./gcn_tools.py\n    function: session_detail\n",
            )
        )
        tool = m.tools[0]
        assert isinstance(tool, PythonTool)
        assert tool.name == "session_detail"
        assert tool.python == "./gcn_tools.py"
        assert tool.function == "session_detail"

    def test_missing_function_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "tools:\n  - {name: t, python: ./tools.py}\n",
        )
        with pytest.raises(ManifestError, match="function:"):
            load_manifest(path)

    def test_invalid_function_name_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "tools:\n  - {name: t, python: ./tools.py, function: '1bad'}\n",
        )
        with pytest.raises(ManifestError, match="valid Python identifier"):
            load_manifest(path)


class TestToolValidation:
    def test_no_tools_key_is_fine(self, tmp_path: Path) -> None:
        m = load_manifest(_write(tmp_path, "name: ok\n"))
        assert m.tools == []

    def test_missing_name_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "tools:\n  - cypher: MATCH (n) RETURN n\n")
        with pytest.raises(ManifestError, match="needs a string `name:`"):
            load_manifest(path)

    def test_invalid_name_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "tools:\n  - {name: '1bad', cypher: 'MATCH (n) RETURN n'}\n")
        with pytest.raises(ManifestError, match="`name:`"):
            load_manifest(path)

    def test_no_kind_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "tools:\n  - {name: bare}\n")
        with pytest.raises(ManifestError, match="needs exactly one of"):
            load_manifest(path)

    def test_multiple_kinds_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "tools:\n  - {name: t, cypher: 'MATCH (n) RETURN n', python: ./x.py, function: f}\n",
        )
        with pytest.raises(ManifestError, match="multiple kinds"):
            load_manifest(path)

    def test_duplicate_names_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "tools:\n  - {name: same, cypher: 'MATCH (n) RETURN n'}\n  - {name: same, cypher: 'MATCH (m) RETURN m'}\n",
        )
        with pytest.raises(ManifestError, match="duplicate tool name"):
            load_manifest(path)

    def test_unknown_tool_key_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "tools:\n  - {name: t, cypher: 'MATCH (n) RETURN n', mystery: yes}\n",
        )
        with pytest.raises(ManifestError, match="unknown keys.*mystery"):
            load_manifest(path)


class TestTopLevelValidation:
    def test_unknown_top_level_key_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "bogus: 1\n")
        with pytest.raises(ManifestError, match="unknown top-level keys.*bogus"):
            load_manifest(path)

    def test_non_dict_top_level_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "- not\n- a\n- mapping\n")
        with pytest.raises(ManifestError, match="top-level must be a mapping"):
            load_manifest(path)

    def test_yaml_parse_error_wrapped(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "name: [unclosed\n")
        with pytest.raises(ManifestError, match="YAML parse error"):
            load_manifest(path)

    def test_error_includes_file_path(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "bogus: 1\n")
        with pytest.raises(ManifestError) as exc_info:
            load_manifest(path)
        assert str(path) in str(exc_info.value)
