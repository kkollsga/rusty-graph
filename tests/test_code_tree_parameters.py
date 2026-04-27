"""Structured parameter list on Function nodes + USES_TYPE on parameter types.

Parameters arrive as a JSON-encoded list on the `parameters` property
(serialized from `Vec<ParameterInfo>` by the builder). USES_TYPE edges should
fire on parameter type annotations even when the type doesn't appear in the
return position.
"""

import json
import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _write(tmp_path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))


def _params(graph, qname_suffix: str) -> list[dict]:
    rows = graph.cypher(
        "MATCH (f:Function) WHERE f.qualified_name ENDS WITH $suf RETURN f.parameters AS params",
        params={"suf": qname_suffix},
    ).to_list()
    assert rows, f"no Function node ending with {qname_suffix!r}"
    raw = rows[0]["params"]
    if not raw:
        return []
    # KGLite's Cypher value converter auto-decodes JSON-ish strings into
    # native Python objects, so `parameters` may come back already parsed.
    if isinstance(raw, list):
        return raw
    return json.loads(raw)


def _uses_type(graph, qname_suffix: str) -> set[str]:
    """Return the set of qualified type names f -[USES_TYPE]-> t."""
    rows = graph.cypher(
        "MATCH (f:Function)-[:USES_TYPE]->(t) WHERE f.qualified_name ENDS WITH $suf RETURN t.qualified_name AS name",
        params={"suf": qname_suffix},
    ).to_list()
    return {r["name"] for r in rows}


class TestPython:
    def test_python_parameters_extracted(self, tmp_path):
        _write(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/m.py": """
                def f(a: int, b: str = "x", *args, **kwargs):
                    return a
                """,
            },
        )
        g = build(str(tmp_path))
        params = _params(g, ".f")
        assert [p["name"] for p in params] == ["a", "b", "args", "kwargs"]
        assert params[0]["type_annotation"] == "int"
        assert params[1]["type_annotation"] == "str"
        assert params[1]["default"] == '"x"'
        assert params[2]["kind"] == "variadic"
        assert params[3]["kind"] == "kw_variadic"

    def test_python_uses_type_on_parameter(self, tmp_path):
        _write(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/types.py": """
                class Widget:
                    pass
                """,
                "pkg/m.py": """
                from pkg.types import Widget
                def use(w: Widget) -> int:
                    return 1
                """,
            },
        )
        g = build(str(tmp_path))
        types = _uses_type(g, ".m.use")
        assert any(t.endswith("Widget") for t in types), types


class TestRust:
    def test_rust_parameters_extracted(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": """
                [package]
                name = "demo"
                version = "0.1.0"
                """,
                "src/lib.rs": """
                pub fn f(a: u32, b: String) -> u32 {
                    a + b.len() as u32
                }
                """,
            },
        )
        g = build(str(tmp_path))
        params = _params(g, "::f")
        assert [p["name"] for p in params] == ["a", "b"]
        assert params[0]["type_annotation"] is not None
        assert params[1]["type_annotation"] is not None

    def test_rust_uses_type_on_parameter(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": """
                [package]
                name = "demo"
                version = "0.1.0"
                """,
                "src/lib.rs": """
                pub struct Widget;
                pub fn use_widget(w: Widget) -> bool { true }
                """,
            },
        )
        g = build(str(tmp_path))
        types = _uses_type(g, "::use_widget")
        assert any(t.endswith("::Widget") for t in types), types
