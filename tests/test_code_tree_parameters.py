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


class TestReceiver:
    def test_go_method_receiver_kind(self, tmp_path):
        """Go method `func (c *Call) lock()` — parameters has one Receiver entry."""
        _write(
            tmp_path,
            {
                "go.mod": "module demo\n\ngo 1.21\n",
                "main.go": """
                package demo
                type Call struct{}
                func (c *Call) lock() {}
                """,
            },
        )
        g = build(str(tmp_path))
        params = _params(g, ".lock")
        assert len(params) == 1, params
        assert params[0]["kind"] == "receiver"
        assert params[0]["name"] == "c"
        assert "Call" in (params[0]["type_annotation"] or "")

    def test_rust_self_parameter_kind_is_receiver(self, tmp_path):
        """Rust &self method — parameters has Receiver entry with owner type."""
        _write(
            tmp_path,
            {
                "Cargo.toml": """
                [package]
                name = "demo"
                version = "0.1.0"
                """,
                "src/lib.rs": """
                pub struct Foo;
                impl Foo {
                    pub fn bar(&self, x: u32) {}
                    pub fn baz(&mut self) {}
                    pub fn consume(self) {}
                }
                """,
            },
        )
        g = build(str(tmp_path))
        bar = _params(g, "::bar")
        assert bar[0]["kind"] == "receiver"
        assert bar[0]["type_annotation"] == "&Foo"
        # Regular x param follows the receiver
        assert bar[1]["name"] == "x" and bar[1]["kind"] == "positional"

        baz = _params(g, "::baz")
        assert baz[0]["kind"] == "receiver"
        assert baz[0]["type_annotation"] == "&mut Foo"

        consume = _params(g, "::consume")
        assert consume[0]["kind"] == "receiver"
        assert consume[0]["type_annotation"] == "Foo"

    def test_param_count_excludes_receiver(self, tmp_path):
        """`param_count` shouldn't include the receiver — it's not user-supplied."""
        _write(
            tmp_path,
            {
                "Cargo.toml": """
                [package]
                name = "demo"
                version = "0.1.0"
                """,
                "src/lib.rs": """
                pub struct Foo;
                impl Foo {
                    pub fn one_arg(&self, x: u32) {}
                }
                """,
            },
        )
        g = build(str(tmp_path))
        rows = g.cypher(
            "MATCH (f:Function) WHERE f.qualified_name ENDS WITH '::one_arg' RETURN f.param_count AS p"
        ).to_list()
        assert rows[0]["p"] == 1, "receiver should not count toward param_count"
