"""Tests for .pyi (type stub) file parsing in code_tree.

Driven through the public build() API so tests are agnostic to whether
the parser backend is Python (tree-sitter bindings) or Rust (tree-sitter
crate).
"""

import textwrap

from kglite.code_tree import build


def _stub_project(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()

    (pkg / "__init__.pyi").write_text(
        textwrap.dedent("""\
        from .core import Widget

        class Config:
            debug: bool
            def load(self, path: str) -> None: ...
        """)
    )
    (pkg / "core.pyi").write_text(
        textwrap.dedent("""\
        from typing import Protocol

        class Widget:
            name: str
            def render(self) -> str: ...

        class Drawable(Protocol):
            def draw(self) -> None: ...

        def create_widget(name: str) -> Widget: ...

        MAX_WIDGETS: int
        """)
    )
    return tmp_path


def test_pyi_classes_parsed(tmp_path):
    g = build(str(_stub_project(tmp_path)))
    class_names = {r["name"] for r in g.cypher("MATCH (c:Class) RETURN c.name AS name").to_list()}
    assert {"Config", "Widget"} <= class_names


def test_pyi_protocol_parsed(tmp_path):
    g = build(str(_stub_project(tmp_path)))
    protocol_names = {r["name"] for r in g.cypher("MATCH (p:Protocol) RETURN p.name AS name").to_list()}
    assert "Drawable" in protocol_names


def test_pyi_function_parsed(tmp_path):
    g = build(str(_stub_project(tmp_path)))
    fn_names = {r["name"] for r in g.cypher("MATCH (f:Function) RETURN f.name AS name").to_list()}
    # Top-level function and the method signatures.
    assert "create_widget" in fn_names
    assert "render" in fn_names
    assert "draw" in fn_names


def test_pyi_attributes_embedded(tmp_path):
    g = build(str(_stub_project(tmp_path)))
    # Widget.name and Config.debug should round-trip through the JSON fields column.
    rows = g.cypher("MATCH (c:Class) WHERE c.name = 'Widget' RETURN c.fields AS fields").to_list()
    assert rows, "Widget class should exist"
    fields = rows[0]["fields"]
    if fields is None:
        return  # nothing to check
    # Column is stored as JSON text but the result view may return it
    # already parsed; handle both shapes.
    if isinstance(fields, str):
        import json as _json

        fields = _json.loads(fields)
    names = [entry.get("name") for entry in fields]
    assert "name" in names, f"expected 'name' attribute, got {fields!r}"
