"""Promoted metadata flags surface as queryable Function/Class properties.

Phase 4.2 promoted is_pymethod, is_pymodule, is_ffi, ffi_kind, is_static,
is_abstract, is_property, is_classmethod (Function) plus is_pyclass (Class)
out of the JSON metadata blob into typed DataFrame columns.
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _write(tmp_path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))


def test_rust_pymethod_flag_promoted(tmp_path):
    _write(
        tmp_path,
        {
            "Cargo.toml": """
            [package]
            name = "demo"
            version = "0.1.0"
            """,
            "src/lib.rs": """
            use pyo3::prelude::*;

            #[pyclass]
            pub struct Foo;

            #[pymethods]
            impl Foo {
                fn bar(&self) {}
            }

            pub fn plain_fn() {}
            """,
        },
    )
    g = build(str(tmp_path))
    py_methods = g.cypher("MATCH (f:Function {is_pymethod: true}) RETURN f.qualified_name AS q").to_list()
    assert any(r["q"].endswith("::Foo::bar") for r in py_methods), py_methods
    # plain_fn must not be flagged.
    plain = g.cypher(
        "MATCH (f:Function {is_pymethod: true}) WHERE f.qualified_name ENDS WITH '::plain_fn' RETURN f"
    ).to_list()
    assert plain == []


def test_rust_pyclass_flag_on_struct(tmp_path):
    _write(
        tmp_path,
        {
            "Cargo.toml": """
            [package]
            name = "demo"
            version = "0.1.0"
            """,
            "src/lib.rs": """
            use pyo3::prelude::*;

            #[pyclass]
            pub struct Exposed;

            pub struct Hidden;
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher("MATCH (s:Struct {is_pyclass: true}) RETURN s.name AS n").to_list()
    names = {r["n"] for r in rows}
    assert "Exposed" in names
    assert "Hidden" not in names


def test_python_property_and_classmethod_flags(tmp_path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/m.py": """
            class C:
                @property
                def computed(self):
                    return 1
                @classmethod
                def make(cls):
                    return cls()
                @staticmethod
                def util():
                    return 2
                def regular(self):
                    return 3
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        "MATCH (f:Function) WHERE f.qualified_name CONTAINS '.pkg.m.C.' "
        "RETURN f.name AS n, f.is_property AS prop, f.is_classmethod AS cm, f.is_static AS st"
    ).to_list()
    by_name = {r["n"]: r for r in rows}
    assert by_name["computed"]["prop"] is True
    assert by_name["make"]["cm"] is True
    assert by_name["util"]["st"] is True
    assert by_name["regular"]["prop"] is False
    assert by_name["regular"]["cm"] is False
