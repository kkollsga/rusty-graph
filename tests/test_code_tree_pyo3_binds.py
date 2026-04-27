"""PyO3 BINDS edges — Python wrapper Function -> Rust pymethod Function.

The resolver matches Python `kglite.Class.method` to Rust
`crate::*::*::Class::method` whenever the Rust function has
`metadata.is_pymethod = true` (or in our promoted column world,
`is_pymethod = true`). Symptom on KGLite: closes the cross-language
dead-code gap where pyapi-exposed functions look uncalled.
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


def test_python_class_method_binds_to_rust_pymethod(tmp_path):
    # Mixed Python + Rust project mimicking KGLite's pyapi pattern.
    _write(
        tmp_path,
        {
            "pyproject.toml": """
            [project]
            name = "mixed"
            version = "0.1.0"
            """,
            "mixed/__init__.pyi": """
            class Widget:
                def add_thing(self, x: int) -> None: ...
                def remove_thing(self, x: int) -> None: ...
            """,
            "Cargo.toml": """
            [package]
            name = "mixed"
            version = "0.1.0"
            """,
            "src/lib.rs": """
            use pyo3::prelude::*;

            #[pyclass]
            pub struct Widget;

            #[pymethods]
            impl Widget {
                fn add_thing(&mut self, x: u32) {}
                fn remove_thing(&mut self, x: u32) {}
            }
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        """
        MATCH (py:Function)-[:BINDS]->(rs:Function)
        WHERE py.qualified_name ENDS WITH ".Widget.add_thing"
           OR py.qualified_name ENDS WITH ".Widget.remove_thing"
        RETURN py.qualified_name AS py, rs.qualified_name AS rs
        """
    ).to_list()
    bound = {r["py"].rsplit(".", 2)[-2] + "." + r["py"].rsplit(".", 2)[-1]: r["rs"] for r in rows}
    assert "Widget.add_thing" in bound, bound
    assert bound["Widget.add_thing"].endswith("::Widget::add_thing")
    assert "Widget.remove_thing" in bound, bound
    assert bound["Widget.remove_thing"].endswith("::Widget::remove_thing")
