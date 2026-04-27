"""Manifest reader — Cargo.toml parsing.

Covers the `[lib] crate-type` capture for issue #15: PyO3 cdylib projects
need this distinction so downstream queries can tell apart `pub fn` as
"real Rust API export" vs. "noise — only #[pyfunction] / #[pymethods]
matters".
"""

from __future__ import annotations

import textwrap

from kglite.code_tree import build


def _write(tmp_path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))


class TestCrateType:
    def test_cdylib_crate_type_captured(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": """
                [package]
                name = "fixture"
                version = "0.0.0"
                edition = "2021"

                [lib]
                crate-type = ["cdylib", "rlib"]
                """,
                "src/lib.rs": "pub fn placeholder() {}\n",
            },
        )
        g = build(str(tmp_path))
        rows = g.cypher("MATCH (p:Project) RETURN p.name AS name, p.crate_type AS crate_type").to_list()
        assert len(rows) == 1
        assert rows[0]["name"] == "fixture"
        assert rows[0]["crate_type"] == "cdylib,rlib"

    def test_no_lib_section_yields_null_crate_type(self, tmp_path):
        # Plain `lib` crate without an explicit [lib] section — crate_type
        # must surface as None / empty so downstream queries can distinguish.
        _write(
            tmp_path,
            {
                "Cargo.toml": """
                [package]
                name = "plain"
                version = "0.0.0"
                edition = "2021"
                """,
                "src/lib.rs": "pub fn placeholder() {}\n",
            },
        )
        g = build(str(tmp_path))
        rows = g.cypher("MATCH (p:Project) RETURN p.crate_type AS crate_type").to_list()
        assert len(rows) == 1
        assert rows[0]["crate_type"] in (None, "")


class TestToolingOnlyPyproject:
    """Regression: a tooling-only pyproject.toml (scripts/lint config, no
    installable Python package, no maturin) used to trap `build()` into
    parsing only `tests/`. The whole-repo fallback now fires when the
    manifest declares zero source roots."""

    def test_pyproject_with_no_source_roots_falls_back_to_whole_repo(self, tmp_path):
        _write(
            tmp_path,
            {
                "pyproject.toml": """
                [tool.poetry]
                name = "tooling-scripts"
                packages = [{ include = "*.py", from = "." }]

                [build-system]
                build-backend = "poetry.core.masonry.api"
                """,
                "src/main.c": "int main(void) { return 0; }\n",
                "tests/test_thing.py": "def test_x():\n    pass\n",
            },
        )
        g = build(str(tmp_path))
        files = [r["path"] for r in g.cypher("MATCH (f:File) RETURN f.path AS path").to_list()]
        assert any(p.endswith("src/main.c") for p in files), f"expected src/main.c in graph, got: {files}"
