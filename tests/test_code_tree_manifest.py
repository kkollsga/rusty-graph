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
