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


class TestSourceRootDiscovery:
    """A pyproject can declare its packages explicitly via several backends.
    Each declaration must produce a source root by design — not by accident
    of the find_python_package convention failing and the whole-repo fallback
    firing. A stub `<name>/__init__.py` next to the declared paths must not
    cause those declared paths to be skipped."""

    def test_poetry_packages_with_explicit_from_dir(self, tmp_path):
        # [project].name + a stub at the conventional location ensures
        # find_python_package succeeds → source_roots is non-empty → the
        # whole-repo fallback DOESN'T fire. Without the poetry-packages
        # discovery strategy, lib/poetryapp/core.py is silently skipped.
        _write(
            tmp_path,
            {
                "pyproject.toml": """
                [project]
                name = "poetryapp"
                version = "0.1.0"

                [tool.poetry]
                packages = [{ include = "poetryapp", from = "lib" }]

                [build-system]
                build-backend = "poetry.core.masonry.api"
                """,
                "poetryapp/__init__.py": "STUB = True\n",
                "lib/poetryapp/__init__.py": "",
                "lib/poetryapp/core.py": "def f(): pass\n",
            },
        )
        g = build(str(tmp_path))
        files = [r["path"] for r in g.cypher("MATCH (f:File) RETURN f.path AS path").to_list()]
        assert any(p.endswith("lib/poetryapp/core.py") for p in files), (
            f"expected lib/poetryapp/core.py in graph, got: {files}"
        )

    def test_tool_poetry_name_falls_back_when_no_project_name(self, tmp_path):
        # Pure poetry projects use [tool.poetry] for everything (no [project]
        # table). Without [tool.poetry].name fallback, `name` defaults to the
        # parent directory, find_python_package always misses, and the
        # whole-repo fallback fires for the wrong reason.
        _write(
            tmp_path,
            {
                "pyproject.toml": """
                [tool.poetry]
                name = "purepoetry"
                version = "0.1.0"

                [build-system]
                build-backend = "poetry.core.masonry.api"
                """,
                "purepoetry/__init__.py": "",
                "purepoetry/core.py": "def f(): pass\n",
                # Other top-level dir that the whole-repo fallback would parse
                # but a properly-resolved pyproject would not.
                "scratch/notes.py": "x = 1\n",
            },
        )
        g = build(str(tmp_path))
        files = [r["path"] for r in g.cypher("MATCH (f:File) RETURN f.path AS path").to_list()]
        assert any(p.endswith("purepoetry/core.py") for p in files), (
            f"expected purepoetry/core.py in graph, got: {files}"
        )
        # scratch/ is outside the declared package — must not bleed in.
        assert not any(p.endswith("scratch/notes.py") for p in files), (
            f"scratch/ should be outside declared roots, got: {files}"
        )

    def test_setuptools_packages_find_where(self, tmp_path):
        _write(
            tmp_path,
            {
                "pyproject.toml": """
                [project]
                name = "myapp"
                version = "0.1.0"

                [tool.setuptools.packages.find]
                where = ["lib"]

                [build-system]
                build-backend = "setuptools.build_meta"
                """,
                "myapp/__init__.py": "STUB = True\n",
                "lib/myapp/__init__.py": "",
                "lib/myapp/core.py": "def f(): pass\n",
            },
        )
        g = build(str(tmp_path))
        files = [r["path"] for r in g.cypher("MATCH (f:File) RETURN f.path AS path").to_list()]
        assert any(p.endswith("lib/myapp/core.py") for p in files), f"expected lib/myapp/core.py in graph, got: {files}"

    def test_setuptools_explicit_packages_list(self, tmp_path):
        _write(
            tmp_path,
            {
                "pyproject.toml": """
                [project]
                name = "explicitapp"
                version = "0.1.0"

                [tool.setuptools]
                packages = ["pkg_a", "pkg_b"]

                [build-system]
                build-backend = "setuptools.build_meta"
                """,
                # Stub at conventional location: defeats the accidental
                # fallback path so the assertions below test the explicit
                # discovery, not the fallback.
                "explicitapp/__init__.py": "STUB = True\n",
                "pkg_a/__init__.py": "",
                "pkg_a/mod.py": "def a(): pass\n",
                "pkg_b/__init__.py": "",
                "pkg_b/mod.py": "def b(): pass\n",
            },
        )
        g = build(str(tmp_path))
        files = [r["path"] for r in g.cypher("MATCH (f:File) RETURN f.path AS path").to_list()]
        assert any(p.endswith("pkg_a/mod.py") for p in files), f"expected pkg_a/mod.py in graph, got: {files}"
        assert any(p.endswith("pkg_b/mod.py") for p in files), f"expected pkg_b/mod.py in graph, got: {files}"

    def test_hatch_packages_declaration(self, tmp_path):
        _write(
            tmp_path,
            {
                "pyproject.toml": """
                [project]
                name = "hatchapp"
                version = "0.1.0"

                [tool.hatch.build.targets.wheel]
                packages = ["custom/hatchapp"]

                [build-system]
                build-backend = "hatchling.build"
                """,
                "hatchapp/__init__.py": "STUB = True\n",
                "custom/hatchapp/__init__.py": "",
                "custom/hatchapp/core.py": "def f(): pass\n",
            },
        )
        g = build(str(tmp_path))
        files = [r["path"] for r in g.cypher("MATCH (f:File) RETURN f.path AS path").to_list()]
        assert any(p.endswith("custom/hatchapp/core.py") for p in files), (
            f"expected custom/hatchapp/core.py in graph, got: {files}"
        )

    def test_pyproject_with_python_pkg_does_not_skip_other_languages(self, tmp_path):
        # Sibling of TestToolingOnlyPyproject: a pyproject WITH an installable
        # Python package, alongside primary code in another language.
        # find_python_package succeeds → source_roots non-empty → 0.8.25
        # fallback never fires → C code silently skipped before this fix.
        _write(
            tmp_path,
            {
                "pyproject.toml": """
                [project]
                name = "linttool"
                version = "0.0.1"

                [build-system]
                build-backend = "setuptools.build_meta"
                """,
                "linttool/__init__.py": "VERSION = '0.0.1'\n",
                "linttool/cli.py": "def main(): pass\n",
                "src/big1.c": "int big1(void) { return 1; }\n",
                "src/big2.c": "int big2(void) { return 2; }\n",
                "include/big.h": "int big1(void); int big2(void);\n",
            },
        )
        g = build(str(tmp_path))
        files = [r["path"] for r in g.cypher("MATCH (f:File) RETURN f.path AS path").to_list()]
        assert any(p.endswith("linttool/cli.py") for p in files), files
        assert any(p.endswith("src/big1.c") for p in files), (
            f"expected src/big1.c in graph (mixed-language safety net), got: {files}"
        )
        assert any(p.endswith("src/big2.c") for p in files), files
        assert any(p.endswith("include/big.h") for p in files), files

    def test_safety_net_does_not_descend_into_venv(self, tmp_path):
        # Regression: 0.8.36 indexed .venv site-packages because the
        # safety net's recursive walk didn't filter ignored subdirs at
        # depth. A single .c file in `subprojects/.venv/site-packages/`
        # caused `subprojects/` to be added as a supplemental root, then
        # parse_directory walked the whole .venv. Reported by an MCP
        # consumer (graph ballooned 7,620 → 70,605 nodes on upgrade).
        # Fix: `walk_filter` skips ignored dir names at every depth.
        _write(
            tmp_path,
            {
                "pyproject.toml": """
                [project]
                name = "main_pkg"
                version = "0.1.0"

                [build-system]
                build-backend = "setuptools.build_meta"
                """,
                "main_pkg/__init__.py": "",
                "main_pkg/core.py": "def f(): pass\n",
                # .venv with C extension source nested inside `subprojects/`
                "subprojects/.venv/site-packages/numpy/_multiarray.c": "int f(){return 0;}\n",
                "subprojects/.venv/site-packages/pkg_a.py": "def a(): pass\n",
                "subprojects/.venv/site-packages/pkg_b.py": "def b(): pass\n",
            },
        )
        g = build(str(tmp_path))
        files = [r["path"] for r in g.cypher("MATCH (f:File) RETURN f.path AS path").to_list()]

        # Only the legitimate package files should be indexed.
        venv_leaks = [p for p in files if ".venv" in p]
        assert venv_leaks == [], f"venv content leaked into graph: {venv_leaks}"
        assert any(p.endswith("main_pkg/core.py") for p in files), files


class TestPathCollision:
    """Regression: every parser computes rel_path against its per-root walk
    directory. Two source roots with same-named files at matching depths
    therefore produce identical f.path keys and dedup drops one. Cargo
    workspaces with shared `src/lib.rs` hit this directly."""

    def test_workspace_crates_with_same_filenames_do_not_collide(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": """
                [workspace]
                members = ["crates/*"]
                """,
                "crates/alpha/Cargo.toml": """
                [package]
                name = "alpha"
                version = "0.1.0"
                edition = "2021"
                """,
                "crates/alpha/src/lib.rs": "pub fn alpha() {}\n",
                "crates/beta/Cargo.toml": """
                [package]
                name = "beta"
                version = "0.1.0"
                edition = "2021"
                """,
                "crates/beta/src/lib.rs": "pub fn beta() {}\n",
            },
        )
        g = build(str(tmp_path))
        files = sorted(r["path"] for r in g.cypher("MATCH (f:File) RETURN f.path AS path").to_list())
        assert any(p.endswith("crates/alpha/src/lib.rs") for p in files), (
            f"expected alpha's lib.rs preserved, got: {files}"
        )
        assert any(p.endswith("crates/beta/src/lib.rs") for p in files), (
            f"expected beta's lib.rs preserved, got: {files}"
        )
