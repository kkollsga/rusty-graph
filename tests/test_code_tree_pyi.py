"""Tests for .pyi (type stub) file parsing in code_tree."""

import textwrap
from pathlib import Path

import pytest

ts = pytest.importorskip("tree_sitter", reason="requires tree-sitter")

from kglite.code_tree.parsers.python import PythonParser  # noqa: E402


@pytest.fixture
def parser():
    return PythonParser()


@pytest.fixture
def stub_project(tmp_path):
    """Create a minimal project with .pyi stub files."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()

    # __init__.pyi — package-level stub
    (pkg / "__init__.pyi").write_text(
        textwrap.dedent("""\
        from .core import Widget

        class Config:
            debug: bool
            def load(self, path: str) -> None: ...
        """)
    )

    # core.pyi — module-level stub
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

    # A regular .py file alongside stubs
    (pkg / "utils.py").write_text(
        textwrap.dedent("""\
        def helper() -> str:
            return "ok"
        """)
    )

    return tmp_path


class TestPyiModulePath:
    """Verify _file_to_module_path handles .pyi correctly."""

    def test_regular_pyi_module_path(self, parser, stub_project):
        pkg = stub_project / "mypkg"
        path = parser._file_to_module_path(pkg / "core.pyi", pkg)
        assert path == "mypkg.core"

    def test_init_pyi_module_path(self, parser, stub_project):
        pkg = stub_project / "mypkg"
        path = parser._file_to_module_path(pkg / "__init__.pyi", pkg)
        assert path == "mypkg"

    def test_regular_py_still_works(self, parser, stub_project):
        pkg = stub_project / "mypkg"
        path = parser._file_to_module_path(pkg / "utils.py", pkg)
        assert path == "mypkg.utils"


class TestPyiParsing:
    """Verify .pyi files are parsed and entities extracted correctly."""

    def test_parse_pyi_file(self, parser, stub_project):
        pkg = stub_project / "mypkg"
        result = parser.parse_file(pkg / "core.pyi", pkg)

        # Should find the file
        assert len(result.files) == 1
        assert result.files[0].module_path == "mypkg.core"

        # Should find Widget class and Drawable protocol
        class_names = {c.name for c in result.classes}
        assert "Widget" in class_names

        protocol_names = {i.name for i in result.interfaces}
        assert "Drawable" in protocol_names

        # Should find the free function
        fn_names = {f.name for f in result.functions}
        assert "create_widget" in fn_names

        # Should find the constant
        const_names = {c.name for c in result.constants}
        assert "MAX_WIDGETS" in const_names

    def test_parse_init_pyi(self, parser, stub_project):
        pkg = stub_project / "mypkg"
        result = parser.parse_file(pkg / "__init__.pyi", pkg)

        assert len(result.files) == 1
        assert result.files[0].module_path == "mypkg"

        class_names = {c.name for c in result.classes}
        assert "Config" in class_names

    def test_pyi_methods_extracted(self, parser, stub_project):
        pkg = stub_project / "mypkg"
        result = parser.parse_file(pkg / "core.pyi", pkg)

        method_names = {f.name for f in result.functions if f.is_method}
        assert "render" in method_names
        assert "draw" in method_names

    def test_pyi_in_directory_scan(self, parser, stub_project):
        """parse_directory should pick up both .py and .pyi files."""
        pkg = stub_project / "mypkg"
        result = parser.parse_directory(pkg)

        file_names = {f.filename for f in result.files}
        assert "__init__.pyi" in file_names
        assert "core.pyi" in file_names
        assert "utils.py" in file_names


class TestPyiTestDetection:
    """Verify .pyi files are marked as test files correctly."""

    def test_test_prefix_pyi(self, parser, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "test_api.pyi").write_text("def test_foo() -> None: ...\n")
        result = parser.parse_file(pkg / "test_api.pyi", pkg)
        assert result.files[0].is_test is True

    def test_test_suffix_pyi(self, parser, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "api_test.pyi").write_text("def test_bar() -> None: ...\n")
        result = parser.parse_file(pkg / "api_test.pyi", pkg)
        assert result.files[0].is_test is True
