"""Tests for kglite.mcp_server.python_tools — trust-gated python: hooks."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from kglite.mcp_server.manifest import ManifestError, PythonTool
from kglite.mcp_server.python_tools import register_python_tools


class _CaptureMcp:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *_args, **_kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def _write_tool_file(tmp_path: Path, body: str, name: str = "tools.py") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


@pytest.fixture
def trust_dir(tmp_path: Path) -> Path:
    """Directory acting as the manifest_dir for path resolution."""
    return tmp_path


class TestTrustGate:
    def test_no_python_tools_no_trust_check(self, trust_dir: Path) -> None:
        mcp = _CaptureMcp()
        count = register_python_tools(
            mcp,
            [],
            manifest_dir=trust_dir,
            trust_flag=False,
            allow_python_tools=False,
        )
        assert count == 0
        assert mcp.tools == {}

    def test_refuses_without_yaml_trust(self, trust_dir: Path) -> None:
        spec = PythonTool(name="t", python="./tools.py", function="t")
        mcp = _CaptureMcp()
        with pytest.raises(ManifestError, match="trust.allow_python_tools"):
            register_python_tools(
                mcp,
                [spec],
                manifest_dir=trust_dir,
                trust_flag=True,
                allow_python_tools=False,
            )
        assert mcp.tools == {}

    def test_refuses_without_cli_flag(self, trust_dir: Path) -> None:
        spec = PythonTool(name="t", python="./tools.py", function="t")
        mcp = _CaptureMcp()
        with pytest.raises(ManifestError, match="--trust-tools"):
            register_python_tools(
                mcp,
                [spec],
                manifest_dir=trust_dir,
                trust_flag=False,
                allow_python_tools=True,
            )
        assert mcp.tools == {}


class TestLoadingErrors:
    def test_file_missing(self, trust_dir: Path) -> None:
        spec = PythonTool(name="t", python="./does_not_exist.py", function="f")
        with pytest.raises(ManifestError, match="does not exist"):
            register_python_tools(
                _CaptureMcp(),
                [spec],
                manifest_dir=trust_dir,
                trust_flag=True,
                allow_python_tools=True,
            )

    def test_function_missing(self, trust_dir: Path) -> None:
        _write_tool_file(trust_dir, "def other(): return 'nope'\n")
        spec = PythonTool(name="t", python="./tools.py", function="missing")
        with pytest.raises(ManifestError, match="not found in"):
            register_python_tools(
                _CaptureMcp(),
                [spec],
                manifest_dir=trust_dir,
                trust_flag=True,
                allow_python_tools=True,
            )

    def test_module_load_error_wrapped(self, trust_dir: Path) -> None:
        _write_tool_file(trust_dir, "raise RuntimeError('boom')\n")
        spec = PythonTool(name="t", python="./tools.py", function="anything")
        with pytest.raises(ManifestError, match="error executing"):
            register_python_tools(
                _CaptureMcp(),
                [spec],
                manifest_dir=trust_dir,
                trust_flag=True,
                allow_python_tools=True,
            )

    def test_non_callable_attribute_rejected(self, trust_dir: Path) -> None:
        _write_tool_file(trust_dir, "value = 42\n")
        spec = PythonTool(name="t", python="./tools.py", function="value")
        with pytest.raises(ManifestError, match="is not callable"):
            register_python_tools(
                _CaptureMcp(),
                [spec],
                manifest_dir=trust_dir,
                trust_flag=True,
                allow_python_tools=True,
            )


class TestSuccessfulRegistration:
    def test_registers_with_user_signature(self, trust_dir: Path) -> None:
        _write_tool_file(
            trust_dir,
            "def session_detail(session_id: str) -> str:\n"
            '    """Return the canonical session record."""\n'
            '    return f"session={session_id}"\n',
        )
        spec = PythonTool(name="session_detail", python="./tools.py", function="session_detail")
        mcp = _CaptureMcp()
        count = register_python_tools(
            mcp,
            [spec],
            manifest_dir=trust_dir,
            trust_flag=True,
            allow_python_tools=True,
        )
        assert count == 1
        assert "session_detail" in mcp.tools
        fn = mcp.tools["session_detail"]
        # User's signature is preserved
        sig = inspect.signature(fn)
        assert "session_id" in sig.parameters
        assert sig.parameters["session_id"].annotation is str
        assert fn(session_id="abc") == "session=abc"

    def test_manifest_name_overrides_function_name(self, trust_dir: Path) -> None:
        _write_tool_file(
            trust_dir,
            "def underlying_impl(x: int) -> int:\n    return x * 2\n",
        )
        spec = PythonTool(name="doubler", python="./tools.py", function="underlying_impl")
        mcp = _CaptureMcp()
        register_python_tools(
            mcp,
            [spec],
            manifest_dir=trust_dir,
            trust_flag=True,
            allow_python_tools=True,
        )
        # Tool registered under the manifest-declared name
        assert "doubler" in mcp.tools
        assert mcp.tools["doubler"](x=5) == 10

    def test_description_overrides_docstring(self, trust_dir: Path) -> None:
        _write_tool_file(
            trust_dir,
            'def f():\n    """Original docstring."""\n    return 1\n',
        )
        spec = PythonTool(
            name="overridden",
            python="./tools.py",
            function="f",
            description="Manifest description.",
        )
        mcp = _CaptureMcp()
        register_python_tools(
            mcp,
            [spec],
            manifest_dir=trust_dir,
            trust_flag=True,
            allow_python_tools=True,
        )
        assert mcp.tools["overridden"].__doc__ == "Manifest description."

    def test_dotdot_path_resolves(self, tmp_path: Path) -> None:
        # Manifest in /tmp/X/graphs/, tools file in /tmp/X/tools/
        manifest_dir = tmp_path / "graphs"
        tools_dir = tmp_path / "tools"
        manifest_dir.mkdir()
        tools_dir.mkdir()
        (tools_dir / "shared.py").write_text("def hi() -> str: return 'hi'\n")
        spec = PythonTool(name="hi", python="../tools/shared.py", function="hi")
        mcp = _CaptureMcp()
        register_python_tools(
            mcp,
            [spec],
            manifest_dir=manifest_dir,
            trust_flag=True,
            allow_python_tools=True,
        )
        assert mcp.tools["hi"]() == "hi"
