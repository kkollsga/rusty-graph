"""Tests for CALLS edge resolution tiers in code_tree builder."""

import pytest

ts = pytest.importorskip("tree_sitter", reason="requires tree-sitter")

from kglite.code_tree.parsers.models import FunctionInfo  # noqa: E402
from kglite.code_tree.builder import _build_call_edges  # noqa: E402


def _fn(name: str, qualified_name: str, file_path: str = "a.py",
        calls: list[tuple[str, int]] | None = None) -> FunctionInfo:
    """Helper to create a minimal FunctionInfo for testing."""
    return FunctionInfo(
        name=name,
        qualified_name=qualified_name,
        visibility="public",
        is_async=False,
        is_method=False,
        signature=f"def {name}()",
        file_path=file_path,
        line_number=1,
        docstring=None,
        return_type=None,
        calls=calls or [],
    )


class TestTierSameOwner:
    """Tier 1: Same owner — prefer targets sharing the caller's qualified prefix."""

    def test_same_class_preferred(self):
        """ClassA.caller calling run() should resolve to ClassA.run, not ClassB.run."""
        funcs = [
            _fn("caller", "mod.ClassA.caller", "a.py",
                calls=[("run", 10)]),
            _fn("run", "mod.ClassA.run", "a.py"),
            _fn("run", "mod.ClassB.run", "b.py"),
        ]
        df = _build_call_edges(funcs)
        assert len(df) == 1
        assert df.iloc[0]["caller"] == "mod.ClassA.caller"
        assert df.iloc[0]["callee"] == "mod.ClassA.run"

    def test_same_module_preferred(self):
        """Rust: crate::server::handle calling process() prefers crate::server::process."""
        funcs = [
            _fn("handle", "crate::server::handle", "src/server.rs",
                calls=[("process", 5)]),
            _fn("process", "crate::server::process", "src/server.rs"),
            _fn("process", "crate::client::process", "src/client.rs"),
        ]
        df = _build_call_edges(funcs)
        assert len(df) == 1
        assert df.iloc[0]["callee"] == "crate::server::process"


class TestTierSameFile:
    """Tier 2: Same file — prefer targets in the same source file."""

    def test_same_file_preferred(self):
        """When no owner match, prefer target in the same file."""
        funcs = [
            _fn("main", "main", "app.py",
                calls=[("helper", 3)]),
            _fn("helper", "utils.helper", "app.py"),
            _fn("helper", "other.helper", "other.py"),
        ]
        df = _build_call_edges(funcs)
        assert len(df) == 1
        assert df.iloc[0]["callee"] == "utils.helper"

    def test_different_files_no_preference(self):
        """When caller's file doesn't match any target, use all targets."""
        funcs = [
            _fn("main", "main", "main.py",
                calls=[("helper", 3)]),
            _fn("helper", "a.helper", "a.py"),
            _fn("helper", "b.helper", "b.py"),
        ]
        df = _build_call_edges(funcs)
        # Both targets returned since neither is in main.py
        assert len(df) == 2


class TestTierSameLanguage:
    """Tier 3: Same language — prefer targets with matching separator convention."""

    def test_rust_prefers_rust(self):
        """Rust caller should prefer :: targets over . targets."""
        funcs = [
            _fn("handle", "crate::api::handle", "src/api.rs",
                calls=[("connect", 10)]),
            _fn("connect", "crate::db::connect", "src/db.rs"),
            _fn("connect", "db.connect", "db.py"),
        ]
        df = _build_call_edges(funcs)
        assert len(df) == 1
        assert df.iloc[0]["callee"] == "crate::db::connect"

    def test_python_prefers_python(self):
        """Python caller should prefer . targets over :: targets."""
        funcs = [
            _fn("run", "app.run", "app.py",
                calls=[("connect", 5)]),
            _fn("connect", "db.connect", "db.py"),
            _fn("connect", "crate::db::connect", "src/db.rs"),
        ]
        df = _build_call_edges(funcs)
        assert len(df) == 1
        assert df.iloc[0]["callee"] == "db.connect"


class TestReceiverHint:
    """Tier 0: Receiver hint — qualified calls narrow by owner name."""

    def test_receiver_hint_narrows(self):
        """Server.start() should resolve to Server's start method."""
        funcs = [
            _fn("main", "app.main", "main.py",
                calls=[("Server.start", 10)]),
            _fn("start", "app.Server.start", "server.py"),
            _fn("start", "app.Client.start", "client.py"),
        ]
        df = _build_call_edges(funcs)
        assert len(df) == 1
        assert df.iloc[0]["callee"] == "app.Server.start"

    def test_receiver_hint_no_match_falls_through(self):
        """If receiver hint matches nothing, continue to tier resolution."""
        funcs = [
            _fn("main", "app.main", "main.py",
                calls=[("Unknown.start", 10)]),
            _fn("start", "app.Server.start", "main.py"),
            _fn("start", "app.Client.start", "client.py"),
        ]
        df = _build_call_edges(funcs)
        # Falls through to same-file tier: prefers main.py target
        assert len(df) == 1
        assert df.iloc[0]["callee"] == "app.Server.start"


class TestGlobalFallback:
    """Tier 4: Global fallback — when no scope narrows, all targets used."""

    def test_fallback_to_all(self):
        """No owner/file/language match → all targets."""
        funcs = [
            _fn("entry", "entry", "entry.c",
                calls=[("init", 1)]),
            # Two targets, both in different files, no separators for owner
            _fn("init", "init_a", "a.c"),
            _fn("init", "init_b", "b.c"),
        ]
        df = _build_call_edges(funcs)
        assert len(df) == 2

    def test_single_target_no_ambiguity(self):
        """Single target — no disambiguation needed."""
        funcs = [
            _fn("main", "mod.main", "main.py",
                calls=[("unique_fn", 1)]),
            _fn("unique_fn", "lib.unique_fn", "lib.py"),
        ]
        df = _build_call_edges(funcs)
        assert len(df) == 1
        assert df.iloc[0]["callee"] == "lib.unique_fn"


class TestMaxTargets:
    """max_targets limit still applies after tier resolution."""

    def test_max_targets_skips(self):
        """When remaining targets exceed max_targets, skip the call."""
        funcs = [
            _fn("caller", "caller", "main.py",
                calls=[("common", 1)]),
        ]
        # Add 6 targets (default max_targets=5)
        for i in range(6):
            funcs.append(_fn("common", f"pkg{i}.common", f"pkg{i}.py"))
        df = _build_call_edges(funcs)
        assert len(df) == 0

    def test_tier_reduces_below_max(self):
        """Tier resolution can reduce targets below max_targets threshold."""
        funcs = [
            _fn("caller", "pkg.caller", "pkg/main.py",
                calls=[("common", 1)]),
        ]
        # 6 total targets, but only 2 in same file
        for i in range(4):
            funcs.append(_fn("common", f"other{i}.common", f"other{i}.py"))
        funcs.append(_fn("common", "pkg.common_a", "pkg/main.py"))
        funcs.append(_fn("common", "pkg.common_b", "pkg/main.py"))
        df = _build_call_edges(funcs)
        # Same-file tier narrows to 2 targets (below max_targets=5)
        assert len(df) == 2
        callees = set(df["callee"])
        assert callees == {"pkg.common_a", "pkg.common_b"}


class TestExcludedNames:
    """excluded_names still filters."""

    def test_excluded_skipped(self):
        funcs = [
            _fn("main", "app.main", "main.py",
                calls=[("print", 1), ("helper", 2)]),
            _fn("print", "builtins.print", "builtins.py"),
            _fn("helper", "app.helper", "main.py"),
        ]
        df = _build_call_edges(funcs, excluded_names=frozenset(["print"]))
        assert len(df) == 1
        assert df.iloc[0]["callee"] == "app.helper"


class TestSelfCallExcluded:
    """Self-calls (recursion markers) are excluded."""

    def test_no_self_edge(self):
        funcs = [
            _fn("recurse", "mod.recurse", "mod.py",
                calls=[("recurse", 5)]),
        ]
        df = _build_call_edges(funcs)
        assert len(df) == 0
