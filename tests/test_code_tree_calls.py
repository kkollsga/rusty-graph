"""CALLS edge resolution — driven through the public build() entry.

Previously these tests imported `_build_call_edges` from the Python
builder directly; the Rust port replaces that private helper with an
equivalent in-crate function. Rewrote to drive the public API on small
in-memory fixtures so the tests stay implementation-agnostic.
"""

import textwrap

from kglite.code_tree import build


def _write(tmp_path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))


def _call_pairs(graph) -> set[tuple[str, str]]:
    """Return the set of (caller, callee) qualified_name pairs."""
    rows = graph.cypher(
        "MATCH (a:Function)-[:CALLS]->(b:Function) RETURN a.qualified_name AS caller, b.qualified_name AS callee"
    ).to_list()
    return {(r["caller"], r["callee"]) for r in rows}


class TestTierSameOwner:
    """Tier 1: Same owner — prefer targets sharing the caller's qualified prefix."""

    def test_same_class_preferred(self, tmp_path):
        # ClassA.caller calling run() should resolve to ClassA.run, not ClassB.run.
        _write(
            tmp_path,
            {
                "mod/__init__.py": "",
                "mod/a.py": """
                class ClassA:
                    def caller(self):
                        self.run()
                    def run(self):
                        pass
                """,
                "mod/b.py": """
                class ClassB:
                    def run(self):
                        pass
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        assert any(callee.endswith(".ClassA.run") and caller.endswith(".ClassA.caller") for caller, callee in pairs)
        assert not any(callee.endswith(".ClassB.run") for _, callee in pairs)


class TestTierSameFile:
    """Tier 2: Same file — prefer targets in the same source file."""

    def test_same_file_preferred(self, tmp_path):
        _write(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/same.py": """
                def caller():
                    helper()
                def helper():
                    pass
                """,
                "pkg/other.py": """
                def helper():
                    pass
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        # caller should resolve to same-file helper.
        caller_edges = {callee for caller, callee in pairs if caller.endswith(".caller")}
        assert any(c.endswith("same.helper") for c in caller_edges)
        assert not any(c.endswith("other.helper") for c in caller_edges)


class TestTierGlobal:
    """Tier 4: Global fallback — single unambiguous target wins."""

    def test_unique_name_resolves(self, tmp_path):
        _write(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/a.py": """
                def caller():
                    unique_target()
                """,
                "pkg/b.py": """
                def unique_target():
                    pass
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        caller_edges = {callee for caller, callee in pairs if caller.endswith(".caller")}
        assert any(c.endswith("unique_target") for c in caller_edges)


class TestNoiseFilter:
    """Calls to common stdlib names are skipped."""

    def test_python_builtins_dropped(self, tmp_path):
        _write(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/a.py": """
                def caller():
                    print("hi")
                    return len([1, 2, 3])
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        caller_pairs = {callee for caller, callee in pairs if caller.endswith(".caller")}
        assert not any(p.endswith(".print") or p.endswith(".len") for p in caller_pairs)
