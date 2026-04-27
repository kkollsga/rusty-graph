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


def _cargo_toml() -> str:
    return textwrap.dedent(
        """
        [package]
        name = "fixture"
        version = "0.0.0"
        edition = "2021"
        """
    )


class TestRustClosureWalking:
    """Calls inside closure bodies must be attributed to the enclosing
    function (issue #9 sub-fix 1). Closures are expressions, not items,
    and don't get their own graph node — so `.map(|x| target(x))` is
    semantically a call from the outer function."""

    def test_call_inside_map_closure_is_attributed_to_outer_fn(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub fn target(x: u32) -> u32 { x + 1 }

                pub fn caller() -> Vec<u32> {
                    vec![1, 2, 3].into_iter().map(|x| target(x)).collect()
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        assert any(caller.endswith("::caller") and callee.endswith("::target") for caller, callee in pairs), (
            f"caller -> target via .map(|x| target(x)) not detected: {pairs}"
        )

    def test_call_inside_and_then_closure(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub fn helper() -> Option<u32> { Some(1) }
                pub fn deeper(_n: u32) -> Option<u32> { None }

                pub fn caller() -> Option<u32> {
                    helper().and_then(|n| deeper(n))
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        assert any(caller.endswith("::caller") and callee.endswith("::deeper") for caller, callee in pairs), (
            f"caller -> deeper inside closure not detected: {pairs}"
        )

    def test_nested_function_item_still_isolated(self, tmp_path):
        # Nested fn (not closure) must keep its own scope — calls inside
        # an inner `fn` belong to that inner fn, not the outer.
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub fn target() {}

                pub fn outer() {
                    fn inner() {
                        target();
                    }
                    inner();
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        # Outer should NOT have a direct CALLS edge to target — that
        # edge belongs to inner.
        assert not any(caller.endswith("::outer") and callee.endswith("::target") for caller, callee in pairs), (
            f"outer must not directly call target (that's inner's call): {pairs}"
        )
