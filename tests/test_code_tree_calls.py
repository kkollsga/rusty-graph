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

    def test_self_method_disambiguates_by_caller_struct(self, tmp_path):
        """When the same method name exists on multiple structs, a bare
        `self.method()` call inside a method of `Foo` must resolve to
        `Foo::method` — not `Bar::method` (issue #9 sub-fix 2)."""
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub struct Foo;
                pub struct Bar;

                impl Foo {
                    pub fn caller(&self) {
                        self.run();
                    }
                    pub fn run(&self) {}
                }

                impl Bar {
                    pub fn run(&self) {}
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        # Foo::caller should resolve self.run() to Foo::run, not Bar::run.
        caller_edges = {callee for caller, callee in pairs if caller.endswith("::Foo::caller")}
        assert any(c.endswith("::Foo::run") for c in caller_edges), f"Foo::caller -> Foo::run should resolve: {pairs}"
        assert not any(c.endswith("::Bar::run") for c in caller_edges), (
            f"Foo::caller must NOT resolve to Bar::run: {pairs}"
        )

    def test_self_method_disambiguates_across_files(self, tmp_path):
        """Both impls live in different files but on the same struct
        (`impl Foo` in lib.rs and `impl Foo` in extras.rs). A self.method
        call must still bind correctly via the implicit receiver hint."""
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub mod extras;

                pub struct Foo;

                impl Foo {
                    pub fn run(&self) {}
                }
                """,
                "src/extras.rs": """
                use crate::Foo;

                impl Foo {
                    pub fn caller(&self) {
                        self.run();
                    }
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        assert any(
            caller.endswith("::extras::Foo::caller") and callee.endswith("::Foo::run") for caller, callee in pairs
        ), f"caller in extras.rs should resolve self.run() to Foo::run: {pairs}"

    def test_function_pointer_argument_emits_references_fn(self, tmp_path):
        """`iter.and_then(some_fn)` — `some_fn` is a function-value
        argument, not a call site. We emit a separate REFERENCES_FN
        edge so the function isn't reported as dead by inbound-CALLS
        analysis (issue #9 sub-fix 3)."""
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub fn helper(n: u32) -> Option<u32> { Some(n + 1) }

                pub fn caller() -> Option<u32> {
                    Some(0u32).and_then(helper)
                }
                """,
            },
        )
        g = build(str(tmp_path))
        rows = g.cypher(
            "MATCH (a:Function)-[:REFERENCES_FN]->(b:Function) "
            "WHERE a.name = 'caller' AND b.name = 'helper' "
            "RETURN count(*) AS c"
        ).to_list()
        assert rows[0]["c"] == 1, "REFERENCES_FN edge for fn-as-value argument missing"

        # Also verify NO direct CALLS edge — caller doesn't actually
        # invoke helper at the reference site.
        rows = g.cypher(
            "MATCH (a:Function)-[:CALLS]->(b:Function) "
            "WHERE a.name = 'caller' AND b.name = 'helper' "
            "RETURN count(*) AS c"
        ).to_list()
        assert rows[0]["c"] == 0, "CALLS edge must NOT fire for fn-as-value argument"

    def test_scoped_function_pointer_resolves(self, tmp_path):
        """Scoped paths like `module::helper` resolve via the terminal
        segment, same as bare identifiers."""
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub mod helpers {
                    pub fn deep(_n: u32) -> Option<u32> { None }
                }

                pub fn caller() -> Option<u32> {
                    Some(0u32).and_then(crate::helpers::deep)
                }
                """,
            },
        )
        g = build(str(tmp_path))
        rows = g.cypher(
            "MATCH (a:Function)-[:REFERENCES_FN]->(b:Function) "
            "WHERE a.name = 'caller' AND b.name = 'deep' "
            "RETURN count(*) AS c"
        ).to_list()
        assert rows[0]["c"] == 1

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


class TestRustMacroCallExtraction:
    """Calls inside macro invocations (`format!`, `vec!`, `json!`, etc.)
    are not parsed as `call_expression` nodes — tree-sitter-rust represents
    them as `identifier` + `token_tree` siblings inside the macro's body.
    The parser walks `macro_invocation` token-trees explicitly so these
    synthetic call sites still produce CALLS edges."""

    def test_call_inside_format_macro(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub fn helper(x: u32) -> u32 { x + 1 }

                pub fn caller(x: u32) -> String {
                    format!("value = {}", helper(x))
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        assert any(caller.endswith("::caller") and callee.endswith("::helper") for caller, callee in pairs), (
            f"caller -> helper inside format!() not detected: {pairs}"
        )

    def test_call_inside_vec_macro(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub fn helper(x: u32) -> u32 { x }

                pub fn caller() -> Vec<u32> {
                    vec![helper(1), helper(2), helper(3)]
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        assert any(caller.endswith("::caller") and callee.endswith("::helper") for caller, callee in pairs), (
            f"caller -> helper inside vec![] not detected: {pairs}"
        )

    def test_nested_call_inside_outer_call_and_macro(self, tmp_path):
        # `Err(format!("{}", helper(x)))` — the outer Err(...) is a real
        # call_expression, format!(...) inside it is a macro_invocation,
        # and helper(x) lives inside the macro's token_tree.
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub fn helper(x: u32) -> u32 { x }

                pub fn caller(x: u32) -> Result<(), String> {
                    Err(format!("oops: {}", helper(x)))
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        assert any(caller.endswith("::caller") and callee.endswith("::helper") for caller, callee in pairs)


class TestRustSelfDispatch:
    """`Self::method(...)` inside an impl block must resolve to the
    enclosing impl's own type. The parser strips the `Self::` prefix so
    the resolver's implicit caller-owner hint kicks in."""

    def test_self_static_dispatch_resolves_to_owner(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub struct Foo;
                pub struct Bar;

                impl Foo {
                    pub fn caller() {
                        Self::method();
                    }
                    pub fn method() {}
                }

                impl Bar {
                    pub fn method() {}
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        caller_edges = {callee for caller, callee in pairs if caller.endswith("::Foo::caller")}
        assert any(c.endswith("::Foo::method") for c in caller_edges), (
            f"Self::method() must resolve to Foo::method: {pairs}"
        )
        assert not any(c.endswith("::Bar::method") for c in caller_edges), (
            f"Self::method() must NOT resolve to Bar::method: {pairs}"
        )

    def test_self_dispatch_inside_macro(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub struct Parser;

                impl Parser {
                    pub fn token_to_display(t: u32) -> String { format!("{}", t) }

                    pub fn caller(t: u32) -> Result<(), String> {
                        Err(format!("got {}", Self::token_to_display(t)))
                    }
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        assert any(
            caller.endswith("::Parser::caller") and callee.endswith("::Parser::token_to_display")
            for caller, callee in pairs
        )


class TestRustTurbofishCalls:
    """Turbofish / `generic_function` call expressions
    (`path::with::<T>(...)`) get a `generic_function` node wrapping the
    inner identifier or scoped_identifier — not bare `identifier` /
    `scoped_identifier`. The parser strips the type-arguments and recurses
    so these calls still produce CALLS edges."""

    def test_bare_turbofish_call(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub fn helper<T>(_x: T) -> u32 { 0 }

                pub fn caller() -> u32 {
                    helper::<u64>(5u64)
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        assert any(caller.endswith("::caller") and callee.endswith("::helper") for caller, callee in pairs), (
            f"helper::<u64>() turbofish not detected: {pairs}"
        )

    def test_self_turbofish_resolves(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub struct Store;

                impl Store {
                    pub fn load_typed_vec<T>(&self) -> Vec<T> { Vec::new() }

                    pub fn caller(&self) -> Vec<u64> {
                        Self::load_typed_vec::<u64>(self)
                    }
                }
                """,
            },
        )
        g = build(str(tmp_path))
        pairs = _call_pairs(g)
        assert any(
            caller.endswith("::Store::caller") and callee.endswith("::Store::load_typed_vec")
            for caller, callee in pairs
        )


class TestRustEnumImplements:
    """Rust enums commonly implement traits — the IMPLEMENTS edge must
    fire from `Enum -> Trait`, not just `Struct/Class -> Trait`."""

    def test_enum_implements_external_trait(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub trait Greeter {
                    fn greet(&self) -> &str;
                }

                pub enum Mood {
                    Happy,
                    Sad,
                }

                impl Greeter for Mood {
                    fn greet(&self) -> &str {
                        match self {
                            Mood::Happy => "hi",
                            Mood::Sad => "oh",
                        }
                    }
                }
                """,
            },
        )
        g = build(str(tmp_path))
        rows = g.cypher("MATCH (e:Enum {title: 'Mood'})-[:IMPLEMENTS]->(t:Trait) RETURN t.title AS trait").to_list()
        traits = {r["trait"] for r in rows}
        assert "Greeter" in traits, f"Mood enum should IMPLEMENTS Greeter: {traits}"
