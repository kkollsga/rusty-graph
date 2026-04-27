"""is_test detection on Rust source — inline `#[cfg(test)] mod tests` blocks
and the `tests.rs` filename convention.

The core gap motivating these tests: functions inside an inline
`#[cfg(test)] mod tests { ... }` block were not flagged `is_test=true`,
which inflated every dead-code / orphan query against a Rust codebase.
"""

from __future__ import annotations

import textwrap

from kglite.code_tree import build


def _write(tmp_path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))


def _is_test_map(graph) -> dict[str, bool]:
    """Return {qualified_name: is_test} for every Function in the graph."""
    rows = graph.cypher("MATCH (f:Function) RETURN f.qualified_name AS qname, f.is_test AS is_test").to_list()
    return {r["qname"]: bool(r["is_test"]) for r in rows}


def _cargo_toml() -> str:
    # Minimal Cargo.toml so the manifest reader recognises the project as
    # a Rust crate and walks src/ as a source root.
    return textwrap.dedent(
        """
        [package]
        name = "fixture"
        version = "0.0.0"
        edition = "2021"
        """
    )


class TestInlineCfgTestModule:
    def test_function_in_cfg_test_mod_is_marked_test(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub fn production_fn() {}

                #[cfg(test)]
                mod tests {
                    fn helper_in_test_mod() {}

                    #[test]
                    fn explicit_test() {}
                }
                """,
            },
        )
        g = build(str(tmp_path))
        flags = _is_test_map(g)

        prod = next(q for q in flags if q.endswith("::production_fn"))
        helper = next(q for q in flags if q.endswith("::helper_in_test_mod"))
        explicit = next(q for q in flags if q.endswith("::explicit_test"))

        assert flags[prod] is False, "production fn must not be flagged is_test"
        assert flags[helper] is True, (
            "helper inside #[cfg(test)] mod tests must inherit is_test even without #[test] attribute"
        )
        assert flags[explicit] is True, "explicit #[test] fn must remain is_test"

    def test_nested_cfg_test_mod_propagates(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                #[cfg(test)]
                mod tests {
                    mod inner {
                        fn deep_helper() {}
                    }
                }
                """,
            },
        )
        g = build(str(tmp_path))
        flags = _is_test_map(g)
        deep = next(q for q in flags if q.endswith("::deep_helper"))
        assert flags[deep] is True, "is_test must propagate into mods nested under #[cfg(test)]"


class TestTestsRsFilename:
    def test_tests_rs_filename_marks_functions_is_test(self, tmp_path):
        # `tests.rs` as a sibling source file (not the dir convention) is
        # overwhelmingly an inline-tests module re-exposed via #[path] or
        # similar. Treat it as test code by filename alone, no #[cfg(test)]
        # required.
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub mod foo;
                """,
                "src/foo.rs": """
                pub fn production_fn() {}
                """,
                "src/tests.rs": """
                fn fixture_helper() {}
                """,
            },
        )
        g = build(str(tmp_path))
        flags = _is_test_map(g)
        prod = next(q for q in flags if q.endswith("::production_fn"))
        fixture = next(q for q in flags if q.endswith("::fixture_helper"))
        assert flags[prod] is False
        assert flags[fixture] is True
