"""Per-Function complexity attributes (branch_count / param_count / max_nesting / is_recursive).

Driven through the public `build()` entry; assertions go through Cypher so the
test stays implementation-agnostic — what matters is that the properties
surface on Function nodes with sensible values.
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _write(tmp_path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))


def _function_props(graph, qname_suffix: str) -> dict:
    rows = graph.cypher(
        "MATCH (f:Function) WHERE f.qualified_name ENDS WITH $suf RETURN "
        "f.qualified_name AS qname, f.branch_count AS branches, "
        "f.param_count AS params, f.max_nesting AS nesting, "
        "f.is_recursive AS recursive",
        params={"suf": qname_suffix},
    ).to_list()
    assert rows, f"no Function node ending with {qname_suffix!r}"
    return rows[0]


class TestPython:
    def test_simple_function_zero_complexity(self, tmp_path):
        _write(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/m.py": """
                def trivial(a, b):
                    return a + b
                """,
            },
        )
        g = build(str(tmp_path))
        p = _function_props(g, ".trivial")
        assert p["branches"] == 0
        assert p["params"] == 2
        assert p["nesting"] == 0
        assert p["recursive"] is False

    def test_branchy_function(self, tmp_path):
        _write(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/m.py": """
                def branchy(x):
                    if x > 0:
                        if x > 10:
                            return "big"
                        return "small"
                    elif x < 0:
                        return "neg"
                    for i in range(10):
                        if i == x:
                            return i
                    return None
                """,
            },
        )
        g = build(str(tmp_path))
        p = _function_props(g, ".branchy")
        # 2 ifs + elif + for + nested if = at least 5 branch points
        assert p["branches"] >= 5
        # if -> if = depth 2 minimum
        assert p["nesting"] >= 2
        assert p["recursive"] is False

    def test_recursion_detected(self, tmp_path):
        _write(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/m.py": """
                def fact(n):
                    if n <= 1:
                        return 1
                    return n * fact(n - 1)
                """,
            },
        )
        g = build(str(tmp_path))
        p = _function_props(g, ".fact")
        assert p["recursive"] is True
        assert p["params"] == 1

    def test_self_param_excluded(self, tmp_path):
        _write(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/m.py": """
                class C:
                    def m(self, x, y):
                        return x + y
                """,
            },
        )
        g = build(str(tmp_path))
        p = _function_props(g, ".C.m")
        # self is excluded from the count.
        assert p["params"] == 2


class TestRust:
    def test_branchy_rust_fn(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": """
                [package]
                name = "demo"
                version = "0.1.0"
                """,
                "src/lib.rs": """
                pub fn classify(x: i32) -> &'static str {
                    if x > 0 {
                        if x > 10 { "big" } else { "small" }
                    } else if x < 0 {
                        "neg"
                    } else {
                        "zero"
                    }
                }
                """,
            },
        )
        g = build(str(tmp_path))
        p = _function_props(g, "::classify")
        assert p["branches"] >= 3
        assert p["nesting"] >= 2
        assert p["params"] == 1

    def test_self_excluded_rust(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": """
                [package]
                name = "demo"
                version = "0.1.0"
                """,
                "src/lib.rs": """
                pub struct S;
                impl S {
                    pub fn add(&self, a: u32, b: u32) -> u32 {
                        a + b
                    }
                }
                """,
            },
        )
        g = build(str(tmp_path))
        p = _function_props(g, "::add")
        assert p["params"] == 2
