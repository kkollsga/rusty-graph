"""C++ macro-decorated function parsing.

tree-sitter-cpp can parse macro-decorated definitions like
`SPDLOG_INLINE void foo()`, but without a heuristic to skip the macro tokens,
the parser may capture `SPDLOG_INLINE` as the return type and `unknown` as the
function name. The `looks_like_macro_decorator` helper recognizes all-caps
identifiers (length ≥ 2, optional underscores/digits) and skips them in
`get_return_type`, `get_name`, and parameter extraction.
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


def _function(graph, name_suffix: str) -> dict:
    rows = graph.cypher(
        "MATCH (f:Function) WHERE f.qualified_name ENDS WITH $suf RETURN f.name AS n, f.return_type AS rt LIMIT 1",
        params={"suf": name_suffix},
    ).to_list()
    return rows[0] if rows else {}


def test_macro_decorated_function_extracts_real_name_and_type(tmp_path):
    """SPDLOG_INLINE void foo() — name=foo, return_type=void (not SPDLOG_INLINE)."""
    _write(
        tmp_path,
        {
            "main.cpp": """
            #define SPDLOG_INLINE inline

            SPDLOG_INLINE void log_msg() {}
            """,
        },
    )
    g = build(str(tmp_path))
    fn = _function(g, "::log_msg")
    assert fn.get("n") == "log_msg", fn
    assert fn.get("rt") not in (
        "SPDLOG_INLINE",
        "FMT_API",
        None,
    ), f"return_type should be 'void', got {fn.get('rt')!r}"


def test_multiple_macros_stripped(tmp_path):
    """FMT_API FMT_INLINE int bar() — both decorators ignored."""
    _write(
        tmp_path,
        {
            "main.cpp": """
            #define FMT_API
            #define FMT_INLINE inline

            FMT_API FMT_INLINE int compute_value() { return 42; }
            """,
        },
    )
    g = build(str(tmp_path))
    fn = _function(g, "::compute_value")
    assert fn.get("n") == "compute_value", fn
    rt = fn.get("rt") or ""
    assert "FMT" not in rt, f"return type should not contain a macro name: {rt!r}"


def test_lowercase_function_name_not_treated_as_macro(tmp_path):
    """Don't false-positive on lowercase function names just because they
    contain uppercase letters in CamelCase or snake_case parts."""
    _write(
        tmp_path,
        {
            "main.cpp": """
            int io_count() { return 0; }
            int Reader_init() { return 0; }
            """,
        },
    )
    g = build(str(tmp_path))
    # Both real functions should be captured with their actual names.
    a = _function(g, "::io_count")
    b = _function(g, "::Reader_init")
    assert a.get("n") == "io_count", a
    assert b.get("n") == "Reader_init", b
