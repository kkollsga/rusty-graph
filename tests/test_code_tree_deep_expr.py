"""Tree-sitter recursive-descent parsers blow their thread stack on
deeply-nested expressions. macOS gives spawned threads ~2 MB of stack by
default, so the rayon worker pool used to SIGBUS the process on inputs
like dotnet/runtime's `JIT/Regression/JitBlue/GitHub_10215.cs` — a
regression test that is literally a chain of thousands of `+` operators.
The parser pool is configured with a 16 MB stack to absorb these; this
test reproduces a similar input and asserts the build does not crash.
"""

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _deep_chain_cs(name: str, depth: int) -> str:
    chain = " + ".join(["b"] * depth)
    return (
        "namespace Stress;\n"
        "using System;\n"
        f"public class {name}\n"
        "{\n"
        "    public static int Compute()\n"
        "    {\n"
        "        int b = 1;\n"
        f"        return {chain};\n"
        "    }\n"
        "}\n"
    )


def test_deeply_nested_csharp_expression_parses_without_sigbus(tmp_path):
    # 6,000 chained `+` operators is enough to overflow a 2 MB worker stack
    # in tree-sitter's recursive-descent grammar; with the 16 MB pool stack
    # it parses in milliseconds.
    src = _deep_chain_cs("Stress", depth=6000)
    # Two files force rayon to dispatch onto a worker thread (single-element
    # par_iter runs on the calling thread, which has a 8 MB stack on macOS
    # and would mask the regression).
    (tmp_path / "a.cs").write_text(src)
    (tmp_path / "b.cs").write_text(src)

    g = build(str(tmp_path))

    rows = g.cypher("MATCH (f:File) RETURN count(f) AS n").to_list()
    assert rows[0]["n"] == 2
