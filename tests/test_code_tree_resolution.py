"""Cross-namespace name resolution for IMPLEMENTS/EXTENDS/CALLS edges.

Regression coverage for the dotnet/runtime evaluation pass:
- C# `class Foo : IBaseInterface, ISecond` must capture every base type
  (the previous parser only recorded the first one when the secondary
  base used a node kind not on a hardcoded allow-list).
- Generic args must be stripped from base type names so
  `class Foo : IEnumerable<int>` resolves against `IEnumerable`.
- `using` directives disambiguate same-named symbols: when two
  namespaces define `Assert.True`, the caller's `using Xunit;` pins
  the call to `Xunit.Assert.True`.
- `extends X` where X is actually an Interface must auto-reroute to
  IMPLEMENTS — fixes the C# parser's "first base is always extends"
  assumption when there is no base class.
- File DEFINES Function must include methods, not just top-level
  free functions (C# has no top-level functions; without this every
  C# method node was edge-less from the file side).
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


def test_secondary_base_type_captured(tmp_path):
    _write(
        tmp_path,
        {
            "Lib.cs": """
                namespace Lib;
                public interface IBase {}
                public interface ISecond {}
                public class Foo : IBase, ISecond {}
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        "MATCH (Foo:Class)-[r:IMPLEMENTS]->(b) "
        "WHERE Foo.qualified_name ENDS WITH 'Foo' "
        "RETURN b.name AS name ORDER BY name"
    ).to_list()
    assert [r["name"] for r in rows] == ["IBase", "ISecond"]


def test_generic_base_strips_to_bare_name(tmp_path):
    _write(
        tmp_path,
        {
            "Lib.cs": """
                namespace System.Collections.Generic;
                public interface IEnumerable<T> {}
            """,
            "User.cs": """
                namespace App;
                using System.Collections.Generic;
                public class Foo : IEnumerable<int> {}
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher("MATCH (:Class {name:'Foo'})-[:IMPLEMENTS]->(i:Interface) RETURN i.qualified_name AS qn").to_list()
    assert rows == [{"qn": "System.Collections.Generic.IEnumerable"}]


def test_extends_an_interface_reroutes_to_implements(tmp_path):
    # `class Foo : IBar` (no base class). The C# parser emits the first
    # base as "extends", but IBar is genuinely an interface and the
    # edge must land as IMPLEMENTS.
    _write(
        tmp_path,
        {
            "Lib.cs": """
                namespace Lib;
                public interface IBar {}
                public class Foo : IBar {}
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        "MATCH (Foo:Class {name:'Foo'})-[r]->(b) RETURN type(r) AS t, labels(b) AS bL, b.name AS bn"
    ).to_list()
    assert any(r["t"] == "IMPLEMENTS" and r["bn"] == "IBar" and r["bL"] == ["Interface"] for r in rows), rows
    # And no spurious EXTENDS to an Interface.
    assert not any(r["t"] == "EXTENDS" and r["bL"] == ["Interface"] for r in rows), rows


def test_using_directive_disambiguates_same_named_calls(tmp_path):
    _write(
        tmp_path,
        {
            "Xunit/Assert.cs": """
                namespace Xunit;
                public class Assert
                {
                    public static void True(bool cond) {}
                }
            """,
            "Other/Assert.cs": """
                namespace Other;
                public class Assert
                {
                    public static void True(bool cond) {}
                }
            """,
            "App.cs": """
                namespace App;
                using Xunit;
                public class Tests
                {
                    public void Run()
                    {
                        Assert.True(true);
                    }
                }
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        "MATCH (caller:Function)-[:CALLS]->(callee:Function) "
        "WHERE caller.name = 'Run' "
        "RETURN callee.qualified_name AS qn"
    ).to_list()
    qnames = sorted(r["qn"] for r in rows)
    # The using-tier should pick Xunit.Assert.True over Other.Assert.True.
    # Both candidates remain unresolved without using-aware filtering — the
    # check is that we converged onto the correct one.
    assert "Xunit.Assert.True" in qnames, qnames
    assert "Other.Assert.True" not in qnames, qnames


def test_file_defines_includes_class_methods(tmp_path):
    # C# has no top-level free functions: every Function node lives
    # inside a class. The DEFINES edge from File must include them or
    # downstream queries that join through (:File)-[:DEFINES]->(:Function)
    # see zero results for an entire C# codebase.
    _write(
        tmp_path,
        {
            "Lib.cs": """
                namespace Lib;
                public class Foo
                {
                    public int Add(int a, int b) { return a + b; }
                }
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher("MATCH (f:File)-[:DEFINES]->(fn:Function) WHERE fn.name = 'Add' RETURN f.path AS p").to_list()
    assert len(rows) == 1, rows
    assert rows[0]["p"].endswith("Lib.cs")


def test_function_is_test_inherits_from_file(tmp_path):
    _write(
        tmp_path,
        {
            "tests/SampleTests.cs": """
                namespace Tests;
                public class SampleTests
                {
                    public void TestSomething() {}
                }
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher("MATCH (fn:Function {name:'TestSomething'}) RETURN fn.is_test AS t").to_list()
    assert rows == [{"t": True}]


def test_save_to_preserves_function_properties(tmp_path):
    # Regression: build(save_to=path) used to skip the prepare_save +
    # enable_columnar steps that g.save() does, so all property columns
    # except id/title/type were stripped from the persisted file.
    import kglite

    _write(
        tmp_path,
        {
            "Lib.cs": """
                namespace Lib;
                public class Foo
                {
                    public async Task<int> ComputeAsync(string name) { return 42; }
                }
            """,
        },
    )
    out = tmp_path / "graph.kglite"
    build(str(tmp_path), save_to=str(out))
    g = kglite.load(str(out))
    rows = g.cypher(
        "MATCH (fn:Function {name:'ComputeAsync'}) "
        "RETURN fn.signature AS sig, fn.return_type AS ret, fn.is_async AS asy"
    ).to_list()
    assert rows, "ComputeAsync not found after save/load"
    assert rows[0]["sig"], rows
    assert rows[0]["ret"], rows
    assert rows[0]["asy"] is True, rows
