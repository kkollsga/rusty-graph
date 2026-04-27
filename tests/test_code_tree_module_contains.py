"""Module HAS_FILE File edges — closes the natural top-down walk.

Phase 4.4 added one edge per file pointing back to its leaf module. With this,
"what's in module X" becomes a one-shot Cypher walk instead of a string-prefix
filter. Edge name is HAS_FILE rather than CONTAINS (which is a reserved Cypher
keyword for substring matching).
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


def test_module_has_file_python(tmp_path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/leaf.py": """
            def hello():
                return 1
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        """
        MATCH (m:Module)-[:HAS_FILE]->(f:File)
        WHERE m.qualified_name ENDS WITH ".pkg.sub.leaf"
        RETURN f.path AS p
        """
    ).to_list()
    assert rows, "expected HAS_FILE edge from leaf module"
    assert rows[0]["p"].endswith("pkg/sub/leaf.py")


def test_top_down_walk_reaches_functions(tmp_path):
    """Module → HAS_FILE → File → DEFINES → Function in one Cypher walk."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/sub/__init__.py": "",
            "pkg/sub/a.py": """
            def first():
                return 1
            """,
            "pkg/sub/b.py": """
            def second():
                return 2
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        """
        MATCH (m:Module)
              -[:HAS_SUBMODULE*0..]->(:Module)-[:HAS_FILE]->(f:File)
              -[:DEFINES]->(fn:Function)
        WHERE m.qualified_name ENDS WITH ".pkg.sub"
        RETURN fn.name AS n ORDER BY n
        """
    ).to_list()
    names = [r["n"] for r in rows]
    assert names == ["first", "second"]
