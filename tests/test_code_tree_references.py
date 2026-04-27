"""Function -[REFERENCES]-> Constant edge resolution.

Issue #14: constants are one of the cleanest forms of dead code (no
dispatch ambiguity, no trait dynamics). The `Constant` node type
previously had only an inbound `DEFINES` edge from File — no usage
edges — so unreferenced constants could only be detected via ripgrep.

The Rust parser now collects identifiers in function bodies that match
`SCREAMING_SNAKE_CASE` and the builder resolves them against the
project's constant set.
"""

from __future__ import annotations

import textwrap

from kglite.code_tree import build


def _write(tmp_path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))


def _cargo_toml() -> str:
    return textwrap.dedent(
        """
        [package]
        name = "fixture"
        version = "0.0.0"
        edition = "2021"
        """
    )


def _references(graph) -> set[tuple[str, str]]:
    """Return {(function_qname, constant_qname)} for every REFERENCES edge."""
    rows = graph.cypher(
        "MATCH (f:Function)-[:REFERENCES]->(c:Constant) RETURN f.qualified_name AS fn, c.qualified_name AS cnst"
    ).to_list()
    return {(r["fn"], r["cnst"]) for r in rows}


class TestRustConstantReferences:
    def test_bare_identifier_resolves_to_const(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub const MAX_RETRIES: u32 = 5;

                pub fn caller() {
                    let limit = MAX_RETRIES;
                    let _ = limit;
                }
                """,
            },
        )
        g = build(str(tmp_path))
        edges = _references(g)
        assert any(fn.endswith("::caller") and cnst.endswith("::MAX_RETRIES") for fn, cnst in edges), (
            f"expected caller -> MAX_RETRIES, got {edges}"
        )

    def test_static_item_also_resolves(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub static GREETING: &str = "hi";

                pub fn say() {
                    println!("{}", GREETING);
                }
                """,
            },
        )
        g = build(str(tmp_path))
        edges = _references(g)
        assert any(fn.endswith("::say") and cnst.endswith("::GREETING") for fn, cnst in edges), (
            f"expected say -> GREETING, got {edges}"
        )

    def test_scoped_identifier_resolves_via_terminal_segment(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub mod cfg {
                    pub const PORT: u32 = 8080;
                }

                pub fn server() {
                    let p = crate::cfg::PORT;
                    let _ = p;
                }
                """,
            },
        )
        g = build(str(tmp_path))
        edges = _references(g)
        assert any(fn.endswith("::server") and cnst.endswith("::PORT") for fn, cnst in edges), (
            f"expected server -> PORT (via crate::cfg::PORT), got {edges}"
        )

    def test_constant_referenced_multiple_times_emits_one_edge(self, tmp_path):
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub const LIMIT: u32 = 3;

                pub fn caller() {
                    let a = LIMIT;
                    let b = LIMIT + LIMIT;
                    let _ = (a, b);
                }
                """,
            },
        )
        g = build(str(tmp_path))
        rows = g.cypher(
            "MATCH (f:Function)-[:REFERENCES]->(c:Constant) "
            "WHERE f.name = 'caller' AND c.name = 'LIMIT' "
            "RETURN count(*) AS c"
        ).to_list()
        assert rows[0]["c"] == 1

    def test_unreferenced_constant_is_detectable_via_query(self, tmp_path):
        """End-to-end: this is the dead-constant query that was previously
        impossible without ripgrep. Now expressible directly."""
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub const USED: u32 = 1;
                pub const UNUSED: u32 = 99;

                pub fn user() {
                    let x = USED;
                    let _ = x;
                }
                """,
            },
        )
        g = build(str(tmp_path))
        rows = g.cypher(
            "MATCH (c:Constant) WHERE NOT EXISTS { (:Function)-[:REFERENCES]->(c) } RETURN c.name AS name"
        ).to_list()
        names = sorted(r["name"] for r in rows)
        assert names == ["UNUSED"], f"only UNUSED should be unreferenced, got {names}"

    def test_local_lowercase_identifier_does_not_emit_edge(self, tmp_path):
        """Parse-time filter: local variables (lowercase) must not be
        promoted to REFERENCES edges even if a constant of the same name
        existed somewhere — and they don't here, so we just verify zero
        REFERENCES edges land for a function with no constants in scope.
        """
        _write(
            tmp_path,
            {
                "Cargo.toml": _cargo_toml(),
                "src/lib.rs": """
                pub fn no_constants_here() {
                    let local_var = 42;
                    let another = local_var + 1;
                    let _ = another;
                }
                """,
            },
        )
        g = build(str(tmp_path))
        rows = g.cypher(
            "MATCH (f:Function {name: 'no_constants_here'})-[:REFERENCES]->(c:Constant) RETURN count(*) AS c"
        ).to_list()
        assert rows[0]["c"] == 0
