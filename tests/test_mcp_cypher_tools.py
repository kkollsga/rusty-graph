"""Tests for kglite.mcp_server.cypher_tools — manifest cypher tool synthesis."""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("jsonschema")

import kglite  # noqa: E402
from kglite.mcp_server.cypher_tools import (  # noqa: E402
    _build_callable,
    _collect_cypher_params,
    _format_inline,
    _validate_spec,
    register_cypher_tools,
)
from kglite.mcp_server.manifest import CypherTool, ManifestError  # noqa: E402


class _CaptureMcp:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *_args, **_kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    """Tiny graph with three Person nodes for cypher-tool integration tests."""
    g = kglite.KnowledgeGraph()
    import pandas as pd

    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "p1", "name": "Alice", "age": 30},
                {"id": "p2", "name": "Bob", "age": 25},
                {"id": "p3", "name": "Carol", "age": 40},
            ]
        ),
        node_type="Person",
        unique_id_field="id",
        node_title_field="name",
    )
    return g


class TestCollectCypherParams:
    def test_simple(self) -> None:
        assert _collect_cypher_params("MATCH (n {id: $id}) RETURN n") == {"id"}

    def test_multiple(self) -> None:
        assert _collect_cypher_params(
            "MATCH (a {name: $name})-[r]->(b) WHERE r.score > $threshold RETURN b LIMIT $top_k"
        ) == {"name", "threshold", "top_k"}

    def test_none(self) -> None:
        assert _collect_cypher_params("MATCH (n) RETURN n") == set()

    def test_repeated_param_counted_once(self) -> None:
        assert _collect_cypher_params("MATCH (a {x: $v}), (b {y: $v}) RETURN a, b") == {"v"}


class TestValidateSpec:
    def test_no_params_no_refs_ok(self) -> None:
        spec = CypherTool(name="t", cypher="MATCH (n) RETURN n")
        _validate_spec(spec)  # no raise

    def test_declared_match_refs(self) -> None:
        spec = CypherTool(
            name="t",
            cypher="MATCH (n {id: $id}) RETURN n",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
        )
        _validate_spec(spec)

    def test_undeclared_ref_rejected(self) -> None:
        spec = CypherTool(
            name="t",
            cypher="MATCH (n {id: $id, name: $name}) RETURN n",
            parameters={"type": "object", "properties": {"id": {"type": "string"}}},
        )
        with pytest.raises(ManifestError, match=r"\$params \['name'\]"):
            _validate_spec(spec)

    def test_no_schema_with_refs_rejected(self) -> None:
        spec = CypherTool(name="t", cypher="MATCH (n {id: $id}) RETURN n")
        with pytest.raises(ManifestError, match=r"\$params \['id'\]"):
            _validate_spec(spec)

    def test_invalid_jsonschema_rejected(self) -> None:
        spec = CypherTool(
            name="t",
            cypher="MATCH (n) RETURN n",
            parameters={"type": "not-a-real-type"},
        )
        with pytest.raises(ManifestError, match="invalid parameters schema"):
            _validate_spec(spec)


class TestBuildCallable:
    def test_no_params(self, graph: kglite.KnowledgeGraph) -> None:
        spec = CypherTool(name="all_people", cypher="MATCH (p:Person) RETURN p.name AS name")
        fn = _build_callable(graph, spec)
        sig = inspect.signature(fn)
        assert list(sig.parameters) == []
        out = fn()
        assert "Alice" in out
        assert "3 row(s)" in out

    def test_with_required_param(self, graph: kglite.KnowledgeGraph) -> None:
        spec = CypherTool(
            name="person_by_id",
            cypher="MATCH (p:Person {id: $person_id}) RETURN p.name AS name",
            parameters={
                "type": "object",
                "properties": {"person_id": {"type": "string"}},
                "required": ["person_id"],
            },
        )
        fn = _build_callable(graph, spec)
        sig = inspect.signature(fn)
        assert "person_id" in sig.parameters
        assert sig.parameters["person_id"].annotation is str
        assert sig.parameters["person_id"].default is inspect.Parameter.empty
        out = fn(person_id="p1")
        assert "Alice" in out

    def test_optional_param_with_default(self, graph: kglite.KnowledgeGraph) -> None:
        spec = CypherTool(
            name="top_people",
            cypher="MATCH (p:Person) RETURN p.name AS name LIMIT $top_k",
            parameters={
                "type": "object",
                "properties": {"top_k": {"type": "integer", "default": 2}},
            },
        )
        fn = _build_callable(graph, spec)
        sig = inspect.signature(fn)
        assert sig.parameters["top_k"].default == 2
        assert sig.parameters["top_k"].annotation is int
        out = fn(top_k=2)
        assert "2 row(s)" in out
        assert "showing first 15" not in out

    def test_description_used_as_doc(self, graph: kglite.KnowledgeGraph) -> None:
        spec = CypherTool(
            name="t",
            cypher="MATCH (n) RETURN n",
            description="My helpful description",
        )
        fn = _build_callable(graph, spec)
        assert fn.__doc__ == "My helpful description"

    def test_cypher_first_line_as_doc_fallback(self, graph: kglite.KnowledgeGraph) -> None:
        spec = CypherTool(name="t", cypher="\n\nMATCH (n) RETURN n\n")
        fn = _build_callable(graph, spec)
        assert fn.__doc__ == "MATCH (n) RETURN n"

    def test_cypher_error_returned_inline(self, graph: kglite.KnowledgeGraph) -> None:
        spec = CypherTool(name="bad", cypher="THIS IS NOT VALID CYPHER")
        fn = _build_callable(graph, spec)
        out = fn()
        assert "Cypher error" in out


class TestFormatInline:
    def test_empty(self) -> None:
        class _Empty:
            def __len__(self):
                return 0

        assert _format_inline(_Empty()) == "No results."

    def test_truncates_at_15(self) -> None:
        class _Many(list):
            pass

        rows = _Many([{"x": i} for i in range(50)])
        out = _format_inline(rows)
        assert "50 row(s)" in out
        assert "showing first 15" in out
        assert out.count("\n") <= 16  # header + 15 rows

    def test_csv_string_truncated(self) -> None:
        big = "a," * 5000  # 10_000 chars
        out = _format_inline(big)
        assert "truncated" in out
        assert "use cypher_query for FORMAT CSV" in out


class TestRegisterCypherTools:
    def test_validates_before_registering(self, graph: kglite.KnowledgeGraph) -> None:
        bad_spec = CypherTool(
            name="bad",
            cypher="MATCH (n {id: $missing}) RETURN n",
        )
        good_spec = CypherTool(name="good", cypher="MATCH (n) RETURN n")
        mcp = _CaptureMcp()
        with pytest.raises(ManifestError):
            register_cypher_tools(mcp, graph, [good_spec, bad_spec])
        # Validation runs in a separate phase before any tool is registered,
        # so a single bad spec aborts the whole batch.
        assert mcp.tools == {}

    def test_registers_each(self, graph: kglite.KnowledgeGraph) -> None:
        specs = [
            CypherTool(name="all_people", cypher="MATCH (p:Person) RETURN p.name AS n"),
            CypherTool(
                name="person_by_id",
                cypher="MATCH (p:Person {id: $id}) RETURN p.name AS n",
                parameters={
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            ),
        ]
        mcp = _CaptureMcp()
        count = register_cypher_tools(mcp, graph, specs)
        assert count == 2
        assert set(mcp.tools) == {"all_people", "person_by_id"}
        assert "Alice" in mcp.tools["all_people"]()
        assert "Bob" in mcp.tools["person_by_id"](id="p2")
