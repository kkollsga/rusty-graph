"""Tests for the kglite.rules sub-namespace."""

import pandas as pd
import pytest

import kglite
from kglite.rules.pack import RulePackLoadError, load_bundled, loads_pack
from kglite.rules.runner import RuleParameterError, compile_rule

# ---------------------------------------------------------------------
# Pack loader


def test_load_bundled_structural_integrity():
    pack = load_bundled("structural_integrity")
    assert pack.name == "structural_integrity"
    # Version is bumped as the bundled pack evolves; just assert it parses.
    assert pack.version
    assert len(pack.rules) >= 5
    assert "orphan_node" in pack.rule_names


def test_loader_rejects_unanchored_match():
    with pytest.raises(RulePackLoadError, match="unanchored"):
        loads_pack(
            """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    match: 'MATCH (n)-[:E]->(n)'
"""
        )


def test_loader_accepts_unsafe_unanchored_override():
    pack = loads_pack(
        """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    unsafe_unanchored: true
    match: 'MATCH (n)-[:E]->(n)'
"""
    )
    assert pack.rules[0].unsafe_unanchored is True


def test_loader_accepts_template_anchored_match():
    pack = loads_pack(
        """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    parameters:
      type: string
    match: 'MATCH (n:{type}) WHERE 1=1'
"""
    )
    assert pack.rules[0].name == "r"


def test_loader_accepts_call_only_rule():
    pack = loads_pack(
        """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    match: 'CALL connected_components() YIELD node, component'
"""
    )
    assert pack.rules[0].name == "r"


def test_loader_rejects_missing_required_keys():
    with pytest.raises(RulePackLoadError, match="version"):
        loads_pack("name: t\nrules: [{name: r, severity: medium, match: 'MATCH (n:T)'}]")


def test_loader_rejects_invalid_severity():
    with pytest.raises(RulePackLoadError, match="severity"):
        loads_pack(
            """
name: t
version: '1.0'
rules:
  - name: r
    severity: critical
    match: 'MATCH (n:T) WHERE 1=1'
"""
        )


# ---------------------------------------------------------------------
# Compiler


def test_compile_substitutes_placeholders():
    pack = loads_pack(
        """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    parameters: {type: string}
    match: 'MATCH (n:{type})'
    columns: [n.id]
"""
    )
    cypher, used = compile_rule(pack.rules[0], {"type": "LawSection"})
    assert "MATCH (n:LawSection)" in cypher
    assert "LIMIT 1000" in cypher
    assert used == {"type": "LawSection"}


def test_compile_caller_limit_overrides_default():
    pack = loads_pack(
        """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    match: 'MATCH (n:T) WHERE 1=1'
"""
    )
    cypher, _ = compile_rule(pack.rules[0], limit=42)
    assert "LIMIT 42" in cypher


def test_compile_missing_param_raises():
    pack = loads_pack(
        """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    parameters: {type: string}
    match: 'MATCH (n:{type}) WHERE 1=1'
"""
    )
    with pytest.raises(RuleParameterError, match="type"):
        compile_rule(pack.rules[0], {})


def test_compile_zero_limit_rejected():
    pack = loads_pack(
        """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    match: 'MATCH (n:T) WHERE 1=1'
"""
    )
    with pytest.raises(RuleParameterError, match=">= 1"):
        compile_rule(pack.rules[0], limit=0)


def test_compile_no_columns_returns_star():
    pack = loads_pack(
        """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    match: 'MATCH (n:T) WHERE 1=1'
"""
    )
    cypher, _ = compile_rule(pack.rules[0])
    assert "RETURN *" in cypher


def test_compile_preserves_cypher_map_syntax():
    pack = loads_pack(
        """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    match: "MATCH (n:T {created: 'today'}) WHERE 1=1"
"""
    )
    cypher, _ = compile_rule(pack.rules[0])
    assert "{created: 'today'}" in cypher


# ---------------------------------------------------------------------
# End-to-end through g.rules


@pytest.fixture
def integrity_graph():
    """Tiny graph with known integrity issues."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "sec1", "title": "§1"},
                {"id": "sec2", "title": "§2"},
                {"id": "sec3", "title": "§3-orphan"},
                {"id": "sec4", "title": "§1"},  # duplicate title with sec1
            ]
        ),
        "LawSection",
        "id",
        "title",
    )
    g.add_nodes(
        pd.DataFrame([{"id": "lov1", "title": "Test Law"}]),
        "Law",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame(
            [
                {"s": "sec1", "t": "lov1"},
                {"s": "sec2", "t": "lov1"},
                {"s": "sec4", "t": "lov1"},
            ]
        ),
        "SECTION_OF",
        "LawSection",
        "s",
        "Law",
        "t",
    )
    return g


def test_rules_accessor_listing_includes_bundled(integrity_graph):
    packs = integrity_graph.rules.list()
    names = [p["name"] for p in packs]
    assert "structural_integrity" in names


def test_rules_accessor_describe(integrity_graph):
    info = integrity_graph.rules.describe("structural_integrity")
    assert info["name"] == "structural_integrity"
    rule_names = {r["name"] for r in info["rules"]}
    assert "orphan_node" in rule_names
    for rule in info["rules"]:
        assert rule["description_for_agent"]  # non-empty


def test_orphan_node_finds_disconnected(integrity_graph):
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["orphan_node"],
    )
    df = report.violations_for("orphan_node")
    assert len(df) == 1
    assert df.iloc[0]["node_id"] == "sec3"


def test_missing_required_edge_finds_no_section_of(integrity_graph):
    # Add a LawSection that has no SECTION_OF edge
    integrity_graph.add_nodes(
        pd.DataFrame([{"id": "lonely", "title": "lonely §"}]),
        "LawSection",
        "id",
        "title",
    )
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["missing_required_edge"],
    )
    df = report.violations_for("missing_required_edge")
    ids = set(df["node_id"])
    assert "lonely" in ids
    # sec3 also has no SECTION_OF (it's our orphan)
    assert "sec3" in ids


def test_duplicate_title_found(integrity_graph):
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["duplicate_title"],
    )
    df = report.violations_for("duplicate_title")
    assert len(df) == 1
    assert df.iloc[0]["title"] == "§1"
    assert df.iloc[0]["dup_count"] == 2


def test_summary_lazy_does_not_materialize_rows(integrity_graph):
    """Reading .summary alone should not populate the .rows DataFrame cache."""
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["duplicate_title"],
    )
    _ = report.summary
    assert report._rows_cache is None
    _ = report.rows
    assert report._rows_cache is not None


def test_truncated_flag_set_when_limit_reached(integrity_graph):
    # Add many nodes so a low-limit run hits truncation
    integrity_graph.add_nodes(
        pd.DataFrame([{"id": f"n{i}", "title": f"X{i}"} for i in range(50)]),
        "Bulk",
        "id",
        "title",
    )
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="Bulk",
        edge="X",
        only=["orphan_node"],
        limit=5,
    )
    info = report.summary["by_rule"]["orphan_node"]
    assert info["truncated"] is True
    assert info["violations"] == 5


def test_cache_returns_same_report_instance(integrity_graph):
    a = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["orphan_node"],
    )
    b = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["orphan_node"],
    )
    assert a is b


def test_cache_separates_by_params(integrity_graph):
    a = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["orphan_node"],
    )
    b = integrity_graph.rules.run(
        "structural_integrity",
        type="Law",
        edge="SECTION_OF",
        only=["orphan_node"],
    )
    assert a is not b


def test_unknown_pack_name_raises(integrity_graph):
    with pytest.raises(KeyError):
        integrity_graph.rules.run("nonexistent_pack")


def test_describe_xml_silent_until_pack_run(integrity_graph):
    """Rule packs are opt-in: cold describe() must NOT include the block.

    Per-graph advertising activates only after the user actually runs
    a pack on that graph.
    """
    xml = integrity_graph.describe()
    assert "<rule_packs>" not in xml

    integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["duplicate_title"],
    )
    xml_after = integrity_graph.describe()
    assert "<rule_packs>" in xml_after
    assert "last_run_violations=" in xml_after


def test_to_markdown_runs_without_error(integrity_graph):
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
    )
    md = report.to_markdown(sample_rows=2)
    assert "structural_integrity" in md
    assert "## " in md  # headings for each rule


def test_violations_for_unknown_rule_raises(integrity_graph):
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["orphan_node"],
    )
    with pytest.raises(KeyError):
        report.violations_for("nonexistent_rule")


def test_has_blockers_false_for_medium_only_violations(integrity_graph):
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
    )
    # All bundled rules are medium/high severity, no blockers
    assert report.has_blockers is False


# ---------------------------------------------------------------------
# Friction-fix coverage (slice 1.1)


def test_missing_inbound_edge_rule_present():
    pack = kglite.rules.load_bundled("structural_integrity")
    assert "missing_inbound_edge" in pack.rule_names


def test_missing_inbound_edge_fires_on_unreferenced_node(integrity_graph):
    # All Laws here are referenced via SECTION_OF — add an unreferenced one.
    integrity_graph.add_nodes(
        pd.DataFrame([{"id": "lov2", "title": "Unreferenced Law"}]),
        "Law",
        "id",
        "title",
    )
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="Law",
        edge="SECTION_OF",
        only=["missing_inbound_edge"],
    )
    df = report.violations_for("missing_inbound_edge")
    assert "lov2" in set(df["node_id"])


def test_summary_any_truncated_flag(integrity_graph):
    integrity_graph.add_nodes(
        pd.DataFrame([{"id": f"b{i}", "title": f"X{i}"} for i in range(50)]),
        "Bulk",
        "id",
        "title",
    )
    truncated_run = integrity_graph.rules.run(
        "structural_integrity",
        type="Bulk",
        edge="X",
        only=["orphan_node"],
        limit=5,
    )
    assert truncated_run.summary["any_truncated"] is True

    clean_run = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["orphan_node"],
    )
    assert clean_run.summary["any_truncated"] is False


def test_is_suspect_returns_rule_severity_pairs(integrity_graph):
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["orphan_node"],
    )
    assert report.is_suspect("sec3") == [("orphan_node", "medium")]
    assert report.is_suspect("sec1") == []
    assert report.is_suspect("nonexistent") == []


def test_is_suspect_handles_int_and_string_ids(integrity_graph):
    integrity_graph.add_nodes(
        pd.DataFrame([{"id": 99, "title": "int-id node"}]),
        "IntKey",
        "id",
        "title",
    )
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="IntKey",
        edge="X",
        only=["orphan_node"],
    )
    assert report.is_suspect(99) == report.is_suspect("99")
    assert report.is_suspect(99) != []


def test_list_uses_lazy_yaml_peek():
    g = kglite.KnowledgeGraph()
    info = next(p for p in g.rules.list() if p["name"] == "structural_integrity")
    assert info["loaded"] is False
    assert info["version"] != "(not yet loaded)"
    assert info["rule_count"] > 0
    assert info["description"]


def test_pack_usage_hint_field():
    pack = kglite.rules.load_bundled("structural_integrity")
    assert pack.usage_hint
    assert "structural" in pack.usage_hint.lower()


def test_describe_dict_includes_usage_hint(integrity_graph):
    info = integrity_graph.rules.describe("structural_integrity")
    assert "usage_hint" in info and info["usage_hint"]


def test_describe_xml_includes_usage_hint(integrity_graph):
    """After advertising is enabled, the cold describe() block carries usage_hint."""
    kglite.rules.advertise()
    try:
        assert "usage_hint=" in integrity_graph.describe()
    finally:
        kglite.rules._disable_advertising()


def test_describe_xml_marks_truncation(integrity_graph):
    integrity_graph.add_nodes(
        pd.DataFrame([{"id": f"q{i}", "title": f"Y{i}"} for i in range(50)]),
        "Slot",
        "id",
        "title",
    )
    integrity_graph.rules.run(
        "structural_integrity",
        type="Slot",
        edge="X",
        only=["orphan_node"],
        limit=5,
    )
    assert 'last_run_truncated="true"' in integrity_graph.describe()


def test_to_markdown_truncates_long_list_cells(integrity_graph):
    integrity_graph.add_nodes(
        pd.DataFrame([{"id": f"big{i}", "title": "shared-title"} for i in range(6)]),
        "BigDup",
        "id",
        "title",
    )
    report = integrity_graph.rules.run(
        "structural_integrity",
        type="BigDup",
        edge="X",
        only=["duplicate_title"],
    )
    md = report.to_markdown(sample_rows=5)
    # 6 ids → preview shows 3 + " (+3 more)"
    assert "+3 more" in md


# ---------------------------------------------------------------------
# Direction validator (slice 1.1.1)


@pytest.fixture
def directional_graph():
    """Wellbore-IN_LICENCE->Licence (outbound from Wellbore)."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": "w1", "title": "W1"},
                {"id": "w2", "title": "W2"},
                {"id": "w3", "title": "W3-no-licence"},
            ]
        ),
        "Wellbore",
        "id",
        "title",
    )
    g.add_nodes(
        pd.DataFrame([{"id": "l1", "title": "PL001"}]),
        "Licence",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame(
            [
                {"s": "w1", "t": "l1"},
                {"s": "w2", "t": "l1"},
            ]
        ),
        "IN_LICENCE",
        "Wellbore",
        "s",
        "Licence",
        "t",
    )
    return g


def test_outbound_direction_passes_when_correct(directional_graph):
    """Wellbore→Licence is outbound; missing_required_edge should run."""
    report = directional_graph.rules.run(
        "structural_integrity",
        type="Wellbore",
        edge="IN_LICENCE",
        only=["missing_required_edge"],
    )
    info = report.summary["by_rule"]["missing_required_edge"]
    assert info["error"] is None
    assert info["violations"] == 1  # w3 has no IN_LICENCE


def test_inbound_direction_passes_when_correct(directional_graph):
    report = directional_graph.rules.run(
        "structural_integrity",
        type="Licence",
        edge="IN_LICENCE",
        only=["missing_inbound_edge"],
    )
    info = report.summary["by_rule"]["missing_inbound_edge"]
    assert info["error"] is None
    # l1 has incoming IN_LICENCE from w1, w2 → 0 violations


def test_outbound_direction_caught_when_wrong(directional_graph):
    """missing_required_edge on Licence/IN_LICENCE — Licence is the target side."""
    report = directional_graph.rules.run(
        "structural_integrity",
        type="Licence",
        edge="IN_LICENCE",
        only=["missing_required_edge"],
    )
    info = report.summary["by_rule"]["missing_required_edge"]
    assert info["error"] is not None
    assert "DirectionMismatch" in info["error"]
    assert "missing_inbound_edge" in info["error"]  # Suggests the right rule
    assert info["violations"] == 0  # No execution → no rows


def test_inbound_direction_caught_when_wrong(directional_graph):
    """missing_inbound_edge on Wellbore/IN_LICENCE — Wellbore is the source side."""
    report = directional_graph.rules.run(
        "structural_integrity",
        type="Wellbore",
        edge="IN_LICENCE",
        only=["missing_inbound_edge"],
    )
    info = report.summary["by_rule"]["missing_inbound_edge"]
    assert info["error"] is not None
    assert "DirectionMismatch" in info["error"]
    assert "missing_required_edge" in info["error"]


def test_unknown_edge_skips_validation(directional_graph):
    """Edges not present in the graph at all should pass validation
    (graph might be empty or building)."""
    report = directional_graph.rules.run(
        "structural_integrity",
        type="Wellbore",
        edge="NOT_A_REAL_EDGE",
        only=["missing_required_edge"],
    )
    info = report.summary["by_rule"]["missing_required_edge"]
    assert info["error"] is None  # Validation skipped
    # The rule itself runs; with no NOT_A_REAL_EDGE edges, all wellbores match


def test_validates_direction_requires_type_and_edge_params():
    """A rule with validates_direction must declare both 'type' and 'edge'."""
    yaml_text = """
name: t
version: '1.0'
rules:
  - name: bad
    severity: medium
    parameters:
      x: string
    validates_direction: outbound
    match: 'MATCH (n:T) WHERE 1=1'
"""
    with pytest.raises(kglite.rules.RulePackLoadError, match="validates_direction"):
        kglite.rules.loads_pack(yaml_text)


def test_describe_is_native_rust_method():
    """describe() must dispatch directly through Rust — no Python wrapper.

    Slice 1.1 used a monkey-patch (replaced KnowledgeGraph.describe with a
    Python wrapper that called _describe_native). Slice 1.1.1 moved the
    rule-pack XML splice into Rust, eliminating the wrapper. This test
    pins the new behavior so the wrapper doesn't sneak back in.
    """
    g = kglite.KnowledgeGraph()
    assert type(g).describe.__qualname__.startswith("KnowledgeGraph.")
    assert not hasattr(type(g), "_describe_native")


def test_describe_cold_silent_by_default():
    """Rule packs are opt-in: a fresh kglite.KnowledgeGraph() and an
    untouched g.rules accessor must NOT surface a <rule_packs> block.

    This is a deliberate change from slice 1.1, where the cold default
    was always advertised. Agents that don't have a rule-pack tool
    available should not see the block at all.
    """
    fresh = kglite.KnowledgeGraph()
    xml = fresh.describe()
    assert "<rule_packs>" not in xml

    # Touching g.rules.list() (lazy peek only) must NOT activate advertising
    _ = fresh.rules.list()
    xml = fresh.describe()
    assert "<rule_packs>" not in xml


def test_advertise_publishes_global_default():
    """kglite.rules.advertise() pushes a module-level default visible to
    every subsequent describe() — including graphs created before the call."""
    g_before = kglite.KnowledgeGraph()
    assert "<rule_packs>" not in g_before.describe()

    kglite.rules.advertise()
    try:
        # Pre-existing graph sees the new default
        assert "<rule_packs>" in g_before.describe()
        # Newly created graph also sees it
        g_after = kglite.KnowledgeGraph()
        assert "<rule_packs>" in g_after.describe()

        # Idempotent
        kglite.rules.advertise()
        assert "<rule_packs>" in g_before.describe()
    finally:
        kglite.rules._disable_advertising()
        # Once disabled, both graphs go silent again
        assert "<rule_packs>" not in g_before.describe()


def test_per_graph_advertising_isolated_from_global():
    """A per-graph push (via .run()) shouldn't leak into other graphs."""
    g1 = kglite.KnowledgeGraph()
    g1.add_nodes(pd.DataFrame([{"id": "a"}]), "T", "id", "id")
    g2 = kglite.KnowledgeGraph()

    g1.rules.run("structural_integrity", type="T", edge="X", only=["orphan_node"])
    assert "<rule_packs>" in g1.describe()
    assert "<rule_packs>" not in g2.describe()


def test_default_timeout_ms_parsed_from_yaml():
    pack = kglite.rules.loads_pack(
        """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    parameters:
      type: string
    match: 'MATCH (n:{type}) WHERE 1=1'
    default_timeout_ms: 5000
"""
    )
    assert pack.rules[0].default_timeout_ms == 5000


def test_default_timeout_ms_defaults_to_none():
    pack = kglite.rules.load_bundled("structural_integrity")
    # Bundled rules don't (yet) declare default_timeout_ms — should be None.
    for rule in pack.rules:
        assert rule.default_timeout_ms is None


def test_default_timeout_ms_invalid_value_rejected():
    yaml_text = """
name: t
version: '1.0'
rules:
  - name: r
    severity: medium
    match: 'MATCH (n:T) WHERE 1=1'
    default_timeout_ms: -5
"""
    with pytest.raises(kglite.rules.RulePackLoadError, match="default_timeout_ms"):
        kglite.rules.loads_pack(yaml_text)


def test_run_timeout_ms_overrides_rule_default(integrity_graph):
    # Caller-supplied timeout_ms should win — verify by checking cache distinguishes
    a = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["orphan_node"],
    )
    b = integrity_graph.rules.run(
        "structural_integrity",
        type="LawSection",
        edge="SECTION_OF",
        only=["orphan_node"],
        timeout_ms=30_000,
    )
    assert a is not b  # different timeout → cache miss


def test_validates_direction_invalid_value_rejected():
    yaml_text = """
name: t
version: '1.0'
rules:
  - name: bad
    severity: medium
    parameters:
      type: string
      edge: string
    validates_direction: sideways
    match: 'MATCH (n:{type}) WHERE 1=1'
"""
    with pytest.raises(kglite.rules.RulePackLoadError, match="validates_direction"):
        kglite.rules.loads_pack(yaml_text)
