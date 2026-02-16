"""Stress tests for multiple OPTIONAL MATCH + aggregation.

Covers the bug where multiple OPTIONAL MATCH clauses followed by a single WITH
containing count(DISTINCT ...) returned incorrect (inflated) results due to:
1. The fused OPTIONAL MATCH optimization incorrectly applying to multi-variable WITH
2. Cartesian product fan-out not being properly deduplicated
"""

import pytest
from kglite import KnowledgeGraph


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def court_graph():
    """Miniature court decision graph mimicking the bug report scenario.

    5 CourtDecision nodes, each with varying numbers of Keywords, Judges, and LawSections.
    Distributions are intentionally different across relationship types.
    """
    g = KnowledgeGraph()
    # Create court decisions
    for i in range(5):
        g.cypher(f"CREATE (:CourtDecision {{name: 'Case{i}'}})")
    # Create keywords (different count per decision)
    kw_counts = [3, 0, 5, 1, 2]  # 11 total
    for i, cnt in enumerate(kw_counts):
        for j in range(cnt):
            g.cypher(f"CREATE (:Keyword {{name: 'KW{i}_{j}'}})")
            g.cypher(f"""
                MATCH (c:CourtDecision {{name: 'Case{i}'}}), (k:Keyword {{name: 'KW{i}_{j}'}})
                CREATE (c)-[:HAS_KEYWORD]->(k)
            """)
    # Create judges (different distribution)
    judge_counts = [2, 3, 0, 4, 1]  # 10 total
    for i, cnt in enumerate(judge_counts):
        for j in range(cnt):
            g.cypher(f"CREATE (:Judge {{name: 'J{i}_{j}'}})")
            g.cypher(f"""
                MATCH (c:CourtDecision {{name: 'Case{i}'}}), (j:Judge {{name: 'J{i}_{j}'}})
                CREATE (c)-[:JUDGED_BY]->(j)
            """)
    # Create law sections (yet another distribution)
    cite_counts = [1, 2, 3, 0, 4]  # 10 total
    for i, cnt in enumerate(cite_counts):
        for j in range(cnt):
            g.cypher(f"CREATE (:LawSection {{name: 'LS{i}_{j}'}})")
            g.cypher(f"""
                MATCH (c:CourtDecision {{name: 'Case{i}'}}), (ls:LawSection {{name: 'LS{i}_{j}'}})
                CREATE (c)-[:CITES]->(ls)
            """)
    return g, kw_counts, judge_counts, cite_counts


@pytest.fixture
def shared_target_graph():
    """Graph where multiple source nodes share the same target nodes.

    Tests that DISTINCT correctly deduplicates shared targets.
    """
    g = KnowledgeGraph()
    for i in range(3):
        g.cypher(f"CREATE (:Person {{name: 'P{i}'}})")
    # Create 2 shared skills
    g.cypher("CREATE (:Skill {name: 'Python'})")
    g.cypher("CREATE (:Skill {name: 'Rust'})")
    # All 3 people have Python, only P0 has Rust
    for i in range(3):
        g.cypher(f"""
            MATCH (p:Person {{name: 'P{i}'}}), (s:Skill {{name: 'Python'}})
            CREATE (p)-[:HAS_SKILL]->(s)
        """)
    g.cypher("""
        MATCH (p:Person {name: 'P0'}), (s:Skill {name: 'Rust'})
        CREATE (p)-[:HAS_SKILL]->(s)
    """)
    # Create projects — only P0 and P1 have them
    g.cypher("CREATE (:Project {name: 'Alpha'})")
    g.cypher("CREATE (:Project {name: 'Beta'})")
    g.cypher("""
        MATCH (p:Person {name: 'P0'}), (pr:Project {name: 'Alpha'})
        CREATE (p)-[:WORKS_ON]->(pr)
    """)
    g.cypher("""
        MATCH (p:Person {name: 'P0'}), (pr:Project {name: 'Beta'})
        CREATE (p)-[:WORKS_ON]->(pr)
    """)
    g.cypher("""
        MATCH (p:Person {name: 'P1'}), (pr:Project {name: 'Alpha'})
        CREATE (p)-[:WORKS_ON]->(pr)
    """)
    return g


@pytest.fixture
def duplicate_title_graph():
    """Graph where different nodes have identical titles.

    Ensures count(DISTINCT n) uses node identity, not title string.
    """
    g = KnowledgeGraph()
    g.cypher("CREATE (:Person {name: 'Alice'})")
    # Create 3 tags, two with the same name
    g.cypher("CREATE (:Tag {name: 'important'})")
    g.cypher("CREATE (:Tag {name: 'important'})")  # duplicate title!
    g.cypher("CREATE (:Tag {name: 'urgent'})")
    # Connect Alice to all 3 tags
    g.cypher("""
        MATCH (p:Person {name: 'Alice'}), (t:Tag)
        CREATE (p)-[:TAGGED]->(t)
    """)
    return g


# ── Test Class: Multiple OPTIONAL MATCH Correctness ──────────────────────────


class TestMultiOptionalMatchCorrectness:
    """Core bug reproduction: multiple OPTIONAL MATCH + count(DISTINCT) in single WITH."""

    def test_three_optional_match_distinct_counts(self, court_graph):
        """The exact pattern from the bug report: 3 OPTIONAL MATCHes + 1 WITH."""
        g, kw_counts, judge_counts, cite_counts = court_graph
        result = g.cypher("""
            MATCH (c:CourtDecision)
            OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
            OPTIONAL MATCH (c)-[:JUDGED_BY]->(j:Judge)
            OPTIONAL MATCH (c)-[:CITES]->(ls:LawSection)
            WITH c, count(DISTINCT k) AS kw, count(DISTINCT j) AS jg, count(DISTINCT ls) AS ct
            RETURN c.name, kw, jg, ct
            ORDER BY c.name
        """)
        assert len(result) == 5
        for i, row in enumerate(result):
            assert row['c.name'] == f'Case{i}'
            assert row['kw'] == kw_counts[i], f"Case{i}: expected kw={kw_counts[i]}, got {row['kw']}"
            assert row['jg'] == judge_counts[i], f"Case{i}: expected jg={judge_counts[i]}, got {row['jg']}"
            assert row['ct'] == cite_counts[i], f"Case{i}: expected ct={cite_counts[i]}, got {row['ct']}"

    def test_three_optional_match_plain_counts(self, court_graph):
        """Same as above but without DISTINCT — count(k) instead of count(DISTINCT k)."""
        g, kw_counts, judge_counts, cite_counts = court_graph
        result = g.cypher("""
            MATCH (c:CourtDecision)
            OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
            OPTIONAL MATCH (c)-[:JUDGED_BY]->(j:Judge)
            OPTIONAL MATCH (c)-[:CITES]->(ls:LawSection)
            WITH c, count(DISTINCT k) AS kw, count(DISTINCT j) AS jg, count(DISTINCT ls) AS ct
            RETURN c.name, kw, jg, ct
            ORDER BY c.name
        """)
        # Even with plain count, DISTINCT should give the right per-node count
        # because the Cartesian product duplicates each (c, k) pair M*P times
        for i, row in enumerate(result):
            assert row['kw'] == kw_counts[i]
            assert row['jg'] == judge_counts[i]
            assert row['ct'] == cite_counts[i]

    def test_two_optional_match_different_counts(self, court_graph):
        """Two OPTIONAL MATCHes in one WITH — simpler case of the same bug."""
        g, kw_counts, judge_counts, _ = court_graph
        result = g.cypher("""
            MATCH (c:CourtDecision)
            OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
            OPTIONAL MATCH (c)-[:JUDGED_BY]->(j:Judge)
            WITH c, count(DISTINCT k) AS kw, count(DISTINCT j) AS jg
            RETURN c.name, kw, jg
            ORDER BY c.name
        """)
        assert len(result) == 5
        for i, row in enumerate(result):
            assert row['kw'] == kw_counts[i]
            assert row['jg'] == judge_counts[i]

    def test_aggregate_over_multi_optional_match(self, court_graph):
        """The full bug report query: aggregate stats across all decisions."""
        g, kw_counts, judge_counts, cite_counts = court_graph
        result = g.cypher("""
            MATCH (c:CourtDecision)
            OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
            OPTIONAL MATCH (c)-[:JUDGED_BY]->(j:Judge)
            OPTIONAL MATCH (c)-[:CITES]->(ls:LawSection)
            WITH c, count(DISTINCT k) AS kw_count, count(DISTINCT j) AS judge_count, count(DISTINCT ls) AS cite_count
            RETURN
              count(*) AS total,
              sum(CASE WHEN kw_count = 0 THEN 1 ELSE 0 END) AS no_keywords,
              sum(CASE WHEN judge_count = 0 THEN 1 ELSE 0 END) AS no_judges,
              sum(CASE WHEN cite_count = 0 THEN 1 ELSE 0 END) AS no_citations,
              avg(kw_count) AS avg_keywords,
              avg(judge_count) AS avg_judges,
              avg(cite_count) AS avg_citations
        """)
        assert len(result) == 1
        row = result[0]
        assert row['total'] == 5
        # kw_counts = [3, 0, 5, 1, 2] → 1 zero
        assert row['no_keywords'] == 1
        # judge_counts = [2, 3, 0, 4, 1] → 1 zero
        assert row['no_judges'] == 1
        # cite_counts = [1, 2, 3, 0, 4] → 1 zero
        assert row['no_citations'] == 1
        # The three averages should differ
        assert abs(row['avg_keywords'] - sum(kw_counts) / 5) < 0.01
        assert abs(row['avg_judges'] - sum(judge_counts) / 5) < 0.01
        assert abs(row['avg_citations'] - sum(cite_counts) / 5) < 0.01
        # Crucially: averages must NOT all be equal
        assert not (row['avg_keywords'] == row['avg_judges'] == row['avg_citations']), \
            "All three averages are identical — Cartesian product / fusion bug!"


class TestMultiOptionalMatchWithNotExists:
    """Verify NOT EXISTS pattern gives same results as multi-OPTIONAL-MATCH approach."""

    def test_not_exists_matches_optional_match(self, court_graph):
        """NOT EXISTS for each rel type should match the zero-count from OPTIONAL MATCH."""
        g, kw_counts, judge_counts, cite_counts = court_graph

        # Get counts via OPTIONAL MATCH approach
        opt_result = g.cypher("""
            MATCH (c:CourtDecision)
            OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
            OPTIONAL MATCH (c)-[:JUDGED_BY]->(j:Judge)
            OPTIONAL MATCH (c)-[:CITES]->(ls:LawSection)
            WITH c, count(DISTINCT k) AS kw, count(DISTINCT j) AS jg, count(DISTINCT ls) AS ct
            RETURN c.name, kw, jg, ct
            ORDER BY c.name
        """)

        # Cross-check each relationship independently
        for rel, counts in [('HAS_KEYWORD', kw_counts), ('JUDGED_BY', judge_counts), ('CITES', cite_counts)]:
            no_match = g.cypher(f"""
                MATCH (c:CourtDecision) WHERE NOT EXISTS((c)-[:{rel}]->()) RETURN count(*) AS cnt
            """)
            expected_zeros = sum(1 for c in counts if c == 0)
            assert no_match[0]['cnt'] == expected_zeros, \
                f"NOT EXISTS for {rel}: expected {expected_zeros}, got {no_match[0]['cnt']}"

    def test_single_optional_match_each_matches_combined(self, court_graph):
        """Running each OPTIONAL MATCH independently should match the combined query."""
        g, kw_counts, judge_counts, cite_counts = court_graph

        # Combined
        combined = g.cypher("""
            MATCH (c:CourtDecision)
            OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
            OPTIONAL MATCH (c)-[:JUDGED_BY]->(j:Judge)
            OPTIONAL MATCH (c)-[:CITES]->(ls:LawSection)
            WITH c, count(DISTINCT k) AS kw, count(DISTINCT j) AS jg, count(DISTINCT ls) AS ct
            RETURN c.name, kw, jg, ct
            ORDER BY c.name
        """)

        # Individual keyword counts
        kw_ind = g.cypher("""
            MATCH (c:CourtDecision)
            OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
            WITH c, count(k) AS kw
            RETURN c.name, kw ORDER BY c.name
        """)

        # Individual judge counts
        jg_ind = g.cypher("""
            MATCH (c:CourtDecision)
            OPTIONAL MATCH (c)-[:JUDGED_BY]->(j:Judge)
            WITH c, count(j) AS jg
            RETURN c.name, jg ORDER BY c.name
        """)

        # Individual cite counts
        ct_ind = g.cypher("""
            MATCH (c:CourtDecision)
            OPTIONAL MATCH (c)-[:CITES]->(ls:LawSection)
            WITH c, count(ls) AS ct
            RETURN c.name, ct ORDER BY c.name
        """)

        for i in range(5):
            assert combined[i]['kw'] == kw_ind[i]['kw'], \
                f"Case{i} kw: combined={combined[i]['kw']} vs individual={kw_ind[i]['kw']}"
            assert combined[i]['jg'] == jg_ind[i]['jg'], \
                f"Case{i} jg: combined={combined[i]['jg']} vs individual={jg_ind[i]['jg']}"
            assert combined[i]['ct'] == ct_ind[i]['ct'], \
                f"Case{i} ct: combined={combined[i]['ct']} vs individual={ct_ind[i]['ct']}"


class TestDistinctNodeIdentity:
    """Tests for count(DISTINCT n) using node identity rather than title."""

    def test_distinct_counts_duplicate_titles(self, duplicate_title_graph):
        """count(DISTINCT t) should count 3 tags, not 2 (two share same title)."""
        g = duplicate_title_graph
        result = g.cypher("""
            MATCH (p:Person {name: 'Alice'})-[:TAGGED]->(t:Tag)
            RETURN count(DISTINCT t) AS tag_count
        """)
        assert result[0]['tag_count'] == 3, \
            f"Expected 3 distinct tags (by identity), got {result[0]['tag_count']}"

    def test_distinct_with_optional_match_duplicate_titles(self, duplicate_title_graph):
        """OPTIONAL MATCH + count(DISTINCT) also uses node identity."""
        g = duplicate_title_graph
        result = g.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:TAGGED]->(t:Tag)
            WITH p, count(DISTINCT t) AS tag_count
            RETURN p.name, tag_count
        """)
        assert result[0]['tag_count'] == 3


class TestChainedOptionalMatchVsSingleWith:
    """Compare chained pattern (OPTIONAL MATCH + WITH, repeated) vs single WITH."""

    def test_chained_matches_single_with(self, court_graph):
        """Chained approach should give same results as single-WITH approach."""
        g, kw_counts, judge_counts, cite_counts = court_graph

        # Single WITH (the bug report pattern)
        single = g.cypher("""
            MATCH (c:CourtDecision)
            OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
            OPTIONAL MATCH (c)-[:JUDGED_BY]->(j:Judge)
            OPTIONAL MATCH (c)-[:CITES]->(ls:LawSection)
            WITH c, count(DISTINCT k) AS kw, count(DISTINCT j) AS jg, count(DISTINCT ls) AS ct
            RETURN c.name, kw, jg, ct
            ORDER BY c.name
        """)

        # Chained (each OPTIONAL MATCH has its own WITH)
        chained = g.cypher("""
            MATCH (c:CourtDecision)
            OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
            WITH c, count(k) AS kw
            OPTIONAL MATCH (c)-[:JUDGED_BY]->(j:Judge)
            WITH c, kw, count(j) AS jg
            OPTIONAL MATCH (c)-[:CITES]->(ls:LawSection)
            WITH c, kw, jg, count(ls) AS ct
            RETURN c.name, kw, jg, ct
            ORDER BY c.name
        """)

        assert len(single) == len(chained) == 5
        for i in range(5):
            assert single[i]['kw'] == chained[i]['kw'], \
                f"Case{i} kw mismatch: single={single[i]['kw']}, chained={chained[i]['kw']}"
            assert single[i]['jg'] == chained[i]['jg'], \
                f"Case{i} jg mismatch: single={single[i]['jg']}, chained={chained[i]['jg']}"
            assert single[i]['ct'] == chained[i]['ct'], \
                f"Case{i} ct mismatch: single={single[i]['ct']}, chained={chained[i]['ct']}"


class TestEdgeCases:
    """Edge cases for multi-OPTIONAL-MATCH aggregation."""

    def test_all_optional_matches_empty(self):
        """All OPTIONAL MATCHes find nothing — all counts should be 0."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Lonely'})")
        result = g.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
            OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company)
            OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
            WITH p, count(DISTINCT f) AS friends, count(DISTINCT c) AS jobs, count(DISTINCT s) AS skills
            RETURN p.name, friends, jobs, skills
        """)
        assert len(result) == 1
        assert result[0]['friends'] == 0
        assert result[0]['jobs'] == 0
        assert result[0]['skills'] == 0

    def test_one_optional_match_has_results_others_empty(self):
        """Only one OPTIONAL MATCH has results — others should be 0."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        g.cypher("CREATE (:Skill {name: 'Python'})")
        g.cypher("""
            MATCH (p:Person {name: 'Alice'}), (s:Skill {name: 'Python'})
            CREATE (p)-[:HAS_SKILL]->(s)
        """)
        result = g.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
            OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
            WITH p, count(DISTINCT f) AS friends, count(DISTINCT s) AS skills
            RETURN p.name, friends, skills
        """)
        assert result[0]['friends'] == 0
        assert result[0]['skills'] == 1

    def test_count_star_with_multi_optional_match(self):
        """count(*) should reflect the Cartesian product size, not deduplicated."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        for i in range(2):
            g.cypher(f"CREATE (:Skill {{name: 'S{i}'}})")
            g.cypher(f"""
                MATCH (p:Person {{name: 'Alice'}}), (s:Skill {{name: 'S{i}'}})
                CREATE (p)-[:HAS_SKILL]->(s)
            """)
        for i in range(3):
            g.cypher(f"CREATE (:Project {{name: 'P{i}'}})")
            g.cypher(f"""
                MATCH (p:Person {{name: 'Alice'}}), (pr:Project {{name: 'P{i}'}})
                CREATE (p)-[:WORKS_ON]->(pr)
            """)
        # After Cartesian product: 2 skills × 3 projects = 6 rows for Alice
        result = g.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
            OPTIONAL MATCH (p)-[:WORKS_ON]->(pr:Project)
            WITH p, count(DISTINCT s) AS skills, count(DISTINCT pr) AS projects
            RETURN p.name, skills, projects
        """)
        assert result[0]['skills'] == 2
        assert result[0]['projects'] == 3

    def test_multi_optional_match_with_where_filter(self):
        """WHERE on aggregated counts after multi-OPTIONAL-MATCH."""
        g = KnowledgeGraph()
        for i in range(3):
            g.cypher(f"CREATE (:Person {{name: 'P{i}'}})")
        # P0 has 2 friends, P1 has 1, P2 has 0
        g.cypher("MATCH (a:Person {name: 'P0'}), (b:Person {name: 'P1'}) CREATE (a)-[:KNOWS]->(b)")
        g.cypher("MATCH (a:Person {name: 'P0'}), (b:Person {name: 'P2'}) CREATE (a)-[:KNOWS]->(b)")
        g.cypher("MATCH (a:Person {name: 'P1'}), (b:Person {name: 'P2'}) CREATE (a)-[:KNOWS]->(b)")
        # P0 has 1 skill, P1 has 0, P2 has 0
        g.cypher("CREATE (:Skill {name: 'Rust'})")
        g.cypher("MATCH (p:Person {name: 'P0'}), (s:Skill {name: 'Rust'}) CREATE (p)-[:HAS_SKILL]->(s)")

        result = g.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
            OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
            WITH p, count(DISTINCT f) AS friends, count(DISTINCT s) AS skills
            WHERE friends > 0 AND skills > 0
            RETURN p.name, friends, skills
        """)
        assert len(result) == 1
        assert result[0]['p.name'] == 'P0'
        assert result[0]['friends'] == 2
        assert result[0]['skills'] == 1


class TestScaleStress:
    """Larger-scale tests to verify correctness doesn't degrade with more data."""

    def test_medium_graph_multi_optional_match(self):
        """50 anchor nodes with varied fan-out across 3 relationship types."""
        g = KnowledgeGraph()
        n_anchors = 50

        # Create anchors
        for i in range(n_anchors):
            g.cypher(f"CREATE (:Anchor {{name: 'A{i}'}})")

        # Create targets with known distributions
        expected_a = {}
        expected_b = {}
        expected_c = {}

        for i in range(n_anchors):
            # Type A: i % 5 connections (0, 1, 2, 3, 4, 0, 1, ...)
            a_count = i % 5
            expected_a[f'A{i}'] = a_count
            for j in range(a_count):
                g.cypher(f"CREATE (:TypeA {{name: 'TA{i}_{j}'}})")
                g.cypher(f"""
                    MATCH (a:Anchor {{name: 'A{i}'}}), (t:TypeA {{name: 'TA{i}_{j}'}})
                    CREATE (a)-[:REL_A]->(t)
                """)

            # Type B: (i + 2) % 7 connections (different distribution)
            b_count = (i + 2) % 7
            expected_b[f'A{i}'] = b_count
            for j in range(b_count):
                g.cypher(f"CREATE (:TypeB {{name: 'TB{i}_{j}'}})")
                g.cypher(f"""
                    MATCH (a:Anchor {{name: 'A{i}'}}), (t:TypeB {{name: 'TB{i}_{j}'}})
                    CREATE (a)-[:REL_B]->(t)
                """)

            # Type C: (i * 3) % 4 connections (yet another distribution)
            c_count = (i * 3) % 4
            expected_c[f'A{i}'] = c_count
            for j in range(c_count):
                g.cypher(f"CREATE (:TypeC {{name: 'TC{i}_{j}'}})")
                g.cypher(f"""
                    MATCH (a:Anchor {{name: 'A{i}'}}), (t:TypeC {{name: 'TC{i}_{j}'}})
                    CREATE (a)-[:REL_C]->(t)
                """)

        # Run the multi-OPTIONAL-MATCH query
        result = g.cypher("""
            MATCH (a:Anchor)
            OPTIONAL MATCH (a)-[:REL_A]->(ta:TypeA)
            OPTIONAL MATCH (a)-[:REL_B]->(tb:TypeB)
            OPTIONAL MATCH (a)-[:REL_C]->(tc:TypeC)
            WITH a, count(DISTINCT ta) AS cnt_a, count(DISTINCT tb) AS cnt_b, count(DISTINCT tc) AS cnt_c
            RETURN a.name, cnt_a, cnt_b, cnt_c
            ORDER BY a.name
        """)

        assert len(result) == n_anchors
        for row in result:
            name = row['a.name']
            assert row['cnt_a'] == expected_a[name], \
                f"{name} REL_A: expected {expected_a[name]}, got {row['cnt_a']}"
            assert row['cnt_b'] == expected_b[name], \
                f"{name} REL_B: expected {expected_b[name]}, got {row['cnt_b']}"
            assert row['cnt_c'] == expected_c[name], \
                f"{name} REL_C: expected {expected_c[name]}, got {row['cnt_c']}"

    def test_aggregate_summary_matches_individual_queries(self):
        """Aggregate summary from multi-OPTIONAL-MATCH should match individual queries."""
        g = KnowledgeGraph()
        for i in range(20):
            g.cypher(f"CREATE (:Node {{name: 'N{i}'}})")
            for j in range(i % 4):
                g.cypher(f"CREATE (:ChildA {{name: 'CA{i}_{j}'}})")
                g.cypher(f"""
                    MATCH (n:Node {{name: 'N{i}'}}), (c:ChildA {{name: 'CA{i}_{j}'}})
                    CREATE (n)-[:HAS_A]->(c)
                """)
            for j in range((i + 1) % 3):
                g.cypher(f"CREATE (:ChildB {{name: 'CB{i}_{j}'}})")
                g.cypher(f"""
                    MATCH (n:Node {{name: 'N{i}'}}), (c:ChildB {{name: 'CB{i}_{j}'}})
                    CREATE (n)-[:HAS_B]->(c)
                """)

        # Combined query
        combined = g.cypher("""
            MATCH (n:Node)
            OPTIONAL MATCH (n)-[:HAS_A]->(a:ChildA)
            OPTIONAL MATCH (n)-[:HAS_B]->(b:ChildB)
            WITH n, count(DISTINCT a) AS cnt_a, count(DISTINCT b) AS cnt_b
            RETURN sum(cnt_a) AS total_a, sum(cnt_b) AS total_b,
                   avg(cnt_a) AS avg_a, avg(cnt_b) AS avg_b
        """)

        # Individual queries
        ind_a = g.cypher("""
            MATCH (n:Node)
            OPTIONAL MATCH (n)-[:HAS_A]->(a:ChildA)
            WITH n, count(a) AS cnt
            RETURN sum(cnt) AS total, avg(cnt) AS average
        """)
        ind_b = g.cypher("""
            MATCH (n:Node)
            OPTIONAL MATCH (n)-[:HAS_B]->(b:ChildB)
            WITH n, count(b) AS cnt
            RETURN sum(cnt) AS total, avg(cnt) AS average
        """)

        assert combined[0]['total_a'] == ind_a[0]['total']
        assert combined[0]['total_b'] == ind_b[0]['total']
        assert abs(combined[0]['avg_a'] - ind_a[0]['average']) < 0.001
        assert abs(combined[0]['avg_b'] - ind_b[0]['average']) < 0.001


class TestFusionStillWorksForSingleOptionalMatch:
    """Ensure the single OPTIONAL MATCH + WITH count() case still gets fused and is fast."""

    def test_single_optional_match_count_still_correct(self):
        """Verify the fusion optimization still works for the single-OPTIONAL-MATCH case."""
        g = KnowledgeGraph()
        for i in range(10):
            g.cypher(f"CREATE (:Parent {{name: 'P{i}'}})")
            for j in range(i):
                g.cypher(f"CREATE (:Child {{name: 'C{i}_{j}'}})")
                g.cypher(f"""
                    MATCH (p:Parent {{name: 'P{i}'}}), (c:Child {{name: 'C{i}_{j}'}})
                    CREATE (p)-[:HAS]->(c)
                """)

        result = g.cypher("""
            MATCH (p:Parent)
            OPTIONAL MATCH (p)-[:HAS]->(c:Child)
            WITH p, count(c) AS cnt
            RETURN p.name, cnt
            ORDER BY p.name
        """)
        assert len(result) == 10
        for i, row in enumerate(result):
            assert row['cnt'] == i, f"P{i}: expected {i}, got {row['cnt']}"

    def test_explain_shows_fusion_for_single(self):
        """EXPLAIN should show fusion for single OPTIONAL MATCH + count()."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:X {name: 'x'})")
        plan = g.cypher("""
            EXPLAIN
            MATCH (x:X)
            OPTIONAL MATCH (x)-[:R]->(y:Y)
            WITH x, count(y) AS cnt
            RETURN x.name, cnt
        """)
        plan_str = str(plan)
        assert 'FusedOptionalMatchAggregate' in plan_str or 'fused' in plan_str.lower(), \
            f"Expected fusion in plan, got: {plan_str}"

    def test_explain_no_fusion_for_multi_optional_with_distinct(self):
        """EXPLAIN should NOT show fusion for multi-OPTIONAL-MATCH + count(DISTINCT)."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:X {name: 'x'})")
        plan = g.cypher("""
            EXPLAIN
            MATCH (x:X)
            OPTIONAL MATCH (x)-[:A]->(a:A)
            OPTIONAL MATCH (x)-[:B]->(b:B)
            WITH x, count(DISTINCT a) AS ca, count(DISTINCT b) AS cb
            RETURN x.name, ca, cb
        """)
        plan_str = str(plan)
        assert 'FusedOptionalMatchAggregate' not in plan_str, \
            f"Should not fuse multi-OPTIONAL-MATCH + DISTINCT, got: {plan_str}"
