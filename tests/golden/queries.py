"""Seed queries pinned for the Phase 10 golden fixture.

Any intentional output change must be re-committed via
``python tests/golden/regenerate.py``. Test-driven comparison lives in
``tests/test_golden.py``.
"""

from __future__ import annotations

# (slug, cypher). Slug becomes the filename stem (cypher_<slug>.json).
CYPHER_QUERIES: list[tuple[str, str]] = [
    (
        "count_by_type",
        "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS c",
    ),
    (
        "count_by_connection",
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c",
    ),
    (
        "oldest_ten_people",
        "MATCH (p:Person) RETURN p.id AS id, p.age AS age ORDER BY p.age DESC, p.id ASC LIMIT 10",
    ),
    (
        "active_filter_count",
        "MATCH (p:Person) WHERE p.active = true RETURN count(p) AS c",
    ),
    (
        "salary_bucket",
        "MATCH (p:Person) WHERE p.salary >= 200000 "
        "RETURN p.id AS id, p.salary AS s ORDER BY p.salary DESC, p.id ASC LIMIT 20",
    ),
    (
        "companies_founded_before_1950",
        "MATCH (c:Company) WHERE c.founded < 1950 "
        "RETURN c.cid AS cid, c.founded AS founded ORDER BY c.founded ASC, c.cid ASC",
    ),
    (
        "person_to_company_degree_two",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company)-[:LOCATED_IN]->(pl:Place) "
        "WHERE pl.population > 500000 "
        "RETURN p.id AS pid, c.cid AS cid, pl.pid AS plid "
        "ORDER BY p.id, c.cid, pl.pid LIMIT 25",
    ),
    (
        "knows_two_hop",
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
        "WHERE a.id = 0 RETURN DISTINCT c.id AS cid ORDER BY cid LIMIT 15",
    ),
    (
        "place_population_stats",
        "MATCH (pl:Place) RETURN count(pl) AS n, min(pl.population) AS pmin, max(pl.population) AS pmax",
    ),
    (
        "top_companies_by_hires",
        "MATCH (c:Company)<-[:WORKS_AT]-(p:Person) "
        "RETURN c.cid AS cid, count(p) AS hires "
        "ORDER BY hires DESC, cid ASC LIMIT 10",
    ),
]

# find(name, node_type=...) — code-entity-only in this build; always empty
# on the social graph, which still pins the API contract (empty list).
FIND_QUERIES: list[tuple[str, str, str | None]] = [
    ("alice_any", "Alice", None),
    ("function_none", "execute", "Function"),
    ("place_oslo", "Oslo", None),
    ("company_labs", "Labs", None),
    ("empty", "zzzzzzzzzz", None),
]
