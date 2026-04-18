"""Deterministic builder for the Phase 10 golden-fixture graph.

Produces a ~1,000-node KG: Person (500) + Company (300) + Place (200),
~3,000 edges across WORKS_AT / LIVES_IN / LOCATED_IN / KNOWS. Mixed
property types exercise every column-storage path:

- strings: name
- integers: age, founded, population
- floats: lat, lon, salary
- bool: active
- datetime: joined_at

Seeded with ``random.Random(42)`` — byte-identical across runs. Used by
both ``tests/test_golden.py`` (comparison) and
``tests/golden/regenerate.py`` (snapshot refresh).
"""

from __future__ import annotations

import datetime as _dt
import random

import pandas as pd

from kglite import KnowledgeGraph

SEED = 42
N_PERSON = 500
N_COMPANY = 300
N_PLACE = 200

PERSON_FIRST = [
    "Alice",
    "Bob",
    "Carol",
    "Dave",
    "Eva",
    "Frank",
    "Grace",
    "Henry",
    "Ivy",
    "Jack",
]
PERSON_LAST = [
    "Anderson",
    "Berg",
    "Chen",
    "Dahl",
    "Eriksen",
    "Foss",
    "Grønn",
    "Hansen",
    "Iversen",
    "Johansen",
]
COMPANY_SUFFIX = ["AS", "Group", "Holdings", "Labs", "Works"]
PLACE_NAMES = [
    "Oslo",
    "Bergen",
    "Trondheim",
    "Stavanger",
    "Kristiansand",
    "Tromsø",
    "Bodø",
    "Ålesund",
    "Drammen",
    "Fredrikstad",
]


def _make_people(rng: random.Random) -> pd.DataFrame:
    rows = []
    for i in range(N_PERSON):
        first = rng.choice(PERSON_FIRST)
        last = rng.choice(PERSON_LAST)
        rows.append(
            {
                "id": i,
                "name": f"{first} {last}",
                "age": rng.randint(18, 90),
                "salary": round(rng.uniform(30_000, 250_000), 2),
                "active": rng.random() > 0.2,
                "joined_at": _dt.datetime(2020, 1, 1) + _dt.timedelta(days=rng.randint(0, 1500)),
            }
        )
    return pd.DataFrame(rows)


def _make_companies(rng: random.Random) -> pd.DataFrame:
    rows = []
    for i in range(N_COMPANY):
        base = rng.choice(PERSON_LAST)
        suffix = rng.choice(COMPANY_SUFFIX)
        rows.append(
            {
                "cid": i,
                "name": f"{base} {suffix}",
                "founded": rng.randint(1900, 2024),
                "employees": rng.randint(5, 5_000),
            }
        )
    return pd.DataFrame(rows)


def _make_places(rng: random.Random) -> pd.DataFrame:
    rows = []
    for i in range(N_PLACE):
        base = rng.choice(PLACE_NAMES)
        rows.append(
            {
                "pid": i,
                "name": f"{base}_{i}",
                "population": rng.randint(1_000, 1_000_000),
                "lat": round(rng.uniform(58.0, 71.0), 6),
                "lon": round(rng.uniform(4.0, 31.0), 6),
            }
        )
    return pd.DataFrame(rows)


def build_golden_graph(kg: KnowledgeGraph) -> KnowledgeGraph:
    """Populate ``kg`` with the deterministic golden fixture.

    Returns ``kg`` for chaining. Idempotent only on a fresh graph —
    calling twice duplicates nodes.
    """
    rng = random.Random(SEED)

    people = _make_people(rng)
    companies = _make_companies(rng)
    places = _make_places(rng)

    kg.add_nodes(people, "Person", "id", "name")
    kg.add_nodes(companies, "Company", "cid", "name")
    kg.add_nodes(places, "Place", "pid", "name")

    # WORKS_AT — every person at a random company (500 edges).
    works_at = pd.DataFrame([{"pid": i, "cid": rng.randrange(N_COMPANY)} for i in range(N_PERSON)])
    kg.add_connections(works_at, "WORKS_AT", "Person", "pid", "Company", "cid")

    # LIVES_IN — every person at a random place (500 edges).
    lives_in = pd.DataFrame([{"pid": i, "plid": rng.randrange(N_PLACE)} for i in range(N_PERSON)])
    kg.add_connections(lives_in, "LIVES_IN", "Person", "pid", "Place", "plid")

    # LOCATED_IN — every company at a random place (300 edges).
    located_in = pd.DataFrame([{"cid": i, "plid": rng.randrange(N_PLACE)} for i in range(N_COMPANY)])
    kg.add_connections(located_in, "LOCATED_IN", "Company", "cid", "Place", "plid")

    # KNOWS — 1,700 random directed person-person edges (not self, not dup).
    knows_pairs: set[tuple[int, int]] = set()
    while len(knows_pairs) < 1_700:
        a = rng.randrange(N_PERSON)
        b = rng.randrange(N_PERSON)
        if a != b:
            knows_pairs.add((a, b))
    knows = pd.DataFrame([{"a": a, "b": b} for a, b in sorted(knows_pairs)])
    kg.add_connections(knows, "KNOWS", "Person", "a", "Person", "b")

    return kg
