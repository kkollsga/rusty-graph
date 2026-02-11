"""Shared fixtures for kglite test suite."""

import pytest
import pandas as pd
from kglite import KnowledgeGraph


@pytest.fixture
def empty_graph():
    """Empty graph for edge case testing."""
    return KnowledgeGraph()


@pytest.fixture
def small_graph():
    """Small graph: 3 Person nodes + 3 KNOWS edges.

    Persons: Alice (age=28, city=Oslo), Bob (age=35, city=Bergen), Charlie (age=42, city=Oslo)
    Edges: Alice->Bob, Bob->Charlie, Alice->Charlie
    """
    graph = KnowledgeGraph()

    people = pd.DataFrame({
        'person_id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [28, 35, 42],
        'city': ['Oslo', 'Bergen', 'Oslo'],
    })
    graph.add_nodes(people, 'Person', 'person_id', 'name')

    edges = pd.DataFrame({
        'from_id': [1, 2, 1],
        'to_id': [2, 3, 3],
        'since': [2020, 2019, 2021],
    })
    graph.add_connections(edges, 'KNOWS', 'Person', 'from_id', 'Person', 'to_id',
                          columns=['since'])

    return graph


@pytest.fixture
def social_graph():
    """Medium graph: 20 Person + 5 Company nodes, KNOWS/WORKS_AT edges.

    Persons: Person_1..Person_20, age=21..40, city in [Oslo, Bergen, Stavanger, Trondheim]
             email is None for odd-numbered persons (nullable field)
    Companies: TechCorp, DataInc, CloudSoft, AILabs, DevHouse
    KNOWS: each person knows next 3 persons (with 'since' property)
    WORKS_AT: each person works at one company (with 'start_year' property)
    """
    graph = KnowledgeGraph()

    people = pd.DataFrame({
        'person_id': list(range(1, 21)),
        'name': [f'Person_{i}' for i in range(1, 21)],
        'age': [20 + i for i in range(1, 21)],
        'city': (['Oslo'] * 5 + ['Bergen'] * 5 + ['Stavanger'] * 5 + ['Trondheim'] * 5),
        'salary': [50000 + i * 5000 for i in range(20)],
        'email': [f'person{i}@test.com' if i % 2 == 0 else None for i in range(1, 21)],
    })
    graph.add_nodes(people, 'Person', 'person_id', 'name')

    companies = pd.DataFrame({
        'company_id': list(range(100, 105)),
        'name': ['TechCorp', 'DataInc', 'CloudSoft', 'AILabs', 'DevHouse'],
        'industry': ['Tech', 'Data', 'Cloud', 'AI', 'Software'],
    })
    graph.add_nodes(companies, 'Company', 'company_id', 'name')

    knows_edges = []
    for i in range(1, 21):
        for j in range(i + 1, min(i + 4, 21)):
            knows_edges.append({'from_id': i, 'to_id': j, 'since': 2015 + (i % 5)})
    knows_df = pd.DataFrame(knows_edges)
    graph.add_connections(knows_df, 'KNOWS', 'Person', 'from_id', 'Person', 'to_id',
                          columns=['since'])

    works_at = pd.DataFrame({
        'person_id': list(range(1, 21)),
        'company_id': [100 + (i % 5) for i in range(20)],
        'start_year': [2018 + (i % 4) for i in range(20)],
    })
    graph.add_connections(works_at, 'WORKS_AT', 'Person', 'person_id',
                          'Company', 'company_id', columns=['start_year'])

    return graph


@pytest.fixture
def petroleum_graph():
    """Domain graph: Play/Prospect/Discovery/Estimate with temporal and spatial data.

    3 Plays with lat/lon
    20 Prospects with status, geoprovince, lat/lon, date_from/date_to
    10 Discoveries with resource_type, lat/lon
    50 Estimates with value, confidence, date_from/date_to (datetime)
    Connections: HAS_PROSPECT, BECAME_DISCOVERY (share_pct), HAS_ESTIMATE (weight)
    """
    graph = KnowledgeGraph()

    plays = pd.DataFrame({
        'play_id': [1, 2, 3],
        'name': ['North Sea Play', 'Atlantic Play', 'Barents Play'],
        'region': ['Norwegian Sea', 'Atlantic', 'Barents Sea'],
        'latitude': [62.0, 64.5, 71.0],
        'longitude': [5.0, 3.0, 25.0],
    })
    graph.add_nodes(plays, 'Play', 'play_id', 'name')

    prospects = pd.DataFrame({
        'prospect_id': list(range(100, 120)),
        'name': [f'Prospect_{i}' for i in range(20)],
        'status': ['Active'] * 10 + ['Closed'] * 5 + ['Matured'] * 5,
        'geoprovince': ['N3'] * 7 + ['M3'] * 7 + ['B1'] * 6,
        'latitude': [60.0 + i * 0.5 for i in range(20)],
        'longitude': [4.0 + i * 0.3 for i in range(20)],
        'date_from': ['2020-01-01'] * 10 + ['2019-01-01'] * 10,
        'date_to': ['2025-12-31'] * 10 + ['2023-12-31'] * 10,
    })
    graph.add_nodes(prospects, 'Prospect', 'prospect_id', 'name')

    discoveries = pd.DataFrame({
        'discovery_id': list(range(200, 210)),
        'name': [f'Discovery_{i}' for i in range(10)],
        'resource_type': ['Oil'] * 5 + ['Gas'] * 5,
        'latitude': [59.0 + i * 0.4 for i in range(10)],
        'longitude': [2.0 + i * 0.2 for i in range(10)],
    })
    graph.add_nodes(discoveries, 'Discovery', 'discovery_id', 'name')

    estimates = pd.DataFrame({
        'estimate_id': list(range(300, 350)),
        'name': [f'Estimate_{i}' for i in range(50)],
        'value': [10.0 + i * 20.0 for i in range(50)],
        'confidence': [0.5 + (i % 10) * 0.05 for i in range(50)],
        'date_from': ['2020-01-01'] * 25 + ['2021-01-01'] * 25,
        'date_to': ['2020-12-31'] * 25 + ['2021-12-31'] * 25,
    })
    graph.add_nodes(estimates, 'Estimate', 'estimate_id', 'name',
                    column_types={'date_from': 'datetime', 'date_to': 'datetime'})

    play_prospect = pd.DataFrame({
        'play_id': [1] * 7 + [2] * 7 + [3] * 6,
        'prospect_id': list(range(100, 120)),
    })
    graph.add_connections(play_prospect, 'HAS_PROSPECT', 'Play', 'play_id',
                          'Prospect', 'prospect_id')

    prospect_discovery = pd.DataFrame({
        'prospect_id': [100, 101, 102, 107, 108, 109, 114, 115, 116, 117],
        'discovery_id': list(range(200, 210)),
        'share_pct': [100.0, 75.0, 50.0, 80.0, 60.0, 40.0, 90.0, 70.0, 55.0, 45.0],
    })
    graph.add_connections(prospect_discovery, 'BECAME_DISCOVERY', 'Prospect', 'prospect_id',
                          'Discovery', 'discovery_id', columns=['share_pct'])

    prospect_estimate = pd.DataFrame({
        'prospect_id': [100 + (i % 20) for i in range(50)],
        'estimate_id': list(range(300, 350)),
        'weight': [0.5 + (i % 10) * 0.05 for i in range(50)],
    })
    graph.add_connections(prospect_estimate, 'HAS_ESTIMATE', 'Prospect', 'prospect_id',
                          'Estimate', 'estimate_id', columns=['weight'])

    return graph
