"""End-to-end tests for persistent disk-backed property indexes.

Verifies that ``create_index`` on a ``storage='disk'`` graph:
  1. Succeeds and reports ``persistent=True``.
  2. Writes four ``property_index_*`` files next to the CSR.
  3. Is consulted by the Cypher planner on ``WHERE n.prop = 'X'`` queries.
  4. Survives a save/load roundtrip (lazy-loaded on first lookup).
"""

from pathlib import Path
import shutil
import tempfile

import pandas as pd
import pytest

from kglite import KnowledgeGraph, load


@pytest.fixture
def disk_dir():
    d = tempfile.mkdtemp(prefix="kglite_prop_idx_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _build_disk_graph(path: str) -> KnowledgeGraph:
    g = KnowledgeGraph(storage="disk", path=path)
    nodes = pd.DataFrame(
        {
            "nid": [f"Q{i}" for i in range(1, 6)],
            "label": ["Norway", "Sweden", "Denmark", "Finland", "Iceland"],
            "type": ["Country"] * 5,
        }
    )
    g.add_nodes(nodes, "Country", "nid", "label")
    return g


class TestPersistentIndexBuild:
    def test_create_index_reports_persistent_on_disk(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        info = g.create_index("Country", "label")
        assert info["persistent"] is True
        assert info["created"] is True
        assert info["unique_values"] == 5

    def test_create_index_writes_files_to_disk(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        d = Path(disk_dir)
        assert (d / "property_index_Country_label_meta.bin").exists()
        assert (d / "property_index_Country_label_keys.bin").exists()
        assert (d / "property_index_Country_label_offsets.bin").exists()
        assert (d / "property_index_Country_label_ids.bin").exists()


class TestPlannerRoutesToIndex:
    def test_equality_lookup_finds_node_via_index(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        result = g.cypher("MATCH (n:Country {label: 'Norway'}) RETURN n.nid").to_df()
        assert len(result) == 1
        assert result["n.nid"][0] == "Q1"

    def test_equality_lookup_no_match_returns_empty(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        result = g.cypher("MATCH (n:Country {label: 'Atlantis'}) RETURN n.nid").to_df()
        assert len(result) == 0

    def test_lookup_without_index_still_works_via_scan(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        # No index created — fallback scan should still find the node.
        result = g.cypher("MATCH (n:Country {label: 'Sweden'}) RETURN n.nid").to_df()
        assert len(result) == 1
        assert result["n.nid"][0] == "Q2"

    def test_starts_with_uses_prefix_index(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        # 4 labels start with {N,S,D,F,I}; only "F" matches Finland.
        result = g.cypher("MATCH (n:Country) WHERE n.label STARTS WITH 'F' RETURN n.nid").to_df()
        assert len(result) == 1
        assert result["n.nid"][0] == "Q4"  # Finland

    def test_starts_with_no_match(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        result = g.cypher("MATCH (n:Country) WHERE n.label STARTS WITH 'Z' RETURN n.nid").to_df()
        assert len(result) == 0

    def test_starts_with_works_without_index(self, disk_dir):
        # STARTS WITH falls back to post-filter scan when no index exists.
        g = _build_disk_graph(disk_dir)
        result = g.cypher("MATCH (n:Country) WHERE n.label STARTS WITH 'I' RETURN n.nid").to_df()
        assert len(result) == 1
        assert result["n.nid"][0] == "Q5"  # Iceland


class TestQueryDiagnostics:
    def test_diagnostics_reports_elapsed_and_timeout(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        r = g.cypher("MATCH (n:Country {label: 'Norway'}) RETURN n.nid")
        d = r.diagnostics
        assert d is not None
        assert d["elapsed_ms"] >= 0
        assert d["timed_out"] is False
        # Disk default timeout = 10_000 ms; user did not override.
        assert d["timeout_ms"] == 10_000

    def test_diagnostics_respects_explicit_timeout(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        r = g.cypher("MATCH (n:Country) RETURN n.nid", timeout_ms=500)
        assert r.diagnostics["timeout_ms"] == 500

    def test_diagnostics_none_when_timeout_disabled(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        r = g.cypher("MATCH (n:Country) RETURN n.nid", timeout_ms=0)
        assert r.diagnostics["timeout_ms"] is None


class TestDescribeAnnotations:
    def test_indexed_regular_property_annotated_memory(self):
        # Memory backend: describe() emits a <properties> block with
        # column stats for every non-title/non-id column, so indexed
        # annotations are verifiable. (Disk inventory skips the block
        # for small types; the annotation plumbing is the same.)
        g = KnowledgeGraph()
        nodes = pd.DataFrame(
            {
                "nid": [f"Q{i}" for i in range(1, 4)],
                "label": ["Alpha", "Beta", "Gamma"],
                "continent": ["Europe"] * 3,
            }
        )
        g.add_nodes(nodes, "Country", "nid", "label")
        g.create_index("Country", "continent")
        d = g.describe()
        # String columns get both equality and prefix indexing.
        assert 'indexed="eq,prefix"' in d

    def test_indexing_hint_in_extensions(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        d = g.describe()
        assert "<indexing hint=" in d


class TestPersistenceAcrossReload:
    def test_index_survives_save_and_reload(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        g.save(disk_dir)
        del g
        reloaded = load(disk_dir)
        # First lookup after reload triggers lazy mmap open of the index.
        result = reloaded.cypher("MATCH (n:Country {label: 'Denmark'}) RETURN n.nid").to_df()
        assert len(result) == 1
        assert result["n.nid"][0] == "Q3"
