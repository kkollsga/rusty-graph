"""Phase 2 tests for the streaming disk-to-disk subgraph filter.

Phase 2 ships only Pass A — the sequential edge-endpoints scan that
builds a kept-nodes bitset and reports filter stats. No output graph is
written yet; later phases plumb the rest of the pipeline.

The Pass A entry point is the debug method ``_scan_edges_filtered``.
"""

import shutil
import tempfile

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def disk_dir():
    d = tempfile.mkdtemp(prefix="kglite_subgraph_streaming_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def build_disk_graph_with_articles_and_authors(path: str) -> KnowledgeGraph:
    """A miniature 'Wikidata-shape' disk graph: Articles, Authors, and
    extra noise types/edges so the filter actually has to discriminate.
    """
    g = KnowledgeGraph(storage="disk", path=path)

    articles = pd.DataFrame(
        {
            "aid": ["a1", "a2", "a3", "a4"],
            "title": ["Paper One", "Paper Two", "Paper Three", "Paper Four"],
        }
    )
    g.add_nodes(articles, "Article", "aid", "title")

    authors = pd.DataFrame(
        {
            "uid": ["p1", "p2", "p3"],
            "name": ["Alice", "Bob", "Carol"],
        }
    )
    g.add_nodes(authors, "Author", "uid", "name")

    venues = pd.DataFrame(
        {
            "vid": ["v1", "v2"],
            "name": ["Venue One", "Venue Two"],
        }
    )
    g.add_nodes(venues, "Venue", "vid", "name")

    # AUTHORED_BY: a1↔p1, a1↔p2, a2↔p2, a3↔p3, a4↔p3.
    # Touches 4 articles + 3 authors = 7 nodes.
    authored = pd.DataFrame({"from_id": ["a1", "a1", "a2", "a3", "a4"], "to_id": ["p1", "p2", "p2", "p3", "p3"]})
    g.add_connections(authored, "AUTHORED_BY", "Article", "from_id", "Author", "to_id")

    # PUBLISHED_IN: a1→v1, a2→v1, a3→v2, a4→v2 (4 edges, mixes Venue in).
    published = pd.DataFrame({"from_id": ["a1", "a2", "a3", "a4"], "to_id": ["v1", "v1", "v2", "v2"]})
    g.add_connections(published, "PUBLISHED_IN", "Article", "from_id", "Venue", "to_id")

    return g


class TestPassAFilters:
    """Pass A correctness: filter only AUTHORED_BY → 4 articles + 3 authors."""

    def test_authored_by_filter_only(self, disk_dir):
        g = build_disk_graph_with_articles_and_authors(disk_dir)
        stats = g._scan_edges_filtered(edge_types=["AUTHORED_BY"])

        assert stats["total_edge_count"] == 9  # 5 AUTHORED_BY + 4 PUBLISHED_IN
        assert stats["kept_edge_count"] == 5
        assert stats["kept_node_count"] == 7  # 4 articles + 3 authors
        assert stats["scan_duration_secs"] >= 0.0

    def test_published_in_filter_only(self, disk_dir):
        g = build_disk_graph_with_articles_and_authors(disk_dir)
        stats = g._scan_edges_filtered(edge_types=["PUBLISHED_IN"])

        assert stats["kept_edge_count"] == 4
        # 4 articles + 2 venues = 6 nodes (no authors)
        assert stats["kept_node_count"] == 6

    def test_multiple_edge_types(self, disk_dir):
        g = build_disk_graph_with_articles_and_authors(disk_dir)
        stats = g._scan_edges_filtered(edge_types=["AUTHORED_BY", "PUBLISHED_IN"])
        assert stats["kept_edge_count"] == 9
        # Every node has at least one edge → all 9 nodes kept
        assert stats["kept_node_count"] == 9

    def test_no_filter_keeps_all(self, disk_dir):
        g = build_disk_graph_with_articles_and_authors(disk_dir)
        stats = g._scan_edges_filtered(edge_types=None)
        assert stats["kept_edge_count"] == stats["total_edge_count"] == 9
        assert stats["kept_node_count"] == 9

    def test_unknown_edge_type_keeps_nothing(self, disk_dir):
        g = build_disk_graph_with_articles_and_authors(disk_dir)
        stats = g._scan_edges_filtered(edge_types=["NOT_A_REAL_EDGE_TYPE"])
        assert stats["kept_edge_count"] == 0
        assert stats["kept_node_count"] == 0


class TestPassAGating:
    """The streaming pipeline is disk-only; in-memory graphs must error
    out so users land on the existing fast `to_subgraph().save()` path.
    """

    def test_in_memory_graph_rejected(self):
        g = KnowledgeGraph()  # default = in-memory
        nodes = pd.DataFrame({"nid": ["A", "B"], "name": ["Alice", "Bob"]})
        g.add_nodes(nodes, "Person", "nid", "name")

        with pytest.raises(ValueError, match="disk-backed"):
            g._scan_edges_filtered(edge_types=["KNOWS"])


class TestPassAFileOutput:
    """Phase 4: Pass A also spills kept edges to a temp file. The file
    is the input handed to the merge-sort builder in subsequent phases.
    """

    def test_kept_edges_file_written_with_correct_count(self, disk_dir):
        import os

        g = build_disk_graph_with_articles_and_authors(disk_dir)
        out_path = os.path.join(disk_dir, "kept_edges.tmp")

        stats = g._scan_edges_filtered(edge_types=["AUTHORED_BY"], kept_edges_out=out_path)

        # Same logical kept counts as the no-file variant.
        assert stats["kept_edge_count"] == 5
        assert stats["kept_node_count"] == 7
        assert stats["kept_edge_records"] == 5

        # File exists and is sized for at least 5 records (the file is
        # pre-allocated for the worst case of all source edges, but the
        # logical record count is what matters).
        assert os.path.exists(out_path)
        # Each record is (u32, u32, u64) = 16 bytes; mmap'd file has
        # capacity ≥ kept_edge_records × 16. Smaller graphs may pad to
        # the OS page size, so just check non-empty.
        assert os.path.getsize(out_path) >= 5 * 16

    def test_rank_index_kept_count_matches_bitset(self, disk_dir):
        import os

        g = build_disk_graph_with_articles_and_authors(disk_dir)
        out_path = os.path.join(disk_dir, "kept_edges.tmp")
        stats = g._scan_edges_filtered(edge_types=["AUTHORED_BY"], kept_edges_out=out_path)

        # Phase 3 RankIndex built from Pass A's bitset must produce the
        # same kept count as Bitset::count_ones — sanity check the rank
        # primitive end-to-end on real disk data.
        assert stats["rank_kept_count"] == stats["kept_node_count"]

    def test_no_filter_writes_all_edges(self, disk_dir):
        import os

        g = build_disk_graph_with_articles_and_authors(disk_dir)
        out_path = os.path.join(disk_dir, "kept_edges_all.tmp")

        stats = g._scan_edges_filtered(edge_types=None, kept_edges_out=out_path)

        assert stats["kept_edge_records"] == 9
        assert stats["kept_edge_count"] == 9
        assert os.path.exists(out_path)
