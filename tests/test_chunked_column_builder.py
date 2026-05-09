"""End-to-end tests for the chunked-and-spill column store materializer.

The chunked builder bounds peak heap by spilling each ``chunk_size_rows``
rows to its own subdirectory and dropping the in-memory ``ColumnStore``
before continuing. The risk is that the merge step's string-offset
rebasing introduces silent off-by-ones; these tests force chunking on a
small graph so any rebase bug shows up as wrong row reads after reload.

The chunk size is controlled by ``KGLITE_SUBSET_CHUNK_ROWS``. Tests set
it to a small value (e.g. ``2`` or ``5``) so a fixture with a few hundred
rows produces dozens of chunks.
"""

from __future__ import annotations

import os
import resource
import shutil
import sys
import tempfile

import pandas as pd
import pytest

from kglite import KnowledgeGraph, load


@pytest.fixture
def disk_dir():
    d = tempfile.mkdtemp(prefix="kglite_chunked_builder_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _rss_mb() -> float:
    """Peak RSS in MB (matches the pattern in test_phase4_parity.py)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def build_multi_property_disk_graph(path: str, n_articles: int, n_authors: int) -> str:
    """Build a disk graph with multi-typed properties + an edge type, so
    the merge logic exercises Int64 / Str / nulls in addition to Mixed.
    Returns the disk graph path (the input ``path``).
    """
    g = KnowledgeGraph(storage="disk", path=path)

    # Articles: id (str), title (str of varying length to stress
    # offset rebasing), year (int64), score (float64).
    article_titles = [
        # Mix of short, medium, long strings interleaved.
        ("X" if i % 4 == 0 else "MM" * 25 if i % 4 == 1 else "L" * 500 if i % 4 == 2 else f"row{i}")
        for i in range(n_articles)
    ]
    articles = pd.DataFrame(
        {
            "aid": [f"a{i:04d}" for i in range(n_articles)],
            "title": article_titles,
            "year": [2000 + (i % 25) for i in range(n_articles)],
            "score": [0.1 * i for i in range(n_articles)],
        }
    )
    g.add_nodes(articles, "Article", "aid", "title")

    # Authors: similar shape.
    authors = pd.DataFrame(
        {
            "uid": [f"p{i:04d}" for i in range(n_authors)],
            "name": [f"Author #{i}" for i in range(n_authors)],
            "h_index": [i % 50 for i in range(n_authors)],
        }
    )
    g.add_nodes(authors, "Author", "uid", "name")

    # AUTHORED_BY: cycle authors across articles so every article has
    # at least one author.
    edges = pd.DataFrame(
        {
            "from_id": [f"a{i:04d}" for i in range(n_articles)],
            "to_id": [f"p{i % n_authors:04d}" for i in range(n_articles)],
        }
    )
    g.add_connections(edges, "AUTHORED_BY", "Article", "from_id", "Author", "to_id")
    g.save(path)
    return path


class TestChunkedBuilderCorrectness:
    """Force multi-chunk merging on a small graph and verify reload
    produces exactly the same row data as the in-memory baseline."""

    def test_articles_authored_by_round_trip_with_forced_chunking(self, disk_dir, monkeypatch):
        # 250 articles, 30 authors → 250 AUTHORED_BY edges. With
        # KGLITE_SUBSET_CHUNK_ROWS=20 the article materialization
        # produces 13 chunks, the author materialization produces 2,
        # exercising offset rebasing for both at varying chunk counts.
        monkeypatch.setenv("KGLITE_SUBSET_CHUNK_ROWS", "20")

        src_path = os.path.join(disk_dir, "src")
        out_path = os.path.join(disk_dir, "out.kgl")
        build_multi_property_disk_graph(src_path, n_articles=250, n_authors=30)

        src = load(src_path)
        src._save_subset_filtered_by_edge_type(out_path, ["AUTHORED_BY"])

        sub = load(out_path)

        # All 250 articles + all 30 authors are kept (every article
        # contributes one P50 edge endpoint).
        rows_a = sub.cypher("MATCH (a:Article) RETURN count(a) AS c").to_df()
        rows_p = sub.cypher("MATCH (p:Author) RETURN count(p) AS c").to_df()
        rows_e = sub.cypher("MATCH ()-[r:AUTHORED_BY]->() RETURN count(r) AS c").to_df()
        assert int(rows_a["c"][0]) == 250
        assert int(rows_p["c"][0]) == 30
        assert int(rows_e["c"][0]) == 250

        # Per-row exact match: pull every article's id, title, year,
        # score from baseline (the source itself, since save_subset
        # keeps every article) and from the chunked subset, sort, and
        # compare.
        baseline_articles = (
            src.cypher("MATCH (a:Article) RETURN a.id AS id, a.title AS title, a.year AS year, a.score AS score")
            .to_df()
            .sort_values("id")
            .reset_index(drop=True)
        )
        sub_articles = (
            sub.cypher("MATCH (a:Article) RETURN a.id AS id, a.title AS title, a.year AS year, a.score AS score")
            .to_df()
            .sort_values("id")
            .reset_index(drop=True)
        )
        # If the offset rebase is broken, titles would come back as
        # garbage / mis-aligned bytes. Stringent equality catches it.
        pd.testing.assert_frame_equal(baseline_articles, sub_articles)

        baseline_authors = (
            src.cypher("MATCH (p:Author) RETURN p.id AS id, p.title AS title, p.h_index AS h_index")
            .to_df()
            .sort_values("id")
            .reset_index(drop=True)
        )
        sub_authors = (
            sub.cypher("MATCH (p:Author) RETURN p.id AS id, p.title AS title, p.h_index AS h_index")
            .to_df()
            .sort_values("id")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(baseline_authors, sub_authors)

    def test_edge_case_single_partial_chunk(self, disk_dir, monkeypatch):
        # Force chunk size larger than total kept rows → single tail
        # chunk that must be flushed by finalize().
        monkeypatch.setenv("KGLITE_SUBSET_CHUNK_ROWS", "10000")

        src_path = os.path.join(disk_dir, "src")
        out_path = os.path.join(disk_dir, "out.kgl")
        build_multi_property_disk_graph(src_path, n_articles=5, n_authors=2)

        src = load(src_path)
        src._save_subset_filtered_by_edge_type(out_path, ["AUTHORED_BY"])
        sub = load(out_path)

        assert int(sub.cypher("MATCH (a:Article) RETURN count(a) AS c").to_df()["c"][0]) == 5
        assert int(sub.cypher("MATCH (p:Author) RETURN count(p) AS c").to_df()["c"][0]) == 2

    def test_multi_chunk_string_offsets_extreme_lengths(self, disk_dir, monkeypatch):
        # Force chunk size very small (3) and use article titles that
        # vary from 1 char to 500 chars to stress the offset rebasing.
        # Any byte-level mismatch in the merged offsets file shows up
        # as the wrong title coming back per row.
        monkeypatch.setenv("KGLITE_SUBSET_CHUNK_ROWS", "3")

        src_path = os.path.join(disk_dir, "src")
        out_path = os.path.join(disk_dir, "out.kgl")
        build_multi_property_disk_graph(src_path, n_articles=40, n_authors=5)

        src = load(src_path)
        src._save_subset_filtered_by_edge_type(out_path, ["AUTHORED_BY"])
        sub = load(out_path)

        baseline = (
            src.cypher("MATCH (a:Article) RETURN a.id AS id, a.title AS title")
            .to_df()
            .sort_values("id")
            .reset_index(drop=True)
        )
        chunked = (
            sub.cypher("MATCH (a:Article) RETURN a.id AS id, a.title AS title")
            .to_df()
            .sort_values("id")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(baseline, chunked)


class TestChunkedBuilderBoundedMemory:
    """Sanity check: pushing a few hundred rows with small chunks should
    not leak the entire row corpus into heap. This isn't a strict cap
    test (CI variance is high) — just a regression guard against an
    accidental return of the v1 mmap-grow-forever pattern.
    """

    def test_bounded_memory_on_small_graph(self, disk_dir, monkeypatch):
        monkeypatch.setenv("KGLITE_SUBSET_CHUNK_ROWS", "10")

        src_path = os.path.join(disk_dir, "src")
        out_path = os.path.join(disk_dir, "out.kgl")
        build_multi_property_disk_graph(src_path, n_articles=200, n_authors=20)

        src = load(src_path)
        pre = _rss_mb()
        src._save_subset_filtered_by_edge_type(out_path, ["AUTHORED_BY"])
        post = _rss_mb()

        # 200 articles + 20 authors with small properties — even with
        # chunk + merge overhead, peak should stay well under 200 MB
        # delta. CI variance on macOS is high, hence the loose cap.
        assert post - pre < 200, (
            f"chunked builder leaked memory: post={post:.1f} MB pre={pre:.1f} MB (delta={post - pre:.1f} MB)"
        )
