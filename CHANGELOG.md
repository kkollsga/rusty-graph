# Changelog

All notable changes to KGLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.5] - 2026-04-10

### Added
- **`describe()` extreme-scale support**: Adaptive output for graphs with thousands or millions of types. Four scale tiers: Small (≤15 types, inline detail), Medium (16-200, compact listing), Large (201-5000, top-50 + search hint), Extreme (5001+, statistical summary).
- **`describe(type_search='...')`**: Find types by name with 1-layer neighborhood fan-out. Returns matching types with their connections plus connected types — enables domain discovery in a single call.
- **`rebuild_caches()`**: Force computation of type connectivity, edge type counts, and connection endpoint types in a single O(E) pass. Caches are persisted by `save()` and restored by `load()`.
- **Type connectivity cache**: Pre-computed type-level graph `(src_type, conn_type, tgt_type, count)` triples. Makes `type_search` and `describe(types=[...])` instant on any scale.
- **Lazy connectivity compute**: For Large/Extreme graphs, type connectivity is computed on first `type_search` call and cached for the session.

### Changed
- **`describe()` on Wikidata**: Output reduced from 2.9MB/508s to 3KB/0.15s. `type_search` with warm cache is sub-millisecond (was 2082s).
- **Performance guards**: Sampled neighbor schema for types >50K nodes, skip join candidates for >200 types, bounded error messages for large graphs.
- **Connection overview capping**: `describe(connections=True)` caps at 50 connection types for graphs with >500 connection types.
- **Empty endpoint resolution**: Disk-imported graphs with empty `connection_type_metadata` endpoints get source/target types resolved via bounded edge scan.
- **CSR build: in_edges merge sort**: Replaced scatter-write with merge sort for in_edges (1407s → 259s, 5.4× faster). Power-law target distributions caused page cache thrashing with scatter; merge sort uses only sequential I/O.
- **CSR build: zero-fill elimination**: `mapped_zeroed` creates mmap files at full size without writing zeros — OS lazy-fills pages on demand. Saves 13.8 GB of writes.
- **CSR build: edge_endpoints reuse**: Steps 3-4 read from edge_endpoints (written in step 1) instead of re-reading pending_edges. Eliminates one redundant 13.8 GB copy.
- **Wikidata build: 93 min → 73 min** (Phase 3 CSR: 1842s → 631s).
- **Cypher backtick type names**: `MATCH (n:\`programming language\`)` now works. Pattern parser re-adds backticks when reconstructing identifiers with spaces.
- **Q-code type resolution**: Post-Phase-1 pass resolves raw Q-code type names (e.g., Q13442814 → "scholarly article") using the complete label cache. Single sequential scan of node_slots.
- **NTriples type connectivity**: Type connectivity triples accumulated inline during edge creation, eliminating separate O(E) rebuild pass for freshly built graphs.
- **`get_edge_type_counts()` memory-safe**: Fallback path uses `edge_endpoint_keys()` (mmap reads) instead of `edge_weights()` (which materialized all EdgeData → OOM on disk graphs).

## [0.7.4] - 2026-04-08

### Changed
- **CsrEdge 16 → 8 bytes**: Removed `conn_type` from CSR edge records. Connection type stored only in `EdgeEndpoints`. Saves ~14 GB on Wikidata (out_edges + in_edges halved).
- **MergeSortEntry 24 → 12 bytes**: Removed `conn_type` from sort entries. 2× more edges per sort chunk during CSR build.
- **Edge conn_type pre-filter**: `DiskEdges` iterator checks `edge_endpoints` before `materialize_edge()`, skipping arena allocation and property HashMap lookup for non-matching edges.
- **Arena clearing at query boundaries**: `reset_arenas()` called at start of every Cypher execution. Prevents unbounded memory growth across queries (was the OOM cause on Wikidata).
- **`node_type_of()` — zero-materialization type check**: Reads directly from mmap'd `node_slots` (16-byte struct). Used in all Cypher executor fast paths and pattern matching hot loops instead of `node_weight()`.
- **Edge properties fast path**: `materialize_edge()` skips HashMap lookup when `edge_properties` is empty (common for Wikidata — 862M edges, zero properties).
- **Source node cap with LIMIT**: Multi-hop patterns with LIMIT N only allocate PatternMatch objects for `N × 10,000` source nodes instead of the full type.
- **`expand_from_node` limit propagation**: Edge expansion stops after collecting enough results instead of eagerly materializing all matching edges.
- **`id_indices` built on load**: Disk graphs build id_indices from column stores during load (no node materialization). Enables O(1) cross-type id lookup.
- **`lookup_by_id_normalized` trusts id_indices**: When id_indices exist for a type, the O(1) lookup result is trusted without falling through to linear scan.

### Added
- `DiskGraph::node_type_of()` — O(1) node type lookup from mmap'd node_slots.
- `DiskGraph::reset_arenas()` — public arena clearing for query boundary use.
- `DiskGraph::edges_directed_filtered_iter()` — pre-filtered edge iteration by connection type.
- `GraphBackend::node_type_of()`, `edges_directed_filtered()`, `reset_arenas()` — backend-agnostic wrappers.
- `DirGraph::build_id_index_from_columns()` — builds id_indices directly from mmap'd column stores without node materialization.
- `WHERE id(n) = X` pushdown in planner — converts `id()` function calls to inline `{id: X}` pattern properties.
- Cross-type id lookup in `find_matching_nodes` — untyped `{id: X}` patterns try all types via id_indices (O(types) × O(1)).
- `estimate_node_selectivity` returns 1 for any `{id: X}` pattern regardless of type.

### Fixed
- **Typed edge queries on disk graphs returning 0 rows**: `has_connection_type()` returned false when `connection_type_metadata` was empty (disk graphs skip O(types²) metadata). Fixed by falling back to interner check.
- **N-Triples build not registering connection type names**: Added lightweight connection type metadata registration (names only, no type×type matrix).

## [0.7.3] - 2026-04-08

### Changed
- **Single-file mmap column storage**: Column stores written to a single `columns.bin` file with mmap-backed reads. Replaces per-type `columns/<type>/columns.zst` layout. Near-instant load via mmap (no decompression).
- **Property log (disk mode)**: Phase 1 serializes properties to a zstd-compressed log file instead of building ColumnStores in-memory. Phase 1b replays the log to build columns in bulk — avoids O(n²) column rebuilds.
- **Partitioned CSR build**: Default CSR algorithm switched to hash-partitioned (Kuzu pattern). Merge-sort still available via `KGLITE_CSR_ALGO=merge_sort`.
- **File-backed pending edges**: `pending_edges` buffer uses mmap-backed `MmapOrVec` instead of heap `Vec`, avoiding ~14 GB heap allocation at Wikidata scale.
- **Auto-typing from P31**: N-Triples loader automatically derives node types from `P31` (instance-of) values, resolving Q-codes to labels. Entities without P31 default to "Entity".
- **Sparse property overflow**: Properties with <5% fill rate stored in a compact overflow bag instead of dense columns, reducing file size for wide schemas.

### Added
- `MmapColumnStore` — mmap-backed column reader for disk mode.
- `BuildColumnStore` — direct column writer that streams to the mmap file.
- `PropertyLogWriter`/`PropertyLogReader` — zstd-compressed property spill log for disk builds.
- `BlockPool`/`BlockColumn` — block-allocated typed column storage.
- `TypeBuildMeta` — per-type metadata for build-time column schema discovery.
- `MmapOrVec::load_mapped_region()`, `from_vec()`, `as_mut_bytes()` — new helpers for region-mapped and bulk byte access.
- `DiskGraph::update_row_id()` — fix per-type row_id mapping after column conversion.
- `ColumnStore::from_mmap_store()`, `from_raw_columns()` — constructors for mmap-backed and direct-built stores.
- `ColumnStore` id/title column accessors now work in disk mode.

### Fixed
- **code_tree stack overflow**: `extract_comment_annotations` switched from recursive to iterative traversal, fixing crashes on deeply nested ASTs.

## [0.7.2] - 2026-04-07

### Fixed
- **code_tree stack overflow**: `extract_comment_annotations` switched from recursive to iterative traversal, fixing crashes on deeply nested ASTs.

## [0.7.1] - 2026-04-06

### Changed
- **CSR build: external merge sort** (DuckDB-inspired). Replaced random-I/O scatter with external merge sort — sort chunks in memory, merge sequentially. All disk I/O is sequential. Phase 3 at Wikidata scale (862M edges, 16 GB RAM): ~16 min vs 90+ min previously.
- **Disk graph auto-persistence**: CSR arrays and metadata written directly to graph dir during build. No separate `save()` step needed. Mutations (`add_node`, `add_edge`, etc.) auto-flush metadata.
- **Disk graph raw storage**: Save/load uses raw `.bin` files (direct mmap) instead of zstd compression. Load is near-instant (mmap, no decompression). Legacy `.bin.zst` files still supported for loading.
- **Mmap-backed edge buffer**: N-Triples loader streams edges to mmap during Phase 1 (0 heap for edge buffer). Eliminates 13.8 GB heap allocation at Wikidata scale.

### Fixed
- **Memory leak in N-Triples loader**: `edge_buffer` (13.8 GB at Wikidata scale) was kept alive during Phase 3 CSR build, doubling peak memory. Now dropped immediately after Phase 2.
- **Disk thrashing during CSR build**: Random writes to mmap caused SSD thrashing. All writes are now sequential.
- **Temp file cleanup**: CSR build temp files cleaned up immediately after merge. Drop impl flushes metadata as safety net.

## [0.7.0] - 2026-04-05

### Added
- **Disk storage mode**: `KnowledgeGraph(storage="disk", path="./my_graph")` — fully disk-backed graph for very large datasets (100M+ nodes, 1B+ edges). Data lives on disk via mmap, using ~10% of equivalent in-memory RAM. The directory IS the graph — no separate save step needed.
- **GraphBackend abstraction**: Unified API across InMemory (petgraph), Mapped, and Disk backends. All Cypher queries, fluent API, and graph algorithms work identically across all three storage modes.
- **CSR edge storage**: Disk mode uses cache-friendly Compressed Sparse Row format. 3-4x faster than default on WHERE filters, SELECT, and SET operations at 100k scale.
- **zstd N-Triples support**: `load_ntriples()` now accepts `.nt.zst` files — 30x faster decompression than bz2.
- **`enable_disk_mode()`** method: Convert existing in-memory graph to disk-backed CSR.
- **`path` parameter** on constructor: Required for `storage="disk"`.

### Changed
- **Mapped mode**: Fixed O(n²) Arc clone bug — 50-300x faster `add_nodes` in mapped mode.
- **N-Triples loader**: 81x faster via bulk columnar conversion, pipeline parallelism, zero-copy parsing, byte-level filtering, and dense Vec edge lookup.

### Fixed
- Schema extension bug in mapped mode incremental `add_nodes`.
- `add_connections()` in disk mode auto-builds CSR so queries work immediately.

## [0.6.18] - 2026-03-30

### Fixed
- **Cypher LIMIT**: 16x faster multi-hop traversals with LIMIT. `MATCH (a)-[:R]->(b)-[:R]->(c) RETURN ... LIMIT 20` now pushes the limit into the pattern matcher — early termination at the last hop, overcommit budgets at intermediate hops. Benchmarks show parity with Neo4j on 2-hop queries.

## [0.6.17] - 2026-03-30

### Added
- `kglite.to_neo4j(graph, uri, ...)` — push graph data directly to a Neo4j database using batched UNWIND operations. Supports `clear`/`merge` modes, selection export, and verbose progress. Requires the `neo4j` package (`pip install neo4j` or `pip install kglite[neo4j]`).
- **ResultView**: Polars-style table display — `repr()` and `print()` now show a bordered table with column headers. Large results show first 10 + last 5 rows with `…` separator.
- **ResultView**: Improved `help(ResultView)` with quick-reference cheat sheet and examples on all methods.

### Fixed
- **code_tree**: Parse output (`Found N files`) now respects `verbose=False` — silent by default.

## [0.6.16] - 2026-03-30

### Changed
- **ResultView**: Polars-style table display — `repr()` and `print()` now show a bordered table with column headers instead of `ResultView(N rows, columns=[...])`. Large results show first 10 + last 5 rows with `…` separator.
- **ResultView**: Improved `help(ResultView)` with quick-reference cheat sheet and examples on all methods.
- **code_tree**: Parse output (`Found N files`) now respects `verbose=False` — silent by default.

## [0.6.15] - 2026-03-30

### Added
- `kglite.repo_tree(repo)` / `code_tree.repo_tree(repo)` — clone a GitHub repository and build a knowledge graph in one call. Cloned files are cleaned up by default; pass `clone_to=` to keep them locally. Supports private repos via `token=` or `GITHUB_TOKEN` env var.

### Fixed
- **code_tree**: Auto-create stub nodes for external base classes, enums, and traits referenced in EXTENDS, IMPLEMENTS, and HAS_METHOD edges — eliminates all "rows skipped: node not found" warnings during graph building.

## [0.6.12] - 2026-03-30

### Fixed
- **BUG-21**: Window functions (`row_number`, `rank`, `dense_rank`) crash with "Window function must appear in RETURN/WITH clause" when query has `WITH` aggregation + `ORDER BY` + `LIMIT`. The planner's `fuse_order_by_top_k` optimization now skips fusion when RETURN contains window functions.

### Changed
- Extracted window function execution into `window.rs` module (~240 lines out of executor.rs)
- Moved `is_aggregate_expression` / `is_window_expression` from executor.rs to ast.rs for cross-module reuse

## [0.6.11] - 2026-03-29

### Fixed

- **19 Cypher engine bugs resolved** — systematic fix of all bugs discovered via legal knowledge graph testing (BUG-01 through BUG-20, except BUG-04 which requires large-graph validation).

#### Critical — Silent wrong results
- **BUG-01**: Equality filter + GROUP BY no longer returns empty results. WHERE clause is now preserved after predicate pushdown to guarantee correctness when fusion fails.
- **BUG-02**: ORDER BY + LIMIT preserves integer types. `count()`, `size()`, `sum()` on integers no longer convert to float through the top-K heap path.
- **BUG-03**: HAVING clause is now propagated when the planner converts RETURN to WITH in fused optional-match aggregation.
- **BUG-05**: `RETURN *` expands to all bound variables (nodes, edges, paths, projected) instead of returning `{'*': 1}`.
- **BUG-06**: Path variable on explicit multi-hop patterns (`p = (a)-[]->(b)-[]->(c)`) now captures all intermediate nodes and relationships. `length(p)`, `nodes(p)`, `relationships(p)` return correct results.
- **BUG-17**: `MATCH (n) WHERE n.type = 'X'` on unlabeled nodes now works. Pattern matcher recognizes `type`/`node_type`/`label` as virtual properties.
- **BUG-18**: `labels()` returns consistent list format in both plain RETURN and GROUP BY contexts. Single-element list comparison (`labels(n) = 'Person'`) now works.

#### High — Errors on valid syntax
- **BUG-07**: `stDev()` / `stdev()` recognized as alias for `std()` aggregate function.
- **BUG-08**: `datetime('2024-03-15T10:30:00')` parses correctly instead of crashing on the time portion.
- **BUG-09**: `date()` returns null on invalid input (`''`, `'2016-00-00'`, `'2016-13-01'`) instead of crashing.
- **BUG-10**: `date('...').year`, `.month`, `.day` property access on function results now works.
- **BUG-11**: `[:TYPE1|TYPE2|TYPE3]` pipe syntax for multiple relationship types in MATCH patterns.
- **BUG-12**: `XOR` logical operator implemented with correct precedence (between OR and AND).
- **BUG-13**: `%` modulo operator implemented for both integer and float operands.
- **BUG-14**: `head()` and `last()` list functions implemented.
- **BUG-15**: `IN` operator accepts variable references, parameters, and function results — not just literal `[...]` lists.

#### Medium — Less common patterns
- **BUG-16**: Boolean/comparison expressions (`STARTS WITH`, `CONTAINS`, `>`, `=~`, etc.) work in RETURN/WITH clauses, evaluating to boolean values.
- **BUG-19**: `null = null` and `null <> null` return null (Cypher three-valued logic) instead of syntax error.
- **BUG-20**: Map all-properties projection `n {.*}` supported.

### Added

- **`Expression::PredicateExpr`** — AST variant bridging the expression/predicate boundary, enabling boolean predicates in RETURN/WITH items.
- **`Expression::ExprPropertyAccess`** — property access on arbitrary expression results (e.g. `date().year`).
- **`Expression::Modulo`** — modulo arithmetic operator.
- **`Predicate::Xor`** — exclusive-or logical operator.
- **`Predicate::InExpression`** — IN with runtime-evaluated list expressions.
- **`MapProjectionItem::AllProperties`** — wildcard map projection.
- **`EdgePattern.connection_types`** — multi-type edge matching for pipe syntax.
- **Performance benchmark suite** (`bench/benchmark_bugs.py`) — 70 targeted benchmarks covering all affected code paths, with CSV output for version-to-version comparison.

## [0.6.10] - 2026-03-29

### Fixed

- **Multi-MATCH empty propagation** — when the first MATCH in a multi-MATCH query returns 0 rows, subsequent MATCH/OPTIONAL MATCH clauses now correctly return 0 rows instead of matching against the entire graph.
- **Planner fusion guard** — MATCH fusion optimizations (FusedNodeScanAggregate, FusedMatchReturnAggregate, FusedMatchWithAggregate) are now restricted to first-clause position, preventing incorrect results when fused clauses ignored pipeline state from prior clauses.

### Changed

- **Retired legacy pytest/ test suite** — migrated unique test coverage (edge cases, subgraph extraction, pattern matching property filters, connection aggregation, connector API) into the official tests/ suite. Test count grew from 1,573 to 1,609.

## [0.6.9] - 2026-03-22

### Added

- **`'poincare'` distance metric** — new metric for `vector_search()`, `text_score()`, `compare()`, and `search_text()`. Computes hyperbolic distance in the Poincaré ball model, ideal for hierarchical data (taxonomies, ontologies). Based on Nickel & Kiela (2017).
- **`embedding_norm()` Cypher function** — returns the L2 norm of a node's embedding vector. In Poincaré embeddings, norm encodes hierarchy depth (0 = root/general, ~1 = leaf/specific).
- **Stored metric on embeddings** — `set_embeddings(..., metric='poincare')` stores the intended distance metric alongside vectors. Queries default to the stored metric when no explicit `metric=` is passed.

## [0.6.8] - 2026-03-19

### Added

- **`compare()` method** — dedicated API for spatial, semantic, and clustering operations. Replaces the overloaded `traverse(..., method=...)` pattern with a clearer `compare(target_type, method)` signature.
- **`collect_grouped()` method** — materialise nodes grouped by parent type as a dict. `collect()` now always returns a flat `ResultView`.
- **`Agg` helper class** — discoverable aggregation expression builders for `add_properties()`: `Agg.count()`, `Agg.sum(prop)`, `Agg.mean(prop)`, `Agg.min(prop)`, `Agg.max(prop)`, `Agg.std(prop)`, `Agg.collect(prop)`.
- **`Spatial` helper class** — spatial compute expression builders for `add_properties()`: `Spatial.distance()`, `Spatial.area()`, `Spatial.perimeter()`, `Spatial.centroid_lat()`, `Spatial.centroid_lon()`.
- **Traversal hierarchy guide** — new conceptual documentation explaining levels, property enrichment, and grouped collection.

### Breaking

- **`traverse()` no longer accepts `method=`** — use `compare(target_type, method)` instead.
- **`collect()` no longer accepts `parent_type`, `parent_info`, `flatten_single_parent`, or `indices`** — use `collect_grouped(group_by)` for grouped output. `collect()` always returns `ResultView`.

## [0.6.7] - 2026-03-18

### Performance

- **31% faster `.kgl` load** — large files are now memory-mapped directly instead of buffered read; small columns (< 256 KB) skip temp file creation and load into heap.
- **28% faster Cypher queries** — `PropertyStorage::get_value()` returns `Value` directly, avoiding `Cow` wrapping/unwrapping overhead on every property access.
- **Zero-alloc string column access** — `TypedColumn::get_str()` returns `&str` slices into mmap'd data without heap allocation, benefiting all WHERE string comparisons.
- **23% faster save** — reduced overhead from mmap threshold optimizations.

## [0.6.6] - 2026-03-18

### Breaking

- **`.kgl` format upgraded to v3** — files saved with older versions (v1/v2) cannot be loaded; rebuild the graph from source data and re-save.
- **`save_mmap()` and `kglite.load_mmap()` removed** — the v3 `.kgl` format replaces the mmap directory format with a single shareable file that supports larger-than-RAM loading.
- `save()` now leaves the graph in columnar mode after saving (previously restored non-columnar state). This avoids an expensive O(N×P) disable step.

### Added

- **v3 unified columnar file format** — `save()` now writes a single `.kgl` file with separated topology and per-type columnar sections (zstd-compressed). On load, column sections are decompressed to temp files and memory-mapped, keeping peak memory to topology + one type's data at a time.
- `save()` automatically enables columnar storage if not already active — no need to call `enable_columnar()` before saving.
- Loaded v3 files are always columnar (`is_columnar` returns `True`).

### Fixed

- **Temp directory leak** — `/tmp/kglite_v3_*` and `/tmp/kglite_spill_*` directories created during `load()` and `enable_columnar()` are now automatically cleaned up when the graph is dropped.
- Reduced save-side memory usage by eliminating double buffering in column packing.

### Removed

- `save_mmap(path)` method — use `save(path)` instead.
- `kglite.load_mmap(path)` function — use `kglite.load(path)` instead.
- v1 and v2 `.kgl` format support (load and save).
- Dead code: `StringInterner::len()`.

## [0.6.5] - 2026-03-18

### Added

- **Columnar property storage** — `enable_columnar()` / `disable_columnar()` convert node properties to per-type column stores, reducing memory usage for homogeneous typed columns (int64, float64, string, etc.). `is_columnar` property reports current storage mode.
- **Memory-mapped directory format** — `save_mmap(path)` / `kglite.load_mmap(path)` persist graphs as mmap-backed column files, enabling instant startup and out-of-core (larger-than-RAM) workloads. Directory layout: `manifest.json` + `topology.zst` + per-type column files.
- **Automatic memory-pressure spill** — `set_memory_limit(limit_bytes)` configures a heap-byte threshold; `enable_columnar()` automatically spills the largest column stores to disk when the limit is exceeded. `graph_info()` now reports `columnar_heap_bytes`, `columnar_is_mapped`, and `memory_limit`.
- **`unspill()`** — move mmap-backed columnar data back to heap memory (e.g., after deleting nodes to free space).
- **`memmap2` dependency** for memory-mapped file I/O.
- Columnar and mmap benchmarks in `test_bench_core.py` (5 new benchmarks).
- Comprehensive test suite for columnar storage and mmap format (28 new Python tests, 30+ new Rust tests).

### Fixed

- **`vacuum()` now rebuilds columnar stores** — previously, deleting nodes left orphaned rows in columnar storage that were never reclaimed. Now `vacuum()` (and auto-vacuum) automatically rebuilds column stores from only live nodes, eliminating the memory leak.
- `graph_info()` reports `columnar_total_rows` and `columnar_live_rows` for diagnosing columnar fragmentation.
- Boolean columns now correctly persist in mmap directory format (`from_type_str` now matches `"boolean"` in addition to `"bool"`).

### Performance

- 4-11x speedup for columnar/mmap operations: eliminated unnecessary full graph clone in `save_mmap()`, bulk memcpy in `materialize_to_heap()`, async flush, aligned pointer reads, direct push in `push_row()`, and skipped UTF-8 re-validation for string columns.

## [0.6.1] - 2026-03-08

### Changed

- **`describe()` default output now shows edge property names** in the `<connections>` section, improving agent discoverability of edge data without requiring `describe(connections=True)`.
- Improved hint text in describe output to guide agents toward `describe(connections=['CONN_TYPE'])` for edge property stats.
- `write_connections_overview` now reuses pre-computed metadata instead of scanning all edges (performance improvement).

## [0.6.0] - 2026-03-07

### Added

- **Python linting (ruff)** — format + lint enforcement for all Python files. `make lint` now checks both Rust and Python. `make fmt-py` auto-fixes.
- **Coverage reporting** — pytest-cov + Codecov integration in CI (informational, not blocking). `make cov` for local reports.
- **Stubtest** — `mypy.stubtest` verifies `.pyi` stubs match the compiled Rust extension. Runs in CI (py3.12). `make stubtest` for local checks.
- **Property-based testing** — Hypothesis tests for graph invariants (node count, filter correctness, index transparency, Cypher-fluent parity, delete consistency, sort correctness, type roundtrip).
- **Historical benchmark tracking** — pytest-benchmark with `github-action-benchmark` for performance regression detection. `make bench-save` / `make bench-compare` for local use.
- **Diátaxis documentation** — restructured docs into Tutorials, How-to Guides, Explanation, and Reference sections. New architecture and design-decisions explanation pages.
- **GitHub scaffolding** — issue templates (YAML forms), PR template, dependabot, security policy, `.editorconfig`, `.codecov.yml`.
- **PEP 561 `py.typed` marker** — type checkers now recognize KGLite's type stubs.
- **`connection_types` parameter** on `betweenness_centrality()`, `pagerank()`, `degree_centrality()` (stub fix — parameter existed at runtime).
- **`titles_only` parameter** on `connected_components()` (stub fix).
- **`timeout_ms` parameter** on `cypher()` (stub fix).

### Changed

- **Tree-sitter is now an optional dependency** — `pip install kglite[code-tree]` for codebase parsing. Core install reduced to just `pandas`.
- **README rewritten** as a keyword-optimized landing page for discoverability.
- Benchmarks CI job now runs on every push to main (was manual dispatch only).

## [0.5.88] - 2026-03-04

### Added

- **MCP Servers guide** — new docs page covering server setup, core tools, FORMAT CSV export, security, semantic search, and a minimal template

## [0.5.87] - 2026-03-04

### Added

- **`FORMAT CSV` Cypher clause** — append `FORMAT CSV` to any query to get results as a CSV string instead of a ResultView. Good for large data transfers and token-efficient output in MCP servers.

## [0.5.86] - 2026-03-03

### Added

- **`add_connections` query mode** — `add_connections(None, ..., query='MATCH ... RETURN ...')` creates edges from Cypher query results instead of a DataFrame. `extra_properties=` stamps static properties onto every edge.
- **`'sum'` conflict handling mode** — `conflict_handling='sum'` adds numeric edge properties on conflict (Int64+Int64, Float64+Float64, mixed promotes to Float64). Non-numeric properties overwrite like `'update'`. For nodes, `'sum'` behaves identically to `'update'`.

### Fixed

- **`add_connections` query-mode param validation** — `columns`, `skip_columns`, and `column_types` now raise `ValueError` in query mode (previously silently ignored)
- **`describe()` incomplete `add_connections` signature** — now shows `query`, `extra_properties`, `conflict_handling` params and query-mode example

## [0.5.84] - 2026-03-03

### Fixed

- **Cypher edge traversal without ORDER BY** — queries like `MATCH (a)-[r:REL]->(b) RETURN ... LIMIT N` returned wrong row count, NULL target/edge properties, and ignored LIMIT. Root cause: `push_limit_into_match` pushed LIMIT into the pattern executor for edge patterns, causing early termination before edge expansion. Now only pushes for node-only patterns.
- **`create_connections()` silently creating 0 edges** — two sub-bugs: (1) `ConnectionBatchProcessor.flush_chunk` used `find_edge()` which matches ANY edge type, so creating PERSON_AT edges would update existing WORKS_AT edges instead. Now uses type-aware `edges_connecting` lookup. (2) Parent map in `maintain_graph::create_connections` used `HashMap<NodeIndex, NodeIndex>` (single parent per child), losing multi-parent relationships. Now uses `Vec<NodeIndex>` per child and iterates group parents directly.
- **`describe(fluent=['loading'])` wrong parameter name** — documented `properties=` for `add_connections()`, actual parameter is `columns=`
- **`traverse()` with `method='contains'` ignoring `target_type=`** — when spatial method was specified, `target_type=` keyword was ignored and only the first positional arg was used as target type. Now prefers explicit `target_type=` over positional arg.
- **`geometry_contains_geometry` missing combinations** — added `(MultiPolygon, LineString)` and `(MultiPolygon, MultiPolygon)` match arms that previously fell through to `false`

## [0.5.83] - 2026-03-03

### Added

- **`fold_or_to_in` optimizer pass** — folds `WHERE n.x = 'A' OR n.x = 'B' OR n.x = 'C'` into `WHERE n.x IN ['A', 'B', 'C']` for pushdown and index acceleration
- **`InLiteralSet` AST node** — pre-evaluated literal IN with HashSet for O(1) membership testing instead of per-row list evaluation
- **TypeSchema-based fast property key discovery** — `to_df()`, `ResultView`, and `describe()` use TypeSchema for O(1) key lookup when all nodes share a type (>50 nodes)
- **Sampled property stats** — `describe()` and `properties()` sample large types (>1000 nodes) for faster response
- **`StringInterner::try_resolve()`** — fallible key resolution for TypeSchema-based paths
- **`rebuild_type_indices_and_compact` metadata fallback** — scans nodes to build TypeSchemas when metadata is empty (loaded from file)

### Fixed

- **FusedMatchReturnAggregate output columns** — built from return clause items instead of reusing pre-existing columns, fixing wrong column names in fused aggregation results
- **FusedMatchReturnAggregate top-k sort order** — removed erroneous `top.reverse()` calls that inverted DESC/ASC order in `ORDER BY ... LIMIT` queries
- **FusedMatchReturnAggregate zero-count rows** — exclude nodes with zero matching edges (MATCH semantics require at least one match)
- **Path binding variable lookup** — path assignments now find the correct variable-length edge variable instead of grabbing the first available binding
- **UNWIND null produces zero rows** — `UNWIND null AS x` now correctly produces no rows per Cypher spec instead of emitting a null row
- **InLiteralSet cross-type equality** — `WHERE n.id IN [1, 2, 3]` now matches float values via `values_equal` fallback
- **NULL = NULL returns false in WHERE** — implements Cypher three-valued logic where NULL comparisons are falsy; grouping/DISTINCT unaffected
- **Property push-down no longer overwrites** — `apply_property_to_patterns` uses `entry().or_insert()` to preserve earlier matchers
- **Pattern reversal skips path assignments** — `optimize_pattern_start_node` no longer reverses patterns bound to path variables
- **Fuse guard: HAVING clause** — `fuse_match_return_aggregate` bails out when HAVING is present
- **Fuse guard: vector score aggregation** — `fuse_vector_score_order_limit` bails out when return items contain aggregate functions
- **Fuse guard: bidirectional edge count** — `fuse_count_short_circuits` skips undirected patterns that could produce wrong counts
- **Fuse guard: dead SKIP check removed** — `fuse_order_by_top_k` no longer checks wrong clause index for SKIP
- **Parallel expansion error propagation** — errors in rayon parallel edge expansion are now propagated instead of silently returning empty results
- **Variable-length paths min_hops=0** — source node is now yielded at depth 0 when `min_hops=0` (e.g., `[*0..2]`)
- **Parallel distinct target dedup** — parallel expansion path now applies `distinct_target_var` deduplication matching the serial path
- **Unterminated string/backtick detection** — tokenizer now returns errors for unclosed string literals and backtick identifiers
- **String reconstruction preserves escapes** — `CypherToken::StringLit` re-escapes quotes and backslashes during reconstruction

## [0.5.82] - 2026-03-03

### Changed

- **zstd compression for save/load** — replaced gzip level 3 with zstd level 1; Save 9.5s → 1.1s (8.6×), Load 2.3s → 1.0s (2.2×), file 7% smaller. Backward-compatible: old gzip files load transparently
- **Vectorized pandas series extraction** — `convert_pandas_series()` uses `series.tolist()` + `PyList.get_item()` instead of per-cell `Series.get_item()`, plus batch `extract::<Vec<Option<T>>>()` for Float64/Boolean/String. Build 24.7s → 19.3s
- **Fast lookup constructors** — `TypeLookup::from_id_indices()` and `CombinedTypeLookup::from_id_indices()` reuse pre-built `DirGraph.id_indices` instead of scanning all nodes
- **Skip edge existence check on initial load** — `ConnectionBatchProcessor.skip_existence_check` flag bypasses `find_edge()` when no edges of that type exist yet
- **Pre-interned property keys** — intern column name strings once before the row loop, use `Vec<(InternedKey, Value)>` instead of per-row `HashMap<String, Value>` for node creation
- **Single-pass load finalize** — `rebuild_type_indices_and_compact()` combines type index rebuild + Map→Compact property conversion in one pass, with TypeSchemas built from metadata instead of scanning nodes
- **Zero-alloc InternedKey deserialization** — custom serde Visitor hashes borrowed `&str` from the decompressed buffer, eliminating ~5.6M String allocations per load
- **Remove unnecessary `.copy()` on first CSV read** in blueprint loader

## [0.5.81] - 2026-03-02

### Added

- **Comparison pushdown into MATCH** — `WHERE n.prop > val` (and `>=`, `<`, `<=`) is now pushed from WHERE into MATCH patterns, filtering during node scan instead of post-expansion. Includes range merging (`year >= 2015 AND year <= 2022` → single Range matcher). Benchmark: filtered 2-hop query 109ms → 14ms (7.6×), property filter 2.5ms → 0.8ms (3×)
- **Range index acceleration** — pushed comparisons now use `create_range_index()` B-Tree indexes via `lookup_range()` for O(log N + k) scans instead of O(N) type scans
- **Reverse fused aggregation** — `MATCH (:A)-[:REL]->(b:B) RETURN b.prop, count(*)` (group by target node) now fuses into a single pass like source-node grouping. In-degree benchmark: 26ms → 9ms (2.8×)
- **EXISTS/NOT EXISTS fast path** — direct edge-existence check for simple EXISTS patterns instead of instantiating a full PatternExecutor per row. NOT EXISTS: 2372ms → 0.3ms (7400×)
- **FusedMatchReturnAggregate top-k** — BinaryHeap-based top-k selection during edge counting, avoiding full materialization + sort. In-degree top-20: 10.5ms → 5.0ms
- **FusedOrderByTopK external sort expression** — ORDER BY on expressions not in RETURN items now fuses into the top-k heap, projecting only surviving rows. UNION per-arm: 5.4ms → 2.3ms
- **FusedNodeScanAggregate** — single-pass node scan with inline accumulators (count/sum/avg/min/max) for `MATCH (n:Type) RETURN group_keys, aggs(...)`, avoiding intermediate ResultRow creation
- **FusedMatchWithAggregate** — fuse `MATCH...WITH count()` into single pass (same as MATCH+RETURN fusion but for pipeline continuation)
- **DISTINCT push-down into MATCH** — when `RETURN DISTINCT` references a single node variable, pre-deduplicate by NodeIndex during pattern matching. Includes intermediate-hop dedup for anonymous nodes. Filtered 2-hop DISTINCT: 15ms → 10ms
- **UNION hash-based dedup** — replace `HashSet<Vec<Value>>` with hash-of-values approach for UNION (non-ALL) deduplication
- **35-query DuckDB/SQLite comparison benchmark** (`bench_graph_traversal.py`)

## [0.5.80] - 2026-03-02

### Added

- **`closeness_centrality(sample_size=…)`** — stride-based node sampling for closeness centrality, matching the existing betweenness pattern; reduces O(N²) to O(k×(N+E)) for approximate results on large graphs
- **`copy()` / `__copy__` / `__deepcopy__`** — deep-copy a `KnowledgeGraph` in memory without disk I/O, useful for running mutations on an independent copy

### Changed

- **`compute_property_stats` value-set cap** — stop cloning values into the uniqueness `HashSet` once `max_values+1` entries are collected, avoiding O(N) clones for high-cardinality properties
- **Closeness centrality Cypher `CALL`** — `CALL closeness({sample_size: 100})` now supported alongside `normalized` and `connection_types`
- **Regex cache in fluent filtering** — pre-compile `Regex` patterns before filter loops (was compiling per-node); `fluent_where_regex` 302 ms → 1 ms
- **Single-pass property stats** — replaced O(N×P) two-pass scan with O(N×avg_props) single-pass accumulator
- **Pre-computed neighbor schemas** — `describe()` scans all edges once instead of per-type

## [0.5.79] - 2026-03-02

### Added

- **Window functions** — `row_number()`, `rank()`, `dense_rank()` with `OVER (PARTITION BY ... ORDER BY ...)` syntax for ranking within result partitions
- **HAVING clause** — post-aggregation filtering on RETURN and WITH (`RETURN n.type, count(*) AS cnt HAVING cnt > 5`)
- **Date arithmetic** — DateTime ± Int64 (add/subtract days), DateTime − DateTime (days between), `date_diff()` function
- **Window function performance** — pre-computed column names, constant folding, OVER spec deduplication, rayon parallelism, fast path for unpartitioned windows

## [0.5.78] - 2026-03-02

### Changed

- **Betweenness BFS inner loop** — merged redundant `dist[w_idx]` loads into cached `if/else if` branch, eliminating a second memory access per edge in both parallel and sequential paths
- **Pre-intern connection types in algorithms** — betweenness, pagerank, degree, closeness, louvain, and label propagation now pre-intern connection type filters once per call instead of hashing per-edge
- **Adjacency list dedup** — undirected adjacency lists are now sorted and deduplicated to prevent double-counting from bidirectional edges (A→B + B→A)
- **3-way traversal benchmark** — added DuckDB (columnar/vectorized) alongside SQLite and KGLite with optimized batch queries

## [0.5.77] - 2026-03-02

### Changed

- **Edge data optimization** — `EdgeData.connection_type` changed from `String` (24 bytes) to `InternedKey` (8 bytes), reducing per-edge overhead by 16 bytes
- **Edge properties compacted** — `EdgeData.properties` changed from `HashMap<InternedKey, Value>` (48 bytes) to `Vec<(InternedKey, Value)>` (24 bytes), saving 24 bytes per edge
- **BFS connection type comparison** — pre-intern connection type before edge loops for `u64 == u64` comparison instead of string equality
- **Static slice in BFS** — `expand_from_node` changed `vec![Direction]` heap allocation to `&[Direction]` static slice
- **Save/load performance** — save time -70% (2,253 → 682 ms), load time -93% (1,676 → 119 ms) on 50k node / 150k edge benchmark
- **Deep traversal speedup** — 8-20 hop citation queries 16-28% faster from interned comparison and eliminated heap allocations

## [0.5.76] - 2026-03-01

### Changed

- **BFS traversal optimization** — replaced `HashSet` visited set with `Vec<bool>` for cache-friendly O(1) lookups during variable-length path expansion
- **Skip redundant node type checks** — planner now marks edges where the connection type guarantees the target node type, avoiding unnecessary `node_weight()` loads during BFS
- **Skip edge data cloning** — unnamed edge variables no longer clone `connection_type` and `properties`, eliminating thousands of heap allocations per traversal
- **DISTINCT dedup optimization** — uses `Value` hash keys instead of `format_value_compact()` string allocation per row

### Added

- **Graph traversal benchmark suite** — SQLite recursive CTE vs KGLite across 15 query types (citation chains, shortest path, reachability, triangles, neighborhood aggregation)

## [0.5.75] - 2026-03-01

### Added

- **`keys()` function** — `keys(n)` / `keys(r)` returns property names of nodes and relationships as a JSON list
- **Math functions** — `log`/`ln`, `log10`, `exp`, `pow`/`power`, `pi`, `rand`/`random` (previously documented but not implemented)
- **`datetime()` alias** — `datetime('2020-01-15')` works identically to `date()`
- **DateTime property accessors** — `d.year`, `d.month`, `d.day` on DateTime values (via WITH alias)
- **Scientific notation** — tokenizer now parses `1e6`, `1.5e-3`, `2E+10` as float literals

### Fixed

- **String function auto-coercion** — `substring`, `left`, `right`, `split`, `replace`, `trim`, `reverse` now auto-coerce DateTime/numeric/boolean values to strings instead of returning NULL
- **`describe()` algorithm hint** — fixed misleading `YIELD node, score|community|cluster` that didn't mention `component`; now shows which yield name belongs to which procedure
- **Spatial coordinate order note** — added documentation clarifying WKT uses (longitude latitude) while `point()` uses (latitude, longitude)

## [0.5.74] - 2026-03-01

### Added

- **Multi-hop traversal benchmarks** — scale-free graph benchmarks at 1K/10K/50K/100K nodes with hop depths 1–8, comparable to TuringDB/Neo4j multi-hop benchmarks
- **Blueprint documentation** — standalone guide page with step-by-step walkthrough, real CSV examples, and troubleshooting

### Changed

- **Variable-length path BFS** — global dedup mode skips path tracking when path info isn't needed (no `p = ...` assignment, no named edge variable), reducing memory and redundant exploration (~4x faster)
- **WHERE IN predicate pushdown** — `WHERE n.id IN [list]` is now pushed into the MATCH pattern and resolved via id-index O(1) lookups instead of post-filtering all nodes (~1,400x faster on 10K 8-hop traversals)

## [0.5.73] - 2026-02-27

### Changed

- **README** — added blueprint loading and code review examples to Quick Start, doc links on each section
- **CLAUDE.md** — simplified and consolidated conventions

## [0.5.72] - 2026-02-27

### Added

- **Documentation site** — Sphinx + Furo docs with auto-generated API reference from `.pyi` stubs, hosted on Read the Docs. Guide pages for Cypher, data loading, querying, semantic search, spatial, timeseries, graph algorithms, import/export, AI agents, and code tree.

## [0.5.71] - 2026-02-27

### Added

- **`traverse()` API improvements:**
  - `target_type` parameter — filter targets to specific node type(s): `traverse('OF_FIELD', direction='incoming', target_type='ProductionProfile')` or `target_type=['ProductionProfile', 'FieldReserves']`
  - `where` parameter — alias for `filter_target`, consistent with the fluent API: `traverse('HAS_LICENSEE', where={'title': 'Equinor'})`
  - `where_connection` parameter — alias for `filter_connection`: `traverse('RATED', where_connection={'score': {'>': 4}})`
  - `help(g.traverse)` now shows a comprehensive docstring with args, examples, and usage patterns
- **Temporal awareness** — first-class support for time-dependent nodes and connections:
  - Declare temporal columns via `column_types={"fldLicenseeFrom": "validFrom", "fldLicenseeTo": "validTo"}` on `add_nodes()` or `add_connections()` — auto-configures temporal filtering behind the scenes (same pattern as spatial `"geometry"` / `"location.lat"`)
  - `date("2013")` sets a temporal context for the entire chain — all subsequent `select()` and `traverse()` calls filter to that date instead of today
  - `date("2010", "2015")` — range mode: include everything valid at any point during the period (overlap check)
  - `date("all")` — disable temporal filtering entirely (show all records regardless of validity dates)
  - `select()` auto-filters temporal nodes to "currently valid" (or the `date()` context). Pass `temporal=False` to include all historic records
  - `traverse()` auto-filters temporal connections to "currently valid". Override with `at="2015"`, `during=("2010", "2020")`, or `temporal=False`
  - `valid_at()` / `valid_during()` auto-detect field names from temporal config; NULL `date_to` treated as "still active"
  - Display (`sample()`, `collect()`) filters connection summaries to temporally valid edges
  - `describe()` includes `temporal_from`/`temporal_to` attributes on configured types and connections
  - Blueprint loader: use `"validFrom"` / `"validTo"` property types to auto-configure temporal filtering
  - `set_temporal(type_name, valid_from, valid_to)` available as low-level API for manual configuration
  - Temporal configs persist through `save()`/`load()` round-trips
- **`show(columns, limit=200)`** — compact display of selected nodes with chosen properties. Single-level shows `Type(val1, val2)` per line; after `traverse()` walks the full chain as `Type1(vals) -> Type2(vals) -> Type3(vals)`. Resolves field aliases and truncates long values

## [0.5.70] - 2026-02-26

### Added

- **`to_str(limit=50)`** — format current selection as a human-readable string with `[Type] title (id: x)` headers and indented properties
- **`print(ResultView)` smart formatting** — `ResultView.__str__` uses multiline card format (properties + connection arrows) for ≤3 rows, compact one-liner for >3. Connections show direction with `◆` as the current node: `◆ --WORKS_AT--> Company(id, title)` for outgoing, `Person(id, title) --WORKS_AT--> ◆` for incoming. Long values (WKT geometries, etc.) are truncated with middle ellipsis
- **`sample()` selection-aware** — `sample()` now works on the current selection (`graph.select('Person').sample(3)`) in addition to the existing `sample('Person', 3)` form
- **`head()`/`tail()` preserve connections** — slicing a ResultView carries connection summaries through

## [0.5.67] - 2026-02-26

### Changed

- **BREAKING: Fluent API method renames** — modernized the fluent API surface to match common query DSL conventions:
  - `type_filter()` → `select()`
  - `filter()` → `where()`
  - `filter_any()` → `where_any()`
  - `filter_orphans()` → `where_orphans()`
  - `has_connection()` → `where_connected()`
  - `max_nodes()` → `limit()`
  - `get_nodes()` → `collect()`
  - `node_count()` → `len()` (also adds `__len__` for `len(graph)`)
  - `id_values()` → `ids()`
  - `max_nodes=` parameter → `limit=` everywhere (select, where, traverse, collect, etc.)
- **BREAKING: Retrieval method renames** — dropped inconsistent `get_` prefix and shortened verbose methods:
  - `get_titles()` → `titles()`
  - `get_connections()` → `connections()`
  - `get_degrees()` → `degrees()`
  - `get_bounds()` → `bounds()`
  - `get_centroid()` → `centroid()`
  - `get_selection()` → `selection()`
  - `get_schema()` → `schema_text()`
  - `get_schema_definition()` → `schema_definition()`
  - `get_last_report()` → `last_report()`
  - `get_operation_index()` → `operation_index()`
  - `get_report_history()` → `report_history()`
  - `get_spatial()` → `spatial()`
  - `get_timeseries()` → `timeseries()`
  - `get_time_index()` → `time_index()`
  - `get_timeseries_config()` → `timeseries_config()`
  - `get_embeddings()` → `embeddings()`
  - `get_embedding()` → `embedding()`
  - `get_node_by_id()` → `node()`
  - `children_properties_to_list()` → `collect_children()` (also `filter=` param → `where=`)

### Removed

- **`get_ids()`** — removed; use `ids()` for flat ID list or `collect()` for full node dicts

## [0.5.66] - 2026-02-26

### Changed

- **Blueprint loader output** — quiet by default (only warnings/errors + summary); verbose mode for per-type detail. Warnings from `add_connections` skips are now tracked in the loader instead of surfacing as raw `UserWarning`s
- **Blueprint settings** — `root` renamed to `input_root`, `output` split into `output_path` (optional directory) + `output_file` (filename or relative path with `../` support). Old keys still accepted for backwards compatibility

### Fixed

- **Float→Int ID coercion** — FK columns with nullable integers (read as float64 by pandas, e.g. `260.0`) are now auto-coerced to int before edge matching. The Rust lookup layer also gained Float64 → Int64/UniqueId fallback as a safety net
- **Timeseries FK edge filtering** — FK edges for timeseries node types now apply the same time-component filter as node creation (e.g. dropping month=0 aggregate rows), preventing "source node not found" warnings for carriers that only have aggregate data

## [0.5.65] - 2026-02-26

### Added

- **`FLUENT.md`** — comprehensive fluent API reference documenting all method-chaining operations: data loading, selection & filtering, spatial, temporal, timeseries, vector search, traversal, algorithms, set operations, indexes, transactions, export, and a fluent-vs-Cypher feature matrix
- **`create_connections()`** — renamed from `selection_to_new_connections` with new capabilities: `properties` dict copies node properties onto new edges (e.g. `properties={'B': ['score']}`), `source_type`/`target_type` override which traversal levels to connect (defaults to first→last level)
- **Comparison-based `traverse(method=...)`** — discover relationships without pre-existing edges. Five methods: `'contains'` (spatial containment), `'intersects'` (geometry overlap), `'distance'` (geodesic proximity), `'text_score'` (semantic similarity via embeddings), `'cluster'` (kmeans/dbscan grouping). `method` accepts a string shorthand (`method='contains'`) or a dict with settings (`method={'type': 'distance', 'max_m': 5000, 'resolve': 'centroid'}`). The `resolve` key controls polygon geometry interpretation: `'centroid'` (force geometry centroid), `'closest'` (nearest boundary point), `'geometry'` (full polygon shape). Produces the same selection hierarchy as edge-based traversal, so all downstream methods work unchanged
- **`add_properties()`** — enrich selected nodes with properties from ancestor nodes in the traversal chain. Supports copy (`['name']`), copy-all (`[]`), rename (`{'new': 'old'}`), aggregate expressions (`'count(*)'`, `'mean(depth)'`, `'sum(production)'`, `'min()'`, `'max()'`, `'std()'`, `'collect()'`), and spatial compute (`'distance'`, `'area'`, `'perimeter'`, `'centroid_lat'`, `'centroid_lon'`)

### Changed

- **`selection_to_new_connections` → `create_connections`** — renamed for brevity. Now defaults to connecting the top-level ancestor to leaf nodes (was parent→child at last level only)

## [0.5.64] - 2026-02-25

### Added

- **List quantifier predicates** — `any(x IN list WHERE pred)`, `all(...)`, `none(...)`, `single(...)` for filtering over lists in WHERE, RETURN, and WITH clauses
- **Exploration hints in `describe()`** — inventory views now surface disconnected node types and join candidates (property value overlaps between unconnected type pairs) to suggest enrichment opportunities
- **Temporal Cypher functions** — `valid_at(entity, date, 'from_field', 'to_field')` and `valid_during(entity, start, end, 'from_field', 'to_field')` for date-range filtering on both nodes and relationships in WHERE clauses. NULL fields treated as open-ended boundaries

### Changed

- **Rewritten examples** — new domain examples: `legal_graph.py` (index-based loading), `code_graph.py` (code tree parsing), `spatial_graph.py` (blueprint loading), `mcp_server.py` (generic MCP server with auto-detected code tools)

## [0.5.63] - 2026-02-25

### Added

- **`export_csv(path)`** — bulk export to organized CSV directory tree with one file per node type and connection type, sub-node nesting, full properties, and a `blueprint.json` for round-trip re-import via `from_blueprint()`
- **Variable binding in MATCH pattern properties** — bare variables from `WITH`/`UNWIND` can now be used in inline pattern properties: `WITH "Oslo" AS city MATCH (n:Person {city: city}) RETURN n`
- **Map literals in Cypher expressions** — `{key: expr, key2: expr}` syntax in `RETURN`/`WITH` for constructing map objects: `RETURN {name: n.name, age: n.age} AS m`
- **WHERE clause inside EXISTS subqueries** — `EXISTS { MATCH (n:Type) WHERE n.prop = expr }` now supports arbitrary WHERE predicates including cross-scope variable references and regex

### Changed

- **Cypher query performance** — eliminated `type_indices` Vec clone on every MATCH (iterate by reference), move-on-last-match optimization to reduce row cloning in joins, pre-allocated result vectors, eliminated unnecessary clone in composite index lookups
- **MERGE index acceleration** — MERGE now uses `id_indices`, `property_indices`, and `composite_indices` for O(1) pattern matching instead of linear scan through all nodes of a type. Orders-of-magnitude faster for batch `UNWIND + MERGE` workloads
- **UNWIND/MERGE clone reduction** — UNWIND moves (instead of cloning) the row for the last unwound item; MERGE iterates source rows by value to avoid per-row cloning

## [0.5.61] - 2026-02-24

### Added

- **PROFILE** prefix for Cypher queries — executes query and collects per-clause statistics (rows_in, rows_out, elapsed_us). Access via `result.profile`
- **Structured EXPLAIN** — `EXPLAIN` now returns a `ResultView` with columns `[step, operation, estimated_rows]` instead of a plain string. Cardinality estimates use type_indices counts
- **Read-only transactions** — `begin_read()` creates an O(1) Arc-backed snapshot (zero memory overhead). Mutations are rejected
- **Optimistic concurrency control** — `commit()` detects graph modifications since `begin()` and raises `RuntimeError` on conflict
- **Transaction timeout** — `begin(timeout_ms=...)` and `begin_read(timeout_ms=...)` set a deadline for all operations within the transaction
- `Transaction.is_read_only` property
- `describe(cypher=['EXPLAIN'])` and `describe(cypher=['PROFILE'])` topic detail pages
- Expanded `<limitations>` section in `describe(cypher=True)` with workarounds for unsupported features
- openCypher compatibility matrix in CYPHER.md

## [0.5.60] - 2026-02-24

### Added

- `describe(cypher=True)` tier 1 hint now highlights KGLite-specific features (||, =~, coalesce, CALL procedures, distance/contains)
- `describe(cypher=True)` tier 2 includes `<not_supported>` section and spatial functions group
- `describe()` overview connection map includes `count` attribute per connection type
- `describe()` connections hint only shown when graph has edges
- `describe(cypher=['spatial'])` topic with distance, contains, intersects, centroid, area, perimeter docs

## [0.5.59] - 2026-02-24

### Added

- `bug_report(query, result, expected, description)` — file Cypher bug reports to `reported_bugs.md`. Timestamped, version-tagged entries prepended to top of file. Input sanitised against HTML/code injection
- `KnowledgeGraph.explain_mcp()` — static method returning a self-contained XML quickstart for setting up a KGLite MCP server (server template, core/optional tools, Claude registration config)

### Fixed

- `collect(node)[0].property` now returns the actual property value instead of the node's title. Previously, `WITH f, collect(fr)[0] AS lr RETURN lr.oil` would return the node title for every property access. Node identity is now preserved through collect→index→WITH pipelines via internal `Value::NodeRef` references

## [0.5.58] - 2026-02-24

### Added

- `CALL cluster()` procedure — general-purpose clustering via Cypher. Supports DBSCAN and K-means methods. Reads nodes from preceding MATCH clause. Spatial mode auto-detects lat/lon from `set_spatial()` config with geometry centroid fallback; property mode clusters on explicit numeric properties with optional normalization. YIELD node, cluster (noise = -1 for DBSCAN)
- `round(x, decimals)` — optional second argument for decimal precision (e.g. `round(3.14159, 2)` → 3.14). Backward compatible: `round(x)` still rounds to integer
- `||` string concatenation operator — concatenates values in expressions (e.g. `n.first || ' ' || n.last`). Null propagates. Non-string values auto-converted
- `describe(cypher=True)` — 3-tier Cypher language reference: compact `<cypher hint/>` in overview (tier 1), full clause/operator/function/procedure listing with `cypher=True` (tier 2), detailed docs with params and examples via `cypher=['cluster','MATCH',...]` (tier 3)
- `describe(connections=True)` — connection type progressive disclosure: overview with `connections=True` (all types, counts, endpoints, property names), deep-dive with `connections=['BELONGS_TO']` (per-pair counts, property stats, sample edges)

## [0.5.56] - 2026-02-23

### Added

- `near_point_m()` — geodesic distance filter in meters (SI units), replaces `near_point_km()` and `near_point_km_from_wkt()`
- Geometry centroid fallback: fluent API spatial methods (`near_point_m`, `within_bounds`, `get_bounds`, `get_centroid`) now fall back to WKT geometry centroid when lat/lon fields are missing but a geometry is configured via `set_spatial` or `column_types`

### Changed

- Cypher `distance(a, b)` returns Null (instead of erroring) when a node has no spatial data, so `WHERE distance(a, b) < X` simply filters those nodes out
- Cypher comparison operators (`<`, `<=`, `>`, `>=`) now follow three-valued logic: comparisons involving Null evaluate to false (previously Null sorted as less-than-everything)

### Removed

- `near_point_km()` — use `near_point_m()` with meters instead (e.g. `max_distance_m=50_000.0` for 50 km)
- `near_point_km_from_wkt()` — subsumed by `near_point_m()` which auto-falls back to geometry centroid

## [0.5.55] - 2026-02-23

### Changed

- Cypher spatial functions now return SI units: `distance()` → meters, `area()` → m², `perimeter()` → meters (were km/km²). Distance uses WGS84 geodesic (Karney algorithm) instead of spherical haversine

### Removed

- `agent_describe()` — replaced by `describe()`. Migration: `graph.agent_describe()` → `graph.describe()`, `graph.agent_describe(detail='full')` → `graph.describe()` (auto-selects detail level)

## [0.5.54] - 2026-02-23

### Added

- `describe(types=None)` — progressive disclosure schema description for AI agents. Inventory mode returns node types grouped by size with property complexity markers and capability flags, connection map, and Cypher extensions. Focused mode (`types=['Field']`) returns detailed properties, connections, timeseries/spatial config, and sample nodes. Automatically inlines full detail for graphs with ≤15 types
- `set_parent_type(node_type, parent_type)` — declare a node type as a supporting child of a core type. Supporting types are hidden from the `describe()` inventory and appear in the `<supporting>` section when the parent is inspected. The `from_blueprint()` loader auto-sets parent types for sub-nodes
- Cypher math functions: `abs()`, `ceil()` / `ceiling()`, `floor()`, `round()`, `sqrt()`, `sign()` — work with Int64 and Float64 values, propagate Null
- String coercion on `+` operator: when one operand is a string, the other is automatically converted (e.g. `2024 + '-06'` → `'2024-06'`). Null still propagates

### Changed

- `describe()` inventory now uses compact descriptor format `TypeName[size,complexity,flags]` instead of size bands. Types listed as flat comma-separated list sorted by count descending. Core types with supporting children show `+N` suffix. Capability flags from supporting types bubble up to their parent descriptor
- `describe()` now shows a `<read-only>` notice listing unsupported Cypher write commands (CREATE, SET, DELETE, REMOVE, MERGE) when the graph is in read-only mode

## [0.5.53] - 2026-02-23

### Added

- `from_blueprint()` — build a complete KnowledgeGraph from a JSON blueprint and CSV files. Supports core nodes, sub-nodes, FK edges, junction edges, timeseries, geometry conversion, filters, manual nodes (from FK values), and auto-generated IDs
- Cypher `date()` function — converts date strings to DateTime values: `date('2020-01-15')`
- `property_types` on blueprint junction edges for automatic type conversion (e.g. epoch millis → DateTime)
- Temporal join support: `ts_*()` functions accept DateTime edge properties and null values as date range arguments
- Cypher `IS NULL` / `IS NOT NULL` now supported as expressions in RETURN/WITH (e.g. `RETURN x IS NULL AS flag`)
- `agent_describe(detail, include_fluent)` — optional detail level adapts output to graph complexity. Graphs with >15 types auto-select compact mode (~5-8x smaller output). Fluent API docs excluded by default (opt-in via `include_fluent=True`)

### Changed

- **Performance**: `agent_describe()` 27x faster (1.3s → 48ms) via property index fast path and scan capping
- **Performance**: `MATCH (n) RETURN count(n)` short-circuits to O(1) via `FusedCountAll` (was ~266ms, now sub-ms)
- **Performance**: `MATCH (n) RETURN n.type, count(n)` short-circuits to O(types) via `FusedCountByType` (was ~727ms, now sub-ms)
- **Performance**: `MATCH ()-[r]->() RETURN type(r), count(*)` short-circuits to O(E) single-pass via `FusedCountEdgesByType` (was ~822ms, now ~3ms)
- **Performance**: `MATCH (n:Type) RETURN count(n)` short-circuits to O(1) via `FusedCountTypedNode` (reads type index length directly)
- **Performance**: `MATCH ()-[r:Type]->() RETURN count(*)` short-circuits via `FusedCountTypedEdge` (single-pass edge filter)
- **Performance**: Edge type counts cached in DirGraph with lazy invalidation on mutations
- **Performance**: Multi-hop fused aggregation for 5-element patterns (e.g. `MATCH (a)-[]->(b)<-[]-(c) RETURN a.x, count(*)`) traverses without materializing intermediate rows
- **Performance**: Regex `=~` operator caches compiled patterns per query execution (compile once, match many)
- **Performance**: PageRank uses pull-based iteration with rayon parallelization for large graphs (3-4x speedup)
- **Performance**: Louvain community detection precomputes loop-invariant division terms
- Timeseries keys stored as `NaiveDate` instead of composite integer arrays (`Vec<Vec<i64>>`)
- `set_time_index()` now accepts date strings (`['2020-01', '2020-02']`) in addition to integer lists
- `get_time_index()` returns ISO date strings (`['2020-01-01', '2020-02-01']`) instead of integer lists
- `get_timeseries()` keys returned as ISO date strings
- `ts_series()` output uses ISO date strings for time keys (e.g. `"2020-01-01"` instead of `[2020, 1]`)
- Null date arguments to `ts_*()` treated as open-ended ranges (no bound)
- Timeseries data format bumped (v2); legacy files skip timeseries loading with a warning

### Fixed

- `MATCH (a)-[]->(b) RETURN count(*)` with all-aggregate RETURN (no group keys) now correctly returns a single row instead of per-node rows
- `ORDER BY` on DateTime properties with `LIMIT` now returns correct results (FusedOrderByTopK optimization extended to handle DateTime, UniqueId, and Boolean sort keys)
- `ORDER BY` on String/Point properties with `LIMIT` now falls back to standard sort instead of returning empty results

## [0.5.52] - 2026-02-22

### Added

- `add_nodes()` now accepts a `timeseries` parameter for inline timeseries loading from flat DataFrames — automatically deduplicates rows per ID and attaches time-indexed channels
- Timeseries resolution extended to support `hour` (depth 4) and `minute` (depth 5) granularity
- `parse_date_string` now handles `'yyyy-mm-dd hh:mm'` and ISO `'yyyy-mm-ddThh:mm'` formats
- **Timeseries support**: per-node time-indexed data channels with resolution-aware date-string queries
- `set_timeseries()` with `resolution` ("year", "month", "day"), `units`, and `bin_type` metadata
- `set_time_index()` / `add_ts_channel()` for per-node timeseries construction
- `add_timeseries()` for bulk DataFrame ingestion with FK-based node matching and resolution validation
- `get_timeseries()` / `get_time_index()` for data extraction with date-string range filters
- Cypher `ts_*()` functions with date-string arguments: `ts_sum(f.oil, '2020')`, `ts_avg(f.oil, '2020-2', '2020-6')`, etc.
- Query precision validation: errors when query detail exceeds data resolution (e.g. `'2020-2-15'` on month data)
- Channel units (e.g. "MSm3", "°C") and bin type ("total", "mean", "sample") metadata
- Timeseries data persisted as a separate section in `.kgl` files (backward compatible)
- `agent_describe()` includes timeseries metadata, resolution, units, and function reference
- Cypher `range(start, end [, step])` function — generates integer lists for use with `UNWIND`

## [0.5.51] - 2026-02-21

### Added

- Fluent API: `filter()` now supports `regex` (or `=~`) operator for pattern matching, e.g. `filter({'name': {'regex': '^A.*'}})`
- Fluent API: `filter()` now supports negated operators: `not_contains`, `not_starts_with`, `not_ends_with`, `not_in`, `not_regex`
- Fluent API: `filter_any()` method for OR logic — keeps nodes matching any of the provided condition sets
- Fluent API: `offset(n)` method for pagination — combine with `max_nodes()` for page-based queries
- Fluent API: `has_connection(type, direction)` method — filter nodes by edge existence without changing the selection target
- Fluent API: `count(group_by='prop')` and `statistics('prop', group_by='prop')` — group by arbitrary property instead of parent hierarchy

## [0.5.50] - 2026-02-21

### Added

- Shapely/geopandas integration for spatial methods — `intersects_geometry()` and `wkt_centroid()` now accept shapely geometry objects as input in addition to WKT strings
- `as_shapely=True` parameter on `get_centroid()`, `get_bounds()`, and `wkt_centroid()` to return shapely geometry objects instead of dicts
- `ResultView.to_gdf()` — converts lazy results to a geopandas GeoDataFrame, parsing a WKT column into shapely geometries with optional CRS
- Spatial type system via `column_types` in `add_nodes()` — declare `location.lat`/`location.lon`, `geometry`, `point.<name>.lat`/`.lon`, and `shape.<name>` types for auto-resolution in Cypher and fluent API methods
- `set_spatial()` / `get_spatial()` for retroactive spatial configuration
- Cypher `distance(a, b)` now auto-resolves via spatial config (location preferred, geometry centroid fallback)
- Virtual spatial properties in Cypher: `n.location` → Point, `n.geometry` → WKT, `n.<point_name>` → Point, `n.<shape_name>` → WKT
- Spatial methods (`within_bounds`, `near_point_km`, `get_bounds`, `get_centroid`, etc.) auto-resolve field names from spatial config when not explicitly provided
- Node-aware spatial Cypher functions: `contains(a, b)`, `intersects(a, b)`, `centroid(n)`, `area(n)`, `perimeter(n)` — auto-resolve geometry via spatial config, also accept WKT strings
- Geometry-aware `distance()` — `distance(a.geometry, b.geometry)` returns 0 if touching; `distance(point(...), n.geometry)` returns 0 if inside, closest boundary distance otherwise

### Removed

- Cypher functions `wkt_contains()`, `wkt_intersects()`, `wkt_centroid()` — replaced by node-aware `contains()`, `intersects()`, `centroid()` which also accept raw WKT strings

### Fixed

- Betweenness centrality now uses undirected BFS — previously only traversed outgoing edges, causing nodes bridging communities via incoming edges to get zero scores

### Performance

- `RETURN ... ORDER BY expr LIMIT k` fused into single-pass top-k heap — O(n log k) instead of O(n log n) sort + O(n) full projection. **5.4x speedup** on `distance()` ORDER BY LIMIT queries (1M pairs: 2627ms → 486ms)
- `WHERE contains(a, b)` fast path (`ContainsFilterSpec`) — extracts contains() patterns and evaluates directly from spatial cache, bypassing expression evaluator chain
- Spatial Cypher functions 6-8x faster for contains/intersects via per-node spatial cache + bounding box pre-filter:
  - Per-node cache (`NodeSpatialData`): resolves each node's spatial data once per query, cached for all cross-product rows (N×M → N+M lookups)
  - Bounding box pre-filter: computes `geo::Rect` alongside cached geometry; rejects non-overlapping pairs in O(1) before expensive polygon tests
  - `resolve_spatial()` skips redundant expression evaluation for Variable/PropertyAccess — goes directly to cached node data
- Spatial resolution uses WKT geometry cache for centroid fallback path — previously re-parsed WKT on every row
- `intersects()` and `centroid()` avoid deep-cloning `Arc<Geometry>` — use references directly
- `geometry_contains_geometry()` uses `geo::Contains` trait instead of point-by-point boundary check

## [0.5.49] - 2026-02-20

### Added

- Python type stub (`.pyi`) files now included in code graph — enables graph coverage of stub-only packages, compiled extensions, and authoritative type contracts

### Fixed

- Cypher parser now accepts reserved words (e.g. `optional`, `match`, `type`) as alias names after `AS` — previously failed with "Expected alias name after AS"
- Betweenness centrality `sample_size` now uses stride-based sampling across the full node range — previously sampled only the first k nodes, which could be non-participating node types (Module/Class) yielding all-zero scores

## [0.5.46] - 2026-02-20

### Fixed

- Decorator property stored as JSON array instead of comma-separated string — fixes fragmentation of decorators with comma-containing arguments (e.g. `@functools.wraps(func, assigned=(...))`)
- `is_test`, `is_async`, `is_method` boolean properties now explicitly `false` on non-matching entities instead of `null` — enables `WHERE f.is_test = false` queries
- Dynamic project versions (setuptools-scm etc.) now stored as `"dynamic"` instead of `null` on the Project node
- CALLS edges now scope-aware — calls inside nested functions, lambdas, and closures are no longer attributed to the enclosing function (fixes over-counted fan-out in all 7 language parsers)
- `collect(x)[0..N]`, `count(x) + 1` and other aggregate-wrapping expressions in RETURN now work — previously errored with "Aggregate function cannot be used outside of RETURN/WITH"
- `size(collect(...))` and other non-aggregate functions wrapping aggregates now evaluate correctly — previously silently returned `null` because the expression was misclassified as non-aggregate

## [0.5.43] - 2026-02-20

### Added

- List slicing in Cypher: `expr[start..end]`, `expr[..end]`, `expr[start..]` — works on `collect()` results and list literals, supports negative indices

### Fixed

- `size()` and `length()` functions on lists now return element count instead of JSON string length — e.g. `size(collect(n.name))` returns 5 instead of 29
- Duplicate nodes when test directory overlaps with source root (e.g. `root/tests/` inside `root/`) — test roots already covered by a parent source root are now skipped, with `is_test` flags applied to the existing entities instead
- Duplicate Dependency ID collision when same package appears in multiple optional groups — IDs now include the group name (e.g. `matplotlib::viz`)

## [0.5.42] - 2026-02-19

### Added

- `connection_types` parameter for `louvain` and `label_propagation` procedures — filter edges by type, matching the existing support in centrality algorithms

### Fixed

- `CALL pagerank({connection_types: ['CALLS']})` list literal syntax now works correctly — was silently serialized as JSON string causing zero edge matches and uniform scores
- Document list comprehension patterns as unsupported in Cypher reference

## [0.5.41] - 2026-02-19

### Added

- Cypher string functions: `split(str, delim)`, `replace(str, search, repl)`, `substring(str, start [, len])`, `left(str, n)`, `right(str, n)`, `trim(str)`, `ltrim(str)`, `rtrim(str)`, `reverse(str)`

### Fixed

- Duplicate File nodes when source and test roots overlap in code_tree (e.g. `xarray/` source root containing `xarray/tests/` + separate test root)
- Empty `Module.path` properties for declared submodules in code_tree — now resolved from parsed files or inferred from parent directory
- Boolean properties (`is_test`, `is_abstract`, `is_async`, etc.) stored as string `'True'` instead of actual booleans — improved pandas `object` dtype detection to recognize boolean-only columns

## [0.5.39] - 2026-02-19

### Added

- `read_only(True/False)` method to disable Cypher mutations (CREATE, SET, DELETE, REMOVE, MERGE). When enabled, `agent_describe()` omits mutation documentation, simplifying the agent interface for read-only use cases

## [0.5.38] - 2026-02-19

### Added

- Cypher `CALL procedure({params}) YIELD columns` for graph algorithms: pagerank, betweenness, degree, closeness, louvain, label_propagation, connected_components. YIELD `node` is a node binding enabling `node.title`, `node.type` etc. in downstream WHERE/RETURN/ORDER BY clauses
- Inline pattern predicates in WHERE clauses — `WHERE (a)-[:REL]->(b)` and `WHERE NOT (a)-[:REL]->(b)` now work as shorthand for `EXISTS { ... }`, matching standard Cypher behavior
- `CALL list_procedures() YIELD name, description, yield_columns` — introspection procedure listing all available graph algorithm procedures with their parameters and descriptions

### Changed

- `build()` now includes test directories by default (`include_tests=True`)
- CALL procedure error message now hints at the correct map syntax when keyword arguments are used instead of `{key: value}` maps

### Fixed

- CALLS edge resolution in code_tree now uses tiered scope-aware matching (same owner > same file > same language > global) instead of flat bare-name lookup — eliminates false cross-class and cross-language edges
- Rust parser now detects test files at the File level (`_test.rs`, `test_*`, `tests/`, `benches/` conventions) — previously only function-level `#[test]` attributes were detected, leaving File nodes untagged

## [0.5.36] - 2026-02-18

### Changed

- Split `mod.rs` (6,742 LOC) into 5 thematic `#[pymethods]` files: algorithms, export, indexes, spatial, vector — mod.rs reduced to 4,005 LOC
- Enabled PyO3 `multiple-pymethods` feature for multi-file `#[pymethods]` blocks
- Documented transaction isolation semantics (snapshot isolation, last-writer-wins)

### Fixed

- `[n IN nodes(p) | n.name]` now correctly extracts node properties in list comprehensions over path functions — previously returned serialized JSON fragments instead of property values
- `parse_list_value` is now brace-aware — splits at top-level commas only, preserving JSON objects and nested structures
- `EXISTS { MATCH (pattern) }` syntax now accepted — the optional `MATCH` keyword inside EXISTS braces is silently skipped, matching standard Cypher behavior

## [0.5.35] - 2026-02-18

### Added

- CALLS edges now carry `call_lines` and `call_count` properties — line numbers where each call occurs in the caller function
- Comment annotation extraction (TODO/FIXME/HACK/NOTE/etc.) for all non-Rust parsers (Python, TypeScript, JavaScript, Java, Go, C, C++, C#)
- Test file detection (`is_test`) for all parsers based on language naming conventions
- Generic/type parameter extraction for Go 1.18+ and Python 3.12+ (PEP 695) parsers

## [0.5.34] - 2026-02-18

### Added

- `toc(file_path)` method: get a table of contents for any source file — all code entities sorted by line number with a type summary
- `find()` now accepts `match_type` parameter: `"exact"` (default), `"contains"` (case-insensitive substring), `"starts_with"` (case-insensitive prefix)
- `file_toc` MCP tool in `examples/mcp_server.py` for file-level exploration
- `find_entity` MCP tool now supports `match_type` parameter
- Qualified name format documented in `agent_describe()` output (Rust: `crate::module::Type::method`, Python: `package.module.Class.method`)
- Block doc comment support (`/** */`) in Rust parser — previously only `///` line comments were captured
- `call_trace` MCP tool in `examples/mcp_server.py` for tracing function call chains (outgoing/incoming, configurable depth)
- Call trace Cypher pattern documented in `agent_describe()` output
- CHANGELOG.md, CONTRIBUTING.md, and CLAUDE.md for project governance

### Changed

- Doc comments added to all critical Rust structs (`KnowledgeGraph`, `DirGraph`, `CypherExecutor`, `PatternExecutor`, `CypherParser`, and 15+ supporting types)
- Rust parser now captures all `use` declarations, not just `crate::` prefixed imports
- MCP tool descriptions improved with workflow guidance (`graph_overview` says "ALWAYS call this first", `cypher_query` mentions label-optional MATCH, etc.)
- GitHub Release workflow now uses CHANGELOG.md content instead of auto-generated notes

## [0.5.31] - 2025-05-15

### Added

- `find(name, node_type=None)` method: search code entities by name across all types
- `source(name)` method: resolve entity names to file paths and line ranges (supports single string or list)
- `context(name, hops=None)` method: get full neighborhood of a code entity grouped by relationship type
- `find_entity`, `read_source`, `entity_context` MCP tools in `examples/mcp_server.py`
- Label-optional MATCH documented in `agent_describe()` — `MATCH (n {name: 'x'})` searches all node types

### Changed

- Code entity helpers (`find`, `context`) moved from Python (`kglite/code_tree/helpers.py`) to native Rust methods for performance
- `agent_describe()` now conditionally shows code entity methods and notes when code entities are present in the graph

### Removed

- `kglite/code_tree/helpers.py` — replaced by native Rust methods on `KnowledgeGraph`

## [0.5.28] - 2025-05-10

### Added

- Manifest-based building: `build(".")` auto-detects `pyproject.toml` / `Cargo.toml` and reads project metadata (name, version, dependencies)
- `Project` and `Dependency` node types with `DEPENDS_ON` and `HAS_SOURCE` edges
- `USES_TYPE` edges: Function → type references in signatures
- `EXPOSES` edges: FFI boundary tracking (PyO3 modules → exposed items)

### Fixed

- Various code tree parser fixes for Rust trait implementations and method resolution

## [0.5.22] - 2025-04-28

### Added

- `kglite.code_tree` module: parse multi-language codebases into knowledge graphs using tree-sitter
- Supported languages: Rust, Python, TypeScript, JavaScript, Go, Java, C++, C#
- Node types: File, Module, Function, Struct, Class, Enum, Trait, Protocol, Interface, Constant
- Edge types: DEFINES, CALLS, HAS_METHOD, HAS_SUBMODULE, IMPLEMENTS, EXTENDS, IMPORTS
- Embedding export support

---

*For versions prior to 0.5.22, see [GitHub Releases](https://github.com/kkollsga/kglite/releases).*

[0.5.35]: https://github.com/kkollsga/kglite/compare/v0.5.34...v0.5.35
[0.5.34]: https://github.com/kkollsga/kglite/compare/v0.5.31...v0.5.34
[0.5.31]: https://github.com/kkollsga/kglite/compare/v0.5.28...v0.5.31
[0.5.28]: https://github.com/kkollsga/kglite/compare/v0.5.22...v0.5.28
[0.5.22]: https://github.com/kkollsga/kglite/releases/tag/v0.5.22
