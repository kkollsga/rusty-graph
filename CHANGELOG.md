# Changelog

All notable changes to KGLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.23] — 2026-04-27

C++ macro-aware parsing, Go method return-type fix, and receiver attribution
across Go and Rust. Cross-library validation: testify signature-fallback
49% → 0%, KGLite self-graph USES_TYPE edges 2078 → 3197 (+54%), spdlog
macro names eliminated from top return-type list.

### Added

- **`ParameterKind::Receiver`** — Go method receivers (`(c *Call)`) and Rust
  `&self`/`&mut self`/`self` are now captured as structured parameters with
  `kind: "receiver"`, distinct from positional/variadic/kw_variadic. Excluded
  from `param_count` (receivers aren't user-supplied arguments). Cypher
  consumers can filter via `parameters` JSON column.

- **`USES_TYPE` `position="receiver"`** — receivers contribute USES_TYPE edges
  with their own position label. A method `(c *Call) Once() *Call` (receiver +
  return) collapses to `position="both"` consistent with existing aggregation.
  On testify, this drops `position="signature"` fallback from 49% → 0% and
  raises total USES_TYPE edges 251 → 365 (+45%). On KGLite self-graph,
  USES_TYPE edges go 2078 → 3197 (+54%) — Rust `&self` methods now surface
  their owner type as a receiver USES_TYPE edge.

### Fixed

- **C++ parser ignores macro decorators** (`SPDLOG_INLINE`, `FMT_API`,
  `FMT_BEGIN_NAMESPACE`, `Q_INVOKABLE`, etc.). Without this, tree-sitter-cpp
  parses `SPDLOG_INLINE void foo()` so that `SPDLOG_INLINE` looks like a
  type and `foo` becomes the return type — producing `name="unknown"` and
  `return_type="SPDLOG_INLINE"`. Heuristic in `parsers/shared.rs::looks_like_macro_decorator`
  matches all-caps identifiers (length ≥ 2, optional underscores/digits) and
  is applied in `cpp.rs::get_return_type`, `get_name`, and parameter
  extraction. `get_return_type` also recovers from tree-sitter `ERROR` wrappers
  around primitive type keywords (`void`, `int`, `bool`, etc.) — common when
  a macro decorator has confused the parser. On spdlog, macro names are
  eliminated from the top return-type frequency list.

## [0.8.22] — 2026-04-27

A code-graph quality round driven by analysing KGLite's own self-graph and
closing every concrete gap that surfaced. Five new node/edge primitives,
seven new properties on existing nodes, and one small dead-code cleanup.

### Added

- **`BINDS` edges — Python wrapper to Rust pymethod.** Closes the cross-language
  gap where `kglite.KnowledgeGraph.add_nodes` (the Python class method) and
  `crate::graph::pyapi::*::KnowledgeGraph::add_nodes` (the Rust `#[pymethods]`
  impl) lived as disconnected `Function` nodes. The resolver indexes Rust
  functions with `is_pymethod = true` by `(parent_struct_short_name,
  method_name)` and emits `Function -[BINDS]-> Function` for each Python
  method that finds a unique match. Cypher: `MATCH (py)-[:BINDS]->(rs)
  -[:CALLS*]->(impl)` traces a request from the Python entry point to deep
  Rust impl. On the KGLite codebase: ~184 BINDS edges; closes the false-positive
  dead-code finding for `load_ntriples` and other pyapi-exposed functions.

- **Promoted metadata flags as typed Function/Class properties.** Eight booleans
  (`is_pymethod`, `is_pymodule`, `is_ffi`, `is_static`, `is_abstract`,
  `is_property`, `is_classmethod`) plus the `ffi_kind` string are now
  Function-node columns; `is_pyclass` is a Class/Struct column. Replaces
  `f.metadata.get("is_pymethod") == true` JSON-parsing gymnastics with
  `MATCH (f:Function {is_pymethod: true})` direct filters.

- **`USES_TYPE` edges carry a `position` property** (`parameter` | `return` |
  `both` | `signature`). Distinguishes consumers from producers — a function
  that takes `Widget` as a parameter and a function that returns `Widget` no
  longer collapse to the same edge shape. Aggregated per `(function, type)`
  so a single transformation `fn f(w: Widget) -> Widget` emits one edge with
  `position: "both"`. Cypher: `WHERE r.position IN ['parameter','both']` to
  find consumers; `IN ['return','both']` for producers.

- **`Module HAS_FILE File` edges** — closes the natural top-down walk from
  Module → File → Function. Was string-prefix gymnastics on `qualified_name`;
  now `MATCH (m:Module)-[:HAS_SUBMODULE*0..]->(:Module)-[:HAS_FILE]->(f:File)
  -[:DEFINES]->(fn:Function)` returns "what's in this module" in one query.
  Edge name avoids `CONTAINS` (a reserved Cypher keyword for substring matching).

- **`Procedure` nodes — annotation-driven, language-agnostic.** Functions
  whose docstring/leading comment contains `@procedure: NAME` (or
  `@cypher_procedure: NAME`) at the start of a line synthesize a `Procedure`
  node with an `IMPLEMENTED_BY` edge to the function. A single function can
  carry multiple annotations to register under aliases (e.g. both
  `betweenness` and `betweenness_centrality` dispatching to the same impl).
  Generic mechanism for surfacing project-specific registries (Cypher CALL
  procedures, RPC method catalogs, command-bus dispatchers) as first-class
  graph entities. Anchored to line start so prose mentions in docs/tests
  don't false-positive.

- **Annotated all 22 KGLite Cypher CALL procedures** in
  `src/graph/algorithms/graph_algorithms.rs`,
  `src/graph/languages/cypher/executor/rule_procedures.rs`, and
  `executor/call_clause.rs::execute_call_cluster`. Activates the `Procedure`
  node mechanism on the KGLite self-graph: `MATCH (p:Procedure {name: 'pagerank'})
  -[:IMPLEMENTED_BY]->(f:Function) RETURN f.qualified_name` resolves a Cypher
  procedure name to its Rust impl in one query. 27 Procedure nodes (including
  aliases) → 22 implementing functions.

- **Function complexity counters** — `branch_count`, `param_count`,
  `max_nesting`, `is_recursive` now populate on every `Function` node
  produced by `kglite.code_tree.build(...)`. Computed from the
  tree-sitter AST in the same walk that gathers `CALLS` edges, so
  there's no extra parse pass. Per-language branch tables in
  `parsers/shared.rs` cover `if`/`for`/`while`/`case`/`catch`/ternary
  and short-circuit `&&`/`||` forms (cyclomatic-style). Enables direct
  Cypher queries for high-complexity hotspots:

  ```cypher
  MATCH (f:Function)
  WHERE f.branch_count > 30 AND NOT EXISTS { ()-[:CALLS]->(f) }
  RETURN f.qualified_name, f.branch_count, f.max_nesting
  ORDER BY f.branch_count DESC
  ```

- **Generated and minified file skipping during ingestion.** The
  builder now content-sniffs each source file's first 2 KiB before
  dispatching to the per-language parser. Files matching codegen
  markers (`auto-generated`, `DO NOT EDIT`, `code generated by`,
  `<auto-generated>`, `@generated`) are skipped, as are minified
  bundles (one extreme line, or average line width above 500 chars
  across the first 50 lines). Skipped files emit only a `File` node
  with `skip_reason: "generated"` or `skip_reason: "minified"` — no
  Function/Class/Constant nodes — so phantom CALLS edges from
  protobuf stubs and webpack bundles no longer pollute the graph.

  ```cypher
  MATCH (f:File) WHERE f.skip_reason IS NOT NULL
  RETURN f.path, f.skip_reason
  ```

- **Structured `parameters` on `Function` nodes** — JSON-serialised
  list of `{name, type_annotation, default, kind}` per declared
  parameter, with `kind ∈ {positional, variadic, kw_variadic}`.
  Implicit receivers (`self`/`cls`/`&self`/`&mut self`) are excluded.
  Promoting parameters out of the signature string also extends
  `USES_TYPE` resolution: parameter type annotations are now scanned
  alongside the signature and return type, so a function that takes a
  `Widget` argument but doesn't return one now emits the expected
  `Function -[USES_TYPE]-> Widget` edge.

## [0.8.21] — 2026-04-27

Code-analysis tooling round. Closes seven issues filed against KGLite's
own MCP server after a self-analysis session surfaced them — all
visible to users running `kglite.code_tree.build(...)` or
`g.cypher("CALL ...")` against a Rust codebase.

### Added

- **`REFERENCES_FN` edge type — Function → Function** for bare or
  scoped identifiers passed as arguments to higher-order calls
  (`iter.and_then(some_fn)`, `Option::map(my_helper)`). Distinct from
  `CALLS` because the referenced function isn't necessarily invoked
  at the reference site. Dead-code analysis can union the two:

  ```cypher
  MATCH (f:Function)
  WHERE NOT EXISTS { ()-[:CALLS]->(f) }
    AND NOT EXISTS { ()-[:REFERENCES_FN]->(f) }
  RETURN f.qualified_name
  ```

- **`REFERENCES` edge type — Function → Constant** for bare or scoped
  identifiers in function bodies that resolve to a known constant.
  The Rust parser uses `SCREAMING_SNAKE_CASE` as the parse-time
  filter, so local variables don't pollute the edge set. Enables
  detecting unreferenced constants directly from the graph rather
  than via ripgrep.

- **`orphan_node` accepts `link_type` and `direction` parameters.**
  The default behaviour (zero edges in any direction) is unchanged;
  the new params let queries express "no inbound matching edge of a
  specific connection type" — the natural shape for "functions never
  called", "files never imported", etc.:

  ```cypher
  CALL orphan_node({type: 'Function', link_type: 'CALLS', direction: 'in'})
  YIELD node RETURN node.qualified_name
  ```

- **`EXISTS { MATCH ... MATCH ... [WHERE ...] }` multi-clause
  subqueries.** The bare-pattern form already worked; the full
  subquery form with multiple `MATCH` clauses (sharing variables)
  and a `WHERE` predicate evaluated against the merged bindings now
  parses and executes. Multi-hop existence checks no longer have to
  be rewritten as `MATCH ... WITH collect(...) AS xs ... AND NOT y IN xs`.

- **`Project.crate_type` column** captured from `[lib] crate-type` in
  `Cargo.toml`. Lets downstream queries distinguish a regular `lib`
  crate (where `pub fn` is a real export) from a `cdylib` PyO3 crate
  (where only `#[pyfunction]` / `#[pymethods]` matter).

- **`Function.is_test` column** surfaced as a queryable property on
  Function nodes (previously only stored in metadata).

### Fixed

- **`CALLS` edges now include calls inside closure bodies.**
  `closure_expression` was on the parser's NESTED_SCOPES skip-list,
  so `.map(|x| foo(x))` / `.and_then(|x| bar(x))` produced zero
  CALLS edges to the inner function. Closures are expressions in
  Rust, not items — calls inside them belong to the enclosing
  function semantically.

- **`self.method()` receiver-type disambiguation.** When the same
  method name exists on multiple structs, a bare `self.method()`
  call inside a method of `Foo` now narrows to `Foo::method` ahead
  of `Bar::method`, even when both are candidates and live in
  different files. Uses the caller's owner short name as an implicit
  receiver hint when no explicit one is present.

- **`is_test` propagates into inline `#[cfg(test)] mod tests` blocks.**
  Previously only `#[test]` / `#[bench]` annotated functions were
  flagged; helpers inside the test mod weren't, inflating every
  dead-code query against a Rust codebase. Files literally named
  `tests.rs` are also flagged at the file level.

- **`CALL` rule procedures list accepted parameters in error
  messages.** Missing-required-parameter errors now show the full
  schema (required + optional names), so first-time use of a
  procedure doesn't cost three error rounds before guessing the
  parameter name.

### Removed

- **`ARCHITECTURE.md`.** A refactor-time artifact from the 0.8.0
  storage refactor; framing was past-tense and the parity test that
  validated its file-path references was removed alongside it. The
  three other parity gates (god-file cap, unsafe-SAFETY comments,
  mod.rs purity) are evergreen and stay.

- **Dead disk-build infrastructure** — `block_pool.rs`,
  `block_column.rs`, `memory/build_column_store.rs`,
  `AsyncPropertyLogWriter`, plus a sweep of unused methods, fields,
  and enum variants. v3 disk pipeline replaced these; ~3000 lines net.

- **Legacy benchmark suites** — `test_nx_comparison.py` (NetworkX
  comparison, required scipy that wasn't in the venv) and
  `test_performance.py` (used the old `result["stats"][...]`
  subscript API, every Cypher-mutation test failed). Superseded by
  `test_bench_core.py` and `test_bench_memory.py`.

The "completeness round" — six phases that round out the Cypher and
Fluent surface so no primitive a user would reasonably expect is
missing. Every domain (legal, code, sodir, Wikidata) gets value from
each addition; none are domain-specific.

### Added

- **Cypher INTERSECT / EXCEPT (Phase 6 of the completeness round).**
  Cypher now exposes the standard set operators:

  ```cypher
  MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
  INTERSECT
  MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name

  MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
  EXCEPT
  MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name
  ```

  `INTERSECT` keeps rows present in both sides; `EXCEPT` keeps rows in
  left but not in right. Both always dedupe, matching SQL/openCypher
  conventions. Internals: new `SetOpKind` enum on `UnionClause` (the
  same `Clause::Union` variant carries all three operators). The
  executor dispatches on `kind`: UNION uses the existing concat-and-
  dedup path; INTERSECT/EXCEPT pre-build a row-hash set from the
  right-side result and filter the left.

  Brings Cypher in line with the fluent-API set ops (`union`,
  `intersection`, `difference`, `symmetric_difference`).

- **Geospatial primitives (Phase 5 of the completeness round).** Round
  out the spatial surface with the standard GIS toolkit operations.

  Cypher scalar functions on WKT/node geometries:
  - `geom_buffer(geom, meters)` — planar buffer.
  - `geom_convex_hull(geoms)` — variadic or list arg.
  - `geom_union(g1, g2)` / `geom_intersection(g1, g2)` /
    `geom_difference(g1, g2)` — boolean ops.
  - `geom_is_valid(geom)` — OGC validity.
  - `geom_length(geom)` — geodesic length for LineStrings; perimeter
    for polygons (sum of rings); 0 for points.

  Cypher CALL procedure:
  - `kg_knn({lat, lon, target_type, k})` YIELD `node, distance_m` —
    *k* nearest nodes of a target type to a coordinate (geodesic;
    location-first, falls back to geometry centroid).

  ```cypher
  CALL kg_knn({lat: 60.4, lon: 5.3, target_type: 'City', k: 5})
  YIELD node, distance_m
  RETURN node.title, round(distance_m / 1000.0, 1) AS km
  ```

  Backed by `geo = "0.33"` (`Buffer`, `BooleanOps`, `ConvexHull`,
  `Validation`, `LengthMeasurable` traits). New helpers in
  `src/graph/features/spatial.rs`; new `geom_arg` resolver in
  `executor/expression.rs` accepts WKT strings, Points, and
  spatial-configured node/property variables.

- **Weighted shortest path (Phase 4 of the completeness round).**
  `graph.shortest_path()` and `graph.shortest_path_length()` now
  accept an optional `weight_property` parameter. When set, the
  search switches from BFS (hop count) to Dijkstra (sum of edge
  weights). Edges missing the property fall back to weight 1.0
  (matching Louvain's existing weighted-adjacency convention);
  negative weights cause the path to be reported as missing.

  ```python
  result = graph.shortest_path(
      "Stop", "A", "Stop", "Z",
      weight_property="cost",
  )
  # {'path': [...], 'connections': [...], 'length': 3, 'weight': 4.7}

  graph.shortest_path_length(
      "Stop", "A", "Stop", "Z",
      weight_property="cost",
  )  # → 4.7 (float; int when unweighted)
  ```

  Internals: new `shortest_path_weighted()` and
  `shortest_path_cost_weighted()` in `algorithms/graph_algorithms.rs`,
  Dijkstra with a `BinaryHeap<State>` keyed on `(distance, node_idx)`.
  The existing BFS path remains the default — no overhead for
  unweighted callers.

- **Structural validators v2 — rule packs extension (Phase 3 of the
  completeness round).** Seven new CALL procedures complete the
  rule-pack family from 0.8.19 by covering *n*-ary and declarative
  checks:

  - `inverse_violation({rel_a, rel_b}) YIELD a, b` — declared-inverse
    relations not symmetric (e.g. `parent_of` without matching
    `child_of`).
  - `transitivity_violation({rel}) YIELD a, b, c` — `(a)-[rel]->(b)
    -[rel]->(c)` chains where the direct `(a)-[rel]->(c)` is absent.
    Generalizes the OCTF subclass-fold audit pattern.
  - `cardinality_violation({type, edge[, min, max]}) YIELD node, count`
    — declarative cardinality. Setting `max:1` catches functional-
    property violations; `min:1` catches missing-required-edge.
  - `type_domain_violation({edge, expected_source}) YIELD source, target`
  - `type_range_violation({edge, expected_target}) YIELD source, target`
    — schema integrity checks on edge endpoints.
  - `parallel_edges({edge}) YIELD a, b, count` — pairs connected by
    more than one edge of the same type (almost always an ETL bug).
  - `null_property({type, property}) YIELD node` — property side of
    `missing_required_edge`.

  All seven follow the existing rule-procedure pattern and surface
  via `CALL list_procedures()` and `describe(cypher=True)`.

- **Lexical text predicates (Phase 2 of the completeness round).** Six
  string-similarity primitives now expressible in Cypher without
  dropping to Python:

  - `text_edit_distance(a, b)` — Levenshtein, UTF-8 aware (uses
    minimum-row DP for O(min(n,m)) memory).
  - `text_normalize(s)` — lowercase, drop punctuation, collapse
    whitespace. The thing every fuzzy-match pipeline reaches for first.
  - `text_jaccard(a, b [, sep])` — token-set Jaccard, default whitespace
    separator.
  - `text_ngrams(s, n)` — character n-grams as a list.
  - `text_contains_any(s, needles)` / `text_starts_with_any(s, prefixes)`
    — variadic or list-argument forms; short-circuit on first match.

  ```cypher
  MATCH (a:Person), (b:Person) WHERE a.id < b.id
  WITH a, b, text_edit_distance(
      text_normalize(a.title), text_normalize(b.title)
  ) AS d
  WHERE d <= 2 RETURN a.title, b.title, d
  ```

- **Expression-engine fundamentals (Phase 1 of the completeness round).**
  The Cypher engine gains the standard scalar/aggregate/list-fold
  primitives that were missing:

  - `properties(n)` / `properties(r)` — full property map of a node or
    relationship (returns a JSON-formatted map; works alongside
    `keys()`).
  - `start_node(r)` / `end_node(r)` — endpoint access on a bound edge
    variable. `start_node(r).name` works via the existing dotted
    property accessor.
  - `reduce(acc = init, x IN list | body)` — list fold with
    accumulator. New `Expression::Reduce` AST variant; mirrors
    openCypher.
  - `percentile_cont(expr, p)` — continuous percentile via linear
    interpolation; `p ∈ [0,1]`.
  - `percentile_disc(expr, p)` — discrete percentile via nearest rank.
  - `median(expr)` — sugar for `percentile_cont(expr, 0.5)`.
  - `variance(expr)` / `var_samp(expr)` — sample variance, n-1
    denominator (matching the existing `std` convention).

  ```cypher
  MATCH (n:Person)
  RETURN median(n.age), percentile_cont(n.age, 0.9), variance(n.age)

  MATCH (n:Person) WITH collect(n.age) AS ages
  RETURN reduce(s = 0, x IN ages | s + x) AS total
  ```

### Changed

- **Example MCP server simplified to two tools.**
  `examples/mcp_server.py` now exposes only `graph_overview` and
  `cypher_query`. The convenience tools (`search`, `find_entity`,
  `read_source`, `entity_context`, `bug_report`) are removed — every
  operation is reachable from Cypher via `MATCH (n) WHERE n.title =
  $text` etc., and the docstring shows the equivalent patterns. Same
  simplification philosophy as the 0.8.19 rule-procedure refactor:
  lean on the Cypher surface.

## [0.8.19] — 2026-04-26

### Changed

- **Rule packs rebuilt as native Cypher CALL procedures.** The
  Python-layer `kglite.rules` package (`g.rules.run(...)`,
  `RuleReport`, YAML packs, ~1,200 lines) is removed; six
  structural-validator procedures live inside the Cypher engine
  alongside `pagerank` / `connected_components`:

  ```cypher
  CALL orphan_node({type: 'Wellbore'}) YIELD node RETURN node
  CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node ...
  CALL missing_inbound_edge({type: 'Discovery', edge: 'IN_DISCOVERY'}) YIELD node ...
  CALL self_loop({type: 'Person', edge: 'KNOWS'}) YIELD node ...
  CALL cycle_2step({type: 'Person', edge: 'KNOWS'}) YIELD node_a, node_b ...
  CALL duplicate_title({type: 'Prospect'}) YIELD node ...
  ```

  Direct graph iteration in Rust replaces the YAML→Cypher→parse round
  trip — single rule on sodir (564k nodes) runs in **<2 ms** vs.
  ~5 ms for the legacy Python pack runner. Composability with
  surrounding Cypher (WHERE / ORDER BY / aggregation) collapses the
  previous two-step `rules_run + cypher_query` flow into a single
  pass.

  Direction validation, anchored type-by-type iteration, and the
  `DirectionMismatch` error survive — ported to Rust. Same agent
  protection without the parallel API.

  Discovery surface: rule procedures appear in
  `describe(cypher=True)` topic list, in the
  `<rules hint="..."/>` extension hint of `describe()`, and in
  `CALL list_procedures() YIELD name`. Per-procedure docs via
  `describe(cypher=['orphan_node'])`. No `<rule_packs>` block. No
  opt-in `advertise()` function. No separate `rules_run` MCP tool —
  agents invoke via `cypher_query`.

  **Breaking change.** Code using `g.rules.run(...)` or any
  `kglite.rules.*` import from 0.8.16–0.8.18 must migrate to the
  CALL syntax. The migration is mechanical: one `CALL` per rule
  with map-syntax parameters and `YIELD node` (or `YIELD node_a,
  node_b` for `cycle_2step`).

  Removed: `kglite/rules/` package, `g.rules` accessor on
  `KnowledgeGraph`, `Rule`/`RulePack`/`RuleReport`/`_RulesAccessor`
  classes, `kglite.rules.advertise()`, `_set_default_rule_pack_xml`
  PyO3 function, `_set_rule_pack_xml` PyO3 method, `rule_packs_xml`
  field on `KnowledgeGraph`, `inject_rule_packs` helper in
  describe.rs, the `<rule_packs>` block in `describe()`, the
  `rules_run` MCP tool from `examples/mcp_server.py` and
  `prospect_mcp_server.py`, `pyyaml>=6.0` runtime dependency.

## [0.8.18] — 2026-04-26

### Changed

- **Rule-pack discovery via `describe()` is now opt-in.** A fresh
  `kglite.load(...)` produces a `describe()` with no `<rule_packs>`
  block — graphs that don't use rule packs incur no agent-facing
  noise. Activation is explicit:
  - `g.rules.run(...)` or `g.rules.load(...)` activates per-graph
    advertising (existing behaviour, unchanged).
  - New `kglite.rules.advertise()` publishes a module-level default
    visible to every subsequent `describe()` across all graphs. Use
    this for MCP servers that expose a rule-pack tool. Idempotent.
  - The `examples/mcp_server.py` `rules_run` tool is now commented
    out; the file documents how to re-enable it for users who want
    rule packs in their MCP surface. The default MCP example is
    rule-pack-free.
- **Per-rule Cypher timeout via `default_timeout_ms`.** Rules can
  declare a YAML-level `default_timeout_ms` and the runner passes it
  as the `timeout_ms` to `g.cypher()` per rule. A caller-supplied
  `timeout_ms` to `g.rules.run(...)` always wins. Lets full-Wikidata
  users set realistic budgets on rules that scan dense node types
  (13M humans, 45M scholarly articles) without affecting the global
  graph timeout.
- **Rule-pack `describe()` integration moved into Rust.** Slice 1.1
  shipped agent-discovery via a Python monkey-patch that wrapped
  `KnowledgeGraph.describe`, called the Rust method (preserved as
  `_describe_native`), and post-processed the XML to splice in a
  `<rule_packs>` block. That dispatch is now native: `describe()`
  is the Rust method again. Per-instance pack XML lives on the
  `KnowledgeGraph` struct (`Mutex<Option<String>>`); a module-level
  default holds the cold bundled-pack inventory. Python's role
  shrinks to rendering the XML on pack `load()` / `run()` and
  pushing it via the new `_set_rule_pack_xml` PyO3 method (and the
  module-level `_set_default_rule_pack_xml`). User-visible XML and
  behaviour are byte-compatible; the wrapper indirection and its
  per-call `str.rfind`/slice/concat are gone.

### Fixed

- **`LIMIT` was applied before `WHERE` filtering, returning fewer rows
  than expected.** A query like
  `MATCH (n:T) WHERE NOT EXISTS { (n)-[:E]->() } RETURN n.id LIMIT 5`
  could return 0 rows when the first 5 candidate nodes all failed the
  WHERE predicate. Root cause: the planner pushed the LIMIT hint into
  `PatternExecutor`, which capped *candidates* before the inline
  WHERE filter ran. Fix: skip the limit hint at pattern-execution
  time when an inline WHERE is present, and apply LIMIT after the
  WHERE filter (as the surrounding executor already attempts to).
  Affects any filtered query with LIMIT, not just `NOT EXISTS`.

### Added

- **Rule packs** — agent-discoverable structural validators. New
  `g.rules` sub-namespace exposes `list()`, `load()`, `run()`, and
  `describe()` for named YAML packs that compile to Cypher and emit a
  structured `RuleReport`. The bundled `structural_integrity` pack
  (v1.1) ships six universal cross-graph rules: orphan nodes,
  self-loops, short cycles, missing-required-edge (outbound),
  missing-inbound-edge, and duplicate titles. `g.describe()` surfaces
  a `<rule_packs>` block so agents discover packs through the same
  XML they consume for schema. See
  [`docs/guides/rules.md`](docs/guides/rules.md). Reports are lazy:
  `.summary` returns counts without materialising rows, and runs are
  cached per `(pack_name, params, graph)`. New runtime dependency:
  `pyyaml>=6.0`.
- **Rule-pack ergonomics:**
  - `summary["any_truncated"]` — top-level boolean so agents can
    one-glance check if any rule hit its LIMIT.
  - `report.is_suspect(node_id)` — O(1) cross-reference helper that
    returns `[(rule_name, severity), ...]` for rules that flagged the
    node. Built lazily; accepts string or int ids.
  - `g.rules.list()` now reads bundled-YAML headers lazily so cold
    inventory shows real version + rule_count + description (no
    placeholders) before any pack has been loaded.
  - Optional `usage_hint:` field on a pack — surfaced via
    `g.rules.describe(name)` and as an XML attribute in
    `g.describe()` so agents can read "use this pack when…" guidance
    inline with the schema.
  - `to_markdown()` truncates list-typed cells (e.g. the `ids`
    column in `duplicate_title`) to 3 elements + " (+N more)" so
    agent-pasted output stays readable.
  - **Direction-aware `missing_*_edge` rules.** New optional
    `validates_direction:` rule field (`"outbound"` or `"inbound"`).
    The runner inspects `g.connection_types()` and refuses to execute
    when the `(type, edge)` pair flows the wrong way in the graph's
    actual schema, surfacing a `DirectionMismatch` error that
    suggests the right rule. The bundled `missing_required_edge` and
    `missing_inbound_edge` opt in. Prevents trivial rule firing where
    e.g. asking for incoming `IN_LICENCE` on a `Wellbore` would have
    matched every wellbore meaninglessly.

## [0.8.17] — 2026-04-26

### Performance

- **Two-MATCH count fusion: top-K-by-degree filtered queries now run
  ~20× faster.** The shape
  `MATCH (w)-[:T]->(b {nid:'X'}) MATCH (w)-[r]-() WITH ...
  count(r) ... ORDER BY count DESC LIMIT k`
  used to materialise one row per edge for every group key (e.g. 4 M
  edge rows for 416 k Wikidata writers — 494 s on the full graph).
  The aggregation-fusion pass at
  `src/graph/languages/cypher/planner/fusion.rs` now also recognises
  `[Match, Match, With(count)]` and folds it into a single
  `FusedMatchWithAggregate` whose secondary pattern drives the
  per-group-key degree count via the existing `count_edges_filtered`
  fast-path. Measured: top-10-by-degree on Wikidata writers
  **494 s → 24 s (20×)**. The remaining time is one degree lookup
  per group key (832 k mmap reads) — further wins live in storage,
  out of scope for this session.
- **`count_edges_filtered` fast-path now handles undirected `[r]-`
  edges.** Previously the fast-path returned `None` for
  `EdgeDirection::Both`, forcing the slow per-edge enumeration. It
  now sums incoming + outgoing `count_edges_filtered` calls — the
  canonical "total degree" pattern. Both the new two-MATCH fusion and
  the existing single-MATCH `WITH count` benefit.
- **Per-group-key count phase in `execute_fused_match_with_aggregate`
  now runs in parallel** above 4 096 group keys. Each
  `count_edges_filtered` call is a read-only mmap lookup, so rayon's
  par_iter overlaps the per-call I/O instead of serialising it.
  Measured on the same Wikidata top-10-by-degree query
  (124 M nodes / 861 M edges): **24 s → 8.5 s (~2.5×)** on top of
  the fusion win. End-to-end wall on `top_writers.py` is now 73 s,
  vs. 510 s before any of this session's work (~7× total).
- **Top-K hint absorbed into `FusedMatchWithAggregate`.** A new
  planner pass `fuse_match_with_aggregate_top_k` recognises the
  shape `[FusedMatchWithAggregate, Return, OrderBy(count_alias),
  Limit(k)]` (where the RETURN is a pure pass-through projection)
  and pushes the K-bound into the fused stage. The executor sorts
  by count first and then evaluates the group-key projection
  expressions only for the K winners — saves N×P
  `evaluate_expression` calls when N is large and K is small. The
  Wikidata top-10-by-degree query goes from materialising 416 k
  rows to 10; per-row property reads stop being the tail cost
  (modest 5% gain on top of the parallel-count win, but principled:
  "only do necessary work" — projection-heavy queries get a much
  larger benefit).
- **Lazy RETURN — defer per-row property evaluation until Python
  reads each cell.** The planner's new `mark_lazy_eligibility`
  pass annotates the terminal RETURN with `lazy_eligible = true`
  when the query is `MATCH … (WHERE …) RETURN <prop access>` and
  there's no downstream operator that needs row values
  (DISTINCT/HAVING/ORDER BY/aggregate/WITH/UNWIND/CALL/UNION/
  mutation all force the eager path). The executor skips
  `execute_return_projection`'s per-row loop and hands the
  pending rows + return items to the Python `ResultView` via a
  side-channel `LazyResultDescriptor`. `ResultView` materialises
  cells on access (memoised via a `Mutex<Vec<Option<…>>>` so
  repeat reads are free), and `__len__` becomes O(1). Measured
  on the same Wikidata script: the find-writers query
  (`MATCH … RETURN nid, title`, used only for `len()` in the
  caller) **57 s → 35 s (~1.6×)**. End-to-end on
  `top_writers.py` is now **49 s** — down from the original
  **510 s** before this session — **~10× total**.

### Performance

- **Phase 1 N-Triples loader is ~1.7× faster.** Steady-state on
  Wikidata's `latest-truthy.nt.bz2` went from ~2.4 M tri/s to
  **~4.1 M tri/s** (`--size 50` build dropped 20.83 s → 15.96 s; full
  Wikidata Phase 1 projects from ~2 h to **~70 min**). Profiling with
  `samply` showed the loader thread spending ~32% of CPU in
  `libsystem_malloc` and ~10.7% in `core::str::pattern::TwoWaySearcher`
  (used by `str::find`). Four targeted changes:
  - **Byte-level `parse_line`** (`src/graph/io/ntriples/parser.rs`):
    swap `line.find("> ")` for `memchr::memchr(b'>')`. URIs in
    N-triples cannot contain `>`, so a single byte scan is sufficient.
  - **`EntityAccumulator` capacity preallocation**
    (`HashMap::with_capacity(32)`, `Vec::with_capacity(8)`):
    eliminates `RawVecInner::finish_grow` reallocs in the per-entity
    accumulator.
  - **`scratch_props` reuse in `flush_entity`**: hoist the
    `Vec<(InternedKey, Value)>` out of the function so the alloc cost
    is paid once per build instead of per entity.
  - **`mimalloc` as global allocator** (`src/lib.rs`): pure Rust /
    build-time-only dependency; ~10% wall-time win on the loader on
    top of the parser-side changes.
- **Reader-thread channel batches 50k → 200k.** Reduces per-batch
  sync overhead 4× without growing peak RSS meaningfully.

### Added

- **Block-level parallel decoder for single-stream `.bz2` files.** Wikidata
  ships `latest-truthy.nt.bz2` as a single bz2 stream, so the existing
  stream-level scanner in `parallel_bz2.rs` was falling through to a
  single-threaded `MultiBzDecoder` (~1 M triples/s ceiling). The new
  single-stream path delegates to `bzip2_rs::ParallelDecoderReader`
  (paolobarbolini/bzip2-rs, MIT/Apache-2.0), which finds bit-aligned
  block magics inside one stream and decodes blocks on rayon workers.
  Measured Phase 1 throughput on Wikidata: **~1.0 M tri/s → ~3.3 M tri/s
  (3.3× speedup)** at the same memory ceiling. Multistream files still
  use the existing stream-level path. Pinned to a git rev because the
  published 0.1.2 crate ships an older `Cargo.toml` without the `rayon`
  feature flag.
- **Phase 1 progress bar now shows ETA when `max_entities` is set.**
  When the caller has set an entity cap, the loader emits the bar
  position as `entities_created` against `total = max_entities`, so
  tqdm can compute ETA from the entity rate. Without a cap the bar
  still tracks triples (no total → no ETA, just rate). The unused
  counter ships in the event's `fields` dict either way.
- **Ctrl+C cancellation of `load_ntriples` builds.** Phase 1 runs
  inside `py.detach()` (GIL released so Python heartbeat threads can
  run), which previously meant SIGINT couldn't reach Python until the
  Rust call returned — i.e. Ctrl+C did nothing during a multi-hour
  Wikidata build. The progress sink now reacquires the GIL on every
  update event and calls `Python::check_signals`; a pending SIGINT
  flows back through a new `Cancelled` marker on `ProgressSink::emit`,
  unwinds the loader cleanly, and surfaces as `KeyboardInterrupt` on
  the Python side. Cancellation requires a `progress=` callback (which
  is the default in the bench script and dataset wrappers).

### Changed

- **`bench/wikidata_e2e.py` CLI overhaul.** `--progress` is now the
  default (use `--legacy-progress` for the old `[Phase X]` stderr
  output). Removed `--quiet` — wrapper-level status messages
  (cooldown checks, cache hits) print regardless; the loader's
  per-phase eplog lines are auto-silenced when tqdm is active so they
  don't fight the bar. Renamed `--size` to `--entities-m` to make it
  clear the cap is in millions of *entities*, not triples (`--size`
  remains as a deprecated alias).
- **`kglite.datasets.wikidata.open` auto-silences the loader when
  `progress=` is set.** Wrapper-level `verbose=True` controls the
  cache-hit / cooldown messages; loader `verbose` is forced off when
  a progress callback is wired so tqdm owns the terminal.

### Added (continued)

- **Structured build-phase progress callback for `load_ntriples`.** New
  `progress=` kwarg accepts a Python callable that receives one dict
  per phase event (`start` / `update` / `complete`) for each of
  `phase1` (streaming), `phase1b` (columnar build), `phase2` (edges),
  `phase3` (CSR), and `finalising`. Phase 1 fires updates every 5M
  triples (decoupled from the 60s stderr gate) so a UI driven by the
  callback stays live. Errors raised by the callback are swallowed so
  a broken UI cannot kill a multi-hour build.
  Pure-Rust trait `ProgressSink` lives in
  `src/graph/io/ntriples/mod.rs`; the PyO3 adapter that translates a
  Python callable into a sink lives in `src/graph/pyapi/kg_core.rs`,
  keeping the loader free of `pyo3` types.
- **`kglite.progress.TqdmBuildProgress`** — drop-in tqdm-backed
  reporter. One bar per phase, with RSS (via `psutil`) and per-phase
  counters in the postfix. `pip install tqdm psutil` to use.
- **`bench/wikidata_e2e.py --progress`** — opt-in flag that wires
  `TqdmBuildProgress` into the e2e benchmark.

### Fixed

- **`load_ntriples` no longer panics with `slice index starts at A
  but ends at B` on large disk/mapped builds.** Wikidata builds past
  ~450 M triples crashed in `MmapColumnStore::read_str` on reload.
  Root cause: `flush_entity` for entities whose ID didn't parse as a
  Q-code wrote `Value::String(acc.id)` into the `nid` column, which
  flipped that column's `id_is_string=true`. Subsequent entities with
  `Value::UniqueId` left their string offsets uninitialised (zero), so
  reload hit `start > end` decoding them. Fix: in disk/mapped mode,
  skip entities whose ID is not a parseable Q-code at the top of
  `flush_entity` (these were unreachable in the canonical Wikidata
  query surface anyway). In-memory mode is unaffected.
- **Cypher `DETACH DELETE` no longer breaks subsequent typed-edge
  traversals.** Pre-fix, after a Cypher `DETACH DELETE`, fluent
  `g.select(t).traverse(conn_type, ...)` (and any `make_traversal`
  caller) would throw *"Connection type 'X' does not exist in
  graph"* even when `X` still had millions of live edges. The
  Cypher executor's `execute_delete` invalidated the
  `edge_type_counts_cache` but left the `connection_types`
  HashSet alone — `has_connection_type()` consults the HashSet
  first and returned a stale negative. Fix: clear the
  `connection_types` cache on Cypher delete, and add a final
  fall-through in `has_connection_type()` to the disk backend's
  authoritative `conn_type_index_*` arrays.
  Surfaced by `bench/benchmark_full.py` against every disk row.

### Performance

- **Parallel multistream `.bz2` decoder for `load_ntriples` —
  closes the gap with `.zst`.** Wikidata / pbzip2 dumps are a
  concatenation of independent bz2 streams; the previous
  `bzip2::read::MultiBzDecoder` walked them sequentially on a
  single core. New `parallel_bz2::open()` (in
  `src/graph/io/ntriples/parallel_bz2.rs`) scans the file for
  `BZh[1-9]` + 6-byte block magic, dispatches streams to a
  worker pool sized by a memory budget (256 MB default, after
  pbzip2's `NumBufferedBlocksMax`), and re-orders the
  decompressed chunks behind a single `Read` surface. Single-
  stream `.bz2` files take a fast path through `MultiBzDecoder`
  with no thread-pool overhead. Workers join before
  `load_ntriples` exits Phase 1, so no parallelism leaks into
  the Phase-2/3 rayon pool.
  Measured: wiki100m bz2 **99 s → 34 s (2.9×)**, wiki200m
  **199 s → 71 s (2.8×)**. bz2/zst ratio 3.4× → 1.19×.
- **`enable_columnar()` is now idempotent on the already-columnar
  fast path.** Previously, every `g.save()` re-ran the full
  per-node columnar rebuild — even when the graph was already
  columnar and unmodified. At wiki100m memory mode this cost
  ~257 s of pure waste on consecutive saves. Now `enable_columnar`
  walks every node once (O(N) cheap matches) and short-circuits
  if all nodes are `PropertyStorage::Columnar` AND their
  `Arc<ColumnStore>` matches `graph.column_stores` for the type
  (the Arc-pointer check catches the common
  `add_nodes(conflict_handling="update")` fork pattern that would
  otherwise lose updates on save). Measured wiki5m: consecutive
  `save()` 455 ms → **177 ms (2.6×)**.

### Changed

- **`verbose=True` on `load_ntriples` is now phase-oriented.** Previous
  output was a mix of `[T+30s]` timestamps and ad-hoc sub-step prints;
  Phase 2/3 output in particular was developer-grade noise. The new
  output is a small set of `[Phase N]` gate messages — open/close
  pairs around each major stage of the build:
  ```
  [Phase 1] Streaming and parsing N-triples (...)
  [Phase 1] 12.3M triples, 2.8M entities, 8.5M edges buffered — 205k triples/s
  [Phase 1] Complete: ... in 47m18s
  [Phase 1b] Building columnar storage (...)
  [Phase 1b] Complete in 8m42s
  [Phase 2] Creating edges
  [Phase 2] Complete: ... edges in 11m22s
  [Phase 3] Building CSR edge index
  [Phase 3] Complete in 2m04s
  [Finalising] Building auxiliary indexes + saving metadata
  [Finalising] Complete in 32s
  [Build] Total elapsed: 1h09m54s
  ```
  Sub-step timings (CSR step 1/4, peer-count histogram, mmap layout,
  per-type flush logs, interner save timings, Q-code resolution
  timings, …) move behind `KGLITE_BUILD_DEBUG=1`. The legacy
  `KGLITE_CSR_VERBOSE` env var is replaced by `KGLITE_BUILD_DEBUG`
  (one flag for all build sub-step output).

### Added

- **`kglite.datasets.sodir.open(workdir, ...)` — one-call lifecycle for
  Sodir factmaps petroleum data.** Resolves CSVs from the public
  ArcGIS FeatureServer at
  `https://factmaps.sodir.no/api/rest/services/DataService`, applies
  the FK pre-processing the existing build script does, and builds
  the graph via the packaged blueprint. Default storage is
  ``memory`` — Sodir is small enough that disk caching adds little
  on top of CSV caching:

      g = sodir.open("/data/sodir")  # memory; index_cooldown_days=14, dataset_cooldown_days=30
      g = sodir.open("/data/sodir", storage="disk")  # opt-in for cross-process reuse

  Workdir layout: `csv/` (fetched datasets, flat), `sodir_index.json`
  (per-dataset row count + timestamps), `graph/` (disk-mode only).
  Index sweep cheaply re-checks remote row counts every 14 days; only
  changed datasets re-download. Hard cooldown forces full per-dataset
  refresh every 30 days even if counts match.

  **Complement blueprints**: pass
  ``complement_blueprint=path/to/extra.json`` to add new node types
  / edges on top of the packaged baseline. The file is persisted to
  ``workdir/blueprint_complement.json`` on first call and auto-loaded
  on subsequent calls. Pass ``use_complement=False`` to skip it for a
  single call, or ``sodir.remove_complement(workdir)`` to drop it
  permanently. Deep merge with **base-wins on key collisions by
  default** (the packaged baseline tracks the canonical Sodir REST
  catalog and stays authoritative); set
  ``complement_overrides=True`` to flip when the complement should
  win.

  The blueprint walker auto-detects which datasets are referenced and
  fetches only those — adding new node types to the blueprint
  triggers fetches the next time `open()` runs. The packaged
  baseline ships only the 33 node types whose CSVs are fetchable
  from REST (no sideloaded prospect / play / ocean data); use a
  complement to layer those in.

  **Parallel fetcher**: ``workers`` parameter (default 4) drives a
  thread-pool that pulls dataset jobs off a shared backlog. tqdm
  progress bar replaces per-dataset prints so verbose output stays
  one line tall. Geometry handler is defensive against empty
  ``coordinates: []`` (some pre-1970 Sodir wellbores) — drops those
  features' geometry without aborting the fetch. KGLite is independent of Sodir / the
  Norwegian Offshore Directorate; see module docstring. Catalog
  (LAYERS / TABLES / FACTMAPS_LAYERS) vendored from
  `kkollsga/factpages-py`.

- **`kglite.datasets.wikidata.open(workdir, ...)` — one-call lifecycle
  for Wikidata `latest-truthy` graphs.** Resolves the dump (download,
  resume `.part`, refresh on cooldown), builds the disk or in-memory
  graph, and returns it. Subsequent calls cache-hit on the saved
  graph at `workdir/graph[_<N>m]/`. Two storage backends:

      g_full = wikidata.open("/data/wd")                        # disk graph
      g_100m = wikidata.open("/data/wd", entity_limit_millions=100)
      g_mem  = wikidata.open("/data/wd", storage="memory",      # rebuild every call
                              entity_limit_millions=10)

  Sized slices (`entity_limit_millions=100/200/...`) live alongside
  the full graph (`graph_100m/`, `graph_200m/`, `graph/`) and all
  share the same `latest-truthy.nt.bz2` dump under `workdir`.

  Also exports `fetch_truthy(workdir, cooldown_days=31)` for the
  dump-only path. KGLite is independent of the Wikimedia Foundation;
  see module docstring.

- **`load_ntriples` releases the GIL.** Multi-minute loads no longer
  block Python threads — heartbeat / progress monitors and other
  background workers run on schedule throughout the build. Required
  for the new minute-cadence reporting in
  `examples/wikidata_disk.py`.
- **`bench/benchmark_full.py`** — full-stack lifecycle benchmark
  (build / save / load / mutate / resave / Cypher / fluent) across
  every storage mode × Wikidata subset. Wide-pivot CSV output —
  one row per (run, mode, dataset). Errors preserved both in the
  `errors` column (truncated, semicolon-separated) and in a
  sidecar `bench/benchmark_full.errors.log` (full text,
  tab-separated for `cut`-friendly inspection).
- **`bench/results.py`** — pandas-backed analysis tool over the
  bench CSV. Three commands: `latest` (most recent measurement
  per cell), `trends` (per-run time-series for filtered cells),
  `deltas` (consecutive-run deltas — surfaces regressions). Use
  `--mode` / `--dataset` / `--cols` filters.
- **`--languages` CLI flag on `bench/wiki_benchmark.py` and
  `bench/api_benchmark.py`** (default `en`). Matches the existing
  flag on `bench/wikidata_e2e.py`. Threads through subprocess
  scenarios via argv (wiki_benchmark) and `KGLITE_BENCH_LANGUAGES`
  env (api_benchmark, which uses positional args). Pass
  `--languages ""` to keep all languages, but the canonical query
  suite expects English type names like `:human` and will return
  zero rows otherwise.
- **Post-build SANITY PROBE in `bench/wikidata_e2e.py`.** Quick Q42
  title/description lookup + top-5 type histogram printed after
  every build, even with `--no-queries`. Catches language-filter or
  auto-type-rename regressions immediately, before query-suite time.

## [0.8.15] — 2026-04-25

### Performance

- **Mapped-mode property index — `MATCH (n:Type {prop: val})` in O(log N).**
  `MappedGraph` now carries a lazy per-`(node_type, property)` and
  cross-type property index alongside the 0.8.15 conn_type index. On
  first `lookup_by_property_eq` / `lookup_by_property_prefix` /
  `*_any_type` hit the backend iterates nodes once, emits a sorted
  `(key, NodeIndex)` array — same layout as disk's persistent
  `PropertyIndex` — and caches it behind an `Arc<RwLock<…>>`;
  subsequent queries binary-search that array. Alias handling
  matches disk (reads from `node.title` for `title`/`label`/`name`
  and `node.id` for `id`/`nid`/`qid` so the `add_nodes(...,
  node_title_field=...)` pipeline "just works"). Invalidated on
  `add_node`/`remove_node`/`node_weight_mut`. Measured on a 938 k
  node / 212 k Q5 subset (wiki100m Wikidata humans):
  `MATCH (n:Q5 {title: 'Douglas Adams'})` — **37 ms first call
  (builds the index), 0.1 ms warm** (index hit). Prefix scans stay
  in the single-digit-ms range. Correctness pinned by
  `tests/test_mapped_property_index.py` against both memory and disk.

### Added

- **Cypher `count { <pattern> }` subquery expression.** `count { ... }`
  in `WITH` / `RETURN` / `ORDER BY` / `WHERE` now evaluates to the
  number of matches of the inner pattern, scoped to the outer row's
  bindings. Previously the parser rejected the shape with
  *"Expected property name or .property in map projection, got
  Some(LParen)"* because the identifier-followed-by-brace dispatch
  routed to map projection (`n { .prop1, .prop2 }`) unconditionally;
  the parser now special-cases `count` and routes to a new
  `parse_count_subquery` that mirrors the existing `EXISTS { ... }`
  grammar. New AST variant `Expression::CountSubquery`; the executor
  runs the pattern via the shared `PatternExecutor`, bindings-
  compatible with the outer row, with optional inline `WHERE`.
  Parity across all three storage modes verified by
  `tests/test_cypher_count_subquery.py`. Cypher shapes like
  `WITH a, count{(a)-[:REL]->()} AS n` now work out of the box.

### Performance

- **Mapped-mode query acceleration — lazy per-connection-type index.**
  `MappedGraph` was a bare `StableDiGraph` wrapper with none of the
  inverted indexes that make disk-mode fast (`conn_type_index_*`,
  `peer_count_*`, CSR sorted by type). Cypher queries that depend on
  those structures on disk did full-graph scans on mapped — 2-10×
  slower than disk despite every byte being in RAM. 0.8.15 adds a lazy
  `MappedTypeIndex` populated on first typed-edge query per connection
  type: CSR-style sorted source lists, per-peer count histograms, and
  per-source edge slices. Overrides `sources_for_conn_type_bounded`,
  `lookup_peer_counts`, and `count_edges_grouped_by_peer` on the
  mapped `GraphRead` impl; `filter_by_connection` (powering
  `where_connected`) now hoists the source list into a `HashSet` once
  per call instead of probing `edges_directed_filtered` per node.
  Measured on wiki1000m (1 B triples):
  - Cypher `P31 class counts`: **150 ms → 0.5 ms (300×)**.
  - Cypher `2-hop P31 + P279`: **180 ms → 0.9 ms (200×)**.
  - Cypher `P31 LIMIT 50`: **5.6 s → 0.8-1.1 s (5-7×)**, now **beats
    disk at every subset** (disk wiki1000m was 1.40 s).
  - Cypher `Q5 (human) lookup`: 77 ms → 50-54 ms (1.5×).
  - Fluent `traverse P31 out unlimited`: unchanged (74 ms; bare
    petgraph scan is already optimal for sparse-degree nodes and
    avoids the ~100 ns/call index-lookup overhead).
  Correctness preserved: same row counts across storage modes on
  every benchmarked query. Index is built lazily per conn_type,
  amortised across subsequent queries of the same type, and
  invalidated on any edge mutation.

- **Mapped-mode `load_ntriples` routes through the disk fast path.**
  Previously `storage="mapped"` fell through to `DirGraph::enable_columnar`,
  which iterates every node once, clones each property map into a `Vec`,
  and pushes row-by-row into per-column `MmapOrVec` instances that grow
  via `set_len` + remap; each schema extension additionally triggered
  `Arc::make_mut` store clones. On wiki50m (377 k nodes / 282 k edges)
  this ran at ~430 k triples/s with a 5.1 GB peak RSS. Mapped now shares
  the disk path's property-log + single-`columns.bin` pipeline:
  properties stream to a zstd-compressed log during Phase 1, Phase 1b
  replays the log once into a pre-allocated mmap, and a new second-pass
  links each node's `PropertyStorage` to the shared
  `Arc<ColumnStore>` by row_id. Measured (bench/wiki_benchmark_mapped):
  - wiki50m build: **116 s → 13.5 s (8.6× faster)**, peak RSS
    **5073 MB → 828 MB (6× less memory)**.
  - wiki100m build: **? → 29 s**, peak RSS **? → 1.5 GB** — now
    within 1% of disk-mode build time.
  - Same Cypher + fluent rowsets across modes (round-trip oracle in
    `tests/test_incremental_columnar.py::TestNTriplesColumnar`).
  Memory-mode N-Triples load (`storage=default`) is unchanged — still
  goes through the non-columnar `PropertyStorage::Map/Compact` path.

- **Fluent traverse + `where_connected` use CSR-filtered edge iterator.**
  `core/traversal.rs` (`make_traversal_fast`, `make_traversal_full`) and
  `core/filtering.rs` (`filter_by_connection`, i.e. `.where_connected()`)
  previously called `graph.edges_directed(node, dir)` and post-filtered on
  `connection_type`. On disk-mode graphs with `csr_sorted_by_type=true`
  (the `merge_sort` algo that the `wikidata_disk.py` example uses), we now
  pass the connection key into `edges_directed_filtered` so the DiskEdges
  iterator can binary-search the CSR range — O(log D) instead of O(D) plus
  per-edge `EdgeData` materialisation. This is the same fast path the
  Cypher executor has used since 0.8.0 and targets the shape that the
  Wikidata fluent suite regressed on: `select("Entity").traverse("P31",
  limit=100)` was ~71 s, `.where_connected("P31")` was ~887 s, and
  `traverse P31 in limit 50` was ~2171 s. Heap backends ignore the hint;
  correctness is preserved by the existing post-filter.

### Changed

- **`load_ntriples` verbose output is no longer per-type.** Phase 1b used
  to print one `N dense cols, M overflow cols` line per type with any
  overflow and one `overflow bag X MB for N sparse cols` line per type
  with a non-empty overflow bag. On Wikidata this was ~90 k lines. Both
  are now collapsed into a single Phase 1b summary per pass (`columns — N
  dense, M overflow across K types with sparse cols` and `overflow bags —
  X.X MB across N types, M sparse cols total`).
- **Load-phase progress lines throttled to time (≥ 15 s), not bucket.**
  The loader used to emit a progress line every 5 M triples, which was
  ~2× per second on a fast machine and scrolled the interesting phase-
  timing lines off screen. Now the 5 M bucket is still a cheap fast-loop
  counter but the line only fires when ≥ 15 s have passed since the last
  one. Lines now prefix `[T+NNNNs]` so they interleave cleanly with the
  existing phase-timing output.

### Fixed

- **`examples/wikidata_disk.py`** — the fluent cases called `.where_(...)`,
  but the PyO3 binding exposes the method as `.where(...)`; on a large
  graph this masked the real traversal slowness behind an
  `AttributeError`. Expanded the Cypher/fluent suites with 23 + 15 more
  diverse queries (typed 1-hop, 2-hop chains, parameter binding,
  `ORDER BY` on bounded scans, and string-prefix/contains filters). The
  fluent suite now introspects via `g.node_type_counts()` and skips
  unavailable types instead of raising.

## [0.8.14] — 2026-04-24

### Performance — disk-graph `kglite.load()` fast-load series

Four independent on-disk format changes aimed at the serde overhead
that dominates `kglite.load()` on large disk-mode graphs. Profiling
the 124 M-node, 863 M-edge Wikidata graph
(`wikidata_disk_graph_0.8.11`, 81 GB on disk) showed zstd
decompression + mmap setup account for only ~5–6 s of a ~77 s cold
load; the remaining ~70 s is serde rebuild cost on three bulk
structures, plus a 266 MB JSON array inside `metadata.json`. Each
format change replaces that cost with flat packed slices + exact
`HashMap::with_capacity` sizing — same in-memory representation,
zero consumer surface change.

- **`type_connectivity` out of `metadata.json` into a packed binary.**
  On the 81 GB graph this field was 266 MB of a 415 MB JSON file
  (3,176,503 `ConnectivityTriple` entries). The new
  `type_connectivity.bin.zst` at the graph root is:

  ```
  [ 0.. 8]  magic       = b"KGLTCN1\0"
  [ 8..12]  version     = u32 LE (= 1)
  [12..16]  num_entries = u32 LE
  [16..n*32+16]  entries: (u64 src_key, u64 conn_key, u64 tgt_key, u64 count) × n
  ```

  Keys are interner hashes (`InternedKey::as_u64()`). Disk-mode save
  strips the field from metadata.json; in-memory `.kgl` saves keep
  embedding it for single-file portability.

- **`type_indices.bin.zst` flat CSR binary, interner-keyed.** Replaces
  bincode `HashMap<String, Vec<NodeIndex>>` with three packed slices:

  ```
  [ 0.. 8]  magic       = b"KGLTIDX1"
  [ 8..12]  version     = u32 LE (= 1)
  [12..16]  num_types   = u32 LE
  [16..24]  total_nodes = u64 LE
  [24..24 + 8·num_types]        type_keys: [u64]
  [next..next + 8·(num_types+1)]  offsets:  [u64]   (CSR)
  [next..next + 4·total_nodes]   nodes:    [u32]
  ```

  HashMap capacity is sized exactly from `num_types`, and each type's
  `Vec<NodeIndex>` is built from a contiguous u32 slice rather than
  bincode's per-field serde calls.

- **`id_indices.bin.zst` per-variant flat binary.** Replaces bincode
  `HashMap<String, TypeIdIndex>` with:

  ```
  [ 0.. 8]  magic     = b"KGLIIDX1"
  [ 8..12]  version   = u32 LE (= 1)
  [12..16]  num_types = u32 LE
  per-type block:
    [ 0.. 8]  type_key:    u64 LE
    [ 8.. 9]  variant_tag: u8  (0 = Integer, 1 = General)
    [ 9..16]  padding:     [u8; 7]
    [16..24]  num_entries: u64 LE
    payload:
      Integer (tag=0):  keys: [u32], node_idxs: [u32]
      General (tag=1):  blob_len: u64 + bincode HashMap<Value, NodeIndex>
  ```

  The Integer variant dominates Wikidata-style graphs (Q-number ids
  strip to u32), so the bulk of the 997 MB decompressed bincode blob
  collapses to two flat u32 arrays per type.

- **`interner.bin.zst` replaces `interner.json`.** The hash→string
  JSON map becomes a zstd-compressed bincode `Vec<String>` of just
  the originals; hashes are re-derived deterministically on load via
  `get_or_intern`. On the 81 GB graph this drops from ~7 MB JSON to
  ~3 MB bincode and eliminates one JSON parse on the critical path.

- **`KGLITE_LOAD_TIMING=1` stage instrumentation.** Gated per-stage
  wall-clock timing in `load_disk_dir`; off by default (zero
  overhead). Emits one `[TIMING] stage=<name> dur_ms=<ms>` line to
  stderr per major phase (`metadata_json`, `interner_load`,
  `disk_graph_load`, `type_indices_load`, `column_stores_load`,
  `id_indices_load`, `type_connectivity_load`). Used as the
  measurement harness for the four format changes.

**Backward compatibility.** All four loaders fall back to the old
format on a missing file or magic-byte mismatch:

- Missing `type_connectivity.bin.zst` → loader reads embedded JSON
  from `metadata.json`, then derives from `connection_type_metadata`.
- `type_indices.bin.zst` without `KGLTIDX1` magic → old bincode
  `HashMap<String, Vec<NodeIndex>>` path.
- `id_indices.bin.zst` without `KGLIIDX1` magic → old bincode
  `HashMap<String, TypeIdIndex>` path.
- Missing `interner.bin.zst` → old `interner.json` path.

Graphs saved by 0.8.11 and 0.8.12 continue to load without a rewrite.
Re-saving an old-format graph with 0.8.13 produces all four new files
automatically.

**Non-goals / out of scope.** No change to query execution, pattern
matcher, mutation paths, `node_mut_cache` / F1 / F2 flow, segmented
CSR layout (`seg_NNN/`), or the `KGLCOLv1` sidecar format. In-memory
representation of every touched structure is byte-identical to
0.8.12 — zero possibility of query-side regression.

## [0.8.12] — 2026-04-24

### Fixed

Seven disk-backend correctness fixes stacked on top of the 0.8.11
segmented-CSR foundation. Five cover latent save/reload regressions
that slipped through 0.8.11's phase-1–8 coverage; two (F1 + F2) close
the pre-existing Cypher `SET` and `DETACH DELETE` mutation holes on
disk graphs.

- **`save_disk` no longer compacts overflow away before seal.**
  The previous `dg.has_overflow()` gate unconditionally compacted
  before `save_to_dir`, which cleared `overflow_out`/`overflow_in`
  and made the phase-6 seal path see empty overflow. Every edge
  added between saves was silently lost on reload. Gating the
  compact on "won't take the seal path" (manifest empty OR no
  tail above the sealed watermark) preserves overflow for seal
  and keeps the compact-rewrite semantics for the non-seal case.

- **Segment-local seals merge `conn_type_index_sources` with global ids.**
  `write_conn_type_index` walks the segment's segment-local
  `out_offsets` (indices 0..tail_len) and stored those local
  indices as source ids. Reload's merge needed to shift each
  entry by `node_lo[seg]` for segment-local seals (full-range
  seals already store global ids). Without this, post-reload
  `MATCH (a)-[:T]->(b) RETURN a.id, b.id` returned no rows even
  though `count(*)` via the histogram reported the correct total.

- **Compact-rewrite after a prior seal cleans up stale `seg_NNN`.**
  When `save_to_dir` falls to compact-rewrite (tombstones, edits,
  or pure edge mutations between existing nodes), it now removes
  every `seg_NNN > 0` under the target dir before rewriting
  `seg_000`. Without this, `enumerate_segment_dirs` picked up the
  stale sealed segments on reload and concat'd them against the
  fresh seg_000, double-counting nodes and edges.

- **Compact-rewrite persists heap-backed core arrays.**
  `reconcile_seg0_csr` (called inside seal) replaces
  `self.{node_slots, out_offsets, in_offsets, edge_endpoints}`
  with heap-backed `MmapOrVec::Heap` copies. The same-dir
  compact-rewrite previously relied on mmap persistence and
  skipped explicit writes, which left the on-disk files at the
  pre-seal trimmed sizes; reload errored with "File too small".
  `save_to_file` is now called unconditionally for every core
  array — it handles both backings.

- **New types added via `add_nodes` persist on disk save.**
  `DirGraph::save_disk` gated the per-type
  `columns/<type>/columns.zst` sidecar write on the absence of
  `columns.bin`. For disk graphs built via `load_ntriples`,
  `columns.bin` is always present, so the sidecar branch was dead
  code and every type added after the initial build lost its
  column data on reload (properties read back as `None`). Save
  now reads `columns_meta.json`/`.bin.zst` to identify the types
  already covered by `columns.bin` and emits sidecars for the
  remainder. Load path additively walks `columns/` after the mmap
  fast-path to pick them up.

- **Cypher `SET n.prop = X` on disk-backed graphs persists through save + reload.**
  Pre-fix, `DiskGraph::node_weight_mut` materialised a `NodeData`
  into `self.node_arena`; the Cypher executor mutated that arena
  copy; `clear_arenas` dropped it without writing back to the
  canonical `ColumnStore`. Affected SET + save + reload was a silent
  no-op for persistence across 0.8.10 and 0.8.11.

  Fix: mirror the proven `batch.rs::flush_chunk` full-`Arc`
  replacement pattern for exact-row mutations.
  `DiskGraph::node_weight_mut` now stages writes in a
  `node_mut_cache` (Map-backed `NodeData`); `clear_arenas` groups
  cached entries by type, deep-clones each affected `ColumnStore`
  **once**, applies every staged title / property write + DELETE
  tombstone to the clone, and replaces both
  `DiskGraph.column_stores[ty]` and (via
  `DirGraph::sync_column_stores_from_disk`)
  `DirGraph.column_stores[ty]` atomically. Avoids the
  `Arc::make_mut` → per-row clone + Arc divergence that doomed the
  earlier attempt. Title writes diff against the current stored
  value before calling `set_title` so that `TypedColumn::Str`'s
  in-place-update offset-corruption bug (pre-existing) doesn't
  trigger on unchanged titles.

- **`DETACH DELETE` on disk preserves surviving nodes' property
  values across save + reload.** Pre-fix, a disk-graph delete cycle
  corrupted `title` reads (garbage bytes) and returned `None` for
  some `age` values on reload. The surviving count and id set were
  always correct — only the column-store-backed property columns
  were affected. Root cause was in the sidecar load path, not in
  mutation routing: `load_column_sidecars` derived `row_count` from
  `type_indices[type].len()` (live rows only), while the sidecar
  blob retains tombstoned rows alongside live ones. The mismatch
  made `ColumnStore::load_packed` walk column blobs at the wrong
  offsets and decode offset bytes as string data.

  Fix: the sidecar `columns.zst` file now starts with an 8-byte
  `KGLCOLv1` magic tag followed by `ColumnStore::row_count` (u32
  LE) before the existing `write_packed` payload, and the loader
  uses that stored count. Old-format sidecars (no magic tag) fall
  through to the `type_indices.len()` derivation for backward
  compat — best effort for legacy graphs, correct for any graph
  saved by 0.8.12+. Locked by
  `test_detach_delete_property_persistence_disk` (was `_xfail`).

### Deferred to 0.8.13+

- **Planner pruning using `SegmentManifest`.** Summaries are
  populated and persisted since 0.8.11; the pattern matcher
  doesn't yet consult them. Initial exploration showed the win
  is small under the current concat-at-load architecture (typed-
  edge queries at wiki1000m already run at sub-ms when the
  histogram path applies, and concat'd reads are uniform).
  Kept as a future option for 200-segment workloads once those
  exist in practice.

## [0.8.11] — 2026-04-23

### Added (disk-graph-improvement-plan PR1, phases 1–8)

This release lands the segmented-CSR foundation on the disk backend
and then fills in the incremental-save and auxiliary-index work on
top of it. Net result: write amplification on incremental ingest
drops from 5–25× to ~2× (target from
`dev-documentation/disk-graph-improvement-plan.md`) while load/save
on Wikidata-scale graphs now beats 0.8.10 across every subset. Every
existing `.kgl` directory still loads byte-for-byte identically.

- **Segment manifest (`seg_manifest.json`).** On-disk JSON listing
  per-segment `node_id_range`, `edge_count`, `conn_types`,
  `node_type_counts`, and `indexed_prop_ranges` summaries. Future
  planner pruning consults this before scanning. Today populated
  as a single-segment descriptor on every save. New module
  `src/graph/storage/disk/segment_summary.rs`.

- **Segmented CSR directory layout (`seg_NNN/`).** The CSR
  binaries, ColumnStore, and per-(type,prop) property indexes now
  live under a per-segment subdirectory. Graph-level metadata
  (`disk_graph_meta.json`, `seg_manifest.json`, DirGraph
  metadata) stays at the graph root. Gated by
  `csr_layout_version` (`#[serde(default)] == 0`) so legacy flat-
  layout `.kgl` directories still load.

- **`segment_subdir(id)` + `enumerate_segment_dirs(root)`.** The
  directory name is now id-parameterised, and load walks a sorted
  `seg_NNN/` enumeration instead of a hardcoded path.

- **Multi-segment read path.** `SegmentCsr` bundles one segment's
  six core CSR arrays (`node_slots`, `out_offsets`, `out_edges`,
  `in_offsets`, `in_edges`, `edge_endpoints`);
  `concat_segment_csrs` stitches them by shifting segment-local
  `edge_idx` onto combined `edge_endpoints`, concatenating
  per-segment `node_slots` and `edge_endpoints`, and welding the
  offset arrays. Single-segment load stays on the direct-mmap
  path for zero overhead vs 0.8.10.

- **Multi-segment write path (`DiskGraph::seal_to_new_segment`).**
  Flushes the still-mutable tail
  (`[sealed_nodes_bound, node_count)` + overflow edges between
  those nodes) to a fresh `seg_NNN/` — with per-segment
  `conn_type_index_*`, `peer_count_*`, and `edge_prop_*`
  alongside the core CSR — appends a `SegmentSummary` to the
  manifest, clears consumed overflow, advances the watermark,
  and rewrites `disk_graph_meta.json`.

- **Full-range-offset mode for cross-segment seal.** The seal
  path accepts overflow whose source or target is below the
  watermark by emitting offsets that span every global node id
  rather than only the new segment's tail. `concat_segment_csrs`
  distinguishes the two modes per segment via
  `out_offsets.len() > node_slots.len() + 1` and unions
  contributions per-node. Lets general incremental ingest (not
  just new-nodes-only batches) take the seal path.

- **Automatic incremental save.** `save_to_dir` now delegates
  to `seal_to_new_segment` whenever a graph has a prior segment
  manifest and a tail above the watermark — the typical
  incremental-ingest shape. Second save on a 10-chunk build
  produces 10 segments instead of rewriting the entire tree
  each time.

- **Per-segment auxiliary indexes survive seal+reload.**
  Multi-segment reload now merges `conn_type_index_*`,
  `peer_count_*`, and `edge_prop_*` across all segments so
  typed-edge matches, `edge_weight()`, and `peer_count`-backed
  aggregates return correct results on sealed segments.

### Performance

- **Save/load regression on Wikidata-scale graphs undone.**
  0.8.11's initial segmented-CSR work regressed `save`/`load`
  6–22× on wiki100m–wiki500m because the
  `dir.join("columns.bin").exists()` guard in `DirGraph` and
  `io::file` didn't know about the phase-4 `seg_000/`
  relocation. Checking both locations restores the 0.8.10
  baseline — and beats it once the rest of the phase work lands
  (wiki500m build −23 %, save −10 %, load −12 % vs 0.8.10;
  wiki100m build −22 %, save −16 %, load −15 %).

- **`MATCH ()-[:T]->(c) RETURN c, count(*)` aggregations
  sub-millisecond at every scale.** The fused MATCH+RETURN
  aggregate path was running
  `PatternExecutor::execute(MATCH (c))` unconditionally to
  enumerate the group target before dispatching to the
  histogram top-K fast path — for an untyped group target,
  a 14.7 M-node full-graph scan on wiki1000m that the fast
  path never reads. Enumeration is now deferred to the one
  node-centric fallback branch that actually needs it.
  `P31 class counts` on wiki1000m: 3702 ms → 0.3 ms (12 300×).
  Same fix applied to `FusedMatchWithAggregate`:
  `WITH P27 count` on wiki1000m 5387 ms → 13 ms (408×), on
  wiki500m 426 ms → 6 ms (71×).

- **Edge-centric fast path for
  `MATCH (src:T1)-[:T]->(tgt) WITH tgt, count(src)`.** Phase
  3 routes this shape through the pre-built
  `peer_count_histogram` when the source-type filter is a
  no-op, or walks `conn_type_index` source lists otherwise —
  both O(|T-sources|) instead of O(|all nodes| × in-degree).
  On wiki500m the query drops from 1210 ms to 445 ms (the
  result is a stepping stone; the full win lands via the
  histogram-routing fix above).

- **Multi-key ORDER BY LIMIT on aggregated counts.** Phase 4
  extends the fused MATCH+RETURN aggregate path so
  `ORDER BY k DESC, c.title` no longer disables fusion.
  Fusion sets a `candidate_emit` descriptor; the executor
  picks the primary-key threshold via a heap and emits the
  qualifying superset (any tie-breaking on secondary keys
  happens in the unchanged downstream `OrderBy + Limit`).
  `P31 class counts` warm-cache on wiki500m dropped from
  1477 ms to 450 ms as a secondary effect, before the
  group-target-scan fix made it sub-millisecond.

### Fixed

- **CSR mmap files now trim in-place on `save_to_dir`.**
  `MmapOrVec::mapped(path, cap)` has a 64-element minimum, so
  small graphs left trailing zeros on disk. The single-segment
  load path masked this by using `meta.*_len`, but the new
  multi-segment load path relies on file-size inference. All six
  core CSR arrays now pass through `save_to_file` on the
  same-dir save path, triggering the `file.set_len(byte_len)`
  truncation. Same bug pattern as the 0.8.10 conn_type_index trim.

## [0.8.10] — 2026-04-20

### Performance

- **GROUP BY aggregation defers property materialization.** Queries of the
  shape `RETURN x.prop, count(*)` now hash by NodeIndex during the per-row
  pass and resolve the property once per resulting group, rather than once
  per input row. For high-fanout aggregations on disk graphs (e.g., walking
  Wikidata's 439K country=Norway entities and grouping by their `instance_of`
  type) this drops O(rows) random-I/O column reads to O(distinct groups).
  Cypher semantics are preserved by re-bucketing on resolved values after —
  two distinct nodes that share a property value still collapse into one
  group. Implementation in `src/graph/languages/cypher/executor/return_clause.rs`.

### Fixed

- **OPTIONAL MATCH + RETURN with PropertyAccess group keys no longer silently
  returns NULL groups.** The fused OPTIONAL MATCH + aggregation path
  evaluated group-key expressions against the source row (pre-OPTIONAL),
  so a query like `OPTIONAL MATCH (p)-[:OWNS]->(pet) RETURN pet.name,
  count(*)` would resolve `pet.name` to NULL for all rows, collapsing every
  result into one wrong group. The fusion check now rejects PropertyAccess
  on variables only bound by the OPTIONAL MATCH itself, falling through to
  the correct (non-fused) aggregation path. `is_fusable_return_clause` in
  `src/graph/languages/cypher/planner/fusion.rs` now takes the OPTIONAL
  MATCH variable set and rejects matching property accesses.

- **Multi-MATCH re-bind no longer full-scans the graph.** When a second
  MATCH clause re-bound a variable from a prior clause
  (`MATCH (f {id: X}) MATCH (f)-[:R]->(c)`), the pattern matcher's
  inverted-index fast path ignored the existing binding and returned
  every source node for the edge type — 20s+ timeouts on Wikidata-scale
  graphs. The fast path now skips when the first node is already bound,
  falling through to `find_matching_nodes` which resolves the variable
  to a single node. `{Gjøa, Norway}` goes from >20s timeout to ~36ms
  on the 124M-node Wikidata graph.

### Changed

- **Graph algorithm procedures (`CALL pagerank/degree/betweenness/
  closeness/louvain/label_propagation/connected_components`) now error
  on timeout instead of silently returning partial results.** Algorithm
  signatures changed to `Result<_, String>`; the `break`-on-deadline
  branches now return `Err`, and the new `algorithm_timeout_err()`
  message points users at `timeout_ms=N` / `timeout_ms=0`. Fixes silent
  half-converged results that looked successful.
- **`CALL` on graphs over 2M nodes now refuses unscoped procedure
  runs up front.** Prior to this, `CALL degree()` on Wikidata (124M
  nodes) ignored its `_deadline` parameter entirely and ran for
  minutes — long enough to exhaust MCP transport timeouts and appear
  to wedge the server. The new guard errors in <1ms with "would scan
  the whole graph. Subgraph scoping is not yet supported — try a
  smaller graph, or pass timeout_ms=0 to override this guard."
- **`degree_centrality` and `weakly_connected_components` now honor
  the 20s Cypher deadline.** Both previously ignored deadlines (the
  former via an unused `_deadline` parameter, the latter by not
  accepting one). Periodic checks every ~1M edges.

## [0.8.9] — 2026-04-20

### Changed

- **Streaming label journal replaces in-memory `label_cache` during
  `load_ntriples`.** The previous `HashMap<u32, String>` grew to ~10GB
  on Wikidata's 124M entities, collapsing streaming throughput from
  1.8M triples/s to 450K/s via swap pressure. Labels now spill to a
  sequential on-disk journal (`{spill_dir}/labels.bin`) — zero heap
  growth during Phase 1. The post-Phase-1 rename pass reads the
  journal once, filtering to the ~tens-of-thousands of Q-numbers
  that actually appear as type names (~3MB final footprint). New
  module: `src/graph/io/ntriples/label_spill.rs`.

### Fixed

- **Typed `MATCH (n:Type {title: 'X'})` now takes the cross-type
  global-index fast path.** Previously only untyped patterns consulted
  the global index; typed patterns fell through to a full-type scan —
  10–14s (and frequent timeouts) on 13M-row types like Wikidata
  `human`. The matcher now consults the global index and post-filters
  by `node_type_of(idx)`, dropping `MATCH (n:human {title: 'Barack
  Obama'})` from 14s to ~25ms. Alias-aware (title↔label↔name).
- **Per-type `{nid: ...}` / `{qid: ...}` anchors hit the id index.**
  Both typed and untyped paths previously only checked the literal
  `"id"` key, so alias queries fell through to full scans. Now
  `id`/`nid`/`qid` all anchor via the same per-type id_index.
- **String-form id anchors (`{nid: 'Q76'}`) hit the id index.**
  `TypeIdIndex::get` now coerces `"Q76"` → `UniqueId(76)` by stripping
  the leading alpha prefix. Works for any `[A-Za-z]+[0-9]+` id
  scheme (Wikidata Q-codes, P-codes, E-codes, ...). Previously the
  lookup fell through to a full-type scan, so
  `MATCH (a:human {nid: 'Q76'})-[r]-(b:human {nid: 'Q13133'})`
  dropped from ~14s to ~300ms on Wikidata. Also fixes the
  correctness bug where `MATCH (a {id: 'Q76'})` silently returned
  `0` rows instead of the matching node.

## [0.8.8] — 2026-04-19

### Fixed

- **EXISTS inline-property filters on target nodes were silently
  dropped.** `WHERE EXISTS { (a)-[:REL]->({id: 20}) }` used the fast
  path's `get_property("id")` which missed the special id_column,
  producing silent zero-row results even when the pattern genuinely
  matched. Ported the same alias resolution that
  `node_matches_properties` uses — `title`/`name`/`label`/`id`/`type`
  all route to the right column via `resolve_alias`. The fast path
  now behaves identically to the slow path for these literal-property
  checks. Regression tests added to `test_where_exists.py`.

### Added

- **Cross-type global property index.** New `create_global_index(property)`
  builds a single mmap'd sorted-string index covering every live node,
  not just one type. On a disk graph, `save()` now auto-builds a global
  title index so `MATCH (n {title: 'X'})` — without a type label — is
  O(log N) out of the box. Solves the "title-to-ID without guessing
  the type" problem that agents hit repeatedly on Wikidata-scale graphs.
  Files: `global_index_{property}_{meta,keys,offsets,ids}.bin`.
- **`g.search(text, property='title', limit=10)` helper.** Returns
  the top-k nodes whose `property` matches `text` (exact match first,
  then prefix fallback) as `[{id, type, title, id_value}]`. Backed by
  the global index. Also exposed as a new MCP tool so agents can skip
  the "guess the type" ceremony entirely: `search('Equinor')` returns
  the right Q-number without `MATCH` or a type label.
- **Alias-aware cross-type lookups.** When the untyped matcher sees
  `{title: 'X'}` and the literal `title` index doesn't exist, it also
  tries the hardcoded title family (`title/label/name`) AND any
  per-type aliases declared via `node_title_field=` at `add_nodes`
  time. Same for `id/nid/qid`. An agent who built the index as
  `create_global_index('label')` still hits the fast path when
  querying `{title: 'X'}`, and vice versa. Derivation is automatic
  from the graph's existing schema — no new config API.

### Changed

- **`save_disk` auto-builds the global title index.** Every call to
  `save()` on a disk graph now produces `global_index_title_*.bin`
  files. Adds a one-pass sweep over `node_slots` at save time —
  negligible on small graphs, ~single-digit minutes on Wikidata-scale
  (124M nodes). Opt-out: delete the files after save.

## [0.8.7] — 2026-04-19

### Added

- **`WHERE n.prop STARTS WITH 'prefix'` now pushes down into the MATCH
  pattern** and routes through the persistent disk prefix index when
  available. New `PropertyMatcher::StartsWith(String)` variant, new
  `apply_prefix_to_patterns` helper in
  `src/graph/languages/cypher/planner/index_selection.rs`, new path in
  `matcher.rs::try_index_lookup` that calls
  `GraphRead::lookup_by_property_prefix`. String indexes are annotated
  `indexed="eq,prefix"` in `describe()` output (previously just `eq`);
  numeric indexes remain `indexed="eq"` only.
- **Deadline polling inside unanchored matcher scans.** Three hot
  loops in `matcher.rs` that used `.filter().collect()` over
  13M+-node type lists now poll the deadline every 4096 rows (via a
  new `check_scan_deadline()` helper with a structured hint message).
  Worst-case overshoot past the deadline drops from 20-60+ s to under
  a few ms. Other scan loops (variable-length paths, CSR edge
  counting, column stats) already polled; this closes the final gaps.
- **MCP `cypher_query` tool accepts `timeout_ms`.** `examples/mcp_server.py`'s
  tool signature now exposes the override so agents can deliberately
  extend or disable the deadline (`timeout_ms=0`) per call after an
  EXPLAIN confirms the plan is anchored. Previously the MCP agent was
  stuck with the backend-aware default.
- **`ResultView.diagnostics` — lightweight execution diagnostics.** Every
  `cypher()` call now attaches an always-on diagnostics dict to the
  returned `ResultView` with `elapsed_ms`, `timed_out`, and the
  `timeout_ms` that was in effect. Gives agents immediate feedback on
  query cost and timeout state without requiring `PROFILE`. The field
  is ``None`` for mutation paths, EXPLAIN, and transaction queries.
- **`describe()` indexed-property annotations.** Properties covered by
  an index (in-memory `property_indices` *or* the new persistent disk
  `PropertyIndex`) are now emitted with an `indexed="eq"` attribute in
  the `<properties>` detail block. A new `<indexing>` hint inside
  `<extensions>` explains the annotation and reminds agents to prefer
  anchored queries over unanchored scans on disk-backed graphs. New
  helper `DirGraph::has_any_index(node_type, property)` consolidates
  the "in-memory or persistent" check.
- **Persistent disk-backed property index.** `create_index('T', 'label')`
  on a `storage='disk'` graph now writes four mmap'd files
  (`property_index_{type}_{property}_{meta,keys,offsets,ids}.bin`)
  next to the CSR instead of rebuilding a `HashMap<Value, Vec<NodeIndex>>`
  on every `load()`. The previous in-memory path consumed ~1-3 GB of
  heap on 13M-row types and made `create_index` effectively unusable
  on Wikidata-scale disk graphs. The new persistent index is lazy-loaded
  on first query after reopen, keys are sorted lexicographically (so
  both equality and prefix can share the same structure), and the
  Cypher planner consults it via a new `GraphRead::lookup_by_property_eq`
  trait method. `MATCH (n {label: 'X'})` now hits the index on disk in
  O(log N + k). Supports string columns and title aliases
  (`node_title_field` at `add_nodes` time — `label`, `name`, etc.).
  Numeric equality and `STARTS WITH` pushdown are follow-ups. In-memory
  graphs are unchanged (keep the existing `property_indices` HashMap).
  The `create_index` return dict grows a `persistent: bool` field
  indicating whether the disk path was taken.
- **Cypher schema validation** at plan time — catches typos in
  pattern-literal property names (`MATCH (n:Person {agee: 30})`) before
  the executor commits to a scan. Returns a `Did you mean 'age'?` hint.
  Runs in O(clauses) against `node_type_metadata`; skipped when a graph
  has no declared schema. Pattern-literal properties are the only v1
  target — unknown node types, connection types, and WHERE/RETURN
  `n.prop` accesses deliberately pass through (existence-check queries
  and virtual columns would otherwise false-positive). Phase 3 will
  surface those as non-fatal diagnostics.

### Changed

- **`cypher()` default timeout is now backend-aware.** Disk-backed
  graphs default to 10 s, Mapped to 60 s, Memory to no deadline. Users
  can override per-call via `timeout_ms=N` or globally via
  `set_default_timeout(ms)`. The documented escape hatch
  `timeout_ms=0` disables the deadline entirely. Previously,
  disk-backed queries without an explicit `timeout_ms` ran until the
  harness killed them; the new default returns a structured timeout
  error after 10 s with hints pointing at anchoring / index usage.
  (Also applies to transaction-level `cypher()`.)
- **Cypher timeout error message now carries remediation hints.**
  Replaces the bare string `Query timed out` with guidance on
  anchoring queries, raising `timeout_ms`, or using the `timeout_ms=0`
  escape hatch.
- **`set_default_timeout(None)` behaviour updated.** Passing `None`
  now falls through to the backend-aware default rather than meaning
  "no timeout". Pass `0` for the old behaviour explicitly.

## [0.8.6] — 2026-04-19

### Performance

- **`describe(connections=['T'])` fast path on disk graphs.** Rewrote
  `write_connections_detail` to use the persisted `conn_type_index_*`
  inverted index instead of three full `edge_references()` sweeps. The
  previous path materialised every visited edge into a per-query
  `edge_arena` that was never cleared mid-call, growing VSZ linearly
  with scanned edges — on Wikidata (863 M edges) a single
  `describe(connections=['P31'])` call was SIGKILLed by the kernel
  after exhausting VM. The new path:
  - Reads pair counts from `type_connectivity_cache` when populated
    (zero edge I/O).
  - Skips the property-stats scan entirely when the connection type's
    metadata declares no edge properties.
  - Walks only matching edges via the inverted index, capped at two
    samples via an early-exit callback.
  - Measured on Wikidata (`wikidata_disk_graph_p12rebuild`, 122 M
    nodes, 863 M edges, cold page cache):
    - `describe(connections=['P170'])` (1.3 M edges): 108 s → **0.24 s**
      (~450× faster; previous in-flight code held VSZ at +27 GB
      after 90 s without completing).
    - `describe(connections=['P31'])` (122 M edges): **0.25 s** (was
      SIGKILLed by OOM killer before this change).
    - `describe(connections=True)` unchanged at ~0.15 s.

### Changed

- **`describe(connections=['T'])` pair-breakdown now capped at 50
  entries by default** (sorted by count desc), overridable via a new
  `max_pairs` keyword argument. Wide fan-out connection types like
  Wikidata's `P31` have tens of thousands of distinct
  `(src_type, tgt_type)` pairs — P31 alone has 191 k — which produced
  ~13 MB of XML that overshot typical MCP response budgets. The cap
  emits `<endpoints total="N" shown="…">` plus a trailing
  `<more pairs="…" edges="…"/>` marker so agents see both the
  dominant relationships and the exact size of the tail. P31 output
  drops 13 MB → ~4 KB by default; pass `max_pairs=500` (or similar)
  to drill into the full distribution on demand.

### Added

- `GraphBackend::for_each_edge_of_conn_type` — monomorphic closure
  iterator yielding `(src, tgt, edge_idx, properties)` per match. On
  disk uses the inverted index and never materialises `EdgeData`; on
  Memory/Mapped filters petgraph's resident `edge_references`. The
  callback returns `bool` so callers can stop after a bounded prefix.
- `DiskGraph::edge_properties_at(edge_idx)` — borrow an edge's
  property slice without going through the `materialize_edge` arena.
- `describe(..., max_pairs=<int>)` keyword argument — controls the
  pair-breakdown cap described above. `None` (default) resolves to 50.

## [0.8.5] — 2026-04-19

Internal: test coverage, SAFETY docs, storage module reorganization.

## [0.8.4] — 2026-04-19

### Performance

- **Correlated-equality pushdown in the Cypher planner.** `WHERE cur.prop =
  prior.other_prop` — where `prior` is a node bound by an earlier MATCH —
  now pushes onto the current MATCH's pattern as a new `EqualsNodeProp`
  matcher that the executor resolves per-row via the bound node's
  property. When the probe-side property is indexed, the pattern executor
  then picks an indexed lookup instead of scanning all nodes of that
  type. Also pushes `cur.prop = scalar_var` (where `scalar_var` is
  projected by a prior WITH/UNWIND) as `EqualsVar`. WHERE stays as a
  safety-net filter. Fallback: unchanged behavior when no index exists.
- **`add_connections(query=...)` now runs the planner.** Previously, the
  query path in `add_connections` went straight from parse → execute,
  skipping the entire planner — so no pushdowns (equality, IN,
  comparison), no spatial-join fusion, no LIMIT/DISTINCT pushdown. It
  now calls `cypher::optimize` like `g.cypher()` does. Combined with
  the correlated-equality pushdown, the Sodir prospect graph's derived
  connections (Phase 7) now build **~8.5× faster** — 29.6 s → 3.5 s:
  - 7a HC_IN_FORMATION (3 UNION ranks): 11.0 s → 1.2 s (~9×)
  - 7b/7c StructuralElement ENCLOSES: 0.6 s → 0.05 s (fuse_spatial_join
    now fires here)
  - 7d PLAY_HAS_FORMATION (primary + fallback): 17.3 s → 1.5 s (~11×)

## [0.8.3] — 2026-04-19

### Performance

- **Spatial-join operator for `MATCH (s:A), (w:B) WHERE contains(s, w)`.**
  A new planner pass (`fuse_spatial_join`) rewrites this two-pattern
  containment shape into `Clause::SpatialJoin`, bypassing the cartesian
  product. The executor builds an R-tree over the container side (via
  the new `rstar` dependency), iterates the probe side once, and emits
  only matching (container, probe) pairs — `O((N+M) log N + K)` rather
  than `O(N·M)`. Speedups on `tests/bench_spatial.py` (release build):
  - `contains 500K pairs` (500 polygons × 1 K points): 86.96 ms → 0.52 ms (**~167×**)
  - `contains 2.6M prospect_shape` (263 complex polygons × 10 K points):
    480.51 ms → 3.32 ms (**~145×**)
  - `contains 100K pairs`: 17.65 ms → 0.55 ms (~32×)
  - Complex polygons (50 vertices): 18.29 ms → 0.24 ms (~76×)

  Fires when both types have `SpatialConfig` (container needs `geometry`,
  probe needs `location`), the two patterns are disjoint typed nodes
  with no edges, and the WHERE is `contains(var, var)` optionally ANDed
  with a residual predicate. Other shapes (`NOT contains`, constant-point
  `contains(a, point(…))`, intra-pattern edges, three-plus patterns,
  disjunctions) fall back to the existing per-row fast path unchanged.

## [0.8.2] — 2026-04-19

### Changed

- **Blueprint loader rewritten in Rust.** `kglite.from_blueprint()` now runs
  entirely in a new `src/graph/blueprint/` module (schema + CSV reader +
  filter DSL + geometry + timeseries + build orchestrator). `pandas` is no
  longer touched during ingestion — CSVs are parsed with the `csv` crate
  straight into the internal columnar `DataFrame`, then handed to
  `mutation::maintain::add_nodes` / `add_connections`.
  - Every parallelisable phase is pipelined: CSV pre-parse, per-spec
    prep (filter + geometry + typed-column build), FK edge DataFrames,
    and junction edge DataFrames all run across threads via `rayon`.
    Only the graph mutation calls (`add_nodes` / `add_connections`)
    stay serial, because the graph is `&mut`. GeoJSON→WKT centroid
    extraction is also parallelised per row.
  - The Python shim (`kglite/blueprint/__init__.py`, ~60 lines) now
    only handles optional save + schema lock on top of the native
    build. The old 831-line `kglite/blueprint/loader.py` is deleted.
  - Sodir blueprint (564 K nodes, 759 K edges): **9.87 s → 1.6 s** (~6×).
  - Node / edge counts match the previous Python loader exactly (parity
    verified per-type across all 90 node types and all 93 edge types).
  - New runtime deps: `csv`, `geojson`, `indexmap` (the last so node /
    sub-node iteration preserves blueprint JSON order, which in turn
    keeps edge counts byte-identical to the old loader).
  - Set `KGLITE_BLUEPRINT_PROFILE=1` for a per-phase / per-sub-phase
    ms breakdown on stderr.

## [0.8.1] — 2026-04-19

### Changed

- **`code_tree` rewritten in Rust.** The polyglot codebase parser previously
  implemented in Python (`kglite/code_tree/*.py`, ~7,500 LOC) is now a
  first-class Rust module (`src/code_tree/`) exposed via PyO3. All eight
  language parsers (Python, Rust, TypeScript/JavaScript, Go, Java, C#, C,
  C++) plus the builder and manifest readers run natively. Tree-sitter
  grammars are bundled into the native extension — no optional dependency
  needed. **`pip install kglite[code-tree]` is no longer required**; the
  `[code-tree]` extras entry has been removed.
- **abi3 wheel — one wheel per platform, Python 3.10+.** PyO3's
  `abi3-py310` stable-ABI target is now enabled, collapsing the CI wheel
  matrix from 20 wheels (5 Python versions × 4 platforms) to 4. Users on
  any Python ≥ 3.10 install the same wheel.
- **Parallel parsing via rayon + thread-local parsers.** File-level
  parsing runs across CPU cores with one tree-sitter `Parser` per thread
  (via `thread_local!`) — no `Mutex` contention.
- **Parallel CALLS-edge tier resolution.** The 5-tier name-matching pass
  (84 K functions → 200 K edges) now runs in parallel via rayon; each
  function's edges are independent.
- **Aho-Corasick for USES_TYPE.** The multi-pattern type-name scan
  replaces a giant regex alternation with an Aho-Corasick automaton,
  yielding ~2.5× faster `USES_TYPE` edge building on Java-scale corpora.
- **End-to-end performance on real repos:**
  - duckdb C++ (2,805 files): **29 s (Python) → 0.63 s (Rust)** — ~46×
  - neo4j Java (7,966 files, 84 K functions): **crashed in Python →
    1.69 s in Rust**
  - KGLite mixed Py+Rust (248 files): 0.17 s
- **New `code_tree` module shape.** `kglite/code_tree/__init__.py` is a
  4-line shim importing from the native `kglite._kglite_code_tree`
  submodule. The previous Python modules under `kglite/code_tree/` have
  been removed.

### Fixed

- **`build()` no longer crashes on pure-Java repos** (e.g. `neo4j/neo4j`)
  with `Source type 'Struct' does not exist in graph`. Edge routing now
  picks source/target node types per-row from the graph schema rather
  than defaulting to hardcoded names.

## [0.8.0] — 2026-04-18

Internal-only storage-architecture refactor plus a handful of disk-mode
bug fixes and large performance wins. **No Python API signature changes.**
`kglite/__init__.pyi` signatures are byte-identical to v0.7.17 (`git diff
v0.7.17 HEAD -- kglite/__init__.pyi` contains only docstring additions).
Users upgrading from 0.7.x will see no behavioural differences other
than the fixes and performance gains listed below.

### Fixed

- **Concurrent `load_ntriples` calls no longer wipe each other's spill
  directories.** The previous cleanup logic deleted *all other*
  `kglite_build_*` directories in `/tmp` at every ingest start. Two
  `load_ntriples` calls running at the same time (e.g. a long Wikidata
  build and a small test suite) would kill each other's property-log
  files, causing the in-flight build to crash at Phase 1b with `No
  such file or directory`. The cleanup now only removes spill dirs
  whose contents haven't been modified in the last hour, so active
  builds are always safe.
- **`save()` on Wikidata-scale disk graphs no longer appears to hang.**
  On large disk graphs (e.g. 124 M nodes / 88 K types from N-Triples
  ingest), `save()` was iterating `self.column_stores` and writing
  each type's columnar data as a separate `columns/<type>/columns.zst`
  zstd file — a multi-hour serial loop, redundant because the v3
  single-file `columns.bin` (written during Phase 1b of the N-Triples
  builder) already contains everything the loader needs. `save_disk`
  now skips the per-type loop when `columns.bin` exists on disk, and
  reduces to the metadata flush it was always meant to be.
  **Measured on Wikidata (124 230 686 nodes, 862 810 243 edges,
  88 931 types): `save()` went from ≥ 60 min → 5.52 s.** In-memory
  graphs persisted to disk still write the per-type files as before
  (that path never produces `columns.bin`).
- **Disk-mode `add_nodes(conflict_handling="update")` now applies
  property updates.** Previously on disk graphs, re-inserting an
  existing node via `add_nodes(..., conflict_handling="update")`
  silently dropped the new values — `node_weight_mut` materialised
  `NodeData` into a per-query arena that `clear_arenas` discarded
  before the next read, so the mutation never reached
  `DiskGraph::column_stores` where reads happen. The batch-update path
  now mutates the per-type column store directly via `Arc::make_mut`
  and re-syncs with `sync_disk_column_stores` at the end of the chunk.
  Memory and mapped graphs are unaffected (they already worked).
- **Disk-mode `add_nodes(conflict_handling="replace")` now clears
  omitted properties.** Same root cause as above; Replace now nulls
  out every previously-set property on the row before writing the new
  set, matching the `PropertyStorage::replace_all` semantics of the
  heap backends.
- **Disk-mode `MERGE` edges are visible to subsequent `MATCH`
  queries.** `DiskGraph` used to default `defer_csr = true` so every
  `add_edge` on a fresh graph queued into `pending_edges`, which
  `edges_directed` never reads. One-off Cypher mutations now route
  directly to the overflow buffer (visible immediately); bulk loaders
  (`add_connections`, ntriples) still use the pending+rebuild path via
  `build_csr_from_pending`.

### Performance

- **Cypher query primitives faster across the board vs v0.7.17** (N=20
  trials, macOS dev box). Memory and mapped modes both win:
  - `pattern_match` at 10 k nodes: −60 %
  - `two_hop_10x`: −24 %
  - `describe()`: −21 % (memory), −24 % (mapped)
  - `pagerank`: −17 %
  - `find_20x`: −13 % (mapped)
  - Construction sweep (1 k / 10 k / 50 k nodes): −11 % to −22 %
  - No memory-mode query regressed above the +5 % gate; only four cells
    flagged under 5 % (`find_20x_memory` +4.7 %, `simple_filter` /
    `multi_predicate` minor noise).
- **N-Triples disk-graph build is 2.5 % faster on Wikidata.** Added
  `#[inline(always)]` on the hot `GraphBackend` → `GraphRead`/`GraphWrite`
  trampolines (`node_type_of`, `edge_endpoint_keys`, `edge_endpoints`,
  `node_weight`) and a new closure-based
  `GraphBackend::for_each_edge_endpoint_key` that bypasses the
  boxed-iterator virtual-dispatch on hot edge iteration. Phase 1b
  (columnar write) −86 s, Phase 2 (edge creation) −32 s, Phase 3
  (CSR build) −55 s on a 7.65 B-triple / 862.8 M-edge build. Total
  `load_ntriples`: 4747 s → 4627 s.
- **`rebuild_caches()` is 28 % faster on large disk graphs.** Two
  fixes: (a) `compute_type_connectivity` is now Rayon-parallel on the
  disk backend — shards the edge range across all cores and merges
  per-shard HashMaps serially, matching `build_peer_count_histogram`'s
  pattern; (b) removed a `madvise(DONT_NEED)` call at the end of
  `build_peer_count_histogram` that was evicting the 13.8 GB
  `edge_endpoints` from page cache right before
  `compute_type_connectivity` had to re-read it. Also reordered
  `rebuild_caches` to run `compute_type_connectivity` first so its
  sequential sweep warms the cache for the histogram builder. Measured
  on Wikidata (862.8 M edges): 235 s → 169 s. Memory and mapped modes
  unaffected (serial path retained).

### Changed

- **Deterministic `.kgl` v3 saves.** `save()` now produces byte-identical
  output for identical graphs regardless of per-process HashMap
  randomisation. `write_graph_v3` iterates `column_stores` in sorted
  order and canonicalises the metadata JSON (object keys sorted). Old
  `.kgl` files load unchanged — the format on the wire is a strict
  subset of the previous format's possible outputs. Enables byte-level
  golden-hash format-drift tests.
- **`ConnectionTypeInfo` serialises with sorted keys.** `source_types`
  and `target_types` (HashSet<String>) and `property_types`
  (HashMap<String, String>) now emit in lexicographic order, hardening
  the v3 golden-hash invariant for fixtures richer than single-element
  sets. Existing `.kgl` files load unchanged.

### Changed (internal, not user-visible)

- **Internal reorganization — `src/graph/` split into domain
  subdirectories.** Code previously flat in `src/graph/` now lives
  under `algorithms/`, `languages/cypher/`, `features/`,
  `introspection/`, `io/`, `mutation/`, `pyapi/`, `core/` (shared
  primitives, was `query/`), and `storage/`. `storage/` further splits
  into `memory/`, `mapped/`, and `disk/` per-backend folders. Pure
  file moves via `git mv` (rename similarity 97–100 %; git blame
  preserved). Filenames cleaned of redundant prefixes / suffixes
  (`pymethods_*` → `*`, `filtering_methods` → `filtering`, etc.). See
  `ARCHITECTURE.md` for the final layout.
- **Every `.rs` under `src/graph/` is now at or under the 2,500-line
  hard cap.** The Phase 9 split carved nine god files (12,144-line
  `executor.rs` down through the 2,610-line `pattern_matching.rs`)
  into themed submodules. `GOD_FILE_EXCEPTIONS` is empty;
  `test_god_file_gate` passes unconditionally.
- **`MappedGraph` promoted to a distinct struct** (was a type alias
  for `MemoryGraph` pre-Phase 5). Per-backend `impl GraphRead` /
  `impl GraphWrite` land in `src/graph/storage/impls.rs`, setting up
  future backend-specific optimizations without breaking callers.
- **`RecordingGraph<G>` ships as a Rust-only validation wrapper.**
  Generic over any `G: GraphRead`, logs every read-path method call.
  Used internally to prove the architecture is actually open/closed —
  adding a new backend is a 3-src-file change. Not exposed to Python.
  See `docs/adding-a-storage-backend.md` for the worked example.
- **Testing envelope hardened.** New parity tests cover zero-node /
  single-edge / 1 000-hop / Unicode / type-promotion / null-NaN /
  100 000-row cypher results across memory / mapped / disk
  (`tests/test_edge_cases_parity.py`). Golden-fixture regression suite
  (`tests/test_golden.py` + `tests/golden/`) pins byte-exact output
  for a deterministic 1 000-node / 3 000-edge graph across every
  storage mode. New `@pytest.mark.stress` tier for the 30 GB mapped
  bench and 10 k-hop traversal.
- **Unsafe-block hygiene.** All 40 `unsafe { ... }` blocks in `src/`
  carry `// SAFETY:` justifications. A module-level invariants block
  at the top of `src/graph/storage/mapped/mmap_vec.rs` documents the
  shared mmap safety contract.
- **Python API docstring clarification.** The `find()` docstring now
  warns that it searches only code-entity node types (`Function`,
  `Struct`, `Class`, `Enum`, `Trait`, `Protocol`, `Interface`,
  `Module`, `Constant`). The signature is unchanged.
- **Deprecated `TempDir::into_path()` calls** migrated to
  `TempDir::keep()` per tempfile 3.14+ API.
- **`pub type Graph = GraphBackend` alias dropped.** Every call site
  uses `GraphBackend` directly. Removes a hygiene wart flagged in the
  Phase 9 report-out.
- **RecordingGraph audit methods (`log`, `log_len`, `drain_log`) are
  now `#[cfg(test)]` rather than `#[allow(dead_code)]`.** Release
  builds no longer compile these helpers at all.

## [0.7.17] — 2026-04-17

### Added
- **Python 3.14 wheels**. CI test matrix and `build_wheels.yml` now cover
  3.14 across Linux/macOS (Intel + arm64)/Windows, alongside 3.10–3.13.
  Full test suite passes on 3.14 (1758 tests, same as 3.12 minus the
  optional `code-tree` tests that require tree-sitter wheels). pyo3 0.28
  (shipped in 0.7.16) enables this via `ABI3_MAX_MINOR = 14`.

## [0.7.16] — 2026-04-17

CI-fix release on top of 0.7.15. No functional changes to the Cypher
engine; dependency bumps + clippy 1.95 compatibility only.

### Dependencies
- **pyo3 0.27 → 0.28**, **geo 0.29 → 0.33**, **wkt 0.11 → 0.14**,
  **bzip2 0.5 → 0.6**. API changes absorbed: `#[pyclass(skip_from_py_object)]`
  on `KnowledgeGraph` (pyo3 0.28 opt-in); `Geodesic` is now a static value
  (call as `Geodesic.distance(...)` / `length(&Geodesic)`) with the
  `LengthMeasurable` trait imported from `geo::line_measures`.
- Clippy 1.95 compat: `sort_by` → `sort_by_key(Reverse)`, collapsed `if`/
  `match` guard patterns, `file_len.checked_div(elem_size)`, removed
  redundant `.into_iter()` in `IntoIterator` args.

## [0.7.15] — 2026-04-17

### Added
- **`WHERE n:Label` predicate**. Cypher now supports label checks as boolean
  predicates (not just MATCH-level filters). Composes with `AND`/`OR`/`NOT`
  and chained `n:A:B` form (`n:A AND n:B`). Example:
  `MATCH (n) WHERE n:Person OR n:Org RETURN count(n)`.
- **`Value::as_str() -> Option<&str>`**. Borrowing companion to the existing
  `as_string()`. Prefer when ownership is not required — avoids the per-call
  `String` clone.

### Changed
- **Function names lowercased at parse time** instead of per-row during
  dispatch. Every Cypher scalar/aggregate dispatch used to call
  `.to_lowercase()` on the function name each time it evaluated a row
  (21+ sites); names are now normalized once in `parse_function_call` and
  compared directly. Pure CPU win on function-heavy queries.
- **`count(DISTINCT n)` uses typed identity sets** — `HashSet<usize>` keyed
  on node/edge indices (with a `HashSet<Value>` fallback for non-binding
  expressions) instead of per-row `format!("n:{}", idx.index())` string
  formatting. ~20–26% faster on DISTINCT-count queries.
- **`substring()` skips intermediate `Vec<char>`** — uses `chars().skip(start)
  .take(len).collect()` instead of materializing the full char vector.
  ~10–18% faster on substring-heavy queries.
- **Zero-allocation property iterators**. `PropertyStorage::keys()` and
  `::iter()` return explicit `PropertyKeyIter` / `PropertyIter` enums instead
  of `Box<dyn Iterator>`. Saves one heap allocation per `keys(n)` /
  `RETURN n {.*}` / property-scan call. ~10% faster on `keys(n)` over
  all nodes.

### Fixed
- **`HAVING` with aggregate expressions**. `HAVING count(m) > 1` was silently
  returning zero rows when the RETURN item was aliased (`count(m) AS c`).
  Root cause: the aggregate function call fell through to per-row scalar
  dispatch, which errored with "Aggregate function cannot be used outside
  of RETURN/WITH", and the error was swallowed by `unwrap_or(false)`,
  dropping every row. Now `HAVING count(m)` and `HAVING c` both resolve
  to the pre-computed aggregate value regardless of aliasing. Unaliased,
  `DISTINCT`, and no-group-by forms all covered.
- **`rand()` / `random()` correctness under tight loops**. The previous
  SystemTime-per-call seeding could return identical values for adjacent
  rows when the system clock resolved two calls to the same nanosecond,
  and constant folding could collapse `rand()` to a single value for the
  whole query. Replaced with a thread-local xorshift64 PRNG, seeded once
  per thread with a splitmix64-avalanched counter (so parallel Rayon
  workers don't collide), and marked as row-dependent so it bypasses
  constant folding. Also uses the top 53 bits of state for full f64
  mantissa precision.

## [0.7.14] — 2026-04-17

### Added
- **Per-(conn_type, peer) edge-count histogram** as a persistent disk cache. Built once at CSR-build time (parallelised via Rayon, single sequential scan of `edge_endpoints.bin`), stored as three flat `peer_count_*.bin` files. Unanchored aggregate queries like `MATCH (a)-[:TYPE]->(b) RETURN b.title, count(a) ORDER BY cnt DESC LIMIT N` now return in ~ms instead of scanning the full 13 GB `edge_endpoints` array. Rebuildable on existing disk graphs via `g.rebuild_caches()` without a full graph rebuild.
- **`FusedCountAnchoredEdges` planner rule + executor**. `MATCH (var)-[r:TYPE?]->({id: V}) RETURN count(var)` (and the three symmetric variants) is now fused into O(log D) CSR offset arithmetic. The anchor is resolved to a `NodeIndex` at plan time via `graph.id_indices`. Combined with the tombstone short-circuit (below) this turns hub-node count queries (e.g. ~40 M incoming edges on Q5) from 100 s TIMEOUTs into sub-second lookups.
- **Tombstone-free short-circuit in `count_edges_filtered`**. When no nodes/edges have been removed and no peer-type filter is set, the function returns `end - start + overflow_count` directly after binary-searching for the connection-type range — skipping the per-edge tombstone check on hot hubs. Adds a `has_tombstones: bool` flag to `DiskGraph` and `DiskGraphMeta` (defaults to conservative `true` on legacy graphs so correctness is preserved; new builds flip it false).
- **Bounded `sources_for_conn_type`**. `DiskGraph::sources_for_conn_type_bounded(conn_type, max)` stops copying source node IDs after `max` entries, avoiding the ~400 MB eager heap allocation on cold-cache `LIMIT`-bounded pattern-matching queries. `pattern_matching.rs` now passes the `source_cap` through so e.g. `LIMIT 10` queries only read 1 000 sources from `conn_type_index_sources.bin` on first access.
- **`FusedCountTypedEdge` uses cached edge-type counts**. A one-liner that had been missed in v0.7.12: `MATCH (_)-[:TYPE]->(_) RETURN count(*)` now returns `edge_type_counts[TYPE]` in O(1) instead of scanning `edge_weights()` (64 s → sub-millisecond on Wikidata's 862 M edges).
- **`rebuild_caches` refreshes the peer-count histogram** on existing disk graphs, so users don't need to rebuild from scratch to get the v0.7.14 aggregate speedups.

### Fixed
- **DataFrame / blueprint disk builds now rebuild indexes at save time**. Previously, the first `add_connections` batch triggered a CSR build (via `ensure_disk_edges_built`) which wrote `conn_type_index` and `peer_count_histogram` reflecting only that first batch's edges. Subsequent batches added edges to overflow but never refreshed those indexes. Fix: `save_disk` now calls `compact()` once when overflow has accumulated, merging overflow back into CSR and rebuilding the indexes from all live edges. The per-batch `ensure_disk_edges_built` is now a no-op for overflow purposes (no O(E²) cost during multi-batch builds).
- **`lookup_peer_counts` returns `None` on type miss**. Previously returned `Some(empty_map)`, which blocked the caller from falling back to the sequential-scan path when the histogram was stale. Now returns `None` so callers see a clean cache miss.
- **Deadline checks in anchored-count paths**. `try_count_simple_pattern` / `count_edges_filtered` now accept an `Option<Instant>` deadline and check it every 1 M iterations. Closes the bypass that let `Q5_count_P31_incoming` run to 100 s past the 20 s default timeout.
- **Deadline check in `expand_var_length_fast` inner loop**. The outer queue loop was already checked every 512 pops, but the per-edge inner loop was unbounded — a single hub expansion could process 100 M+ edges without checking. Added an inner check every 1 M iterations.
- **Benchmark metric: Wikidata `unanchored_P31_count` now returns in 0.7 ms** (was 64 s with a wrong answer on cold deadline checks), `Q5_count_P31_incoming` 615 ms (was 100 s TIMEOUT), `Q5_incoming_all_count` 670 ms (was 20 s TIMEOUT), `cross_type_limited` 3 ms (was 2.5 s), `limit_10_P31` 10 ms cold-cache (was 2.6 s).

## [0.7.12] — 2026-04-16

### Added
- **Parallel Phase 3 CSR build**: The per-node `out_edges` sort-by-connection-type and the `conn_type_index` inverted-index build are now Rayon-parallelised. On a 124 M-node / 862 M-edge Wikidata build, this cuts combined CSR-build wall-clock on the parallelised portions from ~1000 s serial to ~100-200 s on 8+ P-cores. Build output is bit-identical to the serial version (index source lists are sorted post-reduce for determinism).
- **Deadline enforcement on long edge scans**: `count_edges_grouped_by_peer` (used by fused aggregate top-K and streaming HAVING paths) now accepts an optional deadline and checks it every 1 M edges. Pattern-matching's parallel expansion short-circuits when any thread detects a timeout. Together these stop unanchored aggregate queries from running unbounded past the default 20 s timeout.
- **Disk mode iterative updates**: Loaded disk graphs now support `add_connections()` — new edges go directly to overflow and are immediately visible to queries without CSR rebuild.
- **`compact()` method**: Merges overflow edges back into CSR arrays via full rebuild. Call after accumulating significant overflow (e.g., >10% of edges) to restore optimal query performance.
- **Connection-type inverted index for overflow**: `sources_for_conn_type()` now includes nodes with overflow edges, so Cypher queries on new edge types work immediately.
- **Partitioned CSR build parity**: Out-edges now sorted by connection type (enables binary search), and connection-type inverted index built for both CSR algorithms.

### Fixed
- **Streaming HAVING aggregate no longer OOMs**: `MATCH ...-[...]->(...) RETURN group_key, count(...) HAVING ... ORDER BY ...` (without LIMIT) used to materialise all edge rows before grouping — a 10 GB materialisation on Wikidata-scale graphs that triggered macOS OOM kill. The planner now fuses this shape into `FusedMatchReturnAggregate`; the executor's non-top-k path uses edge-centric `count_edges_grouped_by_peer` and applies HAVING post-aggregation on the small group-by map. On 16 GB hosts, queries that previously SIGKILL'd the Python process now return a clean "Query timed out" error.
- **Benchmark CSV preserved across crashes**: `bench/benchmark_wikidata_cypher.py` now streams each query's result to the CSV row-by-row with per-row flushes, instead of batching the write at the end. SIGKILL / OOM / Ctrl-C mid-run leaves every completed row on disk.
- **Mmap lifecycle during CSR build**: CSR build now writes to a temporary directory, then atomically swaps files into place. Fixes panics on large DataFrame builds where CSR output overwrote mmap'd files.
- **Overflow edges missing from `edges_directed`**: The `edges_directed_filtered_iter` iterator now correctly includes overflow edges (was passing `None` for overflow parameter).
- **Column store corruption on save→load→save cycle**: `write_packed()` now handles mmap-backed column stores (from loaded disk graphs) by materializing data from the MmapColumnStore. Also skips writing empty schema columns that duplicate id/title columns.
- **`defer_csr` not reset after CSR build**: After the first CSR build, `defer_csr` stayed `true`, causing all subsequent `add_edge()` calls to route to `pending_edges` instead of overflow. Each CSR rebuild then lost all previous edges. Fixed by setting `defer_csr = false` in `build_csr_from_pending()`.
- **`edge_weight_mut` for disk mode**: Implemented mutable edge property access for disk graphs, required by `add_connections` with duplicate edge handling (e.g., blueprint builds with temporal edge properties).
- **Disk graph `save_to_dir` missing metadata**: `disk_graph_meta.json` and conn_type_index files were only written to `data_dir`, not to `target_dir` when saving to a different directory. Fixed `save_to_dir` to write metadata to the target.
- **N-Triples mapped mode used compact IDs**: Mapped mode incorrectly used disk-style compact integer IDs instead of string IDs. Now matches memory mode behavior.
- **`enable_columnar()` title column mismatch**: Columnar nodes with missing titles in old stores got no title pushed, causing title column length < row_count. Save→load then failed with "blob too small". Fixed by always using node.title as fallback.
- **`edge_weight_mut` arena offset bug**: The flush logic assumed all edge_weight_mut entries were contiguous at the end of the arena, but read-only `edge_weight` calls interspersed between writes caused wrong offsets. Replaced arena-based tracking with a dedicated `edge_mut_cache` HashMap.
- **N-Triples mapped mode used compact edge path**: `use_compact` was true for mapped mode, sending it through `create_edges_compact()` instead of `create_edges_strings()`. Mapped now uses the memory-mode path for everything.
- **`InternedKey` hash is now deterministic across processes**: `InternedKey::from_str` previously used `DefaultHasher` (SipHash with a per-process random seed). Since `DiskNodeSlot.node_type` persists this as raw u64 on disk and the loader resolves it via the freshly-built interner's hashes, disk graphs built in one process couldn't be reliably loaded in another. Replaced with FNV-1a 64-bit (zero-alloc, zero new deps, deterministic). **Breaking change for existing disk graphs** saved with an older kglite: their `node_type` u64 values were hashed with a random SipHash seed and will not resolve against the new interner. Rebuild affected disk graphs.
- **Disk mode save/load loses embeddings, timeseries_store, and parent_types**: `save_disk` and `load_disk_dir` only persisted the `FileMetadata` struct, which didn't include `parent_types` and omitted embeddings/timeseries entirely. Describe() output on reloaded disk graphs was missing the "core vs supporting" tier split and `<embeddings>` section. Fix: added `parent_types` to `FileMetadata`, and save/load `embeddings.bin.zst` and `timeseries.bin.zst` alongside the other disk artifacts.
- **`describe()` non-deterministic across processes**: `compute_join_candidates` iterated `node_type_metadata` (HashMap) and broke `sort_by` ties with insertion order. Different HashMap `RandomState` seeds produced different candidate orderings, making checksums unstable. Property iteration is now sorted by name, and the candidate sort uses `(overlap desc, left_type, right_type, left_prop)` as a stable key.
- **Disk mode DataFrame/blueprint builds: wrong node titles/properties after multiple types**: `batch_operations` assigned `DiskNodeSlot.row_id` by slot index (set in `add_node`) instead of the per-type column store row returned by `push_row`. Pass 2 tried to fix this via `node_weight_mut`, but that call materializes into an arena that gets cleared on the next call — so the correction never persisted. Once a second node type was added, slot indices diverged from column store rows, causing `n.title`/`n.id` to read wrong rows (and `None` for out-of-bounds slots). Fix: batch_operations now also calls `DiskGraph::update_row_id` after each deferred assignment. Raises `api_benchmark.py` from 38/51 to 49/51 across all 3 modes.
- **Column store schema rebuild drops titles**: When batch_operations rebuilds a column store due to schema growth, titles for existing rows could be lost if `get_title()` returned None. Fixed by always pushing Null fallback.
- **`save_disk()` now persists `type_indices.bin.zst` and `id_indices.bin.zst`**: Previously only written by the N-Triples builder, DataFrame/blueprint-built disk graphs now also persist these files for correct and fast reload.
- **`write_packed()` preserves all schema columns**: Empty schema columns are now written with null padding instead of being skipped, ensuring lossless metadata round-trip through save→load cycles.

## [0.7.10] - 2026-04-16

### Added
- **Connection-type inverted index**: Built during CSR construction, maps edge types to source node IDs. Enables instant lookup of "which nodes have P31 outgoing edges" for unanchored edge queries. Cold-cache `MATCH (a)-[:P31]->(b) LIMIT 50` improved from 14.5s to 4.6s.
- **madvise hints for edge scans**: Sequential/DontNeed advisories on edge_endpoints during full-graph aggregation to reduce page cache pollution.

### Changed
- **FusedNodeScanTopK**: New fused clause for `MATCH (n:Type) RETURN n.prop ORDER BY n.prop LIMIT K` — single-pass scan with inline top-K selection, avoids materializing all rows. String sort keys supported.
- **Streaming top-K for FusedMatchReturnAggregate**: Iterates group nodes directly from type_indices instead of materializing all PatternMatch objects.
- **Edge-centric aggregation**: For untyped group nodes, scans edge_endpoints sequentially with HashMap accumulation instead of per-node iteration.
- **Lightweight peer iteration**: `expand_from_node` skips edge_endpoints reads when edge variable is unnamed (disk-only, reduces I/O by ~50%).

## [0.7.9] - 2026-04-16

### Changed
- **Zero-allocation edge counting**: `count()` queries on edge patterns use a new fast path that iterates CSR edges without materializing `EdgeData`. With sorted CSR, uses binary search to narrow to matching edge type. Result: "count instances of City" dropped from 2.3s to 37ms (63x faster).
- **WHERE-MATCH fusion**: The executor detects MATCH followed by WHERE and evaluates the WHERE predicate inline during pattern expansion. Non-matching rows are skipped immediately, and expansion stops after finding exactly LIMIT matching rows. Previously stuck queries (>10 min) now complete within timeout.
- **LIMIT push-down through WHERE**: Extended `push_limit_into_match` to handle `MATCH → WHERE → RETURN → LIMIT` pattern. The executor enforces exact LIMIT during fused WHERE evaluation.
- **Pre-computed edge type counts**: Edge type counts are computed during CSR build (zero overhead — counted inline during endpoint materialization). Persisted to metadata so `FusedCountEdgesByType` is O(1) on reload.

### Fixed
- **Wikidata type merge**: Q-code types (e.g., "Q5") now properly merge into human-readable labels ("human") during N-Triples build. Previously, when both "Q5" and "human" existed as types, the merge was skipped — now indices and column stores are merged correctly.
- **Column store key remapping**: Property log entries with old Q-code InternedKeys are remapped to merged label keys during Phase 1b, ensuring column stores have correct data after type merges.

## [0.7.8] - 2026-04-15

### Added
- **`set_default_timeout(timeout_ms)`**: Set a default per-query timeout (milliseconds) applied to all `cypher()` calls. Per-query `timeout_ms` overrides it.
- **`set_default_max_rows(max_rows)`**: Set a default cap on intermediate result rows. Queries exceeding this return an error with guidance to add LIMIT. Per-query `max_rows` overrides it.
- **`cypher(max_rows=N)`**: Per-query max rows limit parameter.

### Changed
- **Cypher LIMIT push-down**: Tightened source candidate cap from 10,000× to 100× the LIMIT value. Queries like `MATCH (n:Type)-[:EDGE]->(m) RETURN ... LIMIT 10` on large types are now ~100x faster (avoids allocating the full type index).
- **Cypher pattern start-node optimization**: Improved selectivity estimation for property-filtered nodes (equality filters now estimate /100 instead of /10). Lowered reversal threshold from 10× to 5×. Queries with filters on the target node (e.g., `WHERE b.prop = 'X'`) are now 2-3× faster.
- **DiskGraph edge iteration**: DiskEdges iterator now reads CSR edges lazily from the mmap instead of pre-collecting into a Vec. Eliminates O(degree) allocation per iterator — critical for high-degree nodes at Wikidata scale.
- **DiskGraph direct columnar property access**: Property checks in Cypher WHERE clauses and pattern matching now read individual column values directly from the ColumnStore on disk graphs, bypassing full `NodeData` materialization. Eliminates arena allocation and unnecessary id/title reads — ~3x fewer mmap reads per property check.
- **DiskGraph CSR sorted by connection type**: CSR edges are now sorted by `(node, connection_type)` during build. Edge-type filtering uses binary search instead of linear scan — O(log D + matching) instead of O(D) for high-degree nodes. Metadata flag `csr_sorted_by_type` ensures backward compatibility with older graphs.
- **Fused aggregation with WHERE clauses**: `FusedNodeScanAggregate` now activates for queries with property filters (e.g., `MATCH (n:Entity) WHERE n.pop > 1M RETURN n.continent, count(n)`). `FusedMatchReturnAggregate` now supports property filters on the unbound (counted) node. Both avoid materializing intermediate result rows.

## [0.7.7] - 2026-04-15

### Added
- **Schema locking**: `lock_schema()` / `unlock_schema()` enforce the graph's known schema on Cypher mutations (CREATE, SET, MERGE). Invalid writes return descriptive errors with "did you mean?" suggestions via edit-distance matching. Works on any graph — locks against `node_type_metadata` and `connection_type_metadata`.
- **`from_blueprint(lock_schema=True)`**: Convenience parameter to lock the schema immediately after blueprint loading.
- **`schema_locked` property**: Check whether the schema is currently locked.
- **`describe()` schema-locked notice**: When schema is locked, `describe()` includes a `<schema-locked>` element so agents know writes will be validated.

## [0.7.6] - 2026-04-12

### Fixed
- **Silent data loss on incremental save/load**: Loading a `.kgl` file, adding or updating nodes, then saving and loading again would silently lose properties for the new/updated nodes. The v3 column writer now always consolidates all node properties (Compact, Map, and Columnar) into column stores before writing.
- **Corrupt `.kgl` file on re-save**: Simply loading and re-saving a `.kgl` file (with no changes) could produce a corrupt file that failed to load with `blob too small for offsets`. The v3 column loader was building the ColumnStore schema from `node_type_metadata` (which includes id/title fields) instead of from the column section metadata (which only has property columns), creating empty placeholder columns that corrupted on write.
- **`enable_columnar()` dropped Columnar nodes on rebuild**: When rebuilding column stores, nodes already using Columnar storage were skipped, then their old stores were replaced — losing their properties. Now reads properties from old Columnar stores during rebuild and preserves mapped-mode id/title columns.

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
