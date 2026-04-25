# KGLite Backlog

## Completed in 0.8.16

- **Cypher `count { <pattern> }` subquery expression.** `count { ... }`
  in `WITH` / `RETURN` / `ORDER BY` / `WHERE` now evaluates to the
  number of matches of the inner pattern, scoped to the outer row's
  bindings (previously failed the parser on every backend with
  *"Expected property name or .property in map projection"*). New AST
  variant `Expression::CountSubquery`; parser special-cases the
  identifier-followed-by-brace dispatch for `count`; executor runs the
  pattern via the shared `PatternExecutor` with optional inline
  `WHERE`. Parity pinned by `tests/test_cypher_count_subquery.py`.

- **Property index for mapped mode.** `MappedGraph` now carries a
  lazy per-(node_type, property) and cross-type property index
  alongside the 0.8.15 conn_type index. `MATCH (n:Type {prop: val})`
  and `WHERE n.prop STARTS WITH 'x'` hit a binary-search path
  instead of a full scan; built on first query, cached via
  `Arc<RwLock<…>>`, invalidated on node / property mutations. Alias
  handling (title/label/name, id/nid/qid) matches disk. Measured:
  `MATCH (n:Q5 {title: 'Douglas Adams'})` on wiki100m mapped —
  **37 ms first call (builds the index), 0.1 ms warm**. Parity oracle
  in `tests/test_mapped_property_index.py`.

## Open

---

## Untyped `{nid: "Q42"}` / `WHERE n.nid = "Q42"` on mapped

### Context

`load_ntriples` writes `nid` into `NodeData.properties` as
`Value::String("Q42")`, and Columnar property retrieval (`RETURN n.nid`)
reads back `"Q42"` correctly. Yet the untyped equality path fails:

```python
# Mapped: storage="mapped", after load_ntriples
g.cypher('MATCH (n {nid: "Q42"}) RETURN n.title').to_df()      # 0 rows  ✗
g.cypher('MATCH (n:human {nid: "Q42"}) RETURN n.title').to_df() # 1 row  ✓
g.cypher('MATCH (n) WHERE n.nid = "Q42" RETURN n.title').to_df()  # 0 rows ✗
g.cypher('MATCH (n) WHERE n.nid CONTAINS "Q4" RETURN n.title').to_df() # 1 row ✓
```

Typed label + scan works; untyped equality path fails. The
`load_ntriples` → `build_columns_direct` pipeline produces Columnar
storage where `str_prop_eq` / the global-index scan return None (or
empty) for a string-typed column that clearly holds the string. The
`add_nodes` DataFrame path doesn't exhibit this (same value, different
column-build path), so it's specific to `build_columns_direct`.

The 0.8.16 Item 3 attempt (Q-prefix stripping to match disk) was
parked — stripping `id` from `"Q42"` to `UniqueId(42)` is orthogonal
to this bug, and applying it while the underlying lookup is broken
makes the regression harder to notice.

### Impact

Medium. Affects Wikidata workflows on mapped-mode graphs
(`MATCH ({nid:'Q5'})` is the idiomatic shape). Row-count parity
with disk is also broken: same query returns 10 rows on disk, 0 on
mapped at wiki100m.

### Suspected root cause

Two candidates, both in `load_ntriples` / `build_columns_direct`:

1. Column-type inference picks the wrong variant during
   `build_columns_direct` — maybe the first value seen for a
   (type, property) key determines the column type, and an
   interleaved `Value::UniqueId` vs `Value::String` confuses it.
2. The untyped scan in `pattern_matching::matcher.rs:740-760`
   calls `str_prop_eq` which returns `None` for Columnar-backed
   Str columns that were written via `build_columns_direct`
   (vs `add_nodes`, which works).

### Next steps

1. Targeted Rust unit test: build a `ColumnStore` via
   `build_columns_direct` replaying a minimal `PropertyLogWriter`
   stream with one entity whose `id` is a String and `nid` is
   another String. Assert `store.get` returns the expected
   `Value::String` for both keys. If it doesn't, we've isolated the
   bug to `build_columns_direct`'s column-type selection.
2. If `ColumnStore` round-trips correctly: look at the matcher's
   untyped scan fallback. Compare the Columnar path vs the
   Compact/Map path in `PropertyStorage::str_prop_eq` (schema.rs
   around line 294).

### Follow-up (once the above is fixed)

Revisit Q-prefix stripping on mapped — straightforward once the
underlying lookup works:
- Widen `use_compact_ids` gate at `src/graph/io/ntriples/loader.rs:2357`
  to `graph.graph.is_disk() || graph.graph.is_mapped()`.
- Allocate `qnum_to_idx` as a `Vec<u32>` for mapped (mirrors disk's
  `mapped_prefilled`; budget ~600 MB for the Wikidata ID space).
- The single existing test expecting `{id: "Q42"}` string lookup on
  mapped (`tests/test_incremental_columnar.py::TestNTriplesColumnar`)
  needs to be updated to `{id: 42}` or `{nid: "Q42"}`.

### Files to investigate first

- `src/graph/storage/column_store.rs` — `get` / `str_prop_eq` on
  Columnar storage.
- `src/graph/io/ntriples/loader.rs:1660-1700` — column-type inference
  and writes in `build_columns_direct`.
- `src/graph/core/pattern_matching/matcher.rs:740-760` — untyped
  scan fallback path.
- `src/graph/schema.rs:294` — `PropertyStorage::str_prop_eq`
  Columnar arm.

### Out of scope

- Memory mode (`storage="default"`) — doesn't go through
  `load_ntriples`' Columnar pipeline, doesn't share the bug.
- Disk mode — unaffected; the reference implementation.

---

## How to pick up an item

1. Read the item's **Next steps** and **Files to investigate first**.
2. Enter plan mode (`/plan`) — the plan shape used for the 0.8.15
   mapped-build / mapped-query items and 0.8.16's count/property
   items is a good template.
3. Land the fix in one commit with the CHANGELOG bullet in the same
   commit.
4. Update this backlog: move the item to **Completed** with a
   one-line link, or delete if superseded.
