# Cypher Engine Bug Tracker

Discovered 2026-03-29 via systematic testing against the Norwegian legal knowledge
graph (157,742 nodes, 1,008,891 edges) and verified locally on the test graph.

**49 xfail regression tests** in `tests/test_cypher.py` (classes prefixed `TestBug*`)
cover all bugs below. When a bug is fixed the corresponding tests flip to XPASS,
signalling the xfail marker can be removed.

---

## CRITICAL — Silent Wrong Results

These bugs produce **incorrect output without any error**. Users have no way to
know the result is wrong. Fix these first.

### BUG-01: Equality filter + GROUP BY returns empty results

| | |
|---|---|
| **Severity** | CRITICAL |
| **Symptom** | `WHERE k.title = 'X' RETURN k.title, count(c)` → empty result set |
| **Scope** | Only triggers on larger graphs where the planner optimization kicks in |
| **Tests** | None locally (graph-size dependent); verified on legal MCP |
| **Workaround** | Insert `WITH c, k` between WHERE and RETURN |

**Repro (legal MCP graph):**
```cypher
-- Returns empty (BUG):
MATCH (c:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword)
WHERE k.title = 'Narkotika'
RETURN k.title, count(c) AS cnt

-- Returns 1712 (correct — no group-by column):
MATCH (c:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword)
WHERE k.title = 'Narkotika'
RETURN count(c) AS cnt

-- Returns 1712 (correct — WITH workaround):
MATCH (c:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword)
WHERE k.title = 'Narkotika'
WITH c, k
RETURN k.title, count(c) AS cnt
```

**Root cause — two-phase optimization conflict in `planner.rs`:**

1. **Phase 1** — `push_where_into_match()` (line 546) pushes `k.title = 'Narkotika'`
   into the MATCH pattern as inline property `(k {title: 'Narkotika'})`. If all WHERE
   predicates are consumed, the WHERE clause is **deleted entirely** (line 607).

2. **Phase 2** — `fuse_match_return_aggregate()` (line 1107) tries to create a
   FusedMatchReturnAggregate. It calls `try_count_simple_pattern()` (executor.rs:1361)
   which **rejects** patterns with node properties. Fusion fails.

3. **Fallback** — execution falls back to the normal MATCH → RETURN pipeline, but the
   WHERE clause was already deleted in Phase 1. The property filter `{title: ...}` in
   the pattern is not applied by the standard executor path, so all rows pass through
   and the grouping produces either wrong results or nothing.

**Fix direction:** Either (a) don't delete WHERE when fusion might fail (check whether
the post-pushdown pattern is fusable before removing WHERE), or (b) make the standard
MATCH executor honour inline pattern properties that were pushed down.

---

### BUG-02: ORDER BY + LIMIT converts integers to floats

| | |
|---|---|
| **Severity** | CRITICAL |
| **Symptom** | `count()`, `size()`, `sum()` return `591.0` instead of `591` |
| **Scope** | Only when both ORDER BY and LIMIT are present on an aggregated column through WITH |
| **Tests** | `TestBugOrderByIntToFloat` — 4 xfail tests |
| **Workaround** | Use `toInteger()` on the result (though this is also broken in this context) |

**Repro:**
```cypher
-- Returns sections = 91 (int, correct):
MATCH (s:LawSection)-[:SECTION_OF]->(l:Law)
WITH l, count(s) AS sections
RETURN l.title, sections LIMIT 3

-- Returns sections = 591.0 (float, BUG):
MATCH (s:LawSection)-[:SECTION_OF]->(l:Law)
WITH l, count(s) AS sections
RETURN l.title, sections ORDER BY sections DESC LIMIT 3
```

**Root cause — `executor.rs` line 6914-6921:**

When ORDER BY + LIMIT are present, the planner fuses them into
`execute_fused_order_by_top_k()` which uses a min-heap with `f64` scores for efficient
top-K selection. When reconstructing results, it always wraps the score as
`Value::Float64(actual)` regardless of the original type:

```rust
// Line 6914-6921 — always produces Float64
let val = if j == score_item_index && score_is_native_float && !has_external_sort {
    let actual = if descending { winner.score } else { -winner.score };
    Value::Float64(actual)  // ← BUG: loses Int64 type
} else {
    self.evaluate_expression(&folded_exprs[j], row)?
};
```

The `score_is_native_float` check (line 6905) uses
`matches!(probe, Value::Float64(_) | Value::Int64(_))` which returns true for *both*
types, so it always takes the float path.

**Fix:** Before entering the heap path, probe the original value type. If it was
`Value::Int64`, recover as `Value::Int64(actual as i64)` instead of `Value::Float64`.
Same fix needed in `execute_fused_vector_score_top_k()` at line 6742.

---

### BUG-03: HAVING clause filter is non-functional on large graphs

| | |
|---|---|
| **Severity** | CRITICAL |
| **Symptom** | `HAVING cnt > 100000` returns rows with `cnt = 57798` |
| **Scope** | Manifests on larger graphs; works on small test graph |
| **Tests** | None locally (graph-size dependent); verified on legal MCP |
| **Workaround** | Use `WITH ... WHERE` instead of HAVING |

**Repro (legal MCP graph):**
```cypher
-- Returns cnt=57798 even though 57798 < 100000 (BUG):
MATCH (c:CourtDecision) RETURN count(c) AS cnt HAVING cnt > 100000

-- All 8 rows returned including cnt=17 (BUG):
MATCH (n) RETURN labels(n) AS type, count(n) AS cnt HAVING cnt > 10000
```

**Analysis:**

HAVING *is* implemented. The parser stores it in `ReturnClause.having` (ast.rs:350) and
it is applied in two places:
- `execute_fused_match_return_aggregate()` at line 2251: uses `evaluate_predicate`
- `execute_return()` at line 5591: wraps HAVING as WHERE and calls `execute_where()`

However, `execute_with()` at line 6538 creates a `ReturnClause` with `having: None`,
so any HAVING attached to a WITH is silently dropped.

The large-graph failure likely involves a **third code path** — probably
`execute_fused_order_by_top_k` or another fused aggregate path — that skips the HAVING
filter entirely. The HAVING on a direct RETURN might also be lost when the planner
rewrites the query for fusion.

**Fix direction:** Audit all code paths that handle RETURN clauses to ensure `having`
is propagated and applied. Search for `having: None` in executor.rs for clues.

---

### BUG-04: EXISTS subqueries return empty on large graphs

| | |
|---|---|
| **Severity** | CRITICAL |
| **Symptom** | `WHERE EXISTS { pattern }` always yields zero rows |
| **Scope** | Large graphs only; works locally on 5-node graph |
| **Tests** | None locally; verified on legal MCP |
| **Workaround** | Use double-MATCH pattern instead of EXISTS |

**Repro (legal MCP graph):**
```cypher
-- Returns empty (BUG):
MATCH (c:CourtDecision)
WHERE EXISTS { (c)-[:HAS_KEYWORD]->(:Keyword {title: 'Narkotika'}) }
RETURN c.title LIMIT 5

-- Returns results (correct — equivalent double-MATCH):
MATCH (c:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword {title: 'Narkotika'})
RETURN c.title LIMIT 5
```

**Root cause:**

`try_fast_exists_check()` (executor.rs:1183) only handles simple 3-element patterns
(node-edge-node). When properties are present on the target node the fast path likely
rejects the pattern, and the slow-path fallback at line 2870 uses a full
`PatternExecutor`. On larger graphs the variable binding from the outer row may not
propagate correctly into the inner pattern, causing all checks to fail.

**Fix direction:** Trace the slow-path EXISTS evaluation on a case where it should match
to find where the outer variable binding is lost.

---

### BUG-05: RETURN * yields `{'*': 1}` instead of bound variables

| | |
|---|---|
| **Severity** | CRITICAL |
| **Symptom** | `RETURN *` produces `{'*': 1}` for every row |
| **Scope** | Always — reproducible on any graph |
| **Tests** | `TestBugReturnStar` — 3 xfail tests |

**Repro:**
```cypher
MATCH (n:Person) WHERE n.name = 'Alice' RETURN *
-- Actual:   [{'*': 1}]
-- Expected: [{'n': {name: 'Alice', age: 30, ...}}]
```

**Root cause — `executor.rs` line 3515:**

```rust
Expression::Star => Ok(Value::Int64(1)), // For count(*)
```

This line handles `count(*)` correctly (the aggregate function wraps the Star and only
needs a non-null value to count). But when `*` appears as a **top-level RETURN item**
(not inside count), the same code path runs and returns `1`.

**Fix:** In the RETURN projection code, detect when a `ReturnItem` expression is
`Expression::Star` and expand it to all bound variables from `row.node_bindings`,
`row.edge_bindings`, and `row.path_bindings`. The `count(*)` path (inside aggregate
evaluation) should remain unchanged.

---

### BUG-06: Path variable on explicit multi-hop captures only first hop

| | |
|---|---|
| **Severity** | CRITICAL |
| **Symptom** | `length(p)=1`, `nodes(p)` missing intermediate nodes, `relationships(p)` incomplete |
| **Scope** | Always for multi-hop patterns; variable-length `*1..N` paths work correctly |
| **Tests** | `TestBugMultiHopPath` — 5 xfail tests |

**Repro:**
```cypher
MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:PURCHASED]->(pr:Product)
WHERE a.name = 'Alice'
RETURN length(p) AS hops, [n IN nodes(p) | n.title] AS chain
-- Actual:   hops=1, chain=["Alice", "Laptop"]
-- Expected: hops=2, chain=["Alice", "Bob", "Laptop"]
```

**Root cause:**

The `PatternExecutor` builds a `PathBinding` from the matched pattern, but only records
the source node, target node, and first relationship. For patterns with more than one
explicit edge (`-[:R1]->(x)-[:R2]->`), the intermediate nodes and subsequent edges are
not added to the path.

Variable-length paths `*1..N` take a different code path using
`graph_algorithms::get_path_connections()` which correctly traverses the full chain.

**Fix:** When constructing the `PathBinding`, iterate over all matched pattern elements
(alternating Node/Edge/Node/Edge/Node...) and collect every node and relationship into
the path.

---

## HIGH — Errors on Valid Cypher Syntax

These bugs cause hard errors on queries that should be valid. Users get clear error
messages but cannot accomplish their intent.

### BUG-07: stDev() aggregate not recognized

| | |
|---|---|
| **Tests** | `TestBugStDevFunction` — 4 xfail tests |
| **Error** | `RuntimeError: Unknown function: stDev` |

All case variants fail (`stDev`, `stdev`, `stddev`, `STDEV`). Implementation code
exists at executor.rs:6164 but the function name is not matched in the aggregate
dispatch. The `evaluate_scalar_function()` dispatch also doesn't list it.

**Fix:** Register `"stdev"` / `"stDev"` in the aggregate function dispatch table
alongside count, sum, avg, min, max, collect.

---

### BUG-08: datetime() function crashes

| | |
|---|---|
| **Tests** | `TestBugDatetimeFunction` — 2 xfail tests |
| **Error** | `RuntimeError: Invalid day '15T10:30:00' in date string` |

`datetime('2024-03-15T10:30:00')` is routed through the `date()` parser which doesn't
handle the `T` separator or time components.

**Fix:** Add a `datetime()` function that splits on `T`, parses date and time
separately, and returns an ISO datetime string or structured value.

---

### BUG-09: date() crashes on invalid input

| | |
|---|---|
| **Tests** | `TestBugDateInvalidInput` — 3 xfail tests |
| **Error** | `RuntimeError: Invalid date: 2016-0-0 out of range` |

Real-world data contains values like `'2016-00-00'`, `''`, and `'2016-13-01'`. These
crash the entire query. Neo4j would return null for invalid date input.

**Fix:** Wrap date parsing in a try-catch and return `Value::Null` on failure instead
of propagating the error.

---

### BUG-10: date().year property accessor syntax error

| | |
|---|---|
| **Tests** | `TestBugDatePropertyAccessor` — 3 xfail tests |
| **Error** | `ValueError: Unexpected token: Dot` |

`RETURN date('2024-06-15').year` fails because the parser doesn't support property
access chained directly onto function call results. The workaround
`WITH date('...') AS d RETURN d.year` works because the variable `d` is a normal
identifier that supports `.property` access.

**Fix:** In the expression parser, after parsing a function call, check for a trailing
`.identifier` and wrap the result as a property access on the function result.

---

### BUG-11: Pipe `|` in relationship types not parsed

| | |
|---|---|
| **Tests** | `TestBugRelationshipTypePipe` — 2 xfail tests |
| **Error** | `ValueError: Unexpected token in MATCH pattern: Pipe` |

Standard Cypher allows `[:TYPE1|TYPE2|TYPE3]` to match any of several relationship
types. The pattern parser at `pattern_matching.rs:548` only reads a single identifier
after the colon.

**Fix:** After reading the first type name, loop on `|` tokens to collect additional
type names. Store as `Vec<String>` in `EdgePattern.connection_type` (currently
`Option<String>`). During matching, check if the edge type is in the list.

---

### BUG-12: XOR logical operator not implemented

| | |
|---|---|
| **Tests** | `TestBugXorOperator` — 1 xfail test |
| **Error** | `ValueError: Unexpected token: Identifier("XOR")` |

Listed in the Cypher reference (`describe(cypher=True)`) as a supported logical
operator, but not implemented in the parser or AST.

**Fix:** Add `Xor(Box<Predicate>, Box<Predicate>)` to the `Predicate` enum, parse
`XOR` at the same precedence as `OR`, evaluate as `(a || b) && !(a && b)`.

---

### BUG-13: Modulo `%` operator not implemented

| | |
|---|---|
| **Tests** | `TestBugModuloOperator` — 2 xfail tests |
| **Error** | `ValueError: Unexpected character '%'` |

Not in the tokenizer or AST `Expression` enum (only `Add`, `Subtract`, `Multiply`,
`Divide`).

**Fix:** Tokenize `%` as a new token, add `Modulo(Box<Expression>, Box<Expression>)`
to `Expression`, evaluate with Rust `%` operator. Handle int-int and float-float cases.

---

### BUG-14: head() and last() list functions not implemented

| | |
|---|---|
| **Tests** | `TestBugHeadLastFunctions` — 5 xfail tests |
| **Error** | `RuntimeError: Unknown function: head` / `last` |

Standard Cypher list functions. Trivial to implement.

**Fix:** Add to `evaluate_scalar_function()`:
- `head(list)` → `list[0]` or `Value::Null` if empty
- `last(list)` → `list[list.len()-1]` or `Value::Null` if empty

---

### BUG-15: IN operator with variable reference fails

| | |
|---|---|
| **Tests** | `TestBugInWithVariable` — 3 xfail tests |
| **Error** | `ValueError: Expected LBracket, found Identifier("targets")` |

`WHERE n.prop IN targets` fails when `targets` is a variable from WITH/UNWIND. The
parser only accepts a literal `[...]` list after `IN`.

**Fix:** In the predicate parser, after parsing `IN`, accept any expression (variable
reference, function call, parameter) not just list literals. At evaluation time, resolve
the expression and check if it's a list.

---

## MEDIUM — Less Common Patterns

### BUG-16: Boolean/comparison expressions in RETURN clause

| | |
|---|---|
| **Tests** | `TestBugBooleanExpressionsInReturn` — 4 xfail tests |
| **Error** | `ValueError: Unexpected token at start of clause: StartsWith` (etc.) |

`RETURN n.name STARTS WITH 'A' AS x` fails. Also affects `CONTAINS`, `ENDS WITH`,
`>`, `<`, `=~` in RETURN context. These are parsed only as WHERE predicates, not as
general expressions that can appear in RETURN/WITH.

**Fix:** Allow predicate expressions (comparisons, string predicates, regex) to be used
as expressions that return boolean values. This requires unifying the predicate and
expression parsers or wrapping predicates as expressions.

---

### BUG-17: Unlabeled MATCH + type equality returns empty

| | |
|---|---|
| **Tests** | `TestBugUnlabeledMatchTypeFilter` — 3 xfail tests |
| **Symptom** | `MATCH (n) WHERE n.type = 'Person'` returns 0 rows |

Same root cause family as BUG-01 — the equality on the internal `type` property is
pushed into the pattern, but the pattern matcher doesn't apply it correctly for
unlabeled nodes. `CONTAINS` and `STARTS WITH` work because they aren't pushed down.

---

### BUG-18: labels() returns inconsistent types

| | |
|---|---|
| **Tests** | `TestBugLabelsInconsistency` — 2 xfail tests |
| **Symptom** | Returns `['Person']` (list) in plain RETURN, `'Person'` (string) in GROUP BY |

This makes it impossible to reliably compare or filter on `labels()` output. The GROUP
BY path extracts the inner string for grouping efficiency but exposes it to the user.

**Fix:** Always return `Vec<String>` (i.e., a list). The GROUP BY hashing can still use
the inner string but the value returned to the user should remain a list.

---

### BUG-19: null = null and null <> null syntax error

| | |
|---|---|
| **Tests** | `TestBugNullComparison` — 2 xfail tests |
| **Error** | `ValueError: Unexpected token at start of clause: Equals` |

`null + 1` (arithmetic) works and returns null. But `null = null` (comparison) fails
because the parser doesn't recognise `null` as a valid left-hand side in comparison
expressions.

**Fix:** In the expression parser, when encountering a `null` literal followed by a
comparison operator, parse it as a comparison expression. The result should always be
`null` per Cypher three-valued logic semantics.

---

### BUG-20: Map all-properties `{.*}` projection not supported

| | |
|---|---|
| **Tests** | `TestBugMapAllProperties` — 1 xfail test |
| **Error** | `ValueError: Expected property name after '.' in map projection` |

Named projection `n {.title, .id}` works fine. The `.*` variant that includes all
properties is not parsed.

**Fix:** In the map projection parser, when encountering `.*` after `{`, fetch all
property keys from the node (via `keys(n)`) and include them all in the projection.
