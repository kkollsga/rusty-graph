# Rule Packs

Rule packs are **named, agent-discoverable structural validators** —
orphan-node checks, missing-parent detection, cycle finding,
duplicate-title surfacing — that compile to Cypher and produce a
structured `RuleReport`. They're the layer that lets an LLM agent ask
"what should I trust about this graph?" without re-deriving the
validators each session.

```python
import kglite

g = kglite.load("legal.kgl")

# Discover available packs
g.rules.list()
# [{'name': 'structural_integrity', 'version': '1.0', ...}]

# Run the bundled structural-integrity pack against LawSection nodes
report = g.rules.run(
    "structural_integrity",
    type="LawSection",
    edge="SECTION_OF",
)

report.summary                              # counts + severities
report.violations_for("missing_required_edge")  # one rule's rows
report.to_markdown()                        # agent-pasteable
```

## Why this exists

A graph that you didn't build has trust gaps that look invisible from
the outside. The bundled `structural_integrity` pack surfaces them
across any KGLite graph in milliseconds:

| Graph (size) | Rule | Hits |
|---|---|---|
| Norwegian legal (157k nodes) | `missing_required_edge` (LawSection without SECTION_OF) | **5,682** |
| Norwegian legal | `duplicate_title` (LawSections sharing a title) | **1,582** |
| Sodir petroleum (564k nodes) | `missing_required_edge` (Wellbore without IN_LICENCE) | **502** |
| Sodir petroleum | `orphan_node` (Discovery with no edges) | **1** |

These are "latent" findings — Cypher can already retrieve them, but
nobody runs the queries because they're not part of normal workflow.
A pack makes them part of normal workflow.

## Rule packs are opt-in

By default a fresh `kglite.load(...)` produces a `describe()` with
**no** `<rule_packs>` block. Rule packs surface in `describe()` only
in two situations:

1. **Per-graph activation.** After the user calls `g.rules.run(...)`
   or `g.rules.load(...)`, that one graph's `describe()` shows
   per-rule schemas plus the last-run summary. Other graphs are
   unaffected.
2. **Module-level activation.** Calling `kglite.rules.advertise()`
   pushes a global default visible to every subsequent
   `describe()` — including graphs created earlier. Use this for
   MCP servers (or other agent-facing surfaces) that *do* expose
   rule-pack tools and want agents to discover them at first
   contact. Idempotent.

If neither activation has happened, `describe()` stays silent —
graphs that don't use rule packs incur no agent-facing noise.

## How agents discover rule packs (after activation)

Rule packs surface in `g.describe()` so agents see them through the
same XML they already consume for schema discovery — no separate
introspection tool needed:

```python
print(g.describe())
```

```xml
<graph nodes="157742" edges="1008891">
  ...
  <rule_packs>
    <pack name="structural_integrity" version="1.1" loaded="true"
          rule_count="6"
          last_run_violations="7264"
          last_run_severity="medium=7264"
          usage_hint="Run before answering structural questions...">
      <rule name="orphan_node" severity="medium" params="type"/>
      <rule name="self_loop" severity="high" params="type,edge"/>
      <rule name="cycle_2step" severity="high" params="type,edge"/>
      <rule name="missing_required_edge" severity="medium" params="type,edge"/>
      <rule name="missing_inbound_edge" severity="medium" params="type,edge"/>
      <rule name="duplicate_title" severity="medium" params="type"/>
    </pack>
    <hint>Run packs with g.rules.run(name); per-rule schemas appear once a pack is loaded.</hint>
  </rule_packs>
</graph>
```

A cold `describe()` shows pack-level info (name, version, rule count,
usage hint). After the first call to `g.rules.run(...)`, per-rule
parameter schemas appear inline so an agent can call `run()` again
with the right `type` / `edge` arguments without a second
introspection step. After every run, `last_run_violations` and
`last_run_truncated` carry the trust signal.

## Bundled pack: `structural_integrity`

Five universal validators that work on any property graph. Each
parameterises the node type (and edge type, where applicable) so the
same pack covers different domains.

| Rule | What it finds | Parameters |
|---|---|---|
| `orphan_node` | Nodes with zero edges in any direction | `type` |
| `self_loop` | Nodes connected to themselves via the given edge | `type`, `edge` |
| `cycle_2step` | Two-step cycles (A→B→A) via the given edge | `type`, `edge` |
| `missing_required_edge` | Nodes lacking an outgoing edge of the given type | `type`, `edge` |
| `missing_inbound_edge` | Nodes with no incoming edge of the given type | `type`, `edge` |
| `duplicate_title` | Multiple nodes of one type sharing a title | `type` |

The `missing_required_edge` and `missing_inbound_edge` rules carry a
`validates_direction:` field — at run time the runner inspects the
graph's actual edge schema and refuses to execute when the
`(type, edge)` pair flows the wrong way. Example: asking
`missing_inbound_edge(type='Wellbore', edge='IN_LICENCE')` on a graph
where `IN_LICENCE` flows `Wellbore → Licence` raises a
`DirectionMismatch` error suggesting `missing_required_edge` instead.

Run any subset with `only=[...]`:

```python
report = g.rules.run(
    "structural_integrity",
    type="Wellbore", edge="IN_LICENCE",
    only=["orphan_node", "missing_required_edge"],
)
```

## Performance and safety

A 5-rule run against a 157k-node graph finishes in **~30–80 ms**. The
runner has four guardrails:

1. **Anchored MATCH required.** A rule's first MATCH must specify a
   node label (e.g. `(n:Type)` or `(n:{type})`). Unanchored MATCHes
   would scan every edge — catastrophic on Wikidata-scale graphs. To
   override, set `unsafe_unanchored: true` on the rule (use only on
   small graphs).
2. **Default LIMIT 1000.** Compiled queries are bounded to prevent
   accidental 100k-row materialisations into an LLM context. Override
   per call: `g.rules.run(..., limit=10000)`. The
   `RuleReport.summary` carries `truncated: true` per rule when a
   limit is hit.
3. **Lazy materialisation.** `report.summary` returns counts without
   building any DataFrame. `report.rows` materialises only on first
   access. `report.violations_for(rule_name)` returns one rule's
   rows.
4. **Result cache.** Re-running a pack with identical params returns
   the same `RuleReport` instance from an LRU cache keyed on
   `(pack_name, params, graph)`.

## Writing your own pack

A rule pack is a YAML file. Minimal example:

```yaml
name: legal_integrity
version: "1.0"
description: |
  Norwegian-law-specific structural validators.

rules:
  - name: case_cites_only_repealed_law
    description_for_agent: |
      A CourtDecision that only cites LawSections of repealed Laws.
      Suggests outdated precedent. Surface to user.
    severity: medium
    parameters:
      cutoff_date: string
    match: |
      MATCH (c:CourtDecision)-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law)
      WHERE l.repealed_date < $cutoff_date
    columns:
      - c.id AS case_id
      - c.title AS case_title
      - count(s) AS repealed_citations
```

Load and run:

```python
g.rules.load("packs/legal_integrity.yaml")
g.rules.run("legal_integrity", cutoff_date="2020-01-01")
```

Notes:

- Use `{placeholder}` for text substitution at compile time (becomes
  part of the Cypher source).
- Use `$param` for true Cypher parameters (passed at execution time;
  preferred for value-typed inputs).
- The bundled rules use `{type}` and `{edge}` because they're
  substituted into structural positions (node labels, edge types).

### Per-rule timeouts — `default_timeout_ms`

Rules that scan dense node types (e.g. orphan-detection on a 13M-row
type) often need more than the graph's default Cypher timeout. Set
`default_timeout_ms` on the rule:

```yaml
- name: orphan_node
  parameters:
    type: string
  default_timeout_ms: 300000   # 5 minutes
  match: |
    MATCH (n:{type})
    WHERE NOT EXISTS { (n)-[]-() }
```

The runner passes this as the per-call `timeout_ms` to `g.cypher()`.
A caller-supplied `timeout_ms` to `g.rules.run(...)` always wins:

```python
# Rule's default applies
g.rules.run("structural_integrity", type="human", edge="P31")

# Override for one call
g.rules.run("structural_integrity", type="human", edge="P31",
            timeout_ms=600_000)  # 10 min
```

When the timeout fires, the rule's row in the report carries
`error="RuntimeError: Cypher execution error: Query timed out..."`
and `violations=0`. Other rules in the same pack continue executing.

### Direction-aware rules — `validates_direction`

Rules that check for a missing edge in a specific direction can opt
into a runtime sanity check by setting `validates_direction:` to
`outbound` or `inbound`. Both `type` and `edge` parameters become
required, and the runner refuses to execute when the `(type, edge)`
pair points the wrong way in the graph's actual schema:

```yaml
- name: missing_outbound_X
  parameters:
    type: string
    edge: string
  validates_direction: outbound
  match: |
    MATCH (n:{type})
    WHERE NOT EXISTS { (n)-[:{edge}]->() }
```

Without this, the rule would silently produce one violation per node
when the user picked the wrong direction (every node lacks an edge
that never flows that way for its type). The validator looks at
`g.connection_types()` to learn which node types are sources and
targets per edge type, then either runs the rule or surfaces a
`DirectionMismatch` error suggesting the right rule.

## Severity conventions

| Severity | Meaning | Suggested agent behaviour |
|---|---|---|
| `low` | Informational | Mention in passing |
| `medium` | Trust-relevant | Note in user-facing answer |
| `high` | Likely-error | Flag explicitly |
| `blocker` | Refuse-to-answer-without-flagging | `report.has_blockers` returns True; surface the violation prominently |

The bundled pack uses `medium` and `high` only.

## Cross-referencing query results — `is_suspect()`

Once a pack has been run, you can ask whether a node id was flagged
by any rule:

```python
report = g.rules.run("structural_integrity", type="Wellbore", edge="IN_LICENCE")

# In a normal user-query path:
for row in g.cypher("MATCH (w:Wellbore) WHERE ... RETURN w.id"):
    flags = report.is_suspect(row["w.id"])
    if flags:
        # e.g. [('missing_required_edge', 'medium')]
        ...   # surface to the user
```

`is_suspect()` returns an empty list for clean nodes and accepts
either string or int node ids. The index is built lazily on first
call and cached.

## Limitations (slice 1.1)

- **`g.rules.save_query_as_rule()` not yet implemented.** Agents can't
  yet promote ad-hoc Cypher to a saved rule for the next session.
- **No baseline / diff.** Each run is independent; comparing two runs
  of the same pack to surface "new violations since refresh" must be
  done by hand for now.
- **Anchoring validator is conservative.** It checks the first MATCH
  in a rule. Multi-MATCH rules where only the first is anchored will
  pass, even if subsequent ones are unbounded — rule authors are
  responsible for anchoring all MATCHes.
- **No property-existence validation.** A custom rule that references
  `n.entry_date` when the node type has no such property silently
  returns 0 rows. Check property names against
  `g.describe(types=['Type'])` before authoring custom rules.

These are explicit deferrals to slice 2; see the project's `CHANGELOG`
for status.
