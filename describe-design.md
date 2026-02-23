# Describe Method — Agent Interaction Design

## The Setup

- Complex graph: ~100 node types, 50+ properties each, timeseries, spatial, embeddings
- Agent interface: `describe()` + `cypher()`
- Agent knows: what a knowledge graph is, standard Cypher, general world knowledge
- Agent does NOT know: what's in this specific graph

**User prompt**: "How much smaller is the Troll fields production typically
during the summer months compared with the rest of the year?"

---

## Core Principle: Leverage What the Agent Already Knows

An LLM already knows standard Cypher — MATCH, WHERE, RETURN, WITH, UNWIND,
range(), toString(), aggregation, string concatenation. Explaining these
wastes tokens.

The same applies to graph structure. If the agent sees a type called "Field",
it can infer what a field is. It doesn't need 50 property definitions to
start reasoning.

**The describe method should only provide information the agent can't infer:**

- For **Cypher**: "standard Cypher works" + list the non-standard extensions.
- For **node structure**: state the universal conventions once, flag special
  capabilities per type, defer everything else.

---

## The Agent's Cognitive Steps

### Call 1: `describe()` — "What am I looking at?"

The agent needs an inventory, structural conventions, and non-standard
capabilities. Everything stated once, nothing repeated.

**What comes back** (~350 tokens for 100 types):

```
Graph: 50,000 nodes, 200,000 edges

Conventions:
  All nodes have .id and .title
  Some nodes have: location, geometry, timeseries

Node types (100 types):
  Large (>1000):  Person, WellBore (location), License, Company, Reservoir
  Medium (>100):  Field, ProductionProfile (ts), Discovery, Facility (geometry),
                  Pipeline (geometry), ...
  Small:          Stratigraphy, OperatorHistory, SeismicSurvey, ...

Connections:
  OF_FIELD: ProductionProfile → Field
  HAS_LICENSEE: Field → Company
  DRILLED_BY: WellBore → Company
  ... [50 more]

Cypher: standard Cypher supported. Extensions:
  Timeseries:
    ts_avg(n.channel, start?, end?), ts_sum, ts_min, ts_max, ts_count
    ts_first, ts_last, ts_delta, ts_at, ts_series
    Date args: 'YYYY', 'YYYY-M', 'YYYY-M-D', or DateTime properties
  Spatial:
    distance(a, b), contains(geom, point), intersects, centroid, area, perimeter
  Semantic:
    text_score(n, 'column', 'query text')
  Algorithms:
    CALL pagerank/betweenness/degree/closeness/louvain/label_propagation/
         connected_components({params}) YIELD node, score|community

Use describe(types=['TypeName']) to explore specific types in detail.
```

**What the agent now knows from ~350 tokens**:

- Every node has `.title` → it can match `{title: 'TROLL'}` on any type
- `ProductionProfile` is flagged `ts` → it has timeseries channels
- `ProductionProfile` connects to `Field` via `OF_FIELD`
- `ts_avg(n.channel, start, end)` exists with date string args
- `WellBore` has location → `distance()` will work on it
- `Facility` has geometry → `contains()`, `area()` will work
- Rough scale: Person/WellBore are large tables, Field/ProductionProfile
  are medium — useful for knowing what needs filtering

The conventions section (~20 tokens) eliminates the need to list `.title`
and `.id` as properties on every single type. The `ts`/`location`/`geometry`
flags (~1 token each) tell the agent which types have special capabilities
without explaining what those capabilities are — the Cypher extensions
section already did that. Size bands give rough scale without exact counts
— the agent knows Person is a large type (filter it) and Stratigraphy is
small (scan is fine).

**What the agent thinks**:
> "Troll is a Field. I can find it with {title: 'TROLL'} since all nodes
> have .title. ProductionProfile has timeseries and connects via OF_FIELD.
> I know ts_avg takes date ranges. I just need the channel names..."

---

### Path A: The Agent Tries Directly (confident)

The agent knows enough to attempt a query. It guesses a channel name:

```cypher
MATCH (f:Field {title: 'TROLL'})<-[:OF_FIELD]-(p:ProductionProfile)
RETURN f.title, ts_avg(p.production) AS avg
LIMIT 1
```

**If it works** → done in 1 describe call + 1 query.

**If it fails** → error: `"Unknown channel 'production'. Available:
prd_oe_net, prd_oil_net, prd_gas_net, prd_condensate_net, prd_water"`
→ the error itself teaches the agent the channel names → retry succeeds.

---

### Path B: The Agent Drills First (cautious)

The agent knows ProductionProfile has timeseries (from the `ts` flag) but
wants to see the channel names before writing a query.

**Call 2**: `describe(types=["Field", "ProductionProfile"])` — ~400 tokens:

```
Field (340 nodes):
  Properties:
    title (str, 340 unique)
    fldMainArea (str, vals: North Sea|Norwegian Sea|Barents Sea)
    fldStatus (str, vals: Producing|Shut down|PDO approved)
    ...
  Connections:
    out: -[:HAS_LICENSEE]-> Company (2400)
    in:  <-[:OF_FIELD]- ProductionProfile (340)
    in:  <-[:OF_FIELD]- WellBore (4500)
  Sample:
    {title: "TROLL", fldMainArea: "North Sea", fldStatus: "Producing"}

ProductionProfile (340 nodes):
  Properties:
    title (str, 340 unique)
  Timeseries:
    channels: [prd_oe_net, prd_oil_net, prd_gas_net, prd_condensate_net, prd_water]
    resolution: monthly
    units: {prd_oe_net: MSm3, prd_oil_net: MSm3, prd_gas_net: GSm3}
  Connections:
    out: -[:OF_FIELD]-> Field (340)
  Sample:
    {title: "TROLL", channels: {prd_oe_net: 456 points}}
```

**The agent now knows with certainty**:
- Find Troll via `(f:Field {title: 'TROLL'})`
- Link via `<-[:OF_FIELD]-(p:ProductionProfile)`
- Channel: `prd_oe_net` (total oil equivalent, monthly, MSm3)

---

### The Query (either path leads here)

```cypher
MATCH (f:Field {title: 'TROLL'})<-[:OF_FIELD]-(p:ProductionProfile)
UNWIND range(2015, 2024) AS year
WITH p, year, toString(year) AS y
WITH p, year,
     ts_avg(p.prd_oe_net, y) AS yearly_avg,
     ts_avg(p.prd_oe_net, y + '-6', y + '-8') AS summer_avg
WHERE yearly_avg > 0
WITH year, summer_avg / yearly_avg AS ratio
RETURN avg(ratio) AS mean_ratio,
       std(ratio) AS std_ratio,
       count(ratio) AS n_years,
       1.0 - avg(ratio) AS mean_reduction
```

Everything except `ts_avg` is standard Cypher the agent already knows:
UNWIND, range(), toString(), string concatenation, WITH chaining, avg(), std().

---

## Token Cost Comparison

| Approach | Calls | Context tokens |
|----------|-------|----------------|
| `describe()` → try directly (Path A, success) | 1 + 1 | ~550 |
| `describe()` → try → error → retry (Path A, fail) | 1 + 1 + 1 | ~850 |
| `describe()` → drill types → query (Path B) | 1 + 1 + 1 | ~950 |
| Current `agent_describe(detail='full')` | 1 + 1 | ~15,000+ |

All paths are an order of magnitude cheaper than front-loading everything.

---

## The Interface

```python
# Inventory: conventions, types with flags, connections, extensions
describe()

# Focused detail: properties, connections, timeseries config, samples
describe(types=["Field", "ProductionProfile"])
```

No detail levels. No format options. The agent controls scope by naming
what it cares about.

### Behavior at different graph complexities

| Graph | `describe()` no args |
|-------|---------------------|
| Simple (≤15 types) | Returns full detail inline — inventory IS the full description |
| Complex (>15 types) | Returns inventory with flags — agent drills with types= as needed |

Simple graphs pay zero overhead. The method naturally adapts without the
agent needing to know about complexity thresholds.

---

## Information Layering

The design uses three layers, each stated exactly once:

| Layer | What it covers | Tokens | When consumed |
|-------|---------------|--------|---------------|
| **Conventions** | Universal node fields (.id, .title), special capability types (location, geometry, timeseries) | ~20 | Always (in describe()) |
| **Type inventory** | Types grouped by size (Large/Medium/Small) with capability flags: `ProductionProfile (ts)`, `WellBore (location)` | ~100 for 100 types | Always (in describe()) |
| **Cypher extensions** | Non-standard function signatures and date format | ~150 | Always (in describe()) |
| **Type detail** | Properties, values, connections, timeseries channels, samples | ~200 per type | On demand (describe(types=)) |

Conventions eliminate repetition. Flags make types scannable. Extensions
teach only what's new. Detail is deferred until the agent asks for it.

---

## Error-Driven Exploration

The agent should NOT be told what doesn't work. Limitations are discovered
through errors, which serve two purposes:

1. **The agent adapts** — error messages guide it to the right syntax or property
2. **We learn** — errors the agent hits repeatedly reveal gaps we should fix

Good error messages are part of the interface. If the agent tries
`ts_avg(p.production)` and the channel doesn't exist, the error should say
what channels DO exist. The error becomes a mini describe() call.

---

## Design Principles

1. **Leverage existing knowledge** — the agent knows standard Cypher and can
   infer meaning from type names. Only provide what it can't infer.
2. **State conventions once** — universal node fields, capability types, and
   Cypher extensions are each stated a single time, not repeated per type.
3. **Flag, don't explain** — a `ts` flag on a type tells the agent it has
   timeseries. The Cypher extensions section already explained what that means.
4. **Let the agent choose its path** — some agents are bold (try first),
   some are cautious (drill first). The interface supports both.
5. **Errors are features** — they guide the agent and inform the developer.
   Don't pre-emptively limit; let the agent be ambitious.
6. **Scale naturally** — simple graphs get everything in one call. Complex
   graphs get progressive disclosure. Same interface, no mode switching.
