# Cypher Reference

Full Cypher subset supported by KGLite. For a quick overview, see the [README](README.md#cypher-queries).

> **Single-label note:** Each node has exactly one type. `labels(n)` returns a string, not a list. `SET n:OtherLabel` is not supported.

---

## Basic Query

```python
result = graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WHERE p.age > 30 AND f.city = 'Oslo'
    RETURN p.name AS person, f.name AS friend, p.age AS age
    ORDER BY p.age DESC
    LIMIT 10
""")

# Read queries → ResultView (iterate, index, or convert)
for row in result:
    print(f"{row['person']} knows {row['friend']}")

# Pass to_df=True for a DataFrame
df = graph.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age", to_df=True)
```

## WHERE Clause

```python
# Comparisons: =, <>, <, >, <=, >=
graph.cypher("MATCH (n:Product) WHERE n.price >= 500 RETURN n.title, n.price")

# Boolean operators: AND, OR, NOT
graph.cypher("MATCH (n:Person) WHERE n.age > 25 AND NOT n.city = 'Oslo' RETURN n.name")

# Null checks
graph.cypher("MATCH (n:Person) WHERE n.email IS NOT NULL RETURN n.name")

# String predicates: CONTAINS, STARTS WITH, ENDS WITH
graph.cypher("MATCH (n:Person) WHERE n.name CONTAINS 'ali' RETURN n.name")

# IN lists
graph.cypher("MATCH (n:Person) WHERE n.city IN ['Oslo', 'Bergen'] RETURN n.name")

# Regex matching with =~
graph.cypher("MATCH (n:Person) WHERE n.name =~ '(?i)^ali.*' RETURN n.name")
graph.cypher("MATCH (n:Person) WHERE n.email =~ '.*@example\\.com$' RETURN n.name")
```

## Relationship Properties

Relationships can have properties. Access them with `r.property` syntax:

```python
# Create relationships with properties
graph.cypher("""
    MATCH (p:Person {name: 'Alice'}), (m:Movie {title: 'Inception'})
    CREATE (p)-[:RATED {score: 5, comment: 'Excellent'}]->(m)
""")

# Access, filter, aggregate, sort by relationship properties
graph.cypher("MATCH (p)-[r:RATED]->(m) RETURN p.name, r.score, r.comment, type(r)")
graph.cypher("MATCH (p)-[r:RATED]->(m) WHERE r.score >= 4 RETURN p.name, m.title")
graph.cypher("MATCH (p)-[r:RATED]->(m) RETURN avg(r.score) AS avg_rating")
graph.cypher("MATCH ()-[r:RATED]->(m) RETURN m.title, r.score ORDER BY r.score DESC")
```

## Aggregation

```python
graph.cypher("MATCH (n:Person) RETURN n.city, count(*) AS population ORDER BY population DESC")
graph.cypher("MATCH (n:Person) RETURN avg(n.age) AS avg_age, min(n.age), max(n.age)")

# DISTINCT
graph.cypher("MATCH (n:Person) RETURN DISTINCT n.city")
graph.cypher("MATCH (n:Person) RETURN count(DISTINCT n.city) AS unique_cities")
```

## WITH Clause

```python
graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WITH p, count(f) AS friend_count
    WHERE friend_count > 3
    RETURN p.name, friend_count
    ORDER BY friend_count DESC
""")
```

## OPTIONAL MATCH

Left outer join — keeps rows even when no match:

```python
graph.cypher("""
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
    RETURN p.name, count(f) AS friends
""")
```

## Built-in Functions

| Function | Description |
|----------|-------------|
| `toUpper(expr)` | Convert to uppercase |
| `toLower(expr)` | Convert to lowercase |
| `toString(expr)` | Convert to string |
| `toInteger(expr)` | Convert to integer |
| `toFloat(expr)` | Convert to float |
| `size(expr)` | Length of string or list |
| `type(r)` | Relationship type |
| `id(n)` | Node ID |
| `labels(n)` | Node type (string, not list — single-label) |
| `coalesce(a, b, ...)` | First non-null argument |
| `range(start, end [, step])` | Generate integer list (inclusive); default step = 1 |
| `length(p)` | Path hop count |
| `nodes(p)` | Nodes in a path |
| `relationships(p)` | Relationships in a path |
| `point(lat, lon)` | Create a geographic point |
| `distance(a, b)` | Geodesic distance (m); geometry-aware |
| `contains(a, b)` | Does a's geometry contain b? |
| `intersects(a, b)` | Do geometries intersect? |
| `centroid(n)` | Centroid of geometry → Point |
| `area(n)` | Geodesic area (m²) |
| `perimeter(n)` | Geodesic perimeter/length (m) |
| `latitude(point)` | Extract latitude from point |
| `longitude(point)` | Extract longitude from point |
| `text_score(n, prop, query)` | Semantic similarity (auto-embeds query text; requires `set_embedder()`) |
| `text_score(n, prop, query, metric)` | With explicit metric (`'cosine'`, `'dot_product'`, `'euclidean'`) |
| `ts_sum(n.ch [, 'start'] [, 'end'])` | Sum of timeseries values (date-string range) |
| `ts_avg(n.ch [, 'start'] [, 'end'])` | Average of timeseries values |
| `ts_min(n.ch [, 'start'] [, 'end'])` | Minimum timeseries value |
| `ts_max(n.ch [, 'start'] [, 'end'])` | Maximum timeseries value |
| `ts_count(n.ch)` | Count of non-NaN timeseries values |
| `ts_at(n.ch, 'date')` | Exact timeseries key lookup |
| `ts_first(n.ch)` / `ts_last(n.ch)` | First / last non-NaN value |
| `ts_delta(n.ch, 'from', 'to')` | Value change between two time points |
| `ts_series(n.ch [, 'start'] [, 'end'])` | Extract series as `[{time, value}, ...]` |

## Spatial Functions

Built-in spatial functions for geographic queries. All node-aware functions auto-resolve geometry and location via [spatial types](README.md#spatial-types).

| Function | Returns | Description |
|----------|---------|-------------|
| `point(lat, lon)` | Point | Create a geographic point |
| `distance(a, b)` | Float (m) | Geodesic distance (WGS84); geometry-aware (0 if inside/touching) |
| `distance(lat1, lon1, lat2, lon2)` | Float (m) | Geodesic distance (4-arg shorthand) |
| `contains(a, b)` | Boolean | Does a's geometry contain b? (point-in-polygon or geometry containment) |
| `intersects(a, b)` | Boolean | Do geometries intersect? |
| `centroid(n)` | Point | Centroid of geometry (node or WKT string) |
| `area(n)` | Float (m²) | Geodesic area of polygon (node or WKT string) |
| `perimeter(n)` | Float (m) | Geodesic perimeter/length (node or WKT string) |
| `latitude(point)` | Float | Extract latitude component |
| `longitude(point)` | Float | Extract longitude component |

All functions accept both nodes (auto-resolved via spatial config) and raw values (WKT strings, Points).

```python
# Node-aware spatial — with spatial config declared via column_types
graph.cypher("""
    MATCH (c:City), (a:Area)
    WHERE contains(a, c)
    RETURN c.name, a.name
""")

graph.cypher("""
    MATCH (a:Field), (b:Field)
    WHERE intersects(a, b) AND a <> b
    RETURN a.name, b.name
""")

graph.cypher("""
    MATCH (n:Field)
    RETURN n.name, area(n) AS area_m2, centroid(n) AS center
""")

# Geometry-aware distance
graph.cypher("""
    MATCH (a:Field), (b:Field) WHERE a <> b
    RETURN a.name, b.name, distance(a.geometry, b.geometry) AS dist
""")  # 0 if polygons touch, centroid distance otherwise

graph.cypher("""
    MATCH (n:Field)
    WHERE distance(point(60.5, 3.5), n.geometry) < 10000.0
    RETURN n.name
""")  # 0 if point inside polygon, closest boundary otherwise

# Distance filtering — cities within 100 km of Oslo
graph.cypher("""
    MATCH (n:City)
    WHERE distance(n, point(59.91, 10.75)) < 100000.0
    RETURN n.name
    ORDER BY distance(n, point(59.91, 10.75))
""")

# Aggregation with spatial
graph.cypher("""
    MATCH (a:Field), (b:Field) WHERE a <> b
    RETURN avg(distance(a, b)) AS avg_dist, std(distance(a, b)) AS std_dist
""")
```

## Math Functions

| Function | Description |
|----------|-------------|
| `abs(x)` | Absolute value |
| `ceil(x)` / `ceiling(x)` | Round up to integer |
| `floor(x)` | Round down to integer |
| `round(x)` | Round to nearest integer |
| `round(x, d)` | Round to `d` decimal places (e.g. `round(3.14159, 2)` → 3.14) |
| `sqrt(x)` | Square root |
| `sign(x)` | Sign: -1, 0, or 1 |
| `log(x)` | Natural logarithm |
| `exp(x)` | e^x |
| `pow(x, y)` | x^y |
| `pi()` | π constant |
| `rand()` | Random float [0, 1) |

## Arithmetic & String Concatenation

```python
graph.cypher("MATCH (n:Product) RETURN n.title, n.price * 1.25 AS price_with_tax")

# String concatenation with ||
graph.cypher("MATCH (n:Person) RETURN n.first || ' ' || n.last AS fullname")

# || auto-converts non-strings; null propagates
graph.cypher("RETURN 'block-' || 35 AS label")  # → "block-35"
```

## CASE Expressions

```python
# Generic form
graph.cypher("""
    MATCH (n:Person)
    RETURN n.name,
           CASE WHEN n.age >= 18 THEN 'adult' ELSE 'minor' END AS category
""")

# Simple form
graph.cypher("""
    MATCH (n:Person)
    RETURN n.name,
           CASE n.city WHEN 'Oslo' THEN 'capital' WHEN 'Bergen' THEN 'west coast' ELSE 'other' END AS region
""")
```

## List Comprehensions

`[x IN list WHERE predicate | expression]` syntax:

```python
# Map: double each number
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] | x * 2] AS doubled")
# [2, 4, 6, 8, 10]

# Filter only
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 3] AS filtered")
# [4, 5]

# Filter + map
graph.cypher("UNWIND [1] AS _ RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 3 | x * 2] AS result")
# [8, 10]

# With collect() — transform aggregated values
graph.cypher("""
    MATCH (p:Person)
    WITH collect(p.name) AS names
    RETURN [x IN names | toUpper(x)] AS upper_names
""")
```

> **Note:** List comprehensions require at least one row in the pipeline. Use `UNWIND [1] AS _` or a preceding `MATCH`/`WITH` to provide the row context.

## Map Projections

`n {.prop1, .prop2, alias: expr}` syntax — select specific properties from a node:

```python
# Select only name and age (returns a dict per row)
graph.cypher("MATCH (p:Person) RETURN p {.name, .age} AS info")
# [{'info': {'name': 'Alice', 'age': 30}}, {'info': {'name': 'Bob', 'age': 25}}]

# Mix shorthand properties with computed values
graph.cypher("""
    MATCH (p:Person)-[:WORKS_AT]->(c:Company)
    RETURN p {.name, .age, company: c.name} AS info
""")

# System properties (id, type) work too
graph.cypher("MATCH (p:Person) RETURN p {.name, .type, .id} AS info LIMIT 1")
# [{'info': {'name': 'Alice', 'type': 'Person', 'id': 1}}]
```

## Parameters

```python
graph.cypher(
    "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age",
    params={'min_age': 25}
)

# Parameters in inline pattern properties
graph.cypher(
    "MATCH (n:Person {name: $name}) RETURN n.age",
    params={'name': 'Alice'}
)

# Parameters with DataFrame output
df = graph.cypher(
    "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age ORDER BY n.age",
    params={'min_age': 20}, to_df=True
)
```

## UNWIND

Expand a list into rows:

```python
graph.cypher("UNWIND [1, 2, 3] AS x RETURN x, x * 2 AS doubled")
```

## UNION

```python
graph.cypher("""
    MATCH (n:Person) WHERE n.city = 'Oslo' RETURN n.name AS name
    UNION
    MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name
""")
```

## Variable-Length Paths

```python
# 1 to 3 hops
graph.cypher("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) WHERE a.name = 'Alice' RETURN b.name")

# Exact 2 hops
graph.cypher("MATCH (a:Person)-[:KNOWS*2]->(b:Person) RETURN a.name, b.name")
```

## WHERE EXISTS

Check for subpattern existence. Both brace `{ }` and parenthesis `(( ))` syntax are supported:

```python
# Brace syntax
graph.cypher("MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->(:Person) } RETURN p.name")

# Parenthesis syntax (equivalent)
graph.cypher("MATCH (p:Person) WHERE EXISTS((p)-[:KNOWS]->(:Person)) RETURN p.name")

# Negation
graph.cypher("""
    MATCH (p:Person)
    WHERE NOT EXISTS { (p)-[:PURCHASED]->(:Product) }
    RETURN p.name
""")
```

## shortestPath()

BFS shortest path between two nodes. Supports directed (`->`) and undirected (`-`) syntax:

```python
# Directed — only follows edges in their defined direction
result = graph.cypher("""
    MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Dave'}))
    RETURN length(p), nodes(p), relationships(p), a.name, b.name
""")

# Undirected — traverses edges in both directions (same as fluent API)
result = graph.cypher("""
    MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]-(b:Person {name: 'Dave'}))
    RETURN length(p), nodes(p), relationships(p)
""")

# No path → empty list (not an error)
```

**Path functions:** `length(p)` returns hop count, `nodes(p)` returns node list, `relationships(p)` returns edge type list.

## CREATE / SET / DELETE / REMOVE / MERGE

```python
# CREATE — returns ResultView with .stats
result = graph.cypher("CREATE (n:Person {name: 'Alice', age: 30, city: 'Oslo'})")
print(result.stats['nodes_created'])  # 1

# CREATE relationship between existing nodes
graph.cypher("""
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    CREATE (a)-[:KNOWS]->(b)
""")

# SET — update properties
result = graph.cypher("MATCH (n:Person {name: 'Bob'}) SET n.age = 26, n.city = 'Stavanger'")
print(result.stats['properties_set'])  # 2

# DELETE — plain DELETE errors if node has relationships; DETACH removes all
graph.cypher("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n")

# REMOVE — remove properties (id/type are immutable)
graph.cypher("MATCH (n:Person {name: 'Alice'}) REMOVE n.city")

# MERGE — match or create
graph.cypher("""
    MERGE (n:Person {name: 'Alice'})
    ON CREATE SET n.created = 'today'
    ON MATCH SET n.updated = 'today'
""")
```

## Transactions

Group multiple mutations into an atomic unit. On success, all changes apply; on exception, nothing changes.

```python
with graph.begin() as tx:
    tx.cypher("CREATE (:Person {name: 'Alice', age: 30})")
    tx.cypher("CREATE (:Person {name: 'Bob', age: 25})")
    tx.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    # Commits automatically when the block exits normally
    # Rolls back if an exception occurs

# Manual control:
tx = graph.begin()
tx.cypher("CREATE (:Person {name: 'Charlie'})")
tx.commit()   # or tx.rollback()
```

## DataFrame Output

```python
df = graph.cypher("""
    MATCH (p:Person)-[:KNOWS]->(f:Person)
    WITH p, count(f) AS friends
    RETURN p.name, p.city, friends
    ORDER BY friends DESC
""", to_df=True)
```

## EXPLAIN

Prefix any Cypher query with `EXPLAIN` to see the query plan without executing it:

```python
plan = graph.cypher("""
    EXPLAIN
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[:KNOWS]->(f:Person)
    WITH p, count(f) AS friends
    RETURN p.name, friends
""")
print(plan)
# Query Plan:
#   1. NodeScan (MATCH) :Person
#   2. FusedOptionalMatchAggregate (optimized OPTIONAL MATCH + count)
#   3. Projection (RETURN) [p.name, friends]
# Optimizations: optional_match_fusion=1
```

## Timeseries Functions

Query time-indexed numeric data attached to nodes. All date arguments are strings (`'2020'`, `'2020-2'`, `'2020-2-15'`), and precision is validated against the data's resolution.

### Date-string syntax

| String | Depth | Matches resolution |
|--------|-------|--------------------|
| `'2020'` | year | year, month, day |
| `'2020-2'` | month | month, day |
| `'2020-2-15'` | day | day only |

**Precision rule:** Query depth must be ≤ data resolution for range functions (`ts_sum`, `ts_avg`, etc.). For exact-lookup functions (`ts_at`), query depth must equal the data resolution. Querying with day precision on month-resolution data produces an error.

### Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `ts_sum(n.channel)` | 1 | Float | Sum of all values |
| `ts_sum(n.channel, 'start')` | 2 | Float | Sum within prefix range |
| `ts_sum(n.channel, 'start', 'end')` | 3 | Float | Sum in range [start, end] inclusive |
| `ts_avg(n.channel [, 'start'] [, 'end'])` | 1-3 | Float | Average (same range rules as ts_sum) |
| `ts_min(n.channel [, 'start'] [, 'end'])` | 1-3 | Float | Minimum value in range |
| `ts_max(n.channel [, 'start'] [, 'end'])` | 1-3 | Float | Maximum value in range |
| `ts_count(n.channel)` | 1 | Integer | Count of non-NaN values |
| `ts_at(n.channel, 'date')` | 2 | Float/null | Exact key lookup (depth must match resolution) |
| `ts_first(n.channel)` | 1 | Float/null | First non-NaN value in series |
| `ts_last(n.channel)` | 1 | Float/null | Last non-NaN value in series |
| `ts_delta(n.channel, 'from', 'to')` | 3 | Float/null | Value at 'to' minus value at 'from' (prefix match) |
| `ts_series(n.channel [, 'start'] [, 'end'])` | 1-3 | List | Extract `[{time, value}, ...]` as JSON |

NaN values are skipped in all aggregation functions.

### Examples

```python
# Aggregate monthly data by year
graph.cypher("MATCH (f:Field) RETURN f.title, ts_sum(f.oil, '2020') AS prod")

# Range across months
graph.cypher("MATCH (f:Field) RETURN ts_avg(f.oil, '2020-1', '2020-6') AS h1_avg")

# Multi-year range
graph.cypher("MATCH (f:Field) RETURN ts_sum(f.oil, '2018', '2023') AS total")

# Exact month lookup
graph.cypher("MATCH (f:Field) RETURN ts_at(f.oil, '2020-3') AS march_prod")

# Change between two time points
graph.cypher("MATCH (f:Field) RETURN ts_delta(f.oil, '2019', '2021') AS change")

# Top producers
graph.cypher("""
    MATCH (f:Field)
    RETURN f.title, ts_sum(f.oil, '2020') AS prod
    ORDER BY prod DESC LIMIT 10
""")

# Filter by production threshold
graph.cypher("""
    MATCH (f:Field)
    WHERE ts_sum(f.oil, '2020') > 100.0
    RETURN f.title, ts_sum(f.oil, '2020') AS prod
""")

# Extract full series for plotting
graph.cypher("MATCH (f:Field {title: 'TROLL'}) RETURN ts_series(f.oil, '2015', '2020') AS data")

# Latest reading
graph.cypher("MATCH (s:Sensor) RETURN s.title, ts_last(s.temperature) AS latest")
```

### Precision validation

```python
# OK: year query on month data (coarser → aggregates all months)
graph.cypher("MATCH (f:Field) RETURN ts_sum(f.oil, '2020')")

# OK: month query on month data (exact match)
graph.cypher("MATCH (f:Field) RETURN ts_at(f.oil, '2020-3')")

# ERROR: day query on month data (finer than data resolution)
graph.cypher("MATCH (f:Field) RETURN ts_sum(f.oil, '2020-3-15')")
# → "Query precision 'day' (depth 3) exceeds data resolution 'month' (depth 2)"

# ERROR: year query with ts_at on month data (depth must match for exact lookup)
graph.cypher("MATCH (f:Field) RETURN ts_at(f.oil, '2020')")
# → "Exact lookup requires 2 date components for 'month' resolution, got 1"
```

## Supported Cypher Subset

| Category | Supported |
|----------|-----------|
| **Clauses** | `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `ORDER BY`, `SKIP`, `LIMIT`, `UNWIND`, `UNION`/`UNION ALL`, `CREATE`, `SET`, `DELETE`, `DETACH DELETE`, `REMOVE`, `MERGE`, `EXPLAIN` |
| **Patterns** | Node `(n:Type)`, relationship `-[:REL]->`, variable-length `*1..3`, undirected `-[:REL]-`, properties `{key: val}`, `p = shortestPath(...)` |
| **WHERE** | `=`, `<>`, `<`, `>`, `<=`, `>=`, `=~` (regex), `AND`, `OR`, `NOT`, `IS NULL`, `IS NOT NULL`, `IN [...]`, `CONTAINS`, `STARTS WITH`, `ENDS WITH`, `EXISTS { pattern }`, `EXISTS(( pattern ))` |
| **RETURN** | `n.prop`, `r.prop`, `AS` aliases, `DISTINCT`, arithmetic `+`/`-`/`*`/`/`, string concat `\|\|`, map projections `n {.prop1, .prop2}` |
| **Aggregation** | `count(*)`, `count(expr)`, `sum`, `avg`/`mean`, `min`, `max`, `collect`, `std` |
| **Expressions** | `CASE WHEN...THEN...ELSE...END`, `$param`, `[x IN list WHERE ... \| expr]` |
| **Functions** | `toUpper`, `toLower`, `toString`, `toInteger`, `toFloat`, `size`, `length`, `type`, `id`, `labels`, `coalesce`, `nodes(p)`, `relationships(p)` |
| **Spatial** | `point(lat, lon)`, `distance(a, b)`, `contains(a, b)`, `intersects(a, b)`, `centroid(n)`, `area(n)`, `perimeter(n)`, `latitude(point)`, `longitude(point)` |
| **Semantic** | `text_score(n, prop, query [, metric])` — auto-embeds query via `set_embedder()`, cosine/dot_product/euclidean |
| **Timeseries** | `ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_count`, `ts_at`, `ts_first`, `ts_last`, `ts_delta`, `ts_series` — date-string args with resolution validation |
| **Mutations** | `CREATE (n:Label {props})`, `CREATE (a)-[:TYPE]->(b)`, `SET n.prop = expr`, `DELETE`, `DETACH DELETE`, `REMOVE n.prop`, `MERGE ... ON CREATE SET ... ON MATCH SET` |
| **Procedures** | `CALL pagerank/betweenness/degree/closeness() YIELD node, score`, `CALL louvain/label_propagation() YIELD node, community`, `CALL connected_components() YIELD node, component`, `CALL cluster({method, ...}) YIELD node, cluster` |
| **Operators** | `+`, `-`, `*`, `/`, `\|\|` (string concat), `=~` (regex), `IN`, `STARTS WITH`, `ENDS WITH`, `CONTAINS`, `IS NULL`, `IS NOT NULL` |
| **Not supported** | `FOREACH`, subqueries (`CALL {}`), `SET n:Label` (label mutation), `REMOVE n:Label`, multi-label |
