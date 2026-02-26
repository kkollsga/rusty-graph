# Fluent API Reference

Full fluent (method-chaining) API supported by KGLite. For Cypher queries, see [CYPHER.md](CYPHER.md). For a quick overview, see the [README](README.md).

> **Selection model:** The fluent API is selection-based. Most methods return a new `KnowledgeGraph` with an updated selection — no data is materialised until you call a retrieval method (`get_nodes()`, `to_df()`, etc.). This makes query chains fast even on large graphs.

---

## Data Loading

```python
import kglite
import pandas as pd

graph = kglite.KnowledgeGraph()

# Load nodes from a DataFrame
graph.add_nodes(df, 'Person', 'person_id', 'name')

# With column selection and conflict handling
graph.add_nodes(df, 'Person', 'person_id', 'name',
    columns=['name', 'age', 'city'],
    conflict_handling='update')  # 'update' | 'replace' | 'skip' | 'preserve'

# Spatial columns — declare via column_types
graph.add_nodes(df, 'City', 'city_id', 'name',
    column_types={
        'lat': 'location.lat',
        'lon': 'location.lon',
    })

# Geometry columns (WKT polygons)
graph.add_nodes(df, 'Field', 'field_id', 'name',
    column_types={'wkt_geometry': 'geometry'})

# Named points and shapes
graph.add_nodes(df, 'Pipeline', 'id', 'name',
    column_types={
        'start_lat': 'point.start.lat',
        'start_lon': 'point.start.lon',
        'end_lat': 'point.end.lat',
        'end_lon': 'point.end.lon',
        'route_wkt': 'shape.route',
    })

# Inline timeseries — multiple rows per ID, auto-deduplicated
graph.add_nodes(df, 'Production', 'field_id', 'field_name',
    timeseries={
        'time': 'date',                          # or {'year': 'yr', 'month': 'mo'}
        'channels': ['oil', 'gas', 'condensate'],
        'resolution': 'month',                   # auto-detected if omitted
        'units': {'oil': 'MSm3', 'gas': 'BSm3'},
    })

# Load connections (edges)
graph.add_connections(df, 'WORKS_AT',
    source_type='Person', source_id_field='person_id',
    target_type='Company', target_id_field='company_id')

# Bulk loading — multiple types at once
graph.add_nodes_bulk([
    {'node_type': 'Person', 'unique_id_field': 'id', 'data': people_df},
    {'node_type': 'Company', 'unique_id_field': 'id', 'data': companies_df},
])
graph.add_connections_bulk([
    {'source_type': 'Person', 'target_type': 'Company',
     'connection_name': 'WORKS_AT', 'data': works_df},
])

# Auto-filtering — silently skips connections whose types aren't loaded
graph.add_connections_from_source(connection_specs)
```

### Blueprint Loading

```python
# Build from a JSON blueprint + CSVs (round-trips with export_csv)
graph = kglite.from_blueprint("blueprint.json", verbose=True)

# Load from binary file
graph = kglite.load("graph.kgl")
```

---

## Selection & Filtering

### Type Selection

```python
# Select all nodes of a type
people = graph.type_filter('Person')

# With sort and limit
top10 = graph.type_filter('Person', sort='age', max_nodes=10)

# Multi-column sort
graph.type_filter('Person', sort=[('city', True), ('age', False)])  # city ASC, age DESC
```

### Property Filtering

```python
# Exact match
graph.type_filter('Person').filter({'city': 'Oslo'})

# Comparison operators
graph.type_filter('Person').filter({'age': {'>': 25}})
graph.type_filter('Product').filter({'price': {'>=': 100, '<=': 500}})

# String predicates
graph.type_filter('Person').filter({'name': {'contains': 'ali'}})
graph.type_filter('Person').filter({'name': {'starts_with': 'A'}})
graph.type_filter('Person').filter({'email': {'ends_with': '@example.com'}})
graph.type_filter('Person').filter({'name': {'regex': '^A.*'}})

# Negated variants
graph.type_filter('Person').filter({'status': {'not_in': ['inactive', 'banned']}})
graph.type_filter('Person').filter({'name': {'not_contains': 'test'}})

# IN list
graph.type_filter('Person').filter({'city': {'in': ['Oslo', 'Bergen']}})

# Null checks
graph.type_filter('Person').filter({'email': {'is_not_null': True}})
graph.type_filter('Person').filter({'nickname': {'is_null': True}})

# Combined conditions (AND logic within a single dict)
graph.type_filter('Person').filter({
    'age': {'>': 25},
    'city': 'Oslo',
    'name': {'regex': '^A.*'},
})
```

### OR Filtering

```python
# OR logic across condition sets
graph.type_filter('Person').filter_any([
    {'city': 'Oslo'},
    {'city': 'Bergen'},
    {'age': {'>': 60}},
])
```

### Connection-Based Filtering

```python
# Keep only nodes that have a KNOWS connection
graph.type_filter('Person').has_connection('KNOWS')

# Direction-specific
graph.type_filter('Person').has_connection('KNOWS', direction='outgoing')
graph.type_filter('Person').has_connection('KNOWS', direction='incoming')

# Orphan filtering
graph.type_filter('Person').filter_orphans(include_orphans=True)   # only disconnected nodes
graph.type_filter('Person').filter_orphans(include_orphans=False)  # only connected nodes
```

### Sorting & Pagination

```python
# Sort
graph.type_filter('Person').sort('age')
graph.type_filter('Person').sort('age', ascending=False)
graph.type_filter('Person').sort([('city', True), ('age', False)])

# Limit
graph.type_filter('Person').max_nodes(100)

# Pagination (skip + limit)
graph.type_filter('Person').sort('name').offset(20).max_nodes(10)  # page 3 of 10
```

---

## Temporal Filtering

Date-range filtering on node properties. NULL semantics: NULL `from` = valid since beginning, NULL `to` = still valid.

```python
# Nodes valid at a specific date
graph.type_filter('Employee').valid_at('2024-01-15')
# Uses default fields: date_from, date_to

# Custom field names
graph.type_filter('Contract').valid_at('2024-06-01',
    date_from_field='start_date',
    date_to_field='end_date')

# Nodes valid during a range (overlap check)
graph.type_filter('Regulation').valid_during('2020-01-01', '2022-12-31',
    date_from_field='effective_from',
    date_to_field='effective_to')
```

---

## Spatial Filtering

### Point-Based Filters

```python
# Bounding box
graph.type_filter('City').within_bounds(
    min_lat=59.0, max_lat=61.0,
    min_lon=10.0, max_lon=12.0)

# With custom field names
graph.type_filter('City').within_bounds(59.0, 61.0, 10.0, 12.0,
    lat_field='lat', lon_field='lon')

# Distance filter (degrees — fast, approximate)
graph.type_filter('City').near_point(59.91, 10.75, max_distance=1.0)

# Distance filter (meters — geodesic, WGS84)
graph.type_filter('City').near_point_m(59.91, 10.75, max_distance_m=100_000)
# Falls back to geometry centroid when lat/lon fields are missing
```

### Geometry Filters (WKT)

```python
# Point-in-polygon: which fields contain a point?
graph.type_filter('Field').contains_point(60.5, 3.5)

# Custom geometry field
graph.type_filter('Field').contains_point(60.5, 3.5, geometry_field='wkt_geometry')

# Geometry intersection: which fields overlap a query polygon?
graph.type_filter('Field').intersects_geometry(
    'POLYGON((3.0 60.0, 4.0 60.0, 4.0 61.0, 3.0 61.0, 3.0 60.0))')

# Also accepts shapely geometry objects
from shapely.geometry import box
graph.type_filter('Field').intersects_geometry(box(3.0, 60.0, 4.0, 61.0))
```

### Spatial Configuration

```python
# Declare spatial properties (alternative to column_types in add_nodes)
graph.set_spatial('City',
    location=('latitude', 'longitude'))

graph.set_spatial('Field',
    geometry='wkt_geometry')

# Named points and shapes
graph.set_spatial('Pipeline',
    points={'start': ('start_lat', 'start_lon'), 'end': ('end_lat', 'end_lon')},
    shapes={'route': 'route_wkt'})

# Query spatial config
graph.get_spatial('City')     # config for one type
graph.get_spatial()           # all types
```

### Spatial Aggregations

```python
# Geographic bounds of selection
bounds = graph.type_filter('City').get_bounds()
# {'min_lat': 58.1, 'max_lat': 71.1, 'min_lon': 5.3, 'max_lon': 31.0}

# As shapely polygon
poly = graph.type_filter('City').get_bounds(as_shapely=True)

# Centroid (average lat/lon)
center = graph.type_filter('City').get_centroid()
# {'latitude': 63.4, 'longitude': 10.4}

# WKT centroid (static method — does not require selection)
graph.wkt_centroid('POLYGON((3 60, 4 60, 4 61, 3 61, 3 60))')
# {'latitude': 60.5, 'longitude': 3.5}
```

---

## Timeseries

### Configuration

```python
# Declare timeseries metadata for a node type
graph.set_timeseries('Sensor',
    resolution='day',
    channels=['temperature', 'pressure'],
    units={'temperature': '°C', 'pressure': 'bar'},
    bin_type='sample')  # 'total' | 'mean' | 'sample'

graph.get_timeseries_config('Sensor')
graph.get_timeseries_config()  # all types
```

### Bulk Loading from DataFrame

```python
# Bulk load timeseries from a DataFrame
graph.add_timeseries('Field',
    data=production_df,
    fk='field_id',                       # foreign key → node ID
    time_key=['year', 'month'],          # or ['date'] for date strings
    channels=['oil', 'gas', 'condensate'],
    resolution='month',                  # auto-detected if omitted
    units={'oil': 'MSm3'})
```

### Manual Loading

```python
# Set time index for a node
graph.set_time_index('sensor_1', ['2024-01-01', '2024-01-02', '2024-01-03'])

# Add channel data (length must match time index)
graph.add_ts_channel('sensor_1', 'temperature', [20.1, 21.3, 19.8])
graph.add_ts_channel('sensor_1', 'pressure', [1.01, 1.02, 0.99])
```

### Retrieval

```python
# Get all channels for a node
data = graph.get_timeseries('sensor_1')
# {'keys': ['2024-01-01', '2024-01-02', ...],
#  'channels': {'temperature': [20.1, 21.3, ...], 'pressure': [1.01, ...]}}

# Single channel
data = graph.get_timeseries('sensor_1', channel='temperature')
# {'keys': ['2024-01-01', ...], 'values': [20.1, ...]}

# Date range
data = graph.get_timeseries('sensor_1', channel='temperature',
    start='2024-01-01', end='2024-01-02')

# Just the time index
keys = graph.get_time_index('sensor_1')
# ['2024-01-01', '2024-01-02', '2024-01-03']
```

### Timeseries Aggregation via Cypher

The fluent API provides data loading and extraction. For aggregation (`ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_at`, `ts_delta`, `ts_series`, etc.), use Cypher — see [CYPHER.md](CYPHER.md#timeseries-functions).

```python
# Example: top producers in 2020
graph.cypher("""
    MATCH (f:Field)
    RETURN f.title, ts_sum(f.oil, '2020') AS prod
    ORDER BY prod DESC LIMIT 10
""")
```

---

## Embedding / Vector Search

### Setup

```python
# Register an embedding model (must have .dimension and .embed())
graph.set_embedder(my_model)

# Compute embeddings for a text column
graph.embed_texts('Article', 'summary', batch_size=256, show_progress=True)
# Only embeds nodes that don't have embeddings yet; pass replace=True to re-embed all

# Or provide pre-computed embeddings
graph.set_embeddings('Article', 'summary', {
    'article_1': [0.1, 0.2, ...],
    'article_2': [0.3, 0.4, ...],
})
```

### Text Search (auto-embeds query)

```python
# Search using a text query — auto-embeds via set_embedder()
results = graph.type_filter('Article').search_text(
    'summary', 'machine learning advances', top_k=10)

# With explicit metric
results = graph.type_filter('Article').search_text(
    'summary', 'climate change', top_k=5, metric='dot_product')

# As DataFrame
df = graph.type_filter('Article').search_text(
    'summary', 'AI safety', top_k=10, to_df=True)
```

### Vector Search (pre-computed query vector)

```python
# Search with an explicit vector
results = graph.type_filter('Article').vector_search(
    'summary', query_vector=[0.1, 0.2, ...], top_k=10)

# Combine with property filters
results = (graph
    .type_filter('Article')
    .filter({'category': 'politics'})
    .vector_search('summary', query_vec, top_k=10, metric='cosine'))
```

### Semantic Search via Cypher

```python
# text_score() in Cypher queries (requires set_embedder)
graph.cypher("""
    MATCH (n:Article)
    RETURN n.title, text_score(n, 'summary', 'machine learning') AS score
    ORDER BY score DESC LIMIT 10
""")

# With threshold in WHERE
graph.cypher("""
    MATCH (n:Article)
    WHERE text_score(n, 'summary', $query) > 0.8
    RETURN n.title
""", params={'query': 'artificial intelligence'})
```

### Embedding Management

```python
# List all embedding stores
graph.list_embeddings()
# [{'node_type': 'Article', 'text_column': 'summary', 'dimension': 384, 'count': 1000}]

# Retrieve all embeddings
vecs = graph.get_embeddings('Article', 'summary')       # by type
vecs = graph.type_filter('Article').get_embeddings('summary')  # from selection

# Single embedding
vec = graph.get_embedding('Article', 'summary', 'article_1')

# Remove an embedding store
graph.remove_embeddings('Article', 'summary')
```

---

## Traversal

```python
# Follow outgoing connections
graph.type_filter('Person').traverse('WORKS_AT')

# Direction control
graph.type_filter('Person').traverse('KNOWS', direction='incoming')
graph.type_filter('Person').traverse('KNOWS', direction='both')

# Filter target nodes
graph.type_filter('Person').traverse('WORKS_AT',
    filter_target={'city': 'Oslo'})

# Filter edge properties
graph.type_filter('Person').traverse('RATED',
    filter_connection={'score': {'>': 4}})

# Sort and limit targets
graph.type_filter('Person').traverse('KNOWS',
    sort_target='name', max_nodes=5)

# Multi-hop traversal
companies = (graph
    .type_filter('Person')
    .filter({'city': 'Oslo'})
    .traverse('WORKS_AT')
    .traverse('LOCATED_IN'))
```

### Comparison-Based Traversal (Traverse Plugins)

When `method=` is specified, `traverse()` switches from edge-based to comparison-based mode.
The first argument becomes **target node type** (not connection type).

`method` accepts a **string** for simple cases or a **dict** with method-specific settings:

```python
method='contains'                                        # string shorthand
method={'type': 'contains', 'resolve': 'geometry'}       # dict with settings
```

#### Spatial Containment

```python
# Find all wells within each structural element's geometry
# Default: target resolved via location fields → geometry centroid fallback
graph.type_filter('Structure').traverse('Well', method='contains')

# Force polygon-in-polygon containment (target as full geometry)
graph.type_filter('Structure').traverse('Field',
    method={'type': 'contains', 'resolve': 'geometry'})

# Force geometry centroid (even if target has location fields)
graph.type_filter('Structure').traverse('Well',
    method={'type': 'contains', 'resolve': 'centroid'})

# Override geometry field name
graph.type_filter('Zone').traverse('Well',
    method={'type': 'contains', 'geometry': 'wkt_geometry'})
```

#### Spatial Intersection

```python
# Find licences whose geometry overlaps each field (always geometry-to-geometry)
graph.type_filter('Field').traverse('Licence', method='intersects')

# With custom geometry field
graph.type_filter('Field').traverse('Licence',
    method={'type': 'intersects', 'geometry': 'wkt_field'})
```

#### Distance

```python
# Find wells within 5 km of each platform (point-to-point, default resolution)
graph.type_filter('Platform').traverse('Well',
    method={'type': 'distance', 'max_m': 5000})

# Force geometry centroid (even if nodes have location fields)
graph.type_filter('Structure').traverse('Well',
    method={'type': 'distance', 'max_m': 5000, 'resolve': 'centroid'})

# Closest boundary point (min edge-to-edge distance)
graph.type_filter('Structure').traverse('Well',
    method={'type': 'distance', 'max_m': 5000, 'resolve': 'closest'})

# With filter and limit
graph.type_filter('Platform').traverse('Well',
    method={'type': 'distance', 'max_m': 10000},
    filter_target={'status': 'active'}, max_nodes=20)
```

#### Semantic Similarity

```python
# Find articles with similar abstracts (cosine > 0.85)
graph.type_filter('Article').traverse('Article',
    method={'type': 'text_score', 'property': 'abstract', 'threshold': 0.85},
    max_nodes=5)

# Different similarity metric
graph.type_filter('Doc').traverse('Doc',
    method={'type': 'text_score', 'property': 'summary',
            'threshold': 0.7, 'metric': 'dot_product'})
```

#### Clustering

```python
# Group wells into clusters by location
graph.type_filter('Well').traverse(
    method={'type': 'cluster', 'algorithm': 'kmeans', 'k': 5,
            'features': ['latitude', 'longitude']})

# DBSCAN with distance threshold
graph.type_filter('Well').traverse(
    method={'type': 'cluster', 'algorithm': 'dbscan',
            'eps': 5000, 'min_samples': 3,
            'features': ['latitude', 'longitude']})

# Chain: per-cluster statistics
graph.type_filter('Well').traverse(
    method={'type': 'cluster', 'algorithm': 'kmeans', 'k': 10,
            'features': ['latitude', 'longitude', 'depth']}) \
    .statistics('production')
```

#### Resolve modes

The `resolve` key controls how polygon geometries are spatially interpreted.
When omitted, the default is: **location fields → geometry centroid fallback**.

| `resolve` | Behavior | Use case |
|-----------|----------|----------|
| *(omitted)* | location lat/lon → geometry centroid fallback | Default |
| `'centroid'` | Geometry centroid (skips location fields) | Force use of geometry |
| `'closest'` | Nearest point on geometry boundary | Min boundary-to-boundary distance |
| `'geometry'` | Full polygon shape | Polygon-in-polygon containment |

### Add Properties from Traversal Chain

After any traversal (edge-based or comparison-based), `add_properties()` enriches
the selected (leaf) nodes with properties from ancestor nodes in the hierarchy.

```python
# Copy properties from parent type
graph.type_filter('Structure').traverse('Well', method='contains') \
    .add_properties({'Structure': ['name', 'status']})

# Copy all properties
graph.type_filter('Structure').traverse('Well', method='contains') \
    .add_properties({'Structure': []})

# Rename properties
graph.type_filter('Structure').traverse('Well', method='contains') \
    .add_properties({'Structure': {'struct_name': 'name', 'struct_status': 'status'}})

# Aggregate: count and stats from leaf nodes onto ancestors
graph.type_filter('Well').traverse('Structure', method='contains') \
    .add_properties({'Well': {
        'well_count': 'count(*)',
        'avg_depth': 'mean(depth)',
        'max_depth': 'max(depth)',
        'total_prod': 'sum(production)',
    }})

# Spatial computed properties
graph.type_filter('Structure').traverse('Well', method='contains') \
    .add_properties({'Structure': {
        'dist_to_center': 'distance',     # geodesic distance to parent centroid
        'parent_area': 'area',            # parent geometry area in m²
        'parent_perimeter': 'perimeter',  # parent geometry perimeter in m
    }})

# Combined: rename + spatial in one call
graph.type_filter('Structure').traverse('Well', method='contains') \
    .add_properties({
        'Structure': {
            'struct_name': 'name',
            'struct_area': 'area',
            'dist_to_center': 'distance',
        },
    })

# Copy from intermediate node in A → B → C chain
graph.type_filter('A').traverse('REL_AB').traverse('REL_BC') \
    .add_properties({'B': ['score']})
```

**Aggregate functions:** `count(*)`, `sum(prop)`, `mean(prop)`, `min(prop)`, `max(prop)`, `std(prop)`, `collect(prop)`

**Spatial compute functions:** `distance`, `area`, `perimeter`, `centroid_lat`, `centroid_lon`

### Breadth-First Expansion

```python
# Expand selection by N hops (undirected)
expanded = graph.type_filter('Person').filter({'name': 'Alice'}).expand(hops=2)
```

### Create Connections from Traversal

```python
# After A → B → C traversal, create direct A → C edges
graph.type_filter('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('A_TO_C')

# Copy properties from intermediate B nodes onto the new edges
graph.type_filter('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('A_TO_C', properties={'B': ['score', 'weight']})

# Empty list = copy ALL properties from that type
graph.type_filter('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('A_TO_C', properties={'B': []})

# Copy from multiple node types
graph.type_filter('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('A_TO_C', properties={'A': ['name'], 'B': ['score']})

# Override source/target (connect B → C instead of A → C)
graph.type_filter('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('B_TO_C', source_type='B', target_type='C')

# Conflict handling
graph.type_filter('A').traverse('REL_AB').traverse('REL_BC') \
    .create_connections('A_TO_C', conflict_handling='skip')
```

---

## Data Retrieval

### Nodes

```python
# Full node data — returns ResultView (lazy)
result = graph.type_filter('Person').filter({'age': {'>': 25}}).get_nodes()
for node in result:
    print(node['title'], node['age'])

# ResultView supports indexing, len(), bool()
print(len(result))
print(result[0])

# As list of dicts (full materialisation)
nodes = result.to_list()

# Lightweight: id + title + type only
ids = graph.type_filter('Person').get_ids()

# Flat ID list (lightest)
id_list = graph.type_filter('Person').id_values()

# O(1) lookup by type + ID
person = graph.get_node_by_id('Person', 'alice')

# Titles only
titles = graph.type_filter('Person').get_titles()

# Specific properties as tuples
props = graph.type_filter('Person').get_properties(['name', 'age'])
# [('Alice', 30), ('Bob', 25)]
```

### Count & Indices

```python
# Count without materialising (O(1))
n = graph.type_filter('Person').node_count()

# Raw graph indices
idx = graph.type_filter('Person').indices()
```

### DataFrame Export

```python
# Current selection as DataFrame
df = graph.type_filter('Person').to_df()

# Without type/id columns
df = graph.type_filter('Person').to_df(include_type=False, include_id=False)

# GeoDataFrame (from ResultView)
result = graph.cypher("MATCH (n:Field) RETURN n.name, n.wkt_geometry AS geometry")
gdf = result.to_gdf(geometry_column='geometry', crs='EPSG:4326')
```

### ResultView

All `cypher()` calls, `get_nodes()` (flat), centrality methods, and `sample()` return a `ResultView`:

```python
result = graph.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age")

len(result)           # row count (O(1))
bool(result)          # True if non-empty
result[0]             # single row as dict
result.columns        # ['n.name', 'n.age']
result.head(5)        # first 5 rows as new ResultView
result.tail(5)        # last 5 rows
result.to_list()      # list[dict] (full conversion)
result.to_df()        # pandas DataFrame
result.to_gdf()       # GeoDataFrame (with WKT geometry column)
result.stats          # mutation stats (CREATE/SET/DELETE only)
result.profile        # PROFILE stats (PROFILE queries only)

for row in result:
    print(row)
```

---

## Statistics & Calculations

```python
# Descriptive statistics for a numeric property
stats = graph.type_filter('Person').statistics('age')
# {count, mean, std, min, max, sum}

# Group by a property
stats = graph.type_filter('Person').statistics('age', group_by='city')
# {'Oslo': {count, mean, ...}, 'Bergen': {count, mean, ...}}

# Count nodes
n = graph.type_filter('Person').count()

# Count grouped by property
counts = graph.type_filter('Person').count(group_by='city')
# {'Oslo': 150, 'Bergen': 80}

# Math expressions
graph.type_filter('Product').calculate('price * quantity')

# Aggregate functions
graph.type_filter('Person').calculate('mean(age)')

# Store results as new properties
graph.type_filter('Product').calculate('price * 1.25', store_as='price_with_tax')
```

### Unique Values

```python
cities = graph.type_filter('Person').unique_values('city')

# Store as comma-separated list on parent nodes
graph.type_filter('Company').traverse('EMPLOYS').unique_values(
    'skill', store_as='employee_skills')
```

### Children Properties to List

```python
# Collect child titles into comma-separated strings on parents
graph.type_filter('Company').traverse('EMPLOYS').children_properties_to_list(
    property='name', sort='name', store_as='employees')
```

---

## Graph Algorithms

### Path Finding

```python
# Shortest path
path = graph.shortest_path('Person', 'alice', 'Person', 'dave')
# {'path': [{id, title, type}, ...], 'connections': ['KNOWS', 'KNOWS'], 'length': 2}

# With edge type and node type filters
path = graph.shortest_path('Person', 'alice', 'Person', 'dave',
    connection_types=['KNOWS', 'WORKS_WITH'],
    via_types=['Person'],
    timeout_ms=5000)

# Just the hop count (faster)
dist = graph.shortest_path_length('Person', 'alice', 'Person', 'dave')

# Just the IDs
ids = graph.shortest_path_ids('Person', 'alice', 'Person', 'dave')

# Just raw indices (fastest)
indices = graph.shortest_path_indices('Person', 'alice', 'Person', 'dave')

# All paths up to max hops
paths = graph.all_paths('Person', 'alice', 'Person', 'dave',
    max_hops=5, max_results=100, timeout_ms=10000)
```

### Connectivity

```python
# Boolean connectivity check
connected = graph.are_connected('Person', 'alice', 'Person', 'dave')

# Connected components
components = graph.connected_components(weak=True)   # weakly connected (default)
components = graph.connected_components(weak=False)  # strongly connected
# Returns list of components (largest first)

# Degree counts for selected nodes
degrees = graph.type_filter('Person').get_degrees()
# {'Alice': 5, 'Bob': 3, ...}
```

### Centrality

All centrality methods return `ResultView` by default, with optional `as_dict` or `to_df` output:

```python
# Betweenness centrality
result = graph.betweenness_centrality(top_k=10)
for row in result:
    print(row['title'], row['score'])

# As DataFrame
df = graph.betweenness_centrality(top_k=10, to_df=True)

# With sampling for large graphs
result = graph.betweenness_centrality(sample_size=100, timeout_ms=5000)

# PageRank
result = graph.pagerank(damping_factor=0.85, top_k=10)

# Degree centrality
result = graph.degree_centrality(normalized=True, top_k=10)

# Closeness centrality
result = graph.closeness_centrality(top_k=10)

# As dict
scores = graph.pagerank(as_dict=True)
# {'alice': 0.15, 'bob': 0.12, ...}
```

### Community Detection

```python
# Louvain
result = graph.louvain_communities(resolution=1.0)
# {'communities': {0: [{id, title, type}, ...], 1: [...]},
#  'modularity': 0.45, 'num_communities': 3}

# With edge type filtering and weights
result = graph.louvain_communities(
    weight_property='strength',
    connection_types=['KNOWS'],
    timeout_ms=10000)

# Label propagation
result = graph.label_propagation(max_iterations=100)
```

---

## Set Operations

Combine selections from different query chains on the same graph:

```python
young = graph.type_filter('Person').filter({'age': {'<': 25}})
oslo = graph.type_filter('Person').filter({'city': 'Oslo'})

# Union — nodes in either selection
young.union(oslo).get_nodes()

# Intersection — nodes in both selections
young.intersection(oslo).get_nodes()

# Difference — in young but not in oslo
young.difference(oslo).get_nodes()

# Symmetric difference — in exactly one selection
young.symmetric_difference(oslo).get_nodes()
```

---

## Mutation

### Fluent Update

```python
# Batch-update all selected nodes
result = graph.type_filter('Person').filter({'city': 'Oslo'}).update({
    'region': 'Eastern Norway',
    'updated': True,
})
print(result['nodes_updated'])
```

### Cypher Mutations

For CREATE, SET, DELETE, REMOVE, and MERGE — see [CYPHER.md](CYPHER.md#create--set--delete--remove--merge).

---

## Subgraph Extraction

```python
# Extract selected nodes + their inter-edges into a new independent graph
sub = graph.type_filter('Person').filter({'city': 'Oslo'}).to_subgraph()

# Preview what would be extracted
stats = graph.type_filter('Person').expand(2).subgraph_stats()
# {'node_count': 42, 'edge_count': 78, 'node_types': ['Person', 'Company'], ...}
```

---

## Pattern Matching

```python
# Cypher-like pattern matching without full Cypher
matches = graph.match_pattern('(a:Person)-[:KNOWS]->(b:Person)', max_matches=100)
# [{'a': {id, title, type, ...}, 'b': {id, title, type, ...}}, ...]

# Undirected and incoming
graph.match_pattern('(a:Person)-[:KNOWS]-(b:Person)')
graph.match_pattern('(a:Person)<-[:KNOWS]-(b:Person)')

# With inline properties
graph.match_pattern("(a:Person {city: 'Oslo'})-[:KNOWS]->(b:Person)")
```

---

## Indexes

### Equality Indexes

```python
graph.create_index('Person', 'city')
graph.has_index('Person', 'city')        # True
graph.index_stats('Person', 'city')      # {type, property, unique_values}
graph.list_indexes()                     # [{type, property}, ...]
graph.drop_index('Person', 'city')
graph.rebuild_indexes()                  # rebuild all
```

### Range Indexes (B-Tree)

```python
# Fast >, >=, <, <=, BETWEEN queries
graph.create_range_index('Person', 'age')
graph.drop_range_index('Person', 'age')
```

### Composite Indexes

```python
graph.create_composite_index('Person', ['city', 'age'])
graph.has_composite_index('Person', ['city', 'age'])
graph.composite_index_stats('Person', ['city', 'age'])
graph.list_composite_indexes()
graph.drop_composite_index('Person', ['city', 'age'])
```

### Unified Index View

```python
graph.indexes()
# [{'node_type': 'Person', 'property': 'city', 'type': 'equality'},
#  {'node_type': 'Person', 'properties': ['city', 'age'], 'type': 'composite'}]
```

---

## Transactions

```python
# Read-write transaction (snapshot isolation + optimistic concurrency)
with graph.begin() as tx:
    tx.cypher("CREATE (:Person {name: 'Alice', age: 30})")
    tx.cypher("CREATE (:Person {name: 'Bob', age: 25})")
    tx.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    # Auto-commits on success, auto-rolls-back on exception

# Manual control
tx = graph.begin()
tx.cypher("CREATE (:Person {name: 'Charlie'})")
tx.commit()   # or tx.rollback()

# Read-only transaction (O(1) cost, zero memory overhead)
with graph.begin_read() as tx:
    result = tx.cypher("MATCH (n:Person) RETURN n.name")
    # Mutations rejected with RuntimeError
```

---

## Export

```python
# File export (format inferred from extension)
graph.export('graph.graphml')              # GraphML
graph.export('graph.gexf')                 # GEXF
graph.export('graph.json')                 # D3 JSON
graph.export('graph.csv')                  # CSV

# Selection-only export
graph.type_filter('Person').export('people.graphml', selection_only=True)

# String export (no file)
xml = graph.export_string('graphml')

# CSV directory tree with blueprint (round-trip with from_blueprint)
graph.export_csv('output/')
# output/
# ├── nodes/
# │   ├── Person.csv
# │   └── Company.csv
# ├── connections/
# │   └── WORKS_AT.csv
# └── blueprint.json
```

---

## Persistence

```python
graph.save('graph.kgl')
graph = kglite.load('graph.kgl')
```

---

## Schema & Introspection

```python
# Text summary
print(graph.get_schema())

# Full schema dict
schema = graph.schema()
# {'node_types': {'Person': {count, properties}}, 'connection_types': {...},
#  'indexes': [...], 'node_count': 1000, 'edge_count': 2000}

# Node type counts
graph.node_type_counts()
# {'Person': 500, 'Company': 100}

# Property statistics for a type
graph.properties('Person', max_values=20)
# {'age': {'type': 'int', 'non_null': 500, 'unique': 50, 'values': [...]}, ...}

# Connection topology
graph.neighbors_schema('Person')
# {'outgoing': [{'connection_type': 'KNOWS', 'target_type': 'Person', 'count': 800}],
#  'incoming': [...]}

# Connection types with counts
graph.connection_types()

# Quick sample
graph.sample('Person', n=5)

# Selection state
print(graph.get_selection())
graph.clear()  # reset selection

# Execution plan for current chain
print(graph.type_filter('Person').filter({'age': {'>': 25}}).explain())
# TYPE_FILTER Person (500 nodes) -> FILTER (42 nodes)
```

### Schema Definition & Validation

```python
graph.define_schema({
    'nodes': {
        'Person': {
            'properties': {'name': 'string', 'age': 'integer'},
        },
    },
    'connections': {
        'KNOWS': {'source': 'Person', 'target': 'Person'},
    },
})

errors = graph.validate_schema(strict=True)
graph.has_schema()       # True
graph.clear_schema()
```

### AI Agent Introspection

```python
# XML description for AI agents (progressive disclosure)
print(graph.describe())                              # inventory overview
print(graph.describe(types=['Field', 'Well']))        # focused detail
print(graph.describe(connections=True))               # all connection types
print(graph.describe(connections=['BELONGS_TO']))      # deep-dive
print(graph.describe(cypher=True))                    # Cypher reference
print(graph.describe(cypher=['cluster', 'MATCH']))    # detailed topic docs

# Declare child types (bubbles capabilities into parent descriptor)
graph.set_parent_type('ProductionProfile', 'Field')

# MCP server quickstart
print(KnowledgeGraph.explain_mcp())
```

---

## Graph Maintenance

```python
# Rebuild all indexes
graph.reindex()

# Compact graph (remove tombstones from deletions)
result = graph.vacuum()
# {'nodes_remapped': 50, 'tombstones_removed': 50}

# Auto-vacuum after DELETE operations
graph.set_auto_vacuum(0.3)     # trigger at 30% fragmentation (default)
graph.set_auto_vacuum(0.2)     # more aggressive
graph.set_auto_vacuum(None)    # disable

# Storage health diagnostics
info = graph.graph_info()
# {'node_count': 1000, 'node_capacity': 1050, 'node_tombstones': 50,
#  'edge_count': 2000, 'fragmentation_ratio': 0.048, ...}

# Read-only mode (blocks all Cypher mutations)
graph.read_only(True)
graph.read_only()   # → True
graph.read_only(False)
```

---

## Code Entity Methods

Methods for code knowledge graphs built with the `code_tree` subpackage:

```python
# Find code entities by name
matches = graph.find('execute')
matches = graph.find('execute', node_type='Function')
matches = graph.find('exec', match_type='contains')       # substring
matches = graph.find('get_', match_type='starts_with')     # prefix

# Get source location
loc = graph.source('MyClass.my_method')
# {'file_path': 'src/main.py', 'line_number': 42, 'end_line': 55, ...}

# Batch source lookup
locs = graph.source(['func_a', 'func_b'])

# Neighborhood context
ctx = graph.context('MyClass', hops=2)
# {'node': {...}, 'defined_in': 'src/main.py', 'HAS_METHOD': [...], ...}

# Table of contents for a file
toc = graph.toc('src/main.py')
# {'file': 'src/main.py', 'entities': [...], 'summary': {'Function': 5, 'Class': 2}}
```

---

## Operation Reports

```python
graph.get_last_report()       # most recent operation report
graph.get_operation_index()   # sequential operation counter
graph.get_report_history()    # all reports
```

---

## Fluent vs Cypher Feature Matrix

| Feature | Fluent API | Cypher |
|---------|-----------|--------|
| **Spatial — point filters** | `within_bounds`, `near_point`, `near_point_m`, `contains_point` | `distance()`, `contains()` in WHERE |
| **Spatial — geometry** | `intersects_geometry`, `set_spatial`, `get_spatial` | `intersects()`, `centroid()`, `area()`, `perimeter()` |
| **Spatial — bounds/centroid** | `get_bounds()`, `get_centroid()`, `wkt_centroid()` | Manual via `latitude()`, `longitude()` |
| **Temporal — point-in-time** | `valid_at()` | `valid_at(e, date, 'from', 'to')` |
| **Temporal — range overlap** | `valid_during()` | `valid_during(e, start, end, 'from', 'to')` |
| **Timeseries — load** | `add_timeseries`, `add_ts_channel`, `set_time_index` | N/A (load via fluent) |
| **Timeseries — extract** | `get_timeseries()`, `get_time_index()` | `ts_series()` |
| **Timeseries — aggregate** | N/A (use Cypher) | `ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_at`, `ts_delta` |
| **Vector — load** | `set_embeddings`, `embed_texts`, `set_embedder` | N/A (load via fluent) |
| **Vector — search** | `vector_search()`, `search_text()` | `text_score()` in RETURN/WHERE |
| **Path finding** | `shortest_path`, `all_paths` + 3 variants | `shortestPath()` in MATCH |
| **Centrality** | `betweenness_centrality`, `pagerank`, `degree_centrality`, `closeness_centrality` | `CALL pagerank() YIELD ...` etc. |
| **Community** | `louvain_communities`, `label_propagation` | `CALL louvain() YIELD ...` etc. |
| **Pattern matching** | `match_pattern()`, `traverse()`, `expand()` | Full MATCH with patterns |
| **Filtering** | `filter`, `filter_any`, `has_connection`, etc. | WHERE clause |
| **Aggregation** | `statistics()`, `count()`, `calculate()` | `count()`, `sum()`, `avg()` in RETURN |
| **Mutations** | `update()` (batch property update) | CREATE, SET, DELETE, REMOVE, MERGE |
| **Transactions** | `begin()`, `begin_read()` | Implicit (each `cypher()` is atomic) |
| **Schema** | `define_schema()`, `validate_schema()` | N/A |
| **Set operations** | `union`, `intersection`, `difference`, `symmetric_difference` | `UNION` (result-level only) |
| **Indexes** | `create_index`, `create_range_index`, `create_composite_index` | Auto-maintained |
