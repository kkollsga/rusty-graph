# rusty-graph Feature Wishlist

Features that would significantly improve the Norwegian Petroleum Knowledge Graph implementation.

---

## High Priority

### 1. Temporal Query Support

**Problem**: Estimates and relationships have validity periods (`date_from`/`date_to`), but there's no native way to query "what was valid at time X".

**Current workaround**: Manual filtering on date properties.

**Desired API**:
```python
# Get estimates valid on a specific date
graph.type_filter('ProspectEstimate').valid_at('2020-06-15')

# Get relationships valid during a period
graph.type_filter('Play').traverse(
    'HAS_PROSPECT',
    valid_at='2020-06-15'  # Only follow connections valid at this date
)

# Time-travel query: graph state at a point in time
graph.as_of('2020-01-01').type_filter('Prospect').get_nodes()
```

---

### 2. Spatial/Geometry Operations

**Problem**: WKT polygons stored as strings with no query capability. Cannot do intersection, containment, or distance queries.

**Current workaround**: Store geometries in separate nodes, retrieve as strings, process externally.

**Desired API**:
```python
# Find entities within a bounding box
graph.type_filter('Discovery').within_bounds(
    min_lat=58.0, max_lat=62.0,
    min_lon=1.0, max_lon=5.0
)

# Find entities intersecting a geometry
graph.type_filter('Field').intersects(wkt_polygon)

# Distance queries
graph.type_filter('Wellbore').within_distance(
    lat=60.5, lon=3.2,
    radius_km=50
)

# Spatial join
graph.type_filter('Prospect').spatial_join(
    target_type='Block',
    relationship='INTERSECTS'
)
```

---

### 3. Connection Property Filtering in Traversal

**Problem**: Can filter target nodes during traversal, but unclear how to filter based on connection properties.

**Current workaround**: Retrieve all connections, filter in Python.

**Desired API**:
```python
# Filter traversal by connection properties
graph.type_filter('Discovery').traverse(
    'EXTENDS_INTO',
    filter_connection={'share_pct': {'>=': 50.0}}  # Only major shares
)

# Get connections with specific properties
graph.type_filter('Wellbore').traverse(
    'PENETRATES',
    filter_connection={'top_depth': {'<': 2000}},  # Shallow penetrations
    filter_target={'unit_type': 'FORMATION'}
)
```

---

### 4. Set Operations on Selections

**Problem**: No way to combine or intersect multiple query results.

**Desired API**:
```python
# Union: prospects in N3 OR M3
n3_prospects = graph.type_filter('Prospect').filter({'geoprovince': 'N3'})
m3_prospects = graph.type_filter('Prospect').filter({'geoprovince': 'M3'})
combined = n3_prospects.union(m3_prospects)

# Intersection: discoveries that are BOTH oil AND in block 34
oil_discoveries = graph.type_filter('Discovery').filter({'hc_type': 'OIL'})
block34 = graph.type_filter('Discovery').filter({'block_name': '34'})
result = oil_discoveries.intersection(block34)

# Difference: prospects without estimates
all_prospects = graph.type_filter('Prospect')
with_estimates = graph.type_filter('Prospect').traverse('HAS_ESTIMATE', direction='outgoing')
without_estimates = all_prospects.difference(with_estimates)
```

---

### 5. Path Finding and Graph Algorithms

**Problem**: No shortest path, all paths, or common graph algorithms.

**Desired API**:
```python
# Shortest path between nodes
path = graph.shortest_path(
    source_type='Prospect', source_id=12345,
    target_type='Field', target_id=67890
)

# All paths up to N hops
paths = graph.all_paths(
    source_type='Play', source_id=100,
    target_type='Wellbore',
    max_hops=4
)

# Connected components
components = graph.connected_components()

# Centrality measures
centrality = graph.type_filter('Play').betweenness_centrality()
```

---

## Medium Priority

### 6. Native Date/DateTime Type

**Problem**: Dates stored as strings, requiring string comparison for temporal filtering.

**Desired behavior**:
```python
# Native date handling
graph.add_nodes(
    data=df,
    node_type='ProspectEstimate',
    date_columns=['date_officially_valid_from', 'date_officially_valid_to']  # Auto-parse
)

# Date arithmetic in filters
graph.type_filter('Discovery').filter({
    'discovery_year': {'>=': 'now - 10 years'}
})

# Date range queries
graph.type_filter('Wellbore').filter({
    'entry_date': {'between': ['2020-01-01', '2020-12-31']}
})
```

---

### 7. Aggregation on Connection Properties

**Problem**: Can aggregate node properties but not connection properties.

**Desired API**:
```python
# Sum connection property
total_share = graph.type_filter('Discovery').traverse('EXTENDS_INTO').calculate(
    expression='sum(share_pct)',  # Connection property
    store_as='total_share_pct',
    aggregate_connections=True
)

# Statistics on connection properties
depth_stats = graph.type_filter('Wellbore').get_connection_statistics(
    connection_type='PENETRATES',
    property='top_depth'
)
```

---

### 8. Schema Definition and Validation

**Problem**: No way to define expected node types, required properties, or cardinality constraints.

**Desired API**:
```python
# Define schema
graph.define_schema({
    'nodes': {
        'Prospect': {
            'required': ['npdid_prospect', 'prospect_name'],
            'optional': ['prospect_status', 'prospect_geoprovince'],
            'types': {
                'npdid_prospect': 'integer',
                'prospect_name': 'string',
                'prospect_ns_dec': 'float'
            }
        }
    },
    'connections': {
        'HAS_ESTIMATE': {
            'source': 'Prospect',
            'target': 'ProspectEstimate',
            'cardinality': '1:N'
        }
    }
})

# Validate graph against schema
validation_errors = graph.validate_schema()
```

---

### 9. Multi-Hop Pattern Matching

**Problem**: Chaining `traverse()` is verbose for complex patterns.

**Desired API**:
```python
# Path pattern (like Cypher)
results = graph.match_pattern(
    '(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)'
)

# Variable length paths
results = graph.match_pattern(
    '(o:Ocean)-[:CONTAINS_PLAY|HAS_PROSPECT*1..3]->(target)'
)

# With conditions
results = graph.match_pattern(
    '(p:Play)-[:HAS_PROSPECT]->(pr:Prospect {status: "OD-estimat"})'
)
```

---

### 10. Subgraph Extraction

**Problem**: No way to extract a portion of the graph for analysis or export.

**Desired API**:
```python
# Extract subgraph from selection
north_sea_subgraph = (
    graph.type_filter('Ocean')
    .filter({'name': 'Nordsjøen'})
    .expand(hops=3)  # Include all nodes within 3 hops
    .to_subgraph()
)

# Save subgraph
north_sea_subgraph.save('north_sea.bin')

# Export to other formats
north_sea_subgraph.export('north_sea.graphml', format='graphml')
```

---

## Lower Priority (Nice to Have)

### 11. Null/Missing Value Handling

```python
# Explicit null checks
graph.type_filter('Discovery').filter({
    'npdid_field': {'is_null': True}  # Undeveloped discoveries
})

graph.type_filter('Prospect').filter({
    'prospect_geoprovince': {'is_not_null': True}
})
```

---

### 12. Index Management

```python
# Create index for faster filtering
graph.create_index('Prospect', 'prospect_geoprovince')
graph.create_index('Discovery', ['discovery_year', 'discovery_hc_type'])

# List indexes
graph.list_indexes()

# Drop index
graph.drop_index('Prospect', 'prospect_geoprovince')
```

---

### 13. Export to Visualization Formats

```python
# Export for visualization tools
graph.export('petroleum.graphml', format='graphml')
graph.export('petroleum.gexf', format='gexf')
graph.export('petroleum.json', format='d3')  # D3.js compatible

# Export with layout
graph.export('petroleum.svg', format='svg', layout='force-directed')
```

---

### 14. Batch Property Updates

```python
# Update properties on filtered selection
graph.type_filter('Prospect').filter({'prospect_status': 'Avflagget'}).update({
    'is_active': False,
    'deactivation_reason': 'status_avflagget'
})

# Bulk update from DataFrame
graph.update_nodes(
    data=updated_df,
    node_type='Discovery',
    match_on='npdid_discovery'
)
```

---

### 15. Query Explain/Optimization

```python
# Explain query plan
plan = graph.type_filter('Prospect').traverse('HAS_ESTIMATE').explain()
print(plan)
# Output: SCAN Prospect (6775 nodes) -> TRAVERSE HAS_ESTIMATE (10954 edges) -> ...

# Suggest optimizations
graph.optimize_suggestions()
```

---

## Summary by Impact

| Priority | Feature | Impact on This Project |
|----------|---------|----------------------|
| High | Temporal queries | Essential for estimate validity periods |
| High | Spatial operations | Essential for geographic analysis |
| High | Connection property filtering | Needed for share_pct, depth filtering |
| High | Set operations | Needed for complex multi-criteria queries |
| High | Path finding | Useful for exploration lifecycle analysis |
| Medium | Native date types | Cleaner temporal handling |
| Medium | Connection aggregation | Needed for share percentage totals |
| Medium | Schema validation | Data quality assurance |
| Medium | Pattern matching | Cleaner complex queries |
| Medium | Subgraph extraction | Export for external tools |
| Lower | Null handling | Edge case handling |
| Lower | Index management | Performance optimization |
| Lower | Visualization export | Integration with viz tools |
| Lower | Batch updates | Maintenance operations |
| Lower | Query explain | Debugging and optimization |
