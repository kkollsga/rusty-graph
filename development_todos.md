# Rusty Graph Development Plan: 15 Feature Implementations

## Feature Checklist

| # | Feature | Status | Effort | Sprint |
|---|---------|--------|--------|--------|
| 1 | Null/Missing Value Handling | [x] | Very Low | 1 |
| 2 | Document DateTime Type (README) | [x] | Very Low | 1 |
| 3 | Connection Property Filtering | [x] | Low | 2 |
| 4 | Set Operations on Selections | [x] | Low | 2 |
| 5 | Temporal Query Support | [x] | Low | 1 |
| 6 | Connection Property Aggregation | [x] | Medium | 3 |
| 7 | Batch Property Updates | [x] | Low | 1 |
| 8 | Path Finding & Graph Algorithms | [x] | Medium | 3 |
| 9 | Schema Definition & Validation | [x] | Medium | 4 |
| 10 | Subgraph Extraction | [x] | Medium | 3 |
| 11 | Multi-Hop Pattern Matching | [ ] | High | 4 |
| 12 | Export to Visualization Formats | [x] | Medium | 3 |
| 13 | Index Management | [x] | Medium | 4 |
| 14 | Query Explain/Optimization | [x] | Low | 2 |
| 15 | Spatial/Geometry Operations | [ ] | High | 4 |

**Legend:** `[ ]` = Pending | `[x]` = Complete | `[~]` = In Progress

---

## Overview

This document outlines implementation plans for 15 features requested based on user feedback from the Norwegian Petroleum Knowledge Graph implementation. Each feature includes implementation strategy, performance considerations, code structure recommendations, and integration patterns.

---

## Architecture Summary (from Exploration)

### Key Files
- **Filter conditions**: `src/datatypes/values.rs` (FilterCondition enum), `src/datatypes/py_in.rs` (Python parsing)
- **Filtering logic**: `src/graph/filtering_methods.rs` (matches_condition, process_nodes)
- **Traversal**: `src/graph/traversal_methods.rs` (make_traversal)
- **Calculations**: `src/graph/calculations.rs`, `src/graph/equation_parser.rs`
- **Schema**: `src/graph/schema.rs` (DirGraph, NodeData, EdgeData, CurrentSelection)
- **Python interface**: `src/graph/mod.rs` (KnowledgeGraph #[pymethods])

### Existing Patterns
- **FilterCondition enum**: Add variants, update `parse_operator_condition()`, update `matches_condition()`
- **New methods**: Add to `KnowledgeGraph` impl in mod.rs, call internal functions
- **Aggregations**: Use `AggregateType` enum, `Evaluator` pattern
- **Type indices**: `HashMap<String, Vec<NodeIndex>>` for O(1) type lookups

---

## Feature 1: Null/Missing Value Handling

### Desired API
```python
graph.type_filter('Discovery').filter({'npdid_field': {'is_null': True}})
graph.type_filter('Prospect').filter({'geoprovince': {'is_not_null': True}})
```

### Implementation

**Step 1: Extend FilterCondition enum** (`src/datatypes/values.rs:9-18`)
```rust
pub enum FilterCondition {
    // ... existing variants
    IsNull,
    IsNotNull,
}
```

**Step 2: Add parser support** (`src/datatypes/py_in.rs:34-53`)
```rust
fn parse_operator_condition(op: &str, val: &Bound<'_, PyAny>) -> PyResult<FilterCondition> {
    match op {
        // ... existing operators
        "is_null" => Ok(FilterCondition::IsNull),
        "is_not_null" => Ok(FilterCondition::IsNotNull),
        _ => Err(...)
    }
}
```

**Step 3: Add matching logic** (`src/graph/filtering_methods.rs:8-24`)
```rust
pub fn matches_condition(value: &Value, condition: &FilterCondition) -> bool {
    match condition {
        // ... existing
        FilterCondition::IsNull => matches!(value, Value::Null),
        FilterCondition::IsNotNull => !matches!(value, Value::Null),
    }
}
```

**Step 4: Handle missing fields** - Update `filter_nodes_by_conditions` to treat missing fields as Null

### Performance Impact
- **Negligible**: Simple pattern match, no additional memory or computation

### Code Structure
- Follows existing FilterCondition pattern exactly
- No new modules needed
- 3 files modified, ~15 lines total

---

## Feature 2: Document DateTime Type Usage (README Update)

### Current State
- `Value::DateTime(NaiveDate)` exists in `values.rs:28`
- `column_types` parameter exists in `add_nodes()` but undocumented

### Implementation

**Update README.md** - Add section after "Adding Nodes":
```markdown
### Working with Dates

```python
# Specify date columns for automatic parsing
graph.add_nodes(
    data=df,
    node_type='Estimate',
    unique_id_field='estimate_id',
    column_types={'valid_from': 'datetime', 'valid_to': 'datetime'}
)

# Filter on date fields (string comparison works due to ISO format)
graph.type_filter('Estimate').filter({
    'valid_from': {'>=': '2020-01-01'},
    'valid_to': {'<=': '2020-12-31'}
})
```
```

**Also document Operation Reports API**:
```markdown
## Operation Reports

```python
# Get last operation report
report = graph.get_last_report()
print(f"Created {report['nodes_created']} nodes in {report['processing_time_ms']}ms")

# Get full operation history
history = graph.get_report_history()
```
```

### Performance Impact
- None (documentation only)

---

## Feature 3: Connection Property Filtering in Traversal

### Desired API
```python
graph.type_filter('Discovery').traverse(
    'EXTENDS_INTO',
    filter_connection={'share_pct': {'>=': 50.0}}
)
```

### Implementation

**Step 1: Add parameter to traverse** (`src/graph/mod.rs:464-501`)
```rust
fn traverse(
    &mut self,
    connection_type: String,
    // ... existing params
    filter_connection: Option<&Bound<'_, PyDict>>,  // NEW
) -> PyResult<Self>
```

**Step 2: Modify make_traversal** (`src/graph/traversal_methods.rs:11-150`)
```rust
pub fn make_traversal(
    // ... existing params
    filter_connection: Option<&HashMap<String, FilterCondition>>,  // NEW
) -> Result<(), String> {
    // In edge iteration loop:
    for edge in graph.graph.edges_directed(source_node, Direction::Outgoing) {
        if edge.weight().connection_type == connection_type {
            // NEW: Check connection properties
            if let Some(conn_filter) = filter_connection {
                let edge_props = &edge.weight().properties;
                let matches = conn_filter.iter().all(|(key, condition)| {
                    edge_props.get(key)
                        .map(|v| matches_condition(v, condition))
                        .unwrap_or(false)
                });
                if !matches {
                    continue;  // Skip this edge
                }
            }
            targets.insert(edge.target());
        }
    }
}
```

### Performance Impact
- **Minimal overhead**: One HashMap lookup per edge when filter is active
- **Optimization**: Filter is only applied when `filter_connection` is Some

### Code Structure
- Reuses existing `matches_condition()` function
- Reuses existing `pydict_to_filter_conditions()` parser
- Pattern consistent with `filter_target` parameter

---

## Feature 4: Set Operations on Selections

### Desired API
```python
n3_prospects = graph.type_filter('Prospect').filter({'geoprovince': 'N3'})
m3_prospects = graph.type_filter('Prospect').filter({'geoprovince': 'M3'})
combined = n3_prospects.union(m3_prospects)
result = oil_discoveries.intersection(block34)
without_estimates = all_prospects.difference(with_estimates)
```

### Implementation

**Step 1: Add set operation methods** (`src/graph/mod.rs`)
```rust
fn union(&self, other: &Self) -> PyResult<Self> {
    let mut new_kg = self.clone();
    // Merge selections from both graphs
    set_operations::union_selections(
        &mut new_kg.selection,
        &other.selection
    )?;
    Ok(new_kg)
}

fn intersection(&self, other: &Self) -> PyResult<Self> { ... }
fn difference(&self, other: &Self) -> PyResult<Self> { ... }
```

**Step 2: Create set_operations module** (`src/graph/set_operations.rs`)
```rust
use std::collections::HashSet;
use crate::graph::schema::CurrentSelection;

pub fn union_selections(
    target: &mut CurrentSelection,
    source: &CurrentSelection
) -> Result<(), String> {
    let target_level = target.get_level_mut(0)?;
    let source_level = source.get_level(0)?;

    // Collect all nodes from both selections
    let mut all_nodes: HashSet<NodeIndex> = target_level.get_all_nodes().into_iter().collect();
    all_nodes.extend(source_level.get_all_nodes());

    target_level.selections.clear();
    target_level.add_selection(None, all_nodes.into_iter().collect());
    Ok(())
}

pub fn intersection_selections(...) -> Result<(), String> {
    // Use HashSet intersection
    let target_set: HashSet<_> = target_level.get_all_nodes().into_iter().collect();
    let source_set: HashSet<_> = source_level.get_all_nodes().into_iter().collect();
    let result: Vec<_> = target_set.intersection(&source_set).cloned().collect();
    ...
}

pub fn difference_selections(...) -> Result<(), String> {
    // Use HashSet difference
    let result: Vec<_> = target_set.difference(&source_set).cloned().collect();
    ...
}
```

### Performance Impact
- **O(n)** where n is total nodes in both selections
- **Memory**: Temporary HashSet allocation (released after operation)
- **Optimization**: HashSet operations are O(1) per element

### Code Structure
- New module `set_operations.rs` keeps logic separate
- Add `pub mod set_operations;` to `mod.rs`
- Methods are pure functions on selections

---

## Feature 5: Temporal Query Support

### Desired API
```python
graph.type_filter('ProspectEstimate').valid_at('2020-06-15')
graph.type_filter('Play').traverse('HAS_PROSPECT', valid_at='2020-06-15')
```

### Implementation

**Option A: Sugar methods (Recommended - simpler)**

**Step 1: Add valid_at method** (`src/graph/mod.rs`)
```rust
fn valid_at(
    &mut self,
    date: &str,
    date_from_field: Option<&str>,
    date_to_field: Option<&str>,
) -> PyResult<Self> {
    let from_field = date_from_field.unwrap_or("date_from");
    let to_field = date_to_field.unwrap_or("date_to");

    // Build compound filter
    let mut conditions = HashMap::new();
    conditions.insert(from_field.to_string(),
        FilterCondition::LessThanEquals(Value::String(date.to_string())));
    conditions.insert(to_field.to_string(),
        FilterCondition::GreaterThanEquals(Value::String(date.to_string())));

    // Apply filter
    let mut new_kg = self.clone();
    filtering_methods::filter_nodes(&self.inner, &mut new_kg.selection, conditions, None, None)?;
    Ok(new_kg)
}
```

**Step 2: Add valid_during for ranges**
```rust
fn valid_during(
    &mut self,
    start_date: &str,
    end_date: &str,
    date_from_field: Option<&str>,
    date_to_field: Option<&str>,
) -> PyResult<Self>
```

**Option B: Add 'between' operator**

Add to FilterCondition:
```rust
Between(Value, Value),  // (min, max) inclusive
```

### Performance Impact
- **Negligible**: Compiles to existing filter operations
- String comparison works for ISO dates (lexicographic order matches chronological)

### Code Structure
- Sugar methods are simple wrappers around existing filter logic
- No changes to core filtering needed
- Optional: Add true DateTime comparison for better type safety

---

## Feature 6: Aggregation on Connection Properties

### Desired API
```python
total_share = graph.type_filter('Discovery').traverse('EXTENDS_INTO').calculate(
    expression='sum(share_pct)',
    aggregate_connections=True,
    store_as='total_share_pct'
)
```

### Implementation

**Step 1: Add aggregate_connections parameter** (`src/graph/mod.rs:652-747`)
```rust
fn calculate(
    &mut self,
    expression: &str,
    // ... existing
    aggregate_connections: Option<bool>,  // NEW
) -> PyResult<PyObject>
```

**Step 2: Create connection aggregation function** (`src/graph/calculations.rs`)
```rust
pub fn evaluate_connection_equation(
    graph: &DirGraph,
    selection: &CurrentSelection,
    parsed_expr: &Expr,
    level_index: Option<usize>,
) -> Vec<StatResult> {
    let pairs = get_parent_child_pairs(selection, level_index);

    pairs.iter().map(|pair| {
        let parent = pair.parent.expect("Need parent for connection aggregation");

        // Collect connection properties instead of node properties
        let objects: Vec<HashMap<String, Value>> = pair.children.iter()
            .filter_map(|&child_idx| {
                // Find edge between parent and child
                graph.graph.edges_connecting(parent, child_idx)
                    .next()
                    .map(|edge| edge.weight().properties.clone())
            })
            .collect();

        match Evaluator::evaluate(parsed_expr, &objects) {
            Ok(value) => StatResult { parent_idx: Some(parent), value, ... },
            Err(err) => StatResult { error_msg: Some(err), ... }
        }
    }).collect()
}
```

**Step 3: Add edge lookup helper** (`src/graph/schema.rs`)
```rust
impl DirGraph {
    pub fn get_edges_between(&self, source: NodeIndex, target: NodeIndex) -> Vec<&EdgeData> {
        self.graph.edges_connecting(source, target)
            .map(|e| e.weight())
            .collect()
    }
}
```

### Performance Impact
- **Same as node aggregation**: O(n) where n is number of connections
- **Memory**: Temporary HashMap per edge (same pattern as nodes)

### Code Structure
- Reuses existing `Evaluator` completely
- New function parallels `evaluate_equation`
- Follows established aggregation pattern

---

## Feature 7: Batch Property Updates

### Desired API
```python
graph.type_filter('Prospect').filter({'status': 'Avflagget'}).update({
    'is_active': False,
    'deactivation_reason': 'status_avflagget'
})
```

### Implementation

**Step 1: Expose update method** (`src/graph/mod.rs`)
```rust
fn update(
    &mut self,
    properties: &Bound<'_, PyDict>,
    keep_selection: Option<bool>,
) -> PyResult<PyObject> {
    // Convert Python dict to property updates
    let updates: HashMap<String, Value> = properties.iter()
        .map(|(k, v)| Ok((k.extract::<String>()?, py_in::py_value_to_value(&v)?)))
        .collect::<PyResult<_>>()?;

    // Get selected nodes
    let level = self.selection.get_level(self.selection.get_level_count() - 1)?;
    let nodes = level.get_all_nodes();

    // Extract graph
    let mut graph = extract_or_clone_graph(&mut self.inner);

    // Update each property
    for (property, value) in updates {
        let node_values: Vec<_> = nodes.iter()
            .map(|&idx| (Some(idx), value.clone()))
            .collect();
        maintain_graph::update_node_properties(&mut graph, &node_values, &property)?;
    }

    self.inner = Arc::new(graph);
    // Return report
}
```

### Performance Impact
- **O(n * p)** where n = nodes, p = properties being updated
- Uses existing optimized `update_node_properties`

### Code Structure
- Leverages existing `maintain_graph::update_node_properties`
- Consistent with other methods that modify graph state
- Returns operation report like other mutating methods

---

## Feature 8: Path Finding and Graph Algorithms

### Desired API
```python
path = graph.shortest_path(
    source_type='Prospect', source_id=12345,
    target_type='Field', target_id=67890
)
components = graph.connected_components()
```

### Implementation

**Step 1: Add path finding methods** (`src/graph/mod.rs`)
```rust
fn shortest_path(
    &self,
    source_type: &str,
    source_id: &Bound<'_, PyAny>,
    target_type: &str,
    target_id: &Bound<'_, PyAny>,
    max_hops: Option<usize>,
) -> PyResult<PyObject> {
    // Look up source and target nodes
    let source_idx = lookups::find_node_by_id(&self.inner, source_type, source_id)?;
    let target_idx = lookups::find_node_by_id(&self.inner, target_type, target_id)?;

    // Use petgraph's dijkstra (already available!)
    use petgraph::algo::dijkstra;
    let costs = dijkstra(&self.inner.graph, source_idx, Some(target_idx), |_| 1);

    // Reconstruct path
    graph_algorithms::reconstruct_path(&self.inner, source_idx, target_idx, &costs)
}
```

**Step 2: Create graph_algorithms module** (`src/graph/graph_algorithms.rs`)
```rust
use petgraph::algo::{dijkstra, kosaraju_scc, all_simple_paths};

pub fn shortest_path(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
) -> Option<Vec<NodeIndex>> {
    // petgraph provides this!
    let path_map = dijkstra(&graph.graph, source, Some(target), |_| 1);
    reconstruct_path(source, target, &path_map)
}

pub fn connected_components(graph: &DirGraph) -> Vec<Vec<NodeIndex>> {
    // petgraph provides this!
    kosaraju_scc(&graph.graph)
}

pub fn all_paths(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    max_hops: usize,
) -> Vec<Vec<NodeIndex>> {
    all_simple_paths(&graph.graph, source, target, 0, Some(max_hops))
        .collect()
}
```

### Performance Impact
- **Dijkstra**: O(V + E log V) - efficient for sparse graphs
- **Connected components**: O(V + E)
- **All paths**: Exponential worst case, mitigated by max_hops limit

### Code Structure
- New module `graph_algorithms.rs` wraps petgraph algorithms
- Petgraph already included as dependency (0.7.1)
- Algorithms are well-tested in petgraph library

---

## Feature 9: Schema Definition and Validation

### Desired API
```python
graph.define_schema({
    'nodes': {
        'Prospect': {
            'required': ['npdid_prospect', 'prospect_name'],
            'types': {'npdid_prospect': 'integer', 'prospect_name': 'string'}
        }
    },
    'connections': {
        'HAS_ESTIMATE': {'source': 'Prospect', 'target': 'ProspectEstimate'}
    }
})
validation_errors = graph.validate_schema()
```

### Implementation

**Step 1: Add schema definition storage** (`src/graph/schema.rs`)
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDefinition {
    pub node_schemas: HashMap<String, NodeSchema>,
    pub connection_schemas: HashMap<String, ConnectionSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSchema {
    pub required_fields: Vec<String>,
    pub optional_fields: Vec<String>,
    pub field_types: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionSchema {
    pub source_type: String,
    pub target_type: String,
    pub cardinality: Option<String>,
}
```

**Step 2: Add to DirGraph**
```rust
pub struct DirGraph {
    pub(crate) graph: Graph,
    pub(crate) type_indices: HashMap<String, Vec<NodeIndex>>,
    pub(crate) schema_definition: Option<SchemaDefinition>,  // NEW
}
```

**Step 3: Add validation method** (`src/graph/schema_validation.rs`)
```rust
pub fn validate_graph(
    graph: &DirGraph,
    schema: &SchemaDefinition,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Check each node type
    for (node_type, node_schema) in &schema.node_schemas {
        if let Some(nodes) = graph.type_indices.get(node_type) {
            for &node_idx in nodes {
                if let Some(node) = graph.get_node(node_idx) {
                    // Check required fields
                    for required in &node_schema.required_fields {
                        if node.get_field(required).is_none() {
                            errors.push(ValidationError::MissingRequired {
                                node_idx, field: required.clone()
                            });
                        }
                    }
                    // Check field types
                    for (field, expected_type) in &node_schema.field_types {
                        if let Some(value) = node.get_field(field) {
                            if !value_matches_type(value, expected_type) {
                                errors.push(ValidationError::TypeMismatch { ... });
                            }
                        }
                    }
                }
            }
        }
    }
    errors
}
```

### Performance Impact
- **Schema storage**: Negligible memory overhead
- **Validation**: O(n) scan of all nodes - run on-demand only

### Code Structure
- New module `schema_validation.rs`
- SchemaDefinition is Optional - doesn't affect existing code
- Validation is opt-in, not automatic

---

## Feature 10: Subgraph Extraction

### Desired API
```python
subgraph = (
    graph.type_filter('Ocean')
    .filter({'name': 'Nordsjoen'})
    .expand(hops=3)
    .to_subgraph()
)
subgraph.save('north_sea.bin')
```

### Implementation

**Step 1: Add expand method** (`src/graph/mod.rs`)
```rust
fn expand(&mut self, hops: usize) -> PyResult<Self> {
    let mut new_kg = self.clone();
    subgraph::expand_selection(&self.inner, &mut new_kg.selection, hops)?;
    Ok(new_kg)
}
```

**Step 2: Create subgraph module** (`src/graph/subgraph.rs`)
```rust
pub fn expand_selection(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    hops: usize,
) -> Result<(), String> {
    let level = selection.get_level_mut(0)?;
    let mut frontier: HashSet<NodeIndex> = level.get_all_nodes().into_iter().collect();
    let mut visited = frontier.clone();

    for _ in 0..hops {
        let mut next_frontier = HashSet::new();
        for &node in &frontier {
            // Add all neighbors
            for neighbor in graph.graph.neighbors_undirected(node) {
                if visited.insert(neighbor) {
                    next_frontier.insert(neighbor);
                }
            }
        }
        frontier = next_frontier;
    }

    level.selections.clear();
    level.add_selection(None, visited.into_iter().collect());
    Ok(())
}

pub fn extract_subgraph(
    source: &DirGraph,
    selection: &CurrentSelection,
) -> DirGraph {
    let nodes = selection.get_level(0).unwrap().get_all_nodes();
    let node_set: HashSet<_> = nodes.iter().collect();

    let mut new_graph = DirGraph::new();

    // Copy selected nodes
    let mut index_map = HashMap::new();
    for &old_idx in &nodes {
        if let Some(node) = source.get_node(old_idx) {
            let new_idx = new_graph.graph.add_node(node.clone());
            index_map.insert(old_idx, new_idx);
        }
    }

    // Copy edges between selected nodes
    for &old_idx in &nodes {
        for edge in source.graph.edges(old_idx) {
            if node_set.contains(&edge.target()) {
                let new_source = index_map[&old_idx];
                let new_target = index_map[&edge.target()];
                new_graph.graph.add_edge(new_source, new_target, edge.weight().clone());
            }
        }
    }

    new_graph
}
```

**Step 3: Add to_subgraph method** (`src/graph/mod.rs`)
```rust
fn to_subgraph(&self) -> PyResult<Self> {
    let extracted = subgraph::extract_subgraph(&self.inner, &self.selection);
    Ok(KnowledgeGraph {
        inner: Arc::new(extracted),
        selection: CurrentSelection::new(),
        reports: OperationReports::new(),
    })
}
```

### Performance Impact
- **expand()**: O(hops * avg_degree * n) - BFS expansion
- **to_subgraph()**: O(n + e) where n=nodes, e=edges in selection
- **Memory**: New graph allocation for subgraph

### Code Structure
- New module `subgraph.rs`
- Returns new KnowledgeGraph (independent copy)
- expand() is non-destructive (returns new selection)

---

## Feature 11: Multi-Hop Pattern Matching

### Desired API
```python
results = graph.match_pattern(
    '(p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)'
)
```

### Implementation

**This is the most complex feature. Recommend phased approach:**

**Phase 1: Simple path patterns**
```rust
fn match_path(
    &self,
    path_spec: Vec<(&str, &str, &str)>,  // [(source_type, connection_type, target_type), ...]
) -> PyResult<PyObject>
```

**Phase 2: Variable binding and return**
```rust
struct PathPattern {
    steps: Vec<PathStep>,
    return_bindings: Vec<String>,
}

struct PathStep {
    source_binding: String,
    source_type: Option<String>,
    connection_type: String,
    target_binding: String,
    target_type: Option<String>,
}
```

**Phase 3: Full parser for Cypher-lite syntax**
- Use nom or pest for parsing
- Build AST from pattern string
- Execute against graph

### Performance Impact
- **Simple patterns**: O(n * d^k) where n=starting nodes, d=avg degree, k=hops
- **With type constraints**: Significantly faster due to type_indices

### Code Structure
- New module `pattern_matching.rs`
- Consider using existing `traverse()` internally for each hop
- Phase 1 can be done without parser

---

## Feature 12: Export to Visualization Formats

### Desired API
```python
graph.export('petroleum.graphml', format='graphml')
graph.export('petroleum.json', format='d3')
```

### Implementation

**Step 1: Create export module** (`src/graph/export.rs`)
```rust
pub fn to_graphml(graph: &DirGraph, selection: Option<&CurrentSelection>) -> String {
    let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<graphml xmlns=\"http://graphml.graphstorm.org/xmlns\">\n");
    xml.push_str("  <graph id=\"G\" edgedefault=\"directed\">\n");

    // Export nodes
    for (node_type, indices) in &graph.type_indices {
        for &idx in indices {
            if let Some(NodeData::Regular { id, title, properties, .. }) = graph.get_node(idx) {
                xml.push_str(&format!("    <node id=\"{}\">\n", idx.index()));
                xml.push_str(&format!("      <data key=\"type\">{}</data>\n", node_type));
                // ... more properties
                xml.push_str("    </node>\n");
            }
        }
    }

    // Export edges
    for edge in graph.graph.edge_references() {
        xml.push_str(&format!(
            "    <edge source=\"{}\" target=\"{}\">\n",
            edge.source().index(), edge.target().index()
        ));
        // ... edge properties
        xml.push_str("    </edge>\n");
    }

    xml.push_str("  </graph>\n</graphml>");
    xml
}

pub fn to_d3_json(graph: &DirGraph) -> String {
    // Build JSON structure for D3.js force-directed graph
    let nodes: Vec<_> = graph.graph.node_indices()
        .filter_map(|idx| graph.get_node(idx).map(|n| json_node(idx, n)))
        .collect();

    let links: Vec<_> = graph.graph.edge_references()
        .map(|e| json_link(e))
        .collect();

    serde_json::json!({ "nodes": nodes, "links": links }).to_string()
}
```

**Step 2: Add export method** (`src/graph/mod.rs`)
```rust
fn export(&self, path: &str, format: &str) -> PyResult<()> {
    let content = match format {
        "graphml" => export::to_graphml(&self.inner, Some(&self.selection)),
        "d3" | "json" => export::to_d3_json(&self.inner),
        "gexf" => export::to_gexf(&self.inner),
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown format: {}", format)
        ))
    };

    std::fs::write(path, content)?;
    Ok(())
}
```

### Performance Impact
- **O(n + e)**: Single pass through nodes and edges
- **Memory**: String buffer for output

### Code Structure
- New module `export.rs`
- Add `serde_json` dependency (likely already present)
- Format-specific functions keep code organized

---

## Feature 13: Index Management

### Desired API
```python
graph.create_index('Prospect', 'prospect_geoprovince')
graph.list_indexes()
graph.drop_index('Prospect', 'prospect_geoprovince')
```

### Implementation

**Step 1: Add index storage** (`src/graph/schema.rs`)
```rust
pub struct DirGraph {
    // ... existing
    pub(crate) property_indices: HashMap<(String, String), HashMap<Value, Vec<NodeIndex>>>,
}
```

**Step 2: Add index management methods** (`src/graph/mod.rs`)
```rust
fn create_index(&mut self, node_type: &str, property: &str) -> PyResult<()> {
    let mut graph = extract_or_clone_graph(&mut self.inner);

    let mut index: HashMap<Value, Vec<NodeIndex>> = HashMap::new();

    if let Some(nodes) = graph.type_indices.get(node_type) {
        for &idx in nodes {
            if let Some(node) = graph.get_node(idx) {
                if let Some(value) = node.get_field(property) {
                    index.entry(value.clone()).or_default().push(idx);
                }
            }
        }
    }

    graph.property_indices.insert((node_type.to_string(), property.to_string()), index);
    self.inner = Arc::new(graph);
    Ok(())
}
```

**Step 3: Use indexes in filter_nodes** (`src/graph/filtering_methods.rs`)
```rust
// Check for indexed property
if let Some(index) = graph.property_indices.get(&(node_type, property)) {
    if let FilterCondition::Equals(value) = condition {
        if let Some(matching_nodes) = index.get(value) {
            return matching_nodes.clone();
        }
    }
}
```

### Performance Impact
- **Index creation**: O(n) one-time cost
- **Indexed lookups**: O(1) instead of O(n)
- **Memory**: O(n) additional for each index

### Code Structure
- Indexes are opt-in (user creates them)
- Transparent to query code (automatic usage)
- Stored in DirGraph for persistence

---

## Feature 14: Query Explain/Optimization

### Desired API
```python
plan = graph.type_filter('Prospect').traverse('HAS_ESTIMATE').explain()
# Output: SCAN Prospect (6775 nodes) -> TRAVERSE HAS_ESTIMATE (10954 edges)
```

### Implementation

**Step 1: Add explain mode to selection** (`src/graph/schema.rs`)
```rust
pub struct CurrentSelection {
    levels: Vec<SelectionLevel>,
    current_level: usize,
    explain_mode: bool,  // NEW
    execution_plan: Vec<PlanStep>,  // NEW
}

pub struct PlanStep {
    pub operation: String,
    pub estimated_rows: usize,
    pub actual_rows: Option<usize>,
}
```

**Step 2: Record plan steps during operations**
```rust
// In filter_nodes:
if selection.explain_mode {
    selection.execution_plan.push(PlanStep {
        operation: format!("FILTER {} on {}", node_type, conditions),
        estimated_rows: graph.type_indices.get(&node_type).map(|v| v.len()).unwrap_or(0),
        actual_rows: None,
    });
}
```

**Step 3: Add explain method** (`src/graph/mod.rs`)
```rust
fn explain(&self) -> PyResult<String> {
    let mut output = String::new();
    for (i, step) in self.selection.execution_plan.iter().enumerate() {
        if i > 0 { output.push_str(" -> "); }
        output.push_str(&format!("{} ({} rows)", step.operation,
            step.actual_rows.unwrap_or(step.estimated_rows)));
    }
    Ok(output)
}
```

### Performance Impact
- **Negligible**: Just recording operation metadata
- Only active when explain_mode is true

### Code Structure
- PlanStep records are lightweight
- Non-intrusive changes to existing operations

---

## Feature 15: Spatial/Geometry Operations

### Desired API
```python
graph.type_filter('Discovery').within_bounds(min_lat=58.0, max_lat=62.0, min_lon=1.0, max_lon=5.0)
graph.type_filter('Field').intersects(wkt_polygon)
```

### Implementation

**This is the most complex feature requiring external dependencies:**

**Step 1: Add geo dependencies** (`Cargo.toml`)
```toml
[dependencies]
geo = "0.28"
geo-types = "0.7"
wkt = "0.10"
```

**Step 2: Add spatial value type** (`src/datatypes/values.rs`)
```rust
pub enum Value {
    // ... existing
    Geometry(geo_types::Geometry<f64>),
}
```

**Step 3: Create spatial module** (`src/graph/spatial.rs`)
```rust
use geo::{Contains, Intersects, EuclideanDistance};
use wkt::TryFromWkt;

pub fn parse_wkt(wkt_string: &str) -> Result<Geometry<f64>, String> {
    Geometry::try_from_wkt_str(wkt_string)
        .map_err(|e| format!("Invalid WKT: {}", e))
}

pub fn within_bounds(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    lat_field: &str,
    lon_field: &str,
    min_lat: f64, max_lat: f64,
    min_lon: f64, max_lon: f64,
) -> Result<(), String> {
    // Filter nodes by coordinate bounds
    let conditions = HashMap::from([
        (lat_field.to_string(), FilterCondition::GreaterThanEquals(Value::Float64(min_lat))),
        (lat_field.to_string(), FilterCondition::LessThanEquals(Value::Float64(max_lat))),
        // ... similar for lon
    ]);
    filtering_methods::filter_nodes(graph, selection, conditions, None, None)
}

pub fn intersects_geometry(
    graph: &DirGraph,
    nodes: &[NodeIndex],
    geometry_field: &str,
    query_geometry: &Geometry<f64>,
) -> Vec<NodeIndex> {
    nodes.iter()
        .filter(|&&idx| {
            graph.get_node(idx)
                .and_then(|n| n.get_field(geometry_field))
                .and_then(|v| match v {
                    Value::Geometry(g) => Some(g.intersects(query_geometry)),
                    Value::String(wkt) => parse_wkt(wkt).ok().map(|g| g.intersects(query_geometry)),
                    _ => None
                })
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}
```

### Performance Impact
- **Bounding box**: O(n) with simple comparisons
- **Intersects**: O(n * geometry_complexity) - potentially expensive
- **Optimization**: Add R-tree spatial index for large datasets

### Code Structure
- New module `spatial.rs`
- Geometry type optional (parse from WKT strings on demand)
- Consider R-tree index for production use

---

## Implementation Priority & Effort Summary

| # | Feature | Effort | Files Changed | Dependencies |
|---|---------|--------|---------------|--------------|
| 1 | Null handling | Very Low | 3 | None |
| 2 | DateTime docs | Very Low | 1 (README) | None |
| 3 | Connection filter | Low | 2 | None |
| 4 | Set operations | Low | 2 (new module) | None |
| 5 | Temporal queries | Low | 1 | None |
| 6 | Connection aggregation | Medium | 2 | None |
| 7 | Batch updates | Low | 1 | None |
| 8 | Path finding | Medium | 2 (new module) | petgraph (existing) |
| 9 | Schema validation | Medium | 3 (new module) | None |
| 10 | Subgraph extraction | Medium | 2 (new module) | None |
| 11 | Pattern matching | High | 3 (new module) | Optional: nom/pest |
| 12 | Export formats | Medium | 2 (new module) | serde_json |
| 13 | Index management | Medium | 3 | None |
| 14 | Query explain | Low | 2 | None |
| 15 | Spatial operations | High | 4 (new module) | geo, wkt crates |

---

## Recommended Implementation Order

### Sprint 1: Quick Wins (1-2 days)
1. Null handling (#1)
2. DateTime documentation (#2)
3. Batch updates (#7)
4. Temporal query sugar (#5)

### Sprint 2: Core Enhancements (3-5 days)
5. Connection property filtering (#3)
6. Set operations (#4)
7. Query explain (#14)

### Sprint 3: Advanced Features (1-2 weeks)
8. Connection aggregation (#6)
9. Path finding (#8)
10. Subgraph extraction (#10)
11. Export formats (#12)

### Sprint 4: Complex Features (2-4 weeks)
12. Schema validation (#9)
13. Index management (#13)
14. Pattern matching (#11)
15. Spatial operations (#15)

---

## Verification Plan

### Testing Strategy
1. **Unit tests**: Each new function in isolation
2. **Integration tests**: Python API tests in `pytest/` directory
3. **Performance tests**: Benchmark with existing load test data

### Test Commands
```bash
# Build and test
cd /Volumes/EksternalHome/Koding/Rust/rusty_graph
maturin develop
pytest pytest/ -v

# Performance benchmark
python pytest/loadTest.py
```

### Verification Checklist
- [x] All existing tests pass
- [x] New features have test coverage
- [x] README examples work as documented
- [ ] No performance regression on load tests

---

## Implementation Status

### Feature 1: Null/Missing Value Handling - COMPLETE
**API:**
```python
graph.type_filter('Discovery').filter({'npdid_field': {'is_null': True}})
graph.type_filter('Prospect').filter({'geoprovince': {'is_not_null': True}})
```
**Tests:** `pytest/test_null_and_datetime.py`

### Feature 2: DateTime Documentation - COMPLETE
**Added to README:** Working with Dates section documenting `column_types` parameter and date filtering.

### Feature 3: Connection Property Filtering - COMPLETE
**API:**
```python
graph.type_filter('Discovery').traverse(
    'EXTENDS_INTO',
    filter_connection={'share_pct': {'>=': 50.0}}
)
# Supports all filter operators including is_null/is_not_null
```
**Tests:** `pytest/test_connection_filtering.py`

### Feature 4: Set Operations on Selections - COMPLETE
**API:**
```python
# Union - combines all nodes from both selections
combined = selection_a.union(selection_b)

# Intersection - keeps only nodes present in both
common = selection_a.intersection(selection_b)

# Difference - keeps nodes in first but not second
remaining = selection_a.difference(selection_b)

# Symmetric Difference - keeps nodes in exactly one selection
exclusive = selection_a.symmetric_difference(selection_b)

# Chaining is supported
result = selection_a.union(selection_b).intersection(selection_c)
```
**Implementation Files:**
- `src/graph/set_operations.rs` (new module)
- `src/graph/mod.rs` (Python methods)
**Tests:** `pytest/test_set_operations.py`

### Feature 5: Temporal Query Support - COMPLETE
**API:**
```python
# Find entities valid at a specific point in time
valid_estimates = graph.type_filter('Estimate').valid_at('2020-06-15')

# Use custom field names
active_contracts = graph.type_filter('Contract').valid_at(
    '2021-03-01',
    date_from_field='start_date',
    date_to_field='end_date'
)

# Find entities valid during a date range (overlapping periods)
overlapping = graph.type_filter('Estimate').valid_during('2020-01-01', '2020-06-30')

# Chain with other operations
high_value_valid = graph.type_filter('Estimate').valid_at('2020-06-15').filter({'value': {'>=': 100.0}})
```
**Implementation Notes:**
- `valid_at(date)` filters where `date_from <= date <= date_to`
- `valid_during(start, end)` filters where validity period overlaps with range
- Default field names are 'date_from' and 'date_to'
- Fixed a bug in filter chaining that was clearing selections at level 0
**Tests:** `pytest/test_temporal_queries.py`

### Feature 7: Batch Property Updates - COMPLETE

**API:**

```python
# Select nodes and update them in batch
result = graph.type_filter('Prospect').filter({'status': 'Inactive'}).update({
    'is_active': False,
    'deactivation_reason': 'status_inactive'
})

# Access the updated graph and count
updated_graph = result['graph']
nodes_updated = result['nodes_updated']

# Use keep_selection=True to preserve selection for chaining
result = selection.update({'processed': True}, keep_selection=True)

# Supports multiple value types (int, float, bool, string)
```

**Implementation Notes:**

- Returns dictionary with `graph`, `nodes_updated`, and `report_index`
- By default clears selection; use `keep_selection=True` to preserve it
- Raises error if selection is empty (no nodes to update)
- Leverages existing `maintain_graph::update_node_properties` for efficiency

**Tests:** `pytest/test_batch_updates.py`

### Feature 14: Query Explain/Optimization - COMPLETE

**API:**

```python
# Build a query chain
result = (
    graph.type_filter('Prospect')
    .filter({'region': 'North'})
    .traverse('HAS_ESTIMATE')
)

# See the execution plan
print(result.explain())
# Output: TYPE_FILTER Prospect (6775 nodes) -> FILTER (3200 nodes) -> TRAVERSE HAS_ESTIMATE (10954 nodes)

# Works with temporal queries too
valid_estimates = graph.type_filter('Estimate').valid_at('2020-06-15')
print(valid_estimates.explain())
# Output: TYPE_FILTER Estimate (1000 nodes) -> VALID_AT (450 nodes)
```

**Implementation Notes:**

- Added `PlanStep` struct in `schema.rs` to track query operations
- Each query operation (type_filter, filter, traverse, valid_at, valid_during) records a plan step
- `explain()` method returns human-readable execution plan string
- Shows actual node counts at each step in the query chain

**Tests:** `pytest/test_query_explain.py`

### Feature 6: Connection Property Aggregation - COMPLETE

**API:**

```python
# Sum connection properties - aggregate values stored on edges
total_shares = graph.type_filter('Discovery').traverse('EXTENDS_INTO').calculate(
    expression='sum(share_pct)',
    aggregate_connections=True  # Key parameter for connection aggregation
)
# Returns {'Discovery A': 100.0, 'Discovery B': 100.0}

# Average connection properties
avg_ownership = graph.type_filter('Company').traverse('OWNS').calculate(
    expression='avg(ownership_pct)',
    aggregate_connections=True
)

# Count connections
connection_count = graph.type_filter('Parent').traverse('HAS_CHILD').calculate(
    expression='count(property)',
    aggregate_connections=True
)

# Store aggregated results on parent nodes
updated_graph = graph.type_filter('Prospect').traverse('HAS_ESTIMATE').calculate(
    expression='sum(weight)',
    aggregate_connections=True,
    store_as='total_weight'
)
```

**Supported Aggregate Functions:**

- `sum(property)` - Sum of property values
- `avg(property)` / `mean(property)` - Average of property values
- `min(property)` - Minimum value
- `max(property)` - Maximum value
- `count(property)` - Count of connections with non-null property values
- `std(property)` - Standard deviation

**Implementation Notes:**

- Added `aggregate_connections` parameter to `calculate()` method
- Created `evaluate_connection_equation()` function in `calculations.rs`
- Added `Count` variant to `AggregateType` enum in `equation_parser.rs`
- Skips node property schema validation when aggregating connection properties
- Requires a traversal before calculate (at least 2 selection levels)
- Results grouped by parent (source) node of the traversal

**Tests:** `pytest/test_connection_aggregation.py`

### Feature 9: Schema Definition & Validation - COMPLETE

**API:**

```python
# Define schema for the graph
graph.define_schema({
    'nodes': {
        'Person': {
            'required': ['name', 'email'],
            'optional': ['phone'],
            'types': {
                'name': 'string',
                'email': 'string',
                'age': 'integer'
            }
        },
        'Company': {
            'required': ['company_name'],
            'types': {'company_name': 'string', 'founded': 'integer'}
        }
    },
    'connections': {
        'WORKS_AT': {
            'source': 'Person',
            'target': 'Company',
            'cardinality': 'many-to-one',
            'required_properties': ['start_date'],
            'property_types': {'salary': 'float'}
        }
    }
})

# Validate graph against schema
errors = graph.validate_schema()  # Returns list of validation error dicts
errors = graph.validate_schema(strict=True)  # Also reports undefined types

# Check if schema is defined
if graph.has_schema():
    schema = graph.get_schema_definition()  # Get schema as dict

# Clear schema
graph.clear_schema()
```

**Validation Error Types:**

- `missing_required_field` - A required field is missing from a node
- `type_mismatch` - A field has the wrong type
- `invalid_connection_endpoint` - Connection has wrong source/target types
- `missing_connection_property` - Required property missing from connection
- `undefined_node_type` - Node type exists but not in schema (strict mode)
- `undefined_connection_type` - Connection type exists but not in schema (strict mode)

**Supported Type Names:**

- `string`, `str` - String values
- `integer`, `int`, `i64` - Integer values
- `float`, `double`, `f64`, `number` - Numeric values (float or int)
- `boolean`, `bool` - Boolean values
- `datetime`, `date` - DateTime values
- `null` - Null values
- `any` - Any type is valid

**Implementation Notes:**

- Added `SchemaDefinition`, `NodeSchemaDefinition`, `ConnectionSchemaDefinition` structs to `schema.rs`
- Added `ValidationError` enum with Display implementation for human-readable messages
- Added `schema_definition: Option<SchemaDefinition>` field to `DirGraph` (serde-compatible)
- Created `schema_validation.rs` module with validation logic
- Methods: `define_schema()`, `validate_schema()`, `has_schema()`, `get_schema_definition()`, `clear_schema()`
- Schema is persisted when graph is saved/loaded (via serde)
- Validation is opt-in and on-demand (O(n) scan)

**Tests:** `pytest/test_schema_validation.py`

### Feature 8: Path Finding & Graph Algorithms - COMPLETE

**API:**

```python
# Find shortest path between two nodes
path = graph.shortest_path(
    source_type='Prospect', source_id=12345,
    target_type='Field', target_id=67890
)
# Returns: {'path': [node_info_dicts], 'connections': ['CONN_TYPE', ...], 'length': 3}

# Find all paths up to N hops
paths = graph.all_paths(
    source_type='Person', source_id='alice',
    target_type='Person', target_id='bob',
    max_hops=5
)
# Returns: [{'path': [...], 'connections': [...], 'length': 2}, ...]

# Get connected components
components = graph.connected_components(weak=True)  # Weakly connected (default)
components = graph.connected_components(weak=False)  # Strongly connected

# Check if two nodes are connected
connected = graph.are_connected(
    source_type='A', source_id=1,
    target_type='B', target_id=2
)
# Returns: True/False

# Get degrees for selected nodes
degrees = graph.type_filter('Person').get_degrees()
# Returns: {'Alice': 5, 'Bob': 3, ...}
```

**Implementation Notes:**

- Created `graph_algorithms.rs` module with path finding algorithms
- Uses BFS for shortest path (undirected, more appropriate for knowledge graphs)
- Uses DFS for all_paths with max_hops limit
- Uses Kosaraju's algorithm (via petgraph) for strongly connected components
- Custom weakly connected components implementation using BFS
- Pre-allocates HashMaps/HashSets with capacity estimates for performance

**Tests:** `pytest/test_graph_algorithms.py`

### Feature 10: Subgraph Extraction - COMPLETE

**API:**

```python
# Expand selection by N hops (BFS expansion)
expanded = graph.type_filter('Field').filter({'name': 'EKOFISK'}).expand(hops=2)
# Includes all nodes within 2 hops of selected nodes

# Extract selected nodes into independent subgraph
subgraph = (
    graph.type_filter('Field')
    .filter({'region': 'North Sea'})
    .expand(hops=2)
    .to_subgraph()
)
# Returns new KnowledgeGraph with only selected nodes and their connecting edges

# Preview subgraph stats before extraction
stats = graph.type_filter('Central').expand(hops=3).subgraph_stats()
# Returns: {
#     'node_count': 150,
#     'edge_count': 320,
#     'node_types': {'Field': 10, 'Well': 50, ...},
#     'connection_types': {'HAS_WELL': 100, ...}
# }

# Save subgraph for later use
subgraph.save('north_sea_region.bin')
```

**Implementation Notes:**

- Created `subgraph.rs` module with `expand_selection()` and `extract_subgraph()` functions
- `expand()` uses BFS to include all nodes within N hops (undirected)
- `to_subgraph()` creates independent graph copy with only selected nodes and connecting edges
- `subgraph_stats()` previews extraction without creating the subgraph
- Preserves node properties, edge properties, and schema definition
- Pre-allocates HashMaps with capacity for performance
- Supports `explain()` to show EXPAND operation in query plan

**Tests:** `pytest/test_subgraph_extraction.py`

### Feature 12: Export to Visualization Formats - COMPLETE

**API:**

```python
# Export to GraphML (Gephi, yEd, Cytoscape)
graph.export('output.graphml')
graph.export('output.graphml', format='graphml')

# Export to D3.js JSON format
graph.export('visualization.json', format='d3')

# Export to GEXF (Gephi native format)
graph.export('network.gexf', format='gexf')

# Export to CSV (creates _nodes.csv and _edges.csv files)
graph.export('data.csv', format='csv')

# Export selection only
subgraph = graph.type_filter('Field').expand(hops=2)
subgraph.export('fields.json', format='d3')

# Export to string (for web APIs)
json_str = graph.export_string('d3')
graphml_str = graph.export_string('graphml', selection_only=False)
```

**Supported Formats:**

- **graphml** - XML format supported by Gephi, yEd, Cytoscape
- **gexf** - Gephi native format with attribute support
- **d3** / **json** - D3.js force-directed graph compatible JSON
- **csv** - Simple CSV format (nodes and edges in separate files)

**Implementation Notes:**

- Created `export.rs` module with format-specific functions
- `to_graphml()` - Full GraphML 1.0 specification with node/edge attributes
- `to_d3_json()` - D3.js compatible with nodes[] and links[] arrays
- `to_gexf()` - GEXF 1.2 format with attribute definitions
- `to_csv()` - Simple CSV export for spreadsheet import
- All formats filter out Schema metadata nodes (only exports Regular data nodes)
- Format is auto-detected from file extension if not specified
- `export_string()` method for in-memory export (web APIs)
- Pre-allocates string buffers for performance

**Tests:** `pytest/test_export.py`

### Feature 13: Index Management - COMPLETE

**API:**

```python
# Create an index on a property for faster equality lookups
result = graph.create_index('Prospect', 'geoprovince')
# Returns: {'node_type': 'Prospect', 'property': 'geoprovince', 'created': True, 'unique_values': 15}

# Check if an index exists
has_idx = graph.has_index('Prospect', 'geoprovince')  # True/False

# List all indexes
indexes = graph.list_indexes()
# Returns: [{'node_type': 'Prospect', 'property': 'geoprovince'}, ...]

# Get index statistics
stats = graph.index_stats('Prospect', 'geoprovince')
# Returns: {'node_type': 'Prospect', 'property': 'geoprovince',
#           'unique_values': 15, 'total_entries': 6775, 'avg_entries_per_value': 451.67}

# Drop an index
dropped = graph.drop_index('Prospect', 'geoprovince')  # True if existed

# Rebuild all indexes (after bulk data changes)
count = graph.rebuild_indexes()  # Returns number of indexes rebuilt

# Filtering automatically uses indexes for equality conditions
graph.create_index('Item', 'category')
result = graph.type_filter('Item').filter({'category': 'A'})  # Uses index for O(1) lookup
```

**Performance Benefits:**

- Equality lookups (`filter({'field': 'value'})`) go from O(n) to O(1)
- Index creation is O(n) one-time cost per index
- Memory overhead is O(n) per index (stores node references grouped by value)
- Indexes are transparently used during filtering - no API changes needed
- Multi-condition filters use index for first equality condition, then filter remaining

**Implementation Notes:**

- Added `property_indices: HashMap<IndexKey, HashMap<Value, Vec<NodeIndex>>>` to `DirGraph`
- Index methods added to `DirGraph`: `create_index()`, `drop_index()`, `has_index()`, `list_indexes()`, `lookup_by_index()`, `get_index_stats()`
- `IndexStats` struct holds unique values, total entries, and average entries per value
- `filter_nodes_by_conditions()` in `filtering_methods.rs` checks for indexes on equality conditions
- Indexes are persisted via serde when graph is saved/loaded
- Python methods exposed: `create_index()`, `drop_index()`, `has_index()`, `list_indexes()`, `index_stats()`, `rebuild_indexes()`

**Tests:** `pytest/test_index_management.py`
