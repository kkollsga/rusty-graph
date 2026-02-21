// src/graph/graph_algorithms.rs
//! Graph algorithms module providing path finding and connectivity analysis.

use crate::datatypes::values::Value;
use crate::graph::schema::DirGraph;
use crate::graph::value_operations;
use petgraph::algo::kosaraju_scc;
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ============================================================================
// Path Filtering Helpers
// ============================================================================

/// Get undirected neighbors filtered by edge connection type.
/// When connection_types is None, returns all neighbors (equivalent to neighbors_undirected).
fn filtered_neighbors_undirected(
    graph: &DirGraph,
    node: NodeIndex,
    connection_types: Option<&[String]>,
) -> Vec<NodeIndex> {
    use petgraph::Direction;
    match connection_types {
        None => graph.graph.neighbors_undirected(node).collect(),
        Some(types) => {
            let mut neighbors = Vec::new();
            for edge in graph.graph.edges_directed(node, Direction::Outgoing) {
                if types.iter().any(|t| t == &edge.weight().connection_type) {
                    neighbors.push(edge.target());
                }
            }
            for edge in graph.graph.edges_directed(node, Direction::Incoming) {
                if types.iter().any(|t| t == &edge.weight().connection_type) {
                    neighbors.push(edge.source());
                }
            }
            neighbors
        }
    }
}

/// Get directed (outgoing only) neighbors filtered by edge connection type.
fn filtered_neighbors_outgoing(
    graph: &DirGraph,
    node: NodeIndex,
    connection_types: Option<&[String]>,
) -> Vec<NodeIndex> {
    use petgraph::Direction;
    match connection_types {
        None => graph
            .graph
            .neighbors_directed(node, Direction::Outgoing)
            .collect(),
        Some(types) => graph
            .graph
            .edges_directed(node, Direction::Outgoing)
            .filter(|e| types.iter().any(|t| t == &e.weight().connection_type))
            .map(|e| e.target())
            .collect(),
    }
}

/// Check if a node passes the via_types filter.
/// Source and target should be excluded from this check by the caller.
fn node_passes_via_filter(
    graph: &DirGraph,
    node: NodeIndex,
    via_types: &Option<HashSet<&str>>,
) -> bool {
    match via_types {
        None => true,
        Some(types) => {
            if let Some(node_data) = graph.graph.node_weight(node) {
                types.contains(node_data.node_type.as_str())
            } else {
                false
            }
        }
    }
}

/// Result of a path finding operation
#[derive(Debug, Clone)]
pub struct PathResult {
    /// The path as a sequence of node indices
    pub path: Vec<NodeIndex>,
    /// The total cost/length of the path
    pub cost: usize,
}

/// Information about a node in a path (for Python output)
#[derive(Debug, Clone)]
pub struct PathNodeInfo {
    pub node_type: String,
    pub title: String,
    pub id: Value,
}

/// Find the shortest path between two nodes using undirected BFS.
/// This treats the graph as undirected, finding connections in either direction.
/// Returns None if no path exists.
///
/// # Arguments
/// * `connection_types` - Only traverse edges of these types (None = all)
/// * `via_types` - Only traverse through nodes of these types (None = all)
pub fn shortest_path(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    connection_types: Option<&[String]>,
    via_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Option<PathResult> {
    let via_set: Option<HashSet<&str>> =
        via_types.map(|vt| vt.iter().map(|s| s.as_str()).collect());
    let path = reconstruct_path_bfs(graph, source, target, connection_types, &via_set, deadline)?;
    let cost = path.len().saturating_sub(1);

    Some(PathResult { path, cost })
}

/// Find the shortest path LENGTH between two nodes using undirected BFS.
/// Only returns the hop count, avoiding parent tracking and path reconstruction.
/// Uses level-by-level BFS to avoid per-node distance tracking.
pub fn shortest_path_cost(graph: &DirGraph, source: NodeIndex, target: NodeIndex) -> Option<usize> {
    if source == target {
        return Some(0);
    }

    let node_bound = graph.graph.node_bound();
    let mut visited: Vec<bool> = vec![false; node_bound];

    let target_idx = target.index();

    // Level-by-level BFS using two alternating vectors (avoids VecDeque overhead)
    let mut current_level: Vec<usize> = vec![source.index()];
    let mut next_level: Vec<usize> = Vec::new();
    visited[source.index()] = true;
    let mut depth: usize = 0;

    while !current_level.is_empty() {
        depth += 1;
        next_level.clear();

        for &current_idx in &current_level {
            let current = NodeIndex::new(current_idx);

            for neighbor in graph.graph.neighbors_undirected(current) {
                let neighbor_idx = neighbor.index();
                if !visited[neighbor_idx] {
                    if neighbor_idx == target_idx {
                        return Some(depth);
                    }
                    visited[neighbor_idx] = true;
                    next_level.push(neighbor_idx);
                }
            }
        }

        std::mem::swap(&mut current_level, &mut next_level);
    }

    None
}

/// Batch shortest path cost — reuses visited Vec and adjacency list across multiple pairs.
/// Much faster than calling shortest_path_cost N times for large graphs.
pub fn shortest_path_cost_batch(
    graph: &DirGraph,
    pairs: &[(NodeIndex, NodeIndex)],
) -> Vec<Option<usize>> {
    let node_bound = graph.graph.node_bound();

    // Pre-build undirected adjacency list ONCE for all queries
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();
    let mut node_to_idx = vec![usize::MAX; node_bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in graph.graph.edge_references() {
        let src_i = node_to_idx[edge.source().index()];
        let tgt_i = node_to_idx[edge.target().index()];
        if src_i != usize::MAX && tgt_i != usize::MAX {
            adj[src_i].push(tgt_i);
            adj[tgt_i].push(src_i);
        }
    }

    // Reusable visited array — cleared between queries
    let mut visited: Vec<bool> = vec![false; n];
    let mut current_level: Vec<usize> = Vec::new();
    let mut next_level: Vec<usize> = Vec::new();

    let mut results = Vec::with_capacity(pairs.len());

    for &(source, target) in pairs {
        if source == target {
            results.push(Some(0));
            continue;
        }

        let src_i = node_to_idx[source.index()];
        let tgt_i = node_to_idx[target.index()];
        if src_i == usize::MAX || tgt_i == usize::MAX {
            results.push(None);
            continue;
        }

        // Clear visited (only reset nodes we actually touched)
        // Use a generation counter instead of clearing — much faster
        // But for simplicity, track touched nodes
        let mut touched: Vec<usize> = Vec::new();

        current_level.clear();
        current_level.push(src_i);
        visited[src_i] = true;
        touched.push(src_i);
        let mut depth: usize = 0;
        let mut found = false;

        'bfs: while !current_level.is_empty() {
            depth += 1;
            next_level.clear();

            for &current_idx in &current_level {
                for &neighbor_idx in &adj[current_idx] {
                    if !visited[neighbor_idx] {
                        if neighbor_idx == tgt_i {
                            found = true;
                            break 'bfs;
                        }
                        visited[neighbor_idx] = true;
                        touched.push(neighbor_idx);
                        next_level.push(neighbor_idx);
                    }
                }
            }

            std::mem::swap(&mut current_level, &mut next_level);
        }

        results.push(if found { Some(depth) } else { None });

        // Reset only touched nodes (much faster than clearing entire array)
        for &idx in &touched {
            visited[idx] = false;
        }
    }

    results
}

/// Reconstruct path using BFS with Vec-based tracking for O(1) operations.
/// Uses Vec<bool> for visited and Vec<u32> for parent tracking instead of HashMap/HashSet.
fn reconstruct_path_bfs(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    connection_types: Option<&[String]>,
    via_types: &Option<HashSet<&str>>,
    deadline: Option<Instant>,
) -> Option<Vec<NodeIndex>> {
    use std::collections::VecDeque;

    if source == target {
        return Some(vec![source]);
    }

    let node_bound = graph.graph.node_bound();
    let mut visited: Vec<bool> = vec![false; node_bound];
    let mut parent: Vec<u32> = vec![u32::MAX; node_bound];

    let mut queue = VecDeque::with_capacity(node_bound / 4);

    let source_idx = source.index();
    let target_idx = target.index();

    queue.push_back(source_idx);
    visited[source_idx] = true;

    let mut visit_count = 0u32;

    while let Some(current_idx) = queue.pop_front() {
        // Periodic timeout check (every 1000 nodes)
        visit_count += 1;
        if visit_count.is_multiple_of(1000) {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    return None;
                }
            }
        }

        let current = NodeIndex::new(current_idx);

        // Check all neighbors (both directions for undirected path finding)
        let neighbors = filtered_neighbors_undirected(graph, current, connection_types);
        for neighbor in neighbors {
            let neighbor_idx = neighbor.index();

            if !visited[neighbor_idx] {
                // Apply via_types filter (skip if not target and doesn't match)
                if neighbor_idx != target_idx && !node_passes_via_filter(graph, neighbor, via_types)
                {
                    continue;
                }

                visited[neighbor_idx] = true;
                parent[neighbor_idx] = current_idx as u32;
                queue.push_back(neighbor_idx);

                if neighbor_idx == target_idx {
                    // Found target - reconstruct path
                    let mut path = Vec::with_capacity(16);
                    let mut node_idx = target_idx;

                    while node_idx != source_idx {
                        path.push(NodeIndex::new(node_idx));
                        node_idx = parent[node_idx] as usize;
                    }
                    path.push(source);
                    path.reverse();
                    return Some(path);
                }
            }
        }
    }

    None // No path found
}

/// Directed BFS shortest path — only follows outgoing edges.
/// Used by Cypher shortestPath() which respects edge direction.
///
/// # Arguments
/// * `connection_types` - Only traverse edges of these types (None = all)
/// * `via_types` - Only traverse through nodes of these types (None = all)
pub fn shortest_path_directed(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    connection_types: Option<&[String]>,
    via_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Option<PathResult> {
    use std::collections::VecDeque;

    if source == target {
        return Some(PathResult {
            path: vec![source],
            cost: 0,
        });
    }

    let via_set: Option<HashSet<&str>> =
        via_types.map(|vt| vt.iter().map(|s| s.as_str()).collect());

    let node_bound = graph.graph.node_bound();
    let mut visited: Vec<bool> = vec![false; node_bound];
    let mut parent: Vec<u32> = vec![u32::MAX; node_bound];
    let mut queue = VecDeque::with_capacity(node_bound / 4);

    let source_idx = source.index();
    let target_idx = target.index();

    queue.push_back(source_idx);
    visited[source_idx] = true;

    let mut visit_count = 0u32;

    while let Some(current_idx) = queue.pop_front() {
        // Periodic timeout check
        visit_count += 1;
        if visit_count.is_multiple_of(1000) {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    return None;
                }
            }
        }

        let current = NodeIndex::new(current_idx);

        // Only follow outgoing edges
        let neighbors = filtered_neighbors_outgoing(graph, current, connection_types);
        for neighbor in neighbors {
            let neighbor_idx = neighbor.index();

            if !visited[neighbor_idx] {
                // Apply via_types filter (skip if not target and doesn't match)
                if neighbor_idx != target_idx && !node_passes_via_filter(graph, neighbor, &via_set)
                {
                    continue;
                }

                visited[neighbor_idx] = true;
                parent[neighbor_idx] = current_idx as u32;
                queue.push_back(neighbor_idx);

                if neighbor_idx == target_idx {
                    let mut path = Vec::with_capacity(16);
                    let mut node_idx = target_idx;

                    while node_idx != source_idx {
                        path.push(NodeIndex::new(node_idx));
                        node_idx = parent[node_idx] as usize;
                    }
                    path.push(source);
                    path.reverse();

                    let cost = path.len().saturating_sub(1);
                    return Some(PathResult { path, cost });
                }
            }
        }
    }

    None
}

/// Find all paths between two nodes up to a maximum number of hops.
/// Warning: This can be expensive for graphs with many paths!
///
/// # Arguments
/// * `max_results` - Stop after finding this many paths (prevents OOM on dense graphs)
/// * `connection_types` - Only traverse edges of these types (None = all)
/// * `via_types` - Only traverse through nodes of these types (None = all)
#[allow(clippy::too_many_arguments)]
pub fn all_paths(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    max_hops: usize,
    max_results: Option<usize>,
    connection_types: Option<&[String]>,
    via_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Vec<Vec<NodeIndex>> {
    let via_set: Option<HashSet<&str>> =
        via_types.map(|vt| vt.iter().map(|s| s.as_str()).collect());
    let mut results = Vec::new();
    let mut current_path = vec![source];
    let mut visited = HashSet::new();
    visited.insert(source);

    find_all_paths_recursive(
        graph,
        source,
        target,
        max_hops,
        &mut current_path,
        &mut visited,
        &mut results,
        max_results,
        connection_types,
        &via_set,
        deadline,
    );

    results
}

#[allow(clippy::only_used_in_recursion, clippy::too_many_arguments)]
fn find_all_paths_recursive(
    graph: &DirGraph,
    current: NodeIndex,
    target: NodeIndex,
    remaining_hops: usize,
    current_path: &mut Vec<NodeIndex>,
    visited: &mut HashSet<NodeIndex>,
    results: &mut Vec<Vec<NodeIndex>>,
    max_results: Option<usize>,
    connection_types: Option<&[String]>,
    via_types: &Option<HashSet<&str>>,
    deadline: Option<Instant>,
) {
    // Early termination when result limit is hit
    if let Some(max) = max_results {
        if results.len() >= max {
            return;
        }
    }

    // Timeout check at each recursive entry
    if let Some(dl) = deadline {
        if Instant::now() > dl {
            return;
        }
    }

    if current == target {
        results.push(current_path.clone());
        return;
    }

    if remaining_hops == 0 {
        return;
    }

    // Explore all neighbors (undirected), filtered by connection type
    let neighbors = filtered_neighbors_undirected(graph, current, connection_types);
    for neighbor in neighbors {
        // Check limit before exploring deeper
        if let Some(max) = max_results {
            if results.len() >= max {
                return;
            }
        }

        if !visited.contains(&neighbor) {
            // Apply via_types filter (skip if not target and doesn't match)
            if neighbor != target && !node_passes_via_filter(graph, neighbor, via_types) {
                continue;
            }

            visited.insert(neighbor);
            current_path.push(neighbor);

            find_all_paths_recursive(
                graph,
                neighbor,
                target,
                remaining_hops - 1,
                current_path,
                visited,
                results,
                max_results,
                connection_types,
                via_types,
                deadline,
            );

            current_path.pop();
            visited.remove(&neighbor);
        }
    }
}

/// Find all strongly connected components in the graph.
/// Returns a vector of components, each component is a vector of node indices.
pub fn connected_components(graph: &DirGraph) -> Vec<Vec<NodeIndex>> {
    kosaraju_scc(&graph.graph)
}

/// Find weakly connected components (treating graph as undirected).
/// This is often more useful for knowledge graphs.
/// Uses Union-Find (disjoint set) for optimal performance — O(E * α(V)) ≈ O(E).
pub fn weakly_connected_components(graph: &DirGraph) -> Vec<Vec<NodeIndex>> {
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return Vec::new();
    }

    // Use node_bound() not node_count() — StableDiGraph indices can have gaps
    let bound = graph.graph.node_bound();

    // Build compact index mapping: graph NodeIndex → contiguous 0..n
    let mut node_to_idx = vec![0usize; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    // Union-Find with path compression + union by rank
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u8> = vec![0; n];

    // Find with path compression (iterative)
    #[inline]
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        x
    }

    // Union by rank
    #[inline]
    fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return;
        }
        if rank[ra] < rank[rb] {
            parent[ra] = rb;
        } else if rank[ra] > rank[rb] {
            parent[rb] = ra;
        } else {
            parent[rb] = ra;
            rank[ra] += 1;
        }
    }

    // Process all edges — single pass, no adjacency list needed
    for edge in graph.graph.edge_references() {
        let src_i = node_to_idx[edge.source().index()];
        let tgt_i = node_to_idx[edge.target().index()];
        union(&mut parent, &mut rank, src_i, tgt_i);
    }

    // Collect components by root
    let mut component_map: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        let root = find(&mut parent, i);
        component_map.entry(root).or_default().push(node);
    }

    let mut components: Vec<Vec<NodeIndex>> = component_map.into_values().collect();

    // Sort components by size (largest first)
    components.sort_by_key(|b| std::cmp::Reverse(b.len()));

    components
}

/// Get node info for building Python-friendly path output
pub fn get_node_info(graph: &DirGraph, node_idx: NodeIndex) -> Option<PathNodeInfo> {
    let node = graph.get_node(node_idx)?;
    let title_str = match &node.title {
        Value::String(s) => s.clone(),
        _ => format!("{:?}", node.title),
    };
    Some(PathNodeInfo {
        node_type: node.node_type.clone(),
        title: title_str,
        id: node.id.clone(),
    })
}

/// Get information about what connection types link nodes in a path
pub fn get_path_connections(graph: &DirGraph, path: &[NodeIndex]) -> Vec<Option<String>> {
    // Pre-allocate with exact size (one connection per edge = path.len() - 1)
    let mut connections = Vec::with_capacity(path.len().saturating_sub(1));

    for window in path.windows(2) {
        let from = window[0];
        let to = window[1];

        // Find edge between these nodes (either direction)
        let conn_type = graph
            .graph
            .edges(from)
            .find(|e| e.target() == to)
            .map(|e| e.weight().connection_type.clone())
            .or_else(|| {
                graph
                    .graph
                    .edges(to)
                    .find(|e| e.target() == from)
                    .map(|e| e.weight().connection_type.clone())
            });

        connections.push(conn_type);
    }

    connections
}

/// Check if two nodes are connected (directly or indirectly)
pub fn are_connected(graph: &DirGraph, source: NodeIndex, target: NodeIndex) -> bool {
    shortest_path(graph, source, target, None, None, None).is_some()
}

/// Calculate the degree (number of connections) for a node
pub fn node_degree(graph: &DirGraph, node: NodeIndex) -> usize {
    graph.graph.edges(node).count()
        + graph
            .graph
            .neighbors_directed(node, petgraph::Direction::Incoming)
            .count()
}

// ============================================================================
// Centrality Algorithms
// ============================================================================

/// Result of centrality calculation
#[derive(Debug, Clone)]
pub struct CentralityResult {
    pub node_idx: NodeIndex,
    pub score: f64,
}

/// Calculate betweenness centrality for all nodes in the graph.
///
/// Betweenness centrality measures how often a node lies on the shortest path
/// between other pairs of nodes. Higher values indicate nodes that are more
/// important as "bridges" in the network.
///
/// Uses Brandes' algorithm for efficiency: O(V * E) for unweighted graphs.
/// Optimized to use Vec instead of HashMap for O(1) direct indexing.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `normalized` - If true, normalize scores by 2/((n-1)*(n-2)) for directed graphs
/// * `sample_size` - Optional number of source nodes to sample (for large graphs)
pub fn betweenness_centrality(
    graph: &DirGraph,
    normalized: bool,
    sample_size: Option<usize>,
    connection_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Vec<CentralityResult> {
    use std::collections::VecDeque;

    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n <= 2 {
        return nodes
            .iter()
            .map(|&idx| CentralityResult {
                node_idx: idx,
                score: 0.0,
            })
            .collect();
    }

    // Use Vec-based index mapping for O(1) lookup (vs HashMap)
    let bound = graph.graph.node_bound();
    let mut node_to_idx = vec![0usize; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    // Pre-build undirected adjacency list for BFS.
    // Betweenness treats edges as undirected so that nodes bridging
    // communities are detected regardless of edge direction.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in graph.graph.edge_references() {
        if let Some(types) = &connection_types {
            if !types.iter().any(|t| t == &edge.weight().connection_type) {
                continue;
            }
        }
        let src_i = node_to_idx[edge.source().index()];
        let tgt_i = node_to_idx[edge.target().index()];
        adj[src_i].push(tgt_i);
        adj[tgt_i].push(src_i);
    }

    // Initialize betweenness scores using Vec for O(1) access
    let mut betweenness: Vec<f64> = vec![0.0; n];

    // Determine which nodes to use as sources
    // Use stride-based sampling to ensure even coverage across the graph,
    // avoiding bias from sequential selection (e.g. first k nodes being
    // Module/Class containers with no outgoing edges of the filtered type).
    let source_indices: Vec<usize> = if let Some(k) = sample_size {
        let k = k.min(n);
        if k == n {
            (0..n).collect()
        } else {
            let step = n as f64 / k as f64;
            (0..k).map(|i| (i as f64 * step) as usize).collect()
        }
    } else {
        (0..n).collect()
    };

    // Pre-allocate data structures ONCE outside the loop
    let mut stack: Vec<usize> = Vec::with_capacity(n);
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut sigma: Vec<f64> = vec![0.0; n];
    let mut dist: Vec<i64> = vec![-1; n];
    let mut delta: Vec<f64> = vec![0.0; n];
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);

    // Brandes' algorithm - process each source
    for (source_counter, &s_idx) in source_indices.iter().enumerate() {
        // Periodic timeout check (every 10 source nodes)
        if source_counter.is_multiple_of(10) {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    break;
                }
            }
        }

        // Reset data structures (much faster than re-allocating)
        stack.clear();
        queue.clear();
        for i in 0..n {
            pred[i].clear();
            sigma[i] = 0.0;
            dist[i] = -1;
            delta[i] = 0.0;
        }

        // Initialize source
        sigma[s_idx] = 1.0;
        dist[s_idx] = 0;
        queue.push_back(s_idx);

        // BFS phase using pre-built adjacency list (no graph traversal)
        while let Some(v_idx) = queue.pop_front() {
            stack.push(v_idx);
            let v_dist = dist[v_idx];

            for &w_idx in &adj[v_idx] {
                // First visit?
                if dist[w_idx] < 0 {
                    dist[w_idx] = v_dist + 1;
                    queue.push_back(w_idx);
                }
                // Shortest path to w via v?
                if dist[w_idx] == v_dist + 1 {
                    sigma[w_idx] += sigma[v_idx];
                    pred[w_idx].push(v_idx);
                }
            }
        }

        // Accumulation phase - back propagation
        while let Some(w_idx) = stack.pop() {
            for &v_idx in &pred[w_idx] {
                let contribution = (sigma[v_idx] / sigma[w_idx]) * (1.0 + delta[w_idx]);
                delta[v_idx] += contribution;
            }
            if w_idx != s_idx {
                betweenness[w_idx] += delta[w_idx];
            }
        }
    }

    // Undirected BFS counts each (s,t) pair twice, so halve raw scores.
    for score in betweenness.iter_mut() {
        *score /= 2.0;
    }

    // Normalize if requested
    // For undirected graphs: 2 / ((n-1)*(n-2))
    if normalized && n > 2 {
        let scale = 2.0 / ((n - 1) as f64 * (n - 2) as f64);
        for score in betweenness.iter_mut() {
            *score *= scale;
        }
    }

    // If we sampled, scale up the scores
    if let Some(k) = sample_size {
        if k < n {
            let scale = n as f64 / k as f64;
            for score in betweenness.iter_mut() {
                *score *= scale;
            }
        }
    }

    // Convert to sorted results
    let mut results: Vec<CentralityResult> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node_idx)| CentralityResult {
            node_idx,
            score: betweenness[i],
        })
        .collect();

    // Sort by score descending
    results.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Calculate PageRank centrality for all nodes in the graph.
///
/// PageRank measures the importance of nodes based on the structure of incoming links.
/// Originally developed by Google for ranking web pages.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `damping_factor` - Probability of following a link (typically 0.85)
/// * `max_iterations` - Maximum number of iterations (default: 100)
/// * `tolerance` - Convergence threshold (default: 1e-6)
pub fn pagerank(
    graph: &DirGraph,
    damping_factor: f64,
    max_iterations: usize,
    tolerance: f64,
    connection_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Vec<CentralityResult> {
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return Vec::new();
    }

    // Use Vec-based index mapping for O(1) lookup (vs HashMap)
    let bound = graph.graph.node_bound();
    let mut node_to_idx = vec![0usize; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    // Pre-build adjacency list: for each node i, store list of target indices
    // This avoids graph traversal in the hot loop
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut out_degrees: Vec<usize> = vec![0; n];
    for edge in graph.graph.edge_references() {
        if let Some(types) = &connection_types {
            if !types.iter().any(|t| t == &edge.weight().connection_type) {
                continue;
            }
        }
        let src_i = node_to_idx[edge.source().index()];
        let tgt_i = node_to_idx[edge.target().index()];
        adj[src_i].push(tgt_i);
        out_degrees[src_i] += 1;
    }

    // Initialize PageRank scores (uniform distribution)
    let mut pr: Vec<f64> = vec![1.0 / n as f64; n];
    let mut new_pr: Vec<f64> = vec![0.0; n];

    // Precompute inverse out-degree (multiply instead of divide in hot loop)
    let inv_out_degrees: Vec<f64> = out_degrees
        .iter()
        .map(|&d| {
            if d > 0 {
                damping_factor / d as f64
            } else {
                0.0
            }
        })
        .collect();

    // Identify dangling nodes (no outgoing links) — store as bitmask for fast sum
    let is_dangling: Vec<bool> = out_degrees.iter().map(|&d| d == 0).collect();

    let teleport = (1.0 - damping_factor) / n as f64;
    let inv_n = 1.0 / n as f64;

    // Iterative computation
    for _iteration in 0..max_iterations {
        // Timeout check each iteration
        if let Some(dl) = deadline {
            if Instant::now() > dl {
                break;
            }
        }

        // Calculate dangling node contribution (vectorized over bitmask)
        let mut dangling_sum: f64 = 0.0;
        for i in 0..n {
            if is_dangling[i] {
                dangling_sum += pr[i];
            }
        }
        let base_score = teleport + damping_factor * dangling_sum * inv_n;

        // Reset new_pr to base score (teleport + dangling contribution)
        for score in new_pr.iter_mut() {
            *score = base_score;
        }

        // Add contributions from outgoing links
        // Using precomputed inv_out_degrees avoids division in hot loop
        for i in 0..n {
            let contrib = inv_out_degrees[i] * pr[i];
            if contrib > 0.0 {
                for &j in &adj[i] {
                    new_pr[j] += contrib;
                }
            }
        }

        // Check for convergence (L1 norm)
        let mut diff: f64 = 0.0;
        for i in 0..n {
            diff += (pr[i] - new_pr[i]).abs();
        }

        std::mem::swap(&mut pr, &mut new_pr);

        if diff < tolerance {
            break;
        }
    }

    // Convert to results and sort by score
    let mut results: Vec<CentralityResult> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node_idx)| CentralityResult {
            node_idx,
            score: pr[i],
        })
        .collect();

    results.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Calculate degree centrality for all nodes.
///
/// Simply counts the number of connections each node has.
/// Optionally normalized by (n-1) to get values between 0 and 1.
pub fn degree_centrality(
    graph: &DirGraph,
    normalized: bool,
    connection_types: Option<&[String]>,
    _deadline: Option<Instant>,
) -> Vec<CentralityResult> {
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return Vec::new();
    }

    let scale = if normalized && n > 1 {
        1.0 / (n - 1) as f64
    } else {
        1.0
    };

    // Compute all degrees in a single pass over edges instead of per-node traversal
    let bound = graph.graph.node_bound();
    let mut degrees = vec![0usize; bound];
    for edge in graph.graph.edge_references() {
        if let Some(types) = &connection_types {
            if !types.iter().any(|t| t == &edge.weight().connection_type) {
                continue;
            }
        }
        degrees[edge.source().index()] += 1; // out-degree
        degrees[edge.target().index()] += 1; // in-degree
    }

    let mut results: Vec<CentralityResult> = nodes
        .iter()
        .map(|&node_idx| CentralityResult {
            node_idx,
            score: degrees[node_idx.index()] as f64 * scale,
        })
        .collect();

    results.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Calculate closeness centrality for all nodes.
///
/// Closeness centrality measures how close a node is to all other nodes.
/// Defined as the reciprocal of the sum of shortest path distances.
///
/// Note: For disconnected graphs, only reachable nodes are considered.
/// Optimized to use Vec instead of HashMap for O(1) direct indexing.
pub fn closeness_centrality(
    graph: &DirGraph,
    normalized: bool,
    connection_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Vec<CentralityResult> {
    use std::collections::VecDeque;

    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return Vec::new();
    }

    // Use Vec-based index mapping for O(1) lookup (vs HashMap)
    let bound = graph.graph.node_bound();
    let mut node_to_idx = vec![0usize; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    // Pre-build incoming adjacency list: for closeness centrality on directed graphs,
    // we BFS via incoming edges (convention: d(v, u) = how easy for v to reach u)
    let mut adj_incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in graph.graph.edge_references() {
        if let Some(types) = &connection_types {
            if !types.iter().any(|t| t == &edge.weight().connection_type) {
                continue;
            }
        }
        let src_i = node_to_idx[edge.source().index()];
        let tgt_i = node_to_idx[edge.target().index()];
        // For incoming BFS from node u: follow edges pointing INTO u
        // edge: src -> tgt, so tgt has incoming edge from src
        adj_incoming[tgt_i].push(src_i);
    }

    let mut results = Vec::with_capacity(n);

    // Pre-allocate data structures ONCE outside the loop
    // -1 means not visited, >= 0 is the distance
    let mut dist: Vec<i64> = vec![-1; n];
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);

    for (s_idx, &source) in nodes.iter().enumerate() {
        // Periodic timeout check (every 10 source nodes)
        if s_idx.is_multiple_of(10) {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    break;
                }
            }
        }

        // Reset data structures (much faster than re-allocating)
        queue.clear();
        for d in dist.iter_mut() {
            *d = -1;
        }

        // BFS via incoming edges to find distances
        queue.push_back(s_idx);
        dist[s_idx] = 0;

        while let Some(current_idx) = queue.pop_front() {
            let current_dist = dist[current_idx];

            for &neighbor_idx in &adj_incoming[current_idx] {
                if dist[neighbor_idx] < 0 {
                    dist[neighbor_idx] = current_dist + 1;
                    queue.push_back(neighbor_idx);
                }
            }
        }

        // Calculate closeness from distances
        let mut total_distance: i64 = 0;
        let mut reachable: usize = 0;
        for &d in &dist {
            if d >= 0 {
                total_distance += d;
                reachable += 1;
            }
        }

        if reachable > 1 && total_distance > 0 {
            let closeness = (reachable - 1) as f64 / total_distance as f64;

            // Normalize by the fraction of reachable nodes
            let score = if normalized {
                closeness * (reachable - 1) as f64 / (n - 1) as f64
            } else {
                closeness
            };

            results.push(CentralityResult {
                node_idx: source,
                score,
            });
        } else {
            results.push(CentralityResult {
                node_idx: source,
                score: 0.0,
            });
        }
    }

    results.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

// ============================================================================
// Community Detection
// ============================================================================

#[derive(Debug, Clone)]
pub struct CommunityAssignment {
    pub node_idx: NodeIndex,
    pub community_id: usize,
}

#[derive(Debug)]
pub struct CommunityResult {
    pub assignments: Vec<CommunityAssignment>,
    pub num_communities: usize,
    pub modularity: f64,
}

/// Louvain modularity optimization for community detection.
///
/// Each node starts in its own community. Iteratively moves nodes to the
/// neighboring community that yields the largest modularity gain, until
/// no improvement is found.
/// Optimized with pre-built adjacency list and Vec-based community weight tracking.
pub fn louvain_communities(
    graph: &DirGraph,
    weight_property: Option<&str>,
    resolution: f64,
    connection_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> CommunityResult {
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return CommunityResult {
            assignments: Vec::new(),
            num_communities: 0,
            modularity: 0.0,
        };
    }

    // Build compact index mapping
    let bound = graph.graph.node_bound();
    let mut node_to_idx = vec![0usize; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    // Pre-build undirected weighted adjacency list
    // adj[i] = Vec<(neighbor_compact_idx, weight)>
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut total_weight = 0.0f64;
    for edge in graph.graph.edge_references() {
        if let Some(types) = &connection_types {
            if !types.iter().any(|t| t == &edge.weight().connection_type) {
                continue;
            }
        }
        let w = edge_weight(graph, edge.id(), weight_property);
        let src_i = node_to_idx[edge.source().index()];
        let tgt_i = node_to_idx[edge.target().index()];
        adj[src_i].push((tgt_i, w));
        adj[tgt_i].push((src_i, w));
        total_weight += w;
    }

    if total_weight == 0.0 {
        // No edges — each node is its own community
        let assignments: Vec<CommunityAssignment> = nodes
            .iter()
            .enumerate()
            .map(|(i, &idx)| CommunityAssignment {
                node_idx: idx,
                community_id: i,
            })
            .collect();
        return CommunityResult {
            assignments,
            num_communities: n,
            modularity: 0.0,
        };
    }

    // community[i] = community id for compact node i
    let mut community: Vec<usize> = (0..n).collect();

    // Precompute node degrees (undirected: sum of all edge weights touching the node)
    let mut degree: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        for &(_, w) in &adj[i] {
            degree[i] += w;
        }
    }

    // Precompute sum of degrees per community (sigma_tot)
    let mut sigma_tot: Vec<f64> = vec![0.0; n];
    sigma_tot[..n].copy_from_slice(&degree[..n]);

    let m = total_weight;
    let two_m = 2.0 * m;

    // Vec-based community weight tracking (reused across iterations)
    let mut comm_weight: Vec<f64> = vec![0.0; n];
    let mut touched_comms: Vec<usize> = Vec::with_capacity(64);

    // Iterative optimization
    let max_iterations = 100;
    for _ in 0..max_iterations {
        // Timeout check each iteration
        if let Some(dl) = deadline {
            if Instant::now() > dl {
                break;
            }
        }

        let mut improved = false;

        for i in 0..n {
            let current_community = community[i];
            let k_i = degree[i];

            // Compute weight from node i to each neighboring community
            touched_comms.clear();
            for &(neighbor, w) in &adj[i] {
                let c = community[neighbor];
                if comm_weight[c] == 0.0 {
                    touched_comms.push(c);
                }
                comm_weight[c] += w;
            }

            // Weight from i into its own community
            let k_i_in_current = comm_weight[current_community];

            // Find best community to move to
            let mut best_community = current_community;
            let mut best_delta = 0.0f64;

            for &cand_community in &touched_comms {
                if cand_community == current_community {
                    continue;
                }

                let k_i_in_cand = comm_weight[cand_community];
                let sigma_cand = sigma_tot[cand_community];
                let sigma_curr = sigma_tot[current_community] - k_i;

                let gain_add = k_i_in_cand / m - resolution * sigma_cand * k_i / (two_m * two_m);
                let loss_remove =
                    k_i_in_current / m - resolution * sigma_curr * k_i / (two_m * two_m);
                let delta = gain_add - loss_remove;

                if delta > best_delta {
                    best_delta = delta;
                    best_community = cand_community;
                }
            }

            // Reset community weights (only touched entries)
            for &c in &touched_comms {
                comm_weight[c] = 0.0;
            }

            if best_community != current_community {
                sigma_tot[current_community] -= k_i;
                sigma_tot[best_community] += k_i;
                community[i] = best_community;
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    // Convert compact community to bound-sized array for modularity computation
    let mut community_bound: Vec<usize> = vec![0; bound];
    let mut node_exists: Vec<bool> = vec![false; bound];
    for (i, &node) in nodes.iter().enumerate() {
        community_bound[node.index()] = community[i];
        node_exists[node.index()] = true;
    }

    // Renumber communities to be contiguous 0..n
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    for &c in &community {
        let next_id = id_map.len();
        id_map.entry(c).or_insert(next_id);
    }

    let assignments: Vec<CommunityAssignment> = nodes
        .iter()
        .enumerate()
        .map(|(i, &idx)| CommunityAssignment {
            node_idx: idx,
            community_id: *id_map.get(&community[i]).unwrap(),
        })
        .collect();

    let num_communities = id_map.len();
    let modularity = compute_modularity(
        graph,
        &community_bound,
        &node_exists,
        total_weight,
        weight_property,
    );

    CommunityResult {
        assignments,
        num_communities,
        modularity,
    }
}

/// Label propagation for community detection.
///
/// Each node adopts the most frequent label among its neighbors.
/// Converges when no node changes its label.
/// Optimized with pre-built adjacency list and Vec-based label counting.
pub fn label_propagation(
    graph: &DirGraph,
    max_iterations: usize,
    connection_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> CommunityResult {
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return CommunityResult {
            assignments: Vec::new(),
            num_communities: 0,
            modularity: 0.0,
        };
    }

    // Build compact index mapping
    let bound = graph.graph.node_bound();
    let mut node_to_idx = vec![0usize; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    // Pre-build undirected adjacency list (both directions)
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in graph.graph.edge_references() {
        if let Some(types) = &connection_types {
            if !types.iter().any(|t| t == &edge.weight().connection_type) {
                continue;
            }
        }
        let src_i = node_to_idx[edge.source().index()];
        let tgt_i = node_to_idx[edge.target().index()];
        adj[src_i].push(tgt_i);
        adj[tgt_i].push(src_i);
    }

    // Initialize: each node gets a unique label (0..n)
    let mut labels: Vec<usize> = (0..n).collect();

    // Vec-based label counting (reused across iterations)
    // label_count[label] = count for that label among neighbors
    // We use a sparse approach: track which labels were touched and reset only those
    let mut label_count: Vec<usize> = vec![0; n];
    let mut touched_labels: Vec<usize> = Vec::with_capacity(64);

    for _ in 0..max_iterations {
        // Timeout check each iteration
        if let Some(dl) = deadline {
            if Instant::now() > dl {
                break;
            }
        }

        let mut changed = false;

        for i in 0..n {
            let neighbors = &adj[i];
            if neighbors.is_empty() {
                continue; // isolated node keeps its label
            }

            // Count neighbor labels using Vec (O(1) per access)
            touched_labels.clear();
            for &neighbor in neighbors {
                let lbl = labels[neighbor];
                if label_count[lbl] == 0 {
                    touched_labels.push(lbl);
                }
                label_count[lbl] += 1;
            }

            // Find most frequent label
            let mut best_label = labels[i];
            let mut best_count = 0;
            for &lbl in &touched_labels {
                if label_count[lbl] > best_count {
                    best_count = label_count[lbl];
                    best_label = lbl;
                }
            }

            // Reset counts for next node (only touched entries)
            for &lbl in &touched_labels {
                label_count[lbl] = 0;
            }

            if best_label != labels[i] {
                labels[i] = best_label;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Convert compact labels back to bound-sized array for modularity computation
    let mut labels_bound: Vec<usize> = vec![0; bound];
    let mut node_exists: Vec<bool> = vec![false; bound];
    for (i, &node) in nodes.iter().enumerate() {
        labels_bound[node.index()] = labels[i];
        node_exists[node.index()] = true;
    }

    // Renumber labels to be contiguous
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    for &lbl in &labels {
        let next_id = id_map.len();
        id_map.entry(lbl).or_insert(next_id);
    }

    let assignments: Vec<CommunityAssignment> = nodes
        .iter()
        .enumerate()
        .map(|(i, &idx)| CommunityAssignment {
            node_idx: idx,
            community_id: *id_map.get(&labels[i]).unwrap(),
        })
        .collect();

    let total_weight = graph.graph.edge_count() as f64;
    let num_communities = id_map.len();
    let modularity = compute_modularity(graph, &labels_bound, &node_exists, total_weight, None);

    CommunityResult {
        assignments,
        num_communities,
        modularity,
    }
}

/// Get edge weight from a property, or 1.0 if not specified.
fn edge_weight(
    graph: &DirGraph,
    edge_id: petgraph::graph::EdgeIndex,
    weight_property: Option<&str>,
) -> f64 {
    if let Some(prop) = weight_property {
        if let Some(edge_data) = graph.graph.edge_weight(edge_id) {
            if let Some(val) = edge_data.properties.get(prop) {
                return value_operations::value_to_f64(val).unwrap_or(1.0);
            }
        }
    }
    1.0
}

/// Sum of edge weights for all nodes in a community.
/// Compute Newman modularity: Q = (1/2m) * sum [ A_ij - k_i*k_j/(2m) ] * delta(c_i, c_j)
fn compute_modularity(
    graph: &DirGraph,
    community: &[usize],
    node_exists: &[bool],
    total_weight: f64,
    weight_property: Option<&str>,
) -> f64 {
    if total_weight == 0.0 {
        return 0.0;
    }

    let two_m = 2.0 * total_weight;
    let mut q = 0.0f64;

    // Compute degree (sum of edge weights) for each node
    let bound = graph.graph.node_bound();
    let mut degrees: Vec<f64> = vec![0.0; bound];
    for node_idx in graph.graph.node_indices() {
        let i = node_idx.index();
        if !node_exists[i] {
            continue;
        }
        for edge in graph.graph.edges(node_idx) {
            degrees[i] += edge_weight(graph, edge.id(), weight_property);
        }
        for edge in graph
            .graph
            .edges_directed(node_idx, petgraph::Direction::Incoming)
        {
            degrees[i] += edge_weight(graph, edge.id(), weight_property);
        }
    }

    // Sum over all edges
    for edge in graph.graph.edge_references() {
        let u = edge.source().index();
        let v = edge.target().index();
        let w = edge_weight(graph, edge.id(), weight_property);

        if community[u] == community[v] {
            q += w - degrees[u] * degrees[v] / two_m;
        }
    }

    q / two_m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::values::Value;
    use crate::graph::schema::{DirGraph, EdgeData, NodeData};
    use std::collections::HashMap;

    /// Build a linear graph: A -> B -> C -> D -> E
    fn build_chain_graph() -> (DirGraph, Vec<petgraph::graph::NodeIndex>) {
        let mut graph = DirGraph::new();
        let mut indices = Vec::new();
        for i in 0..5 {
            let node = NodeData::new(
                Value::Int64(i),
                Value::String(format!("Node_{}", i)),
                "Chain".to_string(),
                HashMap::new(),
            );
            let idx = graph.graph.add_node(node);
            graph
                .type_indices
                .entry("Chain".to_string())
                .or_default()
                .push(idx);
            indices.push(idx);
        }
        for i in 0..4 {
            let edge = EdgeData::new("NEXT".to_string(), HashMap::new());
            graph.graph.add_edge(indices[i], indices[i + 1], edge);
        }
        (graph, indices)
    }

    /// Build a triangle graph: A -- B -- C -- A
    fn build_triangle_graph() -> (DirGraph, Vec<petgraph::graph::NodeIndex>) {
        let mut graph = DirGraph::new();
        let mut indices = Vec::new();
        for i in 0..3 {
            let node = NodeData::new(
                Value::Int64(i),
                Value::String(format!("N_{}", i)),
                "Node".to_string(),
                HashMap::new(),
            );
            let idx = graph.graph.add_node(node);
            graph
                .type_indices
                .entry("Node".to_string())
                .or_default()
                .push(idx);
            indices.push(idx);
        }
        // A->B, B->C, C->A
        let pairs = [(0, 1), (1, 2), (2, 0)];
        for (from, to) in pairs {
            let edge = EdgeData::new("LINK".to_string(), HashMap::new());
            graph.graph.add_edge(indices[from], indices[to], edge);
        }
        (graph, indices)
    }

    /// Build two disconnected components: {A, B} and {C, D}
    fn build_disconnected_graph() -> (DirGraph, Vec<petgraph::graph::NodeIndex>) {
        let mut graph = DirGraph::new();
        let mut indices = Vec::new();
        for i in 0..4 {
            let node = NodeData::new(
                Value::Int64(i),
                Value::String(format!("N_{}", i)),
                "Node".to_string(),
                HashMap::new(),
            );
            let idx = graph.graph.add_node(node);
            graph
                .type_indices
                .entry("Node".to_string())
                .or_default()
                .push(idx);
            indices.push(idx);
        }
        // Component 1: A-B
        graph.graph.add_edge(
            indices[0],
            indices[1],
            EdgeData::new("LINK".to_string(), HashMap::new()),
        );
        // Component 2: C-D
        graph.graph.add_edge(
            indices[2],
            indices[3],
            EdgeData::new("LINK".to_string(), HashMap::new()),
        );
        (graph, indices)
    }

    // ========================================================================
    // shortest_path
    // ========================================================================

    #[test]
    fn test_shortest_path_adjacent() {
        let (graph, indices) = build_chain_graph();
        let result = shortest_path(&graph, indices[0], indices[1], None, None, None);
        assert!(result.is_some());
        let path = result.unwrap();
        assert_eq!(path.cost, 1);
        assert_eq!(path.path.len(), 2);
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let (graph, indices) = build_chain_graph();
        let result = shortest_path(&graph, indices[0], indices[4], None, None, None);
        assert!(result.is_some());
        let path = result.unwrap();
        assert_eq!(path.cost, 4);
        assert_eq!(path.path.len(), 5);
    }

    #[test]
    fn test_shortest_path_same_node() {
        let (graph, indices) = build_chain_graph();
        let result = shortest_path(&graph, indices[0], indices[0], None, None, None);
        assert!(result.is_some());
        let path = result.unwrap();
        assert_eq!(path.cost, 0);
        assert_eq!(path.path.len(), 1);
    }

    #[test]
    fn test_shortest_path_not_found() {
        let (graph, indices) = build_disconnected_graph();
        let result = shortest_path(&graph, indices[0], indices[2], None, None, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_shortest_path_reverse_direction() {
        // BFS is undirected, so B -> A should find a path even though edge is A -> B
        let (graph, indices) = build_chain_graph();
        let result = shortest_path(&graph, indices[4], indices[0], None, None, None);
        assert!(result.is_some());
        assert_eq!(result.unwrap().cost, 4);
    }

    // ========================================================================
    // all_paths
    // ========================================================================

    #[test]
    fn test_all_paths_basic() {
        let (graph, indices) = build_chain_graph();
        let paths = all_paths(&graph, indices[0], indices[2], 5, None, None, None, None);
        assert!(!paths.is_empty());
        // There should be a path of length 2: A -> B -> C
        assert!(paths.iter().any(|p| p.len() == 3));
    }

    #[test]
    fn test_all_paths_limited_hops() {
        let (graph, indices) = build_chain_graph();
        // With max_hops=1, can only reach adjacent node
        let paths = all_paths(&graph, indices[0], indices[2], 1, None, None, None, None);
        assert!(paths.is_empty()); // Can't reach C in 1 hop
    }

    #[test]
    fn test_all_paths_triangle() {
        let (graph, indices) = build_triangle_graph();
        let paths = all_paths(&graph, indices[0], indices[2], 3, None, None, None, None);
        // Multiple paths possible in a triangle
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_all_paths_max_results() {
        let (graph, indices) = build_triangle_graph();
        // Triangle has multiple paths — limit to 1
        let paths = all_paths(&graph, indices[0], indices[2], 3, Some(1), None, None, None);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_all_paths_max_results_none_unlimited() {
        let (graph, indices) = build_triangle_graph();
        let limited = all_paths(&graph, indices[0], indices[2], 3, Some(1), None, None, None);
        let unlimited = all_paths(&graph, indices[0], indices[2], 3, None, None, None, None);
        assert!(unlimited.len() >= limited.len());
    }

    #[test]
    fn test_shortest_path_connection_type_filter() {
        // Build graph with two edge types: A -NEXT-> B -NEXT-> C and A -SKIP-> C
        let mut graph = DirGraph::new();
        let mut indices = Vec::new();
        for i in 0..3 {
            let node = NodeData::new(
                Value::Int64(i),
                Value::String(format!("Node_{}", i)),
                "Test".to_string(),
                HashMap::new(),
            );
            let idx = graph.graph.add_node(node);
            graph
                .type_indices
                .entry("Test".to_string())
                .or_default()
                .push(idx);
            indices.push(idx);
        }
        graph.graph.add_edge(
            indices[0],
            indices[1],
            EdgeData::new("NEXT".to_string(), HashMap::new()),
        );
        graph.graph.add_edge(
            indices[1],
            indices[2],
            EdgeData::new("NEXT".to_string(), HashMap::new()),
        );
        graph.graph.add_edge(
            indices[0],
            indices[2],
            EdgeData::new("SKIP".to_string(), HashMap::new()),
        );

        // Without filter: shortest path is A->C via SKIP (1 hop)
        let result = shortest_path(&graph, indices[0], indices[2], None, None, None);
        assert_eq!(result.unwrap().cost, 1);

        // With NEXT filter: must go A->B->C (2 hops)
        let next_only = vec!["NEXT".to_string()];
        let result = shortest_path(&graph, indices[0], indices[2], Some(&next_only), None, None);
        assert_eq!(result.unwrap().cost, 2);

        // With SKIP filter: A->C (1 hop)
        let skip_only = vec!["SKIP".to_string()];
        let result = shortest_path(&graph, indices[0], indices[2], Some(&skip_only), None, None);
        assert_eq!(result.unwrap().cost, 1);
    }

    // ========================================================================
    // connected_components / weakly_connected_components
    // ========================================================================

    #[test]
    fn test_weakly_connected_components_connected() {
        let (graph, _) = build_chain_graph();
        let components = weakly_connected_components(&graph);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 5);
    }

    #[test]
    fn test_weakly_connected_components_disconnected() {
        let (graph, _) = build_disconnected_graph();
        let components = weakly_connected_components(&graph);
        assert_eq!(components.len(), 2);
        // Sorted by size descending, both have 2 nodes
        assert_eq!(components[0].len(), 2);
        assert_eq!(components[1].len(), 2);
    }

    #[test]
    fn test_weakly_connected_components_empty() {
        let graph = DirGraph::new();
        let components = weakly_connected_components(&graph);
        assert!(components.is_empty());
    }

    // ========================================================================
    // are_connected
    // ========================================================================

    #[test]
    fn test_are_connected_true() {
        let (graph, indices) = build_chain_graph();
        assert!(are_connected(&graph, indices[0], indices[4]));
    }

    #[test]
    fn test_are_connected_false() {
        let (graph, indices) = build_disconnected_graph();
        assert!(!are_connected(&graph, indices[0], indices[2]));
    }

    // ========================================================================
    // node_degree
    // ========================================================================

    #[test]
    fn test_node_degree() {
        let (graph, indices) = build_chain_graph();
        // First node: 1 outgoing edge
        assert_eq!(node_degree(&graph, indices[0]), 1);
        // Middle node: 1 outgoing + 1 incoming
        assert_eq!(node_degree(&graph, indices[2]), 2);
        // Last node: 1 incoming
        assert_eq!(node_degree(&graph, indices[4]), 1);
    }

    // ========================================================================
    // Centrality algorithms
    // ========================================================================

    #[test]
    fn test_betweenness_centrality_chain() {
        let (graph, indices) = build_chain_graph();
        let results = betweenness_centrality(&graph, false, None, None, None);
        assert_eq!(results.len(), 5);
        // Middle node (index 2) should have highest betweenness in a chain
        let middle_score = results
            .iter()
            .find(|r| r.node_idx == indices[2])
            .unwrap()
            .score;
        let end_score = results
            .iter()
            .find(|r| r.node_idx == indices[0])
            .unwrap()
            .score;
        assert!(middle_score > end_score);
    }

    #[test]
    fn test_betweenness_centrality_with_sampling() {
        let (graph, indices) = build_chain_graph();
        // With sample_size, stride-based sampling should still find the middle node
        let results = betweenness_centrality(&graph, false, Some(3), None, None);
        assert_eq!(results.len(), 5);
        // Middle node should still have a non-zero betweenness score
        let middle_score = results
            .iter()
            .find(|r| r.node_idx == indices[2])
            .unwrap()
            .score;
        assert!(
            middle_score > 0.0,
            "Middle node should have non-zero betweenness with sampling"
        );
    }

    #[test]
    fn test_degree_centrality() {
        let (graph, indices) = build_chain_graph();
        let results = degree_centrality(&graph, false, None, None);
        assert_eq!(results.len(), 5);
        // Middle nodes should have degree 2, end nodes degree 1
        let middle = results.iter().find(|r| r.node_idx == indices[2]).unwrap();
        let end = results.iter().find(|r| r.node_idx == indices[0]).unwrap();
        assert_eq!(middle.score, 2.0);
        assert_eq!(end.score, 1.0);
    }

    #[test]
    fn test_pagerank_basic() {
        let (graph, _) = build_triangle_graph();
        let results = pagerank(&graph, 0.85, 100, 1e-6, None, None);
        assert_eq!(results.len(), 3);
        // All nodes in a symmetric triangle should have roughly equal PageRank
        let scores: Vec<f64> = results.iter().map(|r| r.score).collect();
        let diff = (scores[0] - scores[2]).abs();
        assert!(
            diff < 0.01,
            "Triangle nodes should have similar PageRank: {:?}",
            scores
        );
    }

    #[test]
    fn test_closeness_centrality_chain() {
        let (graph, indices) = build_chain_graph();
        let results = closeness_centrality(&graph, false, None, None);
        assert_eq!(results.len(), 5);
        // Middle node should have highest closeness
        let middle = results
            .iter()
            .find(|r| r.node_idx == indices[2])
            .unwrap()
            .score;
        let end = results
            .iter()
            .find(|r| r.node_idx == indices[0])
            .unwrap()
            .score;
        assert!(middle > end);
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let graph = DirGraph::new();
        let results = pagerank(&graph, 0.85, 100, 1e-6, None, None);
        assert!(results.is_empty());
    }

    // ========================================================================
    // get_node_info / get_path_connections
    // ========================================================================

    #[test]
    fn test_get_node_info() {
        let (graph, indices) = build_chain_graph();
        let info = get_node_info(&graph, indices[0]);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.node_type, "Chain");
        assert_eq!(info.title, "Node_0");
    }

    #[test]
    fn test_get_path_connections() {
        let (graph, indices) = build_chain_graph();
        let path = vec![indices[0], indices[1], indices[2]];
        let connections = get_path_connections(&graph, &path);
        assert_eq!(connections.len(), 2);
        assert_eq!(connections[0], Some("NEXT".to_string()));
        assert_eq!(connections[1], Some("NEXT".to_string()));
    }
}
