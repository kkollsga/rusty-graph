// src/graph/graph_algorithms.rs
//! Graph algorithms module providing path finding and connectivity analysis.

use std::collections::HashMap;
use petgraph::graph::NodeIndex;
use petgraph::algo::kosaraju_scc;
use petgraph::visit::EdgeRef;
use crate::graph::schema::{DirGraph, NodeData};
use crate::datatypes::values::Value;

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
pub fn shortest_path(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
) -> Option<PathResult> {
    // Use BFS for undirected path finding (more appropriate for knowledge graphs)
    let path = reconstruct_path_bfs(graph, source, target)?;
    let cost = path.len().saturating_sub(1); // Cost is number of edges

    Some(PathResult { path, cost })
}

/// Reconstruct path using BFS (more reliable than predecessor map)
fn reconstruct_path_bfs(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
) -> Option<Vec<NodeIndex>> {
    use std::collections::{VecDeque, HashSet};

    if source == target {
        return Some(vec![source]);
    }

    // Pre-allocate with reasonable capacity based on graph size
    let node_count = graph.graph.node_count();
    let estimated_visited = node_count / 4; // Assume we visit ~25% of nodes on average
    let mut visited = HashSet::with_capacity(estimated_visited);
    let mut queue = VecDeque::with_capacity(estimated_visited);
    let mut parent_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(estimated_visited);

    queue.push_back(source);
    visited.insert(source);

    while let Some(current) = queue.pop_front() {
        // Check all neighbors (both directions for undirected path finding)
        for neighbor in graph.graph.neighbors_undirected(current) {
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                parent_map.insert(neighbor, current);
                queue.push_back(neighbor);

                if neighbor == target {
                    // Found target - reconstruct path
                    let mut path = vec![target];
                    let mut node = target;
                    while let Some(&parent) = parent_map.get(&node) {
                        path.push(parent);
                        node = parent;
                    }
                    path.reverse();
                    return Some(path);
                }
            }
        }
    }

    None // No path found
}

/// Find all paths between two nodes up to a maximum number of hops.
/// Warning: This can be expensive for graphs with many paths!
pub fn all_paths(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    max_hops: usize,
) -> Vec<Vec<NodeIndex>> {
    let mut results = Vec::new();
    let mut current_path = vec![source];
    let mut visited = std::collections::HashSet::new();
    visited.insert(source);

    find_all_paths_recursive(
        graph,
        source,
        target,
        max_hops,
        &mut current_path,
        &mut visited,
        &mut results,
    );

    results
}

fn find_all_paths_recursive(
    graph: &DirGraph,
    current: NodeIndex,
    target: NodeIndex,
    remaining_hops: usize,
    current_path: &mut Vec<NodeIndex>,
    visited: &mut std::collections::HashSet<NodeIndex>,
    results: &mut Vec<Vec<NodeIndex>>,
) {
    if current == target {
        results.push(current_path.clone());
        return;
    }

    if remaining_hops == 0 {
        return;
    }

    // Explore all neighbors (undirected)
    for neighbor in graph.graph.neighbors_undirected(current) {
        if !visited.contains(&neighbor) {
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
pub fn weakly_connected_components(graph: &DirGraph) -> Vec<Vec<NodeIndex>> {
    use std::collections::{HashSet, VecDeque};

    let node_count = graph.graph.node_count();
    let mut visited = HashSet::with_capacity(node_count);
    let mut components = Vec::new();

    for node in graph.graph.node_indices() {
        if visited.contains(&node) {
            continue;
        }

        // BFS to find all connected nodes - estimate component size
        let remaining = node_count - visited.len();
        let mut component = Vec::with_capacity(remaining.min(100)); // Cap initial estimate
        let mut queue = VecDeque::with_capacity(remaining.min(100));
        queue.push_back(node);
        visited.insert(node);

        while let Some(current) = queue.pop_front() {
            component.push(current);

            // Add all neighbors (treating as undirected)
            for neighbor in graph.graph.neighbors_undirected(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        components.push(component);
    }

    // Sort components by size (largest first)
    components.sort_by(|a, b| b.len().cmp(&a.len()));

    components
}

/// Get node info for building Python-friendly path output
pub fn get_node_info(graph: &DirGraph, node_idx: NodeIndex) -> Option<PathNodeInfo> {
    match graph.get_node(node_idx)? {
        NodeData::Regular { node_type, title, id, .. } => {
            let title_str = match title {
                Value::String(s) => s.clone(),
                _ => format!("{:?}", title),
            };
            Some(PathNodeInfo {
                node_type: node_type.clone(),
                title: title_str,
                id: id.clone(),
            })
        }
        NodeData::Schema { node_type, title, .. } => {
            let title_str = match title {
                Value::String(s) => s.clone(),
                _ => format!("{:?}", title),
            };
            Some(PathNodeInfo {
                node_type: node_type.clone(),
                title: title_str,
                id: title.clone(),
            })
        }
    }
}

/// Get information about what connection types link nodes in a path
pub fn get_path_connections(
    graph: &DirGraph,
    path: &[NodeIndex],
) -> Vec<Option<String>> {
    // Pre-allocate with exact size (one connection per edge = path.len() - 1)
    let mut connections = Vec::with_capacity(path.len().saturating_sub(1));

    for window in path.windows(2) {
        let from = window[0];
        let to = window[1];

        // Find edge between these nodes (either direction)
        let conn_type = graph.graph.edges(from)
            .find(|e| e.target() == to)
            .map(|e| e.weight().connection_type.clone())
            .or_else(|| {
                graph.graph.edges(to)
                    .find(|e| e.target() == from)
                    .map(|e| e.weight().connection_type.clone())
            });

        connections.push(conn_type);
    }

    connections
}

/// Check if two nodes are connected (directly or indirectly)
pub fn are_connected(graph: &DirGraph, source: NodeIndex, target: NodeIndex) -> bool {
    shortest_path(graph, source, target).is_some()
}

/// Calculate the degree (number of connections) for a node
pub fn node_degree(graph: &DirGraph, node: NodeIndex) -> usize {
    graph.graph.edges(node).count() +
    graph.graph.neighbors_directed(node, petgraph::Direction::Incoming).count()
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
) -> Vec<CentralityResult> {
    use std::collections::VecDeque;

    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n <= 2 {
        return nodes.iter().map(|&idx| CentralityResult { node_idx: idx, score: 0.0 }).collect();
    }

    // Create node index mapping for O(1) lookup (NodeIndex -> array index)
    let node_to_idx: HashMap<NodeIndex, usize> = nodes.iter()
        .enumerate()
        .map(|(i, &node)| (node, i))
        .collect();

    // Initialize betweenness scores using Vec for O(1) access
    let mut betweenness: Vec<f64> = vec![0.0; n];

    // Determine which nodes to use as sources
    let source_indices: Vec<usize> = if let Some(k) = sample_size {
        let k = k.min(n);
        (0..k).collect()
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
    for &s_idx in &source_indices {
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

        // BFS phase
        while let Some(v_idx) = queue.pop_front() {
            stack.push(v_idx);
            let v_dist = dist[v_idx];
            let v_node = nodes[v_idx];

            // Traverse all neighbors
            for w_node in graph.graph.neighbors_undirected(v_node) {
                let w_idx = node_to_idx[&w_node];

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

    // Normalize if requested
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
    let mut results: Vec<CentralityResult> = nodes.iter()
        .enumerate()
        .map(|(i, &node_idx)| CentralityResult { node_idx, score: betweenness[i] })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

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
) -> Vec<CentralityResult> {
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return Vec::new();
    }

    // Create node index mapping for efficient lookup
    let node_to_idx: HashMap<NodeIndex, usize> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    // Initialize PageRank scores (uniform distribution)
    let mut pr: Vec<f64> = vec![1.0 / n as f64; n];
    let mut new_pr: Vec<f64> = vec![0.0; n];

    // Precompute out-degrees
    let out_degrees: Vec<usize> = nodes.iter()
        .map(|&node| graph.graph.neighbors_undirected(node).count())
        .collect();

    // Identify dangling nodes (no outgoing links)
    let dangling_nodes: Vec<usize> = out_degrees.iter()
        .enumerate()
        .filter(|(_, &deg)| deg == 0)
        .map(|(i, _)| i)
        .collect();

    let teleport = (1.0 - damping_factor) / n as f64;

    // Iterative computation
    for _iteration in 0..max_iterations {
        // Calculate dangling node contribution
        let dangling_sum: f64 = dangling_nodes.iter().map(|&i| pr[i]).sum();
        let dangling_contrib = damping_factor * dangling_sum / n as f64;

        // Reset new_pr
        for score in new_pr.iter_mut() {
            *score = teleport + dangling_contrib;
        }

        // Add contributions from incoming links
        for (i, &node) in nodes.iter().enumerate() {
            if out_degrees[i] > 0 {
                let contrib = damping_factor * pr[i] / out_degrees[i] as f64;
                for neighbor in graph.graph.neighbors_undirected(node) {
                    if let Some(&j) = node_to_idx.get(&neighbor) {
                        new_pr[j] += contrib;
                    }
                }
            }
        }

        // Check for convergence
        let diff: f64 = pr.iter().zip(new_pr.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();

        std::mem::swap(&mut pr, &mut new_pr);

        if diff < tolerance {
            break;
        }
    }

    // Convert to results and sort by score
    let mut results: Vec<CentralityResult> = nodes.iter()
        .enumerate()
        .map(|(i, &node_idx)| CentralityResult { node_idx, score: pr[i] })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Calculate degree centrality for all nodes.
///
/// Simply counts the number of connections each node has.
/// Optionally normalized by (n-1) to get values between 0 and 1.
pub fn degree_centrality(graph: &DirGraph, normalized: bool) -> Vec<CentralityResult> {
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    let scale = if normalized && n > 1 { 1.0 / (n - 1) as f64 } else { 1.0 };

    let mut results: Vec<CentralityResult> = nodes.iter()
        .map(|&node_idx| {
            let degree = node_degree(graph, node_idx);
            CentralityResult {
                node_idx,
                score: degree as f64 * scale,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Calculate closeness centrality for all nodes.
///
/// Closeness centrality measures how close a node is to all other nodes.
/// Defined as the reciprocal of the sum of shortest path distances.
///
/// Note: For disconnected graphs, only reachable nodes are considered.
/// Optimized to use Vec instead of HashMap for O(1) direct indexing.
pub fn closeness_centrality(graph: &DirGraph, normalized: bool) -> Vec<CentralityResult> {
    use std::collections::VecDeque;

    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return Vec::new();
    }

    // Create node index mapping for O(1) lookup (NodeIndex -> array index)
    let node_to_idx: HashMap<NodeIndex, usize> = nodes.iter()
        .enumerate()
        .map(|(i, &node)| (node, i))
        .collect();

    let mut results = Vec::with_capacity(n);

    // Pre-allocate data structures ONCE outside the loop
    // -1 means not visited, >= 0 is the distance
    let mut dist: Vec<i64> = vec![-1; n];
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);

    for (s_idx, &source) in nodes.iter().enumerate() {
        // Reset data structures (much faster than re-allocating)
        queue.clear();
        for d in dist.iter_mut() {
            *d = -1;
        }

        // BFS to find distances to all reachable nodes
        queue.push_back(s_idx);
        dist[s_idx] = 0;

        while let Some(current_idx) = queue.pop_front() {
            let current_dist = dist[current_idx];
            let current_node = nodes[current_idx];

            for neighbor in graph.graph.neighbors_undirected(current_node) {
                let neighbor_idx = node_to_idx[&neighbor];
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

            results.push(CentralityResult { node_idx: source, score });
        } else {
            results.push(CentralityResult { node_idx: source, score: 0.0 });
        }
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    results
}

#[cfg(test)]
mod tests {
    // Tests are implemented in Python (pytest/test_comprehensive_integration.py)
}
