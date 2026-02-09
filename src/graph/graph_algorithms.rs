// src/graph/graph_algorithms.rs
//! Graph algorithms module providing path finding and connectivity analysis.

use crate::datatypes::values::Value;
use crate::graph::schema::DirGraph;
use crate::graph::value_operations;
use petgraph::algo::kosaraju_scc;
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};
use std::collections::HashMap;

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
pub fn shortest_path(graph: &DirGraph, source: NodeIndex, target: NodeIndex) -> Option<PathResult> {
    // Use BFS for undirected path finding (more appropriate for knowledge graphs)
    let path = reconstruct_path_bfs(graph, source, target)?;
    let cost = path.len().saturating_sub(1); // Cost is number of edges

    Some(PathResult { path, cost })
}

/// Reconstruct path using BFS with Vec-based tracking for O(1) operations.
/// Uses Vec<bool> for visited and Vec<u32> for parent tracking instead of HashMap/HashSet.
fn reconstruct_path_bfs(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
) -> Option<Vec<NodeIndex>> {
    use std::collections::VecDeque;

    if source == target {
        return Some(vec![source]);
    }

    // Use node_bound() not node_count() — with StableDiGraph, deleted nodes leave
    // holes so indices can exceed node_count(). node_bound() is the upper bound.
    let node_bound = graph.graph.node_bound();

    // Use Vec instead of HashSet/HashMap for O(1) direct indexing
    // visited[i] = true if node i has been visited
    let mut visited: Vec<bool> = vec![false; node_bound];
    // parent[i] = parent node index of node i (u32::MAX means no parent/source)
    let mut parent: Vec<u32> = vec![u32::MAX; node_bound];

    let mut queue = VecDeque::with_capacity(node_bound / 4);

    let source_idx = source.index();
    let target_idx = target.index();

    queue.push_back(source_idx);
    visited[source_idx] = true;

    while let Some(current_idx) = queue.pop_front() {
        let current = NodeIndex::new(current_idx);

        // Check all neighbors (both directions for undirected path finding)
        for neighbor in graph.graph.neighbors_undirected(current) {
            let neighbor_idx = neighbor.index();

            if !visited[neighbor_idx] {
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
pub fn shortest_path_directed(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
) -> Option<PathResult> {
    use petgraph::Direction;
    use std::collections::VecDeque;

    if source == target {
        return Some(PathResult {
            path: vec![source],
            cost: 0,
        });
    }

    let node_bound = graph.graph.node_bound();
    let mut visited: Vec<bool> = vec![false; node_bound];
    let mut parent: Vec<u32> = vec![u32::MAX; node_bound];
    let mut queue = VecDeque::with_capacity(node_bound / 4);

    let source_idx = source.index();
    let target_idx = target.index();

    queue.push_back(source_idx);
    visited[source_idx] = true;

    while let Some(current_idx) = queue.pop_front() {
        let current = NodeIndex::new(current_idx);

        // Only follow outgoing edges
        for neighbor in graph.graph.neighbors_directed(current, Direction::Outgoing) {
            let neighbor_idx = neighbor.index();

            if !visited[neighbor_idx] {
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

#[allow(clippy::only_used_in_recursion)]
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
/// Optimized to use Vec<bool> for O(1) visited tracking.
pub fn weakly_connected_components(graph: &DirGraph) -> Vec<Vec<NodeIndex>> {
    use std::collections::VecDeque;

    // Use node_bound() not node_count() — StableDiGraph indices can have gaps
    let node_bound = graph.graph.node_bound();
    // Use Vec<bool> for O(1) visited tracking instead of HashSet
    let mut visited: Vec<bool> = vec![false; node_bound];
    let mut components = Vec::new();
    let mut visited_count = 0;

    for node in graph.graph.node_indices() {
        let node_idx = node.index();
        if visited[node_idx] {
            continue;
        }

        // BFS to find all connected nodes - estimate component size
        let remaining = graph.graph.node_count() - visited_count;
        let mut component = Vec::with_capacity(remaining.min(100)); // Cap initial estimate
        let mut queue = VecDeque::with_capacity(remaining.min(100));
        queue.push_back(node);
        visited[node_idx] = true;
        visited_count += 1;

        while let Some(current) = queue.pop_front() {
            component.push(current);

            // Add all neighbors (treating as undirected)
            for neighbor in graph.graph.neighbors_undirected(current) {
                let neighbor_idx = neighbor.index();
                if !visited[neighbor_idx] {
                    visited[neighbor_idx] = true;
                    visited_count += 1;
                    queue.push_back(neighbor);
                }
            }
        }

        components.push(component);
    }

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
    shortest_path(graph, source, target).is_some()
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

    // Create node index mapping for O(1) lookup (NodeIndex -> array index)
    let node_to_idx: HashMap<NodeIndex, usize> = nodes
        .iter()
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
    let mut results: Vec<CentralityResult> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node_idx)| CentralityResult {
            node_idx,
            score: betweenness[i],
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| {
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
) -> Vec<CentralityResult> {
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return Vec::new();
    }

    // Create node index mapping for efficient lookup
    let node_to_idx: HashMap<NodeIndex, usize> =
        nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    // Initialize PageRank scores (uniform distribution)
    let mut pr: Vec<f64> = vec![1.0 / n as f64; n];
    let mut new_pr: Vec<f64> = vec![0.0; n];

    // Precompute out-degrees
    let out_degrees: Vec<usize> = nodes
        .iter()
        .map(|&node| graph.graph.neighbors_undirected(node).count())
        .collect();

    // Identify dangling nodes (no outgoing links)
    let dangling_nodes: Vec<usize> = out_degrees
        .iter()
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
        let diff: f64 = pr
            .iter()
            .zip(new_pr.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();

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

    results.sort_by(|a, b| {
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
pub fn degree_centrality(graph: &DirGraph, normalized: bool) -> Vec<CentralityResult> {
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    let scale = if normalized && n > 1 {
        1.0 / (n - 1) as f64
    } else {
        1.0
    };

    let mut results: Vec<CentralityResult> = nodes
        .iter()
        .map(|&node_idx| {
            let degree = node_degree(graph, node_idx);
            CentralityResult {
                node_idx,
                score: degree as f64 * scale,
            }
        })
        .collect();

    results.sort_by(|a, b| {
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
pub fn closeness_centrality(graph: &DirGraph, normalized: bool) -> Vec<CentralityResult> {
    use std::collections::VecDeque;

    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return Vec::new();
    }

    // Create node index mapping for O(1) lookup (NodeIndex -> array index)
    let node_to_idx: HashMap<NodeIndex, usize> = nodes
        .iter()
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

    results.sort_by(|a, b| {
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
pub fn louvain_communities(
    graph: &DirGraph,
    weight_property: Option<&str>,
    resolution: f64,
) -> CommunityResult {
    let bound = graph.graph.node_bound();
    if bound == 0 {
        return CommunityResult {
            assignments: Vec::new(),
            num_communities: 0,
            modularity: 0.0,
        };
    }

    // Build adjacency with weights
    // community[i] = community id for node at index i
    let mut community: Vec<usize> = vec![0; bound];
    let mut node_exists: Vec<bool> = vec![false; bound];

    // Initialize: each node in its own community
    let mut next_id = 0usize;
    for node_idx in graph.graph.node_indices() {
        let i = node_idx.index();
        community[i] = next_id;
        node_exists[i] = true;
        next_id += 1;
    }

    // Compute total edge weight (2m)
    let mut total_weight = 0.0f64;
    for edge in graph.graph.edge_references() {
        let w = edge_weight(graph, edge.id(), weight_property);
        total_weight += w;
    }

    if total_weight == 0.0 {
        // No edges — each node is its own community
        let assignments: Vec<CommunityAssignment> = graph
            .graph
            .node_indices()
            .map(|idx| CommunityAssignment {
                node_idx: idx,
                community_id: community[idx.index()],
            })
            .collect();
        let num_communities = assignments.len();
        return CommunityResult {
            assignments,
            num_communities,
            modularity: 0.0,
        };
    }

    // Precompute node degrees (undirected: sum of all edge weights touching the node)
    let mut degree: Vec<f64> = vec![0.0; bound];
    for edge in graph.graph.edge_references() {
        let w = edge_weight(graph, edge.id(), weight_property);
        degree[edge.source().index()] += w;
        degree[edge.target().index()] += w;
    }

    // Precompute sum of degrees per community (sigma_tot)
    let mut sigma_tot: Vec<f64> = vec![0.0; next_id];
    for node_idx in graph.graph.node_indices() {
        sigma_tot[community[node_idx.index()]] += degree[node_idx.index()];
    }

    // m = total edge weight (each edge counted once)
    let m = total_weight;
    let two_m = 2.0 * m;

    // Iterative optimization
    let max_iterations = 100;
    for _ in 0..max_iterations {
        let mut improved = false;

        for node_idx in graph.graph.node_indices() {
            let i = node_idx.index();
            let current_community = community[i];
            let k_i = degree[i];

            // Compute weight from node i to each neighboring community (undirected)
            let mut community_weights: HashMap<usize, f64> = HashMap::new();

            for edge in graph.graph.edges(node_idx) {
                let neighbor = edge.target();
                let w = edge_weight(graph, edge.id(), weight_property);
                *community_weights
                    .entry(community[neighbor.index()])
                    .or_default() += w;
            }
            for edge in graph
                .graph
                .edges_directed(node_idx, petgraph::Direction::Incoming)
            {
                let neighbor = edge.source();
                let w = edge_weight(graph, edge.id(), weight_property);
                *community_weights
                    .entry(community[neighbor.index()])
                    .or_default() += w;
            }

            // Weight from i into its own community (excluding self)
            let k_i_in_current = *community_weights.get(&current_community).unwrap_or(&0.0);

            // Find best community to move to
            let mut best_community = current_community;
            let mut best_delta = 0.0f64;

            for (&cand_community, &k_i_in_cand) in &community_weights {
                if cand_community == current_community {
                    continue;
                }

                // Standard Louvain modularity gain formula:
                // delta_Q = k_i_in_cand/m - resolution * sigma_tot_cand * k_i / (2m^2)
                //         - (- k_i_in_current/m + resolution * (sigma_tot_current - k_i) * k_i / (2m^2))
                let sigma_cand = sigma_tot[cand_community];
                let sigma_curr = sigma_tot[current_community] - k_i; // exclude node i

                let gain_add = k_i_in_cand / m - resolution * sigma_cand * k_i / (two_m * two_m);
                let loss_remove =
                    k_i_in_current / m - resolution * sigma_curr * k_i / (two_m * two_m);
                let delta = gain_add - loss_remove;

                if delta > best_delta {
                    best_delta = delta;
                    best_community = cand_community;
                }
            }

            if best_community != current_community {
                // Update sigma_tot
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

    // Renumber communities to be contiguous 0..n
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    for node_idx in graph.graph.node_indices() {
        let c = community[node_idx.index()];
        let next_id = id_map.len();
        id_map.entry(c).or_insert(next_id);
    }

    let assignments: Vec<CommunityAssignment> = graph
        .graph
        .node_indices()
        .map(|idx| CommunityAssignment {
            node_idx: idx,
            community_id: *id_map.get(&community[idx.index()]).unwrap(),
        })
        .collect();

    let num_communities = id_map.len();
    let modularity = compute_modularity(
        graph,
        &community,
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
pub fn label_propagation(graph: &DirGraph, max_iterations: usize) -> CommunityResult {
    let bound = graph.graph.node_bound();
    if bound == 0 {
        return CommunityResult {
            assignments: Vec::new(),
            num_communities: 0,
            modularity: 0.0,
        };
    }

    let mut labels: Vec<usize> = vec![0; bound];
    let mut node_exists: Vec<bool> = vec![false; bound];

    // Initialize: each node gets a unique label
    for (i, node_idx) in graph.graph.node_indices().enumerate() {
        labels[node_idx.index()] = i;
        node_exists[node_idx.index()] = true;
    }

    // Collect node indices for iteration
    let node_indices: Vec<NodeIndex> = graph.graph.node_indices().collect();

    for _ in 0..max_iterations {
        let mut changed = false;

        for &node_idx in &node_indices {
            // Count neighbor labels
            let mut label_counts: HashMap<usize, usize> = HashMap::new();

            for neighbor in graph.graph.neighbors_undirected(node_idx) {
                *label_counts.entry(labels[neighbor.index()]).or_default() += 1;
            }

            if label_counts.is_empty() {
                continue; // isolated node keeps its label
            }

            // Find most frequent label
            let &best_label = label_counts
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|(label, _)| label)
                .unwrap();

            if best_label != labels[node_idx.index()] {
                labels[node_idx.index()] = best_label;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Renumber labels to be contiguous
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    for &node_idx in &node_indices {
        let l = labels[node_idx.index()];
        let next_id = id_map.len();
        id_map.entry(l).or_insert(next_id);
    }

    let assignments: Vec<CommunityAssignment> = node_indices
        .iter()
        .map(|&idx| CommunityAssignment {
            node_idx: idx,
            community_id: *id_map.get(&labels[idx.index()]).unwrap(),
        })
        .collect();

    // Compute modularity for result
    let mut total_weight = 0.0f64;
    for _ in graph.graph.edge_references() {
        total_weight += 1.0;
    }

    let num_communities = id_map.len();
    let modularity = compute_modularity(graph, &labels, &node_exists, total_weight, None);

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
        let result = shortest_path(&graph, indices[0], indices[1]);
        assert!(result.is_some());
        let path = result.unwrap();
        assert_eq!(path.cost, 1);
        assert_eq!(path.path.len(), 2);
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let (graph, indices) = build_chain_graph();
        let result = shortest_path(&graph, indices[0], indices[4]);
        assert!(result.is_some());
        let path = result.unwrap();
        assert_eq!(path.cost, 4);
        assert_eq!(path.path.len(), 5);
    }

    #[test]
    fn test_shortest_path_same_node() {
        let (graph, indices) = build_chain_graph();
        let result = shortest_path(&graph, indices[0], indices[0]);
        assert!(result.is_some());
        let path = result.unwrap();
        assert_eq!(path.cost, 0);
        assert_eq!(path.path.len(), 1);
    }

    #[test]
    fn test_shortest_path_not_found() {
        let (graph, indices) = build_disconnected_graph();
        let result = shortest_path(&graph, indices[0], indices[2]);
        assert!(result.is_none());
    }

    #[test]
    fn test_shortest_path_reverse_direction() {
        // BFS is undirected, so B -> A should find a path even though edge is A -> B
        let (graph, indices) = build_chain_graph();
        let result = shortest_path(&graph, indices[4], indices[0]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().cost, 4);
    }

    // ========================================================================
    // all_paths
    // ========================================================================

    #[test]
    fn test_all_paths_basic() {
        let (graph, indices) = build_chain_graph();
        let paths = all_paths(&graph, indices[0], indices[2], 5);
        assert!(!paths.is_empty());
        // There should be a path of length 2: A -> B -> C
        assert!(paths.iter().any(|p| p.len() == 3));
    }

    #[test]
    fn test_all_paths_limited_hops() {
        let (graph, indices) = build_chain_graph();
        // With max_hops=1, can only reach adjacent node
        let paths = all_paths(&graph, indices[0], indices[2], 1);
        assert!(paths.is_empty()); // Can't reach C in 1 hop
    }

    #[test]
    fn test_all_paths_triangle() {
        let (graph, indices) = build_triangle_graph();
        let paths = all_paths(&graph, indices[0], indices[2], 3);
        // Multiple paths possible in a triangle
        assert!(!paths.is_empty());
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
        let results = betweenness_centrality(&graph, false, None);
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
    fn test_degree_centrality() {
        let (graph, indices) = build_chain_graph();
        let results = degree_centrality(&graph, false);
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
        let results = pagerank(&graph, 0.85, 100, 1e-6);
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
        let results = closeness_centrality(&graph, false);
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
        let results = pagerank(&graph, 0.85, 100, 1e-6);
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
