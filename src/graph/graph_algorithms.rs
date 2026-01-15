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

#[cfg(test)]
mod tests {
    use super::*;

    // Tests would require setting up a graph, which we'll do in Python tests
}
