// Vector search module for embedding-based similarity queries.
// Operates on the current graph selection for filtered vector search.

use crate::graph::schema::{CurrentSelection, DirGraph, EmbeddingStore};
use petgraph::graph::NodeIndex;
use std::collections::BinaryHeap;

/// Distance metric for vector similarity search.
#[derive(Clone, Copy, Debug)]
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

/// A single vector search result: node index + similarity score.
#[derive(Clone, Debug)]
pub struct VectorSearchResult {
    pub node_idx: NodeIndex,
    pub score: f32,
}

/// Threshold for switching to parallel search via rayon.
const PARALLEL_THRESHOLD: usize = 10_000;

/// Perform vector search over the current selection.
///
/// Gets candidate nodes from the selection's current level, computes similarity
/// for each candidate that has an embedding, and returns top-k results sorted
/// by score (descending for cosine/dot, ascending for euclidean).
pub fn vector_search(
    graph: &DirGraph,
    selection: &CurrentSelection,
    embedding_property: &str,
    query_vector: &[f32],
    top_k: usize,
    metric: DistanceMetric,
) -> Result<Vec<VectorSearchResult>, String> {
    let level_count = selection.get_level_count();
    if level_count == 0 {
        return Ok(Vec::new());
    }

    let candidates: Vec<NodeIndex> = selection
        .get_level(level_count - 1)
        .map(|l| l.get_all_nodes())
        .unwrap_or_default();

    if candidates.is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }

    // Fast path: check if first candidate's type has an embedding store (common after type_filter)
    let first_type = graph
        .graph
        .node_weight(candidates[0])
        .map(|n| n.node_type.as_str());

    let single_type = first_type.and_then(|ft| {
        let key = (ft.to_string(), embedding_property.to_string());
        graph.embeddings.get(&key).map(|store| (ft, store))
    });

    let results = if let Some((node_type, store)) = single_type {
        // Validate query vector dimension
        if query_vector.len() != store.dimension {
            return Err(format!(
                "Query vector dimension {} does not match embedding dimension {} for '{}.{}'",
                query_vector.len(),
                store.dimension,
                node_type,
                embedding_property
            ));
        }

        let similarity_fn = get_similarity_fn(metric);

        if candidates.len() > PARALLEL_THRESHOLD {
            parallel_search(&candidates, store, query_vector, top_k, similarity_fn)
        } else {
            sequential_search(&candidates, store, query_vector, top_k, similarity_fn)
        }
    } else {
        // Multi-type path: group by node type
        let similarity_fn = get_similarity_fn(metric);
        let mut heap = MinHeap::with_capacity(top_k);

        for &node_idx in &candidates {
            let node_type = match graph.graph.node_weight(node_idx) {
                Some(n) => &n.node_type,
                None => continue,
            };

            let key = (node_type.clone(), embedding_property.to_string());
            let store = match graph.embeddings.get(&key) {
                Some(s) => s,
                None => continue,
            };

            if query_vector.len() != store.dimension {
                return Err(format!(
                    "Query vector dimension {} does not match embedding dimension {} for '{}.{}'",
                    query_vector.len(),
                    store.dimension,
                    node_type,
                    embedding_property
                ));
            }

            if let Some(embedding) = store.get_embedding(node_idx.index()) {
                let score = similarity_fn(query_vector, embedding);
                heap.push_if_better(node_idx, score, top_k);
            }
        }

        heap.into_sorted_results()
    };

    Ok(results)
}

// ─── Similarity Functions ──────────────────────────────────────────────────────

type SimilarityFn = fn(&[f32], &[f32]) -> f32;

fn get_similarity_fn(metric: DistanceMetric) -> SimilarityFn {
    match metric {
        DistanceMetric::Cosine => cosine_similarity,
        DistanceMetric::DotProduct => dot_product,
        DistanceMetric::Euclidean => neg_euclidean_distance,
    }
}

/// Cosine similarity between two f32 slices.
/// Uses chunks_exact(8) for LLVM auto-vectorization (SSE2/AVX2/NEON).
/// Returns similarity in [-1.0, 1.0].
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();

    for (ac, bc) in a_chunks.zip(b_chunks) {
        // Unrolled loop — LLVM recognizes this pattern and vectorizes it
        for i in 0..8 {
            dot += ac[i] * bc[i];
            norm_a += ac[i] * ac[i];
            norm_b += bc[i] * bc[i];
        }
    }
    for (av, bv) in a_rem.iter().zip(b_rem.iter()) {
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// Dot product similarity.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();

    for (ac, bc) in a_chunks.zip(b_chunks) {
        for i in 0..8 {
            sum += ac[i] * bc[i];
        }
    }
    for (av, bv) in a_rem.iter().zip(b_rem.iter()) {
        sum += av * bv;
    }

    sum
}

/// Negative Euclidean distance (higher = more similar).
#[inline]
pub fn neg_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();

    for (ac, bc) in a_chunks.zip(b_chunks) {
        for i in 0..8 {
            let d = ac[i] - bc[i];
            sum += d * d;
        }
    }
    for (av, bv) in a_rem.iter().zip(b_rem.iter()) {
        let d = av - bv;
        sum += d * d;
    }

    -sum.sqrt()
}

// ─── Top-K Min-Heap ────────────────────────────────────────────────────────────

/// Wrapper for min-heap that keeps the top-k highest-scoring results.
struct MinHeap {
    heap: BinaryHeap<ScoredNode>,
}

/// Node with score, ordered so BinaryHeap acts as a min-heap (lowest score at top).
struct ScoredNode {
    score: f32,
    node_idx: NodeIndex,
}

impl PartialEq for ScoredNode {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredNode {}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed: lower score = higher priority in the heap (min-heap)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl MinHeap {
    fn with_capacity(cap: usize) -> Self {
        MinHeap {
            heap: BinaryHeap::with_capacity(cap + 1),
        }
    }

    #[inline]
    fn push_if_better(&mut self, node_idx: NodeIndex, score: f32, top_k: usize) {
        if self.heap.len() < top_k {
            self.heap.push(ScoredNode { score, node_idx });
        } else if let Some(min) = self.heap.peek() {
            if score > min.score {
                self.heap.pop();
                self.heap.push(ScoredNode { score, node_idx });
            }
        }
    }

    fn into_sorted_results(self) -> Vec<VectorSearchResult> {
        let mut results: Vec<VectorSearchResult> = self
            .heap
            .into_vec()
            .into_iter()
            .map(|sn| VectorSearchResult {
                node_idx: sn.node_idx,
                score: sn.score,
            })
            .collect();
        // Sort descending by score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}

// ─── Search Implementations ────────────────────────────────────────────────────

fn sequential_search(
    candidates: &[NodeIndex],
    store: &EmbeddingStore,
    query: &[f32],
    top_k: usize,
    similarity_fn: SimilarityFn,
) -> Vec<VectorSearchResult> {
    let mut heap = MinHeap::with_capacity(top_k);

    for &node_idx in candidates {
        if let Some(embedding) = store.get_embedding(node_idx.index()) {
            let score = similarity_fn(query, embedding);
            heap.push_if_better(node_idx, score, top_k);
        }
    }

    heap.into_sorted_results()
}

fn parallel_search(
    candidates: &[NodeIndex],
    store: &EmbeddingStore,
    query: &[f32],
    top_k: usize,
    similarity_fn: SimilarityFn,
) -> Vec<VectorSearchResult> {
    use rayon::prelude::*;

    let chunk_size = (candidates.len() / rayon::current_num_threads()).max(1024);

    let per_thread_results: Vec<Vec<VectorSearchResult>> = candidates
        .par_chunks(chunk_size)
        .map(|chunk| sequential_search(chunk, store, query, top_k, similarity_fn))
        .collect();

    // Merge per-thread top-k results
    let mut heap = MinHeap::with_capacity(top_k);
    for thread_results in per_thread_results {
        for result in thread_results {
            heap.push_if_better(result.node_idx, result.score, top_k);
        }
    }

    heap.into_sorted_results()
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_large_vector() {
        // Test with >8 elements to exercise chunked path
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| (i * 2) as f32).collect();
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.99); // Nearly parallel vectors
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dp = dot_product(&a, &b);
        assert!((dp - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_neg_euclidean_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let d = neg_euclidean_distance(&a, &b);
        assert!(d.abs() < 1e-6); // Distance 0 → -0.0
    }

    #[test]
    fn test_neg_euclidean_distance_basic() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let d = neg_euclidean_distance(&a, &b);
        assert!((d + 5.0).abs() < 1e-6); // -sqrt(9+16) = -5.0
    }

    #[test]
    fn test_min_heap_top_k() {
        let mut heap = MinHeap::with_capacity(3);
        let scores = [0.5, 0.9, 0.1, 0.8, 0.3, 0.95, 0.2];

        for (i, &score) in scores.iter().enumerate() {
            heap.push_if_better(NodeIndex::new(i), score, 3);
        }

        let results = heap.into_sorted_results();
        assert_eq!(results.len(), 3);
        assert!((results[0].score - 0.95).abs() < 1e-6);
        assert!((results[1].score - 0.9).abs() < 1e-6);
        assert!((results[2].score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_store_basic() {
        let mut store = EmbeddingStore::new(3);
        store.set_embedding(0, &[1.0, 2.0, 3.0]);
        store.set_embedding(5, &[4.0, 5.0, 6.0]);

        assert_eq!(store.len(), 2);
        assert_eq!(store.get_embedding(0), Some([1.0, 2.0, 3.0].as_slice()));
        assert_eq!(store.get_embedding(5), Some([4.0, 5.0, 6.0].as_slice()));
        assert_eq!(store.get_embedding(1), None);
    }

    #[test]
    fn test_embedding_store_replace() {
        let mut store = EmbeddingStore::new(2);
        store.set_embedding(0, &[1.0, 2.0]);
        store.set_embedding(0, &[3.0, 4.0]);

        assert_eq!(store.len(), 1);
        assert_eq!(store.get_embedding(0), Some([3.0, 4.0].as_slice()));
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }
}
