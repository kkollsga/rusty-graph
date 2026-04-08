// src/graph/graph_iterators.rs
//
// Enum iterator wrappers for GraphBackend. These allow the same iterator
// interface to work for both InMemory (petgraph) and Disk backends.
//
// GraphEdgeRef implements petgraph::visit::EdgeRef so callers that import
// that trait can call .source(), .target(), .weight(), .id() unchanged.

use crate::graph::disk_graph::{CsrEdge, DiskGraph, DiskNodeSlot, TOMBSTONE_EDGE};
use crate::graph::mmap_vec::MmapOrVec;
use crate::graph::schema::{EdgeData, NodeData};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

// ============================================================================
// GraphEdgeRef — unified edge reference
// ============================================================================

/// A lightweight edge reference returned by graph edge iterators.
/// Implements `petgraph::visit::EdgeRef` for compatibility with existing code.
#[derive(Clone, Copy)]
pub struct GraphEdgeRef<'a> {
    source_: NodeIndex,
    target_: NodeIndex,
    index_: EdgeIndex,
    weight_: &'a EdgeData,
}

impl<'a> GraphEdgeRef<'a> {
    #[inline]
    pub fn new(
        source: NodeIndex,
        target: NodeIndex,
        index: EdgeIndex,
        weight: &'a EdgeData,
    ) -> Self {
        GraphEdgeRef {
            source_: source,
            target_: target,
            index_: index,
            weight_: weight,
        }
    }

    // Inherent methods matching petgraph's EdgeReference API.
    // These allow callers to use .source()/.target()/.weight()/.id()
    // without importing the EdgeRef trait.

    #[inline]
    pub fn source(&self) -> NodeIndex {
        self.source_
    }

    #[inline]
    pub fn target(&self) -> NodeIndex {
        self.target_
    }

    #[inline]
    pub fn weight(&self) -> &'a EdgeData {
        self.weight_
    }

    #[inline]
    pub fn id(&self) -> EdgeIndex {
        self.index_
    }
}

impl<'a> EdgeRef for GraphEdgeRef<'a> {
    type NodeId = NodeIndex;
    type EdgeId = EdgeIndex;
    type Weight = EdgeData;

    #[inline]
    fn source(&self) -> NodeIndex {
        self.source_
    }

    #[inline]
    fn target(&self) -> NodeIndex {
        self.target_
    }

    #[inline]
    fn weight(&self) -> &EdgeData {
        self.weight_
    }

    #[inline]
    fn id(&self) -> EdgeIndex {
        self.index_
    }
}

// ============================================================================
// GraphNodeIndices
// ============================================================================

pub enum GraphNodeIndices<'a> {
    InMemory(petgraph::stable_graph::NodeIndices<'a, NodeData, u32>),
    Disk(DiskNodeIndices<'a>),
}

/// Iterates alive node slots in the DiskGraph's mmap'd node_slots array.
pub struct DiskNodeIndices<'a> {
    node_slots: &'a MmapOrVec<DiskNodeSlot>,
    pos: usize,
    len: usize,
}

impl<'a> DiskNodeIndices<'a> {
    pub fn new(node_slots: &'a MmapOrVec<DiskNodeSlot>) -> Self {
        let len = node_slots.len();
        DiskNodeIndices {
            node_slots,
            pos: 0,
            len,
        }
    }
}

impl<'a> Iterator for DiskNodeIndices<'a> {
    type Item = NodeIndex;
    fn next(&mut self) -> Option<NodeIndex> {
        while self.pos < self.len {
            let i = self.pos;
            self.pos += 1;
            if self.node_slots.get(i).is_alive() {
                return Some(NodeIndex::new(i));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.len - self.pos))
    }
}

impl<'a> Iterator for GraphNodeIndices<'a> {
    type Item = NodeIndex;

    #[inline]
    fn next(&mut self) -> Option<NodeIndex> {
        match self {
            GraphNodeIndices::InMemory(iter) => iter.next(),
            GraphNodeIndices::Disk(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            GraphNodeIndices::InMemory(iter) => iter.size_hint(),
            GraphNodeIndices::Disk(iter) => iter.size_hint(),
        }
    }
}

// ============================================================================
// GraphEdges — edges from a single node (directed)
// ============================================================================

pub enum GraphEdges<'a> {
    InMemory(petgraph::stable_graph::Edges<'a, EdgeData, petgraph::Directed, u32>),
    Disk(DiskEdges<'a>),
}

/// Iterates CSR edges for a specific node, materializing EdgeData on the fly.
/// When `conn_type_filter` is set, edges with non-matching connection types
/// are skipped before materialization (avoids arena alloc + property lookup).
pub struct DiskEdges<'a> {
    graph: &'a DiskGraph,
    direction: Direction,
    source_node: NodeIndex,
    // Pre-collected CSR edges (avoids lifetime issues with MmapOrVec)
    csr_edges: Vec<CsrEdge>,
    csr_pos: usize,
    // Overflow edges (appended after CSR construction)
    overflow: Option<&'a [CsrEdge]>,
    overflow_pos: usize,
    // Optional connection type pre-filter (u64 for O(1) comparison)
    conn_type_filter: Option<u64>,
}

impl<'a> DiskEdges<'a> {
    pub fn new(
        graph: &'a DiskGraph,
        direction: Direction,
        source_node: NodeIndex,
        edges: &MmapOrVec<CsrEdge>,
        start: usize,
        end: usize,
        overflow: Option<&'a Vec<CsrEdge>>,
    ) -> Self {
        // Pre-collect CSR edges into a Vec to avoid borrow issues
        let mut csr_edges = Vec::with_capacity(end - start);
        for i in start..end {
            csr_edges.push(edges.get(i));
        }
        DiskEdges {
            graph,
            direction,
            source_node,
            csr_edges,
            csr_pos: 0,
            overflow: overflow.map(|v| v.as_slice()),
            overflow_pos: 0,
            conn_type_filter: None,
        }
    }

    pub fn new_empty(graph: &'a DiskGraph, direction: Direction, source_node: NodeIndex) -> Self {
        DiskEdges {
            graph,
            direction,
            source_node,
            csr_edges: Vec::new(),
            csr_pos: 0,
            overflow: None,
            overflow_pos: 0,
            conn_type_filter: None,
        }
    }

    /// Set a connection type pre-filter. Edges whose connection type doesn't
    /// match are skipped before materialization.
    pub fn with_conn_type_filter(mut self, ct: u64) -> Self {
        self.conn_type_filter = Some(ct);
        self
    }

    #[inline]
    fn make_edge_ref(&self, csr: &CsrEdge) -> GraphEdgeRef<'a> {
        let weight = self.graph.materialize_edge(csr.edge_idx);
        let (src, tgt) = match self.direction {
            Direction::Outgoing => (self.source_node, NodeIndex::new(csr.peer as usize)),
            Direction::Incoming => (NodeIndex::new(csr.peer as usize), self.source_node),
        };
        GraphEdgeRef::new(src, tgt, EdgeIndex::new(csr.edge_idx as usize), weight)
    }
}

impl<'a> Iterator for DiskEdges<'a> {
    type Item = GraphEdgeRef<'a>;

    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        // First: iterate CSR edges
        while self.csr_pos < self.csr_edges.len() {
            let e = self.csr_edges[self.csr_pos];
            self.csr_pos += 1;
            if e.edge_idx != TOMBSTONE_EDGE {
                // Pre-filter by connection type before materializing
                if let Some(ct) = self.conn_type_filter {
                    if self
                        .graph
                        .edge_endpoints
                        .get(e.edge_idx as usize)
                        .connection_type
                        != ct
                    {
                        continue;
                    }
                }
                return Some(self.make_edge_ref(&e));
            }
        }
        // Then: iterate overflow edges
        if let Some(overflow) = self.overflow {
            while self.overflow_pos < overflow.len() {
                let e = &overflow[self.overflow_pos];
                self.overflow_pos += 1;
                if e.edge_idx != TOMBSTONE_EDGE {
                    if let Some(ct) = self.conn_type_filter {
                        if self
                            .graph
                            .edge_endpoints
                            .get(e.edge_idx as usize)
                            .connection_type
                            != ct
                        {
                            continue;
                        }
                    }
                    return Some(self.make_edge_ref(e));
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_csr = self.csr_edges.len().saturating_sub(self.csr_pos);
        let remaining_overflow = self
            .overflow
            .map(|o| o.len().saturating_sub(self.overflow_pos))
            .unwrap_or(0);
        (0, Some(remaining_csr + remaining_overflow))
    }
}

impl<'a> Iterator for GraphEdges<'a> {
    type Item = GraphEdgeRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        match self {
            GraphEdges::InMemory(iter) => iter
                .next()
                .map(|er| GraphEdgeRef::new(er.source(), er.target(), er.id(), er.weight())),
            GraphEdges::Disk(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            GraphEdges::InMemory(iter) => iter.size_hint(),
            GraphEdges::Disk(iter) => iter.size_hint(),
        }
    }
}

// ============================================================================
// GraphEdgeReferences — all edges in the graph
// ============================================================================

pub enum GraphEdgeReferences<'a> {
    InMemory(petgraph::stable_graph::EdgeReferences<'a, EdgeData, u32>),
    Disk(DiskEdgeReferences<'a>),
}

/// Iterates all edges in the DiskGraph by scanning edge_endpoints.
pub struct DiskEdgeReferences<'a> {
    graph: &'a DiskGraph,
    pos: u32,
    total: u32,
}

impl<'a> DiskEdgeReferences<'a> {
    pub fn new(graph: &'a DiskGraph) -> Self {
        DiskEdgeReferences {
            graph,
            pos: 0,
            total: graph.next_edge_idx,
        }
    }
}

impl<'a> Iterator for DiskEdgeReferences<'a> {
    type Item = GraphEdgeRef<'a>;
    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        while self.pos < self.total {
            let i = self.pos;
            self.pos += 1;
            let ep = self.graph.edge_endpoints.get(i as usize);
            if ep.source != TOMBSTONE_EDGE {
                let weight = self.graph.materialize_edge(i);
                return Some(GraphEdgeRef::new(
                    NodeIndex::new(ep.source as usize),
                    NodeIndex::new(ep.target as usize),
                    EdgeIndex::new(i as usize),
                    weight,
                ));
            }
        }
        None
    }
}

impl<'a> Iterator for GraphEdgeReferences<'a> {
    type Item = GraphEdgeRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        match self {
            GraphEdgeReferences::InMemory(iter) => iter
                .next()
                .map(|er| GraphEdgeRef::new(er.source(), er.target(), er.id(), er.weight())),
            GraphEdgeReferences::Disk(iter) => iter.next(),
        }
    }
}

// ============================================================================
// GraphEdgeIndices — edge index iteration
// ============================================================================

pub enum GraphEdgeIndices<'a> {
    InMemory(petgraph::stable_graph::EdgeIndices<'a, EdgeData, u32>),
    Disk(DiskEdgeIndices<'a>),
}

/// Iterates valid edge indices by scanning edge_endpoints for non-tombstones.
pub struct DiskEdgeIndices<'a> {
    endpoints: &'a MmapOrVec<crate::graph::disk_graph::EdgeEndpoints>,
    pos: u32,
    total: u32,
}

impl<'a> DiskEdgeIndices<'a> {
    pub fn new(
        total: u32,
        endpoints: &'a MmapOrVec<crate::graph::disk_graph::EdgeEndpoints>,
    ) -> Self {
        DiskEdgeIndices {
            endpoints,
            pos: 0,
            total,
        }
    }
}

impl<'a> Iterator for DiskEdgeIndices<'a> {
    type Item = EdgeIndex;
    fn next(&mut self) -> Option<EdgeIndex> {
        while self.pos < self.total {
            let i = self.pos;
            self.pos += 1;
            let ep = self.endpoints.get(i as usize);
            if ep.source != TOMBSTONE_EDGE {
                return Some(EdgeIndex::new(i as usize));
            }
        }
        None
    }
}

impl<'a> Iterator for GraphEdgeIndices<'a> {
    type Item = EdgeIndex;

    #[inline]
    fn next(&mut self) -> Option<EdgeIndex> {
        match self {
            GraphEdgeIndices::InMemory(iter) => iter.next(),
            GraphEdgeIndices::Disk(iter) => iter.next(),
        }
    }
}

// ============================================================================
// GraphEdgesConnecting — edges between two specific nodes
// ============================================================================

pub enum GraphEdgesConnecting<'a> {
    InMemory(petgraph::stable_graph::EdgesConnecting<'a, EdgeData, petgraph::Directed, u32>),
    Disk(DiskEdgesConnecting<'a>),
}

/// Iterates edges connecting two specific nodes via the outgoing CSR + overflow.
pub struct DiskEdgesConnecting<'a> {
    inner: DiskEdges<'a>,
    target: u32,
}

impl<'a> DiskEdgesConnecting<'a> {
    pub fn new(graph: &'a DiskGraph, a: NodeIndex, b: NodeIndex) -> Self {
        let inner = graph.edges_directed_iter(a, Direction::Outgoing);
        DiskEdgesConnecting {
            inner,
            target: b.index() as u32,
        }
    }
}

impl<'a> Iterator for DiskEdgesConnecting<'a> {
    type Item = GraphEdgeRef<'a>;
    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        self.inner
            .by_ref()
            .find(|edge| edge.target().index() as u32 == self.target)
    }
}

impl<'a> Iterator for GraphEdgesConnecting<'a> {
    type Item = GraphEdgeRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<GraphEdgeRef<'a>> {
        match self {
            GraphEdgesConnecting::InMemory(iter) => iter
                .next()
                .map(|er| GraphEdgeRef::new(er.source(), er.target(), er.id(), er.weight())),
            GraphEdgesConnecting::Disk(iter) => iter.next(),
        }
    }
}

// ============================================================================
// GraphNeighbors — neighbor node iteration
// ============================================================================

pub enum GraphNeighbors<'a> {
    InMemory(petgraph::stable_graph::Neighbors<'a, EdgeData, u32>),
    Disk(DiskNeighbors),
}

/// Iterates neighbor nodes. Uses pre-collected Vec of NodeIndex.
pub struct DiskNeighbors {
    peers: Vec<NodeIndex>,
    pos: usize,
}

impl DiskNeighbors {
    pub fn new(
        edges: &MmapOrVec<CsrEdge>,
        start: usize,
        end: usize,
        overflow: Option<&Vec<CsrEdge>>,
    ) -> Self {
        let mut peers = Vec::with_capacity(end - start);
        for i in start..end {
            let e = edges.get(i);
            if e.edge_idx != TOMBSTONE_EDGE {
                peers.push(NodeIndex::new(e.peer as usize));
            }
        }
        if let Some(list) = overflow {
            for e in list {
                if e.edge_idx != TOMBSTONE_EDGE {
                    peers.push(NodeIndex::new(e.peer as usize));
                }
            }
        }
        DiskNeighbors { peers, pos: 0 }
    }

    pub fn new_empty() -> Self {
        DiskNeighbors {
            peers: Vec::new(),
            pos: 0,
        }
    }

    pub fn from_collected(peers: Vec<NodeIndex>) -> Self {
        DiskNeighbors { peers, pos: 0 }
    }
}

impl Iterator for DiskNeighbors {
    type Item = NodeIndex;
    fn next(&mut self) -> Option<NodeIndex> {
        if self.pos < self.peers.len() {
            let idx = self.peers[self.pos];
            self.pos += 1;
            Some(idx)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.peers.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> Iterator for GraphNeighbors<'a> {
    type Item = NodeIndex;

    #[inline]
    fn next(&mut self) -> Option<NodeIndex> {
        match self {
            GraphNeighbors::InMemory(iter) => iter.next(),
            GraphNeighbors::Disk(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            GraphNeighbors::InMemory(iter) => iter.size_hint(),
            GraphNeighbors::Disk(iter) => iter.size_hint(),
        }
    }
}
