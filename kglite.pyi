"""Type stubs for kglite — a high-performance knowledge graph library."""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd

__version__: str

def load(path: str) -> KnowledgeGraph:
    """Load a graph from a binary file previously saved with ``save()``.

    Args:
        path: Path to the ``.kgl`` file.

    Returns:
        A new KnowledgeGraph with the loaded data.
    """
    ...

class KnowledgeGraph:
    """A high-performance knowledge graph with typed nodes, connections, and
    a fluent query API backed by Rust.
    """

    # ====================================================================
    # Constructor
    # ====================================================================

    def __init__(self) -> None:
        """Create an empty KnowledgeGraph."""
        ...

    # ====================================================================
    # Properties
    # ====================================================================

    @property
    def node_types(self) -> list[str]:
        """List of node type names present in the graph."""
        ...

    @property
    def last_mutation_stats(self) -> Optional[dict[str, int]]:
        """Mutation statistics from the last Cypher mutation query.

        Returns ``None`` if no mutation has been executed yet.
        Keys: ``nodes_created``, ``relationships_created``, ``properties_set``,
        ``nodes_deleted``, ``relationships_deleted``, ``properties_removed``.
        """
        ...

    # ====================================================================
    # Data Loading
    # ====================================================================

    def add_nodes(
        self,
        data: pd.DataFrame,
        node_type: str,
        unique_id_field: str,
        node_title_field: Optional[str] = None,
        columns: Optional[list[str]] = None,
        conflict_handling: Optional[str] = None,
        skip_columns: Optional[list[str]] = None,
        column_types: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Add nodes from a DataFrame.

        String and integer IDs are auto-detected from the DataFrame dtype.
        Non-contiguous DataFrame indexes (e.g. from filtering) are handled
        automatically.

        Args:
            data: DataFrame containing node data.
            node_type: Label for this set of nodes (e.g. ``'Person'``).
            unique_id_field: Column used as unique identifier.
            node_title_field: Column used as display title. Defaults to ``unique_id_field``.
            columns: Whitelist of columns to include. ``None`` = all.
            conflict_handling: ``'update'`` (default), ``'replace'``, ``'skip'``, or ``'preserve'``.
            skip_columns: Columns to exclude.
            column_types: Override column dtypes ``{'col': 'string'|'integer'|'float'|'datetime'|'uniqueid'}``.

        Returns:
            Operation report dict with keys ``nodes_created``,
            ``nodes_updated``, ``nodes_skipped``, ``processing_time_ms``,
            ``has_errors``, and optionally ``errors`` with skip reasons.
        """
        ...

    def add_connections(
        self,
        data: pd.DataFrame,
        connection_type: str,
        source_type: str,
        source_id_field: str,
        target_type: str,
        target_id_field: str,
        source_title_field: Optional[str] = None,
        target_title_field: Optional[str] = None,
        columns: Optional[list[str]] = None,
        skip_columns: Optional[list[str]] = None,
        conflict_handling: Optional[str] = None,
        column_types: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Add connections (edges) between existing nodes.

        Args:
            data: DataFrame containing edge data.
            connection_type: Label for this edge type (e.g. ``'KNOWS'``).
            source_type: Node type of source nodes.
            source_id_field: Column with source node IDs.
            target_type: Node type of target nodes.
            target_id_field: Column with target node IDs.
            source_title_field: Optional title column for source nodes.
            target_title_field: Optional title column for target nodes.
            columns: Whitelist of property columns to include.
            skip_columns: Columns to exclude.
            conflict_handling: ``'update'``, ``'skip'``, or ``None``.
            column_types: Override column dtypes.

        Returns:
            Operation report dict with ``connections_created``, ``connections_skipped``, etc.
        """
        ...

    def add_nodes_bulk(self, nodes: list[dict[str, Any]]) -> dict[str, int]:
        """Add multiple node types at once.

        Each dict in *nodes* must contain ``node_type``, ``unique_id_field``,
        ``node_title_field``, and ``data`` (a DataFrame).

        Returns:
            Mapping of ``node_type`` to count of nodes added.
        """
        ...

    def add_connections_bulk(self, connections: list[dict[str, Any]]) -> dict[str, int]:
        """Add multiple connection types at once.

        Each dict must contain ``source_type``, ``target_type``,
        ``connection_name``, and ``data`` (DataFrame with ``source_id``/``target_id`` columns).

        Returns:
            Mapping of ``connection_name`` to count of connections created.
        """
        ...

    def add_connections_from_source(self, connections: list[dict[str, Any]]) -> dict[str, int]:
        """Add connections, auto-filtering to types already loaded in the graph.

        Same spec format as :meth:`add_connections_bulk`, but silently skips
        connection specs whose source or target type is not in the graph.

        Returns:
            Mapping of ``connection_name`` to count of connections created.
        """
        ...

    # ====================================================================
    # Selection & Filtering
    # ====================================================================

    def type_filter(
        self,
        node_type: str,
        sort: Optional[Union[str, list[tuple[str, bool]]]] = None,
        max_nodes: Optional[int] = None,
    ) -> KnowledgeGraph:
        """Select all nodes of a given type.

        Args:
            node_type: The node type to select (e.g. ``'Person'``).
            sort: Optional sort spec — a property name or list of ``(field, ascending)`` tuples.
            max_nodes: Limit the number of selected nodes.

        Returns:
            A new KnowledgeGraph with the filtered selection.
        """
        ...

    def filter(
        self,
        conditions: dict[str, Any],
        sort: Optional[Union[str, list[tuple[str, bool]]]] = None,
        max_nodes: Optional[int] = None,
    ) -> KnowledgeGraph:
        """Filter the current selection by property conditions.

        Conditions support exact match, comparison operators
        (``'>'``, ``'<'``, ``'>='``, ``'<='``), ``'in'``, ``'is_null'``,
        ``'is_not_null'``, ``'contains'``, ``'starts_with'``, ``'ends_with'``.

        Example::

            graph.type_filter('Person').filter({
                'age': {'>': 25},
                'city': 'Oslo',
            })

        Returns:
            A new KnowledgeGraph with the filtered selection.
        """
        ...

    def filter_orphans(
        self,
        include_orphans: Optional[bool] = None,
        sort: Optional[Union[str, list[tuple[str, bool]]]] = None,
        max_nodes: Optional[int] = None,
    ) -> KnowledgeGraph:
        """Filter nodes based on whether they have connections.

        Args:
            include_orphans: If ``True``, keep only orphan (disconnected) nodes.
                If ``False``, keep only connected nodes. Default ``True``.
            sort: Optional sort spec.
            max_nodes: Limit the number of selected nodes.

        Returns:
            A new KnowledgeGraph with the filtered selection.
        """
        ...

    def sort(
        self,
        sort: Union[str, list[tuple[str, bool]]],
        ascending: Optional[bool] = None,
    ) -> KnowledgeGraph:
        """Sort the current selection.

        Args:
            sort: Property name (string) or list of ``(field, ascending)`` tuples.
            ascending: Direction when *sort* is a single string. Default ``True``.

        Returns:
            A new KnowledgeGraph with the sorted selection.
        """
        ...

    def max_nodes(self, max_per_group: int) -> KnowledgeGraph:
        """Limit the number of nodes per parent group.

        Args:
            max_per_group: Maximum number of nodes to keep per group.

        Returns:
            A new KnowledgeGraph with the limited selection.
        """
        ...

    def valid_at(
        self,
        date: str,
        date_from_field: Optional[str] = None,
        date_to_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes valid at a specific date.

        Keeps nodes where ``date_from <= date <= date_to``.

        Args:
            date: Date string (e.g. ``'2024-01-15'``).
            date_from_field: Name of the start-date property. Default ``'date_from'``.
            date_to_field: Name of the end-date property. Default ``'date_to'``.

        Returns:
            A new KnowledgeGraph with the filtered selection.
        """
        ...

    def valid_during(
        self,
        start_date: str,
        end_date: str,
        date_from_field: Optional[str] = None,
        date_to_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes whose validity period overlaps a date range.

        Args:
            start_date: Start of the query range.
            end_date: End of the query range.
            date_from_field: Name of the start-date property. Default ``'date_from'``.
            date_to_field: Name of the end-date property. Default ``'date_to'``.

        Returns:
            A new KnowledgeGraph with the filtered selection.
        """
        ...

    # ====================================================================
    # Update
    # ====================================================================

    def update(
        self,
        properties: dict[str, Any],
        keep_selection: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Batch-update properties on all selected nodes.

        Args:
            properties: Mapping of property names to new values.
            keep_selection: Preserve the current selection in the returned graph. Default ``False``.

        Returns:
            Dict with ``graph`` (updated KnowledgeGraph), ``nodes_updated`` (int),
            and ``report_index`` (int).
        """
        ...

    # ====================================================================
    # Data Retrieval
    # ====================================================================

    def get_nodes(
        self,
        max_nodes: Optional[int] = None,
        indices: Optional[list[int]] = None,
        parent_type: Optional[str] = None,
        parent_info: Optional[bool] = None,
        flatten_single_parent: bool = True,
    ) -> list[dict[str, Any]]:
        """Materialise selected nodes as a list of property dicts.

        Args:
            max_nodes: Maximum number of nodes to return.
            indices: Specific node indices to return.
            parent_type: Group results by parent of this type.
            parent_info: Include parent info in grouped output.
            flatten_single_parent: Flatten when there is only one parent group. Default ``True``.

        Returns:
            List of node dicts, each containing ``id``, ``title``, ``type``,
            and all stored properties.
        """
        ...

    def to_df(
        self,
        *,
        include_type: bool = True,
        include_id: bool = True,
    ) -> pd.DataFrame:
        """Export current selection as a pandas DataFrame.

        Each node becomes a row with columns for title, type, id, and all
        properties. Missing properties across different node types become None.

        Args:
            include_type: Include ``type`` column. Default ``True``.
            include_id: Include ``id`` column. Default ``True``.

        Returns:
            DataFrame with one row per selected node.
        """
        ...

    def node_count(self) -> int:
        """Count selected nodes without materialising them.

        Much faster than ``len(get_nodes())``.
        """
        ...

    def indices(self) -> list[int]:
        """Return raw graph indices for selected nodes."""
        ...

    def get_ids(self) -> list[dict[str, Any]]:
        """Return only ``id``, ``title``, and ``type`` for each selected node.

        Faster than :meth:`get_nodes` when you only need identification info.
        """
        ...

    def id_values(self) -> list[Any]:
        """Return a flat list of ID values from the current selection.

        The lightest retrieval method — no dict wrapping.
        """
        ...

    def get_node_by_id(self, node_type: str, node_id: Any) -> Optional[dict[str, Any]]:
        """Look up a single node by type and ID. O(1) via hash index.

        Args:
            node_type: The node type (e.g. ``'User'``).
            node_id: The unique ID value.

        Returns:
            Node property dict, or ``None`` if not found.
        """
        ...

    def build_id_indices(self, node_types: Optional[list[str]] = None) -> None:
        """Pre-build ID lookup indices for fast :meth:`get_node_by_id` calls.

        Args:
            node_types: Types to index. ``None`` indexes all types.
        """
        ...

    def node_type_counts(self) -> dict[str, int]:
        """Get node counts per type without materialising nodes.

        Returns:
            Dict mapping node type name to count.
        """
        ...

    # Graph Maintenance
    # -----------------

    def reindex(self) -> None:
        """Rebuild all indexes from the current graph state.

        Reconstructs type_indices, property_indices, and composite_indices by
        scanning all live nodes. Clears lazy caches (id_indices, connection_types)
        so they rebuild on next access.

        Use after bulk mutations (especially Cypher DELETE/REMOVE) to ensure
        index consistency.

        Example::

            graph.reindex()
        """
        ...

    def vacuum(self) -> dict[str, int]:
        """Compact the graph by removing tombstones left by node/edge deletions.

        With StableDiGraph, deletions leave holes in the internal storage.
        Over time, this wastes memory and degrades iteration performance.
        ``vacuum()`` rebuilds the graph with contiguous indices, then rebuilds
        all indexes.

        **Important**: This resets the current selection since node indices change.
        Call this between query chains, not in the middle of one.

        Returns:
            dict with keys:
                - ``nodes_remapped``: Number of nodes that were remapped
                - ``tombstones_removed``: Number of tombstone slots reclaimed

        Example::

            info = graph.graph_info()
            if info['fragmentation_ratio'] > 0.3:
                result = graph.vacuum()
                print(f"Reclaimed {result['tombstones_removed']} slots")
        """
        ...

    def graph_info(self) -> dict[str, Any]:
        """Get diagnostic information about graph storage health.

        Returns a dictionary with storage metrics useful for deciding when
        to call :meth:`vacuum` or :meth:`reindex`.

        Returns:
            dict with keys:
                - ``node_count``: Number of live nodes
                - ``node_capacity``: Upper bound of node indices (includes tombstones)
                - ``node_tombstones``: Number of wasted slots from deletions
                - ``edge_count``: Number of live edges
                - ``fragmentation_ratio``: Ratio of wasted storage (0.0 = clean)
                - ``type_count``: Number of distinct node types
                - ``property_index_count``: Number of single-property indexes
                - ``composite_index_count``: Number of composite indexes

        Example::

            info = graph.graph_info()
            if info['fragmentation_ratio'] > 0.3:
                graph.vacuum()
        """
        ...

    def get_connections(
        self,
        indices: Optional[list[int]] = None,
        parent_info: Optional[bool] = None,
        include_node_properties: Optional[bool] = None,
        flatten_single_parent: bool = True,
    ) -> dict[str, Any]:
        """Get connections for selected nodes.

        Args:
            indices: Specific node indices to query.
            parent_info: Include parent info in output.
            include_node_properties: Include properties of connected nodes. Default ``True``.
            flatten_single_parent: Flatten when only one parent. Default ``True``.

        Returns:
            Nested dict ``{title: {node_id, type, incoming, outgoing}}``.
        """
        ...

    def get_titles(
        self,
        max_nodes: Optional[int] = None,
        indices: Optional[list[int]] = None,
        flatten_single_parent: Optional[bool] = None,
    ) -> Union[list[str], dict[str, list[str]]]:
        """Get titles of selected nodes.

        Without traversal (single parent group), returns a flat list of titles.
        After traversal (multiple parent groups), returns ``{parent_title: [titles]}``.

        Args:
            flatten_single_parent: Flatten single-group results to a list. Default ``True``.

        Returns:
            ``list[str]`` when flattened, ``dict[str, list[str]]`` when grouped.
        """
        ...

    def explain(self) -> str:
        """Return a human-readable execution plan for the current query chain.

        Example output::

            TYPE_FILTER Person (500 nodes) -> FILTER (42 nodes)
        """
        ...

    def get_properties(
        self,
        properties: list[str],
        max_nodes: Optional[int] = None,
        indices: Optional[list[int]] = None,
        flatten_single_parent: Optional[bool] = None,
    ) -> Union[list[tuple[Any, ...]], dict[str, list[tuple[Any, ...]]]]:
        """Get specific properties for selected nodes.

        Without traversal (single parent group), returns a flat list of tuples.
        After traversal (multiple parent groups), returns ``{parent_title: [tuples]}``.

        Args:
            properties: List of property names to retrieve.
            max_nodes: Maximum number of nodes.
            indices: Specific node indices.
            flatten_single_parent: Flatten single-group results to a list. Default ``True``.

        Returns:
            ``list[tuple]`` when flattened, ``dict[str, list[tuple]]`` when grouped.
        """
        ...

    def unique_values(
        self,
        property: str,
        group_by_parent: Optional[bool] = None,
        level_index: Optional[int] = None,
        indices: Optional[list[int]] = None,
        store_as: Optional[str] = None,
        max_length: Optional[int] = None,
        keep_selection: Optional[bool] = None,
    ) -> Any:
        """Get unique values of a property, optionally storing results.

        Args:
            property: Property name to extract unique values from.
            group_by_parent: Group by parent node. Default ``True``.
            level_index: Target level in the selection hierarchy.
            indices: Specific node indices.
            store_as: If set, stores comma-separated unique values as this property on parents.
            max_length: Max string length when storing.
            keep_selection: Preserve selection after store. Default ``False``.

        Returns:
            Dict of unique values per parent, or a KnowledgeGraph if ``store_as`` is set.
        """
        ...

    # ====================================================================
    # Traversal
    # ====================================================================

    def traverse(
        self,
        connection_type: str,
        level_index: Optional[int] = None,
        direction: Optional[str] = None,
        filter_target: Optional[dict[str, Any]] = None,
        filter_connection: Optional[dict[str, Any]] = None,
        sort_target: Optional[Union[str, list[tuple[str, bool]]]] = None,
        max_nodes: Optional[int] = None,
        new_level: Optional[bool] = None,
    ) -> KnowledgeGraph:
        """Traverse connections from the current selection.

        Args:
            connection_type: Edge type to follow (e.g. ``'KNOWS'``).
            level_index: Source level in the hierarchy.
            direction: ``'outgoing'`` (default), ``'incoming'``, or ``'both'``.
            filter_target: Filter conditions for target nodes.
            filter_connection: Filter conditions for edge properties.
            sort_target: Sort target nodes.
            max_nodes: Limit target nodes per source.
            new_level: Add targets as a new hierarchy level. Default ``True``.

        Returns:
            A new KnowledgeGraph with traversed nodes selected.
        """
        ...

    def selection_to_new_connections(
        self,
        connection_type: str,
        keep_selection: Optional[bool] = None,
        conflict_handling: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Create new connections from the current parent-child selection.

        Args:
            connection_type: Label for the new edges.
            keep_selection: Preserve selection. Default ``False``.
            conflict_handling: ``'update'``, ``'skip'``, or ``None``.

        Returns:
            A new KnowledgeGraph with the connections added.
        """
        ...

    def children_properties_to_list(
        self,
        property: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        sort: Optional[Union[str, list[tuple[str, bool]]]] = None,
        max_nodes: Optional[int] = None,
        store_as: Optional[str] = None,
        max_length: Optional[int] = None,
        keep_selection: Optional[bool] = None,
    ) -> Any:
        """Collect child-node property values into comma-separated lists.

        Args:
            property: Child property to collect. Default ``'title'``.
            filter: Filter conditions for children.
            sort: Sort children.
            max_nodes: Limit children per parent.
            store_as: If set, stores the list as this property on parent nodes.
            max_length: Max string length when storing.
            keep_selection: Preserve selection after store. Default ``False``.

        Returns:
            Dict of ``{parent_title: 'val1, val2, ...'}`` or a KnowledgeGraph
            if ``store_as`` is set.
        """
        ...

    # ====================================================================
    # Statistics & Calculations
    # ====================================================================

    def statistics(self, property: str, level_index: Optional[int] = None) -> Any:
        """Compute descriptive statistics for a numeric property.

        Returns per-parent stats including count, mean, std, min, max, sum.

        Args:
            property: Numeric property name.
            level_index: Target level in the hierarchy.
        """
        ...

    def calculate(
        self,
        expression: str,
        level_index: Optional[int] = None,
        store_as: Optional[str] = None,
        keep_selection: Optional[bool] = None,
        aggregate_connections: Optional[bool] = None,
    ) -> Any:
        """Evaluate a mathematical expression on selected nodes.

        Supports property references, arithmetic operators, and aggregate
        functions (``sum``, ``mean``, ``std``, ``min``, ``max``, ``count``).

        Args:
            expression: Expression string, e.g. ``'price * quantity'`` or ``'mean(age)'``.
            level_index: Target level in the hierarchy.
            store_as: If set, stores results as this property on nodes.
            keep_selection: Preserve selection after store. Default ``False``.
            aggregate_connections: Aggregate over connected nodes.

        Returns:
            Computation results, or a KnowledgeGraph if ``store_as`` is set.
        """
        ...

    def count(
        self,
        level_index: Optional[int] = None,
        group_by_parent: Optional[bool] = None,
        store_as: Optional[str] = None,
        keep_selection: Optional[bool] = None,
    ) -> Any:
        """Count nodes, optionally grouped by parent.

        Args:
            level_index: Target level in the hierarchy.
            group_by_parent: Group counts by parent node.
            store_as: Store count as a property on parent nodes.
            keep_selection: Preserve selection after store. Default ``False``.

        Returns:
            An integer count, grouped counts, or a KnowledgeGraph if ``store_as`` is set.
        """
        ...

    # ====================================================================
    # Debugging & Introspection
    # ====================================================================

    def get_schema(self) -> str:
        """Return a text summary of the graph schema (node types, connections)."""
        ...

    def get_selection(self) -> str:
        """Return a text summary of the current selection state."""
        ...

    def clear(self) -> None:
        """Clear the current selection (resets to empty)."""
        ...

    # ====================================================================
    # Schema Introspection
    # ====================================================================

    def schema(self) -> dict[str, Any]:
        """Return a full schema overview of the graph.

        Returns:
            Dict with keys:
                - ``node_types``: ``{type_name: {count, properties: {name: type_str}}}``
                - ``connection_types``: ``{conn_name: {count, source_types: list, target_types: list}}``
                - ``indexes``: list of ``"Type.property"`` strings
                - ``node_count``: total nodes
                - ``edge_count``: total edges

        Note:
            Scans all edges once (O(m)) to compute accurate connection type stats.
        """
        ...

    def connection_types(self) -> list[dict[str, Any]]:
        """Return all connection types with counts and endpoint type sets.

        Returns:
            List of dicts with ``type``, ``count``, ``source_types``, ``target_types``.
        """
        ...

    def properties(self, node_type: str) -> dict[str, dict[str, Any]]:
        """Return property statistics for a node type.

        Args:
            node_type: The node type to inspect.

        Returns:
            Dict mapping property name to stats dict with keys:
                - ``type``: type string (e.g. ``'str'``, ``'int'``, ``'float'``)
                - ``non_null``: count of non-null values
                - ``unique``: count of distinct values
                - ``values``: (optional) list of values when unique count <= 20

        Raises:
            KeyError: If node_type does not exist.
        """
        ...

    def neighbors_schema(self, node_type: str) -> dict[str, list[dict[str, Any]]]:
        """Return connection topology for a node type.

        Args:
            node_type: The node type to inspect.

        Returns:
            Dict with:
                - ``outgoing``: list of ``{connection_type, target_type, count}``
                - ``incoming``: list of ``{connection_type, source_type, count}``

        Raises:
            KeyError: If node_type does not exist.
        """
        ...

    def sample(self, node_type: str, n: int = 5) -> list[dict[str, Any]]:
        """Return a quick sample of nodes for a given type.

        Args:
            node_type: The node type to sample.
            n: Number of nodes to return. Default ``5``.

        Returns:
            List of node dicts (same format as :meth:`get_nodes`).

        Raises:
            KeyError: If node_type does not exist.
        """
        ...

    def indexes(self) -> list[dict[str, Any]]:
        """Return a unified list of all indexes.

        Returns:
            List of dicts, each with:
                - ``node_type``: the indexed node type
                - ``property``: property name (for equality indexes)
                - ``properties``: list of property names (for composite indexes)
                - ``type``: ``'equality'`` or ``'composite'``
        """
        ...

    # ====================================================================
    # Persistence
    # ====================================================================

    def save(self, path: str) -> None:
        """Serialise the graph to a binary file.

        Load it back with :func:`kglite.load`.

        Args:
            path: Output file path (typically ``*.kgl``).
        """
        ...

    # ====================================================================
    # Operation Reports
    # ====================================================================

    def get_last_report(self) -> dict[str, Any]:
        """Get the most recent operation report as a dict.

        Returns an empty dict if no operations have been performed.
        """
        ...

    def get_operation_index(self) -> int:
        """Get the sequential index of the last operation."""
        ...

    def get_report_history(self) -> list[dict[str, Any]]:
        """Get all operation reports as a list of dicts."""
        ...

    # ====================================================================
    # Set Operations
    # ====================================================================

    def union(self, other: KnowledgeGraph) -> KnowledgeGraph:
        """Combine selections from both graphs (set union).

        Returns:
            A new KnowledgeGraph with nodes from either selection.
        """
        ...

    def intersection(self, other: KnowledgeGraph) -> KnowledgeGraph:
        """Keep only nodes present in both selections (set intersection).

        Returns:
            A new KnowledgeGraph with only shared nodes.
        """
        ...

    def difference(self, other: KnowledgeGraph) -> KnowledgeGraph:
        """Keep nodes in ``self`` but not in ``other`` (set difference).

        Returns:
            A new KnowledgeGraph with the difference.
        """
        ...

    def symmetric_difference(self, other: KnowledgeGraph) -> KnowledgeGraph:
        """Keep nodes in exactly one of the selections (symmetric difference).

        Returns:
            A new KnowledgeGraph with nodes exclusive to each side.
        """
        ...

    # ====================================================================
    # Schema Definition & Validation
    # ====================================================================

    def define_schema(self, schema_dict: dict[str, Any]) -> KnowledgeGraph:
        """Define the expected schema for the graph.

        Args:
            schema_dict: Schema definition with ``nodes`` and ``connections`` keys.
                See the Rust docstring for full structure.

        Returns:
            Self with schema defined.
        """
        ...

    def validate_schema(self, strict: Optional[bool] = None) -> list[dict[str, Any]]:
        """Validate the graph against the defined schema.

        Args:
            strict: Report undefined types in the graph. Default ``False``.

        Returns:
            List of validation error dicts. Empty list means valid.
        """
        ...

    def has_schema(self) -> bool:
        """Check if a schema has been defined."""
        ...

    def clear_schema(self) -> KnowledgeGraph:
        """Remove the schema definition from the graph."""
        ...

    def get_schema_definition(self) -> Optional[dict[str, Any]]:
        """Get the current schema definition as a dict, or ``None``."""
        ...

    # ====================================================================
    # Graph Algorithms — Path Finding & Connectivity
    # ====================================================================

    def shortest_path(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
        connection_types: Optional[list[str]] = None,
        via_types: Optional[list[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Find the shortest path between two nodes.

        Args:
            source_type: Source node type.
            source_id: Source node ID.
            target_type: Target node type.
            target_id: Target node ID.
            connection_types: Only traverse edges of these types. Default all.
            via_types: Only traverse through nodes of these types. Default all.
            timeout_ms: Abort after this many milliseconds and return ``None``.

        Returns:
            Dict with ``path`` (list of node info dicts), ``connections``
            (list of edge types), and ``length`` (hop count).
            ``None`` if no path exists or timeout is reached.
        """
        ...

    def shortest_path_length(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
    ) -> Optional[int]:
        """Get just the hop count of the shortest path.

        Faster than :meth:`shortest_path` when you only need the distance.
        Does not support ``connection_types`` or ``via_types`` filtering.

        Returns:
            Number of hops, or ``None`` if no path exists.
        """
        ...

    def shortest_path_ids(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
        connection_types: Optional[list[str]] = None,
        via_types: Optional[list[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> Optional[list[Any]]:
        """Get node IDs along the shortest path.

        Args:
            connection_types: Only traverse edges of these types. Default all.
            via_types: Only traverse through nodes of these types. Default all.
            timeout_ms: Abort after this many milliseconds and return ``None``.

        Returns:
            List of node IDs, or ``None`` if no path exists or timeout is reached.
        """
        ...

    def shortest_path_indices(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
        connection_types: Optional[list[str]] = None,
        via_types: Optional[list[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> Optional[list[int]]:
        """Get raw graph indices along the shortest path.

        Fastest path query — no node data lookup.

        Args:
            connection_types: Only traverse edges of these types. Default all.
            via_types: Only traverse through nodes of these types. Default all.
            timeout_ms: Abort after this many milliseconds and return ``None``.

        Returns:
            List of integer indices, or ``None`` if no path exists or timeout is reached.
        """
        ...

    def all_paths(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
        max_hops: Optional[int] = None,
        max_results: Optional[int] = None,
        connection_types: Optional[list[str]] = None,
        via_types: Optional[list[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Find all paths between two nodes.

        Args:
            source_type: Source node type.
            source_id: Source node ID.
            target_type: Target node type.
            target_id: Target node ID.
            max_hops: Maximum path length. Default ``5``.
            max_results: Stop after finding this many paths. Default unlimited.
                Use to prevent OOM on dense graphs.
            connection_types: Only traverse edges of these types. Default all.
            via_types: Only traverse through nodes of these types. Default all.
            timeout_ms: Abort after this many milliseconds, returning partial results.

        Returns:
            List of path dicts, each with ``path``, ``connections``, ``length``.
        """
        ...

    def connected_components(self, weak: Optional[bool] = None) -> list[list[dict[str, Any]]]:
        """Find connected components in the graph.

        Args:
            weak: If ``True`` (default), find weakly connected components.
                If ``False``, find strongly connected components.

        Returns:
            List of components (largest first), each a list of node info dicts.
        """
        ...

    def are_connected(
        self,
        source_type: str,
        source_id: Any,
        target_type: str,
        target_id: Any,
    ) -> bool:
        """Check if two nodes are connected (directly or indirectly)."""
        ...

    def get_degrees(self) -> dict[str, int]:
        """Get connection count for each selected node.

        Returns:
            ``{node_title: degree}``.
        """
        ...

    # ====================================================================
    # Centrality Algorithms
    # ====================================================================

    def betweenness_centrality(
        self,
        normalized: Optional[bool] = None,
        sample_size: Optional[int] = None,
        top_k: Optional[int] = None,
        as_dict: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        to_df: Optional[bool] = None,
    ) -> Union[list[dict[str, Any]], "pd.DataFrame"]:
        """Calculate betweenness centrality.

        Args:
            normalized: Normalise scores to ``[0, 1]``. Default ``True``.
            sample_size: Sample source nodes for faster computation on large graphs.
            top_k: Return only the top *K* nodes.
            as_dict: Return ``{id: score}`` dict instead of list of dicts.
            timeout_ms: Abort after this many milliseconds, returning partial results.
            to_df: Return a pandas DataFrame with columns ``type``, ``title``, ``id``, ``score``.

        Returns:
            List of dicts with ``type``, ``title``, ``id``, ``score``,
            sorted by score descending. Or a DataFrame if ``to_df=True``.
        """
        ...

    def pagerank(
        self,
        damping_factor: Optional[float] = None,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        top_k: Optional[int] = None,
        as_dict: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        to_df: Optional[bool] = None,
    ) -> Union[list[dict[str, Any]], "pd.DataFrame"]:
        """Calculate PageRank centrality.

        Args:
            damping_factor: Probability of following a link. Default ``0.85``.
            max_iterations: Maximum iterations. Default ``100``.
            tolerance: Convergence threshold. Default ``1e-6``.
            top_k: Return only the top *K* nodes.
            as_dict: Return ``{id: score}`` dict instead of list of dicts.
            timeout_ms: Abort after this many milliseconds, returning partial results.
            to_df: Return a pandas DataFrame with columns ``type``, ``title``, ``id``, ``score``.

        Returns:
            List of dicts with ``type``, ``title``, ``id``, ``score``.
            Or a DataFrame if ``to_df=True``.
        """
        ...

    def degree_centrality(
        self,
        normalized: Optional[bool] = None,
        top_k: Optional[int] = None,
        as_dict: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        to_df: Optional[bool] = None,
    ) -> Union[list[dict[str, Any]], "pd.DataFrame"]:
        """Calculate degree centrality.

        Args:
            normalized: Normalise by ``(n-1)``. Default ``True``.
            top_k: Return only the top *K* nodes.
            as_dict: Return ``{id: score}`` dict instead of list of dicts.
            timeout_ms: Abort after this many milliseconds, returning partial results.
            to_df: Return a pandas DataFrame with columns ``type``, ``title``, ``id``, ``score``.

        Returns:
            List of dicts with ``type``, ``title``, ``id``, ``score``.
            Or a DataFrame if ``to_df=True``.
        """
        ...

    def closeness_centrality(
        self,
        normalized: Optional[bool] = None,
        top_k: Optional[int] = None,
        as_dict: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        to_df: Optional[bool] = None,
    ) -> Union[list[dict[str, Any]], "pd.DataFrame"]:
        """Calculate closeness centrality.

        Args:
            normalized: Adjust for disconnected components. Default ``True``.
            top_k: Return only the top *K* nodes.
            as_dict: Return ``{id: score}`` dict instead of list of dicts.
            timeout_ms: Abort after this many milliseconds, returning partial results.
            to_df: Return a pandas DataFrame with columns ``type``, ``title``, ``id``, ``score``.

        Returns:
            List of dicts with ``type``, ``title``, ``id``, ``score``.
        """
        ...

    # ====================================================================
    # Community Detection
    # ====================================================================

    def louvain_communities(
        self,
        weight_property: Optional[str] = None,
        resolution: Optional[float] = None,
        timeout_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Detect communities using the Louvain algorithm.

        Args:
            weight_property: Edge property to use as weight. Default all edges weight ``1.0``.
            resolution: Resolution parameter (higher = more communities). Default ``1.0``.
            timeout_ms: Abort after this many milliseconds, returning partial results.

        Returns:
            Dict with ``communities`` (dict of community_id to member list),
            ``modularity``, and ``num_communities``.
        """
        ...

    def label_propagation(
        self,
        max_iterations: Optional[int] = None,
        timeout_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Detect communities using label propagation.

        Args:
            max_iterations: Maximum iterations. Default ``100``.
            timeout_ms: Abort after this many milliseconds, returning partial results.

        Returns:
            Dict with ``communities``, ``modularity``, and ``num_communities``.
        """
        ...

    # ====================================================================
    # Subgraph Extraction
    # ====================================================================

    def expand(self, hops: Optional[int] = None) -> KnowledgeGraph:
        """Expand the selection by *N* hops (breadth-first, undirected).

        Args:
            hops: Number of hops to expand. Default ``1``.

        Returns:
            A new KnowledgeGraph with the expanded selection.
        """
        ...

    def to_subgraph(self) -> KnowledgeGraph:
        """Extract selected nodes into a new independent graph.

        The new graph contains only selected nodes and the edges between them.
        """
        ...

    def subgraph_stats(self) -> dict[str, Any]:
        """Get statistics about the subgraph that would be extracted.

        Returns:
            Dict with ``node_count``, ``edge_count``, ``node_types``, ``connection_types``.
        """
        ...

    # ====================================================================
    # Export
    # ====================================================================

    def export(
        self,
        path: str,
        format: Optional[str] = None,
        selection_only: Optional[bool] = None,
    ) -> None:
        """Export the graph to a file.

        Supported formats: ``graphml``, ``gexf``, ``d3``/``json``, ``csv``.
        Format is inferred from the file extension if not specified.

        Args:
            path: Output file path.
            format: Export format. Default: inferred from extension.
            selection_only: Export only selected nodes. Default: ``True`` if selection exists.
        """
        ...

    def export_string(
        self,
        format: str,
        selection_only: Optional[bool] = None,
    ) -> str:
        """Export the graph to a string.

        Supported formats: ``graphml``, ``gexf``, ``d3``/``json``.

        Args:
            format: Export format.
            selection_only: Export only selected nodes.
        """
        ...

    # ====================================================================
    # Property Indexes
    # ====================================================================

    def create_index(self, node_type: str, property: str) -> dict[str, Any]:
        """Create an index on a property for O(1) equality filter lookups.

        Indexes are automatically maintained by Cypher mutations
        (CREATE, SET, REMOVE, DELETE, MERGE).

        Args:
            node_type: Node type to index.
            property: Property name to index.

        Returns:
            Dict with ``type``, ``property``, ``unique_values``, ``created``.
        """
        ...

    def drop_index(self, node_type: str, property: str) -> bool:
        """Remove an index. Returns ``True`` if it existed."""
        ...

    def list_indexes(self) -> list[dict[str, str]]:
        """List all indexes. Each dict has ``type`` and ``property``."""
        ...

    def has_index(self, node_type: str, property: str) -> bool:
        """Check if an index exists."""
        ...

    def index_stats(self, node_type: str, property: str) -> Optional[dict[str, Any]]:
        """Get statistics for an index. Returns ``None`` if not found."""
        ...

    def rebuild_indexes(self) -> int:
        """Rebuild all indexes. Returns the number of indexes rebuilt."""
        ...

    # ====================================================================
    # Composite Indexes
    # ====================================================================

    def create_composite_index(
        self,
        node_type: str,
        properties: list[str],
    ) -> dict[str, Any]:
        """Create a composite index on multiple properties.

        Args:
            node_type: Node type to index.
            properties: List of property names for the composite key.

        Returns:
            Dict with ``type``, ``properties``, ``unique_combinations``.
        """
        ...

    def drop_composite_index(self, node_type: str, properties: list[str]) -> bool:
        """Remove a composite index. Returns ``True`` if it existed."""
        ...

    def list_composite_indexes(self) -> list[dict[str, Any]]:
        """List all composite indexes."""
        ...

    def has_composite_index(self, node_type: str, properties: list[str]) -> bool:
        """Check if a composite index exists."""
        ...

    def composite_index_stats(
        self,
        node_type: str,
        properties: list[str],
    ) -> Optional[dict[str, Any]]:
        """Get statistics for a composite index. Returns ``None`` if not found."""
        ...

    # ====================================================================
    # Pattern Matching & Cypher
    # ====================================================================

    def match_pattern(
        self,
        pattern: str,
        max_matches: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Match a Cypher-like pattern against the graph.

        Supports node patterns ``(a:Type {prop: val})``, directed edges
        ``-[:TYPE]->``, ``<-[:TYPE]-``, and undirected ``-[:TYPE]-``.

        Args:
            pattern: Pattern string, e.g. ``'(a:Person)-[:KNOWS]->(b:Person)'``.
            max_matches: Maximum results to return.

        Returns:
            List of match dicts with variable bindings.
        """
        ...

    def cypher(
        self,
        query: str,
        *,
        to_df: bool = False,
        params: Optional[dict[str, Any]] = None,
    ) -> Union[list[dict[str, Any]], pd.DataFrame]:
        """Execute a Cypher query.

        Supports MATCH, WHERE, RETURN, ORDER BY, LIMIT, SKIP, WITH,
        OPTIONAL MATCH, UNWIND, UNION, CREATE, SET, DELETE, DETACH DELETE,
        REMOVE, MERGE (with ON CREATE SET / ON MATCH SET), CASE expressions,
        WHERE EXISTS, shortestPath(), list comprehensions,
        parameters ($param), ``!=`` operator, and aggregation functions.

        Mutation queries (CREATE, SET, DELETE, REMOVE, MERGE) store
        statistics on ``graph.last_mutation_stats`` with keys
        ``nodes_created``, ``relationships_created``, ``properties_set``,
        ``nodes_deleted``, ``relationships_deleted``, ``properties_removed``.

        Each call is atomic: if any clause fails, the graph is unchanged.
        Property and composite indexes are automatically maintained.

        Args:
            query: Cypher query string.
            to_df: If ``True``, return a pandas DataFrame.
            params: Optional parameter dict for ``$param`` substitution.

        Returns:
            List of row dicts by default, or a DataFrame when ``to_df=True``.

        Example::

            rows = graph.cypher('''
                MATCH (p:Person)-[:KNOWS]->(f:Person)
                WHERE p.age > $min_age
                RETURN p.name, count(f) AS friends
                ORDER BY friends DESC LIMIT 10
            ''', params={'min_age': 25})
            for row in rows:
                print(row['name'], row['friends'])

            # As DataFrame
            df = graph.cypher('MATCH (n:Person) RETURN n.name, n.age', to_df=True)

            # CREATE nodes and edges
            graph.cypher("CREATE (n:Person {name: 'Alice', age: 30})")
            print(graph.last_mutation_stats['nodes_created'])  # 1

            # SET properties
            graph.cypher('''
                MATCH (n:Person) WHERE n.name = 'Alice'
                SET n.city = 'Oslo', n.age = 31
            ''')
        """
        ...

    # ====================================================================
    # Spatial / Geometry
    # ====================================================================

    def within_bounds(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes within a geographic bounding box.

        Args:
            min_lat: South bound latitude.
            max_lat: North bound latitude.
            min_lon: West bound longitude.
            max_lon: East bound longitude.
            lat_field: Latitude property name. Default ``'latitude'``.
            lon_field: Longitude property name. Default ``'longitude'``.

        Returns:
            A new KnowledgeGraph with only nodes in the bounding box.
        """
        ...

    def near_point(
        self,
        center_lat: float,
        center_lon: float,
        max_distance: float,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes within a distance (in degrees) of a point.

        Args:
            center_lat: Center latitude.
            center_lon: Center longitude.
            max_distance: Maximum distance in degrees.
            lat_field: Latitude property name. Default ``'latitude'``.
            lon_field: Longitude property name. Default ``'longitude'``.

        Returns:
            A new KnowledgeGraph with nearby nodes.
        """
        ...

    def near_point_km(
        self,
        center_lat: float,
        center_lon: float,
        max_distance_km: float,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes within a distance (in km) using the Haversine formula.

        More accurate than :meth:`near_point`.

        Args:
            center_lat: Center latitude.
            center_lon: Center longitude.
            max_distance_km: Maximum distance in kilometres.
            lat_field: Latitude property name. Default ``'latitude'``.
            lon_field: Longitude property name. Default ``'longitude'``.

        Returns:
            A new KnowledgeGraph with nearby nodes.
        """
        ...

    def near_point_km_from_wkt(
        self,
        center_lat: float,
        center_lon: float,
        max_distance_km: float,
        geometry_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes by distance using WKT geometry centroids.

        Args:
            center_lat: Center latitude.
            center_lon: Center longitude.
            max_distance_km: Maximum distance in kilometres.
            geometry_field: WKT geometry property name. Default ``'geometry'``.

        Returns:
            A new KnowledgeGraph with nearby nodes.
        """
        ...

    def contains_point(
        self,
        lat: float,
        lon: float,
        geometry_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes whose WKT polygon contains a point.

        Args:
            lat: Query point latitude.
            lon: Query point longitude.
            geometry_field: WKT geometry property name. Default ``'geometry'``.

        Returns:
            A new KnowledgeGraph with containing nodes.
        """
        ...

    def intersects_geometry(
        self,
        query_wkt: str,
        geometry_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes whose geometry intersects a WKT geometry.

        Args:
            query_wkt: WKT string of the query geometry.
            geometry_field: Geometry property name. Default ``'geometry'``.

        Returns:
            A new KnowledgeGraph with intersecting nodes.
        """
        ...

    def get_bounds(
        self,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
    ) -> Optional[dict[str, float]]:
        """Get geographic bounds of selected nodes.

        Returns:
            Dict with ``min_lat``, ``max_lat``, ``min_lon``, ``max_lon``,
            or ``None`` if no valid coordinates found.
        """
        ...

    def get_centroid(
        self,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
    ) -> Optional[dict[str, float]]:
        """Get the geographic centroid (average lat/lon) of selected nodes.

        Returns:
            Dict with ``latitude`` and ``longitude``, or ``None``.
        """
        ...

    def wkt_centroid(self, wkt_string: str) -> dict[str, float]:
        """Calculate the centroid of a WKT geometry string.

        Args:
            wkt_string: WKT geometry (e.g. ``'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'``).

        Returns:
            Dict with ``latitude`` and ``longitude``.
        """
        ...
