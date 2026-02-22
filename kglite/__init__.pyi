"""Type stubs for kglite — a high-performance knowledge graph library."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Union, overload, runtime_checkable

import pandas as pd

__version__: str


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models passed to ``embed_texts`` / ``search_text``.

    **Required** — ``dimension`` and ``embed()`` must be present.

    **Optional** — ``load()`` and ``unload()`` are called automatically if present:

    - ``load()`` is called before each ``embed_texts()`` / ``search_text()`` call.
    - ``unload()`` is called after each call completes (even on error).

    This lets models manage heavyweight resources (GPU memory, large weights)
    on demand.  A common pattern is to implement a cooldown in ``unload()``
    so the model stays warm across rapid successive calls but eventually
    releases memory after a period of inactivity.

    Example::

        import threading
        from sentence_transformers import SentenceTransformer

        class Embedder:
            def __init__(self, model_name="all-MiniLM-L6-v2"):
                self._model_name = model_name
                self._model = None
                self._timer = None
                self.dimension = 384  # known ahead of time, or set in load()

            def load(self):
                if self._timer:
                    self._timer.cancel()
                    self._timer = None
                if self._model is None:
                    self._model = SentenceTransformer(self._model_name)
                    self.dimension = self._model.get_sentence_embedding_dimension()

            def unload(self, cooldown=60):
                def _release():
                    self._model = None
                    self._timer = None
                self._timer = threading.Timer(cooldown, _release)
                self._timer.start()

            def embed(self, texts: list[str]) -> list[list[float]]:
                return self._model.encode(texts).tolist()
    """

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per text."""
        ...

    def load(self) -> None:
        """(Optional) Load model weights / allocate resources.

        Called automatically before ``embed()`` in ``embed_texts()`` and
        ``search_text()``.  If not defined, this step is skipped.
        """
        ...

    def unload(self) -> None:
        """(Optional) Release model weights / free resources.

        Called automatically after ``embed_texts()`` and ``search_text()``
        complete (including on error).  A common pattern is to start a
        cooldown timer here instead of releasing immediately.
        """
        ...


class ResultIter:
    """Iterator for ResultView. Converts one row per step."""

    def __iter__(self) -> ResultIter: ...
    def __next__(self) -> dict[str, Any]: ...


class ResultView:
    """Lazy result container — data stays in Rust until accessed from Python.

    Returned by ``cypher()``, centrality methods, ``get_nodes()`` (flat),
    and ``sample()``.

    Data is only converted to Python objects when you actually access rows
    (via iteration, indexing, ``to_list()``, or ``to_df()``). This makes
    ``cypher()`` calls fast even for large result sets — the cost is deferred
    to when you consume the data.

    Supports:
      - ``len(result)`` — row count (O(1), no conversion)
      - ``bool(result)`` — True if non-empty
      - ``result[i]`` — single row as dict (converts that row only)
      - ``for row in result`` — iterate rows as dicts (one at a time)
      - ``result.head(n)`` / ``result.tail(n)`` — first/last n rows as a new ResultView
      - ``result.to_list()`` — all rows as ``list[dict]`` (full conversion)
      - ``result.to_df()`` — pandas DataFrame (full conversion)
      - ``result.columns`` — column names
      - ``result.stats`` — mutation stats (CREATE/SET/DELETE queries only)
    """

    @property
    def columns(self) -> list[str]:
        """Column names."""
        ...

    @property
    def stats(self) -> Optional[dict[str, int]]:
        """Mutation statistics, or ``None`` for read queries / non-cypher results."""
        ...

    def head(self, n: int = 5) -> ResultView:
        """Return a new ResultView with the first *n* rows (default 5)."""
        ...

    def tail(self, n: int = 5) -> ResultView:
        """Return a new ResultView with the last *n* rows (default 5)."""
        ...

    def to_list(self) -> list[dict[str, Any]]:
        """Convert all rows to a Python list of dicts (full materialization)."""
        ...

    def to_df(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame."""
        ...

    def to_gdf(
        self,
        geometry_column: str = "geometry",
        crs: Optional[str] = None,
    ) -> Any:
        """Convert to a GeoDataFrame with a geometry column parsed from WKT.

        Materializes the data as a DataFrame, then converts the specified
        WKT string column into shapely geometries and returns a
        ``geopandas.GeoDataFrame``.

        Args:
            geometry_column: Column containing WKT strings. Default ``'geometry'``.
            crs: Coordinate reference system (e.g. ``'EPSG:4326'``), or ``None``.

        Returns:
            A ``geopandas.GeoDataFrame``.

        Raises:
            ImportError: If geopandas is not installed.
        """
        ...

    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __getitem__(self, index: int) -> dict[str, Any]: ...
    def __iter__(self) -> ResultIter: ...
    def __repr__(self) -> str: ...


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
        timeseries: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Add nodes from a DataFrame.

        String and integer IDs are auto-detected from the DataFrame dtype.
        Non-contiguous DataFrame indexes (e.g. from filtering) are handled
        automatically.

        When ``timeseries`` is provided, the DataFrame may contain multiple
        rows per unique ID (one per time step). Rows are deduplicated
        automatically — the first occurrence per ID provides static node
        properties, and all rows contribute to the timeseries channels.

        Args:
            data: DataFrame containing node data.
            node_type: Label for this set of nodes (e.g. ``'Person'``).
            unique_id_field: Column used as unique identifier.
            node_title_field: Column used as display title. Defaults to ``unique_id_field``.
            columns: Whitelist of columns to include. ``None`` = all.
            conflict_handling: ``'update'`` (default), ``'replace'``, ``'skip'``, or ``'preserve'``.
            skip_columns: Columns to exclude.
            column_types: Override column dtypes ``{'col': 'string'|'integer'|'float'|'datetime'|'uniqueid'}``.
                Also supports spatial types: ``'location.lat'``, ``'location.lon'``,
                ``'geometry'``, ``'point.<name>.lat'``, ``'point.<name>.lon'``,
                ``'shape.<name>'``.
            timeseries: Inline timeseries configuration dict with keys:

                - ``time`` (required): column name containing date strings
                  (``'yyyy-mm'``, ``'yyyy-mm-dd'``, ``'yyyy-mm-dd hh:mm'``),
                  or a dict mapping ``year``/``month``/``day``/``hour``/``minute``
                  to column names (e.g. ``{'year': 'ar', 'month': 'maned'}``).
                - ``channels`` (required): list of column names for timeseries
                  data (e.g. ``['oil', 'gas', 'condensate']``).
                - ``resolution`` (optional): ``'year'``, ``'month'``, ``'day'``,
                  ``'hour'``, or ``'minute'``. Auto-detected from time format if omitted.
                - ``units`` (optional): dict mapping channel names to unit strings
                  (e.g. ``{'oil': 'MSm3'}``).

        Returns:
            Operation report dict with keys ``nodes_created``,
            ``nodes_updated``, ``nodes_skipped``, ``processing_time_ms``,
            ``has_errors``, and optionally ``errors`` with skip reasons.

        Example::

            graph.add_nodes(df, 'Production', 'field_id', 'field_name',
                timeseries={
                    'time': 'date',
                    'channels': ['oil', 'gas', 'condensate', 'oe'],
                })
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
        ``'is_not_null'``, ``'contains'``, ``'starts_with'``, ``'ends_with'``,
        ``'regex'`` (or ``'=~'``), and negated variants: ``'not_contains'``,
        ``'not_starts_with'``, ``'not_ends_with'``, ``'not_in'``, ``'not_regex'``.

        Example::

            graph.type_filter('Person').filter({
                'age': {'>': 25},
                'city': 'Oslo',
                'name': {'regex': '^A.*'},
                'status': {'not_in': ['inactive', 'banned']},
            })

        Returns:
            A new KnowledgeGraph with the filtered selection.
        """
        ...

    def filter_any(
        self,
        conditions: list[dict[str, Any]],
        sort: Optional[Union[str, list[tuple[str, bool]]]] = None,
        max_nodes: Optional[int] = None,
    ) -> KnowledgeGraph:
        """Filter the current selection with OR logic across multiple condition sets.

        Each dict in *conditions* is a set of AND conditions (same as ``filter()``).
        A node is kept if it matches **any** of the condition sets.

        Args:
            conditions: List of condition dicts. Must contain at least one.
            sort: Optional sort spec.
            max_nodes: Limit the number of selected nodes.

        Returns:
            A new KnowledgeGraph with the filtered selection.

        Example::

            graph.type_filter('Person').filter_any([
                {'city': 'Oslo'},
                {'city': 'Bergen'},
            ])

        Raises:
            ValueError: If *conditions* is empty.
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

    def offset(self, n: int) -> KnowledgeGraph:
        """Skip the first *n* nodes per parent group (pagination).

        Combine with ``max_nodes()`` for pagination:
        ``graph.sort('name').offset(20).max_nodes(10)``

        Args:
            n: Number of nodes to skip.

        Returns:
            A new KnowledgeGraph with the offset selection.
        """
        ...

    def has_connection(
        self,
        connection_type: str,
        direction: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes that have at least one connection of the given type.

        Keeps only nodes from the current selection that participate in
        edges of the specified type and direction.

        Args:
            connection_type: Edge type to check (e.g. ``'KNOWS'``).
            direction: ``'outgoing'``, ``'incoming'``, or ``'any'`` (default).

        Returns:
            A new KnowledgeGraph with only connected nodes.

        Raises:
            ValueError: If *direction* is not one of the valid values.
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
    ) -> Union[ResultView, dict[str, Any]]:
        """Materialise selected nodes as a ResultView (flat) or grouped dict.

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

    def find(
        self,
        name: str,
        node_type: Optional[str] = None,
        match_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Find code entities by name, with disambiguation context.

        Searches across code entity node types (Function, Struct, Class, Enum,
        Trait, Protocol, Interface, Module, Constant) for nodes matching the
        given name.

        Args:
            name: Entity name to search for (e.g. ``"execute"``).
            node_type: Optional filter — only search this node type
                (e.g. ``"Function"``, ``"Struct"``).
            match_type: Matching strategy: ``"exact"`` (default),
                ``"contains"`` (case-insensitive substring), or
                ``"starts_with"`` (case-insensitive prefix).

        Returns:
            List of dicts with: type, name, qualified_name, file_path,
            line_number, and optionally signature and visibility.
        """
        ...

    @overload
    def source(
        self,
        name: str,
        node_type: Optional[str] = None,
    ) -> dict[str, Any]: ...
    @overload
    def source(
        self,
        name: list[str],
        node_type: Optional[str] = None,
    ) -> list[dict[str, Any]]: ...
    def source(
        self,
        name: str | list[str],
        node_type: Optional[str] = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Get the source location of one or more code entities.

        Resolves names or qualified names to code entities and returns
        file paths and line ranges.

        Args:
            name: Entity name, qualified name, or list of names.
            node_type: Optional node type hint.

        Returns:
            Single name: dict with ``file_path``, ``line_number``,
            ``end_line``, ``line_count``, ``name``, ``qualified_name``,
            ``type``, ``signature``.
            List of names: list of such dicts.
        """
        ...

    def context(
        self,
        name: str,
        node_type: Optional[str] = None,
        hops: Optional[int] = None,
    ) -> dict[str, Any]:
        """Get the full neighborhood of a code entity.

        Returns the node's properties and all related entities grouped by
        relationship type. If the name is ambiguous, returns the matches
        so you can refine with a qualified name.

        Args:
            name: Entity name or qualified name.
            node_type: Optional node type hint.
            hops: Max traversal depth (default 1).

        Returns:
            Dict with ``"node"`` (properties), ``"defined_in"`` (file path),
            and relationship groups (e.g. ``"HAS_METHOD"``, ``"CALLS"``,
            ``"called_by"``).
        """
        ...

    def toc(
        self,
        file_path: str,
    ) -> dict[str, Any]:
        """Get a table of contents for a file — all code entities defined in it.

        Returns entities sorted by line number with a type summary.

        Args:
            file_path: Path of the file (the File node's id/path).

        Returns:
            Dict with ``"file"`` (path), ``"entities"`` (list of entity dicts
            sorted by line_number, each with type, name, qualified_name,
            line_number, end_line, and optionally signature), and ``"summary"``
            (dict of type name to count).
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

    def set_auto_vacuum(self, threshold: float | None) -> None:
        """Configure automatic vacuum after DELETE operations.

        When enabled, the graph automatically compacts itself after Cypher DELETE
        operations if the fragmentation ratio exceeds the threshold and there are
        more than 100 tombstones.

        Args:
            threshold: A float between 0.0 and 1.0, or ``None`` to disable.
                Default is ``0.3`` (30% fragmentation triggers vacuum).

        Example::

            graph.set_auto_vacuum(0.2)   # more aggressive — vacuum at 20%
            graph.set_auto_vacuum(None)  # disable auto-vacuum
            graph.set_auto_vacuum(0.3)   # restore default
        """
        ...

    def read_only(self, enabled: bool | None = None) -> bool:
        """Set or query read-only mode for the Cypher layer.

        When enabled, all Cypher mutation queries (CREATE, SET, DELETE, REMOVE,
        MERGE) are rejected, and ``agent_describe()`` omits mutation docs.

        Args:
            enabled: If ``True``, enable read-only mode. If ``False``, disable.
                If omitted, return the current state without changing it.

        Returns:
            The current read-only state (after applying the change, if any).

        Example::

            graph.read_only(True)   # lock the graph
            graph.read_only()       # -> True
            graph.read_only(False)  # unlock
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

    def statistics(
        self,
        property: str,
        level_index: Optional[int] = None,
        group_by: Optional[str] = None,
    ) -> Any:
        """Compute descriptive statistics for a numeric property.

        Returns per-parent stats including count, mean, std, min, max, sum.

        Args:
            property: Numeric property name.
            level_index: Target level in the hierarchy.
            group_by: Group results by this property instead of by parent.
                Returns ``{group_value: {count, sum, mean, min, max, std}}``.
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
        group_by: Optional[str] = None,
    ) -> Any:
        """Count nodes, optionally grouped by parent or by a property.

        Args:
            level_index: Target level in the hierarchy.
            group_by_parent: Group counts by parent node.
            store_as: Store count as a property on parent nodes.
            keep_selection: Preserve selection after store. Default ``False``.
            group_by: Group counts by this property instead of by parent.
                Returns ``{group_value: count}``.

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

    def agent_describe(self) -> str:
        """Return a minimal XML string describing this graph for AI agents.

        The output is a self-contained XML document covering:
        - Graph structure: node types with counts and property schemas,
          connection types with counts and endpoint types, indexes.
        - Supported Cypher subset: clauses, patterns, operators, functions.
        - Key API methods with signatures.

        Designed to be included directly in an LLM prompt so an agent
        can autonomously query the graph using ``cypher()``.

        Example::

            prompt = f"You have a knowledge graph:\\n{graph.agent_describe()}\\nAnswer the question."
        """
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

    def properties(self, node_type: str, max_values: int = 20) -> dict[str, dict[str, Any]]:
        """Return property statistics for a node type.

        Only properties that exist on at least one node are included.

        Args:
            node_type: The node type to inspect.
            max_values: Include ``values`` list when unique count <= this
                threshold. Set to 0 to never include values. Default: 20.

        Returns:
            Dict mapping property name to stats dict with keys:
                - ``type``: type string (e.g. ``'str'``, ``'int'``, ``'float'``)
                - ``non_null``: count of non-null values
                - ``unique``: count of distinct values
                - ``values``: (optional) sorted list of values when unique count <= max_values

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

    def sample(self, node_type: str, n: int = 5) -> ResultView:
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
    ) -> Union[ResultView, dict[Any, float], pd.DataFrame]:
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
    ) -> Union[ResultView, dict[Any, float], pd.DataFrame]:
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
    ) -> Union[ResultView, dict[Any, float], pd.DataFrame]:
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
    ) -> Union[ResultView, dict[Any, float], pd.DataFrame]:
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
        connection_types: Optional[list[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Detect communities using the Louvain algorithm.

        Args:
            weight_property: Edge property to use as weight. Default all edges weight ``1.0``.
            resolution: Resolution parameter (higher = more communities). Default ``1.0``.
            connection_types: Only consider edges of these types. Default all edge types.
            timeout_ms: Abort after this many milliseconds, returning partial results.

        Returns:
            Dict with ``communities`` (dict of community_id to member list),
            ``modularity``, and ``num_communities``.
        """
        ...

    def label_propagation(
        self,
        max_iterations: Optional[int] = None,
        connection_types: Optional[list[str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Detect communities using label propagation.

        Args:
            max_iterations: Maximum iterations. Default ``100``.
            connection_types: Only consider edges of these types. Default all edge types.
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
    # Range Indexes (B-Tree)
    # ====================================================================

    def create_range_index(self, node_type: str, property: str) -> dict[str, Any]:
        """Create a range index (B-Tree) on a property for efficient range queries.

        Enables fast ``>``, ``>=``, ``<``, ``<=``, and ``BETWEEN`` queries
        in ``filter()`` calls.

        Args:
            node_type: Node type to index.
            property: Property name to index.

        Returns:
            Dict with ``type``, ``property``, ``unique_values``, ``created``.

        Example::

            graph.create_range_index('Person', 'age')
            old = graph.filter({'type': 'Person'}).filter({'age': {'>': 60}}).get_nodes()
        """
        ...

    def drop_range_index(self, node_type: str, property: str) -> bool:
        """Remove a range index. Returns ``True`` if it existed."""
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
    ) -> Union[ResultView, pd.DataFrame]:
        """Execute a Cypher query.

        Supports MATCH, WHERE, RETURN, ORDER BY, LIMIT, SKIP, WITH,
        OPTIONAL MATCH, UNWIND, UNION, CREATE, SET, DELETE, DETACH DELETE,
        REMOVE, MERGE (with ON CREATE SET / ON MATCH SET), CASE expressions,
        WHERE EXISTS, shortestPath(), list comprehensions,
        CALL...YIELD (graph algorithms: pagerank, betweenness, degree,
        closeness, louvain, label_propagation, connected_components),
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

            # Semantic search with text_score (requires set_embedder + embed_texts)
            results = graph.cypher('''
                MATCH (n:Article)
                RETURN n.title,
                       text_score(n, 'summary', 'machine learning') AS score
                ORDER BY score DESC LIMIT 10
            ''', to_df=True)

            # text_score with parameter
            graph.cypher('''
                MATCH (n:Article)
                WHERE text_score(n, 'summary', $query) > 0.8
                RETURN n.title
            ''', params={'query': 'artificial intelligence'})

            # CALL graph algorithms
            top = graph.cypher('''
                CALL pagerank() YIELD node, score
                RETURN node.title, score
                ORDER BY score DESC LIMIT 10
            ''')

            # Community detection
            graph.cypher('''
                CALL louvain() YIELD node, community
                RETURN community, count(*) AS size
                ORDER BY size DESC
            ''')
        """
        ...

    # ====================================================================
    # Spatial / Geometry
    # ====================================================================

    def set_spatial(
        self,
        node_type: str,
        *,
        location: Optional[tuple[str, str]] = None,
        geometry: Optional[str] = None,
        points: Optional[dict[str, tuple[str, str]]] = None,
        shapes: Optional[dict[str, str]] = None,
    ) -> None:
        """Configure spatial properties for a node type.

        Args:
            node_type: The node type to configure.
            location: Primary lat/lon pair as ``(lat_field, lon_field)``. At most one per type.
            geometry: Primary WKT geometry field name. At most one per type.
            points: Named lat/lon points as ``{name: (lat_field, lon_field)}``.
            shapes: Named WKT shape fields as ``{name: field_name}``.
        """
        ...

    def get_spatial(
        self,
        node_type: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get spatial configuration for a node type or all types.

        Args:
            node_type: If given, return config for this type only. Otherwise return all.

        Returns:
            Dict with spatial config, or ``None`` if not configured.
        """
        ...

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
        query_wkt: Union[str, Any],
        geometry_field: Optional[str] = None,
    ) -> KnowledgeGraph:
        """Filter nodes whose geometry intersects a WKT geometry.

        Args:
            query_wkt: WKT string or shapely geometry object.
            geometry_field: Geometry property name. Default ``'geometry'``.

        Returns:
            A new KnowledgeGraph with intersecting nodes.
        """
        ...

    def get_bounds(
        self,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
        as_shapely: bool = False,
    ) -> Optional[Union[dict[str, float], Any]]:
        """Get geographic bounds of selected nodes.

        Args:
            lat_field: Latitude property name. Default ``'latitude'``.
            lon_field: Longitude property name. Default ``'longitude'``.
            as_shapely: If ``True``, return a ``shapely.geometry.Polygon``
                (box) instead of a dict.

        Returns:
            Dict with ``min_lat``, ``max_lat``, ``min_lon``, ``max_lon``,
            or a shapely box polygon when ``as_shapely=True``,
            or ``None`` if no valid coordinates found.
        """
        ...

    def get_centroid(
        self,
        lat_field: Optional[str] = None,
        lon_field: Optional[str] = None,
        as_shapely: bool = False,
    ) -> Optional[Union[dict[str, float], Any]]:
        """Get the geographic centroid (average lat/lon) of selected nodes.

        Args:
            lat_field: Latitude property name. Default ``'latitude'``.
            lon_field: Longitude property name. Default ``'longitude'``.
            as_shapely: If ``True``, return a ``shapely.geometry.Point``
                instead of a dict.

        Returns:
            Dict with ``latitude`` and ``longitude``, or a shapely Point
            when ``as_shapely=True``, or ``None``.
        """
        ...

    def wkt_centroid(
        self,
        wkt_string: Union[str, Any],
        as_shapely: bool = False,
    ) -> Union[dict[str, float], Any]:
        """Calculate the centroid of a WKT geometry string.

        Args:
            wkt_string: WKT geometry string or shapely geometry object.
            as_shapely: If ``True``, return a ``shapely.geometry.Point``
                instead of a dict.

        Returns:
            Dict with ``latitude`` and ``longitude``, or a shapely Point
            when ``as_shapely=True``.
        """
        ...

    # ── Timeseries ─────────────────────────────────────────────────────────

    def set_timeseries(
        self,
        node_type: str,
        *,
        resolution: str,
        channels: Optional[list[str]] = None,
        units: Optional[dict[str, str]] = None,
        bin_type: Optional[str] = None,
    ) -> None:
        """Configure timeseries metadata for a node type.

        Args:
            node_type: The node type to configure.
            resolution: Time granularity — ``'year'``, ``'month'``, or ``'day'``.
                Determines key depth (year=1, month=2, day=3).
            channels: Optional list of known channel names.
            units: Optional map of channel name to unit string,
                e.g. ``{'oil': 'MSm3', 'temperature': '°C'}``.
            bin_type: What values represent — ``'total'``, ``'mean'``,
                or ``'sample'``. None if unspecified.
        """
        ...

    def get_timeseries_config(
        self,
        node_type: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get timeseries configuration for a node type or all types.

        Returns a dict with ``resolution``, ``channels``, ``units``, ``bin_type``.
        """
        ...

    def set_time_index(
        self,
        node_id: Any,
        keys: list[list[int]],
    ) -> None:
        """Set the sorted time index for a specific node.

        If the node already has a timeseries, this replaces its time index
        and clears all channels. Key depth must match the resolution set
        via ``set_timeseries()``.

        Args:
            node_id: The node's unique ID.
            keys: Sorted list of composite time keys,
                e.g. ``[[2020, 1], [2020, 2], ...]`` for month resolution.
        """
        ...

    def add_ts_channel(
        self,
        node_id: Any,
        channel_name: str,
        values: list[float],
    ) -> None:
        """Add a timeseries channel to a node.

        The node must already have a time index set (via ``set_time_index``
        or ``add_timeseries``). The values length must match the time index
        length. Use ``float('nan')`` for missing values.

        Args:
            node_id: The node's unique ID.
            channel_name: Channel name (e.g. ``'oil'``, ``'temperature'``).
            values: Float values aligned with the time index.
        """
        ...

    def add_timeseries(
        self,
        node_type: str,
        *,
        data: Any,
        fk: str,
        time_key: list[str],
        channels: Union[dict[str, str], list[str]],
        resolution: Optional[str] = None,
        units: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Bulk-load timeseries data from a DataFrame.

        Groups rows by ``fk``, sorts by ``time_key``, and attaches the
        resulting timeseries to matching nodes (found by node ID).

        Args:
            node_type: Target node type.
            data: Source DataFrame.
            fk: Foreign key column in ``data`` linking to node IDs.
            time_key: Column(s) forming the composite time key.
                Count must match resolution depth (2 for month, 3 for day).
            channels: Either a list of column names (used as channel names)
                or a dict mapping ``{channel_name: column_name}``.
            resolution: Time granularity (``'year'``, ``'month'``, ``'day'``).
                Required if ``set_timeseries()`` has not been called.
            units: Optional channel→unit map, merged into config.

        Returns:
            Summary: ``{'nodes_loaded': N, 'total_records': M, 'total_rows': R}``.
        """
        ...

    def get_timeseries(
        self,
        node_id: Any,
        channel: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Extract timeseries data for a node.

        If ``channel`` is given, returns ``{'keys': [...], 'values': [...]}``.
        Otherwise returns ``{'keys': [...], 'channels': {'name': [...], ...}}``.

        Args:
            node_id: The node's unique ID.
            channel: Optional channel name to extract.
            start: Optional range start as date string (e.g. ``'2020'``, ``'2020-2'``).
            end: Optional range end as date string.

        Returns:
            Dict with keys and channel data, or ``None`` if no timeseries.
        """
        ...

    def get_time_index(
        self,
        node_id: Any,
    ) -> Optional[list[list[int]]]:
        """Get the time index for a node, or ``None``."""
        ...

    # ── Embedding / Vector Search ──────────────────────────────────────────

    def set_embeddings(
        self,
        node_type: str,
        text_column: str,
        embeddings: dict[Any, list[float]],
    ) -> dict[str, int]:
        """Store embeddings for nodes of the given type.

        Embeddings are stored separately from regular node properties and are
        invisible to ``get_nodes()``, ``to_df()``, and other property-based APIs.
        The embedding store key is auto-derived as ``{text_column}_emb``.

        Validates that ``text_column`` exists as a property on the node type
        (builtins ``id``, ``title``, ``type`` are always accepted).

        Args:
            node_type: The node type (e.g. ``'Article'``).
            text_column: Source text column name (e.g. ``'summary'``).
            embeddings: Dict mapping node IDs to embedding vectors.

        Returns:
            Dict with ``embeddings_stored``, ``dimension``, and ``skipped``.
        """
        ...

    def vector_search(
        self,
        text_column: str,
        query_vector: list[float],
        top_k: int = 10,
        metric: str = "cosine",
        to_df: bool = False,
    ) -> list[dict[str, Any]] | pd.DataFrame:
        """Vector similarity search within the current selection.

        Searches for nodes most similar to the query vector among the currently
        selected nodes. Results are ordered by similarity (most similar first).

        Args:
            text_column: Source text column name (e.g. ``'summary'``).
            query_vector: The query embedding vector.
            top_k: Number of results to return (default 10).
            metric: ``'cosine'`` (default), ``'dot_product'``, or ``'euclidean'``.
            to_df: If ``True``, return a pandas DataFrame instead of list of dicts.

        Returns:
            List of dicts with ``id``, ``title``, ``type``, ``score``, and all
            node properties. Or a DataFrame if ``to_df=True``.

        Example::

            results = (graph
                .type_filter('Article')
                .filter({'category': 'politics'})
                .vector_search('summary', query_vec, top_k=10))
        """
        ...

    def list_embeddings(self) -> list[dict[str, Any]]:
        """List all embedding stores in the graph.

        Returns:
            List of dicts with ``node_type``, ``text_column``, ``dimension``, ``count``.
        """
        ...

    def remove_embeddings(self, node_type: str, text_column: str) -> None:
        """Remove an embedding store.

        Args:
            node_type: The node type.
            text_column: Source text column name (e.g. ``'summary'``).
        """
        ...

    @overload
    def get_embeddings(self, node_type: str, text_column: str) -> dict[Any, list[float]]:
        """Retrieve all embeddings for a node type.

        Args:
            node_type: The node type (e.g. 'Article').
            text_column: Source text column name (e.g. 'summary').

        Returns:
            Dict mapping node IDs to embedding vectors.
        """
        ...

    @overload
    def get_embeddings(self, text_column: str) -> dict[Any, list[float]]:
        """Retrieve embeddings for nodes in the current selection.

        Args:
            text_column: Source text column name (e.g. 'summary').

        Returns:
            Dict mapping node IDs to embedding vectors.
        """
        ...

    def get_embeddings(self, *args, **kwargs) -> dict[Any, list[float]]: ...

    def get_embedding(
        self, node_type: str, text_column: str, node_id: Any
    ) -> list[float] | None:
        """Retrieve a single node's embedding vector.

        Args:
            node_type: The node type (e.g. 'Article').
            text_column: Source text column name (e.g. 'summary').
            node_id: The node ID to look up.

        Returns:
            The embedding vector as a list of floats, or None if not found.
        """
        ...

    def set_embedder(self, model: EmbeddingModel) -> None:
        """Register an embedding model on the graph.

        After calling this, ``embed_texts()`` and ``search_text()`` use the
        registered model automatically.  The model is **not** serialized —
        call ``set_embedder()`` again after deserializing.

        If the model has optional ``load()`` / ``unload()`` methods, they are
        called automatically around each embedding operation.

        Args:
            model: An embedding model with ``dimension`` and ``embed()`` — see
                :class:`EmbeddingModel`.

        Example::

            g.set_embedder(my_model)
        """
        ...

    def embed_texts(
        self,
        node_type: str,
        text_column: str,
        batch_size: int = 256,
        show_progress: bool = True,
        replace: bool = False,
    ) -> dict[str, int]:
        """Embed a text column for all nodes of a given type.

        Uses the model registered via ``set_embedder()``.  Reads each node's
        ``text_column`` property, calls ``model.embed()`` in batches, and
        stores the resulting vectors as ``{text_column}_emb``.
        Nodes with missing or non-string text are skipped.

        By default, nodes that already have an embedding are skipped.
        Pass ``replace=True`` to re-embed everything.

        Shows a tqdm progress bar by default (requires ``tqdm``).

        Args:
            node_type: The node type to embed (e.g. ``'Article'``).
            text_column: The node property containing text to embed.
            batch_size: Number of texts per ``model.embed()`` call (default 256).
            show_progress: Show a tqdm progress bar (default ``True``).
                Silently falls back to no bar if ``tqdm`` is not installed.
            replace: Re-embed all nodes, even those with existing embeddings
                (default ``False``).

        Returns:
            Dict with ``embedded``, ``skipped``, ``skipped_existing``, and ``dimension``.

        Example::

            g.set_embedder(my_model)
            g.embed_texts("Article", "summary")
            # Embedding Article.summary: 100%|████████| 1000/1000 [00:05<00:00]

            # Add new articles, then re-run — only new ones get embedded:
            g.embed_texts("Article", "summary")  # skips already-embedded nodes
        """
        ...

    def search_text(
        self,
        text_column: str,
        query: str,
        top_k: int = 10,
        metric: str = "cosine",
        to_df: bool = False,
    ) -> list[dict[str, Any]] | pd.DataFrame:
        """Search embeddings using a text query.

        Uses the model registered via ``set_embedder()`` to embed the query,
        then performs vector search within the current selection.  Refer to
        the text column name (e.g. ``"summary"``); the graph resolves it to
        ``"summary_emb"`` internally.

        Args:
            text_column: Text column whose embeddings to search (e.g. ``'summary'``).
            query: The text query to search for.
            top_k: Number of results (default 10).
            metric: ``'cosine'`` (default), ``'dot_product'``, or ``'euclidean'``.
            to_df: If True, return a pandas DataFrame.

        Returns:
            Same format as ``vector_search()`` — list of dicts or DataFrame.

        Example::

            results = g.type_filter("Article").search_text(
                "summary", "find AI articles", top_k=10
            )
        """
        ...

    def begin(self) -> Transaction:
        """Begin a transaction — returns a Transaction with a working copy of the graph.

        Creates a snapshot (deep-clone) of the current graph state. All mutations
        within the transaction are isolated until ``commit()`` is called. If rolled
        back (or dropped without committing), no changes are applied.

        Note:
            The snapshot is a full deep-clone of the graph, so creating a
            transaction on a very large graph has a one-time memory cost
            proportional to graph size. Embeddings are not cloned.

        Can be used as a context manager::

            with graph.begin() as tx:
                tx.cypher("CREATE (n:Person {name: 'Alice', age: 30})")
                tx.cypher("CREATE (n:Person {name: 'Bob', age: 25})")
                # auto-commits on success, auto-rollbacks on exception
        """
        ...


class Transaction:
    """An isolated transaction on a KnowledgeGraph.

    Created via :meth:`KnowledgeGraph.begin`. Mutations are applied to a
    working copy and only become visible in the original graph after
    :meth:`commit`. If an exception occurs (or :meth:`rollback` is called),
    all changes are discarded.

    Isolation semantics:
        - **Snapshot isolation**: ``begin()`` clones the entire graph. The
          transaction sees a frozen snapshot from the moment it was created.
        - **Write isolation**: mutations inside the transaction modify only the
          working copy. The original graph is untouched until ``commit()``.
        - **Commit**: replaces the original graph's data atomically.
        - **No concurrent-transaction guarantees**: if two transactions exist
          simultaneously, whichever commits last wins (last-writer-wins). There
          is no conflict detection or merge.
    """

    def cypher(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        to_df: bool = False,
    ) -> ResultView | pd.DataFrame:
        """Execute a Cypher query within this transaction.

        Same interface as :meth:`KnowledgeGraph.cypher` but operates on
        the transaction's working copy.
        """
        ...

    def commit(self) -> None:
        """Commit the transaction — apply all changes to the original graph.

        After commit, the transaction cannot be used again.
        """
        ...

    def rollback(self) -> None:
        """Roll back the transaction — discard all changes.

        After rollback, the transaction cannot be used again.
        """
        ...

    def __enter__(self) -> Transaction: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> bool: ...
