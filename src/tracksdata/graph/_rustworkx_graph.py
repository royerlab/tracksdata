import operator
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import bidict
import numpy as np
import polars as pl
import rustworkx as rx

from tracksdata.attrs import AttrComparison, split_attr_comps
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._mapped_graph_mixin import MappedGraphMixin
from tracksdata.graph.filters._base_filter import BaseFilter
from tracksdata.utils._cache import cache_method
from tracksdata.utils._dataframe import unpack_array_attrs
from tracksdata.utils._dtypes import AttrSchema, process_attr_key_args
from tracksdata.utils._logging import LOG
from tracksdata.utils._signal import is_signal_on

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView


def _pop_time_eq(
    attrs: Sequence[AttrComparison],
) -> tuple[list[AttrComparison], int | None]:
    """
    Pop the time equality filter from a list of attribute filters.
    If multiple time equality filters are found, an error is raised.

    Parameters
    ----------
    attrs : Sequence[AttrComparison]
        The attribute filters to pop the time equality filter from.

    Returns
    -------
    tuple[list[AttrComparison], int | None]
        The attribute filters without the time equality filter and the time value.
    """
    out_attrs = []
    time = None
    for attr_comp in attrs:
        if str(attr_comp.column) == DEFAULT_ATTR_KEYS.T and attr_comp.op == operator.eq:
            if time is not None:
                raise ValueError(f"Multiple '{DEFAULT_ATTR_KEYS.T}' equality filters are not allowed\n {attrs}")
            time = int(attr_comp.other)
        else:
            out_attrs.append(attr_comp)

    return out_attrs, time


def _maybe_fill_null(s: pl.Series, schema: AttrSchema) -> pl.Series:
    if s.has_nulls() and schema.default_value is not None:
        if schema.dtype == pl.Object:
            value = pl.lit(schema.default_value, allow_object=True)
        elif isinstance(schema.dtype, pl.Array):
            if isinstance(schema.default_value, np.ndarray):
                value = schema.default_value.tolist()
            else:
                value = schema.default_value
            value = pl.lit(value).cast(schema.dtype)
        else:
            value = schema.default_value
        s = s.fill_null(value)
    return s


def _create_filter_func(
    attr_comps: Sequence[AttrComparison],
    schema: dict[str, AttrSchema],
) -> Callable[[dict[str, Any]], bool]:
    LOG.info(f"Creating filter function for {attr_comps}")

    def _filter(attrs: dict[str, Any]) -> bool:
        for attr_op in attr_comps:
            value = attrs.get(attr_op.column, schema[attr_op.column].default_value)
            if not attr_op.op(value, attr_op.other):
                return False
        return True

    return _filter


class RXFilter(BaseFilter):
    def __init__(
        self,
        *attr_comps: AttrComparison,
        graph: "RustWorkXGraph",
        node_ids: Sequence[int] | None = None,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._attr_comps = attr_comps

        if node_ids is not None and hasattr(node_ids, "tolist"):
            node_ids = node_ids.tolist()

        self._node_ids = node_ids
        self._include_targets = include_targets
        self._include_sources = include_sources
        self._node_attr_comps, self._edge_attr_comps = split_attr_comps(attr_comps)

    @cache_method
    def _current_node_ids(self) -> list[int]:
        """
        Get the node IDs without considering their `source` or `target` neighbors.
        """
        if len(self._node_attr_comps) == 0:
            if self._node_ids is None:
                node_ids = self._graph.rx_graph.node_indices()
            else:
                node_ids = self._node_ids
        else:
            # only nodes are filtered return nodes that pass node filters
            node_ids = self._graph._filter_nodes_by_attrs(*self._node_attr_comps, node_ids=self._node_ids)

        return node_ids

    @cache_method
    def node_ids(self) -> list[int]:
        # if there are no edge filters, we can return the current node ids
        if not self._edge_attr_comps and (not self._include_targets and not self._include_sources):
            return self._current_node_ids()

        # find nodes that are connected to edges that pass the edge filters
        node_ids = []
        edge_node_ids = (
            self._edge_attrs()
            .select(
                DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                DEFAULT_ATTR_KEYS.EDGE_TARGET,
            )
            .to_numpy()
            .ravel()
        )
        node_ids.append(edge_node_ids)

        if self._node_attr_comps:
            # if there are node filters, we need to add the nodes that pass the node filters
            node_ids.append(self._current_node_ids())

        node_ids = [v for v in node_ids if len(v) > 0]

        if len(node_ids) == 0:
            return []

        node_ids = np.unique(np.concatenate(node_ids, dtype=int))
        return node_ids.tolist()

    @cache_method
    def _edge_attrs(self) -> pl.DataFrame:
        node_ids = self._current_node_ids()

        _filter_func = _create_filter_func(self._edge_attr_comps, self._graph._edge_attr_schemas())
        neigh_funcs = [self._graph.rx_graph.out_edges]

        if self._include_sources:
            neigh_funcs.append(self._graph.rx_graph.in_edges)

        # only edges are filtered return nodes that pass edge filters
        sources = []
        targets = []
        data = {k: [] for k in self._graph.edge_attr_keys()}
        data[DEFAULT_ATTR_KEYS.EDGE_ID] = []

        check_node_ids = None
        if self._node_ids is not None and not (self._include_targets or self._include_sources):
            check_node_ids = set(node_ids)

        # TODO: at this point I think we are better creating a rx subgraph
        # and using the filter method
        for node_id in node_ids:
            for nf in neigh_funcs:
                for src, tgt, attr in nf(node_id):
                    if _filter_func(attr):
                        if check_node_ids is not None:
                            if src not in check_node_ids or tgt not in check_node_ids:
                                continue

                        sources.append(src)
                        targets.append(tgt)
                        for k in data.keys():
                            data[k].append(attr.get(k, None))

        for k in data.keys():
            schema = self._graph._edge_attr_schemas()[k]
            s = pl.Series(name=k, values=data[k], dtype=schema.dtype)
            s = _maybe_fill_null(s, schema)
            data[k] = s

        data[DEFAULT_ATTR_KEYS.EDGE_SOURCE] = pl.Series(
            name=DEFAULT_ATTR_KEYS.EDGE_SOURCE, values=sources, dtype=pl.Int64
        )
        data[DEFAULT_ATTR_KEYS.EDGE_TARGET] = pl.Series(
            name=DEFAULT_ATTR_KEYS.EDGE_TARGET, values=targets, dtype=pl.Int64
        )

        df = pl.DataFrame(data)
        return df

    @cache_method
    def edge_ids(self) -> list[int]:
        return self._edge_attrs()[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()

    @cache_method
    def node_attrs(
        self,
        attr_keys: list[str] | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        return self._graph._node_attrs_from_node_ids(
            node_ids=self.node_ids(),
            attr_keys=attr_keys,
            unpack=unpack,
        )

    @cache_method
    def edge_attrs(
        self,
        attr_keys: list[str] | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        df = self._edge_attrs()
        if df.is_empty():
            return df

        if attr_keys is None:
            attr_keys = self._graph.edge_attr_keys()

        attr_keys = [
            DEFAULT_ATTR_KEYS.EDGE_ID,
            DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            DEFAULT_ATTR_KEYS.EDGE_TARGET,
            *attr_keys,
        ]
        attr_keys = list(dict.fromkeys(attr_keys))

        df = df.select(attr_keys)
        if unpack:
            df = unpack_array_attrs(df)
        return df

    @cache_method
    def subgraph(
        self,
        node_attr_keys: Sequence[str] | None = None,
        edge_attr_keys: Sequence[str] | None = None,
    ) -> "GraphView":
        from tracksdata.graph._graph_view import GraphView

        node_ids = self.node_ids()

        rx_graph, node_map = self._graph._rx_subgraph_with_nodemap(node_ids)
        if self._edge_attr_comps:
            _filter_func = _create_filter_func(self._edge_attr_comps, self._graph._edge_attr_schemas())
            for src, tgt, attr in rx_graph.weighted_edge_list():
                if not _filter_func(attr):
                    rx_graph.remove_edge(src, tgt)

        graph_view = GraphView(
            rx_graph,
            node_map_to_root=dict(node_map.items()),
            root=self._graph,
            node_attr_keys=node_attr_keys,
            edge_attr_keys=edge_attr_keys,
        )

        return graph_view


class RustWorkXGraph(BaseGraph):
    """
    High-performance in-memory graph implementation using rustworkx.

    RustWorkXGraph provides a fast, memory-efficient graph backend built on
    rustworkx (a Rust-based graph library). It stores nodes and edges in memory
    with their attributes, making it suitable for moderate-sized graphs that
    fit in RAM. This implementation offers excellent performance for graph
    algorithms and is the recommended choice for most tracking applications.

    Parameters
    ----------
    rx_graph : rx.PyDiGraph | None, optional
        An existing rustworkx directed graph to wrap. If None, creates a new
        empty graph.

    Attributes
    ----------
    rx_graph : rx.PyDiGraph
        The underlying rustworkx directed graph object.
    _time_to_nodes : dict[int, list[int]]
        Mapping from time points to lists of node IDs at that time.
    _node_attr_keys : list[str]
        List of available node attribute keys.
    _edge_attr_keys : list[str]
        List of available edge attribute keys.

    See Also
    --------
    [SQLGraph][tracksdata.graph.SQLGraph]:
        Database-backed graph implementation for larger datasets.

    Examples
    --------
    Create an empty graph:

    ```python
    from tracksdata.graph import RustWorkXGraph

    graph = RustWorkXGraph()
    ```

    Add nodes and edges:

    ```python
    node_id = graph.add_node({"t": 0, "x": 10.5, "y": 20.3})
    edge_id = graph.add_edge(source_id, target_id, {"weight": 0.8})
    ```

    Filter nodes by attributes:

    ```python
    from tracksdata.attrs import NodeAttr

    node_ids = graph.filter(NodeAttr("t") == 0).node_ids()
    ```

    Create subgraphs:

    ```python
    subgraph = graph.filter(NodeAttr("t") == 0).subgraph()
    ```
    """

    def __init__(self, rx_graph: rx.PyDiGraph | None = None) -> None:
        super().__init__()

        self._time_to_nodes: dict[int, list[int]] = {}
        self.__node_attr_schemas: dict[str, AttrSchema] = {}
        self.__edge_attr_schemas: dict[str, AttrSchema] = {}
        self._overlaps: list[list[int, 2]] = []

        # Add default node attributes with inferred schemas
        self.__node_attr_schemas[DEFAULT_ATTR_KEYS.T] = AttrSchema(
            key=DEFAULT_ATTR_KEYS.T,
            dtype=pl.Int32,
        )
        self.__node_attr_schemas[DEFAULT_ATTR_KEYS.NODE_ID] = AttrSchema(
            key=DEFAULT_ATTR_KEYS.NODE_ID,
            dtype=pl.Int64,
        )

        for key in [DEFAULT_ATTR_KEYS.EDGE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]:
            self.__edge_attr_schemas[key] = AttrSchema(
                key=key,
                dtype=pl.Int32 if key == DEFAULT_ATTR_KEYS.EDGE_ID else pl.Int64,
            )

        if rx_graph is None:
            self._graph = rx.PyDiGraph(attrs={})
        else:
            self._graph = rx_graph

            if self._graph.attrs is None:
                self._graph.attrs = {}

            elif not isinstance(self._graph.attrs, dict):
                LOG.warning(
                    "previous attribute %s will be added to key 'old_attrs' of `graph.metadata()`",
                    self._graph.attrs,
                )
                self._graph.attrs = {
                    "old_attrs": self._graph.attrs,
                }

            # Process nodes: build time index and infer schemas
            first_node_attrs = None
            for node_id in self._graph.node_indices():
                node_attrs = self._graph[node_id]
                try:
                    t = node_attrs[DEFAULT_ATTR_KEYS.T]
                except KeyError as e:
                    raise ValueError(
                        f"Node attributes must have a '{DEFAULT_ATTR_KEYS.T}' key. Got {node_attrs.keys()}"
                    ) from e

                self._time_to_nodes.setdefault(int(t), []).append(node_id)

                # Store first node's attrs to infer schemas
                if first_node_attrs is None:
                    first_node_attrs = node_attrs

            # Infer node schemas from first node
            if first_node_attrs is not None:
                for key, value in first_node_attrs.items():
                    if key == DEFAULT_ATTR_KEYS.NODE_ID:
                        continue
                    try:
                        dtype = pl.Series([value]).dtype
                    except (ValueError, TypeError):
                        # If polars can't infer dtype (e.g., for complex objects), use Object
                        dtype = pl.Object
                    self.__node_attr_schemas[key] = AttrSchema(key=key, dtype=dtype)

            # Process edges: set edge IDs and infer schemas
            edge_idx_map = self._graph.edge_index_map()
            first_edge_attrs = None
            for edge_idx, (_, _, attr) in edge_idx_map.items():
                attr[DEFAULT_ATTR_KEYS.EDGE_ID] = edge_idx
                # Store first edge's attrs to infer schemas
                if first_edge_attrs is None:
                    first_edge_attrs = attr

            # Infer edge schemas from first edge
            if first_edge_attrs is not None:
                for key, value in first_edge_attrs.items():
                    # TODO: check if EDGE_SOURCE and EDGE_TARGET should be also ignored or in the schema
                    if key == DEFAULT_ATTR_KEYS.EDGE_ID:
                        continue
                    try:
                        dtype = pl.Series([value]).dtype
                    except (ValueError, TypeError):
                        # If polars can't infer dtype (e.g., for complex objects), use Object
                        dtype = pl.Object
                    self.__edge_attr_schemas[key] = AttrSchema(key=key, dtype=dtype)

    def _node_attr_schemas(self) -> dict[str, AttrSchema]:
        return self.__node_attr_schemas

    def _edge_attr_schemas(self) -> dict[str, AttrSchema]:
        return self.__edge_attr_schemas

    @property
    def rx_graph(self) -> rx.PyDiGraph:
        return self._graph

    def filter(
        self,
        *attr_filters: AttrComparison,
        node_ids: Sequence[int] | None = None,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> RXFilter:
        return RXFilter(
            *attr_filters,
            graph=self,
            node_ids=node_ids,
            include_targets=include_targets,
            include_sources=include_sources,
        )

    def add_node(
        self,
        attrs: dict[str, Any],
        validate_keys: bool = True,
        index: int | None = None,
    ) -> int:
        """
        Add a node to the graph at time t.

        Parameters
        ----------
        attrs : Any
            The attributes of the node to be added, must have a "t" key.
            The keys of the attributes will be used as the attributes of the node.
            For example:
            ```python
            graph.add_node(dict(t=0, label="A", intensity=100))
            ```
        validate_keys : bool
            Whether to check if the attributes keys are valid.
            If False, the attributes keys will not be checked,
            useful to speed up the operation when doing bulk insertions.
        index : int | None
            Optional node index. RustWorkXGraph does not support custom indices
            and will raise an error if this parameter is provided.
        """
        if index is not None:
            raise ValueError("RustWorkXGraph does not support custom node indices. Use IndexedRXGraph instead.")

        # avoiding copying attributes on purpose, it could be a problem in the future
        if validate_keys:
            self._validate_attributes(attrs, self.node_attr_keys(), "node")

            if "t" not in attrs:
                raise ValueError(f"Node attributes must have a 't' key. Got {attrs.keys()}")

        node_id = self.rx_graph.add_node(attrs)
        self._time_to_nodes.setdefault(attrs["t"], []).append(node_id)
        self.node_added.emit_fast(node_id)
        return node_id

    def bulk_add_nodes(self, nodes: list[dict[str, Any]], indices: list[int] | None = None) -> list[int]:
        """
        Faster method to add multiple nodes to the graph with less overhead and fewer checks.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            The data of the nodes to be added.
            The keys of the data will be used as the attributes of the nodes.
            Must have "t" key.
        indices : list[int] | None
            Optional list of node indices. RustWorkXGraph does not support custom indices
            and will raise an error if this parameter is provided.

        Returns
        -------
        list[int]
            The IDs of the added nodes.
        """
        if indices is not None:
            raise ValueError("RustWorkXGraph does not support custom node indices. Use IndexedRXGraph instead.")

        node_indices = list(self.rx_graph.add_nodes_from(nodes))
        for node, index in zip(nodes, node_indices, strict=True):
            self._time_to_nodes.setdefault(node["t"], []).append(index)

        # checking if it has connections to reduce overhead
        if is_signal_on(self.node_added):
            for node_id in node_indices:
                self.node_added.emit_fast(node_id)

        return node_indices

    def remove_node(self, node_id: int) -> None:
        """
        Remove a node from the graph.

        This method removes the specified node and all edges connected to it
        (both incoming and outgoing edges). Also updates the time_to_nodes mapping.

        Parameters
        ----------
        node_id : int
            The ID of the node to remove.

        Raises
        ------
        ValueError
            If the node_id does not exist in the graph.
        """
        if node_id not in self.rx_graph.node_indices():
            raise ValueError(f"Node {node_id} does not exist in the graph.")

        self.node_removed.emit_fast(node_id)

        # Get the time value before removing the node
        t = self.rx_graph[node_id]["t"]

        # Remove the node from the graph (this also removes all connected edges)
        self.rx_graph.remove_node(node_id)

        # Update the time_to_nodes mapping
        self._time_to_nodes[t].remove(node_id)
        # Clean up empty time entries
        if not self._time_to_nodes[t]:
            del self._time_to_nodes[t]

        # Remove from overlaps if present
        if self._overlaps is not None:
            self._overlaps = [overlap for overlap in self._overlaps if node_id != overlap[0] and node_id != overlap[1]]

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        attrs: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        """
        Add an edge to the graph.

        Parameters
        ----------
        source_id : int
            The ID of the source node.
        target_id : int
            The ID of the target node.
        attrs : dict[str, Any]
            The attributes of the edge to be added.
            The keys of the attributes will be used as the attributes of the edge.
        validate_keys : bool
            Whether to check if the attributes keys are valid.
            If False, the attributes keys will not be checked,
            useful to speed up the operation when doing bulk insertions.
        """
        if validate_keys:
            self._validate_attributes(attrs, self.edge_attr_keys(), "edge")
        edge_id = self.rx_graph.add_edge(source_id, target_id, attrs)
        attrs[DEFAULT_ATTR_KEYS.EDGE_ID] = edge_id
        return edge_id

    def bulk_add_edges(self, edges: list[dict[str, Any]], return_ids: bool = False) -> list[int] | None:
        """
        Faster method to add multiple edges to the graph with less overhead and fewer checks.

        Parameters
        ----------
        edges : list[dict[str, Any]]
            The data of the edges to be added.
            The keys of the data will be used as the attributes of the edges.
            Must have "source_id" and "target_id" keys.
        return_ids : bool
            Whether to return the IDs of the added edges.
            If False, the edges are added and the method returns None.

        Examples
        --------
        ```python
        edges = [
            {"source_id": 1, "target_id": 2, "weight": 0.8},
            {"source_id": 2, "target_id": 3, "weight": 0.9"},
        ]
        graph.bulk_add_edges(edges)
        ```

        Returns
        -------
        list[int] | None
            The IDs of the added edges.
        """
        # saving for historical reasons, iterating over edges is faster than using rx.add_edges_from
        # edges_data = [(d.pop(DEFAULT_ATTR_KEYS.EDGE_SOURCE), d.pop(DEFAULT_ATTR_KEYS.EDGE_TARGET), d) for d in edges]
        # indices = self.rx_graph.add_edges_from(edges_data)
        # for i, d in zip(indices, edges, strict=True):
        #     d[DEFAULT_ATTR_KEYS.EDGE_ID] = i
        # return indices
        return super().bulk_add_edges(edges, return_ids=return_ids)

    def remove_edge(
        self,
        source_id: int | None = None,
        target_id: int | None = None,
        *,
        edge_id: int | None = None,
    ) -> None:
        """
        Remove an edge by ID or by endpoints.
        """
        if edge_id is None:
            if source_id is None or target_id is None:
                raise ValueError("Provide either edge_id or both source_id and target_id.")
            try:
                self.rx_graph.remove_edge(source_id, target_id)
            except rx.NoEdgeBetweenNodes as e:
                raise ValueError(f"Edge {source_id}->{target_id} does not exist in the graph.") from e
        else:
            edge_map = self.rx_graph.edge_index_map()
            if edge_id not in edge_map:
                raise ValueError(f"Edge {edge_id} does not exist in the graph.")
            src, tgt, _ = edge_map[edge_id]
            self.rx_graph.remove_edge(src, tgt)

    def add_overlap(
        self,
        source_id: int,
        target_id: int,
    ) -> int:
        """
        Add a new overlap to the graph.
        Overlapping nodes are mutually exclusive.

        Parameters
        ----------
        source_id : int
            The ID of the source node.
        target_id : int
            The ID of the target node.

        Returns
        -------
        int
            The ID of the added overlap.
        """
        self._overlaps.append([source_id, target_id])
        return len(self._overlaps) - 1

    def overlaps(
        self,
        node_ids: list[int] | None = None,
    ) -> list[list[int, 2]]:
        """
        Get the overlaps between the nodes in `node_ids`.
        If `node_ids` is None, all nodes are used.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the nodes to get the overlaps for.
            If None, all nodes are used.

        Returns
        -------
        list[list[int, 2]]
            The overlaps between the nodes in `node_ids`.
        """
        if len(self._overlaps) == 0:
            return []

        if node_ids is None:
            return self._overlaps

        node_ids = np.asarray(node_ids, dtype=int)
        overlaps_arr = np.asarray(self._overlaps, dtype=int)

        is_in_source = np.isin(overlaps_arr[:, 0], node_ids)
        is_in_target = np.isin(overlaps_arr[:, 1], node_ids)
        return overlaps_arr[is_in_source & is_in_target].tolist()

    def has_overlaps(self) -> bool:
        """
        Check if the graph has any overlaps.

        Returns
        -------
        bool
            True if the graph has any overlaps, False otherwise.
        """
        return len(self.overlaps()) > 0

    def _get_neighbors(
        self,
        neighbors_func: Callable[[rx.PyDiGraph, int], rx.NodeIndices],
        node_ids: list[int] | int | None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
        """
        Get the predecessors or sucessors of a list of nodes.
        See more information below.
        """
        single_node = False
        rx_graph = self.rx_graph
        if node_ids is None:
            node_ids = list(rx_graph.node_indices())
        elif isinstance(node_ids, int):
            node_ids = [node_ids]
            single_node = True

        if not return_attrs and attr_keys is not None:
            LOG.warning("attr_keys is ignored when return_attrs is False.")

        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]
        valid_schema = None
        neighbors: dict[int, list[int]] | dict[int, pl.DataFrame] = {}
        for node_id in node_ids:
            neighbors_indices = neighbors_func(rx_graph, node_id)
            if not return_attrs:
                neighbors[node_id] = [int(idx) for idx in neighbors_indices]
            else:
                neighbors_data: list[dict[str, Any]] = [rx_graph[i] for i in neighbors_indices]

                if attr_keys is not None:
                    neighbors_data = [
                        {k: neigh_attr[k] for k in attr_keys if k != DEFAULT_ATTR_KEYS.NODE_ID}
                        for neigh_attr in neighbors_data
                    ]

                if len(neighbors_data) > 0:
                    df = pl.DataFrame(neighbors_data)
                    if attr_keys is None or DEFAULT_ATTR_KEYS.NODE_ID in attr_keys:
                        df = df.with_columns(
                            pl.Series(DEFAULT_ATTR_KEYS.NODE_ID, np.asarray(neighbors_indices, dtype=int)),
                        )
                    neighbors[node_id] = df
                    valid_schema = neighbors[node_id].schema

        if not return_attrs:
            default_value = []
        elif valid_schema is None:
            default_value = pl.DataFrame()
        else:
            default_value = pl.DataFrame(schema=valid_schema)

        if single_node:
            return neighbors.get(node_ids[0], default_value)
        else:
            for node_id in node_ids:
                neighbors.setdefault(node_id, default_value)

            return neighbors

    def successors(
        self,
        node_ids: list[int] | int | None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
        """
        Get the sucessors of a list of nodes.

        Parameters
        ----------
        node_ids : list[int] | int | None
            The IDs of the nodes to get the sucessors for.
            If None, all nodes are used.
        attr_keys : Sequence[str] | str | None
            The attribute keys to retrieve when ``return_attrs`` is True.
            If None, all attributes are included.
        return_attrs : bool, default False
            Whether to return the attributes DataFrame. When False only successor
            node IDs are returned.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]
            When ``return_attrs`` is True, returns a DataFrame for a single node or a dictionary
            mapping each node ID to a DataFrame of neighbor attributes. Otherwise returns a list
            of neighbor node IDs for a single node or a dictionary mapping each node ID to its
            neighbor ID list.
        """
        return self._get_neighbors(
            rx.PyDiGraph.successor_indices,
            node_ids,
            attr_keys,
            return_attrs=return_attrs,
        )

    def predecessors(
        self,
        node_ids: list[int] | int | None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
        """
        Get the predecessors of a list of nodes.

        Parameters
        ----------
        node_ids : list[int] | int | None
            The IDs of the nodes to get the predecessors for.
            If None, all nodes are used.
        attr_keys : Sequence[str] | str | None
            The attribute keys to retrieve when ``return_attrs`` is True.
            If None, all attributes are included.
        return_attrs : bool, default False
            Whether to return the attributes DataFrame. When False only predecessor
            node IDs are returned.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]
            When ``return_attrs`` is True, returns a DataFrame for a single node or a dictionary
            mapping each node ID to a DataFrame of neighbor attributes. Otherwise returns a list
            of neighbor node IDs for a single node or a dictionary mapping each node ID to its
            neighbor ID list.
        """
        return self._get_neighbors(
            rx.PyDiGraph.predecessor_indices,
            node_ids,
            attr_keys,
            return_attrs=return_attrs,
        )

    def _filter_nodes_by_attrs(
        self,
        *attrs: AttrComparison,
        node_ids: Sequence[int] | None = None,
    ) -> list[int]:
        """
        Filter nodes by attributes.

        Parameters
        ----------
        *attrs : AttrComparison
            The attributes to filter by, for example:
        node_ids : list[int] | None
            The IDs of the nodes to include in the filter.
            If None, all nodes are used.

        Returns
        -------
        list[int]
            The IDs of the filtered nodes.
        """
        rx_graph = self.rx_graph
        node_map = None
        # entire graph
        attrs, time = _pop_time_eq(attrs)
        selected_nodes = None

        if time is not None:
            selected_nodes = self._time_to_nodes.get(time, [])

            if node_ids is not None:
                selected_nodes = np.intersect1d(selected_nodes, node_ids).tolist()

            if len(attrs) == 0:
                return selected_nodes

        elif node_ids is not None:
            selected_nodes = node_ids

        if selected_nodes is not None:
            # subgraph of selected nodes
            rx_graph, node_map = rx_graph.subgraph_with_nodemap(selected_nodes)

        _filter_func = _create_filter_func(attrs, self._node_attr_schemas())

        if node_map is None:
            return list(rx_graph.filter_nodes(_filter_func))
        else:
            return [node_map[n] for n in rx_graph.filter_nodes(_filter_func)]

    def node_ids(self) -> list[int]:
        """
        Get the IDs of all nodes in the graph.
        """
        return [int(i) for i in self.rx_graph.node_indices()]

    def edge_ids(self) -> list[int]:
        """
        Get the IDs of all edges in the graph.
        """
        return [int(i) for i in self.rx_graph.edge_indices()]

    def time_points(self) -> list[int]:
        """
        Get the unique time points in the graph.
        """
        return list(self._time_to_nodes.keys())

    def node_attr_keys(self, return_ids: bool = False) -> list[str]:
        """
        Get the keys of the attributes of the nodes.

        Parameters
        ----------
        return_ids : bool, optional
            Whether to include NODE_ID in the returned keys. Defaults to False.
            If True, NODE_ID will be included in the list.
        """
        keys = list(self._node_attr_schemas().keys())
        if not return_ids and DEFAULT_ATTR_KEYS.NODE_ID in keys:
            keys.remove(DEFAULT_ATTR_KEYS.NODE_ID)
        return keys

    def edge_attr_keys(self, return_ids: bool = False) -> list[str]:
        """
        Get the keys of the attributes of the edges.

        Parameters
        ----------
        return_ids : bool, optional
            Whether to include EDGE_ID, EDGE_SOURCE, and EDGE_TARGET in the returned keys.
            Defaults to False. If True, these ID fields will be included in the list.
        """
        keys = list(self.__edge_attr_schemas.keys())
        if not return_ids:
            for id_key in [DEFAULT_ATTR_KEYS.EDGE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]:
                keys.remove(id_key)
        return keys

    def add_node_attr_key(
        self,
        key_or_schema: str | AttrSchema,
        dtype: pl.DataType | None = None,
        default_value: Any = None,
    ) -> None:
        """
        Add a new attribute key to the graph.
        All existing nodes will have the default value for the new attribute key.

        Parameters
        ----------
        key_or_schema : str | AttrSchema
            Either the key name (str) or an AttrSchema object containing key, dtype, and default_value.
        dtype : pl.DataType | None
            The polars data type for this attribute. Required when key_or_schema is a string.
        default_value : Any, optional
            The default value for existing nodes for the new attribute key.
            If None, will be inferred from dtype.
        """
        # Process arguments and create validated schema
        schema = process_attr_key_args(key_or_schema, dtype, default_value, self.__node_attr_schemas)

        # Store schema
        self.__node_attr_schemas[schema.key] = schema

    def remove_node_attr_key(self, key: str) -> None:
        """
        Remove an existing node attribute key from the graph.
        """
        if key not in self.node_attr_keys():
            raise ValueError(f"Node attribute key {key} does not exist")

        if key in (DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.T):
            raise ValueError(f"Cannot remove required node attribute key {key}")

        del self.__node_attr_schemas[key]
        for node_attr in self.rx_graph.nodes():
            node_attr.pop(key, None)

    def add_edge_attr_key(
        self,
        key_or_schema: str | AttrSchema,
        dtype: pl.DataType | None = None,
        default_value: Any = None,
    ) -> None:
        """
        Add a new attribute key to the graph.
        All existing edges will have the default value for the new attribute key.

        Parameters
        ----------
        key_or_schema : str | AttrSchema
            Either the key name (str) or an AttrSchema object containing key, dtype, and default_value.
        dtype : pl.DataType | None
            The polars data type for this attribute. Required when key_or_schema is a string.
        default_value : Any, optional
            The default value for existing edges for the new attribute key.
            If None, will be inferred from dtype.
        """
        # Process arguments and create validated schema
        schema = process_attr_key_args(key_or_schema, dtype, default_value, self.__edge_attr_schemas)

        # Store schema
        self.__edge_attr_schemas[schema.key] = schema

    def remove_edge_attr_key(self, key: str) -> None:
        """
        Remove an existing edge attribute key from the graph.
        """
        if key not in self.edge_attr_keys():
            raise ValueError(f"Edge attribute key {key} does not exist")

        del self.__edge_attr_schemas[key]
        for edge_attr in self.rx_graph.edges():
            edge_attr.pop(key, None)

    def _node_attrs_from_node_ids(
        self,
        *,
        node_ids: list[int] | None = None,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        """
        Get the attributes of the nodes as a polars DataFrame.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the nodes to get the attributesfor.
            If None, all nodes are used.
        attr_keys : Sequence[str] | None
            The attribute keys to get.
            If None, all the attributes of the first node are used.
        unpack : bool
            Whether to unpack array attributes into multiple scalar attributes.

        Returns
        -------
        pl.DataFrame
            A polars DataFrame with the attributes of the nodes.
        """
        rx_graph = self.rx_graph
        # If no node_ids provided, use all nodes
        if node_ids is None:
            node_ids = list(rx_graph.node_indices())

        if attr_keys is None:
            attr_keys = self.node_attr_keys(return_ids=True)

        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        node_attr_schemas = self._node_attr_schemas()
        pl_schema = {k: node_attr_schemas[k].dtype for k in attr_keys}

        if len(node_ids) == 0:
            return pl.DataFrame({key: [] for key in attr_keys}, schema=pl_schema)

        # making them unique
        attr_keys = list(dict.fromkeys(attr_keys))

        # Create columns directly instead of building intermediate dictionaries
        columns = {key: [] for key in attr_keys}

        if DEFAULT_ATTR_KEYS.NODE_ID in attr_keys:
            columns[DEFAULT_ATTR_KEYS.NODE_ID] = pl.Series(
                name=DEFAULT_ATTR_KEYS.NODE_ID,
                values=node_ids,
                dtype=pl.Int64,
            )
            attr_keys.remove(DEFAULT_ATTR_KEYS.NODE_ID)

        # Build columns in a vectorized way
        for node_id in node_ids:
            node_data = rx_graph[node_id]
            for key in attr_keys:
                columns[key].append(node_data.get(key))

        for key in attr_keys:
            schema = node_attr_schemas[key]
            s = pl.Series(name=key, values=columns[key], dtype=schema.dtype)
            s = _maybe_fill_null(s, schema)
            columns[key] = s

        # Create DataFrame and set node_id as index in one shot
        df = pl.DataFrame(columns)

        if unpack:
            df = unpack_array_attrs(df)

        return df

    def node_attrs(
        self,
        *,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        """
        Get the attributes of the nodes as a polars DataFrame.
        """
        return self._node_attrs_from_node_ids(attr_keys=attr_keys, unpack=unpack)

    def edge_attrs(
        self,
        *,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        """
        Get the attributes of the edges as a polars DataFrame.

        Parameters
        ----------
        attr_keys : Sequence[str] | str | None
            The attribute keys to get.
            If None, all attributesare used.
        unpack : bool
            Whether to unpack array attributesinto multiple scalar attributes.
        """
        if attr_keys is None:
            attr_keys = self.edge_attr_keys()

        attr_keys = [DEFAULT_ATTR_KEYS.EDGE_ID, *attr_keys]
        attr_keys = list(dict.fromkeys(attr_keys))

        rx_graph = self.rx_graph

        edge_map = rx_graph.edge_index_map()
        if len(edge_map) == 0:
            return pl.DataFrame(
                {
                    key: []
                    for key in [
                        *attr_keys,
                        DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                        DEFAULT_ATTR_KEYS.EDGE_TARGET,
                    ]
                }
            )

        source, target, data = zip(*edge_map.values(), strict=False)

        columns = {key: [] for key in attr_keys}

        for row in data:
            for key in attr_keys:
                columns[key].append(row.get(key))

        columns[DEFAULT_ATTR_KEYS.EDGE_SOURCE] = source
        columns[DEFAULT_ATTR_KEYS.EDGE_TARGET] = target

        for key in attr_keys:
            schema = self._edge_attr_schemas()[key]
            s = pl.Series(name=key, values=columns[key], dtype=schema.dtype)
            s = _maybe_fill_null(s, schema)
            columns[key] = s

        df = pl.DataFrame(columns)
        if unpack:
            df = unpack_array_attrs(df)
        return df

    def num_edges(self) -> int:
        """
        The number of edges in the graph.
        """
        return self._graph.num_edges()

    def num_nodes(self) -> int:
        """
        The number of nodes in the graph.
        """
        return self._graph.num_nodes()

    def update_node_attrs(
        self,
        *,
        attrs: dict[str, Any],
        node_ids: Sequence[int] | None = None,
    ) -> None:
        """
        Update the attributes of the nodes.

        Parameters
        ----------
        attrs : dict[str, Any]
            The attributes to update.
        node_ids : Sequence[int] | None
            The IDs of the nodes to update or None to update all nodes.
        """
        if node_ids is None:
            node_ids = self.node_ids()

        for key, value in attrs.items():
            if key not in self.node_attr_keys():
                raise ValueError(f"Node attribute key '{key}' not found in graph. Expected '{self.node_attr_keys()}'")

            if not np.isscalar(value) and len(attrs[key]) != len(node_ids):
                raise ValueError(f"Attribute '{key}' has wrong size. Expected {len(node_ids)}, got {len(attrs[key])}")

        for key, value in attrs.items():
            if np.isscalar(value):
                value = [value] * len(node_ids)

            for node_id, v in zip(node_ids, value, strict=False):
                self._graph[node_id][key] = v

    def update_edge_attrs(
        self,
        *,
        attrs: dict[str, Any],
        edge_ids: Sequence[int] | None = None,
    ) -> None:
        """
        Update the attributes of the edges.

        Parameters
        ----------
        attrs : dict[str, Any]
            Attributes to be updated.
        edge_ids : Sequence[int] | None
            The IDs of the edges to update or None to update all edges.
        """
        if edge_ids is None:
            edge_ids = self.edge_ids()

        size = len(edge_ids)
        for key, value in attrs.items():
            if key not in self.edge_attr_keys():
                raise ValueError(f"Edge attribute key '{key}' not found in graph. Expected '{self.edge_attr_keys()}'")

            if np.isscalar(value):
                attrs[key] = [value] * size

            elif len(attrs[key]) != size:
                raise ValueError(f"Attribute '{key}' has wrong size. Expected {size}, got {len(attrs[key])}")

        edge_map = self._graph.edge_index_map()

        for i, edge_id in enumerate(edge_ids):
            edge_attr = edge_map[edge_id][2]  # 0=source, 1=target, 2=attributes
            for key, value in attrs.items():
                edge_attr[key] = value[i]

    def assign_tracklet_ids(
        self,
        output_key: str = DEFAULT_ATTR_KEYS.TRACKLET_ID,
        reset: bool = True,
        tracklet_id_offset: int | None = None,
        node_ids: list[int] | None = None,
        return_id_update: bool = False,
    ) -> rx.PyDiGraph | tuple[rx.PyDiGraph, pl.DataFrame]:
        # local import to avoid circular import
        from tracksdata.functional._rx import _assign_tracklet_ids

        if node_ids is not None:
            track_node_ids = set(self.tracklet_nodes(node_ids))
            return (
                self.filter(node_ids=list(track_node_ids))
                .subgraph(node_attr_keys=[output_key], edge_attr_keys=[])
                .assign_tracklet_ids(
                    output_key=output_key,
                    reset=reset,
                    tracklet_id_offset=tracklet_id_offset,
                    return_id_update=return_id_update,
                )
            )
        else:
            if output_key not in self.node_attr_keys():
                previous_id_df = None
                self.add_node_attr_key(output_key, pl.Int64, -1)
                if tracklet_id_offset is None:
                    tracklet_id_offset = 1
            elif reset:
                if return_id_update:
                    previous_id_df = self.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, output_key])
                else:
                    previous_id_df = None
                self.update_node_attrs(attrs={output_key: -1})
                if tracklet_id_offset is None:
                    tracklet_id_offset = 1
            else:
                previous_id_df = self.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, output_key])
                if tracklet_id_offset is None:
                    tracklet_id_offset = max(previous_id_df[output_key].max(), 0) + 1

            try:
                track_node_ids, tracklet_ids, tracks_graph = _assign_tracklet_ids(self.rx_graph, tracklet_id_offset)
            except RuntimeError as e:
                raise RuntimeError(
                    "Are you sure this graph is a valid lineage graph?\n"
                    "This function expects a solved graph.\n"
                    "Often used from `graph.subgraph(edge_attr_filter={'solution': True})`"
                ) from e

            # Converting to list of int for SQLGraph compatibility (See below)
            tracklet_ids = tracklet_ids.tolist()

            # For the IndexedRXGraph, we need to map the track_node_ids to the external node ids
            if hasattr(self, "_map_to_external"):
                track_node_ids = self._map_to_external(track_node_ids)  # type: ignore

            # mapping to already existing track IDs as much as possible
            id_update_df = None
            if previous_id_df is not None:
                # Entering this block means that either of
                # (`return_id_update == True` and the output_key already existed) or we are reusing existing IDs.
                # So we compute the id_update_df here.
                new_id_df = pl.DataFrame({DEFAULT_ATTR_KEYS.NODE_ID: track_node_ids, output_key + "_new": tracklet_ids})
                id_update_df = new_id_df.join(
                    previous_id_df,
                    left_on=DEFAULT_ATTR_KEYS.NODE_ID,
                    right_on=DEFAULT_ATTR_KEYS.NODE_ID,
                    how="left",
                )
                if reset is False:
                    id_update_df_filtered = id_update_df.filter(pl.col(output_key) != -1)
                    if id_update_df_filtered.height > 0:
                        tracklet_id_map = id_update_df_filtered.unique(output_key + "_new", keep="first").unique(
                            output_key, keep="first"
                        )
                        tracklet_id_map = dict(
                            zip(tracklet_id_map[output_key + "_new"], tracklet_id_map[output_key], strict=True)
                        )
                        # Ensure that the result is a list of integers (using numpy integer causes issues with SQLGraph)
                        # Later on, we will make it safe to use numpy integers everywhere for updating attributes.
                        tracklet_ids = [int(tracklet_id_map.get(tid, tid)) for tid in tracklet_ids]  # type: ignore
                        # Update the value with the reused IDs
                        id_update_df = id_update_df.with_columns(pl.Series(output_key + "_new", tracklet_ids))

            self.update_node_attrs(
                node_ids=track_node_ids,  # type: ignore
                attrs={output_key: tracklet_ids},
            )
            # Return a dataframe with node_id, old_tracklet_id, new_tracklet_id if return_id_update is True
            if return_id_update:
                if id_update_df is None:
                    id_update_df = pl.DataFrame(
                        {DEFAULT_ATTR_KEYS.NODE_ID: track_node_ids, output_key + "_new": tracklet_ids}
                    )
                return tracks_graph, id_update_df
            else:
                return tracks_graph

    def in_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the in-degree of a list of nodes.
        """
        if node_ids is None:
            node_ids = self.node_ids()
        rx_graph = self.rx_graph
        if isinstance(node_ids, int):
            return rx_graph.in_degree(node_ids)
        return [rx_graph.in_degree(node_id) for node_id in node_ids]

    def out_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the out-degree of a list of nodes.
        """
        if node_ids is None:
            node_ids = self.node_ids()
        rx_graph = self.rx_graph
        if isinstance(node_ids, int):
            return rx_graph.out_degree(node_ids)
        return [rx_graph.out_degree(node_id) for node_id in node_ids]

    def contract_nodes(
        self,
        permanent_node_ids: Sequence[int],
    ) -> "GraphView":
        """
        Contract the graph to only include the given `permanent_node_ids`.
        Predecessor and sucessors of removed nodes are connected during contraction.

        Example:

        ```mermaid
        graph TD
            A((t=0)) --> B((t=1))
            A --> C((t=1))
            B --> D((t=2))
            C --> E((t=2))
            C --> F((t=2))
        ```

        After contraction only keeping nodes in `t=0` and `t=2`:
        ```mermaid
        graph TD
            A((t=0)) --> B((t=2))
            A --> C((t=2))
            A --> D((t=2))
        ```

        Parameters
        ----------
        permanent_node_ids : Sequence[int]
            The node ids to keep in the contracted graph.

        Returns
        -------
        GraphView
            A view of the contracted graph.
        """
        from tracksdata.graph._graph_view import GraphView

        all_node_ids = np.asarray(self.node_ids())
        selected_nodes_mask = np.isin(all_node_ids, permanent_node_ids)
        missing_node_ids = all_node_ids[~selected_nodes_mask]

        # must block multigraphs to avoid edge duplication
        rx_graph = rx.PyDiGraph(multigraph=False)
        new_indices = rx_graph.add_nodes_from(self.rx_graph.nodes())
        rx_graph.add_edges_from(self.rx_graph.weighted_edge_list())

        for node_id in missing_node_ids:
            rx_graph.remove_node_retain_edges(
                node_id,
                use_outgoing=True,
                condition=lambda *args: True,
            )

        node_map_to_root = dict(
            zip(
                np.asarray(new_indices)[selected_nodes_mask].tolist(),
                permanent_node_ids,
                strict=True,
            )
        )

        # I'm a bit concerned with the internal booking of indices of rustworkx
        # so I'm adding this sanity check
        assert len(node_map_to_root) == rx_graph.num_nodes()

        graph_view = GraphView(rx_graph, node_map_to_root=node_map_to_root, root=self)

        return graph_view

    def _rx_subgraph_with_nodemap(
        self,
        node_ids: Sequence[int] | None = None,
    ) -> tuple[rx.PyDiGraph, rx.NodeMap]:
        """
        Get a subgraph of the graph.

        Parameters
        ----------
        node_ids : Sequence[int] | None
            The node ids to include in the subgraph.

        Returns
        -------
        tuple[rx.PyDiGraph, rx.NodeMap]
            The subgraph and the node map.
        """
        rx_graph, node_map = self.rx_graph.subgraph_with_nodemap(node_ids)
        return rx_graph, node_map

    def has_node(self, node_id: int) -> bool:
        """
        Check if the graph has a node with the given id.
        """
        return self.rx_graph.has_node(node_id)

    def has_edge(self, source_id: int, target_id: int) -> bool:
        """
        Check if the graph has an edge between two nodes.
        """
        return self.rx_graph.has_edge(source_id, target_id)

    def edge_id(self, source_id: int, target_id: int) -> int:
        """
        Return the edge id between two nodes.
        """
        return self.rx_graph.get_edge_data(source_id, target_id)[DEFAULT_ATTR_KEYS.EDGE_ID]

    def metadata(self) -> dict[str, Any]:
        return self._graph.attrs

    def update_metadata(self, **kwargs) -> None:
        self._graph.attrs.update(kwargs)

    def remove_metadata(self, key: str) -> None:
        self._graph.attrs.pop(key, None)

    def edge_list(self) -> list[list[int, int]]:
        """
        Get the edge list of the graph.
        """
        return self.rx_graph.edge_list()


class IndexedRXGraph(MappedGraphMixin, RustWorkXGraph):
    """
    A graph with arbitrary node indices.

    Parameters
    ----------
    rx_graph : rx.PyDiGraph | None
        The rustworkx graph to index.
    node_id_map : dict[int, int] | None
        A map of external ids (arbitrary) to rx graph ids (0 to N-1), must be provided if `rx_graph` is provided.

    Examples
    --------
    ```python
    graph = IndexedRXGraph(rx_graph=rx_graph, node_id_map=node_id_map)
    ...
    graph = IndexedRXGraph()
    graph.add_node({"t": 0}, index=1355)
    ```
    """

    def __init__(
        self,
        rx_graph: rx.PyDiGraph | None = None,
        node_id_map: dict[int, int] | None = None,
    ) -> None:
        if rx_graph is not None and node_id_map is None:
            raise ValueError("`node_id_map` must be provided when `rx_graph` is provided")

        if rx_graph is None and node_id_map is not None:
            raise ValueError("`rx_graph` must be provided when `node_id_map` is provided")

        # Initialize RustWorkXGraph
        RustWorkXGraph.__init__(self, rx_graph)

        # Initialize MappedGraphMixin with inverted mapping (local -> external)
        inverted_map = None
        if node_id_map is not None:
            # Validate for duplicate values before inverting
            # This will raise bidict.ValueDuplicationError if there are duplicates
            inverted_map = bidict.bidict(node_id_map).inverse
            self._next_external_id = max(node_id_map.keys(), default=0) + 1
        else:
            self._next_external_id = 0

        MappedGraphMixin.__init__(self, inverted_map)

    def _get_next_available_external_id(self) -> int:
        """
        Get the next available external ID in O(1) time using an internal counter.

        Returns
        -------
        int
            The next available external ID.
        """
        next_id = self._next_external_id
        self._next_external_id += 1
        return next_id

    def supports_custom_indices(self) -> bool:
        return True

    def add_node(
        self,
        attrs: dict[str, Any],
        validate_keys: bool = True,
        index: int | None = None,
    ) -> int:
        """
        Add a node to the graph.

        Parameters
        ----------
        attrs : dict[str, Any]
            The attributes of the node.
        validate_keys : bool
            Whether to validate the keys of the attributes.
        index : int | None
            The index of the node. If None, the next available index will be used
            to avoid conflicts with existing node indices.

        Returns
        -------
        int
            The index of the node.
        """
        with self.node_added.blocked():
            node_id = super().add_node(attrs, validate_keys)

        if index is None:
            index = self._get_next_available_external_id()
        else:
            # Update counter if explicit index is higher to avoid future collisions
            self._next_external_id = max(self._next_external_id, index + 1)
        # Add mapping using mixin
        self._add_id_mapping(node_id, index)
        self.node_added.emit_fast(index)
        return index

    def bulk_add_nodes(
        self,
        nodes: list[dict[str, Any]],
        indices: list[int] | None = None,
    ) -> list[int]:
        """
        Add multiple nodes to the graph.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            The attributes of the nodes.
        indices : list[int] | None
            The indices of the nodes. If None, all nodes will get auto-generated indices.
            If provided, must have same length as nodes and all indices must be specified.

        Returns
        -------
        list[int]
            The indices of the nodes.
        """
        if len(nodes) == 0:
            return []

        self._validate_indices_length(nodes, indices)

        with self.node_added.blocked():
            graph_ids = super().bulk_add_nodes(nodes)

        if indices is None:
            # All nodes get auto-generated indices
            indices = [self._get_next_available_external_id() for _ in nodes]
        else:
            # All indices are explicitly provided
            if len(indices) != len(nodes):
                raise ValueError(f"Length of indices ({len(indices)}) must match nodes ({len(nodes)})")

            # Update counter to be after the highest explicit index
            self._next_external_id = max(self._next_external_id, max(indices) + 1)

        self._add_id_mappings(list(zip(graph_ids, indices, strict=True)))

        if is_signal_on(self.node_added):
            for index in indices:
                self.node_added.emit_fast(index)

        return indices

    def _rx_subgraph_with_nodemap(
        self,
        node_ids: Sequence[int] | None = None,
    ) -> tuple[rx.PyDiGraph, rx.NodeMap]:
        """
        Get a subgraph of the graph.

        Parameters
        ----------
        node_ids : Sequence[int] | None
            The node ids to include in the subgraph.

        Returns
        -------
        tuple[rx.PyDiGraph, rx.NodeMap]
            The subgraph and the node map.
        """
        # TODO: fix this, too many back and forth between world and graph ids within FilterRX
        node_ids = self._get_local_ids() if node_ids is None else self._map_to_local(node_ids)
        rx_graph, node_map = super()._rx_subgraph_with_nodemap(node_ids)
        node_map = {k: self._map_to_external(v) for k, v in node_map.items()}
        return rx_graph, node_map

    def node_ids(self) -> list[int]:
        """
        Get the node ids of the graph.

        Returns
        -------
        list[int]
            The node ids of the graph.
        """
        return self._get_external_ids()

    def _node_attrs_from_node_ids(
        self,
        *,
        node_ids: list[int] | None = None,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        node_ids = self._get_local_ids() if node_ids is None else self._map_to_local(node_ids)
        df = super()._node_attrs_from_node_ids(node_ids=node_ids, attr_keys=attr_keys, unpack=unpack)
        df = self._map_df_to_external(df, [DEFAULT_ATTR_KEYS.NODE_ID])
        return df

    def node_attrs(
        self,
        *,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        """
        Get the node attributes of the graph.

        Parameters
        ----------
        attr_keys : Sequence[str] | str | None
            The attributes to include in the subgraph.
        unpack : bool
            Whether to unpack the attributes.

        Returns
        -------
        pl.DataFrame
            The node attributes of the graph.
        """
        node_ids = self._get_local_ids()
        df = super()._node_attrs_from_node_ids(node_ids=node_ids, attr_keys=attr_keys, unpack=unpack)
        df = self._map_df_to_external(df, [DEFAULT_ATTR_KEYS.NODE_ID])
        return df

    def edge_attrs(
        self,
        *,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        """
        Get the edge attributes of the graph.

        Parameters
        ----------
        attr_keys : Sequence[str] | str | None
            The attributes to include in the subgraph.
        unpack : bool
            Whether to unpack the attributes.

        Returns
        -------
        pl.DataFrame
            The edge attributes of the graph.
        """
        node_ids = self._get_local_ids()
        df = super().filter(node_ids=node_ids).edge_attrs(attr_keys=attr_keys, unpack=unpack)
        df = self._map_df_to_external(df, [DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET])
        return df

    def in_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the in degree of the graph.

        Parameters
        ----------
        node_ids : list[int] | int | None
            The node ids to include in the subgraph.

        Returns
        -------
        list[int] | int
            The in degree of the graph.
        """
        node_ids = self._get_local_ids() if node_ids is None else self._map_to_local(node_ids)
        return super().in_degree(node_ids)

    def out_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the out degree of the graph.

        Parameters
        ----------
        node_ids : list[int] | int | None
            The node ids to include in the subgraph.

        Returns
        -------
        list[int] | int
            The out degree of the graph.
        """
        node_ids = self._get_local_ids() if node_ids is None else self._map_to_local(node_ids)
        return super().out_degree(node_ids)

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        attrs: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        """
        Add an edge to the graph.

        Parameters
        ----------
        source_id : int
            The source node id.
        target_id : int
            The target node id.
        attrs : dict[str, Any]
            The attributes of the edge.
        validate_keys : bool
            Whether to validate the keys of the attributes.

        Returns
        -------
        int
            The edge id.
        """
        source_id = self._map_to_local(source_id)
        target_id = self._map_to_local(target_id)
        return super().add_edge(source_id, target_id, attrs, validate_keys)

    def remove_edge(
        self,
        source_id: int | None = None,
        target_id: int | None = None,
        *,
        edge_id: int | None = None,
    ) -> None:
        """
        Remove an edge by endpoints (external IDs) or by edge_id.
        """
        if edge_id is not None:
            return super().remove_edge(edge_id=edge_id)
        if source_id is None or target_id is None:
            raise ValueError("Provide either edge_id or both source_id and target_id.")
        try:
            local_source = self._map_to_local(source_id)
            local_target = self._map_to_local(target_id)
        except KeyError as e:
            raise ValueError(f"Edge {source_id}->{target_id} does not exist in the graph.") from e
        try:
            return super().remove_edge(local_source, local_target)
        except ValueError as e:
            raise ValueError(f"Edge {source_id}->{target_id} does not exist in the graph.") from e

    def add_overlap(self, source_id: int, target_id: int) -> int:
        """
        Add an overlap to the graph.

        Parameters
        ----------
        source_id : int
            The source node id.
        target_id : int
            The target node id.

        Returns
        -------
        int
            The overlap id.
        """
        source_id = self._map_to_local(source_id)
        target_id = self._map_to_local(target_id)
        return super().add_overlap(source_id, target_id)

    def overlaps(self, node_ids: list[int] | int | None = None) -> list[list[int, 2]]:
        """
        Get the overlaps of the graph.

        Parameters
        ----------
        node_ids : list[int] | int | None
            The node ids to include in the subgraph.

        Returns
        -------
        list[list[int, 2]]
            The overlaps of the graph.
        """
        try:
            node_ids = self._get_local_ids() if node_ids is None else self._map_to_local(node_ids)
        except KeyError:
            LOG.warning(f"`node_ids` {node_ids} not found in index map.")
            return []
        overlaps = super().overlaps(node_ids)
        # Convert each pair of node IDs in the overlaps list
        if len(overlaps) > 0:
            overlaps = self._vectorized_map_to_external(np.asarray(overlaps)).tolist()
        return overlaps

    def _get_neighbors(
        self,
        neighbors_func: Callable[[rx.PyDiGraph, int], rx.NodeIndices],
        node_ids: list[int] | int | None = None,
        attr_keys: Sequence[str] | str | None = None,
        *,
        return_attrs: bool = False,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame | dict[int, list[int]] | list[int]:
        node_ids = self._get_local_ids() if node_ids is None else self._map_to_local(node_ids)
        neighbors = super()._get_neighbors(neighbors_func, node_ids, attr_keys, return_attrs=return_attrs)
        if not return_attrs:
            if isinstance(neighbors, list):
                return self._map_to_external(neighbors)
            else:
                return {
                    self._map_to_external(node_id): self._map_to_external(neighbor_ids)
                    for node_id, neighbor_ids in neighbors.items()
                }
        else:
            if isinstance(neighbors, pl.DataFrame):
                return self._map_df_to_external(neighbors, [DEFAULT_ATTR_KEYS.NODE_ID])
            else:
                return {
                    self._map_to_external(node_id): self._map_df_to_external(df, [DEFAULT_ATTR_KEYS.NODE_ID])
                    for node_id, df in neighbors.items()
                }

    def update_node_attrs(
        self,
        *,
        attrs: dict[str, Any],
        node_ids: Sequence[int] | None = None,
    ) -> None:
        """
        Update the node attributes of the graph.

        Parameters
        ----------
        attrs : dict[str, Any]
            The attributes to update.
        node_ids : Sequence[int] | None
            The node ids to update.
        """
        node_ids = self._get_local_ids() if node_ids is None else self._map_to_local(node_ids)
        super().update_node_attrs(attrs=attrs, node_ids=node_ids)

    def remove_node(self, node_id: int) -> None:
        """
        Remove a node from the graph.

        Parameters
        ----------
        node_id : int
            The external ID of the node to remove.

        Raises
        ------
        ValueError
            If the node_id does not exist in the graph.
        """
        if node_id not in self._external_to_local:
            raise ValueError(f"Node {node_id} does not exist in the graph.")

        local_node_id = self._map_to_local(node_id)

        self.node_removed.emit_fast(node_id)
        with self.node_removed.blocked():
            super().remove_node(local_node_id)

        self._remove_id_mapping(external_id=node_id)

    def filter(
        self,
        *attr_filters: AttrComparison,
        node_ids: Sequence[int] | None = None,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> RXFilter:
        from tracksdata.graph.filters._indexed_filter import IndexRXFilter

        return IndexRXFilter(
            *attr_filters,
            graph=self,
            node_ids=None if node_ids is None else self._map_to_local(node_ids),
            include_targets=include_targets,
            include_sources=include_sources,
        )

    def edge_id(self, source_id: int, target_id: int) -> int:
        """
        Return the edge id between two nodes.
        """
        source_id = self._map_to_local(source_id)
        target_id = self._map_to_local(target_id)
        return super().edge_id(source_id, target_id)
