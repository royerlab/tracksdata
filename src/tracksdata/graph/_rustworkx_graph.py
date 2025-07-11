import operator
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import rustworkx as rx

from tracksdata.attrs import (
    AttrComparison,
    attr_comps_to_strs,
    polars_reduce_attr_comps,
    split_attr_comps,
)
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._rx import _assign_track_ids
from tracksdata.graph._base_filter import BaseFilter, cache_method
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._dataframe import unpack_array_attrs
from tracksdata.utils._logging import LOG

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


def _create_filter_func(
    attr_comps: Sequence[AttrComparison],
) -> Callable[[dict[str, Any]], bool]:
    LOG.info(f"Creating filter function for {attr_comps}")

    def _filter(attrs: dict[str, Any]) -> bool:
        for attr_op in attr_comps:
            if not attr_op.op(attrs[str(attr_op.column)], attr_op.other):
                return False
        return True

    return _filter


class RXFilter(BaseFilter):
    def __init__(
        self,
        graph: "RustWorkXGraph",
        *attr_comps: AttrComparison,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._attr_comps = attr_comps
        self._include_targets = include_targets
        self._include_sources = include_sources
        self._node_attr_comps, self._edge_attr_comps = split_attr_comps(attr_comps)

    @cache_method
    def _edge_attrs(self) -> pl.DataFrame:
        return self._graph.edge_attrs()

    @cache_method
    def _current_node_ids(self) -> list[int]:
        """
        Get the node IDs without considering their `source` or `target` neighbors.
        """
        if len(self._node_attr_comps) == 0:
            node_ids = self._graph.rx_graph.node_indices()
        else:
            # only nodes are filtered return nodes that pass node filters
            node_ids = self._graph.filter_nodes_by_attrs(*self._node_attr_comps)

        return node_ids

    @cache_method
    def node_ids(self) -> list[int]:
        # no nodes outside of the filtered nodes are allowed
        if len(self._edge_attr_comps) == 0 and (not self._include_targets and not self._include_sources):
            return self._current_node_ids()

        # find nodes that are connected to edges that pass the edge filters
        edge_node_ids = (
            self._edge_attrs()
            .select(
                DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                DEFAULT_ATTR_KEYS.EDGE_TARGET,
            )
            .to_numpy()
            .ravel()
        )
        node_ids = np.unique(np.concatenate([edge_node_ids, self._current_node_ids()]))
        return np.unique(node_ids).tolist()

    @cache_method
    def _edge_attrs(self) -> pl.DataFrame:
        node_ids = self._current_node_ids()

        _filter_func = _create_filter_func(self._edge_attr_comps)
        neigh_funcs = []
        if self._include_targets:
            neigh_funcs.append(self._graph.rx_graph.out_edges)

        if self._include_sources:
            neigh_funcs.append(self._graph.rx_graph.in_edges)

        # only edges are filtered return nodes that pass edge filters
        sources = []
        targets = []
        data = {k: [] for k in self._graph.edge_attr_keys}
        data[DEFAULT_ATTR_KEYS.EDGE_ID] = []

        for node_id in node_ids:
            for nf in neigh_funcs:
                for src, tgt, attr in nf(node_id):
                    if _filter_func(attr):
                        sources.append(src)
                        targets.append(tgt)
                        for k in data.keys():
                            data[k].append(attr[k])

        df = pl.DataFrame(data).with_columns(
            pl.Series(sources, dtype=pl.Int64).alias(DEFAULT_ATTR_KEYS.EDGE_SOURCE),
            pl.Series(targets, dtype=pl.Int64).alias(DEFAULT_ATTR_KEYS.EDGE_TARGET),
        )
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
        return self._graph.node_attrs(node_ids=self.node_ids(), attr_keys=attr_keys, unpack=unpack)

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
            attr_keys = self._graph.edge_attr_keys

        attr_keys = [
            DEFAULT_ATTR_KEYS.EDGE_ID,
            DEFAULT_ATTR_KEYS.EDGE_SOURCE,
            DEFAULT_ATTR_KEYS.EDGE_TARGET,
            *attr_keys,
        ]
        attr_keys = list(dict.fromkeys(attr_keys))

        df = df.select(attr_keys)
        if unpack:
            df = unpack_array_attrs(df, attr_keys, unpack_array_attrs)
        return df

    @cache_method
    def subgraph(
        self,
        node_attr_keys: Sequence[str] | str | None = None,
        edge_attr_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        from tracksdata.graph._graph_view import GraphView

        node_ids = self.node_ids()

        rx_graph, node_map = self._graph.rx_graph.subgraph_with_nodemap(node_ids)
        if self._edge_attr_comps:
            _filter_func = _create_filter_func(self._edge_attr_comps)
            for src, tgt, attr in rx_graph.weighted_edge_list():
                if not _filter_func(attr):
                    rx_graph.remove_edge(src, tgt)

        graph_view = GraphView(
            rx_graph,
            node_map_to_root=dict(node_map.items()),
            root=self._graph,
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

    node_ids = graph.filter_nodes_by_attrs(NodeAttr("t") == 0)
    ```

    Create subgraphs:

    ```python
    subgraph = graph.subgraph(NodeAttr("t") == 0)
    ```
    """

    def __init__(self, rx_graph: rx.PyDiGraph | None = None) -> None:
        super().__init__()

        if rx_graph is None:
            self._graph = rx.PyDiGraph()
        else:
            self._graph = rx_graph

        self._time_to_nodes: dict[int, list[int]] = {}
        self._node_attr_keys: list[str] = [DEFAULT_ATTR_KEYS.T]
        self._edge_attr_keys: list[str] = []
        self._overlaps: list[list[int, 2]] = []

    @property
    def rx_graph(self) -> rx.PyDiGraph:
        return self._graph

    def filter(
        self,
        *attr_filters: AttrComparison,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> RXFilter:
        return RXFilter(
            self,
            *attr_filters,
            include_targets=include_targets,
            include_sources=include_sources,
        )

    def add_node(
        self,
        attrs: dict[str, Any],
        validate_keys: bool = True,
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
        """
        # avoiding copying attributes on purpose, it could be a problem in the future
        if validate_keys:
            self._validate_attributes(attrs, self.node_attr_keys, "node")

            if "t" not in attrs:
                raise ValueError(f"Node attributes must have a 't' key. Got {attrs.keys()}")

        node_id = self.rx_graph.add_node(attrs)
        self._time_to_nodes.setdefault(attrs["t"], []).append(node_id)
        return node_id

    def bulk_add_nodes(self, nodes: list[dict[str, Any]]) -> list[int]:
        """
        Faster method to add multiple nodes to the graph with less overhead and fewer checks.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            The data of the nodes to be added.
            The keys of the data will be used as the attributes of the nodes.
            Must have "t" key.

        Returns
        -------
        list[int]
            The IDs of the added nodes.
        """
        indices = list(self.rx_graph.add_nodes_from(nodes))
        for node, index in zip(nodes, indices, strict=True):
            self._time_to_nodes.setdefault(node["t"], []).append(index)
        return indices

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
            self._validate_attributes(attrs, self.edge_attr_keys, "edge")
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
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        """
        Get the predecessors or sucessors of a list of nodes.
        See more information below.
        """
        single_node = False
        if isinstance(node_ids, int):
            node_ids = [node_ids]
            single_node = True

        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        rx_graph = self.rx_graph
        valid_schema = None

        neighbors = {}
        for node_id in node_ids:
            neighbors_indices = neighbors_func(rx_graph, node_id)
            neighbors_data: list[dict[str, Any]] = [rx_graph[i] for i in neighbors_indices]

            if attr_keys is not None:
                neighbors_data = [
                    {k: edge_data[k] for k in attr_keys if k != DEFAULT_ATTR_KEYS.NODE_ID}
                    for edge_data in neighbors_data
                ]

            if len(neighbors_data) > 0:
                df = pl.DataFrame(neighbors_data)
                if attr_keys is None or DEFAULT_ATTR_KEYS.NODE_ID in attr_keys:
                    df = df.with_columns(
                        pl.Series(DEFAULT_ATTR_KEYS.NODE_ID, np.asarray(neighbors_indices, dtype=int)),
                    )
                neighbors[node_id] = df
                valid_schema = neighbors[node_id].schema

        if single_node:
            try:
                # could not find sucessors for this node
                return neighbors[node_ids[0]]
            except KeyError:
                return pl.DataFrame()

        for node_id in node_ids:
            if node_id not in neighbors:
                if valid_schema is None:
                    neighbors[node_id] = pl.DataFrame()
                else:
                    neighbors[node_id] = pl.DataFrame(schema=valid_schema)

        return neighbors

    def successors(
        self,
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        """
        Get the sucessors of a list of nodes.

        Parameters
        ----------
        node_ids : list[int] | int
            The IDs of the nodes to get the sucessors for.
        attr_keys : Sequence[str] | str | None
            The attribute keys to get.
            If None, all attributesare used.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame
            The sucessors of the nodes indexed by node ID if a list of nodes is provided.
        """
        return self._get_neighbors(
            rx.PyDiGraph.successor_indices,
            node_ids,
            attr_keys,
        )

    def predecessors(
        self,
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        """
        Get the predecessors of a list of nodes.

        Parameters
        ----------
        node_ids : list[int] | int
            The IDs of the nodes to get the predecessors for.
        attr_keys : Sequence[str] | str | None
            The attribute keys to get.
            If None, all attributesare used.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame
            The predecessors of the nodes indexed by node ID if a list of nodes is provided.
        """
        return self._get_neighbors(
            rx.PyDiGraph.predecessor_indices,
            node_ids,
            attr_keys,
        )

    def filter_nodes_by_attrs(
        self,
        *attrs: AttrComparison,
    ) -> list[int]:
        """
        Filter nodes by attributes.

        Parameters
        ----------
        *attrs : AttrComparison
            The attributes to filter by, for example:
            ```python
            graph.filter_nodes_by_attrs(Attr("t") == 0, Attr("label") == "A")
            ```

        Returns
        -------
        list[int]
            The IDs of the filtered nodes.
        """
        rx_graph = self.rx_graph
        node_map = None
        # entire graph
        attrs, time = _pop_time_eq(attrs)

        if time is not None:
            selected_nodes = self._time_to_nodes.get(time, [])
            if len(attrs) == 0:
                return selected_nodes

            # subgraph of selected nodes
            rx_graph, node_map = rx_graph.subgraph_with_nodemap(selected_nodes)

        _filter_func = _create_filter_func(attrs)

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

    def subgraph(
        self,
        *attr_filters: AttrComparison,
        node_ids: Sequence[int] | None = None,
        node_attr_keys: Sequence[str] | str | None = None,
        edge_attr_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        """
        Create a subgraph from the graph from the given node IDs
        or attributes' filters.

        Node IDs or a single attribute filter can be used to create a subgraph.

        Parameters
        ----------
        *attr_filters : AttrComparison
            The attributes to filter the nodes and edges by.
        node_ids : Sequence[int] | None
            The IDs of the nodes to include in the subgraph.
        node_attr_keys : Sequence[str] | str | None
            The attribute keys to include in the subgraph.
        edge_attr_keys : Sequence[str] | str | None
            The attribute keys to include in the subgraph.

        Returns
        -------
        RustWorkXGraph
            A new graph with the specified nodes.
        """
        from tracksdata.graph._graph_view import GraphView

        node_attr_comps, edge_attr_comps = split_attr_comps(attr_filters)
        self._validate_subgraph_args(node_ids, node_attr_comps, edge_attr_comps)

        if node_attr_comps:
            filtered_node_ids = self.filter_nodes_by_attrs(*node_attr_comps)
            if node_ids is not None:
                node_ids = np.intersect1d(node_ids, filtered_node_ids).tolist()
            else:
                node_ids = filtered_node_ids

        if node_ids is None and edge_attr_comps:
            edges_df = self.edge_attrs(node_ids=node_ids, attr_keys=attr_comps_to_strs(edge_attr_comps))
            mask = polars_reduce_attr_comps(edges_df, edge_attr_comps, operator.and_)
            node_ids = np.unique(
                edges_df.filter(mask)
                .select(
                    DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                    DEFAULT_ATTR_KEYS.EDGE_TARGET,
                )
                .to_numpy()
            )

        rx_graph, node_map = self.rx_graph.subgraph_with_nodemap(node_ids)

        if edge_attr_comps:
            LOG.info("Removing edges without attributes %s", edge_attr_comps)
            _filter_func = _create_filter_func(edge_attr_comps)
            for source, target, edge_attr in rx_graph.weighted_edge_list():
                if not _filter_func(edge_attr):
                    rx_graph.remove_edge(source, target)

        graph_view = GraphView(
            rx_graph=rx_graph,
            node_map_to_root=dict(node_map.items()),
            root=self,
        )

        return graph_view

    def time_points(self) -> list[int]:
        """
        Get the unique time points in the graph.
        """
        return list(self._time_to_nodes.keys())

    @property
    def node_attr_keys(self) -> list[str]:
        """
        Get the keys of the attributes of the nodes.
        """
        return self._node_attr_keys

    @property
    def edge_attr_keys(self) -> list[str]:
        """
        Get the keys of the attributes of the edges.
        """
        return self._edge_attr_keys

    def add_node_attr_key(self, key: str, default_value: Any) -> None:
        """
        Add a new attribute key to the graph.
        All existing nodes will have the default value for the new attribute key.

        Parameters
        ----------
        key : str
            The key of the new attribute.
        default_value : Any
            The default value for existing nodes for the new attribute key.
        """
        if key in self.node_attr_keys:
            raise ValueError(f"Attribute key {key} already exists")

        self._node_attr_keys.append(key)
        rx_graph = self.rx_graph
        for node_id in rx_graph.node_indices():
            rx_graph[node_id][key] = default_value

    def add_edge_attr_key(self, key: str, default_value: Any) -> None:
        """
        Add a new attribute key to the graph.
        All existing edges will have the default value for the new attribute key.

        Parameters
        ----------
        key : str
            The key of the new attribute.
        default_value : Any
            The default value for existing edges for the new attribute key.
        """
        if key in self.edge_attr_keys:
            raise ValueError(f"Attribute key {key} already exists")

        self._edge_attr_keys.append(key)
        for _, _, edge_attr in self.rx_graph.weighted_edge_list():
            edge_attr[key] = default_value

    def node_attrs(
        self,
        *,
        node_ids: Sequence[int] | None = None,
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
            Whether to unpack array attributesinto multiple scalar attributes.

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
            attr_keys = [DEFAULT_ATTR_KEYS.NODE_ID, *self.node_attr_keys]

        if isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        if len(node_ids) == 0:
            return pl.DataFrame({key: [] for key in attr_keys})

        # making them unique
        attr_keys = list(dict.fromkeys(attr_keys))

        # Create columns directly instead of building intermediate dictionaries
        columns = {key: [] for key in attr_keys}

        if DEFAULT_ATTR_KEYS.NODE_ID in attr_keys:
            columns[DEFAULT_ATTR_KEYS.NODE_ID] = np.asarray(node_ids, dtype=int)
            attr_keys.remove(DEFAULT_ATTR_KEYS.NODE_ID)

        # Build columns in a vectorized way
        for node_id in node_ids:
            node_data = rx_graph[node_id]
            for key in attr_keys:
                columns[key].append(node_data[key])

        for key in attr_keys:
            columns[key] = np.asarray(columns[key])

        # Create DataFrame and set node_id as index in one shot
        df = pl.DataFrame(columns)

        if unpack:
            df = unpack_array_attrs(df)

        return df

    def edge_attrs(
        self,
        *,
        node_ids: list[int] | None = None,
        attr_keys: Sequence[str] | str | None = None,
        include_targets: bool = False,
        unpack: bool = False,
    ) -> pl.DataFrame:
        """
        Get the attributes of the edges as a polars DataFrame.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the subgraph to get the edge attributesfor.
            If None, all edges of the graph are used.
        attr_keys : Sequence[str] | str | None
            The attribute keys to get.
            If None, all attributesare used.
        include_targets : bool
            Whether to include edges out-going from the given node_ids even
            if the target node is not in the given node_ids.
        unpack : bool
            Whether to unpack array attributesinto multiple scalar attributes.
        """
        if attr_keys is None:
            attr_keys = self.edge_attr_keys

        attr_keys = [DEFAULT_ATTR_KEYS.EDGE_ID, *attr_keys]
        attr_keys = list(dict.fromkeys(attr_keys))

        if node_ids is None:
            rx_graph = self.rx_graph
            node_map = None
        else:
            if include_targets:
                selected_nodes = set(node_ids)
                for node_id in node_ids:
                    neighbors = self.rx_graph.neighbors(node_id)
                    selected_nodes.update(neighbors)
                node_ids = list(selected_nodes)

            rx_graph, node_map = self.rx_graph.subgraph_with_nodemap(node_ids)

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

        if node_map is not None:
            source = [node_map[s] for s in source]
            target = [node_map[t] for t in target]

        columns = {key: [] for key in attr_keys}

        for row in data:
            for key in attr_keys:
                columns[key].append(row[key])

        columns[DEFAULT_ATTR_KEYS.EDGE_SOURCE] = source
        columns[DEFAULT_ATTR_KEYS.EDGE_TARGET] = target

        columns = {k: np.asarray(v) for k, v in columns.items()}

        df = pl.DataFrame(columns)
        if unpack:
            df = unpack_array_attrs(df)
        return df

    @property
    def num_edges(self) -> int:
        """
        The number of edges in the graph.
        """
        return self._graph.num_edges()

    @property
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
            if key not in self.node_attr_keys:
                raise ValueError(f"Node attribute key '{key}' not found in graph. Expected '{self.node_attr_keys}'")

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
            if key not in self.edge_attr_keys:
                raise ValueError(f"Edge attribute key '{key}' not found in graph. Expected '{self.edge_attr_keys}'")

            if np.isscalar(value):
                attrs[key] = [value] * size

            elif len(attrs[key]) != size:
                raise ValueError(f"Attribute '{key}' has wrong size. Expected {size}, got {len(attrs[key])}")

        edge_map = self._graph.edge_index_map()

        for i, edge_id in enumerate(edge_ids):
            edge_attr = edge_map[edge_id][2]  # 0=source, 1=target, 2=attributes
            for key, value in attrs.items():
                edge_attr[key] = value[i]

    def assign_track_ids(
        self,
        output_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
        reset: bool = True,
    ) -> rx.PyDiGraph:
        """
        Compute and assign track ids to nodes.

        Parameters
        ----------
        output_key : str
            The key of the output track id attribute.
        reset : bool
            Whether to reset the track ids of the graph. If True, the track ids will be reset to -1.

        Returns
        -------
        rx.PyDiGraph
            A compressed graph (parent -> child) with track ids lineage relationships.
        """
        try:
            node_ids, track_ids, tracks_graph = _assign_track_ids(self.rx_graph)
        except RuntimeError as e:
            raise RuntimeError(
                "Are you sure this graph is a valid lineage graph?\n"
                "This function expects a solved graph.\n"
                "Often used from `graph.subgraph(edge_attr_filter={'solution': True})`"
            ) from e

        if output_key not in self.node_attr_keys:
            self.add_node_attr_key(output_key, -1)
        elif reset:
            self.update_node_attrs(node_ids=self.node_ids(), attrs={output_key: -1})

        self.update_node_attrs(
            node_ids=node_ids,
            attrs={output_key: track_ids},
        )

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
