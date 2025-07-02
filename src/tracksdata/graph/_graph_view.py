from collections.abc import Callable, Sequence
from typing import Any, overload

import polars as pl
import rustworkx as rx

from tracksdata.attrs import AttrComparison
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._rx import _assign_track_ids
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._rustworkx_graph import RustWorkXGraph


@overload
def map_ids(map: dict[int, int], indices: Sequence[int]) -> list[int]: ...


@overload
def map_ids(map: dict[int, int], indices: None) -> None: ...


def map_ids(
    map: dict[int, int],
    indices: Sequence[int] | None,
) -> list[int] | None:
    if indices is None:
        return None

    if hasattr(indices, "tolist"):
        indices = indices.tolist()

    return [map[node_id] for node_id in indices]


class GraphView(RustWorkXGraph):
    """
    A filtered view of a graph that maintains bidirectional mapping to the root graph.

    GraphView provides a lightweight way to work with subsets of a larger graph
    while maintaining the ability to synchronize changes back to the original graph.
    It acts as a view layer that maps between local node/edge IDs and the root
    graph's IDs, enabling efficient subgraph operations with minimal data duplication.

    Parameters
    ----------
    rx_graph : rx.PyDiGraph
        The rustworkx graph object representing the subgraph.
    node_map_to_root : dict[int, int]
        Mapping from local node IDs to root graph node IDs.
    root : BaseGraph
        Reference to the root graph that this view is derived from.
    sync : bool, default True
        Whether to automatically synchronize changes in the view.
        By default only the root graph is updated.

    Attributes
    ----------
    _node_map_to_root : dict[int, int]
        Mapping from view node IDs to root graph node IDs.
    _node_map_from_root : dict[int, int]
        Mapping from root graph node IDs to view node IDs.
    _edge_map_to_root : dict[int, int]
        Mapping from view edge IDs to root graph edge IDs.
    _edge_map_from_root : dict[int, int]
        Mapping from root graph edge IDs to view edge IDs.

    See Also
    --------
    [RustWorkXGraph][tracksdata.graph.RustWorkXGraph]:
        The base graph implementation that this view extends.

    [SQLGraph][tracksdata.graph.SQLGraph]:
        Database-backed graph implementation for larger datasets.

    Examples
    --------
    Create a subgraph view filtered by time:

    ```python
    from tracksdata.attrs import NodeAttr

    view = graph.subgraph(NodeAttr("t") == 5)
    ```

    Access nodes in the view:

    ```python
    node_ids = view.node_ids()
    node_attrs = view.node_attrs(node_ids=node_ids)
    ```

    The view automatically maps between local and root IDs when needed.
    """

    def __init__(
        self,
        rx_graph: rx.PyDiGraph,
        node_map_to_root: dict[int, int],
        root: BaseGraph,
        sync: bool = True,
    ) -> None:
        super().__init__(rx_graph=rx_graph)

        # setting up the time_to_nodes mapping
        for idx in rx_graph.node_indices():
            t = self.rx_graph[idx][DEFAULT_ATTR_KEYS.T]
            if t not in self._time_to_nodes:
                self._time_to_nodes[t] = []
            self._time_to_nodes[t].append(idx)

        self._node_map_to_root = node_map_to_root.copy()
        self._node_map_from_root = {v: k for k, v in node_map_to_root.items()}

        self._edge_map_to_root: dict[int, int] = {
            idx: data[DEFAULT_ATTR_KEYS.EDGE_ID] for idx, (_, _, data) in self.rx_graph.edge_index_map().items()
        }
        self._edge_map_from_root: dict[int, int] = {v: k for k, v in self._edge_map_to_root.items()}

        self._root = root
        self._is_root_rx_graph = isinstance(root, RustWorkXGraph)
        self._sync = sync
        self._out_of_sync = False

        # making sure these are not used
        # they should be accessed through the root graph
        self._node_attr_keys = None
        self._edge_attr_keys = None

    @property
    def sync(self) -> bool:
        return self._sync

    @sync.setter
    def sync(self, value: bool) -> None:
        if value and not self._sync:
            raise ValueError("Cannot sync a graph view that is not synced\nRe-create the graph view.")
        self._sync = value

    def node_ids(self) -> list[int]:
        indices = self.rx_graph.node_indices()
        return map_ids(self._node_map_to_root, indices)

    def edge_ids(self) -> list[int]:
        indices = self.rx_graph.edge_indices()
        return map_ids(self._edge_map_to_root, indices)

    def subgraph(
        self,
        *attr_filters: AttrComparison,
        node_ids: Sequence[int] | None = None,
        node_attr_keys: Sequence[str] | str | None = None,
        edge_attr_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        subgraph = super().subgraph(
            *attr_filters,
            node_ids=map_ids(self._node_map_from_root, node_ids),
            node_attr_keys=node_attr_keys,
            edge_attr_keys=edge_attr_keys,
        )

        subgraph._replace_parent_graph_with_root()

        return subgraph

    @property
    def node_attr_keys(self) -> list[str]:
        return self._root.node_attr_keys

    @property
    def edge_attr_keys(self) -> list[str]:
        return self._root.edge_attr_keys

    def add_node_attr_key(self, key: str, default_value: Any) -> None:
        self._root.add_node_attr_key(key, default_value)
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if not self._is_root_rx_graph:
            if self.sync:
                rx_graph = self.rx_graph
                for node_id in rx_graph.node_indices():
                    rx_graph[node_id][key] = default_value
            else:
                self._out_of_sync = True

    def add_edge_attr_key(self, key: str, default_value: Any) -> None:
        self._root.add_edge_attr_key(key, default_value)
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if not self._is_root_rx_graph:
            if self.sync:
                for _, _, edge_attr in self.rx_graph.weighted_edge_list():
                    edge_attr[key] = default_value
            else:
                self._out_of_sync = True

    def add_node(
        self,
        attrs: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        parent_node_id = self._root.add_node(
            attrs=attrs,
            validate_keys=validate_keys,
        )

        if self.sync:
            node_id = super().add_node(
                attrs=attrs,
                validate_keys=validate_keys,
            )
            self._node_map_to_root[node_id] = parent_node_id
            self._node_map_from_root[parent_node_id] = node_id
        else:
            self._out_of_sync = True

        return parent_node_id

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        attrs: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        parent_edge_id = self._root.add_edge(
            source_id=source_id,
            target_id=target_id,
            attrs=attrs,
            validate_keys=validate_keys,
        )
        attrs[DEFAULT_ATTR_KEYS.EDGE_ID] = parent_edge_id

        if self.sync:
            # it does not set the EDGE_ID as attribute as the super().add_edge
            edge_id = self.rx_graph.add_edge(
                self._node_map_from_root[source_id],
                self._node_map_from_root[target_id],
                attrs,
            )
            self._edge_map_to_root[edge_id] = parent_edge_id
            self._edge_map_from_root[parent_edge_id] = edge_id
        else:
            self._out_of_sync = True

        return parent_edge_id

    def _get_neighbors(
        self,
        neighbors_func: Callable[[rx.PyDiGraph, int], rx.NodeIndices],
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        single_node = False
        if isinstance(node_ids, int):
            node_ids = [node_ids]
            single_node = True

        node_ids = map_ids(self._node_map_from_root, node_ids)
        neighbors_data = super()._get_neighbors(neighbors_func, node_ids, attr_keys)

        out_data = {}
        for node_id in node_ids:
            df = neighbors_data[node_id]
            out_data[self._node_map_to_root[node_id]] = self._map_to_root_df_node_ids(df)

        if single_node:
            return next(iter(out_data.values()))

        return out_data

    def sucessors(
        self,
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        if self._out_of_sync:
            raise RuntimeError("Out of sync graph view cannot be used to get sucessors")
        return super().sucessors(node_ids, attr_keys)

    def predecessors(
        self,
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        if self._out_of_sync:
            raise RuntimeError("Out of sync graph view cannot be used to get predecessors")
        return super().predecessors(node_ids, attr_keys)

    def filter_nodes_by_attrs(
        self,
        *attrs: AttrComparison,
    ) -> list[int]:
        node_ids = super().filter_nodes_by_attrs(
            *attrs,
        )
        return map_ids(self._node_map_to_root, node_ids)

    def _map_to_root_df_node_ids(self, df: pl.DataFrame) -> pl.DataFrame:
        if DEFAULT_ATTR_KEYS.NODE_ID in df.columns:
            df = df.with_columns(
                pl.col(DEFAULT_ATTR_KEYS.NODE_ID)
                .map_elements(self._node_map_to_root.get, return_dtype=pl.Int64)
                .alias(DEFAULT_ATTR_KEYS.NODE_ID)
            )
        return df

    def node_attrs(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        node_dfs = super().node_attrs(
            node_ids=map_ids(self._node_map_from_root, node_ids),
            attr_keys=attr_keys,
            unpack=unpack,
        )
        node_dfs = self._map_to_root_df_node_ids(node_dfs)
        return node_dfs

    def edge_attrs(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        attr_keys: Sequence[str] | str | None = None,
        include_targets: bool = False,
        unpack: bool = False,
    ) -> pl.DataFrame:
        edges_df = super().edge_attrs(
            node_ids=map_ids(self._node_map_from_root, node_ids),
            attr_keys=attr_keys,
            include_targets=include_targets,
            unpack=unpack,
        )

        edges_df = edges_df.with_columns(
            *[
                pl.col(key).map_elements(self._node_map_to_root.get, return_dtype=pl.Int64).alias(key)
                for key in [DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]
            ]
        )

        return edges_df

    def update_node_attrs(
        self,
        *,
        attrs: dict[str, Any],
        node_ids: Sequence[int] | None = None,
    ) -> None:
        if node_ids is None:
            node_ids = self.node_ids()

        self._root.update_node_attrs(
            node_ids=node_ids,
            attrs=attrs,
        )
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if not self._is_root_rx_graph:
            if self.sync:
                super().update_node_attrs(
                    node_ids=map_ids(self._node_map_from_root, node_ids),
                    attrs=attrs,
                )
            else:
                self._out_of_sync = True

    def update_edge_attrs(
        self,
        *,
        attrs: dict[str, Any],
        edge_ids: Sequence[int] | None = None,
    ) -> None:
        if edge_ids is None:
            edge_ids = self.edge_ids()

        self._root.update_edge_attrs(
            edge_ids=edge_ids,
            attrs=attrs,
        )
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if not self._is_root_rx_graph:
            if self.sync:
                super().update_edge_attrs(
                    edge_ids=map_ids(self._edge_map_from_root, edge_ids),
                    attrs=attrs,
                )
            else:
                self._out_of_sync = True

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
            Whether to reset all track ids before assigning new ones.

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

        node_ids = map_ids(self._node_map_to_root, node_ids)

        if output_key not in self.node_attr_keys:
            self.add_node_attr_key(output_key, -1)
        elif reset:
            self.update_node_attrs(attrs={output_key: -1})

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
            return rx_graph.in_degree(self._node_map_from_root[node_ids])
        return [rx_graph.in_degree(self._node_map_from_root[node_id]) for node_id in node_ids]

    def out_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the out-degree of a list of nodes.
        """
        if node_ids is None:
            node_ids = self.node_ids()
        rx_graph = self.rx_graph
        if isinstance(node_ids, int):
            return rx_graph.out_degree(self._node_map_from_root[node_ids])
        return [rx_graph.out_degree(self._node_map_from_root[node_id]) for node_id in node_ids]

    def _replace_parent_graph_with_root(self) -> None:
        """
        Replace the parent graph with it's own parent graph (the root graph)
        This is internally called so every view of a graph maps to a single root, skipping intermediate views.
        """
        parent = self._root

        if not isinstance(parent, GraphView):
            raise ValueError(
                f"Parent graph must be a GraphView to have its parent replaced with the root graph. Got {type(parent)}."
            )

        self._root = parent._root
        self._node_map_to_root = {k: parent._node_map_to_root[v] for k, v in self._node_map_to_root.items()}
        self._node_map_from_root = {v: k for k, v in self._node_map_to_root.items()}

        self._edge_map_to_root = {k: parent._edge_map_to_root[v] for k, v in self._edge_map_to_root.items()}
        self._edge_map_from_root = {v: k for k, v in self._edge_map_to_root.items()}

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
        subgraph = super().contract_nodes(
            permanent_node_ids=map_ids(self._node_map_from_root, permanent_node_ids),
        )
        subgraph._replace_parent_graph_with_root()

        return subgraph
