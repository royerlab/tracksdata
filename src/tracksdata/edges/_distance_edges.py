from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from scipy.spatial import KDTree

from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._base_edges import BaseEdgesOperator
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._logging import LOG


class DistanceEdges(BaseEdgesOperator):
    """
    Operator that adds edges to a graph based on Euclidean distance between nodes.

    Creates edges between nodes in consecutive time points by finding the closest
    neighbors within a specified distance threshold using efficient KDTree-based
    spatial indexing. Creates directed edges from nodes in the range t-1
    to t-delta_t to the nodes in the current time point t, representing potential transitions.

    Parameters
    ----------
    distance_threshold : float
        Maximum Euclidean distance for adding edges between nodes.
        Nodes farther apart than this threshold will not be connected.
    n_neighbors : int
        Maximum number of neighbors to consider for each node when adding edges.
        For each node at time t, edges will be created to at most n_neighbors
        closest nodes at time t-1 to t-delta_t.
    delta_t : int
        The number of time points to consider for adding edges.
        For each node at time t, edges will be created to the closest
        n_neighbors nodes at time t-1 to t-delta_t.
    neighbors_per_frame : bool, default False
        Whether to consider the `n_neighbors` as `per_frame` or `total`.
        If True, `n_neighbors` is the number of neighbors per frame, meaning that
        for each node at time t, edges will be created to the closest
        n_neighbors per adjacent frame.
        If False, `n_neighbors` is the number of neighbors in all frames (from t-delta_t to t)
        considering all adjacent frames together.
    output_key : str, default DEFAULT_ATTR_KEYS.EDGE_WEIGHT
        The attribute key to store the distance values in the edges.
    attr_keys : Sequence[str] | None, optional
        The node attribute keys to use for distance calculation. If None,
        defaults to [DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
        if DEFAULT_ATTR_KEYS.Z exists, otherwise [DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X].

    Attributes
    ----------
    distance_threshold : float
        The distance threshold for adding edges.
    n_neighbors : int
        The maximum number of neighbors to consider for adding edges.
        This in respect from the current to the previous frame.
        That means, a node in frame t will have edges to the closest
        n_neighbors nodes in frame t-1.
    delta_t : int, default 1
        The number of time points to consider for adding edges.
        For each node at time t, edges will be created to the closest
        n_neighbors nodes at time t-delta_t to t.
    neighbors_per_frame : bool, default False
        Whether `n_neighbors` is the number of neighbors per frame or all frames (from t-delta_t to t).
    output_key : str
        The key used to store distance values in edges.
    attr_keys : Sequence[str] | None
        The attribute keys to use for the distance calculation.
        When None, DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X are used.
    show_progress : bool
        Whether to print progress of the edges addition.

    Examples
    --------
    Create a distance-based edge operator:

    ```python
    from tracksdata.edges import DistanceEdges

    edge_op = DistanceEdges(distance_threshold=50.0, n_neighbors=3, attr_keys=["x", "y"])
    ```

    Add edges to a graph:

    ```python
    edge_op.add_edges(graph)
    ```

    Add edges for a specific time point:

    ```python
    edge_op.add_edges(graph, t=5)
    ```

    Use custom output key:

    ```python
    edge_op = DistanceEdges(distance_threshold=30.0, n_neighbors=2, output_key="euclidean_distance")
    ```
    """

    output_key: str

    def __init__(
        self,
        distance_threshold: float,
        n_neighbors: int,
        delta_t: int = 1,
        neighbors_per_frame: bool = False,
        output_key: str = DEFAULT_ATTR_KEYS.EDGE_DIST,
        attr_keys: Sequence[str] | None = None,
    ):
        if delta_t < 1:
            raise ValueError(f"'delta_t' must be at least 1, got {delta_t}")

        super().__init__(output_key=output_key)
        self.distance_threshold = distance_threshold
        self.n_neighbors = n_neighbors
        self.delta_t = delta_t
        self.attr_keys = attr_keys
        self.neighbors_per_frame = neighbors_per_frame

    def _init_edge_attrs(self, graph: BaseGraph) -> None:
        """
        Initialize the edge attributes for the graph.
        """
        if self.output_key not in graph.edge_attr_keys():
            graph.add_edge_attr_key(self.output_key, pl.Float64, default_value=-99999.0)

    def _get_spatial_attr_keys(self, graph: BaseGraph) -> list[str]:
        """
        Determine which spatial attribute keys to use for distance calculation.

        Parameters
        ----------
        graph : BaseGraph
            The graph containing node attributes.

        Returns
        -------
        list[str]
            List of attribute keys to use for spatial coordinates.
        """
        if self.attr_keys is None:
            if DEFAULT_ATTR_KEYS.Z in graph.node_attr_keys():
                attr_keys = [DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
            else:
                attr_keys = [DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
        else:
            attr_keys = list(self.attr_keys)

        return attr_keys

    def _build_kdtree_data(
        self, graph: BaseGraph, time_point: int, attr_keys: Sequence[str]
    ) -> tuple[KDTree, Any, list]:
        """
        Build KDTree for a specific time point.

        Parameters
        ----------
        graph : BaseGraph
            The graph to query.
        time_point : int
            The time point to build the KDTree for.
        attr_keys : Sequence[str]
            Attribute keys to use for spatial coordinates.

        Returns
        -------
        tuple[KDTree, GraphView, list]
            A tuple containing:
            - KDTree built from node coordinates
            - Node attributes as numpy array
            - List of node IDs at this time point
        """
        node_filter = graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == time_point)

        if node_filter.is_empty():
            return None, None, []

        node_attrs = node_filter.node_attrs(attr_keys=attr_keys)
        node_ids = list(node_filter.node_ids())
        kdtree = KDTree(node_attrs.to_numpy())

        return kdtree, node_attrs, node_ids

    def _query_neighbors_single_kdtree(
        self,
        kdtree: KDTree,
        source_node_ids: np.ndarray,
        target_coords: np.ndarray,
        target_node_ids: list,
    ) -> list[dict[str, Any]]:
        """
        Query neighbors from a single KDTree and create edge data.

        Parameters
        ----------
        kdtree : KDTree
            KDTree of source nodes to query.
        source_node_ids : np.ndarray
            Array of source node IDs corresponding to KDTree points.
        target_coords : np.ndarray
            Coordinates of target nodes to query for.
        target_node_ids : list
            List of target node IDs.

        Returns
        -------
        list[dict[str, Any]]
            List of edge dictionaries with source_id, target_id, and distance.
        """
        distances, neighbor_indices = kdtree.query(
            target_coords,
            k=self.n_neighbors,
            distance_upper_bound=self.distance_threshold,
        )

        is_valid = ~np.isinf(distances)

        # Convert KDTree indices (0 to n-1) back to actual node IDs
        neighbor_indices_copy = neighbor_indices.copy()
        neighbor_indices_copy[is_valid] = source_node_ids[neighbor_indices_copy[is_valid]]

        edges_data = []
        for target_id, neigh_ids, neigh_dist, neigh_valid in zip(
            target_node_ids, neighbor_indices_copy, distances, is_valid, strict=True
        ):
            for source_id, dist in zip(neigh_ids[neigh_valid].tolist(), neigh_dist[neigh_valid].tolist(), strict=True):
                edges_data.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        self.output_key: dist,
                    }
                )

        return edges_data

    def _add_edges_per_time(
        self,
        t: int,
        *,
        graph: BaseGraph,
    ) -> list[dict[str, Any]]:
        """
        Add distance-based edges between nodes at consecutive time points.

        Finds nodes at time t and previous time points (t-1 to t-delta_t),
        computes pairwise distances using KDTree, and creates edges between
        nearby nodes within the distance threshold.

        The behavior depends on the `neighbors_per_frame` parameter:
        - If False (default): Queries all previous frames as one combined KDTree,
          returning up to `n_neighbors` total connections.
        - If True: Queries each previous frame separately, returning up to
          `n_neighbors` connections per frame.

        Parameters
        ----------
        t : int
            The current time point. Edges will be created from nodes at
            previous time points to nodes at time t.
        graph : BaseGraph
            The graph to add edges to.

        Returns
        -------
        list[dict[str, Any]]
            List of edge dictionaries to be added to the graph.
        """
        attr_keys = self._get_spatial_attr_keys(graph)

        # Get current time point nodes
        current_filter = graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == t)

        if current_filter.is_empty():
            LOG.warning("No nodes found for time point %d", t)
            return []

        cur_attrs = current_filter.node_attrs(attr_keys=attr_keys)
        cur_coords = cur_attrs.to_numpy()
        cur_node_ids = list(current_filter.node_ids())

        edges_data = []

        if self.neighbors_per_frame:
            # Query each previous time frame separately
            for prev_t in range(int(t - self.delta_t), int(t)):
                kdtree, _, prev_node_ids = self._build_kdtree_data(graph, prev_t, attr_keys)

                if kdtree is None:
                    LOG.warning("No nodes found for time point %d", prev_t)
                    continue

                frame_edges = self._query_neighbors_single_kdtree(
                    kdtree,
                    np.asarray(prev_node_ids),
                    cur_coords,
                    cur_node_ids,
                )
                edges_data.extend(frame_edges)
        else:
            # Query all previous frames as one combined KDTree (original behavior)
            if self.delta_t == 1:
                # Faster path for single frame
                prev_filter = graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == t - 1)
            else:
                # Range filter for multiple frames
                prev_filter = graph.filter(
                    NodeAttr(DEFAULT_ATTR_KEYS.T) >= t - self.delta_t,
                    NodeAttr(DEFAULT_ATTR_KEYS.T) < t,
                )

            if prev_filter.is_empty():
                LOG.warning(
                    "No nodes found for time point in range (%d <= t < %d)",
                    t - self.delta_t,
                    t,
                )
                return []

            prev_attrs = prev_filter.node_attrs(attr_keys=attr_keys)
            prev_node_ids = np.asarray(list(prev_filter.node_ids()))
            prev_kdtree = KDTree(prev_attrs.to_numpy())

            edges_data = self._query_neighbors_single_kdtree(
                prev_kdtree,
                prev_node_ids,
                cur_coords,
                cur_node_ids,
            )

        if len(edges_data) == 0:
            LOG.warning("No valid edges found for time point %d", t)

        return edges_data
