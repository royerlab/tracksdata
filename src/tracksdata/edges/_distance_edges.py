from collections.abc import Sequence
from typing import Any

import numpy as np
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
    delta_t : int, default 1
        The number of time points to consider for adding edges.
        For each node at time t, edges will be created to the closest
        n_neighbors nodes at time t-1 to t-delta_t.
    output_key : str, default DEFAULT_ATTR_KEYS.EDGE_WEIGHT
        The attribute key to store the distance values in the edges.
    attr_keys : Sequence[str] | None, optional
        The node attribute keys to use for distance calculation. If None,
        defaults to ["z", "y", "x"] if "z" exists, otherwise ["y", "x"].

    Attributes
    ----------
    distance_threshold : float
        The distance threshold for adding edges.
    n_neighbors : int
        The maximum number of neighbors to consider for adding edges.
        This in respect from the current to the previous frame.
        That means, a node in frame t will have edges to the closest
        n_neighbors nodes in frame t-1.
    output_key : str
        The key used to store distance values in edges.
    attr_keys : Sequence[str] | None
        The attribute keys to use for the distance calculation.
        When None, "z", "y", "x" are used.
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

    def _init_edge_attrs(self, graph: BaseGraph) -> None:
        """
        Initialize the edge attributes for the graph.
        """
        if self.output_key not in graph.edge_attr_keys:
            graph.add_edge_attr_key(self.output_key, default_value=-99999.0)

    def _add_edges_per_time(
        self,
        t: int,
        *,
        graph: BaseGraph,
    ) -> list[dict[str, Any]]:
        """
        Add distance-based edges between nodes at consecutive time points.

        Finds nodes at time t-1 and t, computes pairwise distances using KDTree,
        and creates edges between nearby nodes within the distance threshold.
        Uses bulk edge insertion for efficiency.

        Parameters
        ----------
        t : int
            The current time point. Edges will be created from nodes at
            time t-1 to nodes at time t.
        graph : BaseGraph
            The current time point. Edges will be created from nodes at
            time t-1 to nodes at time t.
        """
        if self.attr_keys is None:
            if "z" in graph.node_attr_keys:
                attr_keys = ["z", "y", "x"]
            else:
                attr_keys = ["y", "x"]
        else:
            attr_keys = self.attr_keys

        if self.delta_t == 1:
            # faster than the range filter
            prev_filter = graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == t - 1)
        else:
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

        current_filter = graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == t)

        if current_filter.is_empty():
            LOG.warning(
                "No nodes found for time point %d",
                t,
            )
            return []

        prev_attrs = prev_filter.node_attrs(attr_keys=attr_keys)
        cur_attrs = current_filter.node_attrs(attr_keys=attr_keys)

        prev_kdtree = KDTree(prev_attrs.to_numpy())

        distances, prev_neigh_ids = prev_kdtree.query(
            cur_attrs.to_numpy(),
            k=self.n_neighbors,
            distance_upper_bound=self.distance_threshold,
        )
        is_valid = ~np.isinf(distances)

        prev_node_ids = np.asarray(prev_filter.node_ids())
        # kdtree return from 0 to n-1
        # converting back to arbitrary indexing
        prev_neigh_ids[is_valid] = prev_node_ids[prev_neigh_ids[is_valid]]

        edges_data = []
        for cur_id, neigh_ids, neigh_dist, neigh_valid in zip(
            current_filter.node_ids(), prev_neigh_ids, distances, is_valid, strict=True
        ):
            for neigh_id, dist in zip(neigh_ids[neigh_valid].tolist(), neigh_dist[neigh_valid].tolist(), strict=True):
                edges_data.append(
                    {
                        "source_id": neigh_id,
                        "target_id": cur_id,
                        self.output_key: dist,
                    }
                )

        if len(edges_data) == 0:
            LOG.warning("No valid edges found for the pair of time point (%d, %d)", t, t - 1)

        return edges_data
