from typing import Any, Literal

import numpy as np
from tqdm import tqdm

from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator


class RandomNodes(BaseNodesOperator):
    """
    Generate random node coordinates for testing and simulation purposes.

    RandomNodes creates nodes with randomly distributed coordinates within the
    unit hypercube [0,1]^n. This is useful for testing tracking algorithms,
    generating synthetic datasets, or creating baseline comparisons. The number
    of nodes per time point can vary randomly within a specified range.

    Parameters
    ----------
    n_time_points : int
        The number of time points to generate nodes for.
    n_nodes_per_tp : tuple[int, int]
        The minimum and maximum number of nodes to generate per time point.
        The actual number is randomly chosen within this range for each time point.
    n_dim : Literal[2, 3], default 3
        The spatial dimensionality of the coordinates:

        - 2: generates (x, y) coordinates
        - 3: generates (x, y, z) coordinates
    random_state : int, default 0
        Random seed for reproducible results.
    show_progress : bool, default False
        Whether to display progress bars during node generation.

    Attributes
    ----------
    n_time_points : int
        Number of time points to generate.
    n_nodes : tuple[int, int]
        Range of nodes per time point.
    spatial_cols : list[str]
        Names of spatial coordinate columns.
    rng : np.random.Generator
        Random number generator instance.

    See Also
    --------
    :class:`tracksdata.nodes.RegionPropsNodes`
        Extract nodes from segmented images using region properties.
    :class:`tracksdata.nodes.Mask`
        Node operator for mask-based objects.

    Examples
    --------
    Generate 2D random nodes:

    ```python
    from tracksdata.nodes import RandomNodes

    node_op = RandomNodes(n_time_points=10, n_nodes_per_tp=(5, 15), n_dim=2, random_state=42)
    ```

    Add nodes to a graph:

    ```python
    node_op.add_nodes(graph)
    ```

    Generate nodes for a specific time point:

    ```python
    node_op.add_nodes(graph, t=5)
    ```

    Use 3D coordinates with consistent node count:

    ```python
    node_op = RandomNodes(
        n_time_points=20,
        n_nodes_per_tp=(10, 10),  # exactly 10 nodes per time point
        n_dim=3,
    )
    ```
    """

    def __init__(
        self,
        n_time_points: int,
        n_nodes_per_tp: tuple[int, int],
        n_dim: Literal[2, 3] = 3,
        random_state: int = 0,
        show_progress: bool = False,
    ):
        super().__init__(show_progress=show_progress)
        if isinstance(n_nodes_per_tp, int):
            raise ValueError("`n_nodes_per_tp` must be a tuple of two integers")

        self.n_time_points = n_time_points
        self.n_nodes = n_nodes_per_tp

        if n_dim == 2:
            self.spatial_cols = ["x", "y"]
        elif n_dim == 3:
            self.spatial_cols = ["x", "y", "z"]
        else:
            raise ValueError(f"Invalid number of dimensions: {n_dim}")

        self.rng = np.random.default_rng(random_state)

    def add_nodes(
        self,
        graph: BaseGraph,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Override the base add_nodes method to handle n_time_points parameter.

        When t=None, iterates over range(n_time_points) instead of graph.time_points().
        When t is specified, uses the base implementation.
        """
        if t is None:
            for t in tqdm(range(self.n_time_points), disable=not self.show_progress, desc="Adding nodes"):
                self._add_nodes_per_time(
                    graph,
                    t=t,
                    **kwargs,
                )
        else:
            self._add_nodes_per_time(
                graph,
                t=t,
                **kwargs,
            )

    def _add_nodes_per_time(
        self,
        graph: BaseGraph,
        *,
        t: int,
        **kwargs: Any,
    ) -> None:
        """
        Add nodes for a specific time point.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add nodes to.
        t : int
            The time point to add nodes for.
        **kwargs : Any
            Additional keyword arguments to pass to add_node.
        """
        # Register each spatial column individually
        for col in self.spatial_cols:
            if col not in graph.node_attr_keys:
                graph.add_node_attr_key(col, None)

        n_nodes_at_t = self.rng.integers(
            self.n_nodes[0],
            self.n_nodes[1],
        )

        coords = self.rng.uniform(
            low=0,
            high=1,
            size=(n_nodes_at_t, len(self.spatial_cols)),
        )

        graph.bulk_add_nodes(
            [{"t": t, **dict(zip(self.spatial_cols, c.tolist(), strict=True)), **kwargs} for c in coords]
        )
