from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._base_node_attrs import BaseNodeAttrsOperator
from tracksdata.utils._logging import LOG

T = TypeVar("T")
R = TypeVar("R")


class GenericFuncNodeAttrs(BaseNodeAttrsOperator):
    """
    Operator to apply a function to a node and insert the result as a new attribute.

    Parameters
    ----------
    func : Callable[[T], R] | Callable[[list[T]], list[R]]
        Function to apply to the node.
        If `frames` is provided when calling `add_node_attrs`,
        the function must accept a single argument for the frame.
        Otherwise, the function must accept two arguments for the mask and the additional arguments.
    output_key : str
        Key of the new attribute to add.
    attr_keys : Sequence[str], optional
        Additional attributes to pass to the `func` as keyword arguments.
    default_value : Any, optional
        Default value to use for the new attribute.
        TODO: this should be replaced by a more advanced typing that takes default values.
    batch_size : int, optional
        Batch size to use for the function.
        If 0, the function will be called for each node separately.
        If > 0, the function will be called for each batch of nodes and return a list of results.
        The batch size is the number of nodes that will be passed to the function at once.
        Batch only contains nodes from the same time point.

    Examples
    --------
    ```python
    video = ...
    graph = ...


    def intensity_median_times_t(image: NDArray, mask: Mask, t: int) -> float:
        cropped_frame = mask.crop(image)
        valid_pixels = cropped_frame[mask.mask]
        return np.median(valid_pixels) * t


    crop_attrs = GenericFuncNodeAttrs(
        func=intensity_median,
        output_key="intensity_median",
        attr_keys=["mask", "t"],
    )

    crop_attrs.add_node_attrs(graph, frames=video)
    ```

    With batching:

    ```python
    video = ...
    graph = ...


    def intensity_median_times_t(image: NDArray, masks: list[Mask], t: list[int]) -> list[float]:
        results = []
        for i in range(len(masks)):
            cropped_frame = masks[i].crop(image)
            valid_pixels = cropped_frame[masks[i].mask]
            value = np.median(valid_pixels) * t[i]
            results.append(value)
        return results


    crop_attrs = GenericFuncNodeAttrs(
        func=intensity_median,
        output_key="intensity_median",
        attr_keys=["mask", "t"],
    )

    crop_attrs.add_node_attrs(graph, frames=video)
    ```
    """

    output_key: str

    def __init__(
        self,
        func: Callable[[T], R] | Callable[[list[T]], list[R]],
        output_key: str,
        default_value: Any = None,
        attr_keys: Sequence[str] = (),
        batch_size: int = 0,
    ) -> None:
        super().__init__(output_key)
        self.func = func
        self.attr_keys = attr_keys
        self.default_value = default_value
        self.batch_size = batch_size

    def _init_node_attrs(self, graph: BaseGraph) -> None:
        """
        Initialize the node attributes for the graph.
        """
        if self.output_key not in graph.node_attr_keys:
            graph.add_node_attr_key(self.output_key, default_value=self.default_value)

    def add_node_attrs(
        self,
        graph: BaseGraph,
        *,
        t: int | None = None,
        frames: NDArray | None = None,
    ) -> None:
        """
        Add attributes to nodes of a graph.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add attributes to.
        t : int | None
            The time point to add attributes for. If None, add attributes for all time points.
        frames : NDArray | None
            The frames to index by time point to pass to the `func` function.
            Such that `frames[t]` will be passed to the `func` function.
        """
        super().add_node_attrs(graph, t=t, frames=frames)

    def _node_attrs_per_time(
        self,
        t: int,
        *,
        graph: BaseGraph,
        frames: NDArray | None = None,
    ) -> tuple[list[int], dict[str, list[Any]]]:
        """
        Add attributes to nodes of a graph.

        Parameters
        ----------
        t : int
            The time point to add attributes for.
        graph : BaseGraph
            The graph to add attributes to.
        frames : NDArray | None
            The frames to index by time point to pass to the `func` function.
            Such that, when provided, `frames[t]` will be passed to the `func` function.
        """
        # Get node IDs for the specified time point
        graph_filter = graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == t)

        if graph_filter.is_empty():
            LOG.warning(f"No nodes at time point {t}")
            return []

        # Get attributes for these nodes
        node_attrs = graph_filter.node_attrs(attr_keys=self.attr_keys)

        args = []
        if frames is not None:
            args.append(np.asarray(frames[t]))

        results = []
        if self.batch_size > 0:
            size = len(node_attrs)
            for i in range(0, size, self.batch_size):
                batch_node_attrs = node_attrs.slice(i, self.batch_size)
                batch_results = self.func(*args, **batch_node_attrs.to_dict())
                results.extend(batch_results)

        else:
            for data_dict in node_attrs.rows(named=True):
                result = self.func(*args, **data_dict)
                results.append(result)

        return graph_filter.node_ids(), {self.output_key: results}
