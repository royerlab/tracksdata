from collections.abc import Callable, Sequence
from enum import Enum
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._base_node_attrs import BaseNodeAttrsOperator
from tracksdata.nodes._mask import Mask
from tracksdata.utils._logging import LOG

R = TypeVar("R")
T = TypeVar("T")


class CropFuncAttrs(BaseNodeAttrsOperator):
    """
    Operator to apply a function to a node and insert the result as a new attribute.

    Parameters
    ----------
    func : Callable[[Mask, NDArray, R], T] | Callable[[Mask, R], T]
        Function to apply to the node.
        If `frames` is provided when calling `add_node_attrs`,
        the function must accept a single argument for the frame.
        Otherwise, the function must accept two arguments for the mask and the additional arguments.
    output_key : str
        Key of the new attribute to add.
    attr_keys : Sequence[str], optional
        Additional attributes to pass to the `func` as keyword arguments.
    show_progress : bool, optional
        Whether to show a progress bar.

    Examples
    --------
    ```python
    video = ...
    graph = ...


    def intensity_median_times_t(mask: Mask, image: NDArray, t: int) -> float:
        cropped_frame = mask.crop(image)
        valid_pixels = cropped_frame[mask.mask]
        return np.median(valid_pixels) * t


    crop_attrs = CropFuncAttrs(
        func=intensity_median,
        output_key="intensity_median",
        attr_keys=["t"],
    )

    crop_attrs.add_node_attrs(graph, frames=video)
    ```

    """

    def __init__(
        self,
        func: Callable[[Mask, NDArray, R], T] | Callable[[Mask, R], T],
        output_key: str,
        attr_keys: Sequence[str] = (),
        show_progress: bool = True,
    ) -> None:
        super().__init__(output_key, show_progress)
        self.func = func
        self.attr_keys = attr_keys

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

    def _add_node_attrs_per_time(
        self,
        graph: BaseGraph,
        *,
        t: int,
        frames: NDArray | None = None,
    ) -> None:
        """
        Add attributes to nodes of a graph.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add attributes to.
        t : int
            The time point to add attributes for.
        frames : NDArray | None
            The frames to index by time point to pass to the `func` function.
            Such that, when provided, `frames[t]` will be passed to the `func` function.
        """
        # Get node IDs for the specified time point
        node_ids = graph.filter_nodes_by_attrs(NodeAttr(DEFAULT_ATTR_KEYS.T) == t)

        if len(node_ids) == 0:
            LOG.warning(f"No nodes at time point {t}")
            return  # No nodes at this time point

        # Get attributes for these nodes
        columns = [DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK, *self.attr_keys]
        node_attrs = graph.node_attrs(node_ids=node_ids, attr_keys=columns)

        args = []
        if frames is not None:
            args.append(np.asarray(frames[t]))

        results = []
        for data_dict in node_attrs.select(columns[1:]).rows(named=True):
            result = self.func(data_dict.pop(DEFAULT_ATTR_KEYS.MASK), *args, **data_dict)
            results.append(result)

        # Add output key as node attribute key and guessing the type
        if self.output_key not in graph.node_attr_keys:
            sample_result = results[0]
            #  if is scalar, add as float, otherwise add as object
            if isinstance(sample_result, int | float | bool | str | Enum):
                default_value = type(sample_result)(-9999)
            else:
                default_value = None

            graph.add_node_attr_key(self.output_key, default_value)

        graph.update_node_attrs(
            attrs={self.output_key: results},
            node_ids=node_attrs[DEFAULT_ATTR_KEYS.NODE_ID],
        )
