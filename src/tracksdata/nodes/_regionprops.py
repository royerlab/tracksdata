from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.measure._regionprops import RegionProperties, regionprops
from toolz import curry
from typing_extensions import override

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.nodes._mask import Mask
from tracksdata.utils._logging import LOG
from tracksdata.utils._multiprocessing import multiprocessing_apply


class RegionPropsNodes(BaseNodesOperator):
    """
    Operator that adds nodes to a graph using scikit-image's regionprops.

    Extracts region properties from labeled images to create graph nodes using
    scikit-image's regionprops function to compute geometric and intensity-based
    features. Automatically adds centroid coordinates and mask information, with
    additional properties computed based on the extra_properties parameter.

    Parameters
    ----------
    extra_properties : list[str | Callable[[RegionProperties], Any]] | None, optional
        Additional properties to compute for each region. Can be:
        - String names of built-in regionprops properties (e.g., 'area', 'perimeter')
        - Callable functions that take a RegionProperties object and return a value
        If None, only centroid coordinates and masks are extracted.
    spacing : tuple[float, float] | None, optional
        Physical spacing between pixels. If provided, affects distance-based
        measurements. Should be (row_spacing, col_spacing) for 2D or
        (depth_spacing, row_spacing, col_spacing) for 3D.

    Attributes
    ----------
    _extra_properties : list
        List of additional properties to compute.
    _spacing : tuple[float, float] | None
        Physical spacing between pixels.

    Examples
    --------
    Create a basic RegionPropsNodes operator:

    ```python
    from tracksdata.nodes import RegionPropsNodes

    node_op = RegionPropsNodes()
    ```

    Add common geometric properties:

    ```python
    node_op = RegionPropsNodes(extra_properties=["area", "perimeter", "eccentricity"])
    ```

    Add custom properties using functions:

    ```python
    def custom_property(region):
        return region.area / region.perimeter


    node_op = RegionPropsNodes(extra_properties=["area", custom_property])
    ```

    Use with physical spacing:

    ```python
    node_op = RegionPropsNodes(
        spacing=(0.5, 0.1, 0.1),  # z, y, x spacing
        extra_properties=["area", "volume"],
    )
    ```

    Add nodes from a time series:

    ```python
    labels_series = np.random.randint(0, 10, (10, 100, 100))
    node_op.add_nodes(graph, labels=labels_series)
    ```
    """

    def __init__(
        self,
        extra_properties: list[str | Callable[[RegionProperties], Any]] | None = None,
        spacing: tuple[float, float] | None = None,
    ):
        super().__init__()
        self._extra_properties = extra_properties or []
        self._spacing = spacing

    def _axis_names(self, labels: NDArray[np.integer]) -> list[str]:
        """
        Get the names of the axes of the labels.

        Parameters
        ----------
        labels : NDArray[np.integer]
            The (t + nD) labels to get the axis names for.

        Returns
        -------
        list[str]
            The names of the axes of the labels.
        """
        if labels.ndim == 3:
            return ["y", "x"]
        elif labels.ndim == 4:
            return ["z", "y", "x"]
        else:
            raise ValueError(f"`labels` must be 't + 2D' or 't + 3D', got '{labels.ndim}' dimensions.")

    def _init_node_attrs(self, graph: BaseGraph, axis_names: list[str]) -> None:
        """
        Initialize the node attributes for the graph.
        """
        for attr_key in [DEFAULT_ATTR_KEYS.MASK, DEFAULT_ATTR_KEYS.BBOX]:
            if attr_key not in graph.node_attr_keys:
                graph.add_node_attr_key(attr_key, None)

        # initialize the attribute keys
        for attr_key in axis_names + self.attr_keys():
            if attr_key not in graph.node_attr_keys:
                graph.add_node_attr_key(attr_key, -1.0)

    def attr_keys(self) -> list[str]:
        """
        Get the keys of the node attributes that will be extracted.

        Returns only the keys for extra_properties. The centroid coordinates
        (x, y, z) and mask are always included but not listed here.

        Returns
        -------
        list[str]
            List of attribute key names that will be added to nodes.

        Examples
        --------
        ```python
        node_op = RegionPropsNodes(extra_properties=["area", "perimeter"])
        keys = node_op.attr_keys()
        print(keys)  # ['area', 'perimeter']
        ```
        """
        return [prop.__name__ if callable(prop) else prop for prop in self._extra_properties]

    @override
    def add_nodes(
        self,
        graph: BaseGraph,
        *,
        labels: NDArray[np.integer],
        t: int | None = None,
        intensity_image: NDArray | None = None,
    ) -> None:
        """
        Add nodes to a graph using region properties from labeled images.

        Extracts region properties from labeled images and creates corresponding
        nodes in the graph. Can handle both single time point and time series data.
        When t is None, the first axis of labels represents time and processes each
        time point sequentially. Automatically initializes required attribute keys
        in the graph schema before adding nodes.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add nodes to.
        labels : NDArray[np.integer]
            Labeled image(s) where each unique positive integer represents
            a different region/object. Can be:
            - 3D array (time, height, width) for 2D time series
            - 4D array (time, depth, height, width) for 3D time series
            When `t` is provided, it should be padded to include the time dimension.
        t : int | None, optional
            Time point for the nodes. If None, labels are treated as a time
            series where the first axis represents time.
        intensity_image : NDArray | None, optional
            Intensity image(s) corresponding to the labels. Used for computing
            intensity-based properties. Must have the same shape as labels
            (excluding the label values).

        Examples
        --------
        Add nodes from a single 2D labeled image:

        ```python
        labels = skimage.measure.label(binary_image)
        node_op.add_nodes(graph, labels=labels, t=0)
        ```

        Add nodes from a time series:

        ```python
        labels_series = np.stack(
            [
                skimage.measure.label(binary_image_t0),
                skimage.measure.label(binary_image_t1),
            ]
        )
        node_op.add_nodes(graph, labels=labels_series)
        ```

        Add nodes with intensity information:

        ```python
        node_op.add_nodes(graph, labels=labels, t=0, intensity_image=fluorescence_image)
        ```
        """
        axis_names = self._axis_names(labels)
        self._init_node_attrs(graph, axis_names)

        if t is None:
            time_points = range(labels.shape[0])
        else:
            time_points = [t]

        node_ids = []
        for nodes_data in multiprocessing_apply(
            func=curry(self._nodes_per_time, labels=labels, intensity_image=intensity_image),
            sequence=time_points,
            desc="Adding region properties nodes",
        ):
            node_ids.extend(graph.bulk_add_nodes(nodes_data))

    def _nodes_per_time(
        self,
        t: int,
        *,
        labels: NDArray[np.integer],
        intensity_image: NDArray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add nodes for a specific time point using region properties.

        Processes a single time point, computing region properties for each labeled
        region and creating corresponding graph nodes. Determines spatial dimensions
        from label shape, ensures required attribute keys exist, computes region
        properties, extracts coordinates and extra properties, creates mask objects,
        and bulk adds all nodes.

        Parameters
        ----------
        t : int
            The time point to assign to the created nodes.
        labels : NDArray[np.integer]
            2D or 3D labeled image for a single time point.
        intensity_image : NDArray | None, optional
            Corresponding intensity image for computing intensity-based properties.

        Returns
        -------
        list[dict[str, Any]]
            The nodes to add to the graph.

        Raises
        ------
        ValueError
            If labels is not 2D or 3D.
        """
        axis_names = self._axis_names(labels)

        labels = np.asarray(labels[t])

        if intensity_image is not None:
            intensity_image = np.asarray(intensity_image[t])

        nodes_data = []

        for obj in regionprops(
            labels,
            intensity_image=intensity_image,
            spacing=self._spacing,
            cache=True,
        ):
            attrs = dict(zip(axis_names, obj.centroid, strict=False))

            for prop in self._extra_properties:
                if callable(prop):
                    attrs[prop.__name__] = prop(obj)
                else:
                    attrs[prop] = getattr(obj, prop)

            attrs[DEFAULT_ATTR_KEYS.MASK] = Mask(obj.image, obj.bbox)
            attrs[DEFAULT_ATTR_KEYS.BBOX] = np.asarray(obj.bbox, dtype=int)
            attrs[DEFAULT_ATTR_KEYS.T] = t

            nodes_data.append(attrs)
            obj._cache.clear()  # clearing to reduce memory footprint

        if len(nodes_data) == 0:
            LOG.warning("No valid nodes found for time point %d", t)

        return nodes_data
