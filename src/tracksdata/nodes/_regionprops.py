from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.measure._regionprops import RegionProperties, regionprops
from tqdm import tqdm
from typing_extensions import override

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.nodes._mask import Mask
from tracksdata.utils._logging import LOG


class RegionPropsNodes(BaseNodesOperator):
    """
    Operator that adds nodes to a graph using scikit-image's regionprops.

    Extracts region properties from labeled images to create graph nodes using
    scikit-image's regionprops function to compute geometric and intensity-based
    features. Automatically adds centroid coordinates and mask information, with
    additional properties computed based on the extra_properties parameter.

    Parameters
    ----------
    cache : bool, default True
        Whether to enable caching of computed properties in regionprops.
        Caching improves performance when accessing multiple properties
        but uses more memory.
    extra_properties : list[str | Callable[[RegionProperties], Any]] | None, optional
        Additional properties to compute for each region. Can be:
        - String names of built-in regionprops properties (e.g., 'area', 'perimeter')
        - Callable functions that take a RegionProperties object and return a value
        If None, only centroid coordinates and masks are extracted.
    spacing : tuple[float, float] | None, optional
        Physical spacing between pixels. If provided, affects distance-based
        measurements. Should be (row_spacing, col_spacing) for 2D or
        (depth_spacing, row_spacing, col_spacing) for 3D.
    show_progress : bool, default True
        Whether to display progress bars during node addition.

    Attributes
    ----------
    _cache : bool
        Whether caching is enabled for regionprops.
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
        cache: bool = True,
        extra_properties: list[str | Callable[[RegionProperties], Any]] | None = None,
        spacing: tuple[float, float] | None = None,
        show_progress: bool = True,
    ):
        super().__init__(show_progress=show_progress)
        self._cache = cache
        self._extra_properties = extra_properties or []
        self._spacing = spacing

    def attrs_keys(self) -> list[str]:
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
        keys = node_op.attrs_keys()
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
            - 2D array (height, width) for single time point
            - 3D array (depth, height, width) for single 3D volume
            - 3D array (time, height, width) for 2D time series
            - 4D array (time, depth, height, width) for 3D time series
            `t` must be provided if the labels does not have a time dimension.
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
        if t is None:
            for t in tqdm(range(labels.shape[0]), disable=not self.show_progress, desc="Adding nodes"):
                if intensity_image is not None:
                    self._add_nodes_per_time(
                        graph=graph,
                        labels=labels[t],
                        t=t,
                        intensity_image=intensity_image[t],
                    )
                else:
                    self._add_nodes_per_time(
                        graph=graph,
                        labels=labels[t],
                        t=t,
                    )
        else:
            self._add_nodes_per_time(
                graph=graph,
                labels=labels,
                t=t,
                intensity_image=intensity_image,
            )

    def _add_nodes_per_time(
        self,
        graph: BaseGraph,
        *,
        labels: NDArray[np.integer],
        t: int,
        intensity_image: NDArray | None = None,
    ) -> None:
        """
        Add nodes for a specific time point using region properties.

        Processes a single time point, computing region properties for each labeled
        region and creating corresponding graph nodes. Determines spatial dimensions
        from label shape, ensures required attribute keys exist, computes region
        properties, extracts coordinates and extra properties, creates mask objects,
        and bulk adds all nodes. Region property caches are cleared for memory efficiency.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add nodes to.
        labels : NDArray[np.integer]
            2D or 3D labeled image for a single time point.
        t : int
            The time point to assign to the created nodes.
        intensity_image : NDArray | None, optional
            Corresponding intensity image for computing intensity-based properties.

        Raises
        ------
        ValueError
            If labels is not 2D or 3D.
        """
        if labels.ndim == 2:
            axis_names = ["y", "x"]
        elif labels.ndim == 3:
            axis_names = ["z", "y", "x"]
        else:
            raise ValueError(f"`labels` must be 2D or 3D, got {labels.ndim} dimensions.")

        if DEFAULT_ATTR_KEYS.MASK not in graph.node_attr_keys:
            graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

        # initialize the attribute keys
        for attr_key in axis_names + [p.__name__ if callable(p) else p for p in self._extra_properties]:
            if attr_key not in graph.node_attr_keys:
                graph.add_node_attr_key(attr_key, -1.0)

        labels = np.asarray(labels)

        nodes_data = []

        for obj in regionprops(
            labels,
            intensity_image=intensity_image,
            spacing=self._spacing,
        ):
            attrs = dict(zip(axis_names, obj.centroid, strict=False))

            for prop in self._extra_properties:
                if callable(prop):
                    attrs[prop.__name__] = prop(obj)
                else:
                    attrs[prop] = getattr(obj, prop)

            attrs[DEFAULT_ATTR_KEYS.MASK] = Mask(obj.image, obj.bbox)
            attrs[DEFAULT_ATTR_KEYS.T] = t

            nodes_data.append(attrs)
            obj._cache.clear()  # clearing to reduce memory footprint

        if len(nodes_data) > 0:
            graph.bulk_add_nodes(nodes_data)
        else:
            LOG.warning("No valid nodes found for time point %d", t)
