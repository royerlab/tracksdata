import time
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.utils._logging import LOG

if TYPE_CHECKING:
    from tracksdata.graph._base_graph import BaseGraph
    from tracksdata.graph.filters._base_filter import BaseFilter


class DataFrameSpatialFilter:
    """
    Internal spatial filter implementation using spatial_graph library.

    This class provides the low-level spatial indexing functionality for efficiently
    querying nodes within spatial regions of interest. It wraps the spatial_graph
    library to create a spatial index from node coordinates.

    Parameters
    ----------
    indices : pl.Series
        Series containing node IDs to be indexed.
    df : pl.DataFrame
        DataFrame containing spatial coordinates for each node.
        Each column represents a spatial dimension.
    """

    def __init__(
        self,
        indices: pl.Series,
        df: pl.DataFrame,
    ) -> None:
        from spatial_graph import PointRTree

        start_time = time.time()
        self._attr_keys = df.columns

        if df.is_empty():
            self._node_rtree = None
            return

        indices = np.ascontiguousarray(indices.to_numpy(), dtype=np.int64).copy()
        node_pos = np.ascontiguousarray(df.to_numpy(), dtype=np.float32)
        self._ndims = node_pos.shape[1]
        self._node_rtree = PointRTree(
            item_dtype="int64",
            coord_dtype="float32",
            dims=self._ndims,
        )
        self._node_rtree.insert_point_items(indices, node_pos)

        end_time = time.time()
        LOG.info(f"Time to create spatial graph: {end_time - start_time} seconds")

    def __getitem__(self, keys: tuple[slice, ...]) -> list[int]:
        """
        Query nodes within a spatial region of interest.

        Uses the spatial index to efficiently find all nodes whose coordinates
        fall within the specified rectangular region defined by the slice bounds.

        Parameters
        ----------
        keys : tuple[slice, ...]
            Tuple of slices defining the spatial bounds for each coordinate dimension.
            Each slice must have both start and stop values defined.
            The number of slices must match the number of spatial dimensions.

        Returns
        -------
        list[int]
            List of node IDs that fall within the specified spatial region.

        Raises
        ------
        ValueError
            If any slice is missing start or stop values, or if the number of
            slices doesn't match the number of spatial dimensions.
        """

        if self._node_rtree is None:
            return []

        for key in keys:
            if key.start is None or key.stop is None:
                raise ValueError(f"Slice {key} must have start and stop")

        if len(keys) != self._ndims:
            raise ValueError(f"Expected {self._ndims} keys, got {len(keys)}")

        start_time = time.time()

        roi = np.stack(
            [[s.start, s.stop] for s in keys],  # subtractring 1e-8 because the spatial graph is inclusive
            axis=1,
            dtype=np.float32,
        )
        node_ids = self._node_rtree.search(*roi)

        end_time = time.time()

        LOG.info(f"Time to query nodes in ROI: {end_time - start_time} seconds")

        return node_ids.tolist()


class SpatialFilter:
    """
    Spatial filtering for graph nodes using spatial indexing.

    This filter creates a spatial index of graph nodes based on their spatial coordinates
    and allows efficient querying of nodes within spatial regions of interest (ROI).

    Parameters
    ----------
    graph : BaseGraph
        The graph containing nodes with spatial coordinates.
    attr_keys : list[str] | None, optional
        List of attribute keys to use as spatial coordinates. If None, defaults to
        ["t", "z", "y", "x"] filtered to only include keys present in the graph.

    Examples
    --------
    ```python
    graph = RustWorkXGraph()
    # Add nodes with spatial coordinates
    graph.add_node({"t": 0, "y": 10, "x": 20})
    graph.add_node({"t": 1, "y": 30, "x": 40})

    # Create spatial filter with 2D coordinates
    spatial_filter = SpatialFilter(graph, attr_keys=["y", "x"])

    # Query nodes in spatial region
    subgraph = spatial_filter[0:50, 0:50]
    ```
    """

    def __init__(
        self,
        graph: "BaseGraph",
        attr_keys: list[str] | None = None,
    ) -> None:
        if attr_keys is None:
            attr_keys = ["t", "z", "y", "x"]
            valid_keys = set(graph.node_attr_keys)
            attr_keys = list(filter(lambda x: x in valid_keys, attr_keys))

        self._graph = graph

        nodes_df = graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, *attr_keys])
        node_ids = nodes_df[DEFAULT_ATTR_KEYS.NODE_ID]

        self._df_filter = DataFrameSpatialFilter(indices=node_ids, df=nodes_df.select(attr_keys))

    def __getitem__(self, keys: tuple[slice, ...]) -> "BaseFilter":
        """
        Query nodes within a spatial region of interest.

        Uses spatial indexing to efficiently find nodes whose coordinates fall within
        the specified bounds for each spatial dimension.

        Parameters
        ----------
        keys : tuple[slice, ...]
            Tuple of slices defining the spatial bounds for each coordinate dimension.
            Must match the number of coordinate dimensions specified in attr_keys.
            Each slice defines [start, stop) bounds for that dimension.

        Returns
        -------
        BaseFilter
            A filter containing only nodes and their edges that fall within the spatial ROI.

        Raises
        ------
        ValueError
            If the number of slices doesn't match the number of coordinate dimensions.

        Examples
        --------
        ```python
        # For 2D spatial filter with ["y", "x"] coordinates
        filter = spatial_filter[10:50, 20:60]

        # For 4D spatial filter with ["t", "z", "y", "x"] coordinates
        subgraph = spatial_filter[0:10, 0:5, 10:50, 20:60].subgraph()
        ```
        """
        node_ids = self._df_filter[keys]
        return self._graph.filter(node_ids=node_ids)


class BBoxSpatialFilter:
    """
    Spatial filter for bounding box queries on graph nodes.

    This filter allows querying nodes within a bounding box defined by minimum and
    maximum coordinates for each spatial dimension.

    Parameters
    ----------
    graph : BaseGraph
        The graph containing nodes with spatial coordinates.
    frame_attr_key : str | None, optional
        The attribute key representing the frame or time dimension.
        Default is None.
        If None it will only use the bounding box coordinates.
    bbox_attr_key : str, optional
        The attribute key representing the bounding box coordinates.
        Defaults to `DEFAULT_ATTR_KEYS.BBOX`.
        The bounding box coordinates should be in the format:
        [min_x, min_y, min_z, ..., max_x, max_y, max_z, ...]
        where each dimension has a min and max value.
    """

    def __init__(
        self,
        graph: "BaseGraph",
        frame_attr_key: str | None = DEFAULT_ATTR_KEYS.T,
        bbox_attr_key: str = DEFAULT_ATTR_KEYS.BBOX,
    ) -> None:
        from spatial_graph import PointRTree

        self._graph = graph

        if frame_attr_key is None:
            attr_keys = [DEFAULT_ATTR_KEYS.NODE_ID, bbox_attr_key]
        else:
            attr_keys = [DEFAULT_ATTR_KEYS.NODE_ID, frame_attr_key, bbox_attr_key]
        nodes_df = graph.node_attrs(attr_keys=attr_keys)
        node_ids = np.ascontiguousarray(nodes_df[DEFAULT_ATTR_KEYS.NODE_ID].to_numpy(), dtype=np.int64).copy()

        if nodes_df.is_empty():
            self._node_rtree = None
        else:
            bboxes = nodes_df[bbox_attr_key].to_numpy()
            if bboxes.shape[1] % 2 != 0:
                raise ValueError(f"Bounding box coordinates must have even number of dimensions, got {bboxes.shape[1]}")
            num_dims = bboxes.shape[1] // 2

            if frame_attr_key is None:
                self._ndims = num_dims
                positions_min = np.ascontiguousarray(bboxes[:, :num_dims], dtype=np.float32)
                positions_max = np.ascontiguousarray(bboxes[:, num_dims:], dtype=np.float32)
            else:
                frames = nodes_df[frame_attr_key].to_numpy()
                self._ndims = num_dims + 1  # +1 for the frame dimension
                positions_min = np.ascontiguousarray(
                    np.hstack((frames[:, np.newaxis], bboxes[:, :num_dims])), dtype=np.float32
                )
                positions_max = np.ascontiguousarray(
                    np.hstack((frames[:, np.newaxis], bboxes[:, num_dims:])), dtype=np.float32
                )
            self._node_rtree = PointRTree(
                item_dtype="int64",
                coord_dtype="float32",
                dims=self._ndims,
            )
            self._node_rtree.insert_bb_items(node_ids, positions_min, positions_max)

    def __getitem__(self, keys: tuple[slice, ...]) -> "BaseFilter":
        """
        Query nodes that overlaps with a spatial region of interest.

        Uses spatial indexing to efficiently find nodes whose bounding boxes fall within
        the specified bounds for each spatial dimension.

        Parameters
        ----------
        keys : tuple[slice, ...]
            Tuple of slices defining the spatial bounds for each coordinate dimension.
            Must match the number of coordinate dimensions specified in attr_keys.
            Each slice defines [start, stop] bounds for that dimension.

        Returns
        -------
        BaseFilter
            A filter containing only nodes and their edges that fall within the spatial ROI.

        Raises
        ------
        ValueError
            If the number of slices doesn't match the number of coordinate dimensions.

        Examples
        --------
        ```python
        # For 2D spatial filter with the bounding boxes
        # overlapping with the region [10, 20] x [50, 60]
        spatial_filter = BBoxSpatialFilter(graph, bbox_attr_key="bbox")
        filter = spatial_filter[10:50, 20:60]

        # For 4D spatial filter with the t in [0, 10] and
        # bounding boxes overlapping with the region [0, 5] x [10, 50] x [20, 60]
        spatial_filter = BBoxSpatialFilter(graph, frame_attr_key="t", bbox_attr_key="bbox")
        subgraph = spatial_filter[0:10, 0:5, 10:50, 20:60].subgraph()
        ```
        """

        if self._node_rtree is None:
            return self._graph.filter(node_ids=[])

        for key in keys:
            if key.start is None or key.stop is None:
                raise ValueError(f"Slice {key} must have start and stop")

        if len(keys) != self._ndims:
            raise ValueError(f"Expected {self._ndims} keys, got {len(keys)}")

        node_ids = self._node_rtree.search(
            *(
                np.stack(
                    [[s.start, s.stop] for s in keys],
                    axis=1,
                    dtype=np.float32,
                )
            )
        )
        return self._graph.filter(node_ids=node_ids)
