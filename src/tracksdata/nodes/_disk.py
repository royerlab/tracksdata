from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import skimage.morphology as morph
from numpy.typing import NDArray

from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._base_node_attrs import BaseNodeAttrsOperator
from tracksdata.utils._logging import LOG


@lru_cache(maxsize=5)
def _spherical_mask(
    radius: int,
    ndim: int,
) -> NDArray[np.bool_]:
    """
    Get a spherical mask of a given radius and dimension.
    """
    if ndim == 2:
        return morph.disk(radius).astype(bool)

    if ndim == 3:
        return morph.ball(radius).astype(bool)

    raise ValueError(f"Spherical is only implemented for 2D and 3D, got ndim={ndim}")


def _create_mask_and_bbox_from_coordinates(
    center: NDArray,
    radius: int,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """
    Create a mask and bounding box from center coordinates and radius.

    Parameters
    ----------
    center : NDArray
        The center of the mask.
    radius : int
        The radius of the mask.
    image_shape : tuple[int, ...] | None
        The shape of the image. When provided, crops regions outside the image.

    Returns
    -------
    tuple[NDArray[np.bool_], NDArray[np.int64]]
        The mask and bounding box arrays.
    """
    mask = _spherical_mask(radius, len(center))
    center = np.round(center).astype(int)

    start = center - np.asarray(mask.shape) // 2
    end = start + mask.shape

    if image_shape is None:
        bbox = np.concatenate([start, end])
    else:
        processed_start = np.maximum(start, 0)
        processed_end = np.minimum(end, image_shape)

        start_overhang = processed_start - start
        end_overhang = end - processed_end

        mask = mask[tuple(slice(s, -e if e > 0 else None) for s, e in zip(start_overhang, end_overhang, strict=True))]

        bbox = np.concatenate([processed_start, processed_end])

    return mask, bbox


class DiskMaskAttrs(BaseNodeAttrsOperator):
    """
    Operator to create disk masks and bounding boxes for each node.

    Creates spherical masks in space, so temporal information should not be provided.

    Parameters
    ----------
    radius : int
        The radius of the mask.
    image_shape : tuple[int, ...]
        The shape of the image, must match the number of attr_keys.
    attr_keys : Sequence[str] | None
        The attributes for the center of the mask.
        If not provided, "z", "y", "x" will be used.
    mask_output_key : str
        The key to store the mask attribute.
    bbox_output_key : str
        The key to store the bounding box attribute.
    """

    def __init__(
        self,
        radius: int,
        image_shape: tuple[int, ...],
        attr_keys: Sequence[str] | None = None,
        mask_output_key: str = DEFAULT_ATTR_KEYS.MASK,
        bbox_output_key: str = DEFAULT_ATTR_KEYS.BBOX,
    ):
        super().__init__(mask_output_key)  # Primary output key for base class

        if attr_keys is None:
            default_columns = ["z", "y", "x"]
            attr_keys = default_columns[-len(image_shape) :]

        if len(attr_keys) != len(image_shape):
            raise ValueError(
                f"Expected image shape {image_shape} to have the same number of dimensions as attr_keys '{attr_keys}'."
            )

        self.radius = radius
        self.image_shape = image_shape
        self.attr_keys = attr_keys
        self.mask_output_key = mask_output_key
        self.bbox_output_key = bbox_output_key

    def _init_node_attrs(self, graph: BaseGraph) -> None:
        """Initialize the node attributes for the graph."""
        if self.mask_output_key not in graph.node_attr_keys:
            graph.add_node_attr_key(self.mask_output_key, default_value=None)
        if self.bbox_output_key not in graph.node_attr_keys:
            graph.add_node_attr_key(self.bbox_output_key, default_value=None)

    def _node_attrs_per_time(
        self,
        t: int,
        *,
        graph: BaseGraph,
        frames: NDArray | None = None,
    ) -> tuple[list[int], dict[str, list]]:
        """
        Add mask and bbox attributes to nodes for a specific time point.
        """
        # Get node IDs for the specified time point
        graph_filter = graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == t)

        if graph_filter.is_empty():
            LOG.warning(f"No nodes at time point {t}")
            return [], {}

        # Get attributes for these nodes
        node_attrs = graph_filter.node_attrs(attr_keys=self.attr_keys)

        masks = []
        bboxes = []

        for data_dict in node_attrs.rows(named=True):
            center = np.asarray([data_dict[key] for key in self.attr_keys])
            mask, bbox = _create_mask_and_bbox_from_coordinates(
                center=center,
                radius=self.radius,
                image_shape=self.image_shape,
            )
            masks.append(mask)
            bboxes.append(bbox)

        return graph_filter.node_ids(), {
            self.mask_output_key: masks,
            self.bbox_output_key: bboxes,
        }
