import numpy as np

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._generic_edges import GenericFuncEdgeAttrs
from tracksdata.functional._mask import mask_iou


class IoUEdgeAttr(GenericFuncEdgeAttrs):
    """
    Add weights to the edges of the graph based on the IoU
    of the masks of the nodes.

    Parameters
    ----------
    output_key : str
        The key to use for the output of the IoU.
    mask_key : str
        The key to use for the masks of the nodes.
    bbox_key : str
        The key to use for the bounding boxes of the nodes.
    """

    def __init__(
        self,
        output_key: str,
        mask_key: str = DEFAULT_ATTR_KEYS.MASK,
        bbox_key: str = DEFAULT_ATTR_KEYS.BBOX,
    ):
        def _compute_iou(source_attrs: dict[str, np.ndarray], target_attrs: dict[str, np.ndarray]) -> float:
            return mask_iou(
                source_attrs[bbox_key], source_attrs[mask_key], target_attrs[bbox_key], target_attrs[mask_key]
            )

        super().__init__(
            func=_compute_iou,
            attr_keys=[mask_key, bbox_key],
            output_key=output_key,
        )
