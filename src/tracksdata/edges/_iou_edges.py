from typing import Any

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._generic_edges import GenericFuncEdgeAttrs
from tracksdata.nodes._mask import Mask, _decode_mask, mask_iou


def _mask_iou(source_mask: "Mask | dict[str, Any]", target_mask: "Mask | dict[str, Any]") -> float:
    """IoU between two mask attribute values (struct dicts or `Mask` values)."""
    return mask_iou(_decode_mask(source_mask), _decode_mask(target_mask))


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
    """

    def __init__(
        self,
        output_key: str,
        mask_key: str = DEFAULT_ATTR_KEYS.MASK,
    ):
        super().__init__(
            func=_mask_iou,
            attr_keys=mask_key,
            output_key=output_key,
        )
