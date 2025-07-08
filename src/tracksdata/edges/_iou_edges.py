from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._generic_edges import GenericFuncEdgeAttrs
from tracksdata.nodes._mask import Mask


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
            func=Mask.iou,
            attr_keys=mask_key,
            output_key=output_key,
        )
