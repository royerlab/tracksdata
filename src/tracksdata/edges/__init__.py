"""Edge operators for creating connections between nodes of a graph."""

from tracksdata.edges._distance_edges import DistanceEdges
from tracksdata.edges._generic_edges import GenericFuncEdgeAttrs
from tracksdata.edges._iou_edges import IoUEdgeAttr

__all__ = ["DistanceEdges", "GenericFuncEdgeAttrs", "IoUEdgeAttr"]
