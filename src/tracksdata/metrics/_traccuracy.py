from typing import TYPE_CHECKING, Any

import networkx as nx

from tracksdata.array._graph_array import GraphArrayView
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph

if TYPE_CHECKING:
    from traccuracy import TrackingGraph
else:
    TrackingGraph = Any


def to_traccuracy_graph(
    graph: BaseGraph,
    array_view_kwargs: dict[str, Any] | None = None,
    location_keys: list[str] | None = None,
) -> "TrackingGraph":
    """
    Convert a tracksdata graph to a traccuracy graph.

    Parameters
    ----------
    graph : BaseGraph
        The graph to convert.
    array_view_kwargs : dict[str, Any] | None
        Additional keyword arguments to pass to the `GraphArrayView` constructor used to create the segmentation.
    location_keys : list[str] | None
        The keys of the location attributes to use for the segmentation.
        If None, the location keys are inferred from the intersection of the graph node attributes and
        the list [DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X].

    Returns
    -------
    TrackingGraph
        A traccuracy graph.
    """

    try:
        from traccuracy import TrackingGraph
    except ImportError as e:
        raise ImportError(
            "`traccuracy` is required to evaluate TRAccuracy metrics.\nPlease install it with `pip install traccuracy`."
        ) from e

    if array_view_kwargs is None:
        array_view_kwargs = {}

    node_attrs = graph.node_attrs()
    nx_graph = nx.DiGraph()

    for node_data in node_attrs.iter_rows(named=True):
        node_data["segmentation_id"] = node_data[DEFAULT_ATTR_KEYS.NODE_ID]
        nx_graph.add_node(node_data[DEFAULT_ATTR_KEYS.NODE_ID], **node_data)

    edge_attrs = graph.edge_attrs()
    for edge_data in edge_attrs.iter_rows(named=True):
        nx_graph.add_edge(edge_data[DEFAULT_ATTR_KEYS.EDGE_SOURCE], edge_data[DEFAULT_ATTR_KEYS.EDGE_TARGET])

    segmentation = GraphArrayView(graph, attr_key=DEFAULT_ATTR_KEYS.NODE_ID, **array_view_kwargs)

    if location_keys is None:
        location_keys = [DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
        node_attr_keys = graph.node_attr_keys()
        location_keys = [key for key in location_keys if key in node_attr_keys]

    return TrackingGraph(nx_graph, segmentation, location_keys=location_keys)
