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
) -> "TrackingGraph":
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

    return TrackingGraph(nx_graph, segmentation)
