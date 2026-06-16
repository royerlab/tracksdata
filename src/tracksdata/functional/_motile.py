from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import networkx as nx

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph

if TYPE_CHECKING:
    import motile
else:
    motile = Any


def to_motile_graph(
    graph: BaseGraph,
    *,
    node_attr_keys: Sequence[str] | None = None,
    edge_attr_keys: Sequence[str] | None = None,
    frame_attribute: str = DEFAULT_ATTR_KEYS.T,
) -> "motile.TrackGraph":
    """
    Convert a tracksdata graph into a [`motile.TrackGraph`](https://funkelab.github.io/motile/).

    Node and edge attributes are copied over so they can be used as `motile` costs.
    Each node keeps its ``frame_attribute`` (time) value, which `motile` requires.

    Parameters
    ----------
    graph : BaseGraph
        The graph to convert.
    node_attr_keys : Sequence[str] | None
        Node attribute keys to copy. If None, all node attributes are copied.
        ``NODE_ID`` and ``frame_attribute`` are always included.
    edge_attr_keys : Sequence[str] | None
        Edge attribute keys to copy. If None, all edge attributes are copied.
    frame_attribute : str
        Node attribute used as the time/frame dimension. Defaults to ``"t"``.

    Returns
    -------
    motile.TrackGraph
        A `motile` track graph with the same nodes, edges, and copied attributes.
    """
    try:
        import motile
    except ImportError as e:
        raise ImportError(
            "`motile` is required to convert a graph to a `motile.TrackGraph`.\n"
            "Please install it with `pip install motile`."
        ) from e

    if node_attr_keys is not None:
        node_attr_keys = list(dict.fromkeys([DEFAULT_ATTR_KEYS.NODE_ID, frame_attribute, *node_attr_keys]))

    nodes_df = graph.node_attrs(attr_keys=node_attr_keys)

    if frame_attribute not in nodes_df.columns:
        raise ValueError(
            f"Frame attribute '{frame_attribute}' not found in the graph node attributes {nodes_df.columns}."
        )

    nx_graph = nx.DiGraph()
    for node_data in nodes_df.iter_rows(named=True):
        nx_graph.add_node(node_data[DEFAULT_ATTR_KEYS.NODE_ID], **node_data)

    # avoid querying edge columns that may not be registered yet when there are no edges
    if graph.num_edges() == 0:
        edge_attr_keys = []
    elif edge_attr_keys is not None:
        edge_attr_keys = list(edge_attr_keys)

    edges_df = graph.edge_attrs(attr_keys=edge_attr_keys)
    for edge_data in edges_df.iter_rows(named=True):
        source = edge_data.pop(DEFAULT_ATTR_KEYS.EDGE_SOURCE)
        target = edge_data.pop(DEFAULT_ATTR_KEYS.EDGE_TARGET)
        edge_data.pop(DEFAULT_ATTR_KEYS.EDGE_ID, None)
        nx_graph.add_edge(source, target, **edge_data)

    return motile.TrackGraph(nx_graph, frame_attribute=frame_attribute)
