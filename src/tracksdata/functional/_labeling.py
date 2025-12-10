import polars as pl
import rustworkx as rx

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._edges import join_node_attrs_to_edges
from tracksdata.graph import BaseGraph
from tracksdata.utils._logging import LOG


def _ancestral_edges(tracklet_graph: rx.PyDiGraph) -> set[tuple[int, int]]:
    """
    Find the ancestral edges of a tracklet subgraph, this includes a self-loop for each node.

    Parameters
    ----------
    tracklet_graph : rx.PyDiGraph
        The tracklet graph.

    Returns
    -------
    set[tuple[int, int]]
        The ancestral edges of the tracklet subgraph.
    """
    # transverse graph backwards to find tracklet paths
    path_edges = set()

    for rx_target_id in tracklet_graph.node_indices():
        target_track_id = tracklet_graph[rx_target_id]
        LOG.info("Finding path edges for %d", target_track_id)
        path_edges.add((target_track_id, target_track_id))  # path to self is considered valid
        for rx_source_id in rx.ancestors(tracklet_graph, rx_target_id):
            source_track_id = tracklet_graph[rx_source_id]
            LOG.info("Adding path edge %d -> %d", source_track_id, target_track_id)
            path_edges.add((source_track_id, target_track_id))

    return path_edges


def _input_graph_ancestral_edges(
    edge_attrs: pl.DataFrame,
    ancestral_edges: set[tuple[int, int]],
) -> pl.DataFrame:
    """
    Selects the edges in the input graph that are part of ancestral paths.

    Parameters
    ----------
    edge_attrs : pl.DataFrame
        The edge attributes dataframe.
    ancestral_edges : set[tuple[int, int]]
        The ancestral edges of the tracklet subgraph.

    Returns
    -------
    list[int]
        The edge indices of the input graph that are part of ancestral paths.
    """
    cols = [f"source_{DEFAULT_ATTR_KEYS.TRACKLET_ID}", f"target_{DEFAULT_ATTR_KEYS.TRACKLET_ID}"]

    valid_edges_df = pl.DataFrame(
        list(ancestral_edges),
        schema=cols,
        orient="row",
    )
    valid_edges_df = valid_edges_df.with_columns(
        pl.lit(True).alias("ancestral_edge"),
    )

    edge_attrs = edge_attrs.join(
        valid_edges_df,
        on=cols,
        how="left",
    ).with_columns(
        pl.col("ancestral_edge").fill_null(False).cast(pl.Boolean),
    )

    return edge_attrs.filter(pl.col("ancestral_edge"))[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()


def ancestral_connected_edges(
    input_graph: BaseGraph,
    reference_graph: BaseGraph,
    match: bool = True,
) -> list[int]:
    """
    Let an ancestral path be any sequence from (target, source)-edges in the `reference_graph`.
    This function returns the subset of edges in the `input_graph` that are
    part of an ancestral path in the reference graph.

    IMPORTANT: This function updates the `input_graph` in place when matching
    with the `reference_graph`.

    Parameters
    ----------
    input_graph : BaseGraph
        The input graph.
    reference_graph : BaseGraph
        The reference graph.
    match : bool, optional
        Whether to match the input graph with the reference graph.
    """
    if DEFAULT_ATTR_KEYS.TRACKLET_ID not in reference_graph.node_attr_keys():
        tracklet_graph = reference_graph.assign_tracklet_ids()
    else:
        tracklet_graph = reference_graph.tracklet_graph()

    if match:
        input_graph.match(reference_graph)

    elif DEFAULT_ATTR_KEYS.MATCHED_NODE_ID not in input_graph.node_attr_keys():
        raise ValueError(
            "`ancestral_connected_edges` requires the input graph to previously matched "
            f"and have a `{DEFAULT_ATTR_KEYS.MATCHED_NODE_ID}` column when `match=False`"
        )

    in_node_attrs = input_graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    ref_node_attrs = reference_graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.TRACKLET_ID])

    in_node_attrs = in_node_attrs.filter(
        pl.col(DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) >= 0,
    ).join(
        ref_node_attrs,
        left_on=DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
        right_on=DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    )

    edge_attrs = input_graph.edge_attrs(attr_keys=[])

    edge_attrs = join_node_attrs_to_edges(
        node_attrs=in_node_attrs.select(DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.TRACKLET_ID),
        edge_attrs=edge_attrs,
    )

    tracklet_ancestral_edges = _ancestral_edges(tracklet_graph)
    input_graph_ancestral_edges = _input_graph_ancestral_edges(
        edge_attrs=edge_attrs,
        ancestral_edges=tracklet_ancestral_edges,
    )

    return input_graph_ancestral_edges
