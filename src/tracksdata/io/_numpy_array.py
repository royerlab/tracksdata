import numpy as np
import polars as pl
from tqdm import tqdm

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph


def _add_edges_from_tracklet_ids(
    graph: BaseGraph,
    tracks_df: pl.DataFrame,
    tracklet_id_graph: dict[int, int],
    tracklet_id_key: str,
) -> None:
    """
    Add edges to a graph based on track ids.

    Parameters
    ----------
    graph : BaseGraph
        The graph to add edges to.
    tracks_df : pl.DataFrame
        The dataframe containing "node_id", "t" and `tracklet_id_key` columns.
    tracklet_id_graph : dict[int, int]
        Mapping of division as child track id (key) to parent track id (value) relationships.
    tracklet_id_key : str
        The column name of the track id in the dataframe.
    """
    nodes_by_tracklet_id = tracks_df.group_by(tracklet_id_key)

    # reducing the nodes to the last node in each track
    # to find the divisions relationships between tracks
    if len(tracklet_id_graph) > 0:
        last_nodes = nodes_by_tracklet_id.map_groups(lambda group: group.sort(DEFAULT_ATTR_KEYS.T).tail(1))
        tracklet_id_to_last_node = dict(
            zip(last_nodes[tracklet_id_key], last_nodes[DEFAULT_ATTR_KEYS.NODE_ID], strict=True)
        )
    else:
        tracklet_id_to_last_node = {}  # for completeness, it won't be used if this is true

    edges = []
    for (tracklet_id,), group in nodes_by_tracklet_id:
        node_ids = group.sort(DEFAULT_ATTR_KEYS.T)[DEFAULT_ATTR_KEYS.NODE_ID].to_list()

        first_node = node_ids[0]
        if tracklet_id in tracklet_id_graph:
            parent_node = tracklet_id_to_last_node[tracklet_id_graph[tracklet_id]]
            edges.append(
                {
                    DEFAULT_ATTR_KEYS.EDGE_SOURCE: parent_node,
                    DEFAULT_ATTR_KEYS.EDGE_TARGET: first_node,
                }
            )

        for i in range(len(node_ids) - 1):
            edges.append(
                {
                    DEFAULT_ATTR_KEYS.EDGE_SOURCE: node_ids[i],
                    DEFAULT_ATTR_KEYS.EDGE_TARGET: node_ids[i + 1],
                }
            )

    graph.bulk_add_edges(edges)


def from_array(
    positions: np.ndarray,
    graph: BaseGraph,
    tracklet_ids: np.ndarray | None = None,
    tracklet_id_graph: dict[int, int] | None = None,
) -> None:
    """
    Load a numpy array content into a graph.

    Parameters
    ----------
    positions : np.ndarray
        (N, 4 or 3) dimensional array of positions.
        Defined by (T, (Z), Y, X) coordinates.
    graph : BaseGraph
        The graph to load the data into.
    tracklet_ids : np.ndarray | None
        Track ids of the nodes if available.
    tracklet_id_graph : dict[int, int] | None
        Mapping of division as child track id (key) to parent track id (value) relationships.

    See Also
    --------
    [BaseGraph.from_array][tracksdata.graph.BaseGraph.from_array]:
        Create a graph from a numpy array.
    """
    positions = np.asarray(positions)

    if positions.shape[1] == 3:
        ndim = 2
        spatial_cols = [DEFAULT_ATTR_KEYS.X, DEFAULT_ATTR_KEYS.Y]
    elif positions.shape[1] == 4:
        ndim = 3
        spatial_cols = [DEFAULT_ATTR_KEYS.X, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.Z]
    else:
        raise ValueError(f"Expected 4 or 5 dimensions, got {positions.shape[1]}.")

    if tracklet_id_graph is not None and tracklet_ids is None:
        raise ValueError("`tracklet_ids` must be provided if `tracks_graph` is provided.")

    if tracklet_id_graph is None:
        tracklet_id_graph = {}

    if tracklet_ids is not None:
        if len(tracklet_ids) != positions.shape[0]:
            raise ValueError(
                "`tracklet_ids` must have the same length as `positions`. "
                f"Expected {positions.shape[0]}, got {len(tracklet_ids)}."
            )
        graph.add_node_attr_key(DEFAULT_ATTR_KEYS.TRACKLET_ID, pl.Int64, -1)
        tracklet_ids = tracklet_ids.tolist()

    for col in spatial_cols:
        graph.add_node_attr_key(col, pl.Float64, -999_999)

    node_attrs = []

    for i, position in tqdm(
        enumerate(positions.tolist()),
        total=len(positions),
        desc="Generating node attributes",
    ):
        attr = {
            DEFAULT_ATTR_KEYS.T: position[0],
            DEFAULT_ATTR_KEYS.X: position[-1],
            DEFAULT_ATTR_KEYS.Y: position[-2],
        }

        if ndim == 3:
            attr[DEFAULT_ATTR_KEYS.Z] = position[1]

        if tracklet_ids is not None:
            attr[DEFAULT_ATTR_KEYS.TRACKLET_ID] = tracklet_ids[i]

        node_attrs.append(attr)

    node_ids = graph.bulk_add_nodes(node_attrs)

    if tracklet_ids is not None:
        tracks_df = pl.DataFrame(
            {
                DEFAULT_ATTR_KEYS.TRACKLET_ID: tracklet_ids,
                DEFAULT_ATTR_KEYS.T: positions[:, 0],
                DEFAULT_ATTR_KEYS.NODE_ID: node_ids,
            }
        )
        _add_edges_from_tracklet_ids(
            graph,
            tracks_df,
            tracklet_id_graph,
            DEFAULT_ATTR_KEYS.TRACKLET_ID,
        )
