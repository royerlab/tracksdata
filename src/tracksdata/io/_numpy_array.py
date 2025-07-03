import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from tqdm import tqdm

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._mask import Mask


def _add_edges_from_track_ids(
    graph: BaseGraph,
    tracks_df: pl.DataFrame,
    track_id_graph: dict[int, int],
    track_id_key: str,
) -> None:
    """
    Add edges to a graph based on track ids.

    Parameters
    ----------
    graph : BaseGraph
        The graph to add edges to.
    tracks_df : pl.DataFrame
        The dataframe containing "node_id", "t" and `track_id_key` columns.
    track_id_graph : dict[int, int]
        Mapping of division as child track id (key) to parent track id (value) relationships.
    track_id_key : str
        The column name of the track id in the dataframe.
    """
    nodes_by_track_id = tracks_df.group_by(track_id_key)

    # reducing the nodes to the last node in each track
    # to find the divisions relationships between tracks
    if len(track_id_graph) > 0:
        last_nodes = nodes_by_track_id.map_groups(lambda group: group.sort(DEFAULT_ATTR_KEYS.T).tail(1))
        track_id_to_last_node = dict(zip(last_nodes[track_id_key], last_nodes[DEFAULT_ATTR_KEYS.NODE_ID], strict=True))
    else:
        track_id_to_last_node = {}  # for completeness, it won't be used if this is true

    edges = []
    for (track_id,), group in nodes_by_track_id:
        node_ids = group.sort(DEFAULT_ATTR_KEYS.T)[DEFAULT_ATTR_KEYS.NODE_ID].to_list()

        first_node = node_ids[0]
        if track_id in track_id_graph:
            parent_node = track_id_to_last_node[track_id_graph[track_id]]
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


def load_array(
    positions: np.ndarray,
    graph: BaseGraph,
    track_ids: np.ndarray | None = None,
    track_id_graph: dict[int, int] | None = None,
    radius: ArrayLike = 1,
    image_shape: tuple[int, ...] | None = None,
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
    track_ids : np.ndarray | None
        Track ids of the nodes if available.
    track_id_graph : dict[int, int] | None
        Mapping of division as child track id (key) to parent track id (value) relationships.
    radius : ArrayLike
        Integer or N-dimensional array of radii.
    image_shape : tuple[int, ...] | None
        Shape of the image if available masks are cropped to fit the image.

    See Also
    --------
    [BaseGraph.from_array][tracksdata.graph.BaseGraph.from_array]:
        Create a graph from a numpy array.
    """
    positions = np.asarray(positions)

    if positions.shape[1] == 3:
        ndim = 2
        spatial_cols = ["x", "y"]
    elif positions.shape[1] == 4:
        ndim = 3
        spatial_cols = ["x", "y", "z"]
    else:
        raise ValueError(f"Expected 4 or 5 dimensions, got {positions.shape[1]}.")

    if image_shape is not None and len(image_shape) != ndim:
        raise ValueError(f"Expected {ndim} dimensions for `image_shape`, got {len(image_shape)}.")

    if np.isscalar(radius):
        radius = np.broadcast_to(radius, (positions.shape[0],))

    if len(radius) != positions.shape[0]:
        raise ValueError(
            "`radius` must be a scalar or have the same length as `positions`. "
            f"Expected {positions.shape[0]}, got {len(radius)}."
        )

    if track_id_graph is not None and track_ids is None:
        raise ValueError("`track_ids` must be provided if `tracks_graph` is provided.")

    if track_id_graph is None:
        track_id_graph = {}

    if track_ids is not None:
        if len(track_ids) != positions.shape[0]:
            raise ValueError(
                "`track_ids` must have the same length as `positions`. "
                f"Expected {positions.shape[0]}, got {len(track_ids)}."
            )
        graph.add_node_attr_key(DEFAULT_ATTR_KEYS.TRACK_ID, -1)
        track_ids = track_ids.tolist()

    for col in spatial_cols:
        graph.add_node_attr_key(col, -999_999)

    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    node_attrs = []

    for i, (position, rad) in tqdm(
        enumerate(zip(positions.tolist(), radius.tolist(), strict=True)),
        total=len(positions),
        desc="Generating node attributes",
    ):
        mask = Mask.from_coordinates(position[1:], rad, image_shape=image_shape)

        attr = {
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.T: position[0],
            "x": position[-1],
            "y": position[-2],
        }

        if ndim == 3:
            attr["z"] = position[1]

        if track_ids is not None:
            attr[DEFAULT_ATTR_KEYS.TRACK_ID] = track_ids[i]

        node_attrs.append(attr)

    node_ids = graph.bulk_add_nodes(node_attrs)

    if track_ids is not None:
        tracks_df = pl.DataFrame(
            {
                DEFAULT_ATTR_KEYS.TRACK_ID: track_ids,
                DEFAULT_ATTR_KEYS.T: positions[:, 0],
                DEFAULT_ATTR_KEYS.NODE_ID: node_ids,
            }
        )
        _add_edges_from_track_ids(
            graph,
            tracks_df,
            track_id_graph,
            DEFAULT_ATTR_KEYS.TRACK_ID,
        )
