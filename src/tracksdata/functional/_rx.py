import numpy as np
import polars as pl
import rustworkx as rx
from numba import njit, typed, types

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.utils._logging import LOG

NO_PARENT = -1


@njit
def _fast_path_transverse(
    node: int,
    dag: dict[int, int],
) -> list[int]:
    """
    Traverse a path in the tracking directed graph following parent→child relationships.

    Parameters
    ----------
    node : int
        Source path node.
    dag : dict[int, int]
        Directed graph (tree) of paths relationships.

    Returns
    -------
    list[int]
        Sequence of nodes in the path.
    """
    path = typed.List.empty_list(types.int64)

    while node != NO_PARENT:
        path.append(node)
        if node in dag:
            node = dag[node]
        else:
            node = NO_PARENT

    return path


@njit
def _fast_dag_transverse(
    starts: np.ndarray,
    dag: dict[int, int],
    track_id_offset: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    """
    Traverse the tracks DAG creating a distinct id to each linear path.

    This algorithm handles asymmetric division by treating each parent→child
    relationship as a single linear path. Dividing edges (nodes with multiple
    children) are processed separately as "long edges" to create proper track
    relationships in the output graph.

    Parameters
    ----------
    starts : np.ndarray
        Track starting points (nodes without parents).
    dag : dict[int, int]
        Directed acyclic graph mapping parent → child for linear paths only.
        Dividing edges are excluded and handled separately.
    track_id_offset : int
        The starting track id, useful when assigning track ids to a subgraph.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[int, int], dict[int, int]]
        - sequence of paths: List of numpy arrays, each containing node IDs in a linear path
        - their respective track_id: Track ID for each path
        - length: Length of each path
        - last_to_track_id: Maps last node in each path to its track_id
        - first_to_track_id: Maps first node in each path to its track_id
    """
    paths = []
    track_ids = []
    lengths = []
    last_to_track_id = {}
    first_to_track_id = {}

    track_id = track_id_offset

    for start in starts:
        path = _fast_path_transverse(start, dag)
        paths.append(path)
        track_ids.append(track_id)
        lengths.append(len(path))
        last_to_track_id[path[-1]] = track_id
        first_to_track_id[path[0]] = track_id
        track_id += 1

    paths = [np.asarray(p) for p in paths]
    track_ids = np.asarray(track_ids)
    lengths = np.asarray(lengths)

    return paths, track_ids, lengths, last_to_track_id, first_to_track_id


@njit
def _numba_build_dict(keys: np.ndarray, values: np.ndarray) -> dict[int, int]:
    """
    Creates a numba-compatible dict from keys and values. When used to build a
    directed acyclic graph (DAG), the keys should be parent nodes and the values should be
    child nodes.

    Important:
    Parent-less (orphan) nodes should not be provided when building the DAG.

    Parameters
    ----------
    keys : np.ndarray
        Nodes indices.
    values : np.ndarray
        Parent indices.

    Returns
    -------
    dict[int, int]
        Dict that maps the keys to values.
    """
    dag = {}

    for i in range(len(keys)):
        dag[keys[i]] = values[i]

    return dag


def _rx_graph_to_dict_dag(graph: rx.PyDiGraph) -> tuple[dict[int, int], np.ndarray, pl.DataFrame]:
    """Creates the DAG of track lineages, separating linear paths from dividing edges.

    This function implements the core logic for handling asymmetric division by:
    1. Identifying dividing edges (nodes with multiple children)
    2. Identifying temporal gaps (edges spanning > 1 time unit)
    3. Creating a simplified DAG with only linear parent→child relationships
    4. Returning dividing/long edges separately for track relationship creation

    Parameters
    ----------
    graph : rx.PyDiGraph
        Directed acyclic graph of nodes with time attributes.

    Returns
    -------
    tuple[dict[int, int], np.ndarray, pl.DataFrame]
        - Linear DAG: Maps parent → child for linear paths only (excludes dividing edges)
        - Starts: Array of track starting node IDs (nodes without parents)
        - Long edges: DataFrame of (source, target) edges that create new track relationships
          (includes both dividing edges and temporal gaps > 1 time unit)
    """
    # target are the children
    # source are the parents
    node_indices = np.asarray(graph.node_indices(), dtype=np.int64)
    times = [n[DEFAULT_ATTR_KEYS.T] for n in graph.nodes()]
    nodes_df = pl.DataFrame(
        {"node_id": node_indices, DEFAULT_ATTR_KEYS.T: times},
    )
    edges_df = pl.from_numpy(
        np.asarray(graph.edge_list(), dtype=np.int64),
        schema=["source", "target"],
    )
    try:
        edges_df = (
            edges_df.join(
                nodes_df,
                left_on="target",
                right_on="node_id",
                how="left",
                validate="1:1",
            )
            .with_columns(pl.col("t").alias("t_target"))
            .join(
                nodes_df,
                left_on="source",
                right_on="node_id",
                how="left",
                validate="m:1",
                suffix="_source",
            )
        )

    except pl.exceptions.ComputeError as e:
        if "join keys did not fulfill 1:1" in str(e):
            raise RuntimeError("Invalid graph structure, found node with multiple parents") from e
        else:
            raise e

    # ASYMMETRIC DIVISION FIX:
    # The key insight is to separate dividing edges (nodes with multiple children)
    # from linear path edges. This prevents the algorithm from incorrectly handling
    # asymmetric branching patterns where one branch continues linearly while
    # another creates a new track.
    dividing_mask = edges_df["source"].is_duplicated()  # Nodes with multiple children
    long_edges_mask = (edges_df["t_target"] - edges_df["t_source"]).abs() > 1  # Temporal gaps
    both_mask = dividing_mask | long_edges_mask

    long_edges_df = edges_df.filter(both_mask)  # Edges that create new track relationships

    edges_df = edges_df.filter(~both_mask)  # Only linear parent→child edges remain
    nodes_df = (
        nodes_df.with_columns(pl.col("node_id").alias("target"))
        .join(edges_df, on="target", how="left", validate="1:1")
        .select("target", "source")
    )

    # above we convert to numpy representation and then create numba dict
    # inside a njit function, otherwise it's very slow
    has_parent = nodes_df["source"].is_not_null()
    starts = nodes_df["target"].filter(~has_parent).cast(pl.Int64).to_numpy()

    # nodes_array is a (target, source) 2xN-array
    # source is before target, and therefore the parent nodes
    nodes_arr = nodes_df.filter(has_parent).to_numpy(order="fortran").T
    linear_dag = _numba_build_dict(nodes_arr[1], nodes_arr[0])

    return linear_dag, starts, long_edges_df


@njit
def _track_id_edges_from_long_edges(
    source: np.ndarray,
    target: np.ndarray,
    first_to_track_id: dict[int, int],
    last_to_track_id: dict[int, int],
    track_id_to_rx_node_id: dict[int, int],
) -> list[tuple[int, int]]:
    """
    Compute the track_id edges from the long edges.

    Parameters
    ----------
    source : np.ndarray
        Source nodes.
    target : np.ndarray
        Target nodes.
    first_to_track_id : dict[int, int]
        First node -> track_id.
    last_to_track_id : dict[int, int]
        Last node -> track_id.
    track_id_to_rx_node_id : dict[int, int]
        Maps track_id to node_id of the rx graph of tracklets.

    Returns
    -------
    list[tuple[int, int]]
        List of track_id edges.
    """
    edges = []
    for i in range(len(source)):
        child_track_id = track_id_to_rx_node_id[first_to_track_id[target[i]]]
        parent_track_id = track_id_to_rx_node_id[last_to_track_id[source[i]]]
        edges.append((parent_track_id, child_track_id))
    return edges


def _assign_track_ids(
    graph: rx.PyDiGraph,
    track_id_offset: int,
) -> tuple[np.ndarray, np.ndarray, rx.PyDiGraph]:
    """
    Assigns an unique `track_id` to each simple path in the graph and
    their respective parent -> child relationships.

    Parameters
    ----------
    graph : rx.PyDiGraph
        Directed acyclic graph of tracks.
    track_id_offset : int
        The starting track id, useful when assigning track ids to a subgraph.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, rx.PyDiGraph]
        - node_ids: Sequence of node ids.
        - track_ids: The respective track_id for each node.
        - tracks_graph: Graph indicating the parent -> child relationships.
    """
    if graph.num_nodes() == 0:
        raise ValueError("Graph is empty")

    LOG.info(f"Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")

    # was it better (faster) when using a numpy array for the digraph as in ultrack?
    linear_dag, starts, long_edges_df = _rx_graph_to_dict_dag(graph)

    paths, track_ids, lengths, last_to_track_id, first_to_track_id = _fast_dag_transverse(
        starts, linear_dag, track_id_offset
    )

    n_tracks = len(track_ids)

    tracks_graph = rx.PyDiGraph(node_count_hint=n_tracks, edge_count_hint=n_tracks)

    node_ids = tracks_graph.add_nodes_from(track_ids)
    track_id_to_rx_node_id = _numba_build_dict(
        np.asarray(track_ids, dtype=np.int64),
        np.asarray(node_ids, dtype=np.int64),
    )
    if len(long_edges_df) > 0:
        edges = _track_id_edges_from_long_edges(
            long_edges_df["source"].to_numpy(),
            long_edges_df["target"].to_numpy(),
            first_to_track_id,
            last_to_track_id,
            track_id_to_rx_node_id,
        )
        tracks_graph.add_edges_from_no_data(edges)

    paths = np.concatenate(paths)
    nodes_track_ids = np.repeat(track_ids, lengths)

    return paths, nodes_track_ids, tracks_graph
