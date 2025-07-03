from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from dask.array.image import imread as dask_imread
from tifffile import imread as tiff_imread

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.io._numpy_array import _add_edges_from_track_ids
from tracksdata.nodes import RegionPropsNodes
from tracksdata.utils._logging import LOG


def compressed_tracks_table(graph: BaseGraph) -> np.ndarray:
    """
    Compress the tracks of a graph into a (n, 4)-tabular format.

    Where
    - n is the number of tracks
    - 4 is the number of columns:
        - track_id: the track ID
        - start: the start frame
        - end: the end frame
        - parent_track_id: the parent track ID

    Parameters
    ----------
    graph : BaseGraph
        The graph to compress the tracks from.

    Returns
    -------
    tracks : np.ndarray
        The compressed tracks.
    """
    nodes_df = graph.node_attrs(
        attr_keys=[
            DEFAULT_ATTR_KEYS.NODE_ID,
            DEFAULT_ATTR_KEYS.T,
            DEFAULT_ATTR_KEYS.TRACK_ID,
        ]
    )

    tracks_data = []
    node_ids = []

    for (track_id,), group in nodes_df.group_by(DEFAULT_ATTR_KEYS.TRACK_ID):
        start = group[DEFAULT_ATTR_KEYS.T].min()
        end = group[DEFAULT_ATTR_KEYS.T].max()
        tracks_data.append([track_id, start, end, 0])
        node_ids.append(group.filter(pl.col(DEFAULT_ATTR_KEYS.T) == start)[DEFAULT_ATTR_KEYS.NODE_ID].item())

    parents = graph.predecessors(
        node_ids,
        attr_keys=[DEFAULT_ATTR_KEYS.TRACK_ID],
    )
    for track_id, node_id in zip(tracks_data, node_ids, strict=True):
        df = parents[node_id]
        if len(df) > 0:
            track_id[3] = df[DEFAULT_ATTR_KEYS.TRACK_ID].item()

    out_array = np.asarray(tracks_data, dtype=int)
    out_array = out_array[np.argsort(out_array[:, 0])]

    return out_array


def _load_tracks_file(tracks_file: Path) -> dict[int, int]:
    """
    Load a CTC tracks file into a graph.

    Parameters
    ----------
    tracks_file : Path
        The path to the CTC tracks .txt file.
        The unnamed columns are:
        - track_id: the track ID
        - start: the start frame
        - end: the end frame
        - parent_track_id: the parent track ID

    Returns
    -------
    track_id_graph : dict[int, int]
        A dictionary mapping track IDs to their parent track IDs.
    """
    track_id_graph = {}

    df = pl.read_csv(
        tracks_file,
        separator=" ",
        has_header=False,
        use_pyarrow=True,
    ).rename(
        {
            "column_1": "track_id",
            "column_4": "parent_track_id",
        }
    )

    df = df.filter(pl.col("parent_track_id") > 0)

    for track_id, parent_track_id in zip(
        df["track_id"],
        df["parent_track_id"],
        strict=True,
    ):
        track_id_graph[track_id] = parent_track_id

    return track_id_graph


def load_ctc(
    data_dir: str | Path,
    graph: BaseGraph,
    region_props_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Load a CTC ground truth file into a graph.
    The resulting graph will have region properties attributesfrom a CTC ground truth file.

    Graph backend method API (e.g. `graph.from_ctc`) is preferred over this function.

    Parameters
    ----------
    data_dir : str | Path
        The path to the CTC ground truth directory.
    graph : BaseGraph
        The graph to load the CTC ground truth into.
    region_props_kwargs : dict[str, Any]
        Keyword arguments to pass to RegionPropsNodes.

    Examples
    --------
    ```python
    from tracksdata.io import load_ctc
    from tracksdata.graph import RustWorkXGraph

    graph = RustWorkXGraph()
    load_ctc("Fluo-N2DL-HeLa/01_GT/TRA", graph)
    ```

    See Also
    --------
    [BaseGraph.from_ctc][tracksdata.graph._base_graph.BaseGraph.from_ctc]:
        Create a graph from a CTC data directory.

    [RegionPropsNodes][tracksdata.nodes.RegionPropsNodes]:
        Operator to create nodes from label images.

    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    if region_props_kwargs is None:
        region_props_kwargs = {}

    if "extra_properties" not in region_props_kwargs:
        region_props_kwargs["extra_properties"] = ["label"]

    elif "label" not in region_props_kwargs["extra_properties"]:
        region_props_kwargs["extra_properties"].append("label")

    tracks_file_found = False
    track_id_graph = {}

    for tracks_file in ["man_track.txt", "res_track.txt"]:
        tracks_file_path = data_dir / tracks_file
        if tracks_file_path.exists():
            tracks_file_found = True
            track_id_graph = _load_tracks_file(tracks_file_path)
            break

    if not tracks_file_found:
        LOG.warning(
            f"Tracks file {data_dir}/man_track.txt and {data_dir}/res_track.txt does not exist.\n"
            "Graph won't have edges."
        )

    labels = dask_imread(str(data_dir / "*.tif"), imread=tiff_imread)

    region_props_nodes = RegionPropsNodes(**region_props_kwargs)
    region_props_nodes.add_nodes(graph, labels=labels)

    nodes_df = graph.node_attrs(
        attr_keys=[
            DEFAULT_ATTR_KEYS.NODE_ID,
            DEFAULT_ATTR_KEYS.T,
            "label",
        ]
    )

    _add_edges_from_track_ids(
        graph,
        nodes_df,
        track_id_graph,
        "label",
    )

    # is duplicating an attribute that bad?
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.TRACK_ID, -1)
    graph.update_node_attrs(
        node_ids=nodes_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
        attrs={
            DEFAULT_ATTR_KEYS.TRACK_ID: nodes_df["label"].to_list(),
        },
    )
