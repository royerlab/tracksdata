import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import tifffile as tiff
from dask.array.image import imread as dask_imread

from tracksdata.array._graph_array import GraphArrayView
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.io._numpy_array import _add_edges_from_tracklet_ids
from tracksdata.nodes import RegionPropsNodes
from tracksdata.utils._logging import LOG
from tracksdata.utils._multiprocessing import multiprocessing_apply


def compressed_tracks_table(graph: BaseGraph) -> np.ndarray:
    """
    Compress the tracks of a graph into a (n, 4)-tabular format.

    Where
    - n is the number of tracks
    - 4 is the number of columns:
        - tracklet_id: the track ID
        - start: the start frame
        - end: the end frame
        - parent_tracklet_id: the parent track ID

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
            DEFAULT_ATTR_KEYS.TRACKLET_ID,
        ]
    )

    tracks_data = []
    node_ids = []

    for (tracklet_id,), group in nodes_df.group_by(DEFAULT_ATTR_KEYS.TRACKLET_ID):
        start = group[DEFAULT_ATTR_KEYS.T].min()
        end = group[DEFAULT_ATTR_KEYS.T].max()
        tracks_data.append([tracklet_id, start, end, 0])
        node_ids.append(group.filter(pl.col(DEFAULT_ATTR_KEYS.T) == start)[DEFAULT_ATTR_KEYS.NODE_ID].item())

    parents = graph.predecessors(
        node_ids,
        attr_keys=[DEFAULT_ATTR_KEYS.TRACKLET_ID],
    )
    for tracklet_id, node_id in zip(tracks_data, node_ids, strict=True):
        df = parents[node_id]
        if len(df) > 0:
            tracklet_id[3] = df[DEFAULT_ATTR_KEYS.TRACKLET_ID].item()

    if len(tracks_data) == 0:
        return np.empty((0, 4), dtype=int)

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
        - tracklet_id: the track ID
        - start: the start frame
        - end: the end frame
        - parent_tracklet_id: the parent track ID

    Returns
    -------
    tracklet_id_graph : dict[int, int]
        A dictionary mapping track IDs to their parent track IDs.
    """
    tracklet_id_graph = {}

    try:
        df = pl.read_csv(
            tracks_file,
            separator=" ",
            has_header=False,
            use_pyarrow=True,
        )
    except pa.ArrowInvalid:
        # pyarrow cannot read csv with a single row
        df = pl.read_csv(
            tracks_file,
            separator=" ",
            has_header=False,
            use_pyarrow=False,
        )

    df = df.rename(
        {
            "column_1": "tracklet_id",
            "column_4": "parent_tracklet_id",
        }
    )

    df = df.filter(pl.col("parent_tracklet_id") > 0)

    for tracklet_id, parent_tracklet_id in zip(
        df["tracklet_id"],
        df["parent_tracklet_id"],
        strict=True,
    ):
        tracklet_id_graph[tracklet_id] = parent_tracklet_id

    return tracklet_id_graph


def from_ctc(
    data_dir: str | Path,
    graph: BaseGraph,
    region_props_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Load a CTC ground truth file into a graph.
    The resulting graph will have region properties attributes from a CTC data directory.

    Graph backend method API (e.g. `graph.from_ctc`) is preferred over this function.

    Parameters
    ----------
    data_dir : str | Path
        The path to the CTC data directory.
    graph : BaseGraph
        The graph to load the CTC data into.
    region_props_kwargs : dict[str, Any]
        Keyword arguments to pass to RegionPropsNodes.

    Examples
    --------
    ```python
    from tracksdata.io import from_ctc
    from tracksdata.graph import RustWorkXGraph

    graph = RustWorkXGraph()
    from_ctc("Fluo-N2DL-HeLa/01_GT/TRA", graph)
    ```

    See Also
    --------
    [BaseGraph.from_ctc][tracksdata.graph.BaseGraph.from_ctc]:
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
    tracklet_id_graph = {}

    for tracks_file in ["man_track.txt", "res_track.txt"]:
        tracks_file_path = data_dir / tracks_file
        if tracks_file_path.exists():
            tracks_file_found = True
            tracklet_id_graph = _load_tracks_file(tracks_file_path)
            break

    if not tracks_file_found:
        LOG.warning(
            f"Tracks file {data_dir}/man_track.txt and {data_dir}/res_track.txt does not exist.\n"
            "Graph won't have edges."
        )

    labels = dask_imread(str(data_dir / "*.tif"), imread=tiff.imread)

    region_props_nodes = RegionPropsNodes(**region_props_kwargs)
    region_props_nodes.add_nodes(graph, labels=labels)

    nodes_df = graph.node_attrs(
        attr_keys=[
            DEFAULT_ATTR_KEYS.NODE_ID,
            DEFAULT_ATTR_KEYS.T,
            "label",
        ]
    )

    _add_edges_from_tracklet_ids(
        graph,
        nodes_df,
        tracklet_id_graph,
        "label",
    )

    # is duplicating an attribute that bad?
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.TRACKLET_ID, -1)
    graph.update_node_attrs(
        node_ids=nodes_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
        attrs={
            DEFAULT_ATTR_KEYS.TRACKLET_ID: nodes_df["label"].to_list(),
        },
    )


def to_ctc(
    graph: BaseGraph,
    shape: tuple[int, ...],
    output_dir: str | Path,
    tracklet_id_key: str = DEFAULT_ATTR_KEYS.TRACKLET_ID,
    overwrite: bool = False,
) -> None:
    """
    Save a graph to a CTC data directory.

    Parameters
    ----------
    graph : BaseGraph
        The graph to save.
    shape : tuple[int, ...]
        The shape of the label images (T, (Z), Y, X)
    output_dir : str | Path
        The directory to save the label images and the tracks graph to.
    tracklet_id_key : str
        The attribute key to use for the track IDs.
    overwrite : bool
        Whether to overwrite the output directory if it exists.

    See Also
    --------
    [BaseGraph.to_ctc][tracksdata.graph.BaseGraph.to_ctc]:
        Functional API to export into a CTC data directory.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        # and not empty
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory {output_dir} already exists.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    view = GraphArrayView(graph, shape=shape, attr_key=tracklet_id_key)

    n_digits = max(len(str(view.shape[0])), 3)

    tracks_table = compressed_tracks_table(graph)

    np.savetxt(output_dir / "res_track.txt", tracks_table, fmt="%d")

    def _write_tiff(t: int) -> None:
        LOG.info(f"Saving label image for time point {t}")
        tiff.imwrite(
            output_dir / f"mask{t:0{n_digits}d}.tif",
            view[t],
            compression="LZW",
        )

    list(
        multiprocessing_apply(
            _write_tiff,
            range(view.shape[0]),
            desc="Saving label images",
            sorted=False,
        )
    )
