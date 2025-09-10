from typing import TYPE_CHECKING, overload

import polars as pl
import rustworkx as rx

from tracksdata.attrs import EdgeAttr, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph

if TYPE_CHECKING:
    from tracksdata.array._graph_array import GraphArrayView


@overload
def to_napari_format(
    graph: BaseGraph,
    shape: tuple[int, ...],
    solution_key: str | None,
    output_track_id_key: str,
    mask_key: None,
) -> tuple[pl.DataFrame, dict[int, int]]: ...


@overload
def to_napari_format(
    graph: BaseGraph,
    shape: tuple[int, ...],
    solution_key: str | None,
    output_track_id_key: str,
    mask_key: str,
) -> tuple[pl.DataFrame, dict[int, int], "GraphArrayView"]: ...


def to_napari_format(
    graph: BaseGraph,
    shape: tuple[int, ...],
    solution_key: str | None = DEFAULT_ATTR_KEYS.SOLUTION,
    output_track_id_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
    mask_key: str | None = None,
    chunk_shape: tuple[int] | None = None,
    buffer_cache_size: int | None = None,
) -> (
    tuple[
        pl.DataFrame,
        dict[int, int],
        "GraphArrayView",
    ]
    | tuple[
        pl.DataFrame,
        dict[int, int],
    ]
):
    """
    Convert the subgraph of solution nodes to a napari-ready format.

    This includes:
    - a tracks layer with the solution tracks
    - a graph with the parent-child relationships for the solution tracks
    - a labels layer with the solution nodes if `mask_key` is provided.

    IMPORTANT: This function will reset the track ids if they already exist.

    Parameters
    ----------
    graph : BaseGraph
        The graph to convert.
    shape : tuple[int, ...]
        The shape of the labels layer.
    solution_key : str, optional
        The key of the solution attribute. If None, the graph is not filtered by the solution attribute.
    output_track_id_key : str, optional
        The key of the output track id attribute.
    mask_key : str | None, optional
        The key of the mask attribute.
    chunk_shape : tuple[int] | None, optional
        The chunk shape for the labels layer. If None, the default chunk size is used.
    buffer_cache_size : int, optional
        The maximum number of buffers to keep in the cache for the labels layer.
        If None, the default buffer cache size is used.

    Examples
    --------

    ```python
    labels = ...
    graph = ...
    tracks_data, dict_graph, array_view = to_napari_format(graph, labels.shape, mask_key="mask")
    ```

    Returns
    -------
    tuple[pl.DataFrame, dict[int, int], GraphArrayView] | tuple[pl.DataFrame, dict[int, int]]
        - tracks_data: The tracks data as a polars DataFrame.
        - dict_graph: A dictionary of parent -> child relationships.
        - array_view: The array view of the solution graph if `mask_key` is provided.
    """
    if solution_key is not None:
        solution_graph = graph.filter(
            NodeAttr(solution_key) == True,
            EdgeAttr(solution_key) == True,
        ).subgraph()

    else:
        solution_graph = graph

    tracks_graph = solution_graph.assign_track_ids(output_track_id_key)
    dict_graph = {tracks_graph[child]: tracks_graph[parent] for parent, child in tracks_graph.edge_list()}

    spatial_cols = ["z", "y", "x"][-len(shape) + 1 :]

    tracks_data = solution_graph.node_attrs(
        attr_keys=[output_track_id_key, DEFAULT_ATTR_KEYS.T, *spatial_cols],
    )

    # sorting columns
    tracks_data = tracks_data.select([output_track_id_key, DEFAULT_ATTR_KEYS.T, *spatial_cols])

    if mask_key is not None:
        from tracksdata.array._graph_array import GraphArrayView

        array_view = GraphArrayView(
            solution_graph,
            shape,
            attr_key=output_track_id_key,
            chunk_shape=chunk_shape,
            buffer_cache_size=buffer_cache_size,
        )

        return tracks_data, dict_graph, array_view

    return tracks_data, dict_graph


def rx_digraph_to_napari_dict(
    tracklet_graph: rx.PyDiGraph,
) -> dict[int, list[int]]:
    """
    Convert a tracklet graph to a napari-ready dictionary.
    The input is a (child -> parent) graph (forward in time) and it is converted
    to a (parent -> child) dictionary (backward in time).

    Parameters
    ----------
    tracklet_graph : rx.PyDiGraph
        The tracklet graph to convert.

    Returns
    -------
    dict[int, list[int]]
        A dictionary of parent -> child relationships.
    """
    dict_graph = {}
    for parent, child in tracklet_graph.edges():
        dict_graph.setdefault(child, []).append(parent)
    return dict_graph
