from typing import TYPE_CHECKING, Optional

import numpy as np

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._edges import join_node_attrs_to_edges
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._logging import LOG

if TYPE_CHECKING:
    import napari


def _vector_kwargs_defaults(vector_kwargs: dict | None) -> dict:
    """
    Set the default values for the vector kwargs.
    """
    if vector_kwargs is None:
        vector_kwargs = {}
    else:
        vector_kwargs = vector_kwargs.copy()

    default_kwargs = {
        "edge_width": 3,
        "vector_style": "arrow",
    }
    for key, value in default_kwargs.items():
        if key not in vector_kwargs:
            vector_kwargs[key] = value

    return vector_kwargs


def _points_kwargs_defaults(points_kwargs: dict | None) -> dict:
    """
    Set the default values for the points kwargs.
    """
    if points_kwargs is None:
        points_kwargs = {}
    else:
        points_kwargs = points_kwargs.copy()

    color_cycle = ["yellow", "blue"]

    default_kwargs = {
        "face_color": "transparent",
        "border_width": 0.15,
        "border_color": "yellow",
        "border_color_cycle": color_cycle,
        "size": 20,
    }
    for key, value in default_kwargs.items():
        if key not in points_kwargs:
            points_kwargs[key] = value

    return points_kwargs


def _add_matching_text_and_color(points_kwargs: dict | None) -> dict:
    """
    Add the matching text and color to the points kwargs.
    """
    if points_kwargs is None:
        points_kwargs = {}
    else:
        points_kwargs = points_kwargs.copy()

    color_cycle = ["red", "green"]

    text_params = {
        "string": "Score:{match_score:.3f} Match:{matched_node_id}",
        "size": 20,
        "color": {"feature": "matched", "colormap": color_cycle},
        "anchor": "upper_left",
    }

    default_kwargs = {
        "border_color": "matched",
        "border_color_cycle": color_cycle,
        "text": text_params,
    }
    for key, value in default_kwargs.items():
        if key not in points_kwargs:
            points_kwargs[key] = value

    points_kwargs = _points_kwargs_defaults(points_kwargs)

    return points_kwargs


def visualize_matches(
    input_graph: BaseGraph,
    ref_graph: BaseGraph,
    matched_node_id_key: str = DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
    match_score_key: str = DEFAULT_ATTR_KEYS.MATCH_SCORE,
    matched_edge_mask_key: str = DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK,
    viewer: Optional["napari.Viewer"] = None,
    points_kwargs: dict | None = None,
    vector_kwargs: dict | None = None,
) -> None:
    """
    Visualize the matches between the graph and the gt_graph.

    Parameters
    ----------
    input_graph : BaseGraph
        The predicted graph to visualize.
    ref_graph : BaseGraph
        The reference (ground truth) graph to visualize.
    matched_node_id_key : str, optional
        The key of the attribute that contains the matched node IDs.
        If not provided, the default key will be used.
    match_score_key : str, optional
        The key of the attribute that contains the match scores.
        If not provided, the default key will be used.
    matched_edge_mask_key : str, optional
        The key of the attribute that contains the matched edge mask.
        If not provided, the default key will be used.
    viewer : napari.Viewer, optional
        The napari viewer to use. If not provided, a new viewer will be created.
    points_kwargs : dict, optional
        Additional keyword arguments to the napari.Viewer.add_points method.
    vector_kwargs : dict, optional
        Additional keyword arguments to the napari.Viewer.add_vectors method.
    """
    try:
        import napari
    except ImportError as e:
        raise ImportError(
            "napari must be installed to use this function.\nPlease install it with `pip install napari`.",
        ) from e

    if viewer is None:
        viewer = napari.Viewer()

    if "z" in input_graph.node_attr_keys:
        pos = ["t", "z", "y", "x"]
    else:
        pos = ["t", "y", "x"]

    node_attrs = input_graph.node_attrs()
    ref_node_attrs = ref_graph.node_attrs()

    matched_points_kwargs = _add_matching_text_and_color(points_kwargs)
    points_kwargs = _points_kwargs_defaults(points_kwargs)

    layer = viewer.add_points(
        node_attrs[pos],
        name="predicted",
        properties={
            "matched_node_id": node_attrs[matched_node_id_key],
            "match_score": node_attrs[match_score_key],
            "matched": node_attrs[matched_node_id_key] >= 0,
        },
        **matched_points_kwargs,
    )
    layer.text.visible = False

    matched_ref_mask = ref_node_attrs[DEFAULT_ATTR_KEYS.NODE_ID].is_in(node_attrs[matched_node_id_key].implode())
    # matched_ref_node_attrs = ref_node_attrs.filter(matched_ref_mask)
    unmatched_ref_node_attrs = ref_node_attrs.filter(~matched_ref_mask)

    viewer.add_points(
        unmatched_ref_node_attrs[pos],
        name="missing objects",
        **points_kwargs,
    )

    edge_attrs = input_graph.edge_attrs(attr_keys=[matched_edge_mask_key])
    source_pos = [f"source_{p}" for p in pos]
    target_pos = [f"target_{p}" for p in pos]
    source_matched_key = f"source_{matched_node_id_key}"
    target_matched_key = f"target_{matched_node_id_key}"

    edge_attrs = join_node_attrs_to_edges(
        node_attrs.select(DEFAULT_ATTR_KEYS.NODE_ID, matched_node_id_key, *pos),
        edge_attrs,
    )

    # target positions is just the difference between source and target
    # not the actual target position
    edge_attrs[target_pos] = edge_attrs[target_pos] - edge_attrs[source_pos]

    mask = edge_attrs[matched_edge_mask_key].to_numpy().astype(bool)

    edge_arr = np.stack(
        [
            edge_attrs.select(*source_pos),
            edge_attrs.select(*target_pos),
        ],
        axis=1,
    )

    matched_edge_arr = edge_arr[mask]
    unmatched_edge_arr = edge_arr[~mask]

    vector_kwargs = _vector_kwargs_defaults(vector_kwargs)

    if matched_edge_arr.shape[0] > 0:
        viewer.add_vectors(
            matched_edge_arr,
            name="matched edges",
            edge_color="green",
            opacity=1,
            **vector_kwargs,
        )
    else:
        LOG.warning("No edges matched to the reference graph")

    if unmatched_edge_arr.shape[0] > 0:
        viewer.add_vectors(
            unmatched_edge_arr,
            name="unmatched input edges",
            edge_color="blue",
            opacity=0.2,
            **vector_kwargs,
        )
    else:
        LOG.warning("All edges matched to the reference graph")

    ref_edge_attrs = ref_graph.edge_attrs(attr_keys=[])

    ref_edge_attrs = ref_edge_attrs.join(
        edge_attrs.select(
            matched_edge_mask_key,
            source_matched_key,
            target_matched_key,
        ),
        left_on=[DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET],
        right_on=[source_matched_key, target_matched_key],
        how="left",
    )

    unmatched_ref_edge_attrs = ref_edge_attrs.filter(ref_edge_attrs[matched_edge_mask_key].is_null())
    unmatched_ref_edge_attrs = join_node_attrs_to_edges(
        ref_node_attrs.select(DEFAULT_ATTR_KEYS.NODE_ID, *pos),
        unmatched_ref_edge_attrs,
    )

    # same as above, target positions is just the difference between source and target
    unmatched_ref_edge_attrs[target_pos] = unmatched_ref_edge_attrs[target_pos] - unmatched_ref_edge_attrs[source_pos]

    unmatched_ref_edge_arr = np.stack(
        [
            unmatched_ref_edge_attrs.select(*source_pos),
            unmatched_ref_edge_attrs.select(*target_pos),
        ],
        axis=1,
    )

    viewer.add_vectors(
        unmatched_ref_edge_arr,
        name="unmatched reference edges",
        edge_color="yellow",
        opacity=1,
        **vector_kwargs,
    )
