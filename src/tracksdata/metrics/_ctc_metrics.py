from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import scipy.sparse as sp
from toolz import curry

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.io._ctc import compressed_tracks_table
from tracksdata.options import get_options
from tracksdata.utils._dtypes import column_from_bytes, column_to_bytes
from tracksdata.utils._logging import LOG
from tracksdata.utils._multiprocessing import multiprocessing_apply

if TYPE_CHECKING:
    from tracksdata.graph import RustWorkXGraph
    from tracksdata.graph._base_graph import BaseGraph


def _fill_empty(weights: sp.csr_array, eps: float = 1e-10) -> None:
    """
    Fill empty rows and columns of a sparse matrix with a small value.
    """
    empty_rows = weights.sum(axis=1) == 0
    if empty_rows.any():
        weights[empty_rows, :] = eps

    empty_cols = weights.sum(axis=0) == 0
    if empty_cols.any():
        weights[:, empty_cols] = eps


def _match_single_frame(
    t: int,
    *,
    groups_by_time: dict[str, dict[int, pl.DataFrame]],
    reference_graph_key: str,
    input_graph_key: str,
    optimal_matching: bool = False,
    min_reference_intersection: float = 0.5,
) -> tuple[list[int], list[int], list[float]]:
    """
    Match the groups of the reference and input graphs for a single time point.

    Parameters
    ----------
    t : int
        The time point to match.
    groups_by_time : dict[str, dict[int, pl.DataFrame]]
        The groups of the reference and input graphs by time point.
    reference_graph_key : str
        The key to obtain the track id from the reference graph.
    input_graph_key : str
        The key to obtain the track id from the input graph.
    optimal_matching : bool, optional
        Whether to solve the optimal matching.
    min_reference_intersection : float, optional
        The minimum intersection between the reference and input masks to be considered as a match.

    Returns
    -------
    tuple[list[int], list[int], list[float]]
        For each matching, their respective reference id, input id and IoU.
    """
    try:
        ref_group = groups_by_time["ref"][t]
        comp_group = groups_by_time["comp"][t]

    except KeyError:
        return [], [], []

    if ref_group[DEFAULT_ATTR_KEYS.MASK].dtype == pl.Binary:
        ref_group = column_from_bytes(ref_group, DEFAULT_ATTR_KEYS.MASK)
        comp_group = column_from_bytes(comp_group, DEFAULT_ATTR_KEYS.MASK)

    _mapped_ref = []
    _mapped_comp = []
    _ious = []
    _rows = []
    _cols = []

    for i, (ref_id, ref_mask) in enumerate(
        zip(ref_group[reference_graph_key], ref_group[DEFAULT_ATTR_KEYS.MASK], strict=True)
    ):
        for j, (comp_id, comp_mask) in enumerate(
            zip(comp_group[input_graph_key], comp_group[DEFAULT_ATTR_KEYS.MASK], strict=True)
        ):
            # intersection over reference is used to select the matches
            inter = ref_mask.intersection(comp_mask)
            ctc_score = inter / ref_mask.size
            if ctc_score > min_reference_intersection:
                _mapped_ref.append(ref_id)
                _mapped_comp.append(comp_id)
                _rows.append(i)
                _cols.append(j)

                # NOTE: there was something weird with IoU, the length when compared with `ctc_metrics`
                #       sometimes it had an extra element
                iou = inter / (ref_mask.size + comp_mask.size - inter)
                _ious.append(iou.item())

    if optimal_matching and len(_rows) > 0:
        LOG.info("Solving optimal matching ...")

        weights = sp.csr_array((_ious, (_rows, _cols)), dtype=np.float32)

        try:
            rows_id, cols_id = sp.csgraph.min_weight_full_bipartite_matching(weights, maximize=True)
        except ValueError:
            _fill_empty(weights)
            rows_id, cols_id = sp.csgraph.min_weight_full_bipartite_matching(weights, maximize=True)

        # loading original group ids and filtering by the matches
        _mapped_ref = ref_group[reference_graph_key][rows_id].to_list()
        _mapped_comp = comp_group[input_graph_key][cols_id].to_list()
        _ious = weights[rows_id, cols_id].tolist()

        LOG.info("Done!")

    return _mapped_ref, _mapped_comp, _ious


def _matching_data(
    input_graph: "BaseGraph",
    reference_graph: "BaseGraph",
    input_graph_key: str,
    reference_graph_key: str,
    optimal_matching: bool = False,
    min_reference_intersection: float = 0.5,
) -> dict[str, list[list]]:
    """
    Compute matching data for CTC metrics.

    This function includes an optional functionality of solving the optimal matching
    to avoid duplicated matches from `input_graph` to `reference_graph`.

    Parameters
    ----------
    input_graph : BaseGraph
        Input graph.
    reference_graph : BaseGraph
        Reference graph.
    input_graph_key : str
        Key to obtain the track id from the input graph.
    reference_graph_key : str
        Key to obtain the track id from the reference graph.
    optimal_matching : bool, optional
        Whether to solve the optimal matching.
    min_reference_intersection : float, optional
        Minimum coverage of a reference mask to be considered as a possible match.

    Returns
    -------
    dict[str, list[list]]
        Dictionary with the matching data, see `compute_ctc_metrics_data` for more information.
    """
    result = {}

    groups_by_time = {
        "ref": {},
        "comp": {},
    }

    n_workers = get_options().n_workers

    # computing unique labels for each graph
    for name, graph, track_id_key in [
        ("ref", reference_graph, reference_graph_key),
        ("comp", input_graph, input_graph_key),
    ]:
        nodes_df = graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.T, track_id_key, DEFAULT_ATTR_KEYS.MASK])
        if n_workers > 1:
            # required by multiprocessing
            nodes_df = column_to_bytes(nodes_df, DEFAULT_ATTR_KEYS.MASK)
        labels = {}

        for (t,), group in nodes_df.group_by(DEFAULT_ATTR_KEYS.T):
            labels[t] = group[track_id_key].sort().to_list()
            # storing masks of each frame
            groups_by_time[name][t] = group

        result[f"labels_{name}"] = labels

    n_time_points = (
        max(
            max(groups_by_time["ref"].keys(), default=-1),
            max(groups_by_time["comp"].keys(), default=-1),
        )
        + 1
    )
    # NOTE: should we also use a min value?

    # converting `labels_{...}` to a list of lists
    # we need n_time_points to create the empty lists
    for k, labels_dict in list(result.items()):
        result[k] = [labels_dict.get(t, []) for t in range(n_time_points)]

    mapped_ref = []
    mapped_comp = []
    ious = []

    match_func = curry(
        _match_single_frame,
        groups_by_time=groups_by_time,
        reference_graph_key=reference_graph_key,
        input_graph_key=input_graph_key,
        optimal_matching=optimal_matching,
        min_reference_intersection=min_reference_intersection,
    )

    for _mapped_ref, _mapped_comp, _ious in multiprocessing_apply(
        func=match_func,
        sequence=range(n_time_points),
        desc="Matching nodes between graphs",
        sorted=True,
    ):
        mapped_ref.append(_mapped_ref)
        mapped_comp.append(_mapped_comp)
        ious.append(_ious)

    result["mapped_ref"] = mapped_ref
    result["mapped_comp"] = mapped_comp
    result["ious"] = ious

    return result


def compute_ctc_metrics_data(
    input_graph: "BaseGraph",
    reference_graph: "BaseGraph",
    input_track_id_key: str,
    reference_track_id_key: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[list]]]:
    """
    Compute intermediate data required for CTC metrics.

    Reference: [ctc_metrics.scripts.evaluate.calculate_metrics](https://github.com/CellTrackingChallenge/py-ctcmetrics/blob/main/ctc_metrics/scripts/evaluate.py)

    Parameters
    ----------
    input_graph : RustWorkXGraph
        Input graph.
    reference_graph : RustWorkXGraph
        Reference graph.
    input_track_id_key : str
        Key to obtain the track id from the input graph.
    reference_track_id_key : str
        Key to obtain the track id from the reference graph.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, list], dict[str, list], list[str]]
        - input_tracks:
            (n, 4) array with reprensenting the (track_id, start_frame, end_frame, parent_track_id) of each track
        - reference_tracks:
            (n, 4) array with reprensenting the (track_id, start_frame, end_frame, parent_track_id) of each track
        - matching_data: Frame-wise matching data, defined as a dict with the following keys:
            - labels_ref: A list of lists containing the labels of the reference masks.
            - labels_comp: A list of lists containing the labels of the computed masks.
            - mapped_ref: A list of lists containing the mapped labels of the reference masks.
            - mapped_comp: A list of lists containing the mapped labels of the computed masks.
            - ious: A list of lists containing the intersection over union values
                    between mapped reference and computed masks.
    """
    input_tracks = compressed_tracks_table(input_graph)
    reference_tracks = compressed_tracks_table(reference_graph)

    matching_data = _matching_data(input_graph, reference_graph, input_track_id_key, reference_track_id_key)

    return input_tracks, reference_tracks, matching_data


def evaluate_ctc_metrics(
    input_graph: "RustWorkXGraph",
    reference_graph: "RustWorkXGraph",
    input_track_id_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
    reference_track_id_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
    input_reset: bool = True,
    reference_reset: bool = False,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """
    Evaluate CTC metrics using `py-ctcmetrics` developed by [Timo Kaiser](https://github.com/TimoK93).

    If you use this function, please cite the respective papers of each metric, as described in
    [here](https://github.com/CellTrackingChallenge/py-ctcmetrics?tab=readme-ov-file#acknowledgement-and-citations).

    IMPORTANT: The `SEG` metric is computed from the TRA masks.

    Parameters
    ----------
    input_graph : RustWorkXGraph
        Input graph.
    reference_graph : RustWorkXGraph
        Reference graph.
    input_track_id_key : str, optional
        Key to obtain the track id from the input graph.
        If key does not exist, it will be created.
    reference_track_id_key : str, optional
        Key to obtain the track id from the reference graph.
        If key does not exist, it will be created.
    input_reset : bool, optional
        Whether to reset the track ids of the input graph. If True, the track ids will be reset to -1.
    reference_reset : bool, optional
        Whether to reset the track ids of the reference graph. If True, the track ids will be reset to -1.
    metrics : list[str] | None, optional
        List of metrics to evaluate. If None, all metrics are evaluated.
        Available metrics:
        "CHOTA", "BC", "CT", "CCA", "TF", "TRA", "DET", "MOTA", "HOTA",
        "IDF1", "MTML", "FAF", "LNK", "OP_CTB", "OP_CSB", "BIO", "OP_CLB"

    Returns
    -------
    dict[str, float]
        Dictionary with the results of the evaluated metrics.
    """
    try:
        from ctc_metrics.metrics import ALL_METRICS
        from ctc_metrics.scripts.evaluate import calculate_metrics
    except ImportError as e:
        raise ImportError(
            "`py-ctcmetrics` is required to evaluate CTC metrics.\nPlease install it with `pip install py-ctcmetrics`."
        ) from e

    if metrics is None:
        metrics: list[str] = ALL_METRICS.copy()

    if "SEG" in metrics:
        LOG.warning("IMPORTANT! 'SEG' metric results are based on TRA masks, not the SEG masks.")

    if input_graph.num_nodes == 0:
        LOG.warning("Input graph has no nodes, returning -1.0 for all metrics.")
        return dict.fromkeys(metrics, -1.0)

    if input_track_id_key not in input_graph.node_attr_keys:
        input_graph.assign_track_ids(input_track_id_key, reset=input_reset)

    if reference_track_id_key not in reference_graph.node_attr_keys:
        reference_graph.assign_track_ids(reference_track_id_key, reset=reference_reset)

    input_tracks, reference_tracks, matching_data = compute_ctc_metrics_data(
        input_graph, reference_graph, input_track_id_key, reference_track_id_key
    )

    results = calculate_metrics(
        comp_tracks=input_tracks,
        ref_tracks=reference_tracks,
        traj=matching_data,
        segm=matching_data,
        metrics=metrics,
        is_valid=True,
    )

    results = {k: v.item() if hasattr(v, "item") else v for k, v in results.items()}

    return results
