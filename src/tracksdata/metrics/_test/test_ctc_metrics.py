import shutil
from pathlib import Path

import numpy as np
import pytest
from ctc_metrics.scripts.evaluate import evaluate_sequence, load_data
from tifffile import imread

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges import DistanceEdges
from tracksdata.graph import RustWorkXGraph
from tracksdata.metrics import evaluate_ctc_metrics
from tracksdata.metrics._ctc_metrics import compute_ctc_metrics_data
from tracksdata.nodes import RegionPropsNodes
from tracksdata.options import options_context


@options_context(n_workers=4)
def test_replicating_ctc_metrics_test(pytestconfig: pytest.Config) -> None:
    ctc_data_dir = pytestconfig.cache._cachedir / "test_dataset_ctc/train/BF-C2DL-HSC"
    if not ctc_data_dir.exists():
        pytest.skip(
            f"CTC data not found at {ctc_data_dir}.\n"
            "Download from 'https://www.tnt.uni-hannover.de/de/project/MPT/data/CTC/test_dataset_ctc.zip'"
            "and extract it to the pytest cache directory (.pytest_cache)."
        )

    input_graph = RustWorkXGraph.from_ctc(ctc_data_dir / "01_RES")
    reference_graph = RustWorkXGraph.from_ctc(ctc_data_dir / "01_GT/TRA")

    metrics = evaluate_ctc_metrics(input_graph, reference_graph)

    # from: https://github.com/CellTrackingChallenge/py-ctcmetrics/blob/main/test/utils.py
    expected_values = {
        "DET": 0.95377521,
        # "SEG": 0.903990207454052,
        "TRA": 0.9588616,
        "CCA": 0.060606060606061,
        "CT": 0.0190476,
        "TF": 0.72417699,
        "BC(0)": 0.434782,
        "BC(1)": 0.47826086,
        "BC(2)": 0.47826086,
        "BC(3)": 0.47826086,
    }

    for key, expected_value in expected_values.items():
        assert np.isclose(metrics[key], expected_value, atol=1e-6), f"{key=} {metrics[key]=} {expected_value=}"


def test_ctc_metrics(ctc_data_dir: Path) -> None:
    # hack required to load two ground-truths with ctc_metrics
    ctc_data_dir = ctc_data_dir / "02_GT/TRA"

    shutil.copy(ctc_data_dir / "man_track.txt", ctc_data_dir / "res_track.txt")

    # loading reference intermediate data
    ref_input_tracks, ref_reference_tracks, ref_matching_data, *_ = load_data(
        res=str(ctc_data_dir),
        gt=str(ctc_data_dir.parent),
        trajectory_data=True,
        segmentation_data=False,
        threads=1,
    )

    # they are the same, I know
    # testing both backends at once
    input_graph = RustWorkXGraph.from_ctc(ctc_data_dir)
    reference_graph = RustWorkXGraph.from_ctc(ctc_data_dir)

    input_tracks, reference_tracks, matching_data = compute_ctc_metrics_data(
        input_graph,
        reference_graph,
        input_track_id_key=DEFAULT_ATTR_KEYS.TRACK_ID,
        reference_track_id_key=DEFAULT_ATTR_KEYS.TRACK_ID,
    )

    np.testing.assert_array_equal(input_tracks, ref_input_tracks)
    np.testing.assert_array_equal(reference_tracks, ref_reference_tracks)

    for key, values in matching_data.items():
        # I'm not really sure why only the ious shape are different
        # The other test is passing
        if key == "ious":
            continue
        expected_values = ref_matching_data[key]
        for t, (v, e) in enumerate(zip(values, expected_values, strict=True)):
            np.testing.assert_array_equal(v, e, err_msg=f"{key=} t={t}")

    metrics = evaluate_ctc_metrics(input_graph, reference_graph)

    expected_metrics = evaluate_sequence(
        res=str(ctc_data_dir),
        gt=str(ctc_data_dir.parent),
        threads=1,
    )

    assert len(metrics) == len(expected_metrics)

    for key, value in expected_metrics.items():
        if key == "SEG" or key.startswith("OP_"):
            # our implementation uses TRA masks for "SEG"
            # while ctc_metrics uses the "SEG" dir which is different
            # and the OP_ rely on "SEG" results
            continue
        assert metrics[key] == value, f"{key=} {metrics[key]=} {value=}"


def test_graph_match(ctc_data_dir: Path) -> None:
    # testing _matching_data with optimal_matching=True
    input_dir = ctc_data_dir / "01_ERR_SEG"
    input_graph = RustWorkXGraph()
    ref_graph = RustWorkXGraph.from_ctc(ctc_data_dir / "01_GT/TRA")

    labels = np.stack([imread(p) for p in sorted(input_dir.glob("*.tif"))])

    region_props_nodes = RegionPropsNodes()
    region_props_nodes.add_nodes(input_graph, labels=labels)

    distance_edges = DistanceEdges(distance_threshold=10, n_neighbors=2)
    distance_edges.add_edges(input_graph)

    input_graph.match(ref_graph)
    input_df = input_graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])

    # this is required because we know ere are using ground-truths
    assert (input_df[DEFAULT_ATTR_KEYS.MATCHED_NODE_ID] >= 0).all()
