from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.io._ctc import _load_tracks_file
from tracksdata.nodes._mask import Mask


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("7 2 5 0\n", {}),
        ("9 6 8 7\n", {9: 7}),
        ("7 2 5 0\n9 6 8 7\n10 6 9 7\n", {9: 7, 10: 7}),
    ],
)
def test_load_tracks_file(tmp_path: Path, contents: str, expected: dict[int, int]) -> None:
    tracks_file = tmp_path / "man_track.txt"
    tracks_file.write_text(contents)

    assert _load_tracks_file(tracks_file) == expected


@pytest.mark.parametrize("metadata_shape", [True, False])
def test_export_from_ctc_roundtrip(tmp_path: Path, metadata_shape: bool) -> None:
    """Test that exporting and loading CTC format preserves graph structure."""
    # Create original graph with nodes and edges
    in_graph = RustWorkXGraph()

    in_graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    in_graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))
    in_graph.add_node_attr_key(DEFAULT_ATTR_KEYS.TRACKLET_ID, pl.Int64, -1)
    in_graph.add_node_attr_key("x", pl.Float64, -999_999)
    in_graph.add_node_attr_key("y", pl.Float64, -999_999)

    node_1 = in_graph.add_node(
        attrs={
            DEFAULT_ATTR_KEYS.T: 0,
            DEFAULT_ATTR_KEYS.TRACKLET_ID: 1,
            "x": 0,
            "y": 0,
            DEFAULT_ATTR_KEYS.MASK: Mask(
                mask=np.ones((2, 2), dtype=bool),
                bbox=np.asarray([0, 0, 2, 2]),
            ),
            DEFAULT_ATTR_KEYS.BBOX: np.array([0, 0, 2, 2]),
        },
    )

    node_2 = in_graph.add_node(
        attrs={
            DEFAULT_ATTR_KEYS.T: 1,
            DEFAULT_ATTR_KEYS.TRACKLET_ID: 2,
            "x": 1,
            "y": 1,
            DEFAULT_ATTR_KEYS.MASK: Mask(
                mask=np.ones((2, 2), dtype=bool),
                bbox=np.asarray([0, 0, 2, 2]),
            ),
            DEFAULT_ATTR_KEYS.BBOX: np.array([0, 0, 2, 2]),
        },
    )

    node_3 = in_graph.add_node(
        attrs={
            DEFAULT_ATTR_KEYS.T: 1,
            DEFAULT_ATTR_KEYS.TRACKLET_ID: 3,
            "x": 2,
            "y": 2,
            DEFAULT_ATTR_KEYS.MASK: Mask(
                mask=np.ones((2, 2), dtype=bool),
                bbox=np.asarray([1, 1, 3, 3]),
            ),
            DEFAULT_ATTR_KEYS.BBOX: np.array([1, 1, 3, 3]),
        },
    )

    in_graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, pl.Float64, 0.0)
    in_graph.add_edge(node_1, node_2, attrs={DEFAULT_ATTR_KEYS.EDGE_DIST: 1.0})
    in_graph.add_edge(node_1, node_3, attrs={DEFAULT_ATTR_KEYS.EDGE_DIST: 1.0})

    if metadata_shape:
        in_graph.metadata.update(shape=(2, 4, 4))
        shape = None
    else:
        shape = (2, 4, 4)

    in_graph.to_ctc(output_dir=tmp_path, shape=shape)

    out_graph = RustWorkXGraph.from_ctc(tmp_path)

    assert out_graph.num_nodes() == in_graph.num_nodes()
    assert out_graph.num_edges() == in_graph.num_edges()

    in_attrs = in_graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.TRACKLET_ID])
    out_attrs = out_graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.TRACKLET_ID])

    assert in_attrs.equals(out_attrs)
