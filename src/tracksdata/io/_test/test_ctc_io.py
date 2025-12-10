from pathlib import Path

import numpy as np
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.nodes._mask import Mask


@pytest.mark.parametrize("metadata_shape", [True, False])
def test_export_from_ctc_roundtrip(tmp_path: Path, metadata_shape: bool) -> None:
    """Test that exporting and loading CTC format preserves graph structure."""
    # Create original graph with nodes and edges
    in_graph = RustWorkXGraph()

    in_graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    in_graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, None)
    in_graph.add_node_attr_key(DEFAULT_ATTR_KEYS.TRACKLET_ID, -1)
    in_graph.add_node_attr_key("x", -999_999)
    in_graph.add_node_attr_key("y", -999_999)

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

    in_graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)
    in_graph.add_edge(node_1, node_2, attrs={DEFAULT_ATTR_KEYS.EDGE_DIST: 1.0})
    in_graph.add_edge(node_1, node_3, attrs={DEFAULT_ATTR_KEYS.EDGE_DIST: 1.0})

    if metadata_shape:
        in_graph.update_metadata(shape=(2, 4, 4))
        shape = None
    else:
        shape = (2, 4, 4)

    in_graph.to_ctc(output_dir=tmp_path, shape=shape)

    out_graph = RustWorkXGraph.from_ctc(tmp_path)

    assert out_graph.num_nodes == in_graph.num_nodes
    assert out_graph.num_edges == in_graph.num_edges

    in_attrs = in_graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.TRACKLET_ID])
    out_attrs = out_graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.TRACKLET_ID])

    assert in_attrs.equals(out_attrs)
