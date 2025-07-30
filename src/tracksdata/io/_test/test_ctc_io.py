from pathlib import Path

import numpy as np

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.utils._test_utils import (
    setup_custom_node_attr,
    setup_edge_distance_attr,
    setup_mask_attrs,
)


def test_export_from_ctc_roundtrip(tmp_path: Path):
    """Test that exporting and loading CTC format preserves graph structure."""
    # Create original graph with nodes and edges
    in_graph = RustWorkXGraph()

    setup_mask_attrs(in_graph)
    setup_custom_node_attr(in_graph, DEFAULT_ATTR_KEYS.TRACK_ID, -1)
    setup_custom_node_attr(in_graph, "x", -999_999)
    setup_custom_node_attr(in_graph, "y", -999_999)

    node_1 = in_graph.add_node(
        attrs={
            DEFAULT_ATTR_KEYS.T: 0,
            DEFAULT_ATTR_KEYS.TRACK_ID: 1,
            "x": 0,
            "y": 0,
            DEFAULT_ATTR_KEYS.MASK: np.ones((2, 2), dtype=bool),
            DEFAULT_ATTR_KEYS.BBOX: np.asarray([0, 0, 2, 2]),
        },
    )

    node_2 = in_graph.add_node(
        attrs={
            DEFAULT_ATTR_KEYS.T: 1,
            DEFAULT_ATTR_KEYS.TRACK_ID: 2,
            "x": 1,
            "y": 1,
            DEFAULT_ATTR_KEYS.MASK: np.ones((2, 2), dtype=bool),
            DEFAULT_ATTR_KEYS.BBOX: np.asarray([0, 0, 2, 2]),
        },
    )

    node_3 = in_graph.add_node(
        attrs={
            DEFAULT_ATTR_KEYS.T: 1,
            DEFAULT_ATTR_KEYS.TRACK_ID: 3,
            "x": 2,
            "y": 2,
            DEFAULT_ATTR_KEYS.MASK: np.ones((2, 2), dtype=bool),
            DEFAULT_ATTR_KEYS.BBOX: np.asarray([1, 1, 3, 3]),
        },
    )

    setup_edge_distance_attr(in_graph)
    in_graph.add_edge(node_1, node_2, attrs={DEFAULT_ATTR_KEYS.EDGE_DIST: 1.0})
    in_graph.add_edge(node_1, node_3, attrs={DEFAULT_ATTR_KEYS.EDGE_DIST: 1.0})

    in_graph.to_ctc(shape=(2, 4, 4), output_dir=tmp_path)

    out_graph = RustWorkXGraph.from_ctc(tmp_path)

    assert out_graph.num_nodes == in_graph.num_nodes
    assert out_graph.num_edges == in_graph.num_edges

    in_attrs = in_graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.TRACK_ID])
    out_attrs = out_graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.T, DEFAULT_ATTR_KEYS.TRACK_ID])

    assert in_attrs.equals(out_attrs)
