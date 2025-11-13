from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.metrics._visualize import visualize_matches
from tracksdata.nodes import MaskDiskAttrs

if TYPE_CHECKING:
    import napari


def test_visualize_matches(make_napari_viewer: Callable[[], "napari.Viewer"]) -> None:
    """Test the visualize_matches function with two small graphs."""
    napari = pytest.importorskip("napari")

    # Create input graph with nodes at different time points
    input_positions = np.array(
        [
            [0, 10, 20],  # t=0, y=10, x=20
            [1, 15, 25],  # t=1, y=15, x=25
            [1, 40, 50],  # t=1, y=40, x=50
        ]
    )

    input_tracklet_ids = np.array([1, 2, 3])
    input_tracklet_id_graph = {2: 1, 3: 1}  # track 2 connects to track 1, track 3 connects to track 1

    input_graph = RustWorkXGraph.from_array(
        input_positions,
        tracklet_ids=input_tracklet_ids,
        tracklet_id_graph=input_tracklet_id_graph,
    )

    # Create reference graph with nodes at different time points
    ref_positions = np.array(
        [
            [0, 12, 18],  # t=0, y=12, x=18 (close to first input node)
            [1, 17, 27],  # t=1, y=17, x=27 (close to second input node)
            [1, 60, 70],  # t=1, y=60, x=70 (no close match in input)
        ]
    )

    ref_tracklet_ids = np.array([1, 1, 2])
    ref_tracklet_id_graph = {1: 1}  # track 1 connects to track 1

    ref_graph = RustWorkXGraph.from_array(
        ref_positions,
        tracklet_ids=ref_tracklet_ids,
        tracklet_id_graph=ref_tracklet_id_graph,
    )

    # Add masks to both graphs
    image_shape = (80, 80)

    # Add masks to input graph
    input_mask_attrs = MaskDiskAttrs(
        radius=5,
        image_shape=image_shape,
        output_key=DEFAULT_ATTR_KEYS.MASK,
    )
    input_mask_attrs.add_node_attrs(input_graph)

    # Add masks to reference graph
    ref_mask_attrs = MaskDiskAttrs(
        radius=3,
        image_shape=image_shape,
        output_key=DEFAULT_ATTR_KEYS.MASK,
    )
    ref_mask_attrs.add_node_attrs(ref_graph)

    # Match the graphs to add matching attributes
    input_graph.match(ref_graph)

    viewer = make_napari_viewer()

    # Test the visualization function
    visualize_matches(
        input_graph=input_graph,
        ref_graph=ref_graph,
        viewer=viewer,
    )

    # Verify that layers were added to the viewer
    assert len(viewer.layers) > 0

    # Check that expected layer names are present
    layer_names = [layer.name for layer in viewer.layers]
    assert "predicted" in layer_names
    assert "missing objects" in layer_names

    # Check that vector layers for edges were added
    # (the exact names depend on whether matches were found)
    vector_layers = [layer for layer in viewer.layers if isinstance(layer, napari.layers.Vectors)]
    assert len(vector_layers) > 0
