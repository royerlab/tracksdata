import numpy as np
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional import to_napari_format
from tracksdata.graph import RustWorkXGraph
from tracksdata.nodes import MaskDiskAttrs


@pytest.mark.parametrize("metadata_shape", [True, False])
def test_napari_conversion(metadata_shape: bool) -> None:
    positions = np.asarray(
        [
            [0, 5, 10, 20],  # t=0, z=5, y=10, x=20
            [1, 6, 15, 25],  # t=1, z=6, y=15, x=25
            [1, 7, 20, 30],  # t=2, z=7, y=20, x=30
        ]
    )

    tracklet_ids = np.asarray([1, 2, 3])
    tracklet_id_graph = {3: 1, 2: 1}

    graph = RustWorkXGraph.from_array(
        positions,
        tracklet_ids=tracklet_ids,
        tracklet_id_graph=tracklet_id_graph,
    )
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.SOLUTION, True)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.SOLUTION, True)

    shape = (2, 10, 22, 32)
    if metadata_shape:
        graph.update_metadata(shape=shape)
        arg_shape = None
    else:
        arg_shape = shape

    mask_attrs = MaskDiskAttrs(
        radius=2,
        image_shape=shape[1:],
        output_key=DEFAULT_ATTR_KEYS.MASK,
    )
    mask_attrs.add_node_attrs(graph)

    # Maybe we should update the MaskDiskAttrs to handle bounding boxes
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, None)
    masks = graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.MASK])[DEFAULT_ATTR_KEYS.MASK]
    graph.update_node_attrs(
        attrs={DEFAULT_ATTR_KEYS.BBOX: [mask.bbox for mask in masks]},
        node_ids=graph.node_ids(),
    )

    tracks_df, dict_graph, array_view = to_napari_format(
        graph,
        shape=arg_shape,
        mask_key=DEFAULT_ATTR_KEYS.MASK,
    )

    assert dict_graph == tracklet_id_graph

    assert tracks_df.shape == (3, 5)

    np.testing.assert_equal(
        tracks_df.to_numpy()[:, 1:],
        positions,
    )

    assert array_view.shape == (2, 10, 22, 32)

    np.testing.assert_equal(np.unique(array_view[0]), [0, 1])
    np.testing.assert_equal(np.unique(array_view[1]), [0, 2, 3])
