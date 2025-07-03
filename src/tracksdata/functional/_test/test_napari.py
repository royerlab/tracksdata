import numpy as np

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional import to_napari_format
from tracksdata.graph import RustWorkXGraph


def test_napari_conversion() -> None:
    positions = np.asarray(
        [
            [0, 5, 10, 20],  # t=0, z=5, y=10, x=20
            [1, 6, 15, 25],  # t=1, z=6, y=15, x=25
            [1, 7, 20, 30],  # t=2, z=7, y=20, x=30
        ]
    )

    track_ids = np.asarray([1, 2, 3])
    track_id_graph = {3: 1, 2: 1}

    radius = np.asarray([1, 3, 5])

    image_shape = (10, 22, 32)

    graph = RustWorkXGraph.from_array(
        positions, track_ids=track_ids, track_id_graph=track_id_graph, radius=radius, image_shape=image_shape
    )
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.SOLUTION, True)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.SOLUTION, True)

    array_view, tracks_df, dict_graph = to_napari_format(graph, (2, *image_shape))

    assert dict_graph == track_id_graph

    assert tracks_df.shape == (3, 5)

    np.testing.assert_equal(
        tracks_df.to_numpy()[:, 1:],
        positions,
    )

    assert array_view.shape == (2, 10, 22, 32)

    np.testing.assert_equal(np.unique(array_view[0]), [0, 1])
    np.testing.assert_equal(np.unique(array_view[1]), [0, 2, 3])
