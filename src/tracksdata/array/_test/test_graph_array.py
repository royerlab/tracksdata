from collections.abc import Sequence

import numpy as np
import pytest
from pytest import fixture

from tracksdata.array import GraphArrayView
from tracksdata.array._graph_array import chain_indices
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.nodes import RegionPropsNodes
from tracksdata.nodes._mask import Mask
from tracksdata.options import Options, get_options

# NOTE: this could be generic test for all array backends
# when more slicing operations are implemented we could test as in:
#  - https://github.com/royerlab/ultrack/blob/main/ultrack/utils/_test/test_utils_array.py


def test_merge_indices() -> None:
    assert chain_indices(slice(3, 20), slice(5, 15)) == slice(8, 18, 1)
    assert chain_indices(slice(3, 20), slice(5, None)) == slice(8, 20, 1)
    assert chain_indices(slice(3, 20), slice(None, 15)) == slice(3, 18, 1)
    assert chain_indices(slice(3, 20), 4) == 7
    assert chain_indices(slice(3, 20, 3), 2) == 9
    assert chain_indices(slice(3, 20, 3), slice(2, 6, 2)) == slice(9, 21, 6)
    assert chain_indices(slice(3, 20), [4, 5]) == [7, 8]
    assert chain_indices((5, 6, 7, 8, 9, 10), [3, 5]) == [8, 10]


def test_graph_array_view_init() -> None:
    """Test GraphArrayView initialization."""
    graph = RustWorkXGraph()

    # Add a attribute key
    graph.add_node_attr_key("label", 0)

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attr_key="label", offset=0)

    assert array_view.graph is graph
    assert array_view.shape == (10, 100, 100)
    assert array_view._attr_key == "label"
    assert array_view._offset == 0
    assert array_view.dtype == get_options().gav_default_dtype
    assert array_view.ndim == 3
    assert len(array_view) == 10


def test_graph_array_view_init_invalid_attr_key() -> None:
    """Test GraphArrayView initialization with invalid attribute key."""
    graph = RustWorkXGraph()

    with pytest.raises(ValueError, match="Attribute key 'invalid_key' not found in graph"):
        GraphArrayView(graph=graph, shape=(10, 100, 100), attr_key="invalid_key")


def test_graph_array_view_getitem_empty_time() -> None:
    """Test __getitem__ with empty time point (no nodes)."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("label", 0)

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attr_key="label")

    # Get data for time point 0 (no nodes)
    result = array_view[0]

    # Should return zeros with correct shape
    assert result.shape == (100, 100)
    assert np.all(np.asarray(result) == 0)
    assert array_view.dtype == get_options().gav_default_dtype


def test_graph_array_view_getitem_with_nodes() -> None:
    """Test __getitem__ with nodes at time point."""
    graph = RustWorkXGraph()

    # Add attribute keys
    graph.add_node_attr_key("label", 0)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, None)
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)

    # Create a mask
    mask_data = np.array([[True, True], [True, False]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([10, 20, 12, 22]))  # y_min, x_min, y_max, x_max

    # Add a node with mask and label
    graph.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 5,
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
            "y": 11,
            "x": 21,
        }
    )

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attr_key="label")

    # Get data for time point 0
    result = array_view[0]

    # Should have correct shape
    assert result.shape == (100, 100)

    # Check that the mask was painted with the label value
    # The mask should be painted at the bbox location
    assert np.asarray(result)[10, 20] == 5  # Top-left of mask
    assert np.asarray(result)[10, 21] == 5  # Top-right of mask
    assert np.asarray(result)[11, 20] == 5  # Bottom-left of mask
    assert np.asarray(result)[11, 21] == 0  # Bottom-right should be 0 (mask is False there)

    # Test indexing on grapharrayview BEFORE conversion to numpy array, especially when slicing a single value
    assert np.array_equal(result[10, 20], 5)
    assert np.array_equal(result[10, 20:22], np.array([5, 5]))

    # Other areas should be 0
    assert np.asarray(result)[0, 0] == 0
    assert np.asarray(result)[50, 50] == 0


def test_graph_array_view_getitem_multiple_nodes() -> None:
    """Test __getitem__ with multiple nodes at same time point."""
    graph = RustWorkXGraph()

    # Add attribute keys
    graph.add_node_attr_key("label", 0)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, None)
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)

    # Create two masks at different locations
    mask1_data = np.array([[True, True]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([10, 20, 11, 22]))

    mask2_data = np.array([[True]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([30, 40, 31, 41]))

    # Add nodes with different labels
    graph.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 3,
            DEFAULT_ATTR_KEYS.MASK: mask1,
            DEFAULT_ATTR_KEYS.BBOX: mask1.bbox,
            "y": 11,
            "x": 21,
        }
    )

    graph.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 7,
            DEFAULT_ATTR_KEYS.MASK: mask2,
            DEFAULT_ATTR_KEYS.BBOX: mask2.bbox,
            "y": 31,
            "x": 41,
        }
    )

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attr_key="label")

    # Get data for time point 0
    result = array_view[0]

    # Check that both masks were painted with their respective labels
    assert np.asarray(result)[10, 20] == 3
    assert np.asarray(result)[10, 21] == 3
    assert np.asarray(result)[30, 40] == 7

    # Other areas should be 0
    assert np.asarray(result)[0, 0] == 0
    assert np.asarray(result)[50, 50] == 0


def test_graph_array_view_getitem_boolean_dtype() -> None:
    """Test __getitem__ with boolean attribute values."""
    graph = RustWorkXGraph()

    # Add attribute keys
    graph.add_node_attr_key("is_active", False)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, None)
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)

    # Create a mask
    mask_data = np.array([[True]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([10, 20, 11, 21]))

    # Add a node with boolean attribute
    graph.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "is_active": True,
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
            "y": 11,
            "x": 21,
        }
    )

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attr_key="is_active")

    # Get data for time point 0
    result = array_view[0]

    # Boolean values should be converted to uint8 for napari
    assert result.dtype == np.uint8
    assert np.asarray(result)[10, 20] == 1  # True -> 1
    assert np.asarray(result)[0, 0] == 0  # False -> 0


def test_graph_array_view_dtype_inference() -> None:
    """Test that dtype is properly inferred from data."""
    graph = RustWorkXGraph()

    # Add attribute keys
    graph.add_node_attr_key("float_label", 0.0)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, None)
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)

    # Create a mask
    mask_data = np.array([[True]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([10, 20, 11, 21]))

    # Add a node with float attribute
    graph.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "float_label": 3.14,
            DEFAULT_ATTR_KEYS.MASK: mask,
            "y": 11,
            "x": 21,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
        }
    )

    array_view = GraphArrayView(graph=graph, shape=(10, 100, 100), attr_key="float_label")

    # Get data to trigger dtype inference
    _ = array_view[0]

    # Dtype should be updated based on the actual data
    assert array_view.dtype == np.float64


@fixture(
    params=[
        (10, 100, 100),
        (10, 100, 100, 100),
    ]
)
def multi_node_graph_from_image(request) -> GraphArrayView:
    """Fixture to create a graph with multiple nodes for testing."""
    shape = request.param
    label = np.zeros(shape, dtype=np.uint8)
    for i in range(shape[0]):
        label[i, 10:20, 10:20] = i + 1
    graph = RustWorkXGraph()
    nodes_operator = RegionPropsNodes(extra_properties=["label"])
    nodes_operator.add_nodes(graph, labels=label)
    return GraphArrayView(graph=graph, shape=shape, attr_key="label"), label


def test_graph_array_view_equal(multi_node_graph_from_image) -> None:
    array_view, label = multi_node_graph_from_image
    assert array_view.shape == label.shape
    for t in range(array_view.shape[0]):
        print(np.unique(array_view[t]), np.unique(label[t]))
        assert np.array_equal(array_view[t], label[t])
    assert np.array_equal(array_view, label)
    assert np.array_equal(array_view[:5], label[:5])
    assert np.array_equal(array_view[[1, 6, 7]], label[[1, 6, 7]])
    assert np.array_equal(array_view[:, 3, 10:20][3, 2:5], label[:, 3, 10:20][3, 2:5])
    assert array_view.ndim == label.ndim
    assert array_view.dtype == np.int64  # fixed
    assert array_view.shape == label.shape
    # assert np.array_equal(array_view[0], label[0])


def test_graph_array_view_getitem_multi_slices(multi_node_graph_from_image) -> None:
    """Test __getitem__ with slices."""
    array_view, label = multi_node_graph_from_image

    for count_slice in range(1, array_view.ndim):
        # Test with slice(10, 20)
        window = tuple([5] + [slice(10, 20)] * count_slice)
        assert np.array_equal(array_view[window], label[window])
        # Test with slice(10, 20, 2)
        window = tuple([5] + [slice(10, 20, 2)] * count_slice)
        assert np.array_equal(array_view[window], label[window])
        # Test with slice(None, 20)
        window = tuple([5] + [slice(None, 20)] * count_slice)
        assert np.array_equal(array_view[window], label[window])
        # Test with slice(10, None)
        window = tuple([5] + [slice(10, None)] * count_slice)
        assert np.array_equal(array_view[window], label[window])
        # Test with slice(None, None)
        window = tuple([5] + [slice(None, None)] * count_slice)
        assert np.array_equal(array_view[window], label[window])


possible_combinations = [
    (slice(3, 20), slice(5, None)),
    (slice(3, 20), slice(None, 15)),
    (slice(3, 20), slice(None, 15)),
    (slice(3, 20, 4), slice(None, 15)),
    (slice(3, 20), 4),
    (slice(3, 20), [4, 5]),
    ([5, 6, 9, 8, 7], slice(1, 3)),
    (4, 0),
]


@pytest.mark.parametrize("index1, index2", possible_combinations)
def test_graph_array_view_getitem_time_index_nested(multi_node_graph_from_image, index1, index2) -> None:
    """Test __getitem__ with nested indices."""
    array_view, label = multi_node_graph_from_image
    msg = f"Failed for index1={index1}, index2={index2}"
    assert np.array_equal(array_view[index1][index2], label[index1][index2]), msg
    assert array_view[index1][index2].shape == label[index1][index2].shape, msg
    assert array_view[index1][index2].ndim == label[index1][index2].ndim, msg

    if isinstance(index1, Sequence) or isinstance(index2, Sequence):
        with pytest.raises(NotImplementedError):
            # This should raise an error to avoid inconsistency with numpy-like indexing
            array_view[(index1, index1)].__array__()
            array_view[(index2, index2)].__array__()
    elif not isinstance(index1, int):
        msg = f"Failed for index1={index1}, index2={index2}"
        expected_array = label[(index1, index1)][(index2, index2)]
        actual_array = array_view[(index1, index1)][(index2, index2)]
        assert np.array_equal(actual_array, expected_array), msg


def test_graph_array_set_options() -> None:
    with Options(gav_chunk_shape=(512, 512), gav_default_dtype=np.int16):
        empty_graph = RustWorkXGraph()
        empty_graph.add_node_attr_key("label", 0)
        array_view = GraphArrayView(graph=empty_graph, shape=(10, 100, 100), attr_key="label")
        assert array_view.chunk_shape == (512, 512)
        assert array_view.dtype == np.int16
