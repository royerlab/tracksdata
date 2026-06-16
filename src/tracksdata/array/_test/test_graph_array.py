from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from pytest import fixture

from tracksdata.array import GraphArrayView
from tracksdata.array._graph_array import chain_indices
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import BaseGraph
from tracksdata.nodes import RegionPropsNodes
from tracksdata.nodes._mask import Mask
from tracksdata.options import Options, get_options

# NOTE: this could be generic test for all array backends
# when more slicing operations are implemented we could test as in:
#  - https://github.com/royerlab/ultrack/blob/main/ultrack/utils/_test/test_utils_array.py


def test_chain_indices() -> None:
    assert chain_indices(slice(3, 20), slice(5, 15)) == slice(8, 18, 1)
    assert chain_indices(slice(3, 20), slice(5, None)) == slice(8, 20, 1)
    assert chain_indices(slice(3, 20), slice(None, 15)) == slice(3, 18, 1)
    assert chain_indices(slice(3, 20), 4) == 7
    assert chain_indices(slice(3, 20, 3), 2) == 9
    assert chain_indices(slice(3, 20, 3), slice(2, 6, 2)) == slice(9, 21, 6)
    assert chain_indices(slice(3, 20), [4, 5]) == [7, 8]
    assert chain_indices((5, 6, 7, 8, 9, 10), [3, 5]) == [8, 10]


def test_graph_array_view_init(graph_backend: BaseGraph) -> None:
    """Test GraphArrayView initialization."""
    # Add a attribute key
    graph_backend.add_node_attr_key("label", dtype=pl.Int64)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 6))

    array_view = GraphArrayView(graph=graph_backend, shape=(10, 100, 100), attr_key="label", offset=0)

    assert array_view.graph is graph_backend
    assert array_view.shape == (10, 100, 100)
    assert array_view._attr_key == "label"
    assert array_view._offset == 0
    assert array_view.dtype == get_options().gav_default_dtype
    assert array_view.ndim == 3
    assert len(array_view) == 10
    assert array_view.size == 10 * 100 * 100


def test_graph_array_view_init_invalid_attr_key(graph_backend: BaseGraph) -> None:
    """Test GraphArrayView initialization with invalid attribute key."""

    with pytest.raises(ValueError, match="Attribute key 'invalid_key' not found in graph"):
        GraphArrayView(graph=graph_backend, shape=(10, 100, 100), attr_key="invalid_key")


def test_graph_array_view_getitem_empty_time(graph_backend: BaseGraph) -> None:
    """Test __getitem__ with empty time point (no nodes)."""

    graph_backend.add_node_attr_key("label", dtype=pl.Int64)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 6))

    array_view = GraphArrayView(graph=graph_backend, shape=(10, 100, 100), attr_key="label")

    # Get data for time point 0 (no nodes)
    result = array_view[0]

    # Should return zeros with correct shape
    assert result.shape == (100, 100)
    assert np.all(np.asarray(result) == 0)
    assert array_view.dtype == get_options().gav_default_dtype


def test_graph_array_view_getitem_with_nodes(graph_backend: BaseGraph) -> None:
    """Test __getitem__ with nodes at time point."""

    # Add attribute keys
    graph_backend.add_node_attr_key("label", dtype=pl.Int64)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))
    graph_backend.add_node_attr_key("y", dtype=pl.Int64)
    graph_backend.add_node_attr_key("x", dtype=pl.Int64)

    # Create a mask
    mask_data = np.array([[True, True], [True, False]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([10, 20, 12, 22]))  # y_min, x_min, y_max, x_max

    # Add a node with mask and label
    graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 5,
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
            "y": 11,
            "x": 21,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(10, 100, 100), attr_key="label")

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


def test_graph_array_view_getitem_multiple_nodes(graph_backend: BaseGraph) -> None:
    """Test __getitem__ with multiple nodes at same time point."""

    # Add attribute keys
    graph_backend.add_node_attr_key("label", dtype=pl.Int64)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))
    graph_backend.add_node_attr_key("y", dtype=pl.Int64)
    graph_backend.add_node_attr_key("x", dtype=pl.Int64)

    # Create two masks at different locations
    mask1_data = np.array([[True, True]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([10, 20, 11, 22]))

    mask2_data = np.array([[True]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([30, 40, 31, 41]))

    # Add nodes with different labels
    graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 3,
            DEFAULT_ATTR_KEYS.MASK: mask1,
            DEFAULT_ATTR_KEYS.BBOX: mask1.bbox,
            "y": 11,
            "x": 21,
        }
    )

    graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 7,
            DEFAULT_ATTR_KEYS.MASK: mask2,
            DEFAULT_ATTR_KEYS.BBOX: mask2.bbox,
            "y": 31,
            "x": 41,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(10, 100, 100), attr_key="label")

    # Get data for time point 0
    result = array_view[0]

    # Check that both masks were painted with their respective labels
    assert np.asarray(result)[10, 20] == 3
    assert np.asarray(result)[10, 21] == 3
    assert np.asarray(result)[30, 40] == 7

    # Other areas should be 0
    assert np.asarray(result)[0, 0] == 0
    assert np.asarray(result)[50, 50] == 0


def test_graph_array_view_getitem_boolean_dtype(graph_backend: BaseGraph) -> None:
    """Test __getitem__ with boolean attribute values."""

    # Add attribute keys
    graph_backend.add_node_attr_key("is_active", pl.Boolean)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))
    graph_backend.add_node_attr_key("y", dtype=pl.Int64)
    graph_backend.add_node_attr_key("x", dtype=pl.Int64)

    # Create a mask
    mask_data = np.array([[True]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([10, 20, 11, 21]))

    # Add a node with boolean attribute
    graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "is_active": True,
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
            "y": 11,
            "x": 21,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(10, 100, 100), attr_key="is_active")

    # Get data for time point 0
    result = array_view[0]

    # Boolean values should be converted to uint8 for napari
    assert result.dtype == np.uint8
    assert np.asarray(result)[10, 20] == 1  # True -> 1
    assert np.asarray(result)[0, 0] == 0  # False -> 0


def test_graph_array_view_dtype_inference(graph_backend: BaseGraph) -> None:
    """Test that dtype is properly inferred from data."""

    # Add attribute keys
    graph_backend.add_node_attr_key("float_label", dtype=pl.Float64)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))
    graph_backend.add_node_attr_key("y", dtype=pl.Int64)
    graph_backend.add_node_attr_key("x", dtype=pl.Int64)

    # Create a mask
    mask_data = np.array([[True]], dtype=bool)
    mask = Mask(mask_data, bbox=np.array([10, 20, 11, 21]))

    # Add a node with float attribute
    graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "float_label": 3.14,
            DEFAULT_ATTR_KEYS.MASK: mask,
            "y": 11,
            "x": 21,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(10, 100, 100), attr_key="float_label")

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
def multi_node_graph_from_image(request, graph_backend) -> tuple[GraphArrayView, np.ndarray]:
    """Fixture to create a graph with multiple nodes for testing."""
    shape = request.param
    label = np.zeros(shape, dtype=np.uint8)
    for i in range(shape[0]):
        label[i, 10:20, 10:20] = i + 1

    nodes_operator = RegionPropsNodes(extra_properties=["label"])
    nodes_operator.add_nodes(graph_backend, labels=label)
    return GraphArrayView(graph=graph_backend, shape=shape, attr_key="label"), label


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


def test_graph_array_set_options(graph_backend: BaseGraph) -> None:
    with Options(gav_chunk_shape=(512, 512), gav_default_dtype=np.int16):
        graph_backend.add_node_attr_key("label", dtype=pl.Int64)
        graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
        graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))
        array_view = GraphArrayView(graph=graph_backend, shape=(10, 100, 100), attr_key="label")
        assert array_view.chunk_shape == (512, 512)
        assert array_view.dtype == np.int16


def test_graph_array_raise_error_on_absent_attr_key(graph_backend: BaseGraph) -> None:
    """Test that GraphArrayView raises error if attr_key is absent in the graph or not specified."""

    # Do not add any attribute keys

    with pytest.raises(ValueError, match="Attribute key 'label' not found in graph"):
        GraphArrayView(graph=graph_backend, shape=(10, 100, 100), attr_key="label")
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'attr_key'"):
        GraphArrayView(graph=graph_backend, shape=(10, 100, 100))  # type: ignore


def test_graph_array_raise_error_on_non_scalar_attr_key(graph_backend: BaseGraph) -> None:
    """Test that GraphArrayView raises error if attr_key values are non-scalar."""

    # Add a attribute key
    graph_backend.add_node_attr_key(
        "label", dtype=pl.Object, default_value=np.array([0, 1])
    )  # Non-scalar default value
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": np.array([1, 2]),  # Non-scalar value
            DEFAULT_ATTR_KEYS.MASK: Mask(np.array([[True]], dtype=bool), bbox=np.array([0, 0, 1, 1])),
        }
    )

    with pytest.raises(ValueError, match="Attribute values for key 'label' must be scalar"):
        GraphArrayView(graph=graph_backend, shape=(10, 100, 100), attr_key="label")


def _add_graph_array_node_attrs(graph_backend: BaseGraph) -> None:
    graph_backend.add_node_attr_key("label", dtype=pl.Int64)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))


def _make_square_mask(y: int, x: int, size: int = 2) -> Mask:
    return Mask(np.ones((size, size), dtype=bool), bbox=np.array([y, x, y + size, x + size]))


def test_graph_array_view_invalidates_only_affected_chunk_on_add(graph_backend: BaseGraph) -> None:
    _add_graph_array_node_attrs(graph_backend)

    first_mask = _make_square_mask(1, 1)
    graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 1,
            DEFAULT_ATTR_KEYS.MASK: first_mask,
            DEFAULT_ATTR_KEYS.BBOX: first_mask.bbox,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(2, 8, 8), attr_key="label", chunk_shape=(4, 4))

    _ = np.asarray(array_view[0])
    np.testing.assert_array_equal(array_view._cache._store[0].ready, np.ones((2, 2), dtype=bool))

    second_mask = _make_square_mask(5, 5)
    graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 2,
            DEFAULT_ATTR_KEYS.MASK: second_mask,
            DEFAULT_ATTR_KEYS.BBOX: second_mask.bbox,
        }
    )

    expected_ready = np.ones((2, 2), dtype=bool)
    expected_ready[1, 1] = False
    np.testing.assert_array_equal(array_view._cache._store[0].ready, expected_ready)

    output = np.asarray(array_view[0])
    assert output[1, 1] == 1
    assert output[5, 5] == 2


def test_graph_array_view_invalidates_old_and_new_chunks_on_update(graph_backend: BaseGraph) -> None:
    _add_graph_array_node_attrs(graph_backend)

    mask = _make_square_mask(1, 1)
    node_id = graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 1,
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(2, 8, 8), attr_key="label", chunk_shape=(4, 4))
    _ = np.asarray(array_view[0])

    moved_mask = _make_square_mask(5, 5)
    graph_backend.update_node_attrs(
        attrs={
            "label": [7],
            DEFAULT_ATTR_KEYS.MASK: [moved_mask],
            DEFAULT_ATTR_KEYS.BBOX: [moved_mask.bbox],
        },
        node_ids=[node_id],
    )

    expected_ready = np.ones((2, 2), dtype=bool)
    expected_ready[0, 0] = False
    expected_ready[1, 1] = False
    np.testing.assert_array_equal(array_view._cache._store[0].ready, expected_ready)

    output = np.asarray(array_view[0])
    assert output[1, 1] == 0
    assert output[5, 5] == 7


def test_graph_array_view_invalidates_once_when_attr_key_changes_but_bbox_unchanged(
    graph_backend: BaseGraph,
) -> None:
    """Updating the displayed attr_key (label) without moving the node should invalidate the region exactly once."""
    _add_graph_array_node_attrs(graph_backend)

    mask = _make_square_mask(1, 1)
    node_id = graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 1,
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(2, 8, 8), attr_key="label", chunk_shape=(4, 4))
    _ = np.asarray(array_view[0])
    np.testing.assert_array_equal(array_view._cache._store[0].ready, np.ones((2, 2), dtype=bool))

    mock_invalidate = MagicMock(wraps=array_view._invalidate_bbox)
    with patch.object(array_view, "_invalidate_bbox", mock_invalidate):
        graph_backend.update_node_attrs(
            attrs={"label": [7]},
            node_ids=[node_id],
        )
        # bbox unchanged, but the displayed attribute changed — invalidate exactly one region
        n_regions = sum(len(call.args[0]) for call in mock_invalidate.call_args_list)
        assert n_regions == 1

    # The affected chunk must be invalidated
    expected_ready = np.ones((2, 2), dtype=bool)
    expected_ready[0, 0] = False
    np.testing.assert_array_equal(array_view._cache._store[0].ready, expected_ready)

    # After recomputation, the new value should be painted
    output = np.asarray(array_view[0])
    assert output[1, 1] == 7


def test_graph_array_view_invalidates_twice_when_attr_key_and_bbox_change(graph_backend: BaseGraph) -> None:
    """Updating both the displayed attr_key and the bbox should invalidate old and new regions."""
    _add_graph_array_node_attrs(graph_backend)

    mask = _make_square_mask(1, 1)
    node_id = graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 1,
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(2, 8, 8), attr_key="label", chunk_shape=(4, 4))
    _ = np.asarray(array_view[0])
    np.testing.assert_array_equal(array_view._cache._store[0].ready, np.ones((2, 2), dtype=bool))

    moved_mask = _make_square_mask(5, 5)
    mock_invalidate = MagicMock(wraps=array_view._invalidate_bbox)
    with patch.object(array_view, "_invalidate_bbox", mock_invalidate):
        graph_backend.update_node_attrs(
            attrs={
                "label": [7],
                DEFAULT_ATTR_KEYS.MASK: [moved_mask],
                DEFAULT_ATTR_KEYS.BBOX: [moved_mask.bbox],
            },
            node_ids=[node_id],
        )
        # bbox changed — must invalidate both old and new regions
        n_regions = sum(len(call.args[0]) for call in mock_invalidate.call_args_list)
        assert n_regions == 2

    expected_ready = np.ones((2, 2), dtype=bool)
    expected_ready[0, 0] = False
    expected_ready[1, 1] = False
    np.testing.assert_array_equal(array_view._cache._store[0].ready, expected_ready)

    output = np.asarray(array_view[0])
    assert output[1, 1] == 0
    assert output[5, 5] == 7


def test_graph_array_view_no_invalidation_when_unrelated_attr_changes(graph_backend: BaseGraph) -> None:
    """Updating an attribute the view doesn't display should not invalidate any chunks."""
    _add_graph_array_node_attrs(graph_backend)
    graph_backend.add_node_attr_key("score", dtype=pl.Float64)

    mask = _make_square_mask(1, 1)
    node_id = graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 1,
            "score": 0.5,
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(2, 8, 8), attr_key="label", chunk_shape=(4, 4))
    _ = np.asarray(array_view[0])
    np.testing.assert_array_equal(array_view._cache._store[0].ready, np.ones((2, 2), dtype=bool))

    mock_invalidate = MagicMock(wraps=array_view._invalidate_bbox)
    with patch.object(array_view, "_invalidate_bbox", mock_invalidate):
        graph_backend.update_node_attrs(
            attrs={"score": [0.9]},
            node_ids=[node_id],
        )
        # Neither bbox nor the displayed attribute changed — no region invalidated
        n_regions = sum(len(call.args[0]) for call in mock_invalidate.call_args_list)
        assert n_regions == 0

    np.testing.assert_array_equal(
        array_view._cache._store[0].ready,
        np.ones((2, 2), dtype=bool),
    )


def test_graph_array_view_invalidates_chunk_on_remove(graph_backend: BaseGraph) -> None:
    _add_graph_array_node_attrs(graph_backend)

    first_mask = _make_square_mask(1, 1)
    graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 1,
            DEFAULT_ATTR_KEYS.MASK: first_mask,
            DEFAULT_ATTR_KEYS.BBOX: first_mask.bbox,
        }
    )
    second_mask = _make_square_mask(5, 5)
    second_node = graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 2,
            DEFAULT_ATTR_KEYS.MASK: second_mask,
            DEFAULT_ATTR_KEYS.BBOX: second_mask.bbox,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(2, 8, 8), attr_key="label", chunk_shape=(4, 4))
    _ = np.asarray(array_view[0])

    graph_backend.remove_node(second_node)

    expected_ready = np.ones((2, 2), dtype=bool)
    expected_ready[1, 1] = False
    np.testing.assert_array_equal(array_view._cache._store[0].ready, expected_ready)

    output = np.asarray(array_view[0])
    assert output[1, 1] == 1
    assert output[5, 5] == 0


def test_graph_array_view_invalidates_whole_volume_when_bbox_missing(graph_backend: BaseGraph) -> None:
    """A node event without a bbox key has an unknown location, so the whole time volume is invalidated."""
    _add_graph_array_node_attrs(graph_backend)

    mask = _make_square_mask(1, 1)
    graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 1,
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(2, 8, 8), attr_key="label", chunk_shape=(4, 4))
    _ = np.asarray(array_view[0])
    np.testing.assert_array_equal(array_view._cache._store[0].ready, np.ones((2, 2), dtype=bool))

    # Emit a node-added event whose attrs lack a bbox key: location unknown.
    array_view._on_node_added([999], [{DEFAULT_ATTR_KEYS.T: 0, "label": 5}])

    # The entire volume for time 0 must be invalidated.
    np.testing.assert_array_equal(array_view._cache._store[0].ready, np.zeros((2, 2), dtype=bool))


def test_graph_array_view_invalidates_once_when_mask_changes_but_bbox_unchanged(graph_backend: BaseGraph) -> None:
    """Swapping the mask pixels without moving the bbox must invalidate the region exactly once."""
    _add_graph_array_node_attrs(graph_backend)

    mask = _make_square_mask(1, 1)  # bbox [1, 1, 3, 3], fully filled
    node_id = graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 1,
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(2, 8, 8), attr_key="label", chunk_shape=(4, 4))
    _ = np.asarray(array_view[0])
    np.testing.assert_array_equal(array_view._cache._store[0].ready, np.ones((2, 2), dtype=bool))

    # Same bbox [1, 1, 3, 3], but only the diagonal pixels are set.
    new_mask = Mask(np.array([[True, False], [False, True]], dtype=bool), bbox=np.array([1, 1, 3, 3]))
    mock_invalidate = MagicMock(wraps=array_view._invalidate_bbox)
    with patch.object(array_view, "_invalidate_bbox", mock_invalidate):
        graph_backend.update_node_attrs(
            attrs={DEFAULT_ATTR_KEYS.MASK: [new_mask]},
            node_ids=[node_id],
        )
        # bbox and label unchanged, but the mask pixels changed — invalidate exactly one region
        n_regions = sum(len(call.args[0]) for call in mock_invalidate.call_args_list)
        assert n_regions == 1

    expected_ready = np.ones((2, 2), dtype=bool)
    expected_ready[0, 0] = False
    np.testing.assert_array_equal(array_view._cache._store[0].ready, expected_ready)

    # After recomputation only the diagonal pixels carry the label.
    output = np.asarray(array_view[0])
    assert output[1, 1] == 1
    assert output[2, 2] == 1
    assert output[1, 2] == 0
    assert output[2, 1] == 0


def test_graph_array_view_no_invalidation_when_mask_unchanged(graph_backend: BaseGraph) -> None:
    """Re-supplying an identical mask (same bbox and pixels) must not invalidate anything."""
    _add_graph_array_node_attrs(graph_backend)

    mask = _make_square_mask(1, 1)
    node_id = graph_backend.add_node(
        {
            DEFAULT_ATTR_KEYS.T: 0,
            "label": 1,
            DEFAULT_ATTR_KEYS.MASK: mask,
            DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
        }
    )

    array_view = GraphArrayView(graph=graph_backend, shape=(2, 8, 8), attr_key="label", chunk_shape=(4, 4))
    _ = np.asarray(array_view[0])
    np.testing.assert_array_equal(array_view._cache._store[0].ready, np.ones((2, 2), dtype=bool))

    # A fresh Mask object with identical bbox and pixels: no rendered change.
    same_mask = Mask(np.ones((2, 2), dtype=bool), bbox=np.array([1, 1, 3, 3]))
    mock_invalidate = MagicMock(wraps=array_view._invalidate_bbox)
    with patch.object(array_view, "_invalidate_bbox", mock_invalidate):
        graph_backend.update_node_attrs(
            attrs={DEFAULT_ATTR_KEYS.MASK: [same_mask]},
            node_ids=[node_id],
        )
        # Nothing affecting the rendered output changed — no region invalidated
        n_regions = sum(len(call.args[0]) for call in mock_invalidate.call_args_list)
        assert n_regions == 0

    np.testing.assert_array_equal(array_view._cache._store[0].ready, np.ones((2, 2), dtype=bool))
