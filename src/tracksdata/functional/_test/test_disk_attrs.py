import numpy as np
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._disk_attrs import (
    MaskDiskAttrs,
    _create_mask_and_bbox_from_coordinates,
    _spherical_mask,
)
from tracksdata.graph import RustWorkXGraph
from tracksdata.utils._test_utils import setup_spatial_attrs_2d, setup_spatial_attrs_3d


def test_spherical_mask_2d() -> None:
    """Test spherical mask creation in 2D."""
    mask = _spherical_mask(radius=2, ndim=2)

    assert mask.dtype == bool
    assert mask.shape == (5, 5)  # Disk radius 2 creates 5x5 mask
    assert mask[2, 2]  # Center should be True


def test_spherical_mask_3d() -> None:
    """Test spherical mask creation in 3D."""
    mask = _spherical_mask(radius=1, ndim=3)

    assert mask.dtype == bool
    assert mask.shape == (3, 3, 3)  # Ball radius 1 creates 3x3x3 mask
    assert mask[1, 1, 1]  # Center should be True


def test_spherical_mask_invalid_ndim() -> None:
    """Test spherical mask with invalid dimensions."""
    with pytest.raises(ValueError, match="Spherical is only implemented for 2D and 3D"):
        _spherical_mask(radius=1, ndim=4)


def test_create_mask_and_bbox_from_coordinates_2d_basic() -> None:
    """Test 2D mask creation and bbox without cropping."""
    center = np.asarray([5, 5])
    radius = 2

    mask, bbox = _create_mask_and_bbox_from_coordinates(center, radius)

    # Should be a disk of radius 2, shape (5,5), centered at (5,5)
    assert mask.shape == (5, 5)
    assert mask[2, 2]  # center pixel is True
    np.testing.assert_array_equal(bbox, [3, 3, 8, 8])


def test_create_mask_and_bbox_from_coordinates_3d_basic() -> None:
    """Test 3D mask creation and bbox without cropping."""
    center = np.asarray([4, 5, 6])
    radius = 1

    mask, bbox = _create_mask_and_bbox_from_coordinates(center, radius)

    # Should be a ball of radius 1, shape (3,3,3), centered at (4,5,6)
    assert mask.shape == (3, 3, 3)
    assert mask[1, 1, 1]  # center voxel is True
    np.testing.assert_array_equal(bbox, [3, 4, 5, 6, 7, 8])


def test_create_mask_and_bbox_from_coordinates_cropping() -> None:
    """Test cropping when mask falls outside the image boundary."""
    center = np.asarray([0, 0])
    radius = 5
    image_shape = (4, 3)

    mask, bbox = _create_mask_and_bbox_from_coordinates(center, radius, image_shape=image_shape)

    # Mask shape should match the bbox size
    expected_shape = (4, 3)
    assert mask.shape == expected_shape

    # Mask should be cropped to fit within image bounds
    np.testing.assert_array_equal(bbox, [0, 0, 4, 3])


def test_create_mask_and_bbox_from_coordinates_partial_cropping() -> None:
    """Test partial cropping when mask partially falls outside boundary."""
    center = np.asarray([2, 8])  # Near right edge
    radius = 2
    image_shape = (10, 10)

    mask, bbox = _create_mask_and_bbox_from_coordinates(center, radius, image_shape=image_shape)

    # Should be cropped on the right side
    assert mask.shape[1] < 5  # Original disk would be 5x5
    assert bbox[3] == 10  # Right edge should be image boundary
    assert bbox[1] == 6  # Left edge should be center - radius


def test_create_mask_and_bbox_from_coordinates_float_center() -> None:
    """Test with float center coordinates (should be rounded)."""
    center = np.asarray([2.7, 3.2])
    radius = 1

    mask, bbox = _create_mask_and_bbox_from_coordinates(center, radius)

    # Center should be rounded to [3, 3]
    expected_bbox = [2, 2, 5, 5]  # center (3,3) with radius 1
    np.testing.assert_array_equal(bbox, expected_bbox)


def test_mask_disk_attrs_init_default() -> None:
    """Test MaskDiskAttrs initialization with default parameters."""
    operator = MaskDiskAttrs(radius=2, image_shape=(10, 10))

    assert operator.radius == 2
    assert operator.image_shape == (10, 10)
    assert operator.attr_keys == ["y", "x"]  # Default for 2D
    assert operator.mask_output_key == DEFAULT_ATTR_KEYS.MASK
    assert operator.bbox_output_key == DEFAULT_ATTR_KEYS.BBOX


def test_mask_disk_attrs_init_custom() -> None:
    """Test MaskDiskAttrs initialization with custom parameters."""
    operator = MaskDiskAttrs(
        radius=3,
        image_shape=(5, 10, 15),
        attr_keys=["z", "y", "x"],
        mask_output_key="custom_mask",
        bbox_output_key="custom_bbox",
    )

    assert operator.radius == 3
    assert operator.image_shape == (5, 10, 15)
    assert operator.attr_keys == ["z", "y", "x"]
    assert operator.mask_output_key == "custom_mask"
    assert operator.bbox_output_key == "custom_bbox"


def test_mask_disk_attrs_init_auto_attr_keys() -> None:
    """Test automatic attr_keys selection based on image_shape."""
    # 2D case
    operator_2d = MaskDiskAttrs(radius=1, image_shape=(10, 20))
    assert operator_2d.attr_keys == ["y", "x"]

    # 3D case
    operator_3d = MaskDiskAttrs(radius=1, image_shape=(5, 10, 15))
    assert operator_3d.attr_keys == ["z", "y", "x"]


def test_mask_disk_attrs_init_dimension_mismatch() -> None:
    """Test error when image_shape and attr_keys have different dimensions."""
    with pytest.raises(ValueError, match="Expected image shape"):
        MaskDiskAttrs(radius=1, image_shape=(10, 20), attr_keys=["z", "y", "x"])  # 3D keys for 2D shape


def test_mask_disk_attrs_add_nodes_2d() -> None:
    """Test adding disk masks to 2D nodes."""
    graph = RustWorkXGraph()

    # Initialize required attributes
    setup_spatial_attrs_2d(graph)

    # Add 2 nodes at t=0
    node_attrs = [{DEFAULT_ATTR_KEYS.T: 0, "y": 5.0, "x": 10.0}, {DEFAULT_ATTR_KEYS.T: 0, "y": 15.0, "x": 20.0}]
    graph.bulk_add_nodes(node_attrs)

    # Add disk masks
    disk_operator = MaskDiskAttrs(radius=2, image_shape=(30, 40))
    disk_operator.add_node_attrs(graph)

    # Check that masks and bboxes were added
    nodes_df = graph.node_attrs()
    assert DEFAULT_ATTR_KEYS.MASK in nodes_df.columns
    assert DEFAULT_ATTR_KEYS.BBOX in nodes_df.columns

    # Check that we have 2 nodes with masks
    assert len(nodes_df) == 2

    # Check mask properties
    masks = nodes_df[DEFAULT_ATTR_KEYS.MASK]
    bboxes = nodes_df[DEFAULT_ATTR_KEYS.BBOX]

    for i in range(len(masks)):
        mask = masks[i]
        bbox = bboxes[i]

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (5, 5)  # Radius 2 creates 5x5 disk

        assert len(bbox) == 4  # [min_y, min_x, max_y, max_x]


def test_mask_disk_attrs_add_nodes_3d() -> None:
    """Test adding disk masks to 3D nodes."""
    graph = RustWorkXGraph()

    # Initialize required attributes
    setup_spatial_attrs_3d(graph)

    # Add 2 nodes at t=0 with 3D coordinates
    node_attrs = [
        {DEFAULT_ATTR_KEYS.T: 0, "z": 2.0, "y": 5.0, "x": 10.0},
        {DEFAULT_ATTR_KEYS.T: 0, "z": 8.0, "y": 15.0, "x": 20.0},
    ]
    graph.bulk_add_nodes(node_attrs)

    # Add disk masks
    disk_operator = MaskDiskAttrs(radius=1, image_shape=(10, 20, 30))
    disk_operator.add_node_attrs(graph)

    # Check that masks and bboxes were added
    nodes_df = graph.node_attrs()
    assert DEFAULT_ATTR_KEYS.MASK in nodes_df.columns
    assert DEFAULT_ATTR_KEYS.BBOX in nodes_df.columns

    # Check mask properties
    masks = nodes_df[DEFAULT_ATTR_KEYS.MASK]
    bboxes = nodes_df[DEFAULT_ATTR_KEYS.BBOX]

    for i in range(len(masks)):
        mask = masks[i]
        bbox = bboxes[i]

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (3, 3, 3)  # Radius 1 creates 3x3x3 ball

        assert len(bbox) == 6  # [min_z, min_y, min_x, max_z, max_y, max_x]


def test_mask_disk_attrs_with_cropping() -> None:
    """Test disk mask creation with image boundary cropping."""
    graph = RustWorkXGraph()

    # Initialize required attributes
    setup_spatial_attrs_2d(graph)

    # Add a node near the edge that will require cropping
    node_attrs = [{DEFAULT_ATTR_KEYS.T: 0, "y": 1.0, "x": 1.0}]  # Near top-left corner
    graph.bulk_add_nodes(node_attrs)

    # Add disk masks with small image shape to force cropping
    disk_operator = MaskDiskAttrs(radius=3, image_shape=(5, 5))
    disk_operator.add_node_attrs(graph)

    # Check that mask was cropped
    nodes_df = graph.node_attrs()
    mask = nodes_df[DEFAULT_ATTR_KEYS.MASK][0]
    bbox = nodes_df[DEFAULT_ATTR_KEYS.BBOX][0]

    # Mask should be smaller than full disk due to cropping
    assert mask.shape[0] < 7  # Full disk would be 7x7
    assert mask.shape[1] < 7

    # Bbox should start at image boundary
    assert bbox[0] == 0  # min_y should be 0 (image boundary)
    assert bbox[1] == 0  # min_x should be 0 (image boundary)


def test_mask_disk_attrs_empty_time_point() -> None:
    """Test behavior when no nodes exist at specified time point."""
    graph = RustWorkXGraph()

    # Don't add any nodes, so t=0 will be empty
    disk_operator = MaskDiskAttrs(radius=1, image_shape=(10, 10))

    # This should handle empty time point gracefully
    node_ids, attrs = disk_operator._node_attrs_per_time(t=0, graph=graph)

    assert node_ids == []
    assert attrs == {}


def test_mask_disk_attrs_multiple_time_points() -> None:
    """Test disk mask creation across multiple time points."""
    graph = RustWorkXGraph()

    # Initialize required attributes
    setup_spatial_attrs_2d(graph)

    # Add nodes at different time points
    node_attrs = [
        {DEFAULT_ATTR_KEYS.T: 0, "y": 5.0, "x": 5.0},  # t=0
        {DEFAULT_ATTR_KEYS.T: 1, "y": 10.0, "x": 10.0},  # t=1
        {DEFAULT_ATTR_KEYS.T: 1, "y": 15.0, "x": 15.0},  # t=1
    ]
    graph.bulk_add_nodes(node_attrs)

    # Add disk masks
    disk_operator = MaskDiskAttrs(radius=1, image_shape=(20, 20))
    disk_operator.add_node_attrs(graph)

    # Check that all nodes got masks
    nodes_df = graph.node_attrs()
    assert len(nodes_df) == 3  # 1 + 2 nodes

    # Check that masks were added for all time points
    assert all(nodes_df[DEFAULT_ATTR_KEYS.MASK].is_not_null())
    assert all(nodes_df[DEFAULT_ATTR_KEYS.BBOX].is_not_null())


def test_mask_disk_attrs_init_graph_attributes() -> None:
    """Test that graph attributes are properly initialized."""
    graph = RustWorkXGraph()

    disk_operator = MaskDiskAttrs(
        radius=1, image_shape=(10, 10), mask_output_key="test_mask", bbox_output_key="test_bbox"
    )

    # Before initialization, attributes shouldn't exist
    assert "test_mask" not in graph.node_attr_keys
    assert "test_bbox" not in graph.node_attr_keys

    # Initialize attributes
    disk_operator._init_node_attrs(graph)

    # After initialization, attributes should exist
    assert "test_mask" in graph.node_attr_keys
    assert "test_bbox" in graph.node_attr_keys
