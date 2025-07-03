import numpy as np

from tracksdata.nodes._mask import Mask


def test_mask_init() -> None:
    """Test Mask initialization."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask = Mask(mask_array, bbox)
    assert np.array_equal(mask._mask, mask_array)
    assert np.array_equal(mask._bbox, bbox)


def test_mask_getstate_setstate() -> None:
    """Test Mask serialization and deserialization."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask = Mask(mask_array, bbox)

    # Test serialization
    state = mask.__getstate__()
    assert "_mask" in state
    assert np.array_equal(state["_bbox"], bbox)

    # Test deserialization
    new_mask = Mask.__new__(Mask)
    new_mask.__setstate__(state)
    # After deserialization, the mask should be restored correctly
    assert np.array_equal(new_mask._mask, mask_array)
    assert np.array_equal(new_mask._bbox, bbox)


def test_mask_indices_no_offset() -> None:
    """Test mask_indices with no offset."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([1, 2, 3, 4])  # min_y, min_x, max_y, max_x

    mask = Mask(mask_array, bbox)
    indices = mask.mask_indices()

    # True values are at positions (0,0) and (1,1) in the mask
    # With bbox offset [1, 2]: (0+1, 0+2) and (1+1, 1+2) = (1, 2) and (2, 3)
    expected_y = np.array([1, 2])  # row indices of True values + bbox[0]
    expected_x = np.array([2, 3])  # col indices of True values + bbox[1]

    assert len(indices) == 2
    assert np.array_equal(indices[0], expected_y)
    assert np.array_equal(indices[1], expected_x)


def test_mask_indices_with_scalar_offset() -> None:
    """Test mask_indices with scalar offset."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([1, 2, 3, 4])

    mask = Mask(mask_array, bbox)
    indices = mask.mask_indices(offset=5)

    # True values at (0,0) and (1,1) in mask
    # With bbox [1, 2] and offset 5: (0+1+5, 0+2+5) and (1+1+5, 1+2+5) = (6, 7) and (7, 8)
    expected_y = np.array([6, 7])  # row indices + bbox[0] + offset
    expected_x = np.array([7, 8])  # col indices + bbox[1] + offset

    assert len(indices) == 2
    assert np.array_equal(indices[0], expected_y)
    assert np.array_equal(indices[1], expected_x)


def test_mask_indices_with_array_offset() -> None:
    """Test mask_indices with array offset."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([1, 2, 3, 4])

    mask = Mask(mask_array, bbox)
    offset = np.array([3, 4])
    indices = mask.mask_indices(offset=offset)

    # True values at (0,0) and (1,1) in mask
    # With bbox [1, 2] and offset [3, 4]: (0+1+3, 0+2+4) and (1+1+3, 1+2+4) = (4, 6) and (5, 7)
    expected_y = np.array([4, 5])  # row indices + bbox[0] + offset[0]
    expected_x = np.array([6, 7])  # col indices + bbox[1] + offset[1]

    assert len(indices) == 2
    assert np.array_equal(indices[0], expected_y)
    assert np.array_equal(indices[1], expected_x)


def test_mask_indices_3d() -> None:
    """Test mask_indices with 3D mask."""
    mask_array = np.array([[[True, False], [False, False]], [[False, False], [False, True]]], dtype=bool)
    bbox = np.array([1, 2, 3, 3, 4, 5])  # min_z, min_y, min_x, max_z, max_y, max_x

    mask = Mask(mask_array, bbox)
    indices = mask.mask_indices()

    # True values at (0,0,0) and (1,1,1) in mask
    # With bbox offset [1,2,3]: (0+1, 0+2, 0+3) and (1+1, 1+2, 1+3) = (1,2,3) and (2,3,4)
    expected_z = np.array([1, 2])
    expected_y = np.array([2, 3])
    expected_x = np.array([3, 4])

    assert len(indices) == 3
    assert np.array_equal(indices[0], expected_z)
    assert np.array_equal(indices[1], expected_y)
    assert np.array_equal(indices[2], expected_x)


def test_paint_buffer() -> None:
    """Test paint_buffer method."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask = Mask(mask_array, bbox)

    # Create a buffer to paint on
    buffer = np.zeros((4, 4), dtype=float)
    mask.paint_buffer(buffer, value=5.0)

    # Check that the correct positions are painted
    expected_buffer = np.zeros((4, 4), dtype=float)
    expected_buffer[0, 0] = 5.0  # First True position
    expected_buffer[1, 1] = 5.0  # Second True position

    assert np.array_equal(buffer, expected_buffer)


def test_paint_buffer_with_offset() -> None:
    """Test paint_buffer method with offset."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask = Mask(mask_array, bbox)

    # Create a buffer to paint on
    buffer = np.zeros((6, 6), dtype=float)
    offset = np.array([2, 3])
    mask.paint_buffer(buffer, value=7.0, offset=offset)

    # Check that the correct positions are painted with offset
    expected_buffer = np.zeros((6, 6), dtype=float)
    expected_buffer[2, 3] = 7.0  # First True position + offset
    expected_buffer[3, 4] = 7.0  # Second True position + offset

    assert np.array_equal(buffer, expected_buffer)


def test_mask_iou() -> None:
    """Test IoU calculation between masks."""
    # Create two overlapping masks
    mask1_array = np.array([[True, True], [True, False]], dtype=bool)
    bbox1 = np.array([0, 0, 2, 2])
    mask1 = Mask(mask1_array, bbox1)

    mask2_array = np.array([[True, False], [True, True]], dtype=bool)
    bbox2 = np.array([0, 0, 2, 2])
    mask2 = Mask(mask2_array, bbox2)

    iou = mask1.iou(mask2)

    # Intersection: positions (0,0) and (1,0) = 2 pixels
    # Union: 3 + 3 - 2 = 4 pixels
    # IoU = 2/4 = 0.5
    expected_iou = 0.5
    assert abs(iou - expected_iou) < 1e-6


def test_mask_iou_no_overlap() -> None:
    """Test IoU calculation with non-overlapping masks."""
    mask1_array = np.array([[True, False], [False, False]], dtype=bool)
    bbox1 = np.array([0, 0, 2, 2])
    mask1 = Mask(mask1_array, bbox1)

    mask2_array = np.array([[False, False], [False, True]], dtype=bool)
    bbox2 = np.array([0, 0, 2, 2])
    mask2 = Mask(mask2_array, bbox2)

    iou = mask1.iou(mask2)
    assert iou == 0.0


def test_mask_iou_identical() -> None:
    """Test IoU calculation with identical masks."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask1 = Mask(mask_array, bbox)
    mask2 = Mask(mask_array.copy(), bbox.copy())

    iou = mask1.iou(mask2)
    assert iou == 1.0


def test_mask_empty() -> None:
    """Test mask with no True values."""
    mask_array = np.array([[False, False], [False, False]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask = Mask(mask_array, bbox)
    indices = mask.mask_indices()

    # Should return empty arrays
    assert len(indices) == 2
    assert len(indices[0]) == 0
    assert len(indices[1]) == 0


def test_mask_all_true() -> None:
    """Test mask with all True values."""
    mask_array = np.array([[True, True], [True, True]], dtype=bool)
    bbox = np.array([1, 1, 3, 3])

    mask = Mask(mask_array, bbox)
    indices = mask.mask_indices()

    # Should return all positions
    expected_y = np.array([1, 1, 2, 2])
    expected_x = np.array([1, 2, 1, 2])

    assert len(indices) == 2
    assert np.array_equal(indices[0], expected_y)
    assert np.array_equal(indices[1], expected_x)


def test_mask_repr() -> None:
    """Test mask representation."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask = Mask(mask_array, bbox)
    assert repr(mask) == "Mask(bbox=[0:2, 0:2])"


def test_mask_crop() -> None:
    """Test mask cropping."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([1, 1, 3, 3])

    mask = Mask(mask_array, bbox)
    image = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    cropped_image = mask.crop(image)
    assert np.array_equal(cropped_image, image[1:3, 1:3])


def test_mask_crop_with_shape() -> None:
    """Test mask cropping with shape."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([1, 1, 3, 3])

    mask = Mask(mask_array, bbox)
    image = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    cropped_image = mask.crop(image, shape=(2, 4))
    assert np.array_equal(cropped_image, image[1:3, 0:4])


def test_mask_from_coordinates_2d_basic() -> None:
    """Test 2D mask creation and bbox without cropping."""
    center = np.asarray([5, 5])
    radius = 2
    mask = Mask.from_coordinates(center, radius)
    # Should be a disk of radius 2, shape (5,5), centered at (5,5)
    assert mask.mask.shape == (5, 5)
    assert mask.mask[2, 2]  # center pixel is True
    np.testing.assert_array_equal(mask.bbox, [3, 3, 8, 8])


def test_mask_from_coordinates_3d_basic() -> None:
    """Test 3D mask creation and bbox without cropping."""
    center = np.asarray([4, 5, 6])
    radius = 1
    mask = Mask.from_coordinates(center, radius)
    # Should be a ball of radius 1, shape (3,3,3), centered at (4,5,6)
    assert mask.mask.shape == (3, 3, 3)
    assert mask.mask[1, 1, 1]  # center voxel is True
    np.testing.assert_array_equal(mask.bbox, [3, 4, 5, 6, 7, 8])


def test_mask_from_coordinates_cropping() -> None:
    """Test cropping when mask falls outside the image boundary."""
    center = np.asarray([0, 0])
    radius = 5
    image_shape = (4, 3)

    mask = Mask.from_coordinates(center, radius, image_shape=image_shape)

    # Mask shape should match the bbox size
    expected_shape = (4, 3)
    assert mask.mask.shape == expected_shape

    # Mask should be cropped to fit within image bounds
    np.testing.assert_array_equal(mask.bbox, [0, 0, 4, 3])
