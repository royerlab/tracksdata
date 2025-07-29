import numpy as np

from tracksdata.functional._mask import (
    crop_image_with_bbox,
    mask_indices,
    mask_intersection,
    mask_iou,
    paint_mask_to_buffer,
)


def test_mask_indices_no_offset() -> None:
    """Test mask_indices with no offset."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([1, 2, 3, 4])  # min_y, min_x, max_y, max_x

    indices = mask_indices(mask_array, bbox)

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

    indices = mask_indices(mask_array, bbox, offset=5)

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

    offset = np.array([3, 4])
    indices = mask_indices(mask_array, bbox, offset=offset)

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

    indices = mask_indices(mask_array, bbox)

    # True values at (0,0,0) and (1,1,1) in mask
    # With bbox offset [1,2,3]: (0+1, 0+2, 0+3) and (1+1, 1+2, 1+3) = (1,2,3) and (2,3,4)
    expected_z = np.array([1, 2])
    expected_y = np.array([2, 3])
    expected_x = np.array([3, 4])

    assert len(indices) == 3
    assert np.array_equal(indices[0], expected_z)
    assert np.array_equal(indices[1], expected_y)
    assert np.array_equal(indices[2], expected_x)


def test_paint_mask_to_buffer() -> None:
    """Test paint_mask_to_buffer function."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    # Create a buffer to paint on
    buffer = np.zeros((4, 4), dtype=float)
    paint_mask_to_buffer(buffer, mask_array, bbox, value=5.0)

    # Check that the correct positions are painted
    expected_buffer = np.zeros((4, 4), dtype=float)
    expected_buffer[0, 0] = 5.0  # First True position
    expected_buffer[1, 1] = 5.0  # Second True position

    assert np.array_equal(buffer, expected_buffer)


def test_paint_mask_to_buffer_with_offset() -> None:
    """Test paint_mask_to_buffer function with offset."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    # Create a buffer to paint on
    buffer = np.zeros((6, 6), dtype=float)
    offset = np.array([2, 3])
    paint_mask_to_buffer(buffer, mask_array, bbox, value=7.0, offset=offset)

    # Check that the correct positions are painted with offset
    expected_buffer = np.zeros((6, 6), dtype=float)
    expected_buffer[2, 3] = 7.0  # First True position + offset
    expected_buffer[3, 4] = 7.0  # Second True position + offset

    assert np.array_equal(buffer, expected_buffer)


def test_mask_iou() -> None:
    """Test IoU calculation between mask/bbox pairs."""
    # Create two overlapping masks
    mask1_array = np.array([[True, True], [True, False]], dtype=bool)
    bbox1 = np.array([0, 0, 2, 2])

    mask2_array = np.array([[True, False], [True, True]], dtype=bool)
    bbox2 = np.array([0, 0, 2, 2])

    iou = mask_iou(bbox1, mask1_array, bbox2, mask2_array)

    # Intersection: positions (0,0) and (1,0) = 2 pixels
    # Union: 3 + 3 - 2 = 4 pixels
    # IoU = 2/4 = 0.5
    expected_iou = 0.5
    assert abs(iou - expected_iou) < 1e-6


def test_mask_iou_no_overlap() -> None:
    """Test IoU calculation with non-overlapping masks."""
    mask1_array = np.array([[True, False], [False, False]], dtype=bool)
    bbox1 = np.array([0, 0, 2, 2])

    mask2_array = np.array([[False, False], [False, True]], dtype=bool)
    bbox2 = np.array([0, 0, 2, 2])

    iou = mask_iou(bbox1, mask1_array, bbox2, mask2_array)
    assert iou == 0.0


def test_mask_iou_identical() -> None:
    """Test IoU calculation with identical masks."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    iou = mask_iou(bbox, mask_array, bbox.copy(), mask_array.copy())
    assert iou == 1.0


def test_mask_intersection() -> None:
    """Test intersection calculation between mask/bbox pairs."""
    # Create two overlapping masks
    mask1_array = np.array([[True, True], [True, False]], dtype=bool)
    bbox1 = np.array([0, 0, 2, 2])

    mask2_array = np.array([[True, False], [True, True]], dtype=bool)
    bbox2 = np.array([0, 0, 2, 2])

    intersection = mask_intersection(bbox1, mask1_array, bbox2, mask2_array)

    # Intersection: positions (0,0) and (1,0) = 2 pixels
    expected_intersection = 2.0
    assert abs(intersection - expected_intersection) < 1e-6


def test_mask_empty() -> None:
    """Test mask with no True values."""
    mask_array = np.array([[False, False], [False, False]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    indices = mask_indices(mask_array, bbox)

    # Should return empty arrays
    assert len(indices) == 2
    assert len(indices[0]) == 0
    assert len(indices[1]) == 0


def test_mask_all_true() -> None:
    """Test mask with all True values."""
    mask_array = np.array([[True, True], [True, True]], dtype=bool)
    bbox = np.array([1, 1, 3, 3])

    indices = mask_indices(mask_array, bbox)

    # Should return all positions
    expected_y = np.array([1, 1, 2, 2])
    expected_x = np.array([1, 2, 1, 2])

    assert len(indices) == 2
    assert np.array_equal(indices[0], expected_y)
    assert np.array_equal(indices[1], expected_x)


def test_crop_image_with_bbox() -> None:
    """Test image cropping with bbox."""
    bbox = np.array([1, 1, 3, 3])
    image = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])

    cropped_image = crop_image_with_bbox(image, bbox)
    expected_crop = image[1:3, 1:3]

    assert np.array_equal(cropped_image, expected_crop)


def test_crop_image_with_bbox_and_shape() -> None:
    """Test image cropping with bbox and specific shape."""
    bbox = np.array([1, 1, 3, 3])
    image = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])

    cropped_image = crop_image_with_bbox(image, bbox, shape=(2, 4))
    expected_crop = image[1:3, 0:4]

    assert np.array_equal(cropped_image, expected_crop)
